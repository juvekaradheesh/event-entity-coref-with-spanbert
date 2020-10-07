import torch
import torch.nn as nn

from transformers import AutoModel
from transformers.modeling_outputs import TokenClassifierOutput

class Coref(nn.Module):
    def __init__(self, config, device, spanbert_save_path=None):
        super(Coref, self).__init__()
        
        self.config = config
        self.device = device
        self.span_rep_size = 3*config['spanbert_out'] + config['width_encoding_feature_size']
        if spanbert_save_path is None:
            self.spanBERT = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")
        else:
            self.spanBERT = AutoModel.from_pretrained(spanbert_save_path)
        self.ffnn_alpha = nn.Linear(config['spanbert_out'], 1)
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.softmax = nn.Softmax(dim=1)
        self.ffnm_m = nn.Linear(self.span_rep_size, 1)
        self.ffnm_c = nn.Linear(self.span_rep_size, self.span_rep_size, bias=False)
        self.ffnm_a = nn.Linear(3*self.span_rep_size, 1)
        self.ffnm_f = nn.Linear(2*self.span_rep_size, self.span_rep_size)
        self.sigmoid = nn.Sigmoid()
        self.width_encoding_features = torch.normal(mean=0, std=0.02,size=(self.config['max_span_width'],config['width_encoding_feature_size'])).to(self.device)
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sentence_map=None,
        gold_starts=None,
        gold_ends=None,
        cluster_ids=None
    ):

        # Get the output from SpanBERT model as x_t^*
        outputs = self.spanBERT(
            input_ids, 
            attention_mask=attention_mask,
        )
        
        sequence_output = self.dropout(outputs[0]) # [num_sentences, max_sent_len, emb_dim]
        
        # Remove padding and flatten the spanBERT embeddings (x_t^*) 

        num_sentences = input_ids.shape[0]
        max_sentence_length = input_ids.shape[1]

        sequence_output = torch.masked_select(sequence_output, attention_mask.view(attention_mask.shape[0], attention_mask.shape[1], 1)>0)
        sequence_output = sequence_output.view(-1, self.config['spanbert_out'])
        sentence_map = torch.masked_select(sentence_map, attention_mask>0)

        num_words =  sequence_output.shape[0]
        
        # Get candidate start indices and end indices as 1d tensors

        candidate_starts = torch.unsqueeze(torch.arange(num_words), 1).repeat(1, self.config['max_span_width']).to(self.device) # [num_words, max_span_width]
        candidate_ends = candidate_starts + torch.unsqueeze(torch.arange(self.config['max_span_width']), 0).to(self.device) # [num_words, max_span_width]
        candidate_start_sentence_indices = sentence_map[candidate_starts] # [num_words, max_span_width]
        candidate_end_sentence_indices = sentence_map[torch.clamp(candidate_ends, max=num_words-1)] # [num_words, max_span_width]
        candidate_mask = torch.logical_and(candidate_ends < num_words, torch.eq(candidate_start_sentence_indices, candidate_end_sentence_indices))
        flattened_candidate_mask = candidate_mask.view(-1, ) # [num_words * max_span_width]
        candidate_starts = torch.masked_select(candidate_starts.view(-1,), flattened_candidate_mask) # [num_candidates]
        candidate_ends = torch.masked_select(candidate_ends.view(-1,), flattened_candidate_mask) # [num_candidates]
        candidate_sentence_indices = torch.masked_select(candidate_start_sentence_indices.view(-1, ), flattened_candidate_mask) # [num_candidates]
        
        candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends, cluster_ids)
        
        # Get Span representations
        span_representations = self.get_span_representations(sequence_output, candidate_starts, candidate_ends)
        
        # Get Mention Scores(s_m)
        span_mention_scores = self.dropout(self.ffnm_m(span_representations))
        span_mention_scores = span_mention_scores.view(-1, )

        # Coarse to fine pruning
        #    Stage 1
        #    Keep top M spans, M = top_span_ratio(lambda) * num_words (T)

        m = int(self.config['top_span_ratio'] * num_words)
        top_span_indices = torch.topk(span_mention_scores, m).indices
        top_span_indices = top_span_indices.sort().values

        top_span_representations = span_representations[top_span_indices]
        top_span_starts = candidate_starts[top_span_indices]
        top_span_ends = candidate_ends[top_span_indices]
        top_span_mention_scores = span_mention_scores[top_span_indices]
        top_span_cluster_ids = candidate_cluster_ids[top_span_indices]

        #    Stage 2
        #    Calculate bilinear score(s_c)
        #    Keep top K antecedents of each span based on first three factors of coreference score
        #    i.e s_m(i) + s_m(j) + s_c(i, j)

        k = min(m, self.config['max_top_antecedents'])
        top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.prune_antecedents(top_span_representations, top_span_mention_scores, k)
        dummy_scores = torch.zeros([m, 1]).to(self.device)

        for i in range(self.config['coref_depth']):
            #    Stage 3
            #    Calculate coreference scores s(i, j)

            top_antecedent_representations = top_span_representations[top_antecedents]
            slow_antecedent_scores = self.get_slow_antecedent_scores(top_span_representations, top_antecedents, top_antecedent_representations, top_antecedent_offsets)
            top_antecedent_scores = top_fast_antecedent_scores + slow_antecedent_scores

            top_antecedent_weights = self.softmax(torch.cat((dummy_scores, top_antecedent_scores), 1)) # [m, k + 1]
            top_antecedent_representations = torch.cat((torch.unsqueeze(top_span_representations, 1), top_antecedent_representations), 1) # [m, k + 1, emb]
            attended_span_emb = torch.sum(torch.unsqueeze(top_antecedent_weights, 2) * top_antecedent_representations, 1) # [m, emb]
            f = self.sigmoid(self.ffnm_f(torch.cat((top_span_representations, attended_span_emb), 1))) # [m, emb]

            top_span_representations = f * attended_span_emb + (1 - f) * top_span_representations # [m, emb]

        top_antecedent_scores = torch.cat((dummy_scores, top_antecedent_scores), 1)

        
        top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedents] 
        top_antecedent_cluster_ids += torch.log(top_antecedents_mask.float()).long()
        
        same_cluster_indicator = torch.eq(top_antecedent_cluster_ids, top_span_cluster_ids.unsqueeze(1)) # [k, c]
        
        non_dummy_indicator = (top_span_cluster_ids > 0).unsqueeze(1) # [m, 1]
        pairwise_labels = torch.logical_and(same_cluster_indicator, non_dummy_indicator) # [m, k]
        dummy_labels = torch.logical_not(torch.sum(pairwise_labels, 1, keepdims=True)) # [m, 1]
        top_antecedent_labels = torch.cat([dummy_labels, pairwise_labels], 1) # [m, k + 1]

        loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels)
        loss = torch.sum(loss.float())
        
        return [candidate_starts, candidate_ends, span_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores], loss

    def softmax_loss(self, antecedent_scores, antecedent_labels):
        gold_scores = antecedent_scores + torch.log(antecedent_labels.float()) # [m, max_ant + 1]
        marginalized_gold_scores = torch.logsumexp(gold_scores, [1]) # [m]
        log_norm = torch.logsumexp(antecedent_scores, [1]) # [m]

        return log_norm - marginalized_gold_scores # [m]

    def get_span_representations(self, sequence_output, span_starts, span_ends):
        
        span_representations = []
        num_candidates = span_starts.shape[0]
        num_words = sequence_output.shape[0]

        start_emb = sequence_output[span_starts]

        end_emb = sequence_output[span_ends]

        doc_range = torch.unsqueeze(torch.arange(0, num_words), 0).repeat(num_candidates, 1).to(self.device) # [num_candidates, num_words]
        mention_mask = torch.logical_and(doc_range >= torch.unsqueeze(span_starts, 1), doc_range <= torch.unsqueeze(span_ends, 1)) # [num_candidates, num_words]
        alphas = torch.squeeze(self.ffnn_alpha(sequence_output), 1)
        mention_word_attn = self.softmax(torch.log(mention_mask.float()) + torch.unsqueeze(alphas, 0)) # [num_candidates, num_words]

        span_emb = torch.matmul(mention_word_attn, sequence_output)
        
        span_widths = (span_ends - span_starts + 1).view(-1,1)

        span_width_embeddings = self.size_encoding(span_widths)

        span_representations = torch.cat((start_emb, end_emb, span_emb, span_width_embeddings), 1)

        return span_representations

    def size_encoding(self, span_widths):
        span_widths = span_widths.view(-1, )
        span_widths = span_widths-1
        span_width_embeddings = self.width_encoding_features[span_widths]
        return span_width_embeddings
    
    def prune_antecedents(self, top_span_emb, top_span_mention_scores, k):
        m = top_span_emb.shape[0]
        top_span_range = torch.arange(m)
        antecedent_offsets = (torch.unsqueeze(top_span_range, 1) - torch.unsqueeze(top_span_range, 0)).to(self.device)
        antecedents_mask = antecedent_offsets >= 1
        
        # s_m(i) + s_m(j)
        fast_antecedent_scores = torch.unsqueeze(top_span_mention_scores, 1) + torch.unsqueeze(top_span_mention_scores, 0)
        fast_antecedent_scores += torch.log(antecedents_mask.float()) 
        
        # bilinear score s_c(i, j)
        source_top_span_emb = self.dropout(self.ffnm_c(top_span_emb))
        target_top_span_emb = self.dropout(top_span_emb)
        fast_antecedent_scores += torch.matmul(source_top_span_emb, torch.transpose(target_top_span_emb, 0, 1))

        top_antecedents = torch.topk(fast_antecedent_scores, k, sorted=False).indices # [m, k]
        top_antecedents_mask = self.batch_gather(antecedents_mask, top_antecedents) # [m, k]
        top_fast_antecedent_scores = self.batch_gather(fast_antecedent_scores, top_antecedents) # [m, k]
        top_antecedent_offsets = self.batch_gather(antecedent_offsets, top_antecedents) # [m, k]

        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets
    
    def batch_gather(self, emb, indices):
        batch_size = emb.shape[0]
        seqlen = emb.shape[1]
        if len(emb.shape) > 2:
            emb_size = emb.shape[2]
        else:
            emb_size = 1
        flattened_emb = torch.reshape(emb, [batch_size * seqlen, emb_size])  # [batch_size * seqlen, emb]
        offset = torch.unsqueeze(torch.arange(batch_size) * seqlen, 1).to(self.device)  # [batch_size, 1]
        gathered = flattened_emb[indices + offset] # [batch_size, num_indices, emb]
        if len(emb.shape) == 2:
            gathered = torch.squeeze(gathered, 2) # [batch_size, num_indices]
        return gathered

    def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets):
        m = top_span_emb.shape[0]
        k = top_antecedents.shape[1]

        target_emb = torch.unsqueeze(top_span_emb, 1) # [m, 1, emb]
        similarity_emb = top_antecedent_emb * target_emb # [m, k, emb]
        target_emb = target_emb.repeat(1, k, 1) # [m, k, emb]

        pair_emb = torch.cat((target_emb, top_antecedent_emb, similarity_emb), 2)
        slow_antecedent_scores = self.ffnm_a(pair_emb)
        
        slow_antecedent_scores = torch.squeeze(slow_antecedent_scores, 2)

        return slow_antecedent_scores
    
    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        same_start = torch.eq(labeled_starts.unsqueeze(1), candidate_starts.unsqueeze(0)) # [num_labeled, num_candidates]
        same_end = torch.eq(labeled_ends.unsqueeze(1), candidate_ends.unsqueeze(0)) # [num_labeled, num_candidates]
        same_span = torch.logical_and(same_start, same_end) # [num_labeled, num_candidates]
        candidate_labels = torch.matmul(labels.unsqueeze(0), same_span.float()) # [1, num_candidates]
        candidate_labels = candidate_labels.squeeze(0)

        return candidate_labels