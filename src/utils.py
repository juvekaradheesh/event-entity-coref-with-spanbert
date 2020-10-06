import os
import xmltodict
import json

import numpy as np
import torch

def process_ecb_plus(data_path, mention_type_start):
    all_sentences = []
    all_tokens = []
    batch_indices = [0]
    mentions = []
    gold_starts = []
    gold_ends = []
    clusters = []
    batch_index = 0
    for dir_ in os.listdir(os.path.join(data_path, 'ECB+')):
        if dir_ in ['15', '17']:
            continue
        dir_path = os.path.join(data_path, 'ECB+', str(dir_))
        for file in os.listdir(dir_path):
            is_plus = True if 'plus' in file else False
            file_path = os.path.join(dir_path, file)
            sentences, tokens,  mens, gold_s, gold_e, clusts = process_ecb_xml(file_path, is_plus=is_plus, mention_type_start=mention_type_start)
            all_sentences.extend(sentences)
            all_tokens.append(tokens)
            batch_index +=len(sentences)
            batch_indices.append(batch_index)
            mentions.append(mens)
            clusters.append(clusts)
            gold_starts.append(gold_s)
            gold_ends.append(gold_e)
            

    return all_sentences, all_tokens, batch_indices, mentions, gold_starts, gold_ends, clusters

def process_ecb_xml(file_path, is_plus, mention_type_start):
    f = open(file_path, 'r')
    obj = xmltodict.parse(f.read())
    sentences = []
    tokens = []
    
    sentence = []
    sent_tokens = []
    
    curr_sentence = 0
    for item in obj['Document']['token']:
        if is_plus and int(item['@sentence']) == 0:
            continue

        if int(item['@sentence']) != curr_sentence:
            
            curr_sentence = int(item['@sentence'])
            sentences.append(sentence)
            tokens.append(sent_tokens)
            sentence = []
            sent_tokens = []
            
        if '#text' in item:
            sentence.append(item['#text'])
            sent_tokens.append(int(item['@t_id']))

    sentences.append(sentence)
    tokens.append(sent_tokens)

    mentions = []
    gold_starts = []
    gold_ends = []
    relation_mentions = []
    clusters = []

    for mention_type in obj['Document']['Markables']:
        if mention_type.startswith(mention_type_start):
            if '@m_id' in obj['Document']['Markables'][mention_type]:
                if 'token_anchor' in obj['Document']['Markables'][mention_type]:
                    mentions.append(int(obj['Document']['Markables'][mention_type]['@m_id']))
                    if '@t_id' in obj['Document']['Markables'][mention_type]['token_anchor']:
                        gold_starts.append(int(obj['Document']['Markables'][mention_type]['token_anchor']['@t_id']))
                        gold_ends.append(int(obj['Document']['Markables'][mention_type]['token_anchor']['@t_id']))
                    else:
                        gold_starts.append(int(obj['Document']['Markables'][mention_type]['token_anchor'][0]['@t_id']))
                        gold_ends.append(int(obj['Document']['Markables'][mention_type]['token_anchor'][-1]['@t_id']))
                else:
                    relation_mentions.append(int(obj['Document']['Markables'][mention_type]['@m_id']))
            else:
                for item in obj['Document']['Markables'][mention_type]:
                    if 'token_anchor' in item:
                        mentions.append(int(item['@m_id']))
                        if '@t_id' in item['token_anchor']:
                            gold_starts.append(int(item['token_anchor']['@t_id']))
                            gold_ends.append(int(item['token_anchor']['@t_id']))
                        else:
                            gold_starts.append(int(item['token_anchor'][0]['@t_id']))
                            gold_ends.append(int(item['token_anchor'][-1]['@t_id']))
                    else:
                        relation_mentions.append(int(item['@m_id']))
                
                for item in obj['Document']['Relations']['CROSS_DOC_COREF']:
                    if int(item['target']['@m_id']) in relation_mentions:
                        cluster = []
                        if '@m_id' in item['source']:
                            cluster.append(item['source']['@m_id'])
                        else:
                            for mention in item['source']:
                                cluster.append(int(mention['@m_id']))
                        clusters.append(cluster)

    return sentences, tokens, mentions, gold_starts, gold_ends, clusters

def fix_tokens_with_offsets(tokens, offset_mapping, batch_indices):
    all_tokens = []
    split_offsets = [offset_mapping[batch_indices[i]: batch_indices[i+1]] for i in range(len(batch_indices)-1)]
    for z, (tok, off_map) in enumerate(zip(tokens, split_offsets)):
        encoded_tokens = []
        for doc_tokens, doc_offset in zip(tok, off_map):
            # create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
            arr_offset = np.array(doc_offset)

            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_tokens
            encoded_tokens.append(doc_enc_labels.tolist())
            
        for i, (doc_tokens, doc_offset) in enumerate(zip(tok, off_map)):
            for j in range(len(doc_offset)):
                if doc_offset[j][0] != 0:
                    encoded_tokens[i][j] = encoded_tokens[i][j-1]
        all_tokens.append(encoded_tokens)
    return all_tokens

def create_unmasked_sentence_map(offset_mapping, batch_indices):
    
    complete_sentence_map = []
    split_offsets = [offset_mapping[batch_indices[i]: batch_indices[i+1]] for i in range(len(batch_indices)-1)]
    for off_map in split_offsets:
        sentence_map = []
        for i, off in enumerate(off_map):
            sentence_map.append([i]*len(off))
        complete_sentence_map.append(sentence_map)
    
    return complete_sentence_map

def process_gold_mentions(all_tokens, all_gold_starts, all_gold_ends, all_mentions, attention_mask, batch_indices):

    attention_masks = [attention_mask[batch_indices[i]: batch_indices[i+1]] for i in range(len(batch_indices)-1)]

    new_tokens = []
    full_gold_starts = []
    full_gold_ends = []
    new_mentions = []

    for tokens, gold_starts, gold_ends, mentions, attention_mask in zip(all_tokens, all_gold_starts, all_gold_ends, all_mentions, attention_masks):

        # Flatten Tokens and remove padding
        tokens = torch.Tensor(tokens)
        gold_starts = torch.Tensor(gold_starts)
        gold_ends = torch.Tensor(gold_ends)
        mentions = torch.Tensor(mentions)
        attention_mask = torch.tensor(attention_mask)
        
        tokens = torch.masked_select(tokens, attention_mask>0)

        sort_indices = mentions.sort().indices
        mentions = mentions[sort_indices]
        gold_starts = gold_starts[sort_indices]
        gold_ends = gold_ends[sort_indices]
        new_gold_starts = []
        for item in gold_starts:
            new_gold_starts.append(torch.nonzero(tokens==item)[0][0])
        new_gold_ends = []
        for item in gold_ends:
            new_gold_ends.append(torch.nonzero(tokens==item)[-1][0])
        
        if len(new_gold_starts) > 0:
            gold_starts = torch.stack(new_gold_starts)
            gold_ends = torch.stack(new_gold_ends)
        else:
            gold_starts = torch.Tensor([])
            gold_ends = torch.Tensor([])

        new_tokens.append(tokens)
        full_gold_starts.append(gold_starts)
        full_gold_ends.append(gold_ends)
        new_mentions.append(mentions)

    return new_tokens, full_gold_starts, full_gold_ends, new_mentions

def get_cluster_ids(all_mentions, all_clusters):
    all_cluster_ids = []
    for mentions,clusters in zip(all_mentions, all_clusters):
        cluster_ids = []
        for mention in mentions:
            mention_appeared=False
            for i, cluster in enumerate(clusters):
                if mention in cluster:
                    mention_appeared = True
                    cluster_ids.append(i+1)
                    break
            if not mention_appeared:
                cluster_ids.append(0)
                # cluster_ids.append(torch.nonzero(clusters==mention)[0][0]+1)
        all_cluster_ids.append(torch.Tensor(cluster_ids))

    return all_cluster_ids