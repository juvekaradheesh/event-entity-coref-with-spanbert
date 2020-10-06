import torch

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.model import Coref
from src.data_loader import ECBDataset
from src.utils import *
from tqdm import tqdm

def train(num_epoch, train_loader, model, optimizer, model_path, device):
    
    writer = SummaryWriter()

    for epoch in range(num_epoch):
        print("Epoch", epoch)
        epoch_loss = 0
        for i, batch in enumerate(tqdm(train_loader)):
            model.train()

            input_ids = batch['input_ids'][0].to(device)
            attention_mask = batch['attention_mask'][0].to(device)
            sentence_map = batch['sentence_map'][0].to(device)
            gold_starts = batch['gold_starts'][0].to(device)
            gold_ends = batch['gold_ends'][0].to(device)
            cluster_ids = batch['cluster_ids'][0].to(device)

            predictions, loss = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                sentence_map=sentence_map,
                gold_starts=gold_starts,
                gold_ends=gold_ends,
                cluster_ids=cluster_ids
                )
            loss.backward()
            optimizer.step()

            epoch_loss += loss
        
        torch.save(model.state_dict(), model_path)
        writer.add_scalar('Loss/train', loss, epoch)
        print("Loss = ", epoch_loss/len(train_loader))
    
    writer.close()

def evaluate(test_loader, model, device):
    epoch_loss = 0
    for i, batch in enumerate(tqdm(test_loader)):
        model.eval()
        
        input_ids = batch['input_ids'][0].to(device)
        attention_mask = batch['attention_mask'][0].to(device)
        sentence_map = batch['sentence_map'][0].to(device)
        gold_starts = batch['gold_starts'][0].to(device)
        gold_ends = batch['gold_ends'][0].to(device)
        cluster_ids = batch['cluster_ids'][0].to(device)

        print(cluster_ids)

        predictions, loss = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            sentence_map=sentence_map,
            gold_starts=gold_starts,
            gold_ends=gold_ends,
            cluster_ids=cluster_ids
            )

        print(predictions)
        print(loss)
        epoch_loss += loss
        if i==2:
            break
    print("Loss = ", epoch_loss/len(test_loader))

    
if __name__ == "__main__":

    device = torch.device("cpu")
    print(device)

    num_epoch = 10
    all_sentences, tokens, batch_indices, mentions, gold_starts, gold_ends, clusters = process_ecb_plus('data/train', 'HUMAN')
    tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased", use_fast=True)
    encodings = tokenizer(all_sentences, return_offsets_mapping=True, is_split_into_words=True, truncation=True, padding=True)
    encoded_tokens = fix_tokens_with_offsets(tokens, encodings.offset_mapping, batch_indices)
    encoded_sentence_map = create_unmasked_sentence_map(encodings.offset_mapping, batch_indices)
    encodings.pop('offset_mapping')
    encoded_tokens, gold_starts, gold_ends, mentions = process_gold_mentions(encoded_tokens, gold_starts, gold_ends, mentions, encodings.attention_mask, batch_indices)
    cluster_ids = get_cluster_ids(mentions, clusters)

    dataset = ECBDataset(
        encodings=encodings, 
        batch_indices=batch_indices,
        sentence_map=encoded_sentence_map,
        gold_starts=gold_starts,
        gold_ends=gold_ends,
        cluster_ids=cluster_ids)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )
    config ={
        'spanbert_out': 768,
        'dropout_rate': 0.2,
        'max_span_width': 10,
        'top_span_ratio': 0.4,
        'max_top_antecedents': 50,
        'coref_depth': 2,
        'width_encoding_feature_size': 20
    }
    model = Coref(config, device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    train(num_epoch, loader, model, optimizer, 'models/temp.pt', device)