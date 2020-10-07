from argparse import ArgumentParser
from config import config

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

from src.train import train, evaluate
from src.model import Coref
from src.data_loader import ECBDataset
from src.utils import *


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-M', '--mode', help='select mode to run', required=True, choices=['train', 'test'])
    parser.add_argument('-m', '--model_path', help='provide save/load path for the model', required=True)
    parser.add_argument('-t', '--mention_type', help='mention type to extract', required=True)
    parser.add_argument('-d', '--data_path', help='provide data path', required=False)

    args = vars(parser.parse_args())
    
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    # device = torch.device("cpu")
    print(device)

    
    if args['mode'] == 'train':
        model = Coref(config, device)
        model.to(device)

        all_sentences, tokens, batch_indices, mentions, gold_starts, gold_ends, clusters = process_ecb_plus(args['data_path'], args['mention_type'])
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
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        spanbert_save_path = 'models/spanbert_' + args['mention_type']
        
        train(config['num_epoch'], loader, model, optimizer, args['model_path'], device, spanbert_save_path)
    
    if args['mode'] == 'test':
        spanbert_save_path = 'models/spanbert_' + args['mention_type']
        model = Coref(config, device, spanbert_save_path)
        model.to(device)

        all_sentences, tokens, batch_indices, mentions, gold_starts, gold_ends, clusters = process_ecb_plus(args['data_path'], args['mention_type'])
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

        model.load_state_dict(torch.load(args['model_path']))

        # print(model)
        print(model.parameters)

        evaluate(loader, model, device)
        evaluate(loader, model, device)
    
    # if args['mode'] == 'predict':
    #     all_sentences, tokens, batch_indices, mentions, gold_starts, gold_ends, clusters = process_ecb_plus(args['data_path'], args['mention_type'])
    #     tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased", use_fast=True)
    #     encodings = tokenizer(all_sentences, return_offsets_mapping=True, is_split_into_words=True, truncation=True, padding=True)
    #     encoded_tokens = fix_tokens_with_offsets(tokens, encodings.offset_mapping, batch_indices)
    #     encoded_sentence_map = create_unmasked_sentence_map(encodings.offset_mapping, batch_indices)
    #     encoded_tokens, gold_starts, gold_ends, mentions = process_gold_mentions(encoded_tokens, gold_starts, gold_ends, mentions, encodings.attention_mask, batch_indices)
    #     cluster_ids = get_cluster_ids(mentions, clusters)

    #     dataset = ECBDataset(
    #         encodings=encodings, 
    #         batch_indices=batch_indices,
    #         sentence_map=encoded_sentence_map,
    #         gold_starts=gold_starts,
    #         gold_ends=gold_ends,
    #         cluster_ids=cluster_ids)

    #     loader = DataLoader(
    #         dataset,
    #         batch_size=1,
    #         shuffle=False
    #     )

    #     model.load_state_dict(torch.load(args['model_path']))

    #     predict(loader, model, device, encodings.offset_mapping, all_sentences, batch_indices)