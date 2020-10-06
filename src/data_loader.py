
import torch

from torch.utils.data import Dataset

from src.utils import *

class ECBDataset(Dataset):
    def __init__(self, encodings, batch_indices, sentence_map, gold_starts, gold_ends, cluster_ids):
        self.encodings = encodings
        # self.data = torch.randn(250, 1)
        self.batch_indices = batch_indices
        self.sentence_map = sentence_map
        self.gold_starts = gold_starts
        self.gold_ends = gold_ends
        self.cluster_ids = cluster_ids

    def __getitem__(self, index):
        start_idx = self.batch_indices[index]
        end_idx = self.batch_indices[index+1]
        item = {key: torch.tensor(val[start_idx:end_idx]) for key, val in self.encodings.items()}
        item['sentence_map'] = torch.tensor(self.sentence_map[index])
        item['gold_starts'] = self.gold_starts[index]
        item['gold_ends'] = self.gold_ends[index]
        item['cluster_ids'] = self.cluster_ids[index]
        
        return item
        
    def __len__(self):
        return len(self.batch_indices) - 1

if __name__ == "__main__":
    print(process_ecb_plus('data', 'HUMAN_PART_PER'))
    
    # dataset = MyDataset()
    # loader = DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=2
    # )

    # for data in loader:
    #     data = data.view(-1, 1)
    #     print(data.shape)