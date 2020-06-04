import json
from torch.utils.data import Dataset


class SummaryDataset(Dataset):
    
    def __init__(self, path):
        
        with open(path, 'r', encoding='utf8') as f:
            self.data = [json.loads(line) for line in f]
        
    def __len__(self):
        """Returns the number of data."""
        return len(self.data)
    
    def __getitem__(self, idx):
        sentences = self.data[idx]['doc'].split('\n')
        labels = self.data[idx]['labels'].split('\n')
        labels = [int(label) for label in labels]
        
        return sentences, labels