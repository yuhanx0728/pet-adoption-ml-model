import pandas as pd
import hyperparameters as hyp
import torch.utils.data as data

class CSVDataset(data.Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def __getitem__(self, index):
        return self.data.iloc[index]

    def __len__(self):
        return len(self.data)