import torch
from torch.utils.data import Dataset

class PhonemeTensorDataset(Dataset):
    def __init__(self, data_dict):
        self.mels = data_dict["mels"]
        self.labels = data_dict["labels"]
        self.input_lengths = data_dict["input_lengths"]
        self.label_lengths = data_dict["label_lengths"]

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.mels[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.input_lengths[idx], dtype=torch.long),
            torch.tensor(self.label_lengths[idx], dtype=torch.long),
        )