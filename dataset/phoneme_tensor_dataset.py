import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any

class PhonemeTensorDataset(Dataset):
    def __init__(self, data_dict: Dict[str, Any], meta_list: Optional[List[str]] = None):
        for k in ["mels", "phonemes", "input_lengths", "label_lengths"]:
            if k not in data_dict:
                raise KeyError(f"Missing key in data_dict: {k}")
            
        self.mels = data_dict["mels"]
        self.phonemes = data_dict["phonemes"].long()
        self.input_lengths = data_dict["input_lengths"]
        self.label_lengths = data_dict["label_lengths"]
        self.metas = meta_list if meta_list is not None else data_dict.get("metas", [None] * len(self.mels))

    def __len__(self) -> int:
        return len(self.mels)

    def __getitem__(self, idx: int):
        return (
            self.mels[idx],
            self.phonemes[idx],
            torch.tensor(self.input_lengths[idx], dtype=torch.long),
            torch.tensor(self.label_lengths[idx], dtype=torch.long),
            self.metas[idx]
        )