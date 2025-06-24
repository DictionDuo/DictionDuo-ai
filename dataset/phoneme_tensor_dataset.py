import torch
from torch.utils.data import Dataset

class PhonemeTensorDataset(Dataset):
    def __init__(self, data_dict, meta_dict_list=None):
        """
        data_dict: .pt로부터 로드된 사전 (mels, phonemes, phones_actual, errors, input_lengths, label_lengths, metas 포함)
        meta_dict_list: 각 샘플에 대응되는 메타 정보 (JSON 경로 등 포함)
        """
        self.mels = data_dict["mels"]
        self.phonemes = data_dict["phonemes"]            # 정답 (prompt 기반)
        self.phones_actual = data_dict["phones_actual"]   # 실제 발음 (phones 기반)
        self.errors = data_dict["errors"]
        self.input_lengths = data_dict["input_lengths"]
        self.label_lengths = data_dict["label_lengths"]
        self.metas = meta_dict_list if meta_dict_list is not None else [None] * len(self.mels)

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        return (
            self.mels[idx],
            self.phonemes[idx],
            self.phones_actual[idx],
            self.errors[idx],
            torch.tensor(self.input_lengths[idx], dtype=torch.long),
            torch.tensor(self.label_lengths[idx], dtype=torch.long),
            self.metas[idx]
        )