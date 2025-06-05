import numpy as np
from .frame_utils import seconds_to_frames

def create_phoneme_label(phones, max_frames, phoneme_dict, sr=16000, hop_length=160):
    label = np.zeros(max_frames, dtype=np.int64)
    for phone in phones:
        start = max(0, seconds_to_frames(phone["start"], sr, hop_length))
        end = min(max_frames, seconds_to_frames(phone["end"], sr, hop_length))
        idx = phoneme_dict.get(phone["phone"], 0)
        label[start:end] = idx
    return label

def create_error_label(errors, max_frames, error_dict, sr=16000, hop_length=160):
    label = np.zeros(max_frames, dtype=np.int64)
    for error in errors:
        start = max(0, seconds_to_frames(error["start"], sr, hop_length))
        end = min(max_frames, seconds_to_frames(error["end"], sr, hop_length))
        idx = error_dict.get(error.get("error_tag"), 0)
        label[start:end] = idx
    return label