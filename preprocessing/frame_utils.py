import numpy as np

def pad_or_truncate_feature(feature, max_frames, fill_value=0):
    feature = np.array(feature)
    if len(feature) > max_frames:
        return feature[:max_frames]
    elif len(feature) < max_frames:
        pad_shape = (max_frames - len(feature), *feature.shape[1:]) if feature.ndim > 1 else (max_frames - len(feature),)
        padding = np.full(pad_shape, fill_value)
        return np.concatenate([feature, padding], axis=0)
    return feature

def seconds_to_frames(seconds, sr=16000, hop_length=160):
    return int(seconds * sr / hop_length)