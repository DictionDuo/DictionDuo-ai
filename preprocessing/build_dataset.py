import glob
import os
import json
from preprocessing.feature_extraction import extract_features
from utils.phoneme_utils import Korean

def build_metadata_list(wav_dir, json_dir):
    metadata_list = []
    for wav_path in sorted(glob.glob(os.path.join(wav_dir, '*.wav'))):
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        json_path = os.path.join(json_dir, f"{basename}.json")
        if os.path.exists(json_path):
            metadata_list.append({"wav": wav_path, "json": json_path})
        else:
            print(f"[Warning] JSON not found for {basename}")
    return metadata_list

def get_max_lengths(metadata_list, phoneme2index):
    max_mel = 0
    max_label = 0
    for meta in metadata_list:
        mel = extract_features(meta["wav"])
        if mel is not None:
            max_mel = max(max_mel, mel.shape[0])
        with open(meta["json"], "r", encoding="utf-8") as f:
            text = json.load(f)["transcription"]["AnswerLabelText"]
        label = Korean.text_to_phoneme_sequence(text, phoneme2index)
        max_label = max(max_label, len(label))
    return max_mel, max_label