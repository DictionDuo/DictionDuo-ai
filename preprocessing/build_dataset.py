import glob
import os
import json

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