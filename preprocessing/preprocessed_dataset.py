import os
import json
import torch
import random
import torchaudio
import numpy as np
from tqdm import tqdm
from preprocessing.feature_extraction import extract_features
from preprocessing.build_dataset import build_metadata_list
from preprocessing.split_dataset import split_metadata
from preprocessing.frame_utils import pad_or_truncate_feature
from utils.phoneme_utils import Korean

WIN_SIZE = 80
STRIDE = 40
HOP_LENGTH = 160
SAMPLING_RATE = 16000

SUBSAMPLING = 4
BLANK_IDX = 0
MAX_LABEL_LEN = 32
DROP_IF_EXCEED = True

FRAME_SEC = HOP_LENGTH / SAMPLING_RATE  # 프레임당 시간(초)

def is_valid_wav(wav_path):
    try:
        torchaudio.load(wav_path)
        return True
    except Exception as e:
        print(f"[Invalid WAV] {wav_path} - {e}")
        return False

def phones_to_seq_in_window(phones, f_start, f_end, phoneme2index, use_midpoint=True, max_len=MAX_LABEL_LEN):
    t0 = f_start * FRAME_SEC
    t1 = f_end * FRAME_SEC

    seq = []
    for ph in phones:
        st = float(ph["start"])
        ed = float(ph["end"])
        if use_midpoint:
            mid = 0.5 * (st + ed)
            include = (t0 <= mid < t1)
        else:
            # 겹침 기준을 쓰고 싶으면 이 조건 사용
            include = not (ed <= t0 or st >= t1)

        if include:
            idx = phoneme2index.get(str(ph["phone"]), BLANK_IDX)
            seq.append(idx)

    # 길이 컷 & 패딩
    if len(seq) > max_len:
        seq = seq[:max_len]
    seq_len = len(seq)
    if seq_len < max_len:
        seq += [BLANK_IDX] * (max_len - seq_len)

    return seq, seq_len

def iter_windows(total_frames, win_size=WIN_SIZE, stride=STRIDE, include_tail=False):
    """
    전체 프레임 길이(total_frames)에 대해 (f_start, f_end) 윈도우 구간을 생성.
    include_tail=True이면 마지막 남은 꼬리 구간도 패딩해서 하나 더 생성.
    """
    if total_frames <= win_size:
        yield 0, min(win_size, total_frames)
        return

    last_full_start = total_frames - win_size
    for f_start in range(0, last_full_start + 1, stride):
        yield f_start, f_start + win_size

    if include_tail and (last_full_start + stride < total_frames):
        # 꼬리 구간: 끝쪽을 WIN_SIZE로 맞춰서 한 구간 더
        yield total_frames - win_size, total_frames

def build_tensor_dataset(split_list, split_name, output_dir):
    skipped = []
    mel_list = []
    phoneme_list = []
    lengths = []
    label_lengths = []
    meta_list = []

    with open("utils/error_class_map.json", encoding="utf-8") as f:
        error_map = json.load(f)

    with open("utils/phoneme2index.json", encoding="utf-8") as f:
        phoneme2index = json.load(f)
        phoneme2index = {k: int(v) for k, v in phoneme2index.items()}

    korean = Korean()

    for meta in tqdm(split_list, desc=f"Processing {split_name}"):
        try:
            # 손상된 WAV 파일 사전 필터링
            if not is_valid_wav(meta["wav"]):
                print(f"[SKIP] Invalid WAV: {meta['wav']}")
                skipped.append(meta)
                continue    

            mel = extract_features(meta["wav"])
            # 추출 실패 또는 NaN/Inf 포함된 경우 필터링
            if mel is None or not torch.isfinite(mel).all():
                print(f"[SKIP] Corrupted features: {meta['wav']}")
                skipped.append(meta)
                continue

            mel_np = mel.cpu().numpy()

            with open(meta["json"], encoding="utf-8") as f:
                meta_json = json.load(f)

            try:
                phones = meta_json["RecordingMetadata"]["phonemic"]["phones"]
            except KeyError:
                print(f"[SKIP] Missing phonemic/phones: {meta['json']}")
                skipped.append(meta)
                continue

            T = mel_np.shape[0]
            if T <= 0:
                print(f"[SKIP] Empty feature: {meta['wav']}")
                skipped.append(meta)
                continue

            for f_start, f_end in iter_windows(T, WIN_SIZE, STRIDE, include_tail=False):
                # 멜 윈도우: 길이가 부족할 경우 패딩해 고정 길이로 맞춤
                mel_chunk = mel_np[f_start:f_end]
                if mel_chunk.shape[0] < WIN_SIZE:
                    mel_chunk = pad_or_truncate_feature(mel_chunk, WIN_SIZE, fill_value=0)

                # 라벨(음소 시퀀스) 생성
                seq, seq_len = phones_to_seq_in_window(
                    phones, f_start, min(f_end, T), phoneme2index,
                    use_midpoint=True, max_len=MAX_LABEL_LEN
                )

                # 빈 라벨은 스킵 (학습 효율)
                if seq_len == 0:
                    continue

                est_out_len = max(1, WIN_SIZE // SUBSAMPLING)
                if DROP_IF_EXCEED and (seq_len > est_out_len):
                    # 이 윈도우는 모델 출력 타임스텝이 부족 → 드랍
                    continue

                mel_list.append(torch.tensor(mel_chunk, dtype=torch.float32))
                phoneme_list.append(torch.tensor(seq, dtype=torch.long))
                lengths.append(WIN_SIZE)
                label_lengths.append(int(seq_len))
                meta_list.append(meta["json"])

        except Exception as e:
            print(f"[Error] {meta['wav']} - {e}")
            skipped.append(meta)

    if mel_list:
        torch.save({
            "mels": torch.stack(mel_list),
            "phonemes": torch.stack(phoneme_list),
            "input_lengths": lengths,
            "label_lengths": label_lengths,
            "metas": meta_list
        }, os.path.join(output_dir, f"{split_name}_dataset.pt"))
        print(f"[Saved] ({len(mel_list)} samples)")
    
    if skipped:
        with open(os.path.join(output_dir, f"{split_name}_skipped.json"), "w", encoding="utf-8") as f:
            json.dump(skipped, f, indent=2, ensure_ascii=False)
        print(f"[Saved] {split_name}_skipped.json → {len(skipped)} skipped samples")

def main():
    wav_dir = "/home/ubuntu/data/wav"
    json_dir = "/home/ubuntu/data/json"
    output_dir = "/home/ubuntu/preprocessed"
    os.makedirs(output_dir, exist_ok=True)

    metadata_list = build_metadata_list(wav_dir, json_dir)
    random.shuffle(metadata_list)

    train, val, test = split_metadata(metadata_list)

    build_tensor_dataset(train, "train", output_dir)
    build_tensor_dataset(val, "val", output_dir)
    build_tensor_dataset(test, "test", output_dir)

if __name__ == "__main__":
    main()