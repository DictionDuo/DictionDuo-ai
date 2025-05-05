import os
import shutil
import random
from preprocessing.build_dataset import build_metadata_list

def split_and_move(wav_dir, json_dir, output_dir,
                   train_ratio=0.85, val_ratio=0.10, seed=42):
    """
    WAV, JSON 데이터를 85:10:5 비율로 나눈 뒤, train/val/test 디렉토리로 이동합니다.

    Args:
        wav_dir (str): 원본 WAV 폴더 경로
        json_dir (str): 원본 JSON 폴더 경로
        output_dir (str): 결과를 저장할 최상위 폴더 경로
        train_ratio (float): 학습 데이터 비율
        val_ratio (float): 검증 데이터 비율
        seed (int): 셔플을 위한 랜덤 시드

    Returns:
        dict: 각 split의 파일 개수
    """
    metadata_list = build_metadata_list(wav_dir, json_dir)
    random.seed(seed)
    random.shuffle(metadata_list)

    total = len(metadata_list)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_list = metadata_list[:train_end]
    val_list = metadata_list[train_end:val_end]
    test_list = metadata_list[val_end:]

    split_map = {
        'train': train_list,
        'val': val_list,
        'test': test_list
    }

    for split, items in split_map.items():
        for subdir in ['wav', 'json']:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)

        for item in items:
            wav_dst = os.path.join(output_dir, split, 'wav', os.path.basename(item["wav"]))
            json_dst = os.path.join(output_dir, split, 'json', os.path.basename(item["json"]))
            shutil.move(item["wav"], wav_dst)
            shutil.move(item["json"], json_dst)

    return {
        'train': len(train_list),
        'val': len(val_list),
        'test': len(test_list)
    }