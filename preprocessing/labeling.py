import pandas as pd
from feature_processing import create_feature_dataframe

def create_labeled_dataframe(wav_path, label_value):
    """
    - wav_path: 단일 WAV 파일 경로
    - label_value: 해당 파일의 모든 데이터에 부여할 라벨 값 (0 또는 1)
    - Mel Spectrogram 프레임 개수만큼 동일한 label 값을 가진 데이터프레임 생성
    """
    df = create_feature_dataframe(wav_path)  # WAV 파일에서 피처 추출하여 데이터프레임 생성
    df["label"] = label_value  # 라벨 값 추가
    return df