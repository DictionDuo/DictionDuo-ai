import feature_extraction as fx
import numpy as np
import pandas as pd

def create_feature_dataframe(wav_path):
    """
    - wav_path: 전처리된 WAV 파일 경로
    """
    y, sr = fx.load_audio(wav_path)
    if y is None or sr is None:
        raise ValueError("오디오 파일 로드 실패")

    # Mel Spectrogram 추출
    mel_spectrogram = fx.extract_mel_spectrogram(y, sr)
    mel_frames = mel_spectrogram.shape[1]  # Mel Spectrogram 프레임 개수

    # F0 추출 (frame_period=10.0ms 설정)
    f0, time_stamps, valid_indices = fx.extract_f0(y, sr, frame_period=10.0)

    # F0_mean, F0_median 계산
    valid_f0 = f0[valid_indices] if f0 is not None and valid_indices.any() else np.array([])
    F0_mean = np.mean(valid_f0) if valid_f0.size > 0 else 0
    F0_median = np.median(valid_f0) if valid_f0.size > 0 else 0

    # 추가 피처 추출
    shimmer = fx.extract_shimmer(y, sr, time_stamps, valid_indices)
    F1, F2, F3 = fx.extract_formants(y, sr)
    jitter_absolute, jitter_relative = fx.extract_jitter(f0, valid_indices)

    # Mel Spectrogram 프레임 수에 맞춰 복제
    df = pd.DataFrame({
        "F0_mean": np.repeat(F0_mean, mel_frames),
        "F0_median": np.repeat(F0_median, mel_frames),
        "shimmer": np.repeat(shimmer, mel_frames),
        "F1": np.repeat(F1, mel_frames),
        "F2": np.repeat(F2, mel_frames),
        "F3": np.repeat(F3, mel_frames),
        "jitter_absolute": np.repeat(jitter_absolute, mel_frames),
        "jitter_relative": np.repeat(jitter_relative, mel_frames),
    })

    # Mel Spectrogram을 데이터프레임에 추가
    mel_spectrogram_df = pd.DataFrame(mel_spectrogram.T)
    df = pd.concat([df, mel_spectrogram_df], axis=1)

    return df