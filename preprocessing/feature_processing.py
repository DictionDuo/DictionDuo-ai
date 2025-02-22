import feature_extraction as fx
from preprocess_audio import process_audio
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def create_feature_dataframe(wav_path):
    """
    단일 WAV 파일에서 피처를 추출하고 DataFrame으로 변환하는 함수
    """
    # 전처리된 음성 데이터 로드
    y, sr = process_audio(wav_path)  # 16kHz로 변환된 신호 사용

    # Mel Spectrogram 추출
    mel_spectrogram = fx.extract_mel_spectrogram(y, sr)
    mel_frames = mel_spectrogram.shape[1]  # 프레임 개수는 mel_spectrogram.shape[1]

    # 다른 피처 추출
    f0, time_stamps, valid_indices = fx.extract_f0(y, sr)
    F0_mean = np.mean(f0[f0 > 0]) if f0 is not None else 0
    F0_median = np.median(f0[f0 > 0]) if f0 is not None else 0
    shimmer = fx.extract_shimmer(y, sr, time_stamps, valid_indices)
    F1, F2, F3 = fx.extract_formants(y, sr)
    jitter_absolute, jitter_relative = fx.extract_jitter(f0, valid_indices)

    # 보간 및 브로드캐스팅 (Mel Spectrogram 프레임 개수에 맞춤)
    original_frames = np.linspace(0, 1, len(y) // 160)  # 샘플링 레이트 반영한 프레임 계산
    target_frames = np.linspace(0, 1, mel_frames)  # Mel Spectrogram 프레임 개수 기준 보간

    def interpolate_feature(feature):
        return interp1d(original_frames, feature, kind='linear', axis=0)(target_frames) if len(feature) > 1 else np.full((mel_frames,), feature)

    # 개별 보간 처리
    shimmer_interp = interpolate_feature(np.array([shimmer]))
    f1_interp = interpolate_feature(np.array([F1]))
    f2_interp = interpolate_feature(np.array([F2]))
    f3_interp = interpolate_feature(np.array([F3]))
    jitter_abs_interp = interpolate_feature(np.array([jitter_absolute]))
    jitter_rel_interp = interpolate_feature(np.array([jitter_relative]))

    # DataFrame으로 변환
    df = pd.DataFrame({
        "F0_mean": np.tile(F0_mean, mel_frames),
        "F0_median": np.tile(F0_median, mel_frames),
        "shimmer": shimmer_interp,
        "F1": f1_interp,
        "F2": f2_interp,
        "F3": f3_interp,
        "jitter_absolute": jitter_abs_interp,
        "jitter_relative": jitter_rel_interp,
    })

    # Mel Spectrogram 추가
    mel_spectrogram_df = pd.DataFrame(mel_spectrogram.T)  # 전치(transpose)하여 프레임이 행이 되도록
    final_df = pd.concat([df, mel_spectrogram_df], axis=1)

    return final_df