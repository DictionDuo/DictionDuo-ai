import librosa
import numpy as np
import pyworld
import parselmouth

def extract_f0(y, sr, frame_period=5.0):
    try:
        f0, time_stamps = pyworld.dio(y, sr, frame_period=frame_period)
        f0 = pyworld.stonemask(y, f0, time_stamps, sr)
        valid_indices = f0 > 0  # 무성음 제거
        return f0, time_stamps, valid_indices
    except:
        return None, None, None

def extract_shimmer(y, sr, time_stamps, valid_indices):
    try:
        if not valid_indices.any():
            return 0.0  # 무성음만 있으면 0 반환

        rms_amplitude = np.array([
            np.sqrt(np.mean(y[max(0, int((t - 0.0025) * sr)):min(len(y), int((t + 0.0025) * sr))] ** 2))
            for t in time_stamps[valid_indices]
        ])
        
        if rms_amplitude.size < 2:
            return 0.0  # 데이터가 부족하면 0 반환

        shimmer_values = np.abs(np.diff(rms_amplitude)) / (rms_amplitude[:-1] + 1e-6)  # 0 나누기 방지
        return np.mean(shimmer_values) if shimmer_values.size > 0 else 0.0
    except:
        return 0.0

def extract_mel_spectrogram(y, sr):
    return librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=160, n_mels=128), ref=np.max)

def extract_formants(y, sr):
    try:
        sound = parselmouth.Sound(y)
        formant = sound.to_formant_burg()
        mid_time = np.median(np.linspace(0, formant.xmax, num=10))  # 중앙값 개선
        return (formant.get_value_at_time(i, mid_time) or 0 for i in range(1, 4))
    except:
        return 0, 0, 0

def extract_jitter(f0):
    try:
        f0_clean = f0[f0 > 0]
        periods = 1 / f0_clean
        jitter_absolute = np.mean(np.abs(np.diff(periods)))
        jitter_relative = (jitter_absolute / np.mean(periods)) * 100
        return jitter_absolute, jitter_relative
    except Exception as e:
        print(f"Jitter extraction error: {e}")
        return None, None

def extract_energy(y, sr, frame_length=2048, hop_length=512):
    return librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length).flatten()