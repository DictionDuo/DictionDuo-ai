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

def extract_formants(audio_file_path, time_step=0.01, max_formants=5, max_freq=5500):
    try:
        sound = parselmouth.Sound(audio_file_path)
        formant = sound.to_formant_burg(time_step=time_step, max_number_of_formants=max_formants, maximum_formant=max_freq)
        times = np.linspace(formant.xmin, formant.xmax, int((formant.xmax - formant.xmin) / time_step) + 1)
        return np.array([
            (t, formant.get_value_at_time(1, t) or 0,
             formant.get_value_at_time(2, t) or 0,
             formant.get_value_at_time(3, t) or 0)
            for t in times
        ], dtype=np.float32)
    except Exception as e:
        print(f"Formant extraction error: {e}")
        return None

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