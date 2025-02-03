import librosa
import librosa.display
import numpy as np
import pyworld
import parselmouth
import matplotlib.pyplot as plt

def load_audio(audio_file_path):
    """Load an audio file and return waveform (y) and sample rate (sr)"""
    try:
        y, sr = librosa.load(audio_file_path, sr=None)
        return y.astype(np.float64), sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None

def extract_f0(y, sr, frame_period=5.0):
    """Extract F0 and timestamps using pyworld"""
    f0, time_stamps = pyworld.dio(y, sr, frame_period=frame_period)
    f0 = pyworld.stonemask(y, f0, time_stamps, sr)
    return f0, time_stamps

def calculate_shimmer(y, sr, f0, time_stamps, frame_period=5.0):
    try:
        valid_indices = f0 > 0
        rms_amplitude = np.array([
            np.sqrt(np.mean(y[int(max(0, (t - frame_period / 2000) * sr)):
                               int(min(len(y), (t + frame_period / 2000) * sr))] ** 2))
            for t in time_stamps[valid_indices]
        ])
        threshold = 0.001
        shimmer_values = np.abs(np.diff(rms_amplitude)) / rms_amplitude[:-1]
        return np.mean(shimmer_values[shimmer_values > threshold]) if len(shimmer_values) > 0 else 0.0
    except Exception as e:
        print(f"Shimmer error: {e}")
        return None

def extract_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)

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