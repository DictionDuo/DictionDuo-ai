import numpy as np
import librosa

def extract_features(wav_path: str, sr: int = 16000, n_mels: int = 80) -> np.ndarray:
    """
    Extract Mel-spectrogram from a WAV file.

    Args:
        wav_path (str): path to the WAV file.
        sr (int): sampling rate.
        n_mels (int): number of Mel filterbanks.

    Returns:
        np.ndarray: Transposed mel-spectrogram with shape (T, n_mels)
    """
    try:
        y, _ = librosa.load(wav_path, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max).T
        
        if not np.isfinite(mel_db).all():
            print(f"[extract_features] Skipped due to non-finite values: {wav_path}")
            return None

        return mel_db.astype(np.float32)

    except Exception as e:
        print(f"[Error] Failed to extract features from {wav_path}: {e}")
        return None