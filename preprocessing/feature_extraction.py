import numpy as np
import librosa
import torchaudio

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
        # 사전 검사: 손상된 wav 미리 차단
        torchaudio.load(wav_path)

        y, _ = librosa.load(wav_path, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max).T
        return mel_db.astype(np.float32)
    except Exception as e:
        print(f"[extract_features ERROR] {wav_path}: {e}")
        return None