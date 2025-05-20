import torch
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=80
).to(device)

db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80).to(device)

def extract_features(wav_path: str) -> torch.Tensor:
    try:
        waveform, sr = torchaudio.load(wav_path)  # waveform: [1, T]
        if sr != 16000:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resample(waveform)
        
        waveform = waveform.to(device)
        mel = mel_transform(waveform)         # [1, 80, T]
        mel_db = db_transform(mel)            # [1, 80, T]
        mel_db = mel_db.squeeze(0).transpose(0, 1).cpu()  # [T, 80]

        if not torch.isfinite(mel_db).all():
            print(f"[extract_features] Skipped non-finite values: {wav_path}")
            return None
        
        return mel_db

    except Exception as e:
        print(f"[extract_features ERROR] {wav_path} - {e}")
        return None