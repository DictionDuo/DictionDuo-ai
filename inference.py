import os
import io
import json
import torch
import torchaudio
import hgtk

from conformer.model import Conformer
from preprocessing.frame_utils import pad_or_truncate_feature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=80
).to(device)

db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80).to(device)

WIN_SIZE = 256
SAMPLING_RATE = 16000

def model_fn(model_dir):
    # Load phoneme2index.json
    phoneme_path = os.path.join(model_dir, "phoneme2index.json")
    with open(phoneme_path, "r", encoding="utf-8") as f:
        phoneme2index = json.load(f)
    index2phoneme = {v: k for k, v in phoneme2index.items()}

    # Load model
    model_path = os.path.join(model_dir, "model.pt")

    model = Conformer(
        input_dim=80,
        num_classes=len(phoneme2index),
        encoder_dim=128,
        num_encoder_layers=2,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Attach phoneme dicts to model
    model.phoneme2index = phoneme2index
    model.index2phoneme = index2phoneme

    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "audio/wav":
        waveform, sr = torchaudio.load(io.BytesIO(request_body))
        return waveform, sr
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")
    
# 추론용 전처리 함수
def extract_features_from_waveform(waveform: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    if sr != SAMPLING_RATE:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLING_RATE)
        waveform = resample(waveform)

    waveform = waveform.to(device)
    mel = mel_transform(waveform)
    mel_db = db_transform(mel)
    mel_db = mel_db.squeeze(0).transpose(0, 1).cpu()  # [T, 80]

    if not torch.isfinite(mel_db).all():
        return None

    return mel_db

def phonemes_to_text(phoneme_seq):
    result = ""
    syllable = []

    for p in phoneme_seq:
        syllable.append(p)
        if len(syllable) == 3 or (len(syllable) == 2 and syllable[1] not in [None, ""]):
            try:
                result += hgtk.letter.compose(*syllable)
            except:
                result += ''.join(syllable)
            syllable = []

    if syllable:
        result += ''.join(syllable)
    return result

def predict_fn(input_data, model):
    waveform, sr = input_data
    mel = extract_features_from_waveform(waveform, sr)
    if mel is None:
        raise ValueError("Invalid mel features from input audio.")

    if mel.size(0) < WIN_SIZE:
        mel_np = pad_or_truncate_feature(mel.numpy(), WIN_SIZE, fill_value=0)
        mel_tensor = torch.tensor(mel_np, dtype=torch.float32).unsqueeze(0)  # [1, T, 80]
        input_length = torch.tensor([mel_tensor.size(1)])

        with torch.no_grad():
            outputs, output_lengths = model(mel_tensor, input_length)
            pred_ids = outputs.argmax(dim=-1).squeeze(0).tolist()

    else:
        pred_ids = []
        for start in range(0, len(mel) - WIN_SIZE + 1, WIN_SIZE):
            mel_chunk = mel[start:start + WIN_SIZE].numpy()
            mel_tensor = torch.tensor(mel_chunk, dtype=torch.float32).unsqueeze(0)
            input_length = torch.tensor([mel_tensor.size(1)])

            with torch.no_grad():
                outputs, output_lengths = model(mel_tensor, input_length)
                chunk_ids = outputs.argmax(dim=-1).squeeze(0).tolist()            

            # CTC Greedy decoding
            prev = None
            for idx in chunk_ids:
                if idx != 0 and idx != prev:
                    pred_ids.append(idx)
                prev = idx

    phoneme_seq = [model.index2phoneme.get(idx, "_") for idx in pred_ids]
    reconstructed_text = phonemes_to_text(phoneme_seq)

    return {"reconstructed_text": reconstructed_text}

def output_fn(prediction, accept):
    return json.dumps(prediction, ensure_ascii=False)