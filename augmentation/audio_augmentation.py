import os
import numpy as np
import librosa
import soundfile as sf
from preprocessing.pronunciation_dataloader import PronunciationDataset

def find_max_length(audio_files):
    """
    모든 오디오 파일에서 최대 길이(샘플 수)를 찾는 함수.

    Args:
        audio_files (list): 오디오 파일 경로 리스트.
    
    Returns:
        int: 가장 긴 오디오 파일의 샘플 수.
    """
    max_length = 0
    for file_path in audio_files:
        y, sr = librosa.load(file_path, sr=None)
        max_length = max(max_length, len(y))
    return max_length

def augment_audio(y, sr, augment_type):
    augmented_y = y.copy()
    
    if augment_type == "time_stretch":
        rate = np.random.uniform(0.8, 1.2)
        augmented_y = librosa.effects.time_stretch(augmented_y, rate)
    elif augment_type == "pitch_shift":
        n_steps = np.random.randint(-2, 3)
        augmented_y = librosa.effects.pitch_shift(augmented_y, sr, n_steps)
    elif augment_type == "add_noise":
        noise = np.random.normal(0, 0.01, len(augmented_y))
        augmented_y = augmented_y + noise

    return augmented_y

if __name__ == "__main__":
    input_dir = "your_input_wav_files"  # 원본 WAV 파일이 있는 디렉토리
    output_dir = "your_augmented_wav_files"  # 증강된 파일을 저장할 디렉토리

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 모든 오디오 파일에서 최대 길이 자동 계산
    audio_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".wav")]
    max_length = find_max_length(audio_files)
    print(f"자동 계산된 max_length: {max_length} samples")

    # PronunciationDataset 사용
    dataset = PronunciationDataset(audio_files=audio_files, max_length=max_length)
    
    for i in range(len(dataset)):
        mel_spectrogram, f0, energy, shimmer, formants, jitter_abs, jitter_rel = dataset[i]

        for augment_type in ["time_stretch", "pitch_shift", "add_noise"]:
            augmented_y = augment_audio(mel_spectrogram.numpy().flatten(), 16000, augment_type)
            output_file = os.path.join(output_dir, f"augmented_{i}_{augment_type}.wav")
            sf.write(output_file, augmented_y, 16000)
    
    print("데이터 증강 완료!")