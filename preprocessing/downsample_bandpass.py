import numpy as np
from scipy.signal import firwin, filtfilt
import librosa
import soundfile as sf
import os

# FIR Bandpass 필터 생성 및 적용 함수
def apply_fir_bandpass_filter(data, lowcut=80, highcut=4000, fs=16000, numtaps=101):
    taps = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)
    filtered_data = filtfilt(taps, [1.0], data)
    return filtered_data

# RMS 에너지 정규화 함수
def normalize_rms(y, target_rms=0.1):
    current_rms = np.sqrt(np.mean(y**2))
    if current_rms > 0:
        y *= target_rms / current_rms
    return y

# 리샘플링 + FIR 필터 + RMS 정규화 적용 함수
def process_audio(file_path, target_sr=16000, lowcut=80, highcut=4000, numtaps=101, target_rms=0.1):
    y, sr = librosa.load(file_path, sr=None)

    # 리샘플링
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # Bandpass 필터 적용
    filtered_y = apply_fir_bandpass_filter(y, lowcut, highcut, target_sr, numtaps)
    
    # RMS 정규화 적용
    normalized_y = normalize_rms(filtered_y, target_rms)

    return normalized_y, target_sr

# 폴더 내 모든 오디오 파일 처리 함수
def process_dataset(input_folder, output_folder, target_sr=16000, lowcut=80, highcut=4000, numtaps=101, target_rms=0.1):
    os.makedirs(output_folder, exist_ok=True)  # 폴더가 없으면 생성

    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # 리샘플링 + Bandpass 필터 + RMS 정규화 적용
            processed_audio, sr = process_audio(input_path, target_sr, lowcut, highcut, numtaps, target_rms)
            
            # 결과 저장
            sf.write(output_path, processed_audio, sr)
            print(f"Processed and saved: {output_path}")