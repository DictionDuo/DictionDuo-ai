# DictionDuo-ai  
Real-Time Korean Pronunciation Correction System using Conformer

---

## Overview  
**DictionDuo** is an AI-powered mobile pronunciation correction system designed for international learners of Korean.  
It analyzes pronunciation in real time at the phoneme level using a Conformer-based model and delivers visual feedback through a connected mobile app.

This project presents a real-time Korean pronunciation correction system powered by a Conformer-based model.  
The system analyzes speech at the phoneme level and provides immediate feedback via text.  
It is designed for mobile use, enabling learners to record, analyze, and review their Korean pronunciation interactively.

---

## Features  

- Real-time Korean pronunciation analysis using phoneme-level feedback  
- Conformer-based acoustic model trained with Mel spectrogram and phoneme labels   
- Serverless cloud-based inference pipeline using AWS (SageMaker, Lambda, API Gateway)  
- Mobile application for recording, reviewing, and learning pronunciation history  

---

## Data  

**1. Source**  
- AI Hub: Korean speech dataset from English-native learners  
- 10,000 pairs of `.wav` and `.json` files  

**2. Preprocessing**  
- Audio resampled to 16kHz  
- Converted to 80-dimensional Mel spectrogram using `torchaudio`  
- Transcripts extracted from JSON `"prompt"` fields  
- Text cleaned and converted into phoneme sequences (초성, 중성, 종성)  
- Phoneme sequences mapped to integer indices  
- All inputs/labels normalized to 512 frames (zero-padding or truncation)

---

## Model  

**Architecture**: Conformer  
- Encoder Dim: 128  
- Encoder Layers: 2  
- Input: 80-dim Mel Spectrogram  
- Label: Phoneme index sequence  
- Loss: CTC Loss  
- Epochs: 20  
- Batch Size: 4  
- Framework: PyTorch  
- Training Environment: AWS SageMaker (ml.g4dn.2xlarge)  

**Hyperparameters**

<img width="400" alt="hyperparameters" src="https://github.com/user-attachments/assets/d23c3ace-8d00-4447-b02c-66078cc126c1">

**Training Summary**:  
- Train Loss: 2.4481 → 0.2698  
- Validation PER: 40.31% → 23.65%  
- Final Test PER: 24.33%  
- Training Time: ~22 minutes  

**Training Curves**  
<img width="600" alt="training_curves" src="https://github.com/user-attachments/assets/3b18cfe1-47e5-48b1-9e0d-d56b3c5b9c5b">

---

## Deployment  

- Real-time inference deployed via AWS serverless architecture  
- Pronunciation feedback integrated into Android app using RESTful API  
- In-app RoomDB stores audio, phoneme sequence, and feedback history  

---

## Applications  

**B2B**  
- SaaS service for EdTech companies (language platforms, tutoring services)  

**B2G**  
- Integration with public education systems (e.g., KIIP programs, multicultural families)  
- Collaboration with government institutions (e.g., National Institute of Korean Language)  

---

## Installation  

```bash
git clone https://github.com/DictionDuo/DictionDuo-ai.git
cd DictionDuo-ai
pip install -r requirements.txt
```