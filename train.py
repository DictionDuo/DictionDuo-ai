import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from conformer.model import Conformer
from dataset.phoneme_dataset import PhonemeDataset
from preprocessing.build_dataset import build_metadata_list, get_max_lengths
from utils.phoneme_utils import phoneme2index
from tqdm import tqdm
from difflib import SequenceMatcher
import json
import boto3

def download_from_s3(bucket_name, s3_folder, local_dir):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        for content in result.get('Contents', []):
            key = content['Key']
            if key.endswith('/'):
                continue
            local_file_path = os.path.join(local_dir, os.path.relpath(key, s3_folder))
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            s3.download_file(bucket_name, key, local_file_path)

def calculate_per(pred_seq, label_seq):
    matcher = SequenceMatcher(None, pred_seq, label_seq)
    matches = sum(triple.size for triple in matcher.get_matching_blocks())
    total = len(label_seq)
    errors = total - matches
    return errors / total if total > 0 else 0

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for features, labels in tqdm(loader, desc="Training"):
        input_lengths = torch.full((features.size(0),), features.size(1), dtype=torch.long)
        label_lengths = torch.sum(labels != 0, dim=1)

        features, labels = features.to(device), labels.to(device)
        input_lengths, label_lengths = input_lengths.to(device), label_lengths.to(device)

        outputs, output_lengths = model(features, input_lengths)
        loss = criterion(outputs.transpose(0, 1), labels, output_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss

def evaluate(model, loader, phoneme2index, device):
    model.eval()
    total_per = 0.0
    total_samples = 0
    index2phoneme = {v: k for k, v in phoneme2index.items()}

    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Evaluating"):
            input_lengths = torch.full((features.size(0),), features.size(1), dtype=torch.long)
            label_lengths = torch.sum(labels != 0, dim=1)

            features, labels = features.to(device), labels.to(device)
            input_lengths, label_lengths = input_lengths.to(device), label_lengths.to(device)

            outputs, output_lengths = model(features, input_lengths)
            preds = outputs.argmax(dim=-1)

            for i in range(features.size(0)):
                pred_seq = [index2phoneme[idx.item()] for idx in preds[i][:label_lengths[i]] if idx.item() != 0]
                label_seq = [index2phoneme[idx.item()] for idx in labels[i][:label_lengths[i]] if idx.item() != 0]

                per = calculate_per(pred_seq, label_seq)
                total_per += per
                total_samples += 1

    avg_per = total_per / total_samples if total_samples > 0 else 0
    return avg_per

def main(args):
    if args.download:
        download_from_s3(args.bucket_name, args.s3_folder, args.train_data)

    wav_dir = os.path.join(args.train_data, 'wav')
    json_dir = os.path.join(args.train_data, 'json')
    dataset = PhonemeDataset(wav_dir, json_dir, phoneme2index)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Conformer(
        input_dim=80,
        num_classes=len(phoneme2index),
        encoder_dim=256,
        num_encoder_layers=4,
    ).to(device)

    criterion = nn.CTCLoss(blank=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, loader, criterion, optimizer, device)
        avg_per = evaluate(model, loader, phoneme2index, device)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | PER: {avg_per*100:.2f}%")

    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'conformer.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--train_data', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--bucket_name', type=str, default='your-bucket-name')
    parser.add_argument('--s3_folder', type=str, default='your-s3-prefix/')
    args = parser.parse_args()

    main(args)