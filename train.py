import os
import torch
import torch.nn as nn
from datetime import datetime
from conformer.model import Conformer
from dataset.phoneme_dataset import PhonemeDataset
from preprocessing.split_dataset import build_and_split
from utils.phoneme_utils import phoneme2index
from utils.seed import set_seed, get_data_loader
from utils.logger import setup_logger
from utils.config_loader import convert_config_to_namespace
from tqdm import tqdm
import Levenshtein
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

def upload_to_s3(file_path, bucket_name, upload_path):
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket_name, upload_path)

def calculate_per(pred_seq, label_seq):
    pred_str = ' '.join(pred_seq)
    label_str = ' '.join(label_seq)
    distance = Levenshtein.distance(pred_str, label_str)
    return distance / max(len(label_seq), 1)

def train_one_epoch(model, loader, criterion, optimizer, device, logger):
    model.train()
    total_loss = 0.0

    for features, labels, input_lengths, label_lengths in tqdm(loader, desc="Training"):
        features, labels = features.to(device), labels.to(device)
        input_lengths, label_lengths = input_lengths.to(device), label_lengths.to(device)

        outputs, output_lengths = model(features, input_lengths)
        loss = criterion(outputs.transpose(0, 1).contiguous(), labels, output_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    logger.info(f"Train Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, loader, phoneme2index, device, logger, stage="Validation"):
    model.eval()
    total_per = 0.0
    total_samples = 0
    index2phoneme = {v: k for k, v in phoneme2index.items()}

    with torch.no_grad():
        for features, labels, input_lengths, label_lengths in tqdm(loader, desc=f"Evaluating {stage}"):
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
    logger.info(f"{stage} PER: {avg_per:.2%}")
    return avg_per

def main():
    args = convert_config_to_namespace("train_config.json")
    set_seed(args.seed)

    logger = setup_logger(os.path.join(args.model_dir, 'train.log'))
    logger.info("===== Training Started =====")
    logger.info(f"Epochs: {args.epochs}, LR: {args.learning_rate}, Batch: {args.batch_size}, Seed: {args.seed}")

    if getattr(args, "download", False):
        download_from_s3(args.bucket_name, args.s3_folder, args.train_data)

    wav_dir = os.path.join(args.train_data, 'wav')
    json_dir = os.path.join(args.train_data, 'json')
    train_list, val_list, test_list = build_and_split(wav_dir, json_dir)

    train_dataset = PhonemeDataset(train_list, phoneme2index)
    val_dataset = PhonemeDataset(val_list, phoneme2index)
    test_dataset = PhonemeDataset(test_list, phoneme2index)

    train_loader = get_data_loader(train_dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = get_data_loader(val_dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    test_loader = get_data_loader(test_dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed)

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
        logger.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, logger)
        avg_per = evaluate(model, val_loader, phoneme2index, device, logger, stage="Validation")

    os.makedirs(args.model_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"conformer_{now}.pt"
    model_path = os.path.join(args.model_dir, model_name)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    if getattr(args, 'upload_model', False):
        upload_to_s3(model_path, args.upload_bucket, args.upload_path)
        logger.info(f"Uploaded to s3://{args.upload_bucket}/{args.upload_path}")

    logger.info("===== Running Final Test Evaluation =====")
    evaluate(model, test_loader, phoneme2index, device, logger, stage="Test")
    logger.info("===== Training + Evaluation Completed =====")

if __name__ == "__main__":
    main()