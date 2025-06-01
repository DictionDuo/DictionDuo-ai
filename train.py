import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.use_deterministic_algorithms(True)

import torch.nn as nn
from datetime import datetime
from conformer.model import Conformer
from dataset.phoneme_tensor_dataset import PhonemeTensorDataset
from utils.phoneme_utils import phoneme2index
from utils.seed import set_seed, get_data_loader
from utils.logger import setup_logger
from utils.config_loader import convert_config_to_namespace
from tqdm import tqdm
import Levenshtein
import json
import boto3

def download_from_s3(bucket_name, s3_key, local_path):
    s3 = boto3.client('s3')
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket_name, s3_key, local_path)

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

def evaluate(model, loader, index2phoneme, device, logger, stage="Validation"):
    model.eval()
    total_per = 0.0
    total_samples = 0

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

def load_dataset_from_s3(s3_path):
    bucket = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:])
    filename = os.path.basename(key)
    local_path = os.path.join("preprocessed", filename)
    download_from_s3(bucket, key, local_path)
    return torch.load(local_path)

def main():
    args = convert_config_to_namespace("train_config.json")
    set_seed(args.seed)

    logger = setup_logger(os.path.join(args.model_dir, 'train.log'))
    logger.info("===== Training Started =====")
    logger.info(f"Epochs: {args.epochs}, LR: {args.learning_rate}, Batch: {args.batch_size}, Seed: {args.seed}")

    os.makedirs("preprocessed", exist_ok=True)
    train_data = load_dataset_from_s3(args.train_dataset_path)
    val_data = load_dataset_from_s3(args.val_dataset_path)
    test_data = load_dataset_from_s3(args.test_dataset_path)

    train_dataset = PhonemeTensorDataset(train_data)
    val_dataset = PhonemeTensorDataset(val_data)
    test_dataset = PhonemeTensorDataset(test_data)

    train_loader = get_data_loader(train_dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = get_data_loader(val_dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    test_loader = get_data_loader(test_dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Using GPU device: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("No GPU device available.")
        
    model = Conformer(
        input_dim=80,
        num_classes=len(phoneme2index),
        encoder_dim=256,
        num_encoder_layers=4,
    ).to(device)

    criterion = nn.CTCLoss(blank=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    index2phoneme = {v: k for k, v in phoneme2index.items()}

    for epoch in range(args.epochs):
        logger.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, logger)

        if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.epochs:
            avg_per = evaluate(model, val_loader, index2phoneme, device, logger, stage="Validation")

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
    evaluate(model, test_loader, index2phoneme, device, logger, stage="Test")
    logger.info("===== Training + Evaluation Completed =====")

if __name__ == "__main__":
    main()