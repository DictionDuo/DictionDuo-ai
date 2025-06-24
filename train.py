import os
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from conformer.model import Conformer
from dataset.phoneme_tensor_dataset import PhonemeTensorDataset
from utils.phoneme_utils import Korean, phoneme2index
from utils.logger import setup_logger
from utils.config_loader import convert_config_to_namespace
from tqdm import tqdm
import Levenshtein
import json
import boto3
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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
    valid_batches = 0

    for features, labels, phones_actual, errors, input_lengths, label_lengths, metas in tqdm(loader, desc="Training"):
        features, labels = features.to(device), labels.to(device)
        input_lengths, label_lengths = input_lengths.to(device), label_lengths.to(device)

        # [1] NaN / Inf 체크
        if torch.isnan(features).any() or torch.isinf(features).any():
            logger.error("[MEL ERROR] NaN or Inf in features")
            continue
        if torch.isnan(labels).any() or torch.isinf(labels).any():
            logger.error("[LABEL ERROR] NaN or Inf in labels")
            continue
        if (input_lengths <= 0).any() or (label_lengths <= 0).any():
            logger.error("[LENGTH ERROR] Non-positive input/label lengths")
            continue

        # [2] 값이 너무 큰 경우 체크
        if (features.abs() > 1e4).any():
            logger.warning("[MEL WARNING] Extremely large values in features")

        # [3] Forward
        try:
            outputs, output_lengths = model(features, input_lengths)
        except Exception as e:
            logger.error(f"[FORWARD ERROR] {e}")
            continue

        # [4] Loss 계산
        try:
            loss = criterion(outputs.transpose(0, 1).contiguous(), labels, output_lengths, label_lengths)
        except Exception as e:
            logger.error(f"[LOSS COMPUTE ERROR] {e}")
            continue

        # [5] NaN or inf loss 체크
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error("[LOSS ERROR] NaN or Inf in loss!")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        valid_batches += 1

    avg_loss = total_loss / valid_batches if valid_batches > 0 else float("inf")
    logger.info(f"Train Loss: {avg_loss:.4f}")
    return avg_loss

def greedy_ctc_decode(pred_tensor, input_len, blank=0):
    decoded = []
    prev = None
    for i in range(input_len):
        curr = pred_tensor[i].item()
        if curr != blank and curr != prev:
            decoded.append(curr)
        prev = curr
    return decoded

def evaluate(model, loader, index2phoneme, phoneme2index, device, logger, stage="Validation"):
    model.eval()
    total_per_label = 0.0
    total_per_actual = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    korean = Korean()

    with torch.no_grad():
        for features, labels, phones_actual, errors, input_lengths, label_lengths, metas in tqdm(loader, desc=f"Evaluating {stage}"):
            features, labels, phones_actual = features.to(device), labels.to(device), phones_actual.to(device)
            input_lengths, label_lengths = input_lengths.to(device), label_lengths.to(device)

            try:
                outputs, output_lengths = model(features, input_lengths)
                preds = outputs.argmax(dim=-1)
            except Exception as e:
                logger.error(f"[EVAL FORWARD ERROR] {e}")
                continue

            for i in range(features.size(0)):
                try:
                    with open(metas[i]["json"], encoding="utf-8") as f:
                        meta_json = json.load(f)
                    prompt_text = meta_json["RecordingMetadata"]["prompt"]
                    target_ids = korean.text_to_phoneme_sequence(prompt_text, phoneme2index)

                    pred_ids = greedy_ctc_decode(preds[i], output_lengths[i], blank=0)

                    pred_seq = [index2phoneme[idx] for idx in pred_ids if idx in index2phoneme]
                    label_seq = [index2phoneme[idx] for idx in target_ids if idx in index2phoneme]
                    actual_seq = [index2phoneme[idx.item()] for idx in phones_actual[i][:input_lengths[i]] if idx.item() in index2phoneme]

                    per_label = calculate_per(pred_seq, label_seq)
                    per_actual = calculate_per(pred_seq, actual_seq)

                    total_per_label += per_label
                    total_per_actual += per_actual
                    total_samples += 1

                    all_preds.extend(pred_ids)
                    all_labels.extend(target_ids)

                    logger.debug(f"Sample {i} | Pred: {''.join(pred_seq)} | Label: {''.join(label_seq)} | Actual: {''.join(actual_seq)} | PER(label): {per_label:.2%}, PER(actual): {per_actual:.2%}")

                except Exception as e:
                    logger.error(f"[EVAL DECODE ERROR] {e}")
                    continue
                
    avg_per_label = total_per_label / total_samples if total_samples > 0 else 0
    avg_per_actual = total_per_actual / total_samples if total_samples > 0 else 0
    logger.info(f"{stage} PER(label): {avg_per_label:.2%}, PER(actual): {avg_per_actual:.2%}")

    cm = confusion_matrix(all_labels, all_preds, labels=sorted(index2phoneme.keys()))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, xticklabels=index2phoneme.values(), yticklabels=index2phoneme.values(), cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix ({stage})')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{stage.lower()}.png")
    logger.info(f"Confusion matrix saved to confusion_matrix_{stage.lower()}.png")

    return avg_per_label, avg_per_actual

def load_dataset_from_s3(s3_path):
    bucket = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:])
    filename = os.path.basename(key)
    local_path = os.path.join("preprocessed", filename)
    download_from_s3(bucket, key, local_path)
    return torch.load(local_path)

def main():
    args = convert_config_to_namespace("train_config.json")

    logger = setup_logger(os.path.join(args.model_dir, 'train.log'))
    logger.info("===== Training Started =====")
    logger.info(f"Epochs: {args.epochs}, LR: {args.learning_rate}, Batch: {args.batch_size}, Seed: {args.seed}")

    os.makedirs("preprocessed", exist_ok=True)
    train_data = load_dataset_from_s3(args.train_dataset_path)
    val_data = load_dataset_from_s3(args.val_dataset_path)
    test_data = load_dataset_from_s3(args.test_dataset_path)

    train_dataset = PhonemeTensorDataset(train_data, meta_dict_list=train_data["metas"])
    val_dataset = PhonemeTensorDataset(val_data, meta_dict_list=val_data["metas"])
    test_dataset = PhonemeTensorDataset(test_data, meta_dict_list=test_data["metas"])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Using GPU device: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("No GPU device available.")
        
    model = Conformer(
        input_dim=80,
        num_classes=len(phoneme2index),
        encoder_dim=128,
        num_encoder_layers=2,
    ).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    index2phoneme = {v: k for k, v in phoneme2index.items()}

    for epoch in range(args.epochs):
        logger.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, logger)

        if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.epochs:
            evaluate(model, val_loader, index2phoneme, phoneme2index, device, logger, stage="Validation")

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
    evaluate(model, test_loader, index2phoneme, phoneme2index, device, logger, stage="Test")
    logger.info("===== Training + Evaluation Completed =====")

if __name__ == "__main__":
    main()