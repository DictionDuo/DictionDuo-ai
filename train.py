import os
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from conformer.model import Conformer
from dataset.phoneme_tensor_dataset import PhonemeTensorDataset
from utils.logger import setup_logger
from utils.config_loader import convert_config_to_namespace
from tqdm import tqdm
import Levenshtein
import json
import boto3

SPECIALS = {"<blank>", "<pad>", "<s>", "</s>", "<unk>", "_"}

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

def collapse_repeats(seq):
    """CTC-style repeat collapse: a a a -> a"""
    out, prev = [], None
    for s in seq:
        if s != prev:
            out.append(s)
        prev = s
    return out

def idx_to_phonemes(idx_seq, index2phoneme):
    """index 시퀀스 -> 음소 문자열 리스트"""
    return [index2phoneme.get(int(i), "<unk>") for i in idx_seq]

def clean_phonemes(ph_seq):
    """스페셜/빈 토큰 제거"""
    return [p for p in ph_seq if p and (p not in SPECIALS) and p.strip() != ""]

def train_one_epoch(model, loader, criterion, optimizer, device, logger):
    model.train()
    total_loss = 0.0
    valid_batches = 0

    for features, labels, input_lengths, label_lengths, metas in tqdm(loader, desc="Training"):
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
    for t in range(int(input_len)):
        curr = int(pred_tensor[t].item())
        if curr != blank and curr != prev:
            decoded.append(curr)
        prev = curr
    return decoded

def evaluate(model, loader, index2phoneme, device, logger, stage="Validation", blank_index=0, log_debug_n=2):
    model.eval()
    total_per = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, labels, input_lengths, label_lengths, metas in tqdm(loader, desc=f"Evaluating {stage}"):
            features, labels = features.to(device), labels.to(device)
            input_lengths, label_lengths = input_lengths.to(device), label_lengths.to(device)

            try:
                outputs, output_lengths = model(features, input_lengths)
                preds = outputs.argmax(dim=-1)
            except Exception as e:
                logger.error(f"[EVAL FORWARD ERROR] {e}")
                continue

            B = features.size(0)
            for i in range(B):
                try:
                    T_out = int(output_lengths[i].item())
                    pred_ids_raw = preds[i]  # [T]
                    pred_ids = greedy_ctc_decode(pred_ids_raw, T_out, blank=blank_index)
                    pred_ph = idx_to_phonemes(pred_ids, index2phoneme)
                    pred_ph = clean_phonemes(collapse_repeats(pred_ph))

                    ref_frame = labels[i].detach().cpu().tolist()
                    ref_noblank = [int(r) for r in ref_frame if int(r) != blank_index]
                    ref_seq_ids = collapse_repeats(ref_noblank)
                    ref_ph = idx_to_phonemes(ref_seq_ids, index2phoneme)
                    ref_ph = clean_phonemes(ref_ph)

                    per = calculate_per(pred_ph, ref_ph)
                    total_per += per
                    total_samples += 1

                    # 디버그 로그 (초기 n개)
                    if total_samples <= log_debug_n:
                        nonzero_cnt = int((labels[i] != blank_index).sum().item())
                        L_lab = int(label_lengths[i].item())
                        logger.info(
                            "[DBG] "
                            f"out_T={T_out} ref_nonzero={nonzero_cnt} label_lengths={L_lab} "
                            f"ref_seq_len={len(ref_seq_ids)} pred_seq_len={len(pred_ids)} "
                            f"PER={per:.2%} | REF={' '.join(ref_ph[:40])} | PRED={' '.join(pred_ph[:40])}"
                        )

                except Exception as e:
                    logger.error(f"[EVAL DECODE ERROR] Sample {i} | Error: {e}")
                    continue
                
    avg_per = total_per / total_samples if total_samples > 0 else 0
    logger.info(f"{stage} PER(label): {avg_per:.2%}")

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

    os.makedirs(args.model_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.model_dir, 'train.log'))
    logger.info("===== Training Started =====")
    logger.info(f"Epochs: {args.epochs}, LR: {args.learning_rate}, Batch: {args.batch_size}, Seed: {args.seed}")

    with open("utils/phoneme2index.json", encoding="utf-8") as f:
        phoneme2index = json.load(f)

    phoneme2index = {k: int(v) for k, v in phoneme2index.items()}
    index2phoneme = {v: k for k, v in phoneme2index.items()}

    os.makedirs("preprocessed", exist_ok=True)
    train_data = load_dataset_from_s3(args.train_dataset_path)
    val_data = load_dataset_from_s3(args.val_dataset_path)
    test_data = load_dataset_from_s3(args.test_dataset_path)

    train_dataset = PhonemeTensorDataset(train_data, meta_list=train_data["metas"])
    val_dataset = PhonemeTensorDataset(val_data, meta_list=val_data["metas"])
    test_dataset = PhonemeTensorDataset(test_data, meta_list=test_data["metas"])

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

    for epoch in range(args.epochs):
        logger.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
        _ = train_one_epoch(model, train_loader, criterion, optimizer, device, logger)

        if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.epochs:
            evaluate(model, val_loader, index2phoneme, device, logger, stage="Validation", blank_index=0)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"conformer_{now}.pt"
    model_path = os.path.join(args.model_dir, model_name)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    if getattr(args, 'upload_model', False):
        upload_to_s3(model_path, args.upload_bucket, args.upload_path)
        logger.info(f"Uploaded to s3://{args.upload_bucket}/{args.upload_path}")

    logger.info("===== Running Final Test Evaluation =====")
    evaluate(model, test_loader, index2phoneme, device, logger, stage="Test", blank_index=0)
    logger.info("===== Training + Evaluation Completed =====")

if __name__ == "__main__":
    main()