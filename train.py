# training loop + evaluation + checkpointing
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
Details:
	•	Student-level split (GroupKFold or fixed list)
	•	Train loop, val loop
	•	Macro F1, balanced acc, confusion matrix
	•	Save best model

"""

"""
train.py

Student-level 4-class classification using:
- BiGRU + additive attention pooling (model.py)
- Per-student cached sequences from preprocess.py (.npz files)
- GroupKFold (5-fold) split by student (client_id) to avoid leakage
- Early stopping on validation macro F1

Metrics each epoch (val):
- macro F1
- balanced accuracy
- confusion matrix

Outputs:
- Best checkpoint info written to:
  /data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/results/best_checkpoint.txt
- Per-student predictions across all folds written to:
  /data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/results/per_student_predictions.csv

Notes:
- This script trains 5 separate fold models. Each fold saves its own best .pt checkpoint.
- The "best_checkpoint.txt" records the overall best fold+epoch and path.

Running Example:
python train.py \
  --cache_dir "/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/preprocessed_output/ast_42" \
  --test_type ast \
  --out_dir "/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/results/trial5_ast" \
  --batch_size 16 \
  --epochs 100 \
  --patience 15 \
  --lr 3e-4 \
  --weight_decay 1e-2 \
  --num_workers 4 \
  --device cpu
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import DatasetOptions, StudentSequenceDataset, collate_student_batch, get_groups_and_labels
from model import ModelConfig, StudentBiGRUAttnClassifier


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic can reduce speed; keep it safe but not overly strict
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_device(batch: Dict, device: torch.device) -> Dict:
    batch["x"] = batch["x"].to(device, non_blocking=True)
    batch["y"] = batch["y"].to(device, non_blocking=True)
    batch["mask"] = batch["mask"].to(device, non_blocking=True)
    batch["lengths"] = batch["lengths"].to(device, non_blocking=True)
    # seg_start_s / seg_end_s are not used in the model; keep on CPU
    return batch


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    model.eval()
    all_y: List[int] = []
    all_pred: List[int] = []
    all_probs: List[np.ndarray] = []
    all_client_ids: List[str] = []

    for batch in loader:
        client_ids = batch["client_ids"]
        batch = to_device(batch, device)

        logits, _ = model(batch["x"], batch["mask"], return_attn=False)
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)

        all_y.extend(batch["y"].detach().cpu().numpy().astype(int).tolist())
        all_pred.extend(pred.detach().cpu().numpy().astype(int).tolist())
        all_probs.extend(probs.detach().cpu().numpy())
        all_client_ids.extend(client_ids)

    y_true = np.array(all_y, dtype=int)
    y_pred = np.array(all_pred, dtype=int)

    if y_true.size == 0:
        return {
            "macro_f1": 0.0,
            "balanced_acc": 0.0,
            "confusion": np.zeros((4, 4), dtype=int),
            "y_true": y_true,
            "y_pred": y_pred,
            "probs": np.array(all_probs) if len(all_probs) else np.zeros((0, 4), dtype=float),
            "client_ids": all_client_ids,
        }

    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).astype(int)

    return {
        "macro_f1": macro_f1,
        "balanced_acc": bal_acc,
        "confusion": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": np.array(all_probs, dtype=float),
        "client_ids": all_client_ids,
    }


def format_confusion(cm: np.ndarray) -> str:
    # simple pretty string
    lines = []
    for row in cm.tolist():
        lines.append("  " + " ".join(f"{v:4d}" for v in row))
    return "\n".join(lines)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    extra: Dict,
) -> None:
    ckpt = {
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "extra": extra,
    }
    torch.save(ckpt, path)


# -----------------------------
# Training
# -----------------------------
def train_one_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    dataset: StudentSequenceDataset,
    device: torch.device,
    args: argparse.Namespace,
    out_dir: Path,
) -> Tuple[Dict, Dict]:
    # Dataloaders
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_student_batch,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_student_batch,
    )

    # Model
    mcfg = ModelConfig(
        in_dim=444,
        proj_hidden=128,
        proj_out=64,
        dropout=args.dropout,
        gru_hidden=64,
        attn_hidden=64,
        num_classes=4,
        layer_norm=True,
    )
    model = StudentBiGRUAttnClassifier(mcfg).to(device)

    # Loss / Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Tracking
    best_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = out_dir / f"fold{fold}_best.pt"
    patience_counter = 0

    history = {
        "fold": fold,
        "epochs": [],
        "best": {},
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        t0 = time.time()
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(batch["x"], batch["mask"], return_attn=False)
            loss = criterion(logits, batch["y"])
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)

        # Validate
        val_res = evaluate(model, val_loader, device)
        val_f1 = val_res["macro_f1"]
        val_bal = val_res["balanced_acc"]
        cm = val_res["confusion"]

        elapsed = time.time() - t0

        # Log
        ep_log = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_macro_f1": float(val_f1),
            "val_balanced_acc": float(val_bal),
            "time_sec": float(elapsed),
        }
        history["epochs"].append(ep_log)

        print(
            f"[Fold {fold}] Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | valF1={val_f1:.4f} | valBalAcc={val_bal:.4f} | {elapsed:.1f}s"
        )
        print(f"[Fold {fold}] Confusion matrix (rows=true, cols=pred):\n{format_confusion(cm)}")

        # Early stopping / checkpointing on macro F1
        improved = val_f1 > best_f1 + 1e-6
        if improved:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            extra = {
                "fold": fold,
                "model_config": asdict(mcfg),
                "train_args": vars(args),
                "val_macro_f1": float(val_f1),
                "val_balanced_acc": float(val_bal),
                "confusion": cm.tolist(),
            }
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, best_f1, extra)

        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[Fold {fold}] Early stopping at epoch {epoch} (best epoch {best_epoch}, bestF1={best_f1:.4f})")
                break

    history["best"] = {
        "best_epoch": int(best_epoch),
        "best_macro_f1": float(best_f1),
        "best_ckpt_path": str(best_ckpt_path),
    }

    # Load best for final fold validation predictions
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    fold_val_res = evaluate(model, val_loader, device)

    # Add fold meta
    fold_val_res["fold"] = fold
    fold_val_res["best_epoch"] = best_epoch
    fold_val_res["best_ckpt_path"] = str(best_ckpt_path)

    return history, fold_val_res


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--cache_dir", type=str, required=True,
                    help="Directory containing cached .npz files from preprocess.py")
    ap.add_argument("--test_type", type=str, default=None,
                    help="Optional: filter cache files by suffix _{test_type}.npz (e.g., dnb, vst, ast, flanker, gng)")

    ap.add_argument("--out_dir", type=str,
                    default="/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/results",
                    help="Directory to save checkpoints and outputs")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)

    ap.add_argument("--patience", type=int, default=15,
                    help="Early stop patience (epochs without val macro F1 improvement)")
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # Dataset cleaning options
    ap.add_argument("--drop_if_low_finite_ratio", type=float, default=None,
                    help="If set (e.g., 0.5), drop students whose X finite ratio is below threshold.")
    ap.add_argument("--max_segments", type=int, default=None,
                    help="Optionally truncate sequences to at most this many segments.")

    # Device
    ap.add_argument("--device", type=str, default="cuda",
                    help="cuda or cpu")

    return ap.parse_args()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    # Dataset
    ds_opts = DatasetOptions(
        feature_dim=444,
        min_segments=1,
        max_segments=args.max_segments,
        nan_to_num=True,
        nan_fill_value=0.0,
        drop_if_low_finite_ratio=args.drop_if_low_finite_ratio,
        in_memory_cache=False,
        sort_by_time=False,
    )

    dataset = StudentSequenceDataset(
        cache_dir=args.cache_dir,
        test_type=args.test_type,
        opts=ds_opts,
    )

    idx, y, groups = get_groups_and_labels(dataset)
    gkf = GroupKFold(n_splits=5)

    # Storage
    fold_histories: List[Dict] = []
    all_fold_preds_rows: List[Dict] = []

    overall_best = {
        "macro_f1": -1.0,
        "fold": None,
        "epoch": None,
        "ckpt_path": None,
    }

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(idx, y, groups), start=1):
        print(f"\n========== Fold {fold}/5 ==========")
        history, fold_val = train_one_fold(
            fold=fold,
            train_idx=tr_idx,
            val_idx=va_idx,
            dataset=dataset,
            device=device,
            args=args,
            out_dir=out_dir,
        )
        fold_histories.append(history)

        # Record fold best to overall best
        if fold_val["macro_f1"] > overall_best["macro_f1"]:
            overall_best["macro_f1"] = float(fold_val["macro_f1"])
            overall_best["fold"] = int(fold_val["fold"])
            overall_best["epoch"] = int(fold_val["best_epoch"])
            overall_best["ckpt_path"] = str(fold_val["best_ckpt_path"])

        # Build per-student prediction rows (val set for this fold)
        probs = fold_val["probs"]  # (N,4)
        y_true = fold_val["y_true"]
        y_pred = fold_val["y_pred"]
        cids = fold_val["client_ids"]

        for i in range(len(cids)):
            row = {
                "client_id": cids[i],
                "fold": fold,
                "y_true": int(y_true[i]),
                "y_pred": int(y_pred[i]),
                "p0": float(probs[i, 0]) if probs.shape[0] else 0.0,
                "p1": float(probs[i, 1]) if probs.shape[0] else 0.0,
                "p2": float(probs[i, 2]) if probs.shape[0] else 0.0,
                "p3": float(probs[i, 3]) if probs.shape[0] else 0.0,
            }
            all_fold_preds_rows.append(row)

    # Save histories
    hist_path = out_dir / "training_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(fold_histories, f, indent=2, ensure_ascii=False)

    # Save predictions CSV (all folds combined; each student appears once as val)
    pred_csv_path = out_dir / "per_student_predictions.csv"
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["client_id", "fold", "y_true", "y_pred", "p0", "p1", "p2", "p3"],
        )
        writer.writeheader()
        for row in all_fold_preds_rows:
            writer.writerow(row)

    # Save best checkpoint info to required text path
    best_txt_path = out_dir / "best_checkpoint.txt"
    with open(best_txt_path, "w", encoding="utf-8") as f:
        f.write(f"overall_best_macro_f1: {overall_best['macro_f1']:.6f}\n")
        f.write(f"overall_best_fold: {overall_best['fold']}\n")
        f.write(f"overall_best_epoch: {overall_best['epoch']}\n")
        f.write(f"overall_best_ckpt_path: {overall_best['ckpt_path']}\n")
        f.write(f"history_json: {str(hist_path)}\n")
        f.write(f"predictions_csv: {str(pred_csv_path)}\n")

    print("\n========== Done ==========")
    print(f"History saved to: {hist_path}")
    print(f"Predictions saved to: {pred_csv_path}")
    print(f"Best checkpoint info saved to: {best_txt_path}")
    print(json.dumps(overall_best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()