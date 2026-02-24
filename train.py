# training loop + evaluation + checkpointing
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py

Student-level 2-class classification using:
- BiGRU + additive attention pooling (model.py)
- Per-student cached sequences from preprocess.py (.npz files)
- GroupKFold (5-fold) split by student (client_id) to avoid leakage
- Early stopping on validation macro F1

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
    --cache_dir "/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/preprocessed_output/ast_filtered" \
    --test_type ast \
    --out_dir "/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/results/trial25_ast_bal" \
    --batch_size 16 \
    --epochs 100 \
    --patience 15 \
    --lr 3e-4 \
    --weight_decay 1e-2 \
    --num_workers 4 \
    --device cpu

CHANGED (minimal):
- Add 80/20 hold-out TEST split by student (GroupShuffleSplit on client_id)
- Run GroupKFold (5-fold) only on the 80% train+val set
- After CV, evaluate best checkpoint on the held-out test set
- Test metrics: micro F1, balanced accuracy, per-class recall, confusion matrix

Class balancing (NEW):
- Weighted loss (class-weighted CrossEntropy)
- Weights are computed from the TRAIN split only (per fold), NOT from val/test

Outputs:
- Best checkpoint info written to:
  /data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/results/best_checkpoint.txt
- Per-student predictions (validation across folds) written to:
  /data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/results/per_student_predictions.csv
- Held-out test metrics written to:
  test_metrics.json
- Held-out test predictions written to:
  test_predictions.csv
"""

import argparse
import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

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
    return batch


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    """
    Used for validation during CV and for test evaluation.
    Returns macro F1 (for early stop), balanced acc, confusion matrix, plus raw arrays.
    """
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
            "confusion": np.zeros((2, 2), dtype=int),
            "y_true": y_true,
            "y_pred": y_pred,
            "probs": np.array(all_probs) if len(all_probs) else np.zeros((0, 2), dtype=float),
            "client_ids": all_client_ids,
        }

    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int)

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
    lines = []
    for row in cm.tolist():
        lines.append("  " + " ".join(f"{v:4d}" for v in row))
    return "\n".join(lines)


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float, extra: Dict) -> None:
    ckpt = {
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "extra": extra,
    }
    torch.save(ckpt, path)


# -----------------------------
# Training (one CV fold)
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

    mcfg = ModelConfig(
        in_dim=258,
        proj_hidden=128,
        proj_out=64,
        dropout=args.dropout,
        gru_hidden=64,
        attn_hidden=64,
        num_classes=2,
        layer_norm=True,
    )
    model = StudentBiGRUAttnClassifier(mcfg).to(device)

    # --- class balancing: weighted loss computed from TRAIN split only ---
    y_train = np.asarray([dataset.labels[i] for i in train_idx.tolist()], dtype=int)
    c0 = int(np.sum(y_train == 0))
    c1 = int(np.sum(y_train == 1))
    total = max(c0 + c1, 1)
    # inverse-frequency weights: total / (num_classes * count)
    w0 = total / (2.0 * max(c0, 1))
    w1 = total / (2.0 * max(c1, 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = out_dir / f"fold{fold}_best.pt"
    patience_counter = 0

    history = {"fold": fold, "epochs": [], "best": {}}

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

        val_res = evaluate(model, val_loader, device)
        val_f1 = val_res["macro_f1"]
        val_bal = val_res["balanced_acc"]
        cm = val_res["confusion"]
        elapsed = time.time() - t0

        history["epochs"].append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_macro_f1": float(val_f1),
            "val_balanced_acc": float(val_bal),
            "time_sec": float(elapsed),
        })

        print(
            f"[Fold {fold}] Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | valMacroF1={val_f1:.4f} | valBalAcc={val_bal:.4f} | {elapsed:.1f}s"
        )
        print(f"[Fold {fold}] Confusion matrix (rows=true, cols=pred):\n{format_confusion(cm)}")

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
                "class_weights": [float(w0), float(w1)],
                "train_counts": {"c0": c0, "c1": c1},
            }
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, best_f1, extra)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[Fold {fold}] Early stopping at epoch {epoch} (best epoch {best_epoch}, bestMacroF1={best_f1:.4f})")
                break

    history["best"] = {
        "best_epoch": int(best_epoch),
        "best_macro_f1": float(best_f1),
        "best_ckpt_path": str(best_ckpt_path),
    }

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    fold_val_res = evaluate(model, val_loader, device)
    fold_val_res["fold"] = fold
    fold_val_res["best_epoch"] = best_epoch
    fold_val_res["best_ckpt_path"] = str(best_ckpt_path)

    return history, fold_val_res


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--test_type", type=str, default=None)

    ap.add_argument("--out_dir", type=str,
                    default="/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/results")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)

    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--drop_if_low_finite_ratio", type=float, default=None)
    ap.add_argument("--max_segments", type=int, default=None)

    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--test_size", type=float, default=0.2,
                    help="Fraction of students to hold out as test set (default: 0.2)")

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

    ds_opts = DatasetOptions(
        feature_dim=258,
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

    # 80/20 hold-out test split by student (group)
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    trainval_pos, test_pos = next(gss.split(idx, y, groups))
    trainval_idx = idx[trainval_pos]
    test_idx = idx[test_pos]

    print(f"[Split] total={len(idx)} | train+val={len(trainval_idx)} | test={len(test_idx)}")

    # CV only on trainval set
    y_trainval = y[trainval_pos]
    groups_trainval = groups[trainval_pos]
    gkf = GroupKFold(n_splits=5)

    fold_histories: List[Dict] = []
    all_fold_preds_rows: List[Dict] = []

    overall_best = {
        "macro_f1": -1.0,
        "fold": None,
        "epoch": None,
        "ckpt_path": None,
    }

    # GroupKFold on the 80% (train+val)
    for fold, (tr_local, va_local) in enumerate(gkf.split(trainval_idx, y_trainval, groups_trainval), start=1):
        tr_idx = trainval_idx[tr_local]
        va_idx = trainval_idx[va_local]

        print(f"\n========== Fold {fold}/5 (on train+val 80%) ==========")
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

        if fold_val["macro_f1"] > overall_best["macro_f1"]:
            overall_best["macro_f1"] = float(fold_val["macro_f1"])
            overall_best["fold"] = int(fold_val["fold"])
            overall_best["epoch"] = int(fold_val["best_epoch"])
            overall_best["ckpt_path"] = str(fold_val["best_ckpt_path"])

        probs = fold_val["probs"]
        y_true = fold_val["y_true"]
        y_pred = fold_val["y_pred"]
        cids = fold_val["client_ids"]

        for i in range(len(cids)):
            all_fold_preds_rows.append({
                "client_id": cids[i],
                "fold": fold,
                "split": "val",
                "y_true": int(y_true[i]),
                "y_pred": int(y_pred[i]),
                "p0": float(probs[i, 0]) if probs.shape[0] else 0.0,
                "p1": float(probs[i, 1]) if probs.shape[0] else 0.0,
            })

    # Save CV history and val predictions
    hist_path = out_dir / "training_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(fold_histories, f, indent=2, ensure_ascii=False)

    pred_csv_path = out_dir / "per_student_predictions.csv"
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["client_id", "fold", "split", "y_true", "y_pred", "p0", "p1"],
        )
        writer.writeheader()
        for row in all_fold_preds_rows:
            writer.writerow(row)

    # Save best checkpoint info
    best_txt_path = out_dir / "best_checkpoint.txt"
    with open(best_txt_path, "w", encoding="utf-8") as f:
        f.write(f"overall_best_macro_f1: {overall_best['macro_f1']:.6f}\n")
        f.write(f"overall_best_fold: {overall_best['fold']}\n")
        f.write(f"overall_best_epoch: {overall_best['epoch']}\n")
        f.write(f"overall_best_ckpt_path: {overall_best['ckpt_path']}\n")
        f.write(f"history_json: {str(hist_path)}\n")
        f.write(f"val_predictions_csv: {str(pred_csv_path)}\n")

    # TEST evaluation on held-out 20%
    print("\n========== Testing on held-out 20% ==========")
    if overall_best["ckpt_path"] is None:
        raise RuntimeError("No best checkpoint found; cannot run test evaluation.")

    # Build model and load checkpoint
    mcfg = ModelConfig(
        in_dim=258,
        proj_hidden=128,
        proj_out=64,
        dropout=args.dropout,
        gru_hidden=64,
        attn_hidden=64,
        num_classes=2,
        layer_norm=True,
    )
    model = StudentBiGRUAttnClassifier(mcfg).to(device)
    ckpt = torch.load(overall_best["ckpt_path"], map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loader = DataLoader(
        Subset(dataset, test_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_student_batch,
    )

    test_res = evaluate(model, test_loader, device)
    y_true = test_res["y_true"]
    y_pred = test_res["y_pred"]
    cm = test_res["confusion"]
    bal_acc = float(test_res["balanced_acc"])

    micro_f1 = float(f1_score(y_true, y_pred, average="micro")) if y_true.size else 0.0
    per_class_recall = recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0).astype(float).tolist()

    print(f"[TEST] micro_f1={micro_f1:.4f} | balanced_acc={bal_acc:.4f}")
    print(f"[TEST] per_class_recall (0..1)={per_class_recall}")
    print(f"[TEST] Confusion matrix (rows=true, cols=pred):\n{format_confusion(cm)}")

    test_metrics = {
        "micro_f1": micro_f1,
        "balanced_accuracy": bal_acc,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
        "n_test_students": int(len(test_idx)),
        "best_ckpt_used": str(overall_best["ckpt_path"]),
    }

    test_metrics_path = out_dir / "test_metrics.json"
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    # Save test predictions
    test_pred_path = out_dir / "test_predictions.csv"
    with open(test_pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["client_id", "split", "y_true", "y_pred", "p0", "p1"],
        )
        writer.writeheader()
        probs = test_res["probs"]
        cids = test_res["client_ids"]
        for i in range(len(cids)):
            writer.writerow({
                "client_id": cids[i],
                "split": "test",
                "y_true": int(y_true[i]),
                "y_pred": int(y_pred[i]),
                "p0": float(probs[i, 0]) if probs.shape[0] else 0.0,
                "p1": float(probs[i, 1]) if probs.shape[0] else 0.0,
            })

    print("\n========== Done ==========")
    print(f"History saved to: {hist_path}")
    print(f"Val predictions saved to: {pred_csv_path}")
    print(f"Best checkpoint info saved to: {best_txt_path}")
    print(f"Test metrics saved to: {test_metrics_path}")
    print(f"Test predictions saved to: {test_pred_path}")
    print(json.dumps(overall_best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()