"""
Training and Evaluation Pipeline for ISL Temporal Models.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

from src.models import GRUClassifier, TCNClassifier, HybridTCNGRU
from src.data.dataset import ISLDataset          # ← new dataset with augmentation
from src.utils.config import load_config, set_seed, get_device
from src.utils.logger import setup_logger


def build_model(cfg: Dict[str, Any], num_classes: int) -> nn.Module:
    name = cfg["model"]["name"].lower()
    input_dim = cfg["model"]["input_dim"]
    dropout = cfg["model"]["dropout"]

    if name == "gru":
        return GRUClassifier(
            input_dim=input_dim,
            hidden_dim=cfg["model"]["hidden_dim"],
            num_layers=cfg["model"].get("num_layers", 2),
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=False,
        )
    elif name == "tcn":
        return TCNClassifier(
            input_dim=input_dim,
            num_channels=[64, 128, 128, 256],
            kernel_size=cfg["model"].get("kernel_size", 3),
            dropout=dropout,
            num_classes=num_classes,
        )
    elif name == "hybrid":
        return HybridTCNGRU(
            input_dim=input_dim,
            tcn_channels=[64, 128, 128],
            kernel_size=cfg["model"].get("kernel_size", 3),
            gru_hidden=cfg["model"]["hidden_dim"],
            num_classes=num_classes,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, inv_label_map=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * X.size(0)

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    report = classification_report(
        all_targets, all_preds,
        target_names=[inv_label_map[i] for i in sorted(inv_label_map.keys())] if inv_label_map else None,
        zero_division=0,
        digits=4,
    )

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
    }


def train_model(
    config_path: str = "configs/default.yaml",
    model_name: str = None,
    epochs: int = None,
):
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    logger = setup_logger("train", "logs/train.log")

    if model_name:
        cfg["model"]["name"] = model_name
    if epochs:
        cfg["training"]["epochs"] = epochs

    device = get_device(cfg["training"]["device"] == "cuda")
    logger.info(f"Using device: {device}")
    logger.info(f"Model: {cfg['model']['name']}")

    # ------------------------------------------------------------------
    # Data (Real data + Augmentation)
    # ------------------------------------------------------------------
    processed_dir = Path(cfg["data"]["processed_dir"])
    seq_len = cfg["data"]["sequence_length"]

    train_dataset = ISLDataset(processed_dir / "train.npz", augment=True,  seq_len=seq_len)
    val_dataset   = ISLDataset(processed_dir / "val.npz",   augment=False, seq_len=seq_len)
    test_dataset  = ISLDataset(processed_dir / "test.npz",  augment=False, seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)

    # Label maps
    with open(processed_dir / "label_map.json") as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    logger.info(f"Classes: {num_classes} | Train samples: {len(train_dataset)}")

    # ------------------------------------------------------------------
    # Class Weights
    # ------------------------------------------------------------------
    train_labels = train_dataset.y
    class_counts = Counter(train_labels)
    total = len(train_labels)

    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weights.append(total / (num_classes * count))

    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    logger.info(f"Class weights: {[round(w, 2) for w in weights]}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(cfg, num_classes).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    best_val_f1 = 0.0
    patience = cfg["training"]["early_stopping_patience"]
    patience_counter = 0

    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save label map
    with open(Path(cfg["paths"]["label_map"]), "w") as f:
        json.dump(label_map, f, indent=2)

    logger.info("Starting training with augmentation + class weights...")
    start_time = time.time()

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device, inv_label_map)

        scheduler.step()

        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": best_val_f1,
                "val_acc": val_metrics["accuracy"],
                "config": cfg,
                "num_classes": num_classes,
                "model_name": cfg["model"]["name"],
            }, checkpoint_dir / "best_model.pth")
            logger.info(f"  -> New best model saved (F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start_time
    logger.info(f"Training finished in {elapsed/60:.1f} minutes")

    # ------------------------------------------------------------------
    # Final Test
    # ------------------------------------------------------------------
    logger.info("Loading best model for test evaluation...")
    ckpt = torch.load(checkpoint_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate_model(model, test_loader, criterion, device, inv_label_map)
    logger.info(f"\n=== TEST RESULTS ===")
    logger.info(f"Accuracy : {test_metrics['accuracy']:.4f}")
    logger.info(f"Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Recall   : {test_metrics['recall']:.4f}")
    logger.info(f"F1-Score : {test_metrics['f1']:.4f}")
    logger.info(f"\n{test_metrics['report']}")

    return {
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":
    train_model(epochs=80)