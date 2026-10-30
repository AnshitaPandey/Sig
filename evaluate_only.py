"""
Evaluate the saved best model on the held-out test set.
Usage:
    python evaluate_only.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src.data.dataset import ISLDataset
from src.utils.config import load_config, get_device
from src.models.transformer import TransformerClassifier

try:
    from src.models.tcn import TCNClassifier
except ImportError:
    TCNClassifier = None

try:
    from src.models.hybrid import HybridTCNGRU
except ImportError:
    HybridTCNGRU = None

try:
    from src.models.lstm import LSTMClassifier
except ImportError:
    LSTMClassifier = None


def build_model_from_ckpt(ckpt, num_classes):
    name = ckpt.get("model_name", "transformer").lower()
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}

    input_dim = model_cfg.get("input_dim", 300)
    dropout = model_cfg.get("dropout", 0.2)

    if name == "transformer":
        model = TransformerClassifier(
            input_dim=input_dim,
            d_model=model_cfg.get("d_model", 160),
            nhead=model_cfg.get("nhead", 5),
            num_layers=model_cfg.get("num_layers", 4),
            dim_feedforward=model_cfg.get("dim_feedforward", 320),
            num_classes=num_classes,
            dropout=dropout,
        )
    elif name == "tcn" and TCNClassifier is not None:
        model = TCNClassifier(
            input_dim=input_dim,
            num_channels=[64, 128, 128, 256],
            kernel_size=3,
            dropout=dropout,
            num_classes=num_classes,
        )
    elif name == "hybrid" and HybridTCNGRU is not None:
        model = HybridTCNGRU(
            input_dim=input_dim,
            num_channels=[64, 128, 128],
            kernel_size=3,
            dropout=dropout,
            gru_hidden=128,
            gru_layers=1,
            num_classes=num_classes,
        )
    elif name == "lstm" and LSTMClassifier is not None:
        model = LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 2),
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=False,
        )
    else:
        raise ValueError(f"Unsupported model in checkpoint: {name}")

    return model


@torch.no_grad()
def evaluate(model, loader, device, inv_label_map):
    model.eval()
    all_preds, all_targets = [], []

    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    labels = list(range(len(inv_label_map)))
    target_names = [inv_label_map[i] for i in labels]

    report = classification_report(
        all_targets,
        all_preds,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        digits=4
    )

    cm = confusion_matrix(all_targets, all_preds, labels=labels)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
        "confusion_matrix": cm,
        "y_true": all_targets,
        "y_pred": all_preds,
    }


def main():
    cfg = load_config("configs/default.yaml")
    device = get_device(False)

    processed_dir = Path(cfg["data"]["processed_dir"])
    checkpoint_path = Path(cfg["paths"]["checkpoint_dir"]) / "best_model.pth"
    label_map_path = processed_dir / "label_map.json"

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    with open(label_map_path) as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    print(f"Classes: {num_classes}")
    print(f"Model name in ckpt: {ckpt.get('model_name', 'unknown')}")
    print(f"Best val F1 in ckpt: {ckpt.get('val_f1', 'N/A')}")

    # Test data
    seq_len = cfg["data"]["sequence_length"]
    test_dataset = ISLDataset(processed_dir / "test.npz", augment=False, seq_len=seq_len)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0
    )
    print(f"Test samples: {len(test_dataset)}")

    # Model
    model = build_model_from_ckpt(ckpt, num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metrics = evaluate(model, test_loader, device, inv_label_map)

    print("\n=== OFFLINE TEST RESULTS ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1']:.4f}")
    print("\n" + metrics["report"])

    # Save report
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "test_report.txt", "w", encoding="utf-8") as f:
        f.write("=== OFFLINE TEST RESULTS ===\n")
        f.write(f"Accuracy : {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall   : {metrics['recall']:.4f}\n")
        f.write(f"F1-Score : {metrics['f1']:.4f}\n\n")
        f.write(metrics["report"])

    np.save(out_dir / "confusion_matrix.npy", metrics["confusion_matrix"])

    print("\nSaved:")
    print("  results/test_report.txt")
    print("  results/confusion_matrix.npy")


if __name__ == "__main__":
    main()