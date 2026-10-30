"""
Hybrid TCN-GRU Training Script for Sign Language Recognition
Author: Prepared for SIG project
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import weight_norm
from tqdm import tqdm

# =========================================================
# 1. CONFIGURATION
# =========================================================
class Config:
    # Data
    data_dir = "./data"                  # folder containing train/ and val/
    train_seq_path = "./data/train_sequences.npy"
    train_label_path = "./data/train_labels.npy"
    val_seq_path = "./data/val_sequences.npy"
    val_label_path = "./data/val_labels.npy"

    # Model
    input_size = 1629                    # MediaPipe Holistic (543 landmarks * 3)
    tcn_channels = [128, 128, 256]
    gru_hidden = 256
    num_classes = 50                     # ← CHANGE THIS
    kernel_size = 3
    dropout = 0.3

    # Training
    batch_size = 32
    epochs = 80
    learning_rate = 1e-3
    weight_decay = 1e-4
    max_grad_norm = 1.0
    early_stop_patience = 15

    # Misc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "best_hybrid_tcn_gru.pth"
    num_workers = 4


# =========================================================
# 2. MODEL DEFINITION
# =========================================================
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = nn.ConstantPad1d((0, -padding), 0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = nn.ConstantPad1d((0, -padding), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                              dilation=dilation,
                              padding=(kernel_size-1)*dilation,
                              dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class HybridTCN_GRU(nn.Module):
    def __init__(self, input_size, tcn_channels, gru_hidden, num_classes,
                 kernel_size=3, dropout=0.3):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, tcn_channels, kernel_size, dropout)
        self.gru = nn.GRU(
            input_size=tcn_channels[-1],
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)               # → (B, C, T)
        y = self.tcn(x)                     # → (B, tcn_channels[-1], T)
        y = y.transpose(1, 2)               # → (B, T, tcn_channels[-1])
        out, _ = self.gru(y)
        out = out[:, -1, :]                 # last time step
        return self.fc(out)


# =========================================================
# 3. DATASET
# =========================================================
class SignDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_data(cfg):
    print("Loading data...")
    train_seq = np.load(cfg.train_seq_path)      # shape: (N, T, 1629)
    train_lbl = np.load(cfg.train_label_path)    # shape: (N,)
    val_seq   = np.load(cfg.val_seq_path)
    val_lbl   = np.load(cfg.val_label_path)

    print(f"Train sequences: {train_seq.shape}")
    print(f"Val sequences  : {val_seq.shape}")

    train_dataset = SignDataset(train_seq, train_lbl)
    val_dataset   = SignDataset(val_seq, val_lbl)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


# =========================================================
# 4. TRAINING & VALIDATION
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, cfg):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for sequences, labels in pbar:
        sequences, labels = sequences.to(cfg.device), labels.to(cfg.device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=loss.item())

    return running_loss / len(loader), 100. * correct / total


@torch.no_grad()
def validate(model, loader, criterion, cfg):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in loader:
        sequences, labels = sequences.to(cfg.device), labels.to(cfg.device)
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total


# =========================================================
# 5. MAIN
# =========================================================
def main():
    cfg = Config()
    print(f"Using device: {cfg.device}")

    train_loader, val_loader = load_data(cfg)

    model = HybridTCN_GRU(
        input_size=cfg.input_size,
        tcn_channels=cfg.tcn_channels,
        gru_hidden=cfg.gru_hidden,
        num_classes=cfg.num_classes,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout
    ).to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=6, verbose=True, min_lr=1e-6)

    best_val_acc = 0.0
    patience_counter = 0

    print("\nStarting training...\n")

    for epoch in range(1, cfg.epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, cfg)
        val_loss, val_acc = validate(model, val_loader, criterion, cfg)

        scheduler.step(val_loss)
        elapsed = time.time() - start

        print(f"Epoch {epoch:03d}/{cfg.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"Time: {elapsed:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': vars(cfg)
            }, cfg.save_path)
            print(f"  → Best model saved! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nTraining finished. Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved at: {cfg.save_path}")


if __name__ == "__main__":
    main()