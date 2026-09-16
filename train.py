#!/usr/bin/env python3
"""
Entry point for training.
Usage:
    python train.py --model tcn --epochs 80
    python train.py --model gru --epochs 40
"""

import argparse
from src.training.train import train_model


def main():
    parser = argparse.ArgumentParser(description="Train ISL Temporal Model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="tcn", choices=["gru", "tcn", "hybrid"])
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    train_model(
        config_path=args.config,
        model_name=args.model,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()