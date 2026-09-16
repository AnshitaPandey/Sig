#!/usr/bin/env python3
"""
Real-time ISL → Text + Speech demo.
Usage:
    python run_realtime.py
    python run_realtime.py --camera 1
"""

import argparse
from pathlib import Path
from src.inference.predictor import RealTimePredictor


def main():
    parser = argparse.ArgumentParser(description="Real-time ISL Translator")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    # Check that a model exists
    ckpt = args.checkpoint or "models/checkpoints/best_model.pth"
    if not Path(ckpt).exists():
        print("=" * 60)
        print("No trained model found!")
        print("Please run training first:")
        print("    python train.py --model tcn --epochs 25")
        print("=" * 60)
        return

    predictor = RealTimePredictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )
    predictor.run_webcam(camera_id=args.camera)


if __name__ == "__main__":
    main()
