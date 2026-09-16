#!/usr/bin/env python3
"""
Quick comparison of GRU vs TCN on the same data.
"""

from src.training.train import train_model

print("=" * 50)
print("Training GRU baseline...")
gru = train_model(model_name="gru", use_synthetic=False, epochs=6)
print("\n" + "=" * 50)
print("Training TCN...")
tcn = train_model(model_name="tcn", use_synthetic=False, epochs=6)

print("\n" + "=" * 50)
print("COMPARISON SUMMARY")
print("=" * 50)
print(f"GRU  | Val F1: {gru['best_val_f1']:.4f} | Test Acc: {gru['test_metrics']['accuracy']:.4f}")
print(f"TCN  | Val F1: {tcn['best_val_f1']:.4f} | Test Acc: {tcn['test_metrics']['accuracy']:.4f}")
print("=" * 50)
