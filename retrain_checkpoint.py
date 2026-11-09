"""Retrain the MNIST MLP base model (seed 42) and save the checkpoint.

Replicates pipeline.py Phase 0 exactly:
  set_seed(42), MLP(), Adam lr=1e-3, 10 epochs, batch size 256.
Saves to outputs/checkpoints/mlp_seed42.pt and verifies FP32 accuracy
on the full 10,000-sample test set (expected: 97.75%).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from src.model import MLP
from src.train import set_seed, get_data_loaders, train, evaluate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_dir = os.path.join("outputs", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "mlp_seed42.pt")

    set_seed(42)
    model = MLP()
    train_loader, test_loader = get_data_loaders("./data", batch_size=256)
    model = train(model, train_loader, device, epochs=10, lr=1e-3)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved -> {ckpt_path}")

    acc = evaluate(model, test_loader, device)
    print(f"FP32 full-test accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()
