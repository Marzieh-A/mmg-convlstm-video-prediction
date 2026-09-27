"""Print the full training history stored in a checkpoint."""

import sys
from pathlib import Path

import torch

path = sys.argv[1] if len(sys.argv) > 1 else \
    r"outputs/moving_mnist_gdl/latest.pth"
ck = torch.load(path, map_location="cpu", weights_only=False)
for r in ck["history"]:
    print(f"ep {r['epoch']:3d} | train {r['train_loss']:.6f} | "
          f"val {r['val_loss']:.6f} | val_mse {r['val_mse']:.6f} | "
          f"gnorm {r['grad_norm']:.4f}")
print(f"epoch stored: {ck['epoch']} | best_val: {ck['best_val']:.6f}")