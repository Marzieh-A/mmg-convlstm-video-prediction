"""Inspect a checkpoint: list keys, check weight finiteness."""

import sys
from pathlib import Path

import torch

path = sys.argv[1] if len(sys.argv) > 1 else \
    r"outputs/moving_mnist_gdl/latest.pth"
ckpt = torch.load(path, map_location="cpu", weights_only=False)

print(f"top-level keys: {list(ckpt.keys())}")
for meta in ("epoch", "best_val", "config"):
    if meta in ckpt:
        print(f"{meta}: {ckpt[meta]}")
if "history" in ckpt:
    print(f"history entries: {len(ckpt['history'])}")
    print(f"last history record: {ckpt['history'][-1]}")

# find the state dict under any common key, or scan all dict values
sd = None
for key in ("model_state_dict", "model", "state_dict", "model_sd",
            "weights", "generator"):
    if key in ckpt and isinstance(ckpt[key], dict):
        sd = ckpt[key]
        print(f"state dict found under key: '{key}'")
        break
if sd is None:
    # fall back: any dict-of-tensors value
    for k, v in ckpt.items():
        if isinstance(v, dict) and v and all(
                torch.is_tensor(t) for t in list(v.values())[:5]):
            sd = v
            print(f"state dict found (fallback) under key: '{k}'")
            break

if sd is None:
    print("NO state dict found in checkpoint!")
    sys.exit(1)

n_nan, total, worst = 0, 0, []
for k2, v in sd.items():
    if not torch.is_tensor(v):
        continue
    total += 1
    if not torch.isfinite(v).all():
        n_nan += 1
        worst.append(k2)
print(f"weight tensors: {total}, non-finite: {n_nan}")
if worst:
    print("non-finite layers:", worst[:10])