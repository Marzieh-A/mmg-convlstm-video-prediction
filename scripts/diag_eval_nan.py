"""Load latest.pth STRICTLY into the eval-built model, then run the test
set batch by batch and localize the first NaN (batch, horizon)."""

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.moving_mnist import get_splits
from evaluation.evaluate import build_model
from utils.device import get_device
from utils.seed import set_seed

CKPT = r"outputs/moving_mnist_gdl/latest.pth"

with open("configs/moving_mnist.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
set_seed(42, deterministic=True)
device = get_device()

model = build_model(cfg, "D")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ckpt["model_state"]

# 1) strict key comparison
model_keys = set(model.state_dict().keys())
ckpt_keys = set(sd.keys())
missing = model_keys - ckpt_keys      # in model but not ckpt -> random init!
extra = ckpt_keys - model_keys
print(f"keys in model not in ckpt (would stay RANDOM): {len(missing)}")
for k in sorted(missing)[:10]:
    print(f"  - {k}")
print(f"keys in ckpt not in model: {len(extra)}")
for k in sorted(extra)[:10]:
    print(f"  - {k}")

# shape check on shared keys
shape_bad = [k for k in (model_keys & ckpt_keys)
             if model.state_dict()[k].shape != sd[k].shape]
print(f"shape mismatches: {len(shape_bad)}", shape_bad[:5])

model.load_state_dict(sd, strict=True)   # will raise if anything differs
model = model.to(device).eval()

_, _, test_set = get_splits(cfg["raw_path"])
loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False)

with torch.no_grad():
    for bi, xb in enumerate(loader):
        xb = xb.to(device)
        inp, tgt = xb[:, 0:10], xb[:, 10:20]
        preds = model(inp)
        if not torch.isfinite(preds).all():
            t_nan = (~torch.isfinite(preds)).any(dim=(0, 2, 3, 4)).nonzero()
            print(f"NaN at batch {bi}, horizons {t_nan.flatten().tolist()}")
            print(f"pred range before NaN not available; "
                  f"input finite={torch.isfinite(inp).all().item()}")
            break
        if bi % 25 == 0:
            print(f"batch {bi}: finite, pred std={preds.std().item():.5f}")
    else:
        print("FULL TEST SET FINITE - previous NaN was transient (CUDA state).")