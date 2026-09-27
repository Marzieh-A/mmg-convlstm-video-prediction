"""STAGE 15 - Full horizon-wise evaluation (t+1 ... t+10 + overall).

Loads a checkpoint, runs the FULLY AUTOREGRESSIVE protocol on the test
split, and computes MSE / MAE / SSIM / LPIPS per horizon.
Results are saved as JSON next to the checkpoint's output dir.

Usage:
    python -m evaluation.evaluate --config configs/moving_mnist.yaml \
        --checkpoint outputs/moving_mnist/best.pth --variant D --use-lpips
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.moving_mnist import get_splits
from evaluation.horizon_metrics import per_horizon_metrics
from utils.device import get_device
from utils.seed import seed_worker, set_seed




def build_model(cfg: dict, variant: str = "D"):
    """Build the model from config; variant chooses the class."""
    if variant == "baseline":
        from baselines.convlstm_baseline import BaselineConfig, ConvLSTMBaseline
        return ConvLSTMBaseline(BaselineConfig(in_channels=cfg["channels"]))
    from models.ablations import AblationConfig, AblationModel
    mc = cfg["model"]
    return AblationModel(AblationConfig(
        in_channels=cfg["channels"],
        enc_c1=mc["enc_c1"], enc_c2=mc["enc_c2"], enc_c3=mc["enc_c3"],
        feat_channels=mc["feat_channels"],
        kernel_size=mc["kernel_size"],
        input_len=cfg["input_len"], pred_len=cfg["pred_len"],
        variant=variant))




def load_weights(model, ckpt_path: str):
    from training.checkpoint import load_checkpoint
    ckpt = load_checkpoint(ckpt_path, model)
    return model, ckpt.get("epoch")


@torch.no_grad()
def evaluate_model(model, dataset, config: dict, use_lpips: bool = True,
                   max_batches: int | None = None) -> dict:
    device = get_device()
    model = model.to(device).eval()


    loader = DataLoader(dataset, batch_size=config.get("batch_size", 8),
                        shuffle=False, num_workers=0, pin_memory=True,
                        worker_init_fn=seed_worker)


    lp_fn = None
    if use_lpips:
        from metrics.lpips_metric import lpips_score
        lp_fn = lpips_score

    metric_keys = ["mse", "mae", "ssim"] + (["lpips"] if lp_fn else [])
    agg = {k: [] for k in metric_keys}
    horizon_sum: dict = {k: None for k in metric_keys}
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        inp, tgt = batch[:, 0:10], batch[:, 10:20]
        preds = model(inp)
        if not torch.isfinite(preds).all():
            raise FloatingPointError("NaN/Inf predictions during evaluation")

        m = per_horizon_metrics(preds, tgt, compute_lpips=lp_fn)
        for k in agg:
            agg[k].append(m[f"overall_{k}"])
            h = torch.tensor(m[k])
            horizon_sum[k] = h if horizon_sum[k] is None else horizon_sum[k] + h

        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break

    result = {k: sum(v) / n_batches for k, v in agg.items()}
    result["n_batches"] = n_batches
    for k in metric_keys:
        result[f"horizon_{k}"] = (horizon_sum[k] / n_batches).tolist()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--variant", default="D",
                    choices=["A", "B", "C", "D", "baseline"])
    ap.add_argument("--use-lpips", action="store_true")
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["training"]["seed"], deterministic=True)

    # dataset dispatch (mirrors scripts/train.py): UCF101 reads avi
    # folders with official split lists; everything else is MM's npy
    if cfg.get("dataset") == "ucf101":
        from datasets.ucf101 import UCF101
        mv = cfg.get("max_videos_per_split")
        test_set = UCF101("test", root=cfg["root"],
                          split_dir=cfg["split_dir"], max_videos=mv)
        print(f"ucf101 test videos: {len(test_set)}")
    else:
        from datasets.moving_mnist import get_splits
        _, _, test_set = get_splits(cfg["raw_path"])

    model = build_model(cfg, args.variant)
    model, epoch = load_weights(model, args.checkpoint)
    print(f"loaded {args.checkpoint} (epoch {epoch})")

    result = evaluate_model(model, test_set, cfg["training"],
                            use_lpips=args.use_lpips)

    print("\n=== TEST RESULTS (overall over horizons) ===")
    for k, v in result.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_{args.variant}.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(f"saved: {out_file}")


if __name__ == "__main__":
    main()