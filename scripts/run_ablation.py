"""STAGE 14 - Run the four ablation experiments under identical conditions.

Usage:
    python scripts/run_ablation.py --config configs/moving_mnist.yaml \
        --variants A B C D [--max-epochs 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.moving_mnist import get_splits
from models.ablations import AblationConfig, AblationModel
from training.trainer import Trainer
from utils.reproducibility import save_run_record
from utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--max-epochs", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    t = cfg["training"]
    if args.max_epochs:
        t = {**t, "epochs": args.max_epochs}

    for variant in args.variants:
        print("=" * 60)
        print(f"ABLATION VARIANT {variant}")
        print("=" * 60)
        # identical seed per variant - conditions are exactly equal
        set_seed(t["seed"], deterministic=True)
        train_set, val_set, _ = get_splits(cfg["raw_path"])
        model = AblationModel(AblationConfig(
            in_channels=cfg["channels"], feat_channels=cfg["model"]["feat_channels"],
            variant=variant, input_len=cfg["input_len"], pred_len=cfg["pred_len"]))
        print(f"params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        out_dir = str(Path(cfg["output_dir"]) / f"ablation_{variant}")
        save_run_record({**cfg, "variant": variant}, t["seed"], out_dir)
        trainer = Trainer(model, train_set, val_set, {
            "epochs": t["epochs"], "batch_size": t["batch_size"], "lr": t["lr"],
            "weight_decay": t["weight_decay"], "scheduler": t["scheduler"],
            "max_grad_norm": t["max_grad_norm"], "num_workers": t["num_workers"],
            "early_stopping_patience": t["early_stopping_patience"],
        }, out_dir=out_dir)
        hist = trainer.fit()
        Path(out_dir, "history.json").write_text(json.dumps(hist, indent=2))


if __name__ == "__main__":
    main()