"""STAGE 10 - Training entry point (dataset-dispatching).

Usage:
    python scripts/train.py --config configs/moving_mnist.yaml
    python scripts/train.py --config configs/ucf101.yaml
    python scripts/train.py --config configs/moving_mnist.yaml --resume outputs/moving_mnist/latest.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.mmg_convlstm import MMGConfig, MMGConvLSTM
from training.trainer import Trainer
from utils.reproducibility import save_run_record
from utils.seed import set_seed


def build_datasets(cfg: dict):
    """Dataset dispatch: Moving MNIST reads one .npy; UCF101 reads
    avi folders with the official split lists. `max_videos_per_split`
    (UCF only) selects the smoke-test subset; null = full split."""
    if cfg.get("dataset") == "ucf101":
        from datasets.ucf101 import UCF101
        mv = cfg.get("max_videos_per_split")
        train_set = UCF101("train", root=cfg["root"],
                           split_dir=cfg["split_dir"], max_videos=mv)
        val_set = UCF101("val", root=cfg["root"],
                         split_dir=cfg["split_dir"], max_videos=mv)
        test_set = UCF101("test", root=cfg["root"],
                          split_dir=cfg["split_dir"], max_videos=mv)
        print(f"ucf101 videos: train={len(train_set)} "
              f"val={len(val_set)} test={len(test_set)}")
    else:
        from datasets.moving_mnist import get_splits
        train_set, val_set, test_set = get_splits(cfg["raw_path"])
    return train_set, val_set, test_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg["training"]
    set_seed(tcfg["seed"], deterministic=True)

    train_set, val_set, _ = build_datasets(cfg)

    model = MMGConvLSTM(MMGConfig(
        in_channels=cfg["channels"],
        enc_c1=cfg["model"]["enc_c1"], enc_c2=cfg["model"]["enc_c2"],
        enc_c3=cfg["model"]["enc_c3"], feat_channels=cfg["model"]["feat_channels"],
        kernel_size=cfg["model"]["kernel_size"],
        input_len=cfg["input_len"], pred_len=cfg["pred_len"],
    ))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params:,}")

    save_run_record(cfg, tcfg["seed"], cfg["output_dir"])
    trainer = Trainer(model, train_set, val_set, {
        "epochs": tcfg["epochs"], "batch_size": tcfg["batch_size"],
        "lr": tcfg["lr"], "weight_decay": tcfg["weight_decay"],
        "scheduler": tcfg["scheduler"], "max_grad_norm": tcfg["max_grad_norm"],
        "num_workers": tcfg["num_workers"],
        "early_stopping_patience": tcfg["early_stopping_patience"],
        "loss_kind": tcfg.get("loss_kind", "composite"),
        "loss_warmup_epochs": tcfg.get("loss_warmup_epochs", 8),
        "seed": tcfg["seed"],
    }, out_dir=cfg["output_dir"], resume_path=args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()