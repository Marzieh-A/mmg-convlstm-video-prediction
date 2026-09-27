"""STAGE 10 - Trainer: AdamW, gradient clipping, validation per epoch,
best/latest checkpoints, early stopping, NaN detection, grad-norm and
GPU-memory monitoring, resume support.

Autoregressive protocol: the model itself implements AR generation
(Stage 6); the trainer only supplies observed inputs and future targets.

Model selection (rev3): the best checkpoint is chosen by *pure MSE* on
validation, independent of the training loss. With the two-phase loss the
criterion scale changes at the warmup boundary, so model selection AND
early stopping are reset there: phase 2 is judged on its own trajectory
(documented from our warmup-run diagnosis, where early stopping fired
against the phase-1 best and cut phase 2 short while it was still
improving).
"""

from __future__ import annotations

import math
import time

import torch
from torch.utils.data import DataLoader

from losses.reconstruction import WarmupLoss, mse_loss
from utils.device import get_device
from utils.seed import seed_worker


class Trainer:
    def __init__(self, model, train_set, val_set, config: dict, out_dir: str,
                 resume_path: str | None = None):
        self.cfg = config
        self.out_dir = out_dir
        self.device = get_device()
        self.bad_epochs = 0

        self.model = model.to(self.device)
        self.train_loader = DataLoader(
            train_set, batch_size=config["batch_size"], shuffle=True,
            num_workers=config.get("num_workers", 0), pin_memory=True,
            worker_init_fn=seed_worker, drop_last=True)
        self.val_loader = DataLoader(
            val_set, batch_size=config["batch_size"], shuffle=False,
            num_workers=config.get("num_workers", 0), pin_memory=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["lr"], weight_decay=config.get("weight_decay", 1e-4))
        from utils.reproducibility import save_run_record
        save_run_record(self.cfg, self.cfg.get("seed", 0), out_dir)
        from training.scheduler import make_scheduler
        self.scheduler = make_scheduler(
            self.optimizer, config.get("scheduler", "cosine"),
            config["epochs"], config["lr"])

        # Two-phase loss (documented diagnosis): pure MSE for the warmup
        # epochs (escapes the constant-gray attractor), then composite
        # MSE+L1+GDL (Mathieu et al. family) for sharpening.
        warmup = int(config.get("loss_warmup_epochs", 8))
        print(f"[trainer] loss: MSE for epochs 0..{warmup - 1}, "
              f"then composite")
        self.criterion = WarmupLoss(warmup)
        self._in_warmup = True

        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        self.history = []
        self.best_val = math.inf
        self.start_epoch = 0
        if resume_path:
            from training.checkpoint import load_checkpoint
            ckpt = load_checkpoint(resume_path, self.model, self.optimizer,
                                   self.scheduler)
            self.start_epoch = ckpt["epoch"]
            self.best_val = ckpt["best_val"]
            self.history = ckpt["history"]
            print(f"resumed from {resume_path} at epoch {self.start_epoch}")

    # ---- helpers ----

    def _clip_and_step(self) -> float:
        gnorm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(gnorm):
            raise FloatingPointError(f"non-finite gradient norm: {gnorm}")
        return float(gnorm)

    def _nan_stop(self, loss: float, where: str):
        if not math.isfinite(loss):
            raise FloatingPointError(
                f"NaN/Inf loss at {where} - STOP and diagnose. "
                f"Checkpoints are NOT overwritten on NaN epochs.")

    def train_one_epoch(self, epoch: int) -> dict:
        self.criterion.set_epoch(epoch)
        # phase boundary: reset model selection and early stopping so
        # phase 2 is judged on its own trajectory
        if self._in_warmup and epoch >= self.criterion.warmup_epochs:
            self.best_val = math.inf
            self.bad_epochs = 0
            print(f"[trainer] phase boundary at epoch {epoch}: "
                  f"model selection / early stopping reset for phase 2")
        self._in_warmup = epoch < self.criterion.warmup_epochs

        self.model.train()
        total, nb = 0.0, 0
        t0 = time.time()
        gnorm_sum = 0.0
        for xb in self.train_loader:
            xb = xb.to(self.device, non_blocking=True)
            inp, tgt = xb[:, 0:10], xb[:, 10:20]

            self.optimizer.zero_grad(set_to_none=True)
            preds = self.model(inp)
            loss = self.criterion(preds, tgt)

            if not torch.isfinite(loss):
                self._nan_stop(float(loss), f"epoch {epoch} train")
            loss.backward()
            gnorm_sum += self._clip_and_step()

            total += loss.item(); nb += 1

        if self.scheduler is not None:
            self.scheduler.step()
        return {
            "train_loss": total / max(1, nb),
            "grad_norm": gnorm_sum / max(1, nb),
            "epoch_time_s": time.time() - t0,
            "lr": self.optimizer.param_groups[0]["lr"],
            "gpu_mem_mb": (torch.cuda.max_memory_allocated() / 1024**2)
                          if self.device.type == "cuda" else 0.0,
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """Returns criterion val_loss AND pure-MSE val_mse (the
        scale-stable model-selection and reporting metric)."""
        self.model.eval()
        total, mse_sum, nb = 0.0, 0.0, 0
        for xb in self.val_loader:
            xb = xb.to(self.device, non_blocking=True)
            inp, tgt = xb[:, 0:10], xb[:, 10:20]
            preds = self.model(inp)
            loss = self.criterion(preds, tgt)
            self._nan_stop(float(loss), "validation")
            mse_sum += mse_loss(preds, tgt).item()
            total += loss.item(); nb += 1
        return {"val_loss": total / max(1, nb),
                "val_mse": mse_sum / max(1, nb)}

    def fit(self) -> list:
        from training.checkpoint import save_checkpoint
        from pathlib import Path
        out = Path(self.out_dir)
        patience = self.cfg.get("early_stopping_patience", 10)

        for epoch in range(self.start_epoch, self.cfg["epochs"]):
            stats = self.train_one_epoch(epoch)
            v = self.validate()
            stats.update({"epoch": epoch, **v})
            self.history.append(stats)
            print(f"epoch {epoch:3d} | train {stats['train_loss']:.6f} | "
                  f"val {v['val_loss']:.6f} | val_mse {v['val_mse']:.6f} | "
                  f"lr {stats['lr']:.2e} | gnorm {stats['grad_norm']:.2f} | "
                  f"{stats['epoch_time_s']:.1f}s | "
                  f"mem {stats['gpu_mem_mb']:.0f} MB")

            save_checkpoint(out / "latest.pth", self.model, self.optimizer,
                            self.scheduler, epoch + 1, self.best_val,
                            self.history, self.cfg)
            if v["val_mse"] < self.best_val:      # selection on pure MSE
                self.best_val = v["val_mse"]
                self.bad_epochs = 0
                save_checkpoint(out / "best.pth", self.model, self.optimizer,
                                self.scheduler, epoch + 1, self.best_val,
                                self.history, self.cfg)
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= patience:
                    print(f"early stopping at epoch {epoch}")
                    break

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

        return self.history