"""Diagnose the freeze on REAL Moving MNIST data, replicating the exact
trainer loop but printing per-term losses and true gradient norms."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.moving_mnist import get_splits
from losses.reconstruction import gdl_loss, l1_loss, mse_loss
from models.mmg_convlstm import MMGConvLSTM
from utils.device import get_device
from utils.seed import set_seed


def main():
    set_seed(42, deterministic=True)
    device = get_device()
    model = MMGConvLSTM().to(device)
    train_set, _, _ = get_splits(
        "datasets/raw/moving_mnist/mnist_test_seq.npy")

    loader = torch.utils.data.DataLoader(train_set, batch_size=8,
                                         shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                           weight_decay=1e-4)

    model.train()
    for step, xb in enumerate(loader):
        xb = xb.to(device)
        inp, tgt = xb[:, 0:10], xb[:, 10:20]

        opt.zero_grad(set_to_none=True)
        preds = model(inp)
        l_mse = mse_loss(preds, tgt)
        l_l1 = l1_loss(preds, tgt)
        l_gdl = gdl_loss(preds, tgt)
        loss = l_l1 + l_gdl
        loss.backward()

        # true total gradient norm BEFORE any clipping
        with torch.no_grad():
            gnorm = torch.sqrt(sum(
                (p.grad ** 2).sum() for p in model.parameters()
                if p.grad is not None))
            n_zero_grads = sum(
                1 for p in model.parameters()
                if p.grad is not None and p.grad.abs().max() == 0)
            n_none = sum(1 for p in model.parameters() if p.grad is None)
            pred_range = (preds.min().item(), preds.max().item())
            pred_std = preds.std().item()

        opt.step()

        if step % 20 == 0:
            print(f"step {step:3d} | loss={loss.item():.6f} "
                  f"(mse={l_mse.item():.5f} l1={l_l1.item():.5f} "
                  f"gdl={l_gdl.item():.5f}) | gnorm={gnorm.item():.8f} | "
                  f"zero-grad params={n_zero_grads} none-grad={n_none} | "
                  f"pred range=[{pred_range[0]:.4f},{pred_range[1]:.4f}] "
                  f"std={pred_std:.5f}")
        if step >= 100:
            break


if __name__ == "__main__":
    main()