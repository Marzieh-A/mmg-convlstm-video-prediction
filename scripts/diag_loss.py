"""Diagnose the stalled GDL-L1 training: per-term losses, gradient norms,
and a single-batch overfit sanity check."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.reconstruction import gdl_loss, l1_loss, mse_loss
from models.mmg_convlstm import MMGConvLSTM
from utils.device import get_device
from utils.seed import set_seed


def main():
    set_seed(42, deterministic=True)
    device = get_device()
    model = MMGConvLSTM().to(device)

    # synthetic batch in the real data regime
    x = torch.rand(4, 10, 1, 64, 64, device=device)
    tgt = torch.rand(4, 10, 1, 64, 64, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("== per-term losses (before any training) ==")
    model.eval()
    with torch.no_grad():
        p = model(x)
        print(f"mse={mse_loss(p, tgt):.6f}  l1={l1_loss(p, tgt):.6f}  "
              f"gdl={gdl_loss(p, tgt):.6f}  "
              f"pred range=[{p.min():.4f},{p.max():.4f}]")

    print("\n== single-batch overfit: 60 Adam steps, loss every 10 ==")
    model.train()
    for step in range(60):
        opt.zero_grad()
        preds = model(x)
        loss = l1_loss(preds, tgt) + gdl_loss(preds, tgt)
        loss.backward()
        total_gnorm = torch.sqrt(sum(
            (p_.grad ** 2).sum() for p_ in model.parameters()
            if p_.grad is not None))
        opt.step()
        if step % 10 == 0 or step == 59:
            print(f"step {step:2d} | loss={loss.item():.6f} | gnorm={total_gnorm:.6f}")

    print("\n== check: does the saved trainer see gradients? (1 epoch mini) ==")
    # gradient magnitude distribution across parameters
    norms = [(n, p_.grad.abs().mean().item())
             for n, p_ in model.named_parameters() if p_.grad is not None]
    norms.sort(key=lambda t: -t[1])
    for n, v in norms[:5]:
        print(f"largest grad mean-abs: {n}: {v:.8f}")
    for n, v in norms[-5:]:
        print(f"smallest grad mean-abs: {n}: {v:.8f}")


if __name__ == "__main__":
    main()