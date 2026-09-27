"""Hunt the NaN: run composite training on real data for several hundred
steps, print per-term losses every 20 steps, and stop at the first NaN
reporting exactly which term and the prediction stats at that moment."""

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
        "C:/Users/pc/Desktop/project_updated/datasets/raw/moving_mnist/mnist_test_seq.npy")
    loader = torch.utils.data.DataLoader(train_set, batch_size=8,
                                         shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    for step, xb in enumerate(loader):
        xb = xb.to(device)
        inp, tgt = xb[:, 0:10], xb[:, 10:20]
        opt.zero_grad(set_to_none=True)
        preds = model(inp)

        finite_preds = torch.isfinite(preds).all().item()
        l_mse = mse_loss(preds, tgt)
        l_l1 = l1_loss(preds, tgt)
        l_gdl = gdl_loss(preds, tgt)
        loss = l_mse + l_l1 + l_gdl

        loss.backward()
        if step % 20 == 0:
            sq = None
            for p in model.parameters():
                if p.grad is not None:
                    s = (p.grad ** 2).sum()
                    sq = s if sq is None else sq + s
            g = torch.sqrt(sq) if sq is not None else torch.tensor(0.0)
            print(f"step {step:3d} | loss={loss.item():.6f} | gnorm={g.item():.6f} | "
                  f"pred std={preds.std().item():.5f}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if not torch.isfinite(loss) or not finite_preds:
            print(f"\n!!! NaN/Inf at step {step}")
            print(f"    preds finite: {finite_preds}")
            print(f"    mse={l_mse.item()} l1={l_l1.item()} gdl={l_gdl.item()}")
            print(f"    pred min={preds.min().item()} max={preds.max().item()}")
            print(f"    input finite={torch.isfinite(inp).all().item()} "
                  f"target finite={torch.isfinite(tgt).all().item()}")
            break




if __name__ == "__main__":
    main()