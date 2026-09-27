"""Diagnose the reproducible-NaN issue: run evaluate_model twice and
inspect WHERE the NaN appears (preds? per-horizon list? aggregation?)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate import evaluate_model
from models.ablations import AblationConfig, AblationModel


class Tiny(torch.utils.data.Dataset):
    def __init__(self, n=4):
        g = torch.Generator().manual_seed(0)
        self.data = torch.rand(n, 20, 1, 64, 64, generator=g)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]


def main():
    torch.manual_seed(0)
    m = AblationModel(AblationConfig(variant="A")).eval()
    ds = Tiny(4)
    x = ds[0].unsqueeze(0)

    for run in (1, 2, 3):
        with torch.no_grad():
            y = m(x)
        print(f"run {run}: preds finite={torch.isfinite(y).all().item()} "
              f"min={y.min().item():.6f} max={y.max().item():.6f}")

    for run in (1, 2):
        r = evaluate_model(m, ds, {"batch_size": 2}, use_lpips=False)
        print(f"eval run {run}: mse={r['mse']:.8f} n_batches={r['n_batches']} "
              f"horizon_mse={[round(v, 6) for v in r['horizon_mse']]}")


if __name__ == "__main__":
    main()