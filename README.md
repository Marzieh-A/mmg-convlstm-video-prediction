# MMG-ConvLSTM: Multi-Step Video Prediction

Multi-step video prediction (10 observed → 10 predicted frames) with a
**Multi-scale encoder**, a **Motion residual module** and a
**Motion-Gated ConvLSTM** (MMG-ConvLSTM), trained with a **two-phase loss
protocol**: MSE warm-up followed by a composite MSE+L1+GDL loss.

- Model (variant D): 835,297 parameters (MM) / 836,451 (UCF101)
- Datasets: Moving MNIST (synthetic) and UCF101 (real-world, official split 01)
- Metrics: MSE, MAE, SSIM, LPIPS (AlexNet) — overall and per prediction horizon

-

## 1. Project Structure


project_updated/
├── models/        # architecture: multiscale encoder, motion module,
│                  # motion-gated ConvLSTM, decoder, ablation variants A–D
├── datasets/      # Moving MNIST and UCF101 loaders (video-level splits)
├── losses/        # MSE, L1, GDL, composite loss, WarmupLoss (two-phase)
├── training/      # training loop, model selection, early stopping,
│                  # phase-boundary reset, resume
├── evaluation/    # metrics (overall + per-horizon), comparison demos,
│                  # qualitative grids, plotting
├── configs/       # YAML configuration files per dataset
├── scripts/       # entry points (train.py and utilities)
├── outputs/       # checkpoints, histories, eval JSONs, figures
│                  # (created automatically — not in the repository)
└── tests/         # pytest suite


## 2. Setup

# create and activate a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

Requirements: Python 3.10+, PyTorch (CUDA build recommended),
lpips, matplotlib, numpy, pyyaml, opencv-python, pytest.

---

## 3. Data Preparation

### Moving MNIST
Download `mnist_test_seq.npy` (standard 20-frame, 64×64 version) and place it
under the raw-data path configured in `configs/moving_mnist.yaml`.

### UCF101
Download the official UCF101 videos and the **split 01** lists
(`trainlist01.txt`, `testlist01.txt` from ucfTrainTestlist) and set
`root` / `split_dir` in `configs/ucf101.yaml` accordingly.

The UCF101 loader uses video-level splits: the official train/test lists,
plus a deterministic validation set (10%) carved out of the official train
list via an MD5 hash of the video filename — no data leakage.
Windows are 20 consecutive frames; videos shorter than 20 frames are
looped deterministically.

---

## 4. Training

All experiments are launched through `scripts/train.py` with a YAML config.

```bash
# Moving MNIST — variant D (MMG-ConvLSTM), two-phase loss protocol
.venv\Scripts\python -m scripts.train --config configs/moving_mnist.yaml --variant D

# Moving MNIST — ablation variants (MSE-only protocol)
.venv\Scripts\python -m scripts.train --config configs/moving_mnist.yaml --variant A
.venv\Scripts\python -m scripts.train --config configs/moving_mnist.yaml --variant B
.venv\Scripts\python -m scripts.train --config configs/moving_mnist.yaml --variant C

# UCF101 — full model, two-phase loss protocol
.venv\Scripts\python -m scripts.train --config configs/ucf101.yaml --variant D
```

**What the trainer does:**
- Phase 1 (epochs 0–7): pure MSE. Phase 2: composite MSE+L1+GDL.
- Best model selected by **validation MSE** (scale-stable across phases);
  best-model state and early-stopping counter are **reset at the phase
  boundary** (epoch 8).
- Gradient clipping (max-norm 1.0), NaN guard, cosine LR schedule,
  AdamW (lr 1e-3, weight decay 1e-4), batch size 8, seed 42.
- Resume: training automatically continues from `latest.pth` if present.
- Outputs land in `outputs/<run>/`: `best.pth`, `latest.pth`,
  `history.json`, `eval_*.json`, run record (config + seed + date).

---

## 5. Evaluation

```bash
# evaluate a trained variant on the test set
# (writes eval_<variant>.json with overall + per-horizon metrics)
.venv\Scripts\python -m evaluation.evaluate --config configs/moving_mnist.yaml --variant-dir outputs/moving_mnist_gdl --variant D
.venv\Scripts\python -m evaluation.evaluate --config configs/ucf101.yaml --variant-dir outputs/ucf101 --variant D
```

The evaluation JSON contains `mse`, `mae`, `ssim`, `lpips` and
`horizon_<metric>` arrays for steps t+1 … t+10.

---

## 6. Qualitative Results and Figures

```bash
# side-by-side comparison demo of two runs (grids of input/GT/predictions)
.venv\Scripts\python -m evaluation.compare_demo --config configs/moving_mnist.yaml --run-a outputs/moving_mnist_mse --run-b outputs/moving_mnist_gdl

# loss-protocol comparison: horizon curves + overall bars for two eval JSONs
.venv\Scripts\python -m evaluation.compare_loss_protocols --json-a outputs/moving_mnist_mse/eval_D.json --json-b outputs/moving_mnist_gdl/eval_D.json --out outputs/compare_loss

# prediction grids + error maps for one run (n samples)
.venv\Scripts\python -m evaluation.predictions_demo --config configs/ucf101.yaml --variant-dir outputs/ucf101 --samples 3

# single-eval horizon curves + overall bars (per dataset)
.venv\Scripts\python -m evaluation.plot_single_eval --json outputs/ucf101/eval_D.json --label "UCF101: D (warmup+composite)" --out outputs/ucf101/plots

# inspect a training history
.venv\Scripts\python -m scripts.print_history outputs/ucf101/history.json
```

---

## 7. Tests

```bash
.venv\Scripts\python -m pytest tests -v
```

---

## 8. Results Summary (test sets, two-phase protocol, variant D)

| Dataset | SSIM ↑ | LPIPS ↓ | MAE ↓ | MSE ↓ |
|---|---|---|---|---|
| Moving MNIST | 0.8178 | 0.2852 | 0.0432 | 0.0288 |
| UCF101 | 0.7817 | 0.1422 | 0.0438 | 0.0089 |

Ablation (Moving MNIST, MSE-only protocol): A 0.5785 / B 0.5680 /
C 0.6694 / D 0.6408 SSIM. Capacity experiment: a 5,076,113-parameter
model performs worse than the base model — the bottleneck is the training
protocol, not capacity.

---

## 9. Reproducibility

- Fixed seed (42) for all stochastic sources
- Deterministic, video-level dataset splits
- Run records auto-saved (config, seed, date) under `outputs/<run>/`
- Resume from `latest.pth` supports interrupted runs

## 10. Citation

If you use this code, please cite the associated thesis:
*Multi-step video prediction with MMG-ConvLSTM: multi-scale encoder,
motion gate and the effect of the loss function* (M.Sc. thesis).