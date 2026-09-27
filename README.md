# MMG-ConvLSTM: Multi-Step Video Prediction

A multi-step video prediction framework based on **MMG-ConvLSTM**, combining a **multi-scale encoder**, a **motion residual module**, and a **motion-gated ConvLSTM** for autoregressive video prediction.

The framework predicts 10 future frames from 10 observed frames and supports experiments on both synthetic and real-world video datasets.

The main model uses a two-phase training protocol consisting of an **MSE warm-up phase** followed by a composite **MSE + L1 + GDL** loss.

---

## 1. Key Features

* Multi-scale spatial feature encoding
* Motion residual modeling from consecutive frame features
* Motion-gated ConvLSTM recurrent module
* Autoregressive multi-step video prediction
* Two-phase loss training protocol
* Ablation variants A–D
* Evaluation at both overall and individual prediction horizons
* Support for Moving MNIST and UCF101
* Deterministic video-level dataset splitting
* Checkpoint-based training resume
* Automated evaluation and visualization tools
* Pytest-based implementation tests

---

## 2. Model

The MMG-ConvLSTM architecture consists of four main components:

1. **Multi-scale Encoder**
   Extracts spatial features at multiple resolutions.

2. **Motion Residual Module**
   Computes temporal feature differences between consecutive frames to explicitly represent motion information.

3. **Motion-Gated ConvLSTM**
   Integrates spatial-temporal features and dynamically modulates the motion residual using a learned spatial gate.

4. **Multi-scale Decoder**
   Reconstructs the predicted video frames from the encoded and recurrent representations.

The model operates autoregressively, generating 10 future frames from 10 observed frames.

---

## 3. Project Structure

```text
project_updated/
├── models/        # model architecture and ablation variants
├── datasets/      # Moving MNIST and UCF101 data loaders
├── losses/        # reconstruction and composite loss functions
├── training/      # training loop, scheduler, checkpoints and trainer
├── evaluation/    # evaluation, metrics and visualization
├── configs/       # YAML configuration files
├── scripts/       # training and diagnostic utilities
├── tests/         # automated test suite
├── metrics/       # MSE, MAE, SSIM and LPIPS implementations
├── baselines/     # baseline implementations
├── utils/         # reproducibility, device and utility functions
├── main.py        # main project entry point
├── requirements.txt
├── README.md
└── .gitignore
```

Training outputs, checkpoints, datasets and generated figures are excluded from the repository.

---

## 4. Model Variants

The repository contains four experimental variants:

| Variant | Description                     |
| ------- | ------------------------------- |
| A       | Baseline configuration          |
| B       | Ablation variant                |
| C       | Ablation variant                |
| D       | Full MMG-ConvLSTM configuration |

Variant D is the complete model combining multi-scale encoding, motion residual modeling and motion-gated ConvLSTM processing.

The main model contains **835,297 parameters** for Moving MNIST and **836,451 parameters** for UCF101.

---

## 5. Requirements

* Python 3.10+
* PyTorch
* torchvision
* NumPy
* PyYAML
* OpenCV
* Matplotlib
* LPIPS
* pytest

A CUDA-enabled PyTorch installation is recommended for training.

---

## 6. Setup

### Create a virtual environment

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 7. Data Preparation

### Moving MNIST

Download the standard Moving MNIST sequence file:

```text
mnist_test_seq.npy
```

Place the file under the raw-data directory configured in:

```text
configs/moving_mnist.yaml
```

The experiments use 20-frame sequences with a spatial resolution of 64×64 pixels.

The first 10 frames are used as input and the following 10 frames are used as prediction targets.

### UCF101

Download the official UCF101 video dataset and the corresponding official split files.

The main experiment uses:

```text
trainlist01.txt
testlist01.txt
```

from:

```text
ucfTrainTestlist/
```

Configure the dataset paths in:

```text
configs/ucf101.yaml
```

The UCF101 pipeline performs splitting at the **video level** to prevent overlap between training, validation and test samples.

A deterministic validation subset is constructed from the official training list using an MD5-based procedure.

---

## 8. Training

All training experiments are launched through:

```text
scripts/train.py
```

using a YAML configuration file.

### Moving MNIST — Variant D

```powershell
.venv\Scripts\python -m scripts.train --config configs/moving_mnist.yaml --variant D
```

### Moving MNIST — Ablation Variants

```powershell
.venv\Scripts\python -m scripts.train --config configs/moving_mnist.yaml --variant A

.venv\Scripts\python -m scripts.train --config configs/moving_mnist.yaml --variant B

.venv\Scripts\python -m scripts.train --config configs/moving_mnist.yaml --variant C
```

### UCF101 — Variant D

```powershell
.venv\Scripts\python -m scripts.train --config configs/ucf101.yaml --variant D
```

---

## 9. Training Protocol

The main training protocol consists of two phases.

### Phase 1 — MSE Warm-up

The first 8 epochs use pure MSE loss to establish a stable reconstruction objective.

### Phase 2 — Composite Loss

After the warm-up phase, training continues using a composite loss consisting of:

```text
MSE + L1 + GDL
```

The best checkpoint is selected using **validation MSE**.

At the phase boundary, the best-model state and early-stopping counter are reset so that model selection in the second phase is performed independently.

Additional training settings include:

* Optimizer: AdamW
* Learning rate: 1e-3
* Weight decay: 1e-4
* Batch size: 8
* Gradient clipping: 1.0
* Learning-rate scheduler: cosine
* Random seed: 42
* Automatic checkpointing
* Resume from `latest.pth`
* NaN/invalid-loss protection

---

## 10. Evaluation

A trained model can be evaluated using:

```powershell
.venv\Scripts\python -m evaluation.evaluate --config configs/moving_mnist.yaml --variant-dir outputs/moving_mnist_gdl --variant D
```

For UCF101:

```powershell
.venv\Scripts\python -m evaluation.evaluate --config configs/ucf101.yaml --variant-dir outputs/ucf101 --variant D
```

The evaluation pipeline reports:

* MSE
* MAE
* SSIM
* LPIPS

Metrics are reported both as overall values and separately for prediction horizons from **t+1 to t+10**.

---

## 11. Visualization

The repository provides utilities for qualitative and quantitative visualization.

### Prediction comparison

```powershell
.venv\Scripts\python -m evaluation.compare_demo --config configs/moving_mnist.yaml --run-a outputs/moving_mnist_mse --run-b outputs/moving_mnist_gdl
```

### Loss protocol comparison

```powershell
.venv\Scripts\python -m evaluation.compare_loss_protocols --json-a outputs/moving_mnist_mse/eval_D.json --json-b outputs/moving_mnist_gdl/eval_D.json --out outputs/compare_loss
```

### Prediction grids and error maps

```powershell
.venv\Scripts\python -m evaluation.predictions_demo --config configs/ucf101.yaml --variant-dir outputs/ucf101 --samples 3
```

### Evaluation plots

```powershell
.venv\Scripts\python -m evaluation.plot_single_eval --json outputs/ucf101/eval_D.json --label "UCF101: D (warmup+composite)" --out outputs/ucf101/plots
```

### Training history

```powershell
.venv\Scripts\python -m scripts.print_history outputs/ucf101/history.json
```

---

## 12. Results

The following results correspond to the test-set evaluation of Variant D using the two-phase training protocol.

| Dataset      | SSIM ↑ | LPIPS ↓ |  MAE ↓ |  MSE ↓ |
| ------------ | -----: | ------: | -----: | -----: |
| Moving MNIST | 0.8178 |  0.2852 | 0.0432 | 0.0288 |
| UCF101       | 0.7817 |  0.1422 | 0.0438 | 0.0089 |

### Moving MNIST Ablation

Under the MSE-only training protocol, the observed SSIM values were:

| Variant |   SSIM |
| ------- | -----: |
| A       | 0.5785 |
| B       | 0.5680 |
| C       | 0.6694 |
| D       | 0.6408 |

### Capacity Experiment

A separate capacity experiment used a 5,076,113-parameter model. Its results were lower than those of the base configuration under the corresponding experimental setting, indicating that increasing parameter count alone did not guarantee improved prediction performance.

---

## 13. Tests

The repository includes an automated pytest suite covering model components, datasets, loss functions, metrics, training behavior, evaluation utilities, reproducibility and related functionality.

Run the complete test suite with:

```powershell
.venv\Scripts\python -m pytest tests -v
```

---

## 14. Reproducibility

The implementation includes several mechanisms intended to facilitate reproducible experiments:

* Fixed random seed: 42
* Deterministic video-level dataset splitting
* YAML-based experiment configuration
* Automatic run records
* Saved training histories
* Checkpoint-based resume
* Validation-based model selection
* Separate evaluation outputs
* Automated test suite

Training records are stored under:

```text
outputs/<run>/
```

Typical output files include:

```text
best.pth
latest.pth
history.json
eval_*.json
```

The `outputs/` directory is excluded from version control.

---

## 15. License

This project is provided for research and educational purposes.

