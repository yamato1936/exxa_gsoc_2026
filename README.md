# EXXA GSoC 2026 Submission

This repository contains my submission for the ML4SCI EXXA GSoC 2026 tests.

It includes:

- **General Test**: unsupervised clustering of synthetic ALMA protoplanetary disk images (`.fits`)
- **Sequential Test**: simulated transit curve classification for exoplanet detection

The code is designed to be runnable with minimal user intervention and to produce clear visual outputs for quick evaluation. For the Sequential Test specifically, the goal is to simulate transit curves, train a classifier for planet presence, and report ROC/AUC while handling noisy observations in a reviewer-friendly way.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── checkpoints/
│   ├── general/
│   └── sequential/
├── outputs/
│   ├── general/
│   └── sequential/
├── General_Test/
│   ├── notebooks/
│   │   └── general.ipynb
│   └── src/
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       ├── cluster.py
│       ├── run_baseline.py
│       └── utils.py
└── Sequential_Test/
    ├── notebooks/
    │   └── train.ipynb
    └── src/
        ├── generate_data.py
        ├── model.py
        ├── train.py
        ├── evaluate.py
        ├── infer.py
        └── utils.py
```

## General Test

The General Test baseline trains a convolutional autoencoder on preprocessed FITS images, then clusters the learned latent vectors with k-means. The pipeline is designed for reviewer-friendly execution and saves all artifacts under `outputs/general/`.

### Run From Repository Root

```bash
python General_Test/src/train.py --data_dir path/to/fits_dir
python General_Test/src/cluster.py --data_dir path/to/fits_dir --checkpoint_path checkpoints/general/best_autoencoder.pt
```

Optional one-shot convenience command:

```bash
python General_Test/src/run_baseline.py --data_dir path/to/fits_dir
```

See `General_Test/README.md` for the focused baseline notes and CLI options.

## Sequential Test

The Sequential Test builds synthetic light curves for binary classification: `1` means a periodic transit signal is present and `0` means no planet transit is present. The pipeline is designed for low-friction review: it generates harder positives and negatives, trains a compact 1D CNN, tunes the final decision threshold on the validation split, and saves plots plus CSV/JSON outputs for quick inspection.

### Run From Repository Root

```bash
python3 Sequential_Test/src/generate_data.py
python3 Sequential_Test/src/train.py
python3 Sequential_Test/src/evaluate.py
python3 Sequential_Test/src/infer.py --input_npy sample_curve.npy
```

### What Each Step Produces

- `generate_data.py` writes `outputs/sequential/X_{train,val,test}.npy`, `y_{train,val,test}.npy`, `synthetic_examples.png`, and `dataset_summary.csv`.
- `train.py` writes `checkpoints/sequential/best_model.pt`, `outputs/sequential/loss_curve.png`, `outputs/sequential/val_auc_curve.png`, and `outputs/sequential/training_summary.json`.
- `evaluate.py` loads the saved best checkpoint, reports ROC-AUC and Average Precision, tunes the classification threshold on the validation split, and writes `metrics.json`, `predictions.csv`, ROC/PR curves, a tuned-threshold confusion matrix, a threshold sweep plot, and false-positive/false-negative example panels.
- `infer.py` loads `checkpoints/sequential/best_model.pt`, normalizes a single input curve, reports the predicted probability plus labels at threshold `0.5` and the tuned threshold from `outputs/sequential/metrics.json` when available, and saves `outputs/sequential/infer_example.png`.

### Reviewer Notes

- Place pretrained Sequential weights at `checkpoints/sequential/best_model.pt` if you want to skip training.
- Evaluation is intended to support low-friction scoring on withheld data: run inference on any single 1D `.npy` light curve and the script will reuse the saved best checkpoint automatically.
- All Sequential outputs are saved under `outputs/sequential/` so reviewers can inspect the full submission without digging through the code.
