# Sequential Test — Transit Detection

## Overview

Detect exoplanet transit signals from synthetic light curves.

We focus on realistic data generation and robust classification under noise.

---

## Problem

Input:
- 1D light curve (flux over time)

Output:
- probability of transit (binary classification)

---

## Key Idea

Performance comes from **data design**, not model complexity.

- realistic noise modeling
- hard negative sampling
- threshold optimization

---

## Data Generation

We simulate realistic observational conditions:

- periodic transit dips (variable depth, duration, phase)
- Gaussian noise
- baseline drift
- random spikes

### Hard Negatives

Negative samples include:

- quasi-periodic signals
- spike-dominated curves
- trend-heavy signals

This prevents trivial classification.

---

## Model

Lightweight 1D CNN

- captures local dip structure
- efficient and stable
- robust to small shifts

---

## Training

- Loss: Binary Cross Entropy
- Early stopping on validation AUC
- Best checkpoint saved

---

## Results

- Test ROC-AUC: 0.9006
- Average Precision: 0.9222
- Best validation AUC: 0.9104

---

## Threshold Optimization

Best threshold: 0.185

Reason:
- improves recall under noisy conditions
- reduces missed weak transit signals

---

## Error Analysis

### False Negatives
- shallow transit
- noise masking signal

### False Positives
- spikes misclassified as transits
- strong trends mistaken as periodicity

---

## Outputs

Stored in:

    outputs/

Includes:

- ROC curve
- PR curve
- confusion matrix
- threshold sweep
- error examples

---

## Reproducibility

Run full pipeline:

```bash
    python src/generate_data.py
    python src/train.py
    python src/evaluate.py
```

Inference:

```bash
    python src/infer.py --input_npy sample_curve.npy
```

---

## Design Summary

- realistic synthetic data > toy data
- robustness to noise is critical
- threshold tuning is essential for deployment