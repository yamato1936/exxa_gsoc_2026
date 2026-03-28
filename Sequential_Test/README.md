# Sequential Test — Transit Detection from Synthetic Light Curves

## Overview

This task focuses on detecting exoplanet transits from 1D light curves.
We simulate realistic transit signals with noise and train a binary classifier to determine whether a planet is present.

The pipeline is fully automated:

* synthetic data generation
* model training
* evaluation with ROC/AUC
* threshold tuning
* inference on new data

---

## Problem Definition

Given a time-series light curve:

* **Input**: normalized flux over time (1D array)
* **Output**: probability of planet transit (binary classification)

Label:

* `1`: transit signal present
* `0`: no transit

---

## Synthetic Data Generation

We generate light curves designed to approximate real observational challenges.

### Signal Modeling

Each positive sample includes:

* periodic transit dips
* varying **depth** (planet size proxy)
* varying **duration**
* random **phase shift**

### Noise & Artifacts

To simulate realistic conditions:

* Gaussian noise
* baseline trends (low-frequency drift)
* random spikes / irregular fluctuations

### Hard Negatives

Negative samples are not purely random:

* quasi-periodic signals
* sharp noise spikes
* trend-dominated curves

This prevents the model from learning trivial patterns.

---

## Model

We use a lightweight **1D CNN**:

* captures local dip structures
* efficient and fast to train
* robust to small temporal shifts

Rationale:

* transit signals are local patterns (dips)
* CNNs are well-suited for this structure

---

## Training

* Loss: Binary Cross Entropy
* Early stopping based on **validation AUC**
* Best model checkpoint saved

```bash
python3 Sequential_Test/src/train.py
```

---

## Evaluation

We evaluate using threshold-independent and threshold-dependent metrics.

### Metrics

* **ROC-AUC** (primary metric)
* **Average Precision (AP)**
* **F1 score (for threshold tuning)**

### Results

* Test ROC-AUC: **0.9006**
* Average Precision: **0.9222**
* Best validation AUC: **0.9104**

```bash
python3 Sequential_Test/src/evaluate.py
```

---

## Threshold Selection

The default threshold (0.5) is suboptimal under noisy conditions.

We tune the threshold on the validation set to maximize F1:

* Best threshold: **0.185**

Reason:

* improves recall for weak or noisy transit signals
* reduces false negatives in ambiguous cases

---

## Error Analysis

We explicitly analyze failure cases.

### False Negatives

Common patterns:

* shallow transit depth
* high noise masking periodic dips

### False Positives

Common patterns:

* sharp noise spikes misclassified as transits
* strong trends mistaken for periodic signals

### Insight

These suggest potential improvements:

* detrending preprocessing
* periodicity-aware models (e.g., Fourier features, transformers)
* harder negative mining

---

## Outputs

All outputs are saved to:

```
outputs/sequential/
```

Includes:

* ROC curve
* Precision-Recall curve
* confusion matrix (tuned threshold)
* threshold sweep
* false positive / negative examples
* training curves

---

## Inference

Run inference on a new light curve:

```bash
python3 Sequential_Test/src/infer.py --input_npy sample_curve.npy
```

Output:

* predicted probability
* label @ 0.5 threshold
* label @ tuned threshold
* visualization plot

---

## Reproducibility

Full pipeline:

```bash
python Sequential_Test/src/generate_data.py
python Sequential_Test/src/train.py
python Sequential_Test/src/evaluate.py
```

Optional:

* use pretrained weights at
  `checkpoints/sequential/best_model.pt`

Notes:

* CPU execution is supported
* CUDA is optional

---

## Design Summary

* realistic synthetic data > toy data
* robustness to noise is prioritized
* threshold tuning reflects real-world usage
* full pipeline reproducibility is enforced

---

## Future Work

* incorporate physically grounded simulators (e.g., PyTransit)
* improve noise modeling
* explore sequence models (RNN / Transformer)
* calibrate probabilities for deployment scenarios

---
