# EXXA GSoC 2026 — Representation Learning for Astronomical Data

## Overview

This repository implements end-to-end pipelines for the EXXA GSoC 2026 tests:

- General Test: Unsupervised clustering of protoplanetary disk images (FITS)
- Sequential Test: Transit detection from synthetic light curves

The goal is to learn representations that capture meaningful structure in astronomical data, not just optimize model performance.

---

## Design Philosophy

Protoplanetary disks exhibit morphological structures (e.g., rings and gaps) that are linked to planet formation.

These structures are:
- Spatially localized
- Invariant to rotation and intensity scaling
- More important than exact pixel reconstruction

The pipeline is designed to:
1. Learn compact representations of disk morphology
2. Cluster disks based on learned features
3. Analyze whether clusters correspond to physically meaningful structures

---

## General Test — Representation Learning & Clustering

### Approach

Two representation learning methods are compared:

Autoencoder (AE)
- Objective: pixel-level reconstruction
- Limitation: sensitive to noise and viewing angle

Contrastive Learning (SimCLR)
- Objective: instance discrimination with augmentations
- Advantage: learns invariances and captures morphology

---

### Results

| Method | Silhouette |
|--------|-----------|
| Autoencoder | 0.362 |
| SimCLR (3 epochs) | 0.459 |

- Improved cluster separability
- Better grouping by disk structure

---

### Key Insight

Contrastive learning improves clustering, but only in early training.

- 3 epochs → best performance  
- Longer training → degradation  

Interpretation:

Early representations capture global structure,  
while longer training overfits to instance identity and harms clustering.

---

## Physical Interpretation of Clusters

Clusters are analyzed visually using representative samples:

- Cluster 0:
  Smooth disks with no clear gaps  
  → likely no planets

- Cluster 1:
  Single ring or gap structure  
  → possible single planet

- Cluster 2:
  Multiple rings or gaps  
  → possible multi-planet systems

This suggests that the learned representation captures physically meaningful structures.

---

## Failure Analysis

Observed limitations:

- Sensitivity to viewing angle
- Low-contrast disks are harder to separate
- Weak structures may be mis-clustered

Potential improvements:

- Rotation-invariant representations
- Improved normalization
- Physics-informed augmentations

---

## Visualization

UMAP projection:

![UMAP](General_Test/experiments/phase3/simclr_rot15_flip_e3_best/clusters/umap.png)

Cluster mean examples:

![Cluster Mean](General_Test/experiments/phase3/simclr_rot15_flip_e3_best/clusters/cluster_0_mean.png)

---

## Sequential Test — Transit Detection

### Approach

- Synthetic light curves generated from transit simulations
- Model: 1D CNN classifier
- Task: detect the presence of a planet

### Results

- Model successfully detects transit signals under noise conditions

---

## Repository Structure

General_Test/ 
- src/ 
- experiments/

Sequential_Test/ 
- src/ 
- outputs/

data/ 
archive/

---

## Quick Start

### General Test

```bash
cd General_Test

python src/run_baseline.py \
  --data_dir ../data/fits \
  --output_dir experiments/baseline
```
---

### Sequential Test

```bash
cd Sequential_Test

python src/generate_data.py
python src/train.py
python src/evaluate.py
```
---

## Reproducibility

- Experiments are organized under the experiments directory
- Each run includes checkpoints, logs, latent representations, and clustering outputs
- Pipelines can be executed without manual intervention

---

## Conclusion

- Representation quality is critical for clustering astronomical data
- Contrastive learning provides better features than reconstruction
- However, excessive training degrades clustering performance

This work demonstrates that properly learned representations can recover meaningful structure in protoplanetary disks.
