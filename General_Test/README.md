# General Test — Representation Learning for Disk Clustering

## Goal

Cluster protoplanetary disk images based on morphology without labels.

---

## Pipeline

FITS → preprocessing → encoder → latent → clustering

---

## Phase 2 — Autoencoder (Baseline)

- Objective: reconstruction
- Latent dim: 128
- L2 normalization

Result:
- Silhouette: 0.362

---

## Phase 3 — Contrastive Learning (SimCLR)

### Method

- SimCLR with augmentation

### Augmentation

- rotation (±15°)
- horizontal / vertical flip

---

## Key Result

| Epoch | Silhouette |
|------|-----------|
| 3 | 0.459 (best) |
| 5 | 0.382 |
| 20 | 0.355 |

---

## Why It Works

Early contrastive training captures global structure.

Longer training shifts toward instance discrimination,
which degrades clustering performance.

---

## Final Model

simclr_rot15_flip_e3_best

---

## Reproducibility

### Train

```bash
    python src/train_contrastive.py \
      --data_dir ../data/fits \
      --output_dir experiments/phase3 \
      --experiment_name simclr_rot15_flip_e3_best \
      --batch_size 16 \
      --epochs 3 \
      --latent_dim 128 \
      --projection_dim 64 \
      --temperature 0.1 \
      --preprocess_mode log_minmax \
      --use_augmentation \
      --rotation_deg 15 \
      --hflip_prob 0.5 \
      --vflip_prob 0.5
```

### Extract latents

```bash
    python src/extract_contrastive_latents.py \
      --data_dir ../data/fits \
      --checkpoint_path experiments/phase3/simclr_rot15_flip_e3_best/checkpoints/best_contrastive.pt \
      --output_dir experiments/phase3/simclr_rot15_flip_e3_best/latents \
      --preprocess_mode log_minmax \
      --lower_percentile 1.0 \
      --upper_percentile 99.5 \
      --l2_normalize_latents
```


### Clustering


```bash
    python src/cluster.py \
      --latent_path experiments/phase3/simclr_rot15_flip_e3_best/latents/phase3/simclr_rot30_latent128/latents/latent_vectors.npy \
      --metadata_csv experiments/phase3/simclr_rot15_flip_e3_best/latents/phase3/simclr_rot30_latent128/latents/latent_metadata.csv \
      --metadata_json experiments/phase3/simclr_rot15_flip_e3_best/latents/phase3/simclr_rot30_latent128/latents/latent_metadata.json \
      --output_dir experiments/phase3/simclr_rot15_flip_e3_best/clusters \
      --phase phase3 \
      --n_clusters 4
```