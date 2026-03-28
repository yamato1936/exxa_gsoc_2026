# General Test Baseline

## Overview

This baseline implements a clean unsupervised pipeline for the EXXA General Test:

1. recursively load synthetic ALMA `.fits` files
2. extract only the first image plane from each cube
3. apply deterministic preprocessing
4. train a convolutional autoencoder
5. cluster the latent vectors with k-means
6. save 2D embeddings and representative examples per cluster

The goal here is clarity and reproducibility rather than a highly optimized method.

## Expected Data

Each FITS file is expected to contain a 4-layer continuum cube, typically shaped like `[4, 600, 600]` or a close variant with singleton dimensions. The pipeline:

- safely squeezes singleton axes
- takes only the first slice
- skips corrupted or unreadable files with warnings
- records skipped files in the output JSON summaries

## Preprocessing

Each image is preprocessed independently:

- convert to `float32`
- replace `NaN` and `inf` values safely
- clip intensities to the 1st and 99th percentiles
- scale the clipped image to `[0, 1]`
- resize to `img_size` for the autoencoder input

This same preprocessing is reused during clustering so training and analysis stay consistent.

## Baseline Model

The model is a small convolutional autoencoder:

- 4 downsampling convolution blocks in the encoder
- latent dimension configurable with `--latent_dim`
- mirrored transposed-convolution decoder
- sigmoid output because inputs are normalized to `[0, 1]`

Light training-time augmentation is enabled by default:

- random horizontal flip
- random vertical flip
- random 90-degree rotation

Disable it with `--disable_augmentation` if you want a stricter deterministic baseline.

## Run From Repository Root

Train the autoencoder:

```bash
python3 General_Test/src/train.py --data_dir path/to/fits_dir
```

Cluster the latent vectors with the trained encoder:

```bash
python3 General_Test/src/cluster.py --data_dir path/to/fits_dir --checkpoint_path checkpoints/general/best_autoencoder.pt
```

Run the full baseline in one command:

```bash
python3 General_Test/src/run_baseline.py --data_dir path/to/fits_dir
```

## Useful Options

Common options for training:

```bash
python3 General_Test/src/train.py \
  --data_dir path/to/fits_dir \
  --epochs 40 \
  --batch_size 16 \
  --latent_dim 64 \
  --img_size 256 \
  --device auto
```

Common options for clustering:

```bash
python3 General_Test/src/cluster.py \
  --data_dir path/to/fits_dir \
  --checkpoint_path checkpoints/general/best_autoencoder.pt \
  --n_clusters 4 \
  --embedding_method umap
```

If UMAP import fails at runtime, the script automatically falls back to PCA and records that in `clustering_summary.json`.

## Outputs

Training writes:

- `checkpoints/general/best_autoencoder.pt`
- `outputs/general/sample_inputs.png`
- `outputs/general/dataset_summary.json`
- `outputs/general/train_history.json`
- `outputs/general/loss_curve.png`
- `outputs/general/reconstruction_examples.png`

Clustering writes:

- `outputs/general/latent_vectors.npy`
- `outputs/general/cluster_labels.npy`
- `outputs/general/cluster_assignments.csv`
- `outputs/general/latent_clusters.png`
- `outputs/general/clustering_summary.json`
- `outputs/general/cluster_<id>_examples.png`

## Notes

- CPU execution is supported.
- CUDA is used automatically when available unless `--device cpu` is passed.
- This is an autoencoder-plus-k-means baseline only. Contrastive learning and stronger invariance methods are intentionally left for later iterations.
