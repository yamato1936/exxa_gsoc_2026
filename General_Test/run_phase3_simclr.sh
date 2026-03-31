#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash General_Test/run_phase3_simclr.sh <data_dir>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_DIR="$1"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/general}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/general}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-simclr_rot30_latent128}"
LATENT_DIM="${LATENT_DIM:-128}"
PROJECTION_DIM="${PROJECTION_DIM:-64}"
N_CLUSTERS="${N_CLUSTERS:-4}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
IMG_SIZE="${IMG_SIZE:-256}"
TEMPERATURE="${TEMPERATURE:-0.1}"
OBJECTIVE="${OBJECTIVE:-simclr}"
PATIENCE="${PATIENCE:-10}"
SEED="${SEED:-42}"
ROTATION_DEG="${ROTATION_DEG:-30}"
HFLIP_PROB="${HFLIP_PROB:-0.5}"
VFLIP_PROB="${VFLIP_PROB:-0.5}"
NOISE_STD="${NOISE_STD:-0.0}"
USE_AUGMENTATION="${USE_AUGMENTATION:-1}"
SELECTION_METRIC="${SELECTION_METRIC:-silhouette}"
SELECTION_CLUSTERS="${SELECTION_CLUSTERS:-${N_CLUSTERS}}"
SELECTION_CLUSTERING_METHOD="${SELECTION_CLUSTERING_METHOD:-kmeans}"
SELECTION_SPLIT="${SELECTION_SPLIT:-val}"
CLUSTERING_METHOD="${CLUSTERING_METHOD:-kmeans}"

EXPERIMENT_ROOT="${OUTPUT_DIR}/phase3/${EXPERIMENT_NAME}"
CHECKPOINT_PATH="${EXPERIMENT_ROOT}/checkpoints/best_contrastive.pt"
LATENT_PATH="${EXPERIMENT_ROOT}/latents/latent_vectors.npy"

TRAIN_CMD=(
  python3 General_Test/src/train_contrastive.py
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --experiment_name "${EXPERIMENT_NAME}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --latent_dim "${LATENT_DIM}"
  --projection_dim "${PROJECTION_DIM}"
  --temperature "${TEMPERATURE}"
  --objective "${OBJECTIVE}"
  --patience "${PATIENCE}"
  --seed "${SEED}"
  --img_size "${IMG_SIZE}"
  --preprocess_mode log_minmax
  --rotation_deg "${ROTATION_DEG}"
  --hflip_prob "${HFLIP_PROB}"
  --vflip_prob "${VFLIP_PROB}"
  --noise_std "${NOISE_STD}"
  --selection_metric "${SELECTION_METRIC}"
  --selection_n_clusters "${SELECTION_CLUSTERS}"
  --selection_clustering_method "${SELECTION_CLUSTERING_METHOD}"
  --selection_split "${SELECTION_SPLIT}"
)

if [[ "${USE_AUGMENTATION}" == "1" ]]; then
  TRAIN_CMD+=(--use_augmentation)
else
  TRAIN_CMD+=(--disable_augmentation)
fi

"${TRAIN_CMD[@]}"

python3 General_Test/src/extract_contrastive_latents.py \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --seed "${SEED}" \
  --l2_normalize_latents

python3 General_Test/src/cluster.py \
  --output_dir "${OUTPUT_DIR}" \
  --phase phase3 \
  --experiment_name "${EXPERIMENT_NAME}" \
  --clustering_method "${CLUSTERING_METHOD}" \
  --latent_path "${LATENT_PATH}" \
  --seed "${SEED}" \
  --n_clusters "${N_CLUSTERS}"
