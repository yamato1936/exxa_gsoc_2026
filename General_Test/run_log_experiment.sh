#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash General_Test/run_log_experiment.sh <data_dir>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_DIR="$1"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/general}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/general}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-log_minmax}"
LATENT_DIM="${LATENT_DIM:-64}"
N_CLUSTERS="${N_CLUSTERS:-4}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMG_SIZE="${IMG_SIZE:-256}"
ROTATION_DEG="${ROTATION_DEG:-90}"
RECON_LOSS="${RECON_LOSS:-mse}"
USE_AUGMENTATION="${USE_AUGMENTATION:-1}"
USE_EDGE_LOSS="${USE_EDGE_LOSS:-0}"
EDGE_LOSS_WEIGHT="${EDGE_LOSS_WEIGHT:-0.1}"

EXPERIMENT_ROOT="${OUTPUT_DIR}/phase2/${EXPERIMENT_NAME}"
CHECKPOINT_PATH="${EXPERIMENT_ROOT}/checkpoints/best_autoencoder.pt"
LATENT_PATH="${EXPERIMENT_ROOT}/latents/latent_vectors.npy"

TRAIN_CMD=(
  python3 General_Test/src/train.py
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --experiment_name "${EXPERIMENT_NAME}"
  --preprocess_mode log_minmax
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --latent_dim "${LATENT_DIM}"
  --img_size "${IMG_SIZE}"
  --rotation_deg "${ROTATION_DEG}"
  --recon_loss "${RECON_LOSS}"
)

if [[ "${USE_AUGMENTATION}" == "1" ]]; then
  TRAIN_CMD+=(--use_augmentation)
else
  TRAIN_CMD+=(--disable_augmentation)
fi

if [[ "${USE_EDGE_LOSS}" == "1" ]]; then
  TRAIN_CMD+=(--use_edge_loss --edge_loss_weight "${EDGE_LOSS_WEIGHT}")
fi

"${TRAIN_CMD[@]}"

python3 General_Test/src/extract_latents.py \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --checkpoint_path "${CHECKPOINT_PATH}"

python3 General_Test/src/cluster.py \
  --output_dir "${OUTPUT_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --latent_path "${LATENT_PATH}" \
  --n_clusters "${N_CLUSTERS}"
