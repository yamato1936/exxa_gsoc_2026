#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash General_Test/run_phase2_ablations.sh <data_dir>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_DIR="$1"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/general}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/general}"
LATENT_DIM="${LATENT_DIM:-64}"
N_CLUSTERS="${N_CLUSTERS:-4}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMG_SIZE="${IMG_SIZE:-256}"
ROTATION_DEG="${ROTATION_DEG:-180}"

run_experiment() {
  local experiment_name="$1"
  local preprocess_mode="$2"
  local augmentation_mode="$3"
  local experiment_root="${OUTPUT_DIR}/phase2/${experiment_name}"
  local checkpoint_path="${experiment_root}/checkpoints/best_autoencoder.pt"
  local latent_path="${experiment_root}/latents/latent_vectors.npy"

  local train_cmd=(
    python3 General_Test/src/train.py
    --data_dir "${DATA_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --checkpoint_dir "${CHECKPOINT_DIR}"
    --experiment_name "${experiment_name}"
    --preprocess_mode "${preprocess_mode}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --latent_dim "${LATENT_DIM}"
    --img_size "${IMG_SIZE}"
    --rotation_deg "${ROTATION_DEG}"
  )

  if [[ "${augmentation_mode}" == "aug" ]]; then
    train_cmd+=(--use_augmentation)
  else
    train_cmd+=(--disable_augmentation)
  fi

  echo ""
  echo "=== ${experiment_name} ==="
  "${train_cmd[@]}"

  python3 General_Test/src/extract_latents.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --experiment_name "${experiment_name}" \
    --checkpoint_path "${checkpoint_path}"

  python3 General_Test/src/cluster.py \
    --output_dir "${OUTPUT_DIR}" \
    --experiment_name "${experiment_name}" \
    --latent_path "${latent_path}" \
    --n_clusters "${N_CLUSTERS}"
}

run_experiment "percentile_minmax_noaug_ld${LATENT_DIM}_k${N_CLUSTERS}" "percentile_minmax" "noaug"
run_experiment "log_minmax_noaug_ld${LATENT_DIM}_k${N_CLUSTERS}" "log_minmax" "noaug"
run_experiment "log_minmax_aug_ld${LATENT_DIM}_k${N_CLUSTERS}" "log_minmax" "aug"
run_experiment "robust_aug_ld${LATENT_DIM}_k${N_CLUSTERS}" "robust" "aug"
