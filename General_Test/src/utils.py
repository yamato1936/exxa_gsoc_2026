import os
import json
import random
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/tmp") / "matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def _cuda_available_safely() -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.cuda.is_available()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> str:
    return str(repo_root().joinpath(*parts))


def experiment_root_dir(
    output_dir: str | Path,
    experiment_name: str | None = None,
    phase: str = "phase2",
) -> Path:
    base_dir = Path(output_dir)
    if experiment_name:
        base_dir = base_dir / phase / experiment_name
    return ensure_dir(base_dir)


def stage_output_dir(
    output_dir: str | Path,
    experiment_name: str | None = None,
    stage: str | None = None,
    phase: str = "phase2",
) -> Path:
    base_dir = experiment_root_dir(output_dir, experiment_name, phase=phase)
    if experiment_name and stage:
        base_dir = base_dir / stage
    return ensure_dir(base_dir)


def checkpoint_dir_for_run(
    output_dir: str | Path,
    checkpoint_dir: str | Path,
    experiment_name: str | None = None,
    phase: str = "phase2",
) -> Path:
    if experiment_name:
        return ensure_dir(experiment_root_dir(output_dir, experiment_name, phase=phase) / "checkpoints")
    return ensure_dir(checkpoint_dir)


def build_experiment_name(
    preprocess_mode: str,
    use_augmentation: bool,
    latent_dim: int,
    n_clusters: int | None = None,
) -> str:
    parts = [
        preprocess_mode,
        "aug" if use_augmentation else "noaug",
        f"ld{int(latent_dim)}",
    ]
    if n_clusters is not None:
        parts.append(f"k{int(n_clusters)}")
    return "_".join(parts)


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_json(path: str | Path, default: Optional[Any] = None) -> Any:
    path_obj = Path(path)
    if not path_obj.exists():
        return default

    with open(path_obj, "r", encoding="utf-8") as handle:
        return json.load(handle)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if _cuda_available_safely():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()

    if requested == "auto":
        return torch.device("cuda" if _cuda_available_safely() else "cpu")

    if requested == "cuda" and not _cuda_available_safely():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")

    return torch.device(requested)


def _truncate_title(title: str, max_chars: int = 24) -> str:
    if len(title) <= max_chars:
        return title
    return f"...{title[-(max_chars - 3):]}"


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite_mask = np.isfinite(image)
    if not finite_mask.any():
        return np.zeros_like(image, dtype=np.float32)

    finite_values = image[finite_mask]
    low = float(np.percentile(finite_values, 1.0))
    high = float(np.percentile(finite_values, 99.0))
    image = np.nan_to_num(image, nan=low, posinf=high, neginf=low)
    if high - low < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def save_loss_curve(
    train_losses: Iterable[float],
    val_losses: Iterable[float],
    out_path: str | Path,
    best_epoch: Optional[int] = None,
    ylabel: str = "MSE",
    title: str = "Autoencoder Reconstruction Loss",
) -> None:
    train_losses = list(train_losses)
    val_losses = list(val_losses)
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", markersize=3, label="train_loss")
    ax.plot(epochs, val_losses, marker="o", markersize=3, label="val_loss")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="tab:green", linestyle="--", alpha=0.7, label="best_epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_reconstruction_examples(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    out_path: str | Path,
    max_images: int = 6,
) -> None:
    originals = np.asarray(originals, dtype=np.float32)[:max_images]
    reconstructions = np.asarray(reconstructions, dtype=np.float32)[:max_images]
    n_images = min(len(originals), len(reconstructions))

    if n_images == 0:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No validation examples available.", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    fig, axes = plt.subplots(2, n_images, figsize=(3.0 * n_images, 6.0))
    axes = np.atleast_2d(axes)

    for idx in range(n_images):
        axes[0, idx].imshow(originals[idx], cmap="inferno")
        axes[0, idx].set_title(f"Original {idx + 1}")
        axes[0, idx].axis("off")

        axes[1, idx].imshow(reconstructions[idx], cmap="inferno")
        axes[1, idx].set_title(f"Reconstruction {idx + 1}")
        axes[1, idx].axis("off")

    fig.suptitle("Validation Reconstructions", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_sample_inputs(
    raw_images: list[np.ndarray],
    processed_images: list[np.ndarray],
    titles: list[str],
    out_path: str | Path,
) -> None:
    n_images = min(len(raw_images), len(processed_images), len(titles))

    if n_images == 0:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No sample FITS inputs available.", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    fig, axes = plt.subplots(2, n_images, figsize=(3.2 * n_images, 6.0))
    axes = np.atleast_2d(axes)

    for idx in range(n_images):
        axes[0, idx].imshow(_normalize_for_display(raw_images[idx]), cmap="inferno")
        axes[0, idx].set_title(_truncate_title(titles[idx]))
        axes[0, idx].axis("off")

        axes[1, idx].imshow(processed_images[idx], cmap="inferno", vmin=0.0, vmax=1.0)
        axes[1, idx].axis("off")

    axes[0, 0].set_ylabel("Raw", fontsize=12)
    axes[1, 0].set_ylabel("Preprocessed", fontsize=12)
    fig.suptitle("Sample General-Test Inputs", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_image_grid(
    images: np.ndarray,
    out_path: str | Path,
    titles: Optional[list[str]] = None,
    suptitle: Optional[str] = None,
    n_cols: int = 3,
) -> None:
    images = np.asarray(images, dtype=np.float32)
    n_images = len(images)

    if n_images == 0:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No examples available.", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    n_cols = max(1, min(n_cols, n_images))
    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, axis in enumerate(axes):
        if idx >= n_images:
            axis.axis("off")
            continue

        axis.imshow(images[idx], cmap="inferno", vmin=0.0, vmax=1.0)
        axis.axis("off")
        if titles is not None and idx < len(titles):
            axis.set_title(_truncate_title(titles[idx]), fontsize=9)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    else:
        fig.tight_layout()

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_single_image(
    image: np.ndarray,
    out_path: str | Path,
    title: Optional[str] = None,
) -> None:
    image = np.asarray(image, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.imshow(image, cmap="inferno", vmin=0.0, vmax=1.0)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_embedding_plot(
    embedding: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    title: str,
    cluster_sizes: Optional[dict[int, int]] = None,
) -> None:
    embedding = np.asarray(embedding, dtype=np.float32)
    labels = np.asarray(labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10")

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        legend_label = f"cluster {int(label)}"
        if cluster_sizes is not None:
            legend_label = f"{legend_label} (n={int(cluster_sizes.get(int(label), 0))})"
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=28,
            alpha=0.85,
            color=cmap(idx % 10),
            label=legend_label,
        )

    ax.set_xlabel("component_1")
    ax.set_ylabel("component_2")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_radial_profile_plot(
    profiles: list[dict[str, Any]],
    out_path: str | Path,
    title: str = "Cluster Mean Radial Profiles",
) -> None:
    if not profiles:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No radial profiles available.", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for profile in profiles:
        ax.plot(
            profile["radius"],
            profile["intensity"],
            linewidth=2.0,
            label=profile["label"],
        )

    ax.set_xlabel("Radius (pixels)")
    ax.set_ylabel("Mean Intensity")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
