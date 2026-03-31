from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .data import FitsImageDataset, scan_fits_directory
    from .extract_latents import resolve_preprocess_config_from_checkpoint, save_latent_artifacts
    from .model import ContrastiveModel
    from .preprocess import PREPROCESS_MODES
    from .utils import (
        experiment_root_dir,
        project_path,
        resolve_device,
        seed_everything,
        seed_worker,
        stage_output_dir,
    )
except ImportError:
    from data import FitsImageDataset, scan_fits_directory
    from extract_latents import resolve_preprocess_config_from_checkpoint, save_latent_artifacts
    from model import ContrastiveModel
    from preprocess import PREPROCESS_MODES
    from utils import (
        experiment_root_dir,
        project_path,
        resolve_device,
        seed_everything,
        seed_worker,
        stage_output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract encoder latents from a Phase 3 contrastive checkpoint.")
    parser.add_argument("--data_dir", required=True, help="Directory containing FITS files.")
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Path to a trained contrastive checkpoint. Defaults to outputs/general/phase3/<experiment_name>/checkpoints/best_contrastive.pt when --experiment_name is set.",
    )
    parser.add_argument("--output_dir", default=project_path("outputs", "general"))
    parser.add_argument(
        "--experiment_name",
        default="simclr_rot30_latent128",
        help="Phase 3 experiment name. Latent artifacts go under outputs/general/phase3/<experiment_name>/latents/.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preprocess_mode", choices=PREPROCESS_MODES, default=None)
    parser.add_argument("--lower_percentile", type=float, default=None)
    parser.add_argument("--upper_percentile", type=float, default=None)
    parser.add_argument("--robust_clip", type=float, default=None)

    normalization_group = parser.add_mutually_exclusive_group()
    normalization_group.add_argument(
        "--l2_normalize_latents",
        dest="l2_normalize_latents",
        action="store_true",
        help="L2-normalize encoder latents before saving.",
    )
    normalization_group.add_argument(
        "--disable_l2_normalize_latents",
        dest="l2_normalize_latents",
        action="store_false",
        help="Save raw encoder latents without L2 normalization.",
    )
    parser.set_defaults(l2_normalize_latents=True)
    return parser


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint_path is not None:
        return Path(args.checkpoint_path)
    if args.experiment_name:
        return (
            Path(args.output_dir)
            / "phase3"
            / args.experiment_name
            / "checkpoints"
            / "best_contrastive.pt"
        )
    raise SystemExit("Provide --checkpoint_path or --experiment_name for latent extraction.")


def load_contrastive_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[ContrastiveModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    input_size = int(config.get("input_size", 256))
    latent_dim = int(config.get("latent_dim", 128))
    projection_dim = int(config.get("projection_dim", 64))

    model = ContrastiveModel(
        input_size=input_size,
        latent_dim=latent_dim,
        projection_dim=projection_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model, checkpoint


def extract_encoder_latents(
    model: ContrastiveModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, list[str], list[str]]:
    latents: list[np.ndarray] = []
    filepaths: list[str] = []
    filenames: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            batch_latents = model.encode(images)
            latents.append(batch_latents.cpu().numpy().astype(np.float32))
            filepaths.extend(batch["filepath"])
            filenames.extend(batch["filename"])

    latent_array = (
        np.concatenate(latents, axis=0)
        if latents
        else np.empty((0, model.latent_dim), dtype=np.float32)
    )
    return latent_array, filepaths, filenames


def l2_normalize_latents(latents: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, dict[str, float]]:
    if latents.size == 0:
        return latents.astype(np.float32), {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

    tensor = torch.from_numpy(latents.astype(np.float32))
    normalized = F.normalize(tensor, p=2, dim=1, eps=eps).cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(normalized, axis=1)
    return normalized, {
        "min": float(norms.min()),
        "max": float(norms.max()),
        "mean": float(norms.mean()),
        "std": float(norms.std()),
    }


def run_contrastive_latent_extraction(args: argparse.Namespace) -> dict[str, str | int]:
    seed_everything(args.seed)

    checkpoint_path = resolve_checkpoint_path(args)
    experiment_root = experiment_root_dir(args.output_dir, args.experiment_name, phase="phase3")
    output_dir = stage_output_dir(args.output_dir, args.experiment_name, "latents", phase="phase3")
    device = resolve_device(args.device)

    model, checkpoint = load_contrastive_model_from_checkpoint(checkpoint_path, device)
    checkpoint_config = checkpoint.get("config", {})
    preprocess_config = resolve_preprocess_config_from_checkpoint(
        checkpoint_config,
        mode=args.preprocess_mode,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        robust_clip=args.robust_clip,
        img_size=model.input_size,
    )

    records, skipped_files = scan_fits_directory(args.data_dir)
    if not records:
        raise SystemExit("No valid FITS files were found in the provided data directory.")

    dataset = FitsImageDataset(
        records,
        img_size=model.input_size,
        preprocess_mode=str(preprocess_config["mode"]),
        lower_percentile=float(preprocess_config["lower_percentile"]),
        upper_percentile=float(preprocess_config["upper_percentile"]),
        robust_clip=float(preprocess_config["robust_clip"]),
        augment=False,
        rotation_deg=0.0,
        hflip_prob=0.0,
        vflip_prob=0.0,
        noise_std=0.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )

    latents, filepaths, filenames = extract_encoder_latents(model, loader, device)
    norm_stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    if args.l2_normalize_latents:
        latents, norm_stats = l2_normalize_latents(latents)

    artifacts = save_latent_artifacts(
        latents,
        filepaths,
        filenames,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        data_dir=args.data_dir,
        preprocess_config=preprocess_config,
        skipped_files=skipped_files,
        checkpoint_config=checkpoint_config,
        extra_metadata={
            "phase": "phase3",
            "model": {
                "input_size": int(checkpoint_config.get("input_size", model.input_size)),
                "latent_dim": int(checkpoint_config.get("latent_dim", model.latent_dim)),
                "projection_dim": int(checkpoint_config.get("projection_dim", model.projection_dim)),
            },
            "contrastive": {
                "temperature": float(checkpoint_config.get("temperature", 0.1)),
                "l2_normalize_latents": bool(args.l2_normalize_latents),
                "latent_norm_stats": norm_stats,
            },
        },
    )

    print(f"Contrastive latent extraction complete. Experiment root: {experiment_root}")
    print(f"latent shape: {latents.shape}")
    if latents.size:
        print(
            "latent stats:",
            {
                "min": float(latents.min()),
                "max": float(latents.max()),
                "mean": float(latents.mean()),
                "std": float(latents.std()),
            },
        )
        if args.l2_normalize_latents:
            print("latent norm stats:", norm_stats)

    return {
        "num_images": len(filepaths),
        "latent_dim": int(latents.shape[1] if latents.ndim == 2 else 0),
        "latent_path": artifacts["latent_path"],
        "metadata_csv": artifacts["metadata_csv"],
        "metadata_json": artifacts["metadata_json"],
    }


def main(cli_args: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(cli_args)
    run_contrastive_latent_extraction(args)


if __name__ == "__main__":
    main()
