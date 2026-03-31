from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .data import FitsImageDataset, scan_fits_directory
    from .model import ConvAutoencoder
    from .preprocess import PREPROCESS_MODES, extract_preprocess_config, update_preprocess_config
    from .utils import (
        ensure_dir,
        experiment_root_dir,
        project_path,
        resolve_device,
        save_json,
        seed_everything,
        seed_worker,
        stage_output_dir,
    )
except ImportError:
    from data import FitsImageDataset, scan_fits_directory
    from model import ConvAutoencoder
    from preprocess import PREPROCESS_MODES, extract_preprocess_config, update_preprocess_config
    from utils import (
        ensure_dir,
        experiment_root_dir,
        project_path,
        resolve_device,
        save_json,
        seed_everything,
        seed_worker,
        stage_output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract latent vectors for the General Test autoencoder.")
    parser.add_argument("--data_dir", required=True, help="Directory containing FITS files.")
    parser.add_argument(
        "--checkpoint_path",
        default=project_path("checkpoints", "general", "best_autoencoder.pt"),
        help="Path to a trained autoencoder checkpoint.",
    )
    parser.add_argument("--output_dir", default=project_path("outputs", "general"))
    parser.add_argument(
        "--experiment_name",
        default=None,
        help="Optional Phase 2 experiment name. When set, outputs go under outputs/general/phase2/<experiment_name>/latents/.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preprocess_mode", choices=PREPROCESS_MODES, default=None)
    parser.add_argument("--lower_percentile", type=float, default=None)
    parser.add_argument("--upper_percentile", type=float, default=None)
    parser.add_argument("--robust_clip", type=float, default=None)
    return parser


def load_autoencoder_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[ConvAutoencoder, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    input_size = int(config.get("input_size", 256))
    latent_dim = int(config.get("latent_dim", 64))

    model = ConvAutoencoder(input_size=input_size, latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def resolve_preprocess_config_from_checkpoint(
    checkpoint_config: dict[str, Any],
    *,
    mode: str | None = None,
    lower_percentile: float | None = None,
    upper_percentile: float | None = None,
    robust_clip: float | None = None,
    img_size: int | None = None,
) -> dict[str, str | float | int | None]:
    base_config = extract_preprocess_config(checkpoint_config, default_img_size=img_size)
    return update_preprocess_config(
        base_config,
        mode=mode,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        robust_clip=robust_clip,
        img_size=img_size,
    )


def extract_latent_vectors(
    model: ConvAutoencoder,
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
            batch_latents = model.encode(images).cpu().numpy().astype(np.float32)

            latents.append(batch_latents)
            filepaths.extend(batch["filepath"])
            filenames.extend(batch["filename"])

    latent_array = (
        np.concatenate(latents, axis=0)
        if latents
        else np.empty((0, model.latent_dim), dtype=np.float32)
    )
    return latent_array, filepaths, filenames


def sample_id_from_filename(filename: str) -> str:
    return Path(filename).stem


def infer_latent_artifact_paths(
    latent_path: str | Path,
    *,
    metadata_csv: str | Path | None = None,
    metadata_json: str | Path | None = None,
    sample_ids_path: str | Path | None = None,
) -> dict[str, Path]:
    latent_path = Path(latent_path)
    base_dir = latent_path.parent
    return {
        "latent_path": latent_path,
        "metadata_csv": Path(metadata_csv) if metadata_csv is not None else base_dir / "latent_metadata.csv",
        "metadata_json": Path(metadata_json) if metadata_json is not None else base_dir / "latent_metadata.json",
        "sample_ids_path": Path(sample_ids_path) if sample_ids_path is not None else base_dir / "sample_ids.npy",
    }


def save_latent_manifest(
    filepaths: list[str],
    filenames: list[str],
    out_path: str | Path,
) -> list[str]:
    sample_ids = [sample_id_from_filename(filename) for filename in filenames]
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "sample_id", "filename", "filepath"])
        for index, (filepath, filename, sample_id) in enumerate(
            zip(filepaths, filenames, sample_ids, strict=True)
        ):
            writer.writerow([index, sample_id, filename, filepath])
    return sample_ids


def load_latent_manifest(
    manifest_path: str | Path,
) -> tuple[list[str], list[str], list[str]]:
    filepaths: list[str] = []
    filenames: list[str] = []
    sample_ids: list[str] = []

    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filepaths.append(row["filepath"])
            filenames.append(row["filename"])
            sample_ids.append(row["sample_id"])

    return filepaths, filenames, sample_ids


def save_latent_artifacts(
    latents: np.ndarray,
    filepaths: list[str],
    filenames: list[str],
    *,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    data_dir: str | Path,
    preprocess_config: dict[str, str | float | int | None],
    skipped_files: list[dict[str, str]],
    checkpoint_config: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    artifact_paths = infer_latent_artifact_paths(output_dir / "latent_vectors.npy")
    sample_ids = save_latent_manifest(filepaths, filenames, artifact_paths["metadata_csv"])

    np.save(artifact_paths["latent_path"], latents.astype(np.float32))
    np.save(artifact_paths["sample_ids_path"], np.asarray(sample_ids))

    stats = {
        "shape": list(latents.shape),
        "min": float(latents.min()) if latents.size else 0.0,
        "max": float(latents.max()) if latents.size else 0.0,
        "mean": float(latents.mean()) if latents.size else 0.0,
        "std": float(latents.std()) if latents.size else 0.0,
    }

    metadata = {
        "data_dir": str(Path(data_dir).resolve()),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "num_images": int(len(filepaths)),
        "num_skipped_files": int(len(skipped_files)),
        "skipped_files": skipped_files,
        "preprocess": preprocess_config,
        "model": {
            "input_size": int(checkpoint_config.get("input_size", preprocess_config.get("img_size", 256))),
            "latent_dim": int(checkpoint_config.get("latent_dim", latents.shape[1] if latents.ndim == 2 else 0)),
        },
        "latent_stats": stats,
        "artifacts": {key: str(value) for key, value in artifact_paths.items()},
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    save_json(metadata, artifact_paths["metadata_json"])

    return {key: str(value) for key, value in artifact_paths.items()}


def run_latent_extraction(args: argparse.Namespace) -> dict[str, str | int]:
    seed_everything(args.seed)

    experiment_root = experiment_root_dir(args.output_dir, args.experiment_name)
    output_dir = stage_output_dir(args.output_dir, args.experiment_name, "latents")
    device = resolve_device(args.device)

    model, checkpoint = load_autoencoder_from_checkpoint(args.checkpoint_path, device)
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
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )

    latents, filepaths, filenames = extract_latent_vectors(model, loader, device)
    artifacts = save_latent_artifacts(
        latents,
        filepaths,
        filenames,
        output_dir=output_dir,
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        preprocess_config=preprocess_config,
        skipped_files=skipped_files,
        checkpoint_config=checkpoint_config,
    )

    print(f"Latent extraction complete. Experiment root: {experiment_root}")
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
    run_latent_extraction(args)


if __name__ == "__main__":
    main()
