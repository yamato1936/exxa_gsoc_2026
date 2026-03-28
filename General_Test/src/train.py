from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .data import FitsImageDataset, save_sample_input_grid, scan_fits_directory, train_val_split
    from .model import ConvAutoencoder
    from .utils import (
        ensure_dir,
        project_path,
        resolve_device,
        save_json,
        save_loss_curve,
        save_reconstruction_examples,
        seed_everything,
        seed_worker,
    )
except ImportError:
    from data import FitsImageDataset, save_sample_input_grid, scan_fits_directory, train_val_split
    from model import ConvAutoencoder
    from utils import (
        ensure_dir,
        project_path,
        resolve_device,
        save_json,
        save_loss_curve,
        save_reconstruction_examples,
        seed_everything,
        seed_worker,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the General Test autoencoder baseline.")
    parser.add_argument("--data_dir", required=True, help="Directory containing FITS files.")
    parser.add_argument("--output_dir", default=project_path("outputs", "general"))
    parser.add_argument("--checkpoint_dir", default=project_path("checkpoints", "general"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lower_percentile", type=float, default=1.0)
    parser.add_argument("--upper_percentile", type=float, default=99.0)
    parser.add_argument(
        "--disable_augmentation",
        action="store_true",
        help="Disable the default light flip/rotation augmentation during training.",
    )
    return parser


def run_epoch(
    model: ConvAutoencoder,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    losses: list[float] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        reconstructions = model(images)
        loss = criterion(reconstructions, images)

        if is_training:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else float("nan")


def collect_reconstructions(
    model: ConvAutoencoder,
    loader: DataLoader,
    device: torch.device,
    max_images: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            reconstructions = model(images)
            originals = images.cpu().numpy().squeeze(1)
            reconstructions = reconstructions.cpu().numpy().squeeze(1)
            return originals[:max_images], reconstructions[:max_images]

    return np.empty((0, 0, 0), dtype=np.float32), np.empty((0, 0, 0), dtype=np.float32)


def train_autoencoder(args: argparse.Namespace) -> dict[str, str | float | int]:
    seed_everything(args.seed)

    output_dir = ensure_dir(args.output_dir)
    checkpoint_dir = ensure_dir(args.checkpoint_dir)
    checkpoint_path = checkpoint_dir / "best_autoencoder.pt"
    device = resolve_device(args.device)

    records, skipped_files = scan_fits_directory(args.data_dir)
    if not records:
        raise SystemExit("No valid FITS files were found in the provided data directory.")

    train_records, val_records, split_info = train_val_split(records, args.val_ratio, args.seed)
    augmentation_enabled = not args.disable_augmentation

    save_sample_input_grid(
        records=records,
        output_path=output_dir / "sample_inputs.png",
        img_size=args.img_size,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
    )

    train_dataset = FitsImageDataset(
        train_records,
        img_size=args.img_size,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        augment=augmentation_enabled,
    )
    val_dataset = FitsImageDataset(
        val_records,
        img_size=args.img_size,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        augment=False,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = ConvAutoencoder(input_size=args.img_size, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(
        f"Training on {len(train_records)} images with validation on {len(val_records)} images "
        f"using device={device}."
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, criterion, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, device, criterion, optimizer=None)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = float(val_loss)
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "config": {
                        "input_size": int(args.img_size),
                        "latent_dim": int(args.latent_dim),
                        "lower_percentile": float(args.lower_percentile),
                        "upper_percentile": float(args.upper_percentile),
                        "augmentation_enabled": bool(augmentation_enabled),
                    },
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping triggered at epoch {epoch} after "
                f"{args.patience} epochs without validation improvement."
            )
            break

    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    originals, reconstructions = collect_reconstructions(model, val_loader, device)

    save_loss_curve(
        train_losses=[entry["train_loss"] for entry in history],
        val_losses=[entry["val_loss"] for entry in history],
        out_path=output_dir / "loss_curve.png",
        best_epoch=best_epoch,
    )
    save_reconstruction_examples(
        originals=originals,
        reconstructions=reconstructions,
        out_path=output_dir / "reconstruction_examples.png",
    )

    train_history = {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "epochs_ran": len(history),
    }
    save_json(train_history, output_dir / "train_history.json")

    dataset_summary = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "num_discovered_fits_files": len(records) + len(skipped_files),
        "num_valid_files": len(records),
        "num_skipped_files": len(skipped_files),
        "skipped_files": skipped_files,
        "split": split_info,
        "normalization": {
            "strategy": "per-image percentile clipping then min-max scaling",
            "lower_percentile": float(args.lower_percentile),
            "upper_percentile": float(args.upper_percentile),
            "output_range": [0.0, 1.0],
        },
        "image_size_used_for_training": int(args.img_size),
        "latent_dim": int(args.latent_dim),
        "augmentation": {
            "enabled": bool(augmentation_enabled),
            "details": "random horizontal/vertical flips and random 90-degree rotations",
        },
        "checkpoint_path": str(checkpoint_path),
        "sample_input_figure": str(output_dir / "sample_inputs.png"),
        "reconstruction_figure": str(output_dir / "reconstruction_examples.png"),
    }
    save_json(dataset_summary, output_dir / "dataset_summary.json")

    print(
        f"Training complete. Best validation loss={best_val_loss:.6f} at epoch {best_epoch}. "
        f"Checkpoint saved to {checkpoint_path}."
    )

    return {
        "checkpoint_path": str(checkpoint_path),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "num_valid_files": len(records),
    }


def main(cli_args: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(cli_args)
    train_autoencoder(args)


if __name__ == "__main__":
    main()
