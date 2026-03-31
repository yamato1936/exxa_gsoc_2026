from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .data import FitsImageDataset, save_sample_input_grid, scan_fits_directory, train_val_split
    from .model import ConvAutoencoder
    from .preprocess import PREPROCESS_MODES, build_preprocess_config
    from .utils import (
        checkpoint_dir_for_run,
        experiment_root_dir,
        project_path,
        resolve_device,
        save_json,
        save_loss_curve,
        save_reconstruction_examples,
        seed_everything,
        seed_worker,
        stage_output_dir,
    )
except ImportError:
    from data import FitsImageDataset, save_sample_input_grid, scan_fits_directory, train_val_split
    from model import ConvAutoencoder
    from preprocess import PREPROCESS_MODES, build_preprocess_config
    from utils import (
        checkpoint_dir_for_run,
        experiment_root_dir,
        project_path,
        resolve_device,
        save_json,
        save_loss_curve,
        save_reconstruction_examples,
        seed_everything,
        seed_worker,
        stage_output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the General Test autoencoder.")
    parser.add_argument("--data_dir", required=True, help="Directory containing FITS files.")
    parser.add_argument("--output_dir", default=project_path("outputs", "general"))
    parser.add_argument("--checkpoint_dir", default=project_path("checkpoints", "general"))
    parser.add_argument(
        "--experiment_name",
        default=None,
        help="Optional Phase 2 experiment name. When set, outputs go under outputs/general/phase2/<experiment_name>/.",
    )
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
    parser.add_argument("--preprocess_mode", choices=PREPROCESS_MODES, default="percentile_minmax")
    parser.add_argument("--lower_percentile", type=float, default=1.0)
    parser.add_argument("--upper_percentile", type=float, default=99.5)
    parser.add_argument("--robust_clip", type=float, default=5.0)
    parser.add_argument(
        "--recon_loss",
        choices=["mse", "l1"],
        default="mse",
        help="Base reconstruction loss used for autoencoder training.",
    )

    augmentation_group = parser.add_mutually_exclusive_group()
    augmentation_group.add_argument(
        "--use_augmentation",
        dest="use_augmentation",
        action="store_true",
        help="Enable random rotation, horizontal flip, and vertical flip during training.",
    )
    augmentation_group.add_argument(
        "--disable_augmentation",
        dest="use_augmentation",
        action="store_false",
        help="Disable training-time augmentation.",
    )
    parser.set_defaults(use_augmentation=True)
    parser.add_argument(
        "--rotation_deg",
        type=float,
        default=90.0,
        help="Maximum absolute rotation angle used during training augmentation.",
    )
    parser.add_argument(
        "--use_edge_loss",
        action="store_true",
        help="Add a Sobel-gradient reconstruction term to encourage sharper structural reconstructions.",
    )
    parser.add_argument(
        "--edge_loss_weight",
        type=float,
        default=0.1,
        help="Weight for the optional Sobel edge reconstruction loss.",
    )
    return parser


class AutoencoderLoss(nn.Module):
    def __init__(
        self,
        *,
        recon_loss: str = "mse",
        use_edge_loss: bool = False,
        edge_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.recon_loss_name = str(recon_loss)
        self.use_edge_loss = bool(use_edge_loss)
        self.edge_loss_weight = float(edge_loss_weight)
        if self.recon_loss_name == "mse":
            self.reconstruction_loss = nn.MSELoss()
        elif self.recon_loss_name == "l1":
            self.reconstruction_loss = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported recon_loss: {recon_loss}")

        sobel_x = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _sobel_gradients(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grad_x = F.conv2d(tensor, self.sobel_x, padding=1)
        grad_y = F.conv2d(tensor, self.sobel_y, padding=1)
        return grad_x, grad_y

    def forward(
        self,
        reconstructions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        recon_loss = self.reconstruction_loss(reconstructions, targets)

        if self.use_edge_loss:
            grad_x_pred, grad_y_pred = self._sobel_gradients(reconstructions)
            grad_x_target, grad_y_target = self._sobel_gradients(targets)
            edge_loss = F.l1_loss(grad_x_pred, grad_x_target) + F.l1_loss(
                grad_y_pred,
                grad_y_target,
            )
            total_loss = recon_loss + self.edge_loss_weight * edge_loss
        else:
            edge_loss = torch.zeros(
                (),
                device=reconstructions.device,
                dtype=reconstructions.dtype,
            )
            total_loss = recon_loss

        return {
            "loss": total_loss,
            "total_loss": total_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "edge_loss": edge_loss.detach(),
        }


def run_epoch(
    model: ConvAutoencoder,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_losses: list[float] = []
    recon_losses: list[float] = []
    edge_losses: list[float] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        reconstructions = model(images)
        loss_outputs = criterion(reconstructions, images)
        loss = loss_outputs["loss"]

        if is_training:
            loss.backward()
            optimizer.step()

        total_losses.append(float(loss_outputs["total_loss"].item()))
        recon_losses.append(float(loss_outputs["recon_loss"].item()))
        edge_losses.append(float(loss_outputs["edge_loss"].item()))

    if not total_losses:
        return {
            "total_loss": float("nan"),
            "recon_loss": float("nan"),
            "edge_loss": float("nan"),
        }

    return {
        "total_loss": float(np.mean(total_losses)),
        "recon_loss": float(np.mean(recon_losses)),
        "edge_loss": float(np.mean(edge_losses)),
    }


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

    experiment_root = experiment_root_dir(args.output_dir, args.experiment_name)
    output_dir = stage_output_dir(args.output_dir, args.experiment_name, "train")
    checkpoint_dir = checkpoint_dir_for_run(args.output_dir, args.checkpoint_dir, args.experiment_name)
    checkpoint_path = checkpoint_dir / "best_autoencoder.pt"
    device = resolve_device(args.device)
    augmentation_enabled = bool(args.use_augmentation)
    preprocess_config = build_preprocess_config(
        mode=args.preprocess_mode,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        robust_clip=args.robust_clip,
        img_size=args.img_size,
    )

    records, skipped_files = scan_fits_directory(args.data_dir)
    if not records:
        raise SystemExit("No valid FITS files were found in the provided data directory.")

    train_records, val_records, split_info = train_val_split(records, args.val_ratio, args.seed)

    save_sample_input_grid(
        records=records,
        output_path=output_dir / "sample_inputs.png",
        img_size=args.img_size,
        preprocess_mode=args.preprocess_mode,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        robust_clip=args.robust_clip,
    )

    train_dataset = FitsImageDataset(
        train_records,
        img_size=args.img_size,
        preprocess_mode=args.preprocess_mode,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        robust_clip=args.robust_clip,
        augment=augmentation_enabled,
        rotation_deg=args.rotation_deg,
    )
    val_dataset = FitsImageDataset(
        val_records,
        img_size=args.img_size,
        preprocess_mode=args.preprocess_mode,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        robust_clip=args.robust_clip,
        augment=False,
        rotation_deg=0.0,
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
    criterion = AutoencoderLoss(
        recon_loss=args.recon_loss,
        use_edge_loss=args.use_edge_loss,
        edge_loss_weight=args.edge_loss_weight,
    ).to(device)

    print(
        f"Training on {len(train_records)} images with validation on {len(val_records)} images "
        f"using device={device}. Outputs: {output_dir}"
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device, criterion, optimizer=None)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["total_loss"]),
                "val_loss": float(val_metrics["total_loss"]),
                "train_recon_loss": float(train_metrics["recon_loss"]),
                "val_recon_loss": float(val_metrics["recon_loss"]),
                "train_edge_loss": float(train_metrics["edge_loss"]),
                "val_edge_loss": float(val_metrics["edge_loss"]),
            }
        )

        log_message = (
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_metrics['total_loss']:.6f} | "
            f"val_loss={val_metrics['total_loss']:.6f} | "
            f"train_recon={train_metrics['recon_loss']:.6f} | "
            f"val_recon={val_metrics['recon_loss']:.6f}"
        )
        if args.use_edge_loss:
            log_message += (
                f" | train_edge={train_metrics['edge_loss']:.6f}"
                f" | val_edge={val_metrics['edge_loss']:.6f}"
            )
        print(log_message)

        if val_metrics["total_loss"] < best_val_loss - 1e-6:
            best_val_loss = float(val_metrics["total_loss"])
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
                        "preprocess_mode": args.preprocess_mode,
                        "lower_percentile": float(args.lower_percentile),
                        "upper_percentile": float(args.upper_percentile),
                        "robust_clip": float(args.robust_clip),
                        "recon_loss": args.recon_loss,
                        "preprocess": preprocess_config,
                        "augmentation_enabled": bool(augmentation_enabled),
                        "augmentation": {
                            "enabled": bool(augmentation_enabled),
                            "rotation_deg": float(args.rotation_deg),
                            "horizontal_flip": True,
                            "vertical_flip": True,
                        },
                        "loss": {
                            "use_edge_loss": bool(args.use_edge_loss),
                            "edge_loss_weight": float(args.edge_loss_weight),
                        },
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
        "experiment_name": args.experiment_name,
        "experiment_root": str(experiment_root),
        "train_output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "num_discovered_fits_files": len(records) + len(skipped_files),
        "num_valid_files": len(records),
        "num_skipped_files": len(skipped_files),
        "skipped_files": skipped_files,
        "split": split_info,
        "normalization": {
            "mode": args.preprocess_mode,
            "lower_percentile": float(args.lower_percentile),
            "upper_percentile": float(args.upper_percentile),
            "robust_clip": float(args.robust_clip),
            "output_range": [0.0, 1.0],
        },
        "reconstruction_loss": args.recon_loss,
        "image_size_used_for_training": int(args.img_size),
        "latent_dim": int(args.latent_dim),
        "augmentation": {
            "enabled": bool(augmentation_enabled),
            "rotation_deg": float(args.rotation_deg),
            "horizontal_flip": True,
            "vertical_flip": True,
        },
        "loss": {
            "use_edge_loss": bool(args.use_edge_loss),
            "edge_loss_weight": float(args.edge_loss_weight),
        },
        "sample_input_figure": str(output_dir / "sample_inputs.png"),
        "reconstruction_figure": str(output_dir / "reconstruction_examples.png"),
        "loss_curve_figure": str(output_dir / "loss_curve.png"),
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
        "output_dir": str(output_dir),
    }


def main(cli_args: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(cli_args)
    train_autoencoder(args)


if __name__ == "__main__":
    main()
