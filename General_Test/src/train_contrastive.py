from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .clustering_utils import assign_clusters, compute_silhouette
    from .data import (
        ContrastivePairDataset,
        FitsImageDataset,
        save_sample_input_grid,
        scan_fits_directory,
        train_val_split,
    )
    from .model import ContrastiveModel
    from .preprocess import PREPROCESS_MODES, build_preprocess_config
    from .utils import (
        checkpoint_dir_for_run,
        experiment_root_dir,
        project_path,
        resolve_device,
        save_image_grid,
        save_json,
        save_loss_curve,
        seed_everything,
        seed_worker,
        stage_output_dir,
    )
except ImportError:
    from clustering_utils import assign_clusters, compute_silhouette
    from data import ContrastivePairDataset, FitsImageDataset, save_sample_input_grid, scan_fits_directory, train_val_split
    from model import ContrastiveModel
    from preprocess import PREPROCESS_MODES, build_preprocess_config
    from utils import (
        checkpoint_dir_for_run,
        experiment_root_dir,
        project_path,
        resolve_device,
        save_image_grid,
        save_json,
        save_loss_curve,
        seed_everything,
        seed_worker,
        stage_output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a SimCLR-style contrastive model for the General Test.")
    parser.add_argument("--data_dir", required=True, help="Directory containing FITS files.")
    parser.add_argument("--output_dir", default=project_path("outputs", "general"))
    parser.add_argument("--checkpoint_dir", default=project_path("checkpoints", "general"))
    parser.add_argument(
        "--experiment_name",
        default="simclr_rot30_latent128",
        help="Phase 3 experiment name. Outputs go under outputs/general/phase3/<experiment_name>/.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--objective", choices=["simclr", "simsiam"], default="simclr")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--preprocess_mode", choices=PREPROCESS_MODES, default="log_minmax")
    parser.add_argument("--lower_percentile", type=float, default=1.0)
    parser.add_argument("--upper_percentile", type=float, default=99.5)
    parser.add_argument("--robust_clip", type=float, default=5.0)

    augmentation_group = parser.add_mutually_exclusive_group()
    augmentation_group.add_argument(
        "--use_augmentation",
        dest="use_augmentation",
        action="store_true",
        help="Enable contrastive view augmentation.",
    )
    augmentation_group.add_argument(
        "--disable_augmentation",
        dest="use_augmentation",
        action="store_false",
        help="Disable contrastive augmentation and use identical paired views.",
    )
    parser.set_defaults(use_augmentation=True)
    parser.add_argument("--rotation_deg", type=float, default=30.0)
    parser.add_argument("--hflip_prob", type=float, default=0.5)
    parser.add_argument("--vflip_prob", type=float, default=0.5)
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument(
        "--selection_metric",
        choices=["val_loss", "silhouette"],
        default="silhouette",
        help="Metric used for checkpoint selection and early stopping.",
    )
    parser.add_argument("--selection_n_clusters", type=int, default=4)
    parser.add_argument(
        "--selection_clustering_method",
        choices=["kmeans", "spectral"],
        default="kmeans",
    )
    parser.add_argument(
        "--selection_split",
        choices=["val", "all"],
        default="val",
        help="Dataset split used to compute the silhouette selection metric.",
    )
    return parser


def nt_xent_loss(
    projections_one: torch.Tensor,
    projections_two: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if projections_one.shape != projections_two.shape:
        raise ValueError("Projection tensors must have matching shapes.")

    batch_size = projections_one.shape[0]
    if batch_size < 2:
        raise ValueError("NT-Xent loss requires batch_size >= 2.")

    z_one = F.normalize(projections_one, dim=1)
    z_two = F.normalize(projections_two, dim=1)
    embeddings = torch.cat([z_one, z_two], dim=0)

    similarity = torch.matmul(embeddings, embeddings.T) / float(temperature)
    similarity = similarity.masked_fill(
        torch.eye(2 * batch_size, device=similarity.device, dtype=torch.bool),
        float("-inf"),
    )

    targets = torch.arange(batch_size, device=similarity.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)
    return F.cross_entropy(similarity, targets)


def simsiam_loss(
    predictions_one: torch.Tensor,
    projections_two: torch.Tensor,
    predictions_two: torch.Tensor,
    projections_one: torch.Tensor,
) -> torch.Tensor:
    predictions_one = F.normalize(predictions_one, dim=1)
    projections_two = F.normalize(projections_two.detach(), dim=1)
    predictions_two = F.normalize(predictions_two, dim=1)
    projections_one = F.normalize(projections_one.detach(), dim=1)

    loss_one = -(predictions_one * projections_two).sum(dim=1).mean()
    loss_two = -(predictions_two * projections_one).sum(dim=1).mean()
    return 0.5 * (loss_one + loss_two)


def compute_ssl_loss(
    model: ContrastiveModel,
    view_one: torch.Tensor,
    view_two: torch.Tensor,
    *,
    objective: str,
    temperature: float,
) -> torch.Tensor:
    _, projections_one = model(view_one)
    _, projections_two = model(view_two)

    if objective == "simclr":
        return nt_xent_loss(projections_one, projections_two, temperature)
    if objective == "simsiam":
        predictions_one = model.predict(projections_one)
        predictions_two = model.predict(projections_two)
        return simsiam_loss(predictions_one, projections_two, predictions_two, projections_one)
    raise ValueError(f"Unsupported objective: {objective}")


def run_epoch(
    model: ContrastiveModel,
    loader: DataLoader,
    device: torch.device,
    objective: str,
    temperature: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    losses: list[float] = []

    for batch in loader:
        view_one = batch["view_one"].to(device, non_blocking=True)
        view_two = batch["view_two"].to(device, non_blocking=True)

        if view_one.shape[0] < 2:
            continue

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        loss = compute_ssl_loss(
            model,
            view_one,
            view_two,
            objective=objective,
            temperature=temperature,
        )

        if is_training:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().item()))

    return {"loss": float(np.mean(losses)) if losses else float("nan")}


def extract_encoder_latents(
    model: ContrastiveModel,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    latents: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            batch_latents = model.encode(images)
            batch_latents = F.normalize(batch_latents, dim=1)
            latents.append(batch_latents.cpu().numpy().astype(np.float32))

    if not latents:
        return np.empty((0, model.latent_dim), dtype=np.float32)
    return np.concatenate(latents, axis=0)


def evaluate_representation(
    model: ContrastiveModel,
    loader: DataLoader,
    device: torch.device,
    *,
    clustering_method: str,
    n_clusters: int,
    seed: int,
) -> dict[str, float | None]:
    latents = extract_encoder_latents(model, loader, device)
    if latents.shape[0] < max(2, n_clusters):
        return {"silhouette": None}

    clustering = assign_clusters(
        latents,
        method=clustering_method,
        n_clusters=n_clusters,
        seed=seed,
    )
    silhouette = compute_silhouette(latents, clustering["labels"])
    return {"silhouette": silhouette}


def save_augmented_pair_preview(
    dataset: ContrastivePairDataset,
    out_path: str | Path,
    max_examples: int = 4,
) -> None:
    images: list[np.ndarray] = []
    titles: list[str] = []

    for index in range(min(max_examples, len(dataset))):
        sample = dataset[index]
        base_image = sample["image"].squeeze(0).numpy()
        view_one = sample["view_one"].squeeze(0).numpy()
        view_two = sample["view_two"].squeeze(0).numpy()
        filename = str(sample["filename"])

        images.extend([base_image, view_one, view_two])
        titles.extend(
            [
                f"{filename} raw",
                f"{filename} view_a",
                f"{filename} view_b",
            ]
        )

    save_image_grid(
        images=np.asarray(images, dtype=np.float32),
        out_path=out_path,
        titles=titles,
        suptitle="Contrastive Augmented Views",
        n_cols=3,
    )


def train_contrastive(args: argparse.Namespace) -> dict[str, str | float | int]:
    seed_everything(args.seed)

    if args.batch_size < 2:
        raise SystemExit("Contrastive training requires --batch_size >= 2.")
    if args.temperature <= 0.0:
        raise SystemExit("Contrastive training requires --temperature > 0.")

    experiment_root = experiment_root_dir(args.output_dir, args.experiment_name, phase="phase3")
    output_dir = stage_output_dir(args.output_dir, args.experiment_name, "train", phase="phase3")
    checkpoint_dir = checkpoint_dir_for_run(
        args.output_dir,
        args.checkpoint_dir,
        args.experiment_name,
        phase="phase3",
    )
    checkpoint_path = checkpoint_dir / "best_contrastive.pt"
    device = resolve_device(args.device)
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
    if len(train_records) < 2:
        raise SystemExit("Contrastive training requires at least two training images after splitting.")
    if len(val_records) < 2:
        val_records = list(train_records)
        split_info["contrastive_validation_reuses_training"] = True
    else:
        split_info["contrastive_validation_reuses_training"] = bool(
            split_info.get("validation_reuses_training", False)
        )

    selection_records = list(val_records) if args.selection_split == "val" else list(records)
    if len(selection_records) <= args.selection_n_clusters:
        selection_records = list(records)
        split_info["selection_metric_reuses_full_dataset"] = True
    else:
        split_info["selection_metric_reuses_full_dataset"] = args.selection_split == "all"

    save_sample_input_grid(
        records=records,
        output_path=output_dir / "sample_inputs.png",
        img_size=args.img_size,
        preprocess_mode=args.preprocess_mode,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        robust_clip=args.robust_clip,
    )

    train_dataset = ContrastivePairDataset(
        train_records,
        img_size=args.img_size,
        preprocess_mode=args.preprocess_mode,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        robust_clip=args.robust_clip,
        use_augmentation=args.use_augmentation,
        rotation_deg=args.rotation_deg,
        hflip_prob=args.hflip_prob,
        vflip_prob=args.vflip_prob,
        noise_std=args.noise_std,
    )
    val_dataset = ContrastivePairDataset(
        val_records,
        img_size=args.img_size,
        preprocess_mode=args.preprocess_mode,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        robust_clip=args.robust_clip,
        use_augmentation=args.use_augmentation,
        rotation_deg=args.rotation_deg,
        hflip_prob=args.hflip_prob,
        vflip_prob=args.vflip_prob,
        noise_std=args.noise_std,
    )
    selection_dataset = FitsImageDataset(
        selection_records,
        img_size=args.img_size,
        preprocess_mode=args.preprocess_mode,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        robust_clip=args.robust_clip,
        augment=False,
        rotation_deg=0.0,
        hflip_prob=0.0,
        vflip_prob=0.0,
        noise_std=0.0,
    )

    save_augmented_pair_preview(train_dataset, output_dir / "augmented_pairs.png")

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
    selection_loader = DataLoader(selection_dataset, shuffle=False, **loader_kwargs)

    model = ContrastiveModel(
        input_size=args.img_size,
        latent_dim=args.latent_dim,
        projection_dim=args.projection_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"Training contrastive model on {len(train_records)} images with validation on {len(val_records)} "
        f"using device={device}. Outputs: {output_dir}"
    )

    best_val_loss = float("inf")
    best_selection_score = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args.objective,
            args.temperature,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            args.objective,
            args.temperature,
            optimizer=None,
        )
        eval_metrics = evaluate_representation(
            model,
            selection_loader,
            device,
            clustering_method=args.selection_clustering_method,
            n_clusters=args.selection_n_clusters,
            seed=args.seed,
        )
        silhouette = eval_metrics["silhouette"]
        selection_value = float(val_metrics["loss"])
        selection_value_for_history: float | None = selection_value
        selection_improved = epoch == 1
        if args.selection_metric == "silhouette":
            selection_value_for_history = None if silhouette is None else float(silhouette)
            if silhouette is not None:
                selection_improved = selection_improved or silhouette > best_selection_score + 1e-6
        else:
            selection_improved = selection_improved or val_metrics["loss"] < best_val_loss - 1e-6

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "val_loss": float(val_metrics["loss"]),
                "silhouette_score": None if silhouette is None else float(silhouette),
                "selection_metric_value": selection_value_for_history,
            }
        )

        log_message = (
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f}"
        )
        if silhouette is not None:
            log_message += f" | silhouette={silhouette:.6f}"
        print(log_message)

        if selection_improved:
            best_val_loss = float(val_metrics["loss"])
            if silhouette is not None:
                best_selection_score = float(silhouette)
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_selection_score": None if best_selection_score == float("-inf") else best_selection_score,
                    "best_epoch": best_epoch,
                    "config": {
                        "phase": "phase3",
                        "model_type": "contrastive",
                        "input_size": int(args.img_size),
                        "latent_dim": int(args.latent_dim),
                        "projection_dim": int(args.projection_dim),
                        "temperature": float(args.temperature),
                        "objective": args.objective,
                        "preprocess_mode": args.preprocess_mode,
                        "lower_percentile": float(args.lower_percentile),
                        "upper_percentile": float(args.upper_percentile),
                        "robust_clip": float(args.robust_clip),
                        "preprocess": preprocess_config,
                        "augmentation": {
                            "enabled": bool(args.use_augmentation),
                            "rotation_deg": float(args.rotation_deg),
                            "hflip_prob": float(args.hflip_prob),
                            "vflip_prob": float(args.vflip_prob),
                            "noise_std": float(args.noise_std),
                        },
                        "selection": {
                            "metric": args.selection_metric,
                            "split": args.selection_split,
                            "n_clusters": int(args.selection_n_clusters),
                            "clustering_method": args.selection_clustering_method,
                            "best_silhouette": None if best_selection_score == float("-inf") else best_selection_score,
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

    save_loss_curve(
        train_losses=[entry["train_loss"] for entry in history],
        val_losses=[entry["val_loss"] for entry in history],
        out_path=output_dir / "loss_curve.png",
        best_epoch=best_epoch,
        ylabel="NT-Xent",
        title="Contrastive Training Loss",
    )

    train_history = {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_silhouette": None if best_selection_score == float("-inf") else float(best_selection_score),
        "epochs_ran": len(history),
    }
    save_json(train_history, output_dir / "train_history.json")

    dataset_summary: dict[str, Any] = {
        "phase": "phase3",
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
        "model": {
            "latent_dim": int(args.latent_dim),
            "projection_dim": int(args.projection_dim),
            "temperature": float(args.temperature),
            "objective": args.objective,
        },
        "augmentation": {
            "enabled": bool(args.use_augmentation),
            "rotation_deg": float(args.rotation_deg),
            "hflip_prob": float(args.hflip_prob),
            "vflip_prob": float(args.vflip_prob),
            "noise_std": float(args.noise_std),
        },
        "selection": {
            "metric": args.selection_metric,
            "split": args.selection_split,
            "n_clusters": int(args.selection_n_clusters),
            "clustering_method": args.selection_clustering_method,
            "best_silhouette": None if best_selection_score == float("-inf") else float(best_selection_score),
        },
        "sample_input_figure": str(output_dir / "sample_inputs.png"),
        "augmented_pair_figure": str(output_dir / "augmented_pairs.png"),
        "loss_curve_figure": str(output_dir / "loss_curve.png"),
    }
    save_json(dataset_summary, output_dir / "dataset_summary.json")

    print(
        f"Contrastive training complete. Best epoch={best_epoch}, "
        f"best_val_loss={best_val_loss:.6f}, "
        f"best_silhouette={None if best_selection_score == float('-inf') else round(best_selection_score, 6)}. "
        f"Checkpoint saved to {checkpoint_path}."
    )

    return {
        "checkpoint_path": str(checkpoint_path),
        "best_val_loss": float(best_val_loss),
        "best_silhouette": None if best_selection_score == float("-inf") else float(best_selection_score),
        "best_epoch": int(best_epoch),
        "num_valid_files": len(records),
        "output_dir": str(output_dir),
    }


def main(cli_args: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(cli_args)
    train_contrastive(args)


if __name__ == "__main__":
    main()
