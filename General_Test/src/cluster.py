from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

try:
    from .data import FitsImageDataset, scan_fits_directory
    from .model import ConvAutoencoder
    from .utils import ensure_dir, project_path, resolve_device, save_embedding_plot, save_image_grid, save_json, seed_everything, seed_worker
except ImportError:
    from data import FitsImageDataset, scan_fits_directory
    from model import ConvAutoencoder
    from utils import ensure_dir, project_path, resolve_device, save_embedding_plot, save_image_grid, save_json, seed_everything, seed_worker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster latent vectors for the General Test baseline.")
    parser.add_argument("--data_dir", required=True, help="Directory containing FITS files.")
    parser.add_argument("--checkpoint_path", default=project_path("checkpoints", "general", "best_autoencoder.pt"))
    parser.add_argument("--output_dir", default=project_path("outputs", "general"))
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding_method", choices=["umap", "pca"], default="umap")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser


def _load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[ConvAutoencoder, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    input_size = int(config.get("input_size", 256))
    latent_dim = int(config.get("latent_dim", 64))

    model = ConvAutoencoder(input_size=input_size, latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def extract_latents(
    model: ConvAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    latents: list[np.ndarray] = []
    images: list[np.ndarray] = []
    filepaths: list[str] = []
    filenames: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_images = batch["image"].to(device, non_blocking=True)
            batch_latents = model.encode(batch_images).cpu().numpy().astype(np.float32)

            latents.append(batch_latents)
            images.append(batch_images.cpu().numpy().astype(np.float32))
            filepaths.extend(batch["filepath"])
            filenames.extend(batch["filename"])

    latent_array = np.concatenate(latents, axis=0) if latents else np.empty((0, model.latent_dim), dtype=np.float32)
    image_array = np.concatenate(images, axis=0) if images else np.empty((0, 1, model.input_size, model.input_size), dtype=np.float32)
    return latent_array, image_array.squeeze(1), filepaths, filenames


def compute_embedding(
    latents: np.ndarray,
    method: str,
    seed: int,
) -> tuple[np.ndarray, str]:
    if latents.shape[0] < 2:
        return np.zeros((latents.shape[0], 2), dtype=np.float32), "degenerate"

    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=seed)
            return reducer.fit_transform(latents).astype(np.float32), "umap"
        except Exception as exc:
            print(f"UMAP unavailable or failed ({exc}); falling back to PCA.")

    reducer = PCA(n_components=2, random_state=seed)
    return reducer.fit_transform(latents).astype(np.float32), "pca"


def save_cluster_assignments(
    filepaths: list[str],
    filenames: list[str],
    labels: np.ndarray,
    out_path: str | Path,
) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filepath", "filename", "cluster_id"])
        for filepath, filename, label in zip(filepaths, filenames, labels, strict=True):
            writer.writerow([filepath, filename, int(label)])


def select_representatives(
    latents: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    top_k: int = 9,
) -> dict[int, np.ndarray]:
    representatives: dict[int, np.ndarray] = {}

    for cluster_id in np.unique(labels):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_latents = latents[cluster_indices]
        distances = np.linalg.norm(cluster_latents - centroids[int(cluster_id)], axis=1)
        order = np.argsort(distances)
        representatives[int(cluster_id)] = cluster_indices[order[:top_k]]

    return representatives


def run_clustering(args: argparse.Namespace) -> dict[str, str | int]:
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = resolve_device(args.device)

    model, checkpoint = _load_model(args.checkpoint_path, device)
    config = checkpoint.get("config", {})
    lower_percentile = float(config.get("lower_percentile", 1.0))
    upper_percentile = float(config.get("upper_percentile", 99.0))

    records, skipped_files = scan_fits_directory(args.data_dir)
    if not records:
        raise SystemExit("No valid FITS files were found in the provided data directory.")
    if args.n_clusters > len(records):
        raise SystemExit(
            f"Requested n_clusters={args.n_clusters}, but only {len(records)} valid images are available."
        )

    dataset = FitsImageDataset(
        records,
        img_size=model.input_size,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )

    latents, images, filepaths, filenames = extract_latents(model, loader, device)

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=20)
    labels = kmeans.fit_predict(latents)

    latent_path = output_dir / "latent_vectors.npy"
    labels_path = output_dir / "cluster_labels.npy"
    assignments_path = output_dir / "cluster_assignments.csv"
    embedding_path = output_dir / "latent_clusters.png"

    np.save(latent_path, latents)
    np.save(labels_path, labels.astype(np.int64))
    save_cluster_assignments(filepaths, filenames, labels, assignments_path)

    embedding, embedding_method_used = compute_embedding(latents, args.embedding_method, args.seed)
    save_embedding_plot(
        embedding=embedding,
        labels=labels,
        out_path=embedding_path,
        title=f"Latent Clusters ({embedding_method_used.upper()})",
    )

    representatives = select_representatives(latents, labels, kmeans.cluster_centers_)
    cluster_sizes: dict[str, int] = {}

    for cluster_id in range(args.n_clusters):
        cluster_mask = labels == cluster_id
        cluster_sizes[str(cluster_id)] = int(cluster_mask.sum())

        representative_indices = representatives.get(cluster_id, np.array([], dtype=int))
        representative_images = images[representative_indices]
        representative_titles = [filenames[idx] for idx in representative_indices]
        save_image_grid(
            images=representative_images,
            out_path=output_dir / f"cluster_{cluster_id}_examples.png",
            titles=representative_titles,
            suptitle=f"Cluster {cluster_id} Representative Examples",
            n_cols=3,
        )

    summary = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "num_images": int(len(records)),
        "num_skipped_files": int(len(skipped_files)),
        "skipped_files": skipped_files,
        "latent_dimension": int(latents.shape[1] if latents.ndim == 2 else 0),
        "clustering_method": "kmeans",
        "number_of_clusters": int(args.n_clusters),
        "embedding_method_requested": args.embedding_method,
        "embedding_method_used": embedding_method_used,
        "cluster_sizes": cluster_sizes,
        "artifacts": {
            "latent_vectors": str(latent_path),
            "cluster_labels": str(labels_path),
            "assignments_csv": str(assignments_path),
            "embedding_plot": str(embedding_path),
        },
    }
    save_json(summary, output_dir / "clustering_summary.json")

    print(
        f"Clustering complete for {len(records)} images with k={args.n_clusters}. "
        f"Embedding saved to {embedding_path}."
    )

    return {
        "num_images": len(records),
        "n_clusters": args.n_clusters,
        "embedding_method_used": embedding_method_used,
    }


def main(cli_args: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(cli_args)
    run_clustering(args)


if __name__ == "__main__":
    main()
