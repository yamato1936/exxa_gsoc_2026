from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

try:
    from .clustering_utils import assign_clusters, compute_silhouette, evaluate_cluster_stability
    from .data import FitsImageDataset, load_fits_first_slice, scan_fits_directory
    from .extract_latents import (
        extract_latent_vectors,
        infer_latent_artifact_paths,
        load_autoencoder_from_checkpoint,
        load_latent_manifest,
        resolve_preprocess_config_from_checkpoint,
        save_latent_artifacts,
        sample_id_from_filename,
    )
    from .preprocess import PREPROCESS_MODES, preprocess_image, update_preprocess_config
    from .radial_profile import radial_profile
    from .utils import (
        experiment_root_dir,
        load_json,
        project_path,
        resolve_device,
        save_embedding_plot,
        save_image_grid,
        save_json,
        save_radial_profile_plot,
        save_single_image,
        seed_everything,
        seed_worker,
        stage_output_dir,
    )
except ImportError:
    from clustering_utils import assign_clusters, compute_silhouette, evaluate_cluster_stability
    from data import FitsImageDataset, load_fits_first_slice, scan_fits_directory
    from extract_latents import (
        extract_latent_vectors,
        infer_latent_artifact_paths,
        load_autoencoder_from_checkpoint,
        load_latent_manifest,
        resolve_preprocess_config_from_checkpoint,
        save_latent_artifacts,
        sample_id_from_filename,
    )
    from preprocess import PREPROCESS_MODES, preprocess_image, update_preprocess_config
    from radial_profile import radial_profile
    from utils import (
        experiment_root_dir,
        load_json,
        project_path,
        resolve_device,
        save_embedding_plot,
        save_image_grid,
        save_json,
        save_radial_profile_plot,
        save_single_image,
        seed_everything,
        seed_worker,
        stage_output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster General Test latent vectors with k-means.")
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Directory containing FITS files. Required when latents need to be extracted on the fly.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=project_path("checkpoints", "general", "best_autoencoder.pt"),
        help="Checkpoint used to extract latents on the fly or recover preprocessing config.",
    )
    parser.add_argument(
        "--latent_path",
        default=None,
        help="Optional latent_vectors.npy produced by extract_latents.py.",
    )
    parser.add_argument(
        "--metadata_csv",
        default=None,
        help="Optional latent manifest CSV. Defaults to latent_metadata.csv next to --latent_path.",
    )
    parser.add_argument(
        "--metadata_json",
        default=None,
        help="Optional latent metadata JSON. Defaults to latent_metadata.json next to --latent_path.",
    )
    parser.add_argument("--output_dir", default=project_path("outputs", "general"))
    parser.add_argument(
        "--phase",
        choices=["phase2", "phase3"],
        default="phase2",
        help="Experiment root namespace used when resolving --experiment_name outputs.",
    )
    parser.add_argument(
        "--experiment_name",
        default=None,
        help="Optional experiment name. When set, cluster outputs go under outputs/general/<phase>/<experiment_name>/clusters/.",
    )
    parser.add_argument("--clustering_method", choices=["kmeans", "spectral"], default="kmeans")
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding_method", choices=["umap", "pca"], default="umap")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=9)
    parser.add_argument("--random_k", type=int, default=9)
    parser.add_argument("--radial_nbins", type=int, default=100)
    parser.add_argument(
        "--stability_seeds",
        nargs="*",
        type=int,
        default=[42, 52, 62],
        help="Seeds used to estimate clustering stability with pairwise ARI.",
    )
    parser.add_argument("--preprocess_mode", choices=PREPROCESS_MODES, default=None)
    parser.add_argument("--lower_percentile", type=float, default=None)
    parser.add_argument("--upper_percentile", type=float, default=None)
    parser.add_argument("--robust_clip", type=float, default=None)
    return parser


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
    sample_ids: list[str],
    labels: np.ndarray,
    distances: np.ndarray,
    out_path: str | Path,
) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_index",
                "sample_id",
                "filepath",
                "filename",
                "cluster_id",
                "distance_to_centroid",
            ]
        )
        for index, (filepath, filename, sample_id, label, distance) in enumerate(
            zip(filepaths, filenames, sample_ids, labels, distances, strict=True)
        ):
            writer.writerow([index, sample_id, filepath, filename, int(label), float(distance)])


def load_processed_image(
    filepath: str,
    preprocess_config: dict[str, str | float | int | None],
    img_size: int,
) -> np.ndarray:
    image, _ = load_fits_first_slice(filepath)
    return preprocess_image(
        image,
        mode=str(preprocess_config["mode"]),
        lower=float(preprocess_config["lower_percentile"]),
        upper=float(preprocess_config["upper_percentile"]),
        robust_clip=float(preprocess_config["robust_clip"]),
        img_size=img_size,
    )


def load_images_for_indices(
    indices: np.ndarray,
    filepaths: list[str],
    preprocess_config: dict[str, str | float | int | None],
    img_size: int,
) -> np.ndarray:
    images = [
        load_processed_image(filepaths[int(index)], preprocess_config, img_size)
        for index in np.asarray(indices, dtype=int)
    ]
    if not images:
        return np.empty((0, img_size, img_size), dtype=np.float32)
    return np.stack(images).astype(np.float32)


def compute_cluster_mean_images(
    labels: np.ndarray,
    filepaths: list[str],
    preprocess_config: dict[str, str | float | int | None],
    img_size: int,
    n_clusters: int,
) -> np.ndarray:
    sums = np.zeros((n_clusters, img_size, img_size), dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)

    for filepath, label in zip(filepaths, labels, strict=True):
        processed = load_processed_image(filepath, preprocess_config, img_size)
        sums[int(label)] += processed
        counts[int(label)] += 1

    means = np.zeros((n_clusters, img_size, img_size), dtype=np.float32)
    for cluster_id in range(n_clusters):
        if counts[cluster_id] > 0:
            means[cluster_id] = (sums[cluster_id] / counts[cluster_id]).astype(np.float32)
    return means


def select_representative_indices(
    labels: np.ndarray,
    distances: np.ndarray,
    *,
    top_k: int,
    random_k: int,
    seed: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    topk: dict[int, np.ndarray] = {}
    random_examples: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(seed)

    for cluster_id in np.unique(labels):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_distances = distances[cluster_indices]
        ordered = cluster_indices[np.argsort(cluster_distances)]
        topk[int(cluster_id)] = ordered[:top_k]

        if len(cluster_indices) == 0:
            random_examples[int(cluster_id)] = np.empty((0,), dtype=int)
        else:
            sample_size = min(random_k, len(cluster_indices))
            random_examples[int(cluster_id)] = np.sort(
                rng.choice(cluster_indices, size=sample_size, replace=False)
            )

    return topk, random_examples


def save_radial_profile_csv(
    radius: np.ndarray,
    intensity: np.ndarray,
    out_path: str | Path,
) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["radius_pixels", "mean_intensity"])
        for r_value, i_value in zip(radius, intensity, strict=True):
            writer.writerow([float(r_value), float(i_value)])


def _load_precomputed_latents(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    artifact_paths = infer_latent_artifact_paths(
        args.latent_path,
        metadata_csv=args.metadata_csv,
        metadata_json=args.metadata_json,
    )

    latents = np.load(artifact_paths["latent_path"]).astype(np.float32)
    filepaths, filenames, sample_ids = load_latent_manifest(artifact_paths["metadata_csv"])
    metadata = load_json(artifact_paths["metadata_json"], default={}) or {}

    if latents.shape[0] != len(filepaths):
        raise SystemExit(
            f"Latent count ({latents.shape[0]}) does not match manifest rows ({len(filepaths)})."
        )

    if metadata:
        model_info = metadata.get("model", {})
        preprocess_config = update_preprocess_config(
            metadata.get("preprocess", {}),
            mode=args.preprocess_mode,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
            robust_clip=args.robust_clip,
            img_size=int(model_info.get("input_size", 256)),
        )
        skipped_files = metadata.get("skipped_files", [])
        checkpoint_path = metadata.get("checkpoint_path", args.checkpoint_path)
    else:
        _, checkpoint = load_autoencoder_from_checkpoint(args.checkpoint_path, device)
        checkpoint_config = checkpoint.get("config", {})
        preprocess_config = resolve_preprocess_config_from_checkpoint(
            checkpoint_config,
            mode=args.preprocess_mode,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
            robust_clip=args.robust_clip,
            img_size=int(checkpoint_config.get("input_size", 256)),
        )
        skipped_files = []
        checkpoint_path = args.checkpoint_path

    return {
        "latents": latents,
        "filepaths": filepaths,
        "filenames": filenames,
        "sample_ids": sample_ids,
        "preprocess_config": preprocess_config,
        "input_size": int(preprocess_config["img_size"]),
        "skipped_files": skipped_files,
        "checkpoint_path": checkpoint_path,
        "latent_artifacts": {key: str(value) for key, value in artifact_paths.items()},
    }


def _extract_latents_on_the_fly(
    args: argparse.Namespace,
    device: torch.device,
    latent_output_dir: Path,
) -> dict[str, Any]:
    if not args.data_dir:
        raise SystemExit("Either --latent_path or --data_dir must be provided.")

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
        output_dir=latent_output_dir,
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        preprocess_config=preprocess_config,
        skipped_files=skipped_files,
        checkpoint_config=checkpoint_config,
    )
    sample_ids = [sample_id_from_filename(filename) for filename in filenames]

    return {
        "latents": latents,
        "filepaths": filepaths,
        "filenames": filenames,
        "sample_ids": sample_ids,
        "preprocess_config": preprocess_config,
        "input_size": model.input_size,
        "skipped_files": skipped_files,
        "checkpoint_path": args.checkpoint_path,
        "latent_artifacts": artifacts,
    }


def run_clustering(args: argparse.Namespace) -> dict[str, str | int]:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    experiment_root = experiment_root_dir(args.output_dir, args.experiment_name, phase=args.phase)
    latent_output_dir = stage_output_dir(args.output_dir, args.experiment_name, "latents", phase=args.phase)
    cluster_output_dir = stage_output_dir(args.output_dir, args.experiment_name, "clusters", phase=args.phase)

    if args.latent_path:
        latent_data = _load_precomputed_latents(args, device)
    else:
        if args.phase == "phase3":
            raise SystemExit(
                "Phase 3 clustering expects precomputed latents. "
                "Run extract_contrastive_latents.py first and pass --latent_path."
            )
        latent_data = _extract_latents_on_the_fly(args, device, latent_output_dir)

    latents = latent_data["latents"]
    filepaths = latent_data["filepaths"]
    filenames = latent_data["filenames"]
    sample_ids = latent_data["sample_ids"]
    preprocess_config = latent_data["preprocess_config"]
    img_size = int(latent_data["input_size"])
    skipped_files = latent_data.get("skipped_files", [])

    if latents.shape[0] == 0:
        raise SystemExit("No latents available for clustering.")
    if args.n_clusters > latents.shape[0]:
        raise SystemExit(
            f"Requested n_clusters={args.n_clusters}, but only {latents.shape[0]} samples are available."
        )

    clustering = assign_clusters(
        latents,
        method=args.clustering_method,
        n_clusters=args.n_clusters,
        seed=args.seed,
    )
    labels = clustering["labels"]
    distances = clustering["distances"]
    clustering_details = clustering["details"]
    silhouette = compute_silhouette(latents, labels)
    stability = evaluate_cluster_stability(
        latents,
        method=args.clustering_method,
        n_clusters=args.n_clusters,
        seeds=list(dict.fromkeys(args.stability_seeds)),
    )

    print("latents shape:", latents.shape)
    print("latents min/max:", float(latents.min()), float(latents.max()))
    print("labels unique:", np.unique(labels))
    print("cluster counts:", {int(i): int((labels == i).sum()) for i in np.unique(labels)})
    if clustering_details.get("inertia") is not None:
        print("kmeans inertia:", float(clustering_details["inertia"]))
    print("silhouette score:", silhouette)

    embedding, embedding_method_used = compute_embedding(latents, args.embedding_method, args.seed)
    cluster_sizes = {
        int(cluster_id): int(size)
        for cluster_id, size in clustering_details["cluster_sizes"].items()
    }

    labels_path = cluster_output_dir / "cluster_labels.npy"
    assignments_path = cluster_output_dir / "cluster_assignments.csv"
    embedding_path = cluster_output_dir / "umap.png"
    cluster_summary_path = cluster_output_dir / "cluster_summary.json"
    clustering_summary_path = cluster_output_dir / "clustering_summary.json"

    np.save(labels_path, labels.astype(np.int64))
    save_cluster_assignments(filepaths, filenames, sample_ids, labels, distances, assignments_path)
    save_embedding_plot(
        embedding=embedding,
        labels=labels,
        out_path=embedding_path,
        title=f"Latent Clusters ({embedding_method_used.upper()})",
        cluster_sizes=cluster_sizes,
    )

    topk_indices, random_indices = select_representative_indices(
        labels,
        distances,
        top_k=args.top_k,
        random_k=args.random_k,
        seed=args.seed,
    )
    mean_images = compute_cluster_mean_images(
        labels,
        filepaths,
        preprocess_config,
        img_size,
        args.n_clusters,
    )

    radial_plot_profiles: list[dict[str, Any]] = []
    cluster_summary: list[dict[str, Any]] = []

    for cluster_id in range(args.n_clusters):
        cluster_mask = labels == cluster_id
        cluster_distance_values = distances[cluster_mask]
        nearest_indices = topk_indices.get(cluster_id, np.empty((0,), dtype=int))
        random_cluster_indices = random_indices.get(cluster_id, np.empty((0,), dtype=int))

        topk_images = load_images_for_indices(nearest_indices, filepaths, preprocess_config, img_size)
        topk_titles = [filenames[int(index)] for index in nearest_indices]
        save_image_grid(
            images=topk_images,
            out_path=cluster_output_dir / f"cluster_{cluster_id}_topk.png",
            titles=topk_titles,
            suptitle=f"Cluster {cluster_id} Nearest to Centroid",
            n_cols=3,
        )

        random_images = load_images_for_indices(
            random_cluster_indices,
            filepaths,
            preprocess_config,
            img_size,
        )
        random_titles = [filenames[int(index)] for index in random_cluster_indices]
        save_image_grid(
            images=random_images,
            out_path=cluster_output_dir / f"cluster_{cluster_id}_random.png",
            titles=random_titles,
            suptitle=f"Cluster {cluster_id} Random Examples",
            n_cols=3,
        )

        mean_image = mean_images[cluster_id]
        save_single_image(
            mean_image,
            cluster_output_dir / f"cluster_{cluster_id}_mean.png",
            title=f"Cluster {cluster_id} Mean Image",
        )

        radius, intensity = radial_profile(mean_image, nbins=args.radial_nbins)
        save_radial_profile_csv(
            radius,
            intensity,
            cluster_output_dir / f"cluster_{cluster_id}_radial_profile.csv",
        )
        save_radial_profile_plot(
            [
                {
                    "radius": radius,
                    "intensity": intensity,
                    "label": f"cluster {cluster_id}",
                }
            ],
            cluster_output_dir / f"cluster_{cluster_id}_radial_profile.png",
            title=f"Cluster {cluster_id} Mean Radial Profile",
        )
        radial_plot_profiles.append(
            {
                "radius": radius,
                "intensity": intensity,
                "label": f"cluster {cluster_id}",
            }
        )

        cluster_summary.append(
            {
                "cluster_id": int(cluster_id),
                "count": int(cluster_mask.sum()),
                "example_filenames": [filenames[int(index)] for index in nearest_indices[: min(5, len(nearest_indices))]],
                "distance_to_centroid": {
                    "min": float(cluster_distance_values.min()) if cluster_distance_values.size else None,
                    "max": float(cluster_distance_values.max()) if cluster_distance_values.size else None,
                    "mean": float(cluster_distance_values.mean()) if cluster_distance_values.size else None,
                    "std": float(cluster_distance_values.std()) if cluster_distance_values.size else None,
                },
                "artifacts": {
                    "topk_grid": str(cluster_output_dir / f"cluster_{cluster_id}_topk.png"),
                    "random_grid": str(cluster_output_dir / f"cluster_{cluster_id}_random.png"),
                    "mean_image": str(cluster_output_dir / f"cluster_{cluster_id}_mean.png"),
                    "radial_profile_csv": str(cluster_output_dir / f"cluster_{cluster_id}_radial_profile.csv"),
                    "radial_profile_plot": str(cluster_output_dir / f"cluster_{cluster_id}_radial_profile.png"),
                },
            }
        )

    save_radial_profile_plot(
        radial_plot_profiles,
        cluster_output_dir / "radial_profiles.png",
        title="Cluster Mean Radial Profiles",
    )
    save_json({"clusters": cluster_summary}, cluster_summary_path)

    overall_summary = {
        "experiment_name": args.experiment_name,
        "experiment_root": str(experiment_root),
        "data_dir": str(Path(args.data_dir).resolve()) if args.data_dir else None,
        "checkpoint_path": str(Path(latent_data["checkpoint_path"]).resolve()) if latent_data.get("checkpoint_path") else None,
        "num_images": int(len(filepaths)),
        "num_skipped_files": int(len(skipped_files)),
        "skipped_files": skipped_files,
        "latent_dimension": int(latents.shape[1] if latents.ndim == 2 else 0),
        "clustering_method": args.clustering_method,
        "number_of_clusters": int(args.n_clusters),
        "silhouette_score": silhouette,
        "stability": stability,
        "embedding_method_requested": args.embedding_method,
        "embedding_method_used": embedding_method_used,
        "preprocess": preprocess_config,
        "cluster_sizes": {str(key): value for key, value in cluster_sizes.items()},
        "clustering_details": clustering_details,
        "artifacts": {
            **latent_data.get("latent_artifacts", {}),
            "cluster_labels": str(labels_path),
            "assignments_csv": str(assignments_path),
            "embedding_plot": str(embedding_path),
            "cluster_summary_json": str(cluster_summary_path),
            "radial_profiles_plot": str(cluster_output_dir / "radial_profiles.png"),
        },
    }
    save_json(overall_summary, clustering_summary_path)

    print(
        f"Clustering complete for {len(filepaths)} images with k={args.n_clusters}. "
        f"Outputs saved to {cluster_output_dir}."
    )

    return {
        "num_images": len(filepaths),
        "n_clusters": args.n_clusters,
        "embedding_method_used": embedding_method_used,
        "output_dir": str(cluster_output_dir),
    }


def main(cli_args: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(cli_args)
    run_clustering(args)


if __name__ == "__main__":
    main()
