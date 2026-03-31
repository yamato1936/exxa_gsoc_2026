from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score


def assign_clusters(
    latents: np.ndarray,
    *,
    method: str = "kmeans",
    n_clusters: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    latents = np.asarray(latents, dtype=np.float32)
    if latents.ndim != 2:
        raise ValueError(f"Expected 2D latent array, got shape {latents.shape}.")
    if latents.shape[0] < n_clusters:
        raise ValueError(
            f"Requested n_clusters={n_clusters}, but only {latents.shape[0]} samples are available."
        )

    method = str(method).lower()
    details: dict[str, Any] = {"method": method, "n_clusters": int(n_clusters), "seed": int(seed)}

    if method == "kmeans":
        estimator = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
        labels = estimator.fit_predict(latents)
        details["inertia"] = float(estimator.inertia_)
    elif method == "spectral":
        n_neighbors = max(2, min(10, latents.shape[0] - 1))
        estimator = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels="kmeans",
            random_state=seed,
        )
        labels = estimator.fit_predict(latents)
        details["n_neighbors"] = int(n_neighbors)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    centroids = np.zeros((n_clusters, latents.shape[1]), dtype=np.float32)
    counts = np.zeros(n_clusters, dtype=np.int64)
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if np.any(mask):
            centroids[cluster_id] = latents[mask].mean(axis=0)
            counts[cluster_id] = int(mask.sum())

    distances = np.linalg.norm(latents - centroids[labels], axis=1).astype(np.float32)
    details["cluster_sizes"] = {int(cluster_id): int(count) for cluster_id, count in enumerate(counts.tolist())}
    return {
        "labels": labels.astype(np.int64),
        "distances": distances,
        "centroids": centroids,
        "details": details,
    }


def compute_silhouette(latents: np.ndarray, labels: np.ndarray) -> float | None:
    latents = np.asarray(latents, dtype=np.float32)
    labels = np.asarray(labels)

    unique_labels = np.unique(labels)
    if latents.shape[0] < 2 or unique_labels.size < 2 or unique_labels.size >= latents.shape[0]:
        return None

    try:
        return float(silhouette_score(latents, labels))
    except Exception:
        return None


def evaluate_cluster_stability(
    latents: np.ndarray,
    *,
    method: str,
    n_clusters: int,
    seeds: list[int],
) -> dict[str, Any]:
    if not seeds:
        return {"seeds": [], "pairwise_ari": [], "mean_ari": None, "std_ari": None}

    label_runs: list[np.ndarray] = []
    for seed in seeds:
        label_runs.append(
            assign_clusters(
                latents,
                method=method,
                n_clusters=n_clusters,
                seed=seed,
            )["labels"]
        )

    pairwise_scores: list[float] = []
    for first, second in combinations(range(len(label_runs)), 2):
        pairwise_scores.append(float(adjusted_rand_score(label_runs[first], label_runs[second])))

    return {
        "seeds": [int(seed) for seed in seeds],
        "pairwise_ari": pairwise_scores,
        "mean_ari": float(np.mean(pairwise_scores)) if pairwise_scores else None,
        "std_ari": float(np.std(pairwise_scores)) if pairwise_scores else None,
    }
