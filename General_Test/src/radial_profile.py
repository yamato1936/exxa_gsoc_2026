from __future__ import annotations

import numpy as np


def radial_profile(
    img: np.ndarray,
    center: tuple[float, float] | None = None,
    nbins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    if nbins <= 0:
        raise ValueError(f"nbins must be positive, got {nbins}.")

    image = np.asarray(img, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if image.ndim != 2:
        raise ValueError(f"radial_profile expects a 2D image, got shape {image.shape}.")

    height, width = image.shape
    if center is None:
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
    else:
        center_x, center_y = center

    y, x = np.indices(image.shape, dtype=np.float32)
    radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_radius = float(radius.max())
    if max_radius <= 0.0:
        return np.zeros(nbins, dtype=np.float32), np.zeros(nbins, dtype=np.float32)

    edges = np.linspace(0.0, max_radius, nbins + 1, dtype=np.float32)
    bin_index = np.digitize(radius.ravel(), edges, right=False) - 1
    bin_index = np.clip(bin_index, 0, nbins - 1)

    radial_sum = np.bincount(bin_index, weights=image.ravel(), minlength=nbins)
    radial_count = np.bincount(bin_index, minlength=nbins)
    profile = np.divide(
        radial_sum,
        radial_count,
        out=np.zeros(nbins, dtype=np.float32),
        where=radial_count > 0,
    )
    radii = 0.5 * (edges[:-1] + edges[1:])

    return radii.astype(np.float32), profile.astype(np.float32)
