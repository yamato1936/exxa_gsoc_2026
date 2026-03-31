from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PREPROCESS_MODES = ("percentile_minmax", "log_minmax", "robust")


def build_preprocess_config(
    mode: str = "percentile_minmax",
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
    robust_clip: float = 5.0,
    img_size: int | None = None,
) -> dict[str, str | float | int | None]:
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"Unsupported preprocessing mode: {mode}")
    if upper_percentile <= lower_percentile:
        raise ValueError(
            "upper_percentile must be greater than lower_percentile, "
            f"got {lower_percentile} and {upper_percentile}."
        )
    if robust_clip <= 0:
        raise ValueError(f"robust_clip must be positive, got {robust_clip}.")

    return {
        "mode": str(mode),
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "robust_clip": float(robust_clip),
        "img_size": None if img_size is None else int(img_size),
    }


def extract_preprocess_config(
    source: Mapping[str, Any] | None,
    *,
    default_mode: str = "percentile_minmax",
    default_img_size: int | None = None,
) -> dict[str, str | float | int | None]:
    source = source or {}
    nested = source.get("preprocess", {})
    if not isinstance(nested, Mapping):
        nested = {}

    return build_preprocess_config(
        mode=str(nested.get("mode", source.get("preprocess_mode", default_mode))),
        lower_percentile=float(
            nested.get("lower_percentile", source.get("lower_percentile", 1.0))
        ),
        upper_percentile=float(
            nested.get("upper_percentile", source.get("upper_percentile", 99.5))
        ),
        robust_clip=float(nested.get("robust_clip", source.get("robust_clip", 5.0))),
        img_size=nested.get("img_size", source.get("input_size", default_img_size)),
    )


def update_preprocess_config(
    base: Mapping[str, Any] | None,
    *,
    mode: str | None = None,
    lower_percentile: float | None = None,
    upper_percentile: float | None = None,
    robust_clip: float | None = None,
    img_size: int | None = None,
) -> dict[str, str | float | int | None]:
    config = extract_preprocess_config(base)
    return build_preprocess_config(
        mode=str(mode if mode is not None else config["mode"]),
        lower_percentile=float(
            lower_percentile
            if lower_percentile is not None
            else config["lower_percentile"]
        ),
        upper_percentile=float(
            upper_percentile
            if upper_percentile is not None
            else config["upper_percentile"]
        ),
        robust_clip=float(
            robust_clip if robust_clip is not None else config["robust_clip"]
        ),
        img_size=img_size if img_size is not None else config["img_size"],
    )


def _sanitize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)


def _resize_image(image: np.ndarray, img_size: int | None) -> np.ndarray:
    if img_size is None or image.shape == (img_size, img_size):
        return image.astype(np.float32)

    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    return tensor.squeeze(0).squeeze(0).numpy().astype(np.float32)


def _percentile_clip(values: np.ndarray, lower: float, upper: float) -> tuple[float, float]:
    low = float(np.percentile(values, lower))
    high = float(np.percentile(values, upper))
    return low, high


def _minmax_rescale(image: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    scaled = np.clip(image, low, high)
    scaled = (scaled - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def preprocess_image(
    image: np.ndarray,
    mode: str = "percentile_minmax",
    lower: float = 1.0,
    upper: float = 99.5,
    img_size: int | None = None,
    robust_clip: float = 5.0,
) -> np.ndarray:
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"Unsupported preprocessing mode: {mode}")
    if upper <= lower:
        raise ValueError(
            "upper must be greater than lower, "
            f"got {lower} and {upper}."
        )
    if robust_clip <= 0:
        raise ValueError(f"robust_clip must be positive, got {robust_clip}.")

    image = _sanitize_image(image)
    finite_values = image[np.isfinite(image)]
    if finite_values.size == 0:
        return _resize_image(np.zeros_like(image, dtype=np.float32), img_size)

    if mode == "percentile_minmax":
        low, high = _percentile_clip(finite_values, lower, upper)
        processed = _minmax_rescale(image, low, high)

    elif mode == "log_minmax":
        image = np.clip(image, a_min=0.0, a_max=None)

        positive_values = image[image > 0]
        if positive_values.size == 0:
            processed = np.zeros_like(image, dtype=np.float32)
        else:
            _, high_raw = _percentile_clip(positive_values, lower, upper)
            scale = max(high_raw, 1e-30)

            scaled = image / scale
            log_scale = 1000.0
            logged = np.log1p(log_scale * scaled).astype(np.float32)

            finite_values = logged[np.isfinite(logged)]
            low, high = _percentile_clip(finite_values, lower, upper)
            processed = _minmax_rescale(logged, low, high)

    elif mode == "robust":
        image = np.clip(image, a_min=0.0, a_max=None)
        finite_values = image[np.isfinite(image)]
        median = float(np.median(finite_values))
        q1 = float(np.percentile(finite_values, 25.0))
        q3 = float(np.percentile(finite_values, 75.0))
        iqr = q3 - q1
        if iqr <= 1e-8:
            processed = np.zeros_like(image, dtype=np.float32)
        else:
            normalized = (image - median) / iqr
            clipped = np.clip(normalized, -robust_clip, robust_clip)
            processed = (clipped + robust_clip) / (2.0 * robust_clip)
            processed = np.clip(processed, 0.0, 1.0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported preprocessing mode: {mode}")

    return _resize_image(processed, img_size)