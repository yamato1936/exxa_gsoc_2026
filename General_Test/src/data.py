from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

try:
    from .utils import save_sample_inputs
except ImportError:
    from utils import save_sample_inputs


@dataclass(frozen=True)
class FitsRecord:
    filepath: str
    filename: str
    original_shape: tuple[int, ...]
    image_shape: tuple[int, int]


def find_fits_files(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    return sorted(path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".fits")


def _load_first_available_hdu(path: str | Path) -> np.ndarray:
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return np.asarray(hdu.data)

    raise ValueError("No image data found in FITS file.")


def extract_first_image_plane(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.size == 0:
        raise ValueError("Empty FITS array.")

    squeezed = np.squeeze(array)
    if squeezed.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions after squeeze, got shape {squeezed.shape}.")

    if squeezed.ndim == 2:
        image = squeezed
    else:
        spatial_axes = sorted(np.argsort(squeezed.shape)[-2:].tolist())
        index = tuple(slice(None) if axis in spatial_axes else 0 for axis in range(squeezed.ndim))
        image = np.squeeze(squeezed[index])

    if image.ndim != 2:
        raise ValueError(f"Could not extract a 2D image plane from shape {array.shape}.")

    return np.asarray(image, dtype=np.float32)


def load_fits_first_slice(path: str | Path) -> tuple[np.ndarray, tuple[int, ...]]:
    cube = _load_first_available_hdu(path)
    image = extract_first_image_plane(cube)
    return image, tuple(np.asarray(cube).shape)


def preprocess_image(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    img_size: int | None = None,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite_mask = np.isfinite(image)

    if not finite_mask.any():
        processed = np.zeros_like(image, dtype=np.float32)
    else:
        finite_values = image[finite_mask]
        low = float(np.percentile(finite_values, lower_percentile))
        high = float(np.percentile(finite_values, upper_percentile))

        # Replace invalid values before clipping so the percentile-based scaling
        # stays deterministic and reusable during training and analysis.
        image = np.nan_to_num(image, nan=low, posinf=high, neginf=low)
        if high - low < 1e-8:
            processed = np.zeros_like(image, dtype=np.float32)
        else:
            processed = np.clip(image, low, high)
            processed = (processed - low) / (high - low)
            processed = processed.astype(np.float32)

    if img_size is not None and processed.shape != (img_size, img_size):
        tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
        processed = tensor.squeeze(0).squeeze(0).numpy().astype(np.float32)

    return processed


def scan_fits_directory(data_dir: str | Path) -> tuple[list[FitsRecord], list[dict[str, str]]]:
    records: list[FitsRecord] = []
    skipped_files: list[dict[str, str]] = []

    for path in find_fits_files(data_dir):
        try:
            image, original_shape = load_fits_first_slice(path)
            records.append(
                FitsRecord(
                    filepath=str(path),
                    filename=path.name,
                    original_shape=original_shape,
                    image_shape=tuple(image.shape),
                )
            )
        except Exception as exc:
            warnings.warn(f"Skipping unreadable FITS file {path}: {exc}")
            skipped_files.append({"filepath": str(path), "reason": str(exc)})

    return records, skipped_files


def train_val_split(
    records: list[FitsRecord],
    val_ratio: float,
    seed: int,
) -> tuple[list[FitsRecord], list[FitsRecord], dict[str, int | bool | float]]:
    if not records:
        raise ValueError("No valid FITS records available for splitting.")

    if len(records) == 1 or val_ratio <= 0.0:
        return records, records, {
            "num_train": len(records),
            "num_val": len(records),
            "val_ratio": float(val_ratio),
            "validation_reuses_training": True,
        }

    n_val = max(1, int(round(len(records) * val_ratio)))
    n_val = min(n_val, len(records) - 1)

    train_records, val_records = train_test_split(records, test_size=n_val, random_state=seed, shuffle=True)
    train_records = sorted(train_records, key=lambda record: record.filepath)
    val_records = sorted(val_records, key=lambda record: record.filepath)

    return train_records, val_records, {
        "num_train": len(train_records),
        "num_val": len(val_records),
        "val_ratio": float(val_ratio),
        "validation_reuses_training": False,
    }


def save_sample_input_grid(
    records: list[FitsRecord],
    output_path: str | Path,
    img_size: int | None,
    lower_percentile: float,
    upper_percentile: float,
    max_examples: int = 6,
) -> None:
    raw_images: list[np.ndarray] = []
    processed_images: list[np.ndarray] = []
    titles: list[str] = []

    for record in records[:max_examples]:
        image, _ = load_fits_first_slice(record.filepath)
        raw_images.append(image)
        processed_images.append(
            preprocess_image(
                image,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                img_size=img_size,
            )
        )
        titles.append(record.filename)

    save_sample_inputs(raw_images, processed_images, titles, output_path)


class FitsImageDataset(Dataset):
    def __init__(
        self,
        records: list[FitsRecord],
        img_size: int,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
        augment: bool = False,
    ) -> None:
        self.records = list(records)
        self.img_size = int(img_size)
        self.lower_percentile = float(lower_percentile)
        self.upper_percentile = float(upper_percentile)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.records)

    def _augment_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return tensor

        if torch.rand(1).item() < 0.5:
            tensor = torch.flip(tensor, dims=[2])
        if torch.rand(1).item() < 0.5:
            tensor = torch.flip(tensor, dims=[1])

        rotation_k = int(torch.randint(0, 4, (1,)).item())
        tensor = torch.rot90(tensor, k=rotation_k, dims=[1, 2])
        return tensor

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image, _ = load_fits_first_slice(record.filepath)
        image = preprocess_image(
            image,
            lower_percentile=self.lower_percentile,
            upper_percentile=self.upper_percentile,
            img_size=self.img_size,
        )
        tensor = torch.from_numpy(image).unsqueeze(0)
        tensor = self._augment_tensor(tensor)

        return {
            "image": tensor,
            "filepath": record.filepath,
            "filename": record.filename,
        }
