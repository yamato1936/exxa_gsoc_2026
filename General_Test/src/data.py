from __future__ import annotations

import math
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
    from .preprocess import build_preprocess_config, preprocess_image
    from .utils import save_sample_inputs
except ImportError:
    from preprocess import build_preprocess_config, preprocess_image
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

    return sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".fits"
    )


def _load_first_available_hdu(path: str | Path) -> np.ndarray:
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return np.asarray(hdu.data)

    raise ValueError("No image data found in FITS file.")


def extract_first_image_plane(array: np.ndarray) -> np.ndarray:
    """
    Extract the first relevant 2D image plane from a FITS cube.

    For the EXXA general test dataset, files may look like:
        (4, 1, 1, 600, 600)

    The task description says only the first layer is relevant, so we:
      1. take array[0]
      2. squeeze singleton axes
      3. verify that the result is 2D
    """
    array = np.asarray(array)
    if array.size == 0:
        raise ValueError("Empty FITS array.")

    if array.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions, got shape {array.shape}.")

    # EXXA note: only the first layer is relevant.
    # For shapes like (4, 1, 1, 600, 600), this gives (1, 1, 600, 600).
    first_layer = array[0] if array.ndim > 2 else array

    image = np.squeeze(first_layer)

    if image.ndim != 2:
        raise ValueError(
            f"Could not extract a 2D image plane from shape {array.shape}. "
            f"After taking first layer and squeeze, got {image.shape}."
        )

    return np.asarray(image, dtype=np.float32)


def load_fits_first_slice(path: str | Path) -> tuple[np.ndarray, tuple[int, ...]]:
    cube = _load_first_available_hdu(path)
    image = extract_first_image_plane(cube)
    return image, tuple(np.asarray(cube).shape)


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

    train_records, val_records = train_test_split(
        records,
        test_size=n_val,
        random_state=seed,
        shuffle=True,
    )
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
    preprocess_mode: str,
    lower_percentile: float,
    upper_percentile: float,
    robust_clip: float = 5.0,
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
                mode=preprocess_mode,
                lower=lower_percentile,
                upper=upper_percentile,
                img_size=img_size,
                robust_clip=robust_clip,
            )
        )
        titles.append(record.filename)

    save_sample_inputs(raw_images, processed_images, titles, output_path)


class ImageAugmenter:
    def __init__(
        self,
        enabled: bool = False,
        rotation_deg: float = 90.0,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        noise_std: float = 0.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.rotation_deg = max(0.0, float(rotation_deg))
        self.hflip_prob = float(np.clip(hflip_prob, 0.0, 1.0))
        self.vflip_prob = float(np.clip(vflip_prob, 0.0, 1.0))
        self.noise_std = max(0.0, float(noise_std))

    def _rotate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.rotation_deg <= 0.0:
            return tensor

        angle = float(torch.empty(1).uniform_(-self.rotation_deg, self.rotation_deg).item())
        radians = math.radians(angle)
        theta = torch.tensor(
            [
                [math.cos(radians), -math.sin(radians), 0.0],
                [math.sin(radians), math.cos(radians), 0.0],
            ],
            dtype=tensor.dtype,
            device=tensor.device,
        ).unsqueeze(0)
        grid = F.affine_grid(
            theta,
            size=(1, tensor.shape[0], tensor.shape[1], tensor.shape[2]),
            align_corners=False,
        )
        rotated = F.grid_sample(
            tensor.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return rotated.squeeze(0)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return tensor

        augmented = tensor
        if torch.rand(1).item() < self.hflip_prob:
            augmented = torch.flip(augmented, dims=[2])
        if torch.rand(1).item() < self.vflip_prob:
            augmented = torch.flip(augmented, dims=[1])

        augmented = self._rotate_tensor(augmented)

        if self.noise_std > 0.0:
            augmented = augmented + torch.randn_like(augmented) * self.noise_std

        return augmented.clamp(0.0, 1.0)


class FitsImageDataset(Dataset):
    def __init__(
        self,
        records: list[FitsRecord],
        img_size: int,
        preprocess_mode: str = "percentile_minmax",
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.5,
        robust_clip: float = 5.0,
        augment: bool = False,
        rotation_deg: float = 90.0,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        noise_std: float = 0.0,
    ) -> None:
        self.records = list(records)
        self.img_size = int(img_size)
        self.preprocess_config = build_preprocess_config(
            mode=preprocess_mode,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            robust_clip=robust_clip,
            img_size=img_size,
        )
        self.augmenter = ImageAugmenter(
            enabled=augment,
            rotation_deg=rotation_deg,
            hflip_prob=hflip_prob,
            vflip_prob=vflip_prob,
            noise_std=noise_std,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image, _ = load_fits_first_slice(record.filepath)
        image = preprocess_image(
            image,
            mode=str(self.preprocess_config["mode"]),
            lower=float(self.preprocess_config["lower_percentile"]),
            upper=float(self.preprocess_config["upper_percentile"]),
            img_size=int(self.preprocess_config["img_size"]),
            robust_clip=float(self.preprocess_config["robust_clip"]),
        )
        tensor = torch.from_numpy(image).unsqueeze(0)
        tensor = self.augmenter(tensor)

        return {
            "image": tensor,
            "filepath": record.filepath,
            "filename": record.filename,
        }


class ContrastivePairDataset(Dataset):
    def __init__(
        self,
        records: list[FitsRecord],
        img_size: int,
        preprocess_mode: str = "log_minmax",
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.5,
        robust_clip: float = 5.0,
        use_augmentation: bool = True,
        rotation_deg: float = 30.0,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        noise_std: float = 0.0,
    ) -> None:
        self.records = list(records)
        self.preprocess_config = build_preprocess_config(
            mode=preprocess_mode,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            robust_clip=robust_clip,
            img_size=img_size,
        )
        self.augmenter = ImageAugmenter(
            enabled=use_augmentation,
            rotation_deg=rotation_deg,
            hflip_prob=hflip_prob,
            vflip_prob=vflip_prob,
            noise_std=noise_std,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image, _ = load_fits_first_slice(record.filepath)
        image = preprocess_image(
            image,
            mode=str(self.preprocess_config["mode"]),
            lower=float(self.preprocess_config["lower_percentile"]),
            upper=float(self.preprocess_config["upper_percentile"]),
            img_size=int(self.preprocess_config["img_size"]),
            robust_clip=float(self.preprocess_config["robust_clip"]),
        )
        base_tensor = torch.from_numpy(image).unsqueeze(0)
        view_one = self.augmenter(base_tensor.clone())
        view_two = self.augmenter(base_tensor.clone())

        return {
            "view_one": view_one,
            "view_two": view_two,
            "image": base_tensor,
            "filepath": record.filepath,
            "filename": record.filename,
        }
