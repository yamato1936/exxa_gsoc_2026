import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> str:
    return str(repo_root().joinpath(*parts))


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str, default: Optional[Any] = None) -> Any:
    path_obj = Path(path)
    if not path_obj.exists():
        return default

    with open(path_obj, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_curve(flux: np.ndarray) -> np.ndarray:
    flux = np.asarray(flux, dtype=np.float32)
    median = np.median(flux)
    std = np.std(flux)
    if std < 1e-8:
        return flux - median
    return (flux - median) / std


def resample_curve(curve: np.ndarray, target_length: int) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32).reshape(-1)
    if len(curve) == target_length:
        return curve

    source_x = np.linspace(0.0, 1.0, len(curve))
    target_x = np.linspace(0.0, 1.0, target_length)
    return np.interp(target_x, source_x, curve).astype(np.float32)


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as exc:
            print(f"CUDA unavailable, falling back to CPU: {exc}")

    return torch.device("cpu")
