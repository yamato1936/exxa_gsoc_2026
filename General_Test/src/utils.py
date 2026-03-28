from pathlib import Path
import numpy as np
from astropy.io import fits

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def load_fits_first_plane(path):
    with fits.open(path) as hdul:
        x = hdul[0].data
    if x.ndim == 3:
        x = x[0]
    return np.asarray(x, dtype=np.float32)
