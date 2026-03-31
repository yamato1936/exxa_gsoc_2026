from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import glob

paths = glob.glob("data/fits/*.fits")
path = paths[0]

print("Loading:", path)

# ===== Load =====
raw = fits.getdata(path)
print("raw shape:", raw.shape)

data = raw[0]              # first layer only
print("after [0]:", data.shape)

data = np.squeeze(data)    # (600,600)
print("squeezed shape:", data.shape)

print("min:", data.min())
print("max:", data.max())
print("mean:", data.mean())

# ===== Preprocess =====

# ① 負値除去
data_nonneg = np.clip(data, a_min=0.0, a_max=None)

# ② percentile clipping（logは一旦使わない）
vmin = np.percentile(data_nonneg, 1)
vmax = np.percentile(data_nonneg, 99.5)

print("vmin:", vmin, "vmax:", vmax)

if vmax > vmin:
    img = np.clip(data_nonneg, vmin, vmax)
    img = (img - vmin) / (vmax - vmin)
else:
    print("Warning: vmax == vmin")
    img = np.zeros_like(data_nonneg)

# ===== ★ここが今回のポイント =====
print("processed min:", img.min())
print("processed max:", img.max())
print("processed mean:", img.mean())

# ===== Visualization =====

plt.figure(figsize=(5,5))
plt.title("Raw")
plt.imshow(data_nonneg, cmap="inferno")
plt.colorbar()
plt.savefig("debug_raw.png")

plt.figure(figsize=(5,5))
plt.title("Processed")
plt.imshow(img, cmap="inferno")
plt.colorbar()
plt.savefig("debug_processed.png")