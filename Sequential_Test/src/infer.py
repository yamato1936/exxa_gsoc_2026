import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import TransitCNN
from utils import (
    ensure_dir,
    get_torch_device,
    load_json,
    normalize_curve,
    project_path,
    resample_curve,
)


CHECKPOINT_PATH = project_path("checkpoints", "sequential", "best_model.pt")
METRICS_PATH = project_path("outputs", "sequential", "metrics.json")
INFER_PLOT_PATH = project_path("outputs", "sequential", "infer_example.png")


def load_input_curve(path: str) -> np.ndarray:
    curve = np.load(path).astype(np.float32)

    if curve.ndim == 1:
        return curve

    if curve.ndim == 2 and 1 in curve.shape:
        return curve.reshape(-1)

    raise ValueError("input_npy must contain a single 1D light curve.")


def save_inference_plot(curve: np.ndarray, prob: float, pred_05: int, pred_tuned, tuned_threshold) -> None:
    ensure_dir(project_path("outputs", "sequential"))

    title = f"p={prob:.3f} | pred@0.5={pred_05}"
    if tuned_threshold is not None and pred_tuned is not None:
        title += f" | pred@{tuned_threshold:.3f}={pred_tuned}"

    plt.figure(figsize=(8, 3.6))
    plt.plot(curve, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Cadence")
    plt.ylabel("Normalized flux")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(INFER_PLOT_PATH, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npy", type=str, required=True, help="Path to 1D light curve .npy")
    args = parser.parse_args()

    device = get_torch_device()
    curve = load_input_curve(args.input_npy)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    input_length = int(ckpt["input_length"])

    if len(curve) != input_length:
        print(
            f"Resampling input curve from length {len(curve)} to {input_length} "
            "to match the trained model."
        )
        curve = resample_curve(curve, input_length)

    curve = normalize_curve(curve)
    x = torch.tensor(curve, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    model = TransitCNN(input_length=input_length).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()

    pred_05 = int(prob >= 0.5)
    metrics = load_json(METRICS_PATH, default={}) or {}
    tuned_threshold = metrics.get("threshold_tuned", {}).get("threshold")
    pred_tuned = None

    print(f"Planet probability: {prob:.6f}")
    print(f"Predicted label @ 0.5: {pred_05} (1=planet transit, 0=no transit)")

    if tuned_threshold is not None:
        tuned_threshold = float(tuned_threshold)
        pred_tuned = int(prob >= tuned_threshold)
        print(
            f"Predicted label @ tuned threshold {tuned_threshold:.3f}: "
            f"{pred_tuned} (1=planet transit, 0=no transit)"
        )
    else:
        print("Tuned threshold unavailable. Run evaluate.py to create outputs/sequential/metrics.json.")

    save_inference_plot(curve, prob, pred_05, pred_tuned, tuned_threshold)
    print("Saved inference plot to outputs/sequential/infer_example.png")


if __name__ == "__main__":
    main()
