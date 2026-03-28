from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import ensure_dir, normalize_curve, project_path, seed_everything


OUTPUT_DIR = project_path("outputs", "sequential")


def smooth_noise_component(
    seq_len: int,
    rng: np.random.Generator,
    amplitude: float,
) -> np.ndarray:
    if amplitude <= 0.0:
        return np.zeros(seq_len, dtype=np.float32)

    noise = rng.normal(0.0, 1.0, size=seq_len)
    kernel_size = int(rng.integers(7, 31))
    if kernel_size % 2 == 0:
        kernel_size += 1

    grid = np.linspace(-2.5, 2.5, kernel_size)
    kernel = np.exp(-0.5 * grid**2)
    kernel /= kernel.sum()

    smooth_noise = np.convolve(noise, kernel, mode="same")
    smooth_std = np.std(smooth_noise) + 1e-8
    return (amplitude * smooth_noise / smooth_std).astype(np.float32)


def low_frequency_trend(
    x: np.ndarray,
    rng: np.random.Generator,
    amplitude: float,
) -> np.ndarray:
    trend = rng.uniform(-1.0, 1.0) * (x - 0.5)
    trend += rng.uniform(-0.8, 0.8) * (x - 0.5) ** 2

    if rng.uniform() < 0.85:
        trend += rng.uniform(0.4, 1.0) * np.sin(
            2 * np.pi * rng.uniform(0.3, 1.8) * x + rng.uniform(0.0, 2 * np.pi)
        )

    trend -= trend.mean()
    trend_std = np.std(trend) + 1e-8
    return amplitude * trend / trend_std


def stellar_variability(
    x: np.ndarray,
    rng: np.random.Generator,
    amplitude: float,
) -> np.ndarray:
    variability = np.zeros_like(x)
    n_components = int(rng.integers(1, 4))

    for idx in range(n_components):
        freq = rng.uniform(1.0, 6.0)
        phase = rng.uniform(0.0, 2 * np.pi)
        component_amp = amplitude * rng.uniform(0.4, 1.0) / (idx + 1)
        variability += component_amp * np.sin(2 * np.pi * freq * x + phase)

    if rng.uniform() < 0.35:
        variability += 0.4 * amplitude * np.sin(
            2 * np.pi * rng.uniform(6.0, 12.0) * x + rng.uniform(0.0, 2 * np.pi)
        )

    return variability


def apply_trapezoid_dip(
    flux: np.ndarray,
    time: np.ndarray,
    center: float,
    duration: float,
    depth: float,
    ingress_fraction: float,
) -> np.ndarray:
    updated_flux = flux.copy()
    duration = max(duration, 1e-3)
    depth = max(depth, 0.0)

    ingress = np.clip(ingress_fraction * duration, 0.05 * duration, 0.35 * duration)
    flat_duration = max(duration - 2.0 * ingress, 0.15 * duration)

    distance = np.abs(time - center)
    in_flat = distance <= flat_duration / 2.0
    in_wings = (distance > flat_duration / 2.0) & (distance <= duration / 2.0)

    profile = np.zeros_like(time)
    profile[in_flat] = 1.0

    wing_width = (duration - flat_duration) / 2.0
    if wing_width > 0.0:
        profile[in_wings] = 1.0 - (distance[in_wings] - flat_duration / 2.0) / wing_width

    updated_flux -= depth * np.clip(profile, 0.0, 1.0)
    return updated_flux


def transit_model(
    time: np.ndarray,
    period: float,
    duration: float,
    depth: float,
    phase: float,
    ingress_fraction: float,
    rng: np.random.Generator,
    timing_jitter_std: float,
    depth_jitter: float,
    duration_jitter: float,
) -> np.ndarray:
    flux = np.ones_like(time)

    first_center = phase
    while first_center > time[0]:
        first_center -= period

    centers = np.arange(first_center, time[-1] + period, period)
    for center in centers:
        local_center = center + rng.normal(0.0, timing_jitter_std)
        local_depth = depth * np.clip(1.0 + rng.normal(0.0, depth_jitter), 0.5, 1.6)
        local_duration = duration * np.clip(
            1.0 + rng.normal(0.0, duration_jitter),
            0.65,
            1.5,
        )
        flux = apply_trapezoid_dip(
            flux=flux,
            time=time,
            center=local_center,
            duration=local_duration,
            depth=local_depth,
            ingress_fraction=ingress_fraction,
        )

    return flux


def generate_positive_sample(
    seq_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    time = np.linspace(0.0, 30.0, seq_len)
    x = np.linspace(0.0, 1.0, seq_len)

    period = rng.uniform(2.5, 9.0)
    duration = rng.uniform(0.18, 0.9)
    depth = rng.uniform(0.003, 0.035)
    phase = rng.uniform(0.0, period)

    flux = transit_model(
        time=time,
        period=period,
        duration=duration,
        depth=depth,
        phase=phase,
        ingress_fraction=rng.uniform(0.12, 0.28),
        rng=rng,
        timing_jitter_std=rng.uniform(0.0, 0.08 * duration) if rng.uniform() < 0.35 else 0.0,
        depth_jitter=rng.uniform(0.03, 0.12) if rng.uniform() < 0.4 else 0.0,
        duration_jitter=rng.uniform(0.02, 0.08) if rng.uniform() < 0.4 else 0.0,
    )

    flux += low_frequency_trend(x, rng, amplitude=rng.uniform(0.0, 0.012))

    if rng.uniform() < 0.7:
        flux += stellar_variability(x, rng, amplitude=rng.uniform(0.001, 0.008))

    if rng.uniform() < 0.35:
        flux += smooth_noise_component(seq_len, rng, amplitude=rng.uniform(0.0005, 0.004))

    flux += rng.normal(0.0, rng.uniform(0.001, 0.012), size=seq_len)
    flux = normalize_curve(flux)
    return flux.astype(np.float32)


def generate_negative_sample(
    seq_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    time = np.linspace(0.0, 30.0, seq_len)
    x = np.linspace(0.0, 1.0, seq_len)
    flux = np.ones_like(time)

    negative_type = rng.choice(
        ["noise_only", "stellar_variability", "trend_dominated", "single_dip", "quasi_transit_like"],
        p=[0.18, 0.25, 0.18, 0.20, 0.19],
    )

    if negative_type == "noise_only":
        flux += smooth_noise_component(seq_len, rng, amplitude=rng.uniform(0.0005, 0.003))
        flux += rng.normal(0.0, rng.uniform(0.002, 0.014), size=seq_len)

    elif negative_type == "stellar_variability":
        flux += low_frequency_trend(x, rng, amplitude=rng.uniform(0.001, 0.010))
        flux += stellar_variability(x, rng, amplitude=rng.uniform(0.003, 0.015))
        flux += smooth_noise_component(seq_len, rng, amplitude=rng.uniform(0.001, 0.004))
        flux += rng.normal(0.0, rng.uniform(0.001, 0.010), size=seq_len)

    elif negative_type == "trend_dominated":
        flux += low_frequency_trend(x, rng, amplitude=rng.uniform(0.006, 0.020))
        flux += smooth_noise_component(seq_len, rng, amplitude=rng.uniform(0.002, 0.008))
        flux += rng.normal(0.0, rng.uniform(0.001, 0.008), size=seq_len)

    elif negative_type == "single_dip":
        flux += low_frequency_trend(x, rng, amplitude=rng.uniform(0.001, 0.012))
        if rng.uniform() < 0.6:
            flux += stellar_variability(x, rng, amplitude=rng.uniform(0.001, 0.008))
        flux += smooth_noise_component(seq_len, rng, amplitude=rng.uniform(0.001, 0.005))
        flux += rng.normal(0.0, rng.uniform(0.001, 0.010), size=seq_len)

        flux = apply_trapezoid_dip(
            flux=flux,
            time=time,
            center=rng.uniform(3.0, 27.0),
            duration=rng.uniform(0.2, 1.5),
            depth=rng.uniform(0.004, 0.030),
            ingress_fraction=rng.uniform(0.20, 0.45),
        )

        if rng.uniform() < 0.35:
            bump_center = rng.uniform(4.0, 26.0)
            bump_width = rng.uniform(0.3, 1.2)
            bump_amp = rng.uniform(0.002, 0.008)
            flux += bump_amp * np.exp(-0.5 * ((time - bump_center) / bump_width) ** 2)

    else:
        flux += low_frequency_trend(x, rng, amplitude=rng.uniform(0.001, 0.010))
        flux += stellar_variability(x, rng, amplitude=rng.uniform(0.001, 0.010))
        flux += smooth_noise_component(seq_len, rng, amplitude=rng.uniform(0.001, 0.004))
        flux += rng.normal(0.0, rng.uniform(0.001, 0.009), size=seq_len)

        n_events = int(rng.integers(2, 4))
        base_gap = rng.uniform(4.0, 8.0)
        centers = rng.uniform(1.0, 5.0) + np.arange(n_events) * base_gap
        centers = centers + rng.normal(0.0, rng.uniform(0.5, 1.5), size=n_events)

        for center in centers:
            flux = apply_trapezoid_dip(
                flux=flux,
                time=time,
                center=float(np.clip(center, time[0] + 1.0, time[-1] - 1.0)),
                duration=rng.uniform(0.15, 0.8),
                depth=rng.uniform(0.002, 0.012),
                ingress_fraction=rng.uniform(0.15, 0.40),
            )

    flux = normalize_curve(flux)
    return flux.astype(np.float32)


def build_dataset(
    n_samples: int = 4000,
    seq_len: int = 256,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    X = []
    y = []

    half = n_samples // 2
    for _ in range(half):
        X.append(generate_positive_sample(seq_len, rng))
        y.append(1)

    for _ in range(n_samples - half):
        X.append(generate_negative_sample(seq_len, rng))
        y.append(0)

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)

    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]
    return X, y


def select_diverse_examples(
    X: np.ndarray,
    indices: np.ndarray,
    n_examples: int,
) -> np.ndarray:
    if len(indices) <= n_examples:
        return indices

    diversity_score = np.std(X[indices], axis=1) + 0.5 * np.abs(np.min(X[indices], axis=1))
    ordered = indices[np.argsort(diversity_score)]
    positions = np.linspace(0, len(ordered) - 1, n_examples, dtype=int)
    return ordered[positions]


def save_examples(
    X: np.ndarray,
    y: np.ndarray,
    out_path: str,
    n_examples: int = 6,
) -> None:
    pos_idx = select_diverse_examples(X, np.where(y == 1)[0], n_examples)
    neg_idx = select_diverse_examples(X, np.where(y == 0)[0], n_examples)

    fig, axes = plt.subplots(2, n_examples, figsize=(3.1 * n_examples, 6.2), sharex=True)
    fig.suptitle("Synthetic Sequential Light Curves", fontsize=14)

    for col, idx in enumerate(pos_idx):
        axes[0, col].plot(X[idx], color="tab:blue", linewidth=1.4)
        axes[0, col].grid(True, alpha=0.25)
        axes[0, col].set_title(f"Positive {col + 1}")

    for col, idx in enumerate(neg_idx):
        axes[1, col].plot(X[idx], color="tab:orange", linewidth=1.4)
        axes[1, col].grid(True, alpha=0.25)
        axes[1, col].set_title(f"Negative {col + 1}")

    axes[0, 0].set_ylabel("Transit\nnormalized flux")
    axes[1, 0].set_ylabel("No transit\nnormalized flux")

    for axis in axes[1]:
        axis.set_xlabel("Cadence")

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    seed_everything(42)

    ensure_dir(OUTPUT_DIR)

    X, y = build_dataset(n_samples=4000, seq_len=256, seed=42)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    np.save(project_path("outputs", "sequential", "X_train.npy"), X_train)
    np.save(project_path("outputs", "sequential", "X_val.npy"), X_val)
    np.save(project_path("outputs", "sequential", "X_test.npy"), X_test)

    np.save(project_path("outputs", "sequential", "y_train.npy"), y_train)
    np.save(project_path("outputs", "sequential", "y_val.npy"), y_val)
    np.save(project_path("outputs", "sequential", "y_test.npy"), y_test)

    save_examples(
        X=X,
        y=y,
        out_path=project_path("outputs", "sequential", "synthetic_examples.png"),
    )

    summary = pd.DataFrame(
        {
            "split": ["train", "val", "test"],
            "size": [len(X_train), len(X_val), len(X_test)],
            "positives": [int(y_train.sum()), int(y_val.sum()), int(y_test.sum())],
            "negatives": [
                int((1 - y_train).sum()),
                int((1 - y_val).sum()),
                int((1 - y_test).sum()),
            ],
            "positive_fraction": [
                float(y_train.mean()),
                float(y_val.mean()),
                float(y_test.mean()),
            ],
            "sequence_length": [X_train.shape[1], X_val.shape[1], X_test.shape[1]],
        }
    )
    summary.to_csv(project_path("outputs", "sequential", "dataset_summary.csv"), index=False)

    print("Saved dataset to outputs/sequential/")


if __name__ == "__main__":
    main()
