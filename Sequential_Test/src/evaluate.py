import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

from model import TransitCNN
from utils import ensure_dir, get_torch_device, project_path, save_json


OUTPUT_DIR = project_path("outputs", "sequential")
CHECKPOINT_PATH = project_path("checkpoints", "sequential", "best_model.pt")


def load_split(split: str, base_dir: str = OUTPUT_DIR):
    X = np.load(os.path.join(base_dir, f"X_{split}.npy"))
    y = np.load(os.path.join(base_dir, f"y_{split}.npy"))

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X, y.astype(np.int64), X_tensor, y_tensor


def predict_probs(model, loader, device):
    probs_all = []
    labels_all = []

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()

            probs_all.append(probs)
            labels_all.append(yb.numpy())

    probs_all = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all).astype(np.int64)
    return labels_all, probs_all


def compute_threshold_metrics(y_true, y_prob, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_f1 = -1.0
    threshold_scores = []

    for threshold in thresholds:
        score = f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        threshold_scores.append(score)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return float(best_threshold), float(best_f1), thresholds, np.array(threshold_scores)


def save_roc_curve(y_true, y_prob, out_path: str) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return float(roc_auc)


def save_pr_curve(y_true, y_prob, out_path: str) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    average_precision = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AP = {average_precision:.4f}", linewidth=2, color="tab:green")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return float(average_precision)


def save_confusion_matrix(y_true, y_pred, threshold: float, out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5.4, 5.0))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix (threshold={threshold:.3f})")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["No transit", "Transit"])
    plt.yticks([0, 1], ["No transit", "Transit"])

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(col, row, cm[row, col], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_threshold_sweep(thresholds, scores, best_threshold: float, out_path: str) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(thresholds, scores, linewidth=2)
    plt.axvline(best_threshold, linestyle="--", color="tab:red", label=f"best={best_threshold:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Validation F1")
    plt.title("Validation Threshold Sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_error_examples(
    curves: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    title: str,
    out_path: str,
) -> None:
    indices = np.where(mask)[0]

    if len(indices) == 0:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No {title.lower()} at tuned threshold.", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return

    error_strength = np.where(y_true[indices] == 0, y_prob[indices], 1.0 - y_prob[indices])
    selected = indices[np.argsort(error_strength)[::-1][:6]]

    n_examples = len(selected)
    n_cols = min(3, n_examples)
    n_rows = math.ceil(n_examples / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.8 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for axis, idx in zip(axes, selected):
        axis.plot(curves[idx], color="tab:red" if y_true[idx] == 1 else "tab:orange", linewidth=1.3)
        axis.set_title(
            f"y={int(y_true[idx])} | p={y_prob[idx]:.3f} | pred={int(y_pred[idx])}",
            fontsize=10,
        )
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("Cadence")
        axis.set_ylabel("Flux")

    for axis in axes[n_examples:]:
        axis.axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ensure_dir(OUTPUT_DIR)

    device = get_torch_device()

    _, _, X_val, y_val = load_split("val")
    X_test_raw, _, X_test, y_test = load_split("test")

    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=128, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model = TransitCNN(input_length=ckpt["input_length"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_labels, val_probs = predict_probs(model, val_loader, device)
    test_labels, test_probs = predict_probs(model, test_loader, device)

    best_threshold, best_val_f1, thresholds, threshold_scores = find_best_threshold(val_labels, val_probs)

    metrics_05 = compute_threshold_metrics(test_labels, test_probs, threshold=0.5)
    metrics_tuned = compute_threshold_metrics(test_labels, test_probs, threshold=best_threshold)

    y_pred_05 = (test_probs >= 0.5).astype(int)
    y_pred_tuned = (test_probs >= best_threshold).astype(int)

    roc_auc = save_roc_curve(
        y_true=test_labels,
        y_prob=test_probs,
        out_path=project_path("outputs", "sequential", "roc_curve.png"),
    )
    average_precision = save_pr_curve(
        y_true=test_labels,
        y_prob=test_probs,
        out_path=project_path("outputs", "sequential", "pr_curve.png"),
    )
    save_confusion_matrix(
        y_true=test_labels,
        y_pred=y_pred_tuned,
        threshold=best_threshold,
        out_path=project_path("outputs", "sequential", "confusion_matrix.png"),
    )
    save_threshold_sweep(
        thresholds=thresholds,
        scores=threshold_scores,
        best_threshold=best_threshold,
        out_path=project_path("outputs", "sequential", "threshold_sweep.png"),
    )
    save_error_examples(
        curves=X_test_raw,
        y_true=test_labels,
        y_prob=test_probs,
        y_pred=y_pred_tuned,
        mask=(test_labels == 0) & (y_pred_tuned == 1),
        title="False Positive Examples",
        out_path=project_path("outputs", "sequential", "false_positive_examples.png"),
    )
    save_error_examples(
        curves=X_test_raw,
        y_true=test_labels,
        y_prob=test_probs,
        y_pred=y_pred_tuned,
        mask=(test_labels == 1) & (y_pred_tuned == 0),
        title="False Negative Examples",
        out_path=project_path("outputs", "sequential", "false_negative_examples.png"),
    )

    predictions = pd.DataFrame(
        {
            "y_true": test_labels.astype(int),
            "y_prob": test_probs,
            "y_pred_05": y_pred_05,
            "y_pred_tuned": y_pred_tuned,
        }
    )
    predictions.to_csv(project_path("outputs", "sequential", "predictions.csv"), index=False)

    metrics = {
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "threshold_0.5": metrics_05,
        "threshold_tuned": {
            **metrics_tuned,
            "validation_f1": float(best_val_f1),
        },
        "validation_threshold_search": {
            "grid_min": 0.05,
            "grid_max": 0.95,
            "num_thresholds": 181,
        },
    }
    save_json(metrics, project_path("outputs", "sequential", "metrics.json"))

    print(
        f"Test ROC-AUC: {roc_auc:.4f} | "
        f"AP: {average_precision:.4f} | "
        f"Tuned threshold: {best_threshold:.3f} (val F1={best_val_f1:.4f})"
    )


if __name__ == "__main__":
    main()
