import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from model import TransitCNN
from utils import (
    ensure_dir,
    get_torch_device,
    project_path,
    save_json,
    seed_everything,
)


OUTPUT_DIR = project_path("outputs", "sequential")
CHECKPOINT_DIR = project_path("checkpoints", "sequential")
CHECKPOINT_PATH = project_path("checkpoints", "sequential", "best_model.pt")


def load_split(split: str, base_dir: str = OUTPUT_DIR):
    X = np.load(os.path.join(base_dir, f"X_{split}.npy"))
    y = np.load(os.path.join(base_dir, f"y_{split}.npy"))

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y


def evaluate(model, loader, device):
    model.eval()
    losses = []
    probs_all = []
    labels_all = []
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            probs = torch.sigmoid(logits)

            losses.append(loss.item())
            probs_all.append(probs.cpu().numpy())
            labels_all.append(yb.cpu().numpy())

    probs_all = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all)
    auc = roc_auc_score(labels_all, probs_all)
    return float(np.mean(losses)), float(auc)


def save_training_plots(
    train_losses,
    val_losses,
    val_aucs,
    best_epoch: int,
) -> None:
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", markersize=3, label="train_loss")
    plt.plot(epochs, val_losses, marker="o", markersize=3, label="val_loss")
    if best_epoch > 0:
        plt.axvline(best_epoch, color="tab:green", linestyle="--", alpha=0.7, label="best_epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("Loss Curves")
    plt.tight_layout()
    plt.savefig(project_path("outputs", "sequential", "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_aucs, marker="o", markersize=3, label="val_auc", color="tab:purple")
    if best_epoch > 0:
        best_auc = val_aucs[best_epoch - 1]
        plt.scatter([best_epoch], [best_auc], color="tab:red", label="best_val_auc", zorder=3)
        plt.axvline(best_epoch, color="tab:green", linestyle="--", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("Validation AUC")
    plt.tight_layout()
    plt.savefig(project_path("outputs", "sequential", "val_auc_curve.png"), dpi=200)
    plt.close()


def main():
    seed_everything(42)

    ensure_dir(CHECKPOINT_DIR)
    ensure_dir(OUTPUT_DIR)

    device = get_torch_device()

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=128,
        shuffle=False,
    )

    model = TransitCNN(input_length=X_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    n_epochs = 40
    patience = 5
    epochs_without_improvement = 0

    best_val_auc = -np.inf
    best_epoch = 0

    train_losses = []
    val_losses = []
    val_aucs = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        val_loss, val_auc = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        print(
            f"Epoch {epoch:02d}/{n_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_auc={val_auc:.4f}"
        )

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_length": X_train.shape[-1],
                    "best_val_auc": best_val_auc,
                    "best_epoch": best_epoch,
                },
                CHECKPOINT_PATH,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"after {patience} epochs without validation AUC improvement."
            )
            break

    epochs_run = len(train_losses)
    save_training_plots(train_losses, val_losses, val_aucs, best_epoch)

    training_summary = {
        "best_val_auc": float(best_val_auc),
        "epoch_of_best_model": int(best_epoch),
        "number_of_epochs_run": int(epochs_run),
        "best_checkpoint_path": CHECKPOINT_PATH,
        "best_epoch_metrics": {
            "train_loss": float(train_losses[best_epoch - 1]),
            "val_loss": float(val_losses[best_epoch - 1]),
            "val_auc": float(val_aucs[best_epoch - 1]),
        },
        "final_epoch_metrics": {
            "epoch": int(epochs_run),
            "train_loss": float(train_losses[-1]),
            "val_loss": float(val_losses[-1]),
            "val_auc": float(val_aucs[-1]),
        },
    }
    save_json(training_summary, project_path("outputs", "sequential", "training_summary.json"))

    print(
        "Training complete. "
        f"Best val AUC: {best_val_auc:.4f} at epoch {best_epoch}. "
        f"Ran for {epochs_run} epochs."
    )


if __name__ == "__main__":
    main()
