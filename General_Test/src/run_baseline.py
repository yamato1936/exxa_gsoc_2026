from __future__ import annotations

import argparse

try:
    from .cluster import run_clustering
    from .preprocess import PREPROCESS_MODES
    from .train import train_autoencoder
    from .utils import project_path
except ImportError:
    from cluster import run_clustering
    from preprocess import PREPROCESS_MODES
    from train import train_autoencoder
    from utils import project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full General Test autoencoder + k-means pipeline.")
    parser.add_argument("--data_dir", required=True, help="Directory containing FITS files.")
    parser.add_argument("--output_dir", default=project_path("outputs", "general"))
    parser.add_argument("--checkpoint_dir", default=project_path("checkpoints", "general"))
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--preprocess_mode", choices=PREPROCESS_MODES, default="percentile_minmax")
    parser.add_argument("--lower_percentile", type=float, default=1.0)
    parser.add_argument("--upper_percentile", type=float, default=99.5)
    parser.add_argument("--robust_clip", type=float, default=5.0)
    parser.add_argument("--recon_loss", choices=["mse", "l1"], default="mse")

    augmentation_group = parser.add_mutually_exclusive_group()
    augmentation_group.add_argument("--use_augmentation", dest="use_augmentation", action="store_true")
    augmentation_group.add_argument("--disable_augmentation", dest="use_augmentation", action="store_false")
    parser.set_defaults(use_augmentation=True)
    parser.add_argument("--rotation_deg", type=float, default=90.0)
    parser.add_argument("--use_edge_loss", action="store_true")
    parser.add_argument("--edge_loss_weight", type=float, default=0.1)

    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--embedding_method", choices=["umap", "pca"], default="umap")
    parser.add_argument("--top_k", type=int, default=9)
    parser.add_argument("--random_k", type=int, default=9)
    parser.add_argument("--radial_nbins", type=int, default=100)
    parser.add_argument("--latent_path", default=None)
    parser.add_argument("--metadata_csv", default=None)
    parser.add_argument("--metadata_json", default=None)
    return parser


def main(cli_args: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(cli_args)

    train_result = train_autoencoder(args)
    args.checkpoint_path = train_result["checkpoint_path"]
    run_clustering(args)


if __name__ == "__main__":
    main()
