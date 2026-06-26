#!/usr/bin/env python3
"""
Article visualizations for 3D-CAE with SAM3D.

The script trains Conv3DAutoencoder with the SAM3D attention blocks on K-fold
splits, saves fold metrics, learning curves, reconstruction examples,
height-wise errors, attention maps and Jacobian norm estimates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import load_data, normalize_data
from metrics import compute_height_metrics, compute_reconstruction_metrics
from models import Conv3DAutoencoder, SpatialAttentionModule3D
from train import WeightedHeightLoss, train_epoch, validate_epoch


METRIC_KEYS = ("mse", "rmse", "mae", "relative_l2", "ssim")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and visualize 3D-CAE-SAM3D for article figures."
    )
    parser.add_argument("--config", default="configs/config.yaml", help="YAML config path")
    parser.add_argument("--output-dir", default="results/sam3d_article", help="Output directory")
    parser.add_argument("--latent-dim", type=int, default=None, help="Latent dimension")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--skip-cv", action="store_true", help="Skip K-fold cross-validation")
    parser.add_argument("--skip-full-training", action="store_true", help="Skip final train/val training run")
    parser.add_argument("--full-val-split", type=float, default=None, help="Validation fraction for final train/val run")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for quicker runs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--log-every", type=int, default=None, help="Epoch logging interval")
    parser.add_argument("--jacobian-samples", type=int, default=32, help="Random validation frames for Jacobian norm distributions")
    parser.add_argument("--reconstruction-examples", type=int, default=5, help="Validation frames for reconstruction figure")
    parser.add_argument(
        "--jacobian-target",
        choices=("encoder", "input_to_latent"),
        default="encoder",
        help="Jacobian target. encoder/input_to_latent means d z / d x.",
    )
    parser.add_argument("--skip-jacobian", action="store_true", help="Skip Jacobian norm estimates")
    parser.add_argument("--skip-attention", action="store_true", help="Skip SAM3D attention map plots")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for a smoke test")
    parser.add_argument("--synthetic-samples", type=int, default=24, help="Synthetic sample count")
    parser.add_argument("--synthetic-shape", nargs=3, type=int, default=[8, 24, 20], help="D H W")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_synthetic_data(n_samples: int, shape: tuple[int, int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    depth, height, width = shape
    z = np.linspace(0.0, 1.0, depth)[:, None, None]
    y = np.linspace(-1.0, 1.0, height)[None, :, None]
    x = np.linspace(-1.0, 1.0, width)[None, None, :]
    data = []
    for i in range(n_samples):
        cx, cy = rng.uniform(-0.45, 0.45, size=2)
        plume = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / rng.uniform(0.03, 0.10))
        vertical = np.exp(-2.2 * z) * (1.0 + 0.25 * np.sin((i + 1) * math.pi * z))
        background = 0.08 * np.sin((i + 1) * x) + 0.05 * np.cos((i + 2) * y)
        noise = rng.normal(0.0, 0.015, size=shape)
        data.append(np.maximum(vertical * plume + background + noise, 0.0))
    return np.asarray(data, dtype=np.float32)


def prepare_data(config: dict, args: argparse.Namespace) -> np.ndarray:
    if args.synthetic:
        print("Synthetic mode: generated data will be used.")
        data = make_synthetic_data(
            args.synthetic_samples,
            tuple(args.synthetic_shape),
            seed=args.seed or config["data"].get("random_seed", 42),
        )
    else:
        data = load_data(
            config["data"]["files"],
            base_dir=config["data"]["dataset_dir"],
            variable_name=config["data"].get("variable_name", "co"),
            verbose=True,
        )
    data = normalize_data(data, verbose=True)
    if args.max_samples is not None and args.max_samples < len(data):
        rng = np.random.default_rng(args.seed or config["data"].get("random_seed", 42))
        selected = np.sort(rng.choice(len(data), size=args.max_samples, replace=False))
        data = data[selected]
        print(f"Using sample subset: {len(data)}")
    return data.astype(np.float32)


def split_train_val(data: np.ndarray, val_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < val_split < 1:
        raise ValueError("--full-val-split must be between 0 and 1")
    if len(data) < 2:
        raise ValueError("Need at least two samples for train/val split")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))
    n_val = max(1, int(round(len(data) * val_split)))
    n_val = min(n_val, len(data) - 1)
    val_indices = np.sort(indices[:n_val])
    train_indices = np.sort(indices[n_val:])
    return train_indices, val_indices


def make_loader(data: np.ndarray, indices: np.ndarray, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    tensor = torch.tensor(data[indices], dtype=torch.float32).unsqueeze(1)
    return DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def unwrap_loader(loader: DataLoader):
    for (batch,) in loader:
        yield batch


def evaluate_model(model: torch.nn.Module, data: np.ndarray, indices: np.ndarray, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(data, indices, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    recon_batches = []
    latent_batches = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstructed, latent = model(batch)
            recon_batches.append(reconstructed.cpu().numpy()[:, 0])
            latent_batches.append(latent.cpu().numpy())
    return np.concatenate(recon_batches, axis=0), np.concatenate(latent_batches, axis=0)


def train_fold(
    data: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    config: dict,
    args: argparse.Namespace,
    fold_id: int,
    output_dir: Path,
    device: torch.device,
    run_name: str | None = None,
    run_label: str | None = None,
) -> dict:
    latent_dim = args.latent_dim or config["model"]["latent_dim"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    epochs = args.epochs or config["training"]["num_epochs"]
    lr = args.learning_rate or config["training"]["learning_rate"]
    patience = args.patience or config["training"]["early_stopping"]["patience"]
    min_delta = config["training"]["early_stopping"].get("min_delta", 0.0)
    log_every = args.log_every or config["logging"].get("log_every_n_epochs", 5)
    input_shape = (1, *data.shape[1:])

    display_name = run_label or f"Fold {fold_id}"
    print(f"\n{display_name}: train={len(train_indices)}, val={len(val_indices)}, latent_dim={latent_dim}")
    model = Conv3DAutoencoder(
        latent_dim=latent_dim,
        input_shape=input_shape,
        dropout_rate=config["model"].get("dropout_rate", 0.1),
        use_attention=True,
    ).to(device)

    criterion = WeightedHeightLoss(
        alpha=config["loss"]["alpha"],
        beta=config["loss"]["beta"],
        weight_first_5=config["loss"]["height_weight"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=config["model"].get("weight_decay", 0.0),
    )

    train_loader = make_loader(data, train_indices, batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(data, val_indices, batch_size, shuffle=False, num_workers=args.num_workers)

    history = {key: [] for key in ("epoch", "train_loss", "train_mse", "train_mae", "val_loss", "val_mse", "val_mae")}
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_mse, train_mae = train_epoch(
            model, unwrap_loader(train_loader), criterion, optimizer, device
        )
        val_loss, val_mse, val_mae = validate_epoch(
            model, unwrap_loader(val_loader), criterion, device
        )

        for key, value in (
            ("epoch", epoch),
            ("train_loss", train_loss),
            ("train_mse", train_mse),
            ("train_mae", train_mae),
            ("val_loss", val_loss),
            ("val_mse", val_mse),
            ("val_mae", val_mae),
        ):
            history[key].append(float(value))

        improved = best_val_loss - val_loss > min_delta
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % log_every == 0 or patience_counter >= patience:
            print(
                f"{display_name} epoch {epoch:03d}/{epochs}: "
                f"train_loss={train_loss:.5f}, val_loss={val_loss:.5f}, "
                f"val_mse={val_mse:.5f}"
            )

        if patience_counter >= patience:
            print(f"{display_name}: early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    reconstructed, latents = evaluate_model(model, data, val_indices, batch_size, device)
    val_target = data[val_indices]
    metrics = compute_reconstruction_metrics(val_target, reconstructed)
    height_metrics = compute_height_metrics(val_target, reconstructed)
    elapsed = time.time() - start

    fold_dir = output_dir / (run_name or f"fold_{fold_id:02d}")
    fold_dir.mkdir(parents=True, exist_ok=True)
    save_history_csv(history, fold_dir / "history.csv")
    save_json(history, fold_dir / "history.json")
    np.save(fold_dir / "val_reconstructed.npy", reconstructed)
    np.save(fold_dir / "val_latents.npy", latents)
    np.save(fold_dir / "val_indices.npy", val_indices)
    np.savez(fold_dir / "height_metrics.npz", **height_metrics)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "latent_dim": latent_dim,
            "input_shape": input_shape,
            "fold": fold_id,
            "best_epoch": best_epoch,
            "metrics": metrics,
        },
        fold_dir / "best_model.pth",
    )

    row = {
        "run": run_label or f"fold_{fold_id}",
        "fold": fold_id,
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_sec": elapsed,
        "latent_dim": latent_dim,
        **{key: metrics[key] for key in METRIC_KEYS},
    }
    return {
        "model": model,
        "history": history,
        "metrics": metrics,
        "row": row,
        "height_metrics": height_metrics,
        "reconstructed": reconstructed,
        "latents": latents,
        "val_indices": val_indices,
    }


def save_json(obj: object, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_history_csv(history: dict[str, list[float]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history.keys()))
        writer.writeheader()
        for values in zip(*history.values()):
            writer.writerow(dict(zip(history.keys(), values)))


def save_rows_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def setup_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def plot_training_curves(fold_results: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    pairs = (
        ("loss", "train_loss", "val_loss", "Loss"),
        ("mse", "train_mse", "val_mse", "MSE"),
        ("mae", "train_mae", "val_mae", "MAE"),
    )
    for ax, (_, train_key, val_key, ylabel) in zip(axes, pairs):
        for result in fold_results:
            history = result["history"]
            fold = result["row"]["fold"]
            ax.plot(history["epoch"], history[train_key], alpha=0.45, linestyle="--", label=f"train fold {fold}")
            ax.plot(history["epoch"], history[val_key], alpha=0.85, label=f"val fold {fold}")
        setup_axes(ax, ylabel, "Epoch", ylabel)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves_by_fold.png", dpi=250)
    plt.close(fig)


def plot_full_training_curves(history: dict[str, list[float]], output_dir: Path) -> None:
    plots = (
        ("loss", "train_loss", "val_loss", "Loss"),
        ("mse", "train_mse", "val_mse", "MSE"),
        ("mae", "train_mae", "val_mae", "MAE"),
    )
    for name, train_key, val_key, ylabel in plots:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(history["epoch"], history[train_key], label="train", linewidth=2.0)
        ax.plot(history["epoch"], history[val_key], label="val", linewidth=2.0)
        setup_axes(ax, f"Training {ylabel}", "Epoch", ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"full_training_{name}.png", dpi=250)
        plt.close(fig)


def plot_cv_metrics(rows: list[dict], output_dir: Path) -> None:
    folds = [row["fold"] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for ax, metric in zip(axes, METRIC_KEYS):
        values = [row[metric] for row in rows]
        ax.plot(folds, values, marker="o", linewidth=1.8)
        mean = float(np.mean(values))
        std = float(np.std(values))
        ax.axhline(mean, color="black", linestyle=":", linewidth=1.2, label=f"mean={mean:.4g}")
        ax.fill_between(folds, mean - std, mean + std, color="gray", alpha=0.15, label="±1 std")
        setup_axes(ax, metric.upper(), "Fold", metric)
        ax.legend(fontsize=8)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "cross_validation_metrics.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    metric_values = [[row[metric] for row in rows] for metric in METRIC_KEYS]
    ax.boxplot(metric_values, labels=[metric.upper() for metric in METRIC_KEYS], showmeans=True)
    setup_axes(ax, "Cross-validation metric spread", "Metric", "Value")
    fig.tight_layout()
    fig.savefig(output_dir / "cross_validation_metric_boxplots.png", dpi=250)
    plt.close(fig)


def plot_reconstruction_examples(data: np.ndarray, fold_result: dict, output_dir: Path, max_examples: int = 5) -> None:
    val_indices = fold_result["val_indices"]
    reconstructed = fold_result["reconstructed"]
    depth = data.shape[1]
    levels = sorted(set([0, min(4, depth - 1), depth // 2, depth - 1]))
    examples = min(max_examples, len(val_indices))
    fig, axes = plt.subplots(
        examples * len(levels),
        3,
        figsize=(10, 2.5 * examples * len(levels)),
        squeeze=False,
    )
    for ex in range(examples):
        target = data[val_indices[ex]]
        recon = reconstructed[ex]
        error = np.abs(recon - target)
        for j, level in enumerate(levels):
            row = ex * len(levels) + j
            panels = (target[level], recon[level], error[level])
            titles = (f"True: sample {val_indices[ex]}, level {level}", "Reconstruction", "Absolute error")
            for col, (panel, title) in enumerate(zip(panels, titles)):
                im = axes[row, col].imshow(panel, cmap="viridis" if col < 2 else "magma")
                axes[row, col].set_title(title, fontsize=9)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_dir / "reconstruction_examples.png", dpi=250)
    plt.close(fig)


def plot_height_errors(fold_results: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for result in fold_results:
        fold = result["row"]["fold"]
        levels = np.arange(len(result["height_metrics"]["mse_by_height"]))
        axes[0].plot(levels, result["height_metrics"]["mse_by_height"], marker="o", markersize=3, label=f"fold {fold}")
        axes[1].plot(levels, result["height_metrics"]["mae_by_height"], marker="o", markersize=3, label=f"fold {fold}")
    setup_axes(axes[0], "MSE by vertical level", "Vertical level", "MSE")
    setup_axes(axes[1], "MAE by vertical level", "Vertical level", "MAE")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "height_error_profiles.png", dpi=250)
    plt.close(fig)


def plot_latent_space(latents: np.ndarray, output_dir: Path) -> None:
    if len(latents) < 3:
        return
    coords = PCA(n_components=2, random_state=0).fit_transform(latents)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=np.arange(len(coords)), cmap="viridis", s=28)
    setup_axes(ax, "Latent space PCA projection", "PC1", "PC2")
    fig.colorbar(scatter, ax=ax, label="Validation sample order")
    fig.tight_layout()
    fig.savefig(output_dir / "latent_space_pca.png", dpi=250)
    plt.close(fig)


def collect_attention_maps(model: torch.nn.Module, sample: torch.Tensor, device: torch.device) -> list[np.ndarray]:
    maps = []
    hooks = []

    def hook(module: SpatialAttentionModule3D, inputs, _output):
        x = inputs[0].detach()
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        std_out = torch.std(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out, std_out], dim=1)
        attn = module.sigmoid(module.conv3d(combined))
        maps.append(attn.detach().cpu().numpy()[0, 0])

    for module in model.modules():
        if isinstance(module, SpatialAttentionModule3D):
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    with torch.no_grad():
        model(sample.to(device))
    for h in hooks:
        h.remove()
    return maps


def plot_attention_maps(model: torch.nn.Module, data: np.ndarray, sample_index: int, output_dir: Path, device: torch.device) -> None:
    sample = torch.tensor(data[sample_index : sample_index + 1], dtype=torch.float32).unsqueeze(1)
    maps = collect_attention_maps(model, sample, device)
    if not maps:
        print("No SAM3D attention maps were captured.")
        return
    cols = min(3, len(maps))
    rows = int(math.ceil(len(maps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), squeeze=False)
    for i, attn in enumerate(maps):
        ax = axes[i // cols, i % cols]
        level = attn.shape[0] // 2
        im = ax.imshow(attn[level], cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"SAM3D layer {i + 1}, level {level}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for i in range(len(maps), rows * cols):
        axes[i // cols, i % cols].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "sam3d_attention_maps.png", dpi=250)
    plt.close(fig)


def compute_input_latent_jacobian_norms(model: torch.nn.Module, x: torch.Tensor) -> dict[str, float]:
    """Compute norms of dz/dx for one frame with torch.autograd.functional.jacobian."""
    x = x.detach().requires_grad_(True)

    def encoder_only(inp: torch.Tensor) -> torch.Tensor:
        _, latent = model(inp)
        return latent.reshape(-1)

    jacobian = torch.autograd.functional.jacobian(
        encoder_only,
        x,
        create_graph=False,
        strict=False,
        vectorize=True,
    )
    jacobian_matrix = jacobian.reshape(jacobian.shape[0], -1)
    spectral_norm = torch.linalg.matrix_norm(jacobian_matrix, ord=2)
    frobenius_norm = torch.linalg.matrix_norm(jacobian_matrix, ord="fro")
    return {
        "spectral_norm": float(spectral_norm.detach().cpu()),
        "frobenius_norm": float(frobenius_norm.detach().cpu()),
        "latent_dim": int(jacobian_matrix.shape[0]),
        "input_size": int(jacobian_matrix.shape[1]),
    }


def estimate_jacobian_norms(
    model: torch.nn.Module,
    data: np.ndarray,
    val_indices: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict]:
    model.eval()
    rng_seed = args.seed if args.seed is not None else 42
    rng = np.random.default_rng(rng_seed)
    sample_count = min(args.jacobian_samples, len(val_indices))
    chosen = np.sort(rng.choice(val_indices, size=sample_count, replace=False))
    rows = []
    target_name = "input_to_latent"

    print("Computing input-to-latent Jacobian norms with torch.autograd.functional.jacobian...")
    for sample_index in tqdm(chosen, desc="Jacobian input_to_latent"):
        x = torch.tensor(data[sample_index : sample_index + 1], dtype=torch.float32, device=device).unsqueeze(1)
        try:
            norms = compute_input_latent_jacobian_norms(model, x)
        except RuntimeError as exc:
            print(f"Jacobian failed for sample {sample_index}: {exc}")
            norms = {
                "spectral_norm": float("nan"),
                "frobenius_norm": float("nan"),
                "latent_dim": int(args.latent_dim or 0),
                "input_size": int(np.prod(data.shape[1:])),
            }
        rows.append(
            {
                "sample_index": int(sample_index),
                "target": target_name,
                **norms,
            }
        )

    save_rows_csv(rows, output_dir / "jacobian_norms.csv")
    with open(output_dir / "jacobian_norms.json", "w") as f:
        json.dump(rows, f, indent=2)

    finite_rows = [row for row in rows if np.isfinite(row["spectral_norm"])]
    plot_jacobian_norm_distributions(finite_rows, output_dir)
    return rows


def plot_jacobian_norm_distributions(rows: list[dict], output_dir: Path) -> None:
    spectral = np.asarray([row["spectral_norm"] for row in rows], dtype=float)
    frobenius = np.asarray([row["frobenius_norm"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if len(rows):
        axes[0].hist(spectral, bins="auto", color="#3b6ea8", alpha=0.82, edgecolor="black")
        axes[0].axvline(float(np.mean(spectral)), color="black", linestyle=":", label=f"mean={np.mean(spectral):.4g}")
        axes[0].legend(fontsize=8)
        axes[1].hist(frobenius, bins="auto", color="#8f5d7a", alpha=0.82, edgecolor="black")
        axes[1].axvline(float(np.mean(frobenius)), color="black", linestyle=":", label=f"mean={np.mean(frobenius):.4g}")
        axes[1].legend(fontsize=8)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No finite Jacobian norm values", ha="center", va="center", transform=ax.transAxes)
    setup_axes(axes[0], "Spectral norm distribution", "Spectral norm", "Frequency")
    setup_axes(axes[1], "Frobenius norm distribution", "Frobenius norm", "Frequency")
    fig.tight_layout()
    fig.savefig(output_dir / "jacobian_norm_distributions.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if len(rows):
        ax.boxplot([spectral, frobenius], labels=["Spectral", "Frobenius"], showmeans=True)
    else:
        ax.text(0.5, 0.5, "No finite Jacobian norm values", ha="center", va="center", transform=ax.transAxes)
    setup_axes(ax, "Input-to-latent Jacobian norm distributions", "Norm", "Value")
    fig.tight_layout()
    fig.savefig(output_dir / "jacobian_norm_boxplots.png", dpi=250)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["data"].get("random_seed", 42)
    set_seed(seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json({"args": vars(args), "config": config}, output_dir / "run_config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    data = prepare_data(config, args)
    if not args.skip_cv and len(data) < args.folds:
        raise ValueError(f"Need at least {args.folds} samples for {args.folds}-fold CV, got {len(data)}")

    fold_results = []
    rows = []
    best_result = None

    if not args.skip_cv:
        kfold = KFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold_id, (train_indices, val_indices) in enumerate(kfold.split(np.arange(len(data))), start=1):
            set_seed(seed + fold_id)
            result = train_fold(data, train_indices, val_indices, config, args, fold_id, output_dir, device)
            fold_results.append(result)
            rows.append(result["row"])
            if best_result is None or result["row"]["mse"] < best_result["row"]["mse"]:
                best_result = result

        save_rows_csv(rows, output_dir / "cv_metrics.csv")
        save_json(rows, output_dir / "cv_metrics.json")

        summary = {}
        for metric in METRIC_KEYS:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values))
            summary[f"{metric}_min"] = float(np.min(values))
            summary[f"{metric}_max"] = float(np.max(values))
        save_json(summary, output_dir / "cv_summary.json")

        plot_training_curves(fold_results, output_dir)
        plot_cv_metrics(rows, output_dir)
        plot_height_errors(fold_results, output_dir)
        if best_result is not None:
            plot_reconstruction_examples(
                data,
                best_result,
                output_dir,
                max_examples=args.reconstruction_examples,
            )
            plot_latent_space(best_result["latents"], output_dir)
            if not args.skip_attention:
                plot_attention_maps(
                    best_result["model"],
                    data,
                    int(best_result["val_indices"][0]),
                    output_dir,
                    device,
                )
            if not args.skip_jacobian:
                estimate_jacobian_norms(
                    best_result["model"],
                    data,
                    best_result["val_indices"],
                    output_dir,
                    args,
                    device,
                )

    if not args.skip_full_training:
        val_split = args.full_val_split
        if val_split is None:
            val_split = config["data"].get("val_split_ratio", 0.1)
        train_indices, val_indices = split_train_val(data, val_split=val_split, seed=seed)
        full_dir = output_dir / "full_training"
        set_seed(seed + 10_000)
        full_result = train_fold(
            data,
            train_indices,
            val_indices,
            config,
            args,
            fold_id=0,
            output_dir=output_dir,
            device=device,
            run_name="full_training",
            run_label="full_train_val",
        )
        save_rows_csv([full_result["row"]], full_dir / "full_metrics.csv")
        save_json(full_result["row"], full_dir / "full_metrics.json")
        save_json(full_result["metrics"], full_dir / "full_reconstruction_metrics.json")
        plot_full_training_curves(full_result["history"], full_dir)
        plot_height_errors([full_result], full_dir)
        plot_reconstruction_examples(
            data,
            full_result,
            full_dir,
            max_examples=args.reconstruction_examples,
        )
        plot_latent_space(full_result["latents"], full_dir)
        if not args.skip_attention:
            plot_attention_maps(
                full_result["model"],
                data,
                int(full_result["val_indices"][0]),
                full_dir,
                device,
            )
        if not args.skip_jacobian:
            estimate_jacobian_norms(
                full_result["model"],
                data,
                full_result["val_indices"],
                full_dir,
                args,
                device,
            )

    print("\nSaved article assets:")
    for path in sorted(output_dir.glob("*")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
