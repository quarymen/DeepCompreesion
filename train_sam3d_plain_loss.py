#!/usr/bin/env python3
"""
Train 3D-CAE-SAM3D with plain reconstruction loss:

    loss = MSE + beta * MAE

This script is intentionally separate from train.py and visualize_sam3d_article.py
so article training curves match the stated loss function exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import load_data, normalize_data
from metrics import compute_reconstruction_metrics
from models import Conv3DAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAM3D autoencoder with MSE + beta * MAE loss")
    parser.add_argument("--config", default="configs/config.yaml", help="YAML config path")
    parser.add_argument("--output-dir", default="results/sam3d_plain_loss", help="Output directory")
    parser.add_argument("--latent-dim", type=int, default=None, help="Latent dimension")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--dropout-rate", type=float, default=None, help="Dropout rate")
    parser.add_argument("--beta", type=float, default=0.2, help="MAE coefficient in MSE + beta * MAE")
    parser.add_argument("--val-split", type=float, default=None, help="Validation fraction")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=None, help="Early stopping min delta")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--log-every", type=int, default=None, help="Epoch logging interval")
    return parser.parse_args()


class PlainReconstructionLoss(nn.Module):
    """Loss = MSE + beta * MAE over the whole 3D field."""

    def __init__(self, beta: float = 0.2):
        super().__init__()
        self.beta = beta
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, reconstructed: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mse = self.mse(reconstructed, target)
        mae = self.mae(reconstructed, target)
        loss = mse + self.beta * mae
        return loss, mse, mae


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(config: dict, args: argparse.Namespace, seed: int) -> np.ndarray:
    data = load_data(
        config["data"]["files"],
        base_dir=config["data"]["dataset_dir"],
        variable_name=config["data"].get("variable_name", "co"),
        verbose=True,
    )
    data = normalize_data(data, verbose=True).astype(np.float32)
    if args.max_samples is not None and args.max_samples < len(data):
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(data), size=args.max_samples, replace=False))
        data = data[indices]
        print(f"Using sample subset: {len(data)}")
    return data


def split_train_val(data: np.ndarray, val_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < val_split < 1:
        raise ValueError("--val-split must be between 0 and 1")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))
    n_val = max(1, int(round(len(data) * val_split)))
    n_val = min(n_val, len(data) - 1)
    return np.sort(indices[n_val:]), np.sort(indices[:n_val])


def make_loader(data: np.ndarray, indices: np.ndarray, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    tensor = torch.tensor(data[indices], dtype=torch.float32).unsqueeze(1)
    return DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: PlainReconstructionLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "mse": 0.0, "mae": 0.0, "n": 0}

    for (batch,) in loader:
        batch = batch.to(device)
        if is_train:
            optimizer.zero_grad()

        reconstructed, _ = model(batch)
        loss, mse, mae = criterion(reconstructed, batch)

        if is_train:
            loss.backward()
            optimizer.step()

        totals["loss"] += float(loss.item())
        totals["mse"] += float(mse.item())
        totals["mae"] += float(mae.item())
        totals["n"] += 1

    n = max(1, totals["n"])
    return totals["loss"] / n, totals["mse"] / n, totals["mae"] / n


def evaluate_reconstruction(model: nn.Module, data: np.ndarray, indices: np.ndarray, batch_size: int, device: torch.device) -> dict[str, float]:
    loader = make_loader(data, indices, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    reconstructed_batches = []
    with torch.no_grad():
        for (batch,) in loader:
            reconstructed, _ = model(batch.to(device))
            reconstructed_batches.append(reconstructed.cpu().numpy()[:, 0])
    reconstructed = np.concatenate(reconstructed_batches, axis=0)
    return compute_reconstruction_metrics(data[indices], reconstructed)


def save_history(history: dict[str, list[float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(output_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history.keys()))
        writer.writeheader()
        for values in zip(*history.values()):
            writer.writerow(dict(zip(history.keys(), values)))


def setup_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()


def plot_curves(history: dict[str, list[float]], output_dir: Path, beta: float) -> None:
    plots = (
        ("loss", "train_loss", "val_loss", f"Loss = MSE + {beta:g} MAE"),
        ("mse", "train_mse", "val_mse", "MSE"),
        ("mae", "train_mae", "val_mae", "MAE"),
    )
    for name, train_key, val_key, ylabel in plots:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(history["epoch"], history[train_key], label="train", linewidth=2.0)
        ax.plot(history["epoch"], history[val_key], label="val", linewidth=2.0)
        setup_axes(ax, f"Training {ylabel}", ylabel)
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}.png", dpi=250)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["data"].get("random_seed", 42)
    set_seed(seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = prepare_data(config, args, seed)
    val_split = args.val_split if args.val_split is not None else config["data"].get("val_split_ratio", 0.1)
    train_indices, val_indices = split_train_val(data, val_split=val_split, seed=seed)
    print(f"Train/val split: train={len(train_indices)}, val={len(val_indices)}")

    latent_dim = args.latent_dim or config["model"]["latent_dim"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    epochs = args.epochs or config["training"]["num_epochs"]
    lr = args.learning_rate or config["training"]["learning_rate"]
    weight_decay = args.weight_decay if args.weight_decay is not None else config["model"].get("weight_decay", 0.0)
    dropout_rate = args.dropout_rate if args.dropout_rate is not None else config["model"].get("dropout_rate", 0.1)
    patience = args.patience or config["training"]["early_stopping"]["patience"]
    min_delta = args.min_delta if args.min_delta is not None else config["training"]["early_stopping"].get("min_delta", 0.0)
    log_every = args.log_every or config["logging"].get("log_every_n_epochs", 5)

    model = Conv3DAutoencoder(
        latent_dim=latent_dim,
        input_shape=(1, *data.shape[1:]),
        dropout_rate=dropout_rate,
        use_attention=True,
    ).to(device)
    criterion = PlainReconstructionLoss(beta=args.beta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = make_loader(data, train_indices, batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(data, val_indices, batch_size, shuffle=False, num_workers=args.num_workers)

    history = {key: [] for key in ("epoch", "train_loss", "train_mse", "train_mae", "val_loss", "val_mse", "val_mae")}
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_mse, train_mae = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_mse, val_mae = run_epoch(model, val_loader, criterion, device)

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

        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % log_every == 0 or patience_counter >= patience:
            print(
                f"Epoch {epoch:03d}/{epochs}: "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"train_mse={train_mse:.6f}, train_mae={train_mae:.6f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - start
    metrics = evaluate_reconstruction(model, data, val_indices, batch_size, device)

    save_history(history, output_dir)
    plot_curves(history, output_dir, beta=args.beta)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "latent_dim": latent_dim,
            "input_shape": (1, *data.shape[1:]),
            "loss": f"MSE + {args.beta} * MAE",
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "metrics": metrics,
        },
        output_dir / "best_model.pth",
    )
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"elapsed_sec": elapsed, "best_epoch": best_epoch, "best_val_loss": best_val_loss, **metrics}, f, indent=2)
    np.save(output_dir / "train_indices.npy", train_indices)
    np.save(output_dir / "val_indices.npy", val_indices)

    print(f"Saved plain-loss training outputs to {output_dir}")
    print(f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
