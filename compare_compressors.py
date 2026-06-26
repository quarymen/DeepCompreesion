#!/usr/bin/env python3
"""
Compare dimensionality reduction baselines on normalized CO fields.

The script evaluates PCA, DCT, Tensor Train and Tucker/HOSVD with a common
metric set and writes article-ready CSV/JSON tables.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dct import analyze_dct_components
from interpolation import analyze_interpolation_components
from matrix_methods import analyze_matrix_components
from pca import analyze_pca_components
from tensor_train import analyze_tt_components
from tucker import analyze_tucker_components
from metrics import compute_reconstruction_metrics
from wavelet import analyze_wavelet_components


DEFAULT_METHODS = (
    "AE",
    "PCA",
    "DCT",
    "Wavelet",
    "Interpolation",
    "TruncatedSVD",
    "RandomProjection",
    "TT-SVD",
)

MATRIX_METHODS = ("TruncatedSVD", "RandomProjection", "UMAP")

METHOD_ALIASES = {
    "ae": "AE",
    "autoencoder": "AE",
    "3d-cae": "AE",
    "cae": "AE",
    "pca": "PCA",
    "dct": "DCT",
    "wavelet": "Wavelet",
    "waves": "Wavelet",
    "interpolation": "Interpolation",
    "interp": "Interpolation",
    "linear-interpolation": "Interpolation",
    "truncatedsvd": "TruncatedSVD",
    "truncated-svd": "TruncatedSVD",
    "svd": "TruncatedSVD",
    "randomprojection": "RandomProjection",
    "random-projection": "RandomProjection",
    "rp": "RandomProjection",
    "umap": "UMAP",
    "tt": "TT-SVD",
    "tt-svd": "TT-SVD",
    "tensor-train": "TT-SVD",
    "tucker": "Tucker",
    "hosvd": "Tucker",
    "tucker-hosvd": "Tucker",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare compression methods")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", default="results/comparison", help="Where to save tables")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for quick experiments")
    parser.add_argument("--seed", type=int, default=42, help="Seed used when max-samples is set")
    parser.add_argument("--eval-split", type=float, default=0.2, help="Held-out fraction used for metrics")
    parser.add_argument("--components", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "Methods to evaluate. Choices/aliases: AE, PCA, DCT, Wavelet, Interpolation, "
            "TruncatedSVD, RandomProjection, UMAP, TT-SVD, Tucker. "
            "Default: AE PCA DCT Wavelet Interpolation TruncatedSVD RandomProjection TT-SVD."
        ),
    )
    parser.add_argument("--skip-ae", action="store_true", help="Do not train/evaluate autoencoders")
    parser.add_argument(
        "--ae-models",
        nargs="+",
        default=None,
        help=(
            "Autoencoder model classes to evaluate. "
            "Examples: PlainConv3DAutoencoder Conv3DAutoencoder SimpleConv3DAutoencoder."
        ),
    )
    parser.add_argument("--ae-epochs", type=int, default=None, help="Autoencoder epochs; defaults to config value")
    parser.add_argument("--ae-batch-size", type=int, default=None, help="Autoencoder batch size; defaults to config value")
    parser.add_argument("--ae-learning-rate", type=float, default=None, help="Autoencoder LR; defaults to config value")
    parser.add_argument("--ae-patience", type=int, default=None, help="Autoencoder early stopping patience")
    parser.add_argument("--tt-ranks", nargs="+", default=["2,4", "4,8", "8,8", "8,16", "16,16"])
    parser.add_argument("--include-tucker", action="store_true", help="Evaluate slower Tucker/HOSVD baseline")
    parser.add_argument("--tucker-ranks", nargs="+", default=["2,8,8", "4,8,8", "4,12,12", "8,12,12"])
    parser.add_argument("--wavelet", default="haar", help="Wavelet family for Wavelet baseline")
    parser.add_argument("--wavelet-level", type=int, default=None, help="Wavelet decomposition level")
    parser.add_argument("--interpolation-order", type=int, default=1, help="Interpolation order: 0 nearest, 1 linear, 3 cubic")
    parser.add_argument(
        "--matrix-methods",
        nargs="+",
        default=["TruncatedSVD", "RandomProjection"],
        help="Additional matrix/manifold baselines to evaluate. Add UMAP explicitly for a slower nonlinear baseline.",
    )
    return parser.parse_args()


def normalize_methods(methods):
    if methods is None:
        return list(DEFAULT_METHODS)

    normalized = []
    for method in methods:
        key = method.strip().lower().replace("_", "-")
        canonical = METHOD_ALIASES.get(key)
        if canonical is None:
            choices = ", ".join(sorted(set(METHOD_ALIASES.values())))
            raise ValueError(f"Unknown method '{method}'. Available methods: {choices}")
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def parse_rank_pairs(values):
    return [tuple(int(part) for part in value.split(",")) for value in values]


def load_netcdf_data(file_paths, base_dir="", variable_name="co"):
    import netCDF4 as ncdf

    all_data = []
    for file_path in tqdm(file_paths, desc="Загрузка файлов"):
        full_path = f"{base_dir}/{file_path}" if base_dir else file_path
        with ncdf.Dataset(full_path) as dataset:
            all_data.append(np.array(dataset.variables[variable_name]))
    if not all_data:
        raise ValueError("Не удалось загрузить ни одного файла")
    combined = np.concatenate(all_data, axis=0)
    print(f"Объединенные данные: {combined.shape}")
    return combined


def normalize_per_sample_height(data, eps=1e-7):
    mins = data.min(axis=(2, 3), keepdims=True)
    maxs = data.max(axis=(2, 3), keepdims=True)
    ranges = maxs - mins
    normalized = np.where(ranges > 0, (data - mins) / (ranges + eps), 0.0)
    print(f"Нормализованные данные: min={normalized.min():.4f}, max={normalized.max():.4f}")
    return normalized


def split_train_eval(data, eval_split=0.2, seed=42):
    if not 0 < eval_split < 1:
        raise ValueError("--eval-split must be between 0 and 1")
    if len(data) < 2:
        raise ValueError("Need at least two samples for train/eval split")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))
    n_eval = max(1, int(round(len(data) * eval_split)))
    n_eval = min(n_eval, len(data) - 1)
    eval_indices = indices[:n_eval]
    train_indices = indices[n_eval:]
    train_data = data[np.sort(train_indices)]
    eval_data = data[np.sort(eval_indices)]
    print(f"Train/eval split: train={len(train_data)}, eval={len(eval_data)} ({eval_split:.0%})")
    return train_data, eval_data


def flatten_results(method, results):
    rows = []
    for setting, metrics in results.items():
        rows.append(
            {
                "method": method,
                "setting": str(setting),
                "compressed_size": metrics.get("compressed_size", ""),
                "compression_ratio": metrics["compression_ratio"],
                "mse": metrics["mse"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "relative_l2": metrics["relative_l2"],
                "ssim": metrics["ssim"],
            }
        )
    return rows


def autoencoder_label(model_name, use_attention=True):
    labels = {
        "PlainConv3DAutoencoder": "3D-CAE",
        "Conv3DAutoencoder": "3D-CAE-SAM3D" if use_attention else "3D-CAE",
        "SimpleConv3DAutoencoder": "Simple-3D-CAE",
    }
    return labels.get(model_name, model_name)


def train_autoencoder_components(train_data, eval_data, config, components, args):
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from models import get_model
        from train import WeightedHeightLoss, train_epoch, validate_epoch
    except ImportError as exc:
        raise ImportError(
            "Autoencoder evaluation requires PyTorch and project model modules. "
            "Install dependencies from requirements.txt or run with --skip-ae."
        ) from exc

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.ae_batch_size or config["training"]["batch_size"]
    num_epochs = args.ae_epochs or config["training"]["num_epochs"]
    learning_rate = args.ae_learning_rate or config["training"]["learning_rate"]
    patience = args.ae_patience or config["training"]["early_stopping"]["patience"]
    min_delta = config["training"]["early_stopping"].get("min_delta", 0.0)

    train_tensor = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)
    eval_tensor = torch.tensor(eval_data, dtype=torch.float32).unsqueeze(1)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(TensorDataset(eval_tensor), batch_size=batch_size, shuffle=False)

    def unwrap_loader(loader):
        for (batch,) in loader:
            yield batch

    criterion = WeightedHeightLoss(
        alpha=config["loss"]["alpha"],
        beta=config["loss"]["beta"],
        weight_first_5=config["loss"]["height_weight"],
    ).to(device)

    model_names = args.ae_models or config["model"].get("compare_models") or [config["model"]["name"]]
    results_by_model = {}
    input_shape = (1, *train_data.shape[1:])
    if tuple(config["model"].get("input_shape", input_shape)) != input_shape:
        print(f"AE input_shape overridden from data: {input_shape}")

    for model_name in model_names:
        use_attention = config["model"].get("use_attention", True)
        label = autoencoder_label(model_name, use_attention=use_attention)
        results_by_model[label] = {}

        for latent_dim in components:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

            print(
                f"\nAutoencoder training: model={model_name}, "
                f"latent_dim={latent_dim}, device={device}"
            )
            model = get_model(
                name=model_name,
                latent_dim=latent_dim,
                input_shape=input_shape,
                dropout_rate=config["model"]["dropout_rate"],
                use_attention=use_attention,
            ).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=config["model"].get("weight_decay", 0.0),
            )

            best_val_loss = float("inf")
            best_state = None
            patience_counter = 0

            for epoch in range(num_epochs):
                train_loss, train_mse, _ = train_epoch(
                    model, unwrap_loader(train_loader), criterion, optimizer, device
                )
                val_loss, val_mse, _ = validate_epoch(
                    model, unwrap_loader(eval_loader), criterion, device
                )

                improved = best_val_loss - val_loss > min_delta
                if improved:
                    best_val_loss = val_loss
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch == 0 or (epoch + 1) % config["logging"].get("log_every_n_epochs", 5) == 0:
                    print(
                        f"AE model={label} latent={latent_dim} epoch {epoch + 1:03d}/{num_epochs} | "
                        f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                        f"train_mse={train_mse:.5f} val_mse={val_mse:.5f}"
                    )

                if patience_counter >= patience:
                    print(f"AE model={label} latent={latent_dim}: early stopping at epoch {epoch + 1}")
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            model.eval()
            reconstructed_batches = []
            latent_batches = []
            with torch.no_grad():
                for (batch,) in eval_loader:
                    batch = batch.to(device)
                    reconstructed, latent = model(batch)
                    reconstructed_batches.append(reconstructed.cpu().numpy()[:, 0])
                    latent_batches.append(latent.cpu().numpy())

            reconstructed_eval = np.concatenate(reconstructed_batches, axis=0)
            latents = np.concatenate(latent_batches, axis=0)
            metrics = compute_reconstruction_metrics(eval_data, reconstructed_eval)
            results_by_model[label][latent_dim] = {
                **metrics,
                "compression_ratio": eval_data[0].size / latent_dim,
                "compressed_size": latent_dim,
                "best_val_loss": best_val_loss,
                "latent_shape": list(latents.shape),
                "model_name": model_name,
            }

            print(
                f"AE model={label} latent={latent_dim}: MSE={metrics['mse']:.6f}, "
                f"Relative L2={metrics['relative_l2']:.6f}, SSIM={metrics['ssim']:.6f}"
            )

    return results_by_model


def main():
    args = parse_args()
    import yaml

    selected_methods = normalize_methods(args.methods)
    if args.skip_ae and "AE" in selected_methods:
        selected_methods.remove("AE")

    matrix_methods = [method for method in MATRIX_METHODS if method in selected_methods]
    if args.methods is None:
        matrix_methods = list(args.matrix_methods)

    effective_methods = [
        method for method in selected_methods if method not in MATRIX_METHODS
    ] + matrix_methods
    print(f"Выбранные методы: {', '.join(effective_methods) if effective_methods else 'нет'}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data = load_netcdf_data(
        config["data"]["files"],
        base_dir=config["data"]["dataset_dir"],
        variable_name=config["data"].get("variable_name", "co"),
    )
    data = normalize_per_sample_height(data)

    if args.max_samples is not None and args.max_samples < len(data):
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(data), size=args.max_samples, replace=False)
        data = data[np.sort(indices)]
        print(f"Используется подвыборка: {len(data)} сэмплов")

    train_data, eval_data = split_train_eval(data, eval_split=args.eval_split, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    if "AE" in selected_methods:
        ae_results = train_autoencoder_components(train_data, eval_data, config, args.components, args)
        for model_label, model_results in ae_results.items():
            all_rows.extend(flatten_results(model_label, model_results))

    if "PCA" in selected_methods:
        pca_results = analyze_pca_components(
            data,
            n_components_list=args.components,
            train_data=train_data,
            eval_data=eval_data,
        )
        all_rows.extend(flatten_results("PCA", pca_results))

    if "DCT" in selected_methods:
        dct_results = analyze_dct_components(eval_data, n_components_list=args.components)
        all_rows.extend(flatten_results("DCT", dct_results))

    if "Wavelet" in selected_methods:
        wavelet_results = analyze_wavelet_components(
            eval_data,
            n_components_list=args.components,
            wavelet=args.wavelet,
            level=args.wavelet_level,
        )
        all_rows.extend(flatten_results("Wavelet", wavelet_results))

    if "Interpolation" in selected_methods:
        interpolation_results = analyze_interpolation_components(
            eval_data,
            n_components_list=args.components,
            order=args.interpolation_order,
        )
        all_rows.extend(flatten_results("Interpolation", interpolation_results))

    if matrix_methods:
        matrix_results = analyze_matrix_components(
            data,
            n_components_list=args.components,
            methods=tuple(matrix_methods),
            train_data=train_data,
            eval_data=eval_data,
            random_state=args.seed,
        )
        for method, results in matrix_results.items():
            all_rows.extend(flatten_results(method, results))

    if "TT-SVD" in selected_methods:
        tt_results = analyze_tt_components(eval_data, tt_ranks_list=parse_rank_pairs(args.tt_ranks))
        all_rows.extend(flatten_results("TT-SVD", tt_results))

    if "Tucker" in selected_methods or (args.methods is None and args.include_tucker):
        tucker_results = analyze_tucker_components(eval_data, ranks_list=parse_rank_pairs(args.tucker_ranks))
        all_rows.extend(flatten_results("Tucker/HOSVD", tucker_results))
    elif args.methods is None:
        print("Tucker/HOSVD пропущен. Добавьте --include-tucker для отдельного запуска.")

    if not all_rows:
        raise ValueError("No methods selected. Pass at least one method with --methods.")

    csv_path = output_dir / "compression_metrics.csv"
    json_path = output_dir / "compression_metrics.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2)

    print(f"✓ CSV сохранён: {csv_path}")
    print(f"✓ JSON сохранён: {json_path}")


if __name__ == "__main__":
    main()
