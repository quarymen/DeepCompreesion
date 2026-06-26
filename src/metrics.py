"""
Reconstruction metrics for compression experiments.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def _global_ssim(target: np.ndarray, reconstructed: np.ndarray, data_range: float, eps: float) -> float:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    target_mean = float(np.mean(target))
    reconstructed_mean = float(np.mean(reconstructed))
    target_var = float(np.var(target))
    reconstructed_var = float(np.var(reconstructed))
    covariance = float(np.mean((target - target_mean) * (reconstructed - reconstructed_mean)))

    numerator = (2.0 * target_mean * reconstructed_mean + c1) * (2.0 * covariance + c2)
    denominator = (target_mean**2 + reconstructed_mean**2 + c1) * (
        target_var + reconstructed_var + c2
    )
    return float(numerator / (denominator + eps))


def _ssim_2d(target: np.ndarray, reconstructed: np.ndarray, data_range: float, eps: float) -> float:
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        return _global_ssim(target, reconstructed, data_range, eps)

    min_side = min(target.shape[-2:])
    if min_side < 3:
        return _global_ssim(target, reconstructed, data_range, eps)

    window_size = min(7, min_side)
    if window_size % 2 == 0:
        window_size -= 1

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    target_mean = uniform_filter(target, size=window_size)
    reconstructed_mean = uniform_filter(reconstructed, size=window_size)

    target_mean_sq = target_mean**2
    reconstructed_mean_sq = reconstructed_mean**2
    mean_product = target_mean * reconstructed_mean

    target_var = uniform_filter(target * target, size=window_size) - target_mean_sq
    reconstructed_var = uniform_filter(reconstructed * reconstructed, size=window_size) - reconstructed_mean_sq
    covariance = uniform_filter(target * reconstructed, size=window_size) - mean_product

    ssim_map = ((2.0 * mean_product + c1) * (2.0 * covariance + c2)) / (
        (target_mean_sq + reconstructed_mean_sq + c1) * (target_var + reconstructed_var + c2)
        + eps
    )
    return float(np.mean(ssim_map))


def compute_ssim(
    target: np.ndarray,
    reconstructed: np.ndarray,
    data_range: float | None = None,
    eps: float = 1e-12,
) -> float:
    """
    Compute mean SSIM over all 2D spatial slices.

    For CO fields shaped as (samples, depth, height, width), SSIM is computed
    for every (height, width) slice and then averaged across samples/depths.
    """
    if target.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {target.shape} vs {reconstructed.shape}")

    target = np.asarray(target, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)

    if data_range is None:
        data_range = float(np.max(target) - np.min(target))
    data_range = max(float(data_range), eps)

    if target.ndim < 2:
        return _global_ssim(target, reconstructed, data_range, eps)

    flat_target = target.reshape((-1, *target.shape[-2:]))
    flat_reconstructed = reconstructed.reshape((-1, *reconstructed.shape[-2:]))
    values = [
        _ssim_2d(target_slice, reconstructed_slice, data_range, eps)
        for target_slice, reconstructed_slice in zip(flat_target, flat_reconstructed)
    ]
    return float(np.mean(values))


def compute_reconstruction_metrics(
    target: np.ndarray,
    reconstructed: np.ndarray,
    data_range: float | None = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Compute scalar reconstruction metrics for arrays with identical shape.
    """
    if target.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {target.shape} vs {reconstructed.shape}")

    target = np.asarray(target, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    error = reconstructed - target

    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))

    if data_range is None:
        data_range = float(np.max(target) - np.min(target))
    nrmse = float(rmse / (data_range + eps))

    target_norm = float(np.linalg.norm(target.ravel()))
    relative_l2 = float(np.linalg.norm(error.ravel()) / (target_norm + eps))
    relative_l1 = float(np.sum(np.abs(error)) / (np.sum(np.abs(target)) + eps))

    sample_axes = tuple(range(1, target.ndim))
    sample_error_norms = np.sqrt(np.sum(error**2, axis=sample_axes))
    sample_target_norms = np.sqrt(np.sum(target**2, axis=sample_axes))
    sample_relative_l2 = sample_error_norms / (sample_target_norms + eps)

    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + eps))

    psnr = float(20.0 * np.log10((data_range + eps) / (rmse + eps)))
    max_abs_error = float(np.max(np.abs(error)))
    ssim = compute_ssim(target, reconstructed, data_range=data_range, eps=eps)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "ssim": ssim,
        "nrmse": nrmse,
        "relative_l2": relative_l2,
        "relative_l1": relative_l1,
        "sample_relative_l2_mean": float(np.mean(sample_relative_l2)),
        "sample_relative_l2_median": float(np.median(sample_relative_l2)),
        "sample_relative_l2_p95": float(np.percentile(sample_relative_l2, 95)),
        "r2": r2,
        "psnr": psnr,
        "max_abs_error": max_abs_error,
    }


def compute_height_metrics(target: np.ndarray, reconstructed: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute MSE and MAE for every vertical level.

    Expected shape is (n_samples, depth, height, width).
    """
    if target.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {target.shape} vs {reconstructed.shape}")
    if target.ndim != 4:
        raise ValueError(f"Expected 4D arrays, got {target.ndim}D")

    error = np.asarray(reconstructed, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "mse_by_height": np.mean(error**2, axis=(0, 2, 3)),
        "mae_by_height": np.mean(np.abs(error), axis=(0, 2, 3)),
    }
