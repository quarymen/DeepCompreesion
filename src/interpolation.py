"""
Interpolation compression baseline for 3D concentration fields.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


class InterpolationCompressor:
    """
    Compress a 3D field by resizing it to a coarse grid and reconstructing it
    with interpolation.
    """

    def __init__(self, n_components: int = 64, order: int = 1):
        if n_components < 1:
            raise ValueError("n_components must be positive")
        self.n_components = int(n_components)
        self.order = int(order)
        self.original_shape: Optional[tuple[int, int, int]] = None
        self.coarse_shape: Optional[tuple[int, int, int]] = None

    def fit(self, data: np.ndarray, verbose: bool = True) -> "InterpolationCompressor":
        self._validate_data(data)
        self.original_shape = tuple(int(v) for v in data.shape[1:])
        self.coarse_shape = self._choose_coarse_shape(self.original_shape, self.n_components)
        if verbose:
            print(
                f"Interpolation baseline: shape={self.original_shape}, coarse_shape={self.coarse_shape}, "
                f"compressed_size={np.prod(self.coarse_shape)}, order={self.order}"
            )
        return self

    def transform(self, data: np.ndarray, verbose: bool = False) -> np.ndarray:
        self._validate_data(data)
        if self.original_shape is None:
            self.fit(data, verbose=False)
        elif tuple(data.shape[1:]) != self.original_shape:
            raise ValueError(f"Ожидалась форма сэмпла {self.original_shape}, получено {data.shape[1:]}")

        factors = tuple(c / o for c, o in zip(self.coarse_shape, self.original_shape))
        compressed = []
        iterator = tqdm(data, desc="Interpolation сжатие", disable=not verbose)
        for sample in iterator:
            compressed.append(zoom(sample, zoom=factors, order=self.order))
        return np.stack(compressed, axis=0)

    def reconstruct(self, compressed: np.ndarray) -> np.ndarray:
        if self.original_shape is None or self.coarse_shape is None:
            raise ValueError("Сначала вызовите fit() или transform()")

        factors = tuple(o / c for o, c in zip(self.original_shape, self.coarse_shape))
        reconstructed = []
        for sample in compressed:
            restored = zoom(sample, zoom=factors, order=self.order)
            slices = tuple(slice(0, size) for size in self.original_shape)
            reconstructed.append(restored[slices])
        return np.stack(reconstructed, axis=0)

    def fit_transform(self, data: np.ndarray, verbose: bool = True) -> np.ndarray:
        self.fit(data, verbose=verbose)
        return self.transform(data, verbose=verbose)

    def compression_ratio(self) -> float:
        if self.original_shape is None or self.coarse_shape is None:
            raise ValueError("Сначала вызовите fit() или transform()")
        return float(np.prod(self.original_shape) / np.prod(self.coarse_shape))

    def compressed_size(self) -> int:
        if self.coarse_shape is None:
            raise ValueError("Сначала вызовите fit() или transform()")
        return int(np.prod(self.coarse_shape))

    @staticmethod
    def _choose_coarse_shape(original_shape: tuple[int, int, int], target_size: int) -> tuple[int, int, int]:
        original_size = int(np.prod(original_shape))
        if target_size >= original_size:
            return original_shape

        scale = (target_size / original_size) ** (1.0 / 3.0)
        coarse = [max(1, min(dim, int(round(dim * scale)))) for dim in original_shape]

        def product(values):
            return int(np.prod(values))

        while product(coarse) > target_size and any(value > 1 for value in coarse):
            axis = int(np.argmax(coarse))
            coarse[axis] -= 1

        while True:
            candidates = []
            for axis, dim in enumerate(original_shape):
                if coarse[axis] < dim:
                    candidate = coarse.copy()
                    candidate[axis] += 1
                    if product(candidate) <= target_size:
                        candidates.append(candidate)
            if not candidates:
                break
            coarse = max(candidates, key=product)

        return tuple(int(value) for value in coarse)

    @staticmethod
    def _validate_data(data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy array")
        if data.ndim != 4:
            raise ValueError(f"Ожидается 4D массив (n_samples, depth, height, width), получено {data.ndim}D")


def analyze_interpolation_components(data: np.ndarray, n_components_list=None, order: int = 1, verbose: bool = True) -> dict:
    try:
        from .metrics import compute_reconstruction_metrics
    except ImportError:
        from metrics import compute_reconstruction_metrics

    if n_components_list is None:
        n_components_list = [16, 32, 64, 128, 256]

    results = {}
    for n_components in tqdm(n_components_list, desc="Анализ Interpolation"):
        compressor = InterpolationCompressor(n_components=n_components, order=order)
        compressed = compressor.fit_transform(data, verbose=False)
        reconstructed = compressor.reconstruct(compressed)
        metrics = compute_reconstruction_metrics(data, reconstructed)
        actual_size = compressor.compressed_size()
        results[n_components] = {
            **metrics,
            "compression_ratio": compressor.compression_ratio(),
            "compressed_size": actual_size,
            "coarse_shape": compressor.coarse_shape,
            "compressor": compressor,
        }

        if verbose:
            print(f"\nInterpolation, target={n_components}, actual_size={actual_size}:")
            print(f"  Coarse shape: {compressor.coarse_shape}")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  Relative L2: {metrics['relative_l2']:.6f}")
            print(f"  R2: {metrics['r2']:.6f}")
            print(f"  Коэффициент сжатия: {compressor.compression_ratio():.1f}x")

    return results
