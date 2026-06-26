"""
Discrete Cosine Transform compression baseline.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.fft import dctn, idctn
from tqdm import tqdm


class DCTCompressor:
    """
    Fixed-basis compressor that keeps the largest-magnitude 3D DCT
    coefficients for every sample.
    """

    def __init__(self, n_components: int = 64):
        if n_components < 1:
            raise ValueError("n_components must be positive")
        self.n_components = int(n_components)
        self.original_shape: Optional[tuple[int, int, int]] = None

    def fit(self, data: np.ndarray, verbose: bool = True) -> "DCTCompressor":
        self._validate_data(data)
        self.original_shape = tuple(int(v) for v in data.shape[1:])
        if self.n_components > int(np.prod(self.original_shape)):
            raise ValueError("n_components cannot exceed the number of tensor elements")
        if verbose:
            print(f"DCT baseline: shape={self.original_shape}, n_components={self.n_components}")
        return self

    def transform(self, data: np.ndarray, verbose: bool = False) -> List[dict]:
        self._validate_data(data)
        if self.original_shape is None:
            self.fit(data, verbose=False)
        elif tuple(data.shape[1:]) != self.original_shape:
            raise ValueError(f"Ожидалась форма сэмпла {self.original_shape}, получено {data.shape[1:]}")

        compressed = []
        iterator = tqdm(data, desc="DCT сжатие", disable=not verbose)
        for sample in iterator:
            coeffs = dctn(sample, norm="ortho")
            flat = coeffs.ravel()
            if self.n_components == flat.size:
                indices = np.arange(flat.size)
            else:
                indices = np.argpartition(np.abs(flat), -self.n_components)[-self.n_components:]
            values = flat[indices]
            compressed.append({"indices": indices.astype(np.int64), "values": values.astype(np.float64)})
        return compressed

    def reconstruct(self, compressed: List[dict]) -> np.ndarray:
        if self.original_shape is None:
            raise ValueError("Сначала вызовите fit() или transform()")

        reconstructed = []
        flat_size = int(np.prod(self.original_shape))
        for item in compressed:
            flat = np.zeros(flat_size, dtype=np.float64)
            flat[item["indices"]] = item["values"]
            coeffs = flat.reshape(self.original_shape)
            reconstructed.append(idctn(coeffs, norm="ortho"))
        return np.stack(reconstructed, axis=0)

    def fit_transform(self, data: np.ndarray, verbose: bool = True) -> List[dict]:
        self.fit(data, verbose=verbose)
        return self.transform(data, verbose=verbose)

    def reconstruction_error(self, data: np.ndarray) -> tuple[float, float]:
        compressed = self.transform(data)
        reconstructed = self.reconstruct(compressed)
        return float(np.mean((data - reconstructed) ** 2)), float(np.mean(np.abs(data - reconstructed)))

    def compression_ratio(self) -> float:
        if self.original_shape is None:
            raise ValueError("Сначала вызовите fit() или transform()")
        return float(np.prod(self.original_shape) / self.n_components)

    @staticmethod
    def _validate_data(data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy array")
        if data.ndim != 4:
            raise ValueError(f"Ожидается 4D массив (n_samples, depth, height, width), получено {data.ndim}D")


def analyze_dct_components(data: np.ndarray, n_components_list=None, verbose: bool = True) -> dict:
    try:
        from .metrics import compute_reconstruction_metrics
    except ImportError:
        from metrics import compute_reconstruction_metrics

    if n_components_list is None:
        n_components_list = [16, 32, 64, 128, 256]

    results = {}
    for n_components in tqdm(n_components_list, desc="Анализ DCT"):
        compressor = DCTCompressor(n_components=n_components)
        compressed = compressor.fit_transform(data, verbose=False)
        reconstructed = compressor.reconstruct(compressed)
        metrics = compute_reconstruction_metrics(data, reconstructed)
        results[n_components] = {
            **metrics,
            "compression_ratio": compressor.compression_ratio(),
            "compressed_size": n_components,
            "compressor": compressor,
        }

        if verbose:
            print(f"\nn_components={n_components}:")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  NRMSE: {metrics['nrmse']:.6f}")
            print(f"  R2: {metrics['r2']:.6f}")
            print(f"  Коэффициент сжатия: {compressor.compression_ratio():.1f}x")

    return results
