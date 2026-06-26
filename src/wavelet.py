"""
Wavelet compression baseline for 3D concentration fields.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from tqdm import tqdm


class WaveletCompressor:
    """
    Per-sample 3D wavelet compressor.

    The compressor applies a multilevel wavelet transform and keeps the
    largest-magnitude coefficients, similarly to the DCT baseline.
    """

    def __init__(self, n_components: int = 64, wavelet: str = "haar", level: Optional[int] = None):
        if n_components < 1:
            raise ValueError("n_components must be positive")
        self.n_components = int(n_components)
        self.wavelet = wavelet
        self.level = level
        self.original_shape: Optional[tuple[int, int, int]] = None
        self.coeff_slices = None
        self.coeff_shape = None

    def fit(self, data: np.ndarray, verbose: bool = True) -> "WaveletCompressor":
        self._validate_data(data)
        self.original_shape = tuple(int(v) for v in data.shape[1:])
        pywt = self._pywt()
        coeffs = pywt.wavedecn(np.zeros(self.original_shape), wavelet=self.wavelet, level=self.level, mode="periodization")
        coeff_array, self.coeff_slices = pywt.coeffs_to_array(coeffs)
        self.coeff_shape = coeff_array.shape
        if self.n_components > coeff_array.size:
            raise ValueError("n_components cannot exceed the number of wavelet coefficients")
        if verbose:
            print(
                f"Wavelet baseline: shape={self.original_shape}, coeff_shape={self.coeff_shape}, "
                f"wavelet={self.wavelet}, n_components={self.n_components}"
            )
        return self

    def transform(self, data: np.ndarray, verbose: bool = False) -> list[dict]:
        self._validate_data(data)
        if self.original_shape is None:
            self.fit(data, verbose=False)
        elif tuple(data.shape[1:]) != self.original_shape:
            raise ValueError(f"Ожидалась форма сэмпла {self.original_shape}, получено {data.shape[1:]}")

        pywt = self._pywt()
        compressed = []
        iterator = tqdm(data, desc="Wavelet сжатие", disable=not verbose)
        for sample in iterator:
            coeffs = pywt.wavedecn(sample, wavelet=self.wavelet, level=self.level, mode="periodization")
            coeff_array, _ = pywt.coeffs_to_array(coeffs)
            flat = coeff_array.ravel()
            if self.n_components == flat.size:
                indices = np.arange(flat.size)
            else:
                indices = np.argpartition(np.abs(flat), -self.n_components)[-self.n_components:]
            compressed.append(
                {
                    "indices": indices.astype(np.int64),
                    "values": flat[indices].astype(np.float64),
                }
            )
        return compressed

    def reconstruct(self, compressed: list[dict]) -> np.ndarray:
        if self.original_shape is None or self.coeff_slices is None or self.coeff_shape is None:
            raise ValueError("Сначала вызовите fit() или transform()")

        pywt = self._pywt()
        reconstructed = []
        flat_size = int(np.prod(self.coeff_shape))
        for item in compressed:
            flat = np.zeros(flat_size, dtype=np.float64)
            flat[item["indices"]] = item["values"]
            coeff_array = flat.reshape(self.coeff_shape)
            coeffs = pywt.array_to_coeffs(coeff_array, self.coeff_slices, output_format="wavedecn")
            sample = pywt.waverecn(coeffs, wavelet=self.wavelet, mode="periodization")
            slices = tuple(slice(0, size) for size in self.original_shape)
            reconstructed.append(sample[slices])
        return np.stack(reconstructed, axis=0)

    def fit_transform(self, data: np.ndarray, verbose: bool = True) -> list[dict]:
        self.fit(data, verbose=verbose)
        return self.transform(data, verbose=verbose)

    def compression_ratio(self) -> float:
        if self.original_shape is None:
            raise ValueError("Сначала вызовите fit() или transform()")
        return float(np.prod(self.original_shape) / self.n_components)

    @staticmethod
    def _pywt():
        try:
            import pywt
        except ImportError as exc:
            raise ImportError("Wavelet baseline requires PyWavelets. Install it with: pip install PyWavelets") from exc
        return pywt

    @staticmethod
    def _validate_data(data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy array")
        if data.ndim != 4:
            raise ValueError(f"Ожидается 4D массив (n_samples, depth, height, width), получено {data.ndim}D")


def analyze_wavelet_components(
    data: np.ndarray,
    n_components_list=None,
    wavelet: str = "haar",
    level: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    try:
        from .metrics import compute_reconstruction_metrics
    except ImportError:
        from metrics import compute_reconstruction_metrics

    if n_components_list is None:
        n_components_list = [16, 32, 64, 128, 256]

    results = {}
    for n_components in tqdm(n_components_list, desc="Анализ Wavelet"):
        compressor = WaveletCompressor(n_components=n_components, wavelet=wavelet, level=level)
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
            print(f"\nWavelet, n_components={n_components}:")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  Relative L2: {metrics['relative_l2']:.6f}")
            print(f"  R2: {metrics['r2']:.6f}")
            print(f"  Коэффициент сжатия: {compressor.compression_ratio():.1f}x")

    return results
