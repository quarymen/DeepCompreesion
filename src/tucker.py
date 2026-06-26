"""
Tucker/HOSVD compression baseline for 3D concentration fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


@dataclass
class TuckerSample:
    core: np.ndarray
    factors: Tuple[np.ndarray, np.ndarray, np.ndarray]

    @property
    def n_parameters(self) -> int:
        return int(self.core.size + sum(factor.size for factor in self.factors))


class TuckerCompressor:
    """
    Per-sample truncated HOSVD compressor for arrays shaped
    (n_samples, depth, height, width).
    """

    def __init__(self, ranks: Tuple[int, int, int] = (8, 16, 16)):
        self.ranks = tuple(int(rank) for rank in ranks)
        if len(self.ranks) != 3 or min(self.ranks) < 1:
            raise ValueError("ranks must be a triple of positive integers")
        self.original_shape: Optional[Tuple[int, int, int]] = None

    def fit(self, data: np.ndarray, verbose: bool = True) -> "TuckerCompressor":
        self._validate_data(data)
        self.original_shape = tuple(int(v) for v in data.shape[1:])
        if verbose:
            print(f"Tucker/HOSVD: shape={self.original_shape}, ranks={self.ranks}")
        return self

    def transform(self, data: np.ndarray, verbose: bool = False) -> List[TuckerSample]:
        self._validate_data(data)
        if self.original_shape is None:
            self.fit(data, verbose=False)
        elif tuple(data.shape[1:]) != self.original_shape:
            raise ValueError(f"Ожидалась форма сэмпла {self.original_shape}, получено {data.shape[1:]}")

        samples = []
        iterator = tqdm(data, desc="Tucker сжатие", disable=not verbose)
        for tensor in iterator:
            samples.append(self._hosvd(tensor.astype(np.float64, copy=False)))
        return samples

    def reconstruct(self, compressed: Sequence[TuckerSample]) -> np.ndarray:
        if not compressed:
            return np.empty((0, *(self.original_shape or (0, 0, 0))), dtype=np.float64)
        return np.stack([self._reconstruct_sample(sample) for sample in compressed], axis=0)

    def fit_transform(self, data: np.ndarray, verbose: bool = True) -> List[TuckerSample]:
        self.fit(data, verbose=verbose)
        return self.transform(data, verbose=verbose)

    def reconstruction_error(self, data: np.ndarray) -> tuple[float, float]:
        compressed = self.transform(data)
        reconstructed = self.reconstruct(compressed)
        return float(np.mean((data - reconstructed) ** 2)), float(np.mean(np.abs(data - reconstructed)))

    def compression_ratio(self, compressed: Optional[Sequence[TuckerSample]] = None) -> float:
        if self.original_shape is None:
            raise ValueError("Сначала вызовите fit() или transform()")

        original_size = int(np.prod(self.original_shape))
        if compressed is not None and len(compressed) > 0:
            compressed_size = float(np.mean([sample.n_parameters for sample in compressed]))
        else:
            n1, n2, n3 = self.original_shape
            r1 = min(self.ranks[0], n1)
            r2 = min(self.ranks[1], n2)
            r3 = min(self.ranks[2], n3)
            compressed_size = r1 * r2 * r3 + n1 * r1 + n2 * r2 + n3 * r3

        return float(original_size / compressed_size)

    def flattened_size(self, compressed: Optional[Sequence[TuckerSample]] = None) -> int:
        if compressed is not None and len(compressed) > 0:
            return int(round(float(np.mean([sample.n_parameters for sample in compressed]))))
        if self.original_shape is None:
            raise ValueError("Сначала вызовите fit() или transform()")
        n1, n2, n3 = self.original_shape
        r1 = min(self.ranks[0], n1)
        r2 = min(self.ranks[1], n2)
        r3 = min(self.ranks[2], n3)
        return int(r1 * r2 * r3 + n1 * r1 + n2 * r2 + n3 * r3)

    def _hosvd(self, tensor: np.ndarray) -> TuckerSample:
        factors = []
        for mode, rank in enumerate(self.ranks):
            unfolded = self._unfold(tensor, mode)
            u, _, _ = np.linalg.svd(unfolded, full_matrices=False)
            factors.append(u[:, : min(rank, u.shape[1])])

        core = tensor
        for mode, factor in enumerate(factors):
            core = self._mode_dot(core, factor.T, mode)

        return TuckerSample(core=core, factors=tuple(factors))

    @classmethod
    def _reconstruct_sample(cls, sample: TuckerSample) -> np.ndarray:
        tensor = sample.core
        for mode, factor in enumerate(sample.factors):
            tensor = cls._mode_dot(tensor, factor, mode)
        return tensor

    @staticmethod
    def _unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
        return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)

    @staticmethod
    def _mode_dot(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
        result = np.tensordot(matrix, tensor, axes=([1], [mode]))
        return np.moveaxis(result, 0, mode)

    @staticmethod
    def _validate_data(data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy array")
        if data.ndim != 4:
            raise ValueError(f"Ожидается 4D массив (n_samples, depth, height, width), получено {data.ndim}D")


def analyze_tucker_components(
    data: np.ndarray,
    ranks_list: List[Tuple[int, int, int]],
    verbose: bool = True,
) -> dict:
    try:
        from .metrics import compute_reconstruction_metrics
    except ImportError:
        from metrics import compute_reconstruction_metrics

    results = {}
    for ranks in tqdm(ranks_list, desc="Анализ Tucker"):
        compressor = TuckerCompressor(ranks=ranks)
        compressed = compressor.fit_transform(data, verbose=False)
        reconstructed = compressor.reconstruct(compressed)
        metrics = compute_reconstruction_metrics(data, reconstructed)
        key = tuple(ranks)
        results[key] = {
            **metrics,
            "compression_ratio": compressor.compression_ratio(compressed),
            "compressed_size": compressor.flattened_size(compressed),
            "compressor": compressor,
        }

        if verbose:
            print(f"\nTucker ranks={key}:")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  NRMSE: {metrics['nrmse']:.6f}")
            print(f"  R2: {metrics['r2']:.6f}")
            print(f"  Коэффициент сжатия: {compressor.compression_ratio(compressed):.1f}x")

    return results
