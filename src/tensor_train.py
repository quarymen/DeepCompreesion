"""
Tensor Train compression for 3D concentration fields.

This module implements TT-SVD for every sample independently. The previous
version averaged TT cores across samples, which is not a valid inverse mapping:
TT cores are sample-specific compressed factors, not trainable basis vectors.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


@dataclass
class TTSample:
    """Compressed representation of one 3D tensor in TT format."""

    cores: List[np.ndarray]

    @property
    def ranks(self) -> Tuple[int, int]:
        return int(self.cores[0].shape[2]), int(self.cores[1].shape[2])

    @property
    def n_parameters(self) -> int:
        return int(sum(core.size for core in self.cores))


class TTCompressor:
    """
    Per-sample Tensor Train compressor for arrays shaped
    (n_samples, depth, height, width).

    For a 3D tensor X with shape (n1, n2, n3), TT-SVD approximates it as
    G1(1, n1, r1), G2(r1, n2, r2), G3(r2, n3, 1). The ranks are capped by
    ``tt_ranks`` and by the unfolding matrix dimensions.
    """

    def __init__(self, tt_ranks: Tuple[int, int] = (16, 16)):
        self.tt_ranks = tuple(int(rank) for rank in tt_ranks)
        if len(self.tt_ranks) != 2 or min(self.tt_ranks) < 1:
            raise ValueError("tt_ranks must be a pair of positive integers")

        self.original_shape: Optional[Tuple[int, int, int]] = None

    def fit(self, data: np.ndarray, verbose: bool = True) -> "TTCompressor":
        """
        TT-SVD is non-parametric: fitting only records and validates shape.
        """
        self._validate_data(data)
        self.original_shape = tuple(int(v) for v in data.shape[1:])

        if verbose:
            print(f"Форма данных: {data.shape}")
            print(f"TT-ранги: {self.tt_ranks}")
            print("TT-SVD не обучает глобальную модель; сжатие выполняется для каждого сэмпла.")

        return self

    def transform(self, data: np.ndarray, verbose: bool = False) -> List[TTSample]:
        """
        Compress each sample into its own TT cores.
        """
        self._validate_data(data)
        if self.original_shape is None:
            self.fit(data, verbose=False)
        elif tuple(data.shape[1:]) != self.original_shape:
            raise ValueError(f"Ожидалась форма сэмпла {self.original_shape}, получено {data.shape[1:]}")

        samples: List[TTSample] = []
        iterator = tqdm(data, desc="TT-SVD сжатие", disable=not verbose)
        for tensor in iterator:
            samples.append(TTSample(self._tt_svd(tensor.astype(np.float64, copy=False))))

        return samples

    def reconstruct(self, compressed: Sequence[TTSample]) -> np.ndarray:
        """
        Reconstruct data from a sequence of TT samples.
        """
        if not compressed:
            return np.empty((0, *(self.original_shape or (0, 0, 0))), dtype=np.float64)

        reconstructed = [self._tt_reconstruct(sample.cores) for sample in compressed]
        return np.stack(reconstructed, axis=0)

    def fit_transform(self, data: np.ndarray, verbose: bool = True) -> List[TTSample]:
        self.fit(data, verbose=verbose)
        return self.transform(data, verbose=verbose)

    def reconstruction_error(self, data: np.ndarray) -> Tuple[float, float]:
        compressed = self.transform(data)
        reconstructed = self.reconstruct(compressed)
        return float(np.mean((data - reconstructed) ** 2)), float(np.mean(np.abs(data - reconstructed)))

    def compression_ratio(self, compressed: Optional[Sequence[TTSample]] = None) -> float:
        """
        Average compression ratio for one sample.

        If ``compressed`` is omitted, the ratio is computed from maximum ranks.
        Actual ranks can be lower near small dimensions, so passing compressed
        samples gives the exact average ratio.
        """
        if self.original_shape is None:
            raise ValueError("Сначала вызовите fit() или transform()")

        original_size = int(np.prod(self.original_shape))

        if compressed is not None and len(compressed) > 0:
            compressed_size = float(np.mean([sample.n_parameters for sample in compressed]))
        else:
            n1, n2, n3 = self.original_shape
            r1 = min(self.tt_ranks[0], n1, n2 * n3)
            r2 = min(self.tt_ranks[1], r1 * n2, n3)
            compressed_size = n1 * r1 + r1 * n2 * r2 + r2 * n3

        return float(original_size / compressed_size)

    def flattened_size(self, compressed: Optional[Sequence[TTSample]] = None) -> int:
        if compressed is not None and len(compressed) > 0:
            return int(round(float(np.mean([sample.n_parameters for sample in compressed]))))

        if self.original_shape is None:
            raise ValueError("Сначала вызовите fit() или transform()")

        n1, n2, n3 = self.original_shape
        r1 = min(self.tt_ranks[0], n1, n2 * n3)
        r2 = min(self.tt_ranks[1], r1 * n2, n3)
        return int(n1 * r1 + r1 * n2 * r2 + r2 * n3)

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump({"tt_ranks": self.tt_ranks, "original_shape": self.original_shape}, f)
        print(f"✓ TT конфигурация сохранена в {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "TTCompressor":
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        compressor = cls(tt_ranks=data["tt_ranks"])
        compressor.original_shape = data["original_shape"]
        print(f"✓ TT конфигурация загружена из {filepath}")
        return compressor

    def _tt_svd(self, tensor: np.ndarray) -> List[np.ndarray]:
        if tensor.ndim != 3:
            raise ValueError(f"Ожидался 3D тензор, получено {tensor.ndim}D")

        n1, n2, n3 = tensor.shape
        max_r1, max_r2 = self.tt_ranks

        r0 = 1
        unfolding = tensor.reshape(r0 * n1, n2 * n3)
        u1, s1, vt1 = self._truncated_svd(unfolding, max_r1)
        r1 = s1.size
        core1 = u1.reshape(r0, n1, r1)

        unfolding = (np.diag(s1) @ vt1).reshape(r1 * n2, n3)
        u2, s2, vt2 = self._truncated_svd(unfolding, max_r2)
        r2 = s2.size
        core2 = u2.reshape(r1, n2, r2)
        core3 = (np.diag(s2) @ vt2).reshape(r2, n3, 1)

        return [core1, core2, core3]

    @staticmethod
    def _truncated_svd(matrix: np.ndarray, max_rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        rank = min(int(max_rank), s.size)
        return u[:, :rank], s[:rank], vt[:rank, :]

    @staticmethod
    def _tt_reconstruct(cores: Sequence[np.ndarray]) -> np.ndarray:
        if len(cores) != 3:
            raise ValueError("Для 3D TT-разложения требуется ровно 3 ядра")

        result = np.tensordot(cores[0], cores[1], axes=([2], [0]))
        result = np.tensordot(result, cores[2], axes=([3], [0]))
        return np.squeeze(result, axis=(0, -1))

    @staticmethod
    def _validate_data(data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy array")
        if data.ndim != 4:
            raise ValueError(f"Ожидается 4D массив (n_samples, depth, height, width), получено {data.ndim}D")


def analyze_tt_components(
    data: np.ndarray,
    tt_ranks_list: List[Tuple[int, int]],
    train_data: Optional[np.ndarray] = None,
    val_data: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> dict:
    """
    Analyze TT-SVD reconstruction quality for several rank pairs.

    ``train_data`` is accepted for API compatibility. TT-SVD does not learn a
    global model, so evaluation is performed on ``val_data`` when provided.
    """
    try:
        from .metrics import compute_reconstruction_metrics
    except ImportError:
        from metrics import compute_reconstruction_metrics

    eval_data = val_data if val_data is not None else data
    results = {}

    for tt_ranks in tqdm(tt_ranks_list, desc="Анализ TT"):
        compressor = TTCompressor(tt_ranks=tt_ranks).fit(eval_data, verbose=False)
        compressed = compressor.transform(eval_data, verbose=False)
        reconstructed = compressor.reconstruct(compressed)
        metrics = compute_reconstruction_metrics(eval_data, reconstructed)

        key = tuple(tt_ranks)
        results[key] = {
            **metrics,
            "compression_ratio": compressor.compression_ratio(compressed),
            "compressed_size": compressor.flattened_size(compressed),
            "compressor": compressor,
        }

        if verbose:
            print(f"\nTT-ранги={key}:")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  NRMSE: {metrics['nrmse']:.6f}")
            print(f"  R2: {metrics['r2']:.6f}")
            print(f"  Коэффициент сжатия: {compressor.compression_ratio(compressed):.1f}x")

    return results


if __name__ == "__main__":
    print("Тест Tensor Train модуля...")

    np.random.seed(42)
    test_data = np.random.randn(4, 8, 12, 10)

    compressor = TTCompressor(tt_ranks=(4, 5))
    compressed = compressor.fit_transform(test_data)
    reconstructed = compressor.reconstruct(compressed)

    print(f"\nОригинал: {test_data.shape}")
    print(f"Сжатых сэмплов: {len(compressed)}")
    print(f"Восстановление: {reconstructed.shape}")

    mse, mae = compressor.reconstruction_error(test_data)
    print(f"\nMSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Коэффициент сжатия: {compressor.compression_ratio(compressed):.1f}x")
