"""
Additional nonlinear and matrix baselines for dimensionality reduction.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from tqdm import tqdm

try:
    from .metrics import compute_reconstruction_metrics
except ImportError:
    from metrics import compute_reconstruction_metrics


class TruncatedSVDCompressor:
    """
    Low-rank SVD compressor for vectorized 3D fields.

    Unlike PCA, this implementation does not subtract the mean before
    decomposition. It is a useful baseline for direct low-rank approximation.
    """

    def __init__(self, n_components: int = 64, random_state: int = 42):
        self.n_components = int(n_components)
        self.random_state = random_state
        self.model: Optional[TruncatedSVD] = None
        self.original_shape: Optional[tuple[int, int, int]] = None

    def fit(self, data: np.ndarray, verbose: bool = True) -> "TruncatedSVDCompressor":
        self._validate_data(data)
        self.original_shape = tuple(int(v) for v in data.shape[1:])
        matrix = data.reshape(data.shape[0], -1)
        self.model = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        self.model.fit(matrix)
        if verbose:
            print(f"✓ TruncatedSVD обучен: n_components={self.n_components}")
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        self._check_ready()
        return self.model.transform(data.reshape(data.shape[0], -1))

    def reconstruct(self, compressed: np.ndarray) -> np.ndarray:
        self._check_ready()
        reconstructed = self.model.inverse_transform(compressed)
        return reconstructed.reshape(compressed.shape[0], *self.original_shape)

    def fit_transform(self, data: np.ndarray, verbose: bool = True) -> np.ndarray:
        self.fit(data, verbose=verbose)
        return self.transform(data)

    def compression_ratio(self) -> float:
        self._check_ready()
        return float(np.prod(self.original_shape) / self.n_components)

    def _check_ready(self) -> None:
        if self.model is None or self.original_shape is None:
            raise ValueError("Сначала вызовите fit()")

    @staticmethod
    def _validate_data(data: np.ndarray) -> None:
        if data.ndim != 4:
            raise ValueError(f"Ожидается 4D массив, получено {data.ndim}D")


class RandomProjectionCompressor:
    """
    Gaussian random projection with linear least-squares reconstruction.

    This is a non-adaptive Johnson-Lindenstrauss-style baseline. Reconstruction
    uses the pseudo-inverse of the projection matrix.
    """

    def __init__(self, n_components: int = 64, random_state: int = 42):
        self.n_components = int(n_components)
        self.random_state = random_state
        self.model: Optional[GaussianRandomProjection] = None
        self.inverse_components: Optional[np.ndarray] = None
        self.original_shape: Optional[tuple[int, int, int]] = None

    def fit(self, data: np.ndarray, verbose: bool = True) -> "RandomProjectionCompressor":
        self._validate_data(data)
        self.original_shape = tuple(int(v) for v in data.shape[1:])
        matrix = data.reshape(data.shape[0], -1)
        self.model = GaussianRandomProjection(n_components=self.n_components, random_state=self.random_state)
        self.model.fit(matrix)
        self.inverse_components = np.linalg.pinv(self.model.components_.T)
        if verbose:
            print(f"✓ GaussianRandomProjection создан: n_components={self.n_components}")
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        self._check_ready()
        return self.model.transform(data.reshape(data.shape[0], -1))

    def reconstruct(self, compressed: np.ndarray) -> np.ndarray:
        self._check_ready()
        reconstructed = compressed @ self.inverse_components
        return reconstructed.reshape(compressed.shape[0], *self.original_shape)

    def fit_transform(self, data: np.ndarray, verbose: bool = True) -> np.ndarray:
        self.fit(data, verbose=verbose)
        return self.transform(data)

    def compression_ratio(self) -> float:
        self._check_ready()
        return float(np.prod(self.original_shape) / self.n_components)

    def _check_ready(self) -> None:
        if self.model is None or self.inverse_components is None or self.original_shape is None:
            raise ValueError("Сначала вызовите fit()")

    @staticmethod
    def _validate_data(data: np.ndarray) -> None:
        if data.ndim != 4:
            raise ValueError(f"Ожидается 4D массив, получено {data.ndim}D")


class UMAPCompressor:
    """
    UMAP manifold-learning compressor with approximate inverse reconstruction.

    UMAP is not primarily a lossy compression algorithm, but it is a useful
    nonlinear manifold baseline against autoencoders. Native
    ``inverse_transform`` can be extremely slow, so reconstruction uses a
    KNN regressor in the latent space by default.
    """

    def __init__(
        self,
        n_components: int = 64,
        random_state: int = 42,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        n_jobs: int = -1,
        inverse_method: str = "knn",
        inverse_neighbors: int = 5,
    ):
        self.n_components = int(n_components)
        self.random_state = None
        self.n_neighbors = int(n_neighbors)
        self.min_dist = float(min_dist)
        self.metric = metric
        self.n_jobs = int(n_jobs)
        self.inverse_method = inverse_method
        self.inverse_neighbors = int(inverse_neighbors)
        self.model = None
        self.decoder = None
        self.original_shape: Optional[tuple[int, int, int]] = None

    def fit(self, data: np.ndarray, verbose: bool = True) -> "UMAPCompressor":
        self._validate_data(data)
        self.original_shape = tuple(int(v) for v in data.shape[1:])
        matrix = data.reshape(data.shape[0], -1)
        try:
            from umap import UMAP
        except ImportError as exc:
            raise ImportError(
                "UMAP baseline requires the optional dependency 'umap-learn'. "
                "Install it with: pip install umap-learn"
            ) from exc

        n_neighbors = min(self.n_neighbors, max(2, data.shape[0] - 1))
        self.model = UMAP(
            n_components=self.n_components,
            n_neighbors=n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=None,
            n_jobs=self.n_jobs,
        )
        if verbose:
            print(
                f"UMAP fit: samples={matrix.shape[0]}, features={matrix.shape[1]}, "
                f"n_components={self.n_components}, n_neighbors={n_neighbors}, n_jobs={self.n_jobs}"
            )
        start_time = time.time()
        self.model.fit(matrix)
        embedding = self.model.embedding_
        if verbose:
            print(
                f"✓ UMAP fit завершен за {time.time() - start_time:.1f}s "
                f"(min_dist={self.min_dist}, metric={self.metric})"
            )

        if self.inverse_method == "knn":
            from sklearn.neighbors import KNeighborsRegressor

            n_decoder_neighbors = min(self.inverse_neighbors, data.shape[0])
            if verbose:
                print(
                    f"UMAP KNN decoder fit: latent_shape={embedding.shape}, "
                    f"n_neighbors={n_decoder_neighbors}"
                )
            start_time = time.time()
            self.decoder = KNeighborsRegressor(
                n_neighbors=n_decoder_neighbors,
                weights="distance",
                n_jobs=self.n_jobs,
            )
            self.decoder.fit(embedding, matrix)
            if verbose:
                print(f"✓ UMAP KNN decoder fit завершен за {time.time() - start_time:.1f}s")
        elif self.inverse_method != "umap":
            raise ValueError("inverse_method must be 'knn' or 'umap'")

        return self

    def transform(self, data: np.ndarray, verbose: bool = False) -> np.ndarray:
        self._check_ready()
        matrix = data.reshape(data.shape[0], -1)
        if verbose:
            print(f"UMAP transform: samples={matrix.shape[0]}")
        start_time = time.time()
        compressed = self.model.transform(matrix)
        if verbose:
            print(f"✓ UMAP transform завершен за {time.time() - start_time:.1f}s")
        return compressed

    def reconstruct(self, compressed: np.ndarray, verbose: bool = False) -> np.ndarray:
        self._check_ready()
        if self.inverse_method == "knn":
            if self.decoder is None:
                raise ValueError("KNN decoder is not fitted")
            if verbose:
                print(f"UMAP KNN reconstruct: samples={compressed.shape[0]}")
            start_time = time.time()
            reconstructed = self.decoder.predict(compressed)
            if verbose:
                print(f"✓ UMAP KNN reconstruct завершен за {time.time() - start_time:.1f}s")
        else:
            if verbose:
                print(f"UMAP inverse_transform: samples={compressed.shape[0]}")
            start_time = time.time()
            reconstructed = self.model.inverse_transform(compressed)
            if verbose:
                print(f"✓ UMAP inverse_transform завершен за {time.time() - start_time:.1f}s")
        return reconstructed.reshape(compressed.shape[0], *self.original_shape)

    def fit_transform(self, data: np.ndarray, verbose: bool = True) -> np.ndarray:
        self.fit(data, verbose=verbose)
        return self.transform(data, verbose=verbose)

    def compression_ratio(self) -> float:
        self._check_ready()
        return float(np.prod(self.original_shape) / self.n_components)

    def _check_ready(self) -> None:
        if self.model is None or self.original_shape is None:
            raise ValueError("Сначала вызовите fit()")

    @staticmethod
    def _validate_data(data: np.ndarray) -> None:
        if data.ndim != 4:
            raise ValueError(f"Ожидается 4D массив, получено {data.ndim}D")


def analyze_matrix_components(
    data: np.ndarray,
    n_components_list=None,
    methods=("TruncatedSVD", "RandomProjection"),
    train_data: Optional[np.ndarray] = None,
    eval_data: Optional[np.ndarray] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    if n_components_list is None:
        n_components_list = [16, 32, 64, 128, 256]
    if train_data is None:
        train_data = data
    if eval_data is None:
        eval_data = data

    classes = {
        "TruncatedSVD": TruncatedSVDCompressor,
        "RandomProjection": RandomProjectionCompressor,
        "UMAP": UMAPCompressor,
    }
    results = {}
    n_samples = train_data.shape[0]
    n_features = train_data[0].size

    for method in methods:
        method_results = {}
        cls = classes[method]
        for n_components in tqdm(n_components_list, desc=f"Анализ {method}"):
            if method == "TruncatedSVD":
                max_components = min(n_samples, n_features)
            else:
                max_components = n_features

            if n_components > max_components:
                if verbose:
                    print(
                        f"\n{method}: пропускаю n_components={n_components}, "
                        f"так как максимум для данных равен {max_components}"
                    )
                continue

            compressor = cls(n_components=n_components, random_state=random_state)
            if method == "UMAP":
                if verbose:
                    print(f"\nUMAP, n_components={n_components}:")
                compressor.fit(train_data, verbose=verbose)
                compressed = compressor.transform(eval_data, verbose=verbose)
                reconstructed = compressor.reconstruct(compressed, verbose=verbose)
            else:
                compressor.fit(train_data, verbose=False)
                compressed = compressor.transform(eval_data)
                reconstructed = compressor.reconstruct(compressed)
            metrics = compute_reconstruction_metrics(eval_data, reconstructed)
            method_results[n_components] = {
                **metrics,
                "compression_ratio": compressor.compression_ratio(),
                "compressed_size": n_components,
                "compressor": compressor,
            }

            if verbose:
                print(f"\n{method}, n_components={n_components}:")
                print(f"  MSE: {metrics['mse']:.6f}")
                print(f"  Relative L2: {metrics['relative_l2']:.6f}")
                print(f"  Sample Relative L2 p95: {metrics['sample_relative_l2_p95']:.6f}")
                print(f"  R2: {metrics['r2']:.6f}")

        results[method] = method_results

    return results
