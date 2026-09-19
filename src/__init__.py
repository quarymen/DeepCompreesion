"""
CO Autoencoder Project
Пакет для обучения 3D автоэнкодеров на данных концентрации CO
"""

from .models import (
    Conv3DAutoencoder,
    SAM3DAutoencoderV2,
    SAM3DAutoencoderV3,
    ResidualSAM3DAutoencoder,
    PlainConv3DAutoencoder,
    SimpleConv3DAutoencoder,
    get_model,
)
from .data import ConcentrationFieldDataset, load_data, normalize_data, create_dataloaders
from .train import Trainer, WeightedHeightLoss, load_model
from .pca import PCACompressor, analyze_pca_components
from .tensor_train import TTCompressor, analyze_tt_components
from .dct import DCTCompressor, analyze_dct_components
from .wavelet import WaveletCompressor, analyze_wavelet_components
from .interpolation import InterpolationCompressor, analyze_interpolation_components
from .tucker import TuckerCompressor, analyze_tucker_components
from .matrix_methods import (
    TruncatedSVDCompressor,
    RandomProjectionCompressor,
    UMAPCompressor,
    analyze_matrix_components,
)
from .metrics import compute_reconstruction_metrics, compute_height_metrics, compute_ssim

__version__ = '1.0.0'
__author__ = 'CO Autoencoder Team'

__all__ = [
    'Conv3DAutoencoder',
    'SAM3DAutoencoderV2',
    'SAM3DAutoencoderV3',
    'ResidualSAM3DAutoencoder',
    'PlainConv3DAutoencoder',
    'SimpleConv3DAutoencoder',
    'get_model',
    'ConcentrationFieldDataset',
    'load_data',
    'normalize_data',
    'create_dataloaders',
    'Trainer',
    'WeightedHeightLoss',
    'load_model',
    'PCACompressor',
    'analyze_pca_components',
    'TTCompressor',
    'analyze_tt_components',
    'DCTCompressor',
    'analyze_dct_components',
    'WaveletCompressor',
    'analyze_wavelet_components',
    'InterpolationCompressor',
    'analyze_interpolation_components',
    'TuckerCompressor',
    'analyze_tucker_components',
    'TruncatedSVDCompressor',
    'RandomProjectionCompressor',
    'UMAPCompressor',
    'analyze_matrix_components',
    'compute_reconstruction_metrics',
    'compute_height_metrics',
    'compute_ssim'
]
