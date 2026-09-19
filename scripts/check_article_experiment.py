"""Small end-to-end test of all article methods and profiling (no CAMS download)."""
import json
from pathlib import Path
import tempfile
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cams_article_experiment import ARTICLE_METHODS, run_article_experiments
from src.article_models import ArticleSAMAutoencoder, ArticlePlainAutoencoder


def main():
    torch.set_num_threads(2)
    for cls in [ArticlePlainAutoencoder, ArticleSAMAutoencoder]:
        model = cls(latent_dim=8, input_shape=(1, 6, 96, 84), dropout_rate=0.1)
        model.eval()
        a = torch.rand(1, 1, 6, 96, 84)
        with torch.no_grad():
            y, z = model(a)
            assert y.shape == a.shape and z.shape == (1, 8)
            torch.testing.assert_close(y, model.decode(z))
        del model, a, y, z
    rng = np.random.default_rng(42)
    x = rng.random((40, 3, 8, 8), dtype=np.float32)
    raw = 10 + x*2
    lo, scale = np.full((40, 3), 10.), np.full((40, 3), 2.)
    frames = pd.DataFrame({'day': ['2020-01-01']*32+['2022-01-01']*4+['2023-01-01']*4})
    splits = dict(train_1=np.arange(32), validation=np.arange(32, 36), test=np.arange(36, 40))
    config = dict(
        methods=ARTICLE_METHODS, epochs=1, latent_dims=[2], tt_ranks=[[2, 2]], seeds=[42],
        batch_size=8, learning_rate=0.001, weight_decay=1e-6, dropout=0.1, mae_weight=0.2,
        wavelet='haar', umap_inverse_method='knn', umap_inverse_neighbors=5,
        umap_evaluation_batch_size=4, device='cpu', cpu_threads=2,
        benchmark_frames=4, benchmark_batch_size=2, benchmark_warmup=1,
        benchmark_repeats=2, memory_sample_interval=0.01, log_every=1,
    )
    with tempfile.TemporaryDirectory(prefix='article-check-') as tmp:
        output = Path(tmp)
        table = run_article_experiments(x, raw, lo, scale, frames, splits, [1], config, output)
        assert len(table) == 20
        assert set(table.split) == {'validation', 'test'}
        for field in ['mse', 'ssim', 'physical_rmse', 'fit_seconds', 'training_seconds',
                      'encode_ms_per_frame', 'decode_ms_per_frame', 'fit_ram_peak_mib']:
            assert np.isfinite(table[field]).all(), field
        assert (table.physical_mse - 4*table.mse).abs().max() < 1e-5
        reference = None
        for path in output.glob('*/validation_benchmark_ids.npy'):
            ids = np.load(path)
            if reference is None:
                reference = ids
            np.testing.assert_array_equal(ids, reference)
        again = run_article_experiments(x, raw, lo, scale, frames, splits, [1], config, output)
        assert len(again) == 20
        for path in output.glob('*/results.json'):
            assert len(json.loads(path.read_text())) == 2
        print('PASS: ten methods, both evaluation splits, timings, memory, payload, reuse.')


if __name__ == '__main__':
    main()
