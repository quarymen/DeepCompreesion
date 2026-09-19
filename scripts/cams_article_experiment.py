"""Article methods on the fixed CAMS temporal split; no fit on validation/test."""
from pathlib import Path
import gc
import hashlib
import json
import time
import threading
import platform
import psutil

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from scripts.cams_experiment import evaluate, fit_ae, seed_all
from src.dct import DCTCompressor
from src.wavelet import WaveletCompressor
from src.interpolation import InterpolationCompressor
from src.tensor_train import TTCompressor
from src.matrix_methods import (
    TruncatedSVDCompressor, RandomProjectionCompressor, UMAPCompressor,
)

ARTICLE_METHODS = [
    'PCA', 'DCT', 'Wavelet', 'Interpolation', 'TruncatedSVD',
    'RandomProjection', 'UMAP', 'TT-SVD',
    'ArticlePlainAutoencoder', 'ArticleSAMAutoencoder',
]


def experiment_directory(root, config, files):
    root = Path(root).resolve()
    code = [Path(__file__).resolve(), root/'scripts/cams_experiment.py']
    code.extend(sorted((root/'src').glob('*.py')))
    import importlib.metadata
    packages = ['torch', 'numpy', 'pandas', 'scipy', 'scikit-learn',
                'netCDF4', 'PyWavelets', 'umap-learn', 'psutil']
    protocol = {
        'config': config,
        'files': [dict(path=str(p), bytes=p.stat().st_size,
                       mtime_ns=p.stat().st_mtime_ns) for p in files],
        'code': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in code},
        'versions': {name: importlib.metadata.version(name) for name in packages},
        'architecture_source': '58f40c4:src/models.py',
        'scope': 'Article architectures and methods on CAMS; configurable epochs without early stopping',
        'hardware': dict(platform=platform.platform(), cpu=platform.processor(),
                         cpu_count=psutil.cpu_count(), ram_bytes=psutil.virtual_memory().total,
                         gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        'benchmark': 'CPU arrays -> encode -> CPU code -> decode -> CPU reconstruction; '
                     'CUDA synchronized; excludes metrics/disk IO; RAM sampled process RSS, '
                     'not isolated model memory; GPU memory is PyTorch allocated/reserved.',
    }
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    output = root/'outputs'/f'cams_article_{key}'
    output.mkdir(parents=True, exist_ok=True)
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    return output


class ResourceMeter:
    """Phase-local wall time, sampled process RSS, and PyTorch CUDA peaks."""

    def __init__(self, cuda=False, interval=0.02):
        self.cuda = cuda
        self.interval = interval
        self.process = psutil.Process()
        self.stop = threading.Event()

    def sample(self):
        self.peak = max(self.peak, self.process.memory_info().rss)

    def poll(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.start_rss = self.peak = self.process.memory_info().rss
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        try:
            if self.cuda:
                torch.cuda.synchronize()
            self.seconds = time.perf_counter() - self.start
            self.sample()
            self.stats = dict(
                seconds=self.seconds,
                ram_start_mib=self.start_rss/2**20,
                ram_peak_mib=self.peak/2**20,
                ram_peak_delta_mib=max(0, self.peak-self.start_rss)/2**20,
                gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if self.cuda else None,
                gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if self.cuda else None,
            )
        finally:
            self.stop.set()
            self.thread.join()


def benchmark(encode, decode, x, ids, config, cuda, path, split):
    count = min(config['benchmark_frames'], len(ids))
    selected = ids[np.linspace(0, len(ids)-1, count).round().astype(int)]
    data = np.array(x[selected], dtype='float32')
    np.save(path/f'{split}_benchmark_ids.npy', selected)
    batch = config['benchmark_batch_size']
    sync = torch.cuda.synchronize if cuda else lambda: None
    for _ in range(config['benchmark_warmup']):
        decoded = decode(encode(data[:batch]))
        del decoded
    records = []
    with ResourceMeter(cuda, config['memory_sample_interval']) as meter:
        for repeat in range(config['benchmark_repeats']):
            enc_seconds = dec_seconds = 0.
            for start in range(0, count, batch):
                a = data[start:start+batch]
                sync()
                t = time.perf_counter()
                code = encode(a)
                sync()
                enc_seconds += time.perf_counter() - t
                t = time.perf_counter()
                decoded = decode(code)
                sync()
                dec_seconds += time.perf_counter() - t
                if decoded.shape != a.shape or not np.isfinite(decoded).all():
                    raise ValueError('Invalid benchmark reconstruction')
                del decoded, code
            records.append(dict(repeat=repeat, encode_seconds=enc_seconds,
                                decode_seconds=dec_seconds, frames=count))
    pd.DataFrame(records).to_csv(path/f'{split}_timing_repeats.csv', index=False)
    enc = np.array([r['encode_seconds'] for r in records])/count
    dec = np.array([r['decode_seconds'] for r in records])/count
    result = dict(
        benchmark_frames=count, benchmark_batch_size=batch,
        benchmark_repeats=len(records),
        encode_ms_per_frame=float(np.median(enc)*1000),
        decode_ms_per_frame=float(np.median(dec)*1000),
        inference_ms_per_frame=float(np.median(enc+dec)*1000),
        inference_frames_per_second=float(1/np.median(enc+dec)),
        inference_ms_per_frame_min=float(np.min(enc+dec)*1000),
        inference_ms_per_frame_max=float(np.max(enc+dec)*1000),
    )
    result.update({f'inference_{k}': v for k, v in meter.stats.items()})
    return result


def encoded_size(code):
    """Numerical payload only; shared basis, shapes and file headers excluded."""
    if isinstance(code, np.ndarray):
        return code.size, code.nbytes
    if hasattr(code, 'cores'):
        return encoded_size(code.cores)
    items = code.values() if isinstance(code, dict) else code
    sizes = [encoded_size(item) for item in items]
    return sum(n for n, _ in sizes), sum(b for _, b in sizes)


def make_baseline(name, dim, seed, ranks, config):
    factories = {
        'DCT': lambda: DCTCompressor(dim),
        'Wavelet': lambda: WaveletCompressor(dim, wavelet=config['wavelet']),
        'Interpolation': lambda: InterpolationCompressor(dim, order=1),
        'TruncatedSVD': lambda: TruncatedSVDCompressor(dim, random_state=seed),
        'RandomProjection': lambda: RandomProjectionCompressor(dim, random_state=seed),
        'UMAP': lambda: UMAPCompressor(
            dim, random_state=seed, n_jobs=1,
            inverse_method=config['umap_inverse_method'],
            inverse_neighbors=config['umap_inverse_neighbors']),
        'TT-SVD': lambda: TTCompressor(tuple(ranks)),
    }
    return factories[name]()


def run_article_experiments(x, raw, lo, scale, frames, splits, counts, config, output):
    from threadpoolctl import threadpool_limits
    previous = torch.get_num_threads()
    torch.set_num_threads(config['cpu_threads'])
    try:
        with threadpool_limits(limits=config['cpu_threads']):
            return _run(x, raw, lo, scale, frames, splits, counts, config, output)
    finally:
        torch.set_num_threads(previous)


def _run(x, raw, lo, scale, frames, splits, counts, config, output):
    methods = config['methods']
    if config.get('evaluation_splits', ['validation', 'test']) != ['validation', 'test']:
        raise ValueError('Article evaluation is restricted to validation and test.')
    if (len(counts) != 1 or not methods or set(methods) - set(ARTICLE_METHODS)
            or len(set(methods)) != len(methods)):
        raise ValueError('Use one training pool and unique names from ARTICLE_METHODS.')
    for key in ['epochs', 'benchmark_frames', 'benchmark_batch_size', 'benchmark_repeats']:
        if config[key] < 1:
            raise ValueError(f'{key} must be positive')
    if config['memory_sample_interval'] <= 0 or config['benchmark_warmup'] < 0:
        raise ValueError('Invalid sampling interval or warmup count')
    n = counts[0]
    train_ids = splits[f'train_{n}']
    train = np.asarray(x[train_ids], dtype='float32')
    val = np.asarray(x[splits['validation']], dtype='float32')
    for split in ['validation', 'test']:
        if not len(splits[split]) or np.intersect1d(train_ids, splits[split]).size:
            raise ValueError('Empty evaluation split or train/evaluation overlap')
    if np.intersect1d(splits['validation'], splits['test']).size:
        raise ValueError('Validation/test overlap')
    if max(config['latent_dims']) >= min(len(train)-1, int(np.prod(x.shape[1:]))):
        raise ValueError('Too few training frames/features for requested dimensions (including UMAP).')
    jobs = []
    for name in methods:
        settings = config['tt_ranks'] if name == 'TT-SVD' else config['latent_dims']
        for setting in settings:
            for seed in config['seeds']:
                jobs.append((name, setting, seed))
    requested_device = config.get('device', 'auto')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if requested_device == 'auto' else requested_device)
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('Profiling supports cpu or cuda')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is not available in this PyTorch environment.')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
    rows = []
    print(f'Neural device: {device}; baselines: cpu; runs: {len(jobs)}', flush=True)
    for number, (name, setting, seed) in enumerate(jobs, 1):
        ranks = setting if name == 'TT-SVD' else None
        dim = None if ranks else int(setting)
        suffix = 'r' + '-'.join(map(str, ranks)) if ranks else f'd{dim}'
        path = output/f'{name}_{suffix}_days{n}_seed{seed}'
        path.mkdir(exist_ok=True)
        completed = path/'results.json'
        if completed.exists():
            rows.extend(json.loads(completed.read_text()))
            print(f'[{number}/{len(jobs)}] Reusing {path.name}', flush=True)
            pd.DataFrame(rows).to_csv(output/'metrics_all_runs.csv', index=False)
            continue
        print(f'[{number}/{len(jobs)}] Starting {path.name}', flush=True)
        seed_all(seed)
        best_epoch, params = None, 0
        neural = name.startswith('Article')
        cuda = neural and device.type == 'cuda'
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        with ResourceMeter(cuda, config['memory_sample_interval']) as fit_meter:
            if neural:
                model, best_epoch = fit_ae(name, dim, train, val, config, path, device)
                params = sum(p.numel() for p in model.parameters())
            elif name == 'PCA':
                model = PCA(n_components=dim, svd_solver='randomized', random_state=seed)
                model.fit(train.reshape(len(train), -1))
            else:
                model = make_baseline(name, dim, seed, ranks, config)
                shape_only = name in {'DCT', 'Wavelet', 'Interpolation', 'TT-SVD'}
                model.fit(train[:1] if shape_only else train, verbose=True)
        if neural:
            def encode(a):
                with torch.inference_mode():
                    return model.encode(torch.from_numpy(np.array(a, dtype='float32'))[:, None]
                                        .to(device)).cpu().numpy()
            def decode(code):
                with torch.inference_mode():
                    return model.decode(torch.from_numpy(np.array(code, dtype='float32'))
                                        .to(device))[:, 0].cpu().numpy()
            history = pd.read_csv(path/'history.csv')
            training_seconds = float(history.train_seconds.sum())
            validation_seconds = float(history.validation_seconds.sum())
        elif name == 'PCA':
            def encode(a):
                return model.transform(a.reshape(len(a), -1))
            def decode(code):
                return model.inverse_transform(code).reshape(len(code), *x.shape[1:])
            training_seconds = fit_meter.seconds
            validation_seconds = 0.
        else:
            encode, decode = model.transform, model.reconstruct
            training_seconds = (0. if name in {'DCT', 'Wavelet', 'Interpolation', 'TT-SVD',
                                                'RandomProjection'} else fit_meter.seconds)
            validation_seconds = 0.
        payload_values, payload_bytes = encoded_size(encode(val[:1]))
        def predict(a):
            return decode(encode(a))
        fit_stats = {f'fit_{k}': v for k, v in fit_meter.stats.items()}
        fit_stats.update(training_seconds=training_seconds,
                         training_validation_seconds=validation_seconds)
        (path/'fit_profile.json').write_text(json.dumps(fit_stats, indent=2))
        print(f'  Fit/setup: {fit_meter.seconds:.2f}s; '
              f'RAM peak: {fit_meter.stats["ram_peak_mib"]:.1f} MiB', flush=True)
        run_rows = []
        for split in ['validation', 'test']:
            ids = splits[split]
            timing = benchmark(encode, decode, x, ids, config, cuda, path, split)
            print(f'  {split}: encode {timing["encode_ms_per_frame"]:.3f}, '
                  f'decode {timing["decode_ms_per_frame"]:.3f} ms/frame', flush=True)
            cache = None
            metric_predict = predict
            if name == 'UMAP':
                cache = np.lib.format.open_memmap(
                    path/f'{split}_reconstruction.npy', mode='w+', dtype='float32',
                    shape=(len(ids), *x.shape[1:]))
                batch = config['umap_evaluation_batch_size']
                for begin in range(0, len(ids), batch):
                    end = min(begin+batch, len(ids))
                    cache[begin:end] = predict(np.asarray(x[ids[begin:end]]))
                    print(f'UMAP {split} reconstruction {end}/{len(ids)}', flush=True)
                cache.flush()
                position = 0
                def metric_predict(a):
                    nonlocal position
                    block = np.asarray(cache[position:position+len(a)])
                    position += len(a)
                    return block
            result = evaluate(metric_predict, x, ids, raw, lo, scale, frames, path, split, path.name)
            result.update(
                method=name, latent_dim=dim, tt_ranks=ranks, train_days=n,
                train_frames=len(train), seed=seed, split=split, evaluation_frames=len(ids),
                best_epoch=best_epoch, epochs=config['epochs'] if neural else 0,
                parameters=params, payload_bytes=int(payload_bytes),
                payload_values=int(payload_values), scaling_bytes_per_frame=x.shape[1]*8,
                device=str(device) if neural else 'cpu', cpu_threads=config['cpu_threads'],
                umap_inverse_method=config['umap_inverse_method'] if name == 'UMAP' else None,
                **fit_stats, **timing,
            )
            run_rows.append(result)
            del cache
            if name == 'UMAP':
                # This file is a temporary bridge to the common metric routine.
                (path/f'{split}_reconstruction.npy').unlink()
        completed.write_text(json.dumps(run_rows, indent=2))
        rows.extend(run_rows)
        pd.DataFrame(rows).to_csv(output/'metrics_all_runs.csv', index=False)
        del model, predict, metric_predict, encode, decode
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)
