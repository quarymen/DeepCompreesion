#!/usr/bin/env python3
"""Run WRF-Chem random-frame and date-grouped compression experiments with physical metrics.

The script reuses the article experiment engine, but prepares WRF-Chem data from
configs/config.yaml and builds two protocols:
  * random: frames are shuffled and split into train/validation/test;
  * date: complete simulation dates are assigned to train/validation/test.

Outputs contain metrics_all_runs.csv with normalized and physical metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import yaml

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'configs/config.yaml').is_file() and (p / 'src/models.py').is_file()), None)
if ROOT is None:
    raise FileNotFoundError('Run this script from inside the DeepCompreesion repository.')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from scripts.cams_article_experiment import run_article_experiments  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='WRF physical metrics for random and date splits')
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--output-dir', default='outputs/wrf_physical_metrics')
    parser.add_argument('--split-mode', choices=['random', 'date', 'both'], default='both')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--latent-dims', type=int, nargs='+', default=[8, 16, 32, 64])
    parser.add_argument('--methods', nargs='+', default=[
        'PCA', 'DCT', 'Wavelet', 'Interpolation', 'TruncatedSVD',
        'RandomProjection', 'UMAP', 'TT-SVD',
        'ArticlePlainAutoencoder', 'ArticleSAMAutoencoder',
    ])
    parser.add_argument('--tt-ranks', nargs='+', default=['2,4', '4,8', '8,8', '8,16', '16,16'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--mae-weight', type=float, default=None)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0, ...')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--random-validation-fraction', type=float, default=None)
    parser.add_argument('--random-test-fraction', type=float, default=0.10)
    parser.add_argument('--date-validation-dates', type=int, default=1)
    parser.add_argument('--date-test-dates', type=int, default=2)
    parser.add_argument('--benchmark-frames', type=int, default=32)
    parser.add_argument('--benchmark-batch-size', type=int, default=4)
    parser.add_argument('--benchmark-repeats', type=int, default=3)
    parser.add_argument('--benchmark-warmup', type=int, default=1)
    parser.add_argument('--umap-inverse-method', default='knn', choices=['knn', 'umap'])
    parser.add_argument('--umap-inverse-neighbors', type=int, default=5)
    parser.add_argument('--umap-evaluation-batch-size', type=int, default=16)
    parser.add_argument('--wavelet', default='haar')
    parser.add_argument('--memory-sample-interval', type=float, default=0.05)
    parser.add_argument('--force', action='store_true', help='Recompute split-level metrics even if metrics_all_runs.csv exists.')
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return yaml.safe_load(path.read_text())


def file_date(path_text: str) -> str:
    match = re.search(r'Nsk_(\d{4}-\d{2}-\d{2})_', path_text)
    if not match:
        raise ValueError(f'Cannot extract simulation date from path: {path_text}')
    return match.group(1)


def timestamps_from_nc(ds: nc.Dataset, n: int) -> list[str | None]:
    if 'Times' in ds.variables:
        raw = np.asarray(ds.variables['Times'][:])
        values = []
        for row in raw:
            if raw.dtype.kind == 'S':
                text = b''.join(row).decode(errors='ignore')
            else:
                text = ''.join(row.astype(str)) if np.ndim(row) else str(row)
            values.append(text.strip('\x00 ').replace('_', 'T'))
        parsed = pd.to_datetime(values, errors='coerce')
        if len(parsed) == n and not parsed.isna().any():
            return [x.isoformat() for x in parsed]
    return [None] * n


def load_wrf_arrays(config: dict, output_dir: Path, eps: float = 1e-7):
    base = Path(config['data'].get('dataset_dir', ''))
    if not base.is_absolute():
        base = (ROOT / base).resolve()
    variable = config['data'].get('variable_name', 'co')
    expected = tuple(config['model']['input_shape'][1:])

    records = []
    samples = []
    raw_blocks = []
    x_blocks = []
    lo_blocks = []
    scale_blocks = []

    for file_id, item in enumerate(config['data']['files']):
        path = (base / item).resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Missing WRF file: {path}')
        group = file_date(item)
        with nc.Dataset(path) as ds:
            var = ds.variables[variable]
            if len(var.shape) != 4 or tuple(var.shape[1:]) != expected:
                raise ValueError(f'{item}: got shape {var.shape}, expected (time, {expected}).')
            raw = np.asarray(var[:], dtype=np.float32)
            if np.ma.isMaskedArray(raw):
                raw = raw.filled(np.nan)
            if not np.isfinite(raw).all():
                raise ValueError(f'{item}: NaN/Inf/masked values in {variable}.')
            lo = raw.min(axis=(2, 3)).astype(np.float64)
            hi = raw.max(axis=(2, 3)).astype(np.float64)
            scale = hi - lo
            x = np.where(scale[:, :, None, None] > 0,
                         (raw.astype(np.float64) - lo[:, :, None, None]) / (scale[:, :, None, None] + eps),
                         0.0).astype(np.float32)
            stamps = timestamps_from_nc(ds, raw.shape[0])
            records.append(dict(file_id=file_id, file=item, resolved_path=str(path), group=group,
                                n_frames=int(raw.shape[0]), units=str(getattr(var, 'units', '')),
                                shape=list(var.shape)))
            start = len(samples)
            for j, stamp in enumerate(stamps):
                samples.append(dict(index=start + j, file_id=file_id, group=group,
                                    local_index=j, timestamp=stamp, day=group))
            raw_blocks.append(raw)
            x_blocks.append(x)
            lo_blocks.append(lo)
            scale_blocks.append(scale)
            print(f'Loaded {group}: {raw.shape[0]} frames, {path.name}', flush=True)

    raw = np.concatenate(raw_blocks, axis=0)
    x = np.concatenate(x_blocks, axis=0)
    lo = np.concatenate(lo_blocks, axis=0)
    scale = np.concatenate(scale_blocks, axis=0)
    frames = pd.DataFrame(samples)
    pd.DataFrame(records).to_csv(output_dir / 'wrf_file_inventory.csv', index=False)
    frames.to_csv(output_dir / 'wrf_samples.csv', index=False)
    print(f'WRF arrays: x={x.shape}, raw={raw.shape}, frames={len(frames)}, dates={frames.group.nunique()}', flush=True)
    return x, raw, lo, scale, frames, records


def make_random_splits(n: int, seed: int, validation_fraction: float, test_fraction: float):
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError('Random split fractions must be positive and sum to less than 1.')
    rng = np.random.default_rng(seed)
    ids = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * validation_fraction)))
    test = np.sort(ids[:n_test])
    validation = np.sort(ids[n_test:n_test + n_val])
    train = np.sort(ids[n_test + n_val:])
    return {f'train_{len(train)}': train, 'validation': validation, 'test': test}, [len(train)]


def make_date_splits(frames: pd.DataFrame, seed: int, validation_dates: int, test_dates: int):
    groups = np.array(sorted(frames.group.unique()))
    if validation_dates < 1 or test_dates < 1 or validation_dates + test_dates >= len(groups):
        raise ValueError('Not enough dates for requested date split.')
    shuffled = np.random.default_rng(seed).permutation(groups)
    test_groups = set(shuffled[:test_dates])
    validation_groups = set(shuffled[test_dates:test_dates + validation_dates])
    train_groups = set(shuffled[test_dates + validation_dates:])

    def ids(selected):
        return np.flatnonzero(frames.group.isin(selected).to_numpy())

    train = ids(train_groups)
    validation = ids(validation_groups)
    test = ids(test_groups)
    return {f'train_{len(train)}': train, 'validation': validation, 'test': test}, [len(train)]


def write_split_summary(output: Path, frames: pd.DataFrame, splits: dict):
    rows = []
    for name, ids in splits.items():
        label = 'train' if name.startswith('train_') else name
        part = frames.iloc[np.asarray(ids, dtype=int)]
        rows.append(dict(split=label, n_frames=len(part), n_dates=part.group.nunique(),
                         dates=', '.join(sorted(part.group.unique()))))
    pd.DataFrame(rows).to_csv(output / 'split_summary.csv', index=False)
    np.savez(output / 'split_indices.npz', **splits)


def run_one(mode: str, x, raw, lo, scale, frames, records, base_config, args):
    seed = int(args.seed if args.seed is not None else base_config['data'].get('random_seed', 42))
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    output = output / mode
    output.mkdir(parents=True, exist_ok=True)

    completed_metrics = output / 'metrics_all_runs.csv'
    if completed_metrics.exists() and not args.force:
        print(f'Reusing completed WRF {mode} metrics: {completed_metrics}', flush=True)
        metrics = pd.read_csv(completed_metrics)
        save_physical_mse_tables(output, metrics.assign(protocol=mode), mode)
        return metrics

    if mode == 'random':
        val_fraction = args.random_validation_fraction
        if val_fraction is None:
            val_fraction = float(base_config['data'].get('val_split_ratio', 0.10))
        splits, counts = make_random_splits(len(frames), seed, val_fraction, args.random_test_fraction)
    elif mode == 'date':
        splits, counts = make_date_splits(frames, seed, args.date_validation_dates, args.date_test_dates)
    else:
        raise ValueError(mode)

    write_split_summary(output, frames, splits)
    article_config = dict(
        methods=args.methods,
        latent_dims=args.latent_dims,
        tt_ranks=[tuple(int(v) for v in item.split(',')) for item in args.tt_ranks],
        seeds=[seed],
        epochs=int(args.epochs if args.epochs is not None else base_config['training']['num_epochs']),
        batch_size=int(args.batch_size if args.batch_size is not None else base_config['training']['batch_size']),
        learning_rate=float(args.learning_rate if args.learning_rate is not None else base_config['training']['learning_rate']),
        weight_decay=float(args.weight_decay if args.weight_decay is not None else base_config['model'].get('weight_decay', 0.0)),
        dropout=float(args.dropout if args.dropout is not None else base_config['model'].get('dropout_rate', 0.0)),
        mae_weight=float(args.mae_weight if args.mae_weight is not None else base_config['loss'].get('beta', 0.0)),
        grad_clip_norm=1.0,
        device=args.device,
        cpu_threads=args.cpu_threads,
        evaluation_splits=['validation', 'test'],
        benchmark_frames=args.benchmark_frames,
        benchmark_batch_size=args.benchmark_batch_size,
        benchmark_repeats=args.benchmark_repeats,
        benchmark_warmup=args.benchmark_warmup,
        memory_sample_interval=args.memory_sample_interval,
        wavelet=args.wavelet,
        umap_inverse_method=args.umap_inverse_method,
        umap_inverse_neighbors=args.umap_inverse_neighbors,
        umap_evaluation_batch_size=args.umap_evaluation_batch_size,
    )
    protocol = dict(split_mode=mode, seed=seed, config_path=str(Path(args.config)),
                    article_config=article_config, records=records)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    print(f'Running WRF {mode} split -> {output}', flush=True)
    print(pd.read_csv(output / 'split_summary.csv').to_string(index=False), flush=True)
    metrics = run_article_experiments(x, raw, lo, scale, frames, splits, counts, article_config, output)
    metrics.to_csv(output / 'metrics_all_runs.csv', index=False)
    return metrics



def save_physical_mse_tables(output: Path, metrics: pd.DataFrame, prefix: str):
    """Save method x latent_dim physical-MSE tables for validation/test splits."""
    required = {'method', 'split', 'latent_dim', 'physical_mse'}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f'Cannot save physical_mse tables; missing columns: {sorted(missing)}')
    for split in sorted(metrics['split'].dropna().unique()):
        subset = metrics[(metrics['split'] == split) & metrics['latent_dim'].notna()].copy()
        if subset.empty:
            continue
        subset['latent_dim'] = subset['latent_dim'].astype(int)
        table = subset.pivot_table(
            index='method', columns='latent_dim', values='physical_mse', aggfunc='mean'
        ).sort_index()
        table.columns = [str(int(c)) for c in table.columns]
        table.to_csv(output / f'physical_mse_{prefix}_{split}.csv')

def main():
    args = parse_args()
    base_config = load_config(args.config)
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    modes = ['random', 'date'] if args.split_mode == 'both' else [args.split_mode]

    summaries = []
    pending = []
    for mode in modes:
        completed_metrics = out / mode / 'metrics_all_runs.csv'
        if completed_metrics.exists() and not args.force:
            print(f'Reusing completed WRF {mode} metrics: {completed_metrics}', flush=True)
            tagged = pd.read_csv(completed_metrics).assign(protocol=mode)
            tagged.to_csv(out / f'{mode}_metrics_all_runs.csv', index=False)
            save_physical_mse_tables(out, tagged, mode)
            save_physical_mse_tables(out / mode, tagged, mode)
            summaries.append(tagged)
        else:
            pending.append(mode)

    if pending:
        x, raw, lo, scale, frames, records = load_wrf_arrays(base_config, out)
        for mode in pending:
            metrics = run_one(mode, x, raw, lo, scale, frames, records, base_config, args)
            tagged = metrics.assign(protocol=mode)
            tagged.to_csv(out / f'{mode}_metrics_all_runs.csv', index=False)
            save_physical_mse_tables(out, tagged, mode)
            save_physical_mse_tables(out / mode, tagged, mode)
            summaries.append(tagged)
    else:
        print('All requested split-level metrics were found; NetCDF loading and model runs were skipped.', flush=True)

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(out / 'wrf_random_date_metrics_all_runs.csv', index=False)
    for protocol, part in combined.groupby('protocol'):
        save_physical_mse_tables(out, part, str(protocol))
    print('Saved combined metrics:', out / 'wrf_random_date_metrics_all_runs.csv', flush=True)
    print('Physical columns:', [c for c in combined.columns if c.startswith('physical_')], flush=True)


if __name__ == '__main__':
    main()
