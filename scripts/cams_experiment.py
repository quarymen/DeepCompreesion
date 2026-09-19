"""CAMS archive inspection and day-grouped learning curves for existing models."""
from pathlib import Path
import gc
import hashlib
import json
import random
import shutil
import time
import zipfile
import re

import netCDF4 as nc
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

from src.models import get_model
from src.dct import DCTCompressor
from src.metrics import compute_ssim


def unpack(folder, selected_levels=None):
    """Stream archives to disk; atomically mark successful extraction."""
    folder = Path(folder)
    selected = ({str(int(level)) for level in selected_levels}
                if selected_levels is not None else None)
    files = list(folder.glob('*.nc')) + list(folder.glob('*.nc4'))
    for archive in sorted(folder.glob('*.zip')):
        manifest = archive.with_suffix('.json')
        if selected is not None and manifest.exists():
            metadata = json.loads(manifest.read_text())
            archive_levels = set(metadata.get('request', {}).get('level', []))
            if archive_levels and archive_levels.isdisjoint(selected):
                print(f'Skipping archive outside selected levels: {archive.name}', flush=True)
                continue
        dest = folder / 'unpacked' / archive.stem
        signature = f'{archive.stat().st_size}:{archive.stat().st_mtime_ns}'
        marker = dest / '.complete'
        if not marker.exists() or marker.read_text() != signature:
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as z:
                for item in z.infolist():
                    target = (dest / item.filename).resolve()
                    if not target.is_relative_to(dest.resolve()):
                        raise ValueError(f'Unsafe archive path: {item.filename}')
                    if item.is_dir():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    print('Extracting:', item.filename, f'({item.file_size / 2**30:.2f} GiB)', flush=True)
                    partial = target.with_name(target.name + '.part')
                    with z.open(item) as source, partial.open('wb') as out:
                        shutil.copyfileobj(source, out, length=2**20)
                    partial.replace(target)
            marker.write_text(signature)
        files.extend(dest.rglob('*.nc'))
        files.extend(dest.rglob('*.nc4'))
    files = sorted(set(p.resolve() for p in files))
    if not files:
        raise ValueError(f'No .zip, .nc or .nc4 files in {folder}')
    return files


def coordinate(ds, kind):
    aliases = {'time': ['time', 'valid_time'], 'level': ['level', 'height', 'lev', 'altitude'],
               'lat': ['latitude', 'lat'], 'lon': ['longitude', 'lon']}
    for name in aliases[kind]:
        if name in ds.variables:
            return name
    raise ValueError(f'Cannot identify {kind}: {list(ds.variables)}')


def infer_single_level(path, ds, variable):
    """Read a height stored in metadata/filename when the file has no level axis."""
    patterns = [
        r'(?:^|[._-])L(?:EVEL)?[_-]?(\d{1,4})(?:m)?(?:[._-]|$)',
        r'\b(?:height|level|at)\D{0,20}(\d{1,4})\s*m(?:etre)?s?\b',
    ]
    allowed = {0., 50., 100., 250., 500., 750., 1000., 2000., 3000., 5000.}
    # The ADS filename is the most specific source. Some generic metadata can
    # still contain the word "surface" even for an elevated requested level.
    for text in [path.name, ' | '.join(str(getattr(ds, name)) for name in ds.ncattrs()),
                 ' | '.join(str(getattr(variable, name)) for name in variable.ncattrs())]:
        matches = {float(value) for pattern in patterns
                   for value in re.findall(pattern, text, flags=re.I)}.intersection(allowed)
        if len(matches) == 1:
            return matches.pop()
        if not matches and re.search(r'\bsurface\b', text, flags=re.I):
            return 0.0
    raise ValueError(
        f'{path.name}: CO has no level dimension and its height could not be identified. '
        f'Filename/metadata must contain Surface or L0/L50/.../L5000. Global attributes: '
        f'{dict((name, getattr(ds, name)) for name in ds.ncattrs())}'
    )


def canonical_unit(value, kind):
    """Canonicalise common CAMS spellings without converting numerical data."""
    text = str(value).strip().lower().replace('μ', 'u').replace('µ', 'u')
    text = re.sub(r'[\s{}^*/_]', '', text)
    if kind == 'co' and text in {'ugm-3', 'ugm3-1', 'ugm3'}:
        return 'ug m-3'
    if kind == 'level' and text in {'m', 'meter', 'meters', 'metre', 'metres',
                                     'mabovesurface', 'metersabovesurface',
                                     'metresabovesurface'}:
        return 'm above surface'
    return text


def same_grid(left, right):
    # CAMS files may encode the same nominal 0.1-degree grid as float32 or
    # float64. Their accumulated representation error is a few microdegrees;
    # 1e-4 degrees remains far below one grid cell and still rejects the
    # historical 0.05-degree grid shift.
    return left.shape == right.shape and np.allclose(left, right, rtol=0, atol=1e-4)


def inspect_files(files, variable=None, selected_levels=None):
    records, rows = [], []
    reference = None
    reference_path = None
    selected = ({float(level) for level in selected_levels}
                if selected_levels is not None else None)
    for path in files:
        with nc.Dataset(path) as ds:
            candidates = [n for n, v in ds.variables.items() if v.ndim >= 3 and
                          (n.lower() in ['co', 'co_conc', 'carbon_monoxide'] or
                           'carbon monoxide' in str(getattr(v, 'long_name', '')).lower())]
            name = variable or (candidates[0] if len(candidates) == 1 else None)
            if name not in ds.variables:
                raise ValueError(f'{path.name}: set VARIABLE explicitly; variables={list(ds.variables)}')
            v = ds[name]
            coords = {k: coordinate(ds, k) for k in ['time', 'lat', 'lon']}
            try:
                coords['level'] = coordinate(ds, 'level')
            except ValueError:
                coords['level'] = None
            cv = {k: ds[n] for k, n in coords.items() if n is not None}
            dims = {k: x.dimensions[0] for k, x in cv.items() if x.ndim == 1}
            expected = [dims[k] for k in ['time', 'lat', 'lon']]
            if 'level' in dims:
                expected.append(dims['level'])
            if set(expected) != set(v.dimensions):
                raise ValueError(f'Unsupported CO dimensions {v.dimensions}, coordinates {dims}')
            if coords['level'] is None:
                levels = np.array([infer_single_level(path, ds, v)])
                level_units = 'm above surface'
            else:
                levels = np.asarray(cv['level'][:]).reshape(-1).astype(float)
                level_units = getattr(cv['level'], 'units', 'UNKNOWN')
            level_indices = [i for i, level in enumerate(levels)
                             if selected is None or float(level) in selected]
            if not level_indices:
                print(f'Skipping file outside selected levels: {path.name}', flush=True)
                continue
            levels = levels[level_indices]
            lat, lon = (np.asarray(cv[k][:]) for k in ['lat', 'lon'])
            units_original = getattr(v, 'units', 'UNKNOWN')
            units = canonical_unit(units_original, 'co')
            level_units = canonical_unit(level_units, 'level')
            if reference is None:
                reference = (lat, lon, units, level_units)
                reference_path = path
            differences = []
            if not same_grid(lat, reference[0]):
                differences.append(f'latitude shape/range={lat.shape}/{float(lat.min())}..{float(lat.max())}')
            if not same_grid(lon, reference[1]):
                differences.append(f'longitude shape/range={lon.shape}/{float(lon.min())}..{float(lon.max())}')
            if units != reference[2]:
                differences.append(f'CO units={units_original!r} (canonical {units!r})')
            if level_units != reference[3]:
                differences.append(f'level units={level_units!r}')
            if differences:
                raise ValueError(
                    f'{path.name} differs from the first selected file {reference_path.name}: '
                    + '; '.join(differences)
                    + f'. Reference grid: latitude {reference[0].shape}/'
                      f'{float(reference[0].min())}..{float(reference[0].max())}, longitude '
                      f'{reference[1].shape}/{float(reference[1].min())}..{float(reference[1].max())}, '
                      f'units={reference[2]!r}, level_units={reference[3]!r}. '
                      'Keep only one CAMS product/grid in the data directory.'
                )
            tv = cv['time']
            dates = nc.num2date(tv[:], tv.units, calendar=getattr(tv, 'calendar', 'standard'))
            dates = [d.isoformat() for d in dates]
            rows.append(dict(file=path.name, variable=name, shape=str(v.shape),
                             dimensions=str(v.dimensions), frames=len(dates),
                             levels=levels.tolist(), units=units_original, first=dates[0], last=dates[-1],
                             disk_GiB=path.stat().st_size / 2**30))
            records.append(dict(path=path, variable=name, dims=dims, axes=v.dimensions,
                                dates=dates, levels=levels, level_indices=level_indices))
    if not records:
        raise ValueError(f'No files match selected_levels={selected_levels}')
    inventory = pd.DataFrame(rows)
    return inventory, records, reference


def prepare(records, reference, config, output):
    lat, lon, units, level_units = reference
    h, w = config['crop_shape']
    center_lat, center_lon = config['center_lat_lon']
    if not (min(lat) <= center_lat <= max(lat) and min(lon) <= center_lon <= max(lon)):
        raise ValueError('Crop center lies outside the dataset domain.')
    if h > len(lat) or w > len(lon):
        raise ValueError('Crop exceeds the native grid.')
    y0 = int(np.clip(np.abs(lat-center_lat).argmin()-h//2, 0, len(lat)-h))
    x0 = int(np.clip(np.abs(lon-center_lon).argmin()-w//2, 0, len(lon)-w))
    stamps = sorted({d for r in records for d in r['dates']})
    levels = sorted({float(z) for r in records for z in r['levels']})
    tid, zid = {t:i for i,t in enumerate(stamps)}, {z:i for i,z in enumerate(levels)}
    shape = (len(stamps), len(levels), h, w)
    seen = np.zeros(shape[:2], bool)
    sources = np.full(shape[:2], '', dtype=object)
    duplicate_files = set()
    raw = np.lib.format.open_memmap(output/'co_physical.npy', mode='w+', dtype='float32', shape=shape)
    for record in records:
        with nc.Dataset(record['path']) as ds:
            v = ds[record['variable']]
            dims = record['dims']
            remaining = [a for a in record['axes'] if a != dims['time']]
            desired = ([dims['level']] if 'level' in dims else []) + [dims['lat'], dims['lon']]
            order = [remaining.index(a) for a in desired]
            for local_t, stamp in enumerate(record['dates']):
                selection = {dims['time']: local_t, dims['lat']: slice(y0,y0+h), dims['lon']: slice(x0,x0+w)}
                block = np.ma.asarray(v[tuple(selection.get(a, slice(None)) for a in record['axes'])])
                block = np.asarray(block.filled(np.nan), dtype='float32').transpose(order)
                if 'level' not in dims:
                    block = block[None]
                else:
                    block = block[record['level_indices']]
                i = tid[stamp]
                zz = [zid[float(z)] for z in record['levels']]
                if not np.isfinite(block).all():
                    raise ValueError(f'Missing/nonfinite CO at {stamp}; inspect input before training.')
                for local_z, global_z in enumerate(zz):
                    if seen[i, global_z]:
                        previous = sources[i, global_z]
                        if np.array_equal(np.asarray(raw[i, global_z]), block[local_z]):
                            duplicate_files.add((previous, record['path'].name))
                            continue
                        raise ValueError(
                            f'Conflicting duplicate at time={stamp}, level={levels[global_z]}: '
                            f'{previous} and {record["path"].name}. Move one product/archive '
                            'out of the configured data directory.'
                        )
                    raw[i, global_z] = block[local_z]
                    seen[i, global_z] = True
                    sources[i, global_z] = record['path'].name
        print('Prepared:', record['path'].name, flush=True)
    for first, duplicate in sorted(duplicate_files):
        print(f'Ignored identical duplicate: {duplicate} (same data as {first})', flush=True)
    if not seen.all():
        raise ValueError('Some timestamps lack heights. Download matching height/time coverage.')
    raw.flush()
    lo = np.asarray(raw.min(axis=(2,3)))
    span = np.asarray(raw.max(axis=(2,3))) - lo
    scale = np.where(span > 0, span, 1).astype('float32')
    normalized = np.lib.format.open_memmap(output/'co_normalized.npy', mode='w+', dtype='float32', shape=shape)
    for i in range(len(raw)):
        normalized[i] = (raw[i]-lo[i,:,None,None])/scale[i,:,None,None]
    normalized.flush()
    np.savez(output/'coordinates_scaling.npz', time=np.array(stamps), levels=levels,
             latitude=lat[y0:y0+h], longitude=lon[x0:x0+w], minima=lo, scales=scale)
    frame_table = pd.DataFrame({'frame':range(len(stamps)), 'time':pd.to_datetime(stamps)})
    frame_table['day'] = frame_table.time.dt.strftime('%Y-%m-%d')
    frame_table.to_csv(output/'frames.csv', index=False)
    summary = dict(shape=list(shape), units=units, level_units=level_units, levels=levels,
                   first=stamps[0], last=stamps[-1], unique_days=frame_table.day.nunique(),
                   latitude=[float(lat[y0]),float(lat[y0+h-1])], longitude=[float(lon[x0]),float(lon[x0+w-1])],
                   missing_hour_intervals=int((frame_table.time.diff().dropna()!=pd.Timedelta(hours=1)).sum()),
                   float32_GiB=raw.nbytes/2**30)
    (output/'data_summary.json').write_text(json.dumps(summary, indent=2))
    return raw, normalized, lo, scale, frame_table, summary


def split_days(frames, config, output):
    days = sorted(frames.day.unique())
    years = config.get('split_years')
    if years:
        train_years = years['train'] if isinstance(years['train'], list) else [years['train']]
        train_years = [int(year) for year in train_years]
        validation_year = int(years['validation'])
        test_year = int(years['test'])
        if len(set(train_years + [validation_year, test_year])) != len(train_years) + 2:
            raise ValueError('Training, validation and test years must be different.')
        per_month = int(config.get('evaluation_days_per_month', 7))
        train_days = [d for d in days if pd.Timestamp(d).year in train_years]
        def evaluation_days(year):
            candidates = [d for d in days if pd.Timestamp(d).year == year]
            selected = []
            for month in sorted({pd.Timestamp(d).month for d in candidates}):
                group = [d for d in candidates if pd.Timestamp(d).month == month]
                if len(group) < per_month:
                    raise ValueError(f'Only {len(group)} days available for {year}-{month:02d}.')
                positions = np.linspace(0, len(group)-1, per_month).round().astype(int)
                selected.extend(group[i] for i in positions)
            return selected
        val_days = evaluation_days(validation_year)
        test_days = evaluation_days(test_year)
    else:
        ntest, nval = max(2, round(len(days)*.16)), max(2, round(len(days)*.16))
        gap = config['gap_days']
        test_start = pd.Timestamp(days[-ntest])
        before_test = [d for d in days if pd.Timestamp(d) < test_start-pd.Timedelta(days=gap)]
        val_days = before_test[-nval:]
        if len(val_days) != nval:
            raise ValueError('Not enough days for validation.')
        val_start = pd.Timestamp(val_days[0])
        train_days = [d for d in before_test if pd.Timestamp(d) < val_start-pd.Timedelta(days=gap)]
        test_days = days[-ntest:]
    if len(train_days) < 2:
        raise ValueError('Need more days for train/validation/test with gaps.')
    counts = sorted(set([n for n in config['train_day_counts'] if n <= len(train_days)] + [len(train_days)]))
    # Nested, reproducible subsets, interleaved across months when a full year is used.
    rng = np.random.default_rng(config['subset_seed'])
    buckets = {}
    for day in train_days:
        buckets.setdefault(pd.Timestamp(day).month, []).append(day)
    for month in buckets:
        buckets[month] = rng.permutation(buckets[month]).tolist()
    order = []
    while any(buckets.values()):
        for month in sorted(buckets):
            if buckets[month]:
                order.append(buckets[month].pop())
    ids = lambda ds: np.flatnonzero(frames.day.isin(ds).to_numpy())
    splits = {'validation': ids(val_days), 'test': ids(test_days)}
    splits.update({f'train_{n}':ids(order[:n]) for n in counts})
    for n in counts:
        assert not set(splits[f'train_{n}']) & set(splits['validation'])
        assert not set(splits[f'train_{n}']) & set(splits['test'])
    np.savez(output/'split_indices.npz', **splits)
    labels = frames.copy()
    labels['partition'] = 'gap'
    for key in ['validation','test',f'train_{max(counts)}']:
        labels.loc[splits[key], 'partition'] = 'train_pool' if key.startswith('train') else key
    labels.to_csv(output/'partitions.csv', index=False)
    (output/'training_days.json').write_text(json.dumps({str(n):order[:n] for n in counts}, indent=2))
    return splits, counts


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fit_ae(name, dim, train, val, config, path, device):
    model = get_model(name, latent_dim=dim, input_shape=(1,*train.shape[1:]), dropout_rate=config['dropout']).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f'  Model architecture ({name}):', flush=True)
    print(model, flush=True)
    print(
        f'  Input shape: {(1, *train.shape[1:])}; latent_dim={dim}; '
        f'parameters={total_parameters:,}; trainable={trainable_parameters:,}',
        flush=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    loaders = [DataLoader(TensorDataset(torch.from_numpy(np.array(a, dtype='float32'))[:,None]),
                          batch_size=config['batch_size'], shuffle=(i==0), num_workers=0)
               for i,a in enumerate([train,val])]
    history, best, best_epoch = [], float('inf'), None
    for epoch in range(1,config['epochs']+1):
        losses = []
        for phase, loader in enumerate(loaders):
            model.train(phase==0)
            total, count = 0., 0
            with torch.set_grad_enabled(phase==0):
                for (x,) in loader:
                    x = x.to(device)
                    pred, _ = model(x)
                    loss = ((pred-x)**2).mean() + config['mae_weight']*(pred-x).abs().mean()
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    if phase==0:
                        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                    total += loss.item()*len(x); count += len(x)
            losses.append(total/count)
        history.append(dict(epoch=epoch, train_loss=losses[0], validation_loss=losses[1]))
        if losses[1] < best:
            best, best_epoch = losses[1], epoch
            torch.save(model.state_dict(), path/'best.pt')
        pd.DataFrame(history).to_csv(path/'history.csv', index=False)
        if epoch==1 or epoch%10==0:
            print(f'{name}, dim={dim}, epoch={epoch}/{config["epochs"]}, train={losses[0]:.6f}, val={losses[1]:.6f}', flush=True)
    model.load_state_dict(torch.load(path/'best.pt', map_location=device, weights_only=True))
    model.eval()
    return model, best_epoch


def evaluate(predict, x, ids, raw, lo, scale, frames, path, split, run_label=''):
    rows = []
    total = len(ids)
    print(f'  Evaluating {split}: {total} frames', flush=True)
    progress_step = max(1, min(100, total // 10))
    for position, i in enumerate(ids, start=1):
        target = np.asarray(x[i], dtype='float64')
        pred = np.asarray(predict(np.asarray(x[i:i+1]))[0], dtype='float64')
        if not np.isfinite(pred).all():
            raise ValueError('Nonfinite prediction')
        error = pred-target
        physical = pred*scale[i,:,None,None]+lo[i,:,None,None]
        # A constant original slice is reconstructed from its stored minimum.
        constant = np.ptp(raw[i], axis=(1,2)) == 0
        physical[constant] = lo[i,constant,None,None]
        pe = physical-np.asarray(raw[i], dtype='float64')
        rows.append(dict(frame=int(i), day=frames.iloc[i].day, split=split,
                         mse=float(np.mean(error**2)), mae=float(np.abs(error).mean()),
                         relative_l2=float(np.linalg.norm(error)/(np.linalg.norm(target)+1e-12)),
                         ssim=compute_ssim(target,pred,data_range=1),
                         physical_mse=float(np.mean(pe**2)), physical_mae=float(np.abs(pe).mean()),
                         physical_relative_l2=float(np.linalg.norm(pe)/(np.linalg.norm(raw[i])+1e-12))))
        if position == total or position % progress_step == 0:
            partial = rows
            running_mse = float(np.mean([row['mse'] for row in partial]))
            running_l2 = float(np.mean([row['relative_l2'] for row in partial]))
            running_ssim = float(np.mean([row['ssim'] for row in partial]))
            running_physical_rmse = float(np.sqrt(np.mean(
                [row['physical_mse'] for row in partial]
            )))
            print(
                f'    {run_label} {split}: {position}/{total} frames | '
                f'MSE={running_mse:.6g}, RMSE={np.sqrt(running_mse):.6g}, '
                f'relL2={running_l2:.6g}, SSIM={running_ssim:.6g}, '
                f'physical_RMSE={running_physical_rmse:.6g}',
                flush=True,
            )
    table = pd.DataFrame(rows)
    table.to_csv(path/f'{split}_per_frame.csv',index=False)
    table.groupby('day')[['mse','mae','relative_l2','ssim','physical_mse','physical_mae','physical_relative_l2']].mean().to_csv(path/f'{split}_per_day.csv')
    result = table.drop(columns=['frame','day','split']).mean().to_dict()
    result['rmse'] = float(np.sqrt(result['mse']))
    result['physical_rmse'] = float(np.sqrt(result['physical_mse']))
    print(
        f'  {split} final: MSE={result["mse"]:.6g}, RMSE={result["rmse"]:.6g}, '
        f'MAE={result["mae"]:.6g}, relL2={result["relative_l2"]:.6g}, '
        f'SSIM={result["ssim"]:.6g}, physical_RMSE={result["physical_rmse"]:.6g}',
        flush=True,
    )
    return result


def run_experiments(x, raw, lo, scale, frames, splits, counts, config, output):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    methods = config.get('methods', ['DCT','PCA','PlainConv3DAutoencoder','Conv3DAutoencoder'])
    allowed_methods = {'DCT','PCA','PlainConv3DAutoencoder','Conv3DAutoencoder'}
    unknown_methods = set(methods) - allowed_methods
    if unknown_methods:
        raise ValueError(f'Unknown methods: {sorted(unknown_methods)}')
    evaluation_splits = config.get('evaluation_splits', ['train', 'validation', 'test'])
    allowed_splits = {'train', 'validation', 'test'}
    unknown_splits = set(evaluation_splits) - allowed_splits
    if unknown_splits or not evaluation_splits:
        raise ValueError(f'Invalid evaluation_splits: {evaluation_splits}')
    total_runs = len(counts)*len(config['latent_dims'])*len(config['seeds'])*len(methods)
    print('Device:', device, '; runs:', total_runs, flush=True)
    run_number = 0
    for n in counts:
        train = np.asarray(x[splits[f'train_{n}']])
        val = np.asarray(x[splits['validation']])
        for dim in config['latent_dims']:
            if dim > min(len(train),int(np.prod(x.shape[1:]))):
                raise ValueError(f'PCA dim={dim} exceeds train rank bound {len(train)}; reduce latent_dims or increase smallest subset.')
            for seed in config['seeds']:
                for name in methods:
                    run_number += 1
                    path = output / f'{name}_d{dim}_days{n}_seed{seed}'
                    path.mkdir(exist_ok=True)
                    completed = path/'results.json'
                    if completed.exists():
                        print(f'[{run_number}/{total_runs}] Reusing completed {path.name}', flush=True)
                        rows.extend(json.loads(completed.read_text())); continue
                    print(f'[{run_number}/{total_runs}] Starting {path.name}: '
                          f'{len(train)} train frames, {len(val)} validation frames', flush=True)
                    seed_all(seed)
                    start = time.perf_counter()
                    best_epoch, params = None, 0
                    model = None
                    if name=='DCT':
                        print('  Fitting DCT baseline', flush=True)
                        model = DCTCompressor(dim).fit(train,verbose=False)
                        predict = lambda a: model.reconstruct(model.transform(a))
                        payload_bytes = dim*16  # Existing implementation: float64 + int64 index.
                    elif name=='PCA':
                        print('  Fitting PCA baseline', flush=True)
                        model = PCA(n_components=dim, svd_solver='randomized', random_state=seed).fit(train.reshape(len(train),-1))
                        predict = lambda a: model.inverse_transform(model.transform(a.reshape(len(a),-1))).reshape(a.shape)
                        payload_bytes = dim*4
                    else:
                        print(f'  Training neural network for {config["epochs"]} epochs', flush=True)
                        model, best_epoch = fit_ae(name,dim,train,val,config,path,device)
                        params = sum(p.numel() for p in model.parameters())
                        def predict(a):
                            with torch.no_grad():
                                return model(torch.from_numpy(np.array(a,dtype='float32'))[:,None].to(device))[0][:,0].cpu().numpy()
                        payload_bytes = dim*4
                    if device.type=='cuda':
                        torch.cuda.synchronize()
                    fit_seconds = time.perf_counter()-start
                    print(f'  Fit completed in {fit_seconds:.1f} s', flush=True)
                    run_rows = []
                    split_indices = {
                        'train': splits[f'train_{n}'],
                        'validation': splits['validation'],
                        'test': splits['test'],
                    }
                    for split in evaluation_splits:
                        ids = split_indices[split]
                        result = evaluate(predict,x,ids,raw,lo,scale,frames,path,split,
                                          run_label=path.name)
                        result.update(method=name, latent_dim=dim, train_days=n, train_frames=len(train),
                                      seed=seed, split=split, evaluation_frames=len(ids), best_epoch=best_epoch,
                                      epochs=config['epochs'] if 'Autoencoder' in name else 0,
                                      fit_seconds=fit_seconds, parameters=params,
                                      payload_bytes=payload_bytes, scaling_bytes_per_frame=x.shape[1]*2*4)
                        run_rows.append(result)
                    completed.write_text(json.dumps(run_rows,indent=2))
                    rows.extend(run_rows)
                    pd.DataFrame(rows).to_csv(output/'metrics_all_runs.csv',index=False)
                    print('Completed:',path.name, flush=True)
                    del predict, model
                    gc.collect()
                    if device.type=='cuda': torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    table.to_csv(output/'metrics_all_runs.csv',index=False)
    return table


def experiment_directory(root, config, files):
    protocol = {'config':config, 'files':[{'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in files],
                'code':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [Path(__file__),root/'src/models.py',root/'src/dct.py',root/'src/metrics.py']},
                'versions':{'torch':torch.__version__,'numpy':np.__version__,'netCDF4':nc.__version__}}
    key = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:12]
    output = root/'outputs'/f'cams_learning_curve_{key}'
    output.mkdir(parents=True,exist_ok=True)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    return output
