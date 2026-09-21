"""Storage accounting from existing metrics; no model fitting or data loading."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd


def article_parameter_counts(shape, dim, attention):
    depth, height, width = shape
    flat = 128 * depth * math.ceil(height/8) * math.ceil(width/8)
    encoder = sum(a*b*27+b for a, b in [(1,32), (32,64), (64,128)])
    decoder = sum(a*b*27+b for a, b in [(128,64), (64,32), (32,1)])
    encoder += flat*dim + dim + (3*81 if attention else 0)
    decoder += dim*flat + flat + (2*81 if attention else 0)
    return encoder, decoder


def storage_tables(config):
    csv_path = Path(config['metrics_csv']).expanduser().resolve()
    metrics = pd.read_csv(csv_path)
    required = ['method','latent_dim','train_frames','train_days','seed','split',
                'parameters','payload_bytes','scaling_bytes_per_frame',
                'mse','mae','relative_l2','ssim']
    missing = set(required)-set(metrics.columns)
    if missing:
        raise ValueError(f'Missing CSV columns: {sorted(missing)}')
    shape = tuple(int(v) for v in config['field_shape'])
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError('field_shape must be positive [levels, latitude, longitude]')
    # The previous experiment stores two float32 scaling values per level.
    if not (metrics.scaling_bytes_per_frame == shape[0]*8).all():
        raise ValueError('field_shape levels do not match scaling_bytes_per_frame in CSV')
    points = math.prod(shape)
    raw_bytes = points*int(config['raw_value_bytes'])
    root = Path(config['checkpoint_root']).expanduser().resolve() if config['checkpoint_root'] else None
    rows = []
    for _, r in metrics.iterrows():
        name = r['method']
        dim = int(r.latent_dim) if pd.notna(r.latent_dim) else None
        frames = int(r.train_frames)
        decoder_bytes = full_bytes = 0
        lower_bound = False
        checkpoint = None
        note = 'Fixed transform metadata excluded; per-frame code includes its stored arrays.'
        if name in ['ArticlePlainAutoencoder', 'ArticleSAMAutoencoder']:
            enc, dec = article_parameter_counts(shape, dim, name == 'ArticleSAMAutoencoder')
            if enc+dec != int(r.parameters):
                raise ValueError(f'{name} d={dim}: shape/architecture mismatch: '
                                 f'calculated {enc+dec}, CSV has {r.parameters}')
            width = int(config['weight_value_bytes'])
            decoder_bytes, full_bytes = dec*width, (enc+dec)*width
            note = 'Estimated weight arrays; decoder includes fc_decoder and decoder_conv.'
            if root:
                checkpoint = root/f'{name}_d{dim}_days{int(r.train_days)}_seed{int(r.seed)}'/'best.pt'
                if not checkpoint.is_file():
                    checkpoint = None
        elif name == 'PCA':
            decoder_bytes = full_bytes = (points*dim+points)*config['basis_value_bytes']
            note = 'Estimated PCA basis plus mean; reused by encoder and decoder.'
        elif name == 'TruncatedSVD':
            decoder_bytes = full_bytes = points*dim*config['basis_value_bytes']
            note = 'Estimated uncentered SVD basis, shared by encoder and decoder.'
        elif name == 'RandomProjection':
            decoder_bytes = points*dim*config['projection_value_bytes']
            full_bytes = 2*decoder_bytes
            note = 'Estimated pseudoinverse for decoding; full kit also has projection matrix.'
        elif name == 'UMAP':
            decoder_bytes = full_bytes = frames*(
                points*config['umap_target_value_bytes'] + dim*config['umap_embedding_value_bytes'])
            lower_bound = True
            note = ('Lower bound: training target fields and embeddings for KNN; excludes '
                    'neighbor index, graph, fitted UMAP state and serialization overhead. '
                    'Full encoder+decoder size is unknown.')
        elif name not in ['DCT', 'Wavelet', 'Interpolation', 'TT-SVD']:
            raise ValueError(f'No reviewed storage rule for {name}')
        row = r.to_dict()
        row.update(
            field_shape=str(shape), grid_values=points, raw_array_bytes_per_frame=raw_bytes,
            per_frame_bytes=float(r.payload_bytes+r.scaling_bytes_per_frame),
            decoder_shared_array_bytes=int(decoder_bytes),
            full_shared_array_bytes=int(full_bytes),
            shared_size_status='lower_bound' if lower_bound else 'array_estimate',
            storage_note=note,
            measured_full_checkpoint_bytes=checkpoint.stat().st_size if checkpoint else np.nan,
            measured_checkpoint_path=str(checkpoint) if checkpoint else '',
        )
        rows.append(row)
    accounting = pd.DataFrame(rows)
    scenarios = []
    for _, r in accounting.iterrows():
        for count in config['archive_frame_counts']:
            count = int(count)
            if count <= 0:
                raise ValueError('archive_frame_counts must be positive')
            for scope, column in [('decoder', 'decoder_shared_array_bytes'),
                                  ('full_codec', 'full_shared_array_bytes')]:
                shared = int(r[column])
                per_frame = float(r.per_frame_bytes + shared/count)
                record = r.to_dict()
                record.update(
                    archive_frames=count, scope=scope,
                    estimated_archive_bytes=shared+count*r.per_frame_bytes,
                    estimated_bytes_per_frame=per_frame,
                    estimated_bits_per_grid_value=8*per_frame/points,
                    estimated_compression_ratio=raw_bytes/per_frame,
                    ratio_is_upper_bound=r.shared_size_status == 'lower_bound',
                )
                # The checkpoint is full-network state, never substitute it for decoder-only.
                record['measured_weights_plus_estimated_codes_bytes_per_frame'] = (
                    r.per_frame_bytes + r.measured_full_checkpoint_bytes/count
                    if scope == 'full_codec' and pd.notna(r.measured_full_checkpoint_bytes)
                    else np.nan)
                scenarios.append(record)
    scenarios = pd.DataFrame(scenarios)
    output = Path(config['output_dir']).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    accounting.to_csv(output/'storage_components.csv', index=False)
    scenarios.to_csv(output/'storage_scenarios.csv', index=False)
    for split in ['validation','test']:
        scenarios[scenarios.split == split].to_csv(output/f'error_vs_bytes_{split}.csv', index=False)
    protocol = dict(
        config=config, metrics_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        counts_are='Array-storage estimates, not measured encoded file sizes',
        exclusions='File headers, coordinates, codec metadata and software; no quantization or entropy coding',
        per_frame_source='payload_bytes + scaling_bytes_per_frame from original metrics CSV',
        checkpoint_measurement='Existing full-network best.pt file size only; not decoder-only',
        umap='Lower bound; full serialized UMAP+KNN object was not saved',
        scenarios='Projected archive sizes at N frames; quality remains that of the original evaluation split',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output/'storage_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    return accounting, scenarios


def plot_storage(scenarios, config):
    import matplotlib.pyplot as plt
    output = Path(config['output_dir'])
    selected = scenarios[scenarios.split == config['plot_split']]
    paths = []
    for scope in ['decoder', 'full_codec']:
        for count in config['archive_frame_counts']:
            table = selected[(selected.scope == scope) & (selected.archive_frames == count)]
            if table.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            for method, group in table.groupby('method', sort=False):
                # Do not join different seeds or training pools into one curve.
                for (seed, days), run in group.groupby(['seed','train_days']):
                    run = run.sort_values('estimated_bytes_per_frame')
                    bound = (run.shared_size_status == 'lower_bound').any()
                    label = f'{method}' + (' (storage lower bound)' if bound else '')
                    if group.groupby(['seed','train_days']).ngroups > 1:
                        label += f' seed={seed}, days={days}'
                    ax.plot(run.estimated_bytes_per_frame, run.mse, marker='o',
                            linestyle='--' if bound else '-', label=label)
            ax.set_xscale('log')
            if (table.mse > 0).all():
                ax.set_yscale('log')
            ax.set_xlabel('Estimated bytes per frame, including shared arrays / N')
            ax.set_ylabel('Normalized MSE')
            ax.set_title(f'{config["plot_split"]}; {scope}; N={count:,}')
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
            fig.tight_layout()
            for extension in ['png','pdf']:
                path = output/f'error_vs_bytes_{config["plot_split"]}_{scope}_N{count}.{extension}'
                fig.savefig(path, dpi=180, bbox_inches='tight')
                paths.append(path)
            plt.show()
            plt.close(fig)
    return paths
