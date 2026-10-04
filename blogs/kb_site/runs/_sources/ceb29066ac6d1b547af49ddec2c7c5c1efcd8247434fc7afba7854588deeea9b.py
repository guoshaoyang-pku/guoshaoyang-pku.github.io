#!/usr/bin/env python3
"""Freeze and analyze existing epoch-4 training-lab evidence; never run training."""
import argparse
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/Users/guoshaoyang/Desktop/workdir/ArchitectureIQ')
LAB = ROOT / 'aiq_rl/data/kb_scratch/kb_lab_pool_v3'
PROOF = HERE.parent / 'probe-science-check-20261005.json'
LABELS = {
    'R63d34ce484': ['G', 'L', 'A', 'H', 'F', 'S'],
    'R5e6506510f': ['D', 'A', 'B', 'E', 'C', 'P3_TFF', 'P3_FFT', 'R3_FTF', 'P2_TT'],
    'Rd2f1faba9e': ['A', 'B'],
}
CONTRASTS = {
    'R63d34ce484': [('H', 'L'), ('F', 'S'), ('F', 'H'), ('S', 'L')],
    'R5e6506510f': [('R3_FTF', 'D'), ('B', 'P2_TT'), ('D', 'P3_FFT'),
                     ('P3_TFF', 'D'), ('A', 'P3_TFF'), ('D', 'B')],
    'Rd2f1faba9e': [('A', 'B')],
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise ValueError('Frozen evidence differs: ' + str(path))
    path.write_bytes(raw)
    return {'path': str(path.relative_to(HERE)), 'sha256': digest(raw), 'bytes': len(raw)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def git_file(repo, commit, filename):
    return subprocess.run(['git', '-C', str(repo), 'show', commit + ':' + filename],
                          capture_output=True, check=True).stdout


def hash_file(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def freeze():
    proof = read(PROOF)
    allowed_raw = (LAB / 'allowed_datasets.txt').read_bytes()
    allowed = set(allowed_raw.decode().split())
    lab_manifest = read(LAB / 'lab_manifest.json')
    assert digest(allowed_raw) == lab_manifest['files']['allowed_datasets.txt']
    manifest = {'captured_at': datetime.now().astimezone().isoformat(timespec='seconds'),
                'analysis_origin_epoch': 5, 'measurement_origin_epoch': 4,
                'kind': 'new analysis of existing measurements, not independent experiments',
                'proof': save(HERE / 'sources/report-proof.json', PROOF.read_bytes()),
                'allowlist': save(HERE / 'sources/allowed_datasets.txt', allowed_raw),
                'groups': []}
    for report in proof['reports']:
        repo = Path(report['repo'])
        commit = report['commit']
        report_raw = git_file(repo, commit, 'report.md')
        manifest['groups'].append({'report_id': report['id'], 'commit': commit,
                                  'job_id': report['job_id'], 'report': save(
                                      HERE / 'sources/reports' / (report['id'] + '.md'), report_raw),
                                  'experiments': []})
        group = manifest['groups'][-1]
        for experiment in report['experiments']:
            expid = experiment['id']
            raw = git_file(repo, commit, 'experiments/' + expid + '.json')
            assert digest(raw) == experiment['sha256']
            receipt = json.loads(raw)
            dataset = receipt['input']['dataset']
            assert dataset in allowed
            ds_path = ROOT / 'aiq_bench_repo/data/datasets' / dataset / 'dataset_spec.json'
            dsraw = ds_path.read_bytes()
            ds = json.loads(dsraw)
            item = {'id': expid, 'dataset': dataset, 'allowed': True,
                    'receipt': save(HERE / 'sources/receipts' / (expid + '.json'), raw),
                    'dataset_spec': save(HERE / 'sources/datasets' / dataset / 'dataset_spec.json', dsraw),
                    'cells': [], 'dataset_files': {}}
            for filename in ['train.pt', 'test.pt']:
                path = ds_path.parent / filename
                assert path.is_file()
                item['dataset_files'][filename] = {'source_path': str(path),
                                                   'sha256': hash_file(path), 'bytes': path.stat().st_size}
            for label, cell in zip(LABELS[report['id']], receipt['result']['results'], strict=True):
                candidate = cell['candidate']
                spec = {'schema_version': '1.0', 'profile': 'v1.5',
                        'dataset_id': ds.get('dataset_id', dataset.split('/')[-1]),
                        'family': ds['family'], 'budget': dict(candidate['budget']),
                        'model': candidate['model'], 'optimizer': candidate['optimizer'],
                        'loss': candidate['loss'], 'execution': {'device': 'cpu'},
                        'files': {'model': 'model.py', 'train': 'train.py',
                                  'loss': 'loss.py', 'optimizer': 'optimizer.py'}}
                spec['budget']['total_samples_seen'] = (spec['budget']['training_steps'] *
                                                       spec['budget']['batch_size'])
                identifier = 'x_' + hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
                spec['candidate_id'] = identifier
                source = LAB / 'experiments' / dataset / identifier
                assert read(source / 'candidate_spec.json') == spec
                files = {}
                for relative in ['candidate_spec.json', 'results/summary.json', 'results/curves.npz',
                                 'model.py', 'train.py', 'loss.py', 'optimizer.py']:
                    path = source / relative
                    meta = save(HERE / 'evidence' / expid / label / relative, path.read_bytes())
                    files[relative] = {**meta, 'source_path': str(path)}
                item['cells'].append({'label': label, 'candidate_id': identifier,
                                      'exact_spec_match': True, 'files': files})
            group['experiments'].append(item)
    write_json(HERE / 'manifest.json', manifest)


def analyze():
    manifest = read(HERE / 'manifest.json')
    results = {'analysis_at': datetime.now().astimezone().isoformat(timespec='seconds'),
               'analysis_origin_epoch': 5, 'measurement_origin_epoch': 4,
               'manifest_sha256': digest((HERE / 'manifest.json').read_bytes()),
               'candidates': [], 'contrasts': []}
    for group in manifest['groups']:
        for experiment in group['experiments']:
            receipt = read(HERE / experiment['receipt']['path'])
            cells = {}
            for entry, reported in zip(experiment['cells'], receipt['result']['results'], strict=True):
                for meta in entry['files'].values():
                    assert digest((HERE / meta['path']).read_bytes()) == meta['sha256']
                summary = read(HERE / entry['files']['results/summary.json']['path'])
                spec = read(HERE / entry['files']['candidate_spec.json']['path'])
                metric = summary['selection_metric']
                seeds = summary['seed_results']
                ids = [r['seed'] for r in seeds]
                assert len(ids) == len(set(ids)) == summary['n_seeds'] == 10
                assert ids == list(range(10))
                assert summary['base_seed'] == 0
                assert summary['failed_seeds'] == sum(r['failed'] for r in seeds) == 0
                assert summary['excluded'] is False
                loss = np.array([r['final_' + metric] for r in seeds], dtype=float)
                assert np.isfinite(loss).all()
                assert math.isclose(float(loss.mean()), summary['mean_' + metric], abs_tol=1e-12)
                assert math.isclose(float(loss.std()), summary['std_' + metric], abs_tol=1e-12)
                assert reported['mean'] == summary['mean_' + metric]
                assert reported['std'] == summary['std_' + metric]
                assert reported.get('failed_seeds') == summary['failed_seeds']
                with np.load(HERE / entry['files']['results/curves.npz']['path'], allow_pickle=False) as arrays:
                    curves = arrays['curves'].copy()
                    samples = arrays['samples'].copy()
                    batch = int(arrays['batch_size'])
                steps = spec['budget']['training_steps']
                assert curves.shape == (10, steps) and np.isfinite(curves).all()
                assert np.array_equal(curves[:, -1], loss)
                assert batch == spec['budget']['batch_size']
                assert np.array_equal(samples, np.arange(1, steps + 1) * batch)
                source = (HERE / entry['files']['model.py']['path']).read_text()
                slopes = sorted(set(float(x) for x in re.findall(r'nn\.LeakyReLU\(([0-9.]+)\)', source)))
                results['candidates'].append({'experiment': experiment['id'], 'dataset': experiment['dataset'],
                                              'label': entry['label'], 'candidate_id': entry['candidate_id'],
                                              'n_seeds': 10, 'seed_ids': ids, 'failed_seeds': 0, 'excluded': False,
                                              'mean': float(loss.mean()), 'population_std': float(loss.std()),
                                              'metric': metric, 'last_step': steps, 'last_samples': int(samples[-1]),
                                              'generated_leaky_relu_slopes': slopes,
                                              'environment': summary['environment']})
                cells[entry['label']] = {'loss': loss, 'curves': curves, 'samples': samples,
                                        'seed_ids': ids, 'entry': entry, 'spec': spec}
            for first, second in CONTRASTS[group['report_id']]:
                a, b = cells[first], cells[second]
                assert a['seed_ids'] == b['seed_ids']
                assert np.array_equal(a['samples'], b['samples'])
                differences = a['loss'] - b['loss']
                trajectory = (a['curves'] - b['curves']).mean(axis=0)
                endpoint_sign = int(np.sign(trajectory[-1]))
                signs = np.sign(trajectory).astype(int)
                last_opposite = np.flatnonzero(signs != endpoint_sign)
                stable_from = int(last_opposite[-1]) + 2 if len(last_opposite) else 1
                opposite = np.flatnonzero(signs == -endpoint_sign)
                checkpoints = sorted({1, 16, 32, 64, 128, 256, 512, 1024, 1536, len(trajectory)}
                                     & set(range(1, len(trajectory) + 1)))
                same_models = a['entry']['files']['model.py']['sha256'] == b['entry']['files']['model.py']['sha256']
                contrast = {'report_id': group['report_id'], 'experiment': experiment['id'],
                            'dataset': experiment['dataset'], 'first': first, 'second': second,
                            'metric': receipt['result']['results'][0]['metric'], 'n_paired_rng_seeds': 10,
                            'difference_first_minus_second': float(differences.mean()),
                            'paired_se': float(differences.std(ddof=1) / np.sqrt(len(differences))),
                            'first_wins': int((differences < 0).sum()), 'second_wins': int((differences > 0).sum()),
                            'ties': int((differences == 0).sum()), 'differences_by_seed': differences.tolist(),
                            'same_generated_model_source': same_models,
                            'same_generated_train_source': a['entry']['files']['train.py']['sha256'] == b['entry']['files']['train.py']['sha256'],
                            'pairing_boundary': 'Paired RNG seed IDs; no recorded initial tensor or batch-stream hashes. Different model shapes consume different RNG before batch sampling.',
                            'curve': {'first_difference': float(trajectory[0]),
                                      'endpoint_difference': float(trajectory[-1]),
                                      'observed_mean_direction_reversal': bool(len(opposite)),
                                      'opposite_direction_steps': int(len(opposite)),
                                      'endpoint_direction_steps': int((signs == endpoint_sign).sum()),
                                      'mean_sign_changes': int((signs[1:] != signs[:-1]).sum()),
                                      'endpoint_direction_persists_from_step': stable_from,
                                      'endpoint_direction_persists_from_samples': int(a['samples'][stable_from - 1]),
                                      'checkpoints': [{'step': s, 'samples': int(a['samples'][s - 1]),
                                                      'first_mean': float(a['curves'][:, s - 1].mean()),
                                                      'second_mean': float(b['curves'][:, s - 1].mean()),
                                                      'difference': float(trajectory[s - 1]),
                                                      'first_seed_wins': int((a['curves'][:, s - 1] < b['curves'][:, s - 1]).sum())}
                                                     for s in checkpoints]}}
                results['contrasts'].append(contrast)
    results['seed_count'] = sum(x['n_seeds'] for x in results['candidates'])
    write_json(HERE / 'analysis.json', results)
    checkpoint_rows = ['dataset,experiment,contrast,step,samples,first_mean,second_mean,difference,first_seed_wins']
    for row in results['contrasts']:
        for point in row['curve']['checkpoints']:
            checkpoint_rows.append(','.join(str(x) for x in [row['dataset'], row['experiment'],
                                   row['first'] + '-' + row['second'], point['step'], point['samples'],
                                   point['first_mean'], point['second_mean'], point['difference'],
                                   point['first_seed_wins']]))
    (HERE / 'curve-checkpoints.csv').write_text('\n'.join(checkpoint_rows) + '\n')
    for row in results['contrasts']:
        c = row['curve']
        print(row['dataset'].split('/')[-1], row['first'] + '-' + row['second'],
              format(row['difference_first_minus_second'], '.8g'), 'SE', format(row['paired_se'], '.6g'),
              'wins', str(row['first_wins']) + '/10', 'reversal', c['observed_mean_direction_reversal'],
              'persistent_from', c['endpoint_direction_persists_from_step'])
    if any(row['curve']['observed_mean_direction_reversal'] for row in results['contrasts']):
        plot_reversals(manifest)


def plot_reversals(manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    selected = [('R63d34ce484', 'H', 'L'), ('R5e6506510f', 'R3_FTF', 'D')]
    for row, (report_id, first, second) in enumerate(selected):
        group = next(g for g in manifest['groups'] if g['report_id'] == report_id)
        for col, experiment in enumerate(group['experiments']):
            ax = axes[row, col]
            cells = {c['label']: c for c in experiment['cells']}
            labels = ['lr=.001', 'lr=.00003'] if row == 0 else ['Residual d3 FTF', 'Plain d3 FTF']
            for name, label, color in zip([first, second], labels, ['#b6422f', '#205a8f']):
                with np.load(HERE / cells[name]['files']['results/curves.npz']['path'], allow_pickle=False) as arrays:
                    values = arrays['curves'].mean(axis=0)
                    steps = arrays['samples'] // int(arrays['batch_size'])
                ax.plot(steps, values, label=label, color=color, linewidth=1.6)
            ax.set_yscale('log')
            ax.set_xlabel('Training step (batch=16)')
            ax.set_ylabel('Mean test MSE' if row == 0 else 'Mean test CE')
            ax.set_title(experiment['dataset'].split('/')[-1], fontsize=11)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color='#dddddd', linewidth=.5)
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle('Recorded test curves, same 10 RNG seed IDs per cell', fontsize=12)
    fig.text(.5, .014, 'Top: AdamW beta2=.999, wd=1e-5. Bottom: AdamW lr=.0003; actual LeakyReLU slope=.1.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96))
    fig.savefig(HERE / 'recorded-reversals.png', dpi=180)
    fig.savefig(HERE / 'recorded-reversals.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    analyze()
