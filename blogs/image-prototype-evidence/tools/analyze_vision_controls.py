#!/usr/bin/env python3
"""Freeze simple ranking heuristics and audit the complete local control graph."""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import statistics

from ntp_template_batch import freeze, read, sha

DEFAULT_HEURISTICS = [
    {'id': name, 'field': field, 'higher_is_better': higher}
    for name, field, higher in (
        ('more_parameters', 'parameter_count', True),
        ('more_training_steps', 'training.training_steps', True),
        ('deeper_model', 'model.depth', True),
        ('larger_kernel', 'model.kernel_size', True),
        ('residual', 'model.residual', True),
        ('smaller_patches', 'model.patch_size', False),
    )
]


def get(row, field):
    value = row
    for part in field.split('.'):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def changes(a, b):
    result = {}
    for section in ('model', 'optimizer', 'loss', 'training'):
        for key in sorted(a[section].keys() | b[section].keys()):
            # These are derived from steps and batch size, not independent changes.
            if section == 'training' and key in {'eval_steps', 'total_samples_seen'}:
                continue
            av, bv = a[section].get(key), b[section].get(key)
            if av != bv:
                result[section + '.' + key] = [av, bv]
    return result


def heuristic_order(choices, rule):
    values = {c['node_id']: get(c, rule['field']) for c in choices}
    if any(not isinstance(v, (int, float)) for v in values.values()):
        return {'status': 'not_applicable', 'values': values}
    if len(set(values.values())) != len(values):
        return {'status': 'tie', 'values': values}
    return {'status': 'ordered', 'values': values,
            'best_to_worst': sorted(values, key=values.get, reverse=rule['higher_is_better'])}


def freeze_design(run):
    plan = read(run / 'resolved_plan.json')
    # A fresh design may not be written after confirmation or LLM answers exist.
    path = run / 'control_audit_design.json'
    if not path.exists() and ((run / 'confirmation_tree').exists() or (run / 'model_evaluation').exists()):
        raise ValueError('Freeze control audit before confirmation and model evaluation')
    rules = plan.get('surface_heuristics') or DEFAULT_HEURISTICS
    instances = []
    for t in plan['templates']:
        cells = t['choices'] + t['diagnostics']
        pairs = []
        for a, b in combinations(cells, 2):
            actual = changes(a, b)
            pairs.append({'a': a['id'], 'b': b['id'], 'a_node': a['node_id'], 'b_node': b['node_id'],
                          'changed_fields': actual, 'single_factor': len(actual) == 1})
        instances.append({'instance_id': t['id'], 'template_id': t['template_id'],
            'pairs': pairs, 'heuristics': {h['id']: heuristic_order(t['choices'], h) for h in rules}})
    body = {'schema_version': 'vision_control_audit_design_v1', 'plan_sha256': sha(run / 'resolved_plan.json'),
            'heuristic_rules': rules, 'instances': instances,
            'scope': 'All local pairwise interventions are descriptive; no post-hoc significance or exclusive mechanism claim.'}
    freeze(path, body)
    return body


def paired_contrast(a, b, seeds):
    gaps = [b[s] - a[s] for s in seeds]
    return {'n': len(gaps), 'mean_b_minus_a': statistics.mean(gaps),
            'sd_b_minus_a': statistics.stdev(gaps) if len(gaps) > 1 else None,
            'min_b_minus_a': min(gaps), 'max_b_minus_a': max(gaps),
            'a_better_seeds': sum(x > 0 for x in gaps), 'b_better_seeds': sum(x < 0 for x in gaps),
            'ties': sum(x == 0 for x in gaps), 'paired_gaps': gaps}


def validate_results(summary, diagnostic, location, metric, seeds):
    if (summary.get('candidate_id') != location['candidate_id']
        or summary.get('selection_metric') != metric.removeprefix('final_')
        or summary.get('n_seeds') != len(seeds) or summary.get('base_seed') != seeds[0]):
        raise ValueError('Control summary candidate, metric or seed identity differs')
    for row in summary['seed_results']:
        seed = row['seed']
        d = diagnostic[seed]
        if (row.get('failed') is not False or d.get('failed') is not False or d.get('seed') != seed
            or not isinstance(row.get(metric), (int,float)) or not math.isfinite(row[metric])
            or row[metric] != d.get('final_test_metric')):
            raise ValueError('Control diagnostics differ from completed finite summary observations')


def analyze(run):
    design, plan, mapping = (read(run / f) for f in (
        'control_audit_design.json', 'resolved_plan.json', 'artifact_mapping.json'))
    if design['plan_sha256'] != sha(run / 'resolved_plan.json'):
        raise ValueError('Frozen control design differs from compiled plan')
    tree, seeds = run / 'confirmation_tree', plan['confirmation_training_seeds']
    metric = 'final_' + plan['selection_metric']
    values, diagnostics, candidates = {}, {}, []
    for node, location in mapping['candidates'].items():
        directory = tree / location['path'] / 'results'
        summary_path = directory / 'summary.json'
        if not summary_path.exists():
            candidates.append({'node_id': node, 'status': 'missing_summary'})
            continue
        summary = read(summary_path)
        rows = summary['seed_results']
        complete = not summary['failed_seeds'] and not summary['excluded'] and [r['seed'] for r in rows] == seeds
        candidates.append({'node_id': node, 'status': 'complete' if complete else 'incomplete',
                           'summary_sha256': sha(summary_path)})
        if complete:
            diagnostics[node] = {s: read(directory / f'diagnostics_seed{s}.json') for s in seeds}
            validate_results(summary, diagnostics[node], location, metric, seeds)
            candidate_spec = read(directory.parent/'candidate_spec.json')
            if (candidate_spec['candidate_id'] != location['candidate_id']
                or candidate_spec['dataset_id'] != mapping['datasets'][location['dataset_node_id']]['dataset_id']
                or any(d.get('training_protocol') != candidate_spec['training']
                       or d.get('optimizer') != candidate_spec['optimizer'] for d in diagnostics[node].values())):
                raise ValueError('Control candidate source or training protocol differs')
            values[node] = {r['seed']: r[metric] for r in rows}
            measured = diagnostics[node].values()
            for key in ('initial_validation_metric', 'final_validation_metric', 'final_test_metric',
                        'final_training_metric', 'ood_metric', 'clipped_step_fraction'):
                numbers = [d[key] for d in measured if isinstance(d.get(key), (float, int))]
                if len(numbers) == len(seeds):
                    candidates[-1]['mean_' + key] = statistics.mean(numbers)
            candidates[-1]['mean_last20_minibatch_loss'] = statistics.mean(
                statistics.mean(d['training_losses'][-20:]) for d in measured)
            if all(d.get('gradient_norms') for d in measured):
                candidates[-1]['mean_preclip_gradient_norm'] = statistics.mean(
                    statistics.mean(d['gradient_norms']) for d in measured)
                candidates[-1]['mean_clipped_step_fraction'] = statistics.mean(
                    statistics.mean(n > d['training_protocol']['clip_norm'] for n in d['gradient_norms']) for d in measured)
            candidates[-1]['training_loss_definition'] = {
                'loss':candidate_spec['loss'],
                **{k:candidate_spec['training'][k] for k in ('prediction_type','loss_weighting','noise_schedule')
                   if k in candidate_spec['training']}}
    stats = read(run / 'confirmation_statistics.json')
    protocol = read(run / 'frozen_protocol.json')
    cid_to_node = {v['candidate_id']: n for n, v in mapping['candidates'].items()}
    gold = {e['instance_id']: [cid_to_node[c] for c in e['candidate_order']] for e in protocol['entries']}
    instances, heuristics, final_hashes = [], {}, {}
    rules = {r['id']: r for r in design['heuristic_rules']}

    def final_state_hash(node, seed):
        key = (node,seed)
        if key not in final_hashes:
            import torch
            path = tree/mapping['candidates'][node]['path']/f'results/checkpoint_seed{seed}.pt'
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            if checkpoint['candidate_spec']['candidate_id'] != mapping['candidates'][node]['candidate_id']:
                raise ValueError('Sampling-control checkpoint candidate identity differs')
            digest = hashlib.sha256()
            for name, tensor in checkpoint['state_dict'].items():
                digest.update(name.encode()); digest.update(tensor.contiguous().numpy().tobytes())
            final_hashes[key] = digest.hexdigest()
        return final_hashes[key]
    for instance in design['instances']:
        pairs = []
        for pair in instance['pairs']:
            a, b = pair['a_node'], pair['b_node']
            if a not in values or b not in values:
                pairs.append(dict(pair, status='missing_measurement'))
                continue
            da, db = diagnostics[a], diagnostics[b]
            # Names vary across trainer versions; report actual evidence only.
            hashes = {}
            for key in ('initial_state_sha256', 'initial_state_hash', 'batch_stream_sha256', 'batch_stream_hash',
                        'diffusion_noise_stream_sha256'):
                if all(key in da[s] and key in db[s] for s in seeds):
                    hashes[key] = sum(da[s][key] == db[s][key] for s in seeds)
            prefix = None
            sampling_fields = {'training.sampling_steps', 'training.sampling_eta'}
            changed = set(pair['changed_fields'])
            if changed and changed <= sampling_fields | {'training.training_steps'}:
                errors = []
                for s in seeds:
                    la, lb = da[s]['training_losses'], db[s]['training_losses']
                    errors.append(max(abs(x-y) for x,y in zip(la, lb)))
                prefix = {'exactly_matching_seeds': sum(x == 0 for x in errors),
                          'max_absolute_loss_difference': max(errors),
                          'scope': 'Actual shared training-loss prefix; bitwise match is reported only when measured.'}
            if changed and changed <= sampling_fields:
                hashes['final_state_sha256'] = sum(final_state_hash(a,s) == final_state_hash(b,s) for s in seeds)
            pairs.append(dict(pair, status='measured', **paired_contrast(values[a], values[b], seeds),
                              matching_hash_seeds=hashes, training_prefix=prefix))
        admitted = stats['admitted'][instance['instance_id']]
        predictions = {}
        for name, prediction in instance['heuristics'].items():
            row = dict(prediction, exact_correct=prediction.get('best_to_worst') == gold[instance['instance_id']]
                       if prediction['status'] == 'ordered' else None)
            gold_positions = {node: index for index,node in enumerate(gold[instance['instance_id']])}
            partial = []
            for a,b in combinations(prediction['values'], 2):
                av,bv = prediction['values'][a],prediction['values'][b]
                if not all(isinstance(v,(int,float)) for v in (av,bv)) or av == bv:
                    continue
                a_first = (av > bv) == rules[name]['higher_is_better']
                partial.append(a_first == (gold_positions[a] < gold_positions[b]))
            row.update(comparable_pairs=len(partial), correct_pairs=sum(partial))
            predictions[name] = row
            if admitted:
                total = heuristics.setdefault(name, {'admitted': 0, 'ordered': 0, 'tie': 0,
                    'not_applicable': 0, 'correct': 0, 'comparable_pairs':0, 'correct_pairs':0})
                total['admitted'] += 1
                total[prediction['status']] += 1
                total['correct'] += row['exact_correct'] is True
                total['comparable_pairs'] += len(partial)
                total['correct_pairs'] += sum(partial)
        instances.append(dict(instance_id=instance['instance_id'], template_id=instance['template_id'],
                              admitted=admitted, pairs=pairs, heuristics=predictions))
    for total in heuristics.values():
        total['accuracy_when_ordered'] = total['correct'] / total['ordered'] if total['ordered'] else None
        total['accuracy_on_comparable_pairs'] = total['correct_pairs']/total['comparable_pairs'] if total['comparable_pairs'] else None
    result = {'schema_version': 'vision_control_audit_v1', 'design_sha256': sha(run / 'control_audit_design.json'),
        'metric': metric, 'seeds': seeds, 'candidates': candidates, 'instances': instances,
        'heuristic_summary': heuristics,
        'interpretation': 'Paired differences describe local interventions on fixed data. Bundled recipe differences do not isolate one cause. Tied heuristics do not produce a full ranking and are not counted as errors.'}
    (run / 'control_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    result = freeze_design(args.run) if args.freeze else analyze(args.run)
    print(json.dumps({'instances': len(result['instances'])}))


if __name__ == '__main__':
    main()
