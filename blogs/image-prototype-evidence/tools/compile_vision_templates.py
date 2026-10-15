#!/usr/bin/env python3
"""Compile explicit image question clusters for the canonical batch pipeline."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from architecture_iq.profile import Profile
from architecture_iq.registry import ensure_registries, get_dataset_family, get_model_type
from architecture_iq.optimizers.factory import build_optimizer, render_optimizer_py
from architecture_iq.losses import render_loss_py
from architecture_iq.util import content_sha256
from compile_ntp_templates import overlay, template_comparison


def smoke_model(family, model, code):
    namespace = {}
    exec(compile(code, 'model.py', 'exec'), namespace)
    with torch.random.fork_rng():
        torch.manual_seed(1901)
        rendered = namespace['Model']()
        torch.manual_seed(1901)
        canonical = family.build_module(model)
        rendered.load_state_dict(canonical.state_dict(), strict=True)
        x = torch.randn(2, *model['input_shape'])
        rendered.eval()
        canonical.eval()
        actual, expected = rendered(x), canonical(x)
        if list(actual.shape) != [2, *model['output_shape']]:
            raise ValueError('Rendered model output shape differs from dataset')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        if not torch.isfinite(actual).all() or any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in rendered.parameters()
        ):
            raise ValueError('Nonfinite model forward or gradients')
        return {'smoke': 'standalone_render_forward_backward_and_canonical_parity',
                'output_shape': list(actual.shape), 'canonical_max_abs_error': 0.0}


def compile_templates(source: Path, smoke=False):
    spec = yaml.safe_load(source.read_text())
    if spec['schema_version'] != 'vision_question_clusters_v1':
        raise ValueError('Unsupported vision cluster schema')
    ensure_registries()
    family = get_dataset_family(spec['dataset_family'])
    metric = family.selection_metric_name()
    if spec['selection_metric'] != metric:
        raise ValueError('DSL metric differs from registered family')
    base_path = (source.parent / spec['base_profile']).resolve()
    base = Profile.load(base_path)
    default = spec['defaults']
    discovery, confirmation = spec['discovery_training_seeds'], spec['confirmation_training_seeds']
    for seeds in (discovery, confirmation):
        if not seeds or any(type(s) is not int for s in seeds) or seeds != list(range(seeds[0], seeds[0] + len(seeds))):
            raise ValueError('Canonical runner requires distinct consecutive integer training seeds')
    if set(discovery) & set(confirmation):
        raise ValueError('Discovery and confirmation seeds must be disjoint')
    statistics = spec['statistics']
    discovery_stats, confirmation_stats = statistics['discovery'], statistics['confirmation']
    if (discovery_stats['n_seeds'] != len(discovery) or discovery_stats['split'] != 'validation'
        or confirmation_stats['n_seeds'] != len(confirmation) or confirmation_stats['split'] != 'test'):
        raise ValueError('Statistics seed counts and splits must match executable protocol')
    if (confirmation_stats['test'] != 'one_sided_paired_sign_test'
        or confirmation_stats['correction'] != 'holm'
        or not 0 < confirmation_stats['alpha'] <= .05):
        raise ValueError('Only one-sided paired sign tests with Holm and alpha <= .05 are supported')
    gate = spec['selection_contract']['minimum_mean_gap']
    if not math.isfinite(gate) or gate < 0 or any(
        confirmation_stats[key] != gate for key in ('min_mean_paired_gap', 'minimum_mean_gap')
        if key in confirmation_stats
    ):
        raise ValueError('Statistical mean-gap metadata differs from executable gate')
    if spec['selection_contract'].get('require_nonoverlap', False):
        raise ValueError('Nonoverlapping seed ranges are not an implemented admission gate')
    if len(set(spec['instance_seeds'])) != len(spec['instance_seeds']):
        raise ValueError('Duplicate dataset seeds')
    datasets, candidates, receipts, templates, ids = {}, {}, {}, [], set()
    for template in spec['templates']:
        tid = template['id']
        if tid in ids:
            raise ValueError(f'Duplicate template {tid}')
        ids.add(tid)
        if len(template['choices']) not in spec.get('allowed_num_choices', [3, 4, 5]):
            raise ValueError('Unsupported choice count')
        semantics = template['budget_semantics']
        if semantics not in {'matched_steps', 'matched_samples', 'per_choice_steps'}:
            raise ValueError('Explicit budget semantics required')
        dataset_config = overlay(default['dataset'], spec['contexts'][template['context_ref']], 'dataset')
        profile = deepcopy(base)
        profile.raw['dataset_configs'][family.name] = dataset_config
        params = family.create_instance(profile, spec['instance_seeds'][0])['params']
        cells, cell_ids = [], set()
        for role in ('choices', 'diagnostics'):
            for choice in template.get(role, []):
                if choice['id'] in cell_ids:
                    raise ValueError(f'Duplicate cell {tid}/{choice["id"]}')
                cell_ids.add(choice['id'])
                model = overlay(spec['models'][choice['model_ref']], choice.get('model_updates', {}), 'model')
                if model['type'] not in family.compatible_model_types():
                    raise ValueError('Model is incompatible with dataset family')
                for key in ('input_shape', 'output_shape', 'output_kind', 'condition_dim', 'noise_steps'):
                    if key in params:
                        if key in choice.get('model_updates', {}):
                            raise ValueError(f'{key} is derived from the dataset, not a model intervention')
                        model[key] = deepcopy(params[key])
                model_family = get_model_type(model['type'])
                model_family.validate(model)
                code = model_family.render_model_py(model)
                key = content_sha256(model)
                if key not in receipts:
                    compile(code, f'{tid}/model.py', 'exec')
                    with torch.random.fork_rng():
                        module = model_family.build_module(model)
                        count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    receipts[key] = {'model_sha256': key, 'parameters': count,
                        'code_sha256': hashlib.sha256(code.encode()).hexdigest()}
                    if smoke:
                        receipts[key].update(smoke_model(model_family, model, code))
                optimizer = overlay(default['optimizer'], choice.get('optimizer_updates', {}), 'optimizer')
                build_optimizer(optimizer, torch.nn.Linear(1, 1))
                compile(render_optimizer_py(optimizer), 'optimizer.py', 'exec')
                loss = overlay(default['loss'], choice.get('loss_updates', {}), 'loss')
                if loss['loss_id'] not in base.pools['losses'][family.name]:
                    raise ValueError('Loss is not declared by the dataset profile')
                compile(render_loss_py(loss), 'loss.py', 'exec')
                changes = choice.get('training_updates', {})
                if set(changes) & {'total_samples_seen', 'eval_steps'}:
                    raise ValueError('Sample budget and evaluation checkpoints are derived')
                training = overlay(default['training'], changes, 'training')
                steps, batch = training['training_steps'], training['batch_size']
                if any(type(v) is not int or v < 1 for v in (steps, batch)):
                    raise ValueError('Positive integer steps and batch size required')
                training['total_samples_seen'] = steps * batch
                training['eval_steps'] = sorted({max(1, steps // 4), max(1, steps // 2), steps})
                if (training['precision'] != 'fp32' or training['gradient_accumulation'] != 1
                    or training['schedule'] != 'constant' or training.get('warmup_updates', 0) != 0):
                    raise ValueError('Vision prototype requires FP32, constant LR, no accumulation or reset warmup')
                if not math.isfinite(training['clip_norm']) or training['clip_norm'] <= 0:
                    raise ValueError('Positive finite gradient clipping required')
                cells.append({'id': choice['id'], 'role': 'primary' if role == 'choices' else 'diagnostic',
                    'model': model, 'optimizer': optimizer, 'loss': loss, 'training': training,
                    'parameter_count': receipts[key]['parameters'], 'model_code_sha256': receipts[key]['code_sha256']})
        primary = [c for c in cells if c['role'] == 'primary']
        if semantics == 'matched_samples' and len({c['training']['total_samples_seen'] for c in primary}) != 1:
            raise ValueError(f'{tid}: unequal sample budgets under matched_samples')
        if semantics == 'matched_steps' and len({(c['training']['training_steps'], c['training']['batch_size']) for c in primary}) != 1:
            raise ValueError(f'{tid}: unequal steps/batches under matched_steps')
        comparison = template_comparison(primary, template['varying_axes'], semantics)
        loss_fields = sorted({key for c in primary for key in c['loss']})
        for key in loss_fields:
            values = {json.dumps(c['loss'].get(key), sort_keys=True) for c in primary}
            if len(values) > 1:
                field = 'loss.' + key
                comparison['changed_fields'].append(field)
                for c in primary:
                    comparison['field_values'][c['id']][field] = c['loss'].get(key)
                if field not in template['varying_axes']:
                    comparison['undeclared_fields'].append(field)
        comparison['actual_changed_fields'] = sorted(comparison['changed_fields'])
        comparison['comparison_class'] = ('single_factor' if len(comparison['changed_fields']) == 1
            else 'bundled_recipe' if comparison['undeclared_fields'] else 'factorial')
        if not comparison['changed_fields']:
            raise ValueError('Identical primary choices are not a question')
        for seed in spec['instance_seeds']:
            dkey = 'dataset_' + content_sha256({'family': family.name, 'config': dataset_config, 'seed': seed})[:16]
            datasets[dkey] = {'node_id': dkey, 'dataset_config': dataset_config,
                              'instance_seed': seed, 'context_ref': template['context_ref']}
            resolved = {k: deepcopy(v) for k, v in template.items() if k not in {'choices', 'diagnostics'}}
            resolved.update(id=f'{tid}_s{seed}', template_id=tid, instance_seed=seed, dataset_node_id=dkey,
                            dataset_config=dataset_config, comparison=comparison, choices=[], diagnostics=[])
            for c in cells:
                semantic = {k: c[k] for k in ('model', 'optimizer', 'loss', 'training')}
                node_id = 'candidate_' + content_sha256({'dataset_node_id': dkey, **semantic})[:20]
                candidates[node_id] = {'node_id': node_id, 'dataset_node_id': dkey, **semantic,
                    'parameter_count': c['parameter_count'], 'model_code_sha256': c['model_code_sha256']}
                resolved['choices' if c['role'] == 'primary' else 'diagnostics'].append(dict(c, node_id=node_id))
            if len({c['node_id'] for c in resolved['choices'] + resolved['diagnostics']}) != len(cells):
                raise ValueError(f'Duplicate effective candidate in {tid}')
            templates.append(resolved)
    protocol = json.loads((ROOT / 'profiles/ntp_ranking3_protocol.json').read_text())
    protocol['schema_version'] = 'vision_ranking_protocol_v1'
    protocol['discovery']['seeds'] = discovery
    protocol['confirmation'].update(seeds=confirmation, metric='final_' + metric,
        minimum_adjacent_mean_gap=gate, alpha=confirmation_stats['alpha'])
    protocol['statistics'] = spec['statistics']
    protocol['answer_format'] = 'best_to_worst_v1'
    protocol['causal_scope'] += ' Image results concern the stated synthetic image distribution only.'
    return {'schema_version': 'vision_question_clusters_resolved_v1', 'dataset_family': family.name,
        'selection_metric': metric, 'profile_prefix': family.name + '_template_',
        'source_path': str(source.resolve().relative_to(ROOT)),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'base_profile': str(base_path.relative_to(ROOT)), 'default_num_choices': 3,
        'discovery_training_seeds': discovery, 'confirmation_training_seeds': confirmation,
        'selection_contract': spec['selection_contract'], 'statistics': spec['statistics'],
        'protocol_config': protocol, 'surface_heuristics': spec.get('surface_heuristics', []),
        'prompt_contract': {'representation': 'shared_complete_base_plus_independent_diff',
            'answer_format': 'best_to_worst_v1', 'analyst_only_fields':
            ['hypothesis', 'shortcut_checks', 'explanation_target', 'admission']},
        'worker_sources': ['tools/ntp_h200_worker.py', 'tools/ntp_anchor_worker.py', 'tools/vision_h200_worker.py'],
        'counts': {'templates': len(ids), 'instances': len(templates), 'dataset_nodes': len(datasets),
                   'candidate_nodes': len(candidates), 'distinct_models': len(receipts)},
        'dataset_nodes': list(datasets.values()), 'candidate_nodes': list(candidates.values()),
        'model_validation_receipts': list(receipts.values()), 'templates': templates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    plan = compile_templates(args.source, args.smoke)
    body = json.dumps(plan, ensure_ascii=False, indent=2) + '\n'
    if args.output.exists() and args.output.read_text() != body:
        raise ValueError('Resolved plan already exists with different contents; use a new run')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(body)
    print(json.dumps(plan['counts']))


if __name__ == '__main__':
    main()
