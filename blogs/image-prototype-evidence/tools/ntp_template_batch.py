#!/usr/bin/env python3
"""Materialize template clusters using canonical dataset, candidate and GT contracts."""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from architecture_iq.candidates.generator import build_candidate_spec, write_candidate  # noqa: E402
from architecture_iq.candidates.sets import make_set_name, write_set_manifest  # noqa: E402
from architecture_iq.profile import Profile  # noqa: E402
from architecture_iq.registry import ensure_registries, get_dataset_family, get_model_type  # noqa: E402
from architecture_iq.util import content_sha256  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path, value):
    body = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists() and path.read_text() != body:
        raise ValueError(f'Frozen artifact differs: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def copy_frozen(source, target):
    if target.exists() and sha(source) != sha(target):
        raise ValueError(f'Frozen source differs: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)


def save_profile(path, raw):
    freeze(path, raw)
    return Profile.load(path)


def jobs_for_tree(tree, mapping, seeds, phase):
    sources = {str(p.relative_to(tree)): sha(p) for directory in ('src', 'tools')
               for p in (tree / directory).rglob('*.py') if '__pycache__' not in p.parts}
    jobs = []
    for dataset_node, dataset in mapping['datasets'].items():
        candidates = [value['path'] for value in mapping['candidates'].values()
                      if value['dataset_node_id'] == dataset_node]
        raw = read(tree / dataset['profile_path'])
        raw['ground_truth'].update(base_seed=seeds[0], n_seeds=len(seeds), max_failed_seeds=1)
        profile_path = f'run_profiles/{phase}_{dataset_node}.json'
        save_profile(tree / profile_path, raw)
        files = [p for p in (tree / dataset['path']).iterdir() if p.is_file()]
        for path in candidates:
            files.extend(p for p in (tree / path).iterdir() if p.is_file())
        files.append(tree / profile_path)
        artifacts = {str(p.relative_to(tree)): sha(p) for p in files}
        jobs.append({'id': f'{phase}_{dataset_node}', 'config': {
            'protocol': 'canonical_dataset_candidate_gt_v1', 'profile_path': profile_path,
            'dataset_path': dataset['path'], 'candidate_paths': candidates,
            'source_hashes': sources, 'artifact_hashes': artifacts,
            'order_seed': seeds[0], 'cpu_threads': 4}})
    return jobs


def prepare(args):
    import torch
    torch.set_num_threads(4)
    ensure_registries()
    plan = read(args.plan)
    out = args.out.resolve()
    tree = out / 'discovery_tree'
    seeds = plan['discovery_training_seeds']
    compute = plan.get('compute', {})
    reuse = plan.get('reuse', {})
    reuse_source_seeds = reuse.get('source_discovery_training_seeds', seeds)

    def node_discovery_seeds(node_id):
        annotation = compute.get(node_id)
        if annotation is None:
            return seeds
        if annotation['discovery'] == 'reuse':
            return list(reuse_source_seeds)
        if annotation['discovery'] is None:
            # confirmation-only context variant: training evidence is the
            # confirmation battery, so provenance carries those seeds
            return list(plan['confirmation_training_seeds'])
        return list(annotation['discovery'])
    freeze(out / 'resolved_plan.json', plan)
    source_files = [p for p in (ROOT / 'src/architecture_iq').rglob('*.py')
                    if not {'prompts', 'questions'} & set(p.parts)]
    source_files += [ROOT / path for path in plan.get('worker_sources',
        ['tools/ntp_anchor_worker.py', 'tools/ntp_h200_worker.py'])]
    for path in source_files:
        copy_frozen(path, tree / path.relative_to(ROOT))
    base = Profile.load(ROOT / plan['base_profile'])
    family = get_dataset_family(plan.get('dataset_family', 'random_teacher_ntp'))
    profile_prefix = plan.get('profile_prefix', 'ntp_template_')
    mapping = {'plan_sha256': sha(out / 'resolved_plan.json'), 'datasets': {}, 'candidates': {}}
    by_dataset = defaultdict(list)
    for node in plan['candidate_nodes']:
        by_dataset[node['dataset_node_id']].append(node)
    for node in plan['dataset_nodes']:
        node_seeds = node_discovery_seeds(node['node_id'])
        raw = deepcopy(base.raw)
        raw.update(profile=profile_prefix + node['node_id'], identity_hash_hex=32)
        raw['pools']['num_choices'] = 3
        raw['dataset_configs'][family.name] = node['dataset_config']
        raw['ground_truth'].update(device='cuda', base_seed=node_seeds[0], n_seeds=len(node_seeds), max_failed_seeds=1)
        raw['prompts'].update(code_format='base_diff', include_training_steps_axis=True)
        profile_rel = 'profiles/' + node['node_id'] + '.json'
        profile = save_profile(tree / profile_rel, raw)
        spec = family.build_spec_with_id(family.create_instance(profile, node['instance_seed']))
        dataset_rel = 'data/datasets/' + family.name + '/' + spec['dataset_id']
        dataset = tree / dataset_rel
        family.materialize(spec, dataset)
        mapping['datasets'][node['node_id']] = {'path': dataset_rel, 'dataset_id': spec['dataset_id'],
            'profile_path': profile_rel, 'instance_seed': node['instance_seed']}
        for candidate in by_dataset[node['node_id']]:
            candidate_raw = deepcopy(raw)
            candidate_raw['profile'] = profile_prefix + candidate['node_id']
            candidate_raw['training_defaults'][family.name] = candidate['training']
            candidate_raw['budgets']['total_samples_seen'] = [candidate['training']['total_samples_seen']]
            candidate_profile_rel = 'profiles/' + candidate['node_id'] + '.json'
            candidate_profile = save_profile(tree / candidate_profile_rel, candidate_raw)
            budget = candidate['training']['total_samples_seen']
            cspec = build_candidate_spec(candidate_profile, dataset_id=spec['dataset_id'], family=family.name,
                budget=budget, batch_size=candidate['training']['batch_size'], model=candidate['model'],
                optimizer=candidate['optimizer'], loss=candidate.get('loss', {'loss_id': 'cross_entropy'}), execution_device='cuda',
                provenance={'template_node_id': candidate['node_id'], 'resolved_plan_sha256': mapping['plan_sha256']})
            if cspec['source_hashes']['model'] != candidate['model_code_sha256']:
                raise ValueError('Compiled model source differs')
            if cspec['trainable_parameter_count'] != candidate['parameter_count']:
                raise ValueError('Compiled parameter count differs')
            if 'teacher' in spec and cspec['trainable_parameter_count'] <= spec['teacher']['teacher_parameters']:
                raise ValueError('Teacher must be smaller than students')
            set_name = make_set_name(budget, frozenset(), salt=candidate['node_id'], execution_device='cuda')
            set_path = dataset / 'candidates' / set_name
            cpath = set_path / cspec['candidate_id']
            write_candidate(cspec, cpath, get_model_type(candidate['model']['type']))
            if not (set_path / 'set.json').exists():
                write_set_manifest(set_path, set_name=set_name, budget=budget, count=1,
                    varying_axes=frozenset(), fixed_shared={}, model_type_counts=None,
                    seed=node['instance_seed'], profile=candidate_profile, dataset_id=spec['dataset_id'], family=family.name)
            mapping['candidates'][candidate['node_id']] = {'path': str(cpath.relative_to(tree)),
                'candidate_id': cspec['candidate_id'], 'dataset_node_id': node['node_id'],
                'profile_path': candidate_profile_rel}
        print(json.dumps({'prepared_dataset': node['node_id'], 'candidates': len(by_dataset[node['node_id']])}), flush=True)
    freeze(out / 'artifact_mapping.json', mapping)
    jobs = [job for dataset_node, annotation in compute.items() if isinstance(annotation.get('discovery'), list)
            for job in jobs_for_tree(tree, {'plan_sha256': mapping['plan_sha256'],
                                            'datasets': {dataset_node: mapping['datasets'][dataset_node]},
                                            'candidates': mapping['candidates']},
                                     annotation['discovery'], 'discovery')] if compute else \
        jobs_for_tree(tree, mapping, seeds, 'discovery')
    freeze(out / 'discovery_jobs.json', jobs)
    freeze(out / 'discovery_inputs.json', {str(p.relative_to(tree)): sha(p) for p in tree.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts})
    print(json.dumps({'prepared': len(mapping['candidates']), 'jobs': len(jobs)}), flush=True)


def prepare_confirmation(args):
    out = args.out.resolve()
    if not (out / 'frozen_protocol.json').exists():
        raise ValueError('Freeze discovery directions and complete family before confirmation')
    plan, mapping = read(out / 'resolved_plan.json'), read(out / 'artifact_mapping.json')
    protocol = read(out / 'frozen_protocol.json')
    if not sha(out / 'resolved_plan.json') == mapping['plan_sha256'] == protocol['template_plan_sha256']:
        raise ValueError('Compiled plan identity changed between phases')
    source, target = out / 'discovery_tree', out / 'confirmation_tree'
    frozen_inputs = read(out / 'discovery_inputs.json')
    for rel, expected in frozen_inputs.items():
        if sha(source / rel) != expected:
            raise ValueError(f'Discovery input changed before confirmation: {rel}')
    for rel in frozen_inputs:
        copy_frozen(source / rel, target / rel)
    compute = plan.get('compute', {})
    confirm_nodes = {node for node, annotation in compute.items() if annotation.get('confirmation') != 'reuse'}
    if compute:
        scoped = {'plan_sha256': mapping['plan_sha256'],
                  'datasets': {n: d for n, d in mapping['datasets'].items() if n in confirm_nodes},
                  'candidates': {c: v for c, v in mapping['candidates'].items() if v['dataset_node_id'] in confirm_nodes}}
        jobs = jobs_for_tree(target, scoped, plan['confirmation_training_seeds'], 'confirmation')
    else:
        jobs = jobs_for_tree(target, mapping, plan['confirmation_training_seeds'], 'confirmation')
    freeze(out / 'confirmation_jobs.json', jobs)
    freeze(out / 'confirmation_inputs.json', {str(p.relative_to(target)): sha(p) for p in target.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts})
    print(json.dumps({'confirmation_jobs': len(jobs), 'seeds': plan['confirmation_training_seeds']}))


def freeze_directions(args):
    from architecture_iq.questions.confirmation import build_frozen_protocol, freeze_sources
    out = args.out.resolve()
    plan, mapping = read(out / 'resolved_plan.json'), read(out / 'artifact_mapping.json')
    tree = out / 'discovery_tree'
    is_v2 = plan['schema_version'] == 'ntp_question_clusters_resolved_v2'
    config = deepcopy(plan.get('protocol_config')) if plan.get('protocol_config') else read(ROOT / 'profiles/ntp_ranking3_protocol.json')
    if is_v2:
        config = dict(config, statistics=plan['statistics'],
                      answer_format=plan['prompt_contract']['answer_format'])
        imported_path = out / 'imported_protocol_entries.json'
        if not imported_path.exists():
            raise ValueError('v2 requires tools/import_ntp_template_reuse.py stage-a before freeze-directions')
        imported = read(imported_path)['entries']
    else:
        imported = []
    reuse_seeds = set(plan.get('reuse', {}).get('instance_seeds', []))
    config['discovery']['seeds'] = plan['discovery_training_seeds']
    config['confirmation']['seeds'] = plan['confirmation_training_seeds']
    entries = []
    for template in plan['templates']:
        if is_v2 and ('context_variant' in template or template['instance_seed'] in reuse_seeds):
            continue  # entries come from the hash-verified import
        dataset = mapping['datasets'][template['dataset_node_id']]
        dpath = tree / dataset['path']
        paths, validation = [], {}
        for choice in template['choices']:
            candidate = mapping['candidates'][choice['node_id']]
            path = tree / candidate['path']
            paths.append(path)
            summary = read(path / 'results/summary.json')
            expected = plan['discovery_training_seeds']
            if summary['failed_seeds'] or [r['seed'] for r in summary['seed_results']] != expected:
                raise ValueError('Discovery must complete all fixed seeds before direction freeze')
            validation[candidate['candidate_id']] = [read(path / f'results/diagnostics_seed{seed}.json')
                ['final_validation_metric'] for seed in expected]
        order = sorted(validation, key=lambda cid: (sum(validation[cid])/len(validation[cid]), cid))
        dataset_files = sorted(p.name for p in dpath.iterdir() if p.is_file())
        entries.append({'instance_id': template['id'], 'cluster_id': template['template_id'],
            'category': template['category'], 'dataset_id': dataset['dataset_id'],
            'candidate_order': order, 'discovery_validation': validation,
            **freeze_sources(dpath, paths, dataset_files)})
    entries.extend(imported)
    config['template_plan_sha256'] = sha(out / 'resolved_plan.json')
    config['evaluation']['choice_label_seed'] = 73013
    config['assembler_sha256'] = sha(Path(__file__))
    freeze(out / 'frozen_protocol.json', build_frozen_protocol(entries, config))
    print(json.dumps({'frozen_instances': len(entries), 'imported_instances': len(imported),
        'hypotheses': sum(len(e['candidate_order'])-1 for e in entries)}))


def make_letter_probe(body):
    """Rotate answer letters by one position. The probe asks the same question
    under a permuted letter assignment; its gold order must remap identically.
    Choice positions are preserved; only the letter labels rotate."""
    probe = deepcopy(body)
    letters = [choice['letter'] for choice in body['choices']]
    remap = dict(zip(letters, letters[1:] + letters[:1]))
    for choice in probe['choices']:
        choice['letter'] = remap[choice['letter']]
    probe['correct_order'] = [remap[letter] for letter in body['correct_order']]
    if body.get('answer_format') == 'best_to_worst_v1':
        probe['answer'] = json.dumps({'best_to_worst': probe['correct_order']}, separators=(',', ':'))
    del probe['question_id']
    probe['probe'] = {'type': 'letter_permutation', 'source_question_id': body['question_id']}
    probe['question_id'] = 'q_' + content_sha256(probe)[:32]
    return probe


def assemble(args):
    import itertools
    import random
    from architecture_iq.questions.confirmation import analyze_confirmation
    from architecture_iq.questions.generator import build_ranking_question_record
    from architecture_iq.prompts.renderer import write_prompt
    ensure_registries()
    out = args.out.resolve()
    tree = out / 'confirmation_tree'
    plan, mapping = read(out / 'resolved_plan.json'), read(out / 'artifact_mapping.json')
    protocol = read(out / 'frozen_protocol.json')
    is_v2 = plan['schema_version'] == 'ntp_question_clusters_resolved_v2'
    by_id = {row['candidate_id']: row for row in mapping['candidates'].values()}
    observations = {}
    for entry in protocol['entries']:
        values = {}
        for cid in entry['candidate_order']:
            path = tree / by_id[cid]['path'] / 'results/summary.json'
            if not path.exists():
                continue
            summary = read(path)
            if summary['failed_seeds'] or summary['excluded']:
                continue
            values[cid] = {str(row['seed']): row[protocol['confirmation']['metric']] for row in summary['seed_results']}
        observations[entry['instance_id']] = values
    freeze(out / 'confirmation_observations.json', observations)
    stats = analyze_confirmation(protocol, observations)
    freeze(out / 'confirmation_statistics.json', stats)
    imported_orders_path = out / 'imported_question_orders.json'
    imported_orders = read(imported_orders_path) if imported_orders_path.exists() else {}
    labels = {}
    for count in sorted({len(t['choices']) for t in plan['templates']}):
        indices = [i for i, t in enumerate(plan['templates']) if len(t['choices']) == count]
        permutations = list(itertools.permutations('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:count]))
        orders = (permutations * ((len(indices) + len(permutations)-1) // len(permutations)))[:len(indices)]
        random.Random(protocol['evaluation']['choice_label_seed']).shuffle(orders)
        assigned = dict(zip(indices, orders))
        for i in indices:
            if plan['templates'][i]['id'] in imported_orders:
                assigned[i] = tuple(imported_orders[plan['templates'][i]['id']])
            if tuple(assigned[i]) not in permutations:
                raise ValueError('Letter preassignment must come from the shared permutation bank')
        labels.update(assigned)
    ledger = []
    for index, template in enumerate(plan['templates']):
        if not stats['admitted'][template['id']]:
            continue
        dataset = mapping['datasets'][template['dataset_node_id']]
        dpath = tree / dataset['path']
        profile = Profile.load(tree / f"run_profiles/confirmation_{template['dataset_node_id']}.json")
        paths = [tree / mapping['candidates'][choice['node_id']]['path'] for choice in template['choices']]
        specs = [read(path / 'candidate_spec.json') for path in paths]
        profile.raw['cross_profile_reuse'] = {'enabled': True, 'transformer_allowlist': [
            {'source_profile': spec['profile'], 'source_profile_hash': spec['profile_hash'],
             'model_types': [spec['model']['type']]} for spec in specs]}
        profile.prompts['code_format'] = 'base_diff'
        evidence = {'instance_id': template['id'], 'protocol': protocol, 'observations': observations}
        for shuffle_seed in range(10000):
            body = build_ranking_question_record(profile, dataset_spec=read(dpath/'dataset_spec.json'),
                dataset_path=dpath, candidate_paths=paths, candidate_set_paths=list(dict.fromkeys(p.parent for p in paths)),
                rng=random.Random(shuffle_seed), artifact_root=tree/'data', confirmation_evidence=evidence)
            if body['correct_order'] == list(labels[index]):
                break
        else:
            raise ValueError('Cannot realize preassigned label order')
        candidate_nodes = {choice['letter']: next(node for node, value in mapping['candidates'].items()
                          if value['candidate_id'] == choice['candidate_id']) for choice in body['choices']}
        variants = [body]
        if is_v2 or plan.get('prompt_contract', {}).get('answer_format') == 'best_to_worst_v1':
            body['comparison'] = template.get('comparison')
            if 'context_variant' in template:
                body['context_variant'] = template['context_variant']
            body['answer_format'] = 'best_to_worst_v1'
            body['answer'] = json.dumps({'best_to_worst': body['correct_order']}, separators=(',', ':'))
            variants.append(make_letter_probe(body))
        for record in variants:
            directory = out / 'questions' / record['question_id']
            freeze(directory / 'question.json', record)
            write_prompt(directory, dataset_path=dpath, artifact_root=tree/'data', task='ranking')
            row = {'question_id': record['question_id'], 'question_path': str(directory),
                'question_sha256': sha(directory/'question.json'), 'prompt_sha256': sha(directory/'prompt.txt'),
                'family': record['family'], 'dataset_id': dataset['dataset_id'], 'instance_id': template['id'],
                'template_id': template['template_id'], 'category': template['category'], 'group': template['category'],
                'task': f"ranking{len(template['choices'])}", 'num_choices': len(template['choices']), 'correct_order': record['correct_order'],
                'candidate_nodes': {c['letter']: candidate_nodes[orig] for c, orig in
                                    zip(record['choices'], [c['letter'] for c in body['choices']])},
                'confirmation_protocol_sha256': protocol['protocol_sha256']}
            if 'probe' in record:
                row['probe'] = record['probe']
            ledger.append(row)
    text = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in ledger)
    path = out / 'questions.jsonl'
    if path.exists() and path.read_text() != text:
        raise ValueError('Frozen ledger differs')
    path.write_text(text)
    print(json.dumps({'admitted': len(ledger), 'total': len(plan['templates'])}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'prepare-confirmation', 'freeze-directions', 'assemble'])
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'prepare-confirmation': prepare_confirmation, 'freeze-directions': freeze_directions, 'assemble': assemble}[args.action](args)


if __name__ == '__main__':
    main()
