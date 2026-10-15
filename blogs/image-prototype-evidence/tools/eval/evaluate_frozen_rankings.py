#!/usr/bin/env python3
"""Evaluate frozen ranking ledgers, preserving completed answers and every attempt."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace

import evaluate_task_expansion as evaluator

ROOT = Path(__file__).resolve().parents[2]


def can_retry(record, path):
    if record.get('status') != 'api_failed' or any(record.get(key) for key in (
        'content', 'reasoning_content', 'partial_model_output_seen',
        'partial_answer_tag_seen', 'partial_assistant_event_count'
    )):
        return False
    if record.get('partial_output_path'):
        partial_path = Path(record['partial_output_path'])
        if not partial_path.is_file():
            return False
        partial = json.loads(partial_path.read_text())
        try:
            events = [json.loads(line) for line in partial.get('stdout', '').splitlines() if line.strip()]
        except ValueError:
            return False
        if any(event.get('type') != 'system' for event in events):
            return False
    if record.get('generation', {}).get('transport') == 'claude_cli':
        events_path = path.with_suffix('.events.json')
        if not events_path.is_file():
            return bool(record.get('partial_output_path'))
        for event in json.loads(events_path.read_text()):
            if event.get('type') == 'assistant':
                message = event.get('message', {})
                if not (event.get('is_api_error_message') and message.get('model') == '<synthetic>'
                        and message.get('usage', {}).get('output_tokens') == 0):
                    return False
            if event.get('type') == 'result' and not event.get('is_error'):
                return False
    return True


def validate_record(record, question, model, route):
    if (record.get('requested_model') != model or record.get('wire_model') != route['wire_model']
        or any(record.get(key) != question.get(key) for key in
               ('question_id', 'question_sha256', 'prompt_sha256', 'correct_order'))
        or ('provider' in route and record.get('provider') != route['provider'])):
        raise ValueError('Attempt model, question, gold or prompt identity differs from frozen request')
    if record['status'] in {'complete', 'parse_failed', 'identity_mismatch'}:
        identity = record.get('response_model') in {model, route['wire_model']}
        scored = evaluator.score_answer(question, record.get('content', ''))
        if (record.get('model_identity_matches') != identity
            or record.get('correct') != (scored['answer_correct'] and identity)
            or any(record.get(key) != value for key, value in scored.items())):
            raise ValueError('Stored answer scoring or model identity differs from actual public response')


def collect(batch, questions, models, routes=None):
    records, missing = [], []
    for model in models:
        for question in questions:
            paths = sorted((batch / 'model_evaluation').glob(
                f'attempt_*/responses/{model}/{question["question_id"]}.json'))
            selected = None
            for path in paths:
                record = json.loads(path.read_text())
                validate_record(record, question, model, routes[model] if routes else {'wire_model':model})
                if not can_retry(record, path):
                    if selected is not None:
                        raise ValueError('Multiple completed/partial answers for one model and question')
                    selected = (record, path)
            if selected and selected[0]['status'] in {'complete', 'parse_failed'} and selected[0].get('model_identity_matches'):
                record, path = selected
                public_record = {key: value for key, value in record.items() if key != 'reasoning_content'}
                records.append(dict(public_record, response_path=str(path),
                    response_sha256=evaluator.digest(path.read_bytes()),
                    instance_id=question['instance_id'], template_id=question['template_id'],
                    category=question['category'], probe=question.get('probe')))
            else:
                missing.append({'requested_model': model, 'question_id': question['question_id'],
                    'instance_id': question['instance_id'], 'probe': question.get('probe'),
                    'reason': 'unverified_model_identity' if selected and selected[0].get('model_identity_matches') is False
                              else 'no_completed_response', 'attempts': [str(p) for p in paths]})
    summaries = {}
    for model in models:
        summaries[model] = {}
        for group, probe in (('main', False), ('permutation', True)):
            expected = [q for q in questions if bool(q.get('probe')) == probe]
            rows = [r for r in records if r['requested_model'] == model and bool(r.get('probe')) == probe]
            correct = sum(bool(r.get('correct')) for r in rows)
            summaries[model][group] = {'expected': len(expected), 'completed': len(rows), 'correct': correct,
                'accuracy_on_completed': correct / len(rows) if rows else None,
                'missing': len(expected)-len(rows),
                'parse_valid': sum(r['status'] == 'complete' for r in rows)}
    return {'records': records, 'missing': missing, 'summary': summaries,
            'policy': 'Missing API responses are not scored as wrong. Completed invalid formats remain scored.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--models', default=','.join(evaluator.MODELS))
    parser.add_argument('--attempt', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout-s', type=float, default=600)
    parser.add_argument('--collect-only', action='store_true')
    parser.add_argument('--routes', type=Path, help='Frozen model to provider/transport/wire_model mapping; no credentials')
    args = parser.parse_args()
    batch = args.batch.resolve()
    models = args.models.split(',')
    if (not models or len(set(models)) != len(models) or set(models)-set(evaluator.MODELS)
        or args.attempt < 0 or args.workers < 1):
        parser.error('Invalid exact model IDs, attempt or concurrency')
    ledger = batch / 'questions.jsonl'
    questions = evaluator.load_questions(ledger)
    out = batch / 'model_evaluation'
    out.mkdir(exist_ok=True)
    lock = (out / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    routes = (json.loads(args.routes.read_text()) if args.routes else {
        model: {'provider': 'vapi' if model.startswith('claude-') else 'cctq',
                'transport': 'chat', 'wire_model': model} for model in evaluator.MODELS})
    if set(routes) != set(evaluator.MODELS):
        raise ValueError('Routes must explicitly specify all four requested models')
    for model, route in routes.items():
        if route.get('wire_model') != model:
            raise ValueError('This frozen evaluation requires the exact requested wire model ID')
        evaluator.MODEL_ALIASES.setdefault(route['provider'], {})[model] = route['wire_model']
    contract = {'ledger_sha256': evaluator.digest(ledger.read_bytes()), 'models': evaluator.MODELS,
                'reasoning_effort': 'medium', 'max_tokens': 8192, 'tools': False,
                'system_prompt': evaluator.SYSTEM_PROMPT,
                'routes': routes,
                'evaluation_source_hashes': {
                    path.name: evaluator.digest(path.read_bytes()) for path in (
                        Path(__file__), Path(evaluator.__file__), Path(evaluator.__file__).with_name('benchmark_eval_runner.py'))},
                'retry_policy': 'Only failures without model output may retry; no answer replacement.'}
    contract_path = out / 'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Frozen evaluation contract changed')
    evaluator.save(contract_path, contract)
    source_dir = out / 'source'
    source_dir.mkdir(exist_ok=True)
    for name, expected in contract['evaluation_source_hashes'].items():
        target = source_dir / name
        if target.exists() and evaluator.digest(target.read_bytes()) != expected:
            raise ValueError('Archived evaluation source changed')
        if not target.exists():
            target.write_bytes(Path(__file__).with_name(name).read_bytes())
    if not args.collect_only:
        keys_path = ROOT / 'data/evals/eval_keys.json'
        keys = json.loads(keys_path.read_text())
        jobs = []
        for question in questions:
            for model in models:
                previous = sorted(out.glob(f'attempt_*/responses/{model}/{question["question_id"]}.json'))
                if any(not can_retry(json.loads(p.read_text()), p) for p in previous):
                    continue
                if args.attempt and not previous:
                    raise ValueError('Retry cannot add an unattempted question')
                jobs.append((model, question))

        def run(job):
            model, question = job
            route = routes[model]
            provider = route['provider']
            endpoint = dict(keys[provider])
            endpoint['api_key'] = endpoint.get('api_key') or os.environ.get(endpoint.get('api_key_env', ''), '')
            if not endpoint['api_key']:
                raise ValueError('Provider credential unavailable')
            settings = SimpleNamespace(provider=provider, transport=route['transport'],
                out_dir=out/f'attempt_{args.attempt:02d}', keys=keys_path, max_tokens=8192,
                reasoning_effort='medium', timeout_s=args.timeout_s, max_retries=0)
            record = evaluator.evaluate(model, question, question['prompt_text'], endpoint, settings)
            print(json.dumps({'model': model, 'question_id': question['question_id'],
                              'status': record['status']}), flush=True)
            return record

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run, jobs))
    result = dict(contract, **collect(batch, questions, evaluator.MODELS, routes))
    evaluator.save(batch / 'evaluation_results.json', result)
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
