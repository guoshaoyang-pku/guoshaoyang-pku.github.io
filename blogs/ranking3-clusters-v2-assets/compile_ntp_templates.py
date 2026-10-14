#!/usr/bin/env python3
"""Resolve reusable NTP question clusters without generating data or training."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from architecture_iq.families.ntp_teacher import resolve_parameters  # noqa: E402
from architecture_iq.optimizers.factory import build_optimizer, render_optimizer_py  # noqa: E402
from architecture_iq.registry import ensure_registries, get_model_type  # noqa: E402
from architecture_iq.util import content_sha256  # noqa: E402


def overlay(base, changes, label):
    if set(changes) - set(base):
        raise ValueError(f"Unknown {label} fields: {sorted(set(changes) - set(base))}")
    return dict(deepcopy(base), **deepcopy(changes))


_DERIVED_TRAINING_FIELDS = {"total_samples_seen", "eval_steps"}
_ABSENT = "<absent>"


def _flatten(section, prefix):
    out = {}
    for key, value in section.items():
        path = f"{prefix}.{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out


def template_comparison(cells, varying_axes, budget_semantics):
    """Machine-readable audit of what actually varies across the primary choices."""
    flattened = []
    for cell in cells:
        values = {}
        for section in ("model", "optimizer", "training"):
            for path, value in _flatten(cell[section], section).items():
                if path.startswith("training.") and path.split(".", 1)[1] in _DERIVED_TRAINING_FIELDS:
                    continue
                values[path] = value
        flattened.append((cell["id"], values))
    paths = sorted({path for _, values in flattened for path in values})
    changed = [path for path in paths
               if len({json.dumps(values.get(path, _ABSENT), sort_keys=True) for _, values in flattened}) > 1]
    declared = set(varying_axes)
    undeclared = sorted(set(changed) - declared)
    if len(changed) <= 1:
        comparison_class = "single_factor"
    elif undeclared:
        comparison_class = "bundled_recipe"
    else:
        comparison_class = "factorial"
    return {"budget_semantics": budget_semantics, "changed_fields": changed,
            "undeclared_fields": undeclared, "comparison_class": comparison_class,
            "field_values": {cell_id: {path: values.get(path, _ABSENT) for path in changed}
                             for cell_id, values in flattened}}


def smoke_model(code, model):
    """Deterministic render/exec/forward/causal/backward pass to catch shape or
    causality errors at compile time. Values are environment evidence, not
    semantic inputs: no candidate or dataset identifier depends on them."""
    namespace = {}
    exec(code, namespace)
    module = namespace["Model"]()
    module.eval()
    torch.manual_seed(0)
    vocab = int(model["vocab_size"])
    seq = min(16, int(model["context_length"]))
    batch = torch.randint(0, vocab, (2, seq))
    out = module(batch)
    loss = torch.nn.functional.cross_entropy(out.reshape(-1, out.size(-1)), batch.reshape(-1))
    loss.backward()
    if not torch.isfinite(loss):
        raise ValueError("Smoke forward produced non-finite loss")
    cut = seq // 2
    altered = batch.clone()
    altered[:, cut:] = torch.randint(0, vocab, (2, seq - cut))
    with torch.no_grad():
        causal_error = float((module(altered)[:, :cut] - out[:, :cut]).abs().max())
    return {"smoke": "render_import_forward_causal_backward_pass",
            "finite_loss": float(loss), "causal_max_abs_error": causal_error}

def compile_templates(source, smoke=False):
    spec = yaml.safe_load(source.read_text())
    if spec["schema_version"] not in {"ntp_question_clusters_v1", "ntp_question_clusters_v2"}:
        raise ValueError("Unsupported question cluster schema")
    is_v2 = spec["schema_version"] == "ntp_question_clusters_v2"
    reuse_seeds = set(spec.get("reuse", {}).get("instance_seeds", []))
    if is_v2 and not reuse_seeds:
        raise ValueError("v2 requires a reuse block listing the source instance seeds")
    ensure_registries()
    defaults = spec["defaults"]
    seen_templates, dataset_nodes, candidate_nodes, resolved = set(), {}, {}, []
    compute = {}
    model_receipts = {}

    def emit_dataset_and_candidates(inst, cells, dataset, instance_seed, context_ref):
        dataset_key = content_sha256({"config": dataset, "seed": instance_seed})
        dataset_node = "dataset_" + dataset_key[:16]
        dataset_nodes[dataset_node] = {"node_id": dataset_node, "dataset_config": dataset,
                                       "instance_seed": instance_seed, "context_ref": context_ref}
        inst["dataset_node_id"] = dataset_node
        inst["dataset_config"] = dataset
        for cell in cells:
            semantic = {k: cell[k] for k in ("model", "optimizer", "training")}
            key = content_sha256({"dataset_node_id": dataset_node, **semantic})
            node_id = "candidate_" + key[:20]
            candidate_nodes[node_id] = {"node_id": node_id, "dataset_node_id": dataset_node,
                                        **semantic, "parameter_count": cell["parameter_count"],
                                        "model_code_sha256": cell["model_code_sha256"]}
            inst["choices" if cell["role"] == "primary" else "diagnostics"].append(dict(cell, node_id=node_id))
        return dataset_node

    for template in spec["templates"]:
        tid = template["id"]
        if tid in seen_templates:
            raise ValueError(f"Duplicate template: {tid}")
        seen_templates.add(tid)
        if len(template["choices"]) not in spec["allowed_num_choices"]:
            raise ValueError(f"Invalid choice count for {tid}")
        budget_semantics = template.get("budget_semantics", "matched_steps")
        if budget_semantics not in {"matched_steps", "matched_samples", "per_choice_steps"}:
            raise ValueError(f"Unknown budget semantics for {tid}: {budget_semantics}")
        dataset = overlay(defaults["dataset"], spec["contexts"][template["context_ref"]], "dataset")
        dataset = resolve_parameters(dataset)
        cells, seen_cells = [], set()
        for role in ("choices", "diagnostics"):
            for choice in template[role]:
                if choice["id"] in seen_cells:
                    raise ValueError(f"Duplicate cell {tid}/{choice['id']}")
                seen_cells.add(choice["id"])
                model = overlay(spec["models"][choice["model_ref"]], choice.get("model_updates", {}), "model")
                model["vocab_size"] = dataset["vocab_size"]
                model["context_length"] = dataset["context_length"]
                family = get_model_type(model["type"])
                family.validate(model)
                model_sha = content_sha256(model)
                if model_sha not in model_receipts:
                    with torch.device("meta"):
                        module = family.build_module(model)
                        parameter_count = sum(p.numel() for p in module.parameters())
                    code = family.render_model_py(model)
                    compile(code, f"{tid}/{choice['id']}/model.py", "exec")
                    receipt = {"model_sha256": model_sha, "parameters": parameter_count,
                               "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                               "validation": "registry_render_compile_meta_parameter_count"}
                    if smoke:
                        receipt.update(smoke_model(code, model))
                    model_receipts[model_sha] = receipt
                optimizer = overlay(defaults["optimizer"], choice.get("optimizer_updates", {}), "optimizer")
                compile(render_optimizer_py(optimizer), "optimizer.py", "exec")
                # Construction validates values against the canonical torch optimizer.
                build_optimizer(optimizer, torch.nn.Linear(1, 1))
                changes = choice.get("training_updates", {})
                if set(changes) & {"total_samples_seen", "eval_steps"}:
                    raise ValueError("Budgets/evaluation steps are derived, not independently overridden")
                training = overlay(defaults["training"], changes, "training")
                steps, batch = training["training_steps"], training["batch_size"]
                if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (steps, batch)):
                    raise ValueError("Positive integer steps and batch required")
                training["total_samples_seen"] = steps * batch
                training["eval_steps"] = sorted({0, min(100, steps), min(300, steps), steps})
                if training["schedule"] != "constant" or training["precision"] != "fp32" or training["gradient_accumulation"] != 1:
                    raise ValueError("Canonical NTP training supports constant LR, FP32 and no accumulation")
                if training["clip_norm"] <= 0:
                    raise ValueError("Positive clip_norm required")
                cells.append({"id": choice["id"], "role": "primary" if role == "choices" else "diagnostic",
                              "model": model, "optimizer": optimizer, "training": training,
                              "parameter_count": model_receipts[model_sha]["parameters"],
                              "model_code_sha256": model_receipts[model_sha]["code_sha256"]})
        if budget_semantics == "matched_samples":
            totals = {cell["training"]["total_samples_seen"] for cell in cells}
            if len(totals) != 1:
                raise ValueError(f"{tid} declares matched_samples but choices see {sorted(totals)}")
        primary = [cell for cell in cells if cell["role"] == "primary"]
        comparison = template_comparison(primary, template["varying_axes"], budget_semantics)
        for instance_seed in spec["instance_seeds"]:
            inst = {k: deepcopy(v) for k, v in template.items() if k not in {"choices", "diagnostics", "context_variants"}}
            inst.update({"id": f"{tid}_s{instance_seed}", "template_id": tid,
                         "instance_seed": instance_seed, "choices": [], "diagnostics": []})
            if is_v2:
                inst["comparison"] = comparison
            dataset_node = emit_dataset_and_candidates(inst, cells, dataset, instance_seed,
                                                       template["context_ref"])
            if len({c["node_id"] for c in inst["choices"] + inst["diagnostics"]}) != len(cells):
                raise ValueError(f"Duplicate effective candidates within {tid}")
            resolved.append(inst)
            if is_v2:
                reused = instance_seed in reuse_seeds
                compute[dataset_node] = ({"discovery": "reuse", "confirmation": "reuse"} if reused else
                                         {"discovery": list(spec["discovery_training_seeds"]),
                                          "confirmation": list(spec["confirmation_training_seeds"]),
                                          "confirmation_candidates": "all"})
            if is_v2:
                for variant in template.get("context_variants", []):
                    if instance_seed not in reuse_seeds:
                        continue
                    if variant.get("gate") != "confirmation_only":
                        raise ValueError("Context variants currently support gate=confirmation_only")
                    context = overlay(spec["contexts"][template["context_ref"]],
                                      variant["context_updates"], "context_variant")
                    variant_dataset = resolve_parameters(overlay(defaults["dataset"], context, "dataset"))
                    vinst = {k: deepcopy(v) for k, v in template.items()
                             if k not in {"choices", "diagnostics", "context_variants"}}
                    vinst.update({"id": f"{tid}__{variant['id']}_s{instance_seed}", "template_id": tid,
                                  "instance_seed": instance_seed, "choices": [], "diagnostics": [],
                                  "comparison": comparison,
                                  "context_variant": {"id": variant["id"], "gate": variant["gate"],
                                                      "hypothesis_source": variant.get("hypothesis_source"),
                                                      "base_instance_id": f"{tid}_s{instance_seed}"}})
                    vnode = emit_dataset_and_candidates(vinst, cells, variant_dataset, instance_seed,
                                                        f"{template['context_ref']}+{variant['id']}")
                    resolved.append(vinst)
                    compute[vnode] = {"discovery": None,
                                      "confirmation": list(spec["confirmation_training_seeds"]),
                                      "confirmation_candidates": "all"}
    plan = {"schema_version": "ntp_question_clusters_resolved_v1",
            "source_path": str(source.resolve().relative_to(ROOT)),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "base_profile": str((source.parent / spec["base_profile"]).resolve().relative_to(ROOT)),
            "default_num_choices": spec["default_num_choices"],
            "discovery_training_seeds": spec["discovery_training_seeds"],
            "confirmation_training_seeds": spec["confirmation_training_seeds"],
            "selection_contract": spec["selection_contract"],
            "prompt_contract": {"representation": "shared_complete_base_plus_independent_diff",
                                "analyst_only_fields": ["hypothesis", "shortcut_checks", "explanation_target", "admission"],
                                "diff_comments": "Describe code changes without expected-benefit language"},
            "supported_operator_families": sorted({m["family"] for m in spec["models"].values()}),
            "unsupported_proposed_anchors": {"mamba": "Not a canonical validated ntp_anchor_lm implementation",
                                            "kda": "Not a canonical validated ntp_anchor_lm implementation"},
            "counts": {"templates": len(seen_templates), "instances": len(resolved),
                       "dataset_nodes": len(dataset_nodes), "candidate_nodes": len(candidate_nodes),
                       "distinct_models": len(model_receipts)},
            "dataset_nodes": list(dataset_nodes.values()), "candidate_nodes": list(candidate_nodes.values()),
            "model_validation_receipts": list(model_receipts.values()), "templates": resolved}
    if is_v2:
        plan.update({"schema_version": "ntp_question_clusters_resolved_v2",
                     "statistics": spec["statistics"], "reuse": spec["reuse"], "compute": compute})
        plan["prompt_contract"] = dict(plan["prompt_contract"], answer_format="best_to_worst_v1")
    return plan

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "profiles/ntp-ranking-clusters-v1.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = compile_templates(args.source, args.smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), **result["counts"]}))


if __name__ == "__main__":
    main()
