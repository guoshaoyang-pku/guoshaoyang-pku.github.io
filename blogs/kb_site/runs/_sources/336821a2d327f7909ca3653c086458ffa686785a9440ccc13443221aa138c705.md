# Reproduction

Run from this repository root:

    python research/audit.py
    python research/audit_extra.py
    python research/analyze.py

Audit inputs are the saved pre-experiment XOR snapshots, not a live load_lab(), which includes newly run experiments. The erroneous checkpoint audit is retained as audit_checkpoint_including_experiments.json and must not be used as baseline evidence. audit_complete.json adds loss, budget and exact LN-mask checks. Analyze reads six raw experiment response files and regenerates summary.json and cells.csv. manifest.json specifies allowed bases, all dotted-key overrides, order-aligned target ablation labels, preregistered predictions and observation pin. To retrain use run_experiment(base=...,variants=...) in chunks of at most 12; the independent replication additionally supplies dataset=xor_classification/xorcls_fd5734. Tool engine supplies 10 seeds per cell; seed-level losses and parameter trajectories are unavailable. Do not treat repeated specifications or within-set cell comparisons as independent datasets.

Snapshot kb.json preserves original claims. refresh_e0002.json records later observations. report.md in repository root is the latest report saved by publish_report.
