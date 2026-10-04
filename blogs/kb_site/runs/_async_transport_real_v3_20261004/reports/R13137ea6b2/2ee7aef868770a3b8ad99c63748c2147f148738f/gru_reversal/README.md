Reproduce endpoint analysis with: python gru_reversal/analysis.py

Replay training with run_experiment using replay_manifest.json entries. The lab API fixes nominal seeds0..9; batch indices and runtime tensor dimensions are not exposed. Preserve dataset IDs and compare only co-materialized set IDs. Nonfinite aggregates are serialized as strings, not successful seed counts. Repeated controls are deduplicated by active model/optimizer configuration.

report.md is the full current report. frozen_lab_bigram.json excludes newly appended experiments; history_bigram_e0002.json preserves historical prompts and provenance. external_provenance_e0002.json is not included in this investigation counts.
