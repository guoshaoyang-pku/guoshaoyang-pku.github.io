#!/usr/bin/env python3
"""Appendix S1 · scaling 三视图绘图（一图一变量：train / val / gap）。

数据源：data/runs_scaling/（pilot run 或 full grid run）。

每张图只变化一个变量，同时画 train / val / gap 三行：
  - epoch 组：固定模块（both），只变 epoch 长度 L（L1/L4 pilot；full 后 L1-L4）
  - table 组：固定模块（bigram-only）+ 固定 L4，只变 table size（1M/128K/16K）
"""
import json
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUNS_DIR = os.environ.get(
    "NGLAB_SCALING_RUNS_DIR",
    os.path.join(REPO_ROOT, "data", "runs_scaling"),
)
FIGS_DIR = os.path.join(HERE, "figs")
os.makedirs(FIGS_DIR, exist_ok=True)


def load_train_log(run_dir):
    path = os.path.join(run_dir, "train_log.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            out.append(e)
    return out


def load_fixed(run_dir):
    path = os.path.join(run_dir, "fixed_train_loss.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            out.append(e)
    return out


def load_exact_last(run_dir):
    path = os.path.join(run_dir, "exact_freq_loss.jsonl")
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    return last


def canonical_run_dirs(runs_dir):
    for run_dir in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        if not os.path.isdir(run_dir):
            continue
        physical_id = os.path.basename(run_dir)
        if physical_id.endswith("_fixed"):
            yield physical_id[:-len("_fixed")], physical_id, run_dir
        elif physical_id.startswith("basic_"):
            yield physical_id, physical_id, run_dir
        else:
            continue


def valid_summary(run_dir, physical_id):
    path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        summary = json.load(f)
    config = summary.get("config", {})
    # Accept the historical appendix contracts for provenance figures only.
    is_basic = physical_id.startswith("basic_")
    interval_ok = (
        config.get("val_interval_steps") in (10, 25)
        and config.get("table_norm_interval_steps") in (10, 25)
        and summary.get("probe_eval_interval") in (10, 25)
    )
    if not (
        summary.get("run_id") == physical_id
        and config.get("table_optimizer") == "rmsprop"
        and config.get("table_lr_scale") in (1.0, 2.0)
        and config.get("table_betas") == [0.0, 0.99]
        and interval_ok
    ):
        return None
    if not is_basic:
        # legacy pilot contract: require the *_fixed naming + val_interval 10
        if not physical_id.endswith("_fixed"):
            return None
        if not (
            config.get("val_interval_steps") == 10
            and config.get("table_norm_interval_steps") == 10
            and summary.get("probe_eval_interval") == 10
        ):
            return None
    return summary


def eligible_freq_gaps(record, branch):
    if not record:
        return []
    train = record.get("train", {}).get(branch, {})
    val = record.get("val", {}).get(branch, {})
    rows = []
    for key in set(train) & set(val):
        f = int(key)
        if f <= 0:
            continue
        t = train[key]
        v = val[key]
        if min(t["token_count"], v["token_count"]) < 1024:
            continue
        if min(t["distinct_contexts"], v["distinct_contexts"]) < 32:
            continue
        rows.append((f, v["mean_loss"] - t["mean_loss"]))
    return sorted(rows)


def log_bin_median(rows, n_bins=14):
    if not rows:
        return np.array([]), np.array([])
    frequencies = np.asarray([row[0] for row in rows])
    gaps = np.asarray([row[1] for row in rows])
    edges = np.geomspace(frequencies.min(), frequencies.max(), n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (frequencies >= lo) & (frequencies < hi)
        if mask.any():
            xs.append(np.exp(np.mean(np.log(frequencies[mask]))))
            ys.append(np.median(gaps[mask]))
    return np.asarray(xs), np.asarray(ys)


def three_panel(axs, xs, train, val, xlabel, label, color):
    axs[0].plot(xs, train, color=color, marker=".", markersize=4)
    axs[0].set_ylabel("train loss")
    axs[0].grid(alpha=0.3)
    axs[1].plot(xs, val, color=color, marker=".", markersize=4)
    axs[1].set_ylabel("val loss")
    axs[1].grid(alpha=0.3)
    axs[2].plot(xs, np.array(val) - np.array(train), color=color, marker=".",
                markersize=4)
    axs[2].set_ylabel("gap (val−train)")
    axs[2].set_xlabel(xlabel)
    axs[2].grid(alpha=0.3)
    axs[2].axhline(0, color="gray", lw=0.6)
    for ax, ys in ((axs[0], train), (axs[1], val),
                   (axs[2], np.array(val) - np.array(train))):
        if len(xs) and len(ys):
            ax.annotate(
                label,
                xy=(xs[-1], ys[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                color=color,
                fontsize=8,
                va="center",
            )


def first_run(runs, *run_ids):
    """Return the first available run, preferring full-grid ids."""
    for run_id in run_ids:
        if run_id in runs:
            return run_id
    return None


def epoch_run_id(runs, L, module, align):
    return first_run(
        runs,
        f"basic_{L}_{module}_{align}",
        f"ep_{L}_{module}_{align}",
        f"pilot_ep_{L}_{module}_{align}",
    )


def table_run_id(runs, mult, module):
    pilot_names = {
        64: "1M",
        8: "128K",
        1: "16K",
    }
    pilot = pilot_names.get(mult)
    candidates = [f"basic_tbl_{pilot}_{module}", f"tbl_{mult}_{module}"]
    if pilot is not None:
        candidates.append(f"tbl_pilot_{pilot}_{module}")
    return first_run(runs, *candidates)


def plot_epoch_fixedstep(runs):
    """固定模块=both，只改变 epoch length L；三视图 vs step."""
    fig, axs = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    colors = {"L1": "#2196F3", "L2": "#4CAF50", "L3": "#FF9800", "L4": "#E91E63"}
    for L, color in colors.items():
        run_id = epoch_run_id(runs, L, "both", "fs")
        if run_id is None:
            continue
        log = runs[run_id]["train_log"]
        if not log:
            continue
        xs = [e["step"] for e in log]
        tr = [e["train_loss"] for e in log]
        va = [e["val_loss"] for e in log]
        three_panel(axs, xs, tr, va, "step", L, color)
    fig.suptitle("Epoch length scaling (fixed-step; module=both)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "epoch_fixedstep_both_train_val_gap.png"), dpi=150)
    plt.close(fig)
    print("saved epoch_fixedstep_both_train_val_gap.png")


def plot_epoch_fixedstep_nogram(runs):
    """backbone safety 对照：固定 no-ngram，只改变 epoch length L."""
    fig, axs = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    colors = {"L1": "#2196F3", "L2": "#4CAF50", "L3": "#FF9800", "L4": "#E91E63"}
    for L, color in colors.items():
        run_id = epoch_run_id(runs, L, "nogram", "fs")
        if run_id is None:
            continue
        log = runs[run_id]["train_log"]
        if not log:
            continue
        xs = [e["step"] for e in log]
        tr = [e["train_loss"] for e in log]
        va = [e["val_loss"] for e in log]
        three_panel(axs, xs, tr, va, "step", L, color)
    fig.suptitle("Backbone safety (fixed-step; module=no-ngram)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "epoch_fixedstep_nogram_train_val_gap.png"), dpi=150)
    plt.close(fig)
    print("saved epoch_fixedstep_nogram_train_val_gap.png")


def plot_epoch_fixed_probe(runs):
    """固定 probe 的 train / val / gap 三视图，只改变 epoch length L."""
    fig, axs = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    colors = {"L1": "#2196F3", "L2": "#4CAF50", "L3": "#FF9800", "L4": "#E91E63"}
    for L, color in colors.items():
        run_id = epoch_run_id(runs, L, "both", "fs")
        if run_id is None:
            continue
        fixed = runs[run_id]["fixed"]
        if not fixed:
            continue
        xs = [e["step"] for e in fixed]
        tr = [e["fixed_train_loss"] for e in fixed]
        va = [e["fixed_val_loss"] for e in fixed]
        three_panel(axs, xs, tr, va, "step", L, color)
    fig.suptitle("Fixed-probe scaling (fixed-step; module=both)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "epoch_fixedprobe_gap_vs_step.png"), dpi=150)
    plt.close(fig)
    print("saved epoch_fixedprobe_gap_vs_step.png")


def plot_table_fixedstep(runs):
    """固定 L4 + bigram-only，只改变 table size；三视图."""
    fig, axs = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    sizes = (
        (64, "1M", "#2196F3"),
        (32, "512K", "#42A5F5"),
        (16, "256K", "#66BB6A"),
        (8, "128K", "#FF9800"),
        (4, "64K", "#FFB74D"),
        (2, "32K", "#AB47BC"),
        (1, "16K", "#E91E63"),
    )
    for mult, label, color in sizes:
        run_id = table_run_id(runs, mult, "bigram")
        if run_id is None:
            continue
        log = runs[run_id]["train_log"]
        if not log:
            continue
        xs = [e["step"] for e in log]
        tr = [e["train_loss"] for e in log]
        va = [e["val_loss"] for e in log]
        three_panel(axs, xs, tr, va, "step", label, color)
    fig.suptitle("Table-size scaling (fixed-step; L4, module=bigram-only)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "table_bigram_train_val_gap.png"), dpi=150)
    plt.close(fig)
    print("saved table_bigram_train_val_gap.png")


def plot_table_final_vs_size(runs):
    """table size 扫描：final train / val / gap vs logical 2R."""
    fig, axs = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
    pts = []
    for mult, label in ((64, "1M"), (32, "512K"), (16, "256K"),
                        (8, "128K"), (4, "64K"), (2, "32K"), (1, "16K")):
        run_id = table_run_id(runs, mult, "bigram")
        if run_id is None:
            continue
        fixed = runs[run_id]["fixed"]
        if not fixed:
            continue
        logical = 2 * 8192 * mult
        last = fixed[-1]
        pts.append((logical, label, last["fixed_train_loss"],
                    last["fixed_val_loss"], last["fixed_gap"]))
    pts.sort()
    xs = [p[0] for p in pts]
    for ax, pos in zip(axs, (2, 3, 4)):
        ys = [p[pos] for p in pts]
        ax.plot(xs, ys, marker="o", color="#9C27B0")
        for x, label, *_ in pts:
            ax.annotate(label, (x, ys[xs.index(x)]), xytext=(4, 3),
                        textcoords="offset points", fontsize=8)
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("final train loss")
    axs[1].set_ylabel("final val loss")
    axs[2].set_ylabel("final gap")
    axs[2].set_xlabel("logical addresses 2R (per n-gram, per layer)")
    for ax in axs:
        ax.set_xscale("log")
    fig.suptitle("Table-size scaling (final fixed probe; L4, module=bigram-only)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "table_final_gap_vs_2R.png"), dpi=150)
    plt.close(fig)
    print("saved table_final_gap_vs_2R.png")


def plot_backbone_safety(runs):
    """Long L1 no-ngram safety run, using the fixed probe."""
    run = runs.get("bb_safety_L1_nogram_5000")
    if not run or not run["fixed"]:
        return
    fixed = run["fixed"]
    xs = [entry["step"] for entry in fixed]
    train = [entry["fixed_train_loss"] for entry in fixed]
    val = [entry["fixed_val_loss"] for entry in fixed]
    fig, axs = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    three_panel(axs, xs, train, val, "step", "L1 no-ngram", "#7B1FA2")
    fig.suptitle(
        "Backbone safety (L1, no-ngram; fixed probe; running snapshot)",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIGS_DIR, "backbone_safety_L1_nogram_long.png"),
        dpi=150,
    )
    plt.close(fig)
    print("saved backbone_safety_L1_nogram_long.png")


def plot_exact_freq_summary(runs):
    """Noise-reduced exact-f summary; only eligible f values are shown."""
    groups = [
        (
            "Epoch scaling by exact f (eligible f; module=both)",
            [
                ("L1", epoch_run_id(runs, "L1", "both", "fs"), "#2196F3"),
                ("L4", epoch_run_id(runs, "L4", "both", "fs"), "#E91E63"),
            ],
        ),
        (
            "Table-size scaling by exact f (eligible f; module=bigram-only)",
            [
                ("1M", table_run_id(runs, 64, "bigram"), "#2196F3"),
                ("128K", table_run_id(runs, 8, "bigram"), "#FF9800"),
                ("16K", table_run_id(runs, 1, "bigram"), "#E91E63"),
            ],
        ),
    ]
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, (title, entries) in zip(axs, groups):
        for label, run_id, color in entries:
            run = runs.get(run_id)
            record = load_exact_last(run["run_dir"]) if run else None
            x, y = log_bin_median(eligible_freq_gaps(record, "bigram"))
            if len(x):
                ax.plot(x, y, "o-", label=label, color=color)
        ax.axhline(0, color="gray", lw=0.7)
        ax.set_xscale("log")
        ax.set_ylabel("gap (val−train) @ f")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    axs[1].set_xlabel("exact bigram frequency f")
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIGS_DIR, "exact_freq_gap_by_f_summary.png"),
        dpi=160,
    )
    plt.close(fig)
    print("saved exact_freq_gap_by_f_summary.png")


def main():
    runs = {}
    ignored_noncanonical = 0
    ignored_invalid = 0
    for run_id, physical_id, run_dir in canonical_run_dirs(RUNS_DIR):
        if not (
            run_id.startswith(("ep_", "pilot_ep_", "tbl_", "basic_"))
            or run_id == "bb_safety_L1_nogram_5000"
        ):
            ignored_noncanonical += 1
            continue
        summary = valid_summary(run_dir, physical_id)
        if summary is None:
            ignored_invalid += 1
            continue
        runs[run_id] = {
            "train_log": load_train_log(run_dir),
            "fixed": load_fixed(run_dir),
            "run_dir": run_dir,
            "summary": summary,
        }
    if ignored_noncanonical:
        print(f"ignored {ignored_noncanonical} non-canonical scaling directories")
    if ignored_invalid:
        print(f"ignored {ignored_invalid} scaling directories with an invalid canonical contract")
    print(f"found {len(runs)} runs")
    plot_epoch_fixedstep(runs)
    plot_epoch_fixedstep_nogram(runs)
    plot_epoch_fixed_probe(runs)
    plot_table_fixedstep(runs)
    plot_table_final_vs_size(runs)
    plot_backbone_safety(runs)
    plot_exact_freq_summary(runs)


if __name__ == "__main__":
    main()
