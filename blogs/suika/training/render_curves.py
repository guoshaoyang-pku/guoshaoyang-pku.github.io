"""Render public Suika CSVs; --export-logs refreshes them from the local logs."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / "dqn_runs"
GROUPS = [
    {
        "label": "Set Transformer", "color": "#16734b", "board": "448x720",
        "runs": ["a100_t1_2/wave6_20260930/w6s_h720",
                 "a100_t1_2/wave6_20260930/w6d_h720",
                 "a100_t1_2/wave6_20260930/w6c_h720"],
    },
    {
        "label": "MLP", "color": "#3e6caa", "board": "448x720",
        "runs": ["a100_perm/wave6_20260930/w6s_m720",
                 "a100_perm/wave6_20260930/w6c_m720"],
    },
    {
        "label": "Set Transformer", "color": "#16734b", "board": "550x720",
        "runs": ["a100_t1_3/wave6_20260930/w6s_var",
                 "a100_t1_3/wave6_20260930/w6c_var"],
    },
    {
        "label": "MLP", "color": "#3e6caa", "board": "550x720",
        "runs": ["a100_t1/wave6_20260930/w6s_mvar",
                 "a100_t1/wave6_20260930/w6c_mvar"],
    },
]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 15, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#a6ada8", "axes.labelcolor": "#313a35",
    "text.color": "#28322c", "xtick.color": "#5b645e",
    "ytick.color": "#5b645e", "figure.facecolor": "white",
    "axes.facecolor": "white", "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def read_log(key, name):
    path = RUNS / key / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(name, records):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def contiguous(rows):
    start = 0
    for i in range(1, len(rows)):
        if rows[i]["env_steps"] < rows[i - 1]["env_steps"]:
            yield rows[start:i]
            start = i
    yield rows[start:]


def save_figure(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_logs():
    evaluations, losses, sources = [], [], []
    summary = {}
    for group in GROUPS:
        clean_points = []
        for key in group["runs"]:
            for name in ["eval.jsonl", "metrics.jsonl"]:
                path = RUNS / key / name
                sources.append({"path": str(path.relative_to(ROOT)),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            rows = read_log(key, "eval.jsonl")
            for index, row in enumerate(rows):
                for board, cell in row["grid"].items():
                    evaluations.append({
                        "run": key.split("/")[-1], "evaluation_index": index,
                        "board": board, "env_steps": row["env_steps"],
                        "grad_steps": row["grad_steps"],
                        "time_utc": datetime.fromtimestamp(row["time"], timezone.utc).isoformat(),
                        "n": cell["n"], "mean_score": cell["mean"],
                        "median_score": cell["median"], "max_score": cell["max"],
                        "mean_moves": cell["moves_mean"],
                        "censored_episodes": cell["censored"],
                    })
                cell = row["grid"][group["board"]]
                if not cell["censored"]:
                    clean_points.append({"run": key.split("/")[-1],
                                         "grad_steps": row["grad_steps"], **cell})
            bins = {}
            for row in read_log(key, "metrics.jsonl"):
                if np.isfinite(row["loss"]) and row["loss"] > 0:
                    bins.setdefault(row["env_steps"] // 10_000_000, []).append(row)
            for bucket, entries in sorted(bins.items()):
                losses.append({
                    "run": key.split("/")[-1], "board": group["board"],
                    "bin_start_env_steps": bucket * 10_000_000,
                    "median_env_steps": float(np.median([r["env_steps"] for r in entries])),
                    "median_loss": float(np.median([r["loss"] for r in entries])),
                    "n_metric_records": len(entries),
                })
        summary[f"{group['board']}/{group['label']}"] = {
            "best_uncensored_evaluation_mean": max(clean_points, key=lambda p: p["mean"]),
            "final_evaluation": clean_points[-1],
        }
    write_csv("evaluation_curves.csv", evaluations)
    write_csv("loss_curves.csv", losses)
    return evaluations, losses, sources, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-logs", action="store_true")
    args = parser.parse_args()
    if args.export_logs:
        evaluations, losses, sources, summary = export_logs()
    else:
        with (OUT / "evaluation_curves.csv").open() as handle:
            evaluations = list(csv.DictReader(handle))
        for point in evaluations:
            for field in ["env_steps", "grad_steps", "evaluation_index", "n", "censored_episodes"]:
                point[field] = int(point[field])
            for field in ["mean_score", "median_score", "max_score", "mean_moves"]:
                point[field] = float(point[field])
        with (OUT / "loss_curves.csv").open() as handle:
            losses = list(csv.DictReader(handle))
        for point in losses:
            for field in ["median_env_steps", "median_loss"]:
                point[field] = float(point[field])
        manifest = json.loads((OUT / "provenance.json").read_text())
        sources, summary = manifest["sources"], manifest["summary"]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))
    for ax, board, subtitle in zip(axes, ["448x720", "550x720"],
                                   ["Fixed board · 12 seeds per checkpoint",
                                    "Variable-board training · 4 seeds per board"]):
        ax.set_title(board.replace("x", " × "), loc="left", pad=26, fontweight="bold")
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=10, color="#666f69")
        for group in (g for g in GROUPS if g["board"] == board):
            labeled = False
            for key in group["runs"]:
                run = key.split("/")[-1]
                rows = [r for r in evaluations if r["run"] == run and r["board"] == board]
                style = "--" if run.startswith("w6c_") else "-"
                for part in contiguous(rows):
                    x = [r["env_steps"] / 1e9 for r in part]
                    y = [r["mean_score"] for r in part]
                    ax.plot(x, y, style, color=group["color"], linewidth=1.5,
                            marker="o", markersize=2, alpha=0.9,
                            label=group["label"] if not labeled else None)
                    labeled = True
                capped = [r for r in rows if r["censored_episodes"]]
                if capped:
                    ax.scatter([r["env_steps"] / 1e9 for r in capped],
                               [r["mean_score"] for r in capped],
                               marker="^", s=66, facecolor="white",
                               edgecolor=group["color"], linewidth=1.7, zorder=5)
        ax.set_xlabel("Reported cumulative environment steps (billions)", labelpad=8)
        ax.set_ylabel("Greedy evaluation mean score")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e7ebe8", linewidth=0.8)
        ax.legend(frameon=False, loc="upper left", fontsize=10)
    axes[1].annotate("3,000-move cap\nScores are lower bounds",
                     xy=(1.019, 22638.25), xytext=(1.65, 20800), fontsize=9,
                     arrowprops={"arrowstyle": "-", "color": "#777f79"}, color="#606963")
    fig.suptitle("Suika: learning to keep merging", x=0.064, y=0.98,
                 ha="left", fontsize=21, fontweight="bold")
    fig.text(0.064, 0.90, "Wave 6 · from scratch, then continued training · every recorded evaluation shown",
             fontsize=11, color="#616a64")
    fig.text(0.064, 0.024,
             "Solid: initial training and DDP resume. Dashed: continuation. △: a capped evaluation, with scores as lower bounds.\n"
             "Greedy Q actions; current + next fruit only. Fixed seed suites are development evaluations, not held-out estimates.",
             fontsize=9, color="#616a64")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.74, bottom=0.19, wspace=0.27)
    save_figure(fig, "training_curves")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7))
    for ax, board in zip(axes, ["448x720", "550x720"]):
        for group in (g for g in GROUPS if g["board"] == board):
            for index, key in enumerate(group["runs"]):
                run = key.split("/")[-1]
                pts = [r for r in losses if r["run"] == run]
                ax.plot([r["median_env_steps"] / 1e9 for r in pts],
                        [r["median_loss"] for r in pts],
                        "--" if run.startswith("w6c_") else "-",
                        color=group["color"], linewidth=1.7,
                        label=group["label"] if index == 0 else None)
        ax.set_title(board.replace("x", " × "), loc="left", fontweight="bold")
        ax.set_yscale("log")
        ax.set_xlabel("Reported cumulative environment steps (billions)")
        ax.set_ylabel("Training loss (log scale)")
        ax.set_xlim(left=0)
        ax.grid(axis="y", which="major", color="#e7ebe8", linewidth=0.8)
        ax.legend(frameon=False, fontsize=10)
    fig.suptitle("Training loss", x=0.064, y=0.99, ha="left", fontsize=20, fontweight="bold")
    fig.text(0.064, 0.025, "Median in 10-million-step bins. Solid: initial training / resume; dashed: continuation.\n"
             "Loss is an optimization diagnostic. It does not measure gameplay quality; use the evaluation curves above.",
             fontsize=9, color="#616a64")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.24, wspace=0.27)
    save_figure(fig, "training_loss")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_policy": "greedy argmax Q; current and next fruit; no future-seed lookahead",
        "evaluation_suites": {"448x720": "seeds 0-11 (12 per checkpoint)",
                              "550x720": "seeds 0-3 (4 per board per checkpoint)"},
        "sampling_note": "Fixed development suites reused during training; means are not held-out estimates.",
        "curve_note": "All evaluation records retained; lines break when the environment-step counter rolls back after a restart.",
        "censoring_note": "Triangle markers denote cells containing capped episodes; those scores are lower bounds.",
        "continuation_note": "w6c runs preserve weights and environment-step counters, restart learning-rate schedules and gradient counters.",
        "loss_note": "Median of finite positive logged losses per 10-million environment-step bin; no gameplay claim follows from loss.",
        "sources": sources, "summary": summary,
    }
    if args.export_logs:
        (OUT / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "evaluation_points": len(evaluations),
                      "loss_bins": len(losses), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
