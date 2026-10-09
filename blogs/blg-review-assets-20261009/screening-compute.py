"""Run frozen, new AB sample conditions using the archived lattice solver."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time


def compute(project, configuration, output):
    sys.path.insert(0, str(project))
    from execution import execute_point, file_sha256
    point = execute_point(configuration, str(output / "raw"))
    if "result_path" not in point:
        raise RuntimeError(json.dumps(point))
    path = Path(point["result_path"])
    record = json.loads(path.read_text())
    if file_sha256(path) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Result checksum mismatch")
    if file_sha256(path.parent / "arrays.npz") != record["arrays_sha256"]:
        raise ValueError("Raw array checksum mismatch")
    return {"point": point, "record": record, "result_sha256": file_sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("This run allocates at most four CPU workers")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(name + " must be one")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("This protocol is CPU only")
    sys.path.insert(0, str(args.project))
    from execution import atomic_json, file_sha256
    from v1_schema import source_hash
    from verification.lattice_atlas import compare
    import numpy
    import scipy
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    if source_hash(args.project) != protocol["source_sha256"]:
        raise ValueError("Solver differs from the frozen training source")
    if file_sha256(Path(__file__)) != protocol["compute_driver_sha256"]:
        raise ValueError("Compute driver differs from frozen protocol")
    args.out.mkdir(parents=True, exist_ok=True)
    frozen_path = args.out / "protocol.json"
    if frozen_path.exists() and frozen_path.read_bytes() != protocol_bytes:
        raise ValueError("Output belongs to a different protocol")
    frozen_path.write_bytes(protocol_bytes)
    records = [None] * len(protocol["configurations"])
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(compute, args.project, c, args.out): i
                for i, c in enumerate(protocol["configurations"])}
        for future in as_completed(jobs):
            records[jobs[future]] = future.result()
            atomic_json(args.out / "progress.json", {"completed": sum(r is not None for r in records),
                "total": len(records), "wall_seconds": time.perf_counter() - started})
    points = []
    for row in protocol["heldout"]:
        coarse, fine = [records[row["grid_indices"][g]] for g in ("coarse", "fine")]
        comparison_inputs = [{"point": r["point"],
            "observables": r["record"]["result"]["observables"],
            "integration": r["record"]["result"]["integration"]} for r in (coarse, fine)]
        observed = fine["record"]["result"]["observables"]
        errors = {name: abs(prediction - observed["internal_U_meV"])
                  for name, prediction in row["predicted_internal_U_meV"].items()}
        points.append({**deepcopy(row), "observables": observed,
            "comparison": compare(*comparison_inputs), "prediction_errors_meV": errors,
            "phase_resolved": False})
    if source_hash(args.project) != protocol["source_sha256"]:
        raise RuntimeError("Solver source changed during computation")
    report = {"schema": "ab-screening-prospective/1", "protocol_sha256": protocol_sha,
        "source_sha256": protocol["source_sha256"], "records": records, "points": points,
        "prediction_summary": {name: {
            "max_error_meV": max(p["prediction_errors_meV"][name] for p in points),
            "rms_error_meV": (sum(p["prediction_errors_meV"][name]**2 for p in points) / len(points))**.5,
            "within_frozen_range": all(p["prediction_errors_meV"][name] <=
                protocol["prediction_tolerance_meV"] for p in points)}
            for name in protocol["models"]},
        "cost": {"workers": args.workers, "wall_seconds": time.perf_counter() - started,
            "cumulative_solver_cpu_seconds": sum(r["record"]["cost"]["cpu_user_s"] +
                r["record"]["cost"]["cpu_system_s"] for r in records),
            "max_worker_RSS_bytes": max(r["record"]["cost"]["max_rss_bytes"] for r in records),
            "gpu_calls": 0},
        "environment": {"host": platform.node(), "python": sys.version.split()[0],
            "numpy": numpy.__version__, "scipy": scipy.__version__,
            "thread_limits": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "scope": protocol["scope"]}
    atomic_json(args.out / "results.json", report)
    (args.out / "results.sha256").write_text(file_sha256(args.out / "results.json") + "\n")
    print(json.dumps({"points": len(points), "comparison_passed": sum(p["comparison"]["response_comparison_passed"] for p in points),
                      "prediction_summary": report["prediction_summary"], "cost": report["cost"]}, indent=2))


if __name__ == "__main__":
    main()
