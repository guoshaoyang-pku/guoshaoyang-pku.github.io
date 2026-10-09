"""Freeze two simple screening hypotheses before computing new samples."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
ATLAS = ROOT / "runs/v1/compute/ab-lattice-atlas-20-reference-corrected-001/lattice-atlas.json"
E2_MEV_NM = 1439.96454784255
THERMAL_MEV = .08617333262 * 20


def kernel(sample):
    d, top, bottom = .335, sample["gate_top_nm"], sample["gate_bottom_nm"]
    z, height = [bottom - d / 2, bottom + d / 2], top + bottom
    return [[4 * math.pi * E2_MEV_NM / sample["epsilon_r"] *
             min(a, b) * (height - max(a, b)) / height for b in z] for a in z]


def predict(sample, density, displacement, c0, c1=0.):
    v = kernel(sample)
    total = .01 * density
    ext = 335 * displacement / sample["epsilon_r"]
    a, b = v[1][0] - v[0][0], v[1][1] - v[0][1]
    center, coupling = ext + (a + b) * total / 2, (a - b) / 2
    def residual(u):
        p = u * (c0 + c1 * math.log(max(abs(u), THERMAL_MEV) / THERMAL_MEV))
        return u - center - coupling * p
    left, right = -1000., 1000.
    if residual(left) >= 0 or residual(right) <= 0:
        raise ValueError("Prediction root is not bracketed")
    for _ in range(90):
        middle = (left + right) / 2
        if residual(middle) > 0:
            right = middle
        else:
            left = middle
    return (left + right) / 2


def main():
    destination = RUN / "protocol.json"
    if destination.exists():
        raise ValueError("Frozen protocol already exists")
    atlas = json.loads(ATLAS.read_bytes())
    training = [p for p in atlas["points"] if p["controls"]["n_1e12_cm2"] == 0
                and p["controls"]["D_V_nm"] > 0]
    x = [math.log(abs(p["fine_observables"]["internal_U_meV"]) / THERMAL_MEV) for p in training]
    y = [p["fine_observables"]["layer_polarization_nm2"] /
         p["fine_observables"]["internal_U_meV"] for p in training]
    xm, ym = math.fsum(x) / len(x), math.fsum(y) / len(y)
    log_slope = (math.fsum((a - xm) * (b - ym) for a, b in zip(x, y)) /
                 math.fsum((a - xm)**2 for a in x))
    log_intercept = ym - log_slope * xm
    linear_slope = (math.fsum(p["fine_observables"]["internal_U_meV"] *
        p["fine_observables"]["layer_polarization_nm2"] for p in training) /
        math.fsum(p["fine_observables"]["internal_U_meV"]**2 for p in training))
    models = {"linear": {"formula": "P=c0*U", "c0_nm2_inv_meV_inv": linear_slope, "c1": 0.},
              "log": {"formula": "P=U*(c0+c1*log(max(abs(U),kBT)/kBT))",
                      "c0_nm2_inv_meV_inv": log_intercept, "c1": log_slope}}
    base = deepcopy(atlas["records"][1]["configuration"])
    samples = [{"epsilon_r": e, "gate_top_nm": 25., "gate_bottom_nm": 45.}
               for e in (5., 10.)]
    controls = [(0., .15), (0., .25), (.1, .15), (-.1, .25)]
    configurations, heldout = [], []
    for sid, sample in enumerate(samples):
        for density, displacement in controls:
            row = {"sample_id": f"new-sample-{sid}", "sample": sample,
                   "controls": {"n_1e12_cm2": density, "D_V_nm": displacement},
                   "grid_indices": {}, "predicted_internal_U_meV": {name: predict(sample, density, displacement,
                       m["c0_nm2_inv_meV_inv"], m["c1"]) for name, m in models.items()},
                   "test_kind": "new neutral sample and control" if density == 0 else "density extrapolation stress test"}
            for name, grid in (("coarse", {"bz_points": 324, "patch_radial_points": 72, "patch_angular_points": 216}),
                               ("fine", {"bz_points": 486, "patch_radial_points": 108, "patch_angular_points": 324})):
                cfg = deepcopy(base)
                cfg["sample"].update(sample)
                cfg["controls"].update(row["controls"])
                cfg["numerics"].update(grid)
                row["grid_indices"][name] = len(configurations)
                configurations.append(cfg)
            heldout.append(row)
    driver = RUN / "compute-screening.py"
    protocol = {"schema": "ab-screening-prospective/1", "source_sha256": atlas["source_sha256"],
        "training_atlas": str(ATLAS.relative_to(ROOT)), "training_atlas_sha256": hashlib.sha256(ATLAS.read_bytes()).hexdigest(),
        "training_points": [{"sample_id": p["sample_id"], "controls": p["controls"], "fine_key": p["fine_key"]} for p in training],
        "training_scope": "Forty neutral, positive-D points only; no doped data used in fitting",
        "models": models, "thermal_energy_meV": THERMAL_MEV, "heldout": heldout,
        "configurations": configurations, "prediction_tolerance_meV": .5,
        "compute_driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
        "freeze_driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "New-sample surrogate prediction against the same frozen lattice Hamiltonian at B=0,T=20K. Fixed-U compressibility; constant-tau bulk conductivity. No experimental validation, blind discovery, HF phase or independent many-body solver."}
    destination.write_text(json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                      "training_points": len(training), "heldout_points": len(heldout), "models": models}, indent=2))


if __name__ == "__main__":
    main()
