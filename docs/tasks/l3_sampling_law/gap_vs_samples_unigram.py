#!/usr/bin/env python3
"""gap_vs_samples_unigram.py

Most-basic N-Gram Gap problem: one context row, true conditional P(·|c),
r iid training draws n ~ Multinomial(r, P).  A unigram/count table fits the
empirical distribution phat = n/r (or its smoothed version q) and is
evaluated on (a) the training empirical distribution and (b) the true P
(test = ground truth).

Question: how does gap = L_val - L_train depend on the number of training
draws r?

Answer (two regimes, both derivable):

  EXACT IDENTITY (any q, term-by-term):
      L_val(q) - L_train(q) = sum_c (phat_c - P_c) log q_c
  With q = phat (count table, full-batch converged):
      gap = KL(P||phat) + H(P) - H(phat)

  (i) RESOLVED REGIME (r P_c >> 1 for all c in support):
      E[L_val]   = H + (K-1)/(2 r) + O(r^-2)    (KL bias, Miller-Madow)
      E[L_train] = H - (K-1)/(2 r) + O(r^-2)    (entropy bias)
      E[gap]     = (K-1)/r + O(r^-2)            (H cancels exactly)
      -> log-log slope of E[gap] vs r tends to -1 asymptotically.
      Constant = support-1 = K-1, NOT exp(H)-1 in general
      (they coincide only for uniform P).

  (ii) UNRESOLVED REGIME (r P_c ~< 1 for some cells; many symbols unseen):
      the gap is dominated by the unseen-symbol penalty
          -sum_{c: n_c=0} P_c log q_c
      For the pure MLE table (q = phat) this is +inf.  With smoothing a, an
      unseen cell has raw validation penalty P_c * log((r + a K) / a).
      A centered unseen-mass approximation can be useful empirically, but is
      not an exact total-gap formula.  The result also depends on the seen-cell
      remainder, not purely on r.

Pure numpy + matplotlib, deterministic, no torch, no external data.

Outputs: docs/figs/theory/fig_gap_vs_samples_{bc11,exact,unresolved,longtail,realgen}.svg/.png
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__import__("os").environ.get(
    "NGLAB_FIG_DIR", REPO_ROOT / "docs" / "figs" / "theory"
))
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260812)

# ---------------------------------------------------------------- helpers
def exact_k2(r: int, P=(0.5, 0.5), a: float = 1e-3):
    """Exact E[L_val], E[L_train], E[gap] for K=2 via the binomial sum.

    q_c = (n_c + a)/(r + 2a);  phat_c = n_c/r.
    gap = [H(P)+KL(P||q)] - [H(phat)+KL(phat||q)]  (exact for any q).
    """
    P = np.asarray(P, float)
    H = -float(np.sum(P * np.log(P)))
    E_lv = 0.0; E_lt = 0.0
    for n0 in range(r + 1):
        w = math.comb(r, n0) * P[0] ** n0 * P[1] ** (r - n0)
        n = np.array([n0, r - n0], float)
        ph = n / r
        q = (n + a) / (r + 2 * a)
        lv = -float(np.sum(P * np.log(q)))
        lt = -float(np.sum(np.where(ph > 0, ph * np.log(q), 0.0)))
        E_lv += w * lv; E_lt += w * lt
    return H, E_lv, E_lt, E_lv - E_lt

def mc_gap(P: np.ndarray, r: int, n_mc: int, a: float = 1e-9):
    """MC E[gap] for arbitrary K.  gap = sum (phat - P) log q (exact identity)."""
    K = len(P)
    counts = rng.multinomial(r, P, size=n_mc)
    q = (counts + a) / (r + a * K)
    ph = counts / r
    gap = np.sum((ph - P) * np.log(q), axis=1)
    return float(gap.mean()), float(gap.std(ddof=1) / np.sqrt(n_mc))

# ---------------------------------------------------------------- section A
print("[A] exact identity: gap = sum_c (phat_c - P_c) log q_c  (any q)")
P = np.array([0.7, 0.2, 0.1])
for q in ([0.5, 0.3, 0.2], [0.8, 0.15, 0.05], [0.6, 0.3, 0.1]):
    q = np.asarray(q, float)
    gap_direct = -float(np.sum(P * np.log(q))) - (-float(np.sum(q * np.log(q))))
    gap_ident = float(np.sum((q - P) * np.log(q)))
    assert abs(gap_direct - gap_ident) < 1e-12
print("  identity holds exactly for all q (asserted)")

# ---------------------------------------------------------------- section B
print("\n[B] user example: P(B)=P(C)=1/2 (K=2), exact closed form")
H2 = math.log(2.0)
rows_b = []
print("  r     E[gap]    1/r      E[Ltr]-H   -1/2r      E[Lval]-H  +1/2r")
for r in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    _, E_lv, E_lt, E_gap = exact_k2(r)
    rows_b.append((r, E_gap, E_lv, E_lt))
    print(f"  {r:4d}  {E_gap:+.6f}  {1/r:.6f}  {E_lt-H2:+.6f}  {-1/(2*r):+.6f}  "
          f"{E_lv-H2:+.6f}  {1/(2*r):+.6f}")

# ---------------------------------------------------------------- section C
print("\n[C] uniform K=8: MC E[gap] vs r (resolved regime ~> r=64)")
K8 = 8
P8 = np.ones(K8) / K8
H8 = math.log(K8)
rs_c = [16, 32, 64, 128, 256, 512, 1024, 2048]
rows_c = []
for r in rs_c:
    Eg, se = mc_gap(P8, r, n_mc=600_000)
    rows_c.append((r, Eg, se))
    print(f"  r={r:5d}  E[gap]={Eg:.5f} +- {se:.5f}  (K-1)/r={(K8-1)/r:.5f}")
xs = np.log(np.array([x[0] for x in rows_c if x[0] >= 64]))
ys = np.log(np.array([x[1] for x in rows_c if x[0] >= 64]))
slope, intercept = np.polyfit(xs, ys, 1)
print(f"  log-log slope (resolved r>=64)={slope:.3f} (expect -1), "
      f"exp(intercept)={math.exp(intercept):.3f} (expect K-1={K8-1})")

# ---------------------------------------------------------------- section D
print("\n[D] non-uniform P=[.7,.2,.1] K=3: constant = K-1, not exp(H)-1")
P3 = np.array([0.7, 0.2, 0.1])
H3 = -np.sum(P3 * np.log(P3))
K3 = len(P3)
print(f"  K-1={K3-1}  exp(H)-1={math.exp(H3)-1:.4f}")
for r in [256, 512, 1024, 2048, 4096]:
    Eg, se = mc_gap(P3, r, n_mc=1_000_000)
    print(f"  r={r:5d}  E[gap]={Eg:.6f} +- {se:.6f}  (K-1)/r={(K3-1)/r:.6f}  "
          f"(expH-1)/r={(math.exp(H3)-1)/r:.6f}")

# ---------------------------------------------------------------- section E
print("\n[E] unresolved regime: unseen-symbol penalty dominates, a-dependent")
# K=2, r small, several smoothing values: gap >> 1/r and depends on a
print("  K=2 exact: r     a=1e-6    a=1e-3    a=0.1     (1/r)")
for r in [1, 2, 4, 8, 16, 32]:
    vals = [exact_k2(r, a=a)[3] for a in (1e-6, 1e-3, 0.1)]
    print(f"       {r:4d}  {vals[0]:8.4f}  {vals[1]:8.4f}  {vals[2]:8.4f}  {1/r:.4f}")

# ---------------------------------------------------------------- figures
PAPER, PANEL, INK, MUTED, BORDER = "#f7f5ef", "#fffdf8", "#232426", "#686d73", "#c8c1b6"
plt.rcParams.update({
    "font.size": 9.5,
    "figure.facecolor": PAPER,
    "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.titleweight": "bold",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": BORDER,
    "grid.alpha": 0.35,
    "axes.grid": True,
})
C_BLUE, C_RED, C_TH = "#2d6f9f", "#c4493d", INK

# fig 1: user example B:C = 1:1 (K=2)
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4))
ax = axes[0]
rr = np.array([x[0] for x in rows_b]); gg = np.array([x[1] for x in rows_b])
ax.semilogx(rr, gg, "o-", color=C_BLUE, lw=1.6, ms=5, label=r"exact $E[\mathrm{gap}]$")
ax.semilogx(rr, 1.0 / rr, "--", color=C_TH, lw=1.4, label=r"$(K-1)/r = 1/r$")
ax.set_xlabel(r"training draws per context $r$ (log)")
ax.set_ylabel(r"$E[\mathrm{gap}]$ (nats)")
ax.set_title(r"(a) B:C $=1:1$ (K=2): $E[\mathrm{gap}] = 1/r + O(r^{-2})$")
ax.legend(fontsize=8, frameon=False)
ax = axes[1]
E_lv = np.array([x[2] for x in rows_b]); E_lt = np.array([x[3] for x in rows_b])
ax.semilogx(rr, E_lv - H2, "o-", color=C_RED, lw=1.6, ms=5,
            label=r"$E[L_{\mathrm{val}}]-H$")
ax.semilogx(rr, E_lt - H2, "o-", color=C_BLUE, lw=1.6, ms=5,
            label=r"$E[L_{\mathrm{train}}]-H$")
ax.semilogx(rr, 0.5 / rr, "--", color="0.6", lw=1.2, label=r"$+1/2r$")
ax.semilogx(rr, -0.5 / rr, "--", color="0.6", lw=1.2, label=r"$-1/2r$")
ax.axhline(0, color="0.8", lw=0.8)
ax.set_xlabel(r"training draws $r$ (log)")
ax.set_ylabel(r"$E[L]-H$ (nats)")
ax.set_title(r"(b) symmetric split $\pm (K-1)/2r$ around $H=\log 2$")
ax.legend(fontsize=8, frameon=False)
fig.suptitle("Most-basic unigram gap: one row, r iid draws, count table", y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "fig_gap_vs_samples_bc11.svg")
fig.savefig(OUT / "fig_gap_vs_samples_bc11.png", dpi=150)

# fig 2: uniform K=8 resolved regime + slope
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4))
ax = axes[0]
rr = np.array([x[0] for x in rows_c]); gg = np.array([x[1] for x in rows_c])
ax.loglog(rr, gg, "o-", color=C_BLUE, lw=1.6, ms=5, label=r"MC $E[\mathrm{gap}]$")
ax.loglog(rr, (K8 - 1) / rr, "--", color=C_TH, lw=1.4, label=r"$(K-1)/r$")
ax.set_xlabel(r"training draws $r$ (log)")
ax.set_ylabel(r"$E[\mathrm{gap}]$ (nats, log)")
ax.set_title(f"(a) uniform K=8 (resolved): slope={slope:.3f}, "
             f"scale={math.exp(intercept):.2f}~K-1")
ax.legend(fontsize=8, frameon=False)
ax = axes[1]
# exact K=2 symmetric split as the canonical picture (K=8 MC too noisy for the split)
rr2 = np.array([x[0] for x in rows_b])
lv2 = np.array([x[2] for x in rows_b]); lt2 = np.array([x[3] for x in rows_b])
ax.semilogx(rr2, lv2 - H2, "s-", color=C_RED, lw=1.5, ms=4, label=r"$E[L_{\mathrm{val}}]-H$")
ax.semilogx(rr2, lt2 - H2, "o-", color=C_BLUE, lw=1.5, ms=4, label=r"$E[L_{\mathrm{train}}]-H$")
ax.semilogx(rr2, 0.5 / rr2, "--", color="0.6", lw=1.2, label=r"$\pm1/2r$")
ax.axhline(0, color="0.8", lw=0.8)
ax.set_xlabel(r"training draws $r$ (log)")
ax.set_ylabel(r"$E[L]-H$ (nats)")
ax.set_title(r"(b) H cancels in the gap: train/val split $\pm (K-1)/2r$")
ax.legend(fontsize=8, frameon=False)
fig.suptitle("Resolved regime: exact 1/r law, log-log slope -1", y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "fig_gap_vs_samples_exact.svg")
fig.savefig(OUT / "fig_gap_vs_samples_exact.png", dpi=150)

# fig 3: unresolved regime -- unseen-symbol penalty
fig, ax = plt.subplots(figsize=(6.6, 4.4))
for a, c, mk in ((1e-6, C_BLUE, "o"), (1e-3, C_RED, "s"), (0.1, "#55a868", "^")):
    vals = [exact_k2(r, a=a)[3] for r in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]]
    rr3 = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    ax.semilogx(rr3, vals, f"{mk}-", color=c, lw=1.4, ms=4,
                label=rf"K=2 exact, $\alpha={a:.0e}$")
ax.semilogx(rr3, 1.0 / rr3, "--", color=C_TH, lw=1.4, label=r"$(K-1)/r$")
ax.set_xlabel(r"training draws $r$ (log)")
ax.set_ylabel(r"$E[\mathrm{gap}]$ (nats)")
ax.set_title(r"Unresolved regime: exact gap depends on smoothing $\alpha$"
             "\n" + r"(the unseen-cell term is not a universal total-gap law)")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_gap_vs_samples_unresolved.svg")
fig.savefig(OUT / "fig_gap_vs_samples_unresolved.png", dpi=150)

# fig 4: long tail -- resolved support
print("\n[F] long-tailed Zipf K=128: leading constant = resolved support")
def zipf(K: int, s: float = 1.0) -> np.ndarray:
    z = 1.0 / np.arange(1, K + 1) ** s
    return z / z.sum()

KZ = 128
PZ = zipf(KZ)
HZ = -np.sum(PZ * np.log(PZ))
print(f"  Zipf K={KZ}, H={HZ:.3f}, exp(H)-1={math.exp(HZ)-1:.1f}, K-1={KZ-1}")
longtail = []
for r in [256, 1024, 4096, 16384, 65536]:
    Eg, se = mc_gap(PZ, r, n_mc=300_000)
    Kres = int(np.sum(PZ > 1.0 / r))
    longtail.append((r, Eg, se, Kres))
    print(f"  r={r:6d}  E[gap]={Eg:.5f} +- {se:.5f}  (K-1)/r={(KZ-1)/r:.5f}  "
          f"resolved K_r={Kres:3d}  (K_r-1)/r={(Kres-1)/r:.5f}")
fig, ax = plt.subplots(figsize=(6.6, 4.4))
rr = np.array([x[0] for x in longtail]); gg = np.array([x[1] for x in longtail])
kr = np.array([x[3] for x in longtail])
ax.loglog(rr, gg, "o-", color=C_BLUE, lw=1.6, ms=5, label=r"MC $E[\mathrm{gap}]$ (Zipf-128)")
ax.loglog(rr, (KZ - 1) / rr, "--", color="0.6", lw=1.2, label=r"$(K-1)/r$ (full support)")
ax.loglog(rr, np.maximum(kr - 1, 1) / rr, "--", color=C_RED, lw=1.4,
          label=r"$(K_r-1)/r$ (resolved support)")
ax.set_xlabel(r"training draws $r$ (log)")
ax.set_ylabel(r"$E[\mathrm{gap}]$ (nats, log)")
ax.set_title("Long tail (Zipf K=128): leading constant is the resolved support")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_gap_vs_samples_longtail.svg")
fig.savefig(OUT / "fig_gap_vs_samples_longtail.png", dpi=150)

# ---------------------------------------------------------------- section G
print("\n[G] REAL generator (vocab 8192, 8 private tokens + Zipf-1.05 tail over ~7900):")
print("    E[gap] is NOT (K-1)/r; compare it with a centered unseen-mass heuristic.")
VOCAB = 8192; HUB = 256; SEP = VOCAB - 1
_idc = np.arange(1, SEP - HUB, dtype=float)
_base = _idc ** -1.05
_base = _base / _base.sum()

def _real_dist(seed: int) -> np.ndarray:
    rr = random.Random(seed)
    support = rr.sample(range(HUB + 1, SEP), 8)
    sw = np.array([1.0 / (i + 1) for i in range(8)])
    sw = sw / sw.sum()
    w = 0.10 * _base.copy()
    for tok, weight in zip(support, sw):
        w[tok - HUB - 1] += 0.90 * weight
    return w / w.sum()

print("  generating 300 real-distribution contexts ...")
real_ctx = [_real_dist(i) for i in range(300)]
real_rs = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
real_rows = []
for a in (1e-6, 1e-3, 1e-1):
    gm = []
    for r in real_rs:
        g = []
        for Pi in real_ctx:
            c = rng.multinomial(r, Pi)
            q = (c + a) / (r + a * VOCAB)
            g.append(-np.sum(Pi * np.log(q)) + np.sum((c / r) * np.log(q)))
        gm.append(float(np.mean(g)))
    sl = float(np.polyfit(np.log(real_rs), np.log(gm), 1)[0])
    print(f"  a={a:.0e}: E[gap](r=128)={gm[0]:.4f}  E[gap](r=32768)={gm[-1]:.5f}  "
          f"log-log slope={sl:.3f}  (expect ~-0.2, NOT -1)")
    real_rows.append((a, gm, sl))

# predicted: U(r)*log(r/(aK)) with U(r)=sum P_c(1-P_c)^r
a_ref = 1e-6
pred_rows = []
for r in real_rs:
    U = float(np.mean([np.sum(Pi * (1 - Pi) ** r) for Pi in real_ctx]))
    pred_rows.append((r, U, U * math.log(r / (a_ref * VOCAB))))
print("  compare gap with U(r)*log(r/(aK)) heuristic  [a=1e-6]:")
g_ref = real_rows[0][1]
for i, r in enumerate(real_rs):
    U, pred = pred_rows[i][1], pred_rows[i][2]
    print(f"    r={r:6d}  E[gap]={g_ref[i]:.4f}  U(r)={U:.5f}  "
          f"U*log(r/(aK))={pred:.4f}  ratio={g_ref[i]/pred:.3f}")

# real-distribution figure
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4))
ax = axes[0]
for a, gm, sl in real_rows:
    ax.loglog(real_rs, gm, "o-", lw=1.5, ms=4, label=rf"$E[\mathrm{{gap}}]$, $\alpha={a:.0e}$ (slope {sl:.2f})")
ax.loglog(real_rs, np.array([(8 - 1) / r for r in real_rs]), "--", color=C_TH,
          lw=1.4, label=r"$(K_{\mathrm{priv}}-1)/r = 7/r$ (if only private)")
ax.loglog(real_rs, np.array([13.03 / r for r in real_rs]), ":", color="0.5",
          lw=1.3, label=r"$(\exp(H)-1)/r \approx 12/r$ (uniform-ish)")
ax.set_xlabel(r"training draws per context $r$ (log)")
ax.set_ylabel(r"$E[\mathrm{gap}]$ (nats, log)")
ax.set_title("(a) real generator: gap is much larger and shallower than 1/r")
ax.legend(fontsize=8, frameon=False)
ax = axes[1]
rrG = np.array(real_rs)
ax.loglog(rrG, [pred_rows[i][1] for i in range(len(rrG))], "o-", color=C_BLUE,
          lw=1.6, ms=5, label=r"$U(r)=\mathbb{E}\sum_c P_c(1-P_c)^r$")
ax.loglog(rrG, [pred_rows[i][2] for i in range(len(rrG))], "s--", color=C_RED,
          lw=1.5, ms=4, label=r"$U(r)\log(r/(\alpha K))$ (centered heuristic)")
ax.loglog(rrG, real_rows[0][1], "^-", color="0.4", lw=1.4, ms=4,
          label=r"$E[\mathrm{gap}]$ ($\alpha=10^{-6}$)")
ax.set_xlabel(r"training draws $r$ (log)")
ax.set_ylabel(r"(nats, log)")
ax.set_title(r"(b) unseen-mass heuristic tracks this finite synthetic range")
ax.legend(fontsize=8, frameon=False)
fig.suptitle("Real sparse distribution (vocab 8192, 8 private + long tail): "
             "unresolved tail dominates, slope ~ -0.2 not -1", y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "fig_gap_vs_samples_realgen.svg")
fig.savefig(OUT / "fig_gap_vs_samples_realgen.png", dpi=150)

print("\nfigures ->", OUT)
print("DONE")
