#!/usr/bin/env python3
"""
Re-analysis of Keppel, Weis, Brown & Welch (2005), Journal of Investigative
Psychology and Offender Profiling, 2(1), 1-21.

Reproduces every number, table and figure reported in the manuscript.
Requirements: Python 3, numpy, scipy, matplotlib.  Run:  python3 analysis.py
"""
import csv
import itertools
import json
import math
import os
from datetime import date

import numpy as np
from scipy.stats import beta
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
R = {}

# ---------------------------------------------------------------------------
# A. HITS frequencies printed in Keppel et al. (2005), Tables 2 and 3
# ---------------------------------------------------------------------------
N = 3359


def cp(x, n, a=0.05):
    """Exact (Clopper-Pearson) two-sided confidence interval for x/n."""
    lo = 0.0 if x == 0 else float(beta.ppf(a / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(1 - a / 2, x + 1, n - x))
    return lo, hi


def upper_zero(n, a=0.05):
    """Exact one-sided upper bound when 0 events are seen in n cases (~3/n)."""
    return 1 - a ** (1 / n)


T2 = [("Female victims", 1120, "33%"),
      ("Stabbing or cutting weapon used", 708, "21.1%"),
      ("Stab wounds", 597, "17.8%"),
      ("Body openly displayed", 258, "7.7%"),
      ("Trophies taken", 222, "6.6%"),
      ("Sex-worker victims ('prostitutes')", 114, "3.4%"),
      ("Cutting or incising wounds", 91, "2.7%"),
      ("Unusual body position (staging/posing)", 57, "1.7%"),
      ("Trauma to neck", 60, "1.6%"),
      ("Trauma to abdomen", 52, "1.5%"),
      ("Explored or mutilated cavities", 9, "0.3%"),
      ("Trauma to genitalia", 2, "10.1%")]
T3 = [("Sex worker + unusual body position", 10, "0.27%", ("sex-worker victims", 114)),
      ("Probed, explored or mutilated cavities", 9, "0.26%", None),
      ("Female + mutilation", 6, "0.17%", ("female victims", 1120)),
      ("Unusual body position + mutilation", 2, "0.05%", ("unusual-position cases", 57)),
      ("Sex worker + mutilation", 1, "0.03%", ("sex-worker victims", 114)),
      ("Sex worker + unusual position + mutilation", 0, "0%", ("sex-worker victims", 114))]

R["table2"] = []
for name, x, rep in T2:
    lo, hi = cp(x, N)
    R["table2"].append(dict(item=name, n=x, reported=rep, pct=100 * x / N,
                            lo=100 * lo, hi=100 * hi))
R["table3"] = []
for name, x, rep, cond in T3:
    lo, hi = cp(x, N)
    d = dict(item=name, n=x, reported=rep, pct=100 * x / N, lo=100 * lo, hi=100 * hi)
    if cond:
        lab, m = cond
        clo, chi = cp(x, m)
        d.update(cond_label=lab, cond_n=m, cond_pct=100 * x / m,
                 cond_lo=100 * clo, cond_hi=100 * chi)
    R["table3"].append(d)
R["zero_all_onesided95"] = 100 * upper_zero(N)
R["zero_all_1_in"] = 1 / upper_zero(N)
R["zero_sw_onesided95"] = 100 * upper_zero(114)
R["zero_sw_cp_upper"] = 100 * cp(0, 114)[1]
R["zero_all_cp_upper"] = 100 * cp(0, N)[1]
R["span_days"] = (date(1891, 2, 13) - date(1888, 4, 3)).days
R["span_months"] = R["span_days"] / 30.4375
R["six_span_days"] = (date(1888, 11, 9) - date(1888, 8, 7)).days

# ---------------------------------------------------------------------------
# B. Behavioural coding of the 11 cases (source: case narratives in
#    Keppel et al., 2005, pp. 3-13). 1 = reported present, 0 = not reported.
# ---------------------------------------------------------------------------
cases = ["Smith", "Tabram", "Nichols", "Chapman", "Stride", "Eddowes",
         "Kelly", "Mylett", "McKenzie", "Pinchin St.", "Coles"]
dates = ["1888-04-03", "1888-08-07", "1888-08-31", "1888-09-08", "1888-09-30",
         "1888-09-30", "1888-11-09", "1888-12-20", "1889-07-17", "1889-09-10",
         "1891-02-13"]
beh = [("B1", "Sharp weapon used"),
       ("B2", "Throat cut"),
       ("B3", "Left lying on back"),
       ("B4", "Lower body or abdomen exposed"),
       ("B5", "Three or more sharp-force wounds"),
       ("B6", "Sharp-force injury to abdomen"),
       ("B7", "Abdominal cavity opened"),
       ("B8", "Genital or reproductive-organ injury"),
       ("B9", "Breast injury"),
       ("B10", "Organ(s) removed"),
       ("B11", "Facial mutilation"),
       ("B12", "Objects or body parts arranged"),
       ("B13", "Killed by blunt force or ligature"),
       ("B14", "Body dismembered or moved")]
X0 = np.array([
    # B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11 B12 B13 B14
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # Smith
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],  # Tabram
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # Nichols
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0],  # Chapman
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Stride
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],  # Eddowes
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # Kelly
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Mylett
    [1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # McKenzie
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],  # Pinchin St. (unidentified)
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Coles
], dtype=int)
notes = {
    "Smith": "pp. 3-4. Blunt instrument forced into the vagina, peritoneum ruptured (B8, B13); no knife; died next day.",
    "Tabram": "p. 4. About 39 stab wounds (9 throat, 17 breast, rest lower body); liver, spleen, stomach penetrated; genitals wounded; face-up, clothing arranged to expose lower body. Throat stabbed, not cut (B2=0); abdomen penetrated, not opened (B7=0).",
    "Nichols": "pp. 4-6. Throat cut; abdomen cut open; two stab wounds to genitals; on back, clothing pushed up; no viscera missing.",
    "Chapman": "pp. 6-8. Throat cut; abdomen opened, intestines and skin flaps placed above the shoulders; uterus, upper vagina, most of bladder missing; pocket contents arranged at feet; on back (B4 implied by evisceration).",
    "Stride": "pp. 8-9. Single incision to the throat; found on left side; clothing undisturbed; no other injury.",
    "Eddowes": "pp. 9-10. Throat cut; face mutilated; abdomen opened; intestines over right shoulder, a piece placed by left arm; left kidney and uterus removed; on back, clothing raised.",
    "Kelly": "pp. 10-11. Throat cut; face mutilated; breasts removed; abdomen emptied; organs placed around body; heart absent; garments arranged; on back.",
    "Mylett": "p. 11. Strangled with a cord; no wounds; on side; clothing not disarranged.",
    "McKenzie": "p. 12. Throat cut; jagged incision chest to below navel (cavity not opened, B7=0); cuts to mons veneris; on right side, clothing pulled up to waist. Eye injury mentioned but coded 0 (conservative).",
    "Pinchin St.": "pp. 12-13. Torso; head and legs removed; killed elsewhere; wound from sternum penetrating bowels and uterus; face down; throat not cut per original. Genital involvement judged accidental by original (B8=0).",
    "Coles": "p. 13. Two cuts to the throat; no other mutilation; position not recorded.",
}
I = {c: i for i, c in enumerate(cases)}
K6 = [I[c] for c in ["Tabram", "Nichols", "Chapman", "Stride", "Eddowes", "Kelly"]]
C5 = [I[c] for c in ["Nichols", "Chapman", "Stride", "Eddowes", "Kelly"]]
ALT6 = [I[c] for c in ["Tabram", "Nichols", "Chapman", "McKenzie", "Eddowes", "Kelly"]]
COLES6 = [I[c] for c in ["Tabram", "Nichols", "Chapman", "Coles", "Eddowes", "Kelly"]]


def jac(X, w=None):
    n, m = X.shape
    w = np.ones(m) if w is None else np.asarray(w, float)
    J = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            both = (X[i] == 1) & (X[j] == 1)
            either = (X[i] == 1) | (X[j] == 1)
            den = float((w * either).sum())
            J[i, j] = J[j, i] = float((w * both).sum()) / den if den > 0 else 0.0
    return J


def coh(J, idx):
    return float(np.mean([J[a, b] for a, b in itertools.combinations(idx, 2)]))


def allsets(J, k):
    return [(c, coh(J, c)) for c in itertools.combinations(range(J.shape[0]), k)]


def ranking(J, idx):
    vals = allsets(J, len(idx))
    arr = np.array([v for _, v in vals])
    t = coh(J, idx)
    gt = int((arr > t + 1e-9).sum())
    ge = int((arr >= t - 1e-9).sum())
    best = max(vals, key=lambda z: z[1])
    return dict(cohesion=t, n_sets=len(vals), n_greater=gt, rank=gt + 1,
                n_ge=ge, p_ge=ge / len(vals),
                best_set=[cases[i] for i in best[0]], best_cohesion=best[1])


def best_k(J, k):
    c, v = max(allsets(J, k), key=lambda z: z[1])
    return list(c), v


def to_core(J, core):
    out = {}
    for i, c in enumerate(cases):
        others = [j for j in core if j != i]
        out[c] = float(np.mean([J[i, j] for j in others]))
    return out


# Baseline
J0 = jac(X0)
CORE, core_coh = best_k(J0, 4)
R["core"] = [cases[i] for i in CORE]
R["core_cohesion"] = core_coh
R["core_pairs"] = {f"{cases[a]}-{cases[b]}": J0[a, b] for a, b in itertools.combinations(CORE, 2)}
R["k_per_behaviour"] = {b[0]: int(X0[:, j].sum()) for j, b in enumerate(beh)}
R["k_per_case"] = {c: int(X0[i].sum()) for i, c in enumerate(cases)}
R["J"] = {cases[i]: {cases[j]: round(J0[i, j], 4) for j in range(11)} for i in range(11)}
key_pairs = [("Stride", "Coles"), ("Tabram", "McKenzie"), ("Nichols", "McKenzie"),
             ("Nichols", "Chapman"), ("Stride", "McKenzie"), ("Stride", "Tabram"),
             ("Smith", "Mylett"), ("Stride", "Eddowes"), ("Stride", "Nichols"),
             ("Tabram", "Nichols"), ("Pinchin St.", "Nichols")]
R["key_pairs"] = {f"{a}-{b}": J0[I[a], I[b]] for a, b in key_pairs}
R["shared_with_core"] = {c: {cases[j]: int(((X0[I[c]] == 1) & (X0[j] == 1)).sum()) for j in CORE}
                         for c in ["Stride", "McKenzie", "Tabram", "Coles", "Pinchin St."]}

w_rare = np.array([math.log(11 / max(1, X0[:, j].sum())) for j in range(X0.shape[1])])
X_kf = X0.copy()                       # 'Keppel-favourable' coding
X_kf[I["Stride"], 2] = 1                # B3 supine (as asserted on p. 16)
X_kf[I["Stride"], 3] = 1                # B4 exposed (as asserted on p. 16)
X_kf[I["Tabram"], 1] = 1                # B2 throat attacked counted as throat cut
X_kf[I["Tabram"], 6] = 1                # B7 stab wounds into abdomen counted as opened
X_lit = X0.copy()                      # 'literal' coding
X_lit[I["McKenzie"], 10] = 1            # B11 eyes mutilated (p. 12)
X_lit[I["Pinchin St."], 7] = 1          # B8 wound extended to labia (p. 13)
scen = {"baseline": (X0, None), "keppel_favourable": (X_kf, None),
        "literal": (X_lit, None), "rarity_weighted": (X0, w_rare)}
R["scenarios"] = {}
for name, (X, w) in scen.items():
    J = jac(X, w)
    b4, b4v = best_k(J, 4)
    R["scenarios"][name] = dict(
        to_core=to_core(J, CORE),
        keppel6=ranking(J, K6),
        canonical5=ranking(J, C5),
        core4=ranking(J, CORE),
        alt6_cohesion=coh(J, ALT6),
        coles6_cohesion=coh(J, COLES6),
        best4=[cases[i] for i in b4], best4_cohesion=b4v,
        mckenzie_gt_stride=to_core(J, CORE)["McKenzie"] > to_core(J, CORE)["Stride"])

# Leave-one-behaviour-out
lofo = []
for f in range(X0.shape[1]):
    X = np.delete(X0, f, axis=1)
    J = jac(X)
    b4, _ = best_k(J, 4)
    tc = to_core(J, CORE)
    lofo.append(dict(dropped=beh[f][0], core_still_best4=sorted(b4) == sorted(CORE),
                     mck_gt_stride=tc["McKenzie"] > tc["Stride"],
                     keppel6_rank=ranking(J, K6)["rank"], canon5_rank=ranking(J, C5)["rank"],
                     best6=ranking(J, K6)["best_set"], to_core=tc))
R["lofo"] = lofo
R["lofo_summary"] = dict(
    core_always_best4=all(d["core_still_best4"] for d in lofo),
    mck_always_gt_stride=all(d["mck_gt_stride"] for d in lofo),
    keppel6_rank_range=[min(d["keppel6_rank"] for d in lofo), max(d["keppel6_rank"] for d in lofo)],
    canon5_rank_range=[min(d["canon5_rank"] for d in lofo), max(d["canon5_rank"] for d in lofo)],
    best6_always_alt=all(sorted(d["best6"]) == sorted(cases[i] for i in ALT6) for d in lofo),
    to_core_range={c: [min(d["to_core"][c] for d in lofo), max(d["to_core"][c] for d in lofo)] for c in cases})

# Top six-case sets (baseline)
six = sorted(allsets(J0, 6), key=lambda z: -z[1])
R["top_six_sets"] = [dict(rank=r + 1, members=[cases[i] for i in c], cohesion=v)
                     for r, (c, v) in enumerate(six[:12])]

# Clustering
Z = linkage(squareform(1 - J0, checks=False), method="average")
R["linkage"] = [[int(a), int(b), float(h), int(n)] for a, b, h, n in Z]

# ---------------------------------------------------------------------------
# C. Likelihood-ratio illustration
# ---------------------------------------------------------------------------
ps = [("HITS, all homicides (9/3,359)", 9 / N),
      ("Plausible upper value, sex-worker victims", 0.03),
      ("Within the 11-case file (5/11)", 5 / 11)]
cs = [0.5, 0.8, 0.95]
R["lr"] = [dict(label=l, p=p, both=[c / p for c in cs], one_missing=[(1 - c) / (1 - p) for c in cs])
           for l, p in ps]
p0, c0 = 9 / N, 0.8
R["lr_interrupt"] = {str(q): (q + (1 - q) * (1 - c0)) / (1 - p0) for q in (0.5, 0.9)}
R["lr_throat_local"] = 0.95 / (7 / 11)
R["lr_absent_max"] = 1 / (1 - p0)

# ---------------------------------------------------------------------------
# D. Figures
# ---------------------------------------------------------------------------
from decimal import Decimal, ROUND_HALF_UP


def fmt2(v):  # conventional (half-up) rounding, as in the tables
    return str(Decimal(repr(round(float(v), 9))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
dn = dendrogram(Z, no_plot=True, labels=cases)
order = [I[c] for c in dn["ivl"]]

fig, ax = plt.subplots(figsize=(6.4, 5.4))
M = J0[np.ix_(order, order)]
im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1)
ax.set_xticks(range(11)); ax.set_yticks(range(11))
ax.set_xticklabels([cases[i] for i in order], rotation=45, ha="right")
ax.set_yticklabels([cases[i] for i in order])
for a in range(11):
    for b in range(11):
        v = M[a, b]
        ax.text(b, a, fmt2(v).lstrip("0") if v < 1 else "1", ha="center", va="center",
                fontsize=7, color="white" if v > 0.6 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Jaccard similarity")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Figure1_similarity.png"), dpi=300)
fig.savefig(os.path.join(OUT, "Figure1_similarity.pdf"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 3.6))
dendrogram(Z, labels=cases, orientation="right", color_threshold=0,
           above_threshold_color="black", ax=ax, leaf_font_size=9)
ax.set_xlabel("Distance (1 \u2212 Jaccard), average linkage")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Figure2_dendrogram.png"), dpi=300)
fig.savefig(os.path.join(OUT, "Figure2_dendrogram.pdf"))
plt.close(fig)

vals = np.array([v for _, v in allsets(J0, 6)])
k6 = coh(J0, K6); a6 = coh(J0, ALT6)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.hist(vals, bins=30, color="#c8c8c8", edgecolor="white")
ax.axvline(k6, color="black", ls="--", lw=1.2)
ax.axvline(a6, color="black", ls="-", lw=1.4)
ymax = ax.get_ylim()[1]
ax.text(k6 - 0.01, ymax * 0.92, "Keppel et al.'s six", ha="right", fontsize=8)
ax.text(a6 - 0.01, ymax * 0.72, "McKenzie in place\nof Stride", ha="right", fontsize=8, color="#333333")
ax.set_xlabel("Group cohesion (mean pairwise Jaccard similarity)")
ax.set_ylabel("Number of six-case sets")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Figure3_randomisation.png"), dpi=300)
fig.savefig(os.path.join(OUT, "Figure3_randomisation.pdf"))
plt.close(fig)

# ---------------------------------------------------------------------------
# E. Save
# ---------------------------------------------------------------------------
with open(os.path.join(OUT, "coding_sheet.csv"), "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["case", "date"] + [f"{b[0]} {b[1]}" for b in beh] + ["coding note (page refs to Keppel et al., 2005)"])
    for i, c in enumerate(cases):
        wr.writerow([c, dates[i]] + list(X0[i]) + [notes[c]])


def conv(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=conv)

# Compact console summary
print("span days/months:", R["span_days"], round(R["span_months"], 1), "| six span:", R["six_span_days"])
for d in R["table2"]:
    print(f"T2 {d['item'][:38]:38s} n={d['n']:5d} rep={d['reported']:>6s} pct={d['pct']:.2f} CI=[{d['lo']:.2f},{d['hi']:.2f}]")
for d in R["table3"]:
    s = f"T3 {d['item'][:42]:42s} n={d['n']} rep={d['reported']:>5s} pct={d['pct']:.3f} CI=[{d['lo']:.3f},{d['hi']:.3f}]"
    if "cond_n" in d:
        s += f" | cond {d['cond_n']}: {d['cond_pct']:.2f} [{d['cond_lo']:.2f},{d['cond_hi']:.2f}]"
    print(s)
print("zero: all 1-sided", round(R["zero_all_onesided95"], 3), "1 in", round(R["zero_all_1_in"]), "CP", round(R["zero_all_cp_upper"], 3),
      "| sw 1-sided", round(R["zero_sw_onesided95"], 2), "CP", round(R["zero_sw_cp_upper"], 2))
print("core:", R["core"], round(core_coh, 3), {k: round(v, 3) for k, v in R["core_pairs"].items()})
print("k per behaviour:", R["k_per_behaviour"])
print("key pairs:", {k: round(v, 3) for k, v in R["key_pairs"].items()})
print("shared with core:", R["shared_with_core"])
for name, d in R["scenarios"].items():
    print(f"\n[{name}] to_core:", {k: round(v, 3) for k, v in d["to_core"].items()})
    for g in ("keppel6", "canonical5", "core4"):
        x = d[g]
        print(f"  {g}: coh={x['cohesion']:.3f} rank={x['rank']}/{x['n_sets']} n_ge={x['n_ge']} p_ge={x['p_ge']:.3f} best={x['best_set']} {x['best_cohesion']:.3f}")
    print(f"  alt6={d['alt6_cohesion']:.3f} coles6={d['coles6_cohesion']:.3f} best4={d['best4']} {d['best4_cohesion']:.3f} McK>Stride={d['mckenzie_gt_stride']}")
print("\nLOFO:", {k: v for k, v in R["lofo_summary"].items() if k != "to_core_range"})
print("LOFO to_core ranges:", {k: [round(a, 3), round(b, 3)] for k, (a, b) in R["lofo_summary"]["to_core_range"].items()})
print("top six sets:")
for t in R["top_six_sets"]:
    print("  ", t["rank"], round(t["cohesion"], 3), t["members"])
print("linkage:", [[cases[a] if a < 11 else a, cases[b] if b < 11 else b, round(h, 3)] for a, b, h, n in R["linkage"]])
for d in R["lr"]:
    print("LR", d["label"], "p=", round(d["p"], 4), "both", [round(v, 1) for v in d["both"]], "missing", [round(v, 3) for v in d["one_missing"]])
print("interrupt:", {k: round(v, 3) for k, v in R["lr_interrupt"].items()}, "throat local:", round(R["lr_throat_local"], 2), "absent max:", round(R["lr_absent_max"], 4))
print("weights:", [round(x, 3) for x in w_rare])
