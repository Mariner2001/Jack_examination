#!/usr/bin/env python3
"""Independent re-check of every number in results.json, using different code paths
(scipy.stats.binomtest for intervals, exact fractions for Jaccard and cohesion)."""
import csv, itertools, json, math, subprocess, sys
from datetime import date
from fractions import Fraction as F
from scipy.stats import binomtest

R = json.load(open("out/results.json"))
N = 3359
problems = []
ok = lambda c, msg: None if c else problems.append(msg)

# 1. Percentages and exact intervals
for d in R["table2"] + R["table3"]:
    ci = binomtest(d["n"], N).proportion_ci(confidence_level=0.95, method="exact")
    ok(abs(100 * d["n"] / N - d["pct"]) < 1e-9, f"pct {d['item']}")
    ok(abs(100 * ci.low - d["lo"]) < 1e-6 and abs(100 * ci.high - d["hi"]) < 1e-6, f"CI {d['item']}")
    if "cond_n" in d:
        c2 = binomtest(d["n"], d["cond_n"]).proportion_ci(confidence_level=0.95, method="exact")
        ok(abs(100 * c2.low - d["cond_lo"]) < 1e-6 and abs(100 * c2.high - d["cond_hi"]) < 1e-6, f"condCI {d['item']}")
        ok(abs(100 * d["n"] / d["cond_n"] - d["cond_pct"]) < 1e-9, f"condpct {d['item']}")
ok(abs(100 * (1 - 0.05 ** (1 / N)) - R["zero_all_onesided95"]) < 1e-9, "zero all")
ok(abs(100 * (1 - 0.05 ** (1 / 114)) - R["zero_sw_onesided95"]) < 1e-9, "zero sw")
ok((date(1891, 2, 13) - date(1888, 4, 3)).days == R["span_days"] == 1046, "span")
ok((date(1888, 11, 9) - date(1888, 8, 7)).days == R["six_span_days"] == 94, "six span")
print("Reported-vs-correct (original tables):")
for d in R["table2"] + R["table3"]:
    print(f"  {d['item'][:44]:44s} n={d['n']:5d} reported={d['reported']:>6s} correct={d['pct']:.3f}%")

# 2. Jaccard with exact fractions, from the coding sheet
rows = list(csv.reader(open("out/coding_sheet.csv")))[1:]
cases = [r[0] for r in rows]
sets = {r[0]: {j for j, v in enumerate(r[2:16]) if v == "1"} for r in rows}
def J(a, b):
    u = sets[a] | sets[b]
    return F(len(sets[a] & sets[b]), len(u)) if u else F(0)
for a in cases:
    for b in cases:
        ok(abs(float(J(a, b)) - R["J"][a][b]) < 1e-4, f"J {a}-{b}")
coh = lambda g: sum(J(a, b) for a, b in itertools.combinations(g, 2)) / F(len(g) * (len(g) - 1), 2)

def rank(target):
    allv = sorted((coh(g) for g in itertools.combinations(cases, len(target))), reverse=True)
    t = coh(target)
    return sum(v > t for v in allv) + 1, sum(v >= t for v in allv), len(allv), float(t), [float(v) for v in allv[:3]]
K6 = ["Tabram", "Nichols", "Chapman", "Stride", "Eddowes", "Kelly"]
C5 = ["Nichols", "Chapman", "Stride", "Eddowes", "Kelly"]
CORE = ["Nichols", "Chapman", "Eddowes", "Kelly"]
for name, g, key in [("Keppel six", K6, "keppel6"), ("canonical five", C5, "canonical5"), ("core four", CORE, "core4")]:
    rk, ge, n, t, top = rank(g)
    b = R["scenarios"]["baseline"][key]
    ok(rk == b["rank"] and ge == b["n_ge"] and n == b["n_sets"] and abs(t - b["cohesion"]) < 1e-9, f"rank {name}")
    print(f"{name}: cohesion={t:.4f} rank={rk}/{n} n_ge={ge} p={ge/n:.4f} top3={[round(x,4) for x in top]}")
tc = {c: float(sum(J(c, k) for k in CORE if k != c) / (4 - (c in CORE))) for c in cases}
for c in cases:
    ok(abs(tc[c] - R["scenarios"]["baseline"]["to_core"][c]) < 1e-9, f"to_core {c}")
four = sorted(((coh(g), g) for g in itertools.combinations(cases, 4)), reverse=True)[:3]
print("Top four-case sets:", [(round(float(v), 4), g) for v, g in four])
print("Tabram-McKenzie J =", J("Tabram", "McKenzie"), "| Stride-Tabram J =", J("Stride", "Tabram"))

# 3. Likelihood ratios
p = 9 / N
exp = [[c / p for c in (.5, .8, .95)], [(1 - c) / (1 - p) for c in (.5, .8, .95)]]
ok(all(abs(a - b) < 1e-9 for a, b in zip(exp[0], R["lr"][0]["both"])), "LR both")
ok(all(abs(a - b) < 1e-9 for a, b in zip(exp[1], R["lr"][0]["one_missing"])), "LR missing")
ok(abs((0.5 + 0.5 * 0.2) / (1 - p) - R["lr_interrupt"]["0.5"]) < 1e-9, "LR interrupt")
print("LR both (HITS):", [round(x, 1) for x in exp[0]], "missing:", [round(x, 3) for x in exp[1]])

# 4. Abstract length in the built manuscript
if len(sys.argv) > 1:
    txt = subprocess.run(["pandoc", "-t", "plain", "--wrap=none", sys.argv[1]], capture_output=True, text=True).stdout
    ab = txt.split("Abstract", 1)[1].split("Keywords", 1)[0]
    print("Abstract words:", len(ab.split()))
print("PROBLEMS:", problems if problems else "none")
