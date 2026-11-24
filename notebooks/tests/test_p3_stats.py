"""Method test for p3_stats.py (analysis_plan.md section 7, last item). Rerun after every edit of the statistics code.
1. The bootstrap code against explicit computations.
2. The false-positive rate of the centred-bootstrap test at 0.05 under GENERATED data with no effect. The generated
   numbers evaluate the method only; they are never data and never appear in the paper as observations.
Plan rule: if the false-positive rate is above 0.075, p-values are not reported (written as a deviation before results).
Run: .venv-local/Scripts/python.exe projects/qpain/paper-03/notebooks/tests/test_p3_stats.py [path to attribution_subset.csv]
Writes tests/p3_stats_method_test.json next to this file.
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import p3_stats as S

# ---- 1. explicit computations
cls = np.array([0, 0, 0, 1, 1, 2])
v = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 7.0])
assert np.isclose(S.class_mean(v, cls), (2.0 + 15.0 + 7.0) / 3)
idx = S.resample_indices(cls, 2000, 1)
assert idx.shape == (2000, 6) and (cls[idx] == cls[None, :]).all(), "a resample left its class"
assert (idx == S.resample_indices(cls, 2000, 1)).all(), "resampling is not reproducible"
reps = S.boot_class_mean(v, cls, idx)
explicit = np.array([np.mean([v[idx[b]][cls == c].mean() for c in (0, 1, 2)]) for b in range(2000)])
assert np.allclose(reps, explicit), "vectorised bootstrap differs from the explicit loop"
two = S.boot_class_mean(np.stack([v, -v], axis=1), cls, idx)
assert np.allclose(two[:, 0], reps) and np.allclose(two[:, 1], -reps), "columns are not resampled together"
# bootstrap SE against the analytic SE of a stratified mean (plug-in variances), large classes
rng = np.random.default_rng(0)
cls_big = np.repeat(np.arange(5), 200)
x = rng.normal(0, 1 + cls_big, 1000)
rep = S.boot_class_mean(x, cls_big, S.resample_indices(cls_big, 4000, 2))
analytic = np.sqrt(sum(x[cls_big == c].var() / 200 for c in range(5))) / 5
assert abs(rep.std(ddof=1) / analytic - 1) < 0.05, (rep.std(ddof=1), analytic)
# p-value: shifted far from zero -> small; centred at zero -> not small
s1 = S.summarise(3.0, rng.normal(3.0, 1.0, 9999)); s0 = S.summarise(0.1, rng.normal(0.1, 1.0, 9999))
assert s1["p"] < 0.01 and s0["p"] > 0.5 and s1["ci_low"] < 3.0 < s1["ci_high"]
# Holm, six slots: against a hand computation, with and without a missing contrast
p = np.array([0.001, 0.04, 0.03, 0.2, 0.01, 0.5])
assert np.allclose(S.holm(p), [0.006, 0.12, 0.12, 0.4, 0.05, 0.5])
pm = np.array([0.001, np.nan, 0.03, 0.2, 0.01, 0.5])
h = S.holm(pm)
assert np.isnan(h[1]) and np.allclose(h[[0, 4, 2, 3, 5]], [0.006, 0.05, 0.12, 0.6, 1.0]), h
# Spearman with ties against a hand value, and the constant case
assert np.isclose(S.spearman([1, 2, 3, 4], [1, 3, 2, 4]), 0.8) and np.isnan(S.spearman([1, 1, 1], [1, 2, 3]))
assert np.isclose(S.spearman([1, 1, 2, 3], [1, 2, 2, 3]), np.corrcoef([1.5, 1.5, 3, 4], [1, 2.5, 2.5, 4])[0, 1])
assert np.isclose(S.tie_share([1, 1, 2, 3, 3, 3]), 5 / 6)
print("explicit checks passed")

# ---- 2. false-positive rate under generated data with no effect
# Class sizes: those of the real attribution subset if its file is given, else 30 classes of 9 to 30 images.
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    import csv
    names = [r["class"] for r in csv.DictReader(open(sys.argv[1], encoding="utf-8"))]
    codes = {c: i for i, c in enumerate(sorted(set(names)))}
    cls_sim = np.array([codes[c] for c in names])
    source = "class sizes of outputs/attribution_subset.csv"
else:
    sizes = np.r_[np.full(20, 30), [27, 18, 17, 16, 26, 14, 10, 9, 19, 29]]
    cls_sim = np.repeat(np.arange(30), sizes)
    source = "planned class sizes (20 classes of 30, 10 classes of 9 to 29)"
N_SIM, B_SIM = 2000, 1999
rng = np.random.default_rng(20261122)
results = {}
for label, draw in (("normal", lambda n: rng.normal(0, 0.2, n)),
                    ("skewed (centred exponential)", lambda n: rng.exponential(0.2, n) - 0.2),
                    ("bounded two-point (+-0.5, as a difference of shares can be)", lambda n: rng.choice([-0.5, 0.5], n)),
                    ("class-dependent spread", lambda n: rng.normal(0, 0.05 + 0.3 * (cls_sim % 3), n)),
                    ("zero-heavy (80% exact zeros, else +-0.4)", lambda n: rng.choice([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4, 0.4], n)),
                    ("bounded and asymmetric (centred beta(1,6))", lambda n: rng.beta(1, 6, n) - 1 / 7)):
    rej = 0
    for s in range(N_SIM):
        v = draw(len(cls_sim))
        th = S.class_mean(v, cls_sim)
        rp = S.boot_class_mean(v, cls_sim, S.resample_indices(cls_sim, B_SIM, 10_000 + s))
        rej += S.summarise(th, rp)["p"] < 0.05
    results[label] = rej / N_SIM
    print(f"false-positive rate at 0.05, {label}: {results[label]:.4f}")
worst = max(results.values())
verdict = "p-values reported" if worst <= 0.075 else "ABOVE 0.075: p-values are not reported (plan section 7); write the deviation before results"
import hashlib
out = {"stats_version": S.STATS_VERSION, "stats_code_sha256": hashlib.sha256(open(os.path.join(HERE, "..", "p3_stats.py"), "rb").read()).hexdigest(),
       "class_counts": sorted(np.bincount(cls_sim).tolist()), "class_sizes": source, "n_images": int(len(cls_sim)), "n_classes": int(len(np.unique(cls_sim))),
       "simulated_datasets": N_SIM, "bootstrap_resamples_per_dataset": B_SIM, "false_positive_rate_at_0.05": results,
       "worst": worst, "rule": "above 0.075 -> no p-values", "verdict": verdict,
       "note": "generated data used only to evaluate the test; not observations"}
json.dump(out, open(os.path.join(HERE, "p3_stats_method_test.json"), "w"), indent=1)
print("worst:", worst, "|", verdict)
