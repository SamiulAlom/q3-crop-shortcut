# %% [markdown]
# # RUNTIME: CPU (local PC) - no GPU is used
#
# Project: `qpain/paper-03` - Stage: `04 statistics and figures`
#
# Reads only files in `outputs/` written by notebooks 01-03 and computes everything the manuscript reports
# (analysis_plan.md sections 6-8 with the deviations and clarifications of section 10): the six pre-specified contrasts
# with a class-stratified paired bootstrap and Holm correction, the secondary outcomes, the sensitivity analyses,
# secondary questions A and B, the classification table and the figures. The statistics live in `notebooks/p3_stats.py`,
# whose method test (`notebooks/tests/test_p3_stats.py`) must have passed on this version of the code.
#
# Source: `notebooks/src/04_statistics_figures_CPU.py` (edit the source; the run script rebuilds the notebook).

# %%
# Setup
import os, sys, json, hashlib, platform, datetime
import importlib.metadata as ilmd
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("PAPER_ROOT", os.path.abspath(".."))
OUT = os.path.join(ROOT, "outputs")
sys.path.insert(0, os.path.join(ROOT, "notebooks"))
import p3_stats as S
NOTEBOOK = "04_statistics_figures_CPU"
SEED = 20261122
B = S.B_DEFAULT                 # 10,000 resamples (plan section 7)
TIE_TOL = 1e-9                  # S counts as tied with the area share a within this (clarification, section 10)
FROZEN = json.load(open(os.path.join(ROOT, "notebooks", "p3_frozen.json")))
MODELS = [("resnet50", "cnn", "ResNet-50"), ("efficientnet_b0", "cnn", "EfficientNet-B0"),
          ("deit_small_patch16_224", "transformer", "DeiT-S/16"), ("vit_small_patch16_224", "transformer", "ViT-S/16")]
NAME = {m: lab for m, _, lab in MODELS}
FAMILY = {m: f for m, f, _ in MODELS}
METHODS = {"GC": "Grad-CAM", "IG": "Integrated Gradients", "ROLL": "Attention rollout", "TA": "Transformer attribution"}
VERSIONS = {m: ilmd.version(m) for m in ("numpy", "pandas", "matplotlib")}
print("Run at", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "| Python", platform.python_version(), VERSIONS, "| stats", S.STATS_VERSION)

def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def save(obj, filename, description, cell=""):
    path = os.path.join(OUT, filename)
    tmp = path + ".tmp"
    if hasattr(obj, "to_csv"):
        obj.to_csv(tmp, index=False, lineterminator="\n")
    elif hasattr(obj, "savefig"):
        obj.savefig(tmp, format=filename.rsplit(".", 1)[1], dpi=600)      # exact page size: no tight bounding box
    else:
        with open(tmp, "w", newline="\n") as f:
            json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)
    man = os.path.join(OUT, "MANIFEST.md")
    rows = [l for l in open(man).read().splitlines() if not l.startswith(f"| {filename} |")]
    rows.append(f"| {filename} | {NOTEBOOK} | {cell} | {description} |")
    with open(man + ".tmp", "w", newline="\n") as f:
        f.write("\n".join(rows) + "\n")
    os.replace(man + ".tmp", man)
    print("saved", filename)

# The method test must have been run on this version of the statistics code, and must allow p-values
mt = json.load(open(os.path.join(ROOT, "notebooks", "tests", "p3_stats_method_test.json")))
assert mt["stats_version"] == S.STATS_VERSION and mt["stats_code_sha256"] == sha256_file(os.path.join(ROOT, "notebooks", "p3_stats.py")), \
    "rerun notebooks/tests/test_p3_stats.py: the statistics code changed since the method test"
REPORT_P = mt["worst"] <= 0.075
print("method test: worst false-positive rate at 0.05 =", mt["worst"], "| p-values reported:", REPORT_P, "|", mt["class_sizes"])

# %%
# Frozen inputs
for f, key in (("pv_split.csv", "split_sha256"), ("attribution_subset.csv", "attribution_sha256"),
               ("model_eligibility.csv", "eligibility_sha256"), ("construction_checks.csv", "construction_checks_sha256")):
    assert sha256_file(os.path.join(OUT, f)) == FROZEN[key], f"{f} is not the frozen file"
sub = pd.read_csv(os.path.join(OUT, "attribution_subset.csv")).sort_values("idx").reset_index(drop=True)
IDX = sub.idx.to_numpy()
CLASSES = sorted(sub["class"].unique())
cls = sub["class"].map(CLASSES.index).to_numpy()
n = len(sub)
assert len(CLASSES) == 30 and sub.idx.is_unique
assert sorted(np.bincount(cls).tolist()) == mt["class_counts"], "the method test was not run with the class sizes of the frozen attribution subset"
CLASS_OF = sub.set_index("idx")["class"]
elig = pd.read_csv(os.path.join(OUT, "model_eligibility.csv"))
checks = pd.read_csv(os.path.join(OUT, "construction_checks.csv"))
for _c in ("bg_pass", "leaf_pass", "pair_eligible", "architecture_eligible", "used_in_outcomes"):
    assert elig[_c].dtype == bool, f"model_eligibility.csv: {_c} was not read as boolean"
assert pd.api.types.is_integer_dtype(elig.seed) and pd.api.types.is_integer_dtype(checks.seed), "seed keys must be integers in both tables"
scores = pd.read_csv(os.path.join(OUT, "attribution_scores.csv.gz"))
effects = pd.read_csv(os.path.join(OUT, "image_effects.csv.gz"))
di = pd.read_csv(os.path.join(OUT, "deletion_insertion.csv.gz"))
cls_tab = pd.read_csv(os.path.join(OUT, "classification_test.csv"))
USED = {m: sorted(elig[(elig.model == m) & elig.used_in_outcomes].seed.tolist()) for m, _, _ in MODELS}   # eligible seed pairs
ELIG = [m for m, _, _ in MODELS if USED[m]]
ELIG_CNN = [m for m in ELIG if FAMILY[m] == "cnn"]
ELIG_TR = [m for m in ELIG if FAMILY[m] == "transformer"]
print("eligible seed pairs:", USED)
print("eligible CNNs:", ELIG_CNN, "| eligible transformers:", ELIG_TR)
RES = S.resample_indices(cls, B, SEED)          # one set of resamples for every model, method and analysis (paired)

def S_of(model, regime, seed, method, target="predicted", image="original", col="S"):
    """Per-image values in the order of the attribution subset."""
    tg = "none" if method == "ROLL" else target
    d = scores[(scores.model == model) & (scores.regime == regime) & (scores.seed == seed) & (scores.method == method)
               & (scores.target == tg) & (scores.image == image)]
    key = (model, regime, seed, method, tg, image)
    assert len(d) == n and d.idx.is_unique and set(d.idx) == set(IDX), f"{key}: rows are not exactly the attribution subset"
    d = d.set_index("idx").loc[IDX]
    assert (d["class"].to_numpy() == CLASS_OF.loc[IDX].to_numpy()).all(), f"{key}: class labels differ from the subset"
    v = d[col].to_numpy(dtype=np.float64)
    assert np.isfinite(v).all(), f"{key}: non-finite {col}"
    return v

def has(model, method):
    return method in ("GC", "IG") or FAMILY[model] == "transformer"

def D_img(model, method, **kw):
    """Per-image detection difference of a method on an architecture: S on the BG model minus S on the LEAF model,
    averaged over the eligible seed pairs (seed k of BG with seed k of LEAF)."""
    return np.mean([S_of(model, "BG", s, method, **kw) - S_of(model, "LEAF", s, method, **kw) for s in USED[model]], axis=0)

def pooled(models, method, **kw):
    return np.mean([D_img(m, method, **kw) for m in models], axis=0)

def contrast_vectors(**kw):
    """The six pre-specified contrasts as one number per image each (None = cannot be estimated)."""
    v = {"H1-GC": pooled(ELIG, "GC", **kw) if ELIG else None,
         "H1-IG": pooled(ELIG, "IG", **kw) if ELIG else None,
         "H1-ROLL": pooled(ELIG_TR, "ROLL", **kw) if ELIG_TR else None,
         "H1-TA": pooled(ELIG_TR, "TA", **kw) if ELIG_TR else None,
         "H2": (pooled(ELIG_TR, "TA", **kw) - pooled(ELIG_TR, "GC", **kw)) if ELIG_TR else None,
         "H3": (pooled(ELIG_TR, "GC", **kw) - pooled(ELIG_CNN, "GC", **kw)) if (ELIG_TR and ELIG_CNN) else None}
    return v

ELIG_NAMES = "; ".join(f"{m} seeds {USED[m]}" for m in ELIG) or "none"      # printed with every contrast: which checkpoints it rests on
LABEL = {"H1-GC": "D of Grad-CAM, eligible architectures", "H1-IG": "D of Integrated Gradients, eligible architectures",
         "H1-ROLL": "D of attention rollout, eligible transformers", "H1-TA": "D of transformer attribution, eligible transformers",
         "H2": "D of transformer attribution minus D of Grad-CAM, eligible transformers",
         "H3": "D of Grad-CAM on transformers minus on CNNs"}

_names = lambda ms: "; ".join(f"{m} seeds {USED[m]}" for m in ms) or "none"
POOL_OF = {"H1-GC": _names(ELIG), "H1-IG": _names(ELIG), "H1-ROLL": _names(ELIG_TR), "H1-TA": _names(ELIG_TR), "H2": _names(ELIG_TR),
           "H3": "transformers: " + _names(ELIG_TR) + " | CNNs: " + _names(ELIG_CNN)}
_no = "no eligible architecture"
WHY_NOT = {"H1-GC": _no, "H1-IG": _no, "H1-ROLL": "no eligible transformer", "H1-TA": "no eligible transformer", "H2": "no eligible transformer",
           "H3": "; ".join(x for x in ("no eligible CNN" if not ELIG_CNN else "", "no eligible transformer" if not ELIG_TR else "") if x)}
PRIMARY_ANSWERABLE = bool(ELIG_CNN and ELIG_TR)      # plan section 4: at least one CNN and one transformer eligible
COLS = ["contrast", "definition", "analysis", "estimable", "estimate", "ci_low", "ci_high", "boot_se", "B", "p", "p_holm", "significant_0.05",
        "eligible_cnns", "eligible_transformers", "primary_question_answerable", "note"]

def contrast_table(vectors, test, tag, notes=None):
    rows = []
    for k in LABEL:
        v = vectors[k]
        if v is None:
            rows.append({"contrast": k, "definition": LABEL[k], "analysis": tag, "estimable": False, "note": (notes or {}).get(k, WHY_NOT[k])})
            continue
        r = S.summarise(S.class_mean(v, cls), S.boot_class_mean(v, cls, RES), test=test and REPORT_P)
        rows.append({"contrast": k, "definition": LABEL[k] + " [" + POOL_OF[k] + "]", "analysis": tag, "estimable": True, **r})
    t = pd.DataFrame(rows).reindex(columns=COLS)
    if test and REPORT_P:
        t["p_holm"] = S.holm(t["p"].to_numpy(dtype=float), m=6)      # the family always counts six
        t["significant_0.05"] = [bool(x < 0.05) if np.isfinite(x) else "" for x in t.p_holm.to_numpy(dtype=float)]     # blank, not False, when not estimable
    t["eligible_cnns"], t["eligible_transformers"] = len(ELIG_CNN), len(ELIG_TR)
    t["primary_question_answerable"] = PRIMARY_ANSWERABLE
    t["eligible_checkpoint_pairs"] = ELIG_NAMES
    return t

# %%
# Primary: the six contrasts (original images, predicted class), and the sensitivity analyses of D (no p-values)
primary = contrast_table(contrast_vectors(), True, "primary")
v_true = contrast_vectors(target="true")
v_true["H1-ROLL"] = None                        # rollout does not depend on the class explained: not a sensitivity result
NA = "not applicable to this analysis"
v_native = {k: None for k in LABEL}             # the native grid concerns Grad-CAM only
if ELIG:
    v_native["H1-GC"] = pooled(ELIG, "GC", col="S_native")
if ELIG_TR and ELIG_CNN:
    v_native["H3"] = pooled(ELIG_TR, "GC", col="S_native") - pooled(ELIG_CNN, "GC", col="S_native")
sens = pd.concat([contrast_table(v_true, False, "sensitivity i: true class", {"H1-ROLL": NA + " (rollout is class-agnostic)"}),
                  contrast_table(v_native, False, "sensitivity ii: Grad-CAM on its native grid", {k: NA for k in ("H1-IG", "H1-ROLL", "H1-TA", "H2")}),
                  contrast_table(contrast_vectors(image="own_regime"), False, "sensitivity iii: images built by the model's own regime")], ignore_index=True)
print(primary.to_string(index=False))
print(sens[sens.estimable.astype(bool)].to_string(index=False))
print("primary question answerable (one CNN and one transformer eligible):", PRIMARY_ANSWERABLE)
save(primary, "contrasts_primary.csv", "The six pre-specified contrasts: estimate (unweighted mean of class means), 95% percentile bootstrap interval, centred-bootstrap p-value, Holm-adjusted p-value over six slots", "primary")
save(sens, "contrasts_sensitivity.csv", "Sensitivity analyses of the contrasts: true class explained, Grad-CAM on its native grid, own-regime images (estimates and intervals, no p-values)", "primary")

# D by architecture and by seed, always printed next to pooled values (plan section 7)
rows = []
for m in ELIG:
    for meth in METHODS:
        if not has(m, meth):
            continue
        v = D_img(m, meth)
        rows.append({"model": m, "family": FAMILY[m], "method": meth, "level": "architecture (eligible seeds)", "seed": "",
                     **S.summarise(S.class_mean(v, cls), S.boot_class_mean(v, cls, RES), test=False)})
        for s in USED[m]:
            vs = S_of(m, "BG", s, meth) - S_of(m, "LEAF", s, meth)
            rows.append({"model": m, "family": FAMILY[m], "method": meth, "level": "seed", "seed": s, "estimate": float(S.class_mean(vs, cls))})
d_by = pd.DataFrame(rows).reindex(columns=["model", "family", "method", "level", "seed", "estimate", "ci_low", "ci_high", "boot_se", "B"])
save(d_by, "detection_difference_by_model.csv", "Detection difference D per architecture (95% interval) and per seed pair, each method", "primary")
print(d_by.to_string(index=False))

# Exploratory, plan section 10 deviation 5 (added after the validation checks were frozen and before any attribution output
# existed): D on original images for EVERY architecture and seed pair, eligible or not. Pointwise 95% intervals over leaves
# for fixed checkpoints; no p-value, no pooling over architectures or families, no family comparison, outside the Holm family.
# In a non-eligible pair D mixes the attribution method with an incompletely planted shortcut, so the pair's shortcut
# strength (validation effects of its BG checkpoint) is printed in the same row.
_el = elig.set_index(["model", "seed"])
_bg = checks[checks.regime == "BG"].set_index(["model", "seed"])
assert _el.index.is_unique and _bg.index.is_unique and len(_el) == len(MODELS) * 3 and len(_bg) == len(MODELS) * 3
rows = []
for m, fam, _ in MODELS:
    for s_ in sorted(elig[elig.model == m].seed.tolist()):
        assert _el.at[(m, s_), "bg_weights_sha256"] == _bg.at[(m, s_), "weights_sha256"], f"{m} seed {s_}: BG checkpoint of the eligibility table is not the checked one"
        for meth in METHODS:
            if not has(m, meth):
                continue
            v = S_of(m, "BG", s_, meth) - S_of(m, "LEAF", s_, meth)
            r = S.summarise(S.class_mean(v, cls), S.boot_class_mean(v, cls, RES), test=False)
            rows.append({"analysis": "exploratory: all pairs (deviation 5)", "model": m, "family": fam, "seed": s_, "method": meth,
                         "pair_eligible": bool(_el.at[(m, s_), "pair_eligible"]), "used_in_confirmatory": bool(_el.at[(m, s_), "used_in_outcomes"]),
                         "D": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "B": r["B"],
                         "bg_background_effect": float(_bg.at[(m, s_), "background_effect"]), "bg_leaf_effect": float(_bg.at[(m, s_), "leaf_effect"]),
                         "bg_accuracy_leaf_removed": float(_bg.at[(m, s_), "accuracy_leaf_removed"])})
d_all = pd.DataFrame(rows)
assert len(d_all) == sum(3 * sum(has(m, meth) for meth in METHODS) for m, _, _ in MODELS)
assert int(d_all.used_in_confirmatory.sum()) == sum(len(USED[m]) * sum(has(m, meth) for meth in METHODS) for m, _, _ in MODELS)
save(d_all, "detection_difference_all_pairs_exploratory.csv", "EXPLORATORY (plan deviation 5): D per architecture, seed pair and method for all pairs, eligible or not; 95% bootstrap interval over leaves for fixed checkpoints; no p-values; with each BG checkpoint's validation background effect, leaf effect and leaf-removed accuracy", "exploratory")
print(d_all.round(4).to_string(index=False))

# %%
# Secondary, reported beside D: mean off-leaf share per regime, excess over the area share, below-uniform rate of BG maps
_a = scores[scores.image == "original"].groupby("idx").area_off_leaf.agg(["min", "max"])
assert set(_a.index) == set(IDX) and (_a["min"] == _a["max"]).all(), "the area share of an image differs between runs"
a_share = _a["min"].loc[IDX].to_numpy(dtype=np.float64)
rows = []
for m, fam, _ in MODELS:
    seeds_all = sorted(scores[scores.model == m].seed.unique())
    for meth in METHODS:
        if not has(m, meth):
            continue
        for regime in ("BG", "LEAF", "ORIG"):
            seeds = USED[m] if regime in ("BG", "LEAF") else seeds_all
            if not seeds:
                continue
            v = np.mean([S_of(m, regime, s, meth) for s in seeds], axis=0)
            r = S.summarise(S.class_mean(v, cls), S.boot_class_mean(v, cls, RES), test=False)
            re_ = S.summarise(S.class_mean(v - a_share, cls), S.boot_class_mean(v - a_share, cls, RES), test=False)
            row = {"model": m, "family": fam, "method": meth, "regime": regime, "seeds": len(seeds), "mean_S": r["estimate"],
                   "ci_low": r["ci_low"], "ci_high": r["ci_high"], "mean_area_share": float(S.class_mean(a_share, cls)),
                   "mean_excess": re_["estimate"], "excess_ci_low": re_["ci_low"], "excess_ci_high": re_["ci_high"]}
            if regime == "BG":   # below-uniform rate: S below the background's area share, ties counted apart
                below = np.mean([(S_of(m, "BG", s, meth) < a_share - TIE_TOL) for s in seeds], axis=0)
                ties = np.mean([(np.abs(S_of(m, "BG", s, meth) - a_share) <= TIE_TOL) for s in seeds], axis=0)
                rb = S.summarise(S.class_mean(below, cls), S.boot_class_mean(below, cls, RES), test=False)
                rt = S.summarise(S.class_mean(ties, cls), S.boot_class_mean(ties, cls, RES), test=False)
                row.update({"below_uniform_rate": rb["estimate"], "below_ci_low": rb["ci_low"], "below_ci_high": rb["ci_high"],
                            "tie_rate": rt["estimate"], "tie_ci_low": rt["ci_low"], "tie_ci_high": rt["ci_high"]})
            rows.append(row)
share_tab = pd.DataFrame(rows)
save(share_tab, "off_leaf_share.csv", "Mean off-leaf share S per architecture, method and regime (eligible seeds for BG and LEAF, all seeds for ORIG), 95% interval, excess over the area share, and for BG the below-uniform rate with its tie rate", "secondary")
print(share_tab.round(3).to_string(index=False))
# Control maps: S is the same for every model, so D is exactly zero; their S is reported once (SIL = 0, UNIF = area share)
print("mean background area share a:", round(float(S.class_mean(a_share, cls)), 4))
zero = pd.read_csv(os.path.join(OUT, "zero_mass_maps.csv"))
print("maps replaced by the uniform map:", int(zero.zero_mass_maps.sum()), "zero mass,", int(zero.non_finite_maps.sum()), "non-finite, of", int(zero.maps.sum()))

# %%
# Secondary A (ORIG models): does the excess off-leaf share track the measured background effect, image by image?
rows = []
for m, fam, _ in MODELS:
    seeds_all = sorted(scores[(scores.model == m) & (scores.regime == "ORIG")].seed.unique())
    eff_m = {s: effects[(effects.model == m) & (effects.regime == "ORIG") & (effects.seed == s)].set_index("idx").reindex(IDX) for s in seeds_all}
    for meth in METHODS:
        if not has(m, meth):
            continue
        ex = {s: S_of(m, "ORIG", s, meth) - a_share for s in seeds_all}
        for col in ("background_effect", "background_effect_margin"):
            be = {s: eff_m[s][col].to_numpy(dtype=np.float64) for s in seeds_all}
            for s in seeds_all:
                assert np.isfinite(be[s]).all(), (m, s, col)
            rho = [S.spearman(ex[s], be[s]) for s in seeds_all]
            defined = not np.any(np.isnan(rho))          # undefined if an input is constant on any seed (stated, never imputed)
            est = float(np.mean(rho)) if defined else float("nan")
            reps = np.full(B, np.nan)
            if defined:
                for b in range(B):
                    reps[b] = np.mean([S.spearman(ex[s][RES[b]], be[s][RES[b]]) for s in seeds_all])   # NaN if constant in this resample
            ok = reps[~np.isnan(reps)]
            full = defined and len(ok) == B                # an interval is given only if every resample was defined
            rows.append({"model": m, "family": fam, "method": meth, "effect": col, "status": "defined" if defined else "undefined (constant input)",
                         "spearman_mean_over_seeds": est, "ci_low": float(np.percentile(ok, 2.5)) if full else float("nan"),
                         "ci_high": float(np.percentile(ok, 97.5)) if full else float("nan"),
                         "valid_resamples": int(len(ok)), "resamples": B, "per_seed": json.dumps([round(float(x), 4) for x in rho]),
                         "effect_median_per_seed": json.dumps([round(float(np.median(be[s])), 5) for s in seeds_all]),
                         "effect_tie_share_per_seed": json.dumps([round(S.tie_share(be[s]), 4) for s in seeds_all]),
                         "effect_median": float(np.median(np.concatenate(list(be.values())))),
                         "effect_p90": float(np.percentile(np.concatenate(list(be.values())), 90)),
                         "effect_share_below_0.01": float(np.mean(np.concatenate(list(be.values())) < 0.01)),
                         "effect_tie_share": float(np.mean([S.tie_share(be[s]) for s in seeds_all]))})
sec_a = pd.DataFrame(rows)
save(sec_a, "secondary_a_dose_response.csv", "Secondary A: Spearman correlation over images between the excess off-leaf share and the background effect (probability and logit margin) on ORIG models, mean over seeds with 95% interval, per-seed values, spread and ties of the effect", "secondary A")
print(sec_a.round(3).to_string(index=False))

# %%
# Secondary B (ORIG models): deletion and insertion against model-independent control maps
rows = []
assert set(di.model) == {m for m, _, _ in MODELS}, "deletion and insertion results are missing for an architecture"
assert di.groupby("model").seed.nunique().eq(3).all() and di.groupby("model").fill.nunique().nunique() == 1
for m, fam, _ in MODELS:
    dm = di[di.model == m]
    seeds_all = sorted(dm.seed.unique())
    for fill in sorted(dm.fill.unique()):
        def auc(map_name, col):
            return np.mean([dm[(dm.seed == s) & (dm["map"] == map_name) & (dm.fill == fill)].set_index("idx")[col].reindex(IDX).to_numpy(dtype=np.float64) for s in seeds_all], axis=0)
        for map_name in sorted(dm["map"].unique()):
            for col in ("deletion_auc", "insertion_auc"):
                v = auc(map_name, col)
                assert np.isfinite(v).all()
                row = {"model": m, "family": fam, "map": map_name, "kind": "method" if map_name in METHODS else "control", "fill": fill,
                       "outcome": col, **S.summarise(S.class_mean(v, cls), S.boot_class_mean(v, cls, RES), test=False)}
                if map_name in METHODS:
                    for ctrl in ("SIL", "EDGE"):
                        dv = v - auc(ctrl, col)
                        r = S.summarise(S.class_mean(dv, cls), S.boot_class_mean(dv, cls, RES), test=False)
                        row.update({f"minus_{ctrl}": r["estimate"], f"minus_{ctrl}_ci_low": r["ci_low"], f"minus_{ctrl}_ci_high": r["ci_high"]})
                rows.append(row)
sec_b = pd.DataFrame(rows)
save(sec_b, "secondary_b_deletion_insertion.csv", "Secondary B: deletion and insertion areas under the curve on ORIG models per architecture, map and fill (mean over seeds, 95% interval); for methods, the control-adjusted score (method minus leaf silhouette, method minus edge map). Lower deletion and higher insertion mean a better map", "secondary B")
print(sec_b.round(4).to_string(index=False))

# %%
# Classification on the test partition, construction checks side by side (Table I), and exploratory analyses (section 8)
ct = (cls_tab.groupby(["model", "family", "regime", "images"]).agg(seeds=("seed", "nunique"), accuracy=("accuracy_image_label", "mean"),
      accuracy_min=("accuracy_image_label", "min"), accuracy_max=("accuracy_image_label", "max"), macro_f1=("macro_f1_image_label", "mean"),
      accuracy_donor_label=("accuracy_donor_label", "mean"), n_test_images=("n", "first")).reset_index())
save(ct, "classification_summary.csv", "Test accuracy and macro F1 per architecture and regime on original, background-swapped, leaf-swapped and leaf-removed images (mean, minimum and maximum over seeds); accuracy against the donor's label for swaps", "tables")
ck = (checks.groupby(["model", "family", "regime"]).agg(val_accuracy_original=("val_accuracy_original", "mean"), background_effect=("background_effect", "mean"),
      leaf_effect=("leaf_effect", "mean"), artefact_effect_own_regime=("artefact_effect_own_regime", "mean"), accuracy_outline_only=("accuracy_outline_only", "mean"),
      accuracy_texture_only=("accuracy_texture_only", "mean"), accuracy_leaf_removed=("accuracy_leaf_removed", "mean"),
      checkpoints=("run", "size")).reset_index())
_passed = checks[checks.regime.isin(["BG", "LEAF"])].assign(ok=lambda t: t["pass"].map(lambda x: str(x).strip().lower() == "true")).groupby(["model", "regime"]).ok.sum()
ck["checkpoints_passed"] = [int(_passed.get((m_, r_), 0)) if r_ in ("BG", "LEAF") else "not applicable" for m_, r_ in zip(ck.model, ck.regime)]
save(ck, "construction_checks_summary.csv", "Construction checks per architecture and regime, mean over seeds, with the number of checkpoints that passed (validation images)", "tables")
print(ct.round(3).to_string(index=False)); print(ck.round(3).to_string(index=False))

# Exploratory (labelled as such in the paper): off-leaf share by class, and for wrong against right predictions of ORIG models
orig = scores[(scores.regime == "ORIG") & (scores.image == "original") & (scores.target.isin(["predicted", "none"]))].copy()
orig["correct"] = orig["class"] == orig.pred_class
by_class = orig.groupby(["family", "method", "class"]).S.mean().reset_index().rename(columns={"S": "mean_S"})
by_class["area_share"] = by_class["class"].map(pd.Series(a_share).groupby(sub["class"].to_numpy()).mean())
right_wrong = orig.groupby(["model", "method", "correct"]).S.agg(["mean", "size"]).reset_index().rename(columns={"mean": "mean_S", "size": "maps"})
save(by_class, "exploratory_off_leaf_by_class.csv", "Exploratory: mean off-leaf share of ORIG models by class, method and family, with the class's mean background area share", "exploratory")
save(right_wrong, "exploratory_off_leaf_right_wrong.csv", "Exploratory: mean off-leaf share of ORIG models for right and wrong predictions (maps pooled over seeds and images)", "exploratory")

# %%
# Figure 2: off-leaf share by method, architecture and regime (points = mean over eligible seeds, bars = 95% interval)
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
COL = {"BG": "#D55E00", "ORIG": "#000000", "LEAF": "#0072B2"}     # Okabe-Ito; marker shape repeats the cue
MARK = {"BG": "s", "ORIG": "o", "LEAF": "^"}
order = [(m, meth) for m, _, _ in MODELS for meth in METHODS if has(m, meth)]
fig, ax = plt.subplots(figsize=(7.16, 2.6), layout="constrained")   # IEEE double-column width, 7.16 in
xt, xl = [], []
for k, (m, meth) in enumerate(order):
    for j, regime in enumerate(("LEAF", "ORIG", "BG")):
        r = share_tab[(share_tab.model == m) & (share_tab.method == meth) & (share_tab.regime == regime)]
        if r.empty:
            continue
        r = r.iloc[0]
        x_ = k + (j - 1) * 0.22
        ax.vlines(x_, r.ci_low, r.ci_high, color=COL[regime], lw=0.8)          # the interval as given, from its endpoints
        ax.plot([x_], [r.mean_S], MARK[regime], color=COL[regime], ms=3.5)
    xt.append(k); xl.append(meth)
a_mean = float(S.class_mean(a_share, cls))
ax.axhline(a_mean, color="0.6", lw=0.7, ls="--")
ax.text(len(order) - 0.45, a_mean + 0.015, "uniform map", ha="right", va="bottom", fontsize=6.5, color="0.4")
edges = np.cumsum([sum(1 for mm, _ in order if mm == m) for m, _, _ in MODELS])
start = 0
for (m, _, lab), e in zip(MODELS, edges):
    if e > start:
        ax.text((start + e - 1) / 2, 1.04, lab + ("" if USED[m] else "\n(ORIG only)"), ha="center", va="bottom", fontsize=7.5, transform=ax.get_xaxis_transform())
        if e < len(order):
            ax.axvline(e - 0.5, color="0.85", lw=0.6)
    start = e
ax.set_xticks(xt); ax.set_xticklabels(xl)
ax.set_ylim(0, 1); ax.set_xlim(-0.6, len(order) - 0.4)
ax.set_ylabel("Off-leaf share of the map, S")
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker=MARK[r_], color=COL[r_], lw=0.8, ms=3.5, label=r_) for r_ in ("LEAF", "ORIG", "BG")]
ax.legend(handles=handles, title="Model trained on (ORIG: mean of all seeds; BG, LEAF: mean of the eligible seed pairs)", frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7, title_fontsize=7)
save(fig, "fig2_off_leaf_share.pdf", "Figure 2: off-leaf share S by method, architecture and training regime (ORIG: mean over all seeds; BG and LEAF: mean over the eligible seed pairs, shown for eligible architectures only; 95% bootstrap interval over leaves); dashed line = uniform map", "figures")
save(fig, "fig2_off_leaf_share.png", "Figure 2 as PNG (600 dpi)", "figures")
plt.close(fig)
save(share_tab[["model", "method", "regime", "seeds", "mean_S", "ci_low", "ci_high"]], "fig2_data.csv", "Numbers plotted in Figure 2; seeds = checkpoints averaged (ORIG: all seeds; BG and LEAF: eligible seed pairs only)", "figures")

# %%
# Environment and what was read
save({"notebook": NOTEBOOK, "python": platform.python_version(), "packages": VERSIONS, "stats_version": S.STATS_VERSION, "B": B, "seed": SEED,
      "tie_tolerance": TIE_TOL, "p_values_reported": REPORT_P, "method_test": mt, "eligible_seed_pairs": USED, "n_images": n, "n_classes": len(CLASSES),
      "primary_question_answerable": PRIMARY_ANSWERABLE, "class_counts": np.bincount(cls).tolist(),
      "inputs_sha256": {f: sha256_file(os.path.join(OUT, f)) for f in ("attribution_scores.csv.gz", "image_effects.csv.gz", "deletion_insertion.csv.gz",
                                                                    "classification_test.csv", "construction_checks.csv", "model_eligibility.csv", "attribution_subset.csv",
                                                                    "zero_mass_maps.csv", "pv_split.csv")},
      "notebook_source_sha256": sha256_file(os.path.join(ROOT, "notebooks", "src", "04_statistics_figures_CPU.py")),
      "stats_code_sha256": sha256_file(os.path.join(ROOT, "notebooks", "p3_stats.py")), "frozen": FROZEN,
      "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
     "statistics_environment.json", "Environment, settings, method-test result, eligible seed pairs and input hashes of the statistics notebook", "environment")
print("DONE")
