# %% [markdown]
# # RUNTIME: local workstation GPU (NVIDIA RTX 3060 12 GB), `.venv-local` (plan section 10)
#
# Project: `qpain/paper-03` - Stage: `03 construction checks and attribution` - Estimated time: about 6 h (to be measured)
#
# Two stages, run separately.
# `STAGE=checks` (validation images only): the construction checks of analysis_plan.md section 4 (as redefined in
# deviation 2) for all 36 checkpoints, the three descriptive checks, and the eligibility table. It stops there. The
# manager commits the table and writes its hash into `notebooks/p3_frozen.json`.
# `STAGE=attr` (refuses to start unless the eligibility table matches the frozen hash): attribution maps on the fixed
# attribution subset, off-leaf shares, sensitivity analyses, background effects, test classification, and deletion
# and insertion curves for the ORIG models.
#
# `SMOKE=1`: seed 11 only, 2 images per class, results to checkpoints/smoke/, nothing to outputs/.
# Source: `notebooks/src/03_attribution_GPU.py` (edit the source; the run script rebuilds the notebook).

# %%
# Setup
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, json, time, hashlib, platform, datetime, subprocess
import importlib.metadata as ilmd
import numpy as np, pandas as pd
import torch, torch.nn.functional as F

ROOT = os.environ.get("PAPER_ROOT", os.path.abspath(".."))
OUT, CKPT, CACHE = (os.path.join(ROOT, d) for d in ("outputs", "checkpoints", "data/cache"))
sys.path.insert(0, os.path.join(ROOT, "notebooks"))
import p3_common as P
import p3_torch as T
SMOKE = os.environ.get("SMOKE", "0") == "1"
STAGE = os.environ.get("STAGE", "checks")
assert STAGE in ("checks", "attr")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
FILLS = os.environ.get("FILLS", "blur,mean").split(",")     # deletion/insertion fills; "mean" is the first cut (plan section 8)
NOTEBOOK = "03_attribution_GPU"
RUN_DIR = os.path.join(CKPT, "smoke" if SMOKE else "runs")          # checkpoints of notebook 02
ATTR_DIR = os.path.join(CKPT, "smoke_attr" if SMOKE else "attr")    # per-run results of this notebook (resumable)
os.makedirs(ATTR_DIR, exist_ok=True)
FROZEN = json.load(open(os.path.join(ROOT, "notebooks", "p3_frozen.json")))

SEED = 20261122
MODELS = [("resnet50", "cnn"), ("efficientnet_b0", "cnn"), ("deit_small_patch16_224", "transformer"), ("vit_small_patch16_224", "transformer")]
REGIMES, SEEDS = ["ORIG", "LEAF", "BG"], [11, 22, 33]
IMG = P.SIDE
BATCH = 64
N_SWAPS = 5                 # plan section 4: 5 fixed swaps per image
CHECK_PER_CLASS = 50        # validation images per class used for the effects (accuracy uses all validation images)
# Pass rules, fixed in analysis_plan.md section 4 before any model existed
RULES = {"BG": {"acc_min": 0.20, "own_effect_min": 0.30, "other_effect_max": 0.10, "artefact_max": 0.10},
         "LEAF": {"acc_min": 0.90, "own_effect_min": 0.30, "other_effect_max": 0.10, "artefact_max": 0.10}}
TAIL = 0.05                 # share of images with an effect below this is reported
DEL_STEPS = 28              # plan section 6: 28 steps of 1/28 of the pixels
DEL_BLUR_SIGMA = 11.0       # pixels; the blur fill of the deletion and insertion curves (clarification, fixed before any curve)
N_EXAMPLES = 6              # images whose maps are kept for the example figure
if SMOKE:
    SEEDS, CHECK_PER_CLASS = [11], 2

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# Scoring precision (Codex review 003, point 4): every forward and backward pass of this notebook is float32, without
# autocast and without TF32, so that predicted classes, effects, gates and curves do not depend on mixed precision.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
BACKEND = {"precision": "float32, no autocast", "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
           "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic}
VERSIONS = {m: ilmd.version(m) for m in ("torch", "timm", "numpy", "pandas", "scipy")}
ENV = {"python": platform.python_version(), "platform": platform.platform(), "device": DEVICE, "packages": VERSIONS,
       "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None, "smoke": SMOKE, "stage": STAGE, "fills": FILLS,
       "common_version": P.COMMON_VERSION, "torch_module_version": T.TORCH_VERSION, "backend": BACKEND}
print(json.dumps(ENV, indent=1))

# %%
# Helpers
def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def atomic(path, write):
    tmp = path + ".tmp"
    write(tmp)
    os.replace(tmp, path)

def write_json(path, obj):
    def w(t):
        with open(t, "w", newline="\n") as f:
            json.dump(obj, f, indent=1, default=str)
    atomic(path, w)

def save_output(obj, filename, description, cell=""):
    assert not SMOKE, "smoke runs never write to outputs/"
    path = os.path.join(OUT, filename)
    if hasattr(obj, "to_csv"):
        atomic(path, lambda t: obj.to_csv(t, index=False, lineterminator="\n",
                                          compression={"method": "gzip", "mtime": 0} if filename.endswith(".gz") else None))
    elif isinstance(obj, dict) and filename.endswith(".npz"):
        def w(t):
            with open(t, "wb") as f:
                np.savez_compressed(f, **obj)
        atomic(path, w)
    else:
        write_json(path, obj)
    man = os.path.join(OUT, "MANIFEST.md")
    head = "| file | notebook | cell | description |\n|---|---|---|---|\n"
    rows = [l for l in (open(man).read() if os.path.exists(man) else head).splitlines() if not l.startswith(f"| {filename} |")]
    rows.append(f"| {filename} | {NOTEBOOK} | {cell} | {description} |")
    def w(t):
        with open(t, "w", newline="\n") as f:
            f.write("\n".join(rows) + "\n")
    atomic(man, w)
    print("saved", filename)
    return path

def tick():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return time.time()

# %%
# Frozen inputs: split, attribution subset, cache rows (verified against the digests in the split), checkpoints
split_path = os.path.join(OUT, "pv_split.csv")
assert sha256_file(split_path) == FROZEN["split_sha256"], "pv_split.csv is not the frozen split"
split = pd.read_csv(split_path)
CLASSES = sorted(split["class"].unique())
assert len(CLASSES) == 30
C = T.Cache(CACHE)
by_idx = split.set_index("idx")

def verify_rows(r):
    for i in range(0, len(r), 512):
        rgb, m, bg = (T.read_rows(a, r[i:i + 512]) for a in (C.rgb, C.mask, C.bg))
        for j in range(len(rgb)):
            assert P.cache_row_sha(rgb[j], m[j], bg[j]) == by_idx.cache_row_sha256.at[r[i + j]], f"cache row {r[i + j]} differs from the frozen split"

def label_of(r):
    return by_idx["class"].loc[r].map(CLASSES.index).to_numpy(dtype=np.int64)

val = split[split.partition == "val"].sort_values("idx").reset_index(drop=True)
runs_table = pd.read_csv(os.path.join(OUT, "train_runs.csv")) if not SMOKE else None

def run_list():
    """The 36 selected runs (notebook 02), in a fixed order, each with its marker."""
    out = []
    for name, family in MODELS:
        for regime in REGIMES:
            for seed in SEEDS:
                if SMOKE:
                    cands = [f[:-5] for f in sorted(os.listdir(RUN_DIR)) if f.startswith(f"{name}_{regime}_s{seed}_lr") and f.endswith(".json")]
                    assert cands, f"no smoke run for {name} {regime} {seed}"
                    run = cands[0]
                else:
                    sel = runs_table[(runs_table.model == name) & (runs_table.regime == regime) & (runs_table.seed == seed)]
                    assert len(sel) == 1
                    run = sel.run.iloc[0]
                marker = json.load(open(os.path.join(RUN_DIR, run + ".json")))
                assert marker["run"] == run and marker["classes"] == CLASSES
                assert (marker["model"], marker["family"], marker["regime"], marker["seed"]) == (name, family, regime, seed), f"{run}: marker does not describe this run"
                if not SMOKE:
                    assert marker["weights_sha256"] == sel.weights_sha256.iloc[0]
                out.append(marker)
    return out

RUNS = run_list()
print(len(RUNS), "checkpoints")

def load_model(marker):
    wpath = os.path.join(RUN_DIR, marker["run"] + ".pt")
    assert sha256_file(wpath) == marker["weights_sha256"], f"{marker['run']}: checkpoint does not match its recorded hash"
    model = T.create(marker["model"], len(CLASSES))
    model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True), strict=True)
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mean = torch.tensor(marker["mean"], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(marker["std"], device=DEVICE).view(1, 3, 1, 1)
    T.gc_layer_check(model, marker["model"], DEVICE)
    return model, mean, std

@torch.no_grad()
def logits_u8(model, mean, std, u8):
    """Logits (float32, on the device) for uint8 images already on the device, in batches."""
    out = []
    for i in range(0, len(u8), BATCH):
        out.append(model(T.to_input(u8[i:i + BATCH], mean, std)).float())
    z = torch.cat(out)
    assert torch.isfinite(z).all()
    return z

def logits_kind(model, mean, std, kind, r, d=None):
    out = []
    for i in range(0, len(r), 256):
        out.append(logits_u8(model, mean, std, T.build(C.fetch(kind, r[i:i + 256], None if d is None else d[i:i + 256]), DEVICE)))
    return torch.cat(out)

def prob_and_margin(z, c):
    """Probability of class c and logit margin (logit of c minus the largest other logit), per image."""
    p = z.softmax(1).gather(1, c.view(-1, 1))[:, 0]
    zc = z.gather(1, c.view(-1, 1))[:, 0]
    other = z.scatter(1, c.view(-1, 1), float("-inf")).amax(1)
    return p.double(), (zc - other).double()

def acc_f1(y, pred):
    f1 = []
    for c in np.union1d(np.unique(y), np.unique(pred)):
        tp = int(((pred == c) & (y == c)).sum()); fn = int(((pred != c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum())
        f1.append(2 * tp / (2 * tp + fn + fp))
    return float((pred == y).mean()), float(np.mean(f1))

def donors_for(pos, n_pool, k, rng):
    """k donors per image, uniform over the pool of n_pool images without the image itself. pos: position of each
    image in the pool. Returns positions in the pool, shape (len(pos), k)."""
    d = rng.integers(0, n_pool - 1, (len(pos), k))
    return d + (d >= np.asarray(pos)[:, None])

def cmean(v, y):
    """Unweighted mean of the class means (plan section 7 weighting), used for every effect reported here."""
    return float(np.mean([np.mean(v[y == c]) for c in np.unique(y)]))

# %%
# STAGE checks. Validation images only. Nothing here reads a test image.
if STAGE == "checks":
    rng = np.random.default_rng(SEED + 21)
    chk = pd.concat([g.sample(min(len(g), CHECK_PER_CLASS), random_state=SEED + 21) for _, g in val.groupby("class")]).sort_values("idx").reset_index(drop=True)
    r_chk = chk.idx.to_numpy(dtype=np.int64); y_chk = label_of(r_chk); n = len(r_chk)
    r_val = val.idx.to_numpy(dtype=np.int64); y_val = label_of(r_val)
    verify_rows(r_val)
    pos_chk = np.searchsorted(r_val, r_chk)               # position of each check image in the validation partition
    assert (r_val[pos_chk] == r_chk).all()
    d_chk = r_val[donors_for(pos_chk, len(r_val), N_SWAPS, rng)]    # cache rows of the donors, (n, 5): whole validation partition, never the image itself
    print("check subset:", n, "validation images (up to", CHECK_PER_CLASS, "per class) |", N_SWAPS, "fixed donors each from all", len(r_val), "validation images")

    # Images that do not depend on the model, built once on the CPU with the reference code (p3_common)
    t0 = time.time()
    rgb, msk, bg = (T.read_rows(a, r_chk) for a in (C.rgb, C.mask, C.bg))
    rgb_d, msk_d, bg_d = (T.read_rows(a, d_chk[:, 0]) for a in (C.rgb, C.mask, C.bg))      # the first donor of each image
    bg_alt = np.stack([P.filled_background(rgb[i], msk[i], P.ALT_SIGMA_FRAC) for i in range(n)])        # fill sigma 1/4
    bg_alt_d = np.stack([P.filled_background(rgb_d[i], msk_d[i], P.ALT_SIGMA_FRAC) for i in range(n)])
    art = {  # own-regime swap with the first donor, three ways (deviation 2): default, fill sigma 1/4, feathered edge
        "BG": {"default": np.stack([P.compose(rgb_d[i], msk_d[i], bg[i]) for i in range(n)]),
               "sigma": np.stack([P.compose(rgb_d[i], msk_d[i], bg_alt[i]) for i in range(n)]),
               "feather": np.stack([P.compose(rgb_d[i], msk_d[i], bg[i], P.FEATHER_PX) for i in range(n)])},
        "LEAF": {"default": np.stack([P.compose(rgb[i], msk[i], bg_d[i]) for i in range(n)]),
                 "sigma": np.stack([P.compose(rgb[i], msk[i], bg_alt_d[i]) for i in range(n)]),
                 "feather": np.stack([P.compose(rgb[i], msk[i], bg_d[i], P.FEATHER_PX) for i in range(n)])}}
    del rgb_d, msk_d, bg_d, bg_alt_d
    assert (art["BG"]["default"] == T.build(C.fetch("BG", r_chk, d_chk[:, 0]), "cpu").numpy()).all()
    assert (art["LEAF"]["default"] == T.build(C.fetch("LEAF", r_chk, d_chk[:, 0]), "cpu").numpy()).all()
    # Descriptive checks (addition to the construction checks): outline only, background texture only
    outline = np.where(msk[..., None], np.uint8(64), np.uint8(128)).repeat(3, axis=-1)
    q = IMG // 4
    corners = [(slice(0, q), slice(0, q)), (slice(0, q), slice(IMG - q, IMG)), (slice(IMG - q, IMG), slice(0, q)), (slice(IMG - q, IMG), slice(IMG - q, IMG))]
    texture, tex_ok = np.zeros_like(rgb), np.zeros(n, dtype=bool)
    for i in range(n):
        cover = [int(msk[i][c].sum()) for c in corners]
        k = int(np.argmin(cover))
        tex_ok[i] = cover[k] == 0
        texture[i] = np.tile(rgb[i][corners[k]], (4, 4, 1))
    print(f"model-independent check images built in {time.time() - t0:.0f} s | texture-only usable: {int(tex_ok.sum())} of {n}")
    dev = lambda a: torch.from_numpy(a).to(DEVICE)

    rows_c, rows_cls = [], []
    for k_run, mk in enumerate(RUNS):
        t0 = tick()
        model, mean, std = load_model(mk)
        z_val = logits_kind(model, mean, std, "ORIG", r_val)
        acc_val, f1_val = acc_f1(y_val, z_val.argmax(1).cpu().numpy())
        z0 = logits_kind(model, mean, std, "ORIG", r_chk)
        c = z0.argmax(1)
        p0, m0 = prob_and_margin(z0, c)
        eff = {}
        for kind, key in (("LEAF", "background"), ("BG", "leaf")):      # LEAF kind = background swap; BG kind = leaf swap
            dp, dm = torch.zeros(n, dtype=torch.float64, device=DEVICE), torch.zeros(n, dtype=torch.float64, device=DEVICE)
            for s in range(N_SWAPS):
                p, m = prob_and_margin(logits_kind(model, mean, std, kind, r_chk, d_chk[:, s]), c)
                dp += (p - p0).abs(); dm += (m - m0).abs()
            eff[key + "_effect"] = (dp / N_SWAPS).cpu().numpy(); eff[key + "_effect_margin"] = (dm / N_SWAPS).cpu().numpy()
        for reg in ("BG", "LEAF"):
            pa, _ = prob_and_margin(logits_u8(model, mean, std, dev(art[reg]["default"])), c)
            for var in ("sigma", "feather"):
                pv, _ = prob_and_margin(logits_u8(model, mean, std, dev(art[reg][var])), c)
                eff[f"artefact_{reg}_{var}"] = (pv - pa).abs().cpu().numpy()
        pred_outline = logits_u8(model, mean, std, dev(outline)).argmax(1).cpu().numpy()
        pred_texture = logits_u8(model, mean, std, dev(texture)).argmax(1).cpu().numpy()
        pred_removed = logits_kind(model, mean, std, "REMOVED", r_chk).argmax(1).cpu().numpy()
        reg = mk["regime"]
        own = reg if reg in ("BG", "LEAF") else None
        # gate value: the larger of the two (class-weighted) mean changes; per image, for the tail: the larger change
        art_eff = {r_: max(cmean(eff[f"artefact_{r_}_sigma"], y_chk), cmean(eff[f"artefact_{r_}_feather"], y_chk)) for r_ in ("BG", "LEAF")}
        art_img = {r_: np.maximum(eff[f"artefact_{r_}_sigma"], eff[f"artefact_{r_}_feather"]) for r_ in ("BG", "LEAF")}
        row = {"run": mk["run"], "model": mk["model"], "family": mk["family"], "regime": reg, "seed": mk["seed"],
               "weights_sha256": mk["weights_sha256"], "mean": json.dumps(mk["mean"]), "std": json.dumps(mk["std"]),
               "val_accuracy_original": acc_val, "val_macro_f1_original": f1_val, "n_check_images": n,
               "background_effect": cmean(eff["background_effect"], y_chk), "leaf_effect": cmean(eff["leaf_effect"], y_chk),
               "background_effect_margin": cmean(eff["background_effect_margin"], y_chk), "leaf_effect_margin": cmean(eff["leaf_effect_margin"], y_chk),
               "artefact_effect_BG_swap": art_eff["BG"], "artefact_effect_LEAF_swap": art_eff["LEAF"],
               "artefact_effect_own_regime": art_eff[own] if own else float("nan"),
               "artefact_BG_sigma": cmean(eff["artefact_BG_sigma"], y_chk), "artefact_BG_feather": cmean(eff["artefact_BG_feather"], y_chk),
               "artefact_LEAF_sigma": cmean(eff["artefact_LEAF_sigma"], y_chk), "artefact_LEAF_feather": cmean(eff["artefact_LEAF_feather"], y_chk),
               "background_effect_share_below_0.05": float((eff["background_effect"] < TAIL).mean()),
               "leaf_effect_share_below_0.05": float((eff["leaf_effect"] < TAIL).mean()),
               "artefact_own_regime_share_above_0.10": float((art_img[own] > 0.10).mean()) if own else float("nan"),
               "accuracy_check_original": float((c.cpu().numpy() == y_chk).mean()),
               "accuracy_outline_only": float((pred_outline == y_chk).mean()),
               "accuracy_texture_only": float((pred_texture[tex_ok] == y_chk[tex_ok]).mean()) if tex_ok.any() else float("nan"),
               "n_texture_only": int(tex_ok.sum()), "accuracy_leaf_removed": float((pred_removed == y_chk).mean())}
        if own:
            rule = RULES[own]
            own_eff, other_eff = (row["background_effect"], row["leaf_effect"]) if own == "BG" else (row["leaf_effect"], row["background_effect"])
            row.update({"pass_accuracy": acc_val >= rule["acc_min"], "pass_own_effect": own_eff >= rule["own_effect_min"],
                        "pass_other_effect": other_eff <= rule["other_effect_max"], "pass_artefact": row["artefact_effect_own_regime"] <= rule["artefact_max"]})
            row["pass"] = bool(row["pass_accuracy"] and row["pass_own_effect"] and row["pass_other_effect"] and row["pass_artefact"])
        rows_c.append(row)
        for ci, cname in enumerate(CLASSES):
            sel = y_chk == ci
            rows_cls.append({"run": mk["run"], "class": cname, "n": int(sel.sum()),
                             "background_effect": float(eff["background_effect"][sel].mean()), "leaf_effect": float(eff["leaf_effect"][sel].mean()),
                             "artefact_BG_swap": float(art_img["BG"][sel].mean()), "artefact_LEAF_swap": float(art_img["LEAF"][sel].mean()),
                             "background_effect_share_below_0.05": float((eff["background_effect"][sel] < TAIL).mean()),
                             "leaf_effect_share_below_0.05": float((eff["leaf_effect"][sel] < TAIL).mean()),
                             "accuracy_original": float((c.cpu().numpy()[sel] == ci).mean())})
        del model
        print(f"[{k_run + 1}/{len(RUNS)}] {mk['run']}: val acc {acc_val:.3f} | bg effect {row['background_effect']:.3f} | leaf effect {row['leaf_effect']:.3f} "
              f"| artefact {row['artefact_effect_own_regime']:.3f} | pass {row.get('pass', '-')} | {tick() - t0:.0f} s", flush=True)

    checks = pd.DataFrame(rows_c)
    # Eligibility (plan section 4): a seed pair is eligible if its BG and its LEAF checkpoint both pass; an architecture
    # is eligible if at least two of its three seed pairs are.
    pairs = (checks[checks.regime.isin(["BG", "LEAF"])].pivot_table(index=["model", "family", "seed"], columns="regime", values="pass", aggfunc="first")
             .reset_index().rename(columns={"BG": "bg_pass", "LEAF": "leaf_pass"}))
    for reg_, col in (("BG", "bg"), ("LEAF", "leaf")):      # the checked checkpoints are named in the frozen table itself
        w = checks[checks.regime == reg_].set_index(["model", "seed"])
        pairs[col + "_run"] = [w.run.at[(m_, s_)] for m_, s_ in zip(pairs.model, pairs.seed)]
        pairs[col + "_weights_sha256"] = [w.weights_sha256.at[(m_, s_)] for m_, s_ in zip(pairs.model, pairs.seed)]
    pairs["pair_eligible"] = pairs.bg_pass.astype(bool) & pairs.leaf_pass.astype(bool)
    n_ok = pairs.groupby("model").pair_eligible.transform("sum")
    pairs["architecture_eligible"] = n_ok >= (2 if not SMOKE else 1)
    pairs["used_in_outcomes"] = pairs.pair_eligible & pairs.architecture_eligible
    print(checks[["run", "val_accuracy_original", "background_effect", "leaf_effect", "artefact_effect_own_regime", "pass"]].to_string(index=False))
    print(pairs.to_string(index=False))
    fam_ok = pairs[pairs.architecture_eligible].groupby("family").model.nunique()
    print("eligible architectures by family:", fam_ok.to_dict(), "| primary question answerable:",
          bool(fam_ok.get("cnn", 0) >= 1 and fam_ok.get("transformer", 0) >= 1))
    if not SMOKE:
        save_output(pd.DataFrame({"idx": r_chk, **{f"donor_idx_{s + 1}": d_chk[:, s] for s in range(N_SWAPS)}, "texture_only_usable": tex_ok}),
                    "check_subset.csv", "Validation images used for the construction checks and their five fixed donors (cache rows); whether the texture-only image could be built", "checks")
        save_output(checks, "construction_checks.csv", "Construction checks per checkpoint on validation images: accuracy on originals, background, leaf and artefact effects (probability and logit margin), tail shares, descriptive outline-only, texture-only and leaf-removed accuracy, pass flags", "checks")
        save_output(pd.DataFrame(rows_cls), "construction_checks_by_class.csv", "Background and leaf effects and accuracy by class, per checkpoint (validation check subset)", "checks")
        save_output(pairs, "model_eligibility.csv", "Eligibility per architecture and seed pair, fixed before any test image is scored (plan section 4)", "checks")
        save_output({**ENV, "rules": RULES, "n_swaps": N_SWAPS, "check_per_class": CHECK_PER_CLASS, "frozen_inputs": FROZEN,
                     "weighting": "every effect is the unweighted mean of class means over the check subset",
                     "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
                    "checks_environment.json", "Environment and fixed rules of the construction-check stage", "checks")
    print("STAGE checks done. Commit outputs/model_eligibility.csv and write its hash into notebooks/p3_frozen.json before STAGE=attr.")

# %%
# STAGE attr: building blocks. Runs only with the frozen eligibility table.
if STAGE == "attr":
    if not SMOKE:
        assert sha256_file(os.path.join(OUT, "model_eligibility.csv")) == FROZEN["eligibility_sha256"], \
            "model_eligibility.csv is not the frozen table: no test image is scored before the construction checks are frozen"
        assert sha256_file(os.path.join(OUT, "construction_checks.csv")) == FROZEN["construction_checks_sha256"]
        _chk = pd.read_csv(os.path.join(OUT, "construction_checks.csv")).set_index("run")
        for mk in RUNS:     # the checkpoints scored now are the ones that were checked: same weights, same normalisation
            assert mk["run"] in _chk.index, f"{mk['run']} was not in the construction checks"
            assert _chk.weights_sha256.at[mk["run"]] == mk["weights_sha256"] and json.loads(_chk["mean"].at[mk["run"]]) == mk["mean"] \
                and json.loads(_chk["std"].at[mk["run"]]) == mk["std"], f"{mk['run']}: checkpoint or normalisation changed since the checks"
        attr_path = os.path.join(OUT, "attribution_subset.csv")
        assert sha256_file(attr_path) == FROZEN["attribution_sha256"]
        sub = pd.read_csv(attr_path).sort_values("idx").reset_index(drop=True)
    else:
        sub = pd.concat([g.head(2) for _, g in split[split.partition == "val"].groupby("class")]).sort_values("idx").reset_index(drop=True)   # smoke: validation images, never test
    r_att = sub.idx.to_numpy(dtype=np.int64); y_att = label_of(r_att); n = len(r_att)
    test = split[split.partition == ("test" if not SMOKE else "val")].sort_values("idx").reset_index(drop=True)
    if SMOKE:
        test = pd.concat([g.head(4) for _, g in test.groupby("class")]).sort_values("idx").reset_index(drop=True)
    r_test = test.idx.to_numpy(dtype=np.int64); y_test = label_of(r_test)
    verify_rows(r_test)
    assert np.isin(r_att, r_test).all() or SMOKE
    rng = np.random.default_rng(SEED + 31)
    pos_att = np.searchsorted(r_test, r_att)
    D_att = r_test[donors_for(pos_att, len(r_test), N_SWAPS, rng)]      # fixed donors from the test partition, never the image itself, (n, 5)
    D_test = r_test[donors_for(np.arange(len(r_test)), len(r_test), 1, rng)[:, 0]]   # one fixed donor per test image
    y_test_donor = label_of(D_test)
    M_att = torch.from_numpy(T.read_rows(C.mask, r_att)).to(DEVICE)
    AREA = T.area_off_leaf(M_att).cpu().numpy()                           # a: the background's area share, from pixel counts
    examples = np.sort(np.random.default_rng(SEED + 33).choice(n, min(N_EXAMPLES, n), replace=False))
    tie_rank = torch.stack([torch.from_numpy(np.random.default_rng([SEED, int(i)]).permutation(IMG * IMG)) for i in r_att]).to(DEVICE)

    _rr = int(np.ceil(3 * DEL_BLUR_SIGMA))
    _kk = torch.exp(-0.5 * (torch.arange(-_rr, _rr + 1, dtype=torch.float32) / DEL_BLUR_SIGMA) ** 2)
    BLUR_K = (_kk / _kk.sum()).to(DEVICE)
    def blur(x):                 # x: (N, 3, 224, 224) in [0, 1]; separable Gaussian, reflect padding
        c = x.shape[1]
        y = F.conv2d(F.pad(x, (_rr, _rr, 0, 0), mode="reflect"), BLUR_K.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
        return F.conv2d(F.pad(y, (0, 0, _rr, _rr), mode="reflect"), BLUR_K.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)

    def maps_for(model, mean, std, name, family, u8, targets):
        """Normalised maps (N, 224, 224) per method for the given target classes, with zero-mass flags, the native
        Grad-CAM map and the Integrated Gradients completeness values."""
        out, raw = {}, {}
        x0n = (torch.zeros(1, 3, IMG, IMG, device=DEVICE) - mean) / std          # black in RGB space, then normalised
        gc_n, ig_n, ig_s, ig_d, ro_n, ta_n = [], [], [], [], [], []
        for i in range(0, len(u8), 32):
            xn = T.to_input(u8[i:i + 32], mean, std)
            t = targets[i:i + 32]
            gc_n.append(T.gradcam(model, xn, t, name))
            m, s_, d_ = T.integrated_gradients(model, xn, x0n, t)
            ig_n.append(m); ig_s.append(s_); ig_d.append(d_)
            if family == "transformer":
                ro_n.append(T.rollout(model, xn)); ta_n.append(T.transformer_attribution(model, xn, t))
        raw["GC"] = torch.cat(gc_n)
        out["GC"] = T.normalise(raw["GC"]); out["IG"] = T.normalise(torch.cat(ig_n))
        if family == "transformer":
            out["ROLL"] = T.normalise(torch.cat(ro_n)); out["TA"] = T.normalise(torch.cat(ta_n))
        return out, raw, torch.cat(ig_s).double().cpu().numpy(), torch.cat(ig_d).double().cpu().numpy()

    @torch.no_grad()
    def curves(model, mean, std, u8, target, maps, fills):
        """Deletion and insertion areas under the curve for one image. maps: {name: (224, 224)}; most relevant pixels
        first, ties by the image's fixed random ranking; DEL_STEPS steps of 1/DEL_STEPS of the pixels."""
        x = u8.permute(2, 0, 1).float().div(255)[None]
        fill_img = {"blur": blur(x)[0], "mean": x.mean(dim=(2, 3), keepdim=True).expand_as(x)[0]}
        ks = torch.arange(0, DEL_STEPS + 1, device=DEVICE) * (IMG * IMG // DEL_STEPS)
        xs = torch.arange(0, DEL_STEPS + 1, dtype=torch.float64) / DEL_STEPS
        res = {}
        for mname, (m, tb) in maps.items():
            order = T.saliency_order(m, tb)                                                   # decreasing map value; exact ties by tb
            rank = torch.empty(IMG * IMG, dtype=torch.long, device=DEVICE)
            rank[order] = torch.arange(IMG * IMG, device=DEVICE)
            gone = (rank.view(1, 1, IMG, IMG) < ks.view(-1, 1, 1, 1))                         # (29, 1, 224, 224): pixel affected at step k
            for fname in fills:
                f = fill_img[fname][None]
                batch = torch.cat([torch.where(gone, f, x), torch.where(gone, x, f)])           # deletion, insertion
                p = []
                for i in range(0, len(batch), BATCH):
                    p.append(model((batch[i:i + BATCH] - mean) / std).float().softmax(1)[:, target])
                p = torch.cat(p).double().cpu().view(2, -1)
                assert torch.isfinite(p).all()
                res[(mname, fname)] = (float(torch.trapezoid(p[0], xs)), float(torch.trapezoid(p[1], xs)), p.numpy())
        return res

    # The ordering rule on tiny cases: larger value first; exactly equal values in the order of the tie ranking; values
    # that differ by less than any fixed tolerance still keep their order
    assert T.saliency_order(torch.tensor([0.5, 0.2, 0.5, 0.0]), torch.tensor([3, 0, 1, 2])).tolist() == [2, 0, 1, 3]
    assert T.saliency_order(torch.tensor([1e-9, 1e-9 + 1e-15, 0.0], dtype=torch.float64), torch.tensor([0, 2, 1])).tolist() == [1, 0, 2]

# %%
# STAGE attr: one checkpoint at a time, resumable
def attr_signature(mk):
    return hashlib.sha256(json.dumps([mk["weights_sha256"], FROZEN.get("split_sha256"), FROZEN.get("attribution_sha256"),
                                      FROZEN.get("eligibility_sha256"), sha256_file(os.path.join(ROOT, "notebooks", "src", "03_attribution_GPU.py")),
                                      sha256_file(os.path.join(ROOT, "notebooks", "p3_torch.py")), sha256_file(os.path.join(ROOT, "notebooks", "p3_common.py")),
                                      FILLS, SMOKE, DEVICE, VERSIONS, BACKEND, mk["model"], mk["regime"], mk["seed"], mk["mean"], mk["std"],
                                      hashlib.sha256(D_att.tobytes() + D_test.tobytes() + r_att.tobytes() + r_test.tobytes()).hexdigest()],
                                     sort_keys=True).encode()).hexdigest()

def run_attr(mk):
    run, name, family, regime = mk["run"], mk["model"], mk["family"], mk["regime"]
    sig, path = attr_signature(mk), os.path.join(ATTR_DIR, mk["run"] + ".json")
    try:
        done = json.load(open(path))
        ok = done.get("signature") == sig and done.get("run") == run and all(
            os.path.exists(os.path.join(ATTR_DIR, f"{run}_{k}.csv" if k != "npz" else run + ".npz"))
            and sha256_file(os.path.join(ATTR_DIR, f"{run}_{k}.csv" if k != "npz" else run + ".npz")) == h for k, h in done["shards"].items())
        if ok:
            print("resume: attribution found, signature and file hashes verified, skipped:", run)
            return done
    except (OSError, ValueError):
        pass
    t0 = tick()
    model, mean, std = load_model(mk)
    base = {"run": run, "model": name, "family": family, "regime": regime, "seed": mk["seed"]}
    u8 = T.build(C.fetch("ORIG", r_att), DEVICE)
    z = logits_u8(model, mean, std, u8)
    pred = z.argmax(1)
    y_t = torch.from_numpy(y_att).to(DEVICE)
    scores, ex_maps = [], {}
    # maps for the predicted class, and for the true class (sensitivity i); identical where the prediction is right
    per_target = {}
    for tname, tg in (("predicted", pred), ("true", y_t)):
        if tname == "true":
            need = (pred != y_t).nonzero()[:, 0]
            maps_t = {k: (v[0].clone(), v[1].clone()) for k, v in per_target["predicted"][0].items()}
            raw_t = {k: v.clone() for k, v in per_target["predicted"][1].items()}
            if len(need):
                o2, r2, _, _ = maps_for(model, mean, std, name, family, u8[need], tg[need])
                for k in maps_t:
                    if k != "ROLL":                      # rollout does not depend on the class
                        maps_t[k][0][need], maps_t[k][1][need] = o2[k][0], o2[k][1]
                raw_t["GC"][need] = r2["GC"]
            per_target[tname] = (maps_t, raw_t, None, None)
        else:
            per_target[tname] = maps_for(model, mean, std, name, family, u8, tg)
        maps_t, raw_t = per_target[tname][0], per_target[tname][1]
        for method, (m, zero) in maps_t.items():
            if method == "ROLL" and tname == "true":
                continue
            S = T.off_leaf_share(m, M_att, zero).cpu().numpy()
            S_nat = T.off_leaf_share_native(raw_t["GC"], M_att).double().cpu().numpy() if method == "GC" else np.full(n, np.nan)
            for i in range(n):
                scores.append({**base, "idx": int(r_att[i]), "class": CLASSES[y_att[i]], "pred_class": CLASSES[int(pred[i])],
                               "method": method, "target": tname if method != "ROLL" else "none", "S": S[i], "S_native": S_nat[i],
                               "map_status": int(zero[i]), "area_off_leaf": AREA[i], "image": "original"})
            if tname == "predicted" and mk["seed"] == SEEDS[0]:
                ex_maps[method] = F.avg_pool2d(m[examples][:, None], 2)[:, 0].cpu().numpy().astype(np.float32)   # 112 x 112
    ig_signed, ig_delta = per_target["predicted"][2], per_target["predicted"][3]

    # sensitivity (iii): BG and LEAF models explained on images built by their own regime (first fixed donor)
    if regime in ("BG", "LEAF"):
        u8o = T.build(C.fetch(regime, r_att, D_att[:, 0]), DEVICE)
        M_o = torch.from_numpy(T.read_rows(C.mask, D_att[:, 0])).to(DEVICE) if regime == "BG" else M_att    # the leaf shown is the donor's in a leaf swap
        area_o = T.area_off_leaf(M_o).cpu().numpy()
        pred_o = logits_u8(model, mean, std, u8o).argmax(1)
        maps_o, raw_o, _, _ = maps_for(model, mean, std, name, family, u8o, pred_o)
        for method, (m, zero) in maps_o.items():
            S = T.off_leaf_share(m, M_o, zero).cpu().numpy()
            for i in range(n):
                scores.append({**base, "idx": int(r_att[i]), "class": CLASSES[y_att[i]], "pred_class": CLASSES[int(pred_o[i])],
                               "method": method, "target": "predicted" if method != "ROLL" else "none", "S": S[i], "S_native": np.nan,
                               "map_status": int(zero[i]), "area_off_leaf": area_o[i], "image": "own_regime"})
        del u8o

    # background and leaf effect of every image of the subset (secondary A uses the ORIG models)
    p0, m0 = prob_and_margin(z, pred)
    eff = {}
    for kind, key in (("LEAF", "background"), ("BG", "leaf")):
        dp, dm = torch.zeros(n, dtype=torch.float64, device=DEVICE), torch.zeros(n, dtype=torch.float64, device=DEVICE)
        for s in range(N_SWAPS):
            p, m = prob_and_margin(logits_kind(model, mean, std, kind, r_att, D_att[:, s]), pred)
            dp += (p - p0).abs(); dm += (m - m0).abs()
        eff[key + "_effect"] = (dp / N_SWAPS).cpu().numpy(); eff[key + "_effect_margin"] = (dm / N_SWAPS).cpu().numpy()
    effects = [{**base, "idx": int(r_att[i]), "class": CLASSES[y_att[i]], "pred_class": CLASSES[int(pred[i])], "p_pred": float(p0[i]),
                **{k: float(v[i]) for k, v in eff.items()}, "ig_signed_sum": float(ig_signed[i]), "ig_logit_difference": float(ig_delta[i])} for i in range(n)]

    # classification on the test partition (plan section 6): original, background swapped, leaf swapped, leaf removed
    cls_rows, pred_rows = [], {"idx": r_test}
    for kind, label in (("ORIG", "original"), ("LEAF", "background_swapped"), ("BG", "leaf_swapped"), ("REMOVED", "leaf_removed")):
        pr = logits_kind(model, mean, std, kind, r_test, D_test).argmax(1).cpu().numpy()
        a_img, f_img = acc_f1(y_test, pr)
        pred_rows["pred_" + label] = pr
        row = {**base, "images": label, "n": int(len(pr)), "accuracy_image_label": a_img, "macro_f1_image_label": f_img,
               "accuracy_donor_label": float((pr == y_test_donor).mean()) if kind in ("LEAF", "BG") else float("nan")}
        cls_rows.append(row)

    # deletion and insertion (secondary B): ORIG models only, predicted class, methods and control maps
    di_rows, curve_sum, curve_n = [], {}, 0
    if regime == "ORIG":
        ctrl = T.control_maps(u8, M_att)
        pm = per_target["predicted"][0]
        for i in range(n):
            maps_i = {k: (v[0][i], tie_rank[i]) for k, v in pm.items()}
            maps_i.update({k: (v[i], tie_rank[i]) for k, v in ctrl.items()})
            for (mname, fname), (d_auc, i_auc, ordinates) in curves(model, mean, std, u8[i], int(pred[i]), maps_i, FILLS).items():
                di_rows.append({**base, "idx": int(r_att[i]), "class": CLASSES[y_att[i]], "map": mname, "fill": fname,
                                "deletion_auc": d_auc, "insertion_auc": i_auc})
                curve_sum[(mname, fname)] = curve_sum.get((mname, fname), 0) + ordinates     # (2, 29): deletion, insertion
            curve_n += 1
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    npz = os.path.join(ATTR_DIR, run + ".npz")
    def w(t):
        with open(t, "wb") as f:
            np.savez_compressed(f, **{"example_" + k: v for k, v in ex_maps.items()})
    atomic(npz, w)
    curve_rows = [{**base, "map": mname, "fill": fname, "curve": cname, "step": k, "mean_probability": float(v[ci][k] / curve_n)}
                  for (mname, fname), v in curve_sum.items() for ci, cname in enumerate(("deletion", "insertion")) for k in range(DEL_STEPS + 1)]
    preds = pd.DataFrame(pred_rows).assign(run=run)
    shards = {}
    for key, rows_ in (("scores", scores), ("effects", effects), ("classification", cls_rows), ("deletion_insertion", di_rows),
                       ("curves", curve_rows), ("test_predictions", preds)):
        f = os.path.join(ATTR_DIR, f"{run}_{key}.csv")
        atomic(f, lambda t, rr=rows_: (rr if hasattr(rr, "to_csv") else pd.DataFrame(rr)).to_csv(t, index=False, lineterminator="\n"))
        shards[key] = sha256_file(f)
    shards["npz"] = sha256_file(npz)
    done = {"run": run, "signature": sig, "seconds": tick() - t0, "n_images": n, "has_examples": bool(ex_maps), "shards": shards}
    write_json(path, done)                                   # completion marker last
    print(f"{run}: done in {done['seconds'] / 60:.1f} min", flush=True)
    return done

if STAGE == "attr":
    t_all = time.time()
    order = sorted(RUNS, key=lambda m: (m["seed"], {"BG": 0, "LEAF": 1, "ORIG": 2}[m["regime"]]))
    dones = []
    for k, mk in enumerate(order):
        print(f"=== {mk['run']} ({k + 1}/{len(order)}) ===", flush=True)
        dones.append(run_attr(mk))

# %%
# STAGE attr: assemble outputs
if STAGE == "attr":
    def gather(key):
        frames = [pd.read_csv(os.path.join(ATTR_DIR, f"{mk['run']}_{key}.csv")) for mk in RUNS if os.path.getsize(os.path.join(ATTR_DIR, f"{mk['run']}_{key}.csv")) > 2]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    scores, effects, cls_tab, di = gather("scores"), gather("effects"), gather("classification"), gather("deletion_insertion")
    curve_tab, test_preds = gather("curves"), gather("test_predictions")
    # Strict assembly (Codex review 003, point 5): exactly the expected groups, each with every image once, all finite
    expected = set()
    for mk in RUNS:
        methods = ["GC", "IG"] + (["ROLL", "TA"] if mk["family"] == "transformer" else [])
        for image in ["original"] + (["own_regime"] if mk["regime"] in ("BG", "LEAF") else []):
            for mth in methods:
                for tg_ in (["none"] if mth == "ROLL" else (["predicted", "true"] if image == "original" else ["predicted"])):
                    expected.add((mk["run"], mth, tg_, image))
    got = scores.groupby(["run", "method", "target", "image"]).idx.agg(["size", "nunique"])
    assert set(got.index) == expected, f"score groups differ from the expected set: {set(got.index) ^ expected}"
    assert (got["size"] == n).all() and (got["nunique"] == n).all(), "a score group does not hold every image exactly once"
    assert np.isfinite(scores.S).all() and scores.S.between(-1e-9, 1 + 1e-9).all()
    nat = (scores.method == "GC") & (scores.image == "original")     # the native-grid share exists for Grad-CAM on original images only
    assert np.isfinite(scores.loc[nat, "S_native"]).all() and scores.loc[~nat, "S_native"].isna().all()
    assert (scores.loc[scores.map_status > 0, "S"] == scores.loc[scores.map_status > 0, "area_off_leaf"]).all(), "a replaced map is not an exact tie with the uniform map"
    assert effects.groupby("run").idx.nunique().eq(n).all() and len(effects) == n * len(RUNS)
    assert len(cls_tab) == 4 * len(RUNS) and len(test_preds) == len(r_test) * len(RUNS)
    n_orig = sum(mk["regime"] == "ORIG" for mk in RUNS)
    if n_orig:
        assert np.isfinite(di[["deletion_auc", "insertion_auc"]].to_numpy()).all()
        assert di.groupby(["run", "map", "fill"]).idx.nunique().eq(n).all() and di.run.nunique() == n_orig
    zero_tab = (scores.assign(zero_mass=scores.map_status == 1, invalid=scores.map_status == 2)
                .groupby(["run", "method", "target", "image"])[["zero_mass", "invalid"]].sum().reset_index()
                .rename(columns={"zero_mass": "zero_mass_maps", "invalid": "non_finite_maps"}).assign(maps=n))
    ig = effects.assign(rel_error=lambda t: (t.ig_signed_sum - t.ig_logit_difference).abs() / t.ig_logit_difference.abs().clip(lower=1e-6))
    ig_tab = ig.groupby("run").rel_error.agg(["median", lambda s: s.quantile(0.95)]).reset_index().rename(columns={"median": "ig_completeness_rel_error_median", "<lambda_0>": "ig_completeness_rel_error_p95"})
    ex = {}
    for mk in RUNS:
        if mk["seed"] == SEEDS[0]:
            z_ = np.load(os.path.join(ATTR_DIR, mk["run"] + ".npz"))
            for k in z_.files:
                ex[f"{mk['model']}__{mk['regime']}__{k[len('example_'):]}"] = z_[k]
    ex["example_idx"] = r_att[examples]
    print(cls_tab.to_string(index=False))
    print(zero_tab[(zero_tab.zero_mass_maps > 0) | (zero_tab.non_finite_maps > 0)].to_string(index=False))
    if SMOKE:
        print("SMOKE TEST OK:", len(scores), "score rows;", len(di), "curve rows; nothing written to outputs/.")
    else:
        save_output(scores, "attribution_scores.csv.gz", "Off-leaf share S per checkpoint, image of the attribution subset, method and target class (predicted or true); native-grid S for Grad-CAM; map status (0 used, 1 zero mass, 2 non-finite); background area share; original images and own-regime images (sensitivity iii)", "assemble")
        save_output(effects, "image_effects.csv.gz", "Per checkpoint and image of the attribution subset: predicted class and its probability, background and leaf effects over five fixed swaps (probability and logit margin), Integrated Gradients completeness values", "assemble")
        save_output(cls_tab, "classification_test.csv", "Test-partition accuracy and macro F1 per checkpoint on original, background-swapped, leaf-swapped and leaf-removed images; for swaps also accuracy against the donor's label", "assemble")
        save_output(di, "deletion_insertion.csv.gz", "Deletion and insertion areas under the curve, ORIG checkpoints, per image, map (methods and model-independent controls) and fill", "assemble")
        save_output(zero_tab, "zero_mass_maps.csv", "Number of maps with zero mass and with non-finite values (both replaced by the uniform map) per checkpoint, method, target and image kind", "assemble")
        save_output(curve_tab, "deletion_insertion_curves.csv", "Mean deletion and insertion curves (probability of the predicted class at each of 29 steps) per ORIG checkpoint, map and fill", "assemble")
        save_output(test_preds, "test_predictions.csv.gz", "Predicted class index per checkpoint and test image on original, background-swapped, leaf-swapped and leaf-removed images (class order = sorted class names)", "assemble")
        save_output(pd.DataFrame({"idx": r_test, "donor_idx": D_test}), "test_donors.csv", "Fixed donor (cache row) of every test image for the swapped test images", "assemble")
        save_output(ig_tab, "ig_completeness.csv", "Integrated Gradients completeness: relative difference between the summed attribution and the logit difference to the black baseline, median and 95th percentile per checkpoint", "assemble")
        save_output(ex, "example_maps.npz", "Maps (112 x 112, predicted class) of six fixed images of the attribution subset for the seed-11 checkpoints, for the example figure", "assemble")
        save_output(pd.DataFrame({"idx": r_att, **{f"donor_idx_{s + 1}": D_att[:, s] for s in range(N_SWAPS)}}), "attribution_donors.csv", "Fixed donors (cache rows) of the attribution-subset images for swaps and own-regime images", "assemble")
        save_output({**ENV, "del_steps": DEL_STEPS, "del_blur_sigma_px": DEL_BLUR_SIGMA, "ig_steps": T.IG_STEPS, "gc_layers": T.GC_LAYER,
                     "frozen": FROZEN, "minutes_this_session": (time.time() - t_all) / 60,
                     "minutes_all_checkpoints": float(sum(d["seconds"] for d in dones)) / 60,
                     "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
                    "attribution_environment.json", "Environment, fixed settings, frozen input hashes and timing of the attribution stage", "assemble")
