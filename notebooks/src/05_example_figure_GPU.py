# %% [markdown]
# # RUNTIME: local workstation GPU (NVIDIA RTX 3060 12 GB), `.venv-local` (plan section 10)
#
# Project: `qpain/paper-03` - Stage: `05 example-map figure (Fig. 3)` - Estimated time: under 2 min
#
# Illustration only, no test and no new estimate (analysis_plan.md section 10, deviation 6 as revised on 2026-11-23).
# Rules fixed before any map was looked at:
# - architecture and seed pair: ViT-S/16, seed pair 22 (the lower-numbered of the two eligible pairs);
# - images: the first three, in order of `idx`, of the six example images drawn by notebook 03 before training
#   (`example_idx` in outputs/example_maps.npz); no image is chosen by its map;
# - maps: predicted class, original image, the same functions and settings as notebook 03 (float32, no autocast).
# The maps saved by notebook 03 are for seed 11 (a pair that is not eligible), so the six maps are recomputed here.
# Every recomputed off-leaf share must equal the value in outputs/attribution_scores.csv (tolerance 1e-3), or the
# notebook stops.
# `SMOKE=1`: computes the maps, runs every check and draws the figure in memory, but writes nothing to outputs/.
# Source: `notebooks/src/05_example_figure_GPU.py`.

# %%
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, json, hashlib, platform, datetime
import importlib.metadata as ilmd
import numpy as np, pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("PAPER_ROOT", os.path.abspath(".."))
OUT, CKPT, CACHE = (os.path.join(ROOT, d) for d in ("outputs", "checkpoints", "data/cache"))
sys.path.insert(0, os.path.join(ROOT, "notebooks"))
import p3_common as P
import p3_torch as T
NOTEBOOK = "05_example_figure_GPU"
SMOKE = os.environ.get("SMOKE", "0") == "1"
SEED = 20261122            # as notebook 03
N_EXAMPLES = 6             # as notebook 03
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
RUN_DIR = os.path.join(CKPT, "runs")
FROZEN = json.load(open(os.path.join(ROOT, "notebooks", "p3_frozen.json")))
MODEL, FAMILY, SEED_PAIR, N_SHOWN = "vit_small_patch16_224", "transformer", 22, 3
METHODS = ["GC", "IG", "ROLL", "TA"]
METHOD_LABEL = {"GC": "Grad-CAM", "IG": "Integrated\nGradients", "ROLL": "Rollout", "TA": "Transformer\nattribution"}
IMG = P.SIDE
TOL = 1e-3

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def manifest(filename, description, cell):
    man = os.path.join(OUT, "MANIFEST.md")
    rows = [l for l in open(man).read().splitlines() if not l.startswith(f"| {filename} |")]
    rows.append(f"| {filename} | {NOTEBOOK} | {cell} | {description} |")
    tmp = man + ".tmp"
    with open(tmp, "w", newline="\n") as f:
        f.write("\n".join(rows) + "\n")
    os.replace(tmp, man)
    print("saved", filename)

# %%
# Frozen inputs and the fixed choice of images
elig = pd.read_csv(os.path.join(OUT, "model_eligibility.csv"))
assert sha256_file(os.path.join(OUT, "model_eligibility.csv")) == FROZEN["eligibility_sha256"]
e = elig[(elig.model == MODEL) & (elig.seed == SEED_PAIR)]
assert len(e) == 1 and bool(e.pair_eligible.iloc[0]) and bool(e.used_in_outcomes.iloc[0]), "the shown seed pair must be an eligible pair"
assert SEED_PAIR == int(elig[(elig.model == MODEL) & elig.used_in_outcomes].seed.min()), "rule: the lower-numbered eligible pair"

split_path = os.path.join(OUT, "pv_split.csv")
assert sha256_file(split_path) == FROZEN["split_sha256"]
split = pd.read_csv(split_path)
CLASSES = sorted(split["class"].unique())
by_idx = split.set_index("idx")
sub = pd.read_csv(os.path.join(OUT, "attribution_subset.csv"))
assert sha256_file(os.path.join(OUT, "attribution_subset.csv")) == FROZEN["attribution_sha256"]
ex_all = np.sort(np.load(os.path.join(OUT, "example_maps.npz"))["example_idx"].astype(np.int64))
r_att = np.sort(sub.idx.to_numpy(dtype=np.int64))
drawn = r_att[np.sort(np.random.default_rng(SEED + 33).choice(len(r_att), N_EXAMPLES, replace=False))]      # the draw of notebook 03
assert len(ex_all) == 6 and (ex_all == drawn).all(), "example images are not the six drawn by notebook 03"
r = ex_all[:N_SHOWN]                         # rule: the first three by idx
assert (by_idx.partition.loc[r] == "test").all()
C = T.Cache(CACHE)
rgb, mask, bg = (T.read_rows(a, r) for a in (C.rgb, C.mask, C.bg))
for j in range(len(r)):
    assert P.cache_row_sha(rgb[j], mask[j], bg[j]) == by_idx.cache_row_sha256.at[r[j]], "cache row differs from the frozen split"
true_class = [by_idx["class"].at[i] for i in r]
print("images (idx):", r.tolist(), "| classes:", true_class)

# %%
# Maps for the predicted class on the original images, BG and LEAF checkpoints of the seed pair
runs = pd.read_csv(os.path.join(OUT, "train_runs.csv"))
scores = pd.read_csv(os.path.join(OUT, "attribution_scores.csv.gz"))
M = torch.from_numpy(mask).to(DEVICE)
u8 = T.build((rgb,), DEVICE)
maps, rows = {}, []
for regime in ("BG", "LEAF"):
    sel = runs[(runs.model == MODEL) & (runs.regime == regime) & (runs.seed == SEED_PAIR)]
    assert len(sel) == 1
    run = sel.run.iloc[0]
    marker = json.load(open(os.path.join(RUN_DIR, run + ".json")))
    assert marker["run"] == run and marker["classes"] == CLASSES and marker["weights_sha256"] == sel.weights_sha256.iloc[0]
    wpath = os.path.join(RUN_DIR, run + ".pt")
    assert sha256_file(wpath) == marker["weights_sha256"]
    col = "bg" if regime == "BG" else "leaf"
    assert run == e[col + "_run"].iloc[0] and marker["weights_sha256"] == e[col + "_weights_sha256"].iloc[0], "not the checkpoint of the eligibility table"
    assert (marker["model"], marker["family"], marker["regime"], marker["seed"]) == (MODEL, FAMILY, regime, SEED_PAIR)
    model = T.create(MODEL, len(CLASSES))
    model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True), strict=True)
    model.to(DEVICE).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    mean = torch.tensor(marker["mean"], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(marker["std"], device=DEVICE).view(1, 3, 1, 1)
    T.gc_layer_check(model, MODEL, DEVICE)
    xn = T.to_input(u8, mean, std)
    with torch.no_grad():
        z = model(xn).float()
    assert torch.isfinite(z).all()
    pred = z.argmax(1)
    x0n = (torch.zeros(1, 3, IMG, IMG, device=DEVICE) - mean) / std
    raw = {"GC": T.gradcam(model, xn, pred, MODEL), "IG": T.integrated_gradients(model, xn, x0n, pred)[0],
           "ROLL": T.rollout(model, xn), "TA": T.transformer_attribution(model, xn, pred)}
    for meth in METHODS:
        m, zero = T.normalise(raw[meth])
        S = T.off_leaf_share(m, M, zero).cpu().numpy()
        maps[(regime, meth)] = m.detach().cpu().numpy()
        for j in range(len(r)):
            ref = scores[(scores.run == run) & (scores.idx == r[j]) & (scores.method == meth) & (scores.image == "original")
                         & (scores.target == ("none" if meth == "ROLL" else "predicted"))]
            assert len(ref) == 1, (run, int(r[j]), meth, len(ref))
            assert ref.pred_class.iloc[0] == CLASSES[int(pred[j])], "predicted class differs from the attribution run"
            assert int(ref.map_status.iloc[0]) == int(zero[j]), "map status differs from the attribution run"
            assert abs(float(ref.S.iloc[0]) - float(S[j])) <= TOL, (run, int(r[j]), meth, float(ref.S.iloc[0]), float(S[j]))
            rows.append({"idx": int(r[j]), "class": true_class[j], "model": MODEL, "seed": SEED_PAIR, "regime": regime, "run": run,
                         "pred_class": CLASSES[int(pred[j])], "method": meth, "S_attribution_run": float(ref.S.iloc[0]), "S_recomputed": float(S[j]),
                         "map_status": int(zero[j]), "area_off_leaf": float(ref.area_off_leaf.iloc[0])})
    del model
fig3 = pd.DataFrame(rows)
print(fig3.round(4).to_string(index=False))
print("largest difference between recomputed and stored S:", float((fig3.S_recomputed - fig3.S_attribution_run).abs().max()))

# %%
# Figure 3: one row per image; the photograph with the mask outline, then the four maps of the BG and of the LEAF checkpoint.
# Each map is shown on its own colour scale (white = 0, darkest = that map's maximum); the number is its off-leaf share S
# from outputs/attribution_scores.csv, three decimals.
plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
ncol = 1 + 2 * len(METHODS)
fig, axes = plt.subplots(len(r), ncol, figsize=(7.16, 7.16 / ncol * len(r) + 0.55), layout="constrained")
for j in range(len(r)):
    ax = axes[j, 0]
    ax.imshow(rgb[j]); ax.contour(mask[j].astype(float), levels=[0.5], colors="white", linewidths=0.6)
    ax.set_ylabel(f"a = {fig3[fig3.idx == r[j]].area_off_leaf.iloc[0]:.3f}", fontsize=6)
    if j == 0:
        ax.set_title("Image, mask outline", fontsize=6.5)
    for b, regime in enumerate(("BG", "LEAF")):
        for k, meth in enumerate(METHODS):
            ax = axes[j, 1 + b * len(METHODS) + k]
            m = maps[(regime, meth)][j]
            ax.imshow(m, cmap="Greys", vmin=0, vmax=float(m.max()) if m.max() > 0 else 1.0, interpolation="nearest")
            ax.contour(mask[j].astype(float), levels=[0.5], colors="#0072B2", linewidths=0.35)
            row_ = fig3[(fig3.idx == r[j]) & (fig3.regime == regime) & (fig3.method == meth)].iloc[0]
            ax.set_xlabel(f"S = {row_.S_attribution_run:.3f}" + (" (uniform)" if row_.map_status else ""), fontsize=6, labelpad=1.5)
            if j == 0:
                lab = METHOD_LABEL[meth] if "\n" in METHOD_LABEL[meth] else METHOD_LABEL[meth] + "\n"
                ax.set_title(("BG-trained\n" if regime == "BG" else "LEAF-trained\n") + lab, fontsize=5.5, linespacing=1.0)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
desc = ("Figure 3 (illustration, no test): maps of ViT-S/16 seed pair 22 (eligible) for the first three of the six pre-fixed example images, "
        "predicted class, original images; each map on its own colour scale; S from attribution_scores.csv")
env = {"notebook": NOTEBOOK, "python": platform.python_version(), "device": DEVICE,
       "packages": {m: ilmd.version(m) for m in ("torch", "timm", "numpy", "pandas", "matplotlib")},
       "model": MODEL, "seed_pair": SEED_PAIR, "images_idx": [int(i) for i in r], "tolerance": TOL,
       "max_abs_difference_S": float((fig3.S_recomputed - fig3.S_attribution_run).abs().max()),
       "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
if SMOKE:
    import io as _io
    fig.savefig(_io.BytesIO(), format="png", dpi=100)        # the figure code runs; nothing is written
    print("SMOKE: all checks passed, nothing written to outputs/")
else:
    staged = []
    for ext in ("pdf", "png"):
        name = f"fig3_example_maps.{ext}"
        fig.savefig(os.path.join(OUT, name + ".tmp"), format=ext, dpi=600)
        staged.append((name, desc, "figures"))
    fig3.to_csv(os.path.join(OUT, "fig3_data.csv.tmp"), index=False, lineterminator="\n")
    staged.append(("fig3_data.csv", "Images, checkpoints, predicted classes and off-leaf shares shown in Figure 3 (stored and recomputed S)", "figures"))
    with open(os.path.join(OUT, "fig3_environment.json.tmp"), "w", newline="\n") as f:
        json.dump(env, f, indent=1)
    staged.append(("fig3_environment.json", "Environment and fixed choices of the Figure 3 notebook", "environment"))
    for name, _, _ in staged:                                  # all files exist before any is put in place
        os.replace(os.path.join(OUT, name + ".tmp"), os.path.join(OUT, name))
    for name, d_, cell in staged:
        manifest(name, d_, cell)
print("DONE")
