# %% [markdown]
# # RUNTIME: CPU (local PC) - no GPU is used
#
# Project: `qpain/paper-03` - Stage: `01 audit, masks, split` - Colab units: 0 (runs on this PC, plan section 10)
#
# Fetches PlantVillage and PlantDoc at pinned commits, pairs each colour image with its segmented twin, builds and
# audits leaf masks, caches 224 x 224 images, masks and filled backgrounds, groups images by physical leaf, profiles
# near duplicates, and writes the frozen split and the fixed attribution subset (analysis_plan.md section 3 and
# deviation 1). No model is trained and no network is used. Only derived metadata and small check images go to `outputs/`.
#
# Run: `jupyter nbconvert --to notebook --execute` with `PAPER_ROOT` set to the project folder.
# Source: `notebooks/src/01_audit_masks_split_CPU.py` (edit the source, rebuild with scripts/py-to-ipynb.py).

# %%
# Setup: paths from PAPER_ROOT, seed, parameters fixed by analysis_plan.md section 3 and deviation 1
import os, sys, json, io, time, hashlib, random, platform, datetime, subprocess, collections
import importlib.metadata as ilmd
from concurrent.futures import ProcessPoolExecutor
ROOT = os.environ.get("PAPER_ROOT", os.path.abspath(".."))
RAW, OUT, CKPT, CACHE = (os.path.join(ROOT, d) for d in ("data/raw", "outputs", "checkpoints", "data/cache"))
for d in (RAW, OUT, CKPT, CACHE):
    os.makedirs(d, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "notebooks"))
NOTEBOOK = "01_audit_masks_split_CPU"
AUDIT_VERSION = "2"           # bump when the audit code changes: the cache is then rebuilt

PV_URL, PV_COMMIT = "https://github.com/spMohanty/PlantVillage-Dataset", "7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"
PD_URL, PD_COMMIT = "https://github.com/pratikkayal/PlantDoc-Dataset", "5467f6012d78d1c446145d5f582da6096f852ae8"
LEAF_MAP_SHA = "3b4b253683c6911744959a3870a92509b1faa1ee34e9f8585e21d0fe6337a25b"   # read 2026-11-22 (data_sources.md)
SEED = 20261122               # plan section 3
FRACTIONS = (0.60, 0.15, 0.25)   # train, validation, test by leaf group (deviation 1)
ATTR_PER_CLASS = 30           # plan section 3
MASK_MIN, MASK_MAX = 0.05, 0.95  # plan section 3: mask area share outside this range excludes the image
MAX_PAIRS = 20_000_000        # safety stop on memory in the pair search
HAMMING_MAX = 8               # near-duplicate candidates are searched and profiled up to this pHash distance
MAX_GROUP_SHARE = 0.01        # plan section 3 stop rule: no merged group may hold more than 1% of the images
N_RANDOM_PAIRS = 2_000_000    # random pairs from different leaves, for the threshold profile
N_OVERLAYS = 200              # plan section 3: overlays saved for a visual check
# Deviation 1: classes without any leaf identifier are not used
DROPPED = ["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
           "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___healthy",
           "Squash___Powdery_mildew", "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus"]
# Near-duplicate grouping threshold. It is NOT chosen here: the first run profiles distances and stops before the split.
# The manager then looks at the pair sheets, writes a decision note and sets the value in this file:
#   null = not decided (profile only), -1 = no near-duplicate merging (stop rule), 0..8 = merge pairs up to this distance
THRESHOLD_FILE = os.path.join(ROOT, "notebooks", "p3_group_threshold.json")
GROUP_HAMMING = json.load(open(THRESHOLD_FILE))["group_hamming"] if os.path.exists(THRESHOLD_FILE) else None

import numpy as np, pandas as pd
from PIL import Image, ImageDraw
import p3_common as P
random.seed(SEED); np.random.seed(SEED)
VERSIONS = {m: ilmd.version(m) for m in ("numpy", "pandas", "pillow", "ImageHash", "scipy")}
VERSIONS["git"] = subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip()
print("Run at", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "| Python", platform.python_version())
print("SEED", SEED, "| FRACTIONS", FRACTIONS, "| GROUP_HAMMING", GROUP_HAMMING, "| common", P.COMMON_VERSION, "| MASK_T", P.MASK_T,
      "| alignment: texture correlation >=", P.ALIGN_MIN_CORR, "and above every", P.ALIGN_SHIFT, "px shift")
print(VERSIONS)

# %%
# Helpers: manifest-recorded outputs, git
def manifest_row(filename, description, cell):
    man = os.path.join(OUT, "MANIFEST.md")
    if not os.path.exists(man):
        with open(man, "w") as f:
            f.write("| file | notebook | cell | description |\n|---|---|---|---|\n")
    rows = [l for l in open(man).read().splitlines() if not l.startswith(f"| {filename} |")]
    rows.append(f"| {filename} | {NOTEBOOK} | {cell} | {description} |")
    with open(man, "w") as f:
        f.write("\n".join(rows) + "\n")
    print("saved", filename)

def save_output(obj, filename, description, cell=""):
    path = os.path.join(OUT, filename)
    tmp = path + ".tmp"
    if hasattr(obj, "to_csv"):       # gzip without a timestamp, so the same table always gives the same file
        obj.to_csv(tmp, index=False, lineterminator="\n",
                   compression={"method": "gzip", "mtime": 0} if filename.endswith(".gz") else None)
    elif isinstance(obj, Image.Image):
        obj.save(tmp, format="JPEG", quality=85)
    else:
        with open(tmp, "w", newline="\n") as f:
            json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)
    manifest_row(filename, description, cell)
    return path

def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def git(*args, cwd, binary=False):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True)
    assert r.returncode == 0, f"git {' '.join(args)} failed: {r.stderr.decode(errors='replace')[-800:]}"
    return r.stdout if binary else r.stdout.decode("utf-8").strip()

def remote_head(url):
    return git("ls-remote", url, "HEAD", cwd=ROOT).split()[0]

def fetch_repo(url, commit, dest, sparse=None, checkout=True):
    """Fetch exactly one commit (depth 1). Reused if already there. Returns nothing; asserts HEAD == commit."""
    have = os.path.isdir(os.path.join(dest, ".git")) and \
        subprocess.run(["git", "rev-parse", "-q", "--verify", f"{commit}^{{commit}}"], cwd=dest, capture_output=True).returncode == 0
    if not have:
        os.makedirs(dest, exist_ok=True)
        git("init", "-q", cwd=dest)
        git("config", "core.longpaths", "true", cwd=dest)
        git("config", "core.autocrlf", "false", cwd=dest)
        if subprocess.run(["git", "remote", "get-url", "origin"], cwd=dest, capture_output=True).returncode == 0:
            git("remote", "set-url", "origin", url, cwd=dest)       # an interrupted earlier attempt left the remote
        else:
            git("remote", "add", "origin", url, cwd=dest)
        if sparse:
            git("sparse-checkout", "set", *sparse, cwd=dest)
        git("fetch", "-q", "--depth", "1", *(["--filter=blob:none"] if sparse else []), "origin", commit, cwd=dest)
    if checkout:
        head = subprocess.run(["git", "rev-parse", "-q", "--verify", "HEAD"], cwd=dest, capture_output=True, text=True).stdout.strip()
        if head != commit:
            git("checkout", "-q", "--detach", commit, cwd=dest)
        assert git("rev-parse", "HEAD", cwd=dest) == commit
        dirty = git("status", "--porcelain", "--untracked-files=no", cwd=dest)
        assert dirty == "", f"working tree differs from the pinned commit: {dirty[:300]}"

# %%
# Download at the pinned commits (data_sources.md). The remote head is read before and after and recorded.
PV, PDR = os.path.join(RAW, "plantvillage"), os.path.join(RAW, "plantdoc")
t0 = time.time()
heads_before = {"plantvillage": remote_head(PV_URL), "plantdoc": remote_head(PD_URL)}
fetch_repo(PV_URL, PV_COMMIT, PV, sparse=["raw/color", "raw/segmented"])
fetch_repo(PD_URL, PD_COMMIT, PDR, checkout=False)   # read from the object store: its names cannot exist on Windows
heads_after = {"plantvillage": remote_head(PV_URL), "plantdoc": remote_head(PD_URL)}
assert heads_before == heads_after, "a repository changed during the download: run again"
print("fetch done in", round(time.time() - t0), "s | remote heads", heads_after)
assert sha256_file(os.path.join(PV, "leaf-map.json")) == LEAF_MAP_SHA, "leaf-map.json differs from the file audited on 2026-11-22"
raw_manifest = [
    {"dataset": "PlantVillage", "url": PV_URL, "commit": PV_COMMIT, "remote_head_at_run": heads_after["plantvillage"],
     "fetched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
     "what": "raw/color, raw/segmented and root files (sparse checkout); per-image SHA-256 in pv_inventory.csv.gz"},
    {"dataset": "PlantDoc", "url": PD_URL, "commit": PD_COMMIT, "remote_head_at_run": heads_after["plantdoc"],
     "fetched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
     "what": "whole commit in the git object store, no checkout; per-image SHA-256 in plantdoc_inventory.csv"}]
with open(os.path.join(RAW, "MANIFEST.md"), "w") as f:
    f.write("| dataset | url | commit | remote head at run | fetched (UTC) | what |\n|---|---|---|---|---|---|\n")
    for r in raw_manifest:
        f.write("| " + " | ".join(str(r[k]) for k in ("dataset", "url", "commit", "remote_head_at_run", "fetched_utc", "what")) + " |\n")
save_output(pd.DataFrame(raw_manifest), "raw_data_manifest.csv", "Datasets: URL, pinned commit, remote head at run time, fetch time", "download")

# %%
# Inventory of PlantVillage from the checked-out folders: classes, twins, leaf identifiers (owners' rule).
leaf_map = json.load(open(os.path.join(PV, "leaf-map.json"), encoding="utf-8"))
# Fixtures for the identifier rule, tested before any file is listed
assert P.ident("00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG") == "FREC_Scab 3335"
assert P.ident("00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335_final_masked.jpg") == "FREC_Scab 3335"
assert P.ident("00031d74-076e-4aef-b040-e068cd3576eb___R.S_HL 8315 copy 2.jpg") == "R.S_HL 8315"
assert P.ident("RS_Rust 1563.JPG") == "RS_Rust 1563"

def stem(f):
    return os.path.splitext(f)[0]

classes_all = sorted(os.listdir(os.path.join(PV, "raw", "color")))
assert len(classes_all) == 38 and sorted(os.listdir(os.path.join(PV, "raw", "segmented"))) == classes_all
assert all(c in classes_all for c in DROPPED)
rows, class_table = [], []
for cls in classes_all:
    col = sorted(os.listdir(os.path.join(PV, "raw", "color", cls)))
    seg = sorted(os.listdir(os.path.join(PV, "raw", "segmented", cls)))
    seg_by_stem = {stem(f): f for f in seg}
    assert len(seg_by_stem) == len(seg)
    seg_by_ident, col_ident_n = collections.defaultdict(list), collections.Counter(P.ident(f) for f in col)
    for f in seg:
        seg_by_ident[P.ident(f)].append(f)
    n_id = 0
    for f in col:
        twin, rule = seg_by_stem.get(stem(f) + "_final_masked"), "exact_name"
        if twin is None:
            cand = seg_by_ident.get(P.ident(f), [])
            if "___" not in f and len(cand) == 1 and col_ident_n[P.ident(f)] == 1:   # only names without a prefix
                twin, rule = cand[0], "identifier"
            else:
                twin, rule = None, ("no_twin" if not cand else "twin_ambiguous")
        lid = P.loader_leaf_id(cls, f, leaf_map)
        n_id += lid is not None
        rows.append({"class": cls, "file": f, "ident": P.ident(f), "leaf_id": lid, "twin": twin, "pair_rule": rule})
    class_table.append({"class": cls, "used": cls not in DROPPED, "color_files": len(col), "segmented_files": len(seg),
                        "with_leaf_id": n_id, "distinct_leaves": len({r["leaf_id"] for r in rows[-len(col):] if r["leaf_id"]})})
inv_all = pd.DataFrame(rows)
_tw = inv_all[inv_all.twin.notna()]
assert not _tw.duplicated(["class", "twin"]).any(), "a segmented file is paired with two colour files"
class_table = pd.DataFrame(class_table)
print(class_table.to_string())
# Deviation 1 rests on these facts: they are asserted, not assumed
assert (class_table.loc[~class_table.used, "with_leaf_id"] == 0).all(), "a dropped class has leaf identifiers"
assert (class_table.loc[class_table.used, "distinct_leaves"] >= 30).all(), "a kept class has fewer than 30 leaves"
assert len(inv_all) == 54305 and int(class_table.with_leaf_id.sum()) == 41111, "file list differs from the audit of 2026-11-22"
inv = inv_all[~inv_all["class"].isin(DROPPED)].reset_index(drop=True)
CLASSES = sorted(inv["class"].unique())
assert len(CLASSES) == 30
inv["idx"] = np.arange(len(inv))
print("images in the 30 classes:", len(inv), "| pairing:", inv.pair_rule.value_counts().to_dict(),
      "| with leaf id:", int(inv.leaf_id.notna().sum()))

# %%
# Per-image audit and cache (parallel): hashes, mask, alignment, 224 x 224 image, mask and filled background.
N = len(inv)
fingerprint = hashlib.sha256(json.dumps([AUDIT_VERSION, P.COMMON_VERSION, P.MASK_T, P.SIDE, P.FILL_SIGMA_FRAC, PV_COMMIT,
                                         inv[["class", "file", "twin"]].fillna("").values.tolist()]).encode()).hexdigest()
meta_path, audit_path = os.path.join(CACHE, "cache_meta.json"), os.path.join(CACHE, "audit_rows.csv")
shape_rgb = (N, P.SIDE, P.SIDE, 3)
def open_cache(mode):
    return (np.lib.format.open_memmap(os.path.join(CACHE, "rgb224.npy"), mode=mode, dtype=np.uint8, shape=shape_rgb if mode == "w+" else None),
            np.lib.format.open_memmap(os.path.join(CACHE, "bg224.npy"), mode=mode, dtype=np.uint8, shape=shape_rgb if mode == "w+" else None),
            np.lib.format.open_memmap(os.path.join(CACHE, "mask224.npy"), mode=mode, dtype=np.bool_, shape=shape_rgb[:3] if mode == "w+" else None))
CACHE_FILES = ["rgb224.npy", "bg224.npy", "mask224.npy"]
def cache_ok():
    try:
        meta = json.load(open(meta_path))
    except (OSError, ValueError):
        return False
    if meta.get("fingerprint") != fingerprint or not os.path.exists(audit_path) or meta.get("audit_sha256") != sha256_file(audit_path):
        return False
    return all(os.path.exists(os.path.join(CACHE, f)) and sha256_file(os.path.join(CACHE, f)) == meta["sha256"].get(f) for f in CACHE_FILES)
done = cache_ok()
if done:
    audit = pd.read_csv(audit_path, keep_default_na=False, na_values=[""],
                        dtype={"error": str, "phash": str, "sha256": str, "seg_sha256": str, "pixel_sha256": str, "cache_row_sha256": str}).fillna({"error": ""})
    print("cache reused (hashes verified):", len(audit), "rows")
else:
    if os.path.exists(meta_path):
        os.remove(meta_path)
    c_rgb, c_bg, c_mask = open_cache("w+")
    jobs = [(int(r.idx), os.path.join(PV, "raw", "color", r["class"], r.file),
             None if pd.isna(r.twin) else os.path.join(PV, "raw", "segmented", r["class"], r.twin))
            for _, r in inv.iterrows()]
    t0, out_rows = time.time(), []
    with ProcessPoolExecutor(max_workers=max(1, min(12, (os.cpu_count() or 4) - 2))) as ex:
        for s0 in range(0, N, 2000):                  # slabs bound the results waiting in memory
            for row, arrs in ex.map(P.audit_one, jobs[s0:s0 + 2000], chunksize=32):
                row["cached"] = arrs is not None
                if arrs is not None:
                    c_rgb[row["idx"]], c_mask[row["idx"]], c_bg[row["idx"]] = arrs[0], arrs[1], arrs[2]
                out_rows.append(row)
            print(len(out_rows), "images,", round(time.time() - t0), "s", flush=True)
    for a in (c_rgb, c_bg, c_mask):
        a.flush()
    del c_rgb, c_bg, c_mask
    audit = pd.DataFrame(out_rows).sort_values("idx").reset_index(drop=True)
    audit["error"] = audit["error"].fillna("")
    tmp = audit_path + ".tmp"
    audit.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, audit_path)
    with open(meta_path + ".tmp", "w") as f:
        json.dump({"fingerprint": fingerprint, "n": N, "side": P.SIDE, "common_version": P.COMMON_VERSION,
                   "sha256": {x: sha256_file(os.path.join(CACHE, x)) for x in CACHE_FILES}, "audit_sha256": sha256_file(audit_path),
                   "row_order": "idx column of pv_inventory.csv.gz"}, f)
    os.replace(meta_path + ".tmp", meta_path)
    print("audit and cache built in", round(time.time() - t0), "s")
assert len(audit) == N and (audit.idx.values == np.arange(N)).all()
df = inv.merge(audit, on="idx", how="left", validate="one_to_one")

# %%
# Exclusions (plan section 3), one reason per image, first rule that applies. Excluded images are in no analysis.
def reason(r):
    if r.pair_rule in ("no_twin", "twin_ambiguous"):
        return r.pair_rule
    if r.error:
        return "size_mismatch" if r.error == "size_mismatch" else "read_error"
    if not P.aligned(r.align_corr, r.align_corr_shifted_max):      # deviation 3; NaN fails
        return "misaligned"
    if min(r.mask_share, r.mask_share_224) < MASK_MIN:      # at the native size and at the model's input size
        return "mask_below_5pct"
    if max(r.mask_share, r.mask_share_224) > MASK_MAX:
        return "mask_above_95pct"
    return ""
df["excluded"] = [reason(r) for r in df.itertuples()]
df["usable"] = df.excluded == ""
assert df.loc[df.usable, "cached"].all()
excl = (df.assign(reason=df.excluded.replace("", "kept")).groupby(["class", "reason"]).size().unstack(fill_value=0).reset_index())
save_output(excl, "pv_exclusions_by_class.csv", "PlantVillage, 30 classes: images kept and excluded by reason and class", "exclusions")
print(df.excluded.replace("", "kept").value_counts().to_string())
print("image sizes:", df.groupby(["width", "height"]).size().to_dict())
qs = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
mask_profile = pd.DataFrame({"quantile": qs, **{c: df[c].quantile(qs).values for c in
                             ("mask_share", "mask_share_t8", "mask_share_t32", "mask_share_224", "align_corr", "align_corr_shifted_max", "align_mad")}})
# Sensitivity of the mask rules to the threshold (clarification C1): images outside the 5-95% range at thresholds 8, 16, 32
_tw = df[df.twin.notna() & (df.error == "")]
thr_sens = pd.DataFrame([{"threshold": t, "below_5pct": int((_tw[c] < MASK_MIN).sum()), "above_95pct": int((_tw[c] > MASK_MAX).sum()),
                          "median_share": float(_tw[c].median()), "mean_abs_change_vs_16": float((_tw[c] - _tw.mask_share).abs().mean())}
                         for t, c in ((8, "mask_share_t8"), (16, "mask_share"), (32, "mask_share_t32"))])
save_output(thr_sens, "pv_mask_threshold_sensitivity.csv", "Mask area rule at thresholds 8, 16 (used) and 32: images below 5% and above 95%, median share, mean absolute change against 16", "exclusions")
print(thr_sens.to_string())
save_output(mask_profile, "pv_mask_profile.csv", "Quantiles of mask area share (threshold 16, and 8 and 32 for sensitivity), share at 224 px, and alignment of the two copies inside the mask", "exclusions")
print(mask_profile.round(4).to_string())

# Overlays for a visual check: N_OVERLAYS usable images drawn with the seed, mask edge in red, 4 sheets of 50
from scipy import ndimage
rgb224 = np.load(os.path.join(CACHE, "rgb224.npy"), mmap_mode="r")
bg224 = np.load(os.path.join(CACHE, "bg224.npy"), mmap_mode="r")
mask224 = np.load(os.path.join(CACHE, "mask224.npy"), mmap_mode="r")
rng_o = np.random.default_rng(SEED + 7)
pick = np.sort(rng_o.choice(df.index[df.usable].values, N_OVERLAYS, replace=False))
def tile(i, t=112):
    a = np.array(rgb224[i]); m = np.array(mask224[i])
    edge = m ^ ndimage.binary_erosion(m, iterations=2)
    a[edge] = (255, 0, 0)
    both = np.concatenate([a, np.array(bg224[i])], axis=1)
    return Image.fromarray(both).resize((2 * t, t), Image.BILINEAR)
for s in range(4):
    sheet = Image.new("RGB", (5 * 224, 10 * 112), "white")
    for k, i in enumerate(pick[s * 50:(s + 1) * 50]):
        sheet.paste(tile(int(i)), ((k % 5) * 224, (k // 5) * 112))
    save_output(sheet, f"mask_overlay_sheet_{s + 1}.jpg", "50 random usable images: colour image with the mask edge in red (left) and its filled background (right)", "exclusions")
save_output(df.loc[pick, ["idx", "class", "file"]], "mask_overlay_list.csv", "The 200 images shown on the overlay sheets, in reading order", "exclusions")
# Questionable and rejected pairs are looked at too: the 50 kept images with the lowest alignment correlation, and up to
# 50 images excluded for alignment or mask area (those that could be read)
def sheet_of(indices, name, description):
    indices = [int(i) for i in indices][:50]
    if not indices:
        return
    sheet = Image.new("RGB", (5 * 224, 10 * 112), "white")
    for k, i in enumerate(indices):
        sheet.paste(tile(i), ((k % 5) * 224, (k // 5) * 112))
    save_output(sheet, name + ".jpg", description, "exclusions")
    save_output(df.loc[indices, ["idx", "class", "file", "excluded", "align_corr", "align_corr_shifted_max", "align_mad", "mask_share"]],
                name + "_list.csv", "Images on " + name + ".jpg, in reading order", "exclusions")
sheet_of(df[df.usable].sort_values(["align_corr", "idx"]).index[:50], "mask_overlay_lowest_alignment",
         "The 50 kept images with the lowest alignment correlation: colour image with mask edge (left), filled background (right)")
_rej = df[df.excluded.isin(["misaligned", "mask_below_5pct", "mask_above_95pct"]) & (df.cached == True)]
sheet_of(_rej.sample(min(50, len(_rej)), random_state=SEED).sort_index().index if len(_rej) else [], "mask_overlay_excluded",
         "Up to 50 images excluded for alignment or mask area")

# %%
# Near-duplicate candidates among usable images (pHash, exact brute force) and the threshold profile.
U = df[df.usable].reset_index(drop=True)          # position in U is used below; U.idx is the cache row
h = np.array([int(x, 16) for x in U.phash], dtype=np.uint64)
def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)
assert popcount64(np.array([0, 1, 0xFF, 2**64 - 1], dtype=np.uint64)).tolist() == [0, 1, 8, 64]
pi, pj, pdist = [], [], []
for s in range(0, len(h), 256):
    d = popcount64(h[s:s + 256, None] ^ h[None, :])
    ii, jj = np.nonzero(d <= HAMMING_MAX)
    keep = (ii + s) < jj
    pi.append(ii[keep] + s); pj.append(jj[keep]); pdist.append(d[ii[keep], jj[keep]])
    assert sum(map(len, pi)) <= MAX_PAIRS, f"more than {MAX_PAIRS} candidate pairs at distance <= {HAMMING_MAX}: lower HAMMING_MAX by a recorded deviation"
pi, pj, pdist = np.concatenate(pi), np.concatenate(pj), np.concatenate(pdist)
print("candidate pairs at distance <=", HAMMING_MAX, ":", len(pi))
lid = np.array([x if isinstance(x, str) else None for x in U.leaf_id], dtype=object)
known = np.array([x is not None for x in lid])
assert known.sum() == U.leaf_id.notna().sum()
cls_u = U["class"].values
lcode = pd.factorize(U.leaf_id)[0]                    # -1 = no identifier
assert ((lcode >= 0) == known).all()
same_leaf = (lcode[pi] >= 0) & (lcode[pi] == lcode[pj])
kind = np.where(same_leaf, "same_leaf",
        np.where(known[pi] & known[pj], "different_known_leaves", np.where(known[pi] | known[pj], "known_and_unknown", "both_unknown")))
pairs = pd.DataFrame({"a": pi, "b": pj, "d": pdist, "kind": kind, "same_class": cls_u[pi] == cls_u[pj],
                      "identical_pixels": U.pixel_sha256.values[pi] == U.pixel_sha256.values[pj]})

class DSU:
    def __init__(self, n): self.p = np.arange(n)
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[max(ra, rb)] = min(ra, rb)

def groups_at(d):
    """Group label per usable image: same leaf identifier OR identical decoded pixels OR pHash distance <= d (d = -1: none)."""
    dsu = DSU(len(U))
    for col in ("leaf_id", "pixel_sha256"):
        for idxs in U.groupby(col).indices.values():      # positions in U; rows without a value form no group
            for k in idxs[1:]:
                dsu.union(int(idxs[0]), int(k))
    if d >= 0:
        for a, b in zip(pi[pdist <= d], pj[pdist <= d]):
            dsu.union(int(a), int(b))
    return np.array([dsu.find(i) for i in range(len(U))])

# Reference distributions: all pairs of images of the same leaf, and random pairs from different known leaves
rng_p = np.random.default_rng(SEED + 3)
kidx = np.flatnonzero(known)
ra, rb = rng_p.choice(kidx, N_RANDOM_PAIRS), rng_p.choice(kidx, N_RANDOM_PAIRS)
ok = lcode[ra] != lcode[rb]
d_rand = popcount64(h[ra[ok]] ^ h[rb[ok]])
sl_a, sl_b = [], []
for idxs in U.groupby("leaf_id").indices.values():
    if len(idxs) > 1:
        a, b = np.triu_indices(len(idxs), 1)
        sl_a.append(idxs[a]); sl_b.append(idxs[b])
sl_a, sl_b = np.concatenate(sl_a), np.concatenate(sl_b)
d_same = popcount64(h[sl_a] ^ h[sl_b])
prof = []
for d in range(-1, HAMMING_MAX + 1):
    g = groups_at(d)
    sizes = np.bincount(g)
    sub = pairs[pairs.d <= d]
    prof.append({"distance_max": d, "groups": int((sizes > 0).sum()), "largest_group": int(sizes.max()),
                 "largest_group_share": float(sizes.max() / len(U)),
                 "pairs_same_leaf": int((sub.kind == "same_leaf").sum()),
                 "pairs_different_known_leaves": int((sub.kind == "different_known_leaves").sum()),
                 "pairs_different_known_leaves_other_class": int(((sub.kind == "different_known_leaves") & ~sub.same_class).sum()),
                 "pairs_known_and_unknown": int((sub.kind == "known_and_unknown").sum()),
                 "pairs_both_unknown": int((sub.kind == "both_unknown").sum()),
                 "share_of_same_leaf_pairs_within": float((d_same <= d).mean()) if d >= 0 else 0.0,
                 "random_different_leaf_pairs_within": int((d_rand <= d).sum()) if d >= 0 else 0,
                 "random_pairs_drawn": int(ok.sum())})
prof = pd.DataFrame(prof)
save_output(prof, "near_duplicate_profile.csv", "pHash threshold profile: groups, largest group, candidate pairs by kind, share of same-leaf pairs and count of random different-leaf pairs within each distance (-1 = identity links only)", "near duplicates")
print(prof.to_string())
hist = pd.DataFrame({"distance": np.arange(65), "same_leaf_pairs": np.bincount(d_same, minlength=65),
                     "random_different_leaf_pairs": np.bincount(d_rand, minlength=65)})
save_output(hist, "phash_distance_histogram.csv", "pHash distance of all same-leaf pairs and of random pairs from different known leaves", "near duplicates")

# Pair sheets for the manager's look: up to 30 random pairs that are NOT the same known leaf, per distance band
rng_s = np.random.default_rng(SEED + 5)
sheet_rows = []
for lo, hi in ((0, 0), (1, 2), (3, 4), (5, 6), (7, 8)):
    sub = pairs[(pairs.d >= lo) & (pairs.d <= hi) & (pairs.kind != "same_leaf")]
    take = sub.loc[np.sort(rng_s.choice(sub.index.values, min(30, len(sub)), replace=False))] if len(sub) else sub
    if not len(take):
        continue
    sheet = Image.new("RGB", (5 * 230, 6 * 124), "white")
    draw = ImageDraw.Draw(sheet)
    for k, r in enumerate(take.itertuples()):
        x, y = (k % 5) * 230, (k // 5) * 124
        for q, u in enumerate((r.a, r.b)):
            sheet.paste(Image.fromarray(np.array(rgb224[int(U.idx[u])])).resize((112, 112)), (x + q * 113, y))
        draw.text((x + 2, y + 112), f"d={r.d} {r.kind[:14]} {'same' if r.same_class else 'OTHER'} class", fill="black")
        sheet_rows.append({"band": f"{lo}-{hi}", "position": k, "d": r.d, "kind": r.kind, "same_class": r.same_class,
                           "class_a": cls_u[r.a], "file_a": U.file[r.a], "class_b": cls_u[r.b], "file_b": U.file[r.b]})
    save_output(sheet, f"near_duplicate_pairs_d{lo}-{hi}.jpg", f"Up to 30 random candidate pairs at pHash distance {lo}-{hi} that do not share a known leaf identifier", "near duplicates")
save_output(pd.DataFrame(sheet_rows), "near_duplicate_pair_sheets.csv", "The pairs shown on the near-duplicate sheets, in reading order", "near duplicates")

# %%
# Groups, split and attribution subset. Runs only once the grouping threshold has been decided (p3_group_threshold.json).
summary = {"notebook": NOTEBOOK, "seed": SEED, "fractions": FRACTIONS, "group_hamming": GROUP_HAMMING,
           "pv_commit": PV_COMMIT, "pd_commit": PD_COMMIT, "common_version": P.COMMON_VERSION, "mask_threshold": P.MASK_T,
           "versions": VERSIONS, "images_38_classes": int(len(inv_all)), "images_30_classes": int(N),
           "usable": int(df.usable.sum()), "excluded": df.excluded[df.excluded != ""].value_counts().to_dict(),
           "cache_fingerprint": fingerprint}
SPLIT_DONE = False
FROZEN_FILE = os.path.join(ROOT, "notebooks", "p3_frozen.json")     # written by the manager after the first split run
if GROUP_HAMMING is None:
    print("STOP BEFORE THE SPLIT: the near-duplicate threshold is not decided. Look at the profile and the pair sheets, "
          "write the decision note, set notebooks/p3_group_threshold.json and run again.")
    assert not os.path.exists(FROZEN_FILE), "a frozen split exists but the threshold file says 'not decided'"
else:
    _thr = json.load(open(THRESHOLD_FILE))
    assert isinstance(GROUP_HAMMING, int) and not isinstance(GROUP_HAMMING, bool) and -1 <= GROUP_HAMMING <= HAMMING_MAX
    assert os.path.exists(os.path.join(ROOT, _thr["decision"])), "the threshold file must name the decision note that set it"
    base_share = np.bincount(groups_at(-1)).max() / len(U)
    assert base_share <= MAX_GROUP_SHARE, "identity links alone give a group above 1% of the images: stop and record a deviation"
    g = groups_at(GROUP_HAMMING)
    sizes = np.bincount(g)
    assert sizes.max() / len(U) <= MAX_GROUP_SHARE, ("stop rule (plan section 3): at this threshold a merged group holds more than 1% "
                                                     "of the images; the plan then allows only -1 (no near-duplicate merging)")
    U["group"] = g
    grp = U.groupby("group")
    g_all_known = grp.leaf_id.apply(lambda s: bool(s.notna().all()))
    g_ncls = grp["class"].nunique()
    g_class = grp["class"].first()
    # A group can be evaluated only if EVERY image in it has a leaf identifier and all are of one class. Any other group
    # goes to training whole: an image with no identifier never reaches validation, test or the attribution subset
    # (plan section 3), and each evaluated group belongs to exactly one class.
    g_eval_ok = g_all_known & (g_ncls == 1)
    rng = np.random.default_rng(SEED)
    part, alloc = {}, []
    for cls in CLASSES:                                   # fixed order
        gs = np.array(sorted(g_eval_ok.index[g_eval_ok & (g_class == cls)]))
        rng.shuffle(gs)
        n_test, n_val = int(round(FRACTIONS[2] * len(gs))), int(round(FRACTIONS[1] * len(gs)))
        assert n_test >= 1 and n_val >= 1 and len(gs) - n_test - n_val >= 1, f"{cls}: too few leaf groups to split ({len(gs)})"
        for k, gid in enumerate(gs):
            part[gid] = "test" if k < n_test else ("val" if k < n_test + n_val else "train")
        alloc.append({"class": cls, "groups_split": len(gs), "groups_test": n_test, "groups_val": n_val})
    U["partition"] = [part.get(x, "train_only") for x in U.group]   # train_only = group not eligible for evaluation
    U["group_all_known_one_class"] = U.group.map(g_eval_ok).values
    # checks, row by row: no group in two partitions; every validation and test image has a leaf identifier
    assert (U.groupby("group").partition.nunique() == 1).all()
    assert U.loc[U.partition.isin(["val", "test"]), "leaf_id"].notna().all()
    assert (U.loc[U.partition.isin(["val", "test"])].groupby("group")["class"].nunique() == 1).all()
    forced = U[(U.partition == "train_only") & U.leaf_id.notna()]
    print("known-leaf images moved to training because their group holds an image without an identifier or two classes:",
          len(forced), "images in", forced.group.nunique(), "groups;", int((g_ncls > 1).sum()), "groups span two classes")
    cnt = U.groupby(["class", "partition"]).size().unstack(fill_value=0).reindex(columns=["train", "train_only", "val", "test"], fill_value=0)
    gcnt = U[U.partition != "train_only"].groupby(["class", "partition"]).group.nunique().unstack(fill_value=0)
    assert (cnt[["train", "val", "test"]] > 0).all().all()
    split = U[["idx", "class", "file", "leaf_id", "group", "partition", "sha256", "pixel_sha256", "phash", "cache_row_sha256"]].copy()
    assert split.cache_row_sha256.str.len().eq(64).all()
    split_table = cnt.add_prefix("images_").join(gcnt.add_prefix("groups_")).reset_index()

    # Attribution subset: up to 30 test groups per class, one image per group, drawn once with the plan's seed before any model
    rng_a = np.random.default_rng(SEED)
    chosen = []
    T = U[U.partition == "test"]
    for cls in CLASSES:
        gs = np.array(sorted(T.group[T["class"] == cls].unique()))      # test groups are single-class
        rng_a.shuffle(gs)
        for gid in gs[:ATTR_PER_CLASS]:
            members = np.sort(T.index[T.group == gid].values)
            chosen.append(int(rng_a.choice(members)))
    attr = U.loc[sorted(chosen), ["idx", "class", "file", "leaf_id", "group", "sha256"]].copy()
    assert attr.group.is_unique and attr.leaf_id.notna().all()
    attr_n = attr.groupby("class").size().reindex(CLASSES, fill_value=0)
    assert (attr_n >= 1).all()

    # Freeze: the split and the subset are written once. A later run must reproduce them byte for byte.
    def frozen_write(table, filename, description, key):
        path, tmp = os.path.join(OUT, filename), os.path.join(CKPT, filename + ".candidate")
        table.to_csv(tmp, index=False, lineterminator="\n")
        new = sha256_file(tmp)
        if os.path.exists(FROZEN_FILE):
            want = json.load(open(FROZEN_FILE))[key]
            assert new == want, f"{filename}: this run gives {new[:12]}, the frozen file is {want[:12]}; the frozen split is never overwritten"
            assert os.path.exists(path) and sha256_file(path) == want, f"{filename} in outputs/ is not the frozen file"
            os.remove(tmp)
            manifest_row(filename, description, "split")
        else:
            os.replace(tmp, path)
            manifest_row(filename, description, "split")
        return new
    split_sha = frozen_write(split, "pv_split.csv", "Frozen split: one row per usable image of the 30 classes (cache row idx, class, file, leaf id, merged group, partition; train_only = the group holds an image without a leaf identifier or two classes, training only)", "split_sha256")
    attr_sha = frozen_write(attr, "attribution_subset.csv", "Fixed attribution subset: up to 30 test images per class, one per merged leaf group", "attribution_sha256")
    save_output(split_table, "pv_split_by_class.csv", "Images and leaf groups per class and partition", "split")
    print(split_table.to_string())
    print("attribution subset:", len(attr), "images | per class min", int(attr_n.min()), "max", int(attr_n.max()))

    # Near-duplicate candidates left unmerged, by partition pair (plan section 3: published either way)
    left = pairs[(pairs.d > GROUP_HAMMING) & (pairs.kind != "same_leaf")].copy()
    pa, pb = U.partition.values[left.a], U.partition.values[left.b]
    left["partitions"] = [" / ".join(sorted(x)) for x in zip(pa, pb)]
    cross = left.groupby(["partitions", "d"]).size().unstack(fill_value=0).reset_index()
    cross.columns = [str(c) for c in cross.columns]
    save_output(cross, "near_duplicate_candidates_by_partition.csv", "Unmerged near-duplicate candidate pairs (pHash distance above the grouping threshold, up to 8) by partition pair and distance", "split")
    print(cross.to_string())
    summary.update({"groups": int(U.group.nunique()), "groups_spanning_classes": int((g_ncls > 1).sum()),
                    "known_leaf_images_forced_to_training": int(len(forced)), "largest_group": int(sizes.max()),
                    "partition_images": U.partition.value_counts().to_dict(), "attribution_images": int(len(attr)),
                    "attribution_min_per_class": int(attr_n.min()), "split_sha256": split_sha, "attribution_sha256": attr_sha,
                    "threshold_decision": _thr["decision"]})
    SPLIT_DONE = True

# %%
# Inventory of the 30 classes (all images, with audit values and exclusion reason) and the class table
keep_cols = ["idx", "class", "file", "ident", "leaf_id", "twin", "pair_rule", "excluded", "width", "height", "mask_share",
             "mask_share_t8", "mask_share_t32", "mask_share_224", "align_corr", "align_corr_shifted_max", "align_mad", "sha256", "seg_sha256", "cache_row_sha256",
             "pixel_sha256", "phash", "error"]
save_output(df[keep_cols], "pv_inventory.csv.gz", "PlantVillage, 30 classes: one row per colour image (row order = cache row), twin, leaf id, mask and alignment values, hashes, exclusion reason", "inventory")
save_output(class_table, "pv_class_table.csv", "PlantVillage, 38 classes: files, images with a leaf identifier, distinct leaves, and whether the class is used (deviation 1)", "inventory")

# %%
# PlantDoc: inventory read from the git object store, and the class mapping (fixed here, before any model exists).
PD_MAP = {  # PlantDoc folder -> PlantVillage class; None = no counterpart among the 30 classes used
    "Apple Scab Leaf": "Apple___Apple_scab", "Apple leaf": "Apple___healthy", "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy", "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy", "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": None, "Corn leaf blight": None, "Corn rust leaf": None,
    "Peach leaf": "Peach___healthy", "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight", "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy", "Squash Powdery mildew leaf": None, "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight", "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy", "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight", "Tomato leaf mosaic virus": None,
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": None, "grape leaf black rot": "Grape___Black_rot"}
assert all(v is None or v in CLASSES for v in PD_MAP.values())
listing = git("ls-tree", "-r", "-z", PD_COMMIT, cwd=PDR, binary=True).decode("utf-8").split("\0")
blobs = P.GitBlobs(PDR)
pd_rows = []
for line in listing:
    if not line:
        continue
    meta, path = line.split("\t", 1)
    parts = path.split("/")
    if len(parts) != 3 or parts[0] not in ("train", "test"):
        continue
    data = blobs.read(meta.split()[2])
    row = {"split_folder": parts[0], "folder": parts[1], "file": parts[2], "git_blob": meta.split()[2],
           "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "width": None, "height": None, "error": ""}
    try:
        with Image.open(io.BytesIO(data)) as im:
            row["width"], row["height"] = im.size
            im.convert("RGB").load()
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:200]
    assert parts[1] in PD_MAP, f"PlantDoc folder without a mapping decision: {parts[1]}"
    row["pv_class"] = PD_MAP[parts[1]]
    pd_rows.append(row)
blobs.close()
pdinv = pd.DataFrame(pd_rows)
assert len(pdinv) == 2578, f"PlantDoc image count differs from the audit of 2026-11-22: {len(pdinv)}"
pdinv["duplicate_content"] = pdinv.duplicated("sha256", keep=False)
save_output(pdinv, "plantdoc_inventory.csv", "PlantDoc: one row per image file (folders, git blob, size, SHA-256, decoded size, read error, mapped PlantVillage class, byte-identical duplicates)", "plantdoc")
pdmap = (pdinv.groupby(["folder", "split_folder"]).size().unstack(fill_value=0).reset_index()
         .assign(pv_class=lambda t: t.folder.map(PD_MAP)))
pdmap["readable_images"] = pdmap.folder.map(pdinv[pdinv.error == ""].groupby("folder").size()).fillna(0).astype(int)
save_output(pdmap, "plantdoc_class_mapping.csv", "PlantDoc folder to PlantVillage class mapping (empty = not used), with image counts", "plantdoc")
print(pdmap.to_string())
summary.update({"plantdoc_images": int(len(pdinv)), "plantdoc_unreadable": int((pdinv.error != "").sum()),
                "plantdoc_mapped_images": int(pdinv.pv_class.notna().sum()), "plantdoc_mapped_classes": int(pdinv.pv_class.nunique()),
                "plantdoc_duplicate_files": int(pdinv.duplicate_content.sum())})

# %%
# Summary and end-of-run checks
summary["split_done"] = SPLIT_DONE
summary["remote_heads"] = heads_after
summary["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
save_output(summary, "audit_summary.json", "Run summary of notebook 01: parameters, versions, counts, exclusion counts, split and subset hashes", "summary")
assert git("rev-parse", "HEAD", cwd=PV) == PV_COMMIT
assert git("status", "--porcelain", "--untracked-files=no", cwd=PV) == "", "raw data changed during the run"
n_tree = sum(1 for l in git("ls-tree", "-r", "--name-only", PV_COMMIT, "raw/color", "raw/segmented", cwd=PV).splitlines() if l)
n_disk = sum(len(fs) for sub in ("color", "segmented") for _, _, fs in os.walk(os.path.join(PV, "raw", sub)))
assert n_tree == n_disk == 54305 + 54306, f"files on disk ({n_disk}) differ from the commit's tree ({n_tree})"
print(json.dumps(summary, indent=2, default=str))
print("DONE" if SPLIT_DONE else "DONE (profile only: split not written)")
