# %% [markdown]
# # RUNTIME: local workstation GPU (NVIDIA RTX 3060 12 GB), `.venv-local` (plan section 10)
#
# Project: `qpain/paper-03` - Stage: `02 train` - Estimated time: about 11 h on the RTX 3060 (to be measured)
#
# Trains the four planned classifiers (analysis_plan.md section 4) under three regimes (ORIG, LEAF, BG), three seeds
# each (36 runs), with one protocol, on the frozen split from notebook 01 (SHA-256 asserted). The learning rate is
# chosen per architecture from two values on the ORIG regime with seed 11 (validation loss). Only training and
# validation images are read here: no test image is scored in this notebook.
#
# Run: `PAPER_ROOT=<project> python -m jupyter nbconvert --to notebook --execute notebooks/02_train_GPU.ipynb ...`
# `SMOKE=1`: 1% of each class and partition, 1 epoch, seed 11, results to checkpoints/smoke/, nothing to outputs/.
# `STAGE=lr`: stop after the learning-rate runs (these time one full run per architecture). `DEVICE=cpu` forces the CPU.
# Source: `notebooks/src/02_train_GPU.py` (edit the source, rebuild with scripts/py-to-ipynb.py).

# %%
# Setup: paths, parameters fixed by analysis_plan.md section 4 (and the clarifications in section 10), determinism
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # deterministic cuBLAS; set before CUDA starts
import sys, json, time, hashlib, random, platform, datetime, subprocess, threading, queue, warnings
import importlib.metadata as ilmd
import numpy as np, pandas as pd, requests
import torch, torch.nn as nn, torch.nn.functional as F
import timm

ROOT = os.environ.get("PAPER_ROOT", os.path.abspath(".."))
OUT, CKPT, CACHE = (os.path.join(ROOT, d) for d in ("outputs", "checkpoints", "data/cache"))
sys.path.insert(0, os.path.join(ROOT, "notebooks"))
import p3_common as P
import p3_torch as T
SMOKE = os.environ.get("SMOKE", "0") == "1"
STAGE = os.environ.get("STAGE", "all")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
NOTEBOOK = "02_train_GPU"
RUN_DIR = os.path.join(CKPT, "smoke" if SMOKE else "runs")
for d in (OUT, CKPT, RUN_DIR, os.path.join(CKPT, "pretrained")):
    os.makedirs(d, exist_ok=True)

FROZEN = json.load(open(os.path.join(ROOT, "notebooks", "p3_frozen.json")))   # written after notebook 01's final run
SPLIT_SHA256 = FROZEN["split_sha256"]
MODELS = [  # (timm name, family, plan name, Hub repository, pinned commit, SHA-256 of model.safetensors); read 2026-11-22
    ("resnet50", "cnn", "ResNet-50", "timm/resnet50.a1_in1k",
     "767268603ca0cb0bfe326fa87277f19c419566ef", "773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb"),
    ("efficientnet_b0", "cnn", "EfficientNet-B0", "timm/efficientnet_b0.ra_in1k",
     "1b5383e5f79cc0f7fc067e372f8f26a5fa73f26a", "d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd"),
    ("deit_small_patch16_224", "transformer", "DeiT-Small/16", "timm/deit_small_patch16_224.fb_in1k",
     "91327a9c99f98fe6b524cd4d397b7226b80e1365", "1e747b4a8d0df2cfbd3c450e8c97685d867448ab0c2ddbfb34b6885f5cb23e5b"),
    ("vit_small_patch16_224", "transformer", "ViT-Small/16", "timm/vit_small_patch16_224.augreg_in21k_ft_in1k",
     "7e2c55630205e1266030f18370f4c6ed1a514b52", "79c03c635cdfd798a364a9d8c4e5c0b7255b975ea2c9616046d4f77ab01435aa"),
]
REGIMES = ["BG", "LEAF", "ORIG"]     # BG first: whether background-only models train is the design's main risk
SEEDS = [11, 22, 33]
LR_GRID, LR_REGIME, LR_SEED = [1e-4, 3e-4], "ORIG", 11
IMG = P.SIDE
EPOCHS, BATCH, WD = 10, 64, 0.05
EVAL_BATCH = 64
JITTER = 0.10               # "mild colour jitter": brightness, contrast and saturation factors drawn from 1 +- 0.10
RELOAD_TOL = 1e-3           # largest logit difference allowed between the best epoch and the reloaded checkpoint
DONOR_SEED = 20261122 + 11  # fixed donors of the validation images
MAX_GPU_SHARE = 0.90
SMOKE_FRACTION, SMOKE_SEED = 0.01, 20261122
if SMOKE:
    SEEDS, EPOCHS = [11], 1

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
DET_WARNINGS = set()        # operations without a deterministic implementation, if any is used (recorded per run)
_showwarning = warnings.showwarning
def _record_warning(message, category, filename, lineno, file=None, line=None):
    if "deterministic" in str(message):
        DET_WARNINGS.add(str(message)[:300])
    _showwarning(message, category, filename, lineno, file, line)
warnings.showwarning = _record_warning
BACKEND = {"cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
           "deterministic_algorithms": "warn_only", "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
           "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32}
VERSIONS = {m: ilmd.version(m) for m in ("torch", "timm", "numpy", "pandas", "safetensors", "requests", "scipy")}
ENV = {"python": platform.python_version(), "platform": platform.platform(), "device": DEVICE, "packages": VERSIONS,
       "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
       "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None, "smoke": SMOKE, "stage": STAGE}
if DEVICE == "cuda":
    ENV["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    ENV["gpu_total_mib"] = torch.cuda.get_device_properties(0).total_memory // 2**20
print(json.dumps(ENV, indent=1))

# %%
# Helpers: atomic writes, manifest-recorded outputs (same format as notebook 01), hashing, metrics
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
        atomic(path, lambda t: obj.to_csv(t, index=False, lineterminator="\n"))
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

def metrics(y, logits):
    """Accuracy, macro F1 over the classes present in y or predicted, and mean cross-entropy (float64)."""
    pred = logits.argmax(1)
    f1 = []
    for c in np.union1d(np.unique(y), np.unique(pred)):
        tp = int(((pred == c) & (y == c)).sum()); fn = int(((pred != c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum())
        f1.append(2 * tp / (2 * tp + fn + fp))
    z = logits.astype(np.float64); z = z - z.max(1, keepdims=True)
    ce = float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean())
    return float((pred == y).mean()), float(np.mean(f1)), ce

_y = np.array([0, 0, 1, 2]); _l = np.log(np.eye(3)[[0, 1, 1, 2]] * 0.9 + 0.1 / 3 * 0 + 1e-9)
assert np.isclose(metrics(_y, _l)[0], 0.75)

# %%
# Frozen inputs: the split (hash asserted), the image cache of notebook 01 (fingerprint recorded), the sets used here
split_path = os.path.join(OUT, "pv_split.csv")
assert sha256_file(split_path) == SPLIT_SHA256, "pv_split.csv is not the frozen split (notebooks/p3_frozen.json)"
split = pd.read_csv(split_path)
CLASSES = sorted(split["class"].unique())
assert len(CLASSES) == 30
CACHE_META = json.load(open(os.path.join(CACHE, "cache_meta.json")))
assert CACHE_META["common_version"] == P.COMMON_VERSION and CACHE_META["fingerprint"] == FROZEN["cache_fingerprint"]
C = T.Cache(CACHE)
assert len(C.rgb) == CACHE_META["n"] and split.idx.is_unique and split.idx.max() < len(C.rgb)

sets = {"train": split[split.partition.isin(["train", "train_only"])], "val": split[split.partition == "val"]}
sets = {k: v.sort_values("idx").reset_index(drop=True) for k, v in sets.items()}
# The cache rows this notebook uses are checked against the digests frozen in the split file. Only training and
# validation rows are read: no byte of a test image is touched here (Codex review 002, points 1 and 2).
t0 = time.time()
for k, v in sets.items():
    r = v.idx.to_numpy(dtype=np.int64)
    for i in range(0, len(r), 512):
        rgb, m, bg = (T.read_rows(a, r[i:i + 512]) for a in (C.rgb, C.mask, C.bg))
        for j in range(len(rgb)):
            assert P.cache_row_sha(rgb[j], m[j], bg[j]) == v.cache_row_sha256.iat[i + j], f"cache row {r[i + j]} differs from the frozen split"
print(f"cache rows of train and validation verified in {time.time() - t0:.0f} s")
ROWS_DIGEST = {k: hashlib.sha256("".join(v.cache_row_sha256).encode()).hexdigest() for k, v in sets.items()}
if SMOKE:   # 1% per class in each set, seeded; at least 2 images per class where the class has them
    sets = {k: pd.concat([g.sample(min(len(g), max(2, int(round(len(g) * SMOKE_FRACTION)))), random_state=SMOKE_SEED) for _, g in v.groupby("class")])
            .sort_values("idx").reset_index(drop=True) for k, v in sets.items()}
rows = {k: v.idx.to_numpy(dtype=np.int64) for k, v in sets.items()}
labels = {k: v["class"].map(CLASSES.index).to_numpy(dtype=np.int64) for k, v in sets.items()}
for k, v in sets.items():
    print(k, len(v), "images | classes", v["class"].nunique(), "| partitions", v.partition.value_counts().to_dict())
assert set(sets["val"].partition) == {"val"} and not set(sets["train"].group) & set(sets["val"].group)
# Fixed donors of the validation images: uniform over the validation set, any class, drawn once (plan section 3)
VAL_DONOR = np.random.default_rng(DONOR_SEED).integers(0, len(rows["val"]), len(rows["val"]))
val_donors = pd.DataFrame({"idx": rows["val"], "donor_idx": rows["val"][VAL_DONOR]})

fetch = C.fetch
def build(parts):
    return T.build(parts, DEVICE)

# The device compositing must equal the reference implementation, tested on real cache rows before any training
_r, _d = rows["val"][:8], rows["val"][VAL_DONOR[:8]]
for _reg in ("LEAF", "BG"):
    _got = build(fetch(_reg, _r, _d)).cpu().numpy()
    for _k in range(8):
        _a, _b = (_r[_k], _d[_k]) if _reg == "LEAF" else (_d[_k], _r[_k])      # a gives leaf and mask, b gives the background
        assert (_got[_k] == P.compose(np.array(C.rgb[_a]), np.array(C.mask[_a]), np.array(C.bg[_b]))).all(), _reg
assert (build(fetch("ORIG", _r, _d)).cpu().numpy() == T.read_rows(C.rgb, _r)).all()

# %%
# Pretrained weights: downloaded with requests from the timm repositories on the Hugging Face Hub at the pinned
# commits (the Hub client fails on this PC's certificate setup), checked against the pinned SHA-256, loaded from file.
def pretrained_file(name, repo, commit, sha):
    path = os.path.join(CKPT, "pretrained", f"{name}.safetensors")
    if not (os.path.exists(path) and sha256_file(path) == sha):
        r = requests.get(f"https://huggingface.co/{repo}/resolve/{commit}/model.safetensors", timeout=600)
        r.raise_for_status()
        def w(t):
            with open(t, "wb") as f:
                f.write(r.content)
        atomic(path, w)
    assert sha256_file(path) == sha, f"{name}: weight file does not match the pinned SHA-256"
    cfg = timm.create_model(name, pretrained=False).pretrained_cfg
    assert repo.endswith("." + cfg["tag"]), f"{name}: timm's default tag {cfg['tag']} differs from the pinned repository {repo}"
    return path, {"model": name, "repo": repo, "commit": commit, "sha256": sha, "mean": list(cfg["mean"]), "std": list(cfg["std"])}

WEIGHTS = {name: pretrained_file(name, repo, commit, sha) for name, _, _, repo, commit, sha in MODELS}
for name in WEIGHTS:
    print(name, WEIGHTS[name][1]["repo"], WEIGHTS[name][1]["commit"][:10], WEIGHTS[name][1]["mean"])

def make_model(name, seed):
    torch.manual_seed(seed)                     # the new 30-class head is initialised from the run's seed
    path, meta = WEIGHTS[name]
    m = T.create(name, len(CLASSES), path)
    return m, torch.tensor(meta["mean"]).view(1, 3, 1, 1), torch.tensor(meta["std"]).view(1, 3, 1, 1)

CODE_SHA = sha256_file(os.path.join(ROOT, "notebooks", "src", "02_train_GPU.py"))
COMMON_SHA = sha256_file(os.path.join(ROOT, "notebooks", "p3_common.py"))
TORCH_SHA = sha256_file(os.path.join(ROOT, "notebooks", "p3_torch.py"))
CONFIG = {"img": IMG, "epochs": EPOCHS, "batch": BATCH, "eval_batch": EVAL_BATCH, "wd": WD, "jitter": JITTER,
          "smoke": SMOKE, "split_sha256": SPLIT_SHA256, "cache_fingerprint": CACHE_META["fingerprint"], "donor_seed": DONOR_SEED,
          "weights": {n: m[1]["sha256"] for n, m in WEIGHTS.items()}, "precision": "bf16 autocast" if DEVICE == "cuda" else "fp32",
          "device": DEVICE, "packages": VERSIONS, "code_sha256": CODE_SHA, "common_sha256": COMMON_SHA, "torch_module_sha256": TORCH_SHA, "rows_digest": ROWS_DIGEST,
          "backend": BACKEND, "reload_tol": RELOAD_TOL,
          "n_train": int(len(rows["train"])), "n_val": int(len(rows["val"]))}
CONFIG_SHA = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
print("run signature", CONFIG_SHA[:16], "| code", CODE_SHA[:16])

# %%
# Training and evaluation (plan section 4)
to_input = T.to_input

def augment(u8, gen):
    """After compositing (plan section 4): horizontal flip with probability 0.5 and mild colour jitter per image
    (brightness, contrast, saturation factors from 1 +- JITTER, in this order). Every step acts on each pixel by
    itself: contrast is taken about mid-grey 0.5, not about the image mean, so that the pasted and the kept pixels of a
    composite cannot influence each other (Codex review 002, point 3). Returns float in [0, 1], NCHW."""
    x = u8.permute(0, 3, 1, 2).float().div_(255)
    n = x.shape[0]
    flip = torch.rand(n, generator=gen, device=x.device) < 0.5
    x = torch.where(flip.view(n, 1, 1, 1), x.flip(3), x)
    f = 1 + JITTER * (2 * torch.rand(3, n, 1, 1, 1, generator=gen, device=x.device) - 1)
    x = (x * f[0]).clamp_(0, 1)
    grey = lambda t: (0.299 * t[:, 0:1] + 0.587 * t[:, 1:2] + 0.114 * t[:, 2:3])
    x = ((x - 0.5) * f[1] + 0.5).clamp_(0, 1)
    g = grey(x)
    return ((x - g) * f[2] + g).clamp_(0, 1)

@torch.no_grad()
def predict(model, regime, r, d, mean, std, bs=EVAL_BATCH):
    """Logits (float32, CPU) for image rows r built by the regime with donor rows d. No augmentation."""
    model.eval()
    out = []
    for i in range(0, len(r), bs):
        xb = to_input(build(fetch(regime, r[i:i + bs], d[i:i + bs])), mean, std)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            out.append(model(xb).float().cpu())
    z = torch.cat(out).numpy()
    assert np.isfinite(z).all(), "non-finite logits"
    return z

def run_id(name, regime, seed, lr):
    return f"{name}_{regime}_s{seed}_lr{lr:g}"

def finished(run):
    """A run counts as finished only if its marker is readable, names this run, carries this signature and its
    checkpoint matches its hash."""
    try:
        done = json.load(open(os.path.join(RUN_DIR, run + ".json")))
    except (OSError, ValueError):
        return None
    f = os.path.join(RUN_DIR, run + ".pt")
    if done.get("run") != run or "log" not in done or not os.path.exists(f) or sha256_file(f) != done.get("weights_sha256"):
        return None
    return done if done.get("config_sha") == CONFIG_SHA else None

def batch_feeder(regime, perm, donors, steps):
    """Reads the batches of one epoch on a background thread (disk only; all random numbers are drawn by the caller).
    Order is first in, first out. If the consumer stops early the thread is told to stop and is joined."""
    q, stop = queue.Queue(maxsize=4), threading.Event()
    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.5)
                return True
            except queue.Full:
                pass
        return False
    def work():
        try:
            for b in range(steps):
                sl = slice(b * BATCH, (b + 1) * BATCH)
                if not put((perm[sl], fetch(regime, rows["train"][perm[sl]], rows["train"][donors[sl]]))):
                    return
            put(None)
        except BaseException as e:      # surfaced in the training loop, never swallowed
            put(e)
    th = threading.Thread(target=work, daemon=True)
    th.start()
    try:
        while True:
            item = q.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        th.join(timeout=30)

def run_one(name, family, regime, seed, lr):
    run = run_id(name, regime, seed, lr)
    done = finished(run)
    if done:
        print("resume: finished run found and verified, skipped:", run)
        return done
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model, mean, std = make_model(name, seed)
    model.to(DEVICE); mean, std = mean.to(DEVICE), std.to(DEVICE)
    n = len(rows["train"])
    ytr = torch.from_numpy(labels["train"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    steps = (n + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps)
    lossf = nn.CrossEntropyLoss()
    g_np = np.random.default_rng(seed)                       # order of images and donors (new random donor each epoch)
    g_dev = torch.Generator(device=DEVICE).manual_seed(seed)  # augmentation
    weights_path = os.path.join(RUN_DIR, run + ".pt")
    val_r, val_d = rows["val"], rows["val"][VAL_DONOR]
    best, best_logits, log = None, None, []
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_run = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train(); t0 = time.time(); seen, wait = 0, 0.0
        loss_sum = torch.zeros((), device=DEVICE, dtype=torch.float64)
        perm = g_np.permutation(n)
        donors = g_np.integers(0, n, n)                      # uniform over the training pool, any class (plan section 3)
        feeder = batch_feeder(regime, perm, donors, steps)
        try:
            tw = time.time()
            for idx, parts in feeder:
                wait += time.time() - tw
                xb = (augment(build(parts), g_dev) - mean) / std
                yb = ytr[torch.from_numpy(idx).to(DEVICE)]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                    loss = lossf(model(xb).float(), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step(); sched.step()
                loss_sum += loss.detach().double() * len(idx); seen += len(idx)   # no host synchronisation per batch
                tw = time.time()
        finally:
            feeder.close()
        assert seen == n
        train_loss = float(loss_sum) / seen
        assert np.isfinite(train_loss), f"{run}: training loss is not finite in epoch {epoch}"
        zv = predict(model, regime, val_r, val_d, mean, std)       # validation images built by the model's own regime
        acc, f1, vloss = metrics(labels["val"], zv)
        row = {"run": run, "model": name, "family": family, "regime": regime, "seed": seed, "lr": lr, "epoch": epoch,
               "train_loss": train_loss, "val_loss": vloss, "val_accuracy": acc, "val_macro_f1": f1,
               "lr_end": sched.get_last_lr()[0], "seconds": time.time() - t0, "seconds_waiting_for_data": wait}
        log.append(row)
        print(f"  {run} epoch {epoch}: loss {train_loss:.4f} | val loss {vloss:.4f} acc {acc:.4f} | {row['seconds']:.0f} s (data wait {wait:.0f} s)", flush=True)
        if best is None or vloss < best["val_loss"]:               # lowest validation loss; ties keep the earlier epoch
            best, best_logits = dict(row), zv
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            atomic(weights_path, lambda t: torch.save(state, t))
    peak = torch.cuda.max_memory_allocated() // 2**20 if DEVICE == "cuda" else None
    del opt
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True), strict=True)
    reloaded = model.state_dict()
    assert all(torch.equal(reloaded[k].cpu(), v) for k, v in state.items()), f"{run}: the saved state did not round-trip"
    z_own = predict(model, regime, val_r, val_d, mean, std)
    repeat = float(np.abs(predict(model, regime, val_r, val_d, mean, std) - z_own).max())   # same weights, same inputs, twice
    diff = float(np.abs(z_own - best_logits).max())
    assert diff <= RELOAD_TOL, f"{run}: reloaded checkpoint differs from the best epoch by {diff:.2e} (logits; repeat-forward difference {repeat:.2e})"
    z_orig = z_own if regime == "ORIG" else predict(model, "ORIG", val_r, val_d, mean, std)
    acc_o, f1_o, loss_o = metrics(labels["val"], z_orig)
    done = {"run": run, "model": name, "family": family, "regime": regime, "seed": seed, "lr": lr, "best_epoch": best["epoch"],
            "val_loss_own_regime": best["val_loss"], "val_accuracy_own_regime": best["val_accuracy"],
            "val_macro_f1_own_regime": best["val_macro_f1"], "val_accuracy_original": acc_o, "val_macro_f1_original": f1_o,
            "val_loss_original": loss_o, "reload_max_abs_logit_diff": diff, "repeat_forward_max_abs_logit_diff": repeat,
            "nondeterministic_warnings": sorted(DET_WARNINGS),
            "params": sum(p.numel() for p in model.parameters()), "train_seconds": time.time() - t_run, "peak_gpu_mib": peak,
            "classes": CLASSES, "img": IMG, "mean": WEIGHTS[name][1]["mean"], "std": WEIGHTS[name][1]["std"],
            "pretrained_repo": WEIGHTS[name][1]["repo"], "pretrained_commit": WEIGHTS[name][1]["commit"],
            "checkpoint_format": "plain state_dict of timm.create_model(name, num_classes=30); load with strict=True",
            "weights_sha256": sha256_file(weights_path), "config_sha": CONFIG_SHA, "log": log}
    write_json(os.path.join(RUN_DIR, run + ".json"), done)          # the completion marker is written last
    with open(os.path.join(RUN_DIR, "progress.txt"), "a") as f:     # for watching a long run from outside
        f.write(f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')} {run} best_epoch {best['epoch']} "
                f"val_acc_own {best['val_accuracy']:.4f} val_acc_orig {acc_o:.4f} {done['train_seconds'] / 60:.1f} min\n")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return done

# %%
# GPU memory probe: each model, two training steps and a few evaluation batches. Stops before any run if one needs
# more than 90% of the GPU.
probe = {}
if DEVICE == "cuda":
    total = torch.cuda.get_device_properties(0).total_memory
    free0 = torch.cuda.mem_get_info()[0]
    print(f"GPU memory free before the probe: {free0 / 2**30:.1f} of {total / 2**30:.1f} GiB")
    lossf = nn.CrossEntropyLoss()
    for name, family, *_ in MODELS:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model, mean, std = make_model(name, 0); model.to(DEVICE); mean, std = mean.to(DEVICE), std.to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=WD)
        g = torch.Generator(device=DEVICE).manual_seed(0)
        nb = min(BATCH, len(rows["train"]))
        r, d = rows["train"][:nb], rows["train"][-nb:]
        yb = torch.from_numpy(labels["train"][:nb]).to(DEVICE)
        for _ in range(2):                                   # the same update path as run_one
            model.train()
            xb = (augment(build(fetch("BG", r, d)), g) - mean) / std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(model(xb).float(), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        ne = min(4 * EVAL_BATCH, len(rows["val"]))
        predict(model, "LEAF", rows["val"][:ne], rows["val"][VAL_DONOR[:ne]], mean, std)
        probe[name] = {"peak_allocated_share": torch.cuda.max_memory_allocated() / total,
                       "peak_reserved_share": torch.cuda.max_memory_reserved() / total, "free_before_share": free0 / total}
        del model, opt
        print(f"memory probe {name}: allocated {probe[name]['peak_allocated_share']:.1%}, reserved {probe[name]['peak_reserved_share']:.1%} of {total / 2**30:.1f} GiB")
    torch.cuda.empty_cache()
    worst = max(v["peak_reserved_share"] for v in probe.values())
    assert worst <= MAX_GPU_SHARE and worst <= free0 / total, f"a model needs more GPU memory than allowed or than is free: {probe}"

# %%
# Learning rate per architecture: ORIG regime, seed 11, both rates; the lower best validation loss wins (ties: the
# smaller rate). The winning run is the ORIG seed-11 run of that architecture; the rate is then fixed for every run.
t_all = time.time()
if not SMOKE:
    save_output(val_donors, "val_donors.csv", "Fixed donor of every validation image (cache rows), used for own-regime validation images in training and reused by notebook 03", "learning rate")
lr_rows, LR = [], {}
for name, family, *_ in MODELS:
    cand = [run_one(name, family, LR_REGIME, LR_SEED, lr) for lr in LR_GRID]
    best = min(cand, key=lambda d: (d["val_loss_own_regime"], d["lr"]))
    LR[name] = best["lr"]
    for d in cand:
        lr_rows.append({"model": name, "lr": d["lr"], "best_epoch": d["best_epoch"], "val_loss": d["val_loss_own_regime"],
                        "val_accuracy": d["val_accuracy_own_regime"], "train_minutes": d["train_seconds"] / 60, "chosen": d["lr"] == best["lr"]})
lr_tab = pd.DataFrame(lr_rows)
print(lr_tab.to_string(index=False))
if not SMOKE:
    save_output(lr_tab, "lr_choice.csv", "Learning-rate choice per architecture: ORIG regime, seed 11, best validation loss at each rate, chosen rate, measured training time", "learning rate")
STOP_AFTER_LR = STAGE == "lr"
if STOP_AFTER_LR:
    print("STAGE=lr: stopping after the learning-rate runs, as asked.")

# %%
# The 36 runs: seed by seed, BG then LEAF then ORIG (a stop loses at most the run in progress; verified runs are skipped)
results = []
if not STOP_AFTER_LR:
    todo = [(name, family, regime, seed) for seed in SEEDS for regime in REGIMES for name, family, *_ in MODELS]
    for k, (name, family, regime, seed) in enumerate(todo):
        print(f"=== {name} {regime} seed {seed} ({k + 1}/{len(todo)}) ===", flush=True)
        results.append(run_one(name, family, regime, seed, LR[name]))
    print(f"all runs done in {(time.time() - t_all) / 60:.1f} min")

# %%
# Assemble outputs: logs, run summary, checkpoint hashes, environment. No test image has been read.
if not STOP_AFTER_LR:
    assert len({d["run"] for d in results}) == len(MODELS) * len(REGIMES) * len(SEEDS)
    log = pd.DataFrame([r for d in results for r in d["log"]])
    summary = pd.DataFrame([{k: v for k, v in d.items() if k not in ("log", "classes", "mean", "std")} for d in results])
    print(summary[["run", "best_epoch", "val_accuracy_own_regime", "val_accuracy_original", "train_seconds", "peak_gpu_mib"]].to_string(index=False))
    pre_tab = pd.DataFrame([WEIGHTS[n][1] for n, *_ in MODELS])
    if SMOKE:
        print("SMOKE TEST OK:", len(results), "runs; nothing written to outputs/.")
    else:
        save_output(log, "train_log.csv", "Per-epoch training loss, validation loss, accuracy and macro F1 (validation images built by the run's own regime), learning rate and time, all 36 runs", "assemble")
        save_output(summary, "train_runs.csv", "One row per run: regime, seed, learning rate, best epoch (lowest validation loss), validation scores on own-regime and on original images, reload check, training time, peak GPU memory, checkpoint hash", "assemble")
        save_output(pre_tab, "pretrained_weights.csv", "Pretrained weight files: Hugging Face repository, pinned commit, SHA-256, normalisation constants", "assemble")
        save_output({**ENV, "config": CONFIG, "config_sha": CONFIG_SHA, "lr": LR, "memory_probe_peak_share": probe,
                     "session_minutes": (time.time() - t_all) / 60, "training_minutes_all_runs": float(sum(d["train_seconds"] for d in results)) / 60,
                     "lr_stage_minutes": float(lr_tab.train_minutes.sum()), "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
                    "train_environment.json", "Training environment, run signature (configuration, input, weight, code and package fingerprints), chosen learning rates, memory probe and total time", "assemble")
