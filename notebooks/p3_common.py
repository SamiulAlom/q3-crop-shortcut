"""Shared image code for paper-03 (analysis_plan.md section 3). Imported by notebooks 01-03 so that the mask,
the filled background and the three constructions are computed by one piece of code everywhere.
No network is used anywhere in this file. Edit here, never in a notebook copy.
"""
import hashlib
import numpy as np
from PIL import Image
from scipy import ndimage

COMMON_VERSION = "2"
SIDE = 224                 # plan section 3: resize to 224 x 224
MASK_T = 16                # a pixel of the segmented copy is leaf if its largest channel is above this (0-255)
MASK_MIN_COMPONENT = 0.05  # components smaller than this share of the largest component are dropped
FILL_SIGMA_FRAC = 1 / 8    # plan section 3: sigma of the normalised convolution, as a share of the image side
ALT_SIGMA_FRAC = 1 / 4     # plan section 4, artefact check (i)
FEATHER_PX = 3             # plan section 4, artefact check (ii)
EPS = 1e-6


def ident(file_name):
    """Image identifier, by the dataset owners' rule (plant_village.py at the pinned commit)."""
    s = file_name.replace("_final_masked", "")
    if "___" in s:
        s = s.split("___")[-1]
    s = s.split("copy")[0]
    for e in (".jpg", ".JPG", ".png", ".PNG"):
        s = s.replace(e, "")
    return s.strip()


def loader_leaf_id(class_name, file_name, leaf_map):
    """Leaf identifier by the owners' rule; None where their loader would write a 'fallback_' value."""
    key = ident(file_name).lower().strip()
    if key in leaf_map:
        sug = leaf_map[key]
        if len(sug) == 1:
            return sug[0]
        for s in sug:
            if class_name in s:
                return s
    return None


def load_rgb(path, side=None):
    im = Image.open(path).convert("RGB")
    if side is not None and im.size != (side, side):
        im = im.resize((side, side), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def leaf_mask(seg_rgb, t=MASK_T):
    """Leaf mask from the segmented copy: non-black pixels, holes filled, small components dropped. One rule for all."""
    m = seg_rgb.max(axis=2) > t
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, index=np.arange(1, n + 1))
        keep = np.flatnonzero(sizes >= MASK_MIN_COMPONENT * sizes.max()) + 1
        m = np.isin(lab, keep)
    return m


def resize_mask(mask, side=SIDE):
    """Mask at the model's input size: area-averaged, then cut at one half."""
    if mask.shape == (side, side):
        return mask.copy()
    im = Image.fromarray((mask * 255).astype(np.uint8)).resize((side, side), Image.BOX)
    return np.asarray(im) >= 128


def filled_background(rgb, mask, sigma_frac=FILL_SIGMA_FRAC):
    """Background of an image with the leaf region filled by normalised convolution of the off-leaf pixels.
    Off-leaf pixels are returned unchanged. Where no off-leaf pixel is within reach of the kernel, the mean off-leaf
    colour is used. uint8, same shape as rgb."""
    bg = (~mask).astype(np.float64)
    sigma = sigma_frac * rgb.shape[0]
    den = ndimage.gaussian_filter(bg, sigma, mode="nearest")
    out = np.empty(rgb.shape, dtype=np.float64)
    n_bg = bg.sum()
    for c in range(3):
        ch = rgb[..., c].astype(np.float64)
        num = ndimage.gaussian_filter(ch * bg, sigma, mode="nearest")
        mean_c = (ch * bg).sum() / n_bg if n_bg > 0 else 0.0
        fill = np.where(den > EPS, num / np.maximum(den, EPS), mean_c)
        out[..., c] = np.where(mask, fill, ch)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def soft_edge(mask, feather_px=FEATHER_PX):
    """Feathered mask in [0, 1] (Gaussian blur of the hard mask), used only in the artefact check."""
    return ndimage.gaussian_filter(mask.astype(np.float64), feather_px, mode="nearest")


def compose(leaf_rgb, leaf_m, background_rgb, feather_px=0):
    """Leaf pixels of one image on a (filled) background. The same function builds a background swap (leaf of x on the
    donor's filled background), a leaf swap (donor's leaf on x's filled background) and a self rebuild."""
    if feather_px:
        a = soft_edge(leaf_m, feather_px)[..., None]
        return np.clip(np.rint(a * leaf_rgb + (1 - a) * background_rgb), 0, 255).astype(np.uint8)
    return np.where(leaf_m[..., None], leaf_rgb, background_rgb).astype(np.uint8)


def pixel_sha(rgb):
    return hashlib.sha256(f"{rgb.shape[1]}x{rgb.shape[0]}:".encode() + rgb.tobytes()).hexdigest()


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class GitBlobs:
    """Reads file contents straight from a git object store (`git cat-file --batch`). Needed for PlantDoc: 87 of its
    file names cannot exist on Windows and 6 pairs differ only by letter case, so the repository cannot be checked out."""

    def __init__(self, repo_dir):
        import subprocess
        self.p = subprocess.Popen(["git", "cat-file", "--batch"], cwd=repo_dir, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE)

    def read(self, blob_sha):
        self.p.stdin.write(blob_sha.encode() + b"\n")
        self.p.stdin.flush()
        head = self.p.stdout.readline().split()
        assert len(head) == 3 and head[1] == b"blob", f"unexpected answer for {blob_sha}: {head}"
        data = self.p.stdout.read(int(head[2]))
        self.p.stdout.read(1)
        return data

    def close(self):
        self.p.stdin.close()
        self.p.wait()


ALIGN_SHIFT = 4        # pixels; the segmented copy is also compared with the colour image moved by this much
ALIGN_HP_SIGMA = 2.0   # the comparison uses grey values minus their Gaussian blur of this sigma (fine texture only)
ALIGN_ERODE = 6        # the comparison stays this many pixels inside the mask
ALIGN_MIN_CORR = 0.15  # deviation 3: true twins >= 0.228, wrong twins <= 0.05 on 1,500 images (decisions/evidence/005)


def _highpass(x):
    g = x.astype(np.float64).mean(axis=2)
    return g - ndimage.gaussian_filter(g, ALIGN_HP_SIGMA, mode="nearest")


def alignment(rgb, seg, mask):
    """Is the segmented copy registered with the colour image? (plan deviation 3)
    The owners' segmented copy has different colours from the colour image, so pixel values cannot be compared.
    align_corr: correlation, inside the eroded mask, of the two images' fine texture (grey minus its blur).
    align_corr_shifted_max: the largest such correlation when the colour image is moved by ALIGN_SHIFT pixels in one
    of four directions. A registered twin has align_corr >= ALIGN_MIN_CORR and above every shifted value.
    align_mad: mean absolute RGB difference inside the mask (descriptive: it measures the colour change)."""
    out = {"align_corr": float("nan"), "align_corr_shifted_max": float("nan"), "align_mad": float("nan")}
    core = ndimage.binary_erosion(mask, iterations=ALIGN_ERODE)       # erosion keeps core away from the border, so rolled-in pixels are never read
    if core.sum() < 64:
        return out
    a, b = _highpass(rgb), _highpass(seg)
    out["align_mad"] = float(np.abs(rgb.astype(np.float64) - seg.astype(np.float64))[core].mean())
    bb = b[core]
    if bb.std() == 0:
        return out

    def corr(x):
        xx = x[core]
        return float(np.corrcoef(xx, bb)[0, 1]) if xx.std() > 0 else float("nan")
    out["align_corr"] = corr(a)
    sh = [corr(np.roll(a, s, axis=(0, 1))) for s in ((ALIGN_SHIFT, 0), (-ALIGN_SHIFT, 0), (0, ALIGN_SHIFT), (0, -ALIGN_SHIFT))]
    out["align_corr_shifted_max"] = float(np.nanmax(sh)) if not np.all(np.isnan(sh)) else float("nan")
    return out


def aligned(corr, corr_shifted_max):
    """The registration rule of deviation 3 (NaN fails)."""
    return bool(corr >= ALIGN_MIN_CORR and corr > corr_shifted_max)


def cache_row_sha(rgb, mask, bg):
    """Digest of one cache row (224 x 224 image, mask, filled background); frozen in the split file."""
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes() + np.ascontiguousarray(mask).tobytes() + np.ascontiguousarray(bg).tobytes()).hexdigest()


def audit_one(args):
    """Worker for notebook 01: one colour image and its segmented twin. Returns a dict of audit values and, when both
    files can be read, the 224 x 224 colour image, mask and filled background (for the cache)."""
    idx, color_path, seg_path = args
    import imagehash
    row = {"idx": idx, "error": ""}
    try:
        row["sha256"] = file_sha(color_path)
        if seg_path is not None:
            row["seg_sha256"] = file_sha(seg_path)
        with Image.open(color_path) as im:
            row["width"], row["height"], row["mode"] = im.width, im.height, im.mode
            rgb_native = np.asarray(im.convert("RGB"), dtype=np.uint8)
            row["phash"] = str(imagehash.phash(im.convert("L")))
        row["pixel_sha256"] = pixel_sha(rgb_native)
        if seg_path is None:
            return row, None
        seg_native = load_rgb(seg_path)
        row["seg_width"], row["seg_height"] = seg_native.shape[1], seg_native.shape[0]
        if seg_native.shape != rgb_native.shape:
            row["error"] = "size_mismatch"
            return row, None
        m = leaf_mask(seg_native)
        row["mask_share"] = float(m.mean())
        for t in (8, 32):
            row[f"mask_share_t{t}"] = float(leaf_mask(seg_native, t).mean())
        row.update(alignment(rgb_native, seg_native, m))
        rgb = load_rgb(color_path, SIDE)
        m224 = resize_mask(m, SIDE)
        row["mask_share_224"] = float(m224.mean())
        bg = filled_background(rgb, m224)
        row["cache_row_sha256"] = cache_row_sha(rgb, m224, bg)
        return row, (rgb, m224, bg)
    except Exception as e:  # recorded and counted; never silently dropped
        row["error"] = f"{type(e).__name__}: {e}"[:200]
        return row, None
