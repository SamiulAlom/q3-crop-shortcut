"""Method tests for p3_common.py on generated arrays (test fixtures only; never used as data).
Run: .venv-local/Scripts/python.exe projects/qpain/paper-03/notebooks/tests/test_p3_common.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p3_common as P

rng = np.random.default_rng(0)
S = P.SIDE

# identifier and leaf-id rule
lm = {"frec_scab 3335": ["Apple___Apple_scab:::7.0"], "x 1": ["A___b:::1.0", "C___d:::2.0"]}
assert P.loader_leaf_id("Apple___Apple_scab", "uuid___FREC_Scab 3335.JPG", lm) == "Apple___Apple_scab:::7.0"
assert P.loader_leaf_id("C___d", "uuid___X 1.JPG", lm) == "C___d:::2.0"
assert P.loader_leaf_id("E___f", "uuid___X 1.JPG", lm) is None          # two suggestions, none of this class
assert P.loader_leaf_id("E___f", "uuid___nothing.JPG", lm) is None

# mask: a bright disc with a dark hole and a small bright speck far away
yy, xx = np.mgrid[:256, :256]
disc = (yy - 128) ** 2 + (xx - 128) ** 2 < 80 ** 2
seg = np.zeros((256, 256, 3), np.uint8)
seg[disc] = (40, 120, 30)
seg[120:136, 120:136] = 0                       # dark lesion inside the leaf: must be filled
seg[5:9, 5:9] = 200                             # speck: 16 px, below 5% of the disc, must be dropped
seg = np.clip(seg.astype(int) + rng.integers(0, 6, seg.shape), 0, 255).astype(np.uint8)   # JPEG-like noise in the black
m = P.leaf_mask(seg)
assert (m == disc).mean() > 0.999 and m[128, 128] and not m[6, 6], "mask rule failed"
m224 = P.resize_mask(m)
assert m224.shape == (S, S) and abs(m224.mean() - disc.mean()) < 0.01

# filled background: off-leaf pixels unchanged; a constant background is reproduced exactly inside the leaf
rgb = np.empty((S, S, 3), np.uint8); rgb[:] = (90, 100, 110); rgb[m224] = (10, 200, 10)
bg = P.filled_background(rgb, m224)
assert (bg[~m224] == rgb[~m224]).all() and (bg[m224] == (90, 100, 110)).all(), "constant background not reproduced"
# a left-to-right gradient background: the fill stays between the extremes and carries no leaf colour
grad = np.tile(np.linspace(50, 200, S)[None, :, None], (S, 1, 3)).astype(np.uint8)
img = grad.copy(); img[m224] = (0, 255, 0)
bg = P.filled_background(img, m224)
assert bg[m224].min() >= 50 and bg[m224].max() <= 200 and (bg[..., 0] == bg[..., 1]).all()
# full mask: no off-leaf pixel anywhere -> no NaN, no crash (such images are excluded by the 95% rule anyway)
assert P.filled_background(img, np.ones((S, S), bool)).shape == (S, S, 3)
# leaf far from any background (kernel support): fallback to the mean off-leaf colour, not zero
big = np.ones((S, S), bool); big[:2] = False
out = P.filled_background(rgb, big)
assert np.isfinite(out).all() and out[S - 1, S // 2].tolist() == [90, 100, 110]

# compose: self rebuild with a hard edge returns the original image exactly
assert (P.compose(rgb, m224, P.filled_background(rgb, m224)) == rgb).all()
# background swap keeps the leaf pixels of x and the off-leaf-of-x pixels of the donor background
donor_bg = np.full((S, S, 3), 33, np.uint8)
sw = P.compose(rgb, m224, donor_bg)
assert (sw[m224] == rgb[m224]).all() and (sw[~m224] == 33).all()
# feathered edge changes only pixels near the mask edge
fe = P.compose(rgb, m224, P.filled_background(rgb, m224), feather_px=P.FEATHER_PX)
from scipy import ndimage
far = ndimage.binary_erosion(m224, iterations=12) | ~ndimage.binary_dilation(m224, iterations=12)
assert (fe[far] == rgb[far]).all() and (fe != rgb).any()
print("p3_common tests passed (version", P.COMMON_VERSION + ")")

# alignment (deviation 3): a recoloured twin passes; a twin moved by 4 px fails; another image fails
tex = rng.integers(0, 255, (256, 256, 3)).astype(np.uint8)
tex = ndimage.gaussian_filter(tex.astype(float), (1, 1, 0)).astype(np.uint8)
recoloured = np.clip(tex.astype(float) * [0.8, 1.05, 0.7] + [10, 5, -8], 0, 255).astype(np.uint8)      # same content, other colours
al = P.alignment(tex, np.where(disc[..., None], recoloured, 0).astype(np.uint8), disc)
assert P.aligned(al["align_corr"], al["align_corr_shifted_max"]) and al["align_corr"] > 0.9 and al["align_mad"] > 5, al
moved = np.roll(recoloured, (4, 0), axis=(0, 1))
al2 = P.alignment(tex, np.where(disc[..., None], moved, 0).astype(np.uint8), disc)
assert not P.aligned(al2["align_corr"], al2["align_corr_shifted_max"]), "a moved twin was accepted"
other = ndimage.gaussian_filter(rng.integers(0, 255, (256, 256, 3)).astype(float), (1, 1, 0)).astype(np.uint8)
al3 = P.alignment(tex, np.where(disc[..., None], other, 0).astype(np.uint8), disc)
assert not P.aligned(al3["align_corr"], al3["align_corr_shifted_max"]), "another image was accepted as the twin"
assert not P.aligned(float("nan"), 0.0)
assert len(P.cache_row_sha(rgb, m224, bg)) == 64 and P.cache_row_sha(rgb, m224, bg) != P.cache_row_sha(rgb, ~m224, bg)
print("alignment tests passed")
