"""Statistics for paper-03 (analysis_plan.md section 7), kept apart from the notebook so that the method test
(tests/test_p3_stats.py) runs on exactly the code that produces the results.

Every confirmatory quantity is a function of one number per image: the statistic is the unweighted mean of the class
means of that number (section 7, "class weighting"). Inference resamples the images within each class, the same
resample for every model and method (paired), with seeds, architectures and checkpoints fixed.
"""
import numpy as np

STATS_VERSION = "1"
B_DEFAULT = 10_000
ALPHA = 0.05


def class_mean(v, cls):
    """Unweighted mean of the class means of v. v: (n,) or (n, k); cls: (n,) integer codes. NaN if a class is empty of
    finite values (reported by the caller, never silently dropped)."""
    v = np.asarray(v, dtype=np.float64)
    assert np.isfinite(v).all(), "non-finite per-image value"
    codes = np.unique(cls)
    return np.mean([v[cls == c].mean(axis=0) for c in codes], axis=0)


def resample_indices(cls, B, seed):
    """(B, n) indices: within each class, n_c draws with replacement from that class's images. One generator, classes
    in sorted order, so the result depends only on (cls, B, seed)."""
    rng = np.random.default_rng(seed)
    cls = np.asarray(cls)
    out = np.empty((B, len(cls)), dtype=np.int64)
    for c in np.unique(cls):
        pos = np.flatnonzero(cls == c)
        out[:, pos] = pos[rng.integers(0, len(pos), (B, len(pos)))]
    return out


def boot_class_mean(v, cls, idx):
    """Bootstrap replicates of class_mean for one or several per-image columns. v: (n,) or (n, k); idx from
    resample_indices. Returns (B,) or (B, k). Works class by class to bound memory."""
    v = np.asarray(v, dtype=np.float64)
    cls = np.asarray(cls)
    codes = np.unique(cls)
    acc = 0.0
    for c in codes:
        pos = np.flatnonzero(cls == c)
        acc = acc + v[idx[:, pos]].mean(axis=1)       # idx[:, pos] holds only images of class c
    return acc / len(codes)


def summarise(theta, reps, test=True):
    """Estimate, percentile interval and, if test, the two-sided p-value of the centred bootstrap (section 7):
    p = (1 + #{|theta* - theta| >= |theta|}) / (B + 1)."""
    reps = np.asarray(reps, dtype=np.float64)
    assert np.isfinite(theta) and np.isfinite(reps).all(), "non-finite estimate or bootstrap replicate"
    lo, hi = np.percentile(reps, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    out = {"estimate": float(theta), "ci_low": float(lo), "ci_high": float(hi), "boot_se": float(reps.std(ddof=1)), "B": int(len(reps))}
    if test:
        out["p"] = float((1 + np.sum(np.abs(reps - theta) >= abs(theta))) / (len(reps) + 1))
    return out


def holm(p, m=None):
    """Holm-adjusted p-values. p may contain NaN for contrasts that could not be estimated: they keep their place in
    the family (m counts them) and are never rejected; estimable ones are adjusted as if the missing p-values were 1."""
    p = np.asarray(p, dtype=np.float64)
    m = len(p) if m is None else m
    adj = np.full(len(p), np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    order = ok[np.argsort(p[ok], kind="stable")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def spearman(x, y):
    """Spearman correlation with average ranks; NaN if either input is constant (reported as undefined)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    assert np.isfinite(x).all() and np.isfinite(y).all(), "non-finite value in a correlation"
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(a):
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(1, len(a) + 1)
    vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return (sums / cnt)[inv]


def tie_share(a):
    """Share of values that are tied with at least one other value."""
    _, cnt = np.unique(np.asarray(a), return_counts=True)
    return float(cnt[cnt > 1].sum() / len(a))
