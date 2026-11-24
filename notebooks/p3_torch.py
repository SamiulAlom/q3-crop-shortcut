"""Shared torch code for paper-03: cache access, device compositing, model loading and the attribution methods
(analysis_plan.md sections 3-5). Imported by notebooks 02 and 03 so that both build images and models the same way.
Edit here, never in a notebook copy.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import timm
import p3_common as P

TORCH_VERSION = "1"
IMG = P.SIDE
IG_STEPS = 32            # plan section 5
GC_LAYER = {             # plan section 5: the last layer whose output is larger than 1 x 1 (checked by gc_layer_check)
    "resnet50": ("layer4", "conv"),
    "efficientnet_b0": ("bn2", "conv"),
    "deit_small_patch16_224": ("blocks.11.norm1", "tokens_cls"),
    "vit_small_patch16_224": ("blocks.11.norm1", "tokens_cls"),
}


# ---------------------------------------------------------------- cache rows and compositing
def read_rows(arr, r):
    """Rows of a memory-mapped cache in the given order (read in sorted order for the disk, then put back)."""
    r = np.asarray(r, dtype=np.int64)
    o = np.argsort(r, kind="stable")
    out = np.empty((len(r),) + arr.shape[1:], dtype=arr.dtype)
    out[o] = arr[r[o]]
    return out


class Cache:
    """The three arrays written by notebook 01 (row = column `idx` of pv_split.csv)."""

    def __init__(self, cache_dir):
        self.rgb = np.load(os.path.join(cache_dir, "rgb224.npy"), mmap_mode="r")
        self.bg = np.load(os.path.join(cache_dir, "bg224.npy"), mmap_mode="r")      # filled backgrounds
        self.mask = np.load(os.path.join(cache_dir, "mask224.npy"), mmap_mode="r")
        assert self.rgb.shape[1:] == (IMG, IMG, 3) and self.bg.shape == self.rgb.shape and self.mask.shape == self.rgb.shape[:3]
        assert self.rgb.dtype == np.uint8 and self.bg.dtype == np.uint8 and self.mask.dtype == np.bool_

    def fetch(self, kind, r, d=None):
        """CPU arrays for a batch. kind: 'ORIG' (image r), 'LEAF' = background swap (leaf of r on the filled background
        of donor d), 'BG' = leaf swap (leaf of donor d on the filled background of r), 'REMOVED' (filled background of r).
        Returns (image,) or (leaf image, leaf mask, background)."""
        if kind == "ORIG":
            return (read_rows(self.rgb, r),)
        if kind == "REMOVED":
            return (read_rows(self.bg, r),)
        if kind == "LEAF":
            return read_rows(self.rgb, r), read_rows(self.mask, r), read_rows(self.bg, d)
        if kind == "BG":
            return read_rows(self.rgb, d), read_rows(self.mask, d), read_rows(self.bg, r)
        raise ValueError(kind)


def build(parts, device):
    """uint8 (N, 224, 224, 3) on the device. Same rule as p3_common.compose with a hard edge."""
    t = [torch.from_numpy(np.ascontiguousarray(a)).to(device, non_blocking=True) for a in parts]
    return t[0] if len(t) == 1 else torch.where(t[1].unsqueeze(-1), t[0], t[2])


def to_input(u8, mean, std):
    """uint8 (N, 224, 224, 3) on the device -> normalised float (N, 3, 224, 224), the model's pretrained mean and std."""
    return (u8.permute(0, 3, 1, 2).float().div_(255) - mean) / std


# ---------------------------------------------------------------- models
def create(name, n_classes, weight_file=None):
    if weight_file is None:
        return timm.create_model(name, pretrained=False, num_classes=n_classes)
    return timm.create_model(name, pretrained=True, num_classes=n_classes, pretrained_cfg_overlay=dict(file=weight_file))


def get_module(model, path):
    m = model
    for p in path.split("."):
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m


def gc_layer_check(model, name, device):
    """The Grad-CAM layer rule of plan section 5, checked on the model: the layer's output is larger than 1 x 1 and
    nothing after it keeps a spatial grid (CNNs), or it is the first normalisation of the last block (transformers)."""
    layer, kind = GC_LAYER[name]
    store = {}
    h = get_module(model, layer).register_forward_hook(lambda mod, i, o: store.__setitem__("a", o))
    try:
        with torch.no_grad():
            model.eval()(torch.zeros(1, 3, IMG, IMG, device=device))
    finally:
        h.remove()
    a = store["a"]
    if kind == "conv":
        assert a.dim() == 4 and a.shape[-1] > 1 and a.shape[-2] > 1, (name, a.shape)
        with torch.no_grad():
            xz = torch.zeros(1, 3, IMG, IMG, device=device)
            same = torch.allclose(model.forward_head(a), model(xz), atol=1e-4, rtol=1e-4)
        assert same, f"{name}: {layer} is not the last spatial stage (the model's head applied to it does not give the logits)"
    else:
        assert a.dim() == 3 and a.shape[1] == 1 + (IMG // 16) ** 2, (name, a.shape)
        assert layer == f"blocks.{len(model.blocks) - 1}.norm1"
    return tuple(a.shape)


# ---------------------------------------------------------------- attribution methods
# Every method returns a non-negative float32 tensor (N, h, w) on the device, NOT yet normalised, plus nothing else.
# normalise() turns it into the map that is scored.
def normalise(m):
    """(N, h, w) non-negative -> (N, 224, 224) summing to one (bilinear upsampling of coarse maps, plan section 5).
    Returns the maps and a tensor of codes per map: 0 = used as computed, 1 = zero mass, 2 = a non-finite value or a
    non-finite sum. Maps with code 1 or 2 are replaced by the uniform map; both are counted and reported."""
    m = m.double()
    bad = ~torch.isfinite(m).flatten(1).all(1)
    m = torch.where(bad.view(-1, 1, 1), torch.zeros_like(m), m).clamp_min(0)
    if m.shape[-1] != IMG or m.shape[-2] != IMG:
        m = F.interpolate(m[:, None], size=(IMG, IMG), mode="bilinear", align_corners=False)[:, 0].clamp_min(0)
    s = m.flatten(1).sum(1)
    bad = bad | ~torch.isfinite(s)
    zero = ~bad & ~(s > 0)
    replace = bad | zero
    m = torch.where(replace.view(-1, 1, 1), torch.ones_like(m), m)
    code = torch.where(bad, 2, torch.where(zero, 1, 0))
    return (m / m.flatten(1).sum(1).view(-1, 1, 1)).float(), code


def gradcam(model, xn, targets, name):
    """Grad-CAM (Selvaraju et al. 2017): ReLU of the gradient-weighted sum of the target layer's channels, at the
    layer's own resolution. xn: normalised input (N, 3, 224, 224); targets: class per image (LongTensor)."""
    layer, kind = GC_LAYER[name]
    store = {}
    h = get_module(model, layer).register_forward_hook(lambda mod, i, o: store.__setitem__("a", o))
    try:
        with torch.enable_grad():
            x = xn.clone().requires_grad_(True)          # makes the graph exist although weights are frozen
            logits = model(x).float()
            a = store["a"]
            g, = torch.autograd.grad(logits.gather(1, targets.view(-1, 1)).sum(), a)
    finally:
        h.remove()
    a, g = a.detach().float(), g.float()
    if kind == "tokens_cls":                             # tokens (B, 1 + s*s, C) -> (B, C, s, s), class token dropped
        a, g = a[:, 1:], g[:, 1:]
        s = int(round(a.shape[1] ** 0.5)); assert s * s == a.shape[1]
        a, g = (t.transpose(1, 2).reshape(t.shape[0], t.shape[2], s, s) for t in (a, g))
    return F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1))


def integrated_gradients(model, xn, x0n, targets, steps=IG_STEPS, chunk=64):
    """Integrated Gradients (Sundararajan et al. 2017), midpoint rule, baseline x0n (black in RGB space, normalised),
    absolute value summed over channels. Also returns the signed sum and the logit difference for the completeness check."""
    maps, signed, delta = [], [], []
    alphas = ((torch.arange(steps, device=xn.device) + 0.5) / steps).view(-1, 1, 1, 1)
    for i in range(len(xn)):
        x, t = xn[i:i + 1], int(targets[i])
        gsum = torch.zeros_like(x[0])
        for c0 in range(0, steps, chunk):
            with torch.enable_grad():
                path = (x0n + alphas[c0:c0 + chunk] * (x - x0n)).requires_grad_(True)
                g, = torch.autograd.grad(model(path).float()[:, t].sum(), path)
            gsum += g.float().sum(0)
        attr = (x - x0n)[0] * gsum / steps
        maps.append(attr.abs().sum(0))
        signed.append(attr.sum())
        with torch.no_grad():
            lg = model(torch.cat([x, x0n])).float()[:, t]
        delta.append(lg[0] - lg[1])
    return torch.stack(maps), torch.stack(signed), torch.stack(delta)


class _Attention:
    """Captures the attention probabilities of every block of a timm VisionTransformer (fused attention switched off
    for the duration), in forward order."""

    def __init__(self, model):
        self.model, self.att, self.hooks = model, [], []

    def __enter__(self):
        self.fused = [blk.attn.fused_attn for blk in self.model.blocks]
        for blk in self.model.blocks:
            blk.attn.fused_attn = False
            self.hooks.append(blk.attn.attn_drop.register_forward_hook(lambda mod, i, o: self.att.append(o)))
        return self

    def __exit__(self, *exc):
        for h in self.hooks:
            h.remove()
        for blk, f0 in zip(self.model.blocks, self.fused):
            blk.attn.fused_attn = f0


def rollout(model, xn):
    """Attention rollout (Abnar and Zuidema 2020): head-averaged attention plus identity, row-normalised, multiplied
    through the blocks; the class-token row over the patches. Does not depend on the class explained."""
    with _Attention(model) as cap, torch.no_grad():
        model(xn)
        R = None
        for a in cap.att:
            a = a.float().mean(1)
            a = a + torch.eye(a.shape[-1], device=a.device)
            a = a / a.sum(-1, keepdim=True)
            R = a if R is None else a @ R
    s = int(round((R.shape[-1] - 1) ** 0.5))
    return R[:, 0, 1:].reshape(-1, s, s)


def transformer_attribution(model, xn, targets):
    """Transformer attribution in the attention-and-gradient form (the rule of Chefer et al. 2021 with relevance
    replaced by attention times its gradient, as in their "generic attention explainability" update): per block, the positive part of
    (gradient of the class logit with respect to the attention) x (attention), averaged over heads; relevance
    R <- R + A_bar R starting from the identity; the class-token row over the patches."""
    with _Attention(model) as cap:
        with torch.enable_grad():
            x = xn.clone().requires_grad_(True)
            logits = model(x).float()
            grads = torch.autograd.grad(logits.gather(1, targets.view(-1, 1)).sum(), cap.att)
        n = cap.att[0].shape[-1]
        R = torch.eye(n, device=xn.device).expand(len(xn), n, n).clone()
        for a, g in zip(cap.att, grads):
            abar = (g.float() * a.detach().float()).clamp_min(0).mean(1)
            R = R + abar @ R
    s = int(round((n - 1) ** 0.5))
    return (R[:, 0, 1:] ).reshape(-1, s, s)


def saliency_order(m, tie_rank):
    """Pixel order for deletion and insertion: decreasing map value; exactly equal values in the order of tie_rank
    (a fixed random ranking of the image's pixels). Two stable sorts, so values that differ by any amount keep their order."""
    v = m.flatten().double()
    first = torch.argsort(tie_rank, stable=True)
    return first[torch.argsort(-v[first], stable=True)]


# ---------------------------------------------------------------- model-independent control maps (plan section 5)
_SOBEL = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])


def control_maps(u8, mask):
    """u8: (N, 224, 224, 3) uint8 on the device; mask: (N, 224, 224) bool. Returns {name: (N, 224, 224) non-negative}."""
    x = u8.float() / 255
    grey = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[:, None]
    k = _SOBEL.to(u8.device)
    p = F.pad(grey, (1, 1, 1, 1), mode="reflect")
    edge = torch.sqrt(F.conv2d(p, k.view(1, 1, 3, 3)) ** 2 + F.conv2d(p, k.t().reshape(1, 1, 3, 3)) ** 2)[:, 0]
    c = (torch.arange(IMG, device=u8.device).float() - (IMG - 1) / 2) / (IMG / 4)
    centre = torch.exp(-0.5 * (c.view(-1, 1) ** 2 + c.view(1, -1) ** 2)).expand(len(u8), -1, -1)
    return {"SIL": mask.float(), "EDGE": edge, "CENTRE": centre, "UNIF": torch.ones_like(centre)}


# ---------------------------------------------------------------- outcome
def area_off_leaf(mask):
    """a of plan section 6: the share of pixels outside the leaf mask, from integer counts, float64."""
    return (~mask).flatten(1).sum(1).double() / mask[0].numel()


def off_leaf_share(maps, mask, status=None):
    """S of plan section 6: the sum of a normalised map outside the leaf mask, computed in float64 (the map is summed
    and renormalised in float64, so float32 round-off of the stored map does not enter). Where status says the map
    was replaced by the uniform map (codes 1 and 2 of normalise), S is set to exactly the area share a, so that a
    replaced map is a tie with the uniform map and never 'below' it. maps (N, 224, 224), mask bool."""
    m = maps.double()
    s = (m * (~mask).double()).flatten(1).sum(1) / m.flatten(1).sum(1)
    if status is not None:
        s = torch.where(status > 0, area_off_leaf(mask), s)
    return s


def off_leaf_share_native(cam, mask):
    """Sensitivity (ii): S on the map's own grid, each cell weighted by the share of its area that is off the leaf.
    cam (N, h, w) non-negative, not upsampled. A map with zero or non-finite mass is replaced by the uniform map on
    its own grid, as in normalise(), so the paired sample never changes."""
    h = cam.shape[-1]
    off = F.adaptive_avg_pool2d((~mask).float()[:, None], (cam.shape[-2], h))[:, 0].double()
    cam = cam.double()
    bad = ~torch.isfinite(cam).flatten(1).all(1)
    cam = torch.where(bad.view(-1, 1, 1), torch.zeros_like(cam), cam).clamp_min(0)
    tot = cam.flatten(1).sum(1)
    replace = bad | ~(tot > 0) | ~torch.isfinite(tot)
    cam = torch.where(replace.view(-1, 1, 1), torch.ones_like(cam), cam)
    return (cam * off).flatten(1).sum(1) / cam.flatten(1).sum(1)
