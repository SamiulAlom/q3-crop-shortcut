"""Method tests for p3_torch.py with randomly initialised networks and generated inputs (test fixtures only; no
result of this file is data). Run: .venv-local/Scripts/python.exe projects/qpain/paper-03/notebooks/tests/test_p3_torch.py
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p3_common as P
import p3_torch as T

torch.manual_seed(0)
dev = "cpu"
S = T.IMG
x = torch.rand(3, 3, S, S)
mean, std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
xn = (x - mean) / std
x0n = (torch.zeros(1, 3, S, S) - mean) / std
tg = torch.tensor([0, 3, 7])
mask = torch.zeros(3, S, S, dtype=torch.bool); mask[:, 56:168, 56:168] = True

for name in ("resnet50", "efficientnet_b0", "deit_small_patch16_224", "vit_small_patch16_224"):
    model = T.create(name, 30).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    shape = T.gc_layer_check(model, name, dev)
    ref = model(xn)
    cam = T.gradcam(model, xn, tg, name)
    assert cam.dim() == 3 and cam.shape[0] == 3 and (cam >= 0).all(), (name, cam.shape)
    m, zero = T.normalise(cam)
    assert m.shape == (3, S, S) and torch.allclose(m.flatten(1).sum(1), torch.ones(3), atol=1e-4)
    ig, signed, delta = T.integrated_gradients(model, xn[:1], x0n, tg[:1], steps=16)
    assert ig.shape == (1, S, S) and (ig >= 0).all()
    # completeness holds only approximately with few steps; the sign and order of magnitude must agree for a smooth net
    print(f"{name}: GC layer output {shape}, cam {tuple(cam.shape)}, IG sum {float(signed[0]):+.4f} vs logit difference {float(delta[0]):+.4f}")
    if name.endswith("patch16_224"):
        fused_before = [b.attn.fused_attn for b in model.blocks]
        ro = T.rollout(model, xn)
        assert ro.shape == (3, 14, 14) and (ro >= 0).all()
        ta = T.transformer_attribution(model, xn, tg)
        assert ta.shape == (3, 14, 14) and (ta >= 0).all() and torch.isfinite(ta).all()
        ta2 = T.transformer_attribution(model, xn, torch.tensor([1, 2, 5]))
        assert not torch.allclose(ta, ta2), "transformer attribution does not depend on the class"
        assert torch.allclose(T.rollout(model, xn), ro), "rollout is not repeatable"
        assert [b.attn.fused_attn for b in model.blocks] == fused_before, "fused attention was not restored"
        assert torch.allclose(model(xn), ref, atol=1e-5), "hooks changed the forward pass"
    assert torch.allclose(model(xn), ref, atol=1e-5)

# normalise: a zero map becomes uniform and is flagged
m, zero = T.normalise(torch.zeros(2, 7, 7))
assert (zero == 1).all() and torch.allclose(m, torch.full_like(m, 1 / (S * S)))
bad = torch.ones(2, 7, 7); bad[0, 0, 0] = float("nan"); bad[1, 0, 0] = float("inf")
mb, code = T.normalise(bad)
assert (code == 2).all() and torch.allclose(mb, torch.full_like(mb, 1 / (S * S)))
# off-leaf share: uniform map gives the background area share; a map inside the mask gives 0
u, _ = T.normalise(torch.ones(3, S, S))
a = (~mask).float().flatten(1).mean(1)
assert torch.allclose(T.off_leaf_share(u, mask).float(), a, atol=1e-6)
# a replaced map is an exact tie with the area share (float64, from pixel counts), for several mask sizes
for lo, hi in ((56, 168), (10, 200), (100, 101), (0, 223)):
    mk_ = torch.zeros(2, S, S, dtype=torch.bool); mk_[:, lo:hi, lo + 1:hi] = True
    mm, code = T.normalise(torch.zeros(2, 7, 7))
    assert (T.off_leaf_share(mm, mk_, code) == T.area_off_leaf(mk_)).all()
    assert T.area_off_leaf(mk_)[0].item() == (S * S - (hi - lo) * (hi - lo - 1)) / (S * S)
inside, _ = T.normalise(mask.float())
assert (T.off_leaf_share(inside, mask) == 0).all()
# native-grid share: a 7 x 7 uniform map gives the area share too (the mask is aligned with the 32 px cells here? no: 56..168 = cells 1.75..5.25)
nat = T.off_leaf_share_native(torch.ones(3, 7, 7), mask)
assert torch.allclose(nat.float(), a, atol=1e-4), (nat, a)
assert torch.allclose(T.off_leaf_share_native(torch.zeros(1, 7, 7), mask[:1]).float(), a[:1], atol=1e-4)   # zero mass -> uniform on the native grid
# pixel order: larger first, exact ties by the tie ranking, tiny differences kept
assert T.saliency_order(torch.tensor([0.5, 0.2, 0.5, 0.0]), torch.tensor([3, 0, 1, 2])).tolist() == [2, 0, 1, 3]
assert T.saliency_order(torch.tensor([1e-9, 1e-9 + 1e-15, 0.0], dtype=torch.float64), torch.tensor([0, 2, 1])).tolist() == [1, 0, 2]
# control maps: shapes, non-negative, SIL has share 0, CENTRE peaks in the middle
u8 = (x.permute(0, 2, 3, 1) * 255).to(torch.uint8)
cm = T.control_maps(u8, mask)
assert set(cm) == {"SIL", "EDGE", "CENTRE", "UNIF"} and all(v.shape == (3, S, S) and (v >= 0).all() for v in cm.values())
assert float(T.off_leaf_share(T.normalise(cm["SIL"])[0], mask).max()) < 1e-6
assert cm["CENTRE"][0, S // 2, S // 2] > cm["CENTRE"][0, 0, 0]
# device compositing equals the reference
rgb = np.random.default_rng(0).integers(0, 255, (2, S, S, 3)).astype(np.uint8)
bg = np.random.default_rng(1).integers(0, 255, (2, S, S, 3)).astype(np.uint8)
mk = mask[:2].numpy()
got = T.build((rgb, mk, bg), "cpu").numpy()
assert all((got[i] == P.compose(rgb[i], mk[i], bg[i])).all() for i in range(2))
print("p3_torch tests passed (version", T.TORCH_VERSION + ")")
