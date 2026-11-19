"""Corruption stream: group-sampled perturbation of the DECODED (VAE round-trip) GT state.

The substrate is decode(encode(GT)) (the gt_* twin in every gen npz, already carrying the VAE bias);
the target is always the RAW GT. Groups are mutually exclusive per step:
    clean_roundtrip 0.2 | root 0.1 | torso 0.1 | upper 0.2 | lower 0.1 | object 0.2 | mixed 0.1
mixed = 2-3 distinct non-clean groups stacked. Noise is temporally correlated (OU, tau 0.3 s) plus
per-segment constant biases (seconds-long systematic errors); the root XZ/yaw OU is anchored at zero
on t = 0 (the window gauge), the root height is ONE offset per window, the object additionally
carries per-segment constant offsets. SIGMA below is the calibrated table recorded in every checkpoint.
"""
import math

import numpy as np
import torch

from pami.refine.base_state import so3_exp
from pami.refine.rollout_m3 import JOINT_GROUPS, _yaw_mat
from pami.rotvote import mat_to_rot6d
from pami.vendor.eval_trans import rotation_6d_to_matrix

FPS = 30.0
GROUPS = ("clean_roundtrip", "root", "torso", "upper", "lower", "object", "mixed")
P_GROUP = dict(clean_roundtrip=0.2, root=0.1, torso=0.1, upper=0.2, lower=0.1, object=0.2, mixed=0.1)
SIGMA = dict(
    root_xz_m=0.047,                       # per-axis, anchored OU
    root_yaw_deg=3.0,
    root_y_m=0.04,                         # per-window vertical offset
    torso_deg=(2.0, 5.0),
    lower_hipknee_deg=(3.0, 7.0), lower_ankle_deg=(2.0, 5.0),
    upper_deg=(3.0, 12.0), upper_bias_deg=(1.0, 6.0),
    finger_deg=(4.0, 15.0), finger_bias_deg=(2.0, 8.0),
    obj=dict(sigma_cm=(1.0, 4.0), sigma_deg=(3.0, 12.0), bias_cm=(1.0, 5.0), bias_deg=(3.0, 15.0)),
)


# ---- primitives -----------------------------------------------------------------------------------
def ou_noise(shape, tau_s=0.3, sigma=1.0, fps=FPS, gen=None, device="cpu"):
    """Stationary Ornstein-Uhlenbeck noise over dim 1 (time), marginal std = sigma."""
    a = float(np.exp(-1.0 / (tau_s * fps)))
    eps = torch.randn(shape, generator=gen, device=device)
    out = torch.empty_like(eps)
    out[:, 0] = eps[:, 0]
    for t in range(1, shape[1]):
        out[:, t] = a * out[:, t - 1] + (1 - a ** 2) ** 0.5 * eps[:, t]
    return out * sigma


def ou_anchored(shape, tau_s=0.3, sigma=1.0, gen=None, device="cpu"):
    """OU with x[0] == 0 exactly (subtract-the-start form keeps the temporal correlation)."""
    x = ou_noise(shape, tau_s=tau_s, sigma=sigma, gen=gen, device=device)
    return x - x[:, :1]


def _u(rng, gen, device):
    """Uniform scalar draw in [lo, hi) on the generator's own device."""
    lo, hi = rng
    return float(torch.rand(1, generator=gen, device=device).item()) * (hi - lo) + lo


def _segments(W, gen, device, n=(1, 2), min_len=10, max_len_s=3.0):
    """1-2 random contiguous segments of 10 .. min(W//2, 3 s) frames -- the unit a constant bias acts on."""
    k = int(torch.randint(n[0], n[1] + 1, (1,), generator=gen, device=device))
    segs = []
    cap = min(W // 2, int(max_len_s * FPS))
    for _ in range(k):
        L = int(torch.randint(min_len, max(min_len + 1, cap), (1,), generator=gen, device=device))
        s = int(torch.randint(0, max(1, W - L), (1,), generator=gen, device=device))
        segs.append((s, min(s + L, W)))
    return segs


# ---- per-group corruptions ------------------------------------------------------------------------
def _corrupt_joints(rot6, joints, gen, dev, sigma_deg, bias_deg=None):
    B, W = rot6.shape[:2]
    sig = math.radians(_u(sigma_deg, gen, dev))
    w = ou_noise((B, W, len(joints), 3), sigma=sig, gen=gen, device=dev)
    if bias_deg is not None:
        for b in range(B):
            for (s, e) in _segments(W, gen, dev):
                bias = torch.randn(len(joints), 3, generator=gen, device=dev)
                bias = bias / bias.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                w[b, s:e] += bias * math.radians(_u(bias_deg, gen, dev))
    out = rot6.clone()
    R = rotation_6d_to_matrix(rot6[:, :, joints])
    out[:, :, joints] = mat_to_rot6d(R @ so3_exp(w))
    return out


def _corrupt_root(pel, rot6, gen, dev):
    B, W = pel.shape[:2]
    dxz = ou_anchored((B, W, 2), sigma=SIGMA["root_xz_m"], gen=gen, device=dev)
    dyaw = ou_anchored((B, W, 1), sigma=math.radians(SIGMA["root_yaw_deg"]), gen=gen, device=dev)
    dy = torch.randn(B, 1, 1, generator=gen, device=dev) * SIGMA["root_y_m"]
    pel_c = pel + torch.stack([dxz[..., 0], dy.expand(B, W, 1)[..., 0], dxz[..., 1]], -1)
    rot6_c = rot6.clone()
    rot6_c[:, :, 0] = mat_to_rot6d(_yaw_mat(dyaw[..., 0]) @ rotation_6d_to_matrix(rot6[:, :, 0]))
    return pel_c, rot6_c


def _corrupt_object(p_obj, R_obj, gen, dev):
    """OU wobble + per-segment constant offsets on the object pose (object-major (B,M,W,...)).

    RNG-STREAM NOTE: the original implementation (corruption.corrupt_object with p_shift = p_drift = 0)
    drew two uniforms per sample for the retiming / drift families and never used them. Those two
    draws are kept so the corruption sequence for a given generator state is bit-identical."""
    S = SIGMA["obj"]
    B, M, W = p_obj.shape[:3]
    sp = _u(S["sigma_cm"], gen, dev) / 100.0
    sr = np.radians(_u(S["sigma_deg"], gen, dev))
    dp = ou_noise((B * M, W, 3), sigma=sp, gen=gen, device=dev).reshape(B, M, W, 3)
    dw = ou_noise((B * M, W, 3), sigma=sr, gen=gen, device=dev).reshape(B, M, W, 3)
    for b in range(B):
        for (s, e) in _segments(W, gen, dev):
            u = torch.randn(M, 1, 3, generator=gen, device=dev)
            u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            dp[b, :, s:e] += u * (_u(S["bias_cm"], gen, dev) / 100.0)
            u = torch.randn(M, 1, 3, generator=gen, device=dev)
            u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            dw[b, :, s:e] += u * np.radians(_u(S["bias_deg"], gen, dev))
    for b in range(B):
        torch.rand(1, generator=gen, device=dev)                       # retiming draw (p_shift = 0), kept
        torch.rand(1, generator=gen, device=dev)                       # drift draw (p_drift = 0), kept
    p_c = p_obj + torch.einsum("bmwij,bmwj->bmwi", R_obj, dp)
    return p_c, R_obj @ so3_exp(dw)


def sample_group(gen, dev):
    p = torch.tensor([P_GROUP[g] for g in GROUPS], device=dev)
    return GROUPS[int(torch.multinomial(p / p.sum(), 1, generator=gen))]


def _pick_mixed(gen, dev):
    pool = [g for g in GROUPS if g not in ("clean_roundtrip", "mixed")]
    k = 2 + int(torch.randint(0, 2, (1,), generator=gen, device=dev))
    perm = torch.randperm(len(pool), generator=gen, device=dev)[:k]
    return [pool[int(i)] for i in perm]


def corrupt_m3(rot6, rot6h, pel, p_obj, R_obj, gen, dev, group=None):
    """Decoded state -> corrupted state (same shapes) + the group label."""
    if group is None:
        group = sample_group(gen, dev)
    apply = [group] if group != "mixed" else _pick_mixed(gen, dev)
    for g in apply:
        if g == "clean_roundtrip":
            continue
        elif g == "root":
            pel, rot6 = _corrupt_root(pel, rot6, gen, dev)
        elif g == "torso":
            rot6 = _corrupt_joints(rot6, JOINT_GROUPS["torso"], gen, dev, SIGMA["torso_deg"])
        elif g == "lower":
            rot6 = _corrupt_joints(rot6, [1, 2, 4, 5], gen, dev, SIGMA["lower_hipknee_deg"])
            rot6 = _corrupt_joints(rot6, [7, 8, 10, 11], gen, dev, SIGMA["lower_ankle_deg"])
        elif g == "upper":
            rot6 = _corrupt_joints(rot6, JOINT_GROUPS["upper"], gen, dev, SIGMA["upper_deg"], SIGMA["upper_bias_deg"])
            rot6h = _corrupt_joints(rot6h, list(range(30)), gen, dev, SIGMA["finger_deg"], SIGMA["finger_bias_deg"])
        elif g == "object":
            p_obj, R_obj = _corrupt_object(p_obj, R_obj, gen, dev)
        else:
            raise ValueError(f"unknown corruption group {g!r}")
    return rot6, rot6h, pel, p_obj, R_obj, group
