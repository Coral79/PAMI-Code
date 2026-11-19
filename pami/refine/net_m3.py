"""PamiRefiner network: 54-token full-body factorized spatial/temporal transformer.

TOKENS (54): 22 body joints (pelvis = context, no output), 30 fingers, 1 object, 1 scene.
HEADS: body head -> joints 1..21 (Lie-algebra residuals); finger head -> 30 joints; object token ->
tanh-ball-bounded local delta-SE(3) (5 cm / 10 deg per step); scene token -> root [dx, dz, dyaw, dy],
Gaussian low-passed over time, XZ/yaw gauged to zero at t = 0, ball/tanh-capped.
INPUT ENCODING: InputNorm (running per-channel standardization, no affine) -> learnable random
Fourier features (32 freqs) -> Linear; + token id, side and sinusoidal time embeddings.
All output heads are zero-initialized (step 0 == identity). The cumulative caps over the T-step
recursion are enforced by rollout_m3.project_caps.
"""
import math

import torch
import torch.nn as nn

from pami.refine.object_state import bounded_delta_se3
from pami.refine.sensor_m3 import GearsSensor

N_BODY, N_FINGER = 22, 30
N_TOKENS_M3 = N_BODY + N_FINGER + 2                        # 54
MAX_W = 512                                                # temporal PE table length
_SIDE_BODY = {1: 0, 4: 0, 7: 0, 10: 0, 13: 0, 16: 0, 18: 0, 20: 0,
              2: 1, 5: 1, 8: 1, 11: 1, 14: 1, 17: 1, 19: 1, 21: 1}   # 0 left, 1 right, 2 central


def token_sides_m3():
    s = [_SIDE_BODY.get(j, 2) for j in range(22)] + [0] * 15 + [1] * 15 + [2, 2]
    return torch.tensor(s, dtype=torch.long)


def _sinusoid(n, d):
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    i = torch.arange(d, dtype=torch.float32)[None, :]
    ang = pos / torch.pow(10000.0, (2 * (i // 2)) / d)
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(ang[:, 0::2])
    pe[:, 1::2] = torch.cos(ang[:, 1::2])
    return pe


class _InputNorm(nn.Module):
    """Per-channel standardization with running statistics, no affine (the raw features carry large
    constant offsets with small variation on top; the following Linear supplies scale and shift)."""

    def __init__(self, d, momentum=0.01, eps=1e-5):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.register_buffer("mean", torch.zeros(d))
        self.register_buffer("var", torch.ones(d))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    def forward(self, x):
        if self.training:
            f = x.reshape(-1, x.shape[-1])
            m, v = f.mean(0), f.var(0, unbiased=False)
            with torch.no_grad():
                if torch.isfinite(m).all() and torch.isfinite(v).all():   # never poison the buffers
                    self.steps += 1
                    r = max(self.momentum, 1.0 / float(self.steps))        # debiased warm-up
                    self.mean.lerp_(m.detach(), r)
                    self.var.lerp_(v.detach(), r)
        else:
            m, v = self.mean, self.var
        return (x - m) / (v + self.eps).sqrt()


class _Block(nn.Module):
    """One factorized layer: attention across tokens within a frame, then across frames within a
    token (bidirectional in time: an offline post-process over a finished window)."""

    def __init__(self, d, heads, ff, dropout=0.0):
        super().__init__()
        self.sp = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.tm = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x):                                # (B,W,N,d)
        B, W, N, d = x.shape
        h = self.n1(x).reshape(B * W, N, d)
        x = x + self.sp(h, h, h, need_weights=False)[0].reshape(B, W, N, d)
        h = self.n2(x).permute(0, 2, 1, 3).reshape(B * N, W, d)
        x = x + self.tm(h, h, h, need_weights=False)[0].reshape(B, N, W, d).permute(0, 2, 1, 3)
        return x + self.ff(self.n3(x))


class _FourierEnc(nn.Module):
    """out = [x, sin(x@B), cos(x@B)], B (d_in, nf) learnable."""

    def __init__(self, d_in, nf=32):
        super().__init__()
        self.B = nn.Parameter(torch.randn(d_in, nf))
        self.d_out = d_in + 2 * nf

    def forward(self, x):
        p = x @ self.B
        return torch.cat([x, torch.sin(p), torch.cos(p)], -1)


class _GaussSmooth(nn.Module):
    """Fixed Gaussian low-pass over the time axis, reflection-padded."""

    def __init__(self, sigma=3.0, radius=6):
        super().__init__()
        t = torch.arange(-radius, radius + 1, dtype=torch.float32)
        k = torch.exp(-0.5 * (t / sigma) ** 2)
        self.register_buffer("k", (k / k.sum())[None, None])
        self.radius = radius

    def forward(self, x):                                   # (B,W,C)
        B, W, C = x.shape
        h = x.permute(0, 2, 1).reshape(B * C, 1, W)
        h = torch.nn.functional.pad(h, (self.radius, self.radius), mode="reflect")
        return torch.nn.functional.conv1d(h, self.k).reshape(B, C, W).permute(0, 2, 1)


class RefineNetM3(nn.Module):
    def __init__(self, d_body, d_finger, d_obj, d_scene, d=256, layers=4, heads=8, ff=1024,
                 dropout=0.0, nf=32, cap_obj_m=0.05, cap_obj_rad=math.radians(10.0),
                 cap_root_m=0.10, cap_root_rad=math.radians(5.0), cap_root_y=0.10, sensor_kw=None):
        super().__init__()
        self.cap_obj_m, self.cap_obj_rad = cap_obj_m, cap_obj_rad
        self.cap_root_m, self.cap_root_rad, self.cap_root_y = cap_root_m, cap_root_rad, cap_root_y
        self.nm = nn.ModuleDict({k: _InputNorm(dd) for k, dd in
                                 (("body", d_body), ("finger", d_finger), ("obj", d_obj), ("scene", d_scene))})
        self.fe = nn.ModuleDict({k: _FourierEnc(dd, nf) for k, dd in
                                 (("body", d_body), ("finger", d_finger), ("obj", d_obj), ("scene", d_scene))})
        self.inp = nn.ModuleDict({k: nn.Linear(self.fe[k].d_out, d) for k in self.fe})
        self.tok_emb = nn.Embedding(N_TOKENS_M3, d)
        self.side_emb = nn.Embedding(3, d)
        self.register_buffer("time_pe", _sinusoid(MAX_W, d), persistent=False)
        self.blocks = nn.ModuleList([_Block(d, heads, ff, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head_body = nn.Linear(d, 3)                    # per body token 1..21
        self.head_finger = nn.Linear(d, 3)
        self.head_obj = nn.Linear(d, 6)
        self.head_root = nn.Linear(d, 4)                    # [dx, dz, dyaw, dy]
        for h in (self.head_body, self.head_finger, self.head_obj, self.head_root):
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)  # step 0 == identity
        self.smooth = _GaussSmooth()
        # built LAST (and inside its own seeded RNG window) so it never shifts the init of the rest
        self.sensor = GearsSensor(d, **(sensor_kw or {}))

    def forward(self, f_body, f_finger, f_obj, f_scene, sides, sensor_in):
        """-> dtheta_body (B,W,21,3) [joints 1..21], dtheta_hand (B,W,30,3), dxi (B,W,M,6) bounded
        local object residual, droot (B,W,4) [dx, dz, dyaw, dy] smoothed + capped (XZ/yaw gauged to 0
        at t=0; dy is NOT gauged: a constant vertical offset is exactly the error it corrects).
        sensor_in = (pj, Rj, pts, nrm, tip_p, tip_R) from the CURRENT state (rollout_m3)."""
        B, W, M = f_obj.shape[0], f_obj.shape[1], f_obj.shape[2]
        assert M == 1, "single-object refiner"
        enc = lambda k, x: self.inp[k](self.fe[k](self.nm[k](x)))
        x_fing = enc("finger", f_finger) + self.sensor(*sensor_in)
        x = torch.cat([enc("body", f_body), x_fing, enc("obj", f_obj), enc("scene", f_scene)[:, :, None]], dim=2)
        ids = torch.arange(N_TOKENS_M3, device=x.device)
        x = x + self.tok_emb(ids)[None, None] + self.side_emb(sides)[None, None]
        assert W <= MAX_W, f"window {W} exceeds temporal PE table {MAX_W}"
        x = x + self.time_pe[:W][None, :, None]
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        dtheta_body = self.head_body(x[:, :, 1:22])
        dtheta_hand = self.head_finger(x[:, :, 22:52])
        dxi = bounded_delta_se3(self.head_obj(x[:, :, 52:52 + M]), self.cap_obj_m, self.cap_obj_rad)
        raw = self.head_root(x[:, :, -1])                                # (B,W,4)
        sm = self.smooth(raw[..., :3])
        sm = sm - sm[:, :1]                                              # gauge: zero at t=0
        xz = sm[..., :2]
        n = xz.norm(dim=-1, keepdim=True)
        s = torch.where(n < 1e-4, 1.0 - n ** 2 / 3.0, torch.tanh(n) / n.clamp(min=1e-12))
        xz = self.cap_root_m * s * xz
        yaw = self.cap_root_rad * torch.tanh(sm[..., 2:3])
        dy = self.cap_root_y * torch.tanh(self.smooth(raw[..., 3:]))
        return dtheta_body, dtheta_hand, dxi, torch.cat([xz, yaw, dy], -1)
