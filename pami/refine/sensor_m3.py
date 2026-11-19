"""GEARS-style finger sensor: per-finger-joint local surface-patch encoder (arm "b2", the shipped one).

For each of the 30 finger joints (+ 10 fingertips), all object surface points inside `radius`
(4 cm, world metres) are found; a UNIFORM-IN-BALL subset of K = 64 is selected by a fixed per-point
priority table (density-independent, stable across steps/epochs/processes); each point is encoded
in the joint's local frame as [unit dir (3), d (1), exp(-5d) (1), normal (3)] ("probe" encoding);
a shared pointwise Conv1d stack + masked MEAN pooling per joint + latent conv yields one feature per
joint, which a zero-initialized bias-free Linear ADDS to that finger token's embedding. A joint with
an empty ball contributes exactly zero. Fingertip features go through a separate zero-init Linear
into their finger's DIP token.

Ball-query indices are discrete (no_grad); everything gathered with them stays live, so gradients
reach the joint position/rotation and the object pose through the sensor. Ported from the GEARS
reference `JointFeatExtractor` (has_point gate applied AFTER latent_conv on purpose).
"""
import torch
import torch.nn as nn

N_FINGER = 30
DEFAULT_RADIUS = 0.04            # metres, world scale (coverage ablation's choice)
DEFAULT_K = 64
_UNIFORM_KEY_CACHE = {}


def _uniform_key(P, device, dtype):
    """(P,) fixed random priorities in [0,1), one per POINT INDEX, drawn from a device-local generator
    seeded by P alone and cached per (P, device, dtype). A ball keeps its K smallest-key members, so
    the sensed subset depends only on which points are in the ball -- not on batch size or order.
    NOTE the table is device-specific (CPU and CUDA generators differ): the shipped checkpoint was
    trained and exported on CUDA."""
    k = (int(P), str(device), str(dtype))
    v = _UNIFORM_KEY_CACHE.get(k)
    if v is None:
        g = torch.Generator(device=device).manual_seed(0x5EED * int(P) + 17)
        v = torch.rand(int(P), generator=g, device=device, dtype=dtype)
        _UNIFORM_KEY_CACHE[k] = v
    return v


def ball_query_idx(joints, pts, K, radius):
    """joints (N,J,3), pts (N,P,3) -> idx (N,J,K) long, empty slots = -1.
    Uniform-in-ball selection over the CLOSED ball d <= radius (see _uniform_key). Discrete, no grad."""
    with torch.no_grad():
        jf, pf = joints.float(), pts.float()
        d = torch.cdist(jf, pf)                                   # (N,J,P)
        key = _uniform_key(pf.shape[1], d.device, d.dtype)
        score = key.expand_as(d).clone()
        score.masked_fill_(d > radius, float("inf"))
        k_ = min(K, pf.shape[1])
        kv, ki = torch.topk(score, k_, dim=-1, largest=False)
        ki = torch.where(torch.isfinite(kv), ki.long(), torch.full_like(ki, -1, dtype=torch.long))
        if k_ < K:
            ki = torch.cat([ki, torch.full((*ki.shape[:2], K - k_), -1, dtype=ki.dtype, device=ki.device)], -1)
        return ki


def encode_points(p_loc, n_loc, valid):
    """p_loc, n_loc (N,J,K,3) joint-local point / normal, valid (N,J,K) bool -> (N,J,K,8)
    [unit_dir (3), d (1), exp(-5d) (1), normal (3)]; empty slots contribute zero (no NaN: d is
    clamped before the division and the result is masked)."""
    m = valid[..., None].to(p_loc.dtype)
    d = p_loc.norm(dim=-1, keepdim=True)
    unit = p_loc / d.clamp(min=1e-8)
    return torch.cat([unit, d, torch.exp(-5.0 * d), n_loc], -1) * m


class _ChannelNorm(nn.Module):
    """GEARS' CustomLayerNorm(dim=1): normalize over the channel axis of an (N,C,L) tensor, no params."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return (x - x.mean(1, keepdim=True)) / torch.sqrt(x.var(1, keepdim=True) + self.eps)


class JointFeatExtractor(nn.Module):
    """Shared pointwise Conv1d stack -> masked MEAN inside each joint's own ball -> latent_conv,
    gated AFTER latent_conv so an empty ball gives exactly zero."""

    def __init__(self, width=64, n_conv=2, d_feat=64):
        super().__init__()
        layers, c_in = [], width + 1                                # +1: the mask input channel
        for _ in range(n_conv):
            layers += [nn.Conv1d(c_in, width, 1), _ChannelNorm(), nn.ReLU()]
            c_in = width
        self.conv_layers = nn.Sequential(*layers)
        self.latent_conv = nn.Conv1d(width, d_feat, 1)
        self.d_feat = d_feat

    def forward(self, x, mask):
        """x (N,J,K,C) projected points, mask (N,J,K) bool -> (N,J,d_feat)."""
        N, J, K, C = x.shape
        h = x.reshape(N, J * K, C).permute(0, 2, 1)
        m = mask.reshape(N, 1, J * K).to(h.dtype)
        h = self.conv_layers(torch.cat([h, m], 1))
        h = h.reshape(h.shape[0], h.shape[1], J, K)
        mj = mask[:, None].to(h.dtype)
        pooled = (h * mj).sum(-1) / mj.sum(-1).clamp(min=1.0)     # masked MEAN per joint
        f = self.latent_conv(pooled)
        has_point = mask.any(-1)[:, None].to(f.dtype)
        return (f * has_point).permute(0, 2, 1).contiguous()


class GearsSensor(nn.Module):
    """Emits an ADDITIVE (B,W,30,d_model) residual for the finger tokens (zero at init).

    The shared modules are built inside a saved/restored global-RNG window seeded with `init_seed`,
    so attaching the sensor does not advance the RNG stream of the rest of the model."""
    arm, select, cube, cube_cap = "b2", "uniform", 0.0, 4000       # the shipped configuration (recorded in ckpts)

    def __init__(self, d_model, K=DEFAULT_K, radius=DEFAULT_RADIUS, width=64, n_conv=2, d_feat=64,
                 init_seed=20260815, tips=True):
        super().__init__()
        self.K, self.radius, self.d_feat, self.tips = int(K), float(radius), d_feat, bool(tips)
        state = torch.random.get_rng_state()
        try:
            torch.manual_seed(init_seed)
            self.pt_in = nn.Linear(8, width)
            self.extractor = JointFeatExtractor(width, n_conv, d_feat)
            self.out = nn.Linear(d_feat, d_model, bias=False)       # bias=False: empty ball -> exactly zero
            nn.init.zeros_(self.out.weight)
            if self.tips:
                self.out_tip = nn.Linear(d_feat, d_model, bias=False)
                nn.init.zeros_(self.out_tip.weight)
        finally:
            torch.random.set_rng_state(state)

    def gather_local(self, pj, Rj, pts, nrm):
        """pj (B,W,J,3), Rj (B,W,J,3,3), pts/nrm (B,W,P,3) -> p_loc, n_loc (B*W,J,K,3), valid (B*W,J,K).
        x_local = R^T (p - p_j); n_local = R^T n. Only the integer indices are detached."""
        B, W, J = pj.shape[:3]
        N = B * W
        j, p, n = pj.reshape(N, J, 3), pts.reshape(N, -1, 3), nrm.reshape(N, -1, 3)
        idx = ball_query_idx(j, p, self.K, self.radius)
        valid = idx >= 0
        flat = idx.clamp(min=0).reshape(N, J * self.K)               # clamp BEFORE gather
        pw = torch.gather(p, 1, flat[..., None].expand(-1, -1, 3)).reshape(N, J, self.K, 3)
        nw = torch.gather(n, 1, flat[..., None].expand(-1, -1, 3)).reshape(N, J, self.K, 3)
        m = valid[..., None].to(pw.dtype)
        R = Rj.reshape(N, J, 3, 3)
        p_loc = torch.einsum("njab,njka->njkb", R, (pw - j[:, :, None]) * m)
        n_loc = torch.einsum("njab,njka->njkb", R, nw * m)
        return p_loc * m, n_loc * m, valid

    def features(self, pj, Rj, pts, nrm):
        B, W, J = pj.shape[:3]
        p_loc, n_loc, valid = self.gather_local(pj, Rj, pts, nrm)
        f = self.extractor(self.pt_in(encode_points(p_loc, n_loc, valid)), valid)
        return f.reshape(B, W, J, self.d_feat)

    def forward(self, pj, Rj, pts, nrm, tip_p=None, tip_R=None):
        """-> (B,W,30,d_model) residual. tip_p (B,W,10,3) / tip_R (B,W,10,3,3) from hand_state.tip_centers,
        ordered [L5|R5]; their features land on the DIP tokens [2,5,8,11,14] + 15."""
        r = self.out(self.features(pj, Rj, pts, nrm))
        if not self.tips:
            assert tip_p is None, "tip centers passed to a sensor built with tips=False"
            return r
        assert tip_p is not None, "sensor built with tips=True but no tip centers were passed"
        f_tip = self.features(tip_p, tip_R, pts, nrm)
        dip = [s_ * 15 + d for s_ in (0, 1) for d in (2, 5, 8, 11, 14)]
        r[:, :, dip] = r[:, :, dip] + self.out_tip(f_tip)
        return r


def sensor_cfg_from_ckpt(ck):
    """ck["sensor"] -> (sensor_kw for GearsSensor, sensor cloud length). Refuses configurations the
    cleaned code no longer implements (arm b1, nearest/p3d_first selection, wrist cube, point subsets)."""
    s = ck.get("sensor") or {}
    if s.get("arm", "none") in (None, "none"):
        raise SystemExit("checkpoint has no finger sensor (arm A2/B0); this code implements the b2 sensor only")
    bad = {k: s.get(k) for k, want in (("arm", "b2"), ("select", "uniform"), ("cube", 0.0), ("points", 0))
           if s.get(k, want) != want}
    if bad:
        raise SystemExit(f"checkpoint sensor config {bad} is an ablation arm the cleaned code does not carry "
                         f"(shipped: arm b2, select uniform, cube 0, points 0)")
    kw = dict(K=int(s["K"]), radius=float(s["radius"]), init_seed=int(s.get("seed", 20260815)),
              tips=bool(s.get("tips", False)))
    return kw, int(s.get("surf", (ck.get("args") or {}).get("surf", 512)))
