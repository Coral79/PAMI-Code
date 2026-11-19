"""Human-root representation dialect `wf_abs` (window-absolute root block) + shared rotation helpers.

The standard 272 stores the root as per-frame (vx, vz, heading-delta 6D). PAMI stores the root block [0:8]
WINDOW-ABSOLUTE: (x_wf, z_wf) = offset from the window's first frame, [2:8] = heading H(t) = ghr(0) ghr(t)^T
(identity at t=0), using exactly the integration data/objframe.py uses for the object window-first frame, so
both streams share one integration chain.
  forward  (numpy, dataset side)  root_delta_to_abs : standard 272 -> wf_abs 272
  inverse  (torch, decode side)   root_abs_to_delta : wf_abs 272 -> standard 272, differentiable, fp32 inside
Round-trip convention: the window's first heading delta D(0) / velocity v(0) are unobservable in this dialect
and irrelevant (the window-first transform is invariant to them); row 0 reads back as (0, 0, identity6D).

Objects live in the window-first frame (`is_wf("window_first") == True`): denormalised robj / p_G are
absolutes in that frame and the combine must use nets/objframe_torch.combine_obj_pos_wf.
"""
import contextlib
import numpy as np
import torch
from .objframe import (_rot6d_to_mat, _mat_to_rot6d, window_heading_transform, window_root_traj)

IDENT6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)          # rows b1,b2 of the identity rotation
WF_FRAMES = ("window_first",)


def is_wf(obj_frame):
    """True when the denormalised object channels are window-first absolutes."""
    return obj_frame in WF_FRAMES


@contextlib.contextmanager
def _fp32(t):
    """Disable autocast for the enclosed block (cumulative rotation/differencing chains need fp32)."""
    dt = t.device.type if torch.is_tensor(t) else "cpu"
    if dt not in ("cuda", "cpu"):
        yield; return
    with torch.autocast(device_type=dt, enabled=False):
        yield


def _t_rot6d_to_mat(x):
    """(...,6) row-6D -> (...,3,3): Gram-Schmidt on the two stored ROWS (vendor.eval_trans convention)."""
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = a1 / a1.norm(dim=-1, keepdim=True)
    a2p = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2p / a2p.norm(dim=-1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def _t_mat_to_rot6d(R):
    """(...,3,3) -> (...,6): first two ROWS."""
    return torch.cat([R[..., 0, :], R[..., 1, :]], dim=-1)


def root_delta_to_abs(h):
    """numpy (...,W,272) STANDARD 272 -> root block [0:8] made WINDOW-ABSOLUTE; channels [8:272] untouched."""
    flat = h.reshape(-1, h.shape[-2], h.shape[-1])
    out = flat.copy()
    for i in range(flat.shape[0]):
        w = flat[i]
        T = window_heading_transform(w)                                # (W,3,3), T[0]=I
        traj = window_root_traj(w)                                     # (W,3) chain frame, Y=0
        ghr0 = _rot6d_to_mat(w[:, 2:8].astype(np.float64))[0]
        off = np.einsum("ij,wj->wi", ghr0, traj)                       # (W,3) window-first frame, off[0]=0
        out[i, :, 0] = off[:, 0]
        out[i, :, 1] = off[:, 2]
        out[i, :, 2:8] = _mat_to_rot6d(T)
    return out.reshape(h.shape).astype(h.dtype)


def root_abs_to_delta(h):
    """torch (...,W,272) WINDOW-ABSOLUTE root -> STANDARD 272 (per-frame deltas). Differentiable.
    Dheading(t) = H(t)^T H(t-1) (t=0 -> identity); v(t) = H(t-1)^T (off(t)-off(t-1)) -> (x,z) (t=0 -> 0)."""
    with _fp32(h):
        return _root_abs_to_delta_f32(h.float()).to(h.dtype)


def _root_abs_to_delta_f32(h):
    H = _t_rot6d_to_mat(h[..., 2:8])                                   # (...,W,3,3)
    zero = torch.zeros_like(h[..., 0])
    off = torch.stack([h[..., 0], zero, h[..., 1]], dim=-1)            # (...,W,3) Y=0 by wf definition
    Hprev = H[..., :-1, :, :]
    Hinv_prev = Hprev.transpose(-1, -2)
    D = H[..., 1:, :, :].transpose(-1, -2) @ Hprev                     # (...,W-1,3,3) = H(t)^T H(t-1)
    d6 = _t_mat_to_rot6d(D)
    ident = torch.tensor(IDENT6D, dtype=h.dtype, device=h.device).expand(*h.shape[:-2], 1, 6)
    v = torch.einsum("...ij,...j->...i", Hinv_prev, off[..., 1:, :] - off[..., :-1, :])
    vxz = torch.stack([v[..., 0], v[..., 2]], dim=-1)                  # (...,W-1,2)
    z2 = torch.zeros_like(vxz[..., :1, :])
    return torch.cat([torch.cat([z2, vxz], dim=-2),                    # [0:2] root XZ velocity
                      torch.cat([ident, d6], dim=-2),                  # [2:8] heading delta 6D
                      h[..., 8:]], dim=-1)                             # [8:272] untouched
