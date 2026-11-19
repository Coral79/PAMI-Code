"""DIFFERENTIABLE torch window-first transform + combine.

Torch twin of pami/data/objframe.py (numpy, dataset-side): given a DENORMALIZED human-272
window, integrate the window's own heading/velocity channels into
    T(t)   = H(s)H(t)^T = ghr(s)·ghr(t)^T          (W,3,3), T(s)=I
    off(t) = H(s)(pelvis(t)-pelvis(s)) = ghr(s)·traj_chain(t)   (W,3), Y=0
so the FK combine's body-part terms (natively per-frame canonical) can be moved into the
window-first frame where the (window_first-dialect) p_G and GT positions live:
    p_obj_wf = Σ_p w_p·( T·(anchor_p + R·Δ_p) + off ) + w_FREE·p_g

Gradients FLOW through T/off into the human branch when fed the recon human (plan R1 locked
decision: combine transform from RECON, NO detach; grad-clip 1.0 is the safety net).
The W-step cumulative matrix product is a python loop over W=64 — cheap vs the 262.7M forward.
"""
import sys, torch
from .. import config as C
from pami.vendor.eval_trans import rotation_6d_to_matrix


def window_transform(h272_denorm):
    """h272_denorm (B,W,272) RAW/denormalized -> (T (B,W,3,3), off (B,W,3)), differentiable."""
    B, W, _ = h272_denorm.shape
    D = rotation_6d_to_matrix(h272_denorm[:, :, 2:8])                  # (B,W,3,3) heading deltas
    ghr = [D[:, 0]]
    for t in range(1, W):
        ghr.append(torch.bmm(D[:, t], ghr[-1]))
    ghr = torch.stack(ghr, 1)                                          # (B,W,3,3)
    T = torch.einsum("bij,bwkj->bwik", ghr[:, 0], ghr)                 # ghr0 @ ghr(t)^T

    v = torch.zeros(B, W, 3, dtype=h272_denorm.dtype, device=h272_denorm.device)
    v = torch.cat([h272_denorm[:, :, 0:1], v[:, :, 1:2], h272_denorm[:, :, 1:2]], -1)  # (vx,0,vz)
    inv = ghr.transpose(-1, -2)                                        # (B,W,3,3)
    v_rot = torch.cat([v[:, :1],                                       # first step dropped by the -traj0 below anyway
                       torch.einsum("bwij,bwj->bwi", inv[:, :-1], v[:, 1:])], 1)
    traj = torch.cumsum(v_rot, 1)
    traj = traj - traj[:, :1]                                          # relative to window start; also kills v[0]
    # Y-zeroing = part of the wf frame DEFINITION (numpy twin objframe.py: traj[:,1]=0). On GT input
    # Y is ~0 anyway (exact yaw heading), but a RECON human's heading 6D is a general rotation and
    # leaks up to ~0.2m into Y early in training. Non-inplace for autograd.
    traj = traj * torch.tensor([1.0, 0.0, 1.0], dtype=traj.dtype, device=traj.device)
    off = torch.einsum("bij,bwj->bwi", ghr[:, 0], traj)                # chain frame -> window-first frame
    return T, off


def combine_obj_pos_wf(delta, pobj_wf, w_logits, anchor_p, anchor_R, T, off):
    """FK combine with body-part terms transformed into the WINDOW-FIRST frame (R1-1b).
    delta (B,M,W,6,3); pobj_wf (B,M,W,3) window-first p_G; w_logits (B,M,W,7);
    anchor_p (B,W,6,3) / anchor_R (B,W,6,3,3) per-frame canonical; T/off from window_transform.
    Returns combined position in the window-first frame (B,M,W,3)."""
    w = w_logits.softmax(-1)
    fk = anchor_p[:, None] + torch.einsum("bwpij,bmwpj->bmwpi", anchor_R, delta)   # per-frame canonical
    fk = torch.einsum("bwij,bmwpj->bmwpi", T, fk) + off[:, None, :, None]          # -> window-first
    p_parts = (w[..., :6, None] * fk).sum(-2)
    p_free = w[..., 6:7] * pobj_wf
    return p_parts + p_free
