"""Object-channel coordinate-frame conversion, per-frame canonical -> WINDOW-FIRST-FRAME.

Stored tae_sample channels are PER-FRAME canonical (each frame de-headed on its own root). The
window-first dialect re-expresses them relative to the WINDOW's first frame s:
    R_g(t) = H(s) H(t)^T R_c(t)          (1a, rotation)
    p_g(t) = H(s) H(t)^T p_c(t) + H(s)(pelvis(t) - pelvis(s))   (1b, position; XZ trajectory, Y absolute)
H(t) = de-heading yaw at frame t. Both are computed ON THE FLY from the raw (denormalized) human-272
window itself — channels [0:2] root XZ velocity, [2:8] heading frame-to-frame 6D delta — so NO data
regeneration is needed and mirrors/augmentation stay consistent (they transform the human window too).

Key identity (why sequence-absolute heading is never needed): with ghr(t) = D(t)···D(s) the window's
own cumulative heading product (D = decoded [2:8] delta of each row, exactly render_recon.to_world's
`accum`), sequence terms cancel:  H(s) H(t)^T = ghr(s) ghr(t)^T  — pure window-local quantities.
At t = s this is exactly I.

All functions are pure numpy on ONE window; loops are over W=64 — negligible cost in the dataset.
"""
import numpy as np


def _rot6d_to_mat(x):
    """(N,6) row-6D -> (N,3,3), Gram-Schmidt on the two stored ROWS (PyTorch3D convention,
    matches utils/eval_trans.rotation_6d_to_matrix; numpy port so the dataset needs no torch)."""
    a1, a2 = x[:, 0:3], x[:, 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2p = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2p / np.linalg.norm(a2p, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)                     # rows = b1,b2,b3


def _mat_to_rot6d(R):
    """(...,3,3) -> (...,6): first two ROWS (inverse of _rot6d_to_mat for exact rotations)."""
    return np.concatenate([R[..., 0, :], R[..., 1, :]], axis=-1)


def window_heading_transform(human_raw):
    """human_raw (W,272) RAW/denormalized window -> T (W,3,3) with T(t) = H(s)H(t)^T = ghr(s)ghr(t)^T.
    ghr = cumulative product of the window's own [2:8] heading deltas (to_world's accum)."""
    assert human_raw.ndim == 2 and human_raw.shape[1] == 272, f"expected (W,272) raw human, got {human_raw.shape}"
    W = human_raw.shape[0]
    D = _rot6d_to_mat(human_raw[:, 2:8].astype(np.float64))    # per-row heading delta
    ghr = np.empty_like(D)
    ghr[0] = D[0]
    for t in range(1, W):
        ghr[t] = D[t] @ ghr[t - 1]
    return (ghr[0] @ np.transpose(ghr, (0, 2, 1))).astype(np.float32)   # (W,3,3), T[0]=I


def window_root_traj(human_raw):
    """human_raw (W,272) -> root XZ trajectory rel. window start, WORLD-orientation of the window
    frame chain (to_world's v-integration): traj (W,3) with Y=0, traj[0]=0."""
    W = human_raw.shape[0]
    D = _rot6d_to_mat(human_raw[:, 2:8].astype(np.float64))
    ghr = np.empty_like(D)
    ghr[0] = D[0]
    for t in range(1, W):
        ghr[t] = D[t] @ ghr[t - 1]
    inv = np.transpose(ghr, (0, 2, 1))
    v = np.zeros((W, 3), np.float64)
    v[:, 0] = human_raw[:, 0]; v[:, 2] = human_raw[:, 1]
    v[1:] = np.einsum("tij,tj->ti", inv[:-1], v[1:])           # rotate each step into the chain frame
    traj = np.cumsum(v, 0); traj -= traj[0]                    # relative to window start
    traj[:, 1] = 0.0
    return traj.astype(np.float32)


def robj_to_window_first(robj_c, human_raw):
    """1a. robj_c (M,W,6) per-frame-canonical row-6D -> window-first-frame row-6D (M,W,6).
    R_g(t) = T(t) @ R_c(t)."""
    T = window_heading_transform(human_raw)                    # (W,3,3)
    M, W, _ = robj_c.shape
    Rc = _rot6d_to_mat(robj_c.reshape(M * W, 6)).reshape(M, W, 3, 3)
    Rg = np.einsum("wij,mwjk->mwik", T, Rc)
    return _mat_to_rot6d(Rg).astype(np.float32)


def robj_to_per_frame(robj_g, human_raw):
    """Inverse of robj_to_window_first (renderer/round-trip): R_c(t) = T(t)^T @ R_g(t)."""
    T = window_heading_transform(human_raw)
    M, W, _ = robj_g.shape
    Rg = _rot6d_to_mat(robj_g.reshape(M * W, 6)).reshape(M, W, 3, 3)
    Rc = np.einsum("wji,mwjk->mwik", T, Rg)                    # T^T via swapped indices
    return _mat_to_rot6d(Rc).astype(np.float32)


def pobj_to_window_first(pobj_c, human_raw):
    """1b. pobj_c (M,W,3) per-frame canonical p_G -> window-first frame.
    p_g(t) = T(t) p_c(t) + H(s)(pelvis(t)-pelvis(s));   H(s)(pelvis(t)-pelvis(s)) = ghr(0) @ traj(t)
    where traj is the window-chain-frame XZ trajectory (Y stays absolute, as stored)."""
    T = window_heading_transform(human_raw)                    # ghr0 @ ghr(t)^T
    traj = window_root_traj(human_raw)                         # (W,3) chain frame, Y=0
    W = human_raw.shape[0]
    D = _rot6d_to_mat(human_raw[:, 2:8].astype(np.float64))
    ghr0 = D[0].astype(np.float32)                             # chain frame -> window-first frame
    off = np.einsum("ij,wj->wi", ghr0, traj)                   # (W,3)
    return (np.einsum("wij,mwj->mwi", T, pobj_c) + off[None]).astype(np.float32)


def pobj_to_per_frame(pobj_g, human_raw):
    """Inverse of pobj_to_window_first: p_c(t) = T(t)^T (p_g(t) - off(t))."""
    T = window_heading_transform(human_raw)
    traj = window_root_traj(human_raw)
    D = _rot6d_to_mat(human_raw[:, 2:8].astype(np.float64))
    ghr0 = D[0].astype(np.float32)
    off = np.einsum("ij,wj->wi", ghr0, traj)
    return np.einsum("wji,mwj->mwi", T, pobj_g - off[None]).astype(np.float32)


def human_root_to_window_first(human_raw):
    """The refine pipeline's frame-mismatch fix.

    The 272's pelvis channel [8:11] is PER-FRAME CANONICAL -- XZ is identically zero (each frame
    sits in its own root-centred, heading-canonicalized frame; the real trajectory lives in the
    root channels [0:8] as velocities). refine/base_state.py assumed [8:11] was "global placement",
    which is wrong for this 272 variant: the human FK came out pinned at the origin while the
    object had been converted to window-first (data_refine), so every joint->object distance was
    off by however far the person walked inside the window (measured: sub5_suitcase_022, 0.38 m
    walked -> wrist-object distance wrong by 43 cm at t=63). The gap eval's <5cm gate used the same
    wrong distances, silently evicting exactly the corrupted frames -- the bug hid itself.

    Fix = give the HUMAN the same window transform the object already gets (pobj_to_window_first):
        pelvis_wf(t)  = T(t) @ pelvis_c(t) + off(t)         off = ghr0 @ traj
        R_root_wf(t)  = T(t) @ R_root_c(t)                  (root rotation channel [140:146])
    Distances are preserved exactly (same rigid transform on both sides), so on a stationary clip
    this is a no-op and on a walking clip it reproduces the per-frame-canonical relative geometry.

    Returns a COPY of human_raw with channels [8:11] and [140:146] rewritten; everything else
    (root velocity channels included) untouched, so pobj_to_window_first(pobj, h) remains valid on
    either the original or the returned array.
    """
    assert human_raw.ndim == 2 and human_raw.shape[1] == 272, f"expected (W,272), got {human_raw.shape}"
    T = window_heading_transform(human_raw)                    # (W,3,3)
    traj = window_root_traj(human_raw)                         # (W,3)
    D = _rot6d_to_mat(human_raw[:, 2:8].astype(np.float64))
    ghr0 = D[0].astype(np.float32)
    off = np.einsum("ij,wj->wi", ghr0, traj)                   # (W,3)
    out = np.array(human_raw, copy=True)
    out[:, 8:11] = (np.einsum("wij,wj->wi", T, human_raw[:, 8:11]) + off).astype(human_raw.dtype)
    R_c = _rot6d_to_mat(human_raw[:, 140:146].astype(np.float64))
    out[:, 140:146] = _mat_to_rot6d(np.einsum("wij,wjk->wik", T.astype(np.float64), R_c)) \
        .astype(human_raw.dtype)
    return out


def human_root_to_per_frame(human_wf):
    """Exact inverse of human_root_to_window_first: pelvis [8:11] -> T^T (pel - off), root rotation
    [140:146] -> T^T @ R_wf. Needed by RENDER paths (M3 P-1 audit fix item 1): to_world_interact /
    _body_aa apply their own `inv = ghr^T` heading on [140:146], so feeding them a window-first
    root applies the heading TWICE (measured 32-64 deg of root yaw error on turning windows).
    T/off come from channels [0:8], which neither direction touches, so the round trip is exact.
    """
    assert human_wf.ndim == 2 and human_wf.shape[1] == 272, f"expected (W,272), got {human_wf.shape}"
    T = window_heading_transform(human_wf)
    traj = window_root_traj(human_wf)
    D = _rot6d_to_mat(human_wf[:, 2:8].astype(np.float64))
    ghr0 = D[0].astype(np.float32)
    off = np.einsum("ij,wj->wi", ghr0, traj)
    out = np.array(human_wf, copy=True)
    out[:, 8:11] = np.einsum("wji,wj->wi", T, human_wf[:, 8:11] - off).astype(human_wf.dtype)
    R_wf = _rot6d_to_mat(human_wf[:, 140:146].astype(np.float64))
    out[:, 140:146] = _mat_to_rot6d(np.einsum("wji,wjk->wik", T.astype(np.float64), R_wf)) \
        .astype(human_wf.dtype)
    return out
