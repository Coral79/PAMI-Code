"""Shared wrist-frame helpers for hand_mode="wlocal".

ONE implementation used by make_stats_hand / HOIWindowDataset / prepare_t2hoi / renderers, so the
wrist-local transform can never diverge across sites (the wpos review flagged the tri-site duplication
of the subtraction as the top structural risk; wlocal adds a rotation, so centralize BOTH).

Frame conventions (must match interact_hand_joints.py + AnchorFK):
- h272 is the RAW/DENORMALIZED 272; positions block [8:74] (joint j at 8+3j) and hand_joints.npz
  positions live in the SAME per-frame canonical frame (de-headed, root-XZ centered, absolute Y).
- rot block [140:272] = 22 parent-relative rot6d with the ROOT'S rotation stored DE-HEADED, so
  accumulating parent chains yields global joint rotations IN the canonical frame (exactly what
  AnchorFK replays; to_world_interact re-applies heading afterwards — we deliberately do not).
- wlocal = Rw^T @ (p - pw): finger joints in each wrist's own rotating frame -> the block carries
  pure articulation (curl/spread); orientation and position ride the body's arm chain instead.
"""
import numpy as np
import torch
from pami.vendor.eval_trans import rotation_6d_to_matrix

# SMPL kinematic chains root -> wrist (parents[:22] = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19])
_CHAIN_L = [0, 3, 6, 9, 13, 16, 18, 20]
_CHAIN_R = [0, 3, 6, 9, 14, 17, 19, 21]


def wrist_frames(h272):
    """(T,272) raw/denorm human -> (Rw (T,2,3,3), pw (T,2,3)): global wrist rotations + positions in
    the per-frame canonical frame. Side 0 = left (joint 20), side 1 = right (joint 21)."""
    h272 = np.asarray(h272)
    T = h272.shape[0]
    rot = rotation_6d_to_matrix(torch.from_numpy(
        np.ascontiguousarray(h272[:, 140:140 + 132], dtype=np.float32)).reshape(T, 22, 6)).numpy()
    Rw = np.zeros((T, 2, 3, 3), np.float32)
    for side, chain in ((0, _CHAIN_L), (1, _CHAIN_R)):
        R = rot[:, chain[0]]
        for j in chain[1:]:
            R = np.matmul(R, rot[:, j])
        Rw[:, side] = R
    pw = np.stack([h272[:, 68:71], h272[:, 71:74]], 1).astype(np.float32)   # wrists = joints 20/21
    return Rw, pw


def to_wlocal(hand, h272):
    """hand (T,30,3) canonical [L15|R15] -> (T,30,3) wrist-local: Rw^T @ (p - pw) per side."""
    Rw, pw = wrist_frames(h272)
    out = np.empty_like(hand, dtype=np.float32)
    for side, sl in ((0, slice(0, 15)), (1, slice(15, 30))):
        out[:, sl] = np.einsum("tji,tkj->tki", Rw[:, side], hand[:, sl] - pw[:, side, None])
    return out


_FINGERS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14)]   # 5x [MCP,PIP,DIP], parent=wrist


def fk_hand_rot6d(rot6d, h272, rest, h0, wrist_idx=(20, 21)):
    """(T,30,6) parent-relative hand rot6d + raw/denorm h272 + rest joints (J,3, gendered zero-pose
    Jtr) -> (T,30,3) hand joint positions in the per-frame canonical frame. The theta-path FK:
    global finger rotation = Rw @ R_MCP @ ..., positions chain rest bone offsets — SMPL's
    rest-aligned local-frame convention (same recursion GEARS/AnchorFK/to_world_interact use).
    Use this to visualize/consume the ROTATION channel of posrot/wlocalrot without touching the
    (noisy) position channel."""
    T = rot6d.shape[0]
    R = rotation_6d_to_matrix(torch.from_numpy(
        np.ascontiguousarray(rot6d, dtype=np.float32))).numpy()            # (T,30,3,3) parent-relative
    Rw, pw = wrist_frames(h272)
    out = np.empty((T, 30, 3), np.float32)
    for side in (0, 1):
        wj = wrist_idx[side]
        rj = rest[h0 + side * 15: h0 + (side + 1) * 15] - rest[wj]         # rest offsets from wrist
        Rws, pws = Rw[:, side], pw[:, side]
        for (m, p_, d_) in _FINGERS:
            o = side * 15
            Rm = np.matmul(Rws, R[:, o + m])
            out[:, o + m] = pws + np.einsum("tij,j->ti", Rws, rj[m])
            out[:, o + p_] = out[:, o + m] + np.einsum("tij,j->ti", Rm, rj[p_] - rj[m])
            Rp = np.matmul(Rm, R[:, o + p_])
            out[:, o + d_] = out[:, o + p_] + np.einsum("tij,j->ti", Rp, rj[d_] - rj[p_])
    return out


def from_wlocal(local, h272):
    """Inverse of to_wlocal: (T,30,3) wrist-local -> canonical, using h272's OWN wrists (GT panel
    uses GT human, PRED panel uses the reconstructed human -> rigid attachment by construction)."""
    Rw, pw = wrist_frames(h272)
    out = np.empty_like(local, dtype=np.float32)
    for side, sl in ((0, slice(0, 15)), (1, slice(15, 30))):
        out[:, sl] = np.einsum("tij,tkj->tki", Rw[:, side], local[:, sl]) + pw[:, side, None]
    return out
