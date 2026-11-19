"""tae_sample.npz assembly: object channels in the per-frame canonical dialect + the anchor-routing
labels.

Per active object mi:
  robj_<mi>   (T,6)    canonical object rotation, row-convention 6D (first two ROWS)
  pobj_<mi>   (T,3)    canonical object centroid p_G
  delta_<mi>  (T,6,3)  per-part offset: anchor-local vector from part anchor to the object centroid
  w_<mi>      (T,7)    binary multi-hot routing over {6 parts, FREE}
  shape_<mi>  (2048,3) canonical surface samples
  contact_d_<mi> (T,6) per-part min distance body-surface <-> object surface   speed_<mi> (T,) centroid speed
An object is ACTIVE if it ever moves (> V_THR m/frame) or is ever touched (< THR m).
"""
import os

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree

from .subsets import ANCH_NAME, PARTS, heading_frame, part_vertex_index

THR = 0.010          # contact threshold, m
V_THR = 0.001        # movement threshold, m/frame (3 cm/s @ 30 fps)
VWIN = 5             # frames to smooth speed
CWIN = 15            # window for the motion-coupling Delta variance (~0.5 s)
HV_DS = 2; OBJ_DS = 2   # downsample factors for the contact KD-tree
D_NEAR = 0.08; RIG = 0.02; RIG_OBJ = 0.02; OBJ_NEAR = 0.25


def rotmat_to_6d(R):
    return R[..., :2, :].reshape(R.shape[0], 6)


def smooth(x, w):
    if len(x) < 2:
        return x
    return uniform_filter1d(x, size=w, mode="nearest")


def assemble(seq, out_dir, hv, model="smplx", h272=None, verbose=True):
    """out_dir must already hold anchor_fk.npz + object_6dof.npz; hv = (T,V,3) world verts (hands posed).
    Writes out_dir/tae_sample.npz (human_272 only if `h272` (T,272) is given)."""
    T = hv.shape[0]
    af = np.load(f"{out_dir}/anchor_fk.npz", allow_pickle=True)
    om = np.load(f"{out_dir}/object_6dof.npz", allow_pickle=True)
    names = [str(x) for x in om["names"]]
    PIDX = part_vertex_index(model)
    assert max(int(v.max()) for v in PIDX.values()) < hv.shape[1], \
        f"{seq}: model={model} segmentation ids exceed mesh verts {hv.shape[1]}"
    Ranch = {p: af[f"R_{ANCH_NAME[p]}"] for p in PARTS}
    panch = {p: af[f"p_{ANCH_NAME[p]}"] for p in PARTS}

    Hrot, shift = heading_frame(af["R_root_pelvis"].astype(np.float64),
                                af["p_root_pelvis"].astype(np.float64), float(af["floor_y"]))

    def to_canon_p(pw): return np.einsum("tij,tj->ti", Hrot, pw - shift)
    def to_canon_R(Rw): return np.einsum("tij,tjk->tik", Hrot, Rw)
    def centroid(j): return np.einsum("tij,j->ti", om[f"R_{j}"], om[f"canon_{j}"].mean(0)) + om[f"t_{j}"]

    def speed(j):
        c = centroid(j); s = np.r_[0.0, np.linalg.norm(np.diff(c, axis=0), axis=1)]
        return smooth(s, VWIN)

    def contact_d(j):
        d = np.full((T, len(PARTS)), 9.0, np.float32)
        Rj, tj, samp = om[f"R_{j}"], om[f"t_{j}"], om[f"samp_{j}"][::OBJ_DS]
        c = centroid(j); bodyc = hv.mean(1); near = np.linalg.norm(c - bodyc, axis=1) < 1.2
        for t in np.where(near)[0]:
            osw = (Rj[t] @ samp.T).T + tj[t]; tree = cKDTree(osw)
            for pi, p in enumerate(PARTS):
                d[t, pi] = tree.query(hv[t][PIDX[p][::HV_DS]])[0].min()
        return d

    active = []; SP = {}; CD = {}
    for j in range(len(names)):
        sp = speed(j); SP[j] = sp
        moved = sp.max() > V_THR
        cd = contact_d(j); CD[j] = cd; contacted = cd.min() < THR
        if moved or contacted:
            active.append(j)

    out = {"part_names": np.array(PARTS, dtype="U8"),
           "active_names": np.array([names[j] for j in active], dtype="U64"), "T": T,
           "robj_convention": np.array("row")}
    if h272 is not None:
        out["human_272"] = np.asarray(h272).astype(np.float32)

    DEL = {}; ROBJ = {}; POBJ = {}; POBJC = {}; SAMP = {}; Wd = {}; CEN = {}; MOV = {}
    for mi, j in enumerate(active):
        Rj, tj, canon, samp = om[f"R_{j}"], om[f"t_{j}"], om[f"canon_{j}"], om[f"samp_{j}"]
        pobj = np.einsum("tij,j->ti", Rj, canon.mean(0)) + tj                 # WORLD centroid
        delta = np.zeros((T, len(PARTS), 3), np.float32)
        for pi, p in enumerate(PARTS):
            delta[:, pi] = np.einsum("tij,ti->tj", Ranch[p], (pobj - panch[p]))
        pobj_c = to_canon_p(pobj).astype(np.float32)
        robj = rotmat_to_6d(to_canon_R(Rj.astype(np.float64))).astype(np.float32)
        d = CD[j]; sp = SP[j]
        m1 = smooth(delta, CWIN); m2 = smooth(delta ** 2, CWIN)
        std = np.sqrt(np.clip(m2 - m1 ** 2, 0, None).sum(2) / 3.0)
        w = np.zeros((T, len(PARTS) + 1), np.float32)
        contact_f = d.min(1) < THR; moving_f = sp > V_THR
        for t in range(T):
            if contact_f[t]:
                w[t, :len(PARTS)] = (d[t] < THR).astype(np.float32)
            elif moving_f[t]:
                cand = np.where((d[t] < D_NEAR) & (std[t] < RIG))[0]
                if len(cand):
                    w[t, int(cand[(std[t, cand] + d[t, cand]).argmin()])] = 1.0
                else:
                    w[t, -1] = 1.0
            else:
                w[t, -1] = 1.0
        DEL[mi] = delta; ROBJ[mi] = robj; POBJ[mi] = pobj; POBJC[mi] = pobj_c
        SAMP[mi] = samp; Wd[mi] = w; CEN[mi] = centroid(j); MOV[mi] = moving_f

    # object-to-object coupling (a freely moving object rigidly attached to a held one inherits its routing)
    W0 = {mi: Wd[mi].copy() for mi in Wd}
    for mi in Wd:
        w = Wd[mi]; free_move = MOV[mi] & (W0[mi][:, -1] > 0)
        if not free_move.any():
            continue
        for t in np.where(free_move)[0]:
            best = None; bestd = 1e9
            for mj in Wd:
                if mj == mi or W0[mj][t, :len(PARTS)].sum() == 0:
                    continue
                rel = CEN[mi] - CEN[mj]
                a = max(0, t - CWIN // 2); b = min(T, t + CWIN // 2 + 1)
                relstd = np.sqrt(rel[a:b].var(0).sum())
                dist = np.linalg.norm(rel[t])
                if relstd < RIG_OBJ and dist < OBJ_NEAR and dist < bestd:
                    bestd = dist; best = mj
            if best is not None:
                w[t] = W0[best][t].copy()

    max_rec_err = 0.0
    for mi in Wd:
        w = Wd[mi]; delta = DEL[mi]; pobj = POBJ[mi]
        wn = w / np.clip(w.sum(1, keepdims=True), 1e-8, None)
        rec = np.zeros((T, 3))
        for pi, p in enumerate(PARTS):
            rec += wn[:, pi:pi + 1] * (panch[p] + np.einsum("tij,tj->ti", Ranch[p], delta[:, pi]))
        rec += wn[:, -1:] * pobj
        max_rec_err = max(max_rec_err, np.abs(rec - pobj).max())
        out[f"robj_{mi}"] = ROBJ[mi]; out[f"pobj_{mi}"] = POBJC[mi]; out[f"delta_{mi}"] = delta
        out[f"w_{mi}"] = w; out[f"shape_{mi}"] = SAMP[mi]
        out[f"contact_d_{mi}"] = CD[active[mi]].astype(np.float32)
        out[f"speed_{mi}"] = SP[active[mi]].astype(np.float32)
    tmp = f"{out_dir}/tae_sample.npz.tmp{os.getpid()}.npz"
    np.savez(tmp, **out); os.replace(tmp, f"{out_dir}/tae_sample.npz")
    if verbose:
        print(f"{seq}: T={T} active={len(active)}/{len(names)} pose-recovery max err {max_rec_err*1000:.5f} mm")
    return out
