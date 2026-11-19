"""Loss terms of the refiner.

CORRUPTION STREAM (answer known -- raw GT), per rollout step, absolute-state supervision:
    lam_v |V - V*|^2 + lam_n |Vt - Vt*|^2 (subsampled body vertices, Vt = per-frame mean-centred)
    + rotation geodesic (root 5.0 / body 2.0 / hand 0.5) + cap-normalized object SE(3) (clamped at
    (3 cap)^2) + cap-normalized root xz and root y regression.
REAL STREAM (no answer -- physical), every mask FROZEN from the bank: attraction of intent&reachable
parts to the per-object contact band, penetration hinge (trusted normals only), cumulative slip on
the frozen anchor correspondences, object smoothness, per-part / per-frame preservation vs the BASE
state, foot ground terms (skate, penetration, horizontal pin, vertical band deadzone + shifted pin,
in-run drift, per-side leg preservation + trust region), object pose / drift preservation, root
preservation, rotation temporal smoothness. The vertical targets are shifted by `lift` = the rigid
vertical correction the frozen planted-foot evidence licenses (0 when the window is fine).
"""
import math
import os

import numpy as np
import torch

from pami.refine.base_state import so3_log
from pami.refine.object_state import nearest_surface, pose_surface
from pami.vendor.eval_trans import rotation_6d_to_matrix

# part -> body joints / finger token rows (52-joint convention, same as build_real_bank)
PART_JOINTS = {0: [20], 1: [21], 2: [7, 10], 3: [8, 11], 4: [0, 1, 2, 3, 6, 9, 12, 15]}
PART_FINGERS = {0: list(range(22, 37)), 1: list(range(37, 52))}
BAND_HI_DEFAULT = {0: 0.021, 1: 0.023, 2: 0.070, 3: 0.068, 4: 0.064}   # ALL-row fallback (GT p75 per part)
# planted foot-JOINT height band (metres above the floor): GT p10 .. GT p99 of the shared planted predicate
FOOT_BAND_Y = (0.0022, 0.0765)
W_ROT = dict(root=5.0, body=2.0, hand=0.5)
_BAND_NAMES = ("L_hand", "R_hand", "L_foot", "R_foot", "torso")
BAND_MIN_N = 50                        # a per-object band row needs this many frames behind it, else ALL row
CONTACT_BANDS_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contact_bands_v2.npz")


def load_band_table(path=CONTACT_BANDS_NPZ, min_n=BAND_MIN_N):
    """(table, all_row): table maps (bucket, part) -> band_hi metres ('obj:<name>' / subset / 'ALL'
    buckets, measured on real bank contacts); all_row is the per-part ALL fallback."""
    z = np.load(path, allow_pickle=True)
    table, all_row = {}, dict(BAND_HI_DEFAULT)
    for k in z.files:
        if k == "meta" or k.startswith("prox|"):
            continue
        bucket, _, pname = k.rpartition("/")
        if pname not in _BAND_NAMES or not np.isfinite(z[k][2]):
            continue
        p = _BAND_NAMES.index(pname)
        if bucket.startswith("obj:") and len(z[k]) > 3 and float(z[k][3]) < min_n:
            continue
        table[(bucket, p)] = float(z[k][2])
        if bucket == "ALL":
            all_row[p] = float(z[k][2])
    return table, all_row


def bands_for(obj_names, table, all_row, device, dtype=torch.float32):
    """(B,5) per-sample band-hi tensor, routed per object name with the ALL row as fallback."""
    rows = [[table.get((f"obj:{n}", p), all_row[p]) for p in range(5)] for n in obj_names]
    return torch.tensor(rows, device=device, dtype=dtype)


def geodesic_sq(R_a, R_b):
    return so3_log(R_a.transpose(-1, -2) @ R_b).pow(2).sum(-1)


def _vertex(V_pred, V_gt, vw=None):
    """(L_V, L_Vt). vw (B,SUB) per-sample per-vertex weight or None. The Vt centering stays over ALL
    vertices; only the error average is weighted."""
    e = (V_pred - V_gt).pow(2).sum(-1)
    et = ((V_pred - V_pred.mean(2, keepdim=True)) - (V_gt - V_gt.mean(2, keepdim=True))).pow(2).sum(-1)
    if vw is None:
        return e.mean(), et.mean()
    w = vw[:, None, :].expand_as(e)
    den = w.sum().clamp(min=1.0)
    return (e * w).sum() / den, (et * w).sum() / den


def corruption_losses(snap, gt, net, lam_v=7.0, lam_n=5.0, lam_rot=1.0, lam_obj=1.0,
                      lam_root=1.0, lam_root_y=1.0, V_pred=None, V_gt=None, hand_w=None, vert_hand=None):
    """One rollout step's absolute-state supervision vs raw GT.
    snap / gt: dicts rot6 (B,W,22,6), rot6h (B,W,30,6), pel (B,W,3), p_obj (B,M,W,3), R_obj (B,M,W,3,3)
    V_pred / V_gt (B,W,SUB,3) vertices (None skips the vertex terms)
    hand_w (B,) finger-supervision trust (0/1) or None; vert_hand (SUB,) bool hand-vertex mask."""
    out = {}
    R_p, R_g = rotation_6d_to_matrix(snap["rot6"]), rotation_6d_to_matrix(gt["rot6"])
    out["rot_root"] = geodesic_sq(R_p[:, :, 0], R_g[:, :, 0]).mean()
    out["rot_body"] = geodesic_sq(R_p[:, :, 1:], R_g[:, :, 1:]).mean()
    g_h = geodesic_sq(rotation_6d_to_matrix(snap["rot6h"]), rotation_6d_to_matrix(gt["rot6h"])).mean((1, 2))
    if hand_w is None:
        out["rot_hand"] = g_h.mean()
    else:
        w = hand_w.to(g_h.dtype)
        out["rot_hand"] = (g_h * w).sum() / w.sum().clamp(min=1e-6) if w.sum() > 0 else g_h.sum() * 0.0
    out["obj"] = ((((snap["p_obj"] - gt["p_obj"]).norm(dim=-1) / net.cap_obj_m).pow(2).clamp(max=9.0)).mean()
                  + ((geodesic_sq(snap["R_obj"], gt["R_obj"]) / net.cap_obj_rad ** 2).clamp(max=9.0)).mean())
    out["root_pos"] = ((snap["pel"][..., [0, 2]] - gt["pel"][..., [0, 2]]).pow(2).sum(-1)
                       / net.cap_root_m ** 2).clamp(max=9.0).mean()
    out["root_y"] = ((snap["pel"][..., 1] - gt["pel"][..., 1]).pow(2) / net.cap_root_y ** 2).clamp(max=9.0).mean()
    if V_pred is not None:
        vw = None
        if hand_w is not None and vert_hand is not None:
            vw = 1.0 - (1.0 - hand_w.to(V_pred.dtype))[:, None] * vert_hand.to(V_pred.dtype)[None]
        out["V"], out["Vt"] = _vertex(V_pred, V_gt, vw)
    total = (lam_v * out.get("V", 0.0) + lam_n * out.get("Vt", 0.0)
             + lam_rot * (W_ROT["root"] * out["rot_root"] + W_ROT["body"] * out["rot_body"]
                          + W_ROT["hand"] * out["rot_hand"])
             + lam_obj * out["obj"] + lam_root * out["root_pos"] + lam_root_y * out["root_y"])
    return total, {k: float(v) for k, v in out.items()}


def part_distance(d_all):
    """d_all (B,W,52) joint->surface distances -> (B,W,5) per-part min distance."""
    return torch.stack([d_all[:, :, PART_JOINTS[p] + PART_FINGERS.get(p, [])].amin(-1) for p in range(5)], -1)


def foot_band_lift(pg_base, fint, band=FOOT_BAND_Y, cap=0.10):
    """The frozen vertical correction the planted-foot evidence licenses, (B,1,1) metres, detached:
    lift the deepest planted foot joint back into the band (clamped at the root head's cap)."""
    y = pg_base[:, :, [10, 11], 1]
    sink = (torch.relu(band[0] - y) * fint).amax((1, 2))
    flt = (torch.relu(y - band[1]) * fint).amax((1, 2))
    return (sink - flt).clamp(-cap, cap).detach()[:, None, None]


def _pres_sq(cur, base, lift):
    """Squared preservation vs the base position with its HEIGHT shifted by `lift` (XZ never shifted)."""
    d = cur - base
    return d[..., [0, 2]].pow(2).sum(-1) + (d[..., 1] - lift).pow(2)


def real_losses(snap, base, masks, anchors, pg, pj, d_all, v_all, n_all, bands, pg_base, pj_base, lift,
                lam_attr=1.0, lam_pen=1.0, lam_slip=0.5, lam_smooth=0.1, lam_pres=1.0, lam_foot=0.5,
                near_m=0.08, lam_footpin=5.0, lam_objpres=1.0, lam_rootpres=1.0, lam_tsmooth=0.5,
                lam_legtrust=1.0, cap_m=0.05, cap_rad=math.radians(10.0),
                cap_root_m=0.10, cap_root_rad=math.radians(5.0), cap_root_y=0.10):
    """Self-supervised physical terms for the real generation stream. All masks FROZEN.
    snap: current snapshot | base: the ORIGINAL coarse state | masks: contact_intent / reachable_mask
    (B,W,5), foot_contact_intent (B,W,2), normals_trusted (B,) | anchors: per sample (S,9) rows
    [part,t0,t1,af,jid, pt_loc(3), err] | pg (B,W,22,3), pj (B,W,30,3), d_all/v_all/n_all from the
    CURRENT probe | bands (B,5) | pg_base/pj_base base FK | lift (B,1,1)."""
    dev = pg.device
    out = {}
    zero = torch.zeros((), device=dev)
    base_po, base_Ro = base["p_obj"][:, 0], base["R_obj"][:, 0]
    R_p, R_b = rotation_6d_to_matrix(snap["rot6"]), rotation_6d_to_matrix(base["rot6"])
    allj = torch.cat([pg, pj], 2)
    dpart = part_distance(d_all)                                       # (B,W,5)
    attr_m = (masks["contact_intent"] & masks["reachable_mask"]).to(pg.dtype)
    band = bands.to(device=dev, dtype=pg.dtype)[:, None, :]            # (B,1,5)
    out["attr"] = (torch.relu(dpart - band).pow(2) * attr_m).sum() / attr_m.sum().clamp(min=1.0)
    # penetration: inside iff v . n > 0 (gated, detached); depth gradient flows through d_all
    inside = ((v_all * n_all).sum(-1) > 0).to(d_all.dtype).detach()
    tw = masks["normals_trusted"].to(d_all.dtype)[:, None, None]
    out["pen"] = (torch.relu(d_all * inside - 0.005) * tw).mean()
    # slip: object-local drift of the contact joint since the anchor frame + instantaneous slip
    Ro, po = snap["R_obj"][:, 0], snap["p_obj"][:, 0]
    slips = []
    for b, rows in enumerate(anchors):
        for r in rows:
            t0s, t1s, af, jid, err = int(r[1]), int(r[2]), int(r[3]), int(r[4]), float(r[8])
            if err < 0 or err > near_m:
                continue
            pt = torch.as_tensor(r[5:8], device=dev, dtype=pg.dtype)
            hj = allj[b, t0s:t1s, jid]
            h_loc = torch.einsum("wji,wj->wi", Ro[b, t0s:t1s], hj - po[b, t0s:t1s])
            rel = h_loc - pt
            a_rel = rel[af - t0s].detach()
            slips.append((rel - a_rel).pow(2).sum(-1).mean() + (rel[1:] - rel[:-1]).pow(2).sum(-1).mean())
    out["slip"] = torch.stack(slips).mean() if slips else torch.zeros((), device=dev)
    # object smoothness: translation accel + jerk, angular accel
    v = po[:, 1:] - po[:, :-1]
    a_ = v[:, 1:] - v[:, :-1]
    j_ = a_[:, 1:] - a_[:, :-1]
    wv = so3_log(Ro[:, :-1].transpose(-1, -2) @ Ro[:, 1:])
    wa = wv[:, 1:] - wv[:, :-1]
    out["smooth"] = a_.pow(2).sum(-1).mean() + j_.pow(2).sum(-1).mean() + wa.pow(2).sum(-1).mean()
    # window-level preservation: parts with NO intent in the whole window keep their base rotations
    R_ph, R_bh = rotation_6d_to_matrix(snap["rot6h"]), rotation_6d_to_matrix(base["rot6h"])
    pres = []
    no_int = ~masks["contact_intent"].any(1)                           # (B,5)
    for p in range(5):
        m = no_int[:, p].to(pg.dtype)
        if m.sum() == 0:
            continue
        js = [j for j in PART_JOINTS[p] if j != 0]                     # pelvis belongs to droot
        g = geodesic_sq(R_p[:, :, js], R_b[:, :, js]).mean((1, 2))
        if p in PART_FINGERS:
            fj = [j - 22 for j in PART_FINGERS[p]]
            g = g + geodesic_sq(R_ph[:, :, fj], R_bh[:, :, fj]).mean((1, 2))
        pres.append((g * m).sum() / m.sum())
    out["pres"] = torch.stack(pres).mean() if pres else torch.zeros((), device=dev)
    # foot / ground terms
    feet = pg[:, :, [10, 11]]
    fint = masks["foot_contact_intent"].to(pg.dtype)
    fden = fint.sum().clamp(min=1.0)
    vxz = (feet[:, 1:, :, [0, 2]] - feet[:, :-1, :, [0, 2]]).pow(2).sum(-1)
    out["foot_skate"] = (vxz * fint[:, 1:]).sum() / fint[:, 1:].sum().clamp(min=1.0)
    out["foot_pen"] = torch.relu(-feet[..., 1]).mean()
    feet0 = pg_base[:, :, [10, 11]]
    out["foot_pin"] = (((feet - feet0)[..., [0, 2]].pow(2).sum(-1)) * fint).sum() / fden
    y = feet[..., 1]
    band_lo, band_hi = FOOT_BAND_Y
    dz = torch.relu(band_lo - y).pow(2) + torch.relu(y - band_hi).pow(2)
    out["foot_pin_y"] = ((dz + (y - feet0[..., 1] - lift[:, :, 0:1]).pow(2)) * fint).sum() / fden
    d_run = torch.zeros_like(fint)                                     # in-run cumulative drift
    cur = torch.zeros_like(feet[:, 0])
    run = torch.zeros_like(fint[:, 0])
    for t in range(fint.shape[1]):
        start = (fint[:, t] > 0) & (run < 0.5)
        cur = torch.where(start[..., None], feet[:, t], cur)
        d_run[:, t] = (feet[:, t] - cur)[..., [0, 2]].pow(2).sum(-1) * fint[:, t]
        run = fint[:, t]
    out["foot_drift"] = d_run.sum() / fden
    legs_side = ([1, 4, 7, 10], [2, 5, 8, 11])
    lp, tr = [], []
    for si, js in enumerate(legs_side):
        free = (1.0 - fint[:, :, si])[..., None]
        lp.append((_pres_sq(pg[:, :, js], pg_base[:, :, js], lift) * free).mean())
        tr.append((geodesic_sq(R_p[:, :, js], R_b[:, :, js]) * fint[:, :, si][..., None]).mean())
    out["leg_pres"] = torch.stack(lp).mean()
    out["leg_trust"] = torch.stack(tr).mean()
    # per-frame preservation of every joint of a part outside its (attraction-masked) contact frames
    ci = attr_m
    pf = []
    for p in range(5):
        js = [j for j in PART_JOINTS[p] if j != 0]
        if js:
            e = _pres_sq(pg[:, :, js], pg_base[:, :, js], lift).mean(-1)
            pf.append((e * (1.0 - ci[:, :, p])).mean())
        if p in PART_FINGERS:
            fj = [j - 22 for j in PART_FINGERS[p]]
            e = _pres_sq(pj[:, :, fj], pj_base[:, :, fj], lift).mean(-1)
            pf.append((e * (1.0 - ci[:, :, p])).mean())
    out["pres_frame"] = torch.stack(pf).mean()
    # object pose preservation + low-frequency drift, cap-normalized
    out["obj_pres"] = ((po - base_po).pow(2).sum(-1) / cap_m ** 2).mean()
    dv = (po[:, 1:] - po[:, :-1]) - (base_po[:, 1:] - base_po[:, :-1])
    out["obj_drift"] = (dv.pow(2).sum(-1) / cap_m ** 2).mean()
    out["obj_rpres"] = (geodesic_sq(Ro, base_Ro) / cap_rad ** 2).mean()
    wv0 = so3_log(base_Ro[:, :-1].transpose(-1, -2) @ base_Ro[:, 1:])
    out["obj_wdrift"] = ((wv - wv0).pow(2).sum(-1) / cap_rad ** 2).mean()
    # root preservation (all channels; the vertical one toward the licensed lift)
    dr = snap["droot"]
    rp = (dr[..., :2].pow(2).sum(-1) / cap_root_m ** 2 + (dr[..., 2] / cap_root_rad) ** 2
          + ((dr[..., 3] - lift[:, :, 0]) / cap_root_y) ** 2)
    out["root_pres"] = rp.mean()
    out["z_lift_cm"] = lift.mean() * 100                               # diagnostics, not summed
    out["z_dy_cm"] = dr[..., 3].mean() * 100
    # human rotation temporal smoothness vs the base
    dR = so3_log(R_p[:, :-1].transpose(-1, -2) @ R_p[:, 1:])
    dR0 = so3_log(R_b[:, :-1].transpose(-1, -2) @ R_b[:, 1:])
    out["t_smooth"] = (dR - dR0).pow(2).sum(-1).mean()
    total = (lam_attr * out["attr"] + lam_pen * out["pen"] + lam_slip * out["slip"]
             + lam_smooth * out["smooth"] + lam_pres * out["pres"]
             + lam_foot * (out["foot_skate"] + out["foot_pen"])
             + lam_footpin * (out["foot_pin"] + out["foot_pin_y"] + out["foot_drift"] + out["leg_pres"])
             + lam_legtrust * out["leg_trust"]
             + lam_pres * out["pres_frame"]
             + lam_objpres * (out["obj_pres"] + out["obj_drift"] + out["obj_rpres"] + out["obj_wdrift"])
             + lam_rootpres * out["root_pres"]
             + lam_tsmooth * out["t_smooth"])
    return total, {k: float(v) for k, v in out.items()}


def probe_all(pg, pj, canon, canon_n, p_obj, R_obj):
    """Joint->surface probe for the loss side: d (B,W,52), v (B,W,52,3), n (B,W,52,3) posed normals."""
    surf = pose_surface(canon, R_obj, p_obj)
    allj = torch.cat([pg, pj], 2)
    dm, vm, im = nearest_surface(allj, surf)
    d, v, i = dm[:, 0], vm[:, 0], im[:, 0]
    S = canon_n.shape[2]
    B, W, J = d.shape
    n_can = torch.gather(canon_n[:, 0][:, None].expand(B, W, S, 3), 2, i[..., None].expand(-1, -1, -1, 3))
    return d, v, torch.einsum("bwik,bwjk->bwji", R_obj[:, 0], n_can)
