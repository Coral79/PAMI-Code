"""PamiRefiner trainer: T=3 recursion with full BPTT, two streams, the shipped recipe.

STREAMS (alternating: every --real_every-th iteration is a real step)
  corruption  roundtrip banks (decode(encode(GT)) windows) -> group-sampled corruption
              (corruption_m3) -> per-step absolute-state supervision vs the RAW GT (corruption_losses,
              vertex + rotation + cap-normalized object/root terms).
  real        gen banks (frozen generator windows, no GT answer) -> physical / preservation losses on
              the bank's FROZEN masks (real_losses).
CURRICULUM: rollout depth T_now = min(T, 1 + int(2 * it / warmup)).
EVAL every --eval_every iterations: per-group acceptance table on the val roundtrip bank (recovery /
harm per side -> `score`) and the real-stream gates on the val gen bank; `best` (<out>) is saved when
the score improves AND the gates pass; periodic `<out>_it<N>.pt` (with optimizer state) every
--save_every regardless. The shipped checkpoint is the periodic it44000 file of a 60k run.

Run: python train_refiner.py --rt_bank ... --gen_bank ... --val_rt_bank ... --val_gen_bank ... --out ...
"""
import argparse
import math
import os
import time

import numpy as np
import torch

from pami.refine.arch_guard import ARCH_VERSION
from pami.refine.base_state import SMPL_PARENTS, so3_log
from pami.refine.corruption_m3 import GROUPS, P_GROUP, SIGMA, corrupt_m3
from pami.refine.data_m3 import BankWindows, ModelBatchSampler, collate_m3, sensor_cloud
from pami.refine.features_m3 import D3_BODY, D3_FINGER, D3_OBJ, D3_SCENE
from pami.refine.hand_state import rest_hand_offsets, tip_rest_offsets
from pami.refine.losses_m3 import (FOOT_BAND_Y, bands_for, corruption_losses, foot_band_lift,
                                   load_band_table, part_distance, probe_all, real_losses)
from pami.refine.net_m3 import RefineNetM3, token_sides_m3
from pami.refine.paths import interact_common
from pami.refine.rollout_m3 import CAPS_DEFAULT, RolloutState, fk_state, rollout, sensor_clouds
from pami.refine.vertex_m3 import _bm, hand_sub_mask, verts_from_state
from pami.rotvote import mat_to_rot6d
from pami.vendor.eval_trans import rotation_6d_to_matrix

FRAME_MODE = "wf_abs"
SIDES = ("body", "hand", "obj_pos", "obj_rot", "root", "root_y")
# acceptance tolerances = measured repeat-noise (p95 |pairwise delta| of the per-side mean error over
# repeat evals with different seeds), cm / deg. eps gates valid recovery, delta is the harm tolerance.
ACC_EPS = dict(body=0.29, hand=0.44, obj_pos=0.22, obj_rot=0.30, root=0.23, root_y=0.11)
ACC_DELTA = dict(ACC_EPS)
# real-stream gate tolerances (same protocol), cm; harm_pct is declared, not measured
GATE_TOL = dict(gap_cm=0.43, pen_cm=0.53, skate_cm=0.12, jerk_cm=0.74, harm_pct=5.0)


# ---- batch -> state / geometry ------------------------------------------------------------------
def to_state(b, dev, prefix=""):
    """Collated batch -> state dict on the refiner's chain ('' = pipeline state, 'gt_' = raw GT).
    The root column of rot6 is the explicit window-first root."""
    W = b[prefix + "h272"].shape[1]
    h = b[prefix + "h272"].to(dev)
    rot6 = mat_to_rot6d(rotation_6d_to_matrix(h[..., 140:272].reshape(-1, W, 22, 6)))
    rot6 = rot6.clone()
    rot6[:, :, 0] = mat_to_rot6d(rotation_6d_to_matrix(b[prefix + "root_wf_rot6"].to(dev)))
    return dict(rot6=rot6, rot6h=mat_to_rot6d(rotation_6d_to_matrix(b[prefix + "rot6h"].to(dev))),
                pel=b[prefix + "root_wf_pos"].to(dev), p_obj=b[prefix + "p_obj"].to(dev),
                R_obj=rotation_6d_to_matrix(b[prefix + "r6_obj"].to(dev)))


_TIP_OFF_CACHE = {}


def _tip_off_batch(b, dev):
    """(B,2,5,3) rest DIP->fingertip offsets, one rest-pose body-model forward per distinct subject."""
    outs = []
    for i in range(b["betas16"].shape[0]):
        nb = int(b["nb"][i])
        key = (b["gender"][i], b["model"], nb, b["betas16"][i, :nb].numpy().tobytes())
        v = _TIP_OFF_CACHE.get(key)
        if v is None:
            bm = _bm(b["gender"][i], b["model"], nb, "cpu")
            z = lambda n: torch.zeros(1, n)
            o = bm(root_orient=z(3), pose_body=z(63), pose_hand=z(90), betas=b["betas16"][i, None, :nb].float())
            v = tip_rest_offsets(o.v, o.Jtr, b["model"])[0]
            _TIP_OFF_CACHE[key] = v
        outs.append(v)
    return torch.stack(outs).to(dev)


# Finger-supervision trust per InterAct subset (1 = the GT fingers supervise rot_hand and the hand
# vertices, 0 = they do not). This is the hand_quality table the shipped refiner was trained with
# (grab strong; omomo/behave/intercap mid; neuraldome/imhd
# weak; chairs none) -- NOT the older quality stamps inside hand_joints.npz that the VAE's hand-loss
# weights used (pami.data_prep.subsets keeps those).
HAND_TRUST = {"grab": 1.0, "omomo": 1.0, "behave": 1.0, "intercap": 1.0, "neuraldome": 0.0, "imhd": 0.0, "chairs": 0.0}
_SEQ_SUBSET = {}


def hand_quality_w(seqs, dev):
    """(B,) finger-supervision trust of each sequence's subset (HAND_TRUST; unknown subset -> 0)."""
    if not _SEQ_SUBSET:
        _SEQ_SUBSET.update(interact_common().seq_subset_map())
    return torch.tensor([HAND_TRUST.get(_SEQ_SUBSET.get(s, ""), 0.0) for s in seqs], device=dev)


def batch_geom(b, dev):
    jrest = b["jrest"].to(dev)
    return dict(jrest=jrest[:, :22], rest_h=rest_hand_offsets(jrest, b["model"]),
                canon=b["canon"].to(dev), canon_n=b["canon_n"].to(dev),
                sensor_canon=(b["sensor_canon"].to(dev) if "sensor_canon" in b else None),
                sensor_canon_n=(b["sensor_canon_n"].to(dev) if "sensor_canon_n" in b else None),
                betas16=b["betas16"].to(dev), genders=b["gender"], nbs=b["nb"], model=b["model"],
                tip_off=_tip_off_batch(b, dev),
                masks=dict(contact_intent=b["contact_intent"].to(dev), reachable_mask=b["reachable_mask"].to(dev),
                           foot_contact_intent=b["foot_contact_intent"].to(dev),
                           normals_trusted=b["normals_trusted"].to(dev)),
                anchors=b["segment_anchor"])


class _Snap:
    """Minimal state shim for fk_state: a snapshot dict whose root offset is already baked in."""

    def __init__(self, d):
        self.rot6, self.rot6h, self.pel = d["rot6"], d["rot6h"], d["pel"]

    def root_applied(self):
        return self.pel, self.rot6


def _fk(d, geom, parents):
    """state dict -> (pg (B,W,22,3), pj (B,W,30,3))."""
    pg, _, pj, _, _, _, _, _ = fk_state(_Snap(d), parents, geom["jrest"], geom["rest_h"])
    return pg, pj


# ---- one training step ---------------------------------------------------------------------------
def train_step(net, b, dev, g, real_step, T_now, parents, sides_tok, band_table, all_row,
               lam_v=7.0, lam_n=5.0, lam_real=1.0, lam_root=1.0, lam_root_y=1.0, lam_foot=2.0):
    """One iteration on one collated batch: zero grads, rollout T_now steps, per-step losses,
    backward. Returns dict(loss=float, parts=dict, tag=str, sat=dict). The caller clips + steps."""
    geom = batch_geom(b, dev)
    dec = to_state(b, dev)
    net.zero_grad()
    if real_step:
        base = {k: v.clone() for k, v in dec.items()}
        with torch.no_grad():                              # base FK + the frozen vertical evidence
            pg_b, pj_b = _fk(base, geom, parents)
            lift = foot_band_lift(pg_b, geom["masks"]["foot_contact_intent"].to(pg_b.dtype), cap=net.cap_root_y)
        st = RolloutState(dec["rot6"], dec["rot6h"], dec["pel"], dec["p_obj"], dec["R_obj"])
        snaps = []
        st, sat = rollout(net, st, sides_tok, parents, geom["jrest"], geom["rest_h"], geom["canon"], geom["canon_n"],
                          geom["betas16"], geom["masks"]["foot_contact_intent"], T=T_now, collect=snaps,
                          **sensor_clouds(geom))
        band_t = bands_for(b["obj_name"], band_table, all_row, dev)
        loss = 0.0
        for snap in snaps:                                 # per-step supervision
            pg, pj = _fk(snap, geom, parents)
            d_all, v_all, n_all = probe_all(pg, pj, geom["canon"], geom["canon_n"], snap["p_obj"], snap["R_obj"])
            l, parts = real_losses(snap, base, geom["masks"], geom["anchors"], pg, pj, d_all, v_all, n_all,
                                   bands=band_t, pg_base=pg_b, pj_base=pj_b, lift=lift, lam_foot=lam_foot,
                                   cap_m=net.cap_obj_m, cap_rad=net.cap_obj_rad)
            loss = loss + lam_real * l / len(snaps)
        tag = "real"
    else:
        gt = to_state(b, dev, "gt_")
        r6c, r6hc, pelc, pc, Rc, group = corrupt_m3(dec["rot6"], dec["rot6h"], dec["pel"], dec["p_obj"],
                                                    dec["R_obj"], g, dev)
        st = RolloutState(r6c, r6hc, pelc, pc, Rc)
        snaps = []
        st, sat = rollout(net, st, sides_tok, parents, geom["jrest"], geom["rest_h"], geom["canon"], geom["canon_n"],
                          geom["betas16"], geom["masks"]["foot_contact_intent"], T=T_now, collect=snaps,
                          **sensor_clouds(geom))
        with torch.no_grad():
            V_gt = verts_from_state(gt["rot6"], gt["rot6h"], gt["pel"], geom["betas16"], geom["genders"],
                                    geom["model"], geom["nbs"])
        hand_w, vert_hand = hand_quality_w(b["seq"], dev), hand_sub_mask(b["model"], dev)
        loss, parts = 0.0, {}
        for snap in snaps:
            V_pred = verts_from_state(snap["rot6"], snap["rot6h"], snap["pel"], geom["betas16"], geom["genders"],
                                      geom["model"], geom["nbs"])
            l, parts = corruption_losses(snap, gt, net, lam_v=lam_v, lam_n=lam_n, lam_root=lam_root,
                                         lam_root_y=lam_root_y, V_pred=V_pred, V_gt=V_gt,
                                         hand_w=hand_w, vert_hand=vert_hand)
            loss = loss + l / len(snaps)
        tag = f"corr:{group}"
    loss.backward()
    return dict(loss=float(loss), parts=parts, tag=tag, sat=sat)


# ---- evaluation: acceptance table (corruption stream) + real-stream gates -------------------------
def side_errors(snap, gt, geom, parents):
    """Per-sample per-side error (means over the window), cm / deg."""
    pg, pj = _fk(snap, geom, parents)
    pgg, pjg = _fk(gt, geom, parents)
    e = dict(body=(pg - pgg).norm(dim=-1).mean((1, 2)) * 100,
             hand=(pj - pjg).norm(dim=-1).mean((1, 2)) * 100,
             obj_pos=(snap["p_obj"] - gt["p_obj"]).norm(dim=-1).mean((1, 2)) * 100,
             obj_rot=torch.rad2deg(so3_log(snap["R_obj"].transpose(-1, -2) @ gt["R_obj"]).norm(dim=-1)).mean((1, 2)),
             root=(snap["pel"][..., [0, 2]] - gt["pel"][..., [0, 2]]).norm(dim=-1).mean(1) * 100,
             root_y=(snap["pel"][..., 1] - gt["pel"][..., 1]).abs().mean(1) * 100)
    return e


def acceptance(net, batches, parents, sides_tok, dev, seed, T=3):
    """Per corruption group x side: recovery (where the corruption exceeded eps), harm rate, gain.
    Ranking score = mean recovery - 5 * mean harm rate."""
    agg = {gname: {s: dict(ec=[], el=[], eo=[]) for s in SIDES} for gname in GROUPS}
    gv = torch.Generator(device=dev).manual_seed(seed)
    net.eval()
    with torch.no_grad():
        for b in batches:
            geom = batch_geom(b, dev)
            dec, gt = to_state(b, dev), to_state(b, dev, "gt_")
            for group in GROUPS:
                r6c, r6hc, pelc, pc, Rc, _ = corrupt_m3(dec["rot6"], dec["rot6h"], dec["pel"], dec["p_obj"],
                                                        dec["R_obj"], gv, dev, group=group)
                st = RolloutState(r6c.clone(), r6hc.clone(), pelc.clone(), pc.clone(), Rc.clone())
                st, _ = rollout(net, st, sides_tok, parents, geom["jrest"], geom["rest_h"], geom["canon"],
                                geom["canon_n"], geom["betas16"], geom["masks"]["foot_contact_intent"], T=T,
                                **sensor_clouds(geom))
                e_c = side_errors(dict(rot6=r6c, rot6h=r6hc, pel=pelc, p_obj=pc, R_obj=Rc), gt, geom, parents)
                e_l = side_errors(dec, gt, geom, parents)
                e_o = side_errors(st.snapshot(), gt, geom, parents)
                for s in SIDES:
                    agg[group][s]["ec"].append(e_c[s].cpu()); agg[group][s]["el"].append(e_l[s].cpu())
                    agg[group][s]["eo"].append(e_o[s].cpu())
    net.train()
    rows, rec_terms, harm_terms = [], [], []
    for gname, per in agg.items():
        cells = []
        for s in SIDES:
            ec, el, eo = (torch.cat(per[s][k]) for k in ("ec", "el", "eo"))
            exc_in, exc_out = (ec - el).clamp(min=0), (eo - el).clamp(min=0)
            valid = exc_in > ACC_EPS[s]
            rec = float((1 - exc_out[valid] / exc_in[valid]).mean()) * 100 if valid.any() else None
            harm = float((eo > ec + ACC_DELTA[s]).float().mean()) * 100
            gain = float((ec - eo).mean())
            cells.append((f"{s}: rec {rec:5.1f}%/{int(valid.sum())}" if rec is not None else f"{s}: rec   N/A")
                         + f" gain {gain:+.2f} harm {harm:4.1f}%")
            if rec is not None:
                rec_terms.append(rec / 100.0)
            harm_terms.append(harm / 100.0)
        rows.append(f"{gname:>15} | " + " | ".join(cells))
    score = (float(np.mean(rec_terms)) if rec_terms else -1e9) - 5.0 * (float(np.mean(harm_terms)) if harm_terms else 0.0)
    summary = dict(mean_recovery=(float(np.mean(rec_terms)) * 100 if rec_terms else None),
                   mean_harm_pct=(float(np.mean(harm_terms)) * 100 if harm_terms else 0.0))
    return rows, score, summary


def _gate_metrics(st, geom, parents, band_t):
    """GT-free per-state gate metrics on the real stream (frozen masks + geometry only), cm."""
    pg, pj = _fk(st, geom, parents)
    d_all, _, _ = probe_all(pg, pj, geom["canon"], geom["canon_n"], st["p_obj"], st["R_obj"])
    dpart = part_distance(d_all)
    m = (geom["masks"]["contact_intent"] & geom["masks"]["reachable_mask"]).to(dpart.dtype)
    gap = float((torch.relu(dpart - band_t[:, None, :]) * m).sum() / m.sum().clamp(min=1.0)) * 100
    feet = pg[:, :, [10, 11]]
    fint = geom["masks"]["foot_contact_intent"].to(dpart.dtype)
    pen = float(torch.relu(-feet[..., 1]).mean()) * 100
    sk = (feet[:, 1:, :, [0, 2]] - feet[:, :-1, :, [0, 2]]).norm(dim=-1)
    fm = fint[:, 1:] > 0
    skate = (sk[fm].flatten() * 100) if fm.any() else torch.zeros(0)
    po = st["p_obj"][:, 0]
    v = po[:, 1:] - po[:, :-1]
    jerk = ((v[:, 1:] - v[:, :-1])[:, 1:] - (v[:, 1:] - v[:, :-1])[:, :-1]).norm(dim=-1) * 100
    return dict(gap=gap, pen=pen, skate=skate.cpu(), jerk=jerk.flatten().cpu())


def real_gates(net, gen_batches, parents, sides_tok, dev, T, band_table, all_row, harm_pct):
    """G1 masked contact gap, G2 foot penetration + skate p90, G3 object jerk p95 + |dSE3| caps,
    G4 roundtrip harm: refined must not exceed base + tolerance. Returns (passed, lines, record)."""
    net.eval()
    agg = {k: dict(base=[], ref=[]) for k in ("gap", "pen", "skate", "jerk")}
    dse_p, dse_w = [], []
    with torch.no_grad():
        for b in gen_batches:
            geom = batch_geom(b, dev)
            dec = to_state(b, dev)
            band_t = bands_for(b["obj_name"], band_table, all_row, dev)
            st = RolloutState(dec["rot6"].clone(), dec["rot6h"].clone(), dec["pel"].clone(),
                              dec["p_obj"].clone(), dec["R_obj"].clone())
            st, _ = rollout(net, st, sides_tok, parents, geom["jrest"], geom["rest_h"], geom["canon"], geom["canon_n"],
                            geom["betas16"], geom["masks"]["foot_contact_intent"], T=T, **sensor_clouds(geom))
            ref = st.snapshot()
            mb, mr = _gate_metrics(dec, geom, parents, band_t), _gate_metrics(ref, geom, parents, band_t)
            for k in agg:
                agg[k]["base"].append(mb[k]); agg[k]["ref"].append(mr[k])
            dse_p.append((ref["p_obj"] - dec["p_obj"]).norm(dim=-1).flatten().cpu() * 100)
            dse_w.append(torch.rad2deg(so3_log(dec["R_obj"].transpose(-1, -2) @ ref["R_obj"]).norm(dim=-1)).flatten().cpu())
    net.train()
    mean = lambda xs: float(np.mean(xs))
    q = lambda xs, p: float(torch.cat(xs).quantile(p)) if xs and torch.cat(xs).numel() else 0.0
    gap_b, gap_r = mean(agg["gap"]["base"]), mean(agg["gap"]["ref"])
    pen_b, pen_r = mean(agg["pen"]["base"]), mean(agg["pen"]["ref"])
    sk_b, sk_r = q(agg["skate"]["base"], 0.9), q(agg["skate"]["ref"], 0.9)
    jk_b, jk_r = q(agg["jerk"]["base"], 0.95), q(agg["jerk"]["ref"], 0.95)
    dp, dw = torch.cat(dse_p), torch.cat(dse_w)
    cap_p, cap_w = net.cap_obj_m * 100, math.degrees(net.cap_obj_rad)
    g = {"G1_gap": gap_r <= gap_b + GATE_TOL["gap_cm"],
         "G2_foot": (pen_r <= pen_b + GATE_TOL["pen_cm"]) and (sk_r <= sk_b + GATE_TOL["skate_cm"]),
         "G3_obj": (jk_r <= jk_b + GATE_TOL["jerk_cm"]) and float(dp.max()) <= cap_p * 1.001 and float(dw.max()) <= cap_w * 1.001,
         "G4_harm": harm_pct <= GATE_TOL["harm_pct"]}
    ok = lambda k: "PASS" if g[k] else "FAIL"
    lines = [f"G1 gap(masked excess, cm) base {gap_b:6.2f} -> ref {gap_r:6.2f} (tol +{GATE_TOL['gap_cm']})  {ok('G1_gap')}",
             f"G2 foot pen {pen_b:5.3f}->{pen_r:5.3f} (tol +{GATE_TOL['pen_cm']}) | skate p90 {sk_b:5.3f}->{sk_r:5.3f} "
             f"(tol +{GATE_TOL['skate_cm']})  {ok('G2_foot')}",
             f"G3 obj jerk p95 {jk_b:5.3f}->{jk_r:5.3f} (tol +{GATE_TOL['jerk_cm']}) | |dSE3| max {float(dp.max()):5.2f}cm/"
             f"{float(dw.max()):5.2f}deg (caps {cap_p:.0f}/{cap_w:.0f})  {ok('G3_obj')}",
             f"G4 roundtrip harm {harm_pct:5.2f}% <= {GATE_TOL['harm_pct']}%  {ok('G4_harm')}"]
    rec = dict(g, gap_base=gap_b, gap_ref=gap_r, pen_base=pen_b, pen_ref=pen_r, skate_p90_base=sk_b, skate_p90_ref=sk_r,
               jerk_p95_base=jk_b, jerk_p95_ref=jk_r, dse3_p_max_cm=float(dp.max()), dse3_w_max_deg=float(dw.max()),
               harm_pct=harm_pct, tol=dict(GATE_TOL))
    return all(g.values()), lines, rec


# ---- main ----------------------------------------------------------------------------------------
def build_argparser():
    ap = argparse.ArgumentParser(description="train the PamiRefiner (defaults = the shipped recipe)")
    ap.add_argument("--rt_bank", nargs="+", required=True, help="roundtrip bank dirs (corruption stream)")
    ap.add_argument("--gen_bank", nargs="+", required=True, help="gen bank dirs (real stream)")
    ap.add_argument("--val_rt_bank", nargs="+", required=True, help="held-out roundtrip bank (acceptance table)")
    ap.add_argument("--val_gen_bank", nargs="+", required=True, help="held-out gen bank (real-stream gates)")
    ap.add_argument("--out", required=True, help="best checkpoint path; periodic files go to <out>_it<N>.pt")
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--surf", type=int, default=512, help="probe cloud: the stored shape_0 strided to N points")
    ap.add_argument("--dense_surf", type=int, default=16384, help="sensor cloud: object mesh resampled at N points")
    ap.add_argument("--dense_seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--T", type=int, default=3, help="rollout steps (training)")
    ap.add_argument("--warmup", type=int, default=12000, help="rollout-depth curriculum length")
    ap.add_argument("--lam_v", type=float, default=7.0)
    ap.add_argument("--lam_n", type=float, default=5.0)
    ap.add_argument("--lam_real", type=float, default=1.0)
    ap.add_argument("--lam_root", type=float, default=1.0)
    ap.add_argument("--lam_root_y", type=float, default=1.0)
    ap.add_argument("--lam_foot", type=float, default=2.0, help="real-stream foot_skate + foot_pen weight")
    ap.add_argument("--real_every", type=int, default=2, help="every k-th iteration is a real-stream step")
    ap.add_argument("--sensor_K", type=int, default=64)
    ap.add_argument("--sensor_radius", type=float, default=0.04)
    ap.add_argument("--sensor_seed", type=int, default=20260815)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=2000, help="periodic checkpoint cadence (multiple of --eval_every)")
    ap.add_argument("--draws", type=int, default=10, help="val batches per eval")
    ap.add_argument("--val_seed", type=int, default=1234)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv=None):
    a = build_argparser().parse_args(argv)
    if a.save_every > 0 and a.save_every % a.eval_every:
        a.save_every = ((a.save_every + a.eval_every - 1) // a.eval_every) * a.eval_every
        print(f"[refiner] --save_every rounded up to {a.save_every} (a multiple of --eval_every)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    S_SURF = a.dense_surf or a.surf                          # the cloud length the sensor actually sees
    dkw = dict(dense_surf=a.dense_surf, dense_seed=a.dense_seed)
    tr_rt = BankWindows(a.rt_bank, a.surf, stream="roundtrip", **dkw)
    PROBE_SURF = int(tr_rt[0]["canon"].shape[-2])
    va_rt = BankWindows(a.val_rt_bank, a.surf, stream="roundtrip", **dkw)
    tr_gen = BankWindows(a.gen_bank, a.surf, stream="gen", **dkw)
    va_gen = BankWindows(a.val_gen_bank, a.surf, stream="gen", **dkw)
    print(f"[refiner] rt {len(tr_rt)} | val_rt {len(va_rt)} | gen {len(tr_gen)} | val_gen {len(va_gen)}"
          f" | varlen={tr_rt.varlen}", flush=True)

    def mk(ds, sh, seed=0, persist=True):
        return torch.utils.data.DataLoader(
            ds, batch_sampler=ModelBatchSampler(getattr(ds, "batch_keys", None) or ds.models, a.batch, shuffle=sh, seed=seed),
            num_workers=a.workers, collate_fn=collate_m3, persistent_workers=persist and a.workers > 0)

    dl_rt, dl_gen = mk(tr_rt, True), mk(tr_gen, True, seed=7)
    val_batches = [b for i, b in zip(range(a.draws), mk(va_rt, True, seed=a.val_seed, persist=False))]
    parents = SMPL_PARENTS.to(dev)
    sides_tok = token_sides_m3().to(dev)
    net = RefineNetM3(D3_BODY, D3_FINGER, D3_OBJ, D3_SCENE, d=a.d, layers=a.layers,
                      sensor_kw=dict(K=a.sensor_K, radius=a.sensor_radius, init_seed=a.sensor_seed, tips=True)).to(dev)
    n_have = int(sensor_cloud(val_batches[0])[0].shape[-2])
    assert n_have == S_SURF, f"sensor cloud is {n_have} points but --dense_surf/--surf say {S_SURF}"
    print(f"[refiner] sensor b2 K={a.sensor_K} r={a.sensor_radius} tips cloud={S_SURF} (dense resample) "
          f"probe cloud={PROBE_SURF} | params {sum(q.numel() for q in net.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    BAND_TABLE, ALL_ROW = load_band_table()
    print(f"[refiner] contact bands(cm) ALL " + " ".join(f"{p}:{v*100:.1f}" for p, v in ALL_ROW.items())
          + f" | {len({k[0] for k in BAND_TABLE if k[0].startswith('obj:')})} per-object rows", flush=True)
    val_gen_batches = [b for i, b in zip(range(a.draws), mk(va_gen, True, seed=a.val_seed, persist=False))]

    it, best, t0 = 0, None, time.time()
    run, sat_last = {}, {}
    it_rt, it_gen = iter(dl_rt), iter(dl_gen)
    while it < a.iters:
        it += 1
        T_now = min(1 + int(2 * min(1.0, it / max(a.warmup, 1))), a.T)
        real_step = it % a.real_every == 0
        try:
            b = next(it_gen if real_step else it_rt)
        except StopIteration:
            if real_step:
                it_gen = iter(dl_gen); b = next(it_gen)
            else:
                it_rt = iter(dl_rt); b = next(it_rt)
        r = train_step(net, b, dev, g, real_step, T_now, parents, sides_tok, BAND_TABLE, ALL_ROW,
                       lam_v=a.lam_v, lam_n=a.lam_n, lam_real=a.lam_real, lam_root=a.lam_root,
                       lam_root_y=a.lam_root_y, lam_foot=a.lam_foot)
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        sat_last = r["sat"]
        run.setdefault("loss", []).append(r["loss"])
        for k, v in r["parts"].items():
            run.setdefault(k, []).append(v)
        if it % 100 == 0:
            msg = " ".join(f"{k} {np.mean(v):.4f}" for k, v in sorted(run.items()))
            sat = " ".join(f"{k}:{v:.2f}" for k, v in sat_last.items() if v > 0.01)
            print(f"it{it} T{T_now} [{r['tag']}] {msg}{(' | sat ' + sat) if sat else ''} | "
                  f"{it/(time.time()-t0):.2f} it/s", flush=True)
            run = {}
        if it % a.eval_every == 0 or it == a.iters:
            rows, score, acc_sum = acceptance(net, val_batches, parents, sides_tok, dev, a.val_seed, T=a.T)
            for row in rows:
                print("   " + row, flush=True)
            gates_ok, glines, grec = real_gates(net, val_gen_batches, parents, sides_tok, dev, a.T, BAND_TABLE, ALL_ROW,
                                                acc_sum["mean_harm_pct"])
            for gl in glines:
                print("   [gates] " + gl, flush=True)
            payload = {"model": net.state_dict(), "args": vars(a), "it": it, "score": score, "rows": rows,
                       "frame_mode": FRAME_MODE, "sigma": SIGMA, "p_group": dict(P_GROUP),
                       "caps": {k: float(v) for k, v in CAPS_DEFAULT.items()}, "arm": "A2",
                       "arch_version": ARCH_VERSION, "caps_obj": {"cm": 5.0, "deg": 10.0},
                       "gates": grec, "gate_tol": dict(GATE_TOL), "acc_eps": dict(ACC_EPS), "acc_delta": dict(ACC_DELTA),
                       "acceptance_summary": acc_sum, "bands": {int(k): float(v) for k, v in ALL_ROW.items()},
                       "foot_band_y": list(FOOT_BAND_Y),
                       "caps_root": {"m": net.cap_root_m, "rad": net.cap_root_rad, "y": net.cap_root_y},
                       "banks": {"rt": a.rt_bank, "gen": a.gen_bank, "val_rt": a.val_rt_bank, "val_gen": a.val_gen_bank},
                       "token_layout": "m3-54",
                       "sensor": {"arm": "b2", "K": a.sensor_K, "radius": a.sensor_radius, "points": 0,
                                  "seed": a.sensor_seed, "select": "uniform", "tips": True, "surf": S_SURF,
                                  "probe_surf": PROBE_SURF, "points_src": "iid_area_weighted:all",
                                  "dense_surf": a.dense_surf, "dense_seed": a.dense_seed, "cube": 0.0, "cube_cap": 4000}}
            if (best is None or score > best) and not gates_ok:
                print(f"   -> NOT saved as best: score {score:.4f} but a real-stream gate FAILED", flush=True)
            if (best is None or score > best) and gates_ok:
                best = score
                torch.save(payload, a.out)
                print(f"   -> saved {a.out} (score {score:.4f})", flush=True)
            if a.save_every > 0 and (it % a.save_every == 0 or it == a.iters):
                per = f"{a.out[:-3] if a.out.endswith('.pt') else a.out}_it{it}.pt"
                torch.save({**payload, "opt": opt.state_dict(), "best_score_so_far": best, "periodic": True,
                            "rng": {"corrupt_g": g.get_state(), "epoch_rt": dl_rt.batch_sampler.epoch,
                                    "epoch_gen": dl_gen.batch_sampler.epoch}}, per)
                print(f"   -> saved periodic {per} (it {it}, score {score:.4f}, gates_ok={gates_ok})", flush=True)
    print("REFINER TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
