"""Reconstruction metrics for PamiVAE validation logging.
HUMAN : MPJPE (mm) — 22 joints recovered from the 272 (recover_from_local_position), pelvis-aligned.
OBJECT: pg_cm (free-head p_G error), rot_deg (geodesic), combine_cm (part-anchored combine, predicted routing,
        anchors from the RECON human = what inference does), contact_cm (combine error on frames where an
        anchor part is GT-assigned), hand_mm (finger-joint position error, quality-weighted sequences only),
        pa/<part>_* per-anchor breakdown + oracle routing lower bound.
"""
import numpy as np, torch
from . import config as C
from pami.vendor.eval_trans import recover_from_local_position, calculate_mpjpe, rotation_6d_to_matrix
from .nets.split_ae import PART_NAMES
from .nets.objframe_torch import window_transform, combine_obj_pos_wf

NJOINT = 22


@torch.no_grad()
def human_mpjpe(pred272, gt272):
    """pred272,gt272 (B,W,272) -> mean MPJPE in mm over the batch."""
    pred = pred272.detach().cpu().numpy(); gt = gt272.detach().cpu().numpy()
    tot, n = 0.0, 0
    for b in range(pred.shape[0]):
        pj = torch.from_numpy(recover_from_local_position(pred[b], NJOINT)).float()
        gj = torch.from_numpy(recover_from_local_position(gt[b], NJOINT)).float()
        tot += calculate_mpjpe(gj, pj).mean().item()
        n += 1
    return tot / max(n, 1) * 1000.0


@torch.no_grad()
def obj_trans_cm(pred_pobj, gt_pobj, omask):
    """(B,M,W,3),(B,M,W,3),(B,M) -> mean translation error in cm over ACTIVE objects."""
    e = (pred_pobj - gt_pobj).norm(dim=-1).mean(-1)
    den = omask.sum().clamp(min=1.0)
    return ((e * omask).sum() / den).item() * 100.0


@torch.no_grad()
def obj_rot_deg(pred_robj, gt_robj, omask):
    """(B,M,W,6),(B,M,W,6),(B,M) -> mean geodesic rotation error in degrees over ACTIVE objects."""
    Rp = rotation_6d_to_matrix(pred_robj); Rg = rotation_6d_to_matrix(gt_robj)
    Rrel = torch.matmul(Rp.transpose(-1, -2), Rg)
    tr = Rrel[..., 0, 0] + Rrel[..., 1, 1] + Rrel[..., 2, 2]
    ang = torch.acos(((tr - 1) / 2).clamp(-1.0, 1.0)).mean(-1) * (180.0 / np.pi)
    den = omask.sum().clamp(min=1.0)
    return ((ang * omask).sum() / den).item()


@torch.no_grad()
def split_evaluate(model, loader, dev, norm, max_batches=None, anchor_fk=None, **_ignored):
    """Mean {mpjpe_mm, pg_cm, rot_deg, combine_cm, contact_cm, hand_mm, pa/*} over the loader."""
    assert anchor_fk is not None
    was_training = model.training
    model.eval()
    acc = {"mpjpe_mm": 0.0, "pg_cm": 0.0, "rot_deg": 0.0, "combine_cm": 0.0, "contact_cm": 0.0, "hand_mm": 0.0}
    n = 0; hand_n = 0.0
    ds = loader.dataset
    hmean = torch.tensor(ds.hand_mean[:90], device=dev); hstd = torch.tensor(ds.hand_std[:90], device=dev)
    for i, b in enumerate(loader):
        if max_batches is not None and i >= max_batches: break
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        out = model(b)
        hr, h = norm.dh(out["human_recon"]), norm.dh(b["human"])
        p_pred, p_gt = norm.do("pobj", out["pobj"]), norm.do("pobj", b["pobj"])
        ro_pred, ro_gt = norm.do("robj", out["robj"]), norm.do("robj", b["robj"])
        acc["mpjpe_mm"] += human_mpjpe(hr, h)
        acc["pg_cm"] += obj_trans_cm(p_pred, p_gt, b["omask"])
        acc["rot_deg"] += obj_rot_deg(ro_pred, ro_gt, b["omask"])
        hp = out["hand_recon"][..., :90] * hstd + hmean
        hg = b["hand"][..., :90] * hstd + hmean
        e = (hp - hg).reshape(*hp.shape[:2], 30, 3).norm(dim=-1).mean(dim=(1, 2))
        hw = (b["hand_w"] > 0).float()
        acc["hand_mm"] += float((e * hw).sum() * 1000); hand_n += float(hw.sum())
        parts = list(model.delta_parts)
        WD = b["w"].shape[-1]
        dfull = torch.zeros(*out["delta"].shape[:3], 6, 3, device=dev)
        dfull[..., parts, :] = out["delta"]
        anchor_p, anchor_R = anchor_fk(hr, jrest=b["jrest"])
        Twf, off_wf = window_transform(hr)
        p_comb = combine_obj_pos_wf(norm.do("delta", dfull), p_pred, out["w"], anchor_p, anchor_R, Twf, off_wf)
        acc["combine_cm"] += obj_trans_cm(p_comb, p_gt, b["omask"])
        sel = (b["omask"][..., None] > 0.5).expand(-1, -1, p_gt.shape[2])
        den = sel.sum().clamp(min=1)
        _m = lambda x: float((x * sel).sum() / den)
        d_dn = norm.do("delta", dfull)
        fkp = anchor_p[:, None] + torch.einsum("bwpij,bmwpj->bmwpi", anchor_R, d_dn)
        fkp = torch.einsum("bwij,bmwpj->bmwpi", Twf, fkp) + off_wf[:, None, :, None]
        e_path = (fkp - p_gt[..., None, :]).norm(dim=-1) * 100.0
        e_dlt = (d_dn - norm.do("delta", b["delta"])).norm(dim=-1) * 100.0
        wsm = out["w"].softmax(-1)
        for pi in parts:
            nm = PART_NAMES[pi]
            acc[f"pa/{nm}_pos_cm"] = acc.get(f"pa/{nm}_pos_cm", 0.0) + _m(e_path[..., pi])
            acc[f"pa/{nm}_delta_cm"] = acc.get(f"pa/{nm}_delta_cm", 0.0) + _m(e_dlt[..., pi])
            acc[f"pa/{nm}_w"] = acc.get(f"pa/{nm}_w", 0.0) + _m(wsm[..., pi])
        acc["pa/FREE_w"] = acc.get("pa/FREE_w", 0.0) + _m(wsm[..., WD - 1])
        cand = torch.cat([e_path[..., parts], ((p_pred - p_gt).norm(dim=-1) * 100.0)[..., None]], -1)
        acc["pa/oracle_pos_cm"] = acc.get("pa/oracle_pos_cm", 0.0) + _m(cand.min(-1).values)
        cmask = (b["w"][..., parts].sum(-1) > 0.1).float()
        e = (p_comb - p_gt).norm(dim=-1)
        cden = (b["omask"][..., None] * cmask).sum().clamp(min=1)
        acc["contact_cm"] += ((e * b["omask"][..., None] * cmask).sum() / cden).item() * 100.0
        n += 1
    model.train(was_training)
    out_acc = {k: v / max(n, 1) for k, v in acc.items() if k != "hand_mm"}
    out_acc["hand_mm"] = (acc["hand_mm"] / hand_n) if hand_n > 0 else float("nan")
    return out_acc
