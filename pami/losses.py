"""PamiVAE training loss (losses.split_ae_loss). All terms in NORMALISED space unless stated.

human : h_recon (272 MSE) + h_root (root block [0:8]) + kl_h + h_hand (hand block, weighted by the
        sequence's hand quality) + h_root_vel (first-difference MSE of the window-absolute root block)
object: o_robj + o_pg (free head p_G) + o_vel (physical p_G velocity) + kl_o (active slots only)
combine (part-anchored voting): o_trans = MSE of the routed combine p = sum_p softmax(w)_p (anchor_p + R_p delta_p)
        + w_FREE p_G against the GT p_G, in physical window-first metres, with anchors FK'd from the model's OWN
        reconstructed human (AnchorFK, per-sequence rest skeleton) -> train == inference;
        o_delta = MSE of the anchor deltas on the anchor parts; o_w = soft routing guide MSE(softmax(w), GT w).
decoder-finetune auxes (0 = off): h_fslide (foot-skate: stance-gated world foot velocity; gate "light" = LIGHT's
        |dfeet|^2 < 0.002 rule on the GT, gate "pred" = model's contact head OR GT stance), h_wvel (world-frame
        22-joint velocity MSE), h_fkcons (position block vs FK of the rotation block), h_ctc (BCE of the
        contact head vs GT foot-contact labels: speed < 0.15 m/s & height < 0.15/0.10 m).
Returns (total, dict-of-float-components).
"""
import numpy as np, torch, torch.nn.functional as F
from . import config as C
from pami.vendor.eval_trans import rotation_6d_to_matrix
from .nets.objframe_torch import window_transform, combine_obj_pos_wf

DEFAULT_LAM = dict(root=C.LAMBDA_ROOT, kl_h=C.SAE_LAMBDA_KL_H, robj=C.SAE_LAMBDA_ROBJ, pg=C.SAE_LAMBDA_PG,
                   vel=C.SAE_LAMBDA_VEL, kl_o=C.SAE_LAMBDA_KL_O, hand=1.0, root_vel=50.0,
                   delta=1.0, trans=1.0, w=C.LAMBDA_W,
                   fslide=0.0, wvel=0.0, fkcons=0.0, ctc=0.0)
FEET = [7, 8, 10, 11]                                                  # ankle L/R, foot L/R


def _feet_world(hd, fj):
    """Denormalised standard-272 (B,W,272) -> WORLD positions of joints fj (B,W,len(fj),3): heading = cumulative
    product of the per-frame rot6d deltas [2:8]; root XZ = cumsum of the velocity [0:2] rotated by the previous
    frame's inverse heading; canonical positions [8:74] rotated back and translated. Differentiable."""
    B, W, _ = hd.shape
    rel = rotation_6d_to_matrix(hd[..., 2:8].reshape(B * W, 6)).reshape(B, W, 3, 3)
    ghr = [rel[:, 0]]
    for t in range(1, W):
        ghr.append(rel[:, t] @ ghr[-1])
    inv = torch.stack(ghr, 1).transpose(-1, -2)                        # (B,W,3,3)
    z = torch.zeros_like(hd[..., 0])
    v = torch.stack([hd[..., 0], z, hd[..., 1]], -1)
    vw = torch.cat([v[:, :1], (inv[:, :-1] @ v[:, 1:, :, None]).squeeze(-1)], 1)
    root = torch.cumsum(vw, 1)
    pos = hd[..., 8:74].reshape(B, W, 22, 3)[:, :, fj]
    return (inv[:, :, None] @ pos[..., None]).squeeze(-1) + root[:, :, None]


def _kl(mu, logvar):
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()


def split_ae_loss(out, b, lam=None, norm=None, anchor_fk=None, fslide_gate="light", **_ignored):
    lam = {**DEFAULT_LAM, **(lam or {})}
    assert norm is not None and anchor_fk is not None, "split_ae_loss needs the Normalizer and an AnchorFK"
    L = {}
    hr, h = out["human_recon"], b["human"]
    L["h_recon"] = F.mse_loss(hr, h)
    L["h_root"] = F.mse_loss(hr[..., :8], h[..., :8])
    L["kl_h"] = _kl(out["mu_h"], out["logvar_h"])
    hw = b["hand_w"]                                                   # (B,) per-sequence hand quality weight
    he = ((out["hand_recon"] - b["hand"]) ** 2).flatten(1).mean(-1)
    L["h_hand"] = (he * hw).sum() / hw.sum().clamp(min=1e-6)
    om = b["omask"]
    den = om.sum().clamp(min=1.0)

    def omse(pred, gt):                                                # masked mean over active objects
        e = ((pred - gt) ** 2).flatten(2).mean(-1)
        return (e * om).sum() / den

    L["o_robj"] = omse(out["robj"], b["robj"])
    L["o_pg"] = omse(out["pobj"], b["pobj"])
    L["o_vel"] = omse(norm.do("pobj", out["pobj"]).diff(dim=-2), norm.do("pobj", b["pobj"]).diff(dim=-2))
    L["h_root_vel"] = F.mse_loss(hr[..., :8].diff(dim=-2), h[..., :8].diff(dim=-2))
    klo = (-0.5 * (1 + out["logvar_o"] - out["mu_o"].pow(2) - out["logvar_o"].exp())).mean(dim=(2, 3))
    L["kl_o"] = (klo * om).sum() / den
    # ---- decoder-finetune auxes (world-frame reintegration of the decoded human) ----
    _scale = 1.0 / float(np.mean(norm.h_std[8:74]))                    # metric -> ~normalised units
    if lam["fslide"] > 0:
        with torch.no_grad():
            pw_gt = _feet_world(norm.dh(h), FEET)                      # (B,W,4,3) GT world feet
        pw_pr = _feet_world(norm.dh(hr), FEET)
        gv = pw_gt[:, 1:] - pw_gt[:, :-1]; pv = pw_pr[:, 1:] - pw_pr[:, :-1]
        if fslide_gate == "pred":                                      # model's contact prediction OR GT stance
            if out.get("contact_logit") is None:
                raise RuntimeError("fslide_gate='pred' needs the contact head")
            pg = (torch.sigmoid(out["contact_logit"].detach()) > 0.5)[:, :-1]
            gy = pw_gt[..., 1]; floor = gy.amin(dim=1, keepdim=True)
            gg = (((gy[:, 1:] - floor) < 0.05) & (gv.norm(dim=-1) < 0.01))
            gate = (pg | gg).float().unsqueeze(-1)
            num = (gate * (_scale * (pv[..., [0, 2]] - gv[..., [0, 2]])) ** 2).sum()
        elif fslide_gate == "light":                                   # LIGHT foot_contact rule on the GT
            gate = ((gv ** 2).sum(-1) < 0.002).float().unsqueeze(-1)
            num = (gate * (_scale * pv) ** 2).sum()
        else:
            raise ValueError(f"fslide_gate must be 'light' or 'pred', got {fslide_gate!r}")
        L["h_fslide"] = num / gate.sum().clamp(min=1.0)
    else:
        L["h_fslide"] = torch.zeros((), device=hr.device)
    if lam["ctc"] > 0:
        if out.get("contact_logit") is None:
            raise RuntimeError("lambda_ctc > 0 needs the contact head")
        with torch.no_grad():
            cw = _feet_world(norm.dh(h), FEET)
            cv = (cw[:, 1:] - cw[:, :-1]).norm(dim=-1) * 30.0
            hth = torch.tensor([0.15, 0.15, 0.10, 0.10], device=cw.device, dtype=cw.dtype)
            cl = ((cv < 0.15) & (cw[:, :-1, :, 1] < hth)).float()
            cl = torch.cat([cl, cl[:, -1:]], 1)                        # (B,W,4)
        L["h_ctc"] = F.binary_cross_entropy_with_logits(out["contact_logit"], cl)
        L["h_ctc_pos"] = cl.mean().detach()
    else:
        L["h_ctc"] = torch.zeros((), device=hr.device)
    if lam["wvel"] > 0:
        all22 = list(range(22))
        with torch.no_grad():
            wg = _feet_world(norm.dh(h), all22)
        wp = _feet_world(norm.dh(hr), all22)
        L["h_wvel"] = F.mse_loss(_scale * (wp[:, 1:] - wp[:, :-1]), _scale * (wg[:, 1:] - wg[:, :-1]))
    else:
        L["h_wvel"] = torch.zeros((), device=hr.device)
    if lam["fkcons"] > 0:
        rd2 = norm.dh(hr)
        jfk = anchor_fk(rd2, jrest=b["jrest"], return_joints=True)[2]  # (B,W,22,3) canonical FK joints
        jpos = rd2[..., 8:74].reshape(rd2.shape[0], rd2.shape[1], 22, 3)
        L["h_fkcons"] = F.mse_loss(_scale * jfk, _scale * jpos)
    else:
        L["h_fkcons"] = torch.zeros((), device=hr.device)
    # ---- part-anchored combine ----
    parts = list(out["delta_parts"])
    assert out["delta"].shape[-2] == len(parts)
    allowed = torch.zeros(C.W_D, device=hr.device)
    allowed[parts + [C.W_D - 1]] = 1.0
    wgt = b["w"] * allowed
    zero = wgt.sum(-1, keepdim=True) == 0
    free = torch.zeros_like(wgt); free[..., -1] = 1.0
    wgt = torch.where(zero, free, wgt)
    wgt = wgt / wgt.sum(-1, keepdim=True).clamp(min=1e-6)              # GT routing target over allowed slots
    L["o_w"] = omse(out["w"].softmax(-1), wgt)
    dfull = torch.zeros(*out["delta"].shape[:3], C.N_PARTS, 3, device=hr.device)
    dfull[..., parts, :] = out["delta"]
    h_anch = norm.dh(hr)                                               # recon human -> anchors + window frame
    anchor_p, anchor_R = anchor_fk(h_anch, jrest=b["jrest"])
    Twf, off_wf = window_transform(h_anch)
    p_comb = combine_obj_pos_wf(norm.do("delta", dfull), norm.do("pobj", out["pobj"]),
                                out["w"], anchor_p, anchor_R, Twf, off_wf)
    L["o_trans"] = omse(p_comb, norm.do("pobj", b["pobj"]))
    de = ((out["delta"] - b["delta"][..., parts, :]) ** 2).sum(-1)     # (B,M,W,|parts|)
    num = de.sum((-1, -2))
    den_pf = torch.full_like(num, float(de.shape[-1] * de.shape[-2] * 3))
    L["o_delta"] = ((num / den_pf) * om).sum() / den
    total = (L["h_recon"] + lam["root"] * L["h_root"] + lam["kl_h"] * L["kl_h"]
             + lam["robj"] * L["o_robj"] + lam["pg"] * L["o_pg"]
             + lam["vel"] * L["o_vel"] + lam["kl_o"] * L["kl_o"]
             + lam["delta"] * L["o_delta"] + lam["trans"] * L["o_trans"]
             + lam["w"] * L["o_w"]
             + lam["hand"] * L["h_hand"]
             + lam["root_vel"] * L["h_root_vel"]
             + lam["fslide"] * L["h_fslide"]
             + lam["wvel"] * L["h_wvel"]
             + lam["fkcons"] * L["h_fkcons"]
             + lam["ctc"] * L["h_ctc"])
    L["total"] = total
    return total, {k: float(v) for k, v in L.items()}
