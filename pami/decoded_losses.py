"""Decoded-space object losses for PamiGen training: the predicted clean latent z0_hat = z_t - t v_hat is
decoded through the FROZEN PamiVAE and compared, in physical window-first metres, with the TWIN target (the
GT latent decoded through the same frozen decoder), so the gradient measures diffusion error only.
  L_pg_pos / L_pg_vel : Huber on the free-head p_G position / first difference
  L_comb              : Huber on the fully-PREDICTED part-anchored combine (anchors + window transform FK'd
                        from the decoded predicted human, predicted delta / w / p_G)
  L_agree             : on frames whose GT routing has mass on an allowed anchor, pull p_G toward the predicted
                        candidates under de-FREE renormalised predicted routing weights
Expects fp32 inputs (call under autocast(enabled=False)).
"""
import torch
import torch.nn.functional as F
from pami.ckpt_io import decode_human
from pami.nets.objframe_torch import window_transform, combine_obj_pos_wf
from pami.vendor.eval_trans import rotation_6d_to_matrix


def _masked_mean_bs(x, mask):
    """x, mask (B,M,W) -> per-sample masked mean (B,), and a has-any-support flag (B,)."""
    num = (x * mask).sum((1, 2))
    den = mask.sum((1, 2))
    return num / den.clamp(min=1), (den > 0).float()


def _huber3(a, b, delta):
    return F.huber_loss(a, b, reduction="none", delta=delta).mean(-1)


def decoded_obj_losses(vae, afk, nrm, zh_hat0, zo_hat0, zh0_dn, zo0_dn, fm, om,
                       wgt=None, jrest=None, huber_delta=0.1, want_pg=True, want_comb=False, want_agree=False):
    """Per-sample decoded-space object losses on an (already t-band-gated) sub-batch.
    zh_hat0 (B,Tr,P,16) / zo_hat0 (B,Tr,M,16): predicted clean latents, VAE space; zh0_dn / zo0_dn: GT latents
    (twin targets, decoded under no_grad); fm (B,W') frame mask; om (B,M) object mask; wgt (B,W,M,7) GT routing
    multi-hot (agree gate); jrest (B,22,3) per-seq rest skeleton.
    Returns (terms: li_* (B,) + has_* (B,), diag: float monitors)."""
    B, Tr, P = zh_hat0.shape[:3]
    M = zo_hat0.shape[2]
    dev = zh_hat0.device
    parts = list(vae.delta_parts)
    hrec = decode_human(vae, zh_hat0, B, Tr)["human_recon"]
    robj, pG, delta, w = vae._decode_object(zo_hat0.permute(0, 2, 1, 3), return_delta=True)
    W = pG.shape[2]
    with torch.no_grad():
        hrec_t = decode_human(vae, zh0_dn, B, Tr)["human_recon"]
        robj_t, pG_t, delta_t, w_t = vae._decode_object(zo0_dn.permute(0, 2, 1, 3), return_delta=True)
    fm = fm[:, :W]
    m3 = fm[:, None, :] * om[:, :, None]                               # (B,M,W) active object-frames
    pGp = nrm.do("pobj", pG)
    with torch.no_grad():
        pGt = nrm.do("pobj", pG_t)
    terms, diag = {}, {}
    if want_pg:
        li, has = _masked_mean_bs(_huber3(pGp, pGt, huber_delta), m3)
        terms["li_pg_pos"], terms["has_pg_pos"] = li, has
        m3v = m3[:, :, 1:] * m3[:, :, :-1]
        li_v, has_v = _masked_mean_bs(
            _huber3(pGp[:, :, 1:] - pGp[:, :, :-1], pGt[:, :, 1:] - pGt[:, :, :-1], huber_delta), m3v)
        terms["li_pg_vel"], terms["has_pg_vel"] = li_v, has_v
        with torch.no_grad():
            e = torch.linalg.norm(pGp - pGt, dim=-1)
            diag["dec_pg_cm"] = float((e * m3).sum() / m3.sum().clamp(min=1) * 100.0)
    if want_comb or want_agree:
        def _chain(hrec_n, delta_n, pG_phys, need_cand):
            h_dn = nrm.dh(hrec_n)
            jr = None
            if jrest is not None:
                nz = jrest.abs().sum((1, 2)) > 0
                if bool(nz.all()):
                    jr = jrest
                else:
                    assert not bool(nz.any()), "mixed zero/nonzero jrest in one sub-batch"
            anchor_p, anchor_R = afk(h_dn, jrest=jr)
            Twf, off = window_transform(h_dn)
            dfull = torch.zeros(B, M, W, 6, 3, device=dev, dtype=pG_phys.dtype)
            dfull[..., parts, :] = delta_n
            dc = nrm.do("delta", dfull)
            cand = None
            if need_cand:
                fk = anchor_p[:, None] + torch.einsum("bwpij,bmwpj->bmwpi", anchor_R, dc)
                cand = torch.einsum("bwij,bmwpj->bmwpi", Twf, fk) + off[:, None, :, None]
            return anchor_p, anchor_R, Twf, off, dc, cand

        anchor_p, anchor_R, Twf, off, dc, cand = _chain(hrec, delta, pGp, need_cand=True)
        p_comb = combine_obj_pos_wf(dc, pGp, w, anchor_p, anchor_R, Twf, off)
        with torch.no_grad():
            ap_t, aR_t, T_t, off_t, dc_t, _ = _chain(hrec_t, delta_t, pGt, need_cand=False)
            p_comb_t = combine_obj_pos_wf(dc_t, pGt, w_t, ap_t, aR_t, T_t, off_t)
        if want_comb:
            li_c, has_c = _masked_mean_bs(_huber3(p_comb, p_comb_t, huber_delta), m3)
            terms["li_comb"], terms["has_comb"] = li_c, has_c
            with torch.no_grad():
                e = torch.linalg.norm(p_comb - p_comb_t, dim=-1)
                diag["dec_comb_cm"] = float((e * m3).sum() / m3.sum().clamp(min=1) * 100.0)
        ws = w.softmax(-1)
        with torch.no_grad():
            diag["dec_w_free"] = float((ws[..., -1] * m3).sum() / m3.sum().clamp(min=1))
        if want_agree:
            assert wgt is not None, "L_agree needs the dataset's GT routing (want_wgt=True)"
            wb = ws[..., parts]
            wb = wb / wb.sum(-1, keepdim=True).clamp(min=1e-6)
            cand_sel = cand[..., parts, :]
            hub = _huber3(pGp[..., None, :].expand_as(cand_sel), cand_sel, huber_delta)
            la = (hub * wb).sum(-1)
            gr = (wgt[:, :W, :, :][..., parts].sum(-1) > 0).float().permute(0, 2, 1)
            m_a = m3 * gr
            li_a, has_a = _masked_mean_bs(la, m_a)
            terms["li_agree"], terms["has_agree"] = li_a, has_a
            with torch.no_grad():
                gap = torch.linalg.norm(pGp - (wb[..., None] * cand_sel).sum(-2), dim=-1)
                diag["dec_gap_cm"] = float((gap * m_a).sum() / m_a.sum().clamp(min=1) * 100.0)
                diag["dec_agree_frames"] = float(m_a.sum() / m3.sum().clamp(min=1))
    with torch.no_grad():                                              # rotation monitor
        Ra = rotation_6d_to_matrix(nrm.do("robj", robj))
        Rb = rotation_6d_to_matrix(nrm.do("robj", robj_t))
        tr = (Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1)
        ang = torch.acos(((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)) * (180.0 / torch.pi)
        diag["dec_rot_deg"] = float((ang * m3).sum() / m3.sum().clamp(min=1))
    return terms, diag
