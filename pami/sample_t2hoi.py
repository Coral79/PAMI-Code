"""PamiGen sampling: rectified-flow Euler ODE in the PamiVAE latent space -> decode (contact-tuned decoder)
-> part-anchored object combine -> gen npz.

Conditions come from test pieces (caption, object BPS, body shape, length), so every sample has a GT twin
(the GT latents decoded through the same decoder) for side-by-side rendering / evaluation.
RF Euler, tau 1 -> 0 over --steps; classifier-free guidance on the CAPTION only (v = v_u + w (v_c - v_u));
object shape and body shape stay identical in both branches.

Output <out>/gen_<i>_<seq>.npz: human_272 (W,272), robj (M,W,6), pobj (M,W,3) [DENORMALISED, piece-first frame],
hand (W,270), contact_logit (W,4), obj_candidates / obj_free / obj_weights (anchor diagnostics) + gt_* twins.
CLI: generate.py (repo root).
"""
import os, json, argparse
import numpy as np
import torch
from pami.data.t2hoi_dataset import T2HOIPieceDataset, TextBank, collate_t2hoi
from pami.nets.t2hoi_denoiser import T2HOIDenoiser
from pami.nets.objframe_torch import window_transform, combine_obj_pos_wf
from pami.ckpt_io import (load_vae_for_gen, decode_human, h_denorm, check_latents_dialect, resolve_ckpt)
from pami import config as C


def to_dev(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


def load_t2hoi(ckpt_path, dev, use_ema=True):
    """THE inference-side entry point for a PamiGen checkpoint. Returns (model.eval(), ck, arch).
    use_ema: load the EMA weights (what the paper samples use)."""
    ckpt_path = resolve_ckpt(ckpt_path)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ck["arch"]
    assert arch.get("arch_type") == "t2hoi", f"{ckpt_path} is not a PamiGen checkpoint"
    for k, v in (("token_layout", "split"), ("use_action", False), ("explicit_root", False), ("obj_cond", False),
                 ("first_cond", False), ("vae_is_hoi_tae", False)):
        if arch.get(k, v) != v:
            raise SystemExit(f"load_t2hoi: {ckpt_path} arch {k}={arch.get(k)!r} not supported (needs {v!r})")
    if arch.get("human_shape") and arch.get("human_shape_src", "jrest") != "jrest":
        raise SystemExit(f"load_t2hoi: human_shape_src={arch.get('human_shape_src')!r} not supported (jrest only)")
    m = T2HOIDenoiser(d=arch["dmodel"], layers=arch["layers"], heads=arch["heads"], ff=arch["ff"],
                      dropout=arch["dropout"], n_obj=arch.get("n_obj", 1), n_hp=arch.get("n_hp", 8),
                      human_shape=bool(arch.get("human_shape", False)))
    sd = ck["model"]
    if use_ema and "ema" in ck:
        sd = {k[len("module."):]: v for k, v in ck["ema"].items() if k.startswith("module.")}
        print(f"[t2hoi] using EMA weights (it={ck.get('it')})")
    m.load_state_dict(sd)
    m.to(dev).eval()
    return m, ck, arch


_JREST_CACHE = {}


def seq_jrest(seq, dev):
    """(1,22,3) per-sequence rest skeleton (canon_anchors.npz) for the AnchorFK combine."""
    if seq not in _JREST_CACHE:
        an = np.load(f"{C.PKG}/output/{seq}/canon_anchors.npz", allow_pickle=True)
        _JREST_CACHE[seq] = an["jrest"].astype(np.float32)
    return torch.from_numpy(_JREST_CACHE[seq])[None].to(dev)


def decode_object_split(vae, nrm, afk, seq, zo_l, hrec, dev, return_diag=False, jrest=None):
    """Self-sufficient part-anchored object decode. zo_l (1,M,Tr,L) VAE object latent; hrec (1,W,272) normalised
    decoded human. robj / p_G / delta / w are ALL predicted; anchors are FK'd from the decoded human with the
    sequence's rest skeleton; combine in the window-first frame. Returns numpy (robj (M,W,6) physical,
    pos (M,W,3) combined physical) [+ diag dict: obj_candidates (M,W,6,3), obj_free (M,W,3), obj_weights (M,W,7)]."""
    robj, pG, delta, w = vae._decode_object(zo_l, return_delta=True)   # w already carries the routing mask
    M, W = robj.shape[1], robj.shape[2]
    dfull = torch.zeros(1, M, W, C.N_PARTS, 3, device=dev)
    dfull[..., list(vae.delta_parts), :] = delta
    jrest = seq_jrest(seq, dev) if jrest is None else jrest
    h272_dn = nrm.dh(hrec)
    anchor_p, anchor_R = afk(h272_dn, jrest=jrest)
    Twf, off_wf = window_transform(h272_dn)
    delta_c = nrm.do("delta", dfull)
    pG_wf = nrm.do("pobj", pG)
    pos = combine_obj_pos_wf(delta_c, pG_wf, w, anchor_p, anchor_R, Twf, off_wf)
    robj_phys = nrm.do("robj", robj)
    out = (robj_phys[0].cpu().numpy(), pos[0].cpu().numpy())
    if not return_diag:
        return out
    cand = anchor_p[:, None] + torch.einsum("bwpij,bmwpj->bmwpi", anchor_R, delta_c)
    cand = torch.einsum("bwij,bmwpj->bmwpi", Twf, cand) + off_wf[:, None, :, None]
    diag = {"obj_candidates": cand[0].cpu().numpy(), "obj_free": pG_wf[0].cpu().numpy(),
            "obj_weights": w[0].softmax(-1).cpu().numpy()}
    return (*out, diag)


@torch.no_grad()
def sample(m, b, bank, arch, dev, steps=50, cfg=2.5, seed=None):
    """b: collated condition batch (zh/zo give the shapes only). Returns (zh, zo) in NORMALISED diffusion space,
    (B,Tr,8,16) / (B,Tr,M,16). One seeded noise tensor is drawn for the WHOLE batch."""
    B, Tr = b["zh"].shape[:2]
    g = torch.Generator(device=dev)
    if seed is not None: g.manual_seed(seed)
    zh = torch.randn(B, Tr, m.n_hp, 16, device=dev, generator=g)
    zo = torch.randn(B, Tr, m.m, 16, device=dev, generator=g)
    cap_pooled, cap_tok, cap_msk = bank.captions(b["cap_tid"].cpu(), dev)
    null_tid = torch.zeros_like(b["cap_tid"].cpu())
    ncap_pooled, ncap_tok, ncap_msk = bank.captions(null_tid, dev)
    sm, om, bps = b["step_mask"], b["omask"], b["bps"]
    jb = b.get("jbone_override", b.get("jbone"))
    jb = jb.to(dev) if (jb is not None and m.human_shape) else None
    taus = torch.linspace(1.0, 0.0, steps + 1, device=dev)
    for tau, tau_next in zip(taus[:-1], taus[1:]):
        t = torch.full((B,), float(tau), device=dev)
        vh_c, vo_c = m(zh, zo, t, om, sm, bps, cap_pooled, cap_tok, cap_msk, jbone=jb)
        if cfg != 1.0:
            vh_u, vo_u = m(zh, zo, t, om, sm, bps, ncap_pooled, ncap_tok, ncap_msk, jbone=jb)
            vh = vh_u + cfg * (vh_c - vh_u)
            vo = vo_u + cfg * (vo_c - vo_u)
        else:
            vh, vo = vh_c, vo_c
        dt = float(tau - tau_next)
        zh = zh - vh * dt
        zo = zo - vo * dt
    return zh, zo


def _existing(path, fallback):
    """A path recorded inside a checkpoint (its location at training time) or the configured fallback."""
    return path if path and os.path.exists(path) else fallback


class Generator:
    """PamiGen + PamiVAE decoder bundle: generate(items) -> per-piece dicts of numpy arrays."""
    def __init__(self, ckpt=C.CKPT_GEN, vae=C.CKPT_VAE_DEC, dev=None, use_ema=True, data_dir=None):
        from pami.nets.anchor_fk import AnchorFK
        self.dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
        self.m, self.ck, self.arch = load_t2hoi(ckpt, self.dev, use_ema=use_ema)
        self.vae, self.nrm = load_vae_for_gen(self.arch, vae_ckpt=vae, dev=self.dev)
        self.data_dir = data_dir or _existing(self.arch.get("data_dir"), C.T2HOI_DATA)
        self.bank = TextBank(self.data_dir)
        self.afk = AnchorFK(self.dev)
        self.sh = torch.tensor(self.arch["stats_h"], device=self.dev)
        self.so = torch.tensor(self.arch["stats_o"], device=self.dev)
        hs = np.load(C.STATS_HAND)
        self.hand_ms = (hs["wlocalrot_mean"].astype(np.float32), np.maximum(hs["wlocalrot_std"].astype(np.float32), 1e-6))

    def dataset(self, split_part="test"):
        split = _existing(self.arch.get("split"), C.SPLIT)              # the checkpoint records its training-time path
        ds = T2HOIPieceDataset(self.arch["cap"], self.arch["vae_tag"], seqs=json.load(open(split))[split_part],
                               data_dir=self.data_dir, n_obj=self.m.m, random_caption=False)
        check_latents_dialect(ds, "Generator.dataset")
        # the archive is encoded by every user (encode_latents.py); another GPU model changes the last digits
        assert np.allclose(self.arch["stats_h"], ds.stats_h, rtol=1e-3, atol=1e-3) and \
            np.allclose(self.arch["stats_o"], ds.stats_o, rtol=1e-3, atol=1e-3), \
            "latents npz stats differ from the checkpoint's training-time stats — regenerate or retrain"
        return ds

    @torch.no_grad()
    def decode(self, zh_n, zo_n, seq, Tr, M):
        """Diffusion-space latents (1,Tr,8,16)/(1,Tr,M,16) -> dict of numpy arrays (denormalised)."""
        zh_l = h_denorm(zh_n, self.sh)
        zo_l = zo_n * self.so[1] + self.so[0]
        dh = decode_human(self.vae, zh_l, 1, Tr)
        hrec = dh["human_recon"]
        W = hrec.shape[1]
        if M:
            r_phys, p_comb, dg = decode_object_split(self.vae, self.nrm, self.afk, seq, zo_l[:, :, :M].permute(0, 2, 1, 3),
                                                     hrec, self.dev, return_diag=True)
        else:                                                          # piece without an active object: empty streams
            r_phys, p_comb = np.zeros((0, W, 6), np.float32), np.zeros((0, W, 3), np.float32)
            dg = {"obj_weights": np.zeros((0, W, 7), np.float32)}
        out = dict(human_272=self.nrm.dh(hrec)[0].cpu().numpy(), robj=r_phys, pobj=p_comb,
                   hand=dh["hand_recon"][0].cpu().numpy() * self.hand_ms[1] + self.hand_ms[0], **dg)
        if dh["contact_logit"] is not None:
            out["contact_logit"] = dh["contact_logit"][0].cpu().numpy().astype(np.float32)
        return out

    @torch.no_grad()
    def generate(self, items, steps=50, cfg=2.5, seed=0):
        """items: list of dataset items (same order = same batch = same seeded noise as the paper draws).
        Returns a list of payload dicts (one per item) with the generated arrays + the GT twin (gt_*)."""
        batch_all = to_dev(collate_t2hoi(items), self.dev)
        zh_all, zo_all = sample(self.m, batch_all, self.bank, self.arch, self.dev, steps=steps, cfg=cfg, seed=seed)
        outs = []
        for bi, it in enumerate(items):
            b = to_dev(collate_t2hoi([it]), self.dev)
            seq, f0, f1 = b["seq"][0], int(b["f0"][0]), int(b["f1"][0])
            Tr = b["zh"].shape[1]; M = int(b["omask"].sum())
            gen = self.decode(zh_all[bi:bi + 1, :Tr], zo_all[bi:bi + 1, :Tr], seq, Tr, M)
            gt = self.decode(b["zh"], b["zo"], seq, Tr, M)
            payload = dict(seq=seq, f0=f0, f1=f1, cap_tid=int(b["cap_tid"][0]), cfg=cfg, steps=steps,
                           obj_frame="window_first", dataset="interact", data_dir=self.data_dir,
                           hand_mode="wlocalrot", human_shape_cond=bool(self.arch.get("human_shape")),
                           human_shape_src="jrest", delta_parts=np.asarray(self.vae.delta_parts, dtype=np.int64),
                           contact_joints=np.asarray([7, 8, 10, 11], dtype=np.int64))
            payload.update(gen)
            payload.update({f"gt_{k}": v for k, v in gt.items()})
            outs.append(payload)
        return outs


def main(argv=None):
    ap = argparse.ArgumentParser(description="PamiGen sampling on test-split pieces (generate.py)")
    ap.add_argument("--ckpt", default=C.CKPT_GEN, help="PamiGen checkpoint")
    ap.add_argument("--vae", default=C.CKPT_VAE_DEC, help="PamiVAE checkpoint used for DECODING (default: contact-tuned decoder)")
    ap.add_argument("--n", type=int, default=4, help="number of random pieces when --pieces is not given")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_part", default="test", choices=["val", "train", "test"])
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--pieces", type=int, nargs="*", default=None, help="explicit piece indices into the split")
    ap.add_argument("--data_dir", default=None, help="latents/texts dir (default: the checkpoint's)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    G = Generator(a.ckpt, a.vae, use_ema=not a.no_ema, data_dir=a.data_dir)
    ds = G.dataset(a.split_part)
    idx = a.pieces if a.pieces else list(np.random.RandomState(a.seed).choice(len(ds), a.n, replace=False))
    items = [ds[int(i)] for i in idx]
    out_dir = a.out or os.path.join("outputs", f"{os.path.basename(a.ckpt).replace('.pt', '')}_cfg{a.cfg}")
    os.makedirs(out_dir, exist_ok=True)
    for i, payload in zip(idx, G.generate(items, steps=a.steps, cfg=a.cfg, seed=a.seed)):
        fp = f"{out_dir}/gen_{int(i)}_{payload['seq']}.npz"
        np.savez(fp, ckpt=a.ckpt, vae=a.vae, **payload)
        print(f"[{int(i)}] {payload['seq']}[{payload['f0']}:{payload['f1']}] -> {fp}")
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
