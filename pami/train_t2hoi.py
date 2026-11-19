"""PamiGen training: rectified-flow latent diffusion over the frozen-PamiVAE latent grid (8 human part tokens
+ 1 object token per latent step), conditioned on the CLIP caption (pooled -> AdaLN, tokens -> cross-attention,
10% CFG dropout), the object dual-BPS and the body shape (rest bones). CLI wrapper: train_gen.py.

Loss = per-sample masked v-MSE (human and object token groups normalised separately, L = mean_h + lambda_obj
mean_o) + decoded-space object losses through the frozen VAE (pami/decoded_losses.py: p_G position / velocity,
fully-predicted combine, p_G-candidate agreement; Huber in metres, t in (dec_t_lo, dec_t_hi), curriculum
ramps). EMA 0.9999; LIGHT-aligned AdamW + warmup/cosine LR; bf16 autocast for the denoiser forward.
Paper run: --iters 400000 (checkpoint at 240k used) with the lambdas in train_gen.py's defaults.
"""
import sys, os, json, argparse, time, math, random, hashlib
from contextlib import nullcontext
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from pami.data.t2hoi_dataset import (T2HOIPieceDataset, TextBank, collate_t2hoi, shape_standardization_stats)
from pami.nets.t2hoi_denoiser import T2HOIDenoiser
from pami import config as C


def to_dev(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


RESUME_KEYS = ("cap", "vae_tag", "dmodel", "layers", "heads", "ff", "lr", "warmup", "lr_cosine_steps", "lr_min",
               "weight_decay", "adam_eps", "adam_beta2", "dropout", "lambda_obj", "cfg_drop", "seed",
               "vae_refined", "latent_dialect", "stats_mode", "stats_h", "stats_o", "stats_population", "split_hash",
               "token_layout", "n_hp", "n_obj", "lambda_pg_pos", "lambda_pg_vel", "lambda_comb", "lambda_agree",
               "dec_t_lo", "dec_t_hi", "dec_huber", "dec_warm_pg", "dec_warm_de",
               "human_shape", "human_shape_src", "shape_in_dim", "shape_parents", "jb_stats_hash", "jrest_mean_hash")


def check_resume_arch(ck_arch, arch):
    for key in RESUME_KEYS:
        if ck_arch.get(key) != arch.get(key):
            raise SystemExit(f"resume arch mismatch: {key} ckpt={ck_arch.get(key)} vs {arch.get(key)}")


def rf_step(model, b, bank, dev, lam_obj, cfg_drop, rng=None, vae=None, stats_h=None, stats_o=None,
            lam_pg_pos=0.0, lam_pg_vel=0.0, lam_comb=0.0, lam_agree=0.0,
            dec_t_lo=0.1, dec_t_hi=0.5, dec_huber=0.1, afk=None, dec_nrm=None, **_ignored):
    """One rectified-flow loss evaluation on batch b. rng: optional torch.Generator (reproducible val t/noise)."""
    zh0, zo0 = b["zh"], b["zo"]
    B, Tr = zh0.shape[:2]
    sm, om = b["step_mask"], b["omask"]
    t = torch.rand(B, device=dev, generator=rng)
    te = t[:, None, None, None]
    eh = torch.randn(zh0.shape, device=dev, generator=rng)
    eo = torch.randn(zo0.shape, device=dev, generator=rng)
    zt_h = (1 - te) * zh0 + te * eh
    zt_o = (1 - te) * zo0 + te * eo
    cap_tid = b["cap_tid"].clone().cpu()
    if cfg_drop > 0:
        drop = torch.rand(B) < cfg_drop
        cap_tid[drop] = 0                                              # null caption row
    cap_pooled, cap_tok, cap_msk = bank.captions(cap_tid, dev)
    vh, vo = model(zt_h, zt_o, t, om, sm, b["bps"], cap_pooled, cap_tok, cap_msk, jbone=b["jbone"].to(dev))
    th, to = eh - zh0, eo - zo0                                        # RF velocity target
    smh = sm[:, :, None, None]
    hden = sm.sum(1)
    lh = ((vh - th) ** 2 * smh).sum((1, 2, 3)) / (hden * vh.shape[2] * 16).clamp(min=1)   # (B,)
    L_h = lh.mean()
    smo = sm[:, :, None, None] * om[:, None, :, None]
    lo = ((vo - to) ** 2 * smo).sum((1, 2, 3)) / (smo.sum((1, 2, 3)) * 16).clamp(min=1)
    has_o = (om.sum(1) > 0).float()
    L_o = (lo * has_o).sum() / has_o.sum().clamp(min=1)
    loss = L_h + lam_obj * L_o
    comp = {"v_h": float(L_h), "v_o": float(L_o)}
    if lam_pg_pos > 0 or lam_pg_vel > 0 or lam_comb > 0 or lam_agree > 0:
        assert vae is not None and dec_nrm is not None and stats_o is not None and afk is not None
        g_band = (t > dec_t_lo) & (t < dec_t_hi)
        comp["dec_band_frac"] = float(g_band.float().mean())
        if bool(g_band.any()):
            sub = g_band.nonzero(as_tuple=True)[0]
            with torch.autocast("cuda", enabled=False):                # fp32: FK / einsum chains
                from pami.ckpt_io import h_denorm as _hd
                from pami.decoded_losses import decoded_obj_losses
                te_s = te[sub].float()
                zh_hat0 = _hd(zt_h[sub].float() - te_s * vh[sub].float(), stats_h)
                zo_hat0 = _hd(zt_o[sub].float() - te_s * vo[sub].float(), stats_o)
                zh_dn = _hd(zh0[sub].float(), stats_h)
                zo_dn = _hd(zo0[sub].float(), stats_o)
                fm_s = sm[sub].repeat_interleave(4, dim=1).float()
                wgt_s = b["wgt"][sub].float() if lam_agree > 0 else None
                jr_s = b["jrest"][sub].float() if "jrest" in b else None
                dterms, ddiag = decoded_obj_losses(
                    vae, afk, dec_nrm, zh_hat0, zo_hat0, zh_dn, zo_dn, fm_s, om[sub].float(),
                    wgt=wgt_s, jrest=jr_s, huber_delta=dec_huber,
                    want_pg=(lam_pg_pos > 0 or lam_pg_vel > 0), want_comb=lam_comb > 0, want_agree=lam_agree > 0)
            for name, lam in (("pg_pos", lam_pg_pos), ("pg_vel", lam_pg_vel), ("comb", lam_comb), ("agree", lam_agree)):
                if lam <= 0:
                    continue
                li, has = dterms[f"li_{name}"], dterms[f"has_{name}"]
                Lx = (li * has).sum() / has.sum().clamp(min=1)
                loss = loss + lam * Lx
                comp[f"dec_{name}"] = float(Lx)
            comp.update(ddiag)
            if "li_pg_pos" in dterms:                                  # t-bucket diagnostic
                t_s, mid = t[sub], 0.5 * (dec_t_lo + dec_t_hi)
                for tag, msk in (("tlow", t_s < mid), ("thigh", t_s >= mid)):
                    mk = msk.float() * dterms["has_pg_pos"]
                    if float(mk.sum()) > 0:
                        comp[f"dec_pg_pos_{tag}"] = float((dterms["li_pg_pos"] * mk).sum() / mk.sum())
    comp["total"] = float(loss)
    return loss, comp


def build_parser():
    ap = argparse.ArgumentParser(description="PamiGen training")
    ap.add_argument("--cap", type=int, default=300, choices=[300, 448, 600])
    ap.add_argument("--vae_tag", type=str, default=C.VAE_TAG,
                    help="latents_cap<cap>_<tag>.npz in --data_dir")
    ap.add_argument("--vae", type=str, default=None, help="PamiVAE checkpoint for the decoded losses "
                    "(default: the one recorded in the latents archive, resolved via PAMI_CKPT_DIR)")
    ap.add_argument("--data_dir", type=str, default=C.T2HOI_DATA)
    ap.add_argument("--split", type=str, default=C.SPLIT)
    ap.add_argument("--iters", type=int, default=400000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--adam_eps", type=float, default=1e-6)
    ap.add_argument("--adam_beta2", type=float, default=0.999)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--amp", choices=["off", "bf16"], default="bf16")
    ap.add_argument("--warmup", type=int, default=2500, help="linear warmup steps (start factor 0.05)")
    ap.add_argument("--lr_cosine_steps", type=int, default=350000)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--dmodel", type=int, default=512)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lambda_obj", type=float, default=1.0)
    ap.add_argument("--cfg_drop", type=float, default=0.1)
    ap.add_argument("--lambda_pg_pos", type=float, default=5.0)
    ap.add_argument("--lambda_pg_vel", type=float, default=5.0)
    ap.add_argument("--lambda_comb", type=float, default=5.0)
    ap.add_argument("--lambda_agree", type=float, default=2.5)
    ap.add_argument("--dec_t_lo", type=float, default=0.1)
    ap.add_argument("--dec_t_hi", type=float, default=0.5)
    ap.add_argument("--dec_huber", type=float, default=0.1)
    ap.add_argument("--dec_warm_pg", type=int, default=10000, help="iterations to ramp lambda_pg_* from 0 to full")
    ap.add_argument("--dec_warm_de", type=int, nargs=2, default=[10000, 30000], help="ramp window of lambda_comb/agree")
    ap.add_argument("--eval_every", type=int, default=4000)
    ap.add_argument("--save_every", type=int, default=20000)
    ap.add_argument("--out", type=str, default=os.path.join("outputs", "gen"))
    ap.add_argument("--run_name", type=str, default="pamigen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_pieces", type=int, default=None, help="debug: cap the number of training pieces")
    ap.add_argument("--resume_full", type=str, default=None)
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = a.data_dir
    with np.load(C.latents_file(data_dir, a.cap, a.vae_tag), allow_pickle=True) as z:
        lat_dialect = str(z["latent_dialect"]) if "latent_dialect" in z else ("post_st" if bool(z.get("refined", False)) else "plain")
        stats_pop = str(z["stats_population"]) if "stats_population" in z else "all"
        zrepr = (str(z["repr_obj_frame"]) if "repr_obj_frame" in z else "window_first",
                 str(z["repr_human_root"]) if "repr_human_root" in z else "delta")
    if lat_dialect != "post_st" or zrepr != ("window_first", "wf_abs"):
        raise SystemExit(f"latents archive dialect {lat_dialect}/{zrepr} unsupported (need post_st, window_first, wf_abs)")
    if not os.path.exists(a.split):
        raise FileNotFoundError(f"training split not found: {a.split}")
    sp = json.load(open(a.split))
    train_seqs, val_seqs = sp.get("train"), sp.get("val")
    want_wgt = a.lambda_comb > 0 or a.lambda_agree > 0
    ds = T2HOIPieceDataset(a.cap, a.vae_tag, seqs=train_seqs, data_dir=data_dir, n_obj=1, want_wgt=want_wgt)
    n_hp = int(ds[0]["zh"].shape[1])
    if a.max_pieces: ds.idx = ds.idx[:a.max_pieces]
    bank = TextBank(data_dir)
    print(f"dataset: {len(ds)} pieces (cap {a.cap}) | run={a.run_name} | texts {bank.pooled.shape[0]}")
    stats_h = torch.tensor(ds.stats_h, device=dev)
    stats_o = torch.tensor(ds.stats_o, device=dev)
    m = T2HOIDenoiser(d=a.dmodel, layers=a.layers, heads=a.heads, ff=a.ff, dropout=a.dropout,
                      n_obj=1, n_hp=n_hp, human_shape=True).to(dev)
    # body-shape condition standardised on the TRAIN split only; stats live in the model buffers / ckpt
    tr_seqs = sorted({ds.seg_seq[k] for k in ds.idx})
    jb = np.stack([ds.shape_feat(s_) for s_ in tr_seqs])
    jb_mean, jb_std = shape_standardization_stats(jb)
    m.jb_mean.copy_(torch.tensor(jb_mean, dtype=torch.float32, device=dev))
    m.jb_std.copy_(torch.tensor(jb_std, dtype=torch.float32, device=dev))
    jb_hash = hashlib.sha1(np.concatenate([jb_mean, jb_std]).astype(np.float32).tobytes()).hexdigest()[:12]
    raws = [ds._jrest_raw(s_) for s_ in tr_seqs]
    shape_parents = raws[0][1].tolist()
    assert all(r[1].tolist() == shape_parents for r in raws), "mixed skeleton topologies across the train split"
    jrest_mean = np.mean([r[0] for r in raws], axis=0).astype(np.float32).tolist()
    jrest_mean_hash = hashlib.sha1(np.asarray(jrest_mean, np.float32).tobytes()).hexdigest()[:12]
    print(f"[human_shape] bone cond on {len(jb)} train subjects | leg-bone std {float(jb.std(0)[[0,3,9,18]].mean()*100):.2f} cm")
    npar = sum(p.numel() for p in m.parameters()) / 1e6
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if (a.amp == "bf16" and dev == "cuda") else nullcontext
    print(f"T2HOIDenoiser params: {npar:.1f}M | amp={a.amp}")
    ema = AveragedModel(m, multi_avg_fn=get_ema_multi_avg_fn(a.ema)) if a.ema > 0 else None
    if ema:
        for p in ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.weight_decay,
                            betas=(0.9, a.adam_beta2), eps=a.adam_eps, fused=(dev == "cuda"))
    ARCH = {"arch_type": "t2hoi", "human_shape": True, "human_shape_src": "jrest", "shape_in_dim": int(m.jb_mean.numel()),
            "shape_parents": shape_parents, "jb_stats_hash": jb_hash, "jrest_mean": jrest_mean, "jrest_mean_hash": jrest_mean_hash,
            "cap": a.cap, "vae_tag": a.vae_tag, "vae_refined": ds.refined, "vae_is_hoi_tae": False,
            "vae_ckpt": ds.vae_ckpt, "dataset": "interact", "token_layout": "split", "n_obj": 1, "use_action": False,
            "n_hp": n_hp, "human_stats": "interact", "data_dir": data_dir, "split": a.split,
            "obj_cond": False, "first_cond": False, "explicit_root": False, "root_dim": 0,
            "dmodel": a.dmodel, "layers": a.layers, "heads": a.heads, "ff": a.ff, "dropout": a.dropout,
            "lr": a.lr, "warmup": a.warmup, "lr_cosine_steps": a.lr_cosine_steps, "lr_min": a.lr_min,
            "weight_decay": a.weight_decay, "adam_eps": a.adam_eps, "adam_beta2": a.adam_beta2, "seed": a.seed,
            "lambda_obj": a.lambda_obj, "cfg_drop": a.cfg_drop,
            "stats_h": np.asarray(ds.stats_h).tolist(), "stats_o": np.asarray(ds.stats_o).tolist(), "stats_mode": "per_token",
            "lambda_pg_pos": a.lambda_pg_pos, "lambda_pg_vel": a.lambda_pg_vel, "lambda_comb": a.lambda_comb,
            "lambda_agree": a.lambda_agree, "dec_t_lo": a.dec_t_lo, "dec_t_hi": a.dec_t_hi, "dec_huber": a.dec_huber,
            "dec_warm_pg": a.dec_warm_pg, "dec_warm_de": list(a.dec_warm_de),
            "latent_dialect": lat_dialect, "scene_st_at_decode": False, "stats_population": stats_pop,
            "split_hash": hashlib.sha1(open(a.split, "rb").read()).hexdigest()[:12]}
    need_dec = a.lambda_pg_pos > 0 or a.lambda_pg_vel > 0 or a.lambda_comb > 0 or a.lambda_agree > 0
    vae = afk = dec_nrm = None
    if need_dec:
        from pami.ckpt_io import load_vae_for_gen
        from pami.nets.anchor_fk import AnchorFK
        vae, dec_nrm = load_vae_for_gen(ARCH, vae_ckpt=a.vae, dev=dev, verbose=False)
        vae.eval()
        for p in vae.parameters(): p.requires_grad_(False)
        afk = AnchorFK(dev)
        print(f"[dec] decoded losses ON: pg_pos {a.lambda_pg_pos} pg_vel {a.lambda_pg_vel} comb {a.lambda_comb} "
              f"agree {a.lambda_agree} | t band ({a.dec_t_lo},{a.dec_t_hi}) | huber {a.dec_huber} m | "
              f"warm pg {a.dec_warm_pg} de {list(a.dec_warm_de)}")
    start_it = 0
    if a.resume_full:
        ck = torch.load(a.resume_full, map_location=dev, weights_only=False)
        check_resume_arch(ck["arch"], ARCH)
        m.load_state_dict(ck["model"])
        if ema and "ema" in ck: ema.load_state_dict(ck["ema"])
        if "opt" in ck: opt.load_state_dict(ck["opt"])
        start_it = int(ck.get("it", 0))
        print(f"resume_full from {a.resume_full}: it={start_it}")
    os.makedirs(a.out, exist_ok=True)
    logf = open(f"{a.out}/log_{a.run_name}.jsonl", "a")

    def jlog(d):
        logf.write(json.dumps(d) + "\n"); logf.flush()

    def lr_at(it):
        """LIGHT's LinearLR warmup + CosineAnnealingLR, computed from the iteration (exact on resume)."""
        if a.warmup and it < a.warmup:
            return a.lr * (0.05 + 0.95 * it / max(a.warmup, 1))
        cstep = min(max(it - a.warmup, 0), a.lr_cosine_steps)
        return a.lr_min + 0.5 * (a.lr - a.lr_min) * (1.0 + math.cos(math.pi * cstep / a.lr_cosine_steps))

    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True,
                    collate_fn=collate_t2hoi, persistent_workers=True)
    val_dl = None
    if val_seqs:
        val_ds = T2HOIPieceDataset(a.cap, a.vae_tag, seqs=val_seqs, data_dir=data_dir, n_obj=1,
                                   random_caption=False, want_wgt=want_wgt)
        val_dl = DataLoader(val_ds, batch_size=a.batch, shuffle=False, collate_fn=collate_t2hoi)
        print(f"val: {len(val_ds)} pieces")
    dec_kw = dict(vae=vae, stats_h=stats_h, stats_o=stats_o, afk=afk, dec_nrm=dec_nrm,
                  dec_t_lo=a.dec_t_lo, dec_t_hi=a.dec_t_hi, dec_huber=a.dec_huber)
    it = start_it; t0 = time.time(); m.train()
    while it < a.iters:
        for b in dl:
            for g in opt.param_groups: g["lr"] = lr_at(it)
            b = to_dev(b, dev)
            r_pg = min(it / a.dec_warm_pg, 1.0) if a.dec_warm_pg > 0 else 1.0     # curriculum ramps
            d0, d1 = a.dec_warm_de
            r_de = min(max((it - d0) / max(d1 - d0, 1), 0.0), 1.0)
            with amp():
                loss, comp = rf_step(m, b, bank, dev, a.lambda_obj, a.cfg_drop,
                                     lam_pg_pos=a.lambda_pg_pos * r_pg, lam_pg_vel=a.lambda_pg_vel * r_pg,
                                     lam_comb=a.lambda_comb * r_de, lam_agree=a.lambda_agree * r_de, **dec_kw)
            opt.zero_grad(); loss.backward()
            if a.clip > 0: torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip)
            opt.step()
            if ema: ema.update_parameters(m)
            it += 1
            if it % 200 == 0:
                rate = (it - start_it) / (time.time() - t0)
                eta = (a.iters - it) / max(rate, 1e-6) / 60
                print(f"it{it} | " + " ".join(f"{k} {v:.4f}" for k, v in comp.items()) + f" | {rate:.1f} it/s | ETA {eta:.0f}min")
                jlog({"it": it, "kind": "train", "lr": opt.param_groups[0]["lr"], **comp})
            if val_dl is not None and it % a.eval_every == 0:
                m.eval()
                vs, n = {}, 0
                with torch.no_grad():
                    g = torch.Generator(device=dev).manual_seed(0)     # fixed t/noise draw -> comparable evals
                    for vb in val_dl:
                        vb = to_dev(vb, dev)
                        with amp():
                            _, vc = rf_step(m, vb, bank, dev, a.lambda_obj, 0.0, rng=g,
                                            lam_pg_pos=a.lambda_pg_pos, lam_pg_vel=a.lambda_pg_vel,
                                            lam_comb=a.lambda_comb, lam_agree=a.lambda_agree, **dec_kw)
                        for k, v in vc.items(): vs[k] = vs.get(k, 0) + v
                        n += 1
                vs = {k: v / n for k, v in vs.items()}
                print(f"  [val it{it}] " + " ".join(f"{k} {v:.4f}" for k, v in vs.items()))
                jlog({"it": it, "kind": "val", **vs})
                m.train()
            if it % a.save_every == 0:
                sd = {"model": m.state_dict(), "opt": opt.state_dict(), "it": it, "arch": ARCH}
                if ema: sd["ema"] = ema.state_dict()
                torch.save(sd, f"{a.out}/pamigen_{a.run_name}_{it}.pt")
                torch.save(sd, f"{a.out}/pamigen_{a.run_name}_resume.pt.tmp")
                os.replace(f"{a.out}/pamigen_{a.run_name}_resume.pt.tmp", f"{a.out}/pamigen_{a.run_name}_resume.pt")
            if it >= a.iters: break
    dt = time.time() - t0
    print(f"[done] {it} iters in {dt/3600:.2f}h")
    jlog({"kind": "summary", "iters": it, "seconds": dt})


if __name__ == "__main__":
    main()
