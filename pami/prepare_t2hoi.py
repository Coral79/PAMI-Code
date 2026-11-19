"""PamiGen data: encode every action-aligned PIECE (<= cap frames, segments_cap<cap>.json) with the frozen
PamiVAE into the latents archive PamiGen trains on.

Each piece is encoded INDEPENDENTLY with the window-first object frame / window-absolute human root anchored
at the PIECE's first frame (exactly how the VAE training windows were built; a generated piece starts at its
own origin). Stored per piece: encoder MU (deterministic) refined by the frozen SceneST (the decoders consume
this "refined" dialect), the piece's active-object index list, caption ids; plus per-token latent stats over
the TRAIN pieces only (--stats_split) for diffusion-space normalisation.

CLI: encode_latents.py (repo root). Output: <data_dir>/latents_cap<cap>_<tag>_refined.npz with
tag = VAE checkpoint basename without ".pt" (a "split_ae_"/"pamivae_" prefix is dropped), e.g. pamivae.pt ->
latents_cap300_pamivae_refined.npz.
"""
import os, json, argparse, hashlib
import numpy as np
from pami import config as C

CAPS = (300, 448, 600)                       # 10 / 15 / 20 s @ 30 fps, all 4-aligned


def _r4(x):                                   # nearest 4-multiple (latent-step grid)
    return int(round(x / 4)) * 4


def latents_tag(ckpt):
    base = os.path.basename(ckpt).replace(".pt", "")
    for pfx in ("split_ae_", "pamivae_"):
        if base.startswith(pfx):
            base = base[len(pfx):]
    return base + "_refined"


def stage_encode(ckpt, caps=(300,), data_dir=None, limit=None, stats_split=None, out_tag=None):
    import torch
    from pami.ckpt_io import load_split_ae, assert_repr_dialect
    from pami.norm import build as build_norm
    from pami.data.hoi_dataset import _seq_to_272
    from pami.data.objframe import robj_to_window_first, pobj_to_window_first
    from pami.data.hand_frames import to_wlocal
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, ck, arch = load_split_ae(ckpt, dev=dev)
    assert_repr_dialect(arch, "encode_latents")
    nrm = build_norm()
    DD = data_dir or C.T2HOI_DATA
    hs = np.load(C.STATS_HAND)
    hand_mean = hs["wlocalrot_mean"].astype(np.float32)
    hand_std = np.maximum(hs["wlocalrot_std"].astype(np.float32), 1e-6)
    hand_cache = {}

    def hand_block(seq):
        if seq not in hand_cache:
            z = np.load(f"{C.PKG}/output/{seq}/hand_joints.npz", allow_pickle=True)
            pos = z["hand"].astype(np.float32).reshape(len(z["hand"]), 90)
            h = np.load(f"{C.PKG}/output/{seq}/tae_sample.npz")["human_272"]
            if len(h) != len(pos):
                raise RuntimeError(f"{seq}: human_272 {len(h)} vs hand_joints {len(pos)} frames — stale package")
            pos = to_wlocal(pos.reshape(-1, 30, 3), h).reshape(-1, 90)
            blk = np.concatenate([pos, z["hand_rot6d"].astype(np.float32).reshape(len(pos), 180)], 1)
            hand_cache[seq] = (blk - hand_mean) / hand_std
        return hand_cache[seq]

    tag = out_tag or latents_tag(ckpt)
    train_set = None; train_sha = None
    if stats_split:
        sp = json.load(open(stats_split))
        train = sorted(set(sp["train"]))
        if not train:
            raise SystemExit(f"--stats_split {stats_split}: empty train list")
        train_set = set(train)
        train_sha = hashlib.sha1("\n".join(train).encode()).hexdigest()[:12]
        print(f"[encode] --stats_split {stats_split}: stats population restricted to {len(train)} train seqs (sha1 {train_sha})")
    s2r = _seq_to_272()
    tj = json.load(open(f"{DD}/texts.json"))
    h_cache, ts_cache = {}, {}

    def enc_piece(seq, f0, f1):
        if seq not in h_cache:
            h_cache[seq] = np.load(s2r[seq]).astype(np.float32)
            ts_cache[seq] = np.load(f"{C.PKG}/output/{seq}/tae_sample.npz", allow_pickle=True)
        h = h_cache[seq][f0:f1]                                        # (W,272) raw
        ts = ts_cache[seq]
        keep = list(range(len(ts["active_names"])))                    # every active object of the sequence
        M = len(keep)
        b = {"human": torch.tensor(nrm.nh(h))[None].to(dev)}
        hand_t = torch.tensor(hand_block(seq)[f0:f1])[None].to(dev)
        with torch.no_grad():
            _, mu_h, _ = m._encode_human(b["human"], hand_t)           # (1,Tl,P*16)
            Tl = mu_h.shape[1]
            zh_t = mu_h.reshape(1, Tl, m.P, C.SAE_LATENT)
            if M:
                robj = robj_to_window_first(np.stack([ts[f"robj_{mi}"][f0:f1] for mi in keep]).astype(np.float32), h)
                pobj = pobj_to_window_first(np.stack([ts[f"pobj_{mi}"][f0:f1] for mi in keep]).astype(np.float32), h)
                delta = np.stack([ts[f"delta_{mi}"][f0:f1] for mi in keep]).astype(np.float32)
                ob = {"robj": torch.tensor(nrm.no("robj", robj))[None].to(dev),
                      "pobj": torch.tensor(nrm.no("pobj", pobj))[None].to(dev),
                      "delta": torch.tensor(nrm.no("delta", delta))[None].to(dev),
                      "omask": torch.ones(1, M, device=dev)}
                mu_o = m._encode_object(ob)[1]                        # (1,M,Tl,16)
            else:
                mu_o = torch.zeros(1, 0, Tl, C.SAE_LATENT, device=dev)
            zo8 = torch.zeros(1, C.MAX_OBJS, Tl, C.SAE_LATENT, device=dev)   # frozen SceneST on the 8-slot grid
            om8 = torch.zeros(1, C.MAX_OBJS, device=dev)
            zo8[:, :M] = mu_o; om8[:, :M] = 1
            grid = m.refiner(torch.cat([zh_t, zo8.permute(0, 2, 1, 3)], dim=2), om8)
            zh = grid[0, :, :m.P].cpu().numpy()                        # (Tl,P,16) refined
            zo = grid[0, :, m.P:m.P + M].cpu().numpy()                 # (Tl,M,16) refined
        return zh, zo, keep

    for cap in caps:
        segs = json.load(open(f"{DD}/segments_cap{cap}.json"))["segments"]
        n0 = len(segs); n_act = {}
        for s in segs:                                                 # SceneST grid is 8-slot: drop pieces with more objects
            if s["seq"] not in n_act:
                n_act[s["seq"]] = len(np.load(f"{C.PKG}/output/{s['seq']}/tae_sample.npz", allow_pickle=True)["active_names"])
        segs = [s for s in segs if n_act[s["seq"]] <= C.MAX_OBJS]
        if n0 - len(segs):
            print(f"  [cap {cap}] dropped {n0 - len(segs)} pieces with more than {C.MAX_OBJS} active objects")
        if limit:
            segs = segs[:limit]
            print(f"  [cap {cap}] --limit {limit}: encoding the first {len(segs)} pieces only")
        n_stat = sum(s["seq"] in train_set for s in segs) if train_set is not None else len(segs)
        if train_set is not None and n_stat == 0:
            raise SystemExit(f"--stats_split {stats_split}: cap{cap} has ZERO pieces in the train list")
        out = {"seg_seq": np.array([s["seq"] for s in segs]),
               "seg_f0": np.array([s["f0"] for s in segs], np.int32),
               "seg_f1": np.array([s["f1"] for s in segs], np.int32),
               "cap_tid": np.array([s.get("cap_tid", tj["captions"].get(s["seq"], 0)) for s in segs], np.int32),
               "vae_ckpt": np.array(os.path.basename(ckpt)), "vae_it": np.array(ck.get("it", -1)),
               "refined": np.array(True), "dataset": np.array("interact"), "latent_dialect": np.array("post_st"),
               "repr_obj_frame": np.array("window_first"), "repr_human_root": np.array("wf_abs"),
               "stats_split_path": np.array(stats_split or ""),
               "stats_population": np.array(f"train_only:{n_stat}:{train_sha}" if train_set is not None else "all")}
        hsum = hsq = osum = osq = 0.0; hn = on = 0
        for i, s in enumerate(segs):
            piece_tid = s.get("cap_tid", tj["captions"].get(s["seq"], 0))
            zh, zo, keep = enc_piece(s["seq"], s["f0"], s["f1"])
            out[f"zh_{i}"] = zh.astype(np.float32)
            out[f"zo_{i}"] = zo.astype(np.float32)
            out[f"keep_{i}"] = np.array(keep, np.int16)
            out[f"cap_tids_{i}"] = np.array(s.get("cap_tids", [piece_tid]), np.int32)
            out[f"astep_{i}"] = np.full((s["f1"] - s["f0"]) // 4, -1, np.int32)   # no action annotation on InterAct
            if train_set is None or s["seq"] in train_set:
                hsum += float(zh.sum()); hsq += float((zh ** 2).sum()); hn += zh.size
                if zo.size:
                    osum += float(zo.sum()); osq += float((zo ** 2).sum()); on += zo.size
            if i % 200 == 0:
                print(f"  [cap {cap}] {i}/{len(segs)}", end="\r")
        hm = hsum / hn; hs_ = max(hsq / hn - hm ** 2, 0.0) ** 0.5
        om_ = osum / max(on, 1); os_ = max(osq / max(on, 1) - om_ ** 2, 0) ** 0.5
        out["stats_h"] = np.array([hm, hs_], np.float32)               # scalar (legacy); PamiGen uses stats_h_tok
        out["stats_o"] = np.array([om_, os_], np.float32)
        P = out["zh_0"].shape[1]
        tsum = np.zeros(P); tsq = np.zeros(P); tn = 0
        for i in range(len(segs)):
            if train_set is not None and segs[i]["seq"] not in train_set:
                continue
            zt = out[f"zh_{i}"].astype(np.float64)
            tsum += zt.sum((0, 2)); tsq += (zt ** 2).sum((0, 2)); tn += zt.shape[0] * zt.shape[2]
        tmean = tsum / tn
        tstd = np.sqrt(np.maximum(tsq / tn - tmean ** 2, 1e-12))
        out["stats_h_tok"] = np.stack([tmean, tstd], 1).astype(np.float32)   # (P,2) per-token
        fp = f"{DD}/latents_cap{cap}_{tag}.npz"
        np.savez(fp, **out)
        print(f"[encode cap {cap}] {len(segs)} pieces -> {fp} ({os.path.getsize(fp)/1e6:.0f} MB) | "
              f"mu_h mean/std {hm:.3f}/{hs_:.3f} | mu_o mean/std {om_:.3f}/{os_:.3f} | stats over "
              f"{n_stat}/{len(segs)} pieces ({'train-only' if train_set is not None else 'all'})")
        return fp


def main(argv=None):
    ap = argparse.ArgumentParser(description="Encode PamiGen training latents with the frozen PamiVAE")
    ap.add_argument("--ckpt", default=C.CKPT_VAE, help="PamiVAE checkpoint (encoder + SceneST)")
    ap.add_argument("--cap", type=int, default=300, help="piece cap (segments_cap<cap>.json)")
    ap.add_argument("--data_dir", default=None, help="texts.json / segments -> latents dir (default config.T2HOI_DATA)")
    ap.add_argument("--stats_split", default=C.SPLIT, help="split json whose TRAIN sequences define the latent stats "
                    "(the paper archive used the official split; '' = all pieces)")
    ap.add_argument("--limit", type=int, default=None, help="encode only the first N pieces (debug)")
    ap.add_argument("--out_tag", default=None, help="override the archive tag (default from the ckpt basename)")
    a = ap.parse_args(argv)
    stage_encode(a.ckpt, [a.cap], data_dir=a.data_dir, limit=a.limit, stats_split=a.stats_split or None, out_tag=a.out_tag)


if __name__ == "__main__":
    main()
