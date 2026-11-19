"""Stage 6 — normalization statistics. Population = the TRAIN sequences of split_official.json
(v1; 8825 seqs), which is what the released checkpoints normalize with. std is floored at 1e-6.

  stats_human_interact.npz          mean/std (272) over all train frames                 [base]
  stats_obj_interact.npz            robj(6)/pobj(3)/delta(18) mean/std over active object-frames,
                                    PER-FRAME canonical dialect                          [base]
  stats_obj_interact_wf.npz         robj/pobj re-expressed in the WINDOW-FIRST frame over the train
                                    64-frame windows (obj_frame="window_first"); delta copied [windows]
  stats_human_interact_rootabs.npz  channels [0:8] recomputed as window-absolute (x_wf, z_wf, heading
                                    6D) over train windows (human_root="wf_abs"); [8:272] copied  [windows]
  stats_hand.npz                    hand block stats: pos / wpos / wlocal / posrot / wlocalrot   [hand]
The final PamiVAE (--obj_frame window_first --human_root wf_abs --hand_mode wlocalrot) reads
stats_obj_interact_wf.npz + stats_human_interact_rootabs.npz + stats_hand.npz["wlocalrot_*"].
"""
import collections
import csv
import json
import os

import numpy as np

from . import paths as P

ROBJ_D, POBJ_D, DELTA_D = 6, 3, 18
MAX_OBJS = 8


def _seq_to_272(index_csv, r272_out):
    return {r["seq_name"]: f"{r272_out}/Representation_272/{r['new_name']}" for r in csv.DictReader(open(index_csv))}


def _train(split_json):
    return json.load(open(split_json))["train"]


# ---------------------------------------------------------------- base (per-frame)
def stats_base(split_json, out_root=P.PAMI_DATA_ROOT, r272_out=P.R272_OUT, index_csv=P.INDEX_CSV):
    train = set(_train(split_json))
    s2r = _seq_to_272(index_csv, r272_out)
    hs = hsq = None; hn = 0; miss = 0
    for seq in sorted(train):
        f = s2r.get(seq)
        if not f or not os.path.exists(f):
            miss += 1; continue
        x = np.load(f).astype(np.float64)
        s = x.sum(0); sq = (x * x).sum(0)
        hs = s if hs is None else hs + s
        hsq = sq if hsq is None else hsq + sq
        hn += x.shape[0]
    hmean = hs / hn; hstd = np.maximum(np.sqrt(np.maximum(hsq / hn - hmean ** 2, 0)), 1e-6)
    np.savez(f"{out_root}/stats_human_interact.npz", mean=hmean.astype(np.float32), std=hstd.astype(np.float32))
    print(f"[stats base] human 272: {hn} frames ({miss} train seqs missing 272) -> stats_human_interact.npz")
    acc = {g: [None, None, 0] for g in ("robj", "pobj", "delta")}
    dims = {"robj": ROBJ_D, "pobj": POBJ_D, "delta": DELTA_D}
    for seq in sorted(train):
        f = f"{out_root}/output/{seq}/tae_sample.npz"
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True)
        mis = sorted({k.split("_")[1] for k in d.files if k.startswith("robj_") and k.split("_", 1)[1].isdigit()})
        for mi in mis:
            for g in ("robj", "pobj", "delta"):
                v = d[f"{g}_{mi}"].astype(np.float64).reshape(-1, dims[g])
                s, sq = v.sum(0), (v * v).sum(0)
                a = acc[g]
                a[0] = s if a[0] is None else a[0] + s
                a[1] = sq if a[1] is None else a[1] + sq
                a[2] += v.shape[0]
    out = {}
    for g in ("robj", "pobj", "delta"):
        s, sq, n = acc[g]; mean = s / n
        std = np.maximum(np.sqrt(np.maximum(sq / n - mean ** 2, 0)), 1e-6)
        out[f"{g}_mean"] = mean.astype(np.float32); out[f"{g}_std"] = std.astype(np.float32)
        print(f"  obj {g}: {n} frames")
    np.savez(f"{out_root}/stats_obj_interact.npz", **out)
    print(f"[stats base] object -> stats_obj_interact.npz")


# ---------------------------------------------------------------- window-based (dialects)
def iter_train_windows(split_json, out_root, r272_out, index_csv, manifest=None, max_objs=MAX_OBJS):
    """Yield (human_raw (64,272) f32, robj (M,64,6) f32, pobj (M,64,3) f32) for every train window in
    manifest order — the same population / order HOIWindowDataset serves (seq in train, 272 file
    present, <= MAX_OBJS active objects; M = number of active objects, may be 0)."""
    manifest = manifest or f"{out_root}/manifest_w64_s32.npz"
    man = np.load(manifest, allow_pickle=True)
    train = set(_train(split_json)); s2r = _seq_to_272(index_csv, r272_out)
    cache = collections.OrderedDict()

    def load(seq):
        if seq not in cache:
            if len(cache) >= 64:
                cache.popitem(last=False)
            cache[seq] = (np.load(s2r[seq]).astype(np.float32),
                          np.load(f"{out_root}/output/{seq}/tae_sample.npz", allow_pickle=True))
        return cache[seq]

    for k in range(len(man["seq"])):
        seq = str(man["seq"][k])
        if seq not in train or seq not in s2r:
            continue
        act = list(man["active"][k])
        if len(act) > max_objs:
            continue
        s, e = int(man["start"][k]), int(man["end"][k])
        h272, ts = load(seq)
        human = h272[s:e]
        robj = np.stack([ts[f"robj_{mi}"][s:e] for mi in act]).astype(np.float32) if act else np.zeros((0, e - s, 6), np.float32)
        pobj = np.stack([ts[f"pobj_{mi}"][s:e] for mi in act]).astype(np.float32) if act else np.zeros((0, e - s, 3), np.float32)
        yield human, robj, pobj


def stats_windows(split_json, out_root=P.PAMI_DATA_ROOT, r272_out=P.R272_OUT, index_csv=P.INDEX_CSV,
                  manifest=None):
    """stats_obj_interact_wf.npz + stats_human_interact_rootabs.npz in ONE pass over the train windows."""
    from ..data.objframe import robj_to_window_first, pobj_to_window_first
    from ..data.dialect import root_delta_to_abs
    base_o = np.load(f"{out_root}/stats_obj_interact.npz")
    base_h = np.load(f"{out_root}/stats_human_interact.npz")
    acc = {g: [np.zeros(d), np.zeros(d), 0] for g, d in (("robj", 6), ("pobj", 3))}
    hs = np.zeros(8); hsq = np.zeros(8); hn = 0
    n = 0
    for human, robj, pobj in iter_train_windows(split_json, out_root, r272_out, index_csv, manifest):
        hv = root_delta_to_abs(human)[:, :8].astype(np.float64)
        hs += hv.sum(0); hsq += (hv * hv).sum(0); hn += hv.shape[0]
        if len(robj):
            wf = {"robj": robj_to_window_first(robj, human), "pobj": pobj_to_window_first(pobj, human)}
            for g in ("robj", "pobj"):
                v = wf[g].reshape(-1, acc[g][0].shape[0]).astype(np.float64)
                acc[g][0] += v.sum(0); acc[g][1] += (v * v).sum(0); acc[g][2] += v.shape[0]
        n += 1
        if n % 5000 == 0:
            print(f"  {n} windows", flush=True)
    out = {}
    for g in ("robj", "pobj"):
        s, sq, m = acc[g]
        mean = s / m; std = np.maximum(np.sqrt(np.maximum(sq / m - mean ** 2, 0)), 1e-6)
        out[g] = (mean.astype(np.float32), std.astype(np.float32))
        print(f"  {g} (window_first): n={m}")
    np.savez(f"{out_root}/stats_obj_interact_wf.npz",
             robj_mean=out["robj"][0], robj_std=out["robj"][1], pobj_mean=out["pobj"][0], pobj_std=out["pobj"][1],
             delta_mean=base_o["delta_mean"], delta_std=base_o["delta_std"])
    hmean = hs / hn; hstd = np.maximum(np.sqrt(np.maximum(hsq / hn - hmean ** 2, 0)), 1e-6)
    hm = base_h["mean"].astype(np.float32).copy(); hd = base_h["std"].astype(np.float32).copy()
    hm[:8] = hmean.astype(np.float32); hd[:8] = hstd.astype(np.float32)
    np.savez(f"{out_root}/stats_human_interact_rootabs.npz", mean=hm, std=hd)
    print(f"[stats windows] {n} train windows -> stats_obj_interact_wf.npz + stats_human_interact_rootabs.npz")


# ---------------------------------------------------------------- hand block
def stats_hand(split_json, out_root=P.PAMI_DATA_ROOT):
    from ..data.hand_frames import to_wlocal
    n1 = np.zeros(270); n2 = np.zeros(270); nfr = 0; nseq = 0
    w1 = np.zeros(90); w2 = np.zeros(90)
    l1 = np.zeros(90); l2 = np.zeros(90)
    for s in _train(split_json):
        p = f"{out_root}/output/{s}/hand_joints.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p)
        pos = z["hand"].astype(np.float64).reshape(-1, 90)
        rot = z["hand_rot6d"].astype(np.float64).reshape(-1, 180)
        blk = np.concatenate([pos, rot], 1)
        h = np.load(f"{out_root}/output/{s}/tae_sample.npz")["human_272"]
        if len(h) != len(pos):
            raise SystemExit(f"{s}: human_272 {len(h)} frames vs hand_joints {len(pos)} — stale package")
        wr = h[:, 68:74].astype(np.float64)
        wblk = (pos.reshape(-1, 30, 3) - np.concatenate(
            [np.repeat(wr[:, None, 0:3], 15, 1), np.repeat(wr[:, None, 3:6], 15, 1)], 1)).reshape(-1, 90)
        lblk = to_wlocal(pos.reshape(-1, 30, 3).astype(np.float32), h).reshape(-1, 90).astype(np.float64)
        n1 += blk.sum(0); n2 += (blk ** 2).sum(0)
        w1 += wblk.sum(0); w2 += (wblk ** 2).sum(0)
        l1 += lblk.sum(0); l2 += (lblk ** 2).sum(0)
        nfr += len(blk); nseq += 1
        if nseq % 1000 == 0:
            print(f"  {nseq} seqs / {nfr} frames", flush=True)
    if nfr == 0:
        raise SystemExit("no hand_joints.npz found in the train split")
    mean = n1 / nfr; std = np.sqrt(np.maximum(n2 / nfr - mean ** 2, 1e-12))
    wmean = w1 / nfr; wstd = np.sqrt(np.maximum(w2 / nfr - wmean ** 2, 1e-12))
    lmean = l1 / nfr; lstd = np.sqrt(np.maximum(l2 / nfr - lmean ** 2, 1e-12))
    out = f"{out_root}/stats_hand.npz"
    np.savez(out,
             pos_mean=mean[:90].astype(np.float32), pos_std=std[:90].astype(np.float32),
             wpos_mean=wmean.astype(np.float32), wpos_std=wstd.astype(np.float32),
             wlocal_mean=lmean.astype(np.float32), wlocal_std=lstd.astype(np.float32),
             posrot_mean=mean.astype(np.float32), posrot_std=std.astype(np.float32),
             wlocalrot_mean=np.concatenate([lmean, mean[90:]]).astype(np.float32),
             wlocalrot_std=np.concatenate([lstd, std[90:]]).astype(np.float32),
             n_seqs=nseq, n_frames=nfr)
    print(f"[stats hand] wrote {out} ({nseq} seqs, {nfr} frames)")


def run(split_json=None, out_root=P.PAMI_DATA_ROOT, r272_out=P.R272_OUT, index_csv=P.INDEX_CSV, which=("base", "windows", "hand")):
    split_json = split_json or f"{out_root}/split_official.json"
    print(f"[stats] population = train of {split_json}")
    if "base" in which:
        stats_base(split_json, out_root, r272_out, index_csv)
    if "windows" in which:
        stats_windows(split_json, out_root, r272_out, index_csv)
    if "hand" in which:
        stats_hand(split_json, out_root)
