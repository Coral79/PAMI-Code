"""Stage 1 — per-sequence conversion.

For each sequence (all subsets), writes <out_root>/output/<seq>/:
  object_6dof.npz   names, n_frames, R_0 (T,3,3), t_0 (T,3), canon_0 (V,3) centered canonical mesh,
                    faces_0, samp_0 (2048,3) surface samples, resid_mm (0: 6-DoF is exact from InterAct)
  anchor_fk.npz     world anchor rotations/positions R_<name>/p_<name> for the 6 part anchors
                    (pelvis, spine3, wrists, ankles) from a rigid FK chain on the body pose + floor_y
  canon_anchors.npz p_anch (T,6,3), R_anch (T,6,3,3) in the per-frame canonical frame, part_names,
                    jrest (22,3) gendered rest skeleton + jrest_parents (AnchorFK per-sequence skeleton)
  tae_sample.npz    object channels + routing labels (assemble_tae.py); human_272 is injected by stage 2
  bps_sample.npz    per active object: bps_<mi> (InterAct's precomputed 1024-pt deltas), bps_norm_<mi>
                    (LIGHT-normalized deltas), scale_<mi>, + basis_file / n_bps / margin
and <r272_out>/smpl_85/<new_name>.npy = [poses66 | zeros6 | trans3 | betas10] (float32) for the 272 chain,
plus the sequence index csv (source_path,start_frame,end_frame,new_name,seq_name), row order =
subsets.all_seqs() (fixed subset order, sorted within subset) — the file names of every 272 product
depend on this order, so it never changes (an existing csv is only ever extended).

Body models: gendered SMPL-X / SMPL-H with the subset's betas (16 or 10); the FK uses the REAL hand
poses for the contact mesh; the 272 / anchors use body joints only.
Determinism: FK on GPU when available (float-eps differences vs CPU).
"""
import csv
import json
import os
import time
import traceback

import numpy as np
import torch

from . import paths as P
from .assemble_tae import assemble
from .fk import bm_cached, fk_global_np, fk_seq
from .objects import MARGIN, obj_cached
from .subsets import ANCH_JOINT, ANCH_NAME, PARTS, SUBSETS, all_seqs, heading_frame, load_seq, seq_subset_map


def atomic_savez(path, **kw):
    tmp = f"{path}.tmp{os.getpid()}.npz"; np.savez(tmp, **kw); os.replace(tmp, path)


def atomic_save(path, arr):
    tmp = f"{path}.tmp{os.getpid()}.npy"; np.save(tmp, arr); os.replace(tmp, path)


def build_index(seqs, index_csv):
    """Deterministic seq -> 000000.npy.. mapping. EXTENDS an existing csv whose rows are a prefix of
    `seqs`; rebuilds only if the rows are not a prefix (never renumbers assigned names)."""
    s2s = seq_subset_map()
    rows = []
    if os.path.exists(index_csv):
        rows = list(csv.DictReader(open(index_csv)))
        if [r["seq_name"] for r in rows] == seqs:
            return {r["seq_name"]: r["new_name"] for r in rows}
        if [r["seq_name"] for r in rows] == seqs[:len(rows)]:
            print(f"[index] extending existing csv ({len(rows)} rows) -> {len(seqs)}")
        else:
            print(f"[index] existing csv ({len(rows)} rows) is NOT a prefix of the expected order -> rebuild")
            rows = []
    mapping = {r["seq_name"]: r["new_name"] for r in rows}
    os.makedirs(os.path.dirname(os.path.abspath(index_csv)), exist_ok=True)
    tmp = index_csv + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, ["source_path", "start_frame", "end_frame", "new_name", "seq_name"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
        for i, s in enumerate(seqs[len(rows):], start=len(rows)):
            src = f"{SUBSETS[s2s[s]]['root']}/sequences_canonical/{s}/human.npz"
            h = np.load(src, allow_pickle=True)
            T = int((h["poses"] if "poses" in h.files else h["global_orient"]).shape[0])
            nn = f"{i:06d}.npy"; mapping[s] = nn
            w.writerow({"source_path": src, "start_frame": 0, "end_frame": T, "new_name": nn, "seq_name": s})
    os.replace(tmp, index_csv)
    print(f"[index] wrote {index_csv} ({len(seqs)} rows)")
    return mapping


def seq_done(seq, new_name, out_root, r272_out):
    od = f"{out_root}/output/{seq}"
    try:
        for fn in ("object_6dof.npz", "anchor_fk.npz", "canon_anchors.npz", "tae_sample.npz", "bps_sample.npz"):
            np.load(f"{od}/{fn}", allow_pickle=True).files
        ca = np.load(f"{od}/canon_anchors.npz", allow_pickle=True)
        if "jrest" not in ca.files:
            return False
        b = np.load(f"{od}/bps_sample.npz", allow_pickle=True)
        n_raw = sum(1 for k in b.files if k.startswith("bps_") and not k.startswith("bps_norm_"))
        n_norm = sum(1 for k in b.files if k.startswith("bps_norm_"))
        if n_norm != n_raw:
            return False
        s85 = np.load(f"{r272_out}/smpl_85/{new_name}", mmap_mode="r")
        t = int(np.load(f"{od}/tae_sample.npz", allow_pickle=True)["T"])
        return s85.shape[0] == t
    except Exception:
        return False


def process_seq(seq, new_name, out_root, r272_out, dev="cpu", keep_verts=False):
    out_dir = f"{out_root}/output/{seq}"; os.makedirs(out_dir, exist_ok=True)
    d = load_seq(seq); T = d["T"]
    ob = obj_cached(d["obj_name"], d["subset"], dev)

    atomic_savez(f"{out_dir}/object_6dof.npz",
                 names=np.array([d["obj_name"]], dtype="U64"), n_frames=T,
                 R_0=d["obj_R"].astype(np.float32), t_0=d["obj_t"].astype(np.float32),
                 canon_0=ob["V"].astype(np.float32), faces_0=ob["F"], samp_0=ob["samp"], resid_mm=np.float32(0.0))

    hv, Jtr, Jrest, parents = fk_seq(d, dev)
    if keep_verts:
        atomic_savez(f"{out_dir}/human_verts.npz", verts=hv,
                     faces=bm_cached(d["gender"], d["model"], d["nb"], dev).f.cpu().numpy().astype(np.int32),
                     joints=Jtr.astype(np.float32))

    Rg, pg = fk_global_np(d["poses"][:, :66], d["trans"], Jrest, parents)
    fk_err = float(np.abs(pg - Jtr).max())
    if fk_err > 1e-3:
        raise RuntimeError(f"FK chain mismatch {fk_err*1000:.2f} mm vs BodyModel Jtr")
    af = {"anchor_names": np.array([ANCH_NAME[p] for p in PARTS], dtype="U32")}
    for p in PARTS:
        j = ANCH_JOINT[p]
        af[f"R_{ANCH_NAME[p]}"] = Rg[:, j].astype(np.float32)
        af[f"p_{ANCH_NAME[p]}"] = pg[:, j].astype(np.float32)
    af["floor_y"] = np.float32(pg[:, :22, 1].min())
    atomic_savez(f"{out_dir}/anchor_fk.npz", **af)

    Hrot, shift = heading_frame(af["R_root_pelvis"].astype(np.float64),
                                af["p_root_pelvis"].astype(np.float64), float(af["floor_y"]))
    p_anch = np.zeros((T, 6, 3), np.float32); R_anch = np.zeros((T, 6, 3, 3), np.float32)
    for pi, p in enumerate(PARTS):
        nm = ANCH_NAME[p]
        p_anch[:, pi] = np.einsum("tij,tj->ti", Hrot, af[f"p_{nm}"] - shift)
        R_anch[:, pi] = np.einsum("tij,tjk->tik", Hrot, af[f"R_{nm}"])
    atomic_savez(f"{out_dir}/canon_anchors.npz", p_anch=p_anch, R_anch=R_anch,
                 part_names=np.array(PARTS, dtype="U8"),
                 jrest=Jrest.astype(np.float32), jrest_parents=parents.astype(np.int64))

    os.makedirs(f"{r272_out}/smpl_85", exist_ok=True)
    smpl85 = np.concatenate([d["poses"][:, :66], np.zeros((T, 6)), d["trans"],
                             np.repeat(d["betas16"][None, :10], T, 0)], 1).astype(np.float32)
    atomic_save(f"{r272_out}/smpl_85/{new_name}", smpl85)

    assemble(seq, out_dir, hv, model=d["model"])

    ts = np.load(f"{out_dir}/tae_sample.npz", allow_pickle=True)
    names = [str(x) for x in ts["active_names"]]
    bout = {"active_names": ts["active_names"], "basis_file": np.array(P.BPS_BASIS),
            "n_bps": np.array(1024), "margin": np.float32(MARGIN)}
    for mi, name in enumerate(names):
        o = obj_cached(name, d["subset"], dev)
        bout[f"bps_{mi}"] = o["raw"]; bout[f"bps_norm_{mi}"] = o["norm"]; bout[f"scale_{mi}"] = o["scale"]
    atomic_savez(f"{out_dir}/bps_sample.npz", **bout)
    return dict(T=T, gender=d["gender"], obj=d["obj_name"], n_active=len(names), subset=d["subset"])


def run(seqs=None, shard=None, force=False, keep_verts=False, out_root=P.PAMI_DATA_ROOT,
        r272_out=P.R272_OUT, index_csv=P.INDEX_CSV, device=None):
    """seqs=None -> all sequences. shard='i/n' -> seqs[i::n]. The FULL index is always (re)built first
    so that new_name assignments are identical no matter which subset is processed."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    every = all_seqs()
    mapping = build_index(every, index_csv)
    todo = list(seqs) if seqs else every
    for s in todo:
        assert s in mapping, f"unknown seq {s}"
    fail_json = f"{out_root}/batch_failures.json"
    if shard:
        si, sn = map(int, shard.split("/")); assert 0 <= si < sn
        todo = todo[si::sn]
        fail_json = f"{out_root}/batch_failures_shard{si}of{sn}.json"
    os.makedirs(out_root, exist_ok=True)
    print(f"[convert] device={dev} seqs={len(todo)}/{len(every)} out={out_root}", flush=True)
    fails = {}; n_skip = n_new = 0; t0 = time.time()
    for i, seq in enumerate(todo):
        if not force and seq_done(seq, mapping[seq], out_root, r272_out):
            n_skip += 1; continue
        try:
            info = process_seq(seq, mapping[seq], out_root, r272_out, dev, keep_verts)
            n_new += 1
            print(f"[{i+1}/{len(todo)}] {seq}: T={info['T']} {info['subset']}/{info['gender']} "
                  f"{info['obj']} active={info['n_active']}", flush=True)
        except Exception as e:
            fails[seq] = repr(e)
            print(f"[{i+1}/{len(todo)}] {seq}: FAILED {e!r}", flush=True)
            traceback.print_exc()
        if (i + 1) % 200 == 0:
            json.dump(fails, open(fail_json, "w"), indent=1)
    json.dump(fails, open(fail_json, "w"), indent=1)
    print(f"[convert] done: new={n_new} skipped={n_skip} failed={len(fails)} ({(time.time()-t0)/60:.1f} min)")
    return fails
