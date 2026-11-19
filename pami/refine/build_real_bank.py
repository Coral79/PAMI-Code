"""Bank builder: frozen generator samples (gen_*.npz) -> refiner training windows with frozen masks.

Input: a PamiGen sample npz (decoded human_272, wlocalrot hand block [wrist-local pos 90 | parent-
relative rot6d 180], window-first robj/pobj, per-frame obj_weights routing) and its gt_* twin
(decode(encode(GT)), the corruption stream's substrate). Output: one item per window (varlen: ONE
window per piece, W = min(512, floor32(T))), the fields listed in bank_schema.BANK_FIELDS:

    contact_intent      (W,5) bool   L_hand R_hand L_foot R_foot torso -- vs the OBJECT:
                                     (routing mass > W_TH OR part-surface distance < NEAR_M) AND
                                     |d dist/dt| < TREND_M, then fill gaps <= GAP_F, drop runs < MIN_F
    reachable_mask      (W,5) bool   part-object distance < REACH_M
    segment_id          (W,5) int32  contact segment index, -1 = none
    segment_anchor      (S,10) f32   part,t0,t1,anchor_f,human_joint, obj_face, bary0..2, err --
                                     the frozen object-local correspondence of each segment
    foot_contact_intent (W,2) bool   planted feet vs the GROUND, per side, judged against a shared
                                     per-window floor estimate (p5 of both feet, capped at +FLOOR_CAP)
    root_wf_pos/rot6                 explicit window-first root of the generated human
    rot6h               (W,30,6)     the DECODED finger rotations (re-orthonormalized), no IK

Geometry is the refiner's own theta -> FK chain. All thresholds (V1) are stamped into every item
(`meta`) and into the compared mask identity (`mask_id`).

Sharded build protocol (what the shipped banks used):
    --init     once, into a FRESH dir: writes build_spec.json (pins every mask/dialect parameter)
    --append   per shard, concurrently: verifies its config against the spec, writes items + its own
               manifest.part-<tag>.json
    --finalize once: merges the parts into manifest.json (every item re-verified, counts recounted)
"""
import argparse
import datetime
import glob
import json
import os
import subprocess
import sys

import numpy as np
import torch
import trimesh

from pami.data.objframe import (human_root_to_window_first, pobj_to_per_frame, pobj_to_window_first,
                                robj_to_per_frame, robj_to_window_first)
from pami.refine.bank_schema import (BANK_FIELDS, BANK_SCHEMA, BUILD_SPEC_NAME, ITEM_STAMP_FIELDS,
                                     MANIFEST_NAME, MASK_VERSION, PART_GLOB, PART_PREFIX,
                                     atomic_write_json, count_items, items_digest, mask_id, verify_items)
from pami.refine.base_state import body_fk
from pami.refine.hand_state import hand_fk, rest_hand_offsets, wrist_from_body
from pami.refine.paths import INTERACT_PKG, interact_common, use_interact_config
from pami.rotvote import mat_to_rot6d
from pami.vendor.eval_trans import rotation_6d_to_matrix

WIN, STRIDE = 64, 32                                          # fixed-window mode (the shipped banks are varlen)
MAX_VARLEN = 512                                              # = net_m3.MAX_W
PARTS = ("L_hand", "R_hand", "L_foot", "R_foot", "torso")
PART_JOINTS = {0: [20], 1: [21], 2: [7, 10], 3: [8, 11], 4: [0, 1, 2, 3, 6, 9, 12, 15]}
PART_SLOT = {0: [2], 1: [3], 2: [4], 3: [5], 4: [0, 1]}       # routing slots per part
FINGER_ROWS = {0: list(range(22, 37)), 1: list(range(37, 52))}
FOOT_JOINTS = (10, 11)                                        # col 0 = LEFT, col 1 = RIGHT
FLOOR_STRATEGY = "shared_capped"
# mask thresholds (metres / frames). The dict is hashed into mask_id: do not reorder or rename.
V1 = dict(W_TH=0.25, NEAR_M=0.08, REACH_M=0.15, GAP_F=5, MIN_F=10,
          FOOT_H=0.06, FOOT_V=0.015, ANCHOR_K=5,
          TREND_M=0.01,          # |d(part-object distance)/dt| < 1 cm/frame -> not approaching / leaving
          FOOT_VY=0.005,         # vertical foot-joint speed gate (m/frame)
          FOOT_SINK=0.03,        # a foot more than this below the floor estimate is a defect, not planted
          FLOOR_CAP=0.05)        # the shared p5 floor may follow the generation DOWN, but is capped above
HAND_MODE = "wlocalrot"                                       # the only finger convention the builder supports
MASK_ID = mask_id(FLOOR_STRATEGY, V1)


def _morph(raw, gap, mn):
    """Fill gaps <= gap, then drop runs < mn, per column."""
    out = raw.copy()
    W = len(out)
    for p in range(out.shape[1]):
        col = out[:, p]
        t = 0
        while t < W:
            if not col[t] and t > 0 and col[t - 1]:
                e = t
                while e < W and not col[e]:
                    e += 1
                if e < W and e - t <= gap:
                    col[t:e] = True
                t = e
            else:
                t += 1
        t = 0
        while t < W:
            if col[t]:
                e = t
                while e < W and col[e]:
                    e += 1
                if e - t < mn:
                    col[t:e] = False
                t = e
            else:
                t += 1
    return out


def window_floor(y, V=V1):
    """y (W,2) foot heights -> shared floor estimate: p5 over both feet, capped at +FLOOR_CAP."""
    return float(min(np.percentile(y, 5), V["FLOOR_CAP"]))


def foot_planted_mask(feet, V=V1, morph=True):
    """feet (W,2,3) window-first foot joints -> (W,2) bool planted per side + diagnostics."""
    y = feet[:, :, 1]
    floor = window_floor(y, V)
    h = y - floor
    vh = np.linalg.norm(np.diff(feet[:, :, [0, 2]], axis=0), axis=-1)
    vh = np.concatenate([vh[:1], vh], 0)
    vy = np.abs(np.diff(y, axis=0))
    vy = np.concatenate([vy[:1], vy], 0)
    raw = (h < V["FOOT_H"]) & (h > -V["FOOT_SINK"]) & (vh < V["FOOT_V"]) & (vy < V["FOOT_VY"])
    m = _morph(raw, V["GAP_F"], V["MIN_F"]) if morph else raw
    return m, dict(floor=floor, height=h, v_horiz=vh, v_vert=vy, raw=raw)


def _segments(intent):
    segs = []
    for p in range(intent.shape[1]):
        t = 0
        while t < len(intent):
            if intent[t, p]:
                e = t
                while e < len(intent) and intent[e, p]:
                    e += 1
                segs.append((p, t, e))
                t = e
            else:
                t += 1
    return segs


def check_hand_mode(hm, ndim=None, where=""):
    """The builder reads block[:, :90] as WRIST-LOCAL positions and block[:, 90:270] as decoded
    rotations: only `wlocalrot` (270 dims) carries both."""
    if hm != HAND_MODE:
        raise SystemExit(f"{where}hand_mode={hm!r} is not supported by build_real_bank (only {HAND_MODE!r})")
    if ndim is not None and ndim != 270:
        raise SystemExit(f"{where}hand_mode={hm!r} expects a 270-dim hand block, npz has {ndim}")
    return hm


def seq_meta(seq, cache, cap=48):
    """Per-sequence body model + rest skeleton, LRU-cached (a BodyModel is ~1 s to build)."""
    if seq in cache:
        cache[seq] = cache.pop(seq)
        return cache[seq]
    while len(cache) >= cap:
        cache.pop(next(iter(cache)))
    ic = interact_common()
    m = ic.load_seq(seq)
    bm = ic.body_model(m["gender"], num_betas=m["nb"], model=m["model"])
    J = bm(betas=torch.from_numpy(m["betas16"][None, : m["nb"]]).float()).Jtr[0].detach()
    parents = torch.tensor(np.asarray(bm.kintree_table[0][:22]).astype(np.int64))
    cache[seq] = dict(bm=bm, betas16=m["betas16"].astype(np.float32), jrest=J,
                      rest_h=rest_hand_offsets(J[None], str(m["model"]))[0], parents=parents, model=str(m["model"]))
    return cache[seq]


def build_item(z, f0, meta, bm_cache, pobj_c, robj_c, K=lambda k: k, floor_strategy=FLOOR_STRATEGY,
               anchors_on=True, hand_mode=HAND_MODE, win=None):
    """One window [f0, f0+W) of gen npz `z` -> bank item dict (or None if the piece is too short).
    K maps logical key -> npz key: identity for the GEN stream, 'gt_'+k for the ROUNDTRIP stream.
    pobj_c/robj_c: the whole piece's object pose converted to PER-FRAME canonical once (objframe), so
    every window gets its own window-first transform from its own human slice."""
    assert floor_strategy == FLOOR_STRATEGY and hand_mode == HAND_MODE
    W = WIN if win is None else win
    h = z[K("human_272")][f0:f0 + W].astype(np.float32)
    if len(h) < W:
        return None
    hw = human_root_to_window_first(h)
    root_wf_pos, root_wf_rot6 = hw[:, 8:11].copy(), hw[:, 140:146].copy()
    p_obj = pobj_to_window_first(pobj_c[:, f0:f0 + W], h).astype(np.float32)   # (M,W,3)
    r_obj = robj_to_window_first(robj_c[:, f0:f0 + W], h).astype(np.float32)   # (M,W,6)
    wgt = z[K("obj_weights")][:, f0:f0 + W].astype(np.float32)                # (M,W,7)
    seq = str(z["seq"])

    # the refiner's own chain: body FK from the decoded rotations + explicit window-first root
    rot6 = torch.from_numpy(h[:, 140:272].reshape(1, W, 22, 6))
    rot6 = mat_to_rot6d(rotation_6d_to_matrix(rot6))
    rot6[:, :, 0] = mat_to_rot6d(rotation_6d_to_matrix(torch.from_numpy(root_wf_rot6)[None]))
    pel = torch.from_numpy(root_wf_pos)[None]
    jrest, rest_h, parents = meta["jrest"], meta["rest_h"], meta["parents"]
    pg, Rg = body_fk(rot6, pel, parents, jrest[None, :22])
    Rw, pw = wrist_from_body(pg, Rg)

    # fingers: decoded rotations used directly; residual = FK(rotations) vs decoded positions (mm)
    hblk = z[K("hand")][f0:f0 + W].astype(np.float32)
    check_hand_mode(hand_mode, hblk.shape[1], where=f"{seq} t{f0}: ")
    local = hblk[:, :90].reshape(W, 30, 3).astype(np.float64)
    j_np, Rg_np = pg[0].numpy().astype(np.float64), Rg[0].numpy().astype(np.float64)
    tgt = np.empty_like(local)
    for side in (0, 1):
        wj = (20, 21)[side]; sl = slice(side * 15, (side + 1) * 15)
        tgt[:, sl] = j_np[:, wj][:, None] + np.einsum("tij,tkj->tki", Rg_np[:, wj], local[:, sl])
    r6 = torch.from_numpy(np.ascontiguousarray(hblk[:, 90:270]).reshape(1, W, 30, 6))
    hand_rot6d = mat_to_rot6d(rotation_6d_to_matrix(r6))
    pj, _ = hand_fk(hand_rot6d, Rw, pw, rest_h[None])
    ik_err = float(np.linalg.norm(pj[0].numpy() - tgt, axis=-1).mean() * 1000)
    hand_source = "decoded_rot6d"

    # part -> object surface distances on the posed shape samples
    ts = np.load(f"{INTERACT_PKG}/output/{seq}/tae_sample.npz", allow_pickle=True)
    samp = torch.from_numpy(ts["shape_0"].astype(np.float32))              # (S,3) canonical
    R = rotation_6d_to_matrix(torch.from_numpy(r_obj[0]))                  # (W,3,3)
    surf = torch.einsum("wij,sj->wsi", R, samp) + torch.from_numpy(p_obj[0])[:, None]
    allj = torch.cat([pg[0], pj[0]], 1)                                    # (W,52,3)
    d_all = torch.cdist(allj, surf).min(-1).values                         # (W,52)
    dpart = np.stack([torch.minimum(d_all[:, PART_JOINTS[p]].min(-1).values,
                                    d_all[:, FINGER_ROWS[p]].min(-1).values).numpy()
                      if p in FINGER_ROWS else d_all[:, PART_JOINTS[p]].min(-1).values.numpy()
                      for p in range(5)], 1)                               # (W,5)

    # intent / reachable / segments + frozen correspondence anchors
    wslot = np.stack([wgt[0][:, PART_SLOT[p]].max(-1) for p in range(5)], 1)
    ddt = np.abs(np.diff(dpart, axis=0, prepend=dpart[:1]))
    steady = ddt < V1["TREND_M"]
    raw = ((wslot > V1["W_TH"]) | (dpart < V1["NEAR_M"])) & steady
    intent = _morph(raw, V1["GAP_F"], V1["MIN_F"])
    reach = dpart < V1["REACH_M"]
    seg_id = np.full((W, 5), -1, np.int32)
    anchors = []
    mesh = None
    if anchors_on:
        o6 = np.load(f"{INTERACT_PKG}/output/{seq}/object_6dof.npz", allow_pickle=True)
        mesh = trimesh.Trimesh(o6["canon_0"].astype(np.float64), o6["faces_0"], process=False) \
            if len(o6["faces_0"]) else None
    for si, (p, t0s, t1s) in enumerate(_segments(intent)):
        seg_id[t0s:t1s, p] = si
        near = [t for t in range(t0s, t1s) if dpart[t, p] < V1["NEAR_M"]]
        pool = near if near else list(range(t0s, t1s))
        k = min(V1["ANCHOR_K"], len(pool))
        cand = pool[:k]
        af = sorted(cand, key=lambda t: dpart[t, p])[k // 2]              # median-of-first-k anchor frame
        jset = PART_JOINTS[p] + FINGER_ROWS.get(p, [])
        jid = jset[int(d_all[af, jset].argmin())]
        hp = allj[af, jid].numpy().astype(np.float64)
        hp_loc = (R[af].numpy().T @ (hp - p_obj[0, af])).astype(np.float64)
        if mesh is not None:
            cp, dist, tid = trimesh.proximity.closest_point(mesh, hp_loc[None])
            bary = trimesh.triangles.points_to_barycentric(mesh.triangles[tid[0]][None], cp)[0]
            anchors.append((p, t0s, t1s, af, jid, int(tid[0]), *bary.tolist(), float(dist[0])))
        else:
            anchors.append((p, t0s, t1s, af, jid, -1, 0, 0, 0, -1))

    # feet vs GROUND, per side
    feet = pg[0][:, list(FOOT_JOINTS)].numpy()                             # (W,2,3)
    foot_int, foot_diag = foot_planted_mask(feet)

    return dict(h272=h, rot6h=hand_rot6d[0].numpy().astype(np.float32),
                root_wf_pos=root_wf_pos, root_wf_rot6=root_wf_rot6, p_obj=p_obj, r6_obj=r_obj,
                contact_intent=intent, reachable_mask=reach, segment_id=seg_id,
                segment_anchor=np.array(anchors, np.float32).reshape(-1, 10),
                foot_contact_intent=foot_int, ik_err_mm=float(ik_err), seq=seq, f0=f0 + int(z["f0"]),
                bank_schema=BANK_SCHEMA, mask_version=MASK_VERSION, mask_id=MASK_ID,
                hand_mode=hand_mode, hand_source=hand_source,
                meta=json.dumps(dict(v1=V1, ckpt=str(z["ckpt"]), gen_f0=int(z["f0"]),
                                     floor_strategy=FLOOR_STRATEGY, floor_y=foot_diag["floor"],
                                     hand_mode=hand_mode, hand_source=hand_source,
                                     bank_schema=BANK_SCHEMA, mask_version=MASK_VERSION, mask_id=MASK_ID)),
                _feet=feet)


# ---- sharded build protocol ---------------------------------------------------------------------
_SPEC_FIELDS = ("bank_schema", "mask_version", "mask_id", "stream", "floor_strategy", "hand_mode",
                "window", "stride", "varlen", "mask_params")


def _git_commit():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as e:
        return f"unknown ({e})"


def _npz_hand_mode(path):
    z = np.load(path, allow_pickle=True)
    return str(z["hand_mode"]) if "hand_mode" in z.files else "wlocal"


def _spec_of(a, hand_mode, files):
    return dict(bank_schema=BANK_SCHEMA, mask_version=MASK_VERSION, mask_id=MASK_ID, stream=a.stream,
                floor_strategy=FLOOR_STRATEGY, hand_mode=hand_mode,
                window=(-1 if a.varlen else WIN), stride=(-1 if a.varlen else STRIDE), varlen=bool(a.varlen),
                mask_params=V1, gen_glob=a.gen_glob, n_gen_files=len(files),
                gen_files=[os.path.basename(f) for f in files], builder_git_commit=_git_commit(),
                created=datetime.datetime.now().isoformat(timespec="seconds"), argv=sys.argv)


def _write_spec(out, spec):
    """O_EXCL: two --init calls racing on one dir is an error, never a half-written spec."""
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f"[refuse] {out} is not empty: bank builds go into a FRESH directory")
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, BUILD_SPEC_NAME)
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise SystemExit(f"[refuse] {p} already exists: this directory has already been --init'ed")
    with os.fdopen(fd, "w") as fh:
        json.dump(spec, fh, indent=1)


def _check_spec(out, a, hand_mode, files):
    """A shard may only write into a dir whose spec matches its own configuration."""
    p = os.path.join(out, BUILD_SPEC_NAME)
    if not os.path.isfile(p):
        raise SystemExit(f"[refuse] {out} has no {BUILD_SPEC_NAME}: run --init first")
    spec = json.load(open(p))
    mine = _spec_of(a, hand_mode, files)
    bad = [(k, spec.get(k), mine[k]) for k in _SPEC_FIELDS
           if json.dumps(spec.get(k), sort_keys=True, default=float) != json.dumps(mine[k], sort_keys=True, default=float)]
    if bad:
        raise SystemExit("[refuse] shard configuration differs from the build spec -- "
                         + "; ".join(f"{k}: dir has {v1!r}, shard would write {v2!r}" for k, v1, v2 in bad))
    unknown = [f for f in mine["gen_files"] if f not in set(spec.get("gen_files", []))]
    if unknown:
        raise SystemExit(f"[refuse] {len(unknown)} gen npz in this shard are not in the spec (e.g. {unknown[0]})")
    return spec


def _finalize(out, allow_partial=False):
    """Merge manifest.part-*.json -> manifest.json, atomically, with a recounted item count."""
    sp = os.path.join(out, BUILD_SPEC_NAME)
    if not os.path.isfile(sp):
        raise SystemExit(f"[refuse] {out} has no {BUILD_SPEC_NAME}: nothing to finalize")
    spec = json.load(open(sp))
    parts = sorted(glob.glob(os.path.join(out, PART_GLOB)))
    if not parts:
        raise SystemExit(f"[refuse] {out} has no {PART_GLOB}: no shard has completed")
    per_sub, ckpts, dialects, srcs, seen, n_claimed, item_windows = {}, set(), set(), set(), {}, 0, {}
    for p in parts:
        j = json.load(open(p))
        item_windows.update(j.get("item_windows") or {})
        for k in _SPEC_FIELDS:
            if json.dumps(j.get(k), sort_keys=True, default=float) != json.dumps(spec.get(k), sort_keys=True, default=float):
                raise SystemExit(f"[refuse] {p}: {k}={j.get(k)!r} but the build spec says {spec.get(k)!r}")
        alien = [f for f in j["gen_files"] if f not in set(spec.get("gen_files", []))]
        if alien:
            raise SystemExit(f"[refuse] {os.path.basename(p)} claims gen npz outside the spec (e.g. {alien[0]})")
        for f in j["gen_files"]:
            if f in seen:
                raise SystemExit(f"[refuse] {f} was processed by two shards ({seen[f]}, {os.path.basename(p)})")
            seen[f] = os.path.basename(p)
        for s, c in j["windows_per_subset"].items():
            per_sub[s] = per_sub.get(s, 0) + c
        ckpts.update(j.get("gen_ckpts", [])); dialects.update(j.get("root_dialect", [])); srcs.update(j.get("hand_source", []))
        n_claimed += j["n_windows"]
    missing = [f for f in spec.get("gen_files", []) if f not in seen]
    if missing and not allow_partial:
        raise SystemExit(f"[refuse] {len(missing)}/{len(spec.get('gen_files', []))} gen npz were never processed "
                         f"(e.g. {missing[:3]}); re-run the missing shards or pass --allow_partial")
    on_disk = count_items(out)
    if on_disk != n_claimed:
        raise SystemExit(f"[refuse] shard manifests account for {n_claimed} items but {on_disk} *.npz are on disk")
    item_files = sorted(glob.glob(os.path.join(out, "*.npz")))
    verify_items(out, tuple(spec.get(k) for k in ITEM_STAMP_FIELDS), files=item_files)
    manifest = dict(spec, bank_schema=BANK_SCHEMA, mask_version=MASK_VERSION, fields=BANK_FIELDS,
                    items_sha256=items_digest(item_files), builder="pami/refine/build_real_bank.py",
                    finalized=datetime.datetime.now().isoformat(timespec="seconds"), finalize_argv=sys.argv,
                    complete=True, partial=bool(missing), missing_gen_files=missing, n_shards=len(parts),
                    shard_parts=[os.path.basename(p) for p in parts], subsets=sorted(per_sub),
                    windows_per_subset=per_sub, n_windows=n_claimed, n_items_on_disk=on_disk,
                    item_windows=(item_windows or None), root_dialect=sorted(dialects), gen_ckpts=sorted(ckpts),
                    hand_source=sorted(srcs), part_names=list(PARTS), foot_joints=list(FOOT_JOINTS))
    atomic_write_json(os.path.join(out, MANIFEST_NAME), manifest)
    print(f"FINALIZED: {on_disk} items from {len(parts)} shards -> {out}/{MANIFEST_NAME} "
          f"({BANK_SCHEMA} / {MASK_ID}, stream={manifest['stream']})", flush=True)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="build a refiner bank from PamiGen samples (sharded protocol)")
    ap.add_argument("--gen_glob", help="gen_*.npz glob (required for --init / --append)")
    ap.add_argument("--out", required=True, help="bank items directory")
    ap.add_argument("--stream", default="gen", choices=["gen", "roundtrip"],
                    help="gen = the generation (real stream); roundtrip = its gt_* twin (corruption stream)")
    ap.add_argument("--init", action="store_true", help="create a FRESH bank dir and write its build spec")
    ap.add_argument("--append", action="store_true", help="write items into an --init'ed dir (concurrent shards ok)")
    ap.add_argument("--finalize", action="store_true", help="merge the shard manifests into manifest.json")
    ap.add_argument("--shard", default=None, metavar="I/N", help="process files[I::N] (with --append)")
    ap.add_argument("--shard_tag", default=None, help="name of this shard's part manifest (default from --shard)")
    ap.add_argument("--allow_partial", action="store_true", help="--finalize: accept missing gen files")
    ap.add_argument("--varlen", action="store_true",
                    help="one window per gen file at min(512, floor32(T)) (what the shipped banks use); "
                         "default = fixed 64-frame windows, stride 32")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N gen files (smoke tests)")
    a = ap.parse_args(argv)
    use_interact_config()
    if a.finalize:
        assert not (a.init or a.append), "--finalize is its own pass"
        _finalize(a.out, a.allow_partial)
        return
    assert a.gen_glob, "--gen_glob is required"
    assert not (a.init and a.append), "--init and --append are separate invocations"
    files_all = sorted(glob.glob(a.gen_glob))[: a.limit]
    assert files_all, f"no gen npz matched {a.gen_glob}"
    files, tag = files_all, a.shard_tag or "single"
    if a.shard:
        i, nsh = (int(x) for x in a.shard.split("/"))
        files = files_all[i::nsh]
        tag = a.shard_tag or f"{i}_of_{nsh}"
        print(f"[shard {i}/{nsh}] {len(files)} files", flush=True)
    hm0 = check_hand_mode(_npz_hand_mode(files_all[0]), where=f"{files_all[0]}: ")
    if a.append:
        _check_spec(a.out, a, hm0, files)
    else:
        _write_spec(a.out, _spec_of(a, hm0, files_all))
        if a.init:
            print(f"INIT: {a.out} ready for {len(files_all)} gen npz ({BANK_SCHEMA} / {MASK_ID}, "
                  f"stream={a.stream}). Now run --append shards, then --finalize.", flush=True)
            return
    s2s = interact_common().seq_subset_map()
    per_sub_n, ckpts, root_dialects, hand_srcs, item_windows, done_files = {}, set(), set(), set(), {}, []
    cache, n = {}, 0
    K = (lambda k: "gt_" + k) if a.stream == "roundtrip" else (lambda k: k)
    for f in files:
        z = np.load(f, allow_pickle=True)
        seq = str(z["seq"])
        if z["pobj"].shape[0] == 0:
            print(f"[skip] {f}: M=0 (no active object)", flush=True)
            done_files.append(os.path.basename(f))
            continue
        assert z["pobj"].shape[0] == 1, f"{f}: M={z['pobj'].shape[0]} objects; builder is M=1-only"
        hm = check_hand_mode(str(z["hand_mode"]) if "hand_mode" in z.files else "wlocal",
                             int(z["hand"].shape[1]), where=f"{f}: ")
        ck = str(z["ckpt"]) if "ckpt" in z.files else ""
        ckpts.add(ck)
        root_dialects.add("rootabs" if "rootabs" in ck else "delta")
        meta = seq_meta(seq, cache)
        hfull = z[K("human_272")].astype(np.float32)
        T = hfull.shape[0]
        pobj_c = pobj_to_per_frame(z[K("pobj")].astype(np.float32), hfull)   # piece -> per-frame canonical ONCE
        robj_c = robj_to_per_frame(z[K("robj")].astype(np.float32), hfull)
        gtag = os.path.basename(f).replace(".npz", "").replace(f"_{seq}", "")
        sub = s2s.get(seq, "?")
        if a.varlen:
            wf = min(MAX_VARLEN, (T // 32) * 32)
            starts = [0] if wf >= 64 else []
            if not starts:
                print(f"[skip] {os.path.basename(f)}: T={T} < 64 frames -- no varlen window", flush=True)
        else:
            wf, starts = None, range(0, max(T - WIN + 1, 1), STRIDE)
        for f0 in starts:
            it = build_item(z, f0, meta, cache, pobj_c, robj_c, K=K, hand_mode=hm, win=wf)
            if it is None:
                continue
            it.pop("_feet")
            per_sub_n[sub] = per_sub_n.get(sub, 0) + 1
            hand_srcs.add(it["hand_source"])
            n += 1
            it["win"] = np.int32(wf if a.varlen else WIN)
            name = f"{seq}_{gtag}_t{it['f0']}.npz"
            item_windows[name] = int(it["win"])
            np.savez_compressed(os.path.join(a.out, name), **it)
            cov = it["contact_intent"].mean(0)
            print(f"{seq} t{it['f0']}: intent% " + " ".join(f"{PARTS[p]}={cov[p]:.2f}" for p in range(5))
                  + f" segs={len(it['segment_anchor'])} fit={it['ik_err_mm']:.1f}mm", flush=True)
        done_files.append(os.path.basename(f))
    spec = json.load(open(os.path.join(a.out, BUILD_SPEC_NAME)))
    part = dict({k: spec[k] for k in _SPEC_FIELDS}, shard_tag=tag, shard=a.shard, builder_git_commit=_git_commit(),
                created=datetime.datetime.now().isoformat(timespec="seconds"), argv=sys.argv,
                gen_files=done_files, n_gen_files=len(done_files), windows_per_subset=per_sub_n, n_windows=n,
                item_windows=item_windows, root_dialect=sorted(root_dialects), gen_ckpts=sorted(ckpts),
                hand_source=sorted(hand_srcs))
    atomic_write_json(os.path.join(a.out, f"{PART_PREFIX}{tag}.json"), part)
    print(f"SHARD DONE [{tag}]: {n} items from {len(done_files)} gen npz -> {a.out}", flush=True)
    if not a.append:                                   # one-shot build: finalize immediately
        _finalize(a.out, a.allow_partial)


if __name__ == "__main__":
    main()
