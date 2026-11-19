"""Stage 4 — shape_aligned.npz.

The refiner needs an outward face normal per stored surface sample (object_6dof samp_k). Instead of
replaying the sampler RNG, the STORED points are projected onto the mesh (trimesh closest_point ->
triangle id); the projection distance doubles as an audit trail (hard-fail > 1 mm).
  face_idx_k (2048,) int32 | normal_k (2048,3) float32 (flipped to outward when signed volume < 0)
  proj_err_k (2048,) float32 | meta_k json: watertight / winding_consistent / signed_volume / flipped /
  normals_trusted (False for winding-inconsistent meshes -> penetration terms skip them)
"""
import hashlib
import json
import os

import numpy as np
import trimesh

from . import paths as P

_CACHE = {}


def _mesh_bundle(V, F):
    key = hashlib.sha1(V.tobytes() + F.tobytes()).hexdigest()
    if key in _CACHE:
        return _CACHE[key]
    m = trimesh.Trimesh(V.astype(np.float64), F.astype(np.int64), process=False)
    vol = float(m.volume)
    flipped = vol < 0
    fn = np.asarray(m.face_normals, dtype=np.float32)
    if flipped:
        fn = -fn
    meta = dict(watertight=bool(m.is_watertight), winding_consistent=bool(m.is_winding_consistent),
                signed_volume=vol, flipped=flipped, normals_trusted=bool(m.is_winding_consistent))
    _CACHE[key] = (m, fn, meta)
    return _CACHE[key]


def build_one(seq, out_root=P.PAMI_DATA_ROOT, force=False):
    src = f"{out_root}/output/{seq}/object_6dof.npz"
    dst = f"{out_root}/output/{seq}/shape_aligned.npz"
    if not os.path.exists(src):
        return "no_6dof"
    if os.path.exists(dst) and not force:
        return "skip"
    z = np.load(src, allow_pickle=True)
    ks = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("samp_"))
    out = {}
    worst = 0.0
    for k in ks:
        V, F, samp = z[f"canon_{k}"], z[f"faces_{k}"], z[f"samp_{k}"].astype(np.float64)
        if len(F) == 0:
            out[f"face_idx_{k}"] = np.full(len(samp), -1, np.int32)
            out[f"normal_{k}"] = np.zeros((len(samp), 3), np.float32)
            out[f"proj_err_{k}"] = np.zeros(len(samp), np.float32)
            out[f"meta_{k}"] = json.dumps(dict(no_faces=True, normals_trusted=False))
            continue
        m, fn, meta = _mesh_bundle(V, F)
        _, dist, tid = trimesh.proximity.closest_point(m, samp)
        out[f"face_idx_{k}"] = tid.astype(np.int32)
        out[f"normal_{k}"] = fn[tid]
        out[f"proj_err_{k}"] = dist.astype(np.float32)
        out[f"meta_{k}"] = json.dumps(meta)
        worst = max(worst, float(dist.max()))
    if worst > 1e-3:
        return f"FAIL proj_err {worst:.2e}"
    np.savez_compressed(dst, **out)
    return f"ok worst={worst:.1e}"


def run(seqs=None, shard=None, force=False, out_root=P.PAMI_DATA_ROOT):
    todo = list(seqs) if seqs else sorted(d for d in os.listdir(f"{out_root}/output")
                                          if os.path.exists(f"{out_root}/output/{d}/object_6dof.npz"))
    if shard:
        i, n = map(int, shard.split("/")); todo = todo[i::n]
    stat = {}
    for j, s in enumerate(todo):
        r = build_one(s, out_root, force)
        stat[r.split()[0]] = stat.get(r.split()[0], 0) + 1
        if r.startswith("FAIL"):
            print(f"[{j}] {s}: {r}", flush=True)
        elif j % 500 == 0:
            print(f"[{j}/{len(todo)}] {s}: {r} | cache={len(_CACHE)}", flush=True)
    print("[shape_aligned] DONE", stat, flush=True)
    return stat
