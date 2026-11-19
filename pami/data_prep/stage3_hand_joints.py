"""Stage 3 — hand_joints.npz.

FK the subset's native body model WITH the real hand poses, take the 30 finger joints (SMPL-H Jtr 52
-> [22:52]; SMPL-X Jtr 55 -> [25:55]; both MANO order: L index/middle/pinky/ring/thumb x3, then R) and
apply EXACTLY the per-frame de-heading transform stage 1 uses for the anchors.

  hand            (T,30,3) canonical-frame finger joint positions
  hand_rot6d      (T,30,6) parent-relative finger rotations, row-6D (first two ROWS; same convention
                  as the 272's body block)
  hand_pose_local (T,90)   the axis-angle actually fed to FK (hand mean applied where needed)
  quality / source / model  metadata (quality -> per-sequence hand-loss weight)
"""
import os

import numpy as np
import torch

from . import paths as P
from .fk import fk_global_np, hand_mean_cached
from .subsets import SUBSETS, body_model, heading_frame, load_seq, seq_subset_map

HAND_SLICE = {"smplh": (52, 22), "smplx": (55, 25)}       # (expected Jtr count, hand start)


def process(seq, out_root=P.PAMI_DATA_ROOT):
    from scipy.spatial.transform import Rotation as Rot
    d = load_seq(seq); T = d["T"]
    out_dir = f"{out_root}/output/{seq}"; os.makedirs(out_dir, exist_ok=True)
    hp = d["poses"][:, 66:156].copy()
    if d["hand_mean"]:
        hp = hp + hand_mean_cached(d["model"], d["gender"])[None]
    bm = body_model(d["gender"], num_betas=d["nb"], model=d["model"])
    betas = torch.from_numpy(np.repeat(d["betas16"][None], T, 0)).float()
    with torch.no_grad():
        o = bm(pose_body=torch.from_numpy(d["poses"][:, 3:66]).float(),
               pose_hand=torch.from_numpy(hp).float(),
               root_orient=torch.from_numpy(d["poses"][:, :3]).float(),
               trans=torch.from_numpy(d["trans"]).float(), betas=betas)
        o0 = bm(betas=betas[:1])
    Jtr = o.Jtr.numpy()
    nJ, h0 = HAND_SLICE[d["model"]]
    assert Jtr.shape[1] == nJ, f"{seq}: {d['model']} Jtr has {Jtr.shape[1]} joints, expected {nJ}"
    handw = Jtr[:, h0:h0 + 30].astype(np.float64)

    Jrest = o0.Jtr.numpy()[0, :22].astype(np.float64)
    parents = bm.kintree_table[0].long().numpy()[:22]
    Rg, pg = fk_global_np(d["poses"][:, :66], d["trans"], Jrest, parents)
    floor = float(pg[:, :22, 1].min())
    Hrot, shift = heading_frame(Rg[:, 0].astype(np.float64), pg[:, 0].astype(np.float64), floor)
    handc = np.einsum("tij,tkj->tki", Hrot, handw - shift[:, None, :])

    Rm = Rot.from_rotvec(hp.reshape(-1, 3)).as_matrix().reshape(T, 30, 3, 3)
    rot6d = Rm[..., :2, :].reshape(T, 30, 6)

    tmp = f"{out_dir}/hand_joints.npz.tmp{os.getpid()}.npz"
    np.savez(tmp, hand=handc.astype(np.float32), hand_rot6d=rot6d.astype(np.float32),
             hand_pose_local=hp.astype(np.float32), quality=np.array(d["hand_quality"]),
             source=np.array(f"{d['subset']}:{d['model']}{d['nb']}"), model=np.array(d["model"]))
    os.replace(tmp, f"{out_dir}/hand_joints.npz")
    return handc


def run(seqs=None, subset=None, shard=None, force=False, out_root=P.PAMI_DATA_ROOT):
    if seqs:
        todo = list(seqs)
    elif subset:
        todo = sorted(os.listdir(f"{SUBSETS[subset]['root']}/sequences_canonical"))
    else:
        todo = sorted(seq_subset_map())
    if shard:
        si, sn = map(int, shard.split("/")); todo = todo[si::sn]
    done = fail = 0
    for i, s in enumerate(todo):
        out = f"{out_root}/output/{s}/hand_joints.npz"
        if not force and os.path.exists(out):
            done += 1; continue
        try:
            process(s, out_root); done += 1
        except Exception as e:
            fail += 1; print(f"[FAIL] {s}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"[hand_joints] {i+1}/{len(todo)} done={done} fail={fail}", flush=True)
    print(f"[hand_joints] FINISHED {done}/{len(todo)} (fail {fail})")
