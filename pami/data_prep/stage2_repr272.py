"""Stage 2 — the human 272-dim representation chain (face-z alignment and the 272-dim representation of
MotionStreamer, https://github.com/zju3dv/MotionStreamer, with gendered body-model joints). Per sequence <new_name> (from the index csv):

  smpl_85/<nn>.npy                       (stage 1)  [poses66 | zeros6 | trans3 | betas10]
  -> smpl_85_face_z_transform/<nn>.npy    root orientation / translation rotated so the first frame faces +Z
  -> smpl_85_face_z_transform_joints/<nn> gendered body-model joints (T,J,3), J = 52 (SMPL-H) / 55 (SMPL-X),
                                           the subset's betas (16 or 10) -- NOT the neutral SMPL-H of the
                                           stock chain, so joints / anchors / contact share ONE body
  -> Representation_272/<nn>.npy          272 = [root xz vel 2 | heading delta 6D 6 | 22 joint pos 66 |
                                           22 joint vel 66 | 22 parent-relative rot6d 132], per-frame
                                           canonical (de-headed, root XZ at origin, floor Y=0)
  -> injected as tae_sample.npz["human_272"] (float32)
"""
import csv
import os

import numpy as np
import torch

from . import paths as P
from .fk import bm_cached
from .subsets import SUBSETS, seq_subset_map
from ..vendor.face_z_align_util import (expmap_to_quaternion, matrix_to_rotation_6d, qmul_np, qrot_np,
                                        quaternion_to_axis_angle, quaternion_to_matrix_np)

from .stage1_convert import atomic_save, atomic_savez


# ---------------------------------------------------------------- face-z alignment (stock math)
def _my_quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


def _calc_heading_quat_inv(q):
    ref_dir = torch.zeros_like(q[..., 0:3]); ref_dir[..., 2] = 1
    rot_dir = _my_quat_rotate(q, ref_dir)
    heading = torch.atan2(rot_dir[..., 0], rot_dir[..., 2])
    axis = torch.zeros_like(q[..., 0:3]); axis[..., 1] = 1
    return -heading, axis


def face_z_transform(smpl_data):
    """(T,85) smpl_85 -> (T,85) with the FIRST frame's heading removed from root orient + trans."""
    seq_len = smpl_data.shape[0]
    pose_body = smpl_data[:, :72].reshape(seq_len, -1, 3)
    trans = smpl_data[:, 72:75]
    beta = smpl_data[:, 75:]
    q0 = expmap_to_quaternion(pose_body[0, 0])                        # wxyz
    q0_xyzw = torch.from_numpy(q0[[1, 2, 3, 0]]).float().unsqueeze(0)
    heading_inv, axis = _calc_heading_quat_inv(q0_xyzw)
    q_diff = expmap_to_quaternion((heading_inv * axis).numpy())
    root_q = qmul_np(q_diff.reshape(1, -1).repeat(seq_len, axis=0), expmap_to_quaternion(pose_body[:, 0]))
    root_aa = quaternion_to_axis_angle(torch.from_numpy(root_q)).numpy()
    trans = qrot_np(q_diff.reshape(1, -1).repeat(seq_len, axis=0), trans)
    return np.concatenate([root_aa, pose_body[:, 1:].reshape(seq_len, -1), trans, beta], axis=-1)


# ---------------------------------------------------------------- gendered joints
def body_joints(fz, betas, gender, model, nb, dev="cpu", chunk=512):
    """(T,85) face-z params -> (T,J,3) float32 joints of the gendered body model (hands zero)."""
    bm = bm_cached(gender, model, nb, dev)
    T = fz.shape[0]
    js = []
    with torch.no_grad():
        for a in range(0, T, chunk):
            b = min(T, a + chunk)
            o = bm(pose_body=torch.from_numpy(fz[a:b, 3:66]).float().to(dev),
                   root_orient=torch.from_numpy(fz[a:b, :3]).float().to(dev),
                   trans=torch.from_numpy(fz[a:b, 72:75]).float().to(dev),
                   betas=torch.from_numpy(np.repeat(betas[None], b - a, 0)).float().to(dev))
            js.append(o.Jtr.cpu().numpy().astype(np.float32))
    return np.concatenate(js)


# ---------------------------------------------------------------- 272 (stock math)
def _rot_yaw(yaw):
    cs = np.cos(yaw); sn = np.sin(yaw)
    return np.array([[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]])


def representation_272(joints, fz):
    """joints (T,J,3) + face-z params (T,85) -> (T,272) float64 (stock representation_272.py)."""
    root_idx = 0
    position_data = joints[:, :22, :3]
    nfrm, njoint, _ = position_data.shape
    rotations_wxyz = expmap_to_quaternion(fz[:, :66].reshape(nfrm, njoint, 3))
    rotations_matrix = quaternion_to_matrix_np(rotations_wxyz)

    ori = position_data[0, root_idx].copy()
    y_min = np.min(position_data[:, :, 1])
    ori[1] = y_min
    position_data = position_data - ori
    velocities_root = position_data[1:, root_idx, :] - position_data[:-1, root_idx, :]

    position_data[:, :, 0] -= position_data[:, 0:1, 0]
    position_data[:, :, 2] -= position_data[:, 0:1, 2]

    global_heading = -np.arctan2(rotations_matrix[:, root_idx, 0, 2], rotations_matrix[:, root_idx, 2, 2])
    global_heading_rot = np.array([_rot_yaw(x) for x in global_heading])
    global_heading_diff = global_heading[1:] - global_heading[:-1]
    global_heading_diff_rot = np.array([_rot_yaw(x) for x in global_heading_diff])

    positions_no_heading = np.matmul(np.repeat(global_heading_rot[:, None, :, :], njoint, axis=1),
                                     position_data[..., None]).squeeze(-1)
    velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1]
    velocities_root_xy_no_heading = np.matmul(global_heading_rot[:-1], velocities_root[:, :, None]).squeeze()[..., [0, 2]]
    rotations_matrix[:, 0, ...] = np.matmul(global_heading_rot, rotations_matrix[:, 0, ...])

    size_frame = 8 + njoint * 3 + njoint * 3 + njoint * 6
    final_x = np.zeros((nfrm, size_frame))
    final_x[0, 2] = 1
    final_x[0, 6] = 1
    final_x[1:, 2:8] = matrix_to_rotation_6d(torch.from_numpy(global_heading_diff_rot)).numpy()
    final_x[1:, :2] = velocities_root_xy_no_heading
    final_x[:, 8:8 + 3 * njoint] = np.reshape(positions_no_heading, (nfrm, -1))
    final_x[1:, 8 + 3 * njoint:8 + 6 * njoint] = np.reshape(velocities_no_heading, (nfrm - 1, -1))
    final_x[:, 8 + 6 * njoint:8 + 12 * njoint] = np.reshape(rotations_matrix[..., :, :2, :], (nfrm, -1))
    return final_x


# ---------------------------------------------------------------- driver
def _source_betas_gender(row):
    src = np.load(row["source_path"], allow_pickle=True)
    betas = src["betas"].astype(np.float64); gender = str(src["gender"])
    if betas.ndim == 2:                                            # grab_smplh: per-frame fitted betas -> median
        betas = np.median(betas, axis=0)
    return betas, gender


def process_row(row, r272_out, out_root, dev="cpu", inject=True):
    nn, seq = row["new_name"], row["seq_name"]
    s85 = np.load(f"{r272_out}/smpl_85/{nn}")
    if s85.shape[0] == 0:
        raise RuntimeError(f"{seq}: empty smpl_85")
    fz = face_z_transform(s85)
    os.makedirs(f"{r272_out}/smpl_85_face_z_transform", exist_ok=True)
    atomic_save(f"{r272_out}/smpl_85_face_z_transform/{nn}", fz)
    betas, gender = _source_betas_gender(row)
    cfg = SUBSETS[seq_subset_map()[seq]]
    J = body_joints(fz, betas, gender, cfg["model"], cfg["nb"], dev)
    os.makedirs(f"{r272_out}/smpl_85_face_z_transform_joints", exist_ok=True)
    atomic_save(f"{r272_out}/smpl_85_face_z_transform_joints/{nn}", J)
    x272 = representation_272(J, fz)
    os.makedirs(f"{r272_out}/Representation_272", exist_ok=True)
    atomic_save(f"{r272_out}/Representation_272/{nn}", x272)
    if inject:
        inject_272(seq, x272, out_root)
    return x272


def inject_272(seq, x272, out_root):
    """tae_sample.npz["human_272"] = x272 (float32); no-op if already identical."""
    tae = f"{out_root}/output/{seq}/tae_sample.npz"
    d = np.load(tae, allow_pickle=True)
    h272 = np.asarray(x272).astype(np.float32)
    if "human_272" in d.files and d["human_272"].shape == h272.shape and np.allclose(d["human_272"], h272):
        return False
    out = {k: d[k] for k in d.files if k != "human_272"}
    out["human_272"] = h272
    atomic_savez(tae, **out)
    return True


def run(seqs=None, out_root=P.PAMI_DATA_ROOT, r272_out=P.R272_OUT, index_csv=P.INDEX_CSV, device=None,
        inject=True):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = list(csv.DictReader(open(index_csv)))
    if seqs:
        want = set(seqs); rows = [r for r in rows if r["seq_name"] in want]
    print(f"[repr272] device={dev} rows={len(rows)} r272_out={r272_out}", flush=True)
    n = 0
    for r in rows:
        if not os.path.exists(f"{r272_out}/smpl_85/{r['new_name']}"):
            continue                                               # seq failed / not converted in stage 1
        process_row(r, r272_out, out_root, dev, inject)
        n += 1
        if n % 500 == 0:
            print(f"  {n}/{len(rows)}", flush=True)
    print(f"[repr272] done: {n} sequences")
