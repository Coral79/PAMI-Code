"""Body-model forward kinematics helpers."""
import numpy as np
import torch

from .subsets import body_model, hand_means, model_npz_path

_BM = {}
_HMEAN = {}


def bm_cached(gender, model="smplx", nb=16, dev="cpu"):
    k = (model, gender, nb, str(dev))
    if k not in _BM:
        _BM[k] = body_model(gender, num_betas=nb, model=model).to(dev)
    return _BM[k]


def hand_mean_cached(model, gender):
    """(90,) MANO hand means for subsets canonicalized with flat_hand_mean=False (chairs, intercap)."""
    k = (model, gender)
    if k not in _HMEAN:
        _HMEAN[k] = hand_means(model_npz_path(model, gender))
    return _HMEAN[k]


def fk_global_np(poses66, trans, Jrest, parents):
    """Rigid FK chain -> per-frame GLOBAL joint rotations (T,22,3,3) + positions (T,22,3) in world
    (SMPL / SMPL-H / SMPL-X share the joint order 0..21)."""
    from scipy.spatial.transform import Rotation as Rot
    T = poses66.shape[0]
    J = 22
    Rloc = Rot.from_rotvec(poses66.reshape(-1, 3)).as_matrix().reshape(T, J, 3, 3)
    Rg = np.zeros((T, J, 3, 3)); pg = np.zeros((T, J, 3))
    Rg[:, 0] = Rloc[:, 0]; pg[:, 0] = Jrest[0]
    for j in range(1, J):
        p = parents[j]
        Rg[:, j] = np.einsum("tij,tjk->tik", Rg[:, p], Rloc[:, j])
        pg[:, j] = pg[:, p] + np.einsum("tij,j->ti", Rg[:, p], (Jrest[j] - Jrest[p]))
    return Rg, pg + trans[:, None, :]


def fk_seq(d, dev="cpu", chunk=512):
    """Gendered BodyModel FK with REAL hand poses (subset-aware: adds the MANO hand mean where the
    stored hand pose excludes it). Returns world verts (T,V,3) f32, body joints Jtr (T,22,3),
    rest joints Jrest (22,3) f64, parents (22,)."""
    bm = bm_cached(d["gender"], d["model"], d["nb"], dev)
    T = d["T"]
    poses = d["poses"]
    if d["hand_mean"]:
        poses = poses.copy()
        poses[:, 66:156] += hand_mean_cached(d["model"], d["gender"])[None]
    Pz = torch.from_numpy(poses).float().to(dev)
    tr = torch.from_numpy(d["trans"]).float().to(dev)
    be = torch.from_numpy(d["betas16"]).float().to(dev)[None]
    vs, js = [], []
    with torch.no_grad():
        for a in range(0, T, chunk):
            b = min(T, a + chunk)
            o = bm(pose_body=Pz[a:b, 3:66], pose_hand=Pz[a:b, 66:156], root_orient=Pz[a:b, :3],
                   trans=tr[a:b], betas=be.expand(b - a, -1))
            vs.append(o.v.cpu().numpy().astype(np.float32))
            js.append(o.Jtr[:, :22].cpu().numpy())
        o0 = bm(betas=be)                                            # zero pose -> rest joints
    hv = np.concatenate(vs); Jtr = np.concatenate(js)
    Jrest = o0.Jtr[0, :22].cpu().numpy().astype(np.float64)
    parents = bm.kintree_table[0].long().cpu().numpy()[:22]
    return hv, Jtr, Jrest, parents
