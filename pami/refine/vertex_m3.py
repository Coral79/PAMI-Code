"""Differentiable SMPL-H/X body-model forward for the vertex losses and the export.

verts_from_state: subsampled (SUB=1024, fixed seeded permutation per model) window-first vertices,
grouped by (gender, num_betas) inside a batch (ModelBatchSampler guarantees one MODEL per batch).
full_verts_joints: the full mesh + LIGHT-52 joints of ONE sample (the export's joints52).
"""
import numpy as np
import torch

from pami.refine.base_state import so3_log
from pami.refine.hand_state import HAND_H0
from pami.refine.paths import interact_common
from pami.vendor.eval_trans import rotation_6d_to_matrix

SUB = 1024
_BM, _SUBIDX, _HANDSUB = {}, {}, {}


def _bm(gender, model, nb, dev):
    """Cached body model (parameters frozen)."""
    key = (gender, model, nb, str(dev))
    if key not in _BM:
        _BM[key] = interact_common().body_model(gender, num_betas=nb, model=model).to(dev)
        for p in _BM[key].parameters():
            p.requires_grad_(False)
    return _BM[key]


def _sub_idx(model, n_verts, dev):
    if model not in _SUBIDX:
        g = np.random.RandomState(0)
        _SUBIDX[model] = torch.from_numpy(np.sort(g.permutation(n_verts)[:SUB]).astype(np.int64))
    return _SUBIDX[model].to(dev)


def _aa(rot6):
    return so3_log(rotation_6d_to_matrix(rot6))


def verts_from_state(rot6, rot6h, pel, betas16, genders, model, nb_list):
    """rot6 (B,W,22,6) | rot6h (B,W,30,6) | pel (B,W,3) | betas16 (B,16) | genders list[str] |
    model 'smplh'|'smplx' | nb_list list[int] -> (B,W,SUB,3) window-first vertices, differentiable.
    The forward runs with trans = 0 and the vertices are shifted so the pelvis joint lands on `pel`."""
    B, W = rot6.shape[:2]
    dev = rot6.device
    aa_root = _aa(rot6[:, :, 0]).reshape(B * W, 3)
    aa_body = _aa(rot6[:, :, 1:22]).reshape(B * W, 63)
    aa_hand = _aa(rot6h).reshape(B * W, 90)
    out = torch.empty(B, W, SUB, 3, device=dev, dtype=rot6.dtype)
    groups = {}
    for i, (g, nb) in enumerate(zip(genders, nb_list)):
        groups.setdefault((g, int(nb)), []).append(i)
    for (g, nb), idxs in groups.items():
        bm = _bm(g, model, nb, dev)
        sub = _sub_idx(model, bm.shapedirs.shape[0], dev)
        ii = torch.tensor(idxs, device=dev)
        fr = (ii[:, None] * W + torch.arange(W, device=dev)[None]).reshape(-1)
        bt = betas16[ii][:, None, :nb].expand(-1, W, nb).reshape(-1, nb)
        o = bm(root_orient=aa_root[fr], pose_body=aa_body[fr], pose_hand=aa_hand[fr], betas=bt)
        shift = pel.reshape(B * W, 3)[fr][:, None] - o.Jtr[:, 0:1]
        out[ii] = (o.v[:, sub] + shift).reshape(len(idxs), W, SUB, 3)
    return out


def hand_sub_mask(model, dev):
    """(SUB,) bool: which of this model's SUB sampled vertices belong to the hand region (nearest
    rest joint is a finger joint; zero betas, male -- membership is insensitive to shape)."""
    if model not in _HANDSUB:
        bm = _bm("male", model, 10, "cpu")
        z = lambda n: torch.zeros(1, n)
        o = bm(root_orient=z(3), pose_body=z(63), pose_hand=z(90), betas=z(10))
        v, j = o.v[0], o.Jtr[0]
        sub = _sub_idx(model, v.shape[0], "cpu")
        _HANDSUB[model] = (torch.cdist(v[sub], j).argmin(1) >= HAND_H0[model])
    return _HANDSUB[model].to(dev)


def full_verts_joints(rot6, rot6h, pel, betas16, gender, model, nb, dev):
    """ONE sample: rot6 (W,22,6), rot6h (W,30,6), pel (W,3), betas16 (16,) -> (verts (W,V,3), faces,
    joints52 (W,52,3)) window-first, pelvis pinned to pel. SMPL-H: Jtr[:, :52] is already the
    LIGHT order; SMPL-X: body 0:22 + hands 25:55."""
    W = rot6.shape[0]
    bm = _bm(gender, model, nb, dev)
    o = bm(root_orient=_aa(rot6[:, 0]), pose_body=_aa(rot6[:, 1:22]).reshape(W, 63),
           pose_hand=_aa(rot6h).reshape(W, 90), betas=betas16[None, :nb].expand(W, nb))
    shift = pel[:, None] - o.Jtr[:, 0:1]
    idx = list(range(22)) + list(range(25, 55)) if model == "smplx" else list(range(52))
    J = (o.Jtr[:, idx] + shift).cpu().numpy()
    return (o.v + shift).cpu().numpy(), np.asarray(bm.f.cpu() if torch.is_tensor(bm.f) else bm.f), J
