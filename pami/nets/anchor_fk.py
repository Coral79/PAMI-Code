"""Torch, DIFFERENTIABLE forward-kinematics: derive per-frame de-headed anchor transforms (p, R) for
the 6 body parts DIRECTLY from a (denormalized) 272 human vector -- GT or model-RECONSTRUCTED -- by
replaying the SAME parent-relative joint-rotation chain SMPL uses (in the 272's already-de-headed local/canonical frame, so no extra transl is added).

Used to compute the object combine's anchor from the model's OWN reconstructed human instead of the
GT anchor_p/anchor_R. Not @torch.no_grad(): gradients CAN flow back
into the human branch from the object position loss when anchor_source="recon" and not detached.
"""
import numpy as np
# chumpy (inside the legacy SMPL pkl) needs numpy aliases removed in numpy>=1.24;
# no-op where the aliases still exist. Placed BEFORE the smplx import (defensive, in case smplx ever
# imports chumpy eagerly).
for _a, _v in (("bool", bool), ("int", int), ("float", float), ("complex", complex),
               ("object", object), ("unicode", str), ("str", str)):
    if not hasattr(np, _a): setattr(np, _a, _v)
import torch, smplx
from .. import config as C
import sys
from pami.vendor.eval_trans import rotation_6d_to_matrix

ANCH_JOINT = {"root": 0, "torso": 9, "Larm": 20, "Rarm": 21, "Lleg": 7, "Rleg": 8}   # SMPL 24-joint idx (smpl_anchor_fk.py)
PARTS = ["root", "torso", "Larm", "Rarm", "Lleg", "Rleg"]
SMPL_DIR = C.SMPL_BODY_DIR


class AnchorFK:
    def __init__(self, dev):
        self.dev = dev
        m = smplx.create(SMPL_DIR, model_type="smpl", gender="neutral", num_betas=10, batch_size=1)
        self.parents = torch.from_numpy(m.parents.cpu().numpy()[:22].astype(np.int64)).to(dev)  # (22,) SMPL joint 0-21
        # fallback rest skeleton (mean shape); every caller passes the sequence's own jrest instead
        with torch.no_grad():
            out0 = m(betas=torch.zeros(1, 10))
        self.Jrest = out0.joints[0, :22].to(dev)                      # (22,3) rest joint offsets, mean shape

    def __call__(self, human272, jrest=None, return_joints=False):
        """human272 (B,W,272) denormalized -> anchor_p (B,W,6,3), anchor_R (B,W,6,3,3). Differentiable.
        jrest (B,22,3) optional PER-SAMPLE rest skeleton (InterAct: per-seq gendered
        SMPL-X betas16 rest joints, served by the dataset from canon_anchors.npz — a single
        mean-shape skeleton is 31-85 mm wrong for omomo subjects). None -> the cached mean-shape buffer, and the
        jrest-None branch below keeps the exact original einsum shapes => bit-identical old behavior.
        NOTE the SMPL and SMPL-X body kinematic chains for joints 0..21 share the SAME parents array
        (verified: smplx.create SMPL parents[:22] == human_body_prior SMPL-X
        kintree_table[0][:22]), so one FK loop serves both skeletons."""
        B, W, _ = human272.shape
        pelvis = human272[..., 8:11]                                              # (B,W,3) canonical pelvis pos
        rot6 = human272[..., 140:140+22*6].reshape(B, W, 22, 6)
        Rloc = rotation_6d_to_matrix(rot6)                                        # (B,W,22,3,3) parent-relative (SMPL pose convention)
        if jrest is None:
            Jr = self.Jrest                                                       # (22,3) mean-shape buffer
            bone = lambda p, j: torch.einsum("bwij,j->bwi", Rg_list[p], (Jr[j] - Jr[p]))
            p0 = Jr[0].expand(B, W, 3)
        else:
            Jr = jrest.to(dtype=human272.dtype, device=human272.device)           # (B,22,3) per-sample
            bone = lambda p, j: torch.einsum("bwij,bj->bwi", Rg_list[p], (Jr[:, j] - Jr[:, p]))
            p0 = Jr[:, None, 0].expand(B, W, 3)
        Rg_list = [None] * 22; pg_list = [None] * 22
        Rg_list[0] = Rloc[:, :, 0]; pg_list[0] = p0
        for j in range(1, 22):
            p = int(self.parents[j].item())
            Rg_list[j] = torch.einsum("bwij,bwjk->bwik", Rg_list[p], Rloc[:, :, j])
            pg_list[j] = pg_list[p] + bone(p, j)
        Rg = torch.stack(Rg_list, 2); pg = torch.stack(pg_list, 2)                 # (B,W,22,3,3),(B,W,22,3)
        pg = pg + (pelvis - pg[:, :, 0])[:, :, None]                              # align rest-pelvis -> canonical pelvis (matches render_recon.smpl_verts fix)
        idx = [ANCH_JOINT[p] for p in PARTS]
        if return_joints:
            # foot-slide FK: expose ALL 22 canonical FK joint positions. Same tensors the
            # anchor path already computes -- zero extra FK cost for a consumer that needs feet 7/8/10/11.
            return pg[:, :, idx], Rg[:, :, idx], pg
        return pg[:, :, idx], Rg[:, :, idx]                                       # (B,W,6,3),(B,W,6,3,3)
