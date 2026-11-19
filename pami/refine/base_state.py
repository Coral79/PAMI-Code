"""The refiner's single authoritative theta -> FK chain for the body (SMPL-H/X joints 0..21).

RefineNet outputs JOINT ROTATION residuals, so every quantity it reads, is supervised on, or exports
comes from ONE chain: parent-relative rot6d -> FK -> joint positions. Only the pelvis translation is
an explicit state (the global placement); the 272 position block is never read.
"""
import torch

from pami.vendor.eval_trans import rotation_6d_to_matrix

# SMPL / SMPL-H / SMPL-X share this kinematic chain for joints 0..21 (verified against both
# body models' kintree_table and pami.nets.anchor_fk.AnchorFK.parents).
SMPL_PARENTS = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
                            dtype=torch.long)


def so3_exp(w, eps=1e-8):
    """Rodrigues: (...,3) axis-angle -> (...,3,3). The Taylor branch keeps the gradient finite at
    w = 0, which is where a zero-initialized residual head starts."""
    th = w.norm(dim=-1, keepdim=True)
    small = th < 1e-4
    a = torch.where(small, 1.0 - th ** 2 / 6.0, torch.sin(th) / th.clamp(min=eps))
    b = torch.where(small, 0.5 - th ** 2 / 24.0, (1.0 - torch.cos(th)) / (th ** 2).clamp(min=eps))
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    O = torch.zeros_like(wx)
    K = torch.stack([torch.stack([O, -wz, wy], -1),
                     torch.stack([wz, O, -wx], -1),
                     torch.stack([-wy, wx, O], -1)], -2)
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    return I + a[..., None] * K + b[..., None] * (K @ K)


def so3_log(R, eps=1e-7):
    """(...,3,3) -> (...,3) axis-angle, valid over the whole range including theta near pi.

    Three branches (series near 0, generic, symmetric-part near pi), computed in fp64 with atan2 so
    the backward pass is finite at both ends (torch.where evaluates every branch, so each branch's
    input is made safe where it is not selected)."""
    dt = R.dtype
    R = R.double()
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], -1)                  # 2 sin(theta) * axis
    vn = (v * v).sum(-1, keepdim=True).clamp(min=1e-24).sqrt()
    th = torch.atan2(0.5 * vn, cos[..., None])                          # (...,1) in [0, pi]
    small = th < 1e-4
    near_pi = th > (torch.pi - 1e-2)
    th_g = torch.where(small | near_pi, torch.ones_like(th), th)
    gen = v * (th_g / (2.0 * torch.sin(th_g).clamp(min=eps)))
    ser = v * 0.5 * (1.0 + th ** 2 / 6.0)
    I3 = torch.eye(3, device=R.device, dtype=R.dtype)
    sym = (R + R.transpose(-1, -2)) * 0.5
    cos_np = torch.where(near_pi[..., 0], cos, torch.full_like(cos, -0.5))
    A = (sym - cos_np[..., None, None] * I3) / (1.0 - cos_np[..., None, None]).clamp(min=eps)
    d = A.diagonal(dim1=-2, dim2=-1).clamp(min=0.0)
    k = d.argmax(-1, keepdim=True)
    col = torch.gather(A, -1, k[..., None].expand(*A.shape[:-1], 1)).squeeze(-1)
    a = col / (col * col).sum(-1, keepdim=True).clamp(min=eps ** 2).sqrt()
    a = a * torch.where((a * v).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    return torch.where(small, ser, torch.where(near_pi, a * th, gen)).to(dt)


def body_fk(rot6, pelvis, parents, jrest):
    """Differentiable full-body FK.

    rot6 (B,W,22,6) parent-relative rotations (root column in the window-first frame)
    pelvis (B,W,3) pelvis position | parents (22,) | jrest (B,22,3) or (22,3) rest skeleton
    -> pg (B,W,22,3) joint positions, Rg (B,W,22,3,3) global joint rotations."""
    B, W = rot6.shape[:2]
    Rloc = rotation_6d_to_matrix(rot6)
    per_sample = jrest.dim() == 3
    Jr = jrest.to(dtype=rot6.dtype, device=rot6.device)
    Rg = [None] * 22
    pg = [None] * 22
    Rg[0] = Rloc[:, :, 0]
    pg[0] = (Jr[:, None, 0].expand(B, W, 3) if per_sample else Jr[0].expand(B, W, 3))
    for j in range(1, 22):
        p = int(parents[j])
        Rg[j] = torch.einsum("bwij,bwjk->bwik", Rg[p], Rloc[:, :, j])
        off = (Jr[:, j] - Jr[:, p]) if per_sample else (Jr[j] - Jr[p])
        pg[j] = pg[p] + (torch.einsum("bwij,bj->bwi", Rg[p], off) if per_sample
                         else torch.einsum("bwij,j->bwi", Rg[p], off))
    pg = torch.stack(pg, 2)
    Rg = torch.stack(Rg, 2)
    pg = pg + (pelvis - pg[:, :, 0])[:, :, None]                        # rest-pelvis -> pelvis
    return pg, Rg
