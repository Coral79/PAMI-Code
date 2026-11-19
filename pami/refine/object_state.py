"""Object half of the state: posed surface samples, joint-to-surface queries, bounded SE(3) residual.

Gradients reach BOTH the joints and the object pose; only the nearest-point ARGMIN is detached
(a discrete choice), the distance to the chosen point is smooth in both.
"""
import torch


def pose_surface(canon, R_obj, p_obj):
    """canon (S,3) | (B,M,S,3) canonical surface samples, R_obj (B,M,W,3,3), p_obj (B,M,W,3)
    -> (B,M,W,S,3) window-first points, differentiable in both pose arguments."""
    if canon.dim() == 2:
        canon = canon[None, None]
    return torch.einsum("bmwij,bmsj->bmwsi", R_obj, canon) + p_obj[..., None, :]


def nearest_surface(joints, surf, chunk=4096):
    """joints (B,W,J,3), surf (B,M,W,S,3) -> d (B,M,W,J), vec (B,M,W,J,3) joint->nearest point,
    idx (B,M,W,J) detached. Chunked over frames ((B,M,W,J,S) at S=2048 is 30 GB at once)."""
    B, M, W, S, _ = surf.shape
    J = joints.shape[2]
    ds, vs, ix = [], [], []
    step = max(1, chunk // max(S, 1))
    for w0 in range(0, W, step):
        w1 = min(w0 + step, W)
        jt = joints[:, None, w0:w1]
        sf = surf[:, :, w0:w1]
        with torch.no_grad():
            i = torch.cdist(jt.expand(-1, M, -1, -1, -1).reshape(B * M * (w1 - w0), J, 3),
                            sf.reshape(B * M * (w1 - w0), S, 3)).argmin(-1)
            i = i.reshape(B, M, w1 - w0, J)
        near = torch.gather(sf, 3, i[..., None].expand(-1, -1, -1, -1, 3))
        v = near - jt
        ds.append(v.norm(dim=-1)); vs.append(v); ix.append(i)
    return torch.cat(ds, 2), torch.cat(vs, 2), torch.cat(ix, 2)


def bounded_delta_se3(raw, cap_m=0.05, cap_rad=0.1745):
    """Raw head output (...,6) -> LOCAL residual [dp (3), dw (3)] with |dp| <= cap_m, |dw| <= cap_rad.
    The bound is on the norm (a ball, not a per-component box); tanh(n)/n is continued to n = 0 by
    its series because the zero-initialized head starts exactly there."""
    def _ball(x, cap):
        n = x.norm(dim=-1, keepdim=True)
        s = torch.where(n < 1e-4, 1.0 - n ** 2 / 3.0, torch.tanh(n) / n.clamp(min=1e-12))
        return cap * s * x
    return torch.cat([_ball(raw[..., :3], cap_m), _ball(raw[..., 3:], cap_rad)], -1)
