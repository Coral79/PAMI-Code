"""Token features of the refiner (54 tokens: 22 body incl. pelvis context, 30 fingers, 1 object, 1 scene).

Every joint token is rot6d + a single nearest-surface probe, nothing else; the network's InputNorm
-> learnable Fourier -> Linear encoding sits on top. All frames are window-first; finger probes are
wrist-local; ground = the canonical y = 0 plane.

  body   (D3_BODY=20)   rot6d 6 | unit dir v/d 3, d 1, exp(-5d) 1, posed normal 3, body-relative
                        probe point R_root^T (p* - pelvis) 3 | height y 1, exp(-5y) 1, foot-contact-intent 1
  finger (D3_FINGER=17) rot6d 6 | wrist-local: v/d 3, d 1, exp(-5d) 1, normal 3, probe point 3
  object (D3_OBJ=9)     rot6d 6 | position 3
  scene  (D3_SCENE=25)  root rot6d 6 | pelvis 3 | betas 16 (zero-padded)
"""
import torch

from pami.refine.object_state import nearest_surface, pose_surface
from pami.rotvote import mat_to_rot6d

D3_BODY, D3_FINGER, D3_OBJ, D3_SCENE = 20, 17, 9, 25
N_BODY, N_FINGER = 22, 30
N_TOKENS_M3 = N_BODY + N_FINGER + 2                       # 54
FOOT_ROWS = {7: 0, 10: 0, 8: 1, 11: 1}                    # ankle/foot joint -> foot_contact column


def assemble_m3(rot6, rot6h, pg, Rg, pj, Rw, pw, p_obj, R_obj, canon_surf, canon_normal,
                beta16, foot_intent=None):
    """State -> (f_body (B,W,22,20), f_finger (B,W,30,17), f_obj (B,W,M,9), f_scene (B,W,25)).

    rot6 (B,W,22,6) | rot6h (B,W,30,6) | pg,Rg body FK | pj finger FK | Rw,pw wrists
    p_obj (B,M,W,3), R_obj (B,M,W,3,3) | canon_surf/canon_normal (B,M,S,3) | beta16 (B,16)
    foot_intent (B,W,2) bool or None (-> zeros). Recomputed from the CURRENT state every step;
    differentiable except the nearest-point argmin."""
    B, W = pg.shape[:2]
    M = p_obj.shape[1]
    surf = pose_surface(canon_surf, R_obj, p_obj)                       # (B,M,W,S,3)
    all_j = torch.cat([pg, pj], 2)                                      # (B,W,52,3)
    dm, vm, im = nearest_surface(all_j, surf)                           # (B,M,W,52[,3])
    k = dm.argmin(1, keepdim=True)                                      # nearest object per joint
    d = torch.gather(dm, 1, k)[:, 0]                                    # (B,W,52)
    v = torch.gather(vm, 1, k.detach()[..., None].expand(-1, -1, -1, -1, 3))[:, 0]
    S = canon_normal.shape[2]
    n_all = torch.gather(canon_normal[:, :, None].expand(B, M, W, S, 3), 3,
                         im[..., None].expand(-1, -1, -1, -1, 3))       # normal at each object's nearest sample
    n_can = torch.gather(n_all, 1, k.detach()[..., None].expand(-1, -1, -1, -1, 3))[:, 0]
    kk = k.detach()[:, 0]
    R_at = R_obj.permute(0, 2, 1, 3, 4)                                 # (B,W,M,3,3)
    R_pick = torch.gather(R_at, 2, kk[..., None, None].expand(-1, -1, -1, 3, 3))
    n_wf = torch.einsum("bwjik,bwjk->bwji", R_pick, n_can)              # rotate only
    dcl = d[..., None].clamp(min=1e-4)
    unit, expd = v / dcl, torch.exp(-5.0 * d)[..., None]
    p_star = all_j + v                                                  # nearest points

    # body tokens
    Rroot, pel = Rg[:, :, 0], pg[:, :, 0]
    p_rel = torch.einsum("bwji,bwkj->bwki", Rroot, p_star[:, :, :22] - pel[:, :, None])
    y = pg[..., 1:2]
    fint = torch.zeros(B, W, 22, 1, device=pg.device, dtype=pg.dtype)
    if foot_intent is not None:
        fi = foot_intent.to(pg.dtype)
        for j, c in FOOT_ROWS.items():
            fint[:, :, j, 0] = fi[:, :, c]
    f_body = torch.cat([rot6, unit[:, :, :22], d[..., :22, None], expd[:, :, :22],
                        n_wf[:, :, :22], p_rel, y, torch.exp(-5.0 * y), fint], -1)

    # finger tokens (wrist-local probes)
    loc_v, loc_n, loc_p = [], [], []
    for s in (0, 1):
        sl = slice(22 + s * 15, 22 + (s + 1) * 15)
        Rws, pws = Rw[:, :, s], pw[:, :, s]
        rot = lambda x: torch.einsum("bwji,bwkj->bwki", Rws, x)
        loc_v.append(rot(v[:, :, sl]))
        loc_n.append(rot(n_wf[:, :, sl]))
        loc_p.append(rot(p_star[:, :, sl] - pws[:, :, None]))
    dcf = d[..., 22:, None].clamp(min=1e-4)
    f_finger = torch.cat([rot6h, torch.cat(loc_v, 2) / dcf, d[..., 22:, None],
                          expd[:, :, 22:], torch.cat(loc_n, 2), torch.cat(loc_p, 2)], -1)

    # object / scene tokens
    f_obj = torch.cat([mat_to_rot6d(R_obj), p_obj], -1).transpose(1, 2)   # (B,W,M,9)
    b16 = beta16[:, None, :].expand(B, W, 16)
    f_scene = torch.cat([mat_to_rot6d(Rroot), pel, b16], -1)
    return f_body, f_finger, f_obj, f_scene


def sensor_points(canon_surf, canon_normal, R_obj, p_obj):
    """Canonical samples (B,M,S,3) + pose -> the sensor's world-frame cloud pts/nrm (B,W,M*S,3),
    differentiable in the object pose. Normals are rotated only (same convention as assemble_m3).
    The point set is fixed (a deterministic sample of the object's own mesh, data_m3), never re-drawn."""
    surf = pose_surface(canon_surf, R_obj, p_obj)
    n_wf = torch.einsum("bmwij,bmsj->bmwsi", R_obj, canon_normal)
    B, M, W, S = surf.shape[:4]
    return (surf.permute(0, 2, 1, 3, 4).reshape(B, W, M * S, 3),
            n_wf.permute(0, 2, 1, 3, 4).reshape(B, W, M * S, 3))
