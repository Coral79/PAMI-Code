"""T-step recursion: re-probe -> net -> stepwise composition -> CUMULATIVE caps.

COMPOSITION (right-multiplied in the joint-/object-local frame, never additive):
    R_{t+1} = R_t @ Exp(dtheta_t)   human      p_{t+1} = p_t + R_t @ dp_t ;  R_{t+1} = R_t @ Exp(dw_t)   object
ROOT: an explicit (dx, dz, dyaw, dy) offset relative to the base root, applied as
    pel = pel_base + [dx, dy, dz];  R_root = Yaw(dyaw) @ R_root_base.
CUMULATIVE CAPS: after every step the TOTAL deviation from the ORIGINAL base state is projected back
inside the caps (joint groups: geodesic angle vs R_0; object: translation norm in the base object
frame + geodesic; root: ball / clamps). Full BPTT: nothing detaches the state between steps.
"""
import math

import torch

from pami.refine.base_state import body_fk, so3_exp, so3_log
from pami.refine.features_m3 import assemble_m3, sensor_points
from pami.refine.hand_state import hand_fk, tip_centers, wrist_from_body
from pami.rotvote import mat_to_rot6d
from pami.vendor.eval_trans import rotation_6d_to_matrix

JOINT_GROUPS = {"torso": [3, 6, 9, 12, 15], "lower": [1, 2, 4, 5, 7, 8, 10, 11],
                "upper": [13, 14, 16, 17, 18, 19, 20, 21]}
CAPS_DEFAULT = dict(torso=math.radians(20.0), lower=math.radians(30.0),
                    upper=math.radians(45.0), finger=math.radians(60.0))


def sensor_clouds(geom):
    """train_m3.batch_geom output -> the kwargs routing the SENSOR cloud + fingertip offsets into rollout()."""
    return dict(sensor_canon=geom.get("sensor_canon"), sensor_canon_n=geom.get("sensor_canon_n"),
                tip_off=geom.get("tip_off"))


def _clamp_ball(x, cap):
    n = x.norm(dim=-1, keepdim=True)
    return x * (cap / n.clamp(min=1e-9)).clamp(max=1.0)


def _yaw_mat(yaw):
    """(...,) -> (...,3,3) rotation about world +Y."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    O, I = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([c, O, s], -1), torch.stack([O, I, O], -1),
                        torch.stack([-s, O, c], -1)], -2)


class RolloutState:
    """The live state one recursion step transforms; the base (t=0) quantities are kept, detached,
    for the cumulative caps; the root deviation is explicit."""

    def __init__(self, rot6, rot6h, pel, p_obj, R_obj):
        self.rot6, self.rot6h, self.pel = rot6, rot6h, pel        # (B,W,22,6) (B,W,30,6) (B,W,3)
        self.p_obj, self.R_obj = p_obj, R_obj                     # (B,M,W,3) (B,M,W,3,3)
        self.R0 = rotation_6d_to_matrix(rot6).detach()
        self.R0h = rotation_6d_to_matrix(rot6h).detach()
        self.pel0, self.p0, self.Ro0 = pel.detach(), p_obj.detach(), R_obj.detach()
        self.droot = torch.zeros(*pel.shape[:2], 4, device=pel.device, dtype=pel.dtype)

    def root_applied(self):
        """(pel, rot6 with the root column) after applying the current droot offset to the base."""
        pel = self.pel + torch.stack([self.droot[..., 0], self.droot[..., 3], self.droot[..., 1]], -1)
        Rr = _yaw_mat(self.droot[..., 2]) @ rotation_6d_to_matrix(self.rot6[:, :, 0])
        rot6 = self.rot6.clone()
        rot6[:, :, 0] = mat_to_rot6d(Rr)
        return pel, rot6

    def snapshot(self):
        pel, rot6 = self.root_applied()
        return dict(rot6=rot6, rot6h=self.rot6h, pel=pel, p_obj=self.p_obj, R_obj=self.R_obj, droot=self.droot)


def project_caps(st, caps, net):
    """Project the TOTAL deviation from the base state back inside every cap. Returns saturation
    stats {name: fraction of frames at >= 99% of the cap}."""
    sat = {}
    R = rotation_6d_to_matrix(st.rot6)
    w = so3_log(st.R0.transpose(-1, -2) @ R)
    for g, joints in JOINT_GROUPS.items():
        cap = caps[g]
        wg = w[:, :, joints]
        sat[g] = float((wg.norm(dim=-1) > 0.99 * cap).float().mean())
        w = w.clone()
        w[:, :, joints] = _clamp_ball(wg, cap)
    st.rot6 = mat_to_rot6d(st.R0 @ so3_exp(w))
    wh = so3_log(st.R0h.transpose(-1, -2) @ rotation_6d_to_matrix(st.rot6h))
    sat["finger"] = float((wh.norm(dim=-1) > 0.99 * caps["finger"]).float().mean())
    st.rot6h = mat_to_rot6d(st.R0h @ so3_exp(_clamp_ball(wh, caps["finger"])))
    dp = torch.einsum("bmwji,bmwj->bmwi", st.Ro0, st.p_obj - st.p0)
    sat["obj_m"] = float((dp.norm(dim=-1) > 0.99 * net.cap_obj_m).float().mean())
    st.p_obj = st.p0 + torch.einsum("bmwij,bmwj->bmwi", st.Ro0, _clamp_ball(dp, net.cap_obj_m))
    dw = so3_log(st.Ro0.transpose(-1, -2) @ st.R_obj)
    sat["obj_rad"] = float((dw.norm(dim=-1) > 0.99 * net.cap_obj_rad).float().mean())
    st.R_obj = st.Ro0 @ so3_exp(_clamp_ball(dw, net.cap_obj_rad))
    sat["root_m"] = float((st.droot[..., :2].norm(dim=-1) > 0.99 * net.cap_root_m).float().mean())
    sat["root_rad"] = float((st.droot[..., 2].abs() > 0.99 * net.cap_root_rad).float().mean())
    sat["root_y"] = float((st.droot[..., 3].abs() > 0.99 * net.cap_root_y).float().mean())
    st.droot = torch.cat([_clamp_ball(st.droot[..., :2], net.cap_root_m),
                          st.droot[..., 2:3].clamp(-net.cap_root_rad, net.cap_root_rad),
                          st.droot[..., 3:].clamp(-net.cap_root_y, net.cap_root_y)], -1)
    return sat


def fk_state(st, parents, jrest, rest_h):
    """state -> (pg, Rg, pj, Rw, pw, pel_eff, rot6_eff, Rj) on the refiner's own FK chain, root offset applied."""
    pel, rot6 = st.root_applied()
    pg, Rg = body_fk(rot6, pel, parents, jrest)
    Rw, pw = wrist_from_body(pg, Rg)
    pj, Rj = hand_fk(st.rot6h, Rw, pw, rest_h)
    return pg, Rg, pj, Rw, pw, pel, rot6, Rj


def rollout(net, st, sides, parents, jrest, rest_h, canon, canon_n, beta16, foot_intent,
            T=3, caps=None, collect=None, sensor_canon=None, sensor_canon_n=None, tip_off=None):
    """Run T recursion steps: assemble(re-probe) -> net -> apply -> project caps. Every intermediate
    state is appended to `collect` (per-step supervision). Returns (final state, last step's saturation).

    canon/canon_n (B,M,S,3) = the 512-point probe cloud (assemble_m3 + losses); sensor_canon(_n) =
    the dense sensor-only cloud (falls back to canon when None). Sensor inputs (finger joints, tips,
    posed points) are rebuilt from the CURRENT state at every step -- never from GT."""
    caps = caps or CAPS_DEFAULT
    sat = {}
    s_canon = canon if sensor_canon is None else sensor_canon
    s_canon_n = canon_n if sensor_canon_n is None else sensor_canon_n
    assert tip_off is not None, "route the call through train_m3.batch_geom + sensor_clouds(geom)"
    for _ in range(T):
        pg, Rg, pj, Rw, pw, pel_eff, rot6_eff, Rj = fk_state(st, parents, jrest, rest_h)
        f = assemble_m3(rot6_eff, st.rot6h, pg, Rg, pj, Rw, pw, st.p_obj, st.R_obj,
                        canon, canon_n, beta16, foot_intent)
        pts, nrm = sensor_points(s_canon, s_canon_n, st.R_obj, st.p_obj)
        tp, tR = tip_centers(pj, Rj, tip_off)
        dth_b, dth_h, dxi, droot = net(*f, sides, sensor_in=(pj, Rj, pts, nrm, tp, tR))
        w = torch.zeros(*st.rot6.shape[:2], 22, 3, device=dth_b.device, dtype=dth_b.dtype)
        w[:, :, 1:] = dth_b
        st.rot6 = mat_to_rot6d(rotation_6d_to_matrix(st.rot6) @ so3_exp(w))
        st.rot6h = mat_to_rot6d(rotation_6d_to_matrix(st.rot6h) @ so3_exp(dth_h))
        dxi_om = dxi.transpose(1, 2)                                   # (B,M,W,6)
        st.p_obj = st.p_obj + torch.einsum("bmwij,bmwj->bmwi", st.R_obj, dxi_om[..., :3])
        st.R_obj = st.R_obj @ so3_exp(dxi_om[..., 3:])
        st.droot = st.droot + droot
        sat = project_caps(st, caps, net)
        if collect is not None:
            collect.append(st.snapshot())
    return st, sat
