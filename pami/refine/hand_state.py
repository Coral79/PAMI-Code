"""Finger half of the theta -> FK chain: 30 finger joints [L15 | R15], 5 fingers x (MCP, PIP, DIP)
per side, hung off the BODY FK's wrist (never off the 272 position block: on generated motion the
two wrists differ by ~23 mm and a hand rigidly attached to the wrong one starts 2 cm off).
Also the fingertip centers the sensor queries from (the chain ends at the DIP, ~2 cm behind the tip).
"""
import torch

from pami.vendor.eval_trans import rotation_6d_to_matrix

FINGERS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14)]   # per side: MCP, PIP, DIP
WRIST_J = (20, 21)                                                       # body joints the hands hang off
HAND_H0 = {"smplh": 22, "smplx": 25}                                     # rest-joint index of the first hand joint


def hand_fk(rot6_hand, Rw, pw, rest_hand):
    """rot6_hand (B,W,30,6) parent-relative | Rw (B,W,2,3,3), pw (B,W,2,3) global wrists from body_fk
    | rest_hand (B,2,15,3) or (2,15,3) rest finger offsets RELATIVE TO THE WRIST
    -> pj (B,W,30,3) finger positions, Rj (B,W,30,3,3) global finger rotations."""
    R = rotation_6d_to_matrix(rot6_hand)
    per_sample = rest_hand.dim() == 4
    rh = rest_hand.to(dtype=rot6_hand.dtype, device=rot6_hand.device)
    pj, Rj = [None] * 30, [None] * 30

    def off(side, j):
        return rh[:, side, j] if per_sample else rh[side, j]

    def rot(Rg, v):
        return (torch.einsum("bwij,bj->bwi", Rg, v) if per_sample
                else torch.einsum("bwij,j->bwi", Rg, v))

    for side in (0, 1):
        o = side * 15
        Rws, pws = Rw[:, :, side], pw[:, :, side]
        for (m, p_, d_) in FINGERS:
            Rm = torch.einsum("bwij,bwjk->bwik", Rws, R[:, :, o + m])
            pm = pws + rot(Rws, off(side, m))
            pp = pm + rot(Rm, off(side, p_) - off(side, m))
            Rp = torch.einsum("bwij,bwjk->bwik", Rm, R[:, :, o + p_])
            pd = pp + rot(Rp, off(side, d_) - off(side, p_))
            Rd = torch.einsum("bwij,bwjk->bwik", Rp, R[:, :, o + d_])
            pj[o + m], pj[o + p_], pj[o + d_] = pm, pp, pd
            Rj[o + m], Rj[o + p_], Rj[o + d_] = Rm, Rp, Rd
    return torch.stack(pj, 2), torch.stack(Rj, 2)


def rest_hand_offsets(rest, model="smplh"):
    """Full rest skeleton (J,3) or (B,J,3) -> (2,15,3) / (B,2,15,3) finger offsets relative to the wrist."""
    h0 = HAND_H0[model]
    if rest.dim() == 2:
        return torch.stack([rest[h0 + s * 15: h0 + (s + 1) * 15] - rest[WRIST_J[s]] for s in (0, 1)])
    return torch.stack([rest[:, h0 + s * 15: h0 + (s + 1) * 15] - rest[:, WRIST_J[s], None]
                        for s in (0, 1)], 1)


def wrist_from_body(pg, Rg):
    """body_fk output -> (Rw (B,W,2,3,3), pw (B,W,2,3)), side order 0 = left, 1 = right."""
    return Rg[:, :, WRIST_J], pg[:, :, WRIST_J]


# fingertip vertex ids (per hand: index, middle, pinky, ring, thumb -- the SMPL-H native slot order)
TIP_VIDS = {"smplh": {0: [2319, 2445, 2673, 2556, 2746], 1: [5782, 5905, 6133, 6016, 6191]},
            "smplx": {0: [4933, 5058, 5286, 5169, 5361], 1: [7669, 7794, 8022, 7905, 8079]}}
DIP_LOCAL = (2, 5, 8, 11, 14)                    # index of each finger's DIP within a hand's 15 joints


def tip_rest_offsets(rest_verts, rest_joints, model="smplh"):
    """(B,V,3) rest vertices + (B,J,3) rest joints -> (B,2,5,3) DIP->fingertip vectors in the rest
    pose (= in the DIP's own frame, since every rest joint rotation is identity)."""
    h0 = HAND_H0[model]
    out = []
    for side in (0, 1):
        vids = TIP_VIDS[model][side]
        dip = [h0 + side * 15 + d for d in DIP_LOCAL]
        out.append(rest_verts[:, vids] - rest_joints[:, dip])
    return torch.stack(out, 1)


def tip_centers(pj, Rj, tip_off):
    """pj (B,W,30,3), Rj (B,W,30,3,3), tip_off (B,2,5,3) -> fingertip positions (B,W,10,3) and the
    DIP frames they borrow (B,W,10,3,3), ordered [L5|R5]."""
    idx = [side * 15 + d for side in (0, 1) for d in DIP_LOCAL]
    p_dip, R_dip = pj[:, :, idx], Rj[:, :, idx]
    off = tip_off.reshape(tip_off.shape[0], 10, 3).to(pj.dtype)
    return p_dip + torch.einsum("bwjxy,bjy->bwjx", R_dip, off), R_dip
