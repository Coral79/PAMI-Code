"""Floor of the exported motion: the InterAct / LIGHT convention.

InterAct places the floor of every sequence at the lowest point of the human and object MESHES over its first
30 frames (InterAct `process/canonicalize_human.py`); the ground truth and the baselines' outputs live in that
world. Our 272-d representation instead puts the floor at the lowest JOINT, which sits a few centimetres above
the dataset's floor. The export therefore shifts every refined sequence rigidly along Y so that

    min over the first n frames of (human mesh vertices  U  object mesh vertices)  ==  0,

with n = 30 (fewer only for clips shorter than 34 frames). The shift is one scalar per sequence, computed from
the exported motion itself and applied to the joints, the pelvis and the object translation alike: poses and
the human-object arrangement are unchanged.
"""
import os

import numpy as np

N_FLOOR = 30          # frames used to place the floor
EVAL_TRIM = 4         # the evaluator scores the first T-4 frames (LIGHT's `lengths-4`)
MAXT = 300            # sequences are capped at 300 frames
SHIFT_KEYS = ("joints52", "pel", "p_obj")

_MESH = {}
_DEFAULT_LIGHT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "LIGHT")


def light_root(arg=None):
    """LIGHT_ROOT (CLI argument, else the environment variable, else <repo>/data/LIGHT) holding InterAct/<subset>/objects/..."""
    root = arg or os.environ.get("LIGHT_ROOT") or _DEFAULT_LIGHT
    if not root or not os.path.isdir(os.path.join(root, "InterAct")):
        raise SystemExit("the export places the floor with the InterAct object meshes and needs LIGHT_ROOT "
                         f"(got {root!r}; pass --light_root or export LIGHT_ROOT, the same folder evaluate.py uses), "
                         "or pass --floor native to keep the joint-based floor of the representation")
    return os.path.abspath(os.path.expanduser(root))


def object_mesh(root, subset, obj):
    """((V,3) float64 vertices, (F,3) faces) of the object mesh, in the frame R_obj / p_obj pose."""
    key = (root, subset, obj)
    if key not in _MESH:
        import trimesh
        p = os.path.join(root, "InterAct", subset, "objects", obj, f"{obj}.obj")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"object mesh not found: {p}")
        m = trimesh.load(p, force="mesh")
        _MESH[key] = (np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces))
    return _MESH[key]


def object_mesh_vertices(root, subset, obj):
    """(V,3) float64 vertices of the object mesh."""
    return object_mesh(root, subset, obj)[0]


def n_floor_frames(W):
    """Frames used for the floor: the first 30 of the W exported frames (the first min(30, W - 4) scored frames
    for very short clips)."""
    W = int(min(W, MAXT))
    L = W - EVAL_TRIM
    return int(min(N_FLOOR, L)) if L >= 4 else int(min(N_FLOOR, W))


def floor_shift(human_verts, R_obj, p_obj, obj_verts):
    """human_verts (n,V,3), R_obj (n,3,3), p_obj (n,3), obj_verts (Vo,3) -> (dy in metres, diagnostics).
    dy = -min(lowest human vertex, lowest object vertex) over the n frames."""
    hmin = float(np.asarray(human_verts)[..., 1].min())
    ov = np.einsum("tij,vj->tvi", np.asarray(R_obj, dtype=np.float64), obj_verts) + np.asarray(p_obj, dtype=np.float64)[:, None]
    omin = float(ov[..., 1].min())
    return -min(hmin, omin), dict(human_min=hmin, obj_min=omin, floor_from="object" if omin < hmin else "human")


def apply_shift(arr, dy):
    """Copy of `arr` with dy added to its Y coordinate (last axis, index 1), dtype preserved."""
    a = np.array(arr, copy=True)
    a[..., 1] += np.asarray(dy, dtype=a.dtype)
    return a
