"""Object mesh assets: canonical mesh, 2048 area-weighted surface samples, raw + normalized BPS."""
import numpy as np
import torch
import trimesh

from . import paths as P
from .subsets import SUBSETS

MARGIN = 0.05                                    # LIGHT: the object fills (1 - margin) of the unit ball
_OBJ = {}
_BASIS = {}


def sample_surface(V, F, M=2048, seed=0):
    """Area-weighted surface point sampling on the canonical mesh (deterministic, seed 0)."""
    if len(F) == 0:                              # fallback: resample vertices
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(V), M)
        return V[idx].astype(np.float32)
    tri = V[F]
    a = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(a, axis=1)
    p = area / (area.sum() + 1e-12)
    rng = np.random.default_rng(seed)
    fi = rng.choice(len(F), size=M, p=p)
    u = rng.random(M); w = rng.random(M)
    over = u + w > 1; u[over] = 1 - u[over]; w[over] = 1 - w[over]
    b0 = (1 - u - w)[:, None]; b1 = u[:, None]; b2 = w[:, None]
    pts = b0 * tri[fi, 0] + b1 * tri[fi, 1] + b2 * tri[fi, 2]
    return pts.astype(np.float32)


def basis(dev="cpu"):
    k = str(dev)
    if k not in _BASIS:
        _BASIS[k] = torch.from_numpy(np.load(P.BPS_BASIS).astype(np.float32)).to(dev)   # (1024,3)
    return _BASIS[k]


def bps_deltas(pts, dev="cpu"):
    """pts (P,3) torch -> (1024,3) np: bps_torch 'deltas' (nearest point minus basis point)."""
    B = basis(dev)
    idx = torch.cdist(B, pts).argmin(1)
    return (pts[idx] - B).cpu().numpy().astype(np.float32)


def obj_cached(name, subset, dev="cpu"):
    """Per-(subset, object) cache: V, F, samp (2048,3), raw BPS (InterAct's precomputed vertex deltas,
    variant A = center-only), norm BPS (LIGHT recipe: center, scale = max||v-c||/(1-MARGIN)), scale.
    The key includes the subset: the same object name can denote different scans across subsets."""
    key = (subset, name)
    if key not in _OBJ:
        root = SUBSETS[subset].get("assets", SUBSETS[subset]["root"])
        mesh = trimesh.load(f"{root}/objects/{name}/{name}.obj", force="mesh", process=False)
        V = np.asarray(mesh.vertices, np.float64); F = np.asarray(mesh.faces, np.int32)
        assert np.abs(V.mean(0)).max() < 1e-6, f"canonical mesh {subset}/{name} not centered?"
        samp = sample_surface(V, F, M=2048, seed=0)
        raw = np.load(f"{root}/objects_bps/{name}/{name}_1024.npy")[0].astype(np.float32)
        pts = torch.from_numpy(V.astype(np.float32)).to(dev)
        pts = pts - pts.mean(0, keepdim=True)
        scale = pts.norm(dim=1).max() / (1.0 - MARGIN)
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError(f"degenerate object {subset}/{name}: scale={float(scale)}")
        _OBJ[key] = dict(V=V, F=F, samp=samp, raw=raw, norm=bps_deltas(pts / scale, dev),
                         scale=np.float32(scale.item()))
    return _OBJ[key]
