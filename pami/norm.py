"""(De)normalisation of the human 272 and object channels — the ONLY boundary between stored data and
network space. Two representation dialects are baked in (they are what every final checkpoint uses):

  human_root = "wf_abs"      nh(): the standard 272 root block [0:8] (per-frame XZ velocity + heading delta)
                             is replaced by the WINDOW-ABSOLUTE root (x_wf, z_wf, heading6D) before z-scoring;
                             dh(): z-score inverse, then differentiate back to the standard 272 (differentiable).
  obj_frame  = "window_first" object robj / p_G are stored in the window's first-frame coordinate frame
                             (converted on the fly by data/objframe.py from the raw human window); the stats
                             file describes that distribution. delta (per-part offsets) is frame-invariant.

numpy nh/no run in the dataset; torch dh/do run on decoded tensors (losses / sampling / eval).
"""
import numpy as np, torch
from . import config as C


class Normalizer:
    def __init__(self, h_mean, h_std, obj_npz=C.STATS_OBJ, obj_frame="window_first", human_root="wf_abs"):
        assert obj_frame == "window_first", f"PAMI supports obj_frame='window_first' only (got {obj_frame!r})"
        assert human_root == "wf_abs", f"PAMI supports human_root='wf_abs' only (got {human_root!r})"
        self.obj_frame = obj_frame; self.human_root = human_root
        self.h_mean = h_mean.astype(np.float32); self.h_std = np.maximum(h_std.astype(np.float32), 1e-6)
        o = np.load(obj_npz)
        self.robj  = (o["robj_mean"],  o["robj_std"])                 # (6,)
        self.pobj  = (o["pobj_mean"],  o["pobj_std"])                 # (3,)
        self.delta = (o["delta_mean"].reshape(6, 3), o["delta_std"].reshape(6, 3))   # (6,3)
        self._cache = {}

    # ---- numpy, dataset side ----
    def nh(self, x):
        from .data.dialect import root_delta_to_abs
        return (root_delta_to_abs(x) - self.h_mean) / self.h_std

    def no(self, g, x):
        m, s = getattr(self, g); return (x - m) / s

    # ---- torch, decode side ----
    def _t(self, name, a, dev):
        k = (name, str(dev))
        if k not in self._cache: self._cache[k] = torch.as_tensor(a, device=dev)
        return self._cache[k]

    def dh(self, x):
        from .data.dialect import root_abs_to_delta
        x = x * self._t("h_std", self.h_std, x.device) + self._t("h_mean", self.h_mean, x.device)
        return root_abs_to_delta(x)

    def do(self, g, x):
        m, s = getattr(self, g)
        return x * self._t(g + "_std", s, x.device) + self._t(g + "_mean", m, x.device)


def build(human_npz=None, obj_npz=None, obj_frame="window_first", human_root="wf_abs", **_ignored):
    """Normalizer from the InterAct stats files (defaults: config.STATS_HUMAN / config.STATS_OBJ)."""
    h = np.load(human_npz or C.STATS_HUMAN); hm, hs = h["mean"], h["std"]
    assert hm.shape[0] == 272, f"human stats dim {hm.shape} != 272"
    return Normalizer(hm, hs, obj_npz or C.STATS_OBJ, obj_frame=obj_frame, human_root=human_root)
