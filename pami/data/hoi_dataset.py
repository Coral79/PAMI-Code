"""PamiVAE training data: one item = one 64-frame window of an InterAct sequence =
  human 272 (W,272) + hand block (W,270) + the window's ACTIVE objects {robj, pobj, delta, w} padded to
  MAX_OBJS + GT canonical anchors + the sequence's rest skeleton (jrest).
Reads the manifest (windows + per-window active object indices), per-seq tae_sample.npz / canon_anchors.npz /
hand_joints.npz, and the human 272 .npy (index csv).
"""
import os, csv, numpy as np, torch
from torch.utils.data import Dataset
from .. import config as C
from .hand_frames import to_wlocal

HAND_W_MAP = {"strong": 1.0, "mid": 1.0, "weak": 0.5, "none": 0.0}    # per-subset hand quality -> hand loss weight


def _seq_to_272():
    """seq_name -> absolute path of its 272 .npy (via the 30fps index csv)."""
    m = {}
    for r in csv.DictReader(open(C.INDEX_CSV)):
        m[r["seq_name"]] = os.path.join(C.R272_DIR, r["new_name"])
    return m


class HOIWindowDataset(Dataset):
    def __init__(self, manifest=C.MANIFEST, window=C.WINDOW, max_objs=C.MAX_OBJS, seqs=None, norm=None,
                 hand_stats=C.STATS_HAND, hand_w_map=None):
        """seqs: sequence-name filter (split list). norm: Normalizer (human_root=wf_abs, obj_frame=window_first);
        None serves raw values. Objects are converted to the window-first frame on the fly from the RAW human
        window (data/objframe.py) before normalisation. The hand block is served WRIST-LOCAL (wlocalrot:
        Rw^T (p - pw) per hand + parent-relative rot6d), normalised with hand_stats."""
        self.hand_w_map = dict(hand_w_map or HAND_W_MAP)
        hs = np.load(hand_stats)
        self.hand_mean = hs["wlocalrot_mean"].astype(np.float32)          # (270,)
        self.hand_std = np.maximum(hs["wlocalrot_std"].astype(np.float32), 1e-6)
        self.norm = norm
        man = np.load(manifest, allow_pickle=True)
        self.seq = man["seq"]; self.start = man["start"]; self.end = man["end"]
        self.active = man["active"]                                   # active[i] = int16 array of object indices
        keep = np.ones(len(self.seq), bool)
        if seqs is not None:
            allow = set(map(str, seqs))
            keep &= np.array([str(s) in allow for s in self.seq])
        self.s2r = _seq_to_272()
        has272 = np.array([str(s) in self.s2r for s in self.seq])
        n_drop = int((keep & ~has272).sum())
        keep &= has272
        if n_drop: print(f"[HOIWindowDataset] dropped {n_drop} windows (seq has no 272 file)")
        fits = np.array([len(a) <= max_objs for a in self.active])
        n_over = int((keep & ~fits).sum())
        keep &= fits
        if n_over: print(f"[HOIWindowDataset] dropped {n_over} windows (>{max_objs} active objects)")
        self.idx = np.where(keep)[0]
        has_hand = np.array([os.path.exists(f"{C.PKG}/output/{s}/hand_joints.npz") for s in self.seq])
        kept = np.array([k for k in self.idx if has_hand[k]], dtype=np.int64)
        n_nohand = len(self.idx) - len(kept)
        self.idx = kept
        if n_nohand: print(f"[HOIWindowDataset] dropped {n_nohand} windows (seq has no hand_joints.npz)")
        self.window = window; self.max_objs = max_objs
        self._ts_cache = {}; self._h_cache = {}; self._anch_cache = {}; self._hand_cache = {}

    def __len__(self): return len(self.idx)

    def _tae(self, seq):
        if seq not in self._ts_cache:
            self._ts_cache[seq] = np.load(f"{C.PKG}/output/{seq}/tae_sample.npz", allow_pickle=True)
        return self._ts_cache[seq]

    def _h272(self, seq):
        if seq not in self._h_cache:
            self._h_cache[seq] = np.load(self.s2r[seq]).astype(np.float32)   # (T,272)
        return self._h_cache[seq]

    def _anch(self, seq):
        if seq not in self._anch_cache:
            self._anch_cache[seq] = np.load(f"{C.PKG}/output/{seq}/canon_anchors.npz", allow_pickle=True)
        return self._anch_cache[seq]                                  # p_anch (T,6,3), R_anch (T,6,3,3), jrest (22,3)

    def _hand(self, seq):
        """(T,270) canonical [Lpos|Rpos|Lrot|Rrot] hand block (positions still canonical), quality weight."""
        if seq not in self._hand_cache:
            z = np.load(f"{C.PKG}/output/{seq}/hand_joints.npz", allow_pickle=True)
            pos = z["hand"].astype(np.float32).reshape(len(z["hand"]), 90)
            rot = z["hand_rot6d"].astype(np.float32).reshape(len(pos), 180)
            q = str(z["quality"]) if "quality" in z.files else "none"
            if q not in self.hand_w_map and not getattr(self, "_warned_quality", False):
                print(f"[HOIWindowDataset] WARNING: unknown hand quality '{q}' ({seq}) -> weight 0.0")
                self._warned_quality = True
            self._hand_cache[seq] = (np.concatenate([pos, rot], 1), self.hand_w_map.get(q, 0.0))
        return self._hand_cache[seq]

    def __getitem__(self, i):
        k = self.idx[i]; seq = str(self.seq[k]); s, e = int(self.start[k]), int(self.end[k])
        W = self.window
        human = self._h272(seq)[s:e]                                  # (W,272) raw
        ts = self._tae(seq); act = list(self.active[k])[:self.max_objs]
        M = len(act)
        robj  = np.zeros((self.max_objs, W, C.ROBJ_D), np.float32)
        pobj  = np.zeros((self.max_objs, W, C.POBJ_D), np.float32)
        delta = np.zeros((self.max_objs, W, C.N_PARTS, 3), np.float32)
        wlab  = np.zeros((self.max_objs, W, C.W_D), np.float32)
        omask = np.zeros((self.max_objs,), np.float32)
        obj_names = [""] * self.max_objs
        anames = ts["active_names"] if "active_names" in ts.files else None
        az = self._anch(seq)
        anchor_p = az["p_anch"][s:e].astype(np.float32)               # (W,6,3)
        anchor_R = az["R_anch"][s:e].astype(np.float32)               # (W,6,3,3)
        jrest = az["jrest"].astype(np.float32)                        # (22,3) per-seq rest skeleton
        for o, mi in enumerate(act):
            robj[o]  = ts[f"robj_{mi}"][s:e]
            pobj[o]  = ts[f"pobj_{mi}"][s:e]
            delta[o] = ts[f"delta_{mi}"][s:e]
            wlab[o]  = ts[f"w_{mi}"][s:e]
            omask[o] = 1.0
            if anames is not None and mi < len(anames): obj_names[o] = str(anames[mi])
        if M > 0:                                                     # per-frame canonical -> window-first frame
            from .objframe import robj_to_window_first, pobj_to_window_first
            robj[:M] = robj_to_window_first(robj[:M], human)
            pobj[:M] = pobj_to_window_first(pobj[:M], human)
        h272_raw = human.copy()                                       # wrist frames come from the RAW human
        if self.norm is not None:
            human = self.norm.nh(human)
            robj  = self.norm.no("robj",  robj)
            pobj  = self.norm.no("pobj",  pobj)
            delta = self.norm.no("delta", delta)
        hblock, hw = self._hand(seq)
        if len(hblock) != len(self._h272(seq)):
            raise RuntimeError(f"{seq}: hand_joints has {len(hblock)} frames but human_272 has "
                               f"{len(self._h272(seq))} — stale hand file")
        hb = hblock[s:e]
        hb = np.concatenate([to_wlocal(hb[:, :90].reshape(W, 30, 3), h272_raw).reshape(W, 90), hb[:, 90:]], 1)
        hb = ((hb - self.hand_mean) / self.hand_std).astype(np.float32)
        return {
            "human": torch.from_numpy(human),                        # (W,272) normalised, wf_abs root
            "hand": torch.from_numpy(hb),                            # (W,270) normalised wrist-local hand block
            "hand_w": torch.tensor(hw, dtype=torch.float32),         # per-seq hand-quality loss weight
            "robj":  torch.from_numpy(robj),                         # (MAX_OBJS,W,6)  window-first
            "pobj":  torch.from_numpy(pobj),                         # (MAX_OBJS,W,3)  window-first
            "delta": torch.from_numpy(delta),                       # (MAX_OBJS,W,6,3) canonical part offsets
            "w":     torch.from_numpy(wlab),                        # (MAX_OBJS,W,7) GT routing
            "omask": torch.from_numpy(omask),                      # (MAX_OBJS,)
            "anchor_p": torch.from_numpy(anchor_p),               # (W,6,3) GT canonical anchors (diagnostics)
            "anchor_R": torch.from_numpy(anchor_R),               # (W,6,3,3)
            "jrest": torch.from_numpy(jrest),                     # (22,3) rest skeleton for AnchorFK
            "obj_names": obj_names,
        }
