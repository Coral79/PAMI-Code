"""PamiGen piece dataset: serves the frozen-PamiVAE latents of each action-aligned PIECE (<= 300 frames)
plus its conditions.
  zh (Tl,8,16), zo (Tl,1,16)  encoder mu after SceneST, z-scored PER TOKEN with the archive's train-only
                              stats (stats_h_tok (8,2)) / the scalar object stats (stats_o)
  cap_tid                     caption text id (0 = null caption for CFG); val/test/sampling use the first
                              caption of the piece, training draws one at random
  bps (1,6145)                object dual BPS [raw 3072 | normalised 3072 | scale] (diffusion-side condition)
  jbone (63,)                 body-shape condition: 21 parent-relative rest-bone offsets of the sequence
  wgt (W,1,7), jrest (22,3)   (want_wgt) GT routing multi-hot + rest skeleton for the decoded object losses
TextBank holds the CLIP caption features (pooled + per-token) and assembles per-batch tensors.
"""
import os, json
import numpy as np
import torch
from torch.utils.data import Dataset
from .. import config as C
from ..nets.t2hoi_denoiser import LAT, CLIP_D, SHAPE_WIDTH

BODY_PARENTS = np.asarray([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19], dtype=np.int64)


def shape_standardization_stats(jb):
    """(mean, std) float32 (63,) of the jrest bone condition over the TRAIN rows jb (N,63); dead channels
    (std 0, structurally zero) keep divisor 1, live channels are floored at 1e-4."""
    jb = np.asarray(jb, np.float32)
    assert jb.ndim == 2 and jb.shape[1] == SHAPE_WIDTH, jb.shape
    mean, std = jb.mean(0), jb.std(0)
    dead = std <= 0
    if dead.any():
        bad = [int(i) for i in np.nonzero(dead)[0] if np.abs(jb[:, i]).max() > 0]
        if bad:
            raise SystemExit(f"human_shape: channels {bad} have zero variance but are not identically zero")
        std[dead] = 1.0
    std[~dead] = np.maximum(std[~dead], 1e-4)
    return mean.astype(np.float32), std.astype(np.float32)


class TextBank:
    """CLIP text features (text_feats.npz): pooled (N,512) fp32 + per-text token features (fp16 on disk)."""
    def __init__(self, data_dir=C.T2HOI_DATA):
        z = np.load(f"{data_dir}/text_feats.npz", allow_pickle=True)
        self.pooled = torch.from_numpy(z["pooled"])
        self.tok_len = z["tok_len"]
        self.toks = [torch.from_numpy(z[f"tok_{i}"].astype(np.float32)) for i in range(len(self.tok_len))]

    def captions(self, tids, dev):
        """tids (B,) long -> pooled (B,512), tok (B,Lmax,512), tok_mask (B,Lmax)."""
        tids = tids.cpu()
        pooled = self.pooled[tids].to(dev)
        L = max(int(self.tok_len[int(t)]) for t in tids)
        tok = torch.zeros(len(tids), L, CLIP_D)
        msk = torch.zeros(len(tids), L)
        for j, t in enumerate(tids):
            tk = self.toks[int(t)]
            tok[j, :len(tk)] = tk; msk[j, :len(tk)] = 1
        return pooled, tok.to(dev), msk.to(dev)


class T2HOIPieceDataset(Dataset):
    def __init__(self, cap, vae_tag, seqs=None, data_dir=C.T2HOI_DATA, n_obj=1, random_caption=True,
                 want_wgt=False, human_shape="jrest"):
        assert human_shape in (True, "jrest"), human_shape
        self.n_obj, self.random_caption = n_obj, random_caption
        lat = C.latents_file(data_dir, cap, vae_tag)
        with np.load(lat, allow_pickle=True) as z:
            self.z = {k: z[k] for k in z.files}                        # materialised (fork-safe)
        tj = json.load(open(f"{data_dir}/texts.json"))
        self.n_texts = len(tj["texts"])
        seg_seq = [str(s) for s in self.z["seg_seq"]]
        keep = list(range(len(seg_seq)))
        if seqs is not None:
            allow = set(map(str, seqs))
            keep = [i for i in keep if seg_seq[i] in allow]
        self.idx = keep
        self.seg_seq = seg_seq
        if "stats_h_tok" not in self.z:
            raise SystemExit(f"{lat} has no 'stats_h_tok' (per-token stats)")
        self.stats_h = self.z["stats_h_tok"]                           # (P,2) per-token [mean,std]
        self.stats_o = self.z["stats_o"]
        self.refined = bool(self.z.get("refined", False))
        self.vae_ckpt = str(self.z["vae_ckpt"]) if "vae_ckpt" in self.z else None
        self.want_wgt = want_wgt
        self.human_shape = "jrest"
        self._gbps_cache, self._wgt_cache, self._jrest_cache, self._JR_CACHE = {}, {}, {}, {}

    def __len__(self):
        return len(self.idx)

    def _pkg_dir(self, seq):
        return f"{C.PKG}/output/{seq}"

    def _wgt(self, seq, keep, f0, f1):
        """(W, M, 7) GT routing multi-hot for the piece (tae_sample.npz w_<mi>), raw."""
        cols = []
        for mi in keep:
            ck = (seq, int(mi))
            if ck not in self._wgt_cache:
                with np.load(f"{self._pkg_dir(seq)}/tae_sample.npz", allow_pickle=True) as tz:
                    self._wgt_cache[ck] = tz[f"w_{int(mi)}"].astype(np.float32)
            w = self._wgt_cache[ck][f0:f1]
            assert w.shape == (f1 - f0, 7), (seq, w.shape)
            cols.append(w)
        return np.stack(cols, 1) if cols else np.zeros((f1 - f0, 0, 7), np.float32)

    def _jrest_raw(self, seq):
        """(J (22,3) float32, parents (22,) int64) rest skeleton from canon_anchors.npz."""
        if seq not in self._JR_CACHE:
            an = np.load(f"{self._pkg_dir(seq)}/canon_anchors.npz", allow_pickle=True)
            J = an["jrest"].astype(np.float32)[:22]
            par = (np.asarray(an["jrest_parents"][:22], dtype=np.int64) if "jrest_parents" in an.files else BODY_PARENTS)
            self._JR_CACHE[seq] = (J, par)
        return self._JR_CACHE[seq]

    def _jrest(self, seq):
        return self._jrest_raw(seq)[0]

    def jrest_bones(self, seq):
        """(63,) parent-relative rest bone offsets for body joints 1..21, absolute scale (the shape condition)."""
        J, par = self._jrest_raw(seq)
        v = (J[1:22] - J[par[1:22]]).reshape(-1).astype(np.float32)
        assert v.shape == (SHAPE_WIDTH,), v.shape
        return v

    def shape_feat(self, seq):
        return self.jrest_bones(seq)

    def _shape_feats(self, seq):
        """All per-sequence dual-BPS rows (n_obj_seq, 6145) from bps_sample.npz."""
        if seq not in self._gbps_cache:
            bz = np.load(f"{self._pkg_dir(seq)}/bps_sample.npz", allow_pickle=True)
            n = len(bz["active_names"]) if "active_names" in bz.files else 1
            rows = [np.concatenate([bz[f"bps_{j}"].reshape(-1), bz[f"bps_norm_{j}"].reshape(-1),
                                    np.asarray([bz[f"scale_{j}"]], np.float32)]) for j in range(n)]
            self._gbps_cache[seq] = (np.stack(rows).astype(np.float32) if rows
                                     else np.zeros((0, 2 * 3 * C.BPS_PTS + 1), np.float32))
        return self._gbps_cache[seq]

    def __getitem__(self, i):
        k = self.idx[i]
        seq = self.seg_seq[k]
        f0, f1 = int(self.z["seg_f0"][k]), int(self.z["seg_f1"][k])
        sh = self.stats_h
        zh = (self.z[f"zh_{k}"] - sh[:, 0][None, :, None]) / sh[:, 1][None, :, None]   # (Tl,P,16) per-token
        zo_raw = self.z[f"zo_{k}"]                                      # (Tl,M,16)
        Tl, M = zo_raw.shape[:2]
        assert M <= self.n_obj, f"{seq}: {M} objects > n_obj={self.n_obj}"
        zo = np.zeros((Tl, self.n_obj, LAT), np.float32)
        if M:
            zo[:, :M] = (zo_raw - self.stats_o[0]) / self.stats_o[1]
        omask = np.zeros(self.n_obj, np.float32); omask[:M] = 1
        bps = np.zeros((self.n_obj, 2 * 3 * C.BPS_PTS + 1), np.float32)
        kk = f"keep_{k}"
        keep = self.z[kk] if kk in self.z else np.arange(M)
        if M:
            bps[:M] = self._shape_feats(seq)[keep]
        ckey = f"cap_tids_{k}"
        choices = self.z[ckey] if ckey in self.z else np.asarray([self.z["cap_tid"][k]])
        ci = np.random.randint(len(choices)) if self.random_caption else 0
        item = {"zh": torch.from_numpy(zh.astype(np.float32)),
                "zo": torch.from_numpy(zo),
                "omask": torch.from_numpy(omask),
                "bps": torch.from_numpy(bps),
                "cap_tid": int(choices[ci]),
                "jbone": torch.from_numpy(self.shape_feat(seq).astype(np.float32)),
                "seq": seq, "f0": f0, "f1": f1}
        if self.want_wgt:
            item["wgt"] = torch.from_numpy(self._wgt(seq, keep, f0, f1))                   # (W,M,7)
            item["jrest"] = torch.from_numpy(self._jrest(seq))                             # (22,3)
        return item


def collate_t2hoi(batch, n_obj=None):
    """Pad pieces to the batch-max latent length; step_mask marks real steps."""
    B = len(batch)
    Tr = max(b["zh"].shape[0] for b in batch)
    n_obj = batch[0]["zo"].shape[1] if n_obj is None else n_obj
    out = {"omask": torch.stack([b["omask"] for b in batch]),
           "bps": torch.stack([b["bps"] for b in batch]),
           "cap_tid": torch.tensor([b["cap_tid"] for b in batch], dtype=torch.long),
           "jbone": torch.stack([b["jbone"] for b in batch]),
           "seq": [b["seq"] for b in batch],
           "f0": torch.tensor([b["f0"] for b in batch]), "f1": torch.tensor([b["f1"] for b in batch])}
    n_hp = batch[0]["zh"].shape[1]
    zh = torch.zeros(B, Tr, n_hp, LAT); zo = torch.zeros(B, Tr, n_obj, LAT)
    sm = torch.zeros(B, Tr)
    for j, b in enumerate(batch):
        L = b["zh"].shape[0]
        zh[j, :L] = b["zh"]; zo[j, :L] = b["zo"]; sm[j, :L] = 1
    out.update(zh=zh, zo=zo, step_mask=sm)
    if "wgt" in batch[0]:
        W = Tr * 4
        wg = torch.zeros(B, W, out["zo"].shape[2], 7)
        for j, b in enumerate(batch):
            w, m = b["wgt"].shape[:2]
            wg[j, :w, :m] = b["wgt"]
        out["wgt"] = wg
        out["jrest"] = torch.stack([b["jrest"] for b in batch])        # (B,22,3)
    return out
