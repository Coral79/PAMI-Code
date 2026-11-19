"""Refiner dataset: bank items (gen or roundtrip stream) joined with their RAW-GT targets.

Every item is one build_real_bank npz = one window of a FROZEN pipeline state (a generation for the
real stream, decode(encode(GT)) for the corruption stream) plus the frozen supervision masks. This
loader adds, per item: the RAW GT target of the same (seq, f0) window (272 index, hand_joints.npz,
tae_sample.npz), the object surface clouds, and the subject's rest skeleton / betas.

TWO CLOUDS: `canon`/`canon_n` = the stored 2048-point shape_0 strided to `surf` points, with the
aligned normals (probes, assemble_m3, the contact/penetration losses). `sensor_canon`/`sensor_canon_n`
(only when dense_surf > 0) = an area-weighted resample of the object's OWN mesh at dense_surf points
with exact face normals, read by the finger sensor only.

Frames: the bank state lives in the window-first frame of ITS OWN first-frame root, the GT target in
that of the GT's first-frame root; both are gauge-fixed at t = 0, height is not gauged away.
"""
import glob
import hashlib
import json
import os

import numpy as np
import torch

from pami.refine.bank_schema import BANK_SCHEMA, MASK_VERSION, check_bank_dir, check_banks_agree
from pami.refine.paths import interact_common, use_interact_config

_SEQ_META = {}
_BM_CACHE = {}


def _meta(seq):
    """interact_common.load_seq, memoized (gender / model / nb / betas per sequence)."""
    if seq not in _SEQ_META:
        _SEQ_META[seq] = interact_common().load_seq(seq)
    return _SEQ_META[seq]


def _sample_surface(V, F, M, seed=0):
    """The upstream sampler's rule: face drawn with probability proportional to its area, then a
    uniform barycentric point in it. -> (M,3) points, (M,3) exact face normals (float32)."""
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    fi = rng.choice(len(F), size=M, p=area / (area.sum() + 1e-12))
    u, w = rng.random(M), rng.random(M)
    over = u + w > 1
    u[over], w[over] = 1 - u[over], 1 - w[over]
    pts = (1 - u - w)[:, None] * tri[fi, 0] + u[:, None] * tri[fi, 1] + w[:, None] * tri[fi, 2]
    n = np.cross(tri[fi, 1] - tri[fi, 0], tri[fi, 2] - tri[fi, 0])
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return pts.astype(np.float32), n.astype(np.float32)


class ModelBatchSampler(torch.utils.data.Sampler):
    """Batches whose windows share ONE key (body model, or (model, W) for varlen banks): SMPL-H and
    SMPL-X rest skeletons cannot be stacked, and varlen items must be equal-length within a batch."""

    def __init__(self, keys, batch_size, shuffle=True, seed=0, drop_last=True):
        self.groups, self.bs, self.shuffle, self.seed, self.drop = {}, batch_size, shuffle, seed, drop_last
        for i, k in enumerate(keys):
            self.groups.setdefault(k, []).append(i)
        self.epoch = 0
        if drop_last:
            lost = {k: len(v) for k, v in self.groups.items() if len(v) < batch_size}
            if lost:
                print(f"[sampler] {sum(lost.values())} items in {len(lost)} groups smaller than "
                      f"batch={batch_size} are never yielded (drop_last): {lost}", flush=True)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        batches = []
        for idx in self.groups.values():
            idx = [idx[k] for k in torch.randperm(len(idx), generator=g)] if self.shuffle else list(idx)
            for s in range(0, len(idx), self.bs):
                b = idx[s: s + self.bs]
                if len(b) == self.bs or not self.drop:
                    batches.append(b)
        order = torch.randperm(len(batches), generator=g) if self.shuffle else range(len(batches))
        for k in order:
            yield batches[k]

    def __len__(self):
        return sum((len(v) // self.bs) if self.drop else -(-len(v) // self.bs) for v in self.groups.values())


class BankWindows(torch.utils.data.Dataset):
    """Items from one or more bank dirs. stream: "gen" | "roundtrip" (pinned against every manifest).
    dense_surf: 0 = no sensor cloud key at all; >0 = ship `sensor_canon(_n)` resampled at that size."""

    def __init__(self, dirs, surf=512, stream=None, dense_surf=0, dense_seed=0):
        self.pkg, self.surf = use_interact_config(), surf
        self.dense_surf, self.dense_seed = int(dense_surf), int(dense_seed)
        self.sensor_surf = self.dense_surf or surf
        if isinstance(dirs, str):
            dirs = [dirs]
        expect = {"stream": stream} if stream else {}
        self.manifests = [check_bank_dir(d, expect=expect) for d in dirs]
        self.stream, self.dirs = stream, list(dirs)
        check_banks_agree(self.manifests, dirs)
        want = (BANK_SCHEMA, MASK_VERSION, self.manifests[0].get("mask_id"))
        for m, d in zip(self.manifests, dirs):
            got = (m.get("bank_schema"), m.get("mask_version"), m.get("mask_id"))
            assert got == want, f"{d}: manifest identity {got} != {want}"
        self.files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "*.npz")))
        assert self.files, f"no bank items under {dirs}"
        from pami.data.hoi_dataset import _seq_to_272
        self.s2r = _seq_to_272()
        self._model_of = {}
        self.models = [self._seq_model(os.path.basename(f).rsplit("_gen_", 1)[0]) for f in self.files]
        # varlen banks: items carry their own window length; the sampler groups by (model, W)
        self.varlen = any(m.get("varlen") for m in self.manifests)
        if self.varlen:
            assert all(m.get("varlen") for m in self.manifests), f"varlen and fixed-window bank dirs mixed: {dirs}"
            iw = {}
            for m in self.manifests:
                iw.update(m.get("item_windows") or {})
            missing = [f for f in self.files if os.path.basename(f) not in iw]
            assert not missing, f"{len(missing)} items missing from manifest item_windows (e.g. {missing[:2]})"
            self.item_w = [int(iw[os.path.basename(f)]) for f in self.files]
            self.batch_keys = list(zip(self.models, self.item_w))
        self._cache_cap, self._cache = 64, {}
        self._rest, self._mesh_cache = {}, {}
        self._dense_cache, self._dense_cap = {}, 32        # keyed by OBJECT (11k sequences share ~74 meshes)

    def _seq_model(self, seq):
        if seq not in self._model_of:
            self._model_of[seq] = str(_meta(seq)["model"])
        return self._model_of[seq]

    def _gt(self, seq):
        if seq in self._cache:
            self._cache[seq] = self._cache.pop(seq)
        else:
            while len(self._cache) >= self._cache_cap:
                self._cache.pop(next(iter(self._cache)))
            z = np.load(f"{self.pkg}/output/{seq}/tae_sample.npz", allow_pickle=True)
            hz = np.load(f"{self.pkg}/output/{seq}/hand_joints.npz", allow_pickle=True)
            al = np.load(f"{self.pkg}/output/{seq}/shape_aligned.npz", allow_pickle=True)
            surf = z["shape_0"].astype(np.float32)
            nrm = al["normal_0"].astype(np.float32)
            try:
                meta = json.loads(str(al["meta_0"]))
            except Exception:
                meta = {}
            trusted = bool(meta.get("normals_trusted", True))     # winding-inconsistent meshes gate the pen hinge
            step = max(1, surf.shape[0] // self.surf)
            can, cn = surf[::step][: self.surf], nrm[::step][: self.surf]
            extra = {}
            if self.dense_surf:
                s_can, s_cn = self._dense_surface(seq, bool(meta.get("flipped", False)))
                extra = dict(sensor_canon=s_can[None], sensor_canon_n=s_cn[None])
            self._cache[seq] = dict(h=np.load(self.s2r[seq]).astype(np.float32),
                                    rot6h=hz["hand_rot6d"].astype(np.float32),
                                    pobj=z["pobj_0"].astype(np.float32), robj=z["robj_0"].astype(np.float32),
                                    canon=can[None], canon_n=cn[None], normals_trusted=trusted, **extra)
        return self._cache[seq]

    def _mesh(self, seq):
        if seq in self._mesh_cache:
            self._mesh_cache[seq] = self._mesh_cache.pop(seq)
        else:
            while len(self._mesh_cache) >= 16:
                self._mesh_cache.pop(next(iter(self._mesh_cache)))
            o6 = np.load(f"{self.pkg}/output/{seq}/object_6dof.npz", allow_pickle=True)
            self._mesh_cache[seq] = (o6["canon_0"].astype(np.float32), o6["faces_0"].astype(np.int64),
                                     str(o6["names"][0]))
        return self._mesh_cache[seq]

    def _dense_surface(self, seq, flipped):
        """(M,3) points + (M,3) OUTWARD normals resampled from the object's own centered canonical mesh.
        Seed = dense_seed ^ digest(object identity): the same cloud in every worker/epoch/process."""
        v, f, name = self._mesh(seq)
        M = int(self.dense_surf)
        key = (name, int(v.shape[0]), int(f.shape[0]), M, self.dense_seed, bool(flipped))
        if key in self._dense_cache:
            self._dense_cache[key] = self._dense_cache.pop(key)
            return self._dense_cache[key]
        if f.shape[0] == 0:
            raise RuntimeError(f"{seq}: object '{name}' has no faces -- dense resampling needs a triangulated mesh")
        assert np.abs(v.mean(0)).max() < 1e-3, f"{seq}: canonical mesh is not centered"
        while len(self._dense_cache) >= self._dense_cap:
            self._dense_cache.pop(next(iter(self._dense_cache)))
        seed = self.dense_seed ^ int.from_bytes(hashlib.blake2b(repr(key[:4]).encode(), digest_size=8).digest(), "big")
        pts, nrm = _sample_surface(v.astype(np.float64), f, M, seed)
        if flipped:
            nrm = -nrm
        self._dense_cache[key] = (pts, nrm.astype(np.float32))
        return self._dense_cache[key]

    def _anchor_pts(self, seq, rows):
        """segment_anchor (S,10) [part,t0,t1,af,jid,face,bary0..2,err] -> (S,9) [part,t0,t1,af,jid,
        pt_loc(3), err]: the frozen object-local correspondence point from face id + barycentrics."""
        mesh_v, mesh_f, _ = self._mesh(seq)
        out = np.zeros((len(rows), 9), np.float32)
        for i, r in enumerate(rows):
            part, t0s, t1s, af, jid, face = (int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5]))
            out[i, :5] = (part, t0s, t1s, af, jid)
            out[i, 8] = r[9]
            if face >= 0 and len(mesh_f):
                out[i, 5:8] = r[6:9] @ mesh_v[mesh_f[face]]
            else:
                out[i, 8] = -1.0
        return out

    def rest(self, seq):
        """(jrest (J,3), model, gender, nb, betas16 (16,)) of the sequence's gendered subject."""
        m = _meta(seq)
        key = (m["gender"], str(m["model"]), m["nb"], tuple(np.round(m["betas16"][: m["nb"]], 5)))
        if key in self._rest:
            return self._rest[key]
        bmk = key[:3]
        if bmk not in _BM_CACHE:
            _BM_CACHE[bmk] = interact_common().body_model(m["gender"], num_betas=m["nb"], model=m["model"])
        b = torch.tensor(np.asarray(m["betas16"], dtype=np.float32)[None, : m["nb"]])
        J = _BM_CACHE[bmk](betas=b).Jtr[0].detach().cpu().numpy().astype(np.float32)
        bt = np.zeros(16, np.float32)
        raw_b = np.asarray(m["betas16"], dtype=np.float32).ravel()
        bt[: min(16, len(raw_b))] = raw_b[:16]
        self._rest[key] = (J, str(m["model"]), m["gender"], int(m["nb"]), bt)
        return self._rest[key]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from pami.data.objframe import human_root_to_window_first, pobj_to_window_first, robj_to_window_first
        b = np.load(self.files[i], allow_pickle=True)
        seq, f0 = str(b["seq"]), int(b["f0"])
        W = b["h272"].shape[0]
        d = self._gt(seq)
        sl = slice(f0, f0 + W)
        hgt = d["h"][sl]
        assert hgt.shape[0] == W, f"{seq} f0={f0}: GT slice {hgt.shape[0]} != {W}"
        hw = human_root_to_window_first(hgt)
        p_gt = pobj_to_window_first(d["pobj"][None, sl], hgt)[0]
        r6_gt = robj_to_window_first(d["robj"][None, sl], hgt)[0]
        jrest, model, gender, nb, betas16 = self.rest(seq)
        it = dict(
            # frozen pipeline state (the refiner INPUT)
            h272=torch.from_numpy(b["h272"].copy()), rot6h=torch.from_numpy(b["rot6h"].copy()),
            root_wf_pos=torch.from_numpy(b["root_wf_pos"].copy()), root_wf_rot6=torch.from_numpy(b["root_wf_rot6"].copy()),
            p_obj=torch.from_numpy(b["p_obj"].copy()), r6_obj=torch.from_numpy(b["r6_obj"].copy()),
            # frozen supervision fields
            contact_intent=torch.from_numpy(b["contact_intent"].copy()),
            reachable_mask=torch.from_numpy(b["reachable_mask"].copy()),
            segment_id=torch.from_numpy(b["segment_id"].copy()),
            foot_contact_intent=torch.from_numpy(b["foot_contact_intent"].copy()),
            # RAW GT target
            gt_h272=torch.from_numpy(hgt.copy()), gt_rot6h=torch.from_numpy(d["rot6h"][sl].copy()),
            gt_root_wf_pos=torch.from_numpy(hw[:, 8:11].copy()), gt_root_wf_rot6=torch.from_numpy(hw[:, 140:146].copy()),
            gt_p_obj=torch.from_numpy(p_gt.copy())[None], gt_r6_obj=torch.from_numpy(r6_gt.copy())[None],
            # geometry + subject
            canon=torch.from_numpy(d["canon"].copy()), canon_n=torch.from_numpy(d["canon_n"].copy()),
            jrest=torch.from_numpy(jrest.copy()), betas16=torch.from_numpy(betas16.copy()),
            normals_trusted=torch.tensor(d["normals_trusted"]),
            seq=seq, f0=f0, model=model, gender=gender, nb=nb,
            obj_name=self._mesh(seq)[2])
        if self.dense_surf:
            it["sensor_canon"] = torch.from_numpy(d["sensor_canon"].copy())
            it["sensor_canon_n"] = torch.from_numpy(d["sensor_canon_n"].copy())
        it["segment_anchor"] = self._anchor_pts(seq, b["segment_anchor"])
        return it


def collate_m3(items):
    models = {it["model"] for it in items}
    assert len(models) == 1, f"mixed body models in one batch: {models} -- use ModelBatchSampler"
    ws = {int(it["h272"].shape[0]) for it in items}
    assert len(ws) == 1, f"mixed window lengths in one batch: {sorted(ws)} -- sampler must key on (model, W)"
    keys = ("h272", "rot6h", "root_wf_pos", "root_wf_rot6", "p_obj", "r6_obj",
            "contact_intent", "reachable_mask", "segment_id", "foot_contact_intent",
            "gt_h272", "gt_rot6h", "gt_root_wf_pos", "gt_root_wf_rot6", "gt_p_obj", "gt_r6_obj",
            "canon", "canon_n", "jrest", "betas16", "normals_trusted")
    out = {k: torch.stack([it[k] for it in items]) for k in keys}
    for k in ("sensor_canon", "sensor_canon_n"):
        have = [k in it for it in items]
        assert all(have) or not any(have), f"{k} present in only some items of a batch"
        if all(have):
            out[k] = torch.stack([it[k] for it in items])
    for k in ("seq", "f0", "gender", "nb", "segment_anchor"):
        out[k] = [it[k] for it in items]
    out["obj_name"] = [it.get("obj_name", "?") for it in items]
    out["model"] = models.pop()
    return out


def sensor_cloud(batch):
    """The (points, normals) pair the sensor reads from a collated batch (canon when no dense cloud)."""
    if "sensor_canon" in batch:
        return batch["sensor_canon"], batch["sensor_canon_n"]
    return batch["canon"], batch["canon_n"]
