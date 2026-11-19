"""Bankless refine + export: PamiGen sample npz in -> refined joints52 / object pose npz out.

Each gen_*.npz is turned into ONE in-memory bank item (build_real_bank.build_item, the exact per-window
builder the disk banks use), joined with its GT/geometry context by BankWindows.__getitem__ (the item
lives in a BytesIO npz instead of a directory), refined with the loaded checkpoint (T-step rollout),
and written as one npz per sequence:

    joints52 (W,52,3)  LIGHT-52 joints of the REFINED state (vertex_m3.full_verts_joints)
    rot6 (W,22,6), rot6h (W,30,6), pel (W,3)   the refined SMPL state (parent-relative rot6d)
    R_obj (W,3,3), p_obj (W,3)                 refined object rotation / centroid (window-first frame)
    obj_name, caption, seq, f0, win, provenance (json) + contact_logit/contact_prob/contact_joints
    pass-through when the generator wrote them.

Floor: by default the exported motion is placed on the InterAct / LIGHT floor (pami/refine/floor.py): one rigid Y
shift per sequence, applied AFTER the refinement to joints52, pel and p_obj, such that the lowest human or object
mesh vertex of the first 30 frames is at y = 0. `floor_dy_m` records the shift; `--floor native` keeps the
joint-based floor of the 272-d representation.

Window: W = min(piece_len - f0, max_frames), one window per sequence at f0 (default 0 = the sequence
head, where window-first coincides with InterAct's canonical world frame). Writes ONLY into out_dir.
"""
import argparse
import glob
import io
import json
import os
import re
import subprocess

import numpy as np
import torch

from pami.data.objframe import pobj_to_per_frame, robj_to_per_frame
from pami.refine.arch_guard import build_m3_net, check_dataset, dataset_cfg_from_ckpt
from pami.refine.base_state import SMPL_PARENTS
from pami.refine.build_real_bank import FLOOR_STRATEGY, build_item, check_hand_mode, seq_meta
from pami.refine import floor as floor_lib
from pami.refine.data_m3 import BankWindows, collate_m3
from pami.refine.net_m3 import token_sides_m3
from pami.refine.paths import interact_common, use_interact_config
from pami.refine.rollout_m3 import RolloutState, rollout, sensor_clouds
from pami.refine.train_m3 import batch_geom, to_state
from pami.refine.vertex_m3 import full_verts_joints

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MemBankWindows(BankWindows):
    """BankWindows minus the disk bank: ONE item at a time, living as an in-memory npz blob.
    __getitem__ (the RAW-GT join, window-first transforms, clouds, anchors, rest metadata) is inherited."""

    def __init__(self, surf, dense_surf, dense_seed):
        self.pkg, self.surf = use_interact_config(), int(surf)
        self.dense_surf, self.dense_seed = int(dense_surf), int(dense_seed)
        self.sensor_surf = self.dense_surf or self.surf
        self.manifests, self.dirs, self.stream, self.files, self.models, self.varlen = [], [], "gen", [], [], True
        from pami.data.hoi_dataset import _seq_to_272
        self.s2r = _seq_to_272()
        self._model_of, self._rest = {}, {}
        self._cache_cap, self._cache = 8, {}
        self._mesh_cache = {}
        self._dense_cache, self._dense_cap = {}, 4

    def set_item(self, it):
        blob = io.BytesIO()
        np.savez(blob, **it)
        self.files = [blob]
        return self

    def __getitem__(self, i):
        self.files[i].seek(0)
        return super().__getitem__(i)


def _git_commit():
    try:
        return subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as e:
        return f"unknown ({e})"


_TEXTS = {}


def _caption(z):
    """gen npz -> caption text via its recorded data_dir/texts.json + cap_tid ('' when unavailable)."""
    if "cap_tid" not in z.files:
        return ""
    try:
        dd = str(z["data_dir"])
        if dd not in _TEXTS:
            _TEXTS[dd] = json.load(open(os.path.join(dd, "texts.json")))["texts"]
        return str(_TEXTS[dd][int(z["cap_tid"])])
    except Exception:
        return ""


def seq_filter(spec):
    """None | comma-separated names | json file (list of names / list of {'name'|'seq': ...} / dict of lists)."""
    if not spec:
        return None
    if os.path.isfile(spec):
        j = json.load(open(spec))
        vals = [v for sub in j.values() for v in sub] if isinstance(j, dict) else j
        return {str(v.get("seq") or v.get("name") if isinstance(v, dict) else v) for v in vals}
    return {s for s in spec.split(",") if s}


def refine_files(files, ckpt, out_dir, seqs=None, f0=0, max_frames=300, T=4, overwrite=False,
                 floor="interact", light_root=None):
    """Refine every gen npz in `files` (one output per sequence; later draws of a sequence are skipped).
    floor: "interact" (default) = InterAct / LIGHT floor, needs LIGHT_ROOT; "native" = joint-based floor.
    Returns (n_written, n_failed)."""
    assert floor in ("interact", "native"), floor
    want = seq_filter(seqs) if isinstance(seqs, str) or seqs is None else set(seqs)
    lroot = floor_lib.light_root(light_root) if floor == "interact" else None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = dataset_cfg_from_ckpt(ck)
    ds = MemBankWindows(cfg["surf"], cfg["dense_surf"], cfg["dense_seed"])
    check_dataset(ck, ckpt, ds=ds)
    net = None
    parents = SMPL_PARENTS.to(dev)
    sides = token_sides_m3().to(dev)
    SM = interact_common().seq_subset_map()
    os.makedirs(out_dir, exist_ok=True)
    commit = _git_commit()
    bm_cache, done, n, n_fail = {}, set(), 0, 0
    with torch.no_grad():
        for f in files:
            m = re.match(r"gen_\d+_(.+)\.npz$", os.path.basename(f))
            fseq = m.group(1) if m else None
            if want is not None and fseq is not None and fseq not in want:
                continue
            try:                                               # per-sequence failure isolation
                z = np.load(f, allow_pickle=True)
                seq = str(z["seq"])
                if want is not None and seq not in want:
                    continue
                if seq in done:
                    print(f"[dup ] {seq}: second draw {os.path.basename(f)} skipped", flush=True)
                    continue
                out = os.path.join(out_dir, f"refined_{seq}.npz")
                if os.path.exists(out) and not overwrite:
                    done.add(seq)
                    print(f"[skip] {seq}: {out} exists", flush=True)
                    continue
                if z["pobj"].shape[0] == 0:
                    print(f"[skip] {seq}: M=0 (no active object)", flush=True)
                    continue
                assert z["pobj"].shape[0] == 1, f"{f}: M={z['pobj'].shape[0]} objects; the refiner is single-object"
                hm = check_hand_mode(str(z["hand_mode"]) if "hand_mode" in z.files else "wlocal",
                                     int(z["hand"].shape[1]), where=f"{f}: ")
                hfull = z["human_272"].astype(np.float32)
                Tn = hfull.shape[0]
                W = min(Tn - f0, max_frames)
                if W <= 0:
                    print(f"[skip] {seq}: T={Tn} <= f0={f0}", flush=True)
                    continue
                pobj_c = pobj_to_per_frame(z["pobj"].astype(np.float32), hfull)   # piece -> per-frame canonical ONCE
                robj_c = robj_to_per_frame(z["robj"].astype(np.float32), hfull)
                meta = seq_meta(seq, bm_cache)
                it = build_item(z, f0, meta, bm_cache, pobj_c, robj_c, floor_strategy=FLOOR_STRATEGY,
                                anchors_on=True, hand_mode=hm, win=W)
                if it is None:
                    print(f"[skip] {seq}: build_item returned None (T={Tn}, W={W})", flush=True)
                    continue
                it.pop("_feet")
                ik_err = float(it["ik_err_mm"])
                it["win"] = np.int32(W)
                b = collate_m3([ds.set_item(it)[0]])
                if net is None:
                    net = build_m3_net(ck, ckpt, dev, batch=b)
                geom = batch_geom(b, dev)
                dec = to_state(b, dev)
                st = RolloutState(dec["rot6"].clone(), dec["rot6h"].clone(), dec["pel"].clone(),
                                  dec["p_obj"].clone(), dec["R_obj"].clone())
                st, _ = rollout(net, st, sides, parents, geom["jrest"], geom["rest_h"], geom["canon"], geom["canon_n"],
                                geom["betas16"], geom["masks"]["foot_contact_intent"], T=T, **sensor_clouds(geom))
                s_ = st.snapshot()
                gender, model, nb = b["gender"][0], b["model"], b["nb"][0]
                verts, _, J52 = full_verts_joints(s_["rot6"][0], s_["rot6h"][0], s_["pel"][0], geom["betas16"][0],
                                                  gender, model, nb, dev)
                J52 = J52.astype(np.float32)
                pel_out = s_["pel"][0].cpu().numpy().astype(np.float32)
                R_out = s_["R_obj"][0, 0].cpu().numpy().astype(np.float32)
                p_out = s_["p_obj"][0, 0].cpu().numpy().astype(np.float32)
                floor_info = dict(output_floor=floor)
                if floor == "interact":                        # rigid Y shift onto the InterAct / LIGHT floor
                    nfl = floor_lib.n_floor_frames(W)
                    ov = floor_lib.object_mesh_vertices(lroot, str(SM[seq]), str(b["obj_name"][0]))
                    dy, fd = floor_lib.floor_shift(verts[:nfl], R_out[:nfl], p_out[:nfl], ov)
                    J52, pel_out, p_out = (floor_lib.apply_shift(a, dy) for a in (J52, pel_out, p_out))
                    floor_info.update(floor_dy_m=float(dy), floor_n_frames=int(nfl), floor_from=fd["floor_from"])
                prov = dict(refine_ckpt=os.path.abspath(ckpt), arm=str(ck.get("arm")), it=int(ck.get("it")),
                            sensor_arm="b2", periodic=bool(ck.get("periodic")),
                            gen_npz=os.path.abspath(f), gen_ckpt=str(z["ckpt"]), f0=int(f0), window=int(W),
                            piece_len=int(Tn), rollout_T=int(T), floor_strategy=FLOOR_STRATEGY, hand_mode=hm,
                            hand_source=str(it["hand_source"]), ik_err_mm=ik_err, **floor_info,
                            builder="pami/refine/export.py (bankless)", git_commit=commit)
                extra = {}
                if "contact_logit" in z.files:                 # generator contact head pass-through
                    cl = np.asarray(z["contact_logit"])[f0:f0 + W].astype(np.float32)
                    extra = dict(contact_logit=cl, contact_prob=(1.0 / (1.0 + np.exp(-cl))).astype(np.float32),
                                 contact_joints=np.asarray([7, 8, 10, 11], dtype=np.int64))
                if floor == "interact":
                    extra["floor_dy_m"] = np.float64(floor_info["floor_dy_m"])
                np.savez(out, joints52=J52,
                         rot6=s_["rot6"][0].cpu().numpy().astype(np.float32),
                         rot6h=s_["rot6h"][0].cpu().numpy().astype(np.float32),
                         pel=pel_out, R_obj=R_out, p_obj=p_out,
                         obj_name=str(b["obj_name"][0]), caption=_caption(z), seq=seq, f0=np.int64(f0),
                         win=np.int32(W), provenance=json.dumps(prov), **extra)
                done.add(seq)
                n += 1
                print(f"[{n}] {SM.get(seq, '?')}/{seq} T={Tn} W={W} fit={ik_err:.1f}mm -> {out}", flush=True)
            except Exception as e:
                n_fail += 1
                print(f"[FAIL] {fseq}: {type(e).__name__}: {e}", flush=True)
    print(f"[refine] wrote {n} refined npz, failures {n_fail} -> {out_dir}", flush=True)
    return n, n_fail


def main(argv=None, default_T=4):
    ap = argparse.ArgumentParser(description="refine PamiGen samples with the PamiRefiner and export joints52 / object pose")
    ap.add_argument("--samples_glob", required=True, help="gen_*.npz glob")
    ap.add_argument("--ckpt", default=os.path.join(_REPO, "checkpoints", "pamirefiner.pt"), help="refiner checkpoint")
    ap.add_argument("--out_dir", required=True, help="ONLY directory this writes into")
    ap.add_argument("--seqs", default=None, help="optional filter: comma-separated seq names, or a json file")
    ap.add_argument("--f0", type=int, default=0, help="window start inside the piece")
    ap.add_argument("--max_frames", type=int, default=300, help="window cap: W = min(piece_len - f0, max_frames)")
    ap.add_argument("--T", type=int, default=default_T, help="refiner recursion steps (paper: 4)")
    ap.add_argument("--overwrite", action="store_true", help="redo sequences whose output already exists")
    ap.add_argument("--floor", choices=["interact", "native"], default="interact",
                    help="interact (default): place the output on the InterAct / LIGHT floor (lowest human or object "
                         "mesh vertex of the first 30 frames at y = 0; needs LIGHT_ROOT); native: keep the joint-based "
                         "floor of the 272-d representation")
    ap.add_argument("--light_root", default=None, help="LIGHT folder with InterAct/<subset>/objects (default: $LIGHT_ROOT)")
    a = ap.parse_args(argv)
    files = sorted(glob.glob(a.samples_glob))
    if not files:
        raise SystemExit(f"no files match {a.samples_glob}")
    refine_files(files, a.ckpt, a.out_dir, seqs=a.seqs, f0=a.f0, max_frames=a.max_frames, T=a.T, overwrite=a.overwrite,
                 floor=a.floor, light_root=a.light_root)


if __name__ == "__main__":
    main()
