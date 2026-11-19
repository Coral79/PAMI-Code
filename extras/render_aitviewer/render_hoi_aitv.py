"""Render InterAct human-object interaction sequences (SMPL-H / SMPL-X body + object mesh) with aitviewer, headless.
One or more panels side by side: the ground truth (mocap) and/or exported results of `refine.py`
(`refined_<sequence>.npz`: rot6 22x6 body, rot6h 30x6 hands, pel, R_obj / p_obj). The object mesh is read from
`$LIGHT_ROOT/InterAct/<subset>/objects`, the caption from the test list of `prepare_data.py test_pool`.

Style: white background, world-fixed checkerboard floor, opaque body and object, key light with short shadows, caption
banner on top. The camera follows a smoothed body + object centre per panel (`--cam fixed` for a tripod fitted to the
whole track). 1200x1100 per panel, 15 fps, libx264 crf 20.

Needs aitviewer (https://github.com/eth-ait/aitviewer) and ffmpeg in addition to the model environment, and the same
environment variables as `refine.py` (INTERACT_ROOT, REPR272_ROOT / BODY_MODELS, LIGHT_ROOT). Run from the repository root:
    # ground truth only
    python extras/render_aitviewer/render_hoi_aitv.py --seq Date03_Sub04_trashbin_1004 --gt
    # ground truth next to a refined draw ({k} = draw index from --rep)
    python extras/render_aitviewer/render_hoi_aitv.py --seq sub17_woodchair_018 --gt --rep 3 \
        --panels outputs/refined_rep{k} --labels 'PAMI'
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, textwrap

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from pami.data_prep import subsets as ic
from pami.refine import floor as floor_lib
from pami.refine.vertex_m3 import full_verts_joints
from pami.vendor.face_z_align_util import rotation_6d_to_matrix

from pami.data_prep import paths as P
POOL_JSON = os.path.join(P.PAMI_DATA_ROOT, "test_pool", "interact_test.json")

from aitviewer.configuration import CONFIG as C
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.plane import ChessboardPlane
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.utils import spherical_coordinates_from_direction

CAM_BASE = np.array([2.6, 0.9, 2.6], np.float32)
FONT_DIR = os.path.join(os.path.dirname(ImageFont.__file__), "..", "matplotlib", "mpl-data", "fonts", "ttf")


def hex_rgb(h, a=1.0):
    h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (a,)


def font(name, size):
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    except Exception:
        return ImageFont.load_default()


def _bm_path(meta):
    return ic.model_npz_path(meta["model"], meta["gender"])


class Seq:
    """Per-sequence body model + object mesh shared by all panels."""

    def __init__(self, seq, dev):
        self.meta = m = ic.load_seq(seq)
        self.dev = dev
        self.nb = int(m["nb"])
        self.bm = ic.body_model(m["gender"], num_betas=self.nb, model=m["model"]).to(dev)
        for p in self.bm.parameters():
            p.requires_grad_(False)
        self.betas = torch.from_numpy(np.asarray(m["betas16"], np.float32)[:self.nb]).to(dev)
        self.faces = np.asarray(self.bm.f.cpu() if torch.is_tensor(self.bm.f) else self.bm.f)
        kt = self.bm.kintree_table
        self.parents = np.asarray(kt.cpu() if torch.is_tensor(kt) else kt)[0]
        ov, self.of = floor_lib.object_mesh(floor_lib.light_root(), m["subset"], m["obj_name"])   # the mesh refine.py places the floor with
        self.ov = ov.astype(np.float32)

    def obj_track(self, R, p, T):
        return np.einsum("tij,nj->tni", np.asarray(R, np.float32)[:T], self.ov) + np.asarray(p, np.float32)[:T, None]

    def gt(self, max_T):
        m = self.meta
        T = min(int(m["T"]), max_T)
        P = np.asarray(m["poses"], np.float32)[:T].copy()
        if m["hand_mean"]:  # chairs / intercap store hands without the MANO mean; BodyModel is flat-hand
            P[:, 66:156] += ic.hand_means(_bm_path(m)).astype(np.float32)
        P = torch.from_numpy(P).to(self.dev)
        tr = torch.from_numpy(np.asarray(m["trans"], np.float32)[:T]).to(self.dev)
        with torch.no_grad():
            o = self.bm(root_orient=P[:, :3], pose_body=P[:, 3:66], pose_hand=P[:, 66:156],
                        betas=self.betas[None].expand(T, self.nb), trans=tr)
        return o.v.cpu().numpy(), self.obj_track(m["obj_R"], m["obj_t"], T), o.Jtr.cpu().numpy()

    def refined(self, z, max_T):
        """refined_<seq>.npz -> (verts, obj verts, joints); pelvis pinned to `pel`."""
        assert int(z["f0"]) == 0, "renderer assumes the window starts at frame 0"
        T = min(int(z["rot6"].shape[0]), max_T)
        t = lambda k: torch.from_numpy(np.asarray(z[k], np.float32)[:T]).to(self.dev)
        m = self.meta
        with torch.no_grad():                                    # the same body-model pass refine.py exports joints52 with
            V, _, J = full_verts_joints(t("rot6"), t("rot6h"), t("pel"), torch.from_numpy(np.asarray(m["betas16"], np.float32)).to(self.dev),
                                        m["gender"], m["model"], self.nb, self.dev)
        return V, self.obj_track(z["R_obj"], z["p_obj"], T), J


def body_facing(J, n_first=10):
    """Horizontal facing direction of the body over the first n_first frames (unit XZ vector), so the opening frame
    is seen from the front. SMPL joints: 1 = left hip, 2 = right hip; forward = up x (right - left) = (r_z, 0, -r_x)."""
    r = J[:n_first, 2] - J[:n_first, 1]
    f = np.stack([r[:, 2], np.zeros(len(r)), -r[:, 0]], 1)
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
    fm = f.mean(0); fm[1] = 0.0
    return fm / (np.linalg.norm(fm) + 1e-8)


def camera_track(V, O, J, mode, zoom=1.0, ema_frames=12, az="auto", az_offset=45.0):
    """Per-frame (target, position). `fixed`: one tripod fitted to the whole track (render_recon.fixed_cam).
    `follow`: target = EMA-smoothed per-frame body+object centre (XZ), constant height and offset.
    Camera azimuth: `auto` = in front of the body's mean facing direction, rotated by az_offset deg (front 3/4 view), so
    panels rendered in different world frames (GT / LIGHT / InterAct / ours) all show the person from the same relative
    viewpoint; `world` = fixed +x+z offset like render_recon.fixed_cam; a number = absolute azimuth in degrees."""
    T = V.shape[0]
    pts = np.concatenate([V.reshape(-1, 3)[::67], O.reshape(-1, 3)[::11]])
    lo, hi = pts.min(0), pts.max(0)
    c_all = (lo + hi) / 2
    if mode == "fixed":
        scale = max(1.0, float(max(hi[0] - lo[0], hi[2] - lo[2])) / 2.4)
        tgt = np.repeat(c_all[None], T, 0)
    else:
        ctr = np.stack([np.concatenate([V[t][::23], O[t][::7]]).mean(0) for t in range(T)])
        ext = max(float(np.max(np.ptp(np.concatenate([V[t][::23], O[t][::7]]), 0)[[0, 2]])) for t in range(T))
        scale = max(1.0, ext / 2.4)
        sm = np.zeros_like(ctr); a = 1.0 / ema_frames; sm[0] = ctr[0]
        for t in range(1, T):
            sm[t] = (1 - a) * sm[t - 1] + a * ctr[t]
        tgt = sm
        tgt[:, 1] = c_all[1]
    dist = float(np.linalg.norm(CAM_BASE)) * scale * zoom
    el = np.arctan2(CAM_BASE[1], np.linalg.norm(CAM_BASE[[0, 2]]))
    if az == "world":
        h = CAM_BASE[[0, 2]] / np.linalg.norm(CAM_BASE[[0, 2]])
    else:
        if az == "auto":
            f = body_facing(J); ang = np.arctan2(f[0], f[2]) + np.deg2rad(az_offset)
        else:
            ang = np.deg2rad(float(az))
        h = np.array([np.sin(ang), np.cos(ang)])
    off = np.array([np.cos(el) * h[0], np.sin(el), np.cos(el) * h[1]], np.float32) * dist
    return tgt, tgt + off[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out_dir", default=os.path.join(REPO, "outputs", "videos"))
    ap.add_argument("--gt", action="store_true", help="add a GT (mocap) panel first")
    ap.add_argument("--panels", default="", help="comma list of refine.py output dirs (absolute or relative to the working directory); '{k}' -> --rep")
    ap.add_argument("--labels", default=None, help="'A|B|...' labels for --panels (default: dir names)")
    ap.add_argument("--rep", type=int, default=None, help="draw index substituted for '{k}' in --panels")
    ap.add_argument("--pool", default=POOL_JSON, help="pool json with per-seq text (caption)")
    ap.add_argument("--fps", type=int, default=15, help="output fps (source is 30; 15 = half-speed playback, no frame drop)")
    ap.add_argument("--w", type=int, default=1200)
    ap.add_argument("--h", type=int, default=1100)
    ap.add_argument("--cam", choices=["follow", "fixed"], default="follow")
    ap.add_argument("--zoom", type=float, default=1.25, help="camera distance multiplier; <0.9 crops the body when it walks toward the camera")
    ap.add_argument("--cam_az", default="auto", help="camera azimuth: 'auto' = front 3/4 view relative to the body's mean facing (comparable across panels), 'world' = fixed +x+z, or degrees")
    ap.add_argument("--cam_az_offset", type=float, default=45.0, help="degrees off the facing direction for 'auto' (0 = straight in front)")
    ap.add_argument("--max_T", type=int, default=300)
    ap.add_argument("--obj_smooth", default="", help="'Label:sigma,...' Gaussian-smooth the object rigid pose (quaternion + translation, sigma in source frames) of the named panels")
    ap.add_argument("--body_smooth", default="", help="'Label:sigma,...' Gaussian-smooth the body pose (per-joint quaternion) + pelvis of the named panels")
    ap.add_argument("--hand_smooth", default="", help="'Label:sigma,...' Gaussian-smooth the finger pose (rot6h) of the named panels")
    ap.add_argument("--panel_max_s", default="", help="'Label:sec,...' per-panel length cap in OUTPUT seconds (raw videos only; e.g. 'Ours:7')")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--skeleton", choices=["full", "body", "none"], default="none",
                    help="full = body + finger joints, body = first 22 joints only (fingers shown by the mesh), none")
    ap.add_argument("--body_color", default="85c7f2", help="hex")
    ap.add_argument("--skel_color", default="295c9e", help="hex")
    ap.add_argument("--obj_color", default="f0955a", help="hex")
    ap.add_argument("--body_alpha", type=float, default=1.0)
    ap.add_argument("--obj_alpha", type=float, default=1.0)
    ap.add_argument("--ambient", type=float, default=1.3, help="scene ambient strength (aitviewer default 2.0 washes colours out)")
    ap.add_argument("--skel_radius", type=float, default=0.012)
    ap.add_argument("--light_pos", default="5,12,9", help="x,y,z of the shadow-casting key light (aitviewer default 0,10,15 gives long shadows)")
    ap.add_argument("--key_strength", type=float, default=0.8, help="key (shadow) light strength; 1.0 over-exposes top surfaces with a steep light")
    ap.add_argument("--floor_diffuse", type=float, default=0.5, help="floor diffuse coeff; higher = darker, more visible shadows (aitviewer default 0.1)")
    ap.add_argument("--floor_size", default="inf", help="'auto' = fit the whole track's XZ extent + margin; a number = fixed side length in m; 'inf' = aitviewer 100 m plane")
    ap.add_argument("--floor_margin", type=float, default=2.0)
    ap.add_argument("--floor_y", default="auto", help="'auto' = per panel, lowest body vertex over the window (lowest foot); a number = fixed height for all panels")
    ap.add_argument("--floor_colors", default="fafafa,e6e6e6", help="hex,hex checkerboard colours; matches the on-screen brightness of the aitviewer default floor under its default ambient 2.0 (~251/230)")
    ap.add_argument("--frames", default=None, help="optional 'start:end' frame range to render (debug)")
    ap.add_argument("--label_gt", default="GT")
    ap.add_argument("--no_compose", action="store_true", help="only write the per-panel raw (text-free) videos")
    ap.add_argument("--fail", default=None, help="JSON (string or @file): {label: [{start, end, type, part}]} severe-failure intervals in OUTPUT-video "
                    "seconds (fps = --fps); inside an interval the object recolours (penetration #E26767 / floating #8E78D2) and the named body part "
                    "is tinted (#E8A1A8 / #D4C8F0); outside, canonical colours. Same rule for every panel")
    ap.add_argument("--fail_obj_mix", type=float, default=1.0, help="tint style: how far the object moves toward the warning colour inside an interval (0 = object keeps its canonical colour, 1 = full)")
    ap.add_argument("--fail_style", choices=["tint", "badge"], default="tint",
                    help="tint: recolour human/object inside the interval; badge: leave the render untouched and show a small corner label "
                         "('Penetration' / 'Floating') that fades in/out with the interval")
    ap.add_argument("--fail_halo", type=float, default=0.35, help="peak opacity of the soft warning-colour floor glow under the object inside an interval (0 = none)")
    ap.add_argument("--fail_fade", type=int, default=3, help="fade-in/out length in output frames at each interval end")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S = Seq(args.seq, dev)
    m = S.meta

    # ---- panels: (label, V, O, J)
    panels = []
    if args.gt:
        panels.append((args.label_gt, *S.gt(args.max_T)))
    dirs = [d for d in args.panels.split(",") if d]
    labels = args.labels.split("|") if args.labels else dirs
    assert len(labels) == len(dirs), "--labels count must match --panels"
    for d, lab in zip(dirs, labels):
        if "{k}" in d:
            assert args.rep is not None, f"--rep needed for {d}"
            d = d.replace("{k}", str(args.rep))
        f = os.path.abspath(d)
        f = f"{f}/refined_{args.seq}.npz"
        if not os.path.exists(f):
            print(f"[skip panel] {lab}: {f}")
            continue
        z = dict(np.load(f, allow_pickle=True))
        sm = {k: float(v) for k, v in (x.split(":") for x in args.obj_smooth.split(",") if x)}
        if sm.get(lab, 0) > 0:
            from scipy.ndimage import gaussian_filter1d
            from scipy.spatial.transform import Rotation as Rot
            q = Rot.from_matrix(np.asarray(z["R_obj"], np.float64)).as_quat()
            for t_ in range(1, len(q)):
                if np.dot(q[t_], q[t_ - 1]) < 0: q[t_] = -q[t_]
            q = gaussian_filter1d(q, sm[lab], axis=0, mode="nearest"); q /= np.linalg.norm(q, axis=1, keepdims=True)
            z["R_obj"] = Rot.from_quat(q).as_matrix().astype(np.float32)
            z["p_obj"] = gaussian_filter1d(np.asarray(z["p_obj"], np.float64), sm[lab], axis=0, mode="nearest").astype(np.float32)
            print(f"  [smooth] {lab}: object pose sigma={sm[lab]} frames")
        # body / hand pose smoothing (per-joint quaternion Gaussian, sign-continuous; pelvis translation Gaussian)
        bsm = {k: float(v) for k, v in (x.split(":") for x in args.body_smooth.split(",") if x)}
        hsm = {k: float(v) for k, v in (x.split(":") for x in args.hand_smooth.split(",") if x)}
        if bsm.get(lab, 0) > 0 or hsm.get(lab, 0) > 0:
            from scipy.ndimage import gaussian_filter1d
            from scipy.spatial.transform import Rotation as Rot
            def smooth_r6(r6, sig):
                r6 = np.asarray(r6, np.float32); Tn, Jn = r6.shape[:2]
                M = rotation_6d_to_matrix(torch.from_numpy(r6).reshape(-1, 6)).numpy().reshape(Tn, Jn, 3, 3).astype(np.float64)
                out = np.empty_like(M)
                for j in range(Jn):
                    q = Rot.from_matrix(M[:, j]).as_quat()
                    for t_ in range(1, Tn):
                        if np.dot(q[t_], q[t_ - 1]) < 0: q[t_] = -q[t_]
                    q = gaussian_filter1d(q, sig, axis=0, mode="nearest"); q /= np.linalg.norm(q, axis=1, keepdims=True)
                    out[:, j] = Rot.from_quat(q).as_matrix()
                return np.ascontiguousarray(out[:, :, :2, :].reshape(Tn, Jn, 6)).astype(np.float32)  # first two rows -> 6D (pytorch3d convention)
            if bsm.get(lab, 0) > 0:
                z["rot6"] = smooth_r6(z["rot6"], bsm[lab])
                z["pel"] = gaussian_filter1d(np.asarray(z["pel"], np.float64), bsm[lab], axis=0, mode="nearest").astype(np.float32)
                print(f"  [smooth] {lab}: body pose + pelvis sigma={bsm[lab]} frames")
            if hsm.get(lab, 0) > 0:
                z["rot6h"] = smooth_r6(z["rot6h"], hsm[lab])
                print(f"  [smooth] {lab}: hand pose sigma={hsm[lab]} frames")
        panels.append((lab, *S.refined(z, args.max_T)))
    assert panels, "nothing to render (use --gt and/or --panels)"
    T = min(p[1].shape[0] for p in panels)
    panels = [(lab, V[:T], O[:T], J[:T]) for lab, V, O, J in panels]
    N = len(panels)
    pool = json.load(open(args.pool)) if os.path.exists(args.pool) else []
    pool = {r["name"]: r for r in (pool["rows"] if isinstance(pool, dict) else pool)}
    caption = pool[args.seq]["text"] if args.seq in pool else ""
    # floor per panel: lowest BODY vertex over the whole (untruncated) sequence (body only, not the object)
    floor_ys = [float(V[:, :, 1].min()) if args.floor_y == "auto" else float(args.floor_y) for _, V, _, _ in panels]
    cap = {k: int(round(float(v) * args.fps)) for k, v in (x.split(":") for x in args.panel_max_s.split(",") if x)}
    panel_T = [min(T, cap.get(lab, T)) for lab, *_ in panels]
    floor_y = min(floor_ys)
    print(f"{args.seq}: {m['subset']}/{m['obj_name']} {m['model']}-{m['gender']} T={T} panels={[p[0] for p in panels]} floor_y={[round(f, 3) for f in floor_ys]}")

    # ---- scene
    C.update_conf({"auto_set_floor": False, "background_color": [1.0, 1.0, 1.0, 1.0], "z_up": False})
    v = HeadlessRenderer(size=(args.w, args.h))
    v.auto_set_floor = False
    v.auto_set_camera_target = False
    v.scene.origin.enabled = False
    v.scene.ambient_strength = args.ambient
    # key light: steeper = shorter shadows. Front light (index 1) is the shadow caster in aitviewer's default scene.
    lp = np.array([float(x) for x in args.light_pos.split(",")])
    key = v.scene.lights[1]
    key.elevation, key.azimuth = spherical_coordinates_from_direction(-lp / np.linalg.norm(lp), degrees=True)
    key.strength = args.key_strength

    fa, fb = (hex_rgb(h) for h in args.floor_colors.split(","))
    TILE = 0.5
    if args.floor_size != "inf":
        pts = np.concatenate([np.concatenate([V.reshape(-1, 3)[::67], O.reshape(-1, 3)[::11]]) for _, V, O, _ in panels])
        side = float(np.ptp(pts[:, [0, 2]], 0).max()) + 2 * args.floor_margin if args.floor_size == "auto" else float(args.floor_size)
        n_tiles = max(2, int(round(side / TILE))); side = n_tiles * TILE
        v.scene.remove(v.scene.floor)
        v.scene.floor = ChessboardPlane(side, n_tiles, fa, fb, "xz", name="Floor")
        v.scene.add(v.scene.floor)
        cxz = (pts[:, [0, 2]].min(0) + pts[:, [0, 2]].max(0)) / 2
        v.scene.floor.position = np.array([cxz[0], floor_y, cxz[1]], np.float32)
    else:
        v.scene.floor.c1, v.scene.floor.c2 = np.array(fa), np.array(fb)
        v.scene.floor.position[1] = floor_y
    v.scene.floor.update_transform(parent_transform=v.scene.model_matrix)
    # floor shading: lit tiles stay at the base colour (ambient + diffuse * sum_lights cos = 1); a shadow removes only the
    # key light's diffuse share, so floor_diffuse * cos_key is how much darker the shadow is.
    cos_sum = sum(max(0.0, float(-l.direction[1])) * l.strength for l in v.scene.lights)
    v.scene.floor.material.diffuse = args.floor_diffuse
    v.scene.floor.material.ambient = max(0.0, 1.0 - args.floor_diffuse * cos_sum) / args.ambient

    # ---- severe-failure intervals -> per-panel per-frame object colour + body vertex colours:
    # the annotated body PART is the primary cue (full warning colour), the object gets a weaker tint
    # (--fail_obj_mix); no part -> object takes the full colour. 3-frame fade in/out inside the exact interval.
    # annotation-overlay look. Human keeps its blue and gets a moderate whole-body tint
    # (#C099AC / #999FD6); the object carries the strong warning colour (#E36C72 / #8E78D2); optional faint edge halo.
    FAIL_COL = {"penetration": np.array(hex_rgb("C099AC")[:3]), "floating": np.array(hex_rgb("999FD6")[:3])}
    FAIL_OBJ = {"penetration": np.array(hex_rgb("E36C72")[:3]), "floating": np.array(hex_rgb("8E78D2")[:3])}
    HALO_COL = {"penetration": (227, 108, 114), "floating": (142, 120, 210)}
    halo_w = {}                                                   # panel -> (T,) halo strength, halo_c -> (T,3) colour
    PART_KEYS = {"R_arm": ["rarm"], "L_arm": ["larm"], "both_arms": ["larm", "rarm"], "root": ["root"], "L_leg": ["lleg"],
                 "R_leg": ["rleg"], "torso": ["torso"], "whole_body": ["root", "torso", "larm", "rarm", "lleg", "rleg"]}
    fail = {}
    if args.fail:
        fail = json.load(open(args.fail[1:])) if args.fail.startswith("@") else json.loads(args.fail)
    obj_col = {}; body_vc = {}
    if fail:
        pidx = ic.part_vertex_index(m["model"])
        base_o = np.array(hex_rgb(args.obj_color)[:3]); base_b = np.array(hex_rgb(args.body_color)[:3])
        for i, (lab, V, O, J) in enumerate(panels):
            ivs = [iv for iv in fail.get(lab, []) if iv.get("type") in FAIL_COL and iv.get("start") is not None]
            if not ivs:
                continue
            oc = np.tile(np.append(base_o, args.obj_alpha).astype(np.float32), (T, 1))
            vc = np.tile(np.append(base_b, args.body_alpha).astype(np.float32), (T, V.shape[1], 1))
            hw = np.zeros(T, np.float32); hc = np.zeros((T, 3), np.uint8); htype = [""] * T
            for iv in ivs:
                f0_, f1_ = max(0, int(round(iv["start"] * args.fps))), min(T, int(round(iv["end"] * args.fps)))
                if f1_ <= f0_:
                    continue
                n = f1_ - f0_; ramp = np.ones(n, np.float32); k = min(args.fail_fade, n // 2)
                if k > 0:
                    ramp[:k] = np.linspace(1.0 / k, 1.0, k); ramp[-k:] = np.linspace(1.0, 1.0 / k, k)
                wc, woc = FAIL_COL[iv["type"]], FAIL_OBJ[iv["type"]]
                parts = PART_KEYS.get(iv.get("part") or "whole_body", PART_KEYS["whole_body"])   # no part -> whole human
                for j, t_ in enumerate(range(f0_, f1_)):
                    w = ramp[j]
                    if args.fail_style == "tint":
                        oc[t_, :3] = base_o * (1 - w * args.fail_obj_mix) + woc * (w * args.fail_obj_mix)
                        for pk in parts:
                            vc[t_, pidx[pk], :3] = base_b * (1 - w) + wc * w
                    hw[t_] = w; hc[t_] = HALO_COL[iv["type"]]; htype[t_] = iv["type"]
                print(f"  [fail] {lab}: {iv['type']} part={iv.get('part') or 'whole_body'} frames {f0_}-{f1_}")
            obj_col[i] = oc; body_vc[i] = vc; halo_w[i] = (hw, hc, htype)
    halo_mask = None
    glow_nodes = {}
    if halo_w and args.fail_halo > 0:
        # soft floor glow under the object (radial alpha disc, unlit), only inside the interval, scaled by the fade ramp
        NS, NR = 48, 4
        ang = np.linspace(0, 2 * np.pi, NS, endpoint=False)
        ring_r = np.linspace(0, 1, NR + 1)[1:]
        pts = [np.zeros(3)]; alphas = [1.0]
        for r in ring_r:
            for a_ in ang:
                pts.append(np.array([np.cos(a_) * r, 0, np.sin(a_) * r])); alphas.append(float((1 - r) ** 1.5))
        pts = np.array(pts, np.float32); alphas = np.array(alphas, np.float32)
        faces = []
        for k in range(NS):                                   # centre fan
            faces.append([0, 1 + k, 1 + (k + 1) % NS])
        for ri in range(NR - 1):                              # ring quads
            b0, b1 = 1 + ri * NS, 1 + (ri + 1) * NS
            for k in range(NS):
                k1 = (k + 1) % NS
                faces += [[b0 + k, b1 + k, b1 + k1], [b0 + k, b1 + k1, b0 + k1]]
        faces = np.array(faces, np.int64)
        for i, (hw, hc) in halo_w.items():
            O_ = panels[i][2]; ctr = O_.mean(1)                                # (T,3) object centre
            ext = float(np.median(np.ptp(O_, axis=1)[:, [0, 2]].max(1)))       # object footprint -> glow radius
            R = np.clip(0.9 * ext + 0.25, 0.35, 1.2)
            verts = ctr[:, None, :] + pts[None] * R
            verts[:, :, 1] = floor_ys[i] + 0.01
            col = np.zeros((T, len(pts), 4), np.float32)
            col[..., :3] = hc[:, None, :].astype(np.float32) / 255.0
            col[..., 3] = alphas[None] * hw[:, None] * args.fail_halo
            from aitviewer.scene.material import Material
            g = Meshes(verts, faces, vertex_colors=col, name=f"glow{i}", material=Material(ambient=1.0 / args.ambient, diffuse=0.0, specular=0.0))
            v.scene.add(g); glow_nodes[i] = g

    # one scene, all panels' nodes; per frame only one panel's nodes are enabled
    nodes, cams = [], []
    for i, (lab, V, O, J) in enumerate(panels):
        body = Meshes(V, S.faces, vertex_colors=body_vc[i], name=f"body{i}") if i in body_vc else \
               Meshes(V, S.faces, color=hex_rgb(args.body_color, args.body_alpha), name=f"body{i}")
        ns = [body, Meshes(O, S.of, color=hex_rgb(args.obj_color, args.obj_alpha), name=f"obj{i}")]
        if args.skeleton != "none":
            nj = 22 if args.skeleton == "body" or J.shape[1] != len(S.parents) else len(S.parents)
            conn = np.array([(p, j) for j, p in enumerate(S.parents[:nj]) if p >= 0])
            ns.append(Skeletons(J[:, :nj], conn, radius=args.skel_radius, color=hex_rgb(args.skel_color), name=f"skel{i}"))
        for n in ns:
            v.scene.add(n)
        nodes.append(ns)
        cams.append(camera_track(V, O, J, args.cam, zoom=args.zoom, az=args.cam_az, az_offset=args.cam_az_offset))
    cam = v.scene.camera

    # ---- banner: caption (wrapped, fixed height for constant frame size) + per-panel label
    # caption = big bold title; panel labels = smaller regular
    fc, fl, ft = font("DejaVuSans-Bold.ttf", 46), font("DejaVuSans.ttf", 36), font("DejaVuSans.ttf", 24)
    CW = args.w * N
    lines = textwrap.wrap(caption, width=max(20, int(CW / (46 * 0.6)) - 4)) or [""]
    LH = 56; CAP_H = 14 + LH * len(lines) + 30; LAB_H = 56; BAN = CAP_H + LAB_H   # 30 px gap title -> method labels

    def slug(x):
        return "".join(ch if ch.isalnum() else "_" for ch in x).strip("_").lower()

    def encode(frames_dir, out):
        ffmpeg = shutil.which("ffmpeg") or os.path.join(sys.prefix, "bin", "ffmpeg")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", f"{frames_dir}/%04d.png",
                        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", str(args.crf),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out], check=True)

    os.makedirs(args.out_dir, exist_ok=True)
    raw_dir = os.path.join(args.out_dir, "raw"); os.makedirs(raw_dir, exist_ok=True)
    stem = f"{m['subset']}_{m['obj_name']}_{args.seq}"
    tmp = tempfile.mkdtemp(prefix="aitvhoi_")
    try:
        for i in range(N):
            os.makedirs(f"{tmp}/p{i}")
        if not args.no_compose:
            os.makedirs(f"{tmp}/cmp")
        f0, f1 = (int(x) for x in args.frames.split(":")) if args.frames else (0, T)
        for t in range(f0, f1):
            if not args.no_compose:
                im = Image.new("RGB", (CW, BAN + args.h), (255, 255, 255)); d = ImageDraw.Draw(im)
                for li, ln in enumerate(lines):
                    wl = d.textlength(ln, font=fc); d.text(((CW - wl) / 2, 14 + li * LH), ln, fill=(0, 0, 0), font=fc)
            v.scene.current_frame_id = t
            for i, (lab, V, O, J) in enumerate(panels):
                for j, ns in enumerate(nodes):
                    for n in ns:
                        n.enabled = (j == i)
                if i in obj_col:
                    nodes[i][1].color = tuple(float(x) for x in obj_col[i][t])
                for gi, g in glow_nodes.items():
                    g.enabled = (gi == i) and halo_w[gi][0][t] > 0
                tgt, pos = cams[i]
                cam.target = tgt[t]; cam.position = pos[t]
                v.scene.floor.position[1] = floor_ys[i]          # floor at this panel's lowest foot
                v.scene.floor.update_transform(parent_transform=v.scene.model_matrix)
                img = v.get_frame().convert("RGB")
                if i in halo_w and halo_w[i][0][t] > 0:   # label in both styles
                    w_ = float(halo_w[i][0][t]); ty = halo_w[i][2][t]
                    BADGE = {"penetration": ("Penetration", (217, 79, 87)), "floating": ("Floating", (118, 95, 192))}
                    txt, col = BADGE[ty]
                    fb = font("DejaVuSans-Bold.ttf", 34)
                    ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
                    tw = dr.textlength(txt, font=fb); x0, y0 = 28, 24; pad = 14
                    dr.rounded_rectangle([x0, y0, x0 + tw + 2 * pad, y0 + 34 + 2 * pad - 4], radius=12, fill=(*col, int(235 * w_)))
                    dr.text((x0 + pad, y0 + pad - 6), txt, fill=(255, 255, 255, int(255 * w_)), font=fb)
                    img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
                if t - f0 < panel_T[i]:
                    img.save(f"{tmp}/p{i}/{t - f0:04d}.png")     # raw panel, no text; per-panel length cap
                if not args.no_compose:
                    wl = d.textlength(lab, font=fl); d.text((i * args.w + (args.w - wl) / 2, CAP_H + 4), lab, fill=(0, 0, 0), font=fl)
                    im.paste(img, (i * args.w, BAN))
            if not args.no_compose:
                d.text((10, BAN + args.h - 30), f"{args.seq}  t={t}", fill=(90, 90, 90), font=ft)
                im.save(f"{tmp}/cmp/{t - f0:04d}.png")
            if t % 50 == 0:
                print(f"  frame {t}/{T}")
        for i, (lab, *_r) in enumerate(panels):
            out = f"{raw_dir}/{stem}__{slug(lab)}.mp4"
            encode(f"{tmp}/p{i}", out); print("wrote", out)
        if not args.no_compose:
            tag = "gt" if (args.gt and N == 1) else f"compare{N}"
            out = f"{args.out_dir}/{stem}_{tag}.mp4"
            encode(f"{tmp}/cmp", out); print("wrote", out, T, "frames")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
