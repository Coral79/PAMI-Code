"""InterAct subset table + per-sequence loader.

Body-model conventions per subset follow InterAct's own process/canonicalize_human.py model zoo:
  omomo      SMPL-X, 16 betas (InterAct FKs omomo with SMPLX_{gender}.npz, num_betas=16; SMPL-H would float 14 cm)
  chairs     SMPL-X, 10 betas, hand pose stored WITHOUT the MANO mean (smplx.create default
             flat_hand_mean=False) -> hand_mean=True adds hands_mean{l,r} before FK
  neuraldome SMPL-H, 16 betas          imhd  SMPL-H, 16 betas
  behave     SMPL-H, 10 betas (HOI-corrected release `behave_correct`; objects from `behave`)
  grab       SMPL-H, 10 betas (`grab_smplh`: SMPL-X->SMPL-H transfer, per-frame betas -> median; objects from `grab`)
  intercap   SMPL-X, 10 betas, 12-dim MANO-PCA hands (`intercap_correct_poses`; objects from `intercap`)

Coordinates: Y-up, floor ~ 0. FPS: 30 everywhere (no resampling).

hand_quality is stamped into hand_joints.npz["quality"] and read by the training data loader as the
per-sequence hand-loss weight (train_split --hand_w_map "strong:1.0,mid:1.0,weak:0.5,none:0.0").
The values here are the ones STORED in every hand_joints.npz the final models were trained with
(grab strong, neuraldome/imhd mid, omomo/behave/intercap weak, chairs none).
NOTE: the table was later edited ("visual judgment":
omomo/behave/intercap -> mid, neuraldome/imhd -> weak) but the files were never regenerated, so that
revision never reached any training run; the stamped values are the effective ones and are kept here.
"""
import json
import os
import sys

import numpy as np

from . import paths as P

SUBSETS = {
    "omomo":      dict(root=f"{P.INTERACT_ROOT}/omomo_correct_v2", model="smplx", nb=16, hand_mean=False,
                       hand_quality="weak"),
    "chairs":     dict(root=f"{P.INTERACT_RAW_ROOT}/chairs", model="smplx", nb=10, hand_mean=True,
                       hand_quality="none"),
    "neuraldome": dict(root=f"{P.INTERACT_RAW_ROOT}/neuraldome", model="smplh", nb=16, hand_mean=False,
                       hand_quality="mid"),
    "imhd":       dict(root=f"{P.INTERACT_RAW_ROOT}/imhd", model="smplh", nb=16, hand_mean=False,
                       hand_quality="mid"),
    "behave":     dict(root=f"{P.INTERACT_ROOT}/behave_correct", model="smplh", nb=10, hand_mean=False,
                       assets=f"{P.INTERACT_ROOT}/behave", hand_quality="weak"),
    "grab":       dict(root=f"{P.INTERACT_ROOT}/grab_smplh", model="smplh", nb=10, hand_mean=False,
                       fields="separate", obj_root=f"{P.INTERACT_ROOT}/grab", assets=f"{P.INTERACT_ROOT}/grab",
                       hand_quality="strong"),
    "intercap":   dict(root=f"{P.INTERACT_ROOT}/intercap_correct_poses", model="smplx", nb=10, hand_mean=True,
                       pca=12, assets=f"{P.INTERACT_ROOT}/intercap", hand_quality="weak"),
}
# FIXED order: it defines the sequence index (000000.npy ...) and therefore every stored file name.
SUBSET_ORDER = ["omomo", "chairs", "neuraldome", "imhd", "behave", "grab", "intercap"]

# where text.txt lives for subsets whose parameter dir is a corrected/converted copy (no HumanML text files)
TEXT_FALLBACK = {
    "behave":   [f"{P.INTERACT_ROOT}/behave/sequences_canonical"],
    "grab":     [f"{P.INTERACT_ROOT}/grab/sequences_canonical"],
    "intercap": [f"{P.INTERACT_ROOT}/intercap/sequences_canonical"],
}

# 6-part grouping of the SMPL(-X) vertex segmentation
PART_MAP = {
    "root":  ["hips"],
    "torso": ["spine", "spine1", "spine2", "neck", "head", "leftShoulder", "rightShoulder"],
    "larm":  ["leftArm", "leftForeArm", "leftHand", "leftHandIndex1"],
    "rarm":  ["rightArm", "rightForeArm", "rightHand", "rightHandIndex1"],
    "lleg":  ["leftUpLeg", "leftLeg", "leftFoot", "leftToeBase"],
    "rleg":  ["rightUpLeg", "rightLeg", "rightFoot", "rightToeBase"],
}
PARTS = ["root", "torso", "larm", "rarm", "lleg", "rleg"]
ANCH_JOINT = {"root": 0, "torso": 9, "larm": 20, "rarm": 21, "lleg": 7, "rleg": 8}       # SMPL joint ids
ANCH_NAME = {"root": "root_pelvis", "torso": "torso_spine3", "larm": "larm_wrist",
             "rarm": "rarm_wrist", "lleg": "lleg_ankle", "rleg": "rleg_ankle"}

_SEQ2SUB = None


def seq_subset_map():
    """seq_name -> subset (names are globally unique across the 7 subsets; asserted)."""
    global _SEQ2SUB
    if _SEQ2SUB is None:
        m = {}
        for sub in SUBSET_ORDER:
            for s in os.listdir(f"{SUBSETS[sub]['root']}/sequences_canonical"):
                assert s not in m, f"seq name collision across subsets: {s} in {m[s]} and {sub}"
                m[s] = sub
        _SEQ2SUB = m
    return _SEQ2SUB


def all_seqs():
    """Every sequence, sorted WITHIN each subset, subsets in SUBSET_ORDER (= index csv row order)."""
    out = []
    for sub in SUBSET_ORDER:
        out += sorted(os.listdir(f"{SUBSETS[sub]['root']}/sequences_canonical"))
    return out


def part_vertex_index(model="smplx"):
    """part -> np.array of vertex ids for the model topology (smplx 10475 / smplh 6890)."""
    f = P.SEG_JSON_SMPLX if model == "smplx" else P.SEG_JSON_SMPL
    if not os.path.isfile(f):
        raise FileNotFoundError(f"{f} not found: download the vertex segmentation files from "
                                "https://github.com/Meshcapade/wiki (assets/SMPL_body_segmentation) into the body-model folder")
    seg = json.load(open(f))
    return {p: np.concatenate([np.asarray(seg[s], np.int64) for s in subs]) for p, subs in PART_MAP.items()}


def _grab_fixed_obj_R(seq, obj_root, R_stored):
    """GRAB object-rotation convention fix. GRAB stores object `global_orient` for ROW-vector use
    (v @ R), InterAct's importer read it as column-convention, so the stored canonical rotation is
    A @ R_raw with A = C_yaw @ Rx(-90) (per-sequence constant) while the correct one is A @ R_raw^T.
    A is recovered per sequence from R_stored(t) @ R_raw(t)^T; the clip's frame offset into the 30 fps
    raw stream is found by matching body_pose rows. Any failure raises (never silently falls back)."""
    from scipy.spatial.transform import Rotation as Rot
    toks = seq.split("_")
    sbj, motion = toks[0], "_".join(toks[1:-1])
    raw_p = f"{obj_root}/raw/grab/{sbj}/{motion}.npz"
    if not os.path.exists(raw_p):
        raise FileNotFoundError(f"grab rotation fix: raw GRAB file missing for {seq}: {raw_p}")
    raw = np.load(raw_p, allow_pickle=True)
    R_raw = Rot.from_rotvec(raw["object"].item()["params"]["global_orient"][::4]).as_matrix()
    bp_raw = raw["body"].item()["params"]["body_pose"][::4]
    cx = np.load(f"{obj_root}/sequences_canonical/{seq}/human.npz", allow_pickle=True)
    bp_can = cx["poses"][:, 3:66]
    T = len(bp_can)
    if len(bp_raw) < T:
        raise RuntimeError(f"grab rotation fix: raw shorter than canonical ({len(bp_raw)} < {T}) for {seq}")
    d0 = np.abs(bp_raw - bp_can[0]).max(1)
    off = int(d0.argmin())
    if off + T > len(bp_raw) or np.abs(bp_raw[off:off + T] - bp_can).max() > 1e-3:
        raise RuntimeError(f"grab rotation fix: body_pose alignment failed for {seq} "
                           f"(off={off}, err={np.abs(bp_raw[off:off + T] - bp_can).max():.2e})")
    Rr = R_raw[off:off + T]
    A = np.einsum("tij,tkj->tik", R_stored, Rr)
    dA = np.einsum("ij,tjk->tik", A[0].T, A)
    ang = np.degrees(np.arccos(np.clip((np.einsum("tii->t", dA) - 1) / 2, -1, 1)))
    if ang.max() > 0.5:
        raise RuntimeError(f"grab rotation fix: A not constant for {seq} (max drift {ang.max():.2f} deg)")
    return np.einsum("ij,tkj->tik", A[0], Rr)


_PCA_BASIS = {}


def _mano_pca_basis():
    """(45,45) left/right MANO PCA components from an smplx-library pkl (intercap 12-dim PCA hands)."""
    if "smplx" not in _PCA_BASIS:
        import pickle
        with open(P.SMPLX_PKL, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        _PCA_BASIS["smplx"] = (np.asarray(d["hands_componentsl"], np.float64),
                               np.asarray(d["hands_componentsr"], np.float64))
    return _PCA_BASIS["smplx"]


def load_seq(seq, subset=None):
    """One canonical sequence -> dict(poses (T,156) axis-angle [root3|body63|hands90], betas16 (nb,),
    trans (T,3), gender, obj_R (T,3,3), obj_t (T,3), obj_name, T, subset, model, nb, hand_mean,
    hand_quality). `betas16` keeps its historical name but is (10,) for nb=10 subsets."""
    from scipy.spatial.transform import Rotation as Rot
    sub = subset or seq_subset_map()[seq]
    cfg = SUBSETS[sub]
    root = cfg["root"]
    h = np.load(f"{root}/sequences_canonical/{seq}/human.npz", allow_pickle=True)
    obj_root = cfg.get("obj_root", root)
    o = np.load(f"{obj_root}/sequences_canonical/{seq}/object.npz", allow_pickle=True)
    if cfg.get("fields") == "separate":                          # grab_smplh layout + per-frame betas
        poses = np.concatenate([h["global_orient"], h["body_pose"],
                                h["left_hand_pose"], h["right_hand_pose"]], 1).astype(np.float64)
        betas = h["betas"].astype(np.float64)
        assert betas.ndim == 2, f"{seq}: expected per-frame betas, got {betas.shape}"
        betas = np.median(betas, axis=0)                         # one shape per sequence (robust to outlier frames)
        trans = h["transl"].astype(np.float64)
    else:
        poses = h["poses"].astype(np.float64)
        betas = h["betas"].astype(np.float64)
        trans = h["trans"].astype(np.float64)
        if cfg.get("pca"):                                       # intercap: [root3|body63|pcaL12|pcaR12]
            k = cfg["pca"]
            assert poses.shape[1] == 66 + 2 * k, f"{seq}: pca poses {poses.shape} != 66+2*{k}"
            cl, cr = poses[:, 66:66 + k], poses[:, 66 + k:66 + 2 * k]
            hl, hr = _mano_pca_basis()
            poses = np.concatenate([poses[:, :66], cl @ hl[:k], cr @ hr[:k]], 1)   # mean EXCLUDED (added at FK)
    assert betas.shape[0] == cfg["nb"], f"{seq}: betas {betas.shape} != subset {sub} nb={cfg['nb']}"
    obj_R = Rot.from_rotvec(o["angles"]).as_matrix()
    if sub == "grab":
        obj_R = _grab_fixed_obj_R(seq, obj_root, obj_R)
    return dict(poses=poses, betas16=betas, trans=trans, gender=str(h["gender"]),
                obj_R=obj_R, obj_t=o["trans"].astype(np.float64), obj_name=str(o["name"]), T=poses.shape[0],
                subset=sub, model=cfg["model"], nb=cfg["nb"], hand_mean=cfg["hand_mean"],
                hand_quality=cfg.get("hand_quality", "none"))


def body_model(gender, num_betas=16, model="smplx"):
    """Gendered human_body_prior BodyModel, exactly InterAct's zoo: smplx -> SMPLX_{GENDER}.npz,
    smplh -> smplh/<gender>/model.npz."""
    if P.HBP_PATH not in sys.path:
        sys.path.insert(0, P.HBP_PATH)
    from human_body_prior.body_model.body_model import BodyModel
    if model == "smplx":
        fn = {"male": "SMPLX_MALE.npz", "female": "SMPLX_FEMALE.npz", "neutral": "SMPLX_NEUTRAL.npz"}[gender]
        path = os.path.join(P.BODY_MODELS, "smplx", fn)
    elif model == "smplh":
        path = os.path.join(P.BODY_MODELS, "smplh", gender, "model.npz")
    else:
        raise ValueError(f"unknown body model {model}")
    return BodyModel(bm_fname=path, num_betas=num_betas)


def model_npz_path(model, gender):
    if model == "smplx":
        fn = {"male": "SMPLX_MALE.npz", "female": "SMPLX_FEMALE.npz", "neutral": "SMPLX_NEUTRAL.npz"}[gender]
        return os.path.join(P.BODY_MODELS, "smplx", fn)
    return os.path.join(P.BODY_MODELS, "smplh", gender, "model.npz")


def hand_means(model_npz):
    """(90,) concat MANO mean hand pose [left|right] from an AMASS-style model npz."""
    d = np.load(model_npz, allow_pickle=True)
    return np.concatenate([d["hands_meanl"].ravel(), d["hands_meanr"].ravel()]).astype(np.float64)


def heading_frame(Rroot, proot, floor):
    """Per-frame canonical (272) frame: de-heading yaw rotation Hrot (T,3,3) + shift (T,3) — the same
    math representation_272.py uses (root XZ at the origin, floor at Y=0, heading removed)."""
    T = Rroot.shape[0]
    gh = -np.arctan2(Rroot[:, 0, 2], Rroot[:, 2, 2])
    cs, sn = np.cos(gh), np.sin(gh)
    Hrot = np.zeros((T, 3, 3))
    Hrot[:, 0, 0] = cs; Hrot[:, 0, 2] = sn
    Hrot[:, 1, 1] = 1
    Hrot[:, 2, 0] = -sn; Hrot[:, 2, 2] = cs
    shift = np.stack([proot[:, 0], np.full(T, floor), proot[:, 2]], 1)
    return Hrot, shift
