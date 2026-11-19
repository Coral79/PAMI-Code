"""Filesystem roots for the data-preparation pipeline (all overridable by environment variables; the defaults
are folders under <repo>/data: InterAct/data, pami_data, repr272, LIGHT).
BPS_BASIS          InterAct's assets/bps_basis_set_1024_1.npy (default: next to INTERACT_ROOT, <InterAct clone>/assets/)

INTERACT_ROOT      InterAct data tree (the `data/` dir of the InterAct clone): <subset>/sequences_canonical/<seq>/{human,object}.npz,
                   <subset>/objects/<name>/<name>.obj, <subset>/objects_bps/<name>/<name>_1024.npy, text.txt
INTERACT_RAW_ROOT  where the un-corrected chairs / neuraldome / imhd subsets live when they are kept in a
                   separate folder (default: INTERACT_ROOT, the layout of a plain InterAct download)
PAMI_DATA_ROOT     the prepared data: output/<seq>/*.npz, manifest, splits, stats (the training code's C.PKG)
REPR272_ROOT       the 272-dim motion-representation workspace: body_models/,
                   output_interact_30fps/{smpl_85,smpl_85_face_z_transform,*_joints,Representation_272},
                   index_interact_30fps.csv
BODY_MODELS        smpl/SMPL_NEUTRAL.pkl, smplh/<gender>/model.npz, smplx/SMPLX_<GENDER>.npz (+ SMPLX_MALE.pkl),
                   smpl_vert_segmentation.json, smplx_vert_segmentation.json (Meshcapade wiki)
                   (default: REPR272_ROOT/body_models)
HBP_PATH           optional directory holding a human_body_prior checkout (not needed when it is pip-installed)
LIGHT_ROOT         the LIGHT clone (same variable as evaluate.py) or directly its InterAct tree
                   (stats/test_seq*.txt + <subset>/sequences_canonical/<seq>/data.npz), used for the official
                   test list and as a caption fallback
SMPLX_PKL          any SMPL-X .pkl that carries the MANO PCA components (intercap hands are 12-dim PCA)
                   (default: BODY_MODELS/smplx/SMPLX_MALE.pkl)
T2HOI_OUT          generator corpus dir (default: PAMI_T2HOI_DATA, else PAMI_DATA_ROOT/gen_corpus)
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAMI = os.path.dirname(_HERE)
_DATA = os.path.join(os.path.dirname(_PAMI), "data")              # default location of every root: <repo>/data


def _env(key, default):
    return os.environ.get(key, default)


INTERACT_ROOT     = _env("INTERACT_ROOT", os.path.join(_DATA, "InterAct", "data"))
INTERACT_RAW_ROOT = _env("INTERACT_RAW_ROOT", INTERACT_ROOT)
PAMI_DATA_ROOT    = _env("PAMI_DATA_ROOT", os.path.join(_DATA, "pami_data"))
REPR272_ROOT      = _env("REPR272_ROOT", os.path.join(_DATA, "repr272"))
R272_OUT          = _env("R272_OUT", f"{REPR272_ROOT}/output_interact_30fps")
INDEX_CSV         = _env("INDEX_CSV", f"{REPR272_ROOT}/index_interact_30fps.csv")
BODY_MODELS       = _env("BODY_MODELS", f"{REPR272_ROOT}/body_models")          # smplx/SMPLX_*.npz, smplh/<gender>/model.npz
HBP_PATH          = _env("HBP_PATH", REPR272_ROOT)                               # dir that contains the human_body_prior package


def _light_tree(p):
    """LIGHT_ROOT may name the LIGHT clone (as evaluate.py expects) or its InterAct data tree; return the tree."""
    if p and not os.path.isdir(os.path.join(p, "stats")) and os.path.isdir(os.path.join(p, "InterAct")):
        return os.path.join(p, "InterAct")
    return p


LIGHT_ROOT        = _light_tree(_env("LIGHT_ROOT", os.path.join(_DATA, "LIGHT")))
SMPLX_PKL         = _env("SMPLX_PKL", f"{BODY_MODELS}/smplx/SMPLX_MALE.pkl")
T2HOI_OUT         = _env("T2HOI_OUT", _env("PAMI_T2HOI_DATA", os.path.join(PAMI_DATA_ROOT, "gen_corpus")))

# vertex -> body-part labels of the Meshcapade wiki (github.com/Meshcapade/wiki, assets/SMPL_body_segmentation)
SEG_JSON_SMPLX    = _env("SEG_JSON_SMPLX", f"{BODY_MODELS}/smplx_vert_segmentation.json")   # 10475 vertices
SEG_JSON_SMPL     = _env("SEG_JSON_SMPL", f"{BODY_MODELS}/smpl_vert_segmentation.json")     # 6890 vertices (SMPL-H shares the SMPL topology)
BPS_BASIS         = _env("BPS_BASIS", os.path.join(os.path.dirname(INTERACT_ROOT.rstrip("/")), "assets", "bps_basis_set_1024_1.npy"))
                                                                                 # InterAct's 1024-point BPS basis (in its repository)

OUTPUT_DIR        = f"{PAMI_DATA_ROOT}/output"
MANIFEST          = f"{PAMI_DATA_ROOT}/manifest_w64_s32.npz"
SPLIT_V1          = f"{PAMI_DATA_ROOT}/split_official.json"
SPLIT_V2          = f"{PAMI_DATA_ROOT}/split_official_v2.json"
SPLIT_V3          = f"{PAMI_DATA_ROOT}/split_official_v3_noval.json"


def describe():
    keys = ["INTERACT_ROOT", "INTERACT_RAW_ROOT", "PAMI_DATA_ROOT", "REPR272_ROOT", "R272_OUT", "INDEX_CSV",
            "BODY_MODELS", "HBP_PATH", "LIGHT_ROOT", "SMPLX_PKL", "T2HOI_OUT", "BPS_BASIS"]
    return "\n".join(f"  {k:18s} {globals()[k]}" for k in keys)
