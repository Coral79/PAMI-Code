"""PAMI configuration: data/checkpoint locations (overridable through environment variables) and the
representation / model dimensions shared by every module.

Environment variables (same names as pami/data_prep/paths.py; the defaults are folders under <repo>/data):
  PAMI_DATA_ROOT   the prepared InterAct data (pami/data_prep output; DATA layout below)   (default: <repo>/data/pami_data)
  REPR272_ROOT     root of the 272-dim motion representation (index csv + per-sequence .npy) (default: <repo>/data/repr272)
  INDEX_CSV        seq_name -> 272 file index                              (default: <REPR272_ROOT>/index_interact_30fps.csv)
  BODY_MODELS      SMPL body-model dir used by nets/anchor_fk.py            (default: <REPR272_ROOT>/body_models)
  PAMI_CKPT_DIR    directory holding pamivae*.pt / pamigen*.pt             (default: <repo>/checkpoints)
  PAMI_T2HOI_DATA  PamiGen data dir: texts.json, text_feats.npz, segments_cap300.json, latents_cap300_<tag>.npz
                   (default: <PAMI_DATA_ROOT>/gen_corpus, where prepare_data.py and encode_latents.py write them)
  PAMI_STATS_DIR   where the normalisation statistics stats_*.npz are read from. Default: <PAMI_CKPT_DIR>/stats when
                   it exists (the statistics of the released models, downloaded with the checkpoints), else
                   <PAMI_DATA_ROOT> (your own, from `prepare_data.py stats`)

DATA layout under <PAMI_DATA_ROOT> (built by pami/data_prep, see docs/DATA.md):
  manifest_w64_s32.npz                  training windows (64 frames, stride 32) + per-window active objects
  split_official_v3_noval.json          train / val(logging only) / test sequence lists
  stats_human_interact_rootabs.npz      human-272 mean/std, window-absolute root dialect (human_root=wf_abs)
  stats_obj_interact_wf.npz             object robj/pobj/delta mean/std, window-first frame (obj_frame=window_first)
  stats_hand.npz                        hand block (wlocalrot) mean/std
  output/<seq>/{tae_sample,bps_sample,canon_anchors,hand_joints}.npz   per-sequence representation
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
_DATA = os.path.join(REPO, "data")

# ---- paths ----
PKG       = os.environ.get("PAMI_DATA_ROOT", os.path.join(_DATA, "pami_data"))    # prepared InterAct data
DATA_ROOT = PKG
REPR272   = os.environ.get("REPR272_ROOT", os.path.join(_DATA, "repr272"))
INDEX_CSV = os.environ.get("INDEX_CSV", f"{REPR272}/index_interact_30fps.csv")   # seq_name -> 272 .npy file
R272_DIR  = f"{REPR272}/output_interact_30fps/Representation_272"
SMPL_BODY_DIR = os.environ.get("BODY_MODELS", f"{REPR272}/body_models")          # SMPL pkl dir (anchor_fk)
CKPT_DIR  = os.environ.get("PAMI_CKPT_DIR", os.path.join(REPO, "checkpoints"))
T2HOI_DATA = os.environ.get("PAMI_T2HOI_DATA", os.path.join(PKG, "gen_corpus"))
_CK_STATS = os.path.join(CKPT_DIR, "stats")                       # statistics of the released models
STATS_DIR = os.environ.get("PAMI_STATS_DIR", _CK_STATS if os.path.isdir(_CK_STATS) else PKG)
STATS_HUMAN = f"{STATS_DIR}/stats_human_interact_rootabs.npz"     # human_root = wf_abs
STATS_OBJ   = f"{STATS_DIR}/stats_obj_interact_wf.npz"            # obj_frame  = window_first
STATS_HAND  = f"{STATS_DIR}/stats_hand.npz"                       # hand_mode  = wlocalrot
SPLIT    = f"{PKG}/split_official_v3_noval.json"                  # written by prepare_data.py manifest_split
MANIFEST = f"{PKG}/manifest_w64_s32.npz"
# released checkpoints
CKPT_VAE     = os.path.join(CKPT_DIR, "pamivae.pt")               # PamiVAE: encoder + latent space (PamiGen was trained on it)
CKPT_VAE_DEC = os.path.join(CKPT_DIR, "pamivae_contact.pt")       # PamiVAE with the contact-tuned decoder (same encoder weights)
CKPT_GEN     = os.path.join(CKPT_DIR, "pamigen.pt")
CKPT_REFINER = os.path.join(CKPT_DIR, "pamirefiner.pt")
VAE_TAG = "pamivae_refined"                                       # latents_cap300_<VAE_TAG>.npz of the released PamiVAE


def latents_file(data_dir, cap, vae_tag):
    """<data_dir>/latents_cap<cap>_<tag>.npz"""
    return f"{data_dir}/latents_cap{cap}_{vae_tag}.npz"


# ---- representation ----
WINDOW   = 64                                         # VAE training window (frames); latent step = 4 frames
STRIDE   = WINDOW // 2
MAX_OBJS = 8                                          # object slots per window (SceneST grid is 8-slot)
BPS_PTS  = 1024                                       # BPS basis size (object shape condition of PamiGen)
N_PARTS  = 6                                          # root, torso, Larm, Rarm, Lleg, Rleg
DIM_272  = 272
ROBJ_D   = 6                                          # object rotation 6D
POBJ_D   = 3                                          # object position p_G (free head)
W_D      = N_PARTS + 1                                # routing over {6 parts, FREE}
HAND_DIM = 270                                        # wlocalrot: [Lpos 45 | Rpos 45 | Lrot6d 90 | Rrot6d 90]
CLIP_RANGE = [-30, 20]                                # logvar clamp of the causal encoders

# ---- PamiVAE (SplitAE) ----
SAE_LATENT      = 16                                  # per-token latent (human part / object slot)
SAE_H_ENC_WIDTH = 1024                                # human part-encoder width
SAE_H_DEC_WIDTH = 1024                                # human decoder width
SAE_O_ENC_WIDTH = 512                                 # object encoder width (shared over slots)
SAE_O_DEC_WIDTH = 256                                 # object decoder width
SAE_MOTIONLIFT  = 256                                 # object per-frame input MLP width
SAE_DMODEL      = 256                                 # SceneST working width
SAE_LAYERS      = 6                                   # SceneST blocks
SAE_SP_HEADS    = 8                                   # spatial attention heads
SAE_TM_HEADS    = 14                                  # temporal heads for a 6+8 grid (hands: one head per token, 16)
ANCHOR_PARTS    = [0, 2, 3, 4, 5]                     # "hands_root_legs": root, Larm, Rarm, Lleg, Rleg carry a delta

# ---- loss weights (defaults of train_vae.py) ----
LAMBDA_ROOT     = 1.0
SAE_LAMBDA_ROBJ = 1.0
SAE_LAMBDA_PG   = 1.0
SAE_LAMBDA_VEL  = 1.0
SAE_LAMBDA_KL_H = 1e-4
SAE_LAMBDA_KL_O = 1e-4
LAMBDA_W        = 0.01                                # soft routing guide

# ---- train ----
DEVICE = "cuda"
BATCH  = 32
LR     = 2e-4
