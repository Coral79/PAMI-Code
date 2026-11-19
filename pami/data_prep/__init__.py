"""PAMI data preparation: InterAct raw parameters -> the training data the models read.

Ordered stages (each is a module with a `run(...)` entry point; `prepare_data.py --stage <name>` at
the repo root wraps them):

  1  convert        per-sequence package files: object_6dof / anchor_fk / canon_anchors / tae_sample
                    (object channels, human_272 injected in stage 2) / bps_sample + smpl_85 for the
                    272 chain + the sequence index csv                       [GPU optional, ~1 s/seq]
  2  repr272        human 272-dim representation: face-z alignment -> gendered body-model joints ->
                    representation_272 -> injected into tae_sample.npz         [GPU optional]
  3  hand_joints    hand_joints.npz: 30 finger joints (canonical frame) + parent-relative rot6d
  4  shape_aligned  shape_aligned.npz: face normals for the 2048 object surface samples (refiner)
  5  manifest_split manifest_w64_s32 windows + official splits (v1 -> v2 -> v3_noval)
  6  stats          normalization stats (human 272 / object / window-first object / root-abs /
                    hand) over the v1 official split's train sequences
  7  text           texts.json + segments_cap*.json + coverage.json (LIGHT-aligned pieces) and the
                    "v2nt" corpus filter (drop truncated 301-400-frame parents from TRAIN)
  8  text_feats     text_feats.npz: CLIP ViT-B/32 token + pooled features (needs the `clip` package)
  9  test_pool      the official evaluation pool + its piece positions

Paths: pami.data_prep.paths (env vars INTERACT_ROOT, INTERACT_RAW_ROOT, PAMI_DATA_ROOT, REPR272_ROOT,
LIGHT_ROOT, ...). Every stage function also accepts explicit path arguments.
See docs/DATA.md for the full description.
"""
