# Data preparation

PAMI is trained and evaluated on [InterAct](https://github.com/wzyabcas/InterAct). `prepare_data.py` converts
InterAct into the files the training, sampling and evaluation code reads. Nothing derived from the dataset is
shipped with this repository.

## 1. What to download

| What | Where |
|---|---|
| InterAct (clone + data) | https://github.com/wzyabcas/InterAct → `data/InterAct` |
| LIGHT's processed InterAct data | https://github.com/wzyabcas/LIGHT#installation → `data/LIGHT` |
| SMPL, SMPL-H, SMPL-X | https://smpl.is.tue.mpg.de/, https://mano.is.tue.mpg.de/, https://smpl-x.is.tue.mpg.de/ → `data/repr272/body_models` |
| SMPL / SMPL-X vertex segmentation | https://github.com/Meshcapade/wiki (`assets/SMPL_body_segmentation`) → `data/repr272/body_models` |

`behave_correct`, `grab_smplh` and `intercap_correct_poses` come from InterAct's correction / conversion code; follow
the data instructions of the InterAct repository. Every subset folder must exist, also when only a few sequences are
converted with `--seqs`.

## 2. Folders

Every location is an environment variable; `python prepare_data.py paths` prints the resolved values.

| variable | default | content |
|---|---|---|
| `INTERACT_ROOT` | `data/InterAct/data` | InterAct's `data/` folder |
| `INTERACT_RAW_ROOT` | `$INTERACT_ROOT` | chairs / neuraldome / imhd, if kept elsewhere |
| `LIGHT_ROOT` | `data/LIGHT` | LIGHT clone |
| `REPR272_ROOT` | `data/repr272` | body models and the 272-dim human files |
| `BODY_MODELS` | `$REPR272_ROOT/body_models` | body models and vertex segmentation |
| `PAMI_DATA_ROOT` | `data/pami_data` | everything `prepare_data.py` writes |
| `PAMI_T2HOI_DATA` | `$PAMI_DATA_ROOT/gen_corpus` | captions, pieces, CLIP features, latents |
| `PAMI_STATS_DIR` | `checkpoints/stats` if present, else `$PAMI_DATA_ROOT` | normalisation statistics |

## 3. Stages

| stage | command | writes |
|---|---|---|
| 1 | `prepare_data.py convert` | `output/<seq>/`: object 6-DoF, body-part anchors, part-anchored object channels, object shape features |
| 2 | `prepare_data.py repr272` | 272-dim human motion (`$REPR272_ROOT/output_interact_30fps`) |
| 3 | `prepare_data.py hand_joints` | finger joints |
| 4 | `prepare_data.py shape_aligned` | object surface normals |
| 5 | `prepare_data.py manifest_split` | training windows, train / test split |
| 6 | `prepare_data.py stats` | normalisation statistics |
| 7 | `prepare_data.py text` | captions and pieces (`gen_corpus/`) |
| 8 | `prepare_data.py text_feats --out $PAMI_DATA_ROOT/gen_corpus` | CLIP caption features |
| 9 | `prepare_data.py test_pool --out $PAMI_DATA_ROOT/test_pool` | test list and its piece indices |

`prepare_data.py all` runs 1–7 and 9. Stages 1–4 are resumable and take `--seqs a,b`; stages 1, 3 and 4 also
`--shard i/n`. A GPU is optional. `python encode_latents.py` then writes the PamiVAE latents into `gen_corpus/`.
