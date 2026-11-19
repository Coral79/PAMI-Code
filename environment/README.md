# Environment

One conda environment (`pami`, Python 3.10, PyTorch 2.7.1) for data preparation, training, sampling, refinement and
evaluation.

## 1. Install

```bash
conda create -n pami python=3.10 -y
conda activate pami
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.22.0 scipy==1.11.4 six && pip install --no-build-isolation chumpy==0.70
pip install -r environment/requirements.txt
# caption features of the data preparation, free-text prompts:
pip install ftfy regex && pip install --no-deps git+https://github.com/openai/CLIP
```

* **RTX 50-series GPUs.** The `cu118` build has no kernels for these cards; use
  `--index-url https://download.pytorch.org/whl/cu128` there (same torch / torchvision versions).
* **`chumpy`** is installed on its own with `--no-build-isolation`: its `setup.py` imports `pip`, which fails inside
  pip's isolated build environment. It is needed only to unpickle `body_models/smpl/SMPL_NEUTRAL.pkl`.
* **CLIP must be installed with `--no-deps`.** Its own requirements are unpinned, so a plain
  `pip install git+https://github.com/openai/CLIP` replaces torch 2.7.1 with the latest release.
* **`human_body_prior` is pinned** in `requirements.txt` to commit `4c246d8` (v2.2.2.0, the version the paper's
  data were built with): the current master of that repository requires Python >= 3.11.

Body models (registration required, not redistributable): put them in one folder and point `BODY_MODELS` at it
(default `$REPR272_ROOT/body_models`):

```
body_models/
  smpl/SMPL_NEUTRAL.pkl                     # SMPL (https://smpl.is.tue.mpg.de/): kinematic tree of the anchor FK
  smplh/{male,female,neutral}/model.npz     # SMPL-H, AMASS "extended" with MANO hands (https://mano.is.tue.mpg.de/)
  smplx/SMPLX_{MALE,FEMALE,NEUTRAL}.npz     # SMPL-X (https://smpl-x.is.tue.mpg.de/)
  smplx/SMPLX_MALE.pkl                      # SMPL-X pickle with the MANO PCA components (data preparation of InterCap)
  smpl_vert_segmentation.json               # vertex -> body part, from https://github.com/Meshcapade/wiki
  smplx_vert_segmentation.json              #   (assets/SMPL_body_segmentation/{smpl,smplx}/)
```

## 2. Evaluation packages

`evaluate.py` imports LIGHT's evaluator, which needs `pytorch3d` and `chamfer_distance` compiled from source:

```bash
conda install -c nvidia/label/cuda-11.8.0 cuda-nvcc cuda-cudart-dev cuda-cccl libcublas-dev libcusparse-dev libcufft-dev libcurand-dev libcusolver-dev -y
pip install fvcore iopath ninja
FORCE_CUDA=1 CUDA_HOME=$CONDA_PREFIX pip install --no-build-isolation \
    "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47" \
    "chamfer_distance @ git+https://github.com/otaheri/chamfer_distance@f86f6f7cadd3aca642704573d1626c67ca2e2846"
pip install -r environment/requirements-eval.txt
```

* **Pin the CUDA toolchain to 11.8** (`-c nvidia/label/cuda-11.8.0`); the plain `nvidia` channel installs the newest
  `nvcc`, which does not match the cu118 torch build.
* **`--no-build-isolation`.** Both packages import torch in their `setup.py` and are built against the installed one.
* **Host compiler.** CUDA 11.8's `nvcc` accepts gcc 11 or older. On a system with a newer gcc install one into the
  environment before the pytorch3d step:
  ```bash
  conda install -c conda-forge gxx_linux-64=11 -y && conda deactivate && conda activate pami
  export NVCC_PREPEND_FLAGS="-ccbin $CC"
  ```
* These steps were run from scratch with the `cu118` build (Debian, gcc 14 with the gcc 11 step, NVIDIA A16);
  `evaluate.py` then gives the same numbers as the environment the paper's numbers were computed in. Building the
  two packages for the `cu128` build (RTX 50-series) was not tested; `--skip_pen --device cpu` evaluates everything
  except the penetration metric without a GPU.

Then download LIGHT's evaluator checkpoint (step 5 of https://github.com/wzyabcas/LIGHT#installation) into the LIGHT
folder of the data preparation and export `LIGHT_ROOT`. Details: `docs/EVAL.md`.

## 3. Mandatory runtime pins for evaluation

```bash
export OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Multi-threaded LAPACK corrupts `scipy.linalg.sqrtm` inside the FID (documented in LIGHT's
`utils/eval_t2m_utils.py`), and OpenBLAS picks a broken kernel on Sapphire-Rapids CPUs (silently wrong
eigen-decompositions; FID 5-6 instead of 0.1). `evaluate.py` sets these itself before numpy is imported, but an
explicit export is the safe habit. On a CPU without AVX2 the Haswell kernels crash with an illegal instruction; use
`OPENBLAS_CORETYPE=Sandybridge` there.
