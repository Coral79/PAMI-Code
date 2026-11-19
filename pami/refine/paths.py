"""Filesystem locations the refiner needs, resolved ONCE here (nothing else in `pami.refine`
hard-codes a path).

Source of truth is `pami.data_prep.paths`, which is environment-overridable:
  PAMI_DATA_ROOT   the prepared data: output/<seq>/{tae_sample,hand_joints,shape_aligned,object_6dof}.npz
  REPR272_ROOT     the 272-dim representation workspace (index_interact_30fps.csv, output_interact_30fps/)
plus, for the refiner only,
  PAMI_CKPT_DIR    checkpoint directory (default <repo>/checkpoints).
"""
import os

from pami import config as C
from pami.data_prep import paths as _dp

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INTERACT_PKG = _dp.PAMI_DATA_ROOT
REPR272 = _dp.REPR272_ROOT
INTERACT_INDEX_CSV = _dp.INDEX_CSV
INTERACT_R272_DIR = os.path.join(_dp.R272_OUT, "Representation_272")
CKPT_DIR = os.environ.get("PAMI_CKPT_DIR", os.path.join(_REPO, "checkpoints"))


def use_interact_config():
    """Point `pami.config` at InterAct (the refiner is InterAct-only) and return INTERACT_PKG.
    `pami.data.hoi_dataset._seq_to_272` reads C.INDEX_CSV / C.R272_DIR, so all three must move."""
    C.PKG, C.INDEX_CSV, C.R272_DIR = INTERACT_PKG, INTERACT_INDEX_CSV, INTERACT_R272_DIR
    return INTERACT_PKG


_IC = [None]


def interact_common():
    """The InterAct helper namespace (`pami.data_prep.subsets`): body_model / load_seq / seq_subset_map / SUBSETS."""
    if _IC[0] is None:
        from pami.data_prep import subsets as ic
        _IC[0] = ic
    return _IC[0]
