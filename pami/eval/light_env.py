"""Runtime environment for the LIGHT evaluator (import this BEFORE numpy / torch).

Two things happen here and both are required for reproducible numbers:

1. BLAS pins (set with ``setdefault`` so an explicit export in the shell wins):
   * ``OPENBLAS_CORETYPE=Haswell`` -- the OpenBLAS build in the evaluation env picks a broken
     kernel on Sapphire-Rapids CPUs and returns silently wrong linear algebra (FID jumps from
     ~0.1 to ~5).
   * ``OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`` -- multi-threaded LAPACK corrupts
     ``scipy.linalg.sqrtm`` inside the FID (documented in LIGHT ``utils/eval_t2m_utils.py``).
   They only work if they are set before numpy is imported, hence this module.

2. LIGHT_ROOT: the evaluator re-uses LIGHT's own code and data (nothing re-implemented):
   * ``tma`` (evaluator networks), ``utils.eval_t2m_utils`` (metric functions);
   * ``assets/evaluator.ckpt`` (the released TMR-style evaluator);
   * ``InterAct/<subset>/sequences_canonical/<seq>/data.npz`` (GT motion + 340 object contact
     points), ``InterAct/<subset>/objects/<obj>/<obj>_{can.npy,sdf.npz}`` (canonical rotation,
     SDF), ``InterAct/stats/bps_basis_set_128_1.npy``.
   ``setup_light_root()`` puts LIGHT_ROOT on ``sys.path`` and ``chdir``s into it, because LIGHT's
   SDF loader (``load_object_sdf_data``) reads ``./InterAct/...`` relative to the working
   directory. Resolve your own paths to absolute ones before calling it (the CLI does).
"""
import os
import sys

if "numpy" in sys.modules:   # pins below would be ineffective
    print("[pami.eval] WARNING: numpy was imported before pami.eval.light_env; "
          "BLAS thread/kernel pins may not apply", file=sys.stderr)
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_LIGHT_ROOT = None


def resolve_light_root(arg=None):
    """LIGHT_ROOT from the CLI argument, else the env var, else error."""
    root = arg or os.environ.get("LIGHT_ROOT")
    if not root:
        raise SystemExit("LIGHT_ROOT is required: pass --light_root or export LIGHT_ROOT "
                         "(a clone of https://github.com/wzyabcas/LIGHT with assets/evaluator.ckpt "
                         "and the InterAct data folder; see docs/EVAL.md)")
    root = os.path.abspath(os.path.expanduser(root))
    for rel in ("tma", "utils/eval_t2m_utils.py", "assets/evaluator.ckpt",
                "InterAct/stats/bps_basis_set_128_1.npy"):
        if not os.path.exists(os.path.join(root, rel)):
            raise SystemExit(f"LIGHT_ROOT={root}: missing {rel} (see docs/EVAL.md)")
    return root


def setup_light_root(root):
    """Make LIGHT importable and its cwd-relative data paths valid. Idempotent."""
    global _LIGHT_ROOT
    if _LIGHT_ROOT is not None:
        assert _LIGHT_ROOT == root, (_LIGHT_ROOT, root)
        return root
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)
    _LIGHT_ROOT = root
    return root


def light_root():
    assert _LIGHT_ROOT is not None, "call setup_light_root() first"
    return _LIGHT_ROOT
