#!/usr/bin/env python
"""Evaluate exported PAMI samples with LIGHT's exact InterAct protocol.

    export LIGHT_ROOT=/path/to/LIGHT            # clone + evaluator.ckpt + InterAct data (docs/EVAL.md)
    export OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python evaluate.py --pred_dir outputs/refined_test --out outputs/eval/refined_test.json

Input: a directory of ``refined_<sequence>.npz`` files (one per test sequence; keys joints52 (T,52,3),
R_obj (T,3,3), p_obj (T,3) in LIGHT's canonical frame) as written by ``refine.py``.
Output: JSON with a ``gt`` row and one row per prediction directory (keys r1/r2/r3/matching/fid/
diversity/foot_skate/pen/c_*/ch_*; the paper's Table 1 columns are r1, fid, matching, foot_skate,
pen*100, c_pct, ch_prec, ch_rec, ch_f1). Needs the evaluation packages of environment/README.md.
"""
import os, sys, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pami.eval.light_env as light_env          # noqa: E402  -- BLAS pins BEFORE numpy/torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True, help="directory of refined_<seq>.npz exports")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--light_root", default=None, help="LIGHT checkout (default: $LIGHT_ROOT)")
    ap.add_argument("--name", default=None, help="row key in the output JSON (default: basename of --pred_dir)")
    ap.add_argument("--test_list", default=None, help="test-list JSON (default: $PAMI_DATA_ROOT/test_pool/interact_test.json, written by prepare_data.py test_pool)")
    ap.add_argument("--rprec", default="light64", choices=["light64", "light"],
                    help="R-precision protocol. light64 (paper): shuffled, 5 passes, batch 64, remainder dropped. "
                         "light: LIGHT's public-code protocol (length-sorted, batch 256, repeat-filled last batch)")
    ap.add_argument("--rprec_bs", type=int, default=None, help="override the batch size of the chosen protocol")
    ap.add_argument("--rprec_seed", type=int, default=0)
    ap.add_argument("--rprec_passes", type=int, default=5)
    ap.add_argument("--skip_pen", action="store_true", help="skip the penetration metric (allows CPU-only runs)")
    ap.add_argument("--allow_pen_skip", action="store_true", help="tolerate rows whose SDF lookup fails (default: error)")
    ap.add_argument("--text_emb_cache", default=None,
                    help="text-embedding cache .pt (built if missing; default outputs/eval/text_emb_<test list sha>.pt)")
    ap.add_argument("--evaluator_ckpt", default="assets/evaluator.ckpt", help="relative to LIGHT_ROOT or absolute")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: stop after N kept rows (numbers not comparable)")
    ap.add_argument("--per_seq", action="store_true", help="also store per-sequence geometric metrics")
    ap.add_argument("--device", default=None, help="cuda (default if available) or cpu (needs --skip_pen)")
    a = ap.parse_args()

    from pami.eval import evaluator as E
    root = light_env.resolve_light_root(a.light_root)
    test_list = os.path.abspath(a.test_list or E.DEFAULT_TEST_LIST)
    cache = a.text_emb_cache
    if cache is None:
        cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "eval",
                             f"text_emb_{E.sha1_file(test_list)}.pt")
    E.evaluate(a.pred_dir, a.out, root, test_list=test_list, name=a.name, rprec=a.rprec, rprec_bs=a.rprec_bs,
               rprec_seed=a.rprec_seed, rprec_passes=a.rprec_passes, skip_pen=a.skip_pen,
               allow_pen_skip=a.allow_pen_skip, text_emb_cache=cache, evaluator_ckpt=a.evaluator_ckpt,
               limit=a.limit, per_seq=a.per_seq, device=a.device)


if __name__ == "__main__":
    main()
