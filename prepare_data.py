#!/usr/bin/env python
"""PAMI data preparation driver: InterAct raw parameters -> training data. See docs/DATA.md.

  python prepare_data.py paths                                   # print the resolved roots
  python prepare_data.py convert        [--seqs a,b] [--shard i/n] [--force] [--device cuda|cpu]
  python prepare_data.py repr272        [--seqs a,b] [--device ...]
  python prepare_data.py hand_joints    [--seqs a,b | --subset omomo] [--shard i/n] [--force]
  python prepare_data.py shape_aligned  [--seqs a,b] [--shard i/n] [--force]
  python prepare_data.py manifest_split
  python prepare_data.py stats          [--split <split_official.json>] [--which base,windows,hand]
  python prepare_data.py text           [--split <split_official_v3_noval.json>] [--out <dir>] [--fps 30]
  python prepare_data.py text_feats     --out <dir>              # needs the `clip` package
  python prepare_data.py test_pool      --out <dir> [--segments <segments_cap300.json>] [--latents <npz>]
  python prepare_data.py all            # 1..7 + 9 in order (text_feats needs its own env; run it separately)
Roots come from env vars (INTERACT_ROOT, INTERACT_RAW_ROOT, PAMI_DATA_ROOT, REPR272_ROOT, LIGHT_ROOT, ...),
see pami/data_prep/paths.py; `python prepare_data.py paths` prints the resolved values.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STAGES = ["paths", "convert", "repr272", "hand_joints", "shape_aligned", "manifest_split", "stats", "text",
          "text_feats", "test_pool", "all"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--seqs", default=None, help="comma-separated sequence subset")
    ap.add_argument("--subset", default=None, help="hand_joints: one InterAct subset")
    ap.add_argument("--shard", default=None, metavar="I/N")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--keep_verts", action="store_true", help="convert: also persist human_verts.npz (large)")
    ap.add_argument("--split", default=None)
    ap.add_argument("--which", default="base,windows,hand")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--segments", default=None)
    ap.add_argument("--latents", default=None)
    a = ap.parse_args()
    from pami.data_prep import paths as P
    seqs = a.seqs.split(",") if a.seqs else None
    st = a.stage
    if st == "paths":
        print(P.describe()); return
    if st in ("convert", "all"):
        from pami.data_prep import stage1_convert
        stage1_convert.run(seqs=seqs, shard=a.shard, force=a.force, keep_verts=a.keep_verts, device=a.device)
    if st in ("repr272", "all"):
        from pami.data_prep import stage2_repr272
        stage2_repr272.run(seqs=seqs, device=a.device)
    if st in ("hand_joints", "all"):
        from pami.data_prep import stage3_hand_joints
        stage3_hand_joints.run(seqs=seqs, subset=a.subset, shard=a.shard, force=a.force)
    if st in ("shape_aligned", "all"):
        from pami.data_prep import stage4_shape_aligned
        stage4_shape_aligned.run(seqs=seqs, shard=a.shard, force=a.force)
    if st in ("manifest_split", "all"):
        from pami.data_prep import stage5_manifest_split
        stage5_manifest_split.run()
    if st in ("stats", "all"):
        from pami.data_prep import stage6_stats
        stage6_stats.run(split_json=a.split, which=tuple(a.which.split(",")))
    if st in ("text", "all"):
        from pami.data_prep import stage7_text
        stage7_text.run(split_json=a.split, out=a.out, fps=a.fps)
    if st == "text_feats":
        from pami.data_prep import stage8_text_feats
        stage8_text_feats.run(a.out or P.T2HOI_OUT, device=a.device)
    if st in ("test_pool", "all"):
        from pami.data_prep import stage9_test_pool
        stage9_test_pool.run(a.out or f"{P.PAMI_DATA_ROOT}/test_pool", split_json=a.split,
                             segments_json=a.segments, latents=a.latents)


if __name__ == "__main__":
    main()
