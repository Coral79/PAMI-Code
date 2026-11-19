#!/usr/bin/env python
"""Build a PamiRefiner training bank from PamiGen samples of the TRAIN split (sharded protocol).

    python build_bank.py --init     --out <bank>/items --gen_glob '<samples>/gen_*.npz' --stream gen --varlen
    python build_bank.py --append   --out <bank>/items --gen_glob '<samples>/gen_*.npz' --stream gen --varlen --shard i/N   (i = 0..N-1, concurrently)
    python build_bank.py --finalize --out <bank>/items

Run once with --stream gen (the real stream) and once with --stream roundtrip (the gt_* twins, the
corruption stream) into two separate directories. See pami/refine/build_real_bank.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pami.refine.build_real_bank import main

if __name__ == "__main__":
    main()
