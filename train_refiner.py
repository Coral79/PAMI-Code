#!/usr/bin/env python
"""Train the PamiRefiner (defaults = the shipped recipe: A2 full body, b2 finger sensor K=64 r=4cm
with fingertips, 16384-point sensor cloud, T=3 curriculum, batch 4, 60k iterations, periodic saves).

    python train_refiner.py --rt_bank <rt banks...> --gen_bank <gen banks...> \
        --val_rt_bank <val rt bank> --val_gen_bank <val gen bank> --out outputs/refiner/pamirefiner.pt

Periodic checkpoints land at <out>_it<N>.pt every --save_every iterations (the released pamirefiner.pt is
the it44000 file of a 60k run). See pami/refine/train_m3.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pami.refine.train_m3 import main

if __name__ == "__main__":
    main()
