#!/usr/bin/env python
"""Refine PamiGen samples with the PamiRefiner and export joints52 / object pose (one npz per sequence).

    python refine.py --samples_glob 'outputs/samples/gen_*.npz' --ckpt checkpoints/pamirefiner.pt \
        --out_dir outputs/refined --seqs data/pami_data/test_pool/interact_test.json [--T 4]

Writes <out_dir>/refined_<seq>.npz with joints52 (W,52,3), R_obj (W,3,3), p_obj (W,3), rot6/rot6h/pel,
contact_prob pass-through, provenance. See pami/refine/export.py.

The output is placed on the InterAct / LIGHT floor (lowest human or object mesh vertex of the first 30 frames at
height 0; needs LIGHT_ROOT for the object meshes). `--floor native` keeps the floor of the 272-d representation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pami.refine.export import main

if __name__ == "__main__":
    main(default_T=4)
