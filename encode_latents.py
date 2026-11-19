#!/usr/bin/env python
"""Encode the PamiGen training latents with the frozen PamiVAE (pami/prepare_t2hoi.py).

    python encode_latents.py [--ckpt checkpoints/pamivae.pt] [--cap 300] [--data_dir <dir>]
    -> <data_dir>/latents_cap300_pamivae_refined.npz   (the archive generate.py and train_gen.py read)
The archive tag is derived from the checkpoint basename ("_refined" appended: the latents are the encoder means
after the PamiVAE's scene transformer); use --out_tag to write under another name and pass the same tag to
train_gen.py --vae_tag.
"""
import sys
from pami.prepare_t2hoi import main

if __name__ == "__main__":
    main(sys.argv[1:])
