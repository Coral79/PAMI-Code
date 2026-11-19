#!/usr/bin/env python
"""Train PamiGen (pami/train_t2hoi.py) on the PamiVAE latents archive.

Paper run (the released pamigen.pt is iteration 240k of a 400k run): --cap 300 --dmodel 512 --layers 10 --heads 8
--ff 2048 --lambda_obj 1.0 --lambda_pg_pos 5.0 --lambda_pg_vel 5.0 --lambda_comb 5.0 --lambda_agree 2.5
--human_shape --human_shape_src jrest:
    python train_gen.py --iters 400000 --save_every 20000 --seed 0 --run_name main
    -> outputs/gen/pamigen_main_<iteration>.pt
All of those values are the defaults of train_gen.py; --vae_tag / --data_dir / --split follow pami/config.py
(PAMI_T2HOI_DATA, PAMI_DATA_ROOT). The frozen VAE for the decoded object losses is resolved from the
latents archive's recorded checkpoint through PAMI_CKPT_DIR (override with --vae).
"""
import sys
from pami.train_t2hoi import main

if __name__ == "__main__":
    main(sys.argv[1:])
