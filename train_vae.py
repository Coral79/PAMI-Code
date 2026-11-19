#!/usr/bin/env python
"""Train PamiVAE (pami/train_split.py).

Main run (the released pamivae.pt is iteration 200k of this run):
    python train_vae.py --iters 200000 --save_every 25000 --seed 0 --run_name main [--lambda_fslide 3.0 --fslide_gate light]
    -> outputs/vae/pamivae_main_200000.pt (+ pamivae_main_resume.pt with the optimizer state)

Contact-tuned decoder (the released pamivae_contact.pt): 25k more iterations with the encoder frozen, a contact
head and the contact / foot-slide / wrist-velocity / FK-consistency losses
(= --freeze_encoder --contact_head --lambda_ctc 1 --lambda_fslide 30 --fslide_gate pred --lambda_wvel 30
 --lambda_fkcons 1 --iters 225000 --save_every 5000):
    python train_vae.py --finetune_decoder_contact --resume_full outputs/vae/pamivae_main_resume.pt \
        --seed 0 --run_name contact
Data locations come from pami/config.py (PAMI_DATA_ROOT ...).
"""
import sys
from pami.train_split import main

PRESET_DEC_CTC = ["--freeze_encoder", "--contact_head", "--lambda_fslide", "30", "--fslide_gate", "pred",
                  "--lambda_wvel", "30", "--lambda_fkcons", "1", "--lambda_ctc", "1.0",
                  "--iters", "225000", "--save_every", "5000"]

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--finetune_decoder_contact" in argv:
        argv = [x for x in argv if x != "--finetune_decoder_contact"]
        if "--resume_full" not in argv:
            sys.exit("--finetune_decoder_contact needs --resume_full <pamivae_<run>_resume.pt of the 200k main run>")
        argv = PRESET_DEC_CTC + argv                                   # explicit flags after the preset override it
    main(argv)
