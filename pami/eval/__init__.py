"""PAMI evaluation: LIGHT's exact InterAct text-to-HOI protocol on exported (refined) samples.

Import order matters: ``pami.eval.light_env`` must be imported before numpy/torch so the
OpenBLAS / thread pins take effect (see that module). ``evaluate.py`` at the repo root does this.
"""
