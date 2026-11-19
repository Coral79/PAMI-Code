"""PamiVAE training (main run and the contact-tuned decoder finetune). CLI wrapper: train_vae.py.

Main run (paper, 200k iters, batch 32, AdamW lr 2e-4 wd 0.01, grad clip 1.0, seed 0):
    train_vae.py --split <split> --iters 200000 --save_every 25000 --seed 0 --run_name main
Decoder finetune (+25k iters on the 200k weights, encoders + SceneST frozen, fresh optimiser):
    train_vae.py --finetune_decoder_contact --resume_full <pamivae_main_resume.pt> --iters 225000 ...
Checkpoints: <out>/pamivae_<run>_<it>.pt (model + arch) every --save_every, <out>/pamivae_<run>_resume.pt
(+ optimizer). Validation (split "val" = logging-only subset) every --eval_every iterations.
"""
import sys, os, json, argparse, time, hashlib, torch
from torch.utils.data import DataLoader
from pami.data.hoi_dataset import HOIWindowDataset
from pami.nets.split_ae import SplitAE
from pami.losses import split_ae_loss, DEFAULT_LAM
from pami.metrics import split_evaluate
from pami.norm import build as build_norm
from pami.nets.anchor_fk import AnchorFK
from pami import config as C


def to_dev(b, dev): return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


def _load_split(path):
    if not os.path.exists(path):
        raise SystemExit(f"--split {path!r}: no such file")
    s = json.load(open(path))
    if not s.get("train"):
        raise SystemExit(f"--split {path!r}: no non-empty 'train' list")
    return s["train"], s.get("val")


def _sha256_file(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


def build_parser():
    ap = argparse.ArgumentParser(description="PamiVAE training")
    ap.add_argument("--iters", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=C.BATCH)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--split", type=str, default=C.SPLIT)
    ap.add_argument("--max_windows", type=int, default=None, help="debug: cap the number of training windows")
    ap.add_argument("--eval_every", type=int, default=4000)
    ap.add_argument("--eval_batches", type=int, default=None)
    ap.add_argument("--save_every", type=int, default=25000)
    ap.add_argument("--out", type=str, default=os.path.join("outputs", "vae"))
    ap.add_argument("--run_name", type=str, default="pamivae")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume_full", type=str, default=None, help="continue from a *_resume.pt (arch-guarded)")
    ap.add_argument("--freeze_encoder", action="store_true",
                    help="decoder-only finetune: freeze human/object encoders + SceneST (latents stay identical); "
                         "the optimiser is rebuilt over the trainable parameters")
    ap.add_argument("--contact_head", action="store_true", help="add the 4-logit foot-contact head (train with --lambda_ctc)")
    ap.add_argument("--fslide_gate", default="light", choices=["light", "pred"],
                    help="foot-skate gate: light = LIGHT's GT foot-contact rule; pred = contact head OR GT stance")
    for k, v in DEFAULT_LAM.items():
        ap.add_argument(f"--lambda_{'ow' if k == 'w' else k}", type=float, default=v)
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.seed is not None:
        import random, numpy as np
        random.seed(a.seed); np.random.seed(a.seed)
        torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
        print(f"seeded everything with --seed {a.seed}")
    if a.save_every <= 0: ap.error("--save_every must be > 0")
    if a.lambda_ctc > 0 and not a.contact_head: ap.error("--lambda_ctc needs --contact_head")
    if a.fslide_gate == "pred" and not a.contact_head: ap.error("--fslide_gate pred needs --contact_head")
    lam = {k: getattr(a, f"lambda_{'ow' if k == 'w' else k}") for k in DEFAULT_LAM}
    dev = C.DEVICE if torch.cuda.is_available() else "cpu"
    norm = build_norm()
    train_seqs, val_seqs = _load_split(a.split)
    ds = HOIWindowDataset(manifest=C.MANIFEST, window=C.WINDOW, seqs=train_seqs, norm=norm)
    if a.max_windows: ds.idx = ds.idx[:a.max_windows]
    print(f"dataset: {len(ds)} windows | run={a.run_name}")
    ARCH = {"arch_type": "split_ae", "sae_st": True, "sae_shape": "raw", "st_pos": "table",
            "sae_dmodel": C.SAE_DMODEL, "sae_layers": C.SAE_LAYERS, "sae_latent": C.SAE_LATENT,
            "sae_sp_heads": C.SAE_SP_HEADS, "sae_tm_heads": C.SAE_TM_HEADS,
            "sae_h_enc_width": C.SAE_H_ENC_WIDTH, "sae_h_dec_width": C.SAE_H_DEC_WIDTH,
            "human_stats": "interact", "dataset": "interact", "obj_frame": "window_first", "human_root": "wf_abs",
            "obj_delta": True, "delta_input": True, "predict_w": True, "sae_combine": True,
            "sae_anchor_set": "hands_root_legs", "delta_parts": list(C.ANCHOR_PARTS), "anchor_source": "recon",
            "hand_mode": "wlocalrot", "lambda_hand": lam["hand"], "lambda_root_vel": lam["root_vel"],
            "split": a.split, "split_sha": _sha256_file(a.split), "seed": a.seed}
    if lam["fslide"] > 0: ARCH.update({"lambda_fslide": lam["fslide"], "fslide_gate": a.fslide_gate, "fslide_src": "pos"})
    if lam["wvel"] > 0: ARCH["lambda_wvel"] = lam["wvel"]
    if lam["fkcons"] > 0: ARCH["lambda_fkcons"] = lam["fkcons"]
    if a.contact_head: ARCH.update({"contact_head": True, "lambda_ctc": lam["ctc"]})
    m = SplitAE(delta_parts=C.ANCHOR_PARTS, contact_head=a.contact_head).to(dev)
    anchor_fk = AnchorFK(dev)
    npar = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"SplitAE params: {npar:.1f}M | lambdas {lam}")
    if a.freeze_encoder:
        for mod in (m.h_encoders, m.o_encoder, m.motion_lift, m.refiner):
            for p_ in mod.parameters(): p_.requires_grad_(False)
        ntr = sum(p_.numel() for p_ in m.parameters() if p_.requires_grad) / 1e6
        print(f"[freeze_encoder] frozen human_enc/obj_enc/motion_lift/scene_st; trainable {ntr:.1f}M", flush=True)
        ARCH["freeze_encoder"] = True
    opt = torch.optim.AdamW([p_ for p_ in m.parameters() if p_.requires_grad], lr=a.lr, weight_decay=a.weight_decay)
    start_it = 0
    if a.resume_full:
        ck = torch.load(a.resume_full, map_location=dev, weights_only=False)
        rk = ck.get("arch", {})
        for key in ("arch_type", "sae_st", "sae_shape", "sae_dmodel", "sae_layers", "sae_latent", "sae_h_enc_width",
                    "sae_h_dec_width", "st_pos", "obj_frame", "human_root", "delta_parts"):
            if rk.get(key, ARCH[key]) != ARCH[key]:
                raise SystemExit(f"--resume_full arch mismatch: ckpt {key}={rk.get(key)} vs current {ARCH[key]}")
        if a.contact_head and not rk.get("contact_head", False):     # head added on top of a head-less checkpoint
            mk, uk = m.load_state_dict(ck["model"], strict=False)
            assert not uk and all(k.startswith("contact_head.") for k in mk), (mk, uk)
            print(f"[resume] non-strict: {len(mk)} new contact_head tensors init fresh", flush=True)
        else:
            m.load_state_dict(ck["model"])
        if "opt" in ck and not a.freeze_encoder: opt.load_state_dict(ck["opt"])
        start_it = int(ck.get("it", 0))
        print(f"resume_full from {a.resume_full}: it={start_it} (opt {'RESTORED' if ('opt' in ck and not a.freeze_encoder) else 'FRESH'})")
    os.makedirs(a.out, exist_ok=True)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True)
    val_dl = None
    if val_seqs:
        val_ds = HOIWindowDataset(manifest=C.MANIFEST, window=C.WINDOW, seqs=val_seqs, norm=norm)
        val_dl = DataLoader(val_ds, batch_size=a.batch, shuffle=False, num_workers=4)
        print(f"val dataset: {len(val_ds)} windows")
    logf = open(f"{a.out}/log_{a.run_name}.jsonl", "a"); print(f"jsonl log -> {a.out}/log_{a.run_name}.jsonl")
    def jlog(d): logf.write(json.dumps(d) + "\n"); logf.flush()
    it = start_it; t0 = time.time(); n_skip = 0; m.train()
    done = False
    for ep in range(10 ** 9):
        for b in dl:
            b = to_dev(b, dev); out = m(b)
            loss, comp = split_ae_loss(out, b, lam, norm, anchor_fk=anchor_fk, fslide_gate=a.fslide_gate)
            opt.zero_grad(); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip) if a.clip > 0 else None
            if gn is not None and not torch.isfinite(gn):               # skip the step on a non-finite gradient
                n_skip += 1
                if n_skip <= 5 or n_skip % 100 == 0:
                    print(f"  [it{it}] non-finite grad norm ({gn}) -- step SKIPPED (total {n_skip})")
                opt.zero_grad()
            else:
                opt.step()
            it += 1
            if it % 200 == 0:
                rate = (it - start_it) / (time.time() - t0); eta = (a.iters - it) / max(rate, 1e-6) / 60
                keys = ["total", "h_recon", "h_root", "kl_h", "h_hand", "o_robj", "o_pg", "o_vel", "kl_o", "h_root_vel",
                        "o_trans", "o_delta", "o_w"] + [k for k, lk in (("h_fslide", "fslide"), ("h_wvel", "wvel"),
                                                                   ("h_fkcons", "fkcons"), ("h_ctc", "ctc")) if lam[lk] > 0]
                print(f"it{it} ep{ep} | " + " ".join(f"{k} {comp[k]:.4f}" for k in keys) + f" | {rate:.1f} it/s | ETA {eta:.0f}min")
                jlog({"it": it, "kind": "train", **comp})
            if val_dl is not None and it % a.eval_every == 0:
                ev = split_evaluate(m, val_dl, dev, norm, max_batches=a.eval_batches, anchor_fk=anchor_fk)
                print(f"  [val it{it}] MPJPE {ev['mpjpe_mm']:.1f}mm | p_G {ev['pg_cm']:.2f}cm | rot {ev['rot_deg']:.1f}deg "
                      f"| combine {ev['combine_cm']:.2f}cm | contact {ev['contact_cm']:.2f}cm | hand {ev['hand_mm']:.1f}mm")
                jlog({"it": it, "kind": "val", **ev})
            if it % a.save_every == 0:
                torch.save({"model": m.state_dict(), "it": it, "arch": ARCH}, f"{a.out}/pamivae_{a.run_name}_{it}.pt")
                torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "it": it, "arch": ARCH},
                           f"{a.out}/pamivae_{a.run_name}_resume.pt.tmp")
                os.replace(f"{a.out}/pamivae_{a.run_name}_resume.pt.tmp", f"{a.out}/pamivae_{a.run_name}_resume.pt")
            if it >= a.iters:
                done = True; break
        if done: break
    dt = time.time() - t0
    print(f"[done] {it} iters in {dt/3600:.2f}h")
    jlog({"kind": "summary", "iters": it, "seconds": dt})


if __name__ == "__main__":
    main()
