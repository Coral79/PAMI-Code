"""Checkpoint loading + the latent-space helpers shared by encode / train_gen / generate / eval.

Checkpoints record their architecture in ck["arch"]; every loader rebuilds the module from those keys and
loads the weights strictly, so a checkpoint can never be silently read under the wrong configuration.
VAE checkpoints are resolved through config.CKPT_DIR (`resolve_ckpt`): PamiGen checkpoints and latents
archives record the file name of the PamiVAE they were built on (pamivae.pt for the released models).
"""
import os
import torch
from . import config as C
from .nets.split_ae import SplitAE

_SPLIT_AE_REQUIRED = {"arch_type": "split_ae", "sae_st": True, "sae_shape": "raw", "obj_delta": True,
                      "delta_input": True, "predict_w": True, "hand_mode": "wlocalrot", "st_pos": "table",
                      "obj_frame": "window_first", "human_root": "wf_abs"}
_SPLIT_AE_UNSUPPORTED = ("rotvote", "explicit_root", "wholebody", "w_uniform", "rootcond")


def resolve_ckpt(path):
    """A checkpoint path or basename -> an existing file: absolute/existing paths pass through, otherwise the
    basename is looked up in config.CKPT_DIR."""
    if path and os.path.exists(path):
        return path
    base = os.path.basename(str(path))
    cand = os.path.join(C.CKPT_DIR, base)
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"checkpoint {path!r} not found (looked in {C.CKPT_DIR}; set PAMI_CKPT_DIR)")


def load_split_ae(ckpt_path, dev="cpu", verbose=True):
    """Load a PamiVAE (SplitAE) checkpoint honouring its saved arch. Returns (model.eval(), ck, arch)."""
    ckpt_path = resolve_ckpt(ckpt_path)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", {})
    for k, v in _SPLIT_AE_REQUIRED.items():
        got = arch.get(k, {"st_pos": "table", "obj_frame": "window_first", "human_root": "delta"}.get(k))
        if got != v:
            raise SystemExit(f"load_split_ae: {ckpt_path} arch {k}={got!r} is not supported by PAMI (needs {v!r})")
    for k in _SPLIT_AE_UNSUPPORTED:
        if arch.get(k):
            raise SystemExit(f"load_split_ae: {ckpt_path} uses the removed ablation arch flag {k}={arch[k]!r}")
    C.SAE_DMODEL = arch.get("sae_dmodel", C.SAE_DMODEL)
    C.SAE_LAYERS = arch.get("sae_layers", C.SAE_LAYERS)
    C.SAE_LATENT = arch.get("sae_latent", C.SAE_LATENT)
    C.SAE_SP_HEADS = arch.get("sae_sp_heads", C.SAE_SP_HEADS)
    C.SAE_TM_HEADS = arch.get("sae_tm_heads", C.SAE_TM_HEADS)
    C.SAE_H_ENC_WIDTH = arch.get("sae_h_enc_width", 1024)
    C.SAE_H_DEC_WIDTH = arch.get("sae_h_dec_width", 1024)
    m = SplitAE(delta_parts=arch.get("delta_parts"), contact_head=arch.get("contact_head", False))
    if arch.get("decoder_only"):                                       # release export (export_decoder_only)
        missing, unexpected = m.load_state_dict(ck["model"], strict=False)
        assert not unexpected and all(not k.startswith(DECODER_PREFIXES) for k in missing), (missing, unexpected)
        m.decoder_only = True
    else:
        m.load_state_dict(ck["model"])
    m.to(dev).eval()
    if verbose:
        print(f"[ckpt_io] {ckpt_path}: it={ck.get('it')} SplitAE dmodel={C.SAE_DMODEL} layers={C.SAE_LAYERS} "
              f"delta_parts={m.delta_parts} contact_head={m.contact_head is not None}")
    return m, ck, arch


def assert_repr_dialect(vae_arch, ctx):
    """The representation dialect every PAMI consumer is wired for: window-first objects, wf_abs human root."""
    of = vae_arch.get("obj_frame", "window_first"); hr = vae_arch.get("human_root", "delta")
    if (of, hr) != ("window_first", "wf_abs"):
        raise SystemExit(f"{ctx}: VAE arch dialect obj_frame={of!r} human_root={hr!r} — PAMI decodes "
                         f"('window_first', 'wf_abs') only.")
    return of, hr


# ---------------------------------------------------------------- diffusion-space latent normalisation
def _stats_ms(stats, ref):
    """Per-token stats (P,2) -> broadcastable (mean, std) over trailing (P, L) dims (scalar (2,) also accepted)."""
    t = torch.as_tensor(stats, dtype=ref.dtype, device=ref.device)
    if t.ndim == 1:
        return t[0], t[1]
    return t[:, 0].view(-1, 1), t[:, 1].view(-1, 1)


def h_norm(z, stats):
    m, s = _stats_ms(stats, z); return (z - m) / s


def h_denorm(z, stats):
    m, s = _stats_ms(stats, z); return z * s + m


# ---------------------------------------------------------------- PamiGen -> PamiVAE plumbing
def load_vae_for_gen(arch, vae_ckpt=None, dev="cpu", verbose=True):
    """Load the frozen PamiVAE a PamiGen checkpoint decodes through. vae_ckpt (path) overrides the VAE the
    generator ARCH records (arch['vae_ckpt']); the paper's samples decode through the contact-tuned decoder
    (config.CKPT_VAE_DEC) whose encoder/latents are identical to the training VAE. Returns (vae, Normalizer)."""
    from .norm import build as build_norm
    path = vae_ckpt or arch.get("vae_ckpt") or C.CKPT_VAE
    vae, _, va = load_split_ae(path, dev=dev, verbose=verbose)
    assert_repr_dialect(va, "load_vae_for_gen")
    return vae, build_norm()


def check_latents_dialect(arch_or_ds, ctx):
    """PamiGen latents must be the post-SceneST ("refined") dialect: the decoders consume exactly that space."""
    refined = arch_or_ds.get("vae_refined", arch_or_ds.get("refined")) if isinstance(arch_or_ds, dict) \
        else getattr(arch_or_ds, "refined", None)
    if not refined:
        raise SystemExit(f"{ctx}: latents are not the refined (post-SceneST) dialect — re-run encode_latents.py")


def decode_human(vae, z_latent, B, Tr):
    """(B,Tr,P,16) VAE latent (already de-normalised from diffusion space) -> dict(human_recon (B,W,272),
    hand_recon (B,W,270), contact_logit (B,W,4)|None) through the (contact-tuned) decoder."""
    return vae.decode_human_full(z_latent.reshape(B, Tr, vae.P * C.SAE_LATENT))


# ---------------------------------------------------------------- release exports
DECODER_PREFIXES = ("h_fusion.", "h_decoder.", "contact_head.", "o_dec_proj.", "o_decoder.", "w_mask")


def export_decoder_only(src, dst):
    """Write a decoder-only PamiVAE checkpoint (h_fusion, h_decoder, contact_head, o_dec_proj, o_decoder, w_mask):
    everything generate.py / the refiner need to DECODE stored or generated latents. The encoders + SceneST
    (~2.8 GB) are only needed by encode_latents.py / training. arch['decoder_only']=True marks the file;
    load_split_ae accepts it (strict load of the decoder subset, encoders left at their random init)."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ck["model"].items() if k.startswith(DECODER_PREFIXES)}
    arch = dict(ck["arch"]); arch["decoder_only"] = True
    torch.save({"model": sd, "it": ck.get("it"), "arch": arch, "exported_from": os.path.basename(src)}, dst)
    return dst


def export_gen_ema_only(src, dst):
    """Write an inference-only PamiGen checkpoint: EMA weights as 'model' (no optimizer, no raw weights)."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = {k[len("module."):]: v for k, v in ck["ema"].items() if k.startswith("module.")}
    torch.save({"model": sd, "it": ck.get("it"), "arch": ck["arch"], "ema_only": True,
                "exported_from": os.path.basename(src)}, dst)
    return dst


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="PAMI checkpoint exports")
    ap.add_argument("--export_decoder", nargs=2, metavar=("SRC", "DST"), help="decoder-only PamiVAE export")
    ap.add_argument("--export_gen_ema", nargs=2, metavar=("SRC", "DST"), help="EMA-only PamiGen export")
    a = ap.parse_args()
    if a.export_decoder: print("wrote", export_decoder_only(*a.export_decoder))
    if a.export_gen_ema: print("wrote", export_gen_ema_only(*a.export_gen_ema))
