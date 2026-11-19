#!/usr/bin/env python
"""Text-to-HOI generation with PamiGen + PamiVAE (pami/sample_t2hoi.py).

Test-split pieces (caption / object / body shape / length from the piece; GT twin stored alongside):
    python generate.py --split_part test --pieces 2 4 10 --seed 6 --cfg 2.5 --steps 50 --out DIR
  (defaults: --ckpt checkpoints/pamigen.pt, --vae checkpoints/pamivae_contact.pt = the contact-tuned decoder,
  or their inference-only exports pamigen_ema_only.pt / pamivae_contact_decoder_only.pt when only those were
  downloaded -- same samples; the ODE draws ONE seeded noise tensor for the whole --pieces batch, so the
  paper draws are reproduced only with the same piece list per call: rep3/4/5 = shards NR%3==0/1/2 of
  $PAMI_DATA_ROOT/test_pool/pieces_test.txt with seeds 6,7,8 / 9,10,11 / 12,13,14.)

Free text (--text): the caption is encoded with CLIP ViT-B/32 (needs the `clip` package; the weights are
downloaded to ~/.cache/clip/ViT-B-32.pt on first use — the same encoder that produced text_feats.npz); the
object, the piece length and the body shape are taken from a reference test piece (--ref_piece, default
piece 0), so the prompt should be about that piece's object (piece 10 is a baseball bat).
    python generate.py --text "Swing the baseball bat with both hands." --ref_piece 10 --out DIR
"""
import os, sys, argparse
import numpy as np
import torch
from pami import config as C
from pami.sample_t2hoi import Generator


class _TextBankWithPrompt:
    """Wraps the archive TextBank so a fresh CLIP-encoded prompt can be addressed as caption id N (new row)."""
    def __init__(self, bank, pooled, tok):
        self.base = bank
        self.pooled = torch.cat([bank.pooled, pooled[None]], 0)
        self.tok_len = np.concatenate([bank.tok_len, [len(tok)]])
        self.toks = bank.toks + [tok]
        self.new_id = len(bank.toks)

    def captions(self, tids, dev):
        from pami.nets.t2hoi_denoiser import CLIP_D
        tids = tids.cpu()
        pooled = self.pooled[tids].to(dev)
        L = max(int(self.tok_len[int(t)]) for t in tids)
        tok = torch.zeros(len(tids), L, CLIP_D); msk = torch.zeros(len(tids), L)
        for j, t in enumerate(tids):
            tk = self.toks[int(t)]
            tok[j, :len(tk)] = tk; msk[j, :len(tk)] = 1
        return pooled, tok.to(dev), msk.to(dev)


def encode_prompt(text, dev):
    """CLIP ViT-B/32 features of one caption exactly as pami/data_prep's text pass computes them:
    per-token ln_final features (SOT..EOT) and the pooled EOT @ text_projection."""
    try:
        import clip
    except ImportError:
        sys.exit("--text needs the OpenAI `clip` package: pip install ftfy regex && "
                 "pip install --no-deps git+https://github.com/openai/CLIP   (--no-deps keeps the installed torch)")
    model, _ = clip.load("ViT-B/32", device=dev, download_root=os.path.expanduser("~/.cache/clip"))
    model.eval()
    tk = clip.tokenize([text], truncate=True).to(dev)
    with torch.no_grad():
        x = model.token_embedding(tk).type(model.dtype) + model.positional_embedding.type(model.dtype)
        x = model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = model.ln_final(x).float()
        eot = int(tk.argmax(dim=-1)[0])
        tok = x[0, :eot + 1].cpu().half().float()                    # fp16 on disk in text_feats.npz -> same rounding
        pooled = (x[0, eot] @ model.text_projection.float()).cpu()
    return pooled, tok


def _default_ckpt(full, export):
    """The full checkpoint when it is there, else its inference-only export (identical samples)."""
    return full if os.path.isfile(full) or not os.path.isfile(export) else export


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=_default_ckpt(C.CKPT_GEN, C.CKPT_GEN.replace(".pt", "_ema_only.pt")))
    ap.add_argument("--vae", default=_default_ckpt(C.CKPT_VAE_DEC, C.CKPT_VAE_DEC.replace(".pt", "_decoder_only.pt")),
                    help="PamiVAE checkpoint used for decoding")
    ap.add_argument("--split_part", default="test", choices=["val", "train", "test"])
    ap.add_argument("--pieces", type=int, nargs="*", default=None, help="piece indices into the split (one batch)")
    ap.add_argument("--n", type=int, default=4, help="random pieces when --pieces is not given")
    ap.add_argument("--text", default=None, help="free-text prompt (uses --ref_piece for object/shape/length)")
    ap.add_argument("--ref_piece", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    G = Generator(a.ckpt, a.vae, use_ema=not a.no_ema, data_dir=a.data_dir)
    ds = G.dataset(a.split_part)
    if a.text:
        pooled, tok = encode_prompt(a.text, G.dev)
        G.bank = _TextBankWithPrompt(G.bank, pooled, tok)
        item = ds[a.ref_piece]; item["cap_tid"] = G.bank.new_id
        items, idx = [item], [a.ref_piece]
    else:
        idx = a.pieces if a.pieces else list(np.random.RandomState(a.seed).choice(len(ds), a.n, replace=False))
        items = [ds[int(i)] for i in idx]
    out_dir = a.out or os.path.join("outputs", f"{os.path.basename(a.ckpt).replace('.pt', '')}_cfg{a.cfg}")
    os.makedirs(out_dir, exist_ok=True)
    for i, payload in zip(idx, G.generate(items, steps=a.steps, cfg=a.cfg, seed=a.seed)):
        if a.text:
            payload["text"] = a.text
        fp = f"{out_dir}/gen_{int(i)}_{payload['seq']}.npz"
        np.savez(fp, ckpt=a.ckpt, vae=a.vae, **payload)
        print(f"[{int(i)}] {payload['seq']}[{payload['f0']}:{payload['f1']}] -> {fp}")
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
