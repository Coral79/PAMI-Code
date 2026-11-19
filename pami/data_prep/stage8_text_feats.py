"""Stage 8 — CLIP text features. Standalone: needs the OpenAI `clip` package (+ torch), see the optional
CLIP install in environment/README.md.

Reads  <data_dir>/texts.json  (texts[0] = "" null caption)
Writes <data_dir>/text_feats.npz:
  tok_<i>  (L_i,512) fp16  per-token features after ln_final, SOT..EOT inclusive (denoiser cross-attention)
  pooled   (N,512)   fp32  EOT feature @ text_projection (standard CLIP pooled; row 0 = null caption -> CFG)
  tok_len  (N,)      int32
ViT-B/32 weights from the local cache (~/.cache/clip/ViT-B-32.pt) or downloaded by `clip` on first use.

Usage: python -m pami.data_prep.stage8_text_feats --data_dir <dir>
"""
import argparse
import json
import os

import numpy as np
import torch


def encode_texts(texts, batch=256, device=None):
    import clip
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = clip.load("ViT-B/32", device=dev, download_root=os.path.expanduser("~/.cache/clip"))
    model.eval()
    toks = clip.tokenize(texts, truncate=True).to(dev)                 # (N,77)
    out, pooled, lens = {}, [], []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            tk = toks[s:s + batch]
            x = model.token_embedding(tk).type(model.dtype)
            x = x + model.positional_embedding.type(model.dtype)
            x = model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = model.ln_final(x).float()                              # (B,77,512)
            eot = tk.argmax(dim=-1)                                    # EOT = highest token id
            for j in range(tk.shape[0]):
                i = s + j
                L = int(eot[j]) + 1
                out[f"tok_{i}"] = x[j, :L].cpu().numpy().astype(np.float16)
                pooled.append((x[j, eot[j]] @ model.text_projection.float()).cpu().numpy())
                lens.append(L)
    out["pooled"] = np.stack(pooled).astype(np.float32)
    out["tok_len"] = np.array(lens, np.int32)
    return out


def run(data_dir, batch=256, device=None):
    texts = json.load(open(f"{data_dir}/texts.json"))["texts"]
    out = encode_texts(texts, batch, device)
    fp = f"{data_dir}/text_feats.npz"
    np.savez(fp, **out)
    lens = out["tok_len"]
    print(f"[text_feats] {len(texts)} texts -> {fp} ({os.path.getsize(fp)/1e6:.0f} MB) | "
          f"tok_len min/med/max {lens.min()}/{int(np.median(lens))}/{lens.max()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    run(a.data_dir, a.batch, a.device)
