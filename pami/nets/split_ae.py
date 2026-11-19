"""PamiVAE (SplitAE-ST): the part-anchored human-object interaction autoencoder.

STREAMS
  human : 272 -> 6 anatomical part slices [8,72,48,48,48,48] + 2 hand slices (wrist-local hand block, 135 each)
          -> 8 causal part encoders -> z_h (B,Tr,8,16)          (Tr = W/4 latent steps)
  object: per slot [robj6 | pG3 | delta(5 anchor parts x 3)] = 24/frame -> MotionLift -> shared causal
          encoder -> z_o (B,Tr,8,16). The 8 slots share one weight set; padded slots are masked.
COUPLING  SceneST x6 over the 16-token grid [z_h 8 | z_o 8] (spatial MSA with padding mask, causal temporal
          MSA over the flattened grid, FFN), zero-init residual write-back.
DECODE
  human : refined z_h -> Linear(128->1024) -> causal decoder -> 272 body + 270 hand channels
          (+ optional 4-logit foot-contact head, the contact-tuned decoder checkpoint).
  object: refined z_o -> causal decoder -> [robj 6 | p_G 3 | delta 5x3 | w 7] per slot.
          The final object position is the routed combine  p = sum_p softmax(w)_p (anchor_p + R_p delta_p)
          + w_FREE p_G  with anchors FK'd from the decoded human (losses.py / sample_t2hoi.py).

Checkpoint arch keys read by ckpt_io.load_split_ae: delta_parts, hand_mode (wlocalrot), contact_head, sae_*.
"""
import torch, torch.nn as nn
from .. import config as C
from pami.vendor.causal_cnn import CausalConv1d, CausalEncoder, CausalDecoder
from pami.vendor.tae_part import get_part_indices_272_smplh

OBJ_MOTION_IN = C.ROBJ_D + C.POBJ_D          # [robj 6, pG 3] = 9
HAND_PART_SLICES = [list(range(0, 45)) + list(range(90, 180)),      # left hand : wrist-local pos 45 + rot6d 90
                    list(range(45, 90)) + list(range(180, 270))]    # right hand
PART_NAMES = ["root", "torso", "Larm", "Rarm", "Lleg", "Rleg"]


class SceneST(nn.Module):
    """Symmetric Scene-ST refiner over the (B, Tr, n_h + n_o, L) latent grid.
    Per layer (pre-LN residual): (1) spatial MSA over the tokens of one step (padded object slots masked),
    (2) causal temporal MSA over the flattened (n_tok*D) step vector, (3) token FFN.
    out_proj is zero-initialised so a fresh model is the identity. Padded object tokens are held at zero at
    every layer boundary so they never leak into the temporal attention."""
    def __init__(self, n_h, n_o=C.MAX_OBJS, lat=None, d=None, layers=None, sp_heads=None, tm_heads=None):
        super().__init__()
        self.n_h = n_h; self.n_o = n_o; self.n_tok = n_h + n_o
        L = lat if lat is not None else C.SAE_LATENT
        D = d if d is not None else C.SAE_DMODEL
        self.layers_n = layers if layers is not None else C.SAE_LAYERS
        self.sp_heads = sp_heads if sp_heads is not None else C.SAE_SP_HEADS
        self.tm_heads = tm_heads if tm_heads is not None else C.SAE_TM_HEADS
        self.L = L; self.D = D
        assert D % self.sp_heads == 0, f"D={D} not divisible by sp_heads={self.sp_heads}"
        assert (self.n_tok * D) % self.tm_heads == 0, f"n_tok*D={self.n_tok*D} not divisible by tm_heads={self.tm_heads}"
        self.lift = nn.Linear(L, D)
        self.tok_emb = nn.Parameter(torch.zeros(self.n_tok, D)); nn.init.normal_(self.tok_emb, std=0.02)
        self.time_table = nn.Parameter(torch.zeros(256, D)); nn.init.normal_(self.time_table, std=0.02)   # absolute step embedding (<=256 steps)
        self.sp_ln, self.sp_mha = nn.ModuleList(), nn.ModuleList()
        self.tm_ln, self.tm_mha = nn.ModuleList(), nn.ModuleList()
        self.ff_ln, self.ff = nn.ModuleList(), nn.ModuleList()
        for _ in range(self.layers_n):
            self.sp_ln.append(nn.LayerNorm(D))
            self.sp_mha.append(nn.MultiheadAttention(D, self.sp_heads, batch_first=True))
            self.tm_ln.append(nn.LayerNorm(self.n_tok * D))
            self.tm_mha.append(nn.MultiheadAttention(self.n_tok * D, self.tm_heads, batch_first=True))
            self.ff_ln.append(nn.LayerNorm(D))
            self.ff.append(nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)))
        self.out_ln = nn.LayerNorm(D)
        self.out_proj = nn.Linear(D, L)
        nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)

    def forward(self, grid, omask):
        """grid (B, Tr, n_tok, L); omask (B, n_o) 1.0 on active object slots -> refined (B, Tr, n_tok, L)."""
        B, Tr, N, L = grid.shape
        assert N == self.n_tok, f"grid has {N} tokens, expected {self.n_tok}"
        tok_valid = torch.ones(B, N, device=grid.device, dtype=grid.dtype)
        tok_valid[:, self.n_h:] = omask
        key_pad_sp = (tok_valid == 0)[:, None, :].expand(B, Tr, N).reshape(B * Tr, N)
        causal = torch.triu(torch.full((Tr, Tr), float("-inf"), device=grid.device), diagonal=1)
        x = self.lift(grid) + self.tok_emb[None, None] + self.time_table[:Tr][None, :, None, :]
        x = x * tok_valid[:, None, :, None]
        for i in range(self.layers_n):
            xn = self.sp_ln[i](x).reshape(B * Tr, N, self.D)
            a, _ = self.sp_mha[i](xn, xn, xn, key_padding_mask=key_pad_sp, need_weights=False)
            x = x + a.reshape(B, Tr, N, self.D)
            x = x * tok_valid[:, None, :, None]
            xf = x.reshape(B, Tr, N * self.D)
            xfn = self.tm_ln[i](xf)
            t, _ = self.tm_mha[i](xfn, xfn, xfn, attn_mask=causal, need_weights=False)
            x = (xf + t).reshape(B, Tr, N, self.D)
            x = x + self.ff[i](self.ff_ln[i](x))
            x = x * tok_valid[:, None, :, None]
        upd = self.out_proj(self.out_ln(x))
        return grid + upd * tok_valid[:, None, :, None]


class SplitAE(nn.Module):
    def __init__(self, delta_parts=None, hand_mode="wlocalrot", contact_head=False):
        """delta_parts: body parts (indices into PART_NAMES) that carry an anchor delta + routing slot
        (paper: [0,2,3,4,5] = root + both arms + both legs). hand_mode: "wlocalrot" (the only supported hand
        block). contact_head: add the 4-logit foot-contact head (contact-tuned decoder checkpoint)."""
        super().__init__()
        assert hand_mode == "wlocalrot", hand_mode
        self.hand_mode = hand_mode
        self.hand_dim = C.HAND_DIM
        self.st_on = True; self.obj_delta = True; self.delta_input = True; self.predict_w = True
        self.use_bps = False; self.explicit_root = False; self.rotvote = False
        self.contact_head_on = bool(contact_head)
        self.delta_parts = list(delta_parts) if delta_parts is not None else list(C.ANCHOR_PARTS)
        nd = 3 * len(self.delta_parts)
        self.obj_out = OBJ_MOTION_IN + nd + C.W_D                 # [robj 6 | pG 3 | delta | w 7]
        self.obj_in = OBJ_MOTION_IN + nd                          # encoder also reads the GT part deltas
        wm = torch.full((C.W_D,), float("-inf"))                  # routing restricted to delta_parts + FREE
        for p in self.delta_parts + [C.W_D - 1]:
            wm[p] = 0.0
        self.register_buffer("w_mask", wm)
        self.part_indices = get_part_indices_272_smplh()
        self.part_dims = [len(p) for p in self.part_indices]
        self.hand_part_slices = HAND_PART_SLICES
        self.part_dims += [len(s) for s in self.hand_part_slices]
        self.P = len(self.part_dims)                              # 8
        self.M = C.MAX_OBJS
        L = C.SAE_LATENT
        # --- human stream ---
        HEW, HDW = C.SAE_H_ENC_WIDTH, C.SAE_H_DEC_WIDTH
        self.h_encoders = nn.ModuleList(
            CausalEncoder(input_emb_width=pd, hidden_size=HEW, width=HEW, latent_dim=L, clip_range=C.CLIP_RANGE)
            for pd in self.part_dims)
        self.h_fusion = nn.Linear(L * self.P, HDW)
        self.h_decoder = CausalDecoder(input_emb_width=C.DIM_272 + self.hand_dim, hidden_size=HDW, width=HDW)
        self.contact_head = (nn.Sequential(CausalConv1d(C.DIM_272 + self.hand_dim, 64, 3), nn.GELU(), CausalConv1d(64, 4, 1))
                             if self.contact_head_on else None)
        # --- object stream ---
        OEW, ODW, ML = C.SAE_O_ENC_WIDTH, C.SAE_O_DEC_WIDTH, C.SAE_MOTIONLIFT
        self.motion_lift = nn.Sequential(nn.Linear(self.obj_in, ML), nn.ReLU(), nn.Linear(ML, ML))
        self.o_encoder = CausalEncoder(input_emb_width=ML, hidden_size=OEW, width=OEW, latent_dim=L, clip_range=C.CLIP_RANGE)
        self.o_dec_proj = nn.Linear(L, ODW)
        self.o_decoder = CausalDecoder(input_emb_width=self.obj_out, hidden_size=ODW, width=ODW)
        # --- coupling (built last; one temporal head per token for the 8+8 grid) ---
        self.refiner = SceneST(n_h=self.P, n_o=self.M, lat=L, tm_heads=self.P + self.M)

    def _encode_human(self, human, hand):
        """human (B,W,272), hand (B,W,270) -> z_h (B,Tr,P,L), mu (B,Tr,P*L), logvar (B,Tr,P*L)."""
        if hand is None:
            raise RuntimeError("SplitAE needs the hand block (batch key 'hand')")
        x = human.permute(0, 2, 1); xh = hand.permute(0, 2, 1)
        zs, mus, lvs = [], [], []
        for i, idx in enumerate(self.part_indices):
            z, mu, lv = self.h_encoders[i](x[:, idx, :])
            zs.append(z); mus.append(mu); lvs.append(lv)
        for j, sl in enumerate(self.hand_part_slices):
            z, mu, lv = self.h_encoders[len(self.part_indices) + j](xh[:, sl, :])
            zs.append(z); mus.append(mu); lvs.append(lv)
        return torch.stack(zs, 2), torch.cat(mus, 2), torch.cat(lvs, 2)

    def _encode_object(self, b):
        """b: robj (B,M,W,6), pobj (B,M,W,3), delta (B,M,W,6,3), omask (B,M) (normalised)
        -> z_o (B,M,Tr,L), mu_o, logvar_o (B,M,Tr,L), None (no shape code)."""
        robj, pobj, om = b["robj"], b["pobj"], b["omask"]
        B, M, W, _ = robj.shape
        dh = b["delta"][..., self.delta_parts, :].reshape(B, M, W, -1)
        f = self.motion_lift(torch.cat([robj, pobj, dh], dim=-1))          # (B,M,W,ML)
        z, mu, lv = self.o_encoder(f.reshape(B * M, W, -1).permute(0, 2, 1))
        Tr = z.shape[1]
        z = z.reshape(B, M, Tr, -1) * om[:, :, None, None]                  # padded slots -> 0
        return z, mu.reshape(B, M, Tr, -1), lv.reshape(B, M, Tr, -1), None

    def _decode_object(self, z_o, e_m=None, return_delta=False):
        """z_o (B,M,Tr,L) -> robj (B,M,W,6), pobj (B,M,W,3) [, delta (B,M,W,|parts|,3), w (B,M,W,7) masked logits]."""
        B, M, Tr, L = z_o.shape
        h = self.o_dec_proj(z_o).reshape(B * M, Tr, -1)
        out = self.o_decoder(h).permute(0, 2, 1).reshape(B, M, -1, self.obj_out)
        robj = out[..., :C.ROBJ_D]
        pobj = out[..., C.ROBJ_D:C.ROBJ_D + C.POBJ_D]
        if not return_delta:
            return robj, pobj
        d0 = C.ROBJ_D + C.POBJ_D
        delta = out[..., d0:d0 + 3 * len(self.delta_parts)].reshape(B, M, -1, len(self.delta_parts), 3)
        w = out[..., -C.W_D:] + self.w_mask
        return robj, pobj, delta, w

    def decode_human_full(self, zf):
        """zf (B,Tr,P*L) fused refined latents -> dict(dec_raw (B,W,542), human_recon (B,W,272),
        hand_recon (B,W,270), contact_logit (B,W,4) | None)."""
        dec = self.h_decoder(self.h_fusion(zf)).permute(0, 2, 1)
        contact_logit = self.contact_head(dec.permute(0, 2, 1)).permute(0, 2, 1) if self.contact_head is not None else None
        return dict(dec_raw=dec, human_recon=dec[..., :C.DIM_272], hand_recon=dec[..., C.DIM_272:],
                    contact_logit=contact_logit)

    def refine(self, z_h, z_o, omask):
        """SceneST on (z_h (B,Tr,P,L), z_o (B,M,Tr,L)) -> refined (z_h, z_o) with padded slots re-zeroed."""
        B, Tr, P, L = z_h.shape
        grid = self.refiner(torch.cat([z_h, z_o.permute(0, 2, 1, 3)], dim=2), omask)
        return grid[:, :, :P, :], grid[:, :, P:, :].permute(0, 2, 1, 3) * omask[:, :, None, None]

    def forward(self, b):
        z_h, mu_h, logvar_h = self._encode_human(b["human"], b.get("hand"))
        z_o, mu_o, logvar_o, _ = self._encode_object(b)
        B, Tr, P, L = z_h.shape
        z_h_r, z_o_r = self.refine(z_h, z_o, b["omask"])
        dh = self.decode_human_full(z_h_r.reshape(B, Tr, P * L))
        robj, pobj, delta, w = self._decode_object(z_o_r, return_delta=True)
        return {"human_recon": dh["human_recon"], "hand_recon": dh["hand_recon"], "contact_logit": dh["contact_logit"],
                "mu_h": mu_h, "logvar_h": logvar_h,
                "robj": robj, "pobj": pobj, "delta": delta, "delta_parts": self.delta_parts, "w": w,
                "mu_o": mu_o, "logvar_o": logvar_o}
