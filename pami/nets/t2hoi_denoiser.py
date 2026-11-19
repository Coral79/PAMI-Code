"""PamiGen denoiser: rectified-flow velocity network over the PamiVAE latent grid.

Per latent step k (9 tokens): [ h_0..h_7 | o_0 ]
  h_p : human part-p latent z_h[k,p] in R^16 (6 body parts + 2 hands)   — DIFFUSED
  o_0 : object latent z_o[k,0] in R^16 (InterAct: one object per piece)  — DIFFUSED
Sequence = interleave over k -> (B, Tr*9, D). Backbone (from LIGHT's rope_decoder lineage): modality-aware
RoPE (the 9 tokens of one step share one time position), AdaLN(zero-init) modulation from (timestep, pooled
CLIP caption) with separate human / object parameter sets, self-attention over the grid (padding + step
masks), cross-attention to the CLIP caption tokens, FFN.
Conditions: per-slot object dual-BPS (6145 -> 256 -> 64 -> D, added to the object token); body shape
(63 rest-bone offsets, standardised with train stats, MLP -> D, zero-init, added to every human token);
caption pooled -> AdaLN, caption tokens -> cross-attention memory.
Output: zero-init Linear(D->16) on every diffusion token -> velocity for z_h / z_o.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_HP = 8                     # human part tokens (6 body + 2 hands)
LAT = 16                     # PamiVAE per-token latent
CLIP_D = 512                 # CLIP ViT-B/32 text width
BPS_DUAL = 6145              # raw 3072 + normalised 3072 + scale 1
SHAPE_WIDTH = 63             # body-shape condition: 21 parent-relative rest bone offsets


def zero_module(m):
    for p in m.parameters():
        p.detach().zero_()
    return m


# ---- RoPE ----------------------------------------------------------------------------------------
def build_rope_cache(seq_len, n_elem, dtype, device, base=10000):
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))
    idx_theta = torch.outer(torch.arange(seq_len, dtype=dtype, device=device), theta)
    return torch.polar(torch.ones_like(idx_theta), idx_theta)          # complex64 (T, n_elem/2)


def apply_rope(x, rope_cache, start_pos=0):
    """x (B, H, T, hd) -> RoPE-rotated, rope_cache (>=T, hd/2) complex."""
    x = x.transpose(1, 2)                                              # (B,T,H,hd)
    T = x.size(1)
    rc = rope_cache[start_pos:start_pos + T]
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(xc * rc.view(1, T, 1, xc.size(3))).flatten(3)
    return x_out.transpose(1, 2).type_as(x)


class AdaLN(nn.Module):
    def __init__(self, d, cond_dim):
        super().__init__()
        self.emb = nn.Sequential(nn.SiLU(), zero_module(nn.Linear(cond_dim, 2 * d)))
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)

    def forward(self, h, c):                                           # h (...,d), c (B,cond)
        scale, shift = self.emb(c).chunk(2, dim=-1)
        while scale.dim() < h.dim():
            scale = scale.unsqueeze(1); shift = shift.unsqueeze(1)
        return self.norm(h) * (1 + scale) + shift


class MHAWithRoPE(nn.Module):
    def __init__(self, d, heads, dropout=0.0, num_modalities=1):
        super().__init__()
        assert d % heads == 0 and (d // heads) % 2 == 0
        self.d, self.h, self.hd = d, heads, d // heads
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.dropout = dropout
        self.nm = num_modalities
        self._rope = torch.zeros(0)                                    # lazy cache (plain attribute: EMA-safe)

    def _rope_cache(self, T_tok, dev):
        steps = (T_tok + self.nm - 1) // self.nm
        if self._rope.numel() == 0 or self._rope.shape[0] < steps * max(self.nm, 1) or self._rope.device != dev:
            base = build_rope_cache(steps, self.hd, torch.float32, dev)
            self._rope = torch.repeat_interleave(base, self.nm, dim=0) if self.nm > 1 else base
        return self._rope

    def forward(self, x_q, x_kv, attn_mask=None, use_rope=True):
        B, Tq, _ = x_q.shape; Tk = x_kv.shape[1]
        q = self.q(x_q).view(B, Tq, self.h, self.hd).transpose(1, 2)
        k = self.k(x_kv).view(B, Tk, self.h, self.hd).transpose(1, 2)
        v = self.v(x_kv).view(B, Tk, self.h, self.hd).transpose(1, 2)
        if use_rope:
            rc = self._rope_cache(Tq, x_q.device)
            q = apply_rope(q, rc); k = apply_rope(k, rc)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.o(y.transpose(1, 2).contiguous().view(B, Tq, self.d))


class GridBlock(nn.Module):
    """AdaLN(human / object groups) -> self-attn (RoPE, grid mask) -> cross-attn (text) -> FFN.
    Pre-modulation feeds only the self-attn QKV; residuals run from the un-modulated stream."""
    def __init__(self, d, heads, cond_dim, ff, dropout, n_hp=N_HP, m=1):
        super().__init__()
        self.n_hp, self.m = n_hp, m
        self.n_tok = n_hp + m
        self.ada_h = AdaLN(d, cond_dim)
        self.ada_o = AdaLN(d, cond_dim)
        self.self_attn = MHAWithRoPE(d, heads, dropout, num_modalities=self.n_tok)
        self.cross_attn = MHAWithRoPE(d, heads, dropout, num_modalities=1)
        self.n2 = nn.LayerNorm(d); self.n3 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ff, bias=False), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ff, d, bias=False))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, c, text_mem, grid_mask, text_mask):
        B, L, D = x.shape
        Tr = L // self.n_tok
        xg = x.view(B, Tr, self.n_tok, D)
        h = torch.cat([self.ada_h(xg[:, :, :self.n_hp], c), self.ada_o(xg[:, :, self.n_hp:], c)], dim=2).view(B, L, D)
        x = x + self.drop(self.self_attn(h, h, attn_mask=grid_mask))
        x = x + self.drop(self.cross_attn(self.n2(x), text_mem, attn_mask=text_mask, use_rope=False))
        x = x + self.drop(self.ffn(self.n3(x)))
        return x


class TimestepEmbedder(nn.Module):
    """Sinusoidal t in [0,1] -> MLP (DiT-style, continuous)."""
    def __init__(self, d, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):                                              # t (B,) in [0,1]
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None] * 1000.0
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class T2HOIDenoiser(nn.Module):
    def __init__(self, d=512, layers=10, heads=8, ff=2048, dropout=0.1, cond_dim=None,
                 bps_hid=256, bps_code=64, n_obj=1, n_hp=N_HP, human_shape=True):
        super().__init__()
        cond_dim = cond_dim or d
        self.d = d
        self.n_hp, self.m = n_hp, n_obj
        self.n_tok = self.n_hp + n_obj
        self.token_layout = "split"
        self.human_shape = bool(human_shape)
        self.human_shape_src = "jrest" if self.human_shape else None
        self.tok_in = nn.Linear(LAT, d)
        self.type_emb = nn.Parameter(torch.zeros(self.n_tok, d)); nn.init.normal_(self.type_emb, std=0.02)
        self.bps_mlp = nn.Sequential(nn.Linear(BPS_DUAL, bps_hid), nn.ReLU(), nn.Linear(bps_hid, bps_code))
        self.bps_proj = nn.Linear(bps_code, d)
        self.text_proj = nn.Linear(CLIP_D, d)                          # caption tokens -> cross-attn memory
        self.t_emb = TimestepEmbedder(cond_dim)
        self.cap_emb = nn.Linear(CLIP_D, cond_dim)                     # pooled caption -> AdaLN cond
        if self.human_shape:
            # RNG discipline (kept from the original): jb_mlp is drawn from a dedicated generator so the
            # backbone init at a given seed equals that of an unconditioned model; output zero-init.
            _rng = torch.get_rng_state()
            self.jb_mlp = nn.Sequential(nn.Linear(SHAPE_WIDTH, 128), nn.SiLU(), nn.Linear(128, d))
            torch.set_rng_state(_rng)
            _g = torch.Generator().manual_seed(20260815)
            with torch.no_grad():
                for lin in (self.jb_mlp[0], self.jb_mlp[2]):
                    bd = 1.0 / math.sqrt(lin.weight.shape[1])
                    lin.weight.uniform_(-bd, bd, generator=_g)
                    lin.bias.uniform_(-bd, bd, generator=_g)
            nn.init.zeros_(self.jb_mlp[-1].weight); nn.init.zeros_(self.jb_mlp[-1].bias)
            self.register_buffer("jb_mean", torch.zeros(SHAPE_WIDTH))   # train-split stats, saved in the ckpt
            self.register_buffer("jb_std", torch.ones(SHAPE_WIDTH))
        self.blocks = nn.ModuleList(GridBlock(d, heads, cond_dim, ff, dropout, n_hp=self.n_hp, m=n_obj)
                                    for _ in range(layers))
        self.out_norm = nn.LayerNorm(d)
        self.out = zero_module(nn.Linear(d, LAT))

    def _shape_emb(self, jbone):
        if not self.human_shape:
            return None
        if jbone is None:
            raise RuntimeError("human_shape model called without jbone (the dataset serves it)")
        return self.jb_mlp((jbone - self.jb_mean) / self.jb_std)[:, None]

    def forward(self, zh, zo, t, omask, step_mask, bps_dual, cap_pooled, cap_tok, cap_tok_mask, jbone=None):
        """zh (B,Tr,8,16) zo (B,Tr,M,16) noisy state; t (B,) in [0,1]; omask (B,M); step_mask (B,Tr) 1=valid;
        bps_dual (B,M,6145); cap_pooled (B,512); cap_tok (B,Lt,512); cap_tok_mask (B,Lt) 1=real; jbone (B,63).
        Returns (vh (B,Tr,8,16), vo (B,Tr,M,16))."""
        B, Tr = zh.shape[:2]
        dev = zh.device
        se = self._shape_emb(jbone)                                    # (B,1,d) | None
        toks = [self.tok_in(zh) + self.type_emb[None, None, :self.n_hp]]
        if se is not None:
            toks[0] = toks[0] + se[:, :, None]
        e_s = self.bps_mlp(bps_dual)                                   # (B,M,64) per-slot shape id
        toks.append(self.tok_in(zo) + self.type_emb[None, None, self.n_hp:self.n_hp + self.m]
                    + self.bps_proj(e_s)[:, None])                     # (B,Tr,M,D)
        x = torch.cat(toks, dim=2).view(B, Tr * self.n_tok, self.d)
        tok_valid = torch.ones(B, self.n_tok, device=dev, dtype=torch.bool)
        tok_valid[:, self.n_hp:self.n_hp + self.m] = omask.bool()
        valid = (step_mask.bool()[:, :, None] & tok_valid[:, None, :]).view(B, Tr * self.n_tok)
        grid_mask = valid[:, None, None, :]                            # SDPA bool key mask (True = attend)
        text_mem = self.text_proj(cap_tok)
        text_mask = cap_tok_mask.bool()[:, None, None, :]
        c = self.t_emb(t) + self.cap_emb(cap_pooled)
        for blk in self.blocks:
            x = blk(x, c, text_mem, grid_mask, text_mask)
        x = self.out(self.out_norm(x)).view(B, Tr, self.n_tok, LAT)
        return x[:, :, :self.n_hp], x[:, :, self.n_hp:self.n_hp + self.m]
