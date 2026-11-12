"""Decoder-only Transformer with interchangeable KV-cache-efficient attention.

Attention variants (H query heads, G key/value groups, head width dh, rope width dr):

  gqa        Grouped-query attention. Caches grouped K (rotated) and V.          cache = 2*G*dh
  sharedkv   K = V (GVA paper's failed first cut). Caches V; RoPE applied to K.  cache = G*dh
  gva        Grouped Value Attention, K_h = V_g M_h, ordinary RoPE on K_h.        cache = G*dh
             (M sits between rotations -> cannot be absorbed at decode.)
  gva_drope  GVA + shared decoupled RoPE slice (the GVA paper's serving design).  cache = G*dh + dr
             q~_h = q_h M_h^T is absorbed into the query at decode.
  gka        Grouped Key Attention (ours), V_h = K_g N_h, ordinary RoPE on K.    cache = G*dh
             The cache holds the *unrotated* K; values are reconstructed from it,
             and N_h is absorbed into the output projection at decode.
             RoPE is applied to cached keys on the fly (pure elementwise op), so no
             decoupled slice is needed.
  gka_drope  GKA + shared decoupled RoPE slice (direct mirror of gva_drope).     cache = G*dh + dr

Every model exposes:
  forward(idx, targets)               training / full-sequence scoring (SDPA)
  setup_cache(B, Tmax) + step(idx)    inference with a persistent cache, using the
                                      absorbed decode path for each variant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ["gqa", "sharedkv", "gva", "gva_drope", "gka", "gka_drope"]


@dataclass
class Config:
    attn: str = "gqa"
    vocab_size: int = 50304
    d_model: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_group: int = 2
    head_dim: int = 64
    rope_dim: int = 16  # only used by *_drope variants
    seq_len: int = 512
    rope_base: float = 10000.0

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------------- RoPE

class Rotary(nn.Module):
    def __init__(self, dim: int, max_len: int, base: float):
        super().__init__()
        self.dim, self.base, self.ntk = dim, base, 1.0
        self._build(max_len)

    def _build(self, n, device=None):
        # NTK-aware scaling: stretch the base so the lowest frequencies cover ntk x the trained length
        # while the highest (local) frequencies stay almost unchanged.
        base = self.base * self.ntk ** (self.dim / (self.dim - 2))
        inv = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        freqs = torch.outer(torch.arange(n, device=device).float(), inv)  # (T, dim/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def set_ntk(self, factor: float):
        """factor=1 is plain RoPE; factor=s targets contexts ~s x the training length."""
        self.ntk = factor
        self._build(len(self.cos), self.cos.device)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Rotate x[..., T, dim] using positions pos (T,). Half-split convention."""
        if pos.max() >= len(self.cos):  # grow the tables for long-context evaluation
            self._build(2 * int(pos.max() + 1), self.cos.device)
        cos = self.cos[pos].to(x.dtype)
        sin = self.sin[pos].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ----------------------------------------------------------------------------- attention

class Attention(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        assert cfg.attn in VARIANTS, cfg.attn
        self.cfg = cfg
        self.kind = cfg.attn
        H, G, dh, dr, d = cfg.n_head, cfg.n_group, cfg.head_dim, cfg.rope_dim, cfg.d_model
        assert H % G == 0
        self.H, self.G, self.dh, self.dr, self.rep = H, G, dh, dr, H // G
        self.drope = self.kind.endswith("_drope")

        # query: content slice dh per head (+ rope slice dr per head for drope variants)
        self.q_proj = nn.Linear(d, H * (dh + (dr if self.drope else 0)), bias=False)
        if self.kind == "gqa":
            self.k_proj = nn.Linear(d, G * dh, bias=False)
            self.v_proj = nn.Linear(d, G * dh, bias=False)
        elif self.kind in ("sharedkv", "gva", "gva_drope"):
            self.v_proj = nn.Linear(d, G * dh, bias=False)  # only V is projected / cached
        elif self.kind in ("gka", "gka_drope"):
            self.k_proj = nn.Linear(d, G * dh, bias=False)  # only K is projected / cached
        if self.kind in ("gva", "gva_drope"):
            self.M = nn.Parameter(torch.empty(H, dh, dh))  # K_h = V_g M_h
        if self.kind in ("gka", "gka_drope"):
            self.N = nn.Parameter(torch.empty(H, dh, dh))  # V_h = K_g N_h
        if self.drope:
            self.kr_proj = nn.Linear(d, dr, bias=False)  # shared positional key
        self.o_proj = nn.Linear(H * dh, d, bias=False)
        self.o_proj.is_residual = True

        self.rope = Rotary(dr if self.drope else dh, cfg.seq_len * 8, cfg.rope_base)
        self.cache = None

    # -- helpers -------------------------------------------------------------
    def _split_q(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.H, -1).transpose(1, 2)  # (B,H,T,dq)
        if self.drope:
            return q[..., : self.dh], q[..., self.dh :]
        return q, None

    def _group(self, t):  # (B,G,T,D) -> (B,H,T,D)
        return t.repeat_interleave(self.rep, dim=1)

    # -- training / full-sequence forward -------------------------------------
    def forward(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device)
        G, dh = self.G, self.dh
        qc, qr = self._split_q(x)

        if self.kind == "gqa":
            k = self.k_proj(x).view(B, T, G, dh).transpose(1, 2)
            v = self.v_proj(x).view(B, T, G, dh).transpose(1, 2)
            q, k = self.rope(qc, pos), self.rope(k, pos)
            y = F.scaled_dot_product_attention(q, self._group(k), self._group(v), is_causal=True)
        elif self.kind == "sharedkv":
            v = self.v_proj(x).view(B, T, G, dh).transpose(1, 2)
            q, k = self.rope(qc, pos), self.rope(v, pos)
            y = F.scaled_dot_product_attention(q, self._group(k), self._group(v), is_causal=True)
        elif self.kind == "gva":
            v = self.v_proj(x).view(B, T, G, dh).transpose(1, 2)
            k = torch.einsum("bhtd,hde->bhte", self._group(v), self.M)  # per-head keys
            q, k = self.rope(qc, pos), self.rope(k, pos)
            y = F.scaled_dot_product_attention(q, k, self._group(v), is_causal=True)
        elif self.kind == "gva_drope":
            v = self.v_proj(x).view(B, T, G, dh).transpose(1, 2)
            kc = torch.einsum("bhtd,hde->bhte", self._group(v), self.M)
            kr = self.rope(self.kr_proj(x).unsqueeze(1), pos).expand(B, self.H, T, self.dr)
            q = torch.cat([qc, self.rope(qr, pos)], -1)
            k = torch.cat([kc, kr], -1)
            y = F.scaled_dot_product_attention(q, k, self._group(v), is_causal=True)
        elif self.kind == "gka":
            k = self.k_proj(x).view(B, T, G, dh).transpose(1, 2)  # unrotated content key
            v = torch.einsum("bhtd,hde->bhte", self._group(k), self.N)  # per-head values
            q, kr = self.rope(qc, pos), self.rope(k, pos)
            y = F.scaled_dot_product_attention(q, self._group(kr), v, is_causal=True)
        elif self.kind == "gka_drope":
            k = self.k_proj(x).view(B, T, G, dh).transpose(1, 2)
            v = torch.einsum("bhtd,hde->bhte", self._group(k), self.N)
            kr = self.rope(self.kr_proj(x).unsqueeze(1), pos).expand(B, self.H, T, self.dr)
            q = torch.cat([qc, self.rope(qr, pos)], -1)
            kk = torch.cat([self._group(k), kr], -1)
            y = F.scaled_dot_product_attention(q, kk, v, is_causal=True)

        y = y.transpose(1, 2).reshape(B, T, self.H * dh)
        return self.o_proj(y)

    # -- inference with persistent cache (absorbed decode paths) ---------------
    def cache_scalars_per_token(self) -> int:
        G, dh, dr = self.G, self.dh, self.dr
        return {
            "gqa": 2 * G * dh,
            "sharedkv": G * dh,
            "gva": G * dh,
            "gva_drope": G * dh + dr,
            "gka": G * dh,
            "gka_drope": G * dh + dr,
        }[self.kind]

    def setup_cache(self, B, Tmax, dtype, device):
        G, dh, dr = self.G, self.dh, self.dr
        c = {}
        if self.kind == "gqa":
            c["k"] = torch.zeros(B, G, Tmax, dh, dtype=dtype, device=device)
            c["v"] = torch.zeros(B, G, Tmax, dh, dtype=dtype, device=device)
        elif self.kind in ("sharedkv", "gva", "gva_drope"):
            c["v"] = torch.zeros(B, G, Tmax, dh, dtype=dtype, device=device)
        else:
            c["k"] = torch.zeros(B, G, Tmax, dh, dtype=dtype, device=device)
        if self.drope:
            c["kr"] = torch.zeros(B, 1, Tmax, dr, dtype=dtype, device=device)
        self.cache = c
        # Absorbed weights (fixed at inference).
        H = self.H
        if self.kind == "gva_drope":
            self.M_T = self.M.detach().transpose(1, 2).to(dtype)  # q~ = q M^T
        if self.kind in ("gka", "gka_drope"):
            # y = sum_h (A_h K_g) N_h W_O^h  ->  W_O' = blockdiag(N) @ W_O^T
            Wo = self.o_proj.weight.detach().T.view(H, dh, -1)  # (H, dh, d)
            self.Wo_abs = torch.einsum("hde,hef->hdf", self.N.detach(), Wo).reshape(H * dh, -1).to(dtype)

    def _grouped_scores(self, q, kc):
        """q (B,H,Tq,D), kc (B,G,L,D) -> scores (B,H,Tq,L) without expanding the cache."""
        B, H, Tq, D = q.shape
        qg = q.reshape(B, self.G, self.rep * Tq, D)
        return (qg @ kc.transpose(-1, -2)).view(B, H, Tq, -1)

    def _grouped_av(self, a, vc):
        """a (B,H,Tq,L), vc (B,G,L,D) -> (B,H,Tq,D)."""
        B, H, Tq, L = a.shape
        return (a.reshape(B, self.G, self.rep * Tq, L) @ vc).view(B, H, Tq, -1)

    def step(self, x, start: int):
        """Process a chunk x (B,T,d) at absolute positions [start, start+T) using the cache."""
        B, T, _ = x.shape
        c, G, H, dh = self.cache, self.G, self.H, self.dh
        pos = torch.arange(start, start + T, device=x.device)
        L = start + T
        qc, qr = self._split_q(x)

        # 1) write the new tokens into the cache
        if self.kind == "gqa":
            k = self.k_proj(x).view(B, T, G, dh).transpose(1, 2)
            c["k"][:, :, start:L] = self.rope(k, pos)
            c["v"][:, :, start:L] = self.v_proj(x).view(B, T, G, dh).transpose(1, 2)
        elif self.kind in ("sharedkv", "gva", "gva_drope"):
            c["v"][:, :, start:L] = self.v_proj(x).view(B, T, G, dh).transpose(1, 2)
        else:
            c["k"][:, :, start:L] = self.k_proj(x).view(B, T, G, dh).transpose(1, 2)
        if self.drope:
            c["kr"][:, :, start:L] = self.rope(self.kr_proj(x).unsqueeze(1), pos)

        # 2) scores against the full prefix
        allpos = torch.arange(L, device=x.device)
        if self.kind == "gqa":
            s = self._grouped_scores(self.rope(qc, pos), c["k"][:, :, :L])
            scale = dh
        elif self.kind in ("sharedkv", "gka"):
            # rotate cached (unrotated) keys on the fly; a fused kernel does this in registers
            krot = self.rope(c["k" if self.kind == "gka" else "v"][:, :, :L], allpos)
            s = self._grouped_scores(self.rope(qc, pos), krot)
            scale = dh
        elif self.kind == "gva":
            # cannot absorb: reconstruct + rotate all H keys every step
            k = torch.einsum("bhtd,hde->bhte", self._group(c["v"][:, :, :L]), self.M.to(x.dtype))
            s = self.rope(qc, pos) @ self.rope(k, allpos).transpose(-1, -2)
            scale = dh
        elif self.kind == "gva_drope":
            qt = torch.einsum("bhtd,hde->bhte", qc, self.M_T)  # absorbed query
            s = self._grouped_scores(qt, c["v"][:, :, :L])
            s = s + self.rope(qr, pos) @ c["kr"][:, :, :L].transpose(-1, -2)
            scale = dh + self.dr
        elif self.kind == "gka_drope":
            s = self._grouped_scores(qc, c["k"][:, :, :L])
            s = s + self.rope(qr, pos) @ c["kr"][:, :, :L].transpose(-1, -2)
            scale = dh + self.dr

        s = s.float() / math.sqrt(scale)
        if T > 1:
            mask = allpos[None, :] > pos[:, None]
            s = s.masked_fill(mask, float("-inf"))
        a = s.softmax(-1).to(x.dtype)

        # 3) aggregate
        if self.kind in ("gka", "gka_drope"):
            o = self._grouped_av(a, c["k"][:, :, :L])  # A_h K_g ; N_h absorbed into W_O
            o = o.transpose(1, 2).reshape(B, T, H * dh)
            return o @ self.Wo_abs
        vc = c["v"][:, :, :L]
        o = self._grouped_av(a, vc)
        o = o.transpose(1, 2).reshape(B, T, H * dh)
        return self.o_proj(o)


# ----------------------------------------------------------------------------- block / model

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        hidden = int(8 * d / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        self.down.is_residual = True

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        self.n1 = nn.RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, i)
        self.n2 = nn.RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))

    def step(self, x, start):
        x = x + self.attn.step(self.n1(x), start)
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init)
        for b in self.blocks:
            a = b.attn
            # Scale-matched init of the reconstruction maps (GVA paper, App. A):
            # sigma_M = sigma_target / (sigma_in * sqrt(d_in)). Both the source (V or K) and the
            # target (K or V) come from projections with the same std, so sigma_M = 1/sqrt(dh)
            # keeps the reconstructed stream at the same RMS as a directly projected one.
            if hasattr(a, "M"):
                nn.init.normal_(a.M, std=1 / math.sqrt(a.dh))
            if hasattr(a, "N"):
                nn.init.normal_(a.N, std=1 / math.sqrt(a.dh))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02
            if getattr(m, "is_residual", False):
                std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(m.weight, std=std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        return n - self.wte.weight.numel() if non_embedding else n

    def cache_scalars_per_token(self):
        return sum(b.attn.cache_scalars_per_token() for b in self.blocks)

    def set_rope_ntk(self, factor: float):
        """Inference-time NTK RoPE scaling for every layer (1.0 restores plain RoPE)."""
        for b in self.blocks:
            b.attn.rope.set_ntk(factor)

    def hidden(self, idx):
        """Final normalised hidden states (B, T, d), i.e. everything but the LM head."""
        x = self.wte(idx)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

    def forward(self, idx, targets=None):
        logits = self.lm_head(self.hidden(idx))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # -- inference ------------------------------------------------------------
    @torch.no_grad()
    def setup_cache(self, B, Tmax, dtype=torch.bfloat16):
        dev = self.wte.weight.device
        for b in self.blocks:
            b.attn.setup_cache(B, Tmax, dtype, dev)

    def free_cache(self):
        for b in self.blocks:
            a = b.attn
            a.cache = None
            for n in ("M_T", "Wo_abs"):
                if hasattr(a, n):
                    delattr(a, n)

    @torch.no_grad()
    def step(self, idx, start):
        x = self.wte(idx)
        for b in self.blocks:
            x = b.step(x, start)
        return self.lm_head(self.norm(x[:, -1:]))
