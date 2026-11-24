"""A small decoder-only transformer with modern (Llama-style) components.

Compared with the original GPT-2 block this model uses:
  * RMSNorm instead of LayerNorm (no mean-centering, no bias)
  * Rotary position embeddings (RoPE) instead of learned absolute positions
  * Grouped-query attention (GQA): fewer key/value heads than query heads
  * SwiGLU feed-forward instead of GELU MLP
  * Fused scaled_dot_product_attention (Flash / memory-efficient kernels on GPU)
  * A KV-cache for fast autoregressive generation

Each component can be switched back to its GPT-2 counterpart through the
config, which is what the ablation script uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 4  # == n_head -> standard MHA, 1 -> multi-query attention
    n_embd: int = 512
    dropout: float = 0.0
    rope_theta: float = 10000.0
    # ablation switches
    norm: str = "rmsnorm"  # "rmsnorm" | "layernorm"
    mlp: str = "swiglu"  # "swiglu" | "gelu"
    pos: str = "rope"  # "rope" | "learned"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------- pieces
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for stability under mixed precision.
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def make_norm(cfg: GPTConfig) -> nn.Module:
    return RMSNorm(cfg.n_embd) if cfg.norm == "rmsnorm" else nn.LayerNorm(cfg.n_embd)


def rope_cache(seq_len: int, head_dim: int, theta: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D). Rotates each consecutive (even, odd) pair of channels."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class KVCache:
    """Pre-allocated key/value buffers, one pair per layer."""

    def __init__(self, n_layer: int, batch: int, n_kv_head: int, max_len: int, head_dim: int, device, dtype):
        shape = (batch, n_kv_head, max_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.pos = 0
        self.max_len = max_len

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T = k.size(2)
        self.k[layer][:, :, self.pos : self.pos + T] = k
        self.v[layer][:, :, self.pos : self.pos + T] = v
        end = self.pos + T
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0 and cfg.n_head % cfg.n_kv_head == 0
        self.layer_idx = layer_idx
        self.n_head, self.n_kv_head = cfg.n_head, cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.use_rope = cfg.pos == "rope"
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(cfg.n_embd, 2 * cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache: KVCache | None = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(x).view(B, T, 2, self.n_kv_head, self.head_dim).permute(2, 0, 3, 1, 4)

        if self.use_rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)

        # GQA: share each key/value head across a group of query heads.
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # With a cache and a single new token, every cached key is in the past,
        # so no mask is needed. For a multi-token prefill into an empty cache the
        # usual causal mask applies.
        causal = cache is None or T > 1
        if cache is not None and T > 1 and cache.pos > 0:
            raise ValueError("multi-token decode into a non-empty cache is not supported")
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, dropout_p=self.dropout if self.training else 0.0
        )
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.kind = cfg.mlp
        if cfg.mlp == "swiglu":
            # 2/3 * 4d keeps the parameter count equal to a 4d GELU MLP.
            hidden = int(2 * 4 * cfg.n_embd / 3)
            hidden = 64 * ((hidden + 63) // 64)
            self.gate_up = nn.Linear(cfg.n_embd, 2 * hidden, bias=False)
            self.down = nn.Linear(hidden, cfg.n_embd, bias=False)
        else:
            self.up = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
            self.down = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "swiglu":
            gate, up = self.gate_up(x).chunk(2, dim=-1)
            return self.drop(self.down(F.silu(gate) * up))
        return self.drop(self.down(F.gelu(self.up(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.norm1 = make_norm(cfg)
        self.attn = Attention(cfg, layer_idx)
        self.norm2 = make_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------- model
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.norm_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = rope_cache(cfg.block_size, head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style scaled init for residual projections.
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx, targets=None, loss_mask=None, cache: KVCache | None = None):
        B, T = idx.shape
        start = cache.pos if cache is not None else 0
        assert start + T <= self.cfg.block_size, f"sequence length {start + T} > block_size"

        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(start, start + T, device=idx.device))
        x = self.drop(x)
        cos, sin = self.rope_cos[start : start + T], self.rope_sin[start : start + T]
        for block in self.blocks:
            x = block(x, cos, sin, cache)
        if cache is not None:
            cache.pos += T
        x = self.norm_f(x)

        if targets is None:
            return self.lm_head(x[:, -1:, :]), None  # only the last position is needed

        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-1, reduction="none"
        )
        if loss_mask is not None:
            m = loss_mask.reshape(-1).float()
            loss = (loss * m).sum() / m.sum().clamp(min=1.0)
        else:
            valid = (targets.reshape(-1) != -1).float()
            loss = loss.sum() / valid.sum().clamp(min=1.0)
        return logits, loss

    # ---------------------------------------------------------------- optim
    def configure_optimizer(self, lr: float, weight_decay: float, betas=(0.9, 0.95), device_type="cpu"):
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    # ------------------------------------------------------------- generate
    def new_cache(self, batch: int, max_len: int | None = None) -> KVCache:
        cfg = self.cfg
        p = next(self.parameters())
        return KVCache(
            cfg.n_layer, batch, cfg.n_kv_head, max_len or cfg.block_size, cfg.n_embd // cfg.n_head, p.device, p.dtype
        )

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop_ids: set[int] | None = None,
        use_cache: bool = True,
    ):
        """Yields one token id tensor of shape (B, 1) at a time."""
        self.eval()
        idx = idx[:, -self.cfg.block_size :]
        cache = self.new_cache(idx.size(0)) if use_cache else None
        x = idx
        for _ in range(max_new_tokens):
            if cache is not None and cache.pos + x.size(1) > self.cfg.block_size:
                break  # context full
            if cache is None:
                x = idx[:, -self.cfg.block_size :]
            logits, _ = self(x, cache=cache)
            logits = logits[:, -1, :].float()
            nxt = sample_next(logits, temperature, top_k, top_p)
            yield nxt
            if stop_ids and nxt.numel() == 1 and nxt.item() in stop_ids:
                break
            idx = torch.cat([idx, nxt], dim=1)
            x = nxt if cache is not None else idx


def sample_next(logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float | None) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = sorted_logits.softmax(-1)
        cum = probs.cumsum(-1)
        remove = cum - probs > top_p
        sorted_logits[remove] = -float("inf")
        logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_idx, sorted_logits)
    return torch.multinomial(logits.softmax(-1), 1)
