"""
Qwen3-style decoder, the "modern architecture" control asked for in the ICLR 2027 reviews:
RMSNorm pre-norm, grouped-query attention with per-head QK-norm, rotary position
embeddings, SwiGLU MLP, no biases, tied input/output embedding (as in Qwen3-0.6B).

Tokenizer, data pipeline, parameter routing and training loop are those of GPTBase, so
the only thing that changes between `--model base` and `--model qwen` is the architecture.
Routing is unchanged as well: every weight matrix inside the blocks is 2-D with fewer than
10,000 rows and takes the matrix / sign update; the tied token embedding and all 1-D
parameters (RMSNorm gains, QK-norm gains) take AdamW.
"""

import math

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import GPTBase


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q: (B, nh, T, hd), k: (B, nkv, T, hd), cos / sin: (T, hd); rotate in fp32
    q32, k32 = q.float(), k.float()
    q32 = q32 * cos + rotate_half(q32) * sin
    k32 = k32 * cos + rotate_half(k32) * sin
    return q32.type_as(q), k32.type_as(k)


class QwenAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head else max(1, config.n_head // 2)
        assert self.n_head % self.n_kv_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        # named c_proj so GPTBase's residual-projection init (std / sqrt(2 n_layer)) applies
        self.c_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rmsnorm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rmsnorm_eps)
        self.dropout = config.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q = self.q_norm(self.q_proj(x).view(B, T, self.n_head, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
            enable_gqa=self.n_kv_head != self.n_head,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = int(2 * (4 * config.n_embd) / 3)
        hidden = config.multiple_of * ((hidden + config.multiple_of - 1) // config.multiple_of)
        self.hidden = hidden
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.attn = QwenAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)
        return x + self.mlp(self.ln_2(x))


class Qwen(GPTBase):
    def __init__(self, config):
        nn.Module.__init__(self)
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([QwenBlock(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd, eps=config.rmsnorm_eps),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # tied, as in Qwen3-0.6B

        head_dim = config.n_embd // config.n_head
        theta = float(getattr(config, "rope_theta", 1e6))
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(config.sequence_length).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", emb.cos(), persistent=False)
        self.register_buffer("rope_sin", emb.sin(), persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=config.init_std / math.sqrt(2 * config.n_layer)
                )

    def get_num_params(self, non_embedding=False):
        # no position table to subtract; the tied embedding is counted once
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, get_logits=False):
        b, t = idx.size()
        assert (
            t <= self.config.sequence_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        x = self.transformer.drop(self.transformer.wte(idx))
        cos, sin = self.rope_cos[:t], self.rope_sin[:t]
        for block in self.transformer.h:
            x = block(x, cos, sin)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return {"logits": logits if get_logits else None, "loss": loss}
