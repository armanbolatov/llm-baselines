"""Dion (Ahn et al., 2025, arXiv:2504.05295) — single-device reimplementation
of Algorithm 2 (unsharded) for the single-node comparison at 124M.

Per 2D parameter it keeps a momentum buffer M and warm-started right vectors
V (n x r). Each step: accumulate gradient into M, one power-iteration step
(P = M V, U = QR(P), W = M^T U), error feedback M -= beta * U W^T (beta=0.05),
column-normalize W into V, and apply the orthonormal update U V^T.

To make the tuned Muon learning rate transferable, the update uses the same
Moonshot RMS scaling 0.2 * sqrt(max(m, n)) as the Muon / LionMuon runs in this
repo (instead of Dion's native sqrt(m/n)); non-2D parameters fall back to AdamW
exactly as in muon.py / lion_muon.py.
"""

import math

import torch


class Dion(torch.optim.Optimizer):
    def __init__(
        self,
        muon_params,
        lr=1e-3,
        rank_frac=0.25,
        ef_beta=0.05,
        weight_decay=0.0,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.8, 0.999),
        adamw_eps=1e-8,
        adamw_wd=0.1,
    ):
        defaults = dict(
            lr=lr,
            rank_frac=rank_frac,
            ef_beta=ef_beta,
            weight_decay=weight_decay,
            adamw_lr_ratio=adamw_lr / lr if lr > 0 else 1.0,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        for p in muon_params:
            self.state[p]["use_muon"] = p.ndim >= 2 and p.size(0) < 10000
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:

            ############################
            #      Dion (2D params)    #
            ############################

            lr = group["lr"]
            beta = group["ef_beta"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if not self.state[p]["use_muon"]:
                    continue
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                m_rows, n_cols = g.shape

                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(g, dtype=torch.float32)
                    r = max(1, int(round(group["rank_frac"] * min(m_rows, n_cols))))
                    v = torch.randn(n_cols, r, device=g.device, dtype=torch.float32)
                    state["V"], _ = torch.linalg.qr(v)

                M, V = state["momentum"], state["V"]
                M.add_(g.float())                       # accumulate gradient
                P = M @ V                               # m x r
                U, _ = torch.linalg.qr(P)               # orthonormal columns
                W = M.t() @ U                           # n x r
                M.sub_(U @ W.t(), alpha=beta)           # error feedback
                V.copy_(W / W.norm(dim=0, keepdim=True).clamp_min(1e-8))
                O = U @ V.t()                           # orthonormal update
                O *= 0.2 * math.sqrt(max(m_rows, n_cols))  # Moonshot RMS scaling

                if wd > 0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(O.view_as(p.data).type_as(p.data), alpha=-lr)

            ############################
            #       AdamW backup       #
            ############################

            adamw_lr = group["adamw_lr_ratio"] * lr
            beta1, beta2 = group["adamw_betas"]
            eps, awd = group["adamw_eps"], group["adamw_wd"]

            for p in group["params"]:
                if self.state[p]["use_muon"] or p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1, buf2 = state["moment1"], state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)
                g = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - adamw_lr * awd)
                p.data.add_(g, alpha=-adamw_lr / scale)
