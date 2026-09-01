"""MuonBP (arXiv:2510.16981), single-device version.

Newton-Schulz runs per column block on most steps (no cross-shard
communication) and on the full matrix every P-th step, with lr scaled by
block_lr_ratio on block steps. RMS scaling matches muon.py so tuned LRs
transfer; non-2D parameters fall back to AdamW.
"""

import math

import torch

from .muon import zeropower_via_newtonschulz5


class MuonBP(torch.optim.Optimizer):
    def __init__(
        self,
        muon_params,
        lr=1e-3,
        momentum=0.9,
        n_blocks=4,
        period=5,
        block_lr_ratio=0.5,
        ns_steps=5,
        weight_decay=0.0,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.8, 0.999),
        adamw_eps=1e-8,
        adamw_wd=0.1,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            n_blocks=n_blocks,
            period=period,
            block_lr_ratio=block_lr_ratio,
            ns_steps=ns_steps,
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

        self._step_count = 0

    @torch.no_grad()
    def step(self):
        is_full = (self._step_count % self.param_groups[0]["period"]) == 0
        self._step_count += 1

        for group in self.param_groups:

            ############################
            #    MuonBP (2D params)    #
            ############################

            beta = group["momentum"]
            lr = group["lr"] if is_full else group["block_lr_ratio"] * group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if not self.state[p]["use_muon"]:
                    continue
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(beta).add_(g, alpha=1 - beta)  # EMA momentum

                if is_full:
                    u = zeropower_via_newtonschulz5(buf, steps=group["ns_steps"])
                    u = u * (0.2 * math.sqrt(max(u.size(0), u.size(1))))
                else:
                    # column blocks = simulated TP shards, orthogonalized locally
                    chunks = torch.chunk(buf, group["n_blocks"], dim=1)
                    outs = []
                    for c in chunks:
                        oc = zeropower_via_newtonschulz5(c, steps=group["ns_steps"])
                        outs.append(oc * (0.2 * math.sqrt(max(oc.size(0), oc.size(1)))))
                    u = torch.cat(outs, dim=1)

                if wd > 0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(u.view_as(p.data).type_as(p.data), alpha=-lr)

            ############################
            #       AdamW backup       #
            ############################

            adamw_lr = group["adamw_lr_ratio"] * group["lr"]
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
