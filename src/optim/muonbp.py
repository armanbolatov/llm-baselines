"""MuonBP (arXiv:2510.16981).

Newton-Schulz runs per column block on most steps (no cross-shard
communication) and on the full matrix every P-th step, with lr scaled by
block_lr_ratio on block steps. RMS scaling matches muon.py so tuned LRs
transfer; non-2D parameters fall back to AdamW.

Under DDP (WORLD_SIZE > 1) the momentum is replicated: block steps are
computed locally on every rank with no communication, and the full step
shards Newton-Schulz across ranks and all-reduces the update, exactly like
muon.py. This gives MuonBP the same communication profile as LionMuon at the
same period.
"""

import math
import os

import torch
import torch.distributed as dist

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
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        else:
            self.world_size = 1
            self.rank = 0

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

            params = [p for p in group["params"]
                      if self.state[p]["use_muon"] and p.grad is not None]
            # Full step under DDP: shard NS across ranks and all-reduce the
            # concatenated update (muon.py pattern). Block steps are local.
            shard = is_full and self.world_size > 1
            if shard:
                updates_flat = torch.zeros(
                    sum(p.numel() for p in params), device="cuda", dtype=torch.bfloat16
                )
            curr_idx = 0

            for i, p in enumerate(params):
                g = p.grad
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(beta).add_(g, alpha=1 - beta)  # EMA momentum (replicated)

                if shard:
                    if i % self.world_size == self.rank:
                        u = zeropower_via_newtonschulz5(buf, steps=group["ns_steps"])
                        u = u * (0.2 * math.sqrt(max(u.size(0), u.size(1))))
                        updates_flat[curr_idx : curr_idx + p.numel()] = u.flatten()
                    curr_idx += p.numel()
                    continue

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

            if shard:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
                curr_idx = 0
                for p in params:
                    u = updates_flat[curr_idx : curr_idx + p.numel()].view_as(p.data).type_as(p.data)
                    if wd > 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.add_(u, alpha=-lr)
                    curr_idx += p.numel()

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
