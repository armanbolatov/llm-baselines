"""Dion (arXiv:2504.05295).

Per 2D parameter: momentum M, warm-started right vectors V. Each step does one
power iteration (U = QR(MV), W = M^T U), error feedback M -= 0.05 * U W^T, and
applies U V^T scaled by 0.2 * sqrt(max(m, n)) so tuned Muon LRs transfer.
Non-2D parameters fall back to AdamW as in muon.py.

Under DDP (WORLD_SIZE > 1) this is Dion's data-parallel algorithm: gradient
synchronization is disabled (main.py sets distributed_backend.skip_grad_sync),
each rank keeps a local momentum M_i of its own gradients, and only the
low-rank factors are averaged across ranks: one fused all-reduce of all
P_i = M_i V, then QR, then one fused all-reduce of all W_i = M_i^T U. Error
feedback uses the local W_i, so the mean of the local momenta follows the
single-device recursion exactly. Gradients of the AdamW parameters (embeddings,
1D) are averaged with one additional fused all-reduce.
"""

import math
import os

import torch
import torch.distributed as dist


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

        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        else:
            self.world_size = 1
            self.rank = 0

    @staticmethod
    def _fused_all_reduce_mean(tensors, world_size):
        """One all-reduce for a list of tensors; returns them averaged."""
        flat = torch.cat([t.reshape(-1) for t in tensors])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        out, idx = [], 0
        for t in tensors:
            out.append(flat[idx : idx + t.numel()].view_as(t))
            idx += t.numel()
        return out

    @torch.no_grad()
    def step(self):
        ws = self.world_size
        for group in self.param_groups:

            ############################
            #      Dion (2D params)    #
            ############################

            lr = group["lr"]
            beta = group["ef_beta"]
            wd = group["weight_decay"]

            params = [p for p in group["params"]
                      if self.state[p]["use_muon"] and p.grad is not None]
            Ms, Vs, Ps = [], [], []
            for p in params:
                g = p.grad
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                m_rows, n_cols = g.shape

                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(g, dtype=torch.float32)
                    r = max(1, int(round(group["rank_frac"] * min(m_rows, n_cols))))
                    gen = torch.Generator(device=g.device).manual_seed(0)  # same V on all ranks
                    v = torch.randn(n_cols, r, device=g.device, dtype=torch.float32, generator=gen)
                    state["V"], _ = torch.linalg.qr(v)

                M, V = state["momentum"], state["V"]
                M.add_(g.float())                       # accumulate (local) gradient
                Ms.append(M); Vs.append(V); Ps.append(M @ V)   # m x r

            if ws > 1:
                Ps = self._fused_all_reduce_mean(Ps, ws)   # P = mean_i M_i V
            Us = [torch.linalg.qr(P)[0] for P in Ps]       # orthonormal columns
            Ws_local = [M.t() @ U for M, U in zip(Ms, Us)]  # n x r (local)
            Ws = self._fused_all_reduce_mean(Ws_local, ws) if ws > 1 else Ws_local

            for p, M, V, U, W_loc, W in zip(params, Ms, Vs, Us, Ws_local, Ws):
                m_rows, n_cols = M.shape
                M.sub_(U @ W_loc.t(), alpha=beta)       # error feedback (local W_i)
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

            aw_params = [p for p in group["params"]
                         if not self.state[p]["use_muon"] and p.grad is not None]
            if ws > 1:
                # DDP grad sync is disabled for Dion: average these grads here.
                for p, g in zip(aw_params, self._fused_all_reduce_mean([p.grad for p in aw_params], ws)):
                    p.grad.copy_(g)

            for p in aw_params:
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
