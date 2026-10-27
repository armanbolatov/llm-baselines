# What one step costs at 124M on four GPUs

Measured with `scripts/cost_bench.py` at the 124M DDP config (global batch 32×512, the flags of
`scripts/ddp_train_124m.sh`). Raw JSON in `results/benchmarks/cost/`. These are the per-step
numbers of the paper's cost table; the losses and the cost to reach Muon's loss are in
`results/ddp124/` and the paper's appendix tables.

## What each optimizer sends

DDP all-reduces the fp32 gradients, 124M values = 495.9 MB per step, bucket by bucket during the
backward pass, so most of it hides behind compute. What comes after the backward pass is exposed.

| optimizer | after the backward pass | why |
|---|---|---|
| AdamW, Lion, Signum | nothing | the update is elementwise, every rank computes it on its own |
| Muon | 170.7 MB bf16, every step | each matrix is orthogonalized on one rank and the updates are all-reduced (round-robin) |
| LionMuon / SignMuon, period P | 170.7 MB every P-th step | sign steps are local, only Muon steps pay the all-reduce |
| MuonBP, P = 5 | 170.7 MB every 5th step | block steps are local, the full step is Muon's |
| Dion, rank min(m,n)/4 | 268.5 MB fp32, every step, and no gradient sync | see below |

Dion turns DDP's gradient sync off for the whole model and all-reduces, per matrix, the factors
P = MV (m×r) and then W = MᵀU (n×r), with a QR in between. That is 28,475,392 fp32 values
(the 48 block matrices and the position embedding), 113.9 MB. The parameters it does not
factorize (the 50304×768 token embedding and the norms, 38,652,672 values) still need their
gradients averaged, which costs one more all-reduce of 154.6 MB inside the optimizer step.
microsoft/dion does the same in its data-parallel mode (`no_sync()` and
`replicate_mesh_grad_sync=True`). None of it can start before the backward pass ends, so all
268.5 MB is exposed. At rank min(m,n)/2 the factors double: 382.4 MB.

## Measured per step

| optimizer | optimizer FLOPs | bytes, total | exposed | time / Muon, PCIe | time / Muon, NVLink |
|---|---|---|---|---|---|
| AdamW | 0 | 495.9 | 0 | 0.748 | 0.846 |
| Muon | 1.527e12 | 666.6 | 170.7 | 1.000 | 1.000 |
| LionMuon P=2 | 7.637e11 | 581.3 | 85.3 | 0.880 | 0.924 |
| LionMuon P=5 | 3.055e11 | 530.1 | 34.1 | 0.807 | 0.889 |
| MuonBP P=5 | 1.011e12 | 530.1 | 34.1 | 0.937 | 1.254 |
| Dion, rank 1/4 | 1.309e11 | 268.5 | 268.5 | 0.756 | 1.502 |

- FLOPs: `torch.utils.flop_counter` on the code that runs, matmul-type ops, QR added for Dion.
  Forward and backward are 1.3230e13 per step for every method. `flops_124m.json`.
- Bytes: DDP buckets through a counting comm hook, optimizer all-reduces through a counting
  wrapper. Payload per call. Identical on both nodes. `comm_124m_4gpu.json` (PCIe),
  `comm_124m_4gpu_h200.json` and `comm_124m_4gpu_r2.json` (NVLink, the second with Dion at
  rank 1/2).
- Time: every round runs each method once, in a random order shared by all ranks, for 10 steps;
  the ratio to Muon is taken inside the round, so load from other jobs on a shared node falls on
  every method alike. 4×A100 PCIe with `NCCL_P2P_DISABLE=1`, 30 rounds, Muon's median step
  1.563 s (`time_124m_4xA100_pcie.json`); 4×H200 NVLink, 40 rounds, Muon's median step 0.242 s
  (`time_124m_4xH200_nvlink.json`). The 95% intervals are within ±0.02 on PCIe and ±0.09 on
  NVLink.

Dion and MuonBP are cheaper than Muon on PCIe and slower on NVLink: both trade bytes for extra
arithmetic (QR and power iteration, block Newton-Schulz on every step), which stops paying once
communication is cheap. A LionMuon step does less of both.

`ddp_bench_124m.json` is an earlier, uncontrolled step-time benchmark that the plotting script can
still read with `DDP_X=time`; the paper uses the interleaved measurement above.

## Reproduce

```
python scripts/cost_bench.py --mode flops --out results/benchmarks/cost/flops_124m.json
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=4 scripts/cost_bench.py --mode comm --out results/benchmarks/cost/comm_124m_4gpu.json
torchrun --nproc_per_node=4 scripts/cost_bench.py --mode time --rounds 30 --out results/benchmarks/cost/time_124m_4xA100_pcie.json
```
