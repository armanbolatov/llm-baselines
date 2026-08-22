"""2-GPU NCCL benchmark of Muon's optimizer all-reduce (updates_flat pattern).
Measures ms per all_reduce of the 2D-param flat vector (85M bf16, as in muon.py).
Run: torchrun --nproc_per_node=2 scripts/comms_bench.py"""
import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank = dist.get_rank(); torch.cuda.set_device(rank)
n = 85_000_000  # ~2D params of the 124M model (muon-updated matrices)
x = torch.randn(n, device=f"cuda:{rank}", dtype=torch.bfloat16)
for _ in range(10): dist.all_reduce(x)
torch.cuda.synchronize()
t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
t0.record()
for _ in range(100): dist.all_reduce(x)
t1.record(); torch.cuda.synchronize()
if rank == 0:
    ms = t0.elapsed_time(t1) / 100
    print(f"all_reduce 85M bf16 (170MB): {ms:.2f} ms per call")
    print(f"comms/step: Muon(P=1)={ms:.2f} ms, LionMuon/SignMuon P=2={ms/2:.2f} ms avg, Lion=0 ms")
dist.destroy_process_group()
