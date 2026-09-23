# Distributed communication comparison at 124M

**Dion / LionMuon / MuonBP (+ Muon, AdamW/Lion baselines) · DDP · 4× A100 80GB PCIe**

Companion to `scripts/comms_compare.py` (measurement), `results/benchmarks/comms_124m.json`
(raw runs), and `scripts/plotting/plot_comms.py` (figure). 2026-09-03.

## What each optimizer must communicate per DDP step

Every optimizer except Dion first pays **gradient sync**: DDP all-reduces all
124M gradients in fp32 (param dtype — `src/distributed/ddp.py` registers no
compression hook) = **496 MB/step**. On top of that:

| optimizer | extra communication | why |
|---|---|---|
| AdamW / Lion / Signum | none | update is elementwise, computed identically on every rank |
| Muon | 170 MB **every step** | Newton–Schulz is sharded across ranks (`muon.py:139`), so the orthogonalized 85M-param 2D update (`updates_flat`, bf16) is all-reduced back (`muon.py:158`) |
| LionMuon / SignMuon (period P) | 170 MB **every P-th step** | sign steps are elementwise → zero extra comms; only Muon steps pay the all-reduce → amortized 170/P MB |
| MuonBP (period P) | 170 MB **every P-th step** | block steps orthogonalize per-block with no cross-rank comms; the periodic full-matrix step pays Muon's all-reduce |
| Dion (rank r = min(m,n)/4) | 269 MB/step, **replacing** grad sync | syncs low-rank factors P (m×r) and W (n×r) per matrix instead of gradients; momentum stays local via error feedback. Σ(m+n)·r = 28.3M fp32 = 113 MB, **plus** one all-reduce of the non-matrix gradients (wte and the 1D parameters, 155 MB fp32) that `dion.py:141` also has to do: 113 + 155 = 269 MB, as measured |

Volumes are exact arithmetic at the paper's 124M config
(12 layers × [2304×768, 768×768, 3072×768, 768×3072]) — hardware-independent.

## Measured NCCL times (4× A100 80GB PCIe, `NCCL_P2P_DISABLE=1`)

Mean over 3 runs × 50 all-reduces (min–max in brackets); shared node, so
absolutes are conservative:

| primitive | volume | time |
|---|---|---|
| gradient sync (124M fp32) | 496 MB | 627.9 ms [603–654] |
| Muon update (85M bf16) | 170 MB | 228.3 ms [220–233] |
| Dion factors, fused | 113 MB | 144.3 ms [141–149] |
| Dion factors, naive 96 calls | 113 MB | 1012.7 ms [956–1057] |

## The comparison table (candidate for the paper)

Comms columns are exact (volume) and measured (time). **Exposed** = the
optimizer-step all-reduce that cannot overlap with backward (grad sync
overlaps via DDP bucketing; Dion's factor sync replaces grad sync but itself
sits on the critical path — see caveat 2). State = per-2D-param buffers.

The loss column is a **same-protocol** comparison: WikiText-103, 124M, one
machine (the 5090 box, `exps_124m/wt5090_*`), batch 32×1, seed 0, and the
same learning rate 1e-3 for every spectral method (Muon's tuned FineWeb value;
Dion/MuonBP use Muon's RMS scaling so its LR transfers). Dion and MuonBP were
not tuned beyond this, so their tuned numbers may improve.

| optimizer | val loss @ lr 1e-3 | comms MB/step | measured ms/step | exposed ms/step | optimizer state |
|---|---|---|---|---|---|
| Muon (P=1) | 2.885 | 666 | 856 | 228 | 1 buffer |
| **LionMuon P=2** | 2.886 | 581 | 742 | 114 | 1 buffer |
| **LionMuon P=5** | — | 530 | 674 | 46 | 1 buffer |
| **LionMuon P=20** | — | 505 | 639 | 11 | 1 buffer |
| MuonBP (P=5, block LR ×0.5 — superseded, see note) | 2.914 | 530 | 674 | 46 | 1 buffer |
| Dion (r=¼, fused) | 2.888 | 269 = 113 factors + 155 non-matrix | 144 (factors only — the 155 MB non-matrix all-reduce was not timed) | 144+ | 1 buffer + V (+15%) |
| AdamW | — | 496 | 628 | ~0 | 2 buffers (2×) |

Seed noise at this setting is ~0.003 (Muon seeds 2.883–2.885, LionMuon P=2
seeds 2.882–2.887), so Muon, LionMuon P=2 and Dion are tied; MuonBP trails by
0.03. With per-method tuning (main-machine grid, 64k steps, interior optima)
LionMuon P=5 reaches 2.825 vs 2.838 for Muon; Dion/MuonBP have no tuned
numbers yet (see "Open items").

**Headline (what the evidence supports):** LionMuon keeps Muon's quality at
1/P of its exposed communication, and improves on Muon's loss when both are
tuned (2.825 vs 2.838). MuonBP has the same communication profile as LionMuon
P=5 but 0.03 worse loss at equal LR. Dion also keeps Muon's quality and moves
the fewest bytes in total, but its factor sync is on the critical path, so
LionMuon P≥5 has 3× less exposed communication and keeps the exact gradient.
**Do not claim LionMuon beats Dion on loss** — at equal LR they tie.

### Open items

- **MuonBP correction (Sep 14):** the block-step LR multiplier 0.5 used for the 2.914 number
  is not in the MuonBP paper, which scales block updates by block dimensions only (our RMS
  rule already does that). Default is now 1.0; the number must be re-run before it is reported.

- Dion and MuonBP on **FineWeb** on the same 5090 box at lr 1e-3 (Muon's tuned
  FineWeb LR) and 3e-3: `bash scripts/run_experiments.sh extra fineweb 124m`
  gives the 1e-3 pair; a 3e-3 pair needs `--muon_lr_factor 3e-3`.
- A tuned WikiText number for Dion (3e-3, where Muon's optimum is) to close the
  tuned-vs-tuned comparison.

## End-to-end wall-clock under real 4-GPU DDP

`scripts/ddp_bench.sh` trains each optimizer for 400 steps with `torchrun`
on 4×A100 PCIe (`NCCL_P2P_DISABLE=1`, shared node) and takes the median
synchronized wall-clock per step over steps ≥100 (`wall=` stamp in the log,
which sits after `loss.item()` and therefore after all NCCL work). Two runs at
batch 32, one at batch 128. Raw: `results/benchmarks/ddp_bench*/`, parsed:
`results/benchmarks/ddp_bench_124m.json`, figure `results/figures/ddp_bench_124m.png`.

Distributed implementations used (all in `src/optim/`, active only when
`WORLD_SIZE>1`):
- **LionMuon / Muon**: momentum replicated on every rank; sign steps applied
  locally with no communication; Muon steps shard Newton–Schulz across ranks
  (`i % world_size == rank`) and all-reduce the concatenated bf16 update.
  *Before this change the repo all-reduced on every step, sign steps
  included, so LionMuon under DDP saved nothing — fixed in `lion_muon.py`.*
- **MuonBP**: block steps computed locally on every rank (all blocks); full
  step sharded + all-reduce, like Muon.
- **Dion**: DDP gradient sync disabled (`distributed_backend.skip_grad_sync`),
  local momentum, one fused all-reduce of all P = MV, QR, one fused
  all-reduce of all W = MᵀU, local error feedback; AdamW-parameter grads
  (embeddings, 1D — 154 MB fp32) averaged with one more fused all-reduce.

| optimizer | batch 32 (8/GPU) s/step | batch 128 (32/GPU) s/step |
|---|---|---|
| AdamW | 0.63 ± 0.05 | 0.75 |
| Muon (P=1) | 0.84 ± 0.06 | 0.96 |
| **LionMuon P=2** | 0.71 ± 0.03 | 0.86 |
| **LionMuon P=5** | **0.65** ± 0.01 | **0.75** |
| MuonBP (P=5) | 0.77 ± 0.00 | 0.90 |
| Dion (r=¼) | 0.66 ± 0.00 | 0.90 |

Reading: Muon costs +0.21 s/step over AdamW (= the update all-reduce);
LionMuon removes 60% of that at P=2 and 93% at P=5, where it trains as fast
as AdamW. MuonBP has LionMuon's communication profile but pays block-wise
Newton–Schulz on every step (+0.12 s). Dion is cheap at 8 seqs/GPU but its
factor sync cannot overlap with backward, so at 32 seqs/GPU it is as slow as
MuonBP (0.90 s) while LionMuon P=5 stays at AdamW's 0.75 s.

Caveats: shared node (± from other jobs ≈ 7–15%; run-to-run spread shown);
MuonBP in DDP computes all blocks on every rank (its native TP setting
computes one per rank); Dion's per-matrix QR is not batched. Neither caveat
changes the communication picture, only the compute part of those two bars.

## Fairness caveats (state these in the paper)

1. **Shared node, no P2P.** GPUs were 70–100% busy with other jobs and
   `NCCL_P2P_DISABLE=1` (no NVLink on this box). Absolute times are
   setup-specific; the claim rests on the exact volume column, with measured
   times as evidence the ratios survive real NCCL. Error bars over 3 runs.
2. **Overlap accounting.** DDP grad sync overlaps backward, so "exposed"
   counts only the optimizer-step all-reduce. Dion's factor syncs are
   counted fully exposed here; a pipelined implementation could overlap
   per-matrix factor syncs with backward the way DDP buckets gradients, which
   would shrink Dion's exposed cost — state this when comparing.
3. **Dion must be quoted fused.** The naive per-matrix pattern (96 small
   all-reduces) costs 1013 ms — pure launch/latency overhead that any real
   implementation buckets away. Footnote the naive number at most.
4. **Dion and MuonBP losses are single-device reference implementations**
   (`src/optim/dion.py`, `src/optim/muonbp.py`, no `dist.` code), seed 0,
   WikiText only, run on a different machine (`wt5090_` prefix) than the main
   table — cross-machine differences of ~0.03 are possible (see batch-split
   note in the experiment log). Their comms rows are analytic + primitive
   measurements, not end-to-end training. MuonBP's 0.089 gap exceeds that
   margin; Dion's 0.063 gap is close to it — soften the Dion loss claim
   accordingly or rerun both on the main machine.
5. **Dtypes follow the implementations**: fp32 grad sync (DDP default),
   bf16 `updates_flat` (`muon.py`/`lion_muon.py`), fp32 Dion factors
   (`dion.py`). A bf16 comm hook would halve the grad-sync bars for everyone
   equally.
6. **World size**: measured at 4 ranks. Ring all-reduce volume per rank is
   ~constant in world size (2·(N−1)/N · bytes), so 8-GPU numbers would be
   close; latency grows mildly.

## Reproduce

```bash
export NCCL_P2P_DISABLE=1   # no NVLink on this box
torchrun --nproc_per_node=4 scripts/comms_compare.py   # ~3 min
python scripts/plotting/plot_comms.py                  # reads comms_124m.json
```

## Table or plot?

**Table in the main text** — the comparison has five axes (loss, volume,
time, exposed time, memory) and the volume column is exact, which a table
conveys with authority; cost accounting is conventionally tabular. Keep
`results/figures/comms_124m.png` for the appendix/talks: it shows (left) the
exposed optimizer communication per step as plain bars and (right) WikiText-103
loss vs. that exposed communication, where LionMuon sits alone in the
lower-left corner.

## Measured per step, September 17 (`scripts/cost_bench.py`)

Everything below is counted or measured on the code that runs, at the 124M DDP config
(global batch 32×512, the flags of `scripts/ddp_train_124m.sh`). Raw JSON in
`results/benchmarks/cost/`. This replaces the earlier practice of quoting an *estimated*
training time: FLOPs and bytes are exact, and the time column is a ratio measured in
interleaved rounds, which is what survives a shared node.

**FLOPs per step** (`--mode flops`, `torch.utils.flop_counter`, matmul-class ops, QR added for
Dion; elementwise updates count zero by that convention, which is worth <0.1% here). The count
confirms the analytic Newton–Schulz term used in the paper's FLOPs formula: 1.527e12 measured
against 1.522e12 from the formula. Forward+backward is 1.3230e13 per step (6NT gives 1.2188e13;
the counter also sees attention and the head).

| optimizer | optimizer FLOPs/step | total FLOPs/step | over AdamW |
|---|---|---|---|
| AdamW | 0 | 1.3230e13 | — |
| Muon (P=1) | 1.527e12 | 1.4757e13 | +11.5% |
| LionMuon / SignMuon P=2 | 7.637e11 | 1.3993e13 | +5.8% |
| LionMuon P=5 | 3.055e11 | 1.3535e13 | +2.3% |
| MuonBP (P=5) | 1.011e12 | 1.4240e13 | +7.6% |
| Dion (r=¼) | 1.309e11 | 1.3360e13 | +1.0% |

**Communication per step** (`--mode comm`, 4 ranks: DDP gradient buckets through a counting comm
hook, optimizer all-reduces through a counting wrapper of `dist.all_reduce`). Payload per call; a
ring all-reduce puts 2(N−1)/N of it on the wire per rank.

| optimizer | gradient sync | optimizer | total MB/step |
|---|---|---|---|
| AdamW | 495.9 fp32, 13 buckets | — | 495.9 |
| Muon (P=1) | 495.9 | 170.7 bf16, every step | 666.6 |
| LionMuon / SignMuon P=2 | 495.9 | 85.3 (170.7 every 2nd) | 581.3 |
| LionMuon P=5 | 495.9 | 34.1 (170.7 every 5th) | 530.1 |
| MuonBP (P=5) | 495.9 | 34.1 | 530.1 |
| Dion (r=¼) | none (skipped) | 268.5 fp32, 3 calls | 268.5 |

Gradient sync is fp32 because the repo registers no compression hook; a bf16 hook would cut
495.9 to 248 MB for everyone except Dion, which raises the optimizer's share of the total
(Muon 418 vs LionMuon P=5 282 MB, i.e. 33% less instead of 20%).

**Step time** (`--mode time`, 4×A100 PCIe, `NCCL_P2P_DISABLE=1`, 30 rounds × 10 steps per method,
methods interleaved in a random order per round, slowest rank per block, ratios taken within a
round). The node was shared and much busier than on Sep 7, so the absolute medians are ~1.8× the
earlier `ddp_bench.sh` numbers while the **ratios agree within 4%** — that is the point of
interleaving, and the reason the paper should quote ratios.

| optimizer | s/step median [IQR] | vs Muon [95% CI] | vs AdamW [95% CI] |
|---|---|---|---|
| AdamW | 1.153 [1.137–1.199] | 0.748 [0.744, 0.755] | 1.000 |
| Muon (P=1) | 1.563 [1.510–1.600] | 1.000 | 1.336 [1.325, 1.344] |
| LionMuon P=2 | 1.362 [1.335–1.401] | 0.880 [0.878, 0.886] | 1.173 [1.165, 1.180] |
| LionMuon P=5 | 1.251 [1.223–1.291] | **0.807** [0.804, 0.815] | 1.079 [1.072, 1.087] |
| MuonBP (P=5) | 1.456 [1.423–1.483] | 0.937 [0.933, 0.943] | 1.249 [1.244, 1.259] |
| Dion (r=¼) | 1.180 [1.167–1.194] | 0.756 [0.751, 0.769] | 1.019 [1.005, 1.026] |

**Cost to reach Muon's final loss (3.418)**, steps from the 150k-step runs × the per-step numbers
above. Dion never reaches it (it ends at 3.429).

| optimizer | steps | FLOPs | communication | time vs Muon |
|---|---|---|---|---|
| Muon (P=1) | 147k | 2.17e18 | 98.0 TB | 1.000 |
| LionMuon P=2 | 126k | 1.76e18 (0.81×) | 73.2 TB (0.75×) | 0.754 |
| LionMuon P=5 | 127k | 1.72e18 (0.79×) | 67.3 TB (0.69×) | **0.697** |
| SignMuon P=2 | 136k | 1.90e18 (0.88×) | 79.1 TB (0.81×) | 0.814 |
| AdamW | 137k | 1.81e18 (0.84×) | 67.9 TB (0.69×) | 0.697 |
| MuonBP (P=5) | 142k | 2.02e18 (0.93×) | 75.3 TB (0.77×) | 0.905 |

Reading: against Muon, LionMuon P=5 needs 14% fewer steps, 21% fewer FLOPs, 31% less
communication and 30% less time, all measured. Against AdamW the time to this target is the same
(0.697 both) and the bytes are the same, so the claim there is not speed but the loss at equal
budget: 3.377 vs 3.407 at 150k steps. A hardware-independent floor for the comparison with Muon:
a LionMuon step does at most the work of a Muon step (the sign step replaces Newton-Schulz and
its all-reduce with an elementwise update, plus one elementwise interpolation), so the 14% step
saving holds on any interconnect, and the rest grows as communication gets relatively more
expensive.

### The same step time on NVLink (opt2, 4×H200, `NCCL_P2P_DISABLE=0`, 40 rounds)

Same script and design as the PCIe table above, on the machine where communication is cheap.
Raw JSON on opt2: `results/benchmarks/cost/time_124m_4xH200_nvlink.json`. The byte counts came
out **identical** to the PCIe run (495.9 / 666.6 / 581.3 / 530.1 / 530.1 / 268.5 MB per step),
which is the check that the volume column does not depend on the hardware.

| optimizer | s/step median [IQR] | vs Muon [95% CI] | vs Muon on PCIe |
|---|---|---|---|
| AdamW | 0.205 [0.187–0.225] | 0.846 [0.798, 0.907] | 0.748 |
| Muon (P=1) | 0.242 [0.235–0.260] | 1.000 | 1.000 |
| LionMuon P=2 | 0.224 [0.217–0.234] | 0.924 [0.895, 0.959] | 0.880 |
| LionMuon P=5 | 0.216 [0.194–0.234] | **0.889** [0.806, 0.924] | 0.807 |
| MuonBP (P=5) | 0.307 [0.301–0.322] | **1.254** [1.230, 1.292] | 0.937 |
| Dion (r=¼) | 0.367 [0.363–0.370] | **1.502** [1.444, 1.539] | 0.756 |

Two things to say in the paper, one of them in our favour and one against:

1. **Our saving shrinks but survives.** With NVLink, LionMuon P=5 saves 11% of Muon's step
   instead of 19%. Combined with the 14% fewer steps to Muon's final loss, the time to that loss
   is 0.77× Muon's on NVLink against 0.70× on PCIe. The interconnect sets the size of the prize,
   never its sign, because a LionMuon step never does more work than a Muon step.
2. **Dion and MuonBP are slower than Muon here** (1.50× and 1.25×), although both are cheaper
   than Muon on PCIe (0.76× and 0.94×). They buy fewer bytes with more arithmetic — Dion's
   per-matrix QR and power iteration, MuonBP's block Newton–Schulz on every step — so once
   communication is cheap the trade stops paying. LionMuon instead *removes* work: a sign step
   costs less than a Muon step in both arithmetic and bytes. That is the cleanest argument for
   the design, and it only became visible by measuring both interconnects.

Against AdamW on NVLink the comparison is a tie within the intervals (AdamW 0.846 per step,
0.79 to the target against our 0.77), so the claim against AdamW stays what it was: lower loss
at an equal token budget, not less time.
