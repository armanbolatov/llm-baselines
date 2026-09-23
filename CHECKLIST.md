# LionMuon — ICLR 2027 checklist

One list, every point raised by a reviewer. Source in brackets.
Reviewers: **GKBV** (3), **aN6P** (3), **NQtP** (2, conf. 5), **docx** (fourth review).
Deadlines: abstract **Sep 18**, paper **Sep 25**.

Legend: ✅ done · 🔄 running · ⏳ queued · ❌ not started

---

## Experiments

### Headline: 124M at 1× Chinchilla under real 4-GPU DDP (opt2, 4×H200 NVLink)
Communication efficiency is the main claim now. Loss vs wall-clock, end to end,
2.46B tokens (150k steps × 32 × 512), 124M-tuned hyperparameters, split 4×8.
Methods run round-robin on the same GPUs (2,3,5,7) in 2 h slices so shared-node load
averages out; a slice that dies within 2 min (OOM, port clash) backs off 5 min.
AdamW 0.11 s/step, Muon 0.17 → 5–7 h per method. Launched Sep 14 10:48 UTC, restarted 13:17
with the 2 h slices. Dion and MuonBP kill/resume verified on Sep 14.
- ✅ AdamW — **3.407** (Sep 15) [aN6P W2, GKBV W5]
- ✅ Muon — **3.418** (Sep 15); note AdamW beats Muon at this horizon with the 64k-step LRs
- ✅ LionMuon P=5 — **3.377** (Sep 15): 0.041 below Muon, 0.030 below AdamW
- ✅ LionMuon P=2 — **3.376** (best of all)
- ✅ SignMuon P=2 — **3.395**
- ✅ Decision (Sep 16, agreed): the shared H200 node's own step times measure the neighbours,
  not the optimizer (LionMuon P=2 hit a 0.70 s/step slot, P=5 0.18). The figure therefore
  plots loss vs tokens (opt2 runs) and loss vs wall-clock = iterations × the controlled PCIe
  benchmark step time (opt). Stated plainly in Section 5.1. Iterations to reach Muon's final
  3.418 (first eval at or below it, evals every 1k, Sep 17): LionMuon P=2 126k, P=5 127k,
  SignMuon P=2 136k, AdamW 137k, MuonBP 142k, Muon 147k, Dion never. Node step times per 2 h
  slice vary 2-3x within one method (Muon 0.09-0.19 s, LionMuon P=2 0.04-0.13 s), so they are
  not usable.
- ✅ MuonBP P=5 at LR 1e-3 — **3.414**, level with Muon, 32 h against Muon's 35 h [GKBV W5, aN6P W5]
- ✅ Dion 1e-3 — **3.429**, 3e-3 — **3.451** (both finished Sep 17 13:14 UTC). Lower LR is
  better, so Dion's optimum is at or below 1e-3. Dion 1e-3 sits 0.011 above Muon and 0.052
  above LionMuon P=5. `results/ddp124/final_losses.json` updated; figure redrawn (baselines at their best LR,
  time axis labelled as an estimate).
- ✅ Headline figure now has an exact x axis (Sep 17 evening): the right panel plots loss against
  **measured communication** (`DDP_X=comm`, TB for the 2.46B-token run), with a counted-FLOPs
  variant in `loss_vs_flops_355m_ddp.png` for the appendix. LionMuon P=5 dominates Muon
  (79.5 TB / 3.377 against 100.0 TB / 3.418), MuonBP (same 79.5 TB, 3.414) and SignMuon
  (87.2 TB, 3.395). **Honest reading of the rest:** AdamW sits at 74.4 TB / 3.407, so against
  AdamW we trade 7% more bytes for 0.030 lower loss, not a cost win; Dion moves the fewest bytes
  by far (40.3 TB) and has the worst loss (3.429). The time axis is gone from the main figure.
- 🛑 **The DDP campaign was stopped on Sep 18 11:55 UTC** to keep our share of opt2 at two GPUs.
  Seed 0 of every method is complete, so the comparison stands; what is lost is the low-rate check
  for the two external baselines — Dion 3e-4 (37k/150k) and MuonBP 3e-4 (35k/150k) — and Muon 3e-3
  (96k/150k). Checkpoints for all three are kept, so they can resume if four GPUs ever free up.
  Dion is therefore reported as the better of the two rates that finished (1e-3, 3e-3), which the
  text now says. Seeds 1--2 were dropped by decision, not by failure
- 🔄 Queue replaced Sep 17 18:30 UTC (`ddp124_queue3_opt2.sh`; queue2 driver and the slice loop
  stopped, the running Muon 3e-3 slice left to finish at 84.8k and resumed later as step 4).
  Order: 0a NVLink step time (interleaved, 40 rounds), 0b NVLink byte count (cross-machine
  check), 0c re-score every finished run on 4.1M val tokens (only 4 of the 8 finished runs still
  have a checkpoint), then 1 Muon 3e-4, 2 Dion 3e-4 + MuonBP 3e-4, 3 seed 1, 4 Muon 3e-3 finish,
  5 seed 2. Replaces the plan below:
  1. Muon at LR 3e-3, the transfer check — 🔄 57.8k/150k at 16:40 UTC, 0.18 s/step, done
     ~22:00 UTC. **Worse than Muon 1e-3 at every eval** (57k: 3.960 vs 3.818; gap steady at
     0.14 since 30k), so 1e-3 is not too low at this horizon;
  2. seed 1 of the five core methods (~35 h, → Sep 19 morning) [GKBV W2];
  3. Dion 3e-4 + MuonBP 3e-3 (~21 h, → Sep 20). **Dion 1e-2 dropped**: the sweep shows the
     optimum is below 1e-3, so 1e-2 was the wrong direction;
  4. seed 2 of the five core methods (~35 h, → Sep 21) if the freeze allows.
  A run short of the full token budget is left out of the figure (it would squash the axis);
  `scripts/collect_ddp124.sh` regenerates `results/ddp124/final_losses.json` on the node and
  the figure picks up each run as its `summary.json` appears.
- ✅ MuonBP re-checked against the paper: block-step LR = RMS scaling by block dims only;
  our extra ×0.5 on block steps was not in the paper — removed (default 1.0). The old
  WikiText number 2.914 was with 0.5 and must not be reported.
- ✅ Dion re-checked against the paper: momentum accumulate, QR power iteration, error
  feedback 0.05, column-normalized V, factor all-reduce with local error feedback, AdamW for
  non-matrix — all match. LR convention is Muon's RMS scaling (paper: √(d_out/d_in)); hence the sweep.
- ❌ Same four core methods on opt (4×A100 PCIe, where comms are exposed) — needs all
  4 GPUs on opt for ~5 days; decide
- ✅ Step-time benchmark on NVLink (opt2, 4×H200, 40 interleaved rounds, Sep 17 19:00 UTC).
  Ratios to Muon: AdamW 0.846, LionMuon P=2 0.924, P=5 **0.889**, MuonBP **1.254**, Dion
  **1.502**. Bytes per step came out identical to the PCIe run, which is the hardware-independence
  check. Two conclusions, table and wording in `README_comms_124m.md`: (a) our saving shrinks
  from 19% to 11% per step, so time to Muon's loss is 0.77× instead of 0.70×, and the
  interconnect sets the size of the prize but not its sign; (b) **Dion and MuonBP are slower than
  Muon on NVLink** although both beat it on PCIe — they buy bytes with arithmetic, while a sign
  step removes work of both kinds. That is the cleanest argument for the design and it only
  showed up because both interconnects were measured
- ✅ **Muon's LR at this horizon is interior, so the baseline is not under-tuned** [GKBV W5]:
  1e-3 gives 3.418, 3e-4 gives **3.467** (finished Sep 18 04:06 UTC), 3e-3 is far worse (3.756 at
  96k, still running as queue step 4). This closes the "Muon might just need a different LR over
  150k steps" objection in the direction that helps us
- ✅ **The 131k-token eval subset is representative: the full split is harder by a uniform
  +0.008** (done Sep 18 05:34 UTC, `results/ddp124/fullval.json`). Every finished run that still
  had a checkpoint, re-scored on 4.1M tokens of the same split with the same loss computation:
  MuonBP 1e-3 3.4143 → 3.4223, Dion 1e-3 3.4287 → 3.4371, Dion 3e-3 3.4505 → 3.4581, Muon 3e-4
  3.4672 → 3.4730. The shift is +0.006 to +0.008 for four different models, so **the margins in
  the paper are unaffected** and only the absolute level moves. Together with the GPT-2 anchor
  (3.5626 full vs 3.5652 on the subset) this closes the "your eval set is tiny" objection.
  Validation that the pipeline is right: Dion's checkpoint reproduces its reported 3.4287 exactly
  on the same windows. The first attempt had failed on every run — it asked the model for logits
  without targets, so it got the last position only.
  Note: the five headline runs (adamw, muon, lionmuon_p2, lionmuon_p5, signmuon_p2) no longer have
  checkpoints, so they are re-scored from **seed 1** instead (queue step 3, ~Sep 20); with a
  uniform offset this is now confirmation rather than a dependency
- ✅ Figures refreshed Sep 17, naming unified as `<metric>_<setting>.png`
  (`results/figures/README.md` lists every figure and the script that makes it):
  `loss_vs_flops_124m.png` (tuned grid, 3 seeds) and the new two-panel
  `loss_vs_cost_355m_ddp.png` — 355M vs FLOPs on the left, 124M under 4-GPU DDP vs
  wall-clock on the right, replacing the 720M panel. Right panel: LionMuon P=5 at
  27 h / 3.377 against Muon at 35 h / 3.418, i.e. below it on both axes.
  Three bugs fixed while doing it: `plot_scaling.py` averaged an in-progress seed into a
  finished method and counted the same run on two machines as two seeds;
  `collect_scaling.sh` read the whole queue log instead of the section of the run it was
  reporting.
- ✅ Plot script `scripts/plotting/plot_ddp124.py` (tested on a synthetic kill/resume log):
  x-axis = iterations × the method's median s/step over all its slices (default), or the
  actual summed training time (`--time actual`); the summary table has both. Waiting for
  the logs (opt2 link drops after one connection per check).

### Scale
- ✅ 355M SignMuon P=2 — **3.008** [GKBV W1, aN6P W1]
- ✅ 355M LionMuon P=2 — **3.020** [GKBV W1, aN6P W1]
- ✅ 355M LionMuon P=5 — **3.023** [NQtP W3]
- ✅ 355M Muon — **3.039** (opt2) / **3.040** (opt), machines agree to 0.001 [GKBV W1]
- ✅ 355M AdamW — **3.064**
- ✅ 720M Muon — **2.930**, 720M LionMuon P=2 — **2.945** (single seed, Muon ahead by 0.015) [GKBV W1]
- 🛑 720M SignMuon P=2 (10700/14000) and AdamW (9750/14000) — stopped Sep 14, 720M dropped.
  **Checkpoints deleted Sep 19 07:39 UTC** (approved) to free disk: all four `exps/fw720*/ckpts`,
  25 GB. The two finished runs keep their `summary.json`, so Muon 2.930 and LionMuon P=2 2.945
  survive; the two stopped runs had no summary, so they can no longer be resumed — 720M is now
  final at one seed for two methods
- ❌ 720M at full Chinchilla instead of 1/4 — **superseded**: the rerun is at 1.00× (20 TPP) [NQtP]

All scale runs transfer the 124M optimum (η_M, α, β₁, β₂) verbatim; no re-tuning.
355M is 23 TPP (1.16× Chinchilla), 720M is 20 TPP (1.00×).

### Seeds and error bars
- 📌 **Decision Sep 18: the 124M DDP campaign stays at one seed** so the freed time goes to
  experiments the table is missing. Error bars for the method comparison come from the tuned 124M
  grid (3 seeds, spread 0.002) and from 355M; the DDP section says single seed plainly
- 🔄 **355M SignMuon P=5 seed 0** (opt2 GPU 0, started Sep 18 11:33 UTC) — the missing row of the
  355M table, η_M 1e-2 / α 300 transferred from 124M. 98.7% at Sep 20 11:11 → **~11:58 UTC**
- 🔄 **355M LionMuon P=2 seed 1** (opt2 GPU 6, started Sep 18 11:33 UTC) — error bar for the
  headline method at scale; seed 0 ran on opt2, so the seed runs there too. 97.4% → **~12:20 UTC**
- ⏳ **355M Lion and Signum, seed 0** — queued on opt2 (`queue_baselines355.sh`, Sep 20 11:00
  UTC) to start on GPUs 0 and 6 the moment the two runs above finish, lr 3e-4 transferred from
  124M (rows added to the seeds table in `run_experiments.sh`). These are the baselines the 124M
  comparison has and 355M lacked, and the reason the abstract's "beats Lion and Signum" had to
  be limited to 124M. ~40–48 h each → **~Sep 22 12:00 UTC**. Not fitting before the freeze with
  two GPUs: P=20 and LionMuon P=1 at 355M
- **BUG. 355M Lion and Signum, first attempt: invalid, rerunning.** The rows I added to the
  seeds table used `-` in the alpha column, so `run()` passed no `--sign_lr`, and `lion_muon.py`
  silently falls back to `lr/100`. Both runs trained their sign step at 3e-6 instead of 3e-4 and
  came out at 3.6864 and 3.7298. Fixed: alpha is `1` in both pure-sign rows, the docstring now
  states the real fallback, and the two run directories are renamed `BADLR_*` so the `fw355t_*`
  glob cannot pick them up. Relaunched on opt2 GPUs 0 and 6 at Sep 23 09:20 UTC, ~40 h, so
  **~Sep 25 01:00 UTC**. Rule: never let a run that takes sign steps leave `--sign_lr` unset.
- **BUG. 124M DDP Lion is unstable, not misconfigured.** Its flags do carry `--sign_lr 3e-4`.
  The curve spikes (3.836 at 40k, 4.477 at 80k, 3.895 at 139k) while signum, signmuon_p5 and
  adamw descend monotonically, so the 64k rate does not survive 150k steps at the DDP batch.
  A `lion_1e-4` row is in `ddp_train_124m.sh`; run it on the four GPUs the campaign frees and
  report Lion at the better of the two rates.
- ⚠️ opt2 `/data` is **99% full (213 GB left)** and every GPU is 80–95% occupied by other jobs;
  the two 355M runs took the last ~29 GB of GPUs 0 and 6. Watch for OOM and for the disk
- ✅ 124M: 16/16 configs × 3 seeds, all optima interior [GKBV W2, docx]
- ✅ 355M Muon seed 1 — **3.037** (seed 0: 3.040) [GKBV W2]
- ✅ 355M SignMuon P=2 seed 1 — **3.0092** (finished Sep 19 ~07:00 UTC; seed 0 is 3.0076, so the
  seed spread at 355M is **0.0016**, against a 0.032 gap to Muon) [GKBV W2]
- 🔄 355M Muon seed 2 — **opt rebooted ~19:00 UTC Sep 19** and killed it at 8700 with its
  launcher. Resumed Sep 20 10:35 UTC from the 8500 checkpoint on GPU 2 (eval 3.166 there, seed 0
  had 3.168), so ~200 iterations lost. 7150 left at ~21.5 s/it → **~Sep 22 05:00 UTC**
- ⏳ 355M SignMuon P=2 seed 2 — also killed by the reboot at 2000 (checkpoint intact). Its retry
  launcher is polling GPU 1 for 55 GB (44.6 free at 10:35) and will start on its own; ~83 h from
  the start → Sep 24 at best. Lowest-value run we have; let it go if the deadline needs the GPU
- ⚠️ opt2 `/data` hit **100% (61 GB free)** on Sep 20 10:34 UTC while both 355M runs there were
  at 96–98%. Our whole footprint is 20 GB (7.7 GB of DDP checkpoints, 12 GB of exps), so deleting
  ours would not help; the growth is other users. A failed final checkpoint write would cost at
  most 250 iterations, and the run's summary.json is a few KB
- ⏳ 355M SignMuon P=2 seed 2 — queued behind seed 1, so it starts ~Sep 19 17:00 and needs ~4
  days at the current speed → **~Sep 23**, which now fits before the freeze if opt stays this
  fast (it sped up from 34 to 22 s/step when the node emptied; at 34 s/step it misses)
- ❌ 355M seeds for LionMuon P=2 (must run on opt2) — after the DDP campaign, if time allows
- 🔄 Baseline error bars at 124M, **WikiText half** running on q95 (the 5090 box whose grid made
  seed 0) since Sep 22 12:20 UTC: AdamW 3e-3 seeds 1–2, Lion 3e-4 seeds 1–2, Signum 1e-3 seeds
  1–2, ~1.3 h each → done ~Sep 22 19:00 UTC. Code there synced to ours (backup in
  `~/lion-muon_backup_0922_1219`)
- ❌ Baseline error bars at 124M, **FineWeb half**: seed 0 of the FineWeb 64k grid ran on the
  Windows box, not q95, so these 6 runs need that machine or a stated cross-machine caveat
  (Muon agreed within 0.005 across the 5090 and the 100BT box)
- 🔄 **124M DDP baselines** — Lion, Signum, LionMuon P=1, SignMuon P=5 at their 64k optima, queued
  on opt2 (`ddp124_baselines_opt2.sh`) to start on four free GPUs once Signum-355M lands (~11:00
  UTC Sep 22); ~37 h round-robin → ~Sep 24 00:00 UTC. Five GPUs on opt2 until Lion-355M ends,
  approved by the user

Seeds must run on the machine and batch split of their seed 0.

### Efficiency and distributed
- ✅ **Measured cost per step, Sep 17 (`scripts/cost_bench.py`, JSON in `results/benchmarks/cost/`)**
  — replaces the "estimated time" axis with exact quantities [aN6P W2, GKBV W5]:
  - FLOPs counted on the running code (`torch.utils.flop_counter`): fwd+bwd 1.3230e13/step;
    optimizer AdamW 0, Muon 1.53e12, LionMuon P=2 7.6e11, P=5 3.1e11, MuonBP 1.01e12, Dion 1.3e11.
    Confirms the paper's formula (NS term 1.527e12 measured vs 1.522e12 analytic).
  - Communication counted at every collective: AdamW 495.9 MB/step (fp32 grad sync, 13 buckets),
    Muon 666.6, LionMuon P=2 581.3, P=5 530.1, MuonBP P=5 530.1, Dion 268.5 (no grad sync).
    **Correction: Dion is 269 MB/step, not 113** — it also all-reduces its non-matrix gradients
    (155 MB); `README_comms_124m.md` fixed.
  - Step time, 30 interleaved rounds on 4×A100 PCIe: ratios to Muon are AdamW 0.748
    [0.744, 0.755], LionMuon P=2 0.880 [0.878, 0.886], P=5 0.807 [0.804, 0.815], MuonBP 0.937,
    Dion 0.756. Absolute medians (1.15–1.56 s/step) are ~1.8× the Sep 7 run because the node was
    busier, while the ratios agree within 4% — quote ratios, not absolutes.
  - Cost to reach Muon's 3.418: LionMuon P=5 needs 0.86× the steps, 0.79× FLOPs, 0.69× bytes,
    0.70× time. Against AdamW the time and bytes tie, so the claim there is loss at equal budget.
- ✅ Loss level sanity-checked against the literature (Sep 17): public GPT-2 124M on our val file
  scores 3.5626 (seq 512, full 10.4M tokens) / 3.4759 (seq 1024); our eval subset of 256 windows
  gives 3.5652, i.e. the subset is **not** easier (+0.003). Our AdamW 3.407 beats the GPT-2
  checkpoint by 0.156 with 2.46B tokens, where llm.c's reproduction beats it by ~0.2 with 10B, so
  the level is in line. Note the DDP runs' evals score 131k tokens and the grid runs' 524k
  **different** tokens — never compare those absolutes across settings [`scripts/eval_val_anchor.py`]
- ✅ End-to-end 4-GPU DDP wall-clock, 124M, 400 steps: AdamW 0.63 s/step, Muon 0.84,
  LionMuon P=2 0.71, P=5 0.65, MuonBP 0.77, Dion 0.66 [GKBV W5, aN6P W2]
- ✅ Distributed Dion and MuonBP implemented and measured [GKBV W5, aN6P W5]
- ✅ Isolated NCCL primitives: grad sync 628 ms, Muon update 228 ms, Dion factors 144 ms
- ✅ Per-step GPU time and peak memory at 124M (4090): Muon step 28.5 ms, sign 9 ms,
  14.1 GB vs AdamW 14.5 GB [aN6P W2]
- ✅ Dion / MuonBP loss at one shared LR, FineWeb 124M: Muon 3.527, LionMuon P=2 3.498,
  Dion 3.550, MuonBP 3.546 [GKBV W5]

### Paper edits, Sep 18 (working copy in the session scratchpad, `iclr/iclr2027.tex`)
- ✅ **The tuning appendix contradicted the main text and is rewritten.** Section 5.1 said every
  optimizer was tuned on the full 64k horizon, while Appendix C still described 3k-step *pilot*
  runs on a **LLaMA** model and listed selected pairs — (3e-3, 2e-5) and the like — that **no
  current run uses**. That is the mismatch NQtP raised. It now describes the real protocol, gives
  the swept ranges, the chosen cell and its neighbours as two tables (`tab:grid-fw`, `tab:grid-wt`,
  from `scripts/plotting/grid_table.py`), and says plainly why a cheap pilot was abandoned [GKBV
  minor, NQtP]
- ✅ Heatmap figure dropped (the 64k sweep is a diagonal band with pruned cells, so a grid figure
  advertises cells that were never run); the tables carry it instead
- ✅ Step-time figure dropped. `tab:cost` now reports step time **relative to Muon with 95% CIs on
  both interconnects** instead of PCIe seconds, and the Dion row is corrected to 269 MB
- ✅ Section 5 rewritten around measured bytes: the right panel is MB/step, the cost-to-target
  paragraph uses the measured steps (127k vs 147k), and the Dion/MuonBP inversion on NVLink is
  stated [aN6P W2, GKBV W5]
- ✅ "AdamW ends below Muon at this horizon" is no longer a guess: the LR check (3e-4 → 3.467)
  shows Muon's rate is interior, so AdamW's win at 2.46B tokens is real
- ✅ P recipe corrected: α is 10 → 30 → **100** at P=2/5/20 (the text said 300)
- ❌ Page count: no LaTeX locally or on opt, so the 9-page limit has to be checked in Overleaf
- ❌ Abstract still the old one; swap in whichever of the three new versions is chosen

### Ablations and tuning
- ✅ Nesterov-Muon baseline: 2.8895 vs plain 2.8845, does not help [NQtP W1]
- ✅ Matched beta plane, FineWeb P=2, each pair tuned separately: best off-diagonal
  (0.9, 0.99) 3.5006 beats best diagonal (0.95, 0.95) 3.5092; all three off-diagonal
  beat all three diagonal [NQtP after rebuttal, docx]
- ✅ Full-budget LR grid for every optimizer; the 3k-step pilot mis-ranks LRs [GKBV minor]
- ✅ AdamW betas (0.8, 0.999) vs (0.9, 0.95): the default costs 0.17 [NQtP question]
- ✅ Tail index estimated, κ = 2 at batch 32 [GKBV minor]
- ✅ Dense-approximation check: ‖U‖₂/‖U‖∞ ≈ 410 (FineWeb), 520 (WikiText) vs worst case 768 [GKBV W3]
- ❌ Trajectory-curvature smoothness validation instead of G_{t+1} − G_t [NQtP W2]
- ❌ 1D parameters with Lion instead of AdamW [NQtP W4]
- ❌ Probabilistic P instead of periodic [NQtP question]

---

## Text

- ✅ ICLR 2027 template: `iclr2027.tex` in Overleaf, style files added, reproducibility and
  AI-use statements in place. Main text ends on page 9; the headline figure will push it
  to ~9.4, so ~0.4 page still has to go once the figure is in.
- ✅ Reframed around communication efficiency (Sep 14): new abstract, contributions,
  Section 3.1 "Cost of a step" with the volume/time table, Section 5 rebuilt as
  DDP wall-clock → 124M loss vs FLOPs (3 seeds) → 355M table; 720M and the six-panel
  pilot-tuned grid moved to the appendix; constants figure moved to the appendix.
  Red `\pending{}` marks = numbers waiting for the opt2 campaign.
- ✅ Scaling claim rewritten to the numbers: alternating variants beat Muon at 124M and
  355M; 720M single seed, Muon ahead by 0.015, reported in Appendix as a limitation
  [GKBV W1, aN6P W1]
- ✅ Practical recipe for P written into Section 5.3 [NQtP W3]
- ✅ Fine-tuning-style honesty: limitations sentence in the conclusion (720M, exact-NS
  analysis, fixed P)
- ❌ Fill `\pending{}` items when waves 1–5 finish; then trim to 9 pages
- ❌ Practical recipe for P: flat between P=2 and P=5, degrades at P=20; tuned α grows
  with P (3 → 300); P=5 costs 9% fewer FLOPs than P=1 [NQtP W3]
- ❌ State that the clean bound uses the dense approximation, and report how close it is [GKBV W3]
- ❌ Say what the metric guarantees for the Lion iterates, not just the Muon subsequence
  [GKBV W4, docx]
- ❌ Corollary 1 collapses LionMuon into SignMuon as ε → 0; answer with the beta plane [docx]
- ❌ Theorem allows momentum values where terms are undefined; the proof needs a stronger
  conditional-noise assumption than stated [aN6P W3]
- ❌ Analysis treats the Muon step as exact; Newton–Schulz error is not modelled [aN6P W4]
- ❌ Justify Assumption 3, or show it is not needed [NQtP, docx]
- ❌ Explain how the theory covers the Lion and Muon boundary cases; compare with prior
  work in an appendix [NQtP]
- ❌ Explain why the theory cannot give two learning rates; the tuned ratio is not fixed,
  it runs 3 → 300 and grows with P [NQtP]
- ❌ Narrow the claim "no concentration results exist in other norms" — Hübler et al.
  (Schatten) and Yu et al. (ℓ1, nuclear) exist [NQtP after rebuttal]
- ❌ Compare the convergence rate with Lion and Muon under the same settings [NQtP]
- ❌ Report fine-tuning as a limitation: AdamW wins fine-tuning even from a LionMuon
  checkpoint (3.2241 vs 3.2766); the pretraining edge shrinks 0.095 → 0.065
- ❌ Redraw Figure 5 with the wider grid (η_M 1e-4…1e-1, α 1…3000); rename k to P [docx]

### Typos
- ✅ "Figures 4 to 3" — no longer present after the rewrite [GKBV]
- ✅ Table 1 lists Muon under SignMuon — table now reads correctly [GKBV]
- ❌ ρ barred and unbarred in Theorem 1 — theorem now uses only unbarred ρ; check the
  proof in the appendix once more [GKBV]
- ✅ Frobenius norm ⟨X,Y⟩ → ⟨X,X⟩ [docx]
- ✅ Lemma 1 (A1): η_t restored on the gradient-noise term in the statement (the proof had it) [docx]
- ✅ Hübler duplicate — only one entry in references.bib now [docx]
- ✅ Newton–Schulz with an en dash everywhere; hyperlinks compile without warnings [NQtP]

---

## Machines

`opt` (4×A100 PCIe, shared, 2 GPUs for 355M seeds) and `opt2` (8×H200 NVLink, shared;
GPUs 1,2,3,7 for the DDP campaign, ~7 GB each).
355M: 31 h per run on opt2, 58–72 h on opt. 124M DDP: ~8.5 h per method on opt2.
Splits: 355M is 16×32, 720M is 32×62, DDP 124M is 4×8 — never mix splits or machines
within a comparison.
opt2 is reachable only through the Windows OpenSSH client, not Git Bash.
Checkpoint saves are atomic now (tmp + rename), so killing a run mid-save is safe.
