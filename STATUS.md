# LionMuon: reviewer objections and what we have

Four reviews: GKBV (3), aN6P (3), NQtP (2, confidence 5) from NeurIPS, plus a
fourth in `Review LionMuon (1) (2).docx`. Two reviewers replied after the
rebuttal; neither raised their score.

The paper text has not been changed. "Done" below means we have the
experiment and the number; "text" means the objection can only be closed by
rewriting, and that has not happened yet.

Legend: **[x]** done, evidence in hand · **[~]** partly · **[ ]** not done

---

## GKBV

- [ ] **W1. Abstract's scaling claim contradicts Table 2.** At 720M every LionMuon
  variant is worse than Muon (3.315 vs 3.291); only SignMuon beats it (3.271), same
  at 355M. → 355M and 720M have not been rerun. Text: rewrite the claim around
  SignMuon, explain the dual-EMA degradation.
- [x] **W2. One seed, small margins; add three seeds.** → 124M: 16/16 configs at
  3 seeds. LionMuon beats SignMuon in all eight pairs by 0.010–0.028, which is 5–20x
  the seed spread (0.0006–0.0059). Not done at 720M.
- [~] **W3. Interpolation is not rigorous; state that the clean bound uses the
  dense approximation and report how close it is.** → Measured: `||U||_2 / ||U||_inf`
  is about 410 (FineWeb) and 520 (WikiText) against the worst case 768. Text: not
  written.
- [ ] **W4. The metric mostly bounds Muon steps; state what is guaranteed for the
  Lion iterates.** → text only.
- [~] **W5. No comparison with MuonBP and Dion; no distributed numbers.** →
  Single-node at 124M WikiText (paper config): Dion 2.8884, MuonBP 2.9136, Muon
  2.8845. All-reduce of the Muon-updated matrices on 2 GPUs: 40.7 ms/step, halved
  at P=2. **Not done:** end-to-end distributed run, and Dion in distributed mode
  (our `dion.py` has no distributed path; needs the authors' code on 8xA100).
- [x] **Minor. Tail index kappa never estimated.** → Hill estimator: kappa = 2 at
  batch 32; heavier tails at batch 8 and 2.
- [x] **Minor. LR tuned on one 3k-step pilot; why 2D sweep for ours vs 1D for
  baselines.** → Full-budget grid for every optimizer, all optima interior, AdamW
  on its own ladder. The pilot was shown to mis-rank LRs (picks 2e-3, full horizon
  picks 3e-3, paper used 1e-3). Text: Section 5.1 and Appendix E not rewritten.
- [ ] **Typos.** "Figures 4 to 3"; Table 1 lists Muon under SignMuon; rho barred
  and unbarred in Theorem 1. → text.

### GKBV, after the rebuttal

- [x] **Retuning removes the original gaps.** → With every method at its own
  interior optimum: LionMuon 3.4983 vs AdamW 3.5482 (FineWeb), 2.8251 vs 2.8681
  (WikiText). Margins are 0.050 and 0.043, honest but smaller than the paper's.
- [x] **Seeds cover one setting and don't use the retuned LRs.** → All 16
  configs, retuned, 3 seeds.
- [ ] **720M is single-seed.** → not rerun.
- [ ] **Communication number is a microbenchmark, not end-to-end.** → DDP path
  verified to work (world_size detected, NCCL up with `NCCL_P2P_DISABLE=1`, model
  builds); the timed run was never made, needs two free GPUs.
- [ ] **No distributed Dion.** → see W5.

---

## aN6P

- [~] **W1. Claims stronger than results; dual-EMA advantage doesn't reproduce at
  720M.** → At 124M it reproduces everywhere (8/8 pairs, 3 seeds). 720M: not
  rerun.
- [~] **W2. FLOP accounting only; want GPU time, throughput, memory,
  communication.** → Per-step on RTX 4090 at 124M: Muon optimizer step 28.5 ms,
  sign step 9 ms, full step 5 % faster at P=2. Peak memory 14.1 GB vs AdamW's
  14.5 GB. All-reduce 40.7 ms/step. **Not done:** end-to-end distributed
  throughput.
- [ ] **W3. Theorem allows momentum values where terms are undefined; proof needs
  stronger conditional-noise assumption than stated.** → text only.
- [ ] **W4. Analysis treats the Muon step as exact; Newton-Schulz error not
  included.** → text only.
- [~] **W5. Missing Muon variants and efficiency baselines.** → Nesterov-Muon,
  Dion, MuonBP run at 124M (see GKBV W5). Not run at scale.

---

## NQtP

Empirics

- [x] **W1. Compared against Muon without Nesterov.** → Nesterov-Muon 2.8895 vs
  plain 2.8845 at the tuned LR; it does not help in this setup.
- [ ] **W2. Smoothness validation uses `G_{t+1} - G_t`, so both model and batch
  change; use trajectory-curvature validation as in [1,2,3].** → not run.
- [~] **W3. No practical recipe for P; and if P=1,2 are best, why not just
  Muon.** → 124M data gives the recipe: LionMuon is flat from P=2 to P=5 (FineWeb
  3.4984 / 3.4983, WikiText 2.8275 / 2.8251), degrades at P=20; P=5 costs 9 %
  fewer FLOPs than P=1 and is better. The tuned alpha grows with P (3 → 300),
  which is the usable rule. Text: not written.
- [ ] **W4. 1D parameters use AdamW; replace with Lion.** → ablation not run.

Theory

- [ ] **Assumption 3 introduced only to make the proof go through.** → text only.
- [ ] **Lines 179–180: how does the theory cover the Lion and Muon boundary
  cases; compare with prior work in an appendix.** → text only.
- [~] **Fixed ratio alpha = 0.01; why can't the theory give two learning
  rates.** → Empirically the ratio is not constant: the tuned alpha runs from 3 to
  300 and grows with P. The paper's fixed alpha was wrong as a description of its
  own runs (Appendix E pairs don't match the configs). Text: not written.
- [~] **Not convinced by the sign design or the Muon baseline.** → Beta plane and
  Nesterov results speak to this (below). Text: no.

Questions

- [x] **Why AdamW betas (0.8, 0.999) instead of (0.9, 0.95).** → (0.8, 0.999) is
  the benchmark's tuned pair and it is better: with the (0.9, 0.95) default AdamW
  loses 0.17 on FineWeb (3.7486 vs 3.5812 at lr=3e-4).
- [ ] **Replace periodic P by a probabilistic one.** → not run.
- [x] **LionMuon is sometimes beaten by SignMuon although it contains it.** →
  Only when both share one beta setting. With each pair tuned over its own
  `(eta_M, alpha)`, all three off-diagonal pairs beat all three diagonal ones
  (best 3.5006 vs 3.5092, seed spread 0.002).
- [ ] **Why 1/4 Chinchilla at 720M.** → 720M not rerun. Note: 124M is 8 TPP, i.e.
  0.4x Chinchilla, following the benchmark protocol; 355M is ~1x.
- [ ] **Compare the convergence rate with Lion and Muon under the same
  settings.** → text only.
- [ ] **Formatting: broken hyperlinks, Newton–Schulz needs an en dash.** → text.

### NQtP, after the rebuttal

- [x] **Beta comparison uses one diagonal and one off-diagonal pair; need a
  matched sweep.** → Done, 3 vs 3, FineWeb P=2, every pair tuned separately:
  diagonal 3.5113 / 3.5092 / 3.5102, off-diagonal 3.5084 / 3.5006 / 3.5016.
- [ ] **"No concentration results exist in other norms" is wrong; see Hübler et
  al. (Schatten norms) and Yu et al. (l1 and nuclear).** → Factually correct; the
  papers are arXiv:2606.14560 (June 2026, after our submission) and
  arXiv:2602.07425. Text: claim not yet narrowed.
- [~] **Nesterov, smoothness proxies, full-Chinchilla belong in a resubmission.**
  → Nesterov done; the other two not.

---

## Fourth review (docx)

- [ ] **Theorem 1's metric is dominated by the nuclear norm on the sparse Muon
  subsequence; the frequent sign steps are uncontrolled.** → text only (same
  point as GKBV W4).
- [ ] **Corollary 1 collapses LionMuon into SignMuon.** The condition
  `|1 - b1/b2| <= eps / (16 rho sigma)` forces `b1 = b2` as `eps -> 0`; the
  reviewer offers this as the real reason LionMuon's gain is modest, rather than
  the noise story in Section 5.3. → text only. The beta plane is the empirical
  counterweight: with betas tuned separately, dual-EMA wins consistently.
- [ ] **Assumption 3 is a narrow case of norm equivalence; is it needed?** →
  text only.
- [x] **Margins below 0.02 need error bars.** → 3 seeds everywhere at 124M.
- [~] **Figure 5 sweeps one order of magnitude of the Muon LR, so it cannot show
  the diagonal of optimal pairs; what is k?** → The new grid spans `eta_M` from
  1e-4 to 1e-1 and alpha from 1 to 3000; the optimum does sit on a diagonal, alpha
  rising with P. Figure not redrawn; `k` is `P`, rename it.
- [ ] **Typos.** Frobenius norm on line 118; Lemma 1 (A1) missing the factor
  `eta_t`; Hübler cited twice (411–412 and 438–439). → text.

---

## Not asked by anyone, but relevant

- [x] **Fine-tuning.** AdamW wins fine-tuning in both rows of a 2x2
  pretrain/fine-tune matrix, even from a LionMuon checkpoint (3.2241 vs 3.2766).
  Matches arXiv:2605.10468. The pretraining advantage does transfer: 0.095 at
  pretraining survives as 0.065 after fine-tuning. Report as a limitation.
- [ ] **Baseline error bars.** AdamW, Lion, Signum have seed 0 only. 12 runs,
  ~16 h. The figure shows error bars for our method and none for baselines.

---

## 124M in one table

64k steps, 3 seeds, every optimum interior. Pairs are `(eta_M, alpha)`.

| P | FineWeb SignMuon | FineWeb LionMuon | WikiText SignMuon | WikiText LionMuon |
| --- | --- | --- | --- | --- |
| 1 | 3.5271 ± 0.0006 (1e-3) | 3.5096 ± 0.0023 (1e-3) | 2.8354 ± 0.0020 (3e-3) | 2.8255 ± 0.0006 (3e-3) |
| 2 | 3.5094 ± 0.0017 (3e-3, 30) | **3.4984** ± 0.0020 (1e-3, 10) | 2.8395 ± 0.0059 (3e-3, 3) | 2.8275 ± 0.0012 (3e-3, 10) |
| 5 | 3.5072 ± 0.0019 (1e-2, 300) | **3.4983** ± 0.0018 (3e-3, 30) | 2.8406 ± 0.0019 (1e-2, 30) | **2.8251** ± 0.0010 (1e-2, 30) |
| 20 | 3.5386 ± 0.0024 (1e-2, 100) | 3.5103 ± 0.0009 (1e-2, 100) | 2.8589 ± 0.0002 (3e-2, 100) | 2.8404 ± 0.0014 (3e-2, 100) |

Baselines, seed 0, ladders closed: FineWeb AdamW 3.5482 (1e-3), Lion 3.5672
(3e-4), Signum 3.5898 (3e-4); WikiText AdamW 2.8681 (3e-3), Lion 2.8889 (3e-4),
Signum 2.8947 (1e-3).

Figure: `results/figures/124m_loss_vs_flops.png`, same style and FLOPs formula as
the paper's Pareto figure. Regenerate with `python scripts/plotting/plot_124m.py`.

Committed and pushed: `github.com/armanbolatov/llm-baselines`, branch `limuon`,
head `2b7227b`. Working tree clean.

---

## Handover

```bash
export FINEWEB_SAMPLE=sample-10BT GRID_ITERS=64000 GRID_WU=3000

# 1. baseline seeds, the only gap in 124M (12 runs, ~16 h)
LRS="1e-3" bash scripts/run_experiments.sh base fineweb 124m "1 2"
GSEEDS="1 2" PS=10000000 LRS=3e-4 ALPHAS=1 BETAS="0.9:0.99 0.9:0.9" \
  bash scripts/run_experiments.sh grid fineweb 124m
LRS="3e-3" bash scripts/run_experiments.sh base wikitext 124m "1 2"
GSEEDS="1 2" PS=10000000 LRS=3e-4 ALPHAS=1 BETAS=0.9:0.99 \
  bash scripts/run_experiments.sh grid wikitext 124m
GSEEDS="1 2" PS=10000000 LRS=1e-3 ALPHAS=1 BETAS=0.9:0.9 \
  bash scripts/run_experiments.sh grid wikitext 124m

# 2. 355M, transfers the 124M (eta_M, alpha), no re-tuning (~14 h per run)
bash scripts/run_experiments.sh seeds fineweb 355m "0 1 2"

# 3. on the 8xA100 box
bash scripts/run_experiments.sh extra fineweb 124m
torchrun --nproc_per_node=8 scripts/comms_bench.py
```

Traps that each cost hours here:

1. `FINEWEB_SAMPLE=sample-10BT` always, or the loader pulls 450 GB.
2. Never mix batch splits (`16x2` vs `32x1` differ by 0.026) or machines within
   one comparison. Seeds must match their seed 0.
3. Run seeds only after `close_edges.sh` says "all optima interior"; when a winner
   moves, delete its old `*t_*_seed{1,2}` dirs by hand, `main.py` skips on
   `summary.json` alone.
4. AdamW needs betas (0.8, 0.999) and ~15 GB at bs32.
5. `NCCL_P2P_DISABLE=1` on GPUs without NVLink.
6. Never `git add -A` after a run with `--latest_ckpt_interval`.

---

## Update, September 7 (ICLR 2027 push)

### Newly closed

- [x] **GKBV W5 / aN6P W2, W5: end-to-end distributed numbers, distributed
  Dion.** Real 4-GPU DDP on `opt` (A100 PCIe, no NVLink), 124M, batch 32 (8/GPU):
  AdamW 0.63 s/step, Muon 0.84, LionMuon P=2 0.71, P=5 0.65, MuonBP P=5 0.77,
  Dion 0.66; at 32/GPU: 0.75 / 0.96 / 0.86 / 0.75 / 0.90 / 0.90. Isolated
  primitives: grad sync fp32 496 MB 628 ms, Muon update bf16 170 MB 228 ms, Dion
  factors fp32 113 MB 144 ms. `dion.py` and `muonbp.py` now have DDP paths;
  `lion_muon.py` replicates momentum so sign steps are local (the old sharded
  version all-reduced on every step and saved nothing). Files:
  `results/benchmarks/README_comms_124m.md`, `ddp_bench_124m.json`,
  `scripts/ddp_bench.sh`, `scripts/comms_compare.py`. Code uncommitted.
- [x] **Dion / MuonBP on FineWeb, same protocol** (`opt`, 100BT sample, lr 1e-3,
  32x1, seed 0, 64k): Muon 3.5267, LionMuon P=2 3.4979, Dion 3.5501, MuonBP
  3.5456. Cross-machine check: the 10BT numbers on the 5090 box (3.5278 / 3.5022)
  agree within 0.005. WikiText at the same LR: Muon 2.885, LionMuon P=2 2.886,
  Dion 2.888, MuonBP 2.914 (single untuned LR for Dion/MuonBP; do not compare to
  tuned numbers).
- [x] **OPT2026 workshop version** submitted (`workshop.tex` in the Overleaf
  project, official `opt2026.cls`).

### 355M / 720M campaign (started Sept 7, deadline Sept 25)

Rule: no tuning; every method uses its 124M full-horizon optimum `(eta_M, alpha)`
verbatim (the paper's 355M/720M used x0.75 and alpha = 20-300 from 500-step
pilots, which is where the dual-EMA advantage was lost). FineWeb-100BT on both
machines, 1x Chinchilla, split 16x32 at 355M on both `opt` and `opt2` (29 GB,
~10 s/it under the shared load, ~43 h per run). At most 2 GPUs per cluster.
Checkpoint every 250 iters, `launch355.sh` relaunches on OOM (auto_resume).

Wave 1 (running): `opt2` GPU7 LionMuon P=2 (1e-3, 10), GPU6 Muon (1e-3);
`opt` GPU3 SignMuon P=2 (3e-3, 30), GPU1 Muon (1e-3, cross-machine calibration).
Wave 2: AdamW 1e-3, LionMuon P=5 (3e-3, 30), LionMuon P=1, Muon 3e-3 (baseline
insurance). Wave 3: seeds 1-2 for the four headline rows. Then 720M on `opt2`
only: LionMuon P=2, Muon, SignMuon P=2, AdamW (16x124, 14k iters, 700 warm-up).

Commands (per run, in a retry loop):
`FINEWEB_SAMPLE=sample-100BT DATASETS_DIR=<dir> BS=16 ACC=32 CKPT_EVERY=250 DEV=cuda:N PS=<P> LBLS=<lionmuon|signmuon> bash scripts/run_experiments.sh seeds fineweb 355m "0"`
(`run_experiments.sh` gained a `720m` scale, `DATASETS_DIR`, `CKPT_EVERY`, `LBLS`.)
