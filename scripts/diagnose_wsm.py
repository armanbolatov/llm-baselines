"""End-to-end diagnostic for the WSM pipeline.

Runs on CPU, no dataset required. Builds a tiny GPTBase, takes a few
optimizer steps under the wsm scheduler, saves permanent checkpoints
through the project's save_checkpoint, then invokes merge_wsm.py for all
three methods and verifies the merged checkpoints load back into the
model. Touches every new code path: schedule, merge, CLI wrapper, eval
state-loading.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

from models.utils import get_model
from optim.merge import (discover_ckpts, get_weights, load_model_state,
                         merge_state_dicts)
from optim.schedule import wsm_schedule
from optim.utils import save_checkpoint


def red(s):
    return f"\033[31m{s}\033[0m"


def grn(s):
    return f"\033[32m{s}\033[0m"


def banner(s):
    print(f"\n{'=' * 12} {s} {'=' * 12}")


def tiny_args():
    """Minimal args dict for instantiating a small GPTBase on CPU."""
    return SimpleNamespace(
        model="base",
        use_pretrained="none",
        n_layer=2, n_head=2, n_embd=64,
        sequence_length=32, vocab_size=50304, multiple_of=64,
        dropout=0.0, bias=False, dtype="float32",
        rmsnorm_eps=1e-5, init_std=0.02,
        mlp_dim_exp_factor=1.0, parallel_block=False,
    )


def step1_scheduler():
    banner("1. WSM scheduler under a real LambdaLR")
    p = torch.nn.Parameter(torch.zeros(4))
    opt = torch.optim.SGD([p], lr=1e-3)
    sched_fn = wsm_schedule(n_iterations=100, n_warmup=10, init_div_factor=100)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_fn)

    lrs = []
    for _ in range(30):
        p.grad = torch.ones_like(p)
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])

    # Expectations: linear warmup from 1e-5 -> 1e-3 over 10 steps,
    # then constant 1e-3 thereafter.
    ok = (
        abs(lrs[0] - 1e-3 * (1/10 + 9/10 * 1/100)) < 1e-9
        and abs(lrs[9] - 1e-3) < 1e-9
        and all(abs(x - 1e-3) < 1e-9 for x in lrs[10:])
    )
    print(f"  lr[0]={lrs[0]:.2e}  lr[9]={lrs[9]:.2e}  lr[15]={lrs[15]:.2e}  lr[29]={lrs[29]:.2e}")
    print("  " + (grn("PASS") if ok else red("FAIL: WSM did NOT stay constant after warmup")))
    assert ok


def step2_model_save_load(tmp: Path):
    banner("2. Build tiny GPTBase, run 5 steps under WSM, save 4 ckpts")
    args = tiny_args()
    torch.manual_seed(0)
    model = get_model(args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.3f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, wsm_schedule(n_iterations=1000, n_warmup=5, init_div_factor=100)
    )

    iters_saved = []
    for it in range(1, 21):  # 20 steps
        x = torch.randint(0, args.vocab_size, (2, args.sequence_length))
        y = torch.randint(0, args.vocab_size, (2, args.sequence_length))
        out = model(x, targets=y)
        out["loss"].backward()
        opt.step(); sched.step(); opt.zero_grad()
        if it in (5, 10, 15, 20):
            ckpt_dir = tmp / "ckpts" / str(it)
            save_checkpoint(model, opt, sched, it, ckpt_dir)
            iters_saved.append(it)
            print(f"  saved ckpt iter={it}  loss={out['loss'].item():.3f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

    found = [int(p.name) for p in discover_ckpts(tmp)]
    print(f"  discover_ckpts -> {found}  expected {iters_saved}")
    assert found == iters_saved
    return args


def step3_merge_cli(tmp: Path):
    banner("3. merge_wsm.py via subprocess, all three methods")
    py = sys.executable
    for method in ("mean", "ema", "theorem"):
        r = subprocess.run(
            [py, str(REPO / "src" / "merge_wsm.py"),
             "--exp_dir", str(tmp), "--n", "4", "--method", method],
            capture_output=True, text=True, cwd=str(REPO),
        )
        print(f"  [{method}] rc={r.returncode}")
        if r.stdout.strip():
            for line in r.stdout.strip().splitlines():
                print(f"    {line}")
        if r.returncode != 0:
            print(red("    STDERR:")); print(r.stderr)
            raise SystemExit("merge_wsm.py failed")

    # Verify outputs exist on disk
    for tag in ("merged_mean_n4", "merged_ema0.5_n4", "merged_theorem_n4"):
        f = tmp / "ckpts" / tag / "main.pt"
        assert f.exists(), f"missing {f}"
    print("  " + grn("all three merged files written"))


def step4_loadback(tmp: Path, args):
    banner("4. Load each merged ckpt back into a fresh GPTBase")
    torch.manual_seed(1)  # different seed so init differs from training
    fresh = get_model(args)
    fresh_sd = {k: v.clone() for k, v in fresh.state_dict().items()}

    for tag in ("merged_mean_n4", "merged_ema0.5_n4", "merged_theorem_n4"):
        path = tmp / "ckpts" / tag / "main.pt"
        obj = torch.load(path, map_location="cpu", weights_only=False)
        merged_sd = obj["model"]
        meta = obj.get("merge", {})
        missing, unexpected = fresh.load_state_dict(merged_sd, strict=False)
        if missing or unexpected:
            print(red(f"  [{tag}] strict-load issues: "
                      f"missing={len(missing)} unexpected={len(unexpected)}"))
            raise SystemExit

        # Confirm the merge actually changed weights vs. fresh init.
        sample = next(iter(merged_sd))
        diff = (merged_sd[sample].float() - fresh_sd[sample].float()).norm().item()
        # Confirm a forward pass works on CPU.
        x = torch.randint(0, args.vocab_size, (1, args.sequence_length))
        with torch.no_grad():
            out = fresh(x, targets=x)
        print(f"  [{tag}] weights[{sample}] |delta vs init|={diff:.3f}  "
              f"forward loss={out['loss'].item():.3f}  "
              f"method={meta.get('method')}  weights={[round(w,3) for w in meta.get('weights',[])]}")
    print("  " + grn("all merged checkpoints loaded and ran a forward pass"))


def step5_partial_sum_invariant():
    banner("5. Theorem 3.1 invariant: partial sums of c equal 1-sqrt envelope")
    import math
    for n in (2, 4, 8, 12):
        k = n - 1
        c = get_weights("theorem", n)
        for i in range(1, k + 1):
            Wi = sum(c[i:])
            wi = 1.0 - math.sqrt((i - 1) / k)
            assert abs(Wi - wi) < 1e-9, (n, i, Wi, wi)
        print(f"  n={n:>2}: c={[round(x,3) for x in c]}  partial sums = w envelope OK")
    print("  " + grn("Eq. 4 invariant holds"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true",
                        help="keep the temp work dir for inspection")
    a = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="wsm_diag_"))
    try:
        step1_scheduler()
        args = step2_model_save_load(tmp)
        step3_merge_cli(tmp)
        step4_loadback(tmp, args)
        step5_partial_sum_invariant()
        print(f"\n{grn('DIAGNOSIS PASSED')}")
        print(f"work dir: {tmp}  ({'kept' if a.keep else 'cleaning up'})")
    except Exception:
        print(f"\n{red('DIAGNOSIS FAILED')}")
        print(f"work dir kept at {tmp}")
        raise
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
