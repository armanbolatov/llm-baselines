"""Qwen fine-tuning benchmark.

Fine-tunes a small Qwen model on Alpaca with one of AdamW / Muon / SignMuon /
LionMuon, and reports best eval loss. The Muon-family optimizers are imported
from this repo's src/optim and handle 2D matrices (Muon step) vs 1D/embedding
(AdamW backup) internally. Keep it small and fast.

Usage:
    python scripts/finetune_qwen.py --opt lion_muon --lr 1e-3 --model Qwen/Qwen3-0.6B
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from optim.muon import Muon  # noqa: E402
from optim.lion_muon import LionMuon  # noqa: E402
from optim.sign_muon import SignMuon  # noqa: E402


def build_optimizer(name, model, lr, muon_p=2):
    params = list(model.parameters())
    adamw_kw = dict(adamw_params=None, adamw_lr=lr, adamw_betas=(0.9, 0.95),
                    adamw_eps=1e-8, adamw_wd=0.0)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    if name == "muon":
        return Muon(muon_params=params, lr=lr, momentum=0.9, nesterov=False,
                    ns_steps=5, weight_decay=0.0, **adamw_kw)
    if name == "sign_muon":
        return SignMuon(muon_params=params, lr=lr, cheap_lr=lr * 0.02, momentum=0.9,
                        nesterov=False, ns_steps=5, muon_every_k=muon_p, cheap_mode="sign",
                        sign_scaling="muon", weight_decay=0.0, srank_alpha=0.0, **adamw_kw)
    if name == "lion_muon":
        return LionMuon(muon_params=params, lr=lr, lion_lr=lr * 0.02, beta1=0.9,
                        beta2=0.99, ns_steps=5, muon_every_k=muon_p, nesterov=False,
                        weight_decay=0.0, srank_alpha=0.0, **adamw_kw)
    raise ValueError(name)


def format_alpaca(ex):
    ins, inp, out = ex["instruction"], ex.get("input", ""), ex["output"]
    prompt = f"### Instruction:\n{ins}\n\n"
    if inp:
        prompt += f"### Input:\n{inp}\n\n"
    return prompt + f"### Response:\n{out}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", required=True,
                    choices=["adamw", "muon", "sign_muon", "lion_muon"])
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--muon_p", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()

    torch.manual_seed(cli.seed)
    dev = torch.device(cli.device)
    torch.cuda.set_device(dev)

    tok = AutoTokenizer.from_pretrained(cli.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cli.model, dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.train()

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=42).select(range(cli.n_train + cli.n_eval))
    texts = [format_alpaca(ex) for ex in ds]
    train_texts, eval_texts = texts[: cli.n_train], texts[cli.n_train :]

    def batch_tokens(text_list, idx):
        chunk = [text_list[i % len(text_list)] for i in idx]
        enc = tok(chunk, return_tensors="pt", padding="max_length", truncation=True,
                  max_length=cli.seq_len)
        ids = enc["input_ids"].to(dev)
        labels = ids.clone()
        labels[enc["attention_mask"].to(dev) == 0] = -100
        return ids, enc["attention_mask"].to(dev), labels

    opt = build_optimizer(cli.opt, model, cli.lr, cli.muon_p)

    def lr_factor(step):
        if step < cli.warmup:
            return (step + 1) / cli.warmup
        p = (step - cli.warmup) / max(1, cli.steps - cli.warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    base_lrs = [g["lr"] for g in opt.param_groups]
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(cli.seed)

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        for s in range(0, min(cli.n_eval, 256), cli.batch_size):
            idx = list(range(s, s + cli.batch_size))
            ids, am, labels = batch_tokens(eval_texts, idx)
            with ctx:
                out = model(input_ids=ids, attention_mask=am, labels=labels)
            tot += out.loss.item(); n += 1
        model.train()
        return tot / max(1, n)

    best = float("inf")
    for step in range(cli.steps):
        f = lr_factor(step)
        for grp, b in zip(opt.param_groups, base_lrs):
            grp["lr"] = b * f
        idx = torch.randint(0, len(train_texts), (cli.batch_size,), generator=g).tolist()
        ids, am, labels = batch_tokens(train_texts, idx)
        with ctx:
            out = model(input_ids=ids, attention_mask=am, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % cli.eval_every == 0 or step == cli.steps - 1:
            el = evaluate()
            best = min(best, el)
            print(f"[{cli.opt} lr={cli.lr:.0e}] step {step+1}/{cli.steps} "
                  f"train {out.loss.item():.4f} eval {el:.4f} best {best:.4f}", flush=True)

    print(f"RESULT opt={cli.opt} p={cli.muon_p} lr={cli.lr:.0e} best_eval={best:.4f}")


if __name__ == "__main__":
    main()
