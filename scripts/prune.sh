#!/bin/bash
# Stop a sweep cell once it is clearly out of the running.
#
#   bash scripts/prune.sh logs/grid64k_fw.log
#
# Gates come from 34 finished 64k curves: no eventual winner was ever more than
# 0.067 behind at 6k steps, or 0.154 at 16k. Cells are compared only against
# finished cells of their own (P, betas, budget) group.
set -u
cd "$(dirname "$0")/.."
exec python3 - "${1:?usage: prune.sh <sweep log>}" <<'PY'
import json, os, re, signal, sys, time, glob

LOG = sys.argv[1]
GATES = {6000: 0.10, 16000: 0.20}   # step -> max deficit vs group best
EVAL_EVERY = 500
IDLE_STOP = 1200                    # seconds of silence before giving up

def group_of(name):
    m = re.match(r'\w+g_p(\d+)_.*_b([\d-]+)', name)
    if not m:
        return None
    b = re.search(r'_i(\d+)k', name)          # budgets have different schedules
    return m.group(1), m.group(2), b.group(1) if b else '20'

def group_best_at(group, step):
    idx, out = step // EVAL_EVERY, []
    for f in glob.glob('exps/*/summary.json'):
        if group_of(os.path.basename(os.path.dirname(f))) != group:
            continue
        try:
            s = json.load(open(f))
            vl = s['val_loss']
        except Exception:
            continue
        if s.get('pruned') or len(vl) <= idx or vl[idx] != vl[idx]:
            continue
        out.append(vl[idx])
    return min(out) if out else None

def pid_of(name):
    for p in os.listdir('/proc'):
        if not p.isdigit():
            continue
        try:
            cmd = open(f'/proc/{p}/cmdline').read().split('\0')
        except Exception:
            continue
        if 'src/main.py' in ' '.join(cmd) and name in cmd:
            return int(p)
    return None

cur, curve, last = None, [], time.time()
with open(LOG) as fh:
    fh.seek(0, 2)
    while True:
        line = fh.readline()
        if not line:
            if time.time() - last > IDLE_STOP:
                break
            time.sleep(5)
            continue
        last = time.time()

        m = re.match(r'\[[\d\- :]+\] (\S+_seed\d)\s*$', line)
        if m:
            cur, curve = m.group(1), []
            continue
        m = re.search(r'>Eval: Iter=(\d+).*val_loss=([\d.]+|nan)', line)
        if not (m and cur):
            continue

        step, val = int(m.group(1)), float(m.group(2))
        curve.append(val)
        if val != val:                      # diverged
            reason = 'nan'
        else:
            if step not in GATES:
                continue
            ref = group_best_at(group_of(cur), step)
            if ref is None or val <= ref + GATES[step]:
                continue
            reason = f'{val:.4f} vs {ref:.4f} (+{val - ref:.4f})'

        pid = pid_of(cur)
        print(f'[prune] {cur}: {reason} at {step}, pid={pid}', flush=True)
        os.makedirs(f'exps/{cur}', exist_ok=True)
        json.dump({'val_loss': curve, 'pruned': True, 'pruned_at': step},
                  open(f'exps/{cur}/summary.json', 'w'))
        if pid:
            os.kill(pid, signal.SIGTERM)
        cur = None
PY
