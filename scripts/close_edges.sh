#!/bin/bash
# Extend the grid until every (P, beta) optimum is interior on the eta_M and
# alpha ladders: find edge optima, run one neighbour beyond each, repeat.
#   bash scripts/close_edges.sh fineweb 124m     (budget via GRID_ITERS)
set -u
cd "$(dirname "$0")/.."
DS=${1:-fineweb}; SCALE=${2:-124m}
TAG=$([ "$DS" = fineweb ] && echo fw || echo wt)
P=${TAG}${SCALE%m}
GB=${GRID_ITERS:-20000}   # only touch cells of this budget
SUF=""; [ "$GB" != 20000 ] && SUF="_i$((GB/1000))k"
for round in 1 2 3 4 5; do
  NEXT=$(python3 - "$P" "$SUF" <<'PY'
import json, glob, sys, re
p, suf = sys.argv[1], sys.argv[2]
LR = ['1e-4','3e-4','1e-3','3e-3','1e-2','3e-2','1e-1']   # ordered ladders
AL = ['1','3','10','30','100','300','1000','3000']
def best(f):
    s=json.load(open(f)); vl=s['val_loss']
    v = min(vl) if not isinstance(vl[0],dict) else min(x.get('val/loss',9) for x in vl)
    return float('inf') if v!=v else v   # diverged = worse neighbour
R={}
for src in ('exps_124m','exps'):
    for f in glob.glob(f'{src}/{p}g_*/summary.json'):
        k = f.split('/')[1].replace('_seed0','')
        if suf:
            if not k.endswith(suf): continue
            k = k[:-len(suf)]
        elif re.search(r'_i\d+k$', k): continue
        R[k]=best(f)
jobs=set()
# group into 1-D sweeps: over LR at fixed alpha, and over alpha at fixed LR
groups={}
for k in R:
    m=re.match(rf'{p}g_p(\d+)_lr([\w.-]+?)(?:_a([\w.]+))?_b(\S+)$', k)
    if not m: continue
    P_,lr,al,b = m.group(1), m.group(2), m.group(3), m.group(4)
    groups.setdefault(('lr',P_,al,b),{})[lr]=R[k]
    if al: groups.setdefault(('al',P_,lr,b),{})[al]=R[k]
for (axis,P_,other,b),cells in groups.items():
    ladder = LR if axis=='lr' else AL
    pts=[x for x in ladder if x in cells]
    if len(pts)<2: continue
    bi=min(range(len(pts)), key=lambda i: cells[pts[i]])
    if 0<bi<len(pts)-1: continue
    idx=ladder.index(pts[bi])
    tgt = ladder[idx-1] if bi==0 else (ladder[idx+1] if idx+1<len(ladder) else None)
    if tgt is None or tgt in cells: continue
    if axis=='lr':  jobs.add((P_, tgt, other or '-', b))
    else:           jobs.add((P_, other, tgt, b))
for j in sorted(jobs): print(' '.join(j))
PY
)
  [ -z "$NEXT" ] && { echo "[$(date +%H:%M)] round $round: all optima interior"; break; }
  echo "[$(date +%H:%M)] round $round: closing $(echo "$NEXT" | wc -l) edge(s)"
  echo "$NEXT" | while read p_ lr al b; do
    # names drop the dots in betas: 09-099 -> 0.9:0.99
    b1=${b%%-*}; b2=${b##*-}
    bb="${b1:0:1}.${b1:1}:${b2:0:1}.${b2:1}"
    if [ "$al" = "-" ]; then
      env GRID_ITERS="$GB" GRID_WU="${GRID_WU:-1000}" PS="$p_" LRS="$lr" BETAS="$bb" \
        bash scripts/run_experiments.sh grid "$DS" "$SCALE"
    else
      env GRID_ITERS="$GB" GRID_WU="${GRID_WU:-1000}" PS="$p_" LRS="$lr" ALPHAS="$al" BETAS="$bb" \
        bash scripts/run_experiments.sh grid "$DS" "$SCALE"
    fi
  done
done
