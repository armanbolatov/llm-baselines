r"""The constants of the bound as numbers: one row per run, from norm_diag.json files.

Each record already holds the median over the 2D parameters; a constant is the mean over the records of a run,
exactly as in plot_norm_diagnostics.py. Columns: alpha_G, rho_nuc, rho_1, L_2, L_inf, alpha_U and the two ratios
r_L = L_inf / (alpha_G^2 L_2) and r_rho = rho_1 / (alpha_G rho_nuc) of the same run. With --ref <label> two more
columns compare each run with the reference run (the P=1 run in the paper):
r_L(P) = L_inf(P) / (alpha_G(P)^2 L_2(ref)) and r_rho(P) = rho_1(P) / (alpha_G(P) rho_nuc(ref)).

    D=results/additional/constants_124m
    python scripts/plotting/constants_table.py --ref P=1 P=1=$D/ndL12_fw_p1_seed0.norm_diag.json \
        P=2=$D/ndL12_fw_p2_seed0.norm_diag.json P=5=$D/ndL12_fw_p5_seed0.norm_diag.json \
        Lion=$D/ndL12_fw_pinf_seed0.norm_diag.json
"""
import json
import statistics as st
import sys

# key in the record -> is the stored ratio the inverse of the constant?
KEYS = {"alpha_G": ("grad_nuc_over_1", True), "rho_nuc": ("errmom_F_over_nuc", True), "rho_1": ("errmom_F_over_1", True),
        "L_2": ("smooth_nuc_over_2", False), "L_inf": ("smooth_1_over_inf", False), "alpha_U": ("update_2_over_inf", False)}


def constants(path):
    d = json.load(open(path))
    rows = d["records"] if isinstance(d, dict) and "records" in d else d
    out = {}
    for name, (key, inv) in KEYS.items():
        # the two smoothness proxies are differences between consecutive records and have gaps
        s = [(1 / r[key]) if inv else r[key] for r in rows if r.get(key) and r[key] == r[key]]
        out[name] = st.mean(s)
    out["records"] = len(rows)
    return out


def main(args):
    ref = None
    if args and args[0] == "--ref":
        ref, args = args[1], args[2:]
    runs = {a.rsplit("=", 1)[0]: constants(a.rsplit("=", 1)[1]) for a in args}
    head = f"{'run':<12}{'recs':>5}{'alpha_G':>9}{'rho_nuc':>9}{'rho_1':>8}{'L_2':>8}{'L_inf':>9}{'alpha_U':>9}{'r_L':>7}{'r_rho':>7}"
    print(head + (f"{'r_L(P)':>9}{'r_rho(P)':>10}" if ref else ""))
    for label, v in runs.items():
        a = v["alpha_G"]
        line = (f"{label:<12}{v['records']:>5}{a:>9.1f}{v['rho_nuc']:>9.1f}{v['rho_1']:>8.0f}{v['L_2']:>8.2f}{v['L_inf']:>9.0f}"
                f"{v['alpha_U']:>9.0f}{v['L_inf'] / (a * a * v['L_2']):>7.2f}{v['rho_1'] / (a * v['rho_nuc']):>7.2f}")
        if ref:
            r = runs[ref]
            line += f"{v['L_inf'] / (a * a * r['L_2']):>9.2f}{v['rho_1'] / (a * r['rho_nuc']):>10.2f}"
        print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
