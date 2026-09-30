"""check_campaign_tables.py -- recompute the training tables (Tables 1-6) and the
numbers quoted with them from the per-run files in results/campaign.

  python analysis/check_campaign_tables.py

Each res_<tag>.json is one training run (tags are explained in README.md).
recov_05 is T_GE<0.5 at W = 3000 and is null when the run does not recover.
Recovery medians are taken over successful runs only, and failures are counted
separately (R4). Quartiles use numpy's default linear interpolation.
"""
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMP = os.path.join(ROOT, "results", "campaign")
EPOCH_LINE = re.compile(r"^\[ep (\d+)/\d+\] loss=([\d.]+) acc=([\d.]+)")


def load_runs():
    runs = {}
    for path in glob.glob(os.path.join(CAMP, "res_*.json")):
        with open(path) as fh:
            run = json.load(fh)
        runs[run["tag"]] = run
    return runs


def arm(runs, prefix, seeds):
    return [runs[f"{prefix}_s{s}"] for s in seeds]


def recovered(rs):
    return sorted(r["recov_05"] for r in rs if r["recov_05"] is not None)


def cell(rs):
    """Median T_GE<0.5 over successful runs, or '--' with the median minimum GE."""
    ok = recovered(rs)
    if ok:
        count = f" ({len(ok)}/{len(rs)} recover)" if len(rs) > 1 else ""
        return f"{np.median(ok):g}{count}"
    return f"-- ({np.median([r['ge_min'] for r in rs]):.1f})"


def per_run(rs):
    return "/".join(str(r["recov_05"]) if r["recov_05"] is not None
                    else f"--({r['ge_min']:.1f})" for r in rs)


def last_epoch(tag):
    """(epoch, training loss, training accuracy) of the last trained epoch."""
    with open(os.path.join(CAMP, "logs", f"{tag}.log")) as fh:
        rows = [m.groups() for m in map(EPOCH_LINE.match, fh) if m]
    ep, loss, acc = rows[-1]
    return int(ep), float(loss), float(acc)


def table1(runs):
    print("Table 1 (tab:stages): T_GE<0.5, or -- (minimum GE), W = 3000")
    for label, f, v in [("S1", ["s1_f_pure_s0"], ["s1_v_pure_s0"]),
                        ("S2", ["s2_f_hybrid_s0"], ["s2_v_hybrid_s0"])]:
        print(f"  {label:<8} fixed {cell([runs[t] for t in f]):<22} "
              f"variable {cell([runs[t] for t in v])}")
    for label, p in [("S3-QCNN", "quantum_qcnn"), ("S3-SEL", "quantum_sel")]:
        f = arm(runs, f"s3_f_{p}", range(10) if p == "quantum_qcnn" else range(3))
        v = arm(runs, f"s3_v_{p}", range(3))
        print(f"  {label:<8} fixed {cell(f):<22} variable {cell(v)}")
    kept = runs["s1_f_pure_s0"]["var_kept"], runs["s1_v_pure_s0"]["var_kept"]
    print(f"  PCA variance kept: fixed {kept[0]:.1%}, variable {kept[1]:.1%}")


def table2(_runs):
    with open(os.path.join(CAMP, "grad_variance.json")) as fh:
        gv = json.load(fh)
    print("Table 2 (tab:gradvar): Var[dC/dtheta_1] over 300 draws")
    for name in ("qcnn", "sel"):
        vals = "  ".join(f"n={n}: {gv[name][n]['var']:.3g}" for n in ("4", "6", "8", "10", "12"))
        print(f"  {name.upper():<5} {vals}")


def table3(runs):
    print("Table 3 (tab:variance): ASCAD fixed, ten paired seeds")
    for label, p in [("QCNN", "quantum"), ("Dense control", "clswide"),
                     ("Rank-1 control", "clsnarrow")]:
        rs = arm(runs, f"s3_f_{p}_qcnn", range(10))
        ok = recovered(rs)
        mins = [r["wallclock_s"] / 60 for r in rs]
        spread = (f"{np.median(ok):g} [IQR {np.percentile(ok, 25):g}-{np.percentile(ok, 75):g};"
                  f" range {ok[0]}-{ok[-1]}]" if ok else "--")
        print(f"  {label:<15} mid-block params {rs[0]['mid_params']:>3}  "
              f"failures {len(rs) - len(ok)}/10  T_GE {spread:<38} "
              f"time median {np.median(mins):.1f} min (mean {np.mean(mins):.1f})")


def table4(runs):
    print("Table 4 (tab:main): classical reference models, one run each")
    for label, p in [("MLP_best", "mlp"), ("CNN_best", "cnnbest"), ("RL-CNN", "rijsdijk")]:
        f, v = runs[f"{p}_f_s0"], runs[f"{p}_v_s0"]
        print(f"  {label:<9} fixed {cell([f]):<14} variable {cell([v]):<14} "
              f"(fixed GE<1 at {f['recov_1']})")


def table5(runs):
    print("Table 5 (tab:scan): whole-model params for 1400/700-sample inputs")
    rows = [("n=8,  l=2", "quantum_qcnn", range(10)),
            ("n=8,  l=3", "quantum_qcnn_q8c3", range(1)),
            ("n=10, l=3", "quantum_qcnn_q10c3", range(1))]
    for label, p, fixed_seeds in rows:
        f = arm(runs, f"s3_f_{p}", fixed_seeds)
        v = arm(runs, f"s3_v_{p}", range(3) if "q10" not in p else range(1))
        print(f"  {label}  params {v[0]['params']}/{f[0]['params']}  "
              f"fixed {cell(f):<22} variable {per_run(v)}")


def table6(runs):
    print("Table 6 (tab:desync): minimum GE, median [min-max] over the listed runs")
    for label, p, seeds in [("S3-QCNN", "s3_{d}_quantum_qcnn", range(3)),
                            ("Dense control", "s3_{d}_clswide_qcnn", range(3)),
                            ("S2", "s2_{d}_hybrid", range(1)),
                            ("RL-CNN", "rijsdijk_{d}", range(1))]:
        cols = []
        for d in ("d50", "d100"):
            g = [r["ge_min"] for r in arm(runs, p.format(d=d), seeds)]
            span = f" [{min(g):.1f}-{max(g):.1f}]" if len(g) > 1 else ""
            cols.append(f"{d}: {np.median(g):.1f}{span} ({len(g)} run{'s' * (len(g) > 1)})")
        print(f"  {label:<13} " + "   ".join(cols))


def text_numbers(runs):
    print("Numbers quoted in Sec. 3")
    for tag in ("s1_f_pure_s0", "s2_f_hybrid_s0"):
        ep, loss, acc = last_epoch(tag)
        print(f"  {tag}: last trained epoch {ep}, loss {loss:.2f}, accuracy {acc:.2%} "
              f"(checkpoint restored from epoch {runs[tag]['best_epoch']})")
    q = arm(runs, "s3_f_quantum_qcnn", range(10))
    for name, grp in (("successful", [r for r in q if r["recov_05"] is not None]),
                      ("failed", [r for r in q if r["recov_05"] is None])):
        last = [last_epoch(r["tag"]) for r in grp]
        print(f"  S3-QCNN {name} runs ({len(grp)}): last-epoch loss "
              f"{np.mean([x[1] for x in last]):.2f}, accuracy {np.mean([x[2] for x in last]):.2%}, "
              f"mean best proxy {np.mean([r['val_ge_auc'] for r in grp]):.1f}")
    low = min(q, key=lambda r: last_epoch(r["tag"])[1])
    print(f"  lowest last-epoch loss {last_epoch(low['tag'])[1]:.2f}: {low['tag']}, "
          f"T_GE {low['recov_05']}")
    sel = arm(runs, "s3_f_quantum_sel", range(3)) + arm(runs, "s3_v_quantum_sel", range(3))
    print("  S3-SEL last-epoch loss: " + ", ".join(
        f"{r['tag'][3:]} {last_epoch(r['tag'])[1]:.4f}" for r in sel))


def main():
    runs = load_runs()
    for show in (table1, table2, table3, table4, table5, table6, text_numbers):
        show(runs)
        print()


if __name__ == "__main__":
    main()
