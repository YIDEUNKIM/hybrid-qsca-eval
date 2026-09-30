"""make_fig_recovery_budget.py -- Figure 2: fixed-key recovery against the attack-trace
budget over ten paired training seeds.

For each Stage-3 middle block (S3-QCNN, dense control, rank-1 control) the curve counts
the training runs whose recovery threshold T_GE<0.5 is at most B, for B = 0..3000.
Thresholds are read from the original ten-seed runs in results/campaign
(res_s3_f_<block>_qcnn_s0..s9.json, computed once with W = 3000). Failed runs never
count, and the weight-saving rerun s3_f_quantum_qcnn_s0_w is not used.

The script also prints the numbers quoted with the figure: the counts at B = 250, 1000
and 3000, the paired outcomes of S3-QCNN and the dense control, and the two-sided exact
McNemar p-value for recovery within W = 3000.

  python figures/make_fig_recovery_budget.py [--out out/fig_recovery_budget.pdf]
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results" / "campaign"
SEEDS = range(10)
HORIZON = 3000
QUOTED_BUDGETS = (250, 1000, HORIZON)

ARMS = [  # (label, run-tag prefix, line style, colour)
    ("S3-QCNN", "s3_f_quantum_qcnn", "-", "C0"),
    ("Dense control", "s3_f_clswide_qcnn", "--", "C1"),
    ("Rank-1 control", "s3_f_clsnarrow_qcnn", ":", "C2"),
]

# Figure and axes geometry in points; the manuscript scales the figure by 0.994
# when it includes it at 0.94\textwidth.
FIG_W, FIG_H = 477.0, 169.25
AXES_RECT = (37.10 / FIG_W, 34.67 / FIG_H, 428.26 / FIG_W, 112.29 / FIG_H)
LEGEND_ANCHOR_Y = 0.9703   # legend bottom, in axes-height units
STYLE = {"font.size": 8, "axes.labelsize": 9, "lines.linewidth": 1.8,
         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
         "pdf.fonttype": 42}   # TrueType instead of Type 3 fonts


def load_thresholds(prefix):
    """T_GE<0.5 of seeds 0-9 of one configuration, None for a failed run."""
    thresholds = []
    for seed in SEEDS:
        tag = f"{prefix}_s{seed}"
        with open(CAMPAIGN / f"res_{tag}.json") as fh:
            run = json.load(fh)
        if run["tag"] != tag:
            raise ValueError(f"res_{tag}.json holds run {run['tag']}")
        t = run["recov_05"]
        if t is not None and not 1 <= t <= HORIZON:
            raise ValueError(f"{tag}: threshold {t} outside 1..{HORIZON}")
        thresholds.append(t)
    return thresholds


def step_points(thresholds):
    """Vertices of the count-of-recovered-runs curve for where='post' steps."""
    ok = sorted(t for t in thresholds if t is not None)
    return [0] + ok + [HORIZON], list(range(len(ok) + 1)) + [len(ok)]


def recovered_by(thresholds, budget):
    return sum(t is not None and t <= budget for t in thresholds)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def print_quoted(runs):
    for label, _, _, _ in ARMS:
        counts = ", ".join(f"B={b}: {recovered_by(runs[label], b)}" for b in QUOTED_BUDGETS)
        print(f"{label:<15} {counts}")
    pairs = list(zip(runs["S3-QCNN"], runs["Dense control"]))
    both = [(q, d) for q, d in pairs if q is not None and d is not None]
    only_q = sum(q is not None and d is None for q, d in pairs)
    only_d = sum(q is None and d is not None for q, d in pairs)
    print(f"paired seeds: both recover {len(both)} "
          f"(QCNN lower {sum(q < d for q, d in both)}, dense lower "
          f"{sum(d < q for q, d in both)}, ties {sum(q == d for q, d in both)}); "
          f"only QCNN {only_q}; only dense {only_d}; "
          f"neither {len(pairs) - len(both) - only_q - only_d}")
    print(f"two-sided exact McNemar p = {mcnemar_exact(only_q, only_d):.3f}")


def plot(runs, out):
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(FIG_W / 72, FIG_H / 72))
        ax = fig.add_axes(AXES_RECT)
        for label, _, style, colour in ARMS:
            xs, ys = step_points(runs[label])
            ax.step(xs, ys, where="post", linestyle=style, color=colour, label=label)
        ax.set_xlim(0, HORIZON + 50)
        ax.set_ylim(-0.35, len(SEEDS) + 0.45)
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.set_xlabel("Attack-trace budget B")
        ax.set_ylabel(f"Recovered runs (out of {len(SEEDS)})")
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, LEGEND_ANCHOR_Y), ncol=3,
                  frameon=False)
        fig.savefig(out)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "fig_recovery_budget.pdf")
    args = ap.parse_args()
    runs = {label: load_thresholds(prefix) for label, prefix, _, _ in ARMS}
    print_quoted(runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(runs, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
