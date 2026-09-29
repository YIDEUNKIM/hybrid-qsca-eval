"""check_tables.py -- recompute the execution tables (Tables 7-9) and the numbers
quoted with them from the result files in sr/results.

  python analysis/check_tables.py

The exact 1500-trace reference of Table 8 is recomputed from
sr/results/probs_exact_s0.npy when that file exists (written by sr/exact.py);
this needs the ASCAD fixed-key file (ASCAD_H5 or data/ASCAD.h5).
"""
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "sr", "results")


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def med_range(values):
    v = sorted(values)
    return f"{st.median(v):g} [{v[0]:g}-{v[-1]:g}]"


def table7():
    exact = load("res_exact.json")["pool10000_H3000"]
    shots = load("res_shots.json")
    print("Table 7: full 10k pool, W=3000, five sampling realizations")
    print(f"  exact  T_GE={exact['T_ge05']}  T_SR>=0.9={exact['T_sr90']}  T_SR=1={exact['T_sr100']}")
    for S in sorted(shots, key=int, reverse=True):
        t = [r["T_ge05"] for r in shots[S]]
        s = [r["T_sr90"] for r in shots[S]]
        tm, sm = st.median(t), st.median(s)
        print(f"  S={S:>5}  T_GE={med_range(t):<18} {tm / exact['T_ge05']:.2f}x  "
              f"T_SR>=0.9={med_range(s):<20} {sm / exact['T_sr90']:.2f}x  "
              f"2*S*T_GE={2 * int(S) * tm / 1e6:.2f} M")
    sr_at = [r["SR_at_T_ge05"] for recs in shots.values() for r in recs]
    print(f"  SR(T_GE) over {len(sr_at)} realizations: {min(sr_at):.2f}-{max(sr_at):.2f}")
    t1 = [r["T_sr100"] for r in shots["512"]]
    if None in t1:
        print(f"  S=512 T_SR=1: {t1.count(None)} realization(s) never reach SR=1")
    else:
        print(f"  S=512 T_SR=1: {med_range(t1)}")


def exact_prefix(n):
    """T_GE<0.5 of the exact checkpoint scored on the first n attack traces, W=n."""
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "sr"))
    import srlib as L
    probs = np.load(os.path.join(RES, "probs_exact_s0.npy"))
    _, pa, key, _ = L.load_data()
    ge, sr, _ = L.ge_sr(probs[:n], pa[:n], key, nmax=n)
    return L.metrics(ge, sr)["T_ge05"]


def table8():
    pool = load("res_shots_pool1500.json")
    ibm = load("res_devnoise.json")
    ionq = load("res_ionq_forte-enterprise-1.json")
    print("Table 8: first 1500 attack traces, W=1500, five realizations")
    if os.path.exists(os.path.join(RES, "probs_exact_s0.npy")):
        print(f"  exact  T_GE={exact_prefix(1500)}")
    else:
        print("  exact  (run sr/exact.py first to recompute this row)")
    for S in ("512", "2048"):
        rows = {
            "sampling only": [r["T_ge05"] for r in pool[S]],
            "+ IBM model": [v["T_ge05"] for k, v in ibm.items() if k.startswith(f"S{S}_")],
            "+ IonQ model": [v["T_ge05"] for k, v in ionq.items() if f"_S{S}_" in k],
        }
        for label, t in rows.items():
            print(f"  S={S:>4}  {label:<14} T_GE={med_range(t)}  (n={len(t)})")


def table9():
    hw = load("res_hw_export.json")
    exact = load("res_exact.json")
    fez100, fez650 = hw["fez_S512_n100"], hw["fez_S256_n650"]
    ranks = [d["GE_end"] for d in fez100["devnoise_same_pool"]]
    print("Table 9: hardware and matched simulation, W equal to the pool")
    print(f"  100 traces, S=512: device-noise rank {med_range(ranks)}  "
          f"ibm_fez rank {fez100['hw']['GE_end']:g}  exact rank {fez100['exact_same_pool']['GE_end']:g}")
    e, h = fez650["exact_same_pool"], fez650["hw"]
    samp = load("res_shots_pool650.json")["256"]
    print(f"  650 traces: exact T_GE={e['T_ge05']} (SR {e['SR_at_T_ge05']}) T_SR>=0.9={e['T_sr90']}")
    print(f"              sampling S=256 T_GE={med_range([r['T_ge05'] for r in samp])} "
          f"T_SR>=0.9={med_range([r['T_sr90'] for r in samp])}")
    print(f"              ibm_fez S=256 T_GE={h['T_ge05']} (SR {h['SR_at_T_ge05']}) T_SR>=0.9={h['T_sr90']}")
    print(f"              ratios hw/exact: T_GE {h['T_ge05'] / e['T_ge05']:.2f}, "
          f"T_SR>=0.9 {h['T_sr90'] / e['T_sr90']:.2f}")
    pools = ", ".join(f"{k}: {v['T_ge05']}" for k, v in exact.items())
    print(f"  exact thresholds by pool: {pools}")


if __name__ == "__main__":
    table7()
    table8()
    table9()
