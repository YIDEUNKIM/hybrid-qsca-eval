"""shots.py -- replay of shot_noise.py (same weights, default.qubit sampled at S
shots, measurement seed 1000+rep, eval batch 250) with the SR column added.
Every rep's T_GE<0.5 is checked against res_shots_f_s0.json and the rep-0 GE
curves against ge_shots_f_s0_S*.npy.

  python shots.py                     # all 7 settings x 5 reps, 10000 pool / H=3000
  python shots.py --pool 1500         # rescored on the device-noise pool (Table 8)
"""
import os, sys, json, time, argparse
import numpy as np, torch
from srlib import C, RES, OUT, ge_sr, metrics, fmt, load_data, load_state
from qcnn_stage3 import Model
from shot_noise import attack_probs_sampled

ap = argparse.ArgumentParser()
ap.add_argument("--shots", type=int, nargs="+", default=[8192, 4096, 2048, 1024, 512, 256, 128])
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--pool", type=int, default=None, help="score only the first POOL traces, nmax=POOL")
args = ap.parse_args()

torch.set_num_threads(3)
Xa, pa, real_key, trace_len = load_data()
state = load_state()
stored = json.load(open(os.path.join(RES, "res_shots_f_s0.json")))["results"]
suffix = f"_pool{args.pool}" if args.pool else ""
res = {}
for S in args.shots:
    res[str(S)] = []
    for rep in range(args.reps):
        t0 = time.time()
        torch.manual_seed(0); np.random.seed(0)
        m = Model(trace_len, nq=8, conv_layers=2, head="quantum", ansatz="qcnn",
                  shots=S, rng_seed=1000 + rep)
        m.load_state_dict(state)
        probs = attack_probs_sampled(m, Xa, 250)        # full 10k pass: same RNG stream
        if args.pool:
            ge, sr, ranks = ge_sr(probs[:args.pool], pa[:args.pool], real_key, nmax=args.pool)
        else:
            ge, sr, ranks = ge_sr(probs, pa, real_key)
        mt = metrics(ge, sr); mt["rep"] = rep
        line = f"[S={S} rep={rep}{suffix}] " + fmt(mt)
        if not args.pool:
            mt["archived_T_ge05"] = stored[str(S)]["reps"][rep]
            line += f"  | archived T={mt['archived_T_ge05']}"
            if rep == 0:
                g0 = np.load(os.path.join(RES, f"ge_shots_f_s0_S{S}.npy"))
                mt["max_abs_dGE_vs_archived"] = float(np.abs(ge - g0).max())
                line += f" max|dGE|={mt['max_abs_dGE_vs_archived']:.4f}"
        print(line + f"  ({time.time()-t0:.0f}s)", flush=True)
        np.savez(os.path.join(OUT, f"curves_shots_S{S}_r{rep}{suffix}.npz"), ge=ge, sr=sr, ranks=ranks)
        res[str(S)].append(mt)
        json.dump(res, open(os.path.join(OUT, f"res_shots{suffix}.json"), "w"), indent=1)
