"""exact.py -- S=inf re-evaluation of the seed-0 checkpoint on the three pools
the paper scores (10000/3000, 3000/3000, 650/650). Reproduces 332 / 296 / 163
bit-for-bit against ge_shots_f_s0_exact.npy and ge_exact_f_first650.npy, and
adds the SR column."""
import os, json
import numpy as np, torch
from srlib import C, RES, OUT, ge_sr, metrics, fmt, load_data, load_state
from qcnn_stage3 import Model, probs_of

torch.set_num_threads(3)
Xa, pa, real_key, trace_len = load_data()
torch.manual_seed(0)
m = Model(trace_len, nq=8, conv_layers=2, head="quantum", ansatz="qcnn", shots=None)
m.load_state_dict(load_state())
probs = probs_of(m, Xa)
np.save(os.path.join(OUT, "probs_exact_s0.npy"), probs)

res = {}
for name, n_pool, nmax, ref in [("pool10000_H3000", None, 3000, "ge_shots_f_s0_exact.npy"),
                                ("pool3000_H3000", 3000, 3000, None),
                                ("pool650_H650", 650, 650, "ge_exact_f_first650.npy")]:
    P, p = (probs, pa) if n_pool is None else (probs[:n_pool], pa[:n_pool])
    ge, sr, ranks = ge_sr(P, p, real_key, nmax=nmax)
    mt = metrics(ge, sr)
    line = f"[exact {name}] " + fmt(mt)
    if ref:
        g0 = np.load(os.path.join(RES, ref))
        mt["max_abs_dGE_vs_archived"] = float(np.abs(ge - g0).max())
        line += f"  | vs archived {ref}: max|dGE|={mt['max_abs_dGE_vs_archived']:.4f}"
    print(line, flush=True)
    np.savez(os.path.join(OUT, f"curves_exact_{name}.npz"), ge=ge, sr=sr, ranks=ranks)
    res[name] = mt
json.dump(res, open(os.path.join(OUT, "res_exact.json"), "w"), indent=1)
