"""devnoise.py -- fresh, SEEDED realisations of the device-noise experiment
(FakeSherbrooke calibration noise + S-shot sampling) through hardware_attack.py's
own circuit / observable / estimator path, with GE and SR reported together.

The archived device_noise.py run was unseeded and went through pennylane-qiskit,
so this is an independent draw of the same experiment, not a replay.

  python devnoise.py --shots 512 2048 --seeds 1 2 3 4 5 --n-attack 1500
"""
import os, json, time, argparse
import numpy as np, torch
from srlib import C, OUT, ge_sr, metrics, fmt, load_data, load_state
from qcnn_stage3 import Model
from hardware_attack import build_parameterised_circuit, flatten_weights, observables, NQ
from qiskit import transpile
from qiskit.primitives import BackendEstimatorV2
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

ap = argparse.ArgumentParser()
ap.add_argument("--shots", type=int, nargs="+", default=[512, 2048])
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
ap.add_argument("--n-attack", type=int, default=1500)
args = ap.parse_args()

torch.set_num_threads(3)
Xa, pa, real_key, trace_len = load_data()
n = min(args.n_attack, len(Xa)); Xa, pa = Xa[:n], pa[:n]
state = load_state()
local = Model(trace_len, nq=NQ, conv_layers=2, head="quantum", ansatz="qcnn")
local.load_state_dict(state); local.eval()
with torch.no_grad():
    angles = local.front(torch.tensor(Xa)).numpy().astype(np.float64)

qc, x, w = build_parameterised_circuit()
qc = qc.assign_parameters({w: flatten_weights(state)})
be = FakeSherbrooke()
tqc = transpile(qc, backend=be, optimization_level=2, seed_transpiler=0)
two = sum(v for g, v in tqc.count_ops().items() if g in ("cx", "cz", "ecr", "crz", "rzz"))
print(f"[backend] {be.name}: depth {tqc.depth()}, 2q {two}, ops {dict(tqc.count_ops())}", flush=True)
obs_t = [o.apply_layout(tqc.layout) for o in observables()]
_o = np.empty((len(obs_t), 1), dtype=object)
for i, ob in enumerate(obs_t):
    _o[i, 0] = ob

p = os.path.join(OUT, "res_devnoise.json")
res = json.load(open(p)) if os.path.exists(p) else {}
for S in args.shots:
    for seed in args.seeds:
        est = BackendEstimatorV2(backend=be)
        est.options.default_precision = 1.0 / np.sqrt(S)
        est.options.seed_simulator = seed
        t0 = time.time()
        ev = np.array(est.run([(tqc, _o, angles)]).result()[0].data.evs).T
        with torch.no_grad():
            probs = torch.softmax(local.fc(local.bn(torch.tensor(ev, dtype=torch.float32))), 1) \
                         .numpy().astype(np.float64)
        ge, sr, ranks = ge_sr(probs, pa, real_key, nmax=n)
        mt = metrics(ge, sr)
        mt.update(S=S, n=n, sim_seed=seed, depth=tqc.depth(), two_qubit=two, minutes=round((time.time()-t0)/60, 2))
        print(f"[devnoise S={S} seed={seed}] " + fmt(mt) + f"  GE(100)={mt['GE_100']:.2f}", flush=True)
        np.savez(os.path.join(OUT, f"curves_devnoise_S{S}_n{n}_seed{seed}.npz"),
                 ge=ge, sr=sr, ranks=ranks, ev=ev, probs=probs)
        res[f"S{S}_n{n}_seed{seed}"] = mt
        json.dump(res, open(p, "w"), indent=1)
