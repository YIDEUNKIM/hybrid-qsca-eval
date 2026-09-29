"""
hardware_attack.py -- run the attack's quantum block on a real device.

The deployment split is the point: the trained model is a classical front-end,
a circuit, and a classical head. Here the front-end and head stay local and only
the circuit is executed on hardware -- which is what an evaluator holding
simulator-trained weights and QPU access would actually do.

  # validate the whole path locally, no account needed
  python hardware_attack.py --backend aer --n-attack 40 --shots 512

  # same code against real hardware (token from the environment, never stored here)
  set QISKIT_IBM_TOKEN=...
  python hardware_attack.py --backend ibm:ibm_sherbrooke --n-attack 520 --shots 512

Cost model: one attack trace = 8 expectation values (Z and X on 4 kept wires);
EstimatorV2 batches all traces of a run into a single job via parameter binding.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "qcnn"))
import sca_common as C
from qcnn_stage3 import Model

NQ = 8
KEPT = [0, 2, 4, 6]


def build_parameterised_circuit():
    """Stage-3 conv/pool ansatz. Inputs are free parameters (one per qubit);
    the trained circuit weights are bound as literals at build time."""
    x = ParameterVector("x", NQ)
    w = ParameterVector("w", 4 * 3 + 4 + 2 * 3)   # conv1, pool1, conv2
    qc = QuantumCircuit(NQ)
    for i in range(NQ):
        qc.ry(x[i], i)
    k = 0
    for a, b in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        qc.ry(w[k], a); qc.ry(w[k + 1], b); qc.cx(a, b); qc.ry(w[k + 2], b); k += 3
    for j, (a, b) in enumerate([(0, 1), (2, 3), (4, 5), (6, 7)]):
        qc.crz(w[k + j], b, a)
    k += 4
    for a, b in [(0, 2), (4, 6)]:
        qc.ry(w[k], a); qc.ry(w[k + 1], b); qc.cx(a, b); qc.ry(w[k + 2], b); k += 3
    return qc, x, w


def flatten_weights(state):
    """TorchLayer stores conv1 (4,3), pool1 (4,), conv2 (2,3) -- flatten in the
    order the circuit above consumes them."""
    c1 = state["mid.conv1"].detach().numpy().reshape(-1)
    p1 = state["mid.pool1"].detach().numpy().reshape(-1)
    c2 = state["mid.conv2"].detach().numpy().reshape(-1)
    return np.concatenate([c1, p1, c2])


def observables():
    """<Z> and <X> on the kept wires, as SparsePauliOp over NQ qubits."""
    obs = []
    for p in ("Z", "X"):
        for wire in KEPT:
            lbl = ["I"] * NQ
            lbl[NQ - 1 - wire] = p          # qiskit is little-endian
            obs.append(SparsePauliOp("".join(lbl)))
    return obs


def get_backend(spec):
    if spec == "aer":
        from qiskit_aer import AerSimulator
        return AerSimulator(), "aer"
    if spec.startswith("fake:"):
        from qiskit_ibm_runtime import fake_provider
        be = getattr(fake_provider, spec.split(":", 1)[1])()
        return be, be.name
    if spec.startswith("ibm:"):
        from qiskit_ibm_runtime import QiskitRuntimeService
        # token from the environment, else from a local json the user placed
        # there for this purpose; it is never printed or logged.
        tok = os.environ.get("QISKIT_IBM_TOKEN")
        if not tok:
            import json as _j
            for c in (os.environ.get("IBM_APIKEY_JSON", ""),
                      os.path.expanduser("~/apikey.json")):
                if os.path.exists(c):
                    tok = _j.load(open(c, encoding="utf-8"))["apikey"]
                    break
        svc = QiskitRuntimeService(channel="ibm_cloud", token=tok)
        be = svc.backend(spec.split(":", 1)[1])
        return be, be.name
    raise ValueError(f"unknown backend spec {spec!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["fixed", "variable"], default="fixed")
    ap.add_argument("--data-dir", default="D:/qml-sca-run/data")
    ap.add_argument("--out-dir", default="D:/qml-sca-run/out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="aer",
                    help="aer | fake:FakeSherbrooke | ibm:<backend_name>")
    ap.add_argument("--shots", type=int, default=512)
    ap.add_argument("--n-attack", type=int, default=520)
    ap.add_argument("--opt-level", type=int, default=2)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    torch.set_num_threads(2)
    d = args.dataset[0]
    src = os.path.join(args.out_dir, f"w_s3_{d}_quantum_qcnn_s{args.seed}_w.pt")
    if not os.path.exists(src):
        print(f"[error] no checkpoint at {src}"); sys.exit(1)
    state = torch.load(src, map_location="cpu", weights_only=True)

    db = C.db_path(args.data_dir, args.dataset)
    (Xtr, ytr), _, (Xa, pa), real_key = C.load(db, 45000, seed=args.seed, n_val=0)
    trace_len = Xtr.shape[1]
    del Xtr, ytr
    n = min(args.n_attack, len(Xa))
    Xa, pa = Xa[:n], pa[:n]

    # local half of the model, used both to produce the circuit inputs and to
    # consume its outputs
    local = Model(trace_len, nq=NQ, conv_layers=2, head="quantum", ansatz="qcnn")
    local.load_state_dict(state); local.eval()
    with torch.no_grad():
        angles = local.front(torch.tensor(Xa)).numpy().astype(np.float64)   # (n, 8)
    print(f"[local] front-end produced {angles.shape} angles in "
          f"[{angles.min():.3f}, {angles.max():.3f}]", flush=True)

    qc, x, w = build_parameterised_circuit()
    wv = flatten_weights(state)
    qc = qc.assign_parameters({w: wv})
    be, bename = get_backend(args.backend)
    tqc = transpile(qc, backend=be, optimization_level=args.opt_level, seed_transpiler=0)
    two = sum(v for g, v in tqc.count_ops().items()
              if g in ("cx", "cz", "ecr", "crz", "rzz"))
    print(f"[backend] {bename}: transpiled depth {tqc.depth()}, 2q gates {two}", flush=True)

    obs = observables()
    obs_t = [o.apply_layout(tqc.layout) for o in obs] if tqc.layout else obs
    # EstimatorV2 broadcasts the observable shape against the parameter shape.
    # Observables are (8,) and the angle array is (n, 8) -> parameter shape (n,),
    # which does not broadcast against (8,). Reshaping observables to (8, 1)
    # gives (8, 1) x (n,) -> (8, n): every observable on every attack trace.
    # (np.array would iterate into each SparsePauliOp and yield its coefficients,
    # so fill an empty object array instead.)
    _o = np.empty((len(obs_t), 1), dtype=object)
    for _i, _ob in enumerate(obs_t):
        _o[_i, 0] = _ob
    obs_t = _o

    t0 = time.time()
    if args.backend.startswith("ibm:"):
        from qiskit_ibm_runtime import EstimatorV2
        # Job mode, not Session: the IBM open plan forbids sessions.
        est = EstimatorV2(mode=be)
        est.options.default_shots = args.shots
        job = est.run([(tqc, obs_t, angles)])
        print(f"[job] submitted id={job.job_id()}", flush=True)
        res_ = job.result()
        ev = np.array(res_[0].data.evs).T                      # -> (n, 8)
        try:
            qs = job.metrics()["usage"]["quantum_seconds"]
            print(f"[usage] QPU time consumed: {qs:.1f} s", flush=True)
        except Exception:
            pass
    else:
        from qiskit.primitives import BackendEstimatorV2
        est = BackendEstimatorV2(backend=be)
        est.options.default_precision = 1.0 / np.sqrt(args.shots)
        job = est.run([(tqc, obs_t, angles)])
        ev = np.array(job.result()[0].data.evs).T
    dt = time.time() - t0
    print(f"[run] {ev.shape} expectation values in {dt/60:.2f} min", flush=True)

    # classical head, on the hardware-measured features
    with torch.no_grad():
        feats = torch.tensor(ev, dtype=torch.float32)
        probs = torch.softmax(local.fc(local.bn(feats)), 1).numpy().astype(np.float64)

    ge = C.guessing_entropy(probs, pa, real_key, nmax=n)
    r = C.recov(ge)
    tag = args.tag or f"hw_{d}_{bename}_S{args.shots}"
    np.save(os.path.join(args.out_dir, f"ge_{tag}.npy"), ge)
    with open(os.path.join(args.out_dir, f"res_{tag}.json"), "w") as fh:
        json.dump(dict(tag=tag, backend=bename, shots=args.shots, n_attack=n,
                       depth=tqc.depth(), two_qubit=two, recov_05=r,
                       ge_min=float(ge.min()), minutes=round(dt / 60, 2),
                       total_shots=n * 8 * args.shots), fh, indent=1)
    print(f"[GE] {tag} recov[GE<0.5]={r} minGE={ge.min():.3f} over {n} traces | "
          f"{n*8*args.shots:,} shots", flush=True)


if __name__ == "__main__":
    main()
