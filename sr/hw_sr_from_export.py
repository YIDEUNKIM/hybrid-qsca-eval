"""hw_sr_from_export.py -- exact GE + SR for the ibm_fez runs, from the IBM dashboard export.

The dashboard's per-job download (hardware/exports/job-<id>/, account id removed) holds:
  job-<id>-info.json    submitted pub: transpiled circuit (QPY), 8 observables,
                        (n, 8) angle array, options.default_shots
  job-<id>-result.json  PrimitiveResult: evs (8, n), stds, ensemble_standard_error

Before scoring, three checks tie the export to this package:
  angles   submitted parameter values  vs  local.front(Xa[:n]) of the seed-0 checkpoint
  circuit  transpiled circuit, idle qubits dropped, noiseless on the submitted angles
           and observables  vs  the PennyLane block local.mid on the same angles
  GE       GE curve from the exported evs  vs  archived results/ge_hw_fez_S*.npy
Then GE + SR (srlib.ge_sr) on the device, and on the same pool (first n traces) under
exact simulation (probs_exact_s0.npy from exact.py) and under the seeded fake_sherbrooke
realisations of devnoise.py at the same S, where those exist.

The circuits were serialised with symengine parameter expressions (QPY 10-12), so
decoding needs `symengine<0.14` importable.

  python hw_sr_from_export.py ../hardware/exports/job-*
"""
import argparse, glob, json, os, sys
import numpy as np, torch
from qiskit.circuit import Qubit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import RuntimeDecoder
from srlib import C, RES, OUT, ge_sr, metrics, fmt, load_data, load_state
from qcnn_stage3 import Model

NQ = 8
LOGICAL = ["Z0", "Z2", "Z4", "Z6", "X0", "X2", "X4", "X6"]      # local.mid output order
ARCHIVED = {(512, 10): ("res_HWCAL.json", None),                 # (shots, n) -> archive
            (512, 100): ("res_hw_fez_S512.json", "ge_hw_fez_S512.npy"),
            (256, 650): ("res_hw_fez_S256.json", "ge_hw_fez_S256.npy")}


def load_export(d):
    (info_p,) = glob.glob(os.path.join(d, "job-*-info.json"))
    (res_p,) = glob.glob(os.path.join(d, "job-*-result.json"))
    with open(info_p) as fh:
        info = json.load(fh, cls=RuntimeDecoder)
    with open(res_p) as fh:
        res = json.load(fh, cls=RuntimeDecoder)
    circ, obs, angles, _ = info["params"]["pubs"][0]
    obs = [SparsePauliOp.from_list(list(o[0].items())) for o in obs]     # (8, 1) -> 8
    return dict(job_id=info["id"], backend=info["backend"], created=info["created"],
                shots=int(info["params"]["options"]["default_shots"]),
                circ=circ, obs=obs, angles=np.asarray(angles, dtype=np.float64),
                ev=np.asarray(res[0].data.evs, dtype=np.float64).T,      # (n, 8)
                std=np.asarray(res[0].data.stds, dtype=np.float64).T,
                options=res.metadata, pub_meta=res[0].metadata)


def support(op):
    """(physical qubit, Pauli letter) of a single-qubit Pauli string."""
    (lab,) = op.paulis.to_labels()
    ((q, p),) = [(len(lab) - 1 - i, ch) for i, ch in enumerate(lab) if ch != "I"]
    return q, p


def logical_labels(circ, obs):
    """Observables in virtual-qubit terms, via the layout stored in the QPY."""
    virt = {p: v for v, p in enumerate(circ.layout.final_index_layout())}
    return [f"{p}{virt[q]}" for q, p in map(support, obs)]


def compact(circ, obs):
    """Drop idle qubits so the 156-qubit transpiled circuit fits a statevector."""
    dag = circuit_to_dag(circ)
    idle = [w for w in dag.idle_wires() if isinstance(w, Qubit)]
    keep = [circ.find_bit(q).index for q in circ.qubits if q not in set(idle)]
    assert all(support(o)[0] in keep for o in obs), "observable on an idle qubit"
    dag.remove_qubits(*idle)
    small_obs = []
    for o in obs:
        (lab,) = o.paulis.to_labels()
        small_obs.append(SparsePauliOp("".join(lab[len(lab) - 1 - q] for q in reversed(keep)),
                                       o.coeffs))
    return dag_to_circuit(dag), small_obs, keep


def noiseless_evs(circ, obs, angles):
    small, sobs, keep = compact(circ, obs)
    o = np.empty((len(sobs), 1), dtype=object)      # (8, 1) x (n,) -> (8, n), as submitted
    for i, s in enumerate(sobs):
        o[i, 0] = s
    ev = StatevectorEstimator().run([(small, o, angles)]).result()[0].data.evs
    return np.asarray(ev, dtype=np.float64).T, keep


def head_probs(m, ev):
    """Classical head on measured features, exactly as hardware_attack.py."""
    with torch.no_grad():
        feats = torch.tensor(ev, dtype=torch.float32)
        return torch.softmax(m.fc(m.bn(feats)), 1).numpy().astype(np.float64)


def devnoise_same_pool(S, n, pa, real_key):
    """devnoise.py's seeded realisations (first n of Xa, like the device) rescored on
    the device's pool. At N = n every ordering holds the same traces, so GE_end is
    the rank of the key."""
    out = []
    for p in sorted(glob.glob(os.path.join(OUT, f"curves_devnoise_S{S}_n*_seed*.npz"))):
        probs = np.load(p)["probs"]
        if len(probs) < n:
            continue
        ge, sr, _ = ge_sr(probs[:n], pa[:n], real_key, nmax=n)
        out.append(dict(file=os.path.basename(p), T_ge05=C.recov(ge),
                        GE_end=float(ge[-1]), SR_end=float(sr[-1])))
    return out


def score(job, m, Xa, pa, real_key, probs_exact):
    n, S, tag = len(job["ev"]), job["shots"], f"fez_S{job['shots']}_n{len(job['ev'])}"
    chk = dict(logical_obs=logical_labels(job["circ"], job["obs"]))
    assert chk["logical_obs"] == LOGICAL, chk["logical_obs"]
    with torch.no_grad():
        ang = m.front(torch.tensor(Xa[:n])).numpy().astype(np.float64)
        mid = m.mid(torch.tensor(job["angles"], dtype=torch.float32)).numpy().astype(np.float64)
    chk["max_abs_dangle_vs_front"] = float(np.abs(ang - job["angles"]).max())
    ev0, keep = noiseless_evs(job["circ"], job["obs"], job["angles"])
    chk["physical_qubits"] = keep
    chk["max_abs_dev_transpiled_vs_pennylane"] = float(np.abs(ev0 - mid).max())

    d = job["ev"] - ev0                               # device + shot error, per feature
    noise = dict(rms_dev=float(np.sqrt((d ** 2).mean())),
                 rms_shot_expected=float(np.sqrt(((1 - ev0 ** 2) / S).mean())),
                 mean_reported_std=float(job["std"].mean()),
                 slope_hw_on_exact=float((job["ev"] * ev0).sum() / (ev0 ** 2).sum()),
                 rms_dev_per_obs=dict(zip(LOGICAL, np.sqrt((d ** 2).mean(0)).round(4).tolist())))

    ge, sr, ranks = ge_sr(head_probs(m, job["ev"]), pa[:n], real_key, nmax=n)
    hw = metrics(ge, sr)
    res_json, ge_npy = ARCHIVED.get((S, n), (None, None))
    if ge_npy:
        chk["max_abs_dGE_vs_archived"] = float(np.abs(ge - np.load(os.path.join(RES, ge_npy))).max())
    if res_json:
        a = json.load(open(os.path.join(RES, res_json)))
        chk["archived_recov_05"], chk["archived_ge_min"] = a["recov_05"], a["ge_min"]
    ge_x, sr_x, _ = ge_sr(probs_exact[:n], pa[:n], real_key, nmax=n)
    np.savez(os.path.join(OUT, f"curves_hw_{tag}.npz"), ev=job["ev"], ev_noiseless=ev0,
             ge=ge, sr=sr, ranks=ranks, ge_exact=ge_x, sr_exact=sr_x)
    return tag, dict(job_id=job["job_id"], backend=job["backend"], created=job["created"],
                     shots=S, n_attack=n, runtime_options=job["options"],
                     pub_metadata=job["pub_meta"], checks=chk, noise=noise,
                     hw=hw, exact_same_pool=metrics(ge_x, sr_x),
                     devnoise_same_pool=devnoise_same_pool(S, n, pa, real_key))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="folders holding job-<id>-{info,result}.json")
    args = ap.parse_args()
    torch.set_num_threads(2)                           # as hardware_attack.py
    p_exact = os.path.join(OUT, "probs_exact_s0.npy")
    if not os.path.exists(p_exact):
        sys.exit(f"[error] {p_exact} missing -- run exact.py first")
    probs_exact = np.load(p_exact)
    Xa, pa, real_key, trace_len = load_data()
    m = Model(trace_len, nq=NQ, conv_layers=2, head="quantum", ansatz="qcnn")
    m.load_state_dict(load_state()); m.eval()

    out = {}
    for d in sorted(set(args.dirs)):
        tag, r = score(load_export(d), m, Xa, pa, real_key, probs_exact)
        if tag in out:
            print(f"[skip] {d}: {tag} already scored ({out[tag]['job_id']})"); continue
        out[tag] = r
        c, nz = r["checks"], r["noise"]
        print(f"[{tag}] job {r['job_id']}  obs={','.join(c['logical_obs'])} on {c['physical_qubits']}\n"
              f"  check: max|d angle|={c['max_abs_dangle_vs_front']:.2e}  "
              f"max|d ev| transpiled vs PennyLane={c['max_abs_dev_transpiled_vs_pennylane']:.2e}  "
              f"max|dGE| vs archived={c.get('max_abs_dGE_vs_archived', 'n/a')}\n"
              f"  noise: rms(hw-exact)={nz['rms_dev']:.4f}  shot-only={nz['rms_shot_expected']:.4f}  "
              f"slope={nz['slope_hw_on_exact']:.3f}\n"
              f"  hw    {fmt(r['hw'])}\n  exact {fmt(r['exact_same_pool'])}", flush=True)
        if r["devnoise_same_pool"]:
            g = [x["GE_end"] for x in r["devnoise_same_pool"]]
            print(f"  fake_sherbrooke S={r['shots']}, same pool, {len(g)} seeds: GE(H) {g}  "
                  f"median {np.median(g):g} [{min(g):g}-{max(g):g}]", flush=True)
    with open(os.path.join(OUT, "res_hw_export.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
