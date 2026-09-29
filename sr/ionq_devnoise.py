"""ionq_devnoise.py -- Table 8 row for a second vendor: S-shot sampling plus IonQ's own
Forte noise model, executed on the official IonQ cloud simulator (backend "simulator",
noise {model, seed}). Nothing is sent to a QPU and nothing is charged.

Protocol, matched to the Table 8 device-noise row (device_noise.py / devnoise.py):
  * seed-0 Stage-3 weights (checkpoint/w_s3_f_quantum_qcnn_s0_w.pt); nothing retrained
  * first N=1500 attack traces in file order; standardisation from the seed-0 45k subset
  * two measurement circuits per trace: joint Z on the kept wires {0,2,4,6} and joint X
    (H then measure); each circuit sampled with S shots -> S shots per expectation value
  * features Z0,Z2,Z4,Z6,X0,X2,X4,X6 -> float32 BatchNorm + head -> softmax -> ge_sr(nmax=N)
  * five seeded realisations per S; the cloud noise seed is derived per chunk from
    (BASE_SEED, S, rep, chunk) and recorded

Circuits are submitted as logical QIS gates (RY / CNOT / CRZ / H) without local
transpilation; the IonQ cloud compiler maps them to the native gate set. Debiasing is off.
Submission goes through qml_sca.ionq.simulate_measurements (vendored in sr/qml_sca;
journaled, one POST per chunk, resumable): re-running the same command never
duplicates a cloud job.

  ASCAD_H5=/path/ASCAD.h5 python ionq_devnoise.py \
      --env-file <file with IONQ_API_KEY> --noise-model forte-enterprise-1 \
      --shots 512 2048 --seeds 1 2 3 4 5
  python ionq_devnoise.py --dry-run          # build + parity checks only, no network
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from qiskit.circuit import ClassicalRegister
from qiskit.quantum_info import Statevector

from srlib import OUT, ge_sr, metrics, fmt, load_data, load_state
from qcnn_stage3 import Model
from hardware_attack import build_parameterised_circuit, flatten_weights, NQ, KEPT

import qml_sca.ionq as ionq
from qml_sca.ionq import CloudError, JobPending, SubmissionAmbiguous, simulate_measurements
from qml_sca.circuits import counts_to_expectations, expectations_from_counts

BASE_SEED = 20260922
NOISE_MODELS = ("forte-1", "forte-enterprise-1", "aria-1", "aria-2")
PARITY_TOL = 1e-6
FEATURES = 2 * len(KEPT)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noise-model", default="forte-enterprise-1", choices=NOISE_MODELS)
    ap.add_argument("--shots", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                    help="realisation index; the cloud noise seed is derived from it")
    ap.add_argument("--n-attack", type=int, default=1500)
    ap.add_argument("--chunk-traces", type=int, default=250, help="traces per cloud job (2 circuits each)")
    ap.add_argument("--env-file", default=None, help="file holding IONQ_API_KEY (else the environment)")
    ap.add_argument("--poll-seconds", type=float, default=20.0)
    ap.add_argument("--wait-minutes", type=float, default=240.0, help="give up (resumably) after this")
    ap.add_argument("--parity-traces", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true", help="build circuits and run parity checks only")
    ap.add_argument("--out-dir", default=OUT, help="results/journal directory (default: sr/results)")
    args = ap.parse_args()
    if args.n_attack < 1 or args.chunk_traces < 1 or any(s < 1 for s in args.shots):
        ap.error("--n-attack, --chunk-traces and --shots must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        ap.error("--seeds must be distinct nonnegative integers")
    return args


def chunk_seed(shots, rep, chunk):
    """Distinct, reproducible cloud noise seed in [1, 2**31) per (S, realisation, chunk)."""
    state = np.random.SeedSequence([BASE_SEED, shots, rep, chunk]).generate_state(1, dtype=np.uint32)[0]
    return int(state % (2**31 - 1)) + 1


def measurement_templates(bound_template):
    """Z and X readout copies of the weight-bound template; classical bit j <- wire KEPT[j]."""
    out = {}
    for basis in ("Z", "X"):
        qc = bound_template.copy(name=f"qcnn_{basis}")
        qc.add_register(ClassicalRegister(len(KEPT), "readout"))
        if basis == "X":
            for wire in KEPT:
                qc.h(wire)
        for index, wire in enumerate(KEPT):
            qc.measure(wire, index)
        out[basis] = qc
    return out


def bind(template, x, angles):
    return template.assign_parameters({x: [float(a) for a in angles]})


def exact_features_qiskit(bound_template, x, angles):
    """Exact Z/X expectations on KEPT, computed through the same bit convention as the decoder:
    Statevector.probabilities_dict(qargs=KEPT) keys bit j (from the right) with KEPT[j], and
    counts_to_expectations reads bit j of int(key, 2) -- exactly what decode_counts produces."""
    with_h = bound_template.copy()
    for wire in KEPT:
        with_h.h(wire)
    out = np.empty((len(angles), FEATURES))
    for i, a in enumerate(angles):
        pz = Statevector(bind(bound_template, x, a)).probabilities_dict(qargs=list(KEPT))
        px = Statevector(bind(with_h, x, a)).probabilities_dict(qargs=list(KEPT))
        out[i] = np.concatenate([counts_to_expectations(pz, len(KEPT)),
                                 counts_to_expectations(px, len(KEPT))])
    return out


def exact_features_pennylane(state, trace_len, angles):
    """The collaborator's own PennyLane model, in double precision, on the same angles."""
    ref = Model(trace_len, nq=NQ, conv_layers=2, head="quantum", ansatz="qcnn")
    ref.load_state_dict(state)
    ref.eval().double()
    with torch.no_grad():
        return ref.mid(torch.tensor(angles, dtype=torch.float64)).numpy()


def head_probabilities(local, ev):
    """Identical to devnoise.py / hardware_attack.py: float32 BN + head, float64 softmax output."""
    with torch.no_grad():
        feats = torch.tensor(ev, dtype=torch.float32)
        return torch.softmax(local.fc(local.bn(feats)), 1).numpy().astype(np.float64)


def run_realisation(args, S, rep, templates, x, angles, n, local, pa, real_key, journal_root):
    tag = f"{args.noise_model}_S{S}_n{n}_seed{rep}"
    jdir = os.path.join(journal_root, f"S{S}_rep{rep}")
    os.makedirs(jdir, exist_ok=True)
    starts = list(range(0, n, args.chunk_traces))
    done = {}
    deadline = time.time() + args.wait_minutes * 60
    t0 = time.time()
    while len(done) < len(starts):
        for k, start in enumerate(starts):
            if k in done:
                continue
            stop = min(start + args.chunk_traces, n)
            circuits = []
            for i in range(start, stop):
                circuits.append(bind(templates["Z"], x, angles[i]))
                circuits.append(bind(templates["X"], x, angles[i]))
            seed = chunk_seed(S, rep, k)
            path = os.path.join(jdir, f"chunk_{k:03d}.json")
            try:
                counts, meta = simulate_measurements(
                    circuits, "ionq_forte1_cloud", S, seed, journal_path=path,
                    env_file=args.env_file, timeout=0, poll_interval=5)
            except JobPending:
                continue
            except SubmissionAmbiguous as err:
                raise SystemExit(f"[{tag}] chunk {k}: {err}\nLocate the job by its name in IonQ "
                                 "Cloud and resume with adopt_job_id; do not resubmit.")
            except CloudError as err:
                print(f"[{tag}] chunk {k}: transient cloud error, will retry: {err}", flush=True)
                continue
            reported = (meta.get("noise_reported") or {}).get("model")
            if reported not in (None, args.noise_model):
                raise SystemExit(f"[{tag}] chunk {k}: service reports noise model {reported!r}")
            if len(counts) != 2 * (stop - start):
                raise SystemExit(f"[{tag}] chunk {k}: {len(counts)} histograms for {stop - start} traces")
            done[k] = (counts, meta, seed)
            print(f"[{tag}] chunk {k + 1}/{len(starts)} complete, job {meta['job_id']} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if len(done) < len(starts):
            if time.time() > deadline:
                raise SystemExit(f"[{tag}] {len(starts) - len(done)} chunks still pending; "
                                 "re-run the same command to resume from the journals.")
            time.sleep(args.poll_seconds)

    ev = np.empty((n, FEATURES))
    chunk_meta = []
    for k, start in enumerate(starts):
        counts, meta, seed = done[k]
        stop = min(start + args.chunk_traces, n)
        for j, i in enumerate(range(start, stop)):
            ev[i] = expectations_from_counts(counts[2 * j], counts[2 * j + 1], len(KEPT))
        chunk_meta.append({
            "chunk": k, "traces": [start, stop], "noise_seed": seed, "job_id": meta["job_id"],
            "child_jobs": len(meta.get("child_job_ids") or []),
            "noise_reported": meta.get("noise_reported"), "settings_reported": meta.get("settings_reported"),
            "execution_duration_ms": meta.get("execution_duration_ms"),
            "submitted_at": meta.get("submitted_at"), "completed_at": meta.get("completed_at"),
        })
    probs = head_probabilities(local, ev)
    ge, sr, ranks = ge_sr(probs, pa, real_key, nmax=n)
    mt = metrics(ge, sr)
    mt.update(S=S, n=n, rep=rep, noise_model=args.noise_model, backend="simulator",
              minutes=round((time.time() - t0) / 60, 2), chunks=chunk_meta)
    ge100 = "n/a" if mt["GE_100"] is None else f"{mt['GE_100']:.2f}"
    print(f"[{tag}] " + fmt(mt) + f"  GE(100)={ge100}", flush=True)
    np.savez(os.path.join(args.out_dir, f"curves_ionq_{tag}.npz"), ge=ge, sr=sr, ranks=ranks, ev=ev, probs=probs)
    return tag, mt


def main():
    args = parse_args()
    torch.set_num_threads(3)
    ionq.NOISE_MODEL = args.noise_model  # the frozen package hard-codes forte-1; select the vendor model here
    assert ionq.BACKEND == "simulator"

    Xa, pa, real_key, trace_len = load_data()
    n = min(args.n_attack, len(Xa))
    Xa, pa = Xa[:n], pa[:n]
    state = load_state()
    local = Model(trace_len, nq=NQ, conv_layers=2, head="quantum", ansatz="qcnn")
    local.load_state_dict(state)
    local.eval()
    with torch.no_grad():
        angles = local.front(torch.tensor(Xa)).numpy().astype(np.float64)
    print(f"[local] {angles.shape} angles in [{angles.min():.3f}, {angles.max():.3f}]", flush=True)

    qc, x, w = build_parameterised_circuit()
    bound = qc.assign_parameters({w: flatten_weights(state)})
    templates = measurement_templates(bound)
    ops_z, ops_x = dict(templates["Z"].count_ops()), dict(templates["X"].count_ops())
    print(f"[circuit] Z ops {ops_z} | X ops {ops_x}", flush=True)

    # Parity 1: bound Qiskit circuit == collaborator's PennyLane model (exact expectation values).
    # Parity 2 (implicit): the decoder bit convention reproduces those values from probabilities.
    m = min(args.parity_traces, n)
    fq = exact_features_qiskit(bound, x, angles[:m])
    fp = exact_features_pennylane(state, trace_len, angles[:m])
    gap = float(np.abs(fq - fp).max())
    print(f"[parity] {m} traces, max |qiskit - pennylane| = {gap:.2e}", flush=True)
    if gap > PARITY_TOL:
        raise SystemExit("Circuit/decoder parity failed; nothing submitted.")

    # Serialization check on one chunk, offline.
    sample = [bind(templates[b], x, angles[i]) for i in range(min(args.chunk_traces, n)) for b in ("Z", "X")]
    payload, readouts = ionq.build_payload(sample, args.shots[0], 1)
    print(f"[payload] {len(sample)} circuits, {sum(len(e['circuit']) for e in payload['input']['circuits'])} gates, "
          f"noise {payload['noise']['model']}, debiasing {payload['settings']['error_mitigation']['debiasing']}, "
          f"readout map {readouts[0]['classical_to_qubit']}", flush=True)
    if any(r["classical_to_qubit"] != list(KEPT) for r in readouts):
        raise SystemExit("Readout map differs from KEPT; nothing submitted.")
    if args.dry_run:
        print("[dry-run] no cloud calls made")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    journal_root = os.path.join(args.out_dir, "ionq", args.noise_model)
    res_path = os.path.join(args.out_dir, f"res_ionq_{args.noise_model}.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for S in args.shots:
        for rep in args.seeds:
            tag = f"{args.noise_model}_S{S}_n{n}_seed{rep}"
            if tag in res and os.path.exists(os.path.join(args.out_dir, f"curves_ionq_{tag}.npz")):
                print(f"[{tag}] already complete: " + fmt(res[tag]), flush=True)
                continue
            tag, mt = run_realisation(args, S, rep, templates, x, angles, n, local, pa, real_key, journal_root)
            res[tag] = mt
            json.dump(res, open(res_path, "w"), indent=1)


if __name__ == "__main__":
    main()
