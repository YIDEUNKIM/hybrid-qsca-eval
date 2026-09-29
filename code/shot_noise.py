"""
shot_noise.py -- what the trained hybrid attack costs on a sampled device.

Everything in the main campaign uses exact expectation values, which no physical
quantum device provides: a real machine estimates each <Z>/<X> from a finite
number of shots. This script takes a Stage-3 model trained under the campaign
protocol and re-evaluates the SAME weights with the circuit sampled at
S shots per expectation value, then reports key recovery as a function of S.

No retraining happens between shot settings -- the model is trained once, in
simulation, and then deployed onto a sampled readout, which is the situation an
evaluator with simulator-trained weights and real hardware would face.

  python shot_noise.py --seed 0 --shots 128 256 512 1024 2048 4096 8192
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "qcnn"))
import sca_common as C
from qcnn_stage3 import Model, probs_of


def attack_probs_sampled(model, Xa, batch):
    """Forward the attack set through a sampled circuit.

    Double precision is required, not cosmetic: with a float32 statevector the
    derived outcome probabilities sum to 1 only to ~1e-7, and the sampler
    rejects them outright ("probabilities do not sum to 1"). Inference cost is
    unaffected.
    """
    model.eval().double()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xa), batch):
            xb = torch.tensor(Xa[i:i + batch]).double()
            out.append(torch.softmax(model(xb), 1).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["fixed", "variable"], default="fixed")
    ap.add_argument("--data-dir", default="D:/qml-sca-run/data")
    ap.add_argument("--out-dir", default="D:/qml-sca-run/out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--conv-layers", type=int, default=2)
    ap.add_argument("--shots", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--eval-batch", type=int, default=250)
    ap.add_argument("--reps", type=int, default=1,
                    help="independent sampling realisations per shot setting")
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    d = args.dataset[0]
    src = os.path.join(args.out_dir, f"w_s3_{d}_quantum_qcnn_s{args.seed}_w.pt")
    tag = f"shots_{d}_s{args.seed}"

    db = C.db_path(args.data_dir, args.dataset)
    (Xtr, ytr), _, (Xa, pa), real_key = C.load(db, 45000, seed=args.seed, n_val=0)
    trace_len = Xtr.shape[1]
    del Xtr, ytr

    if not os.path.exists(src):
        print(f"[error] no trained checkpoint at {src}.\n"
              f"        Run qcnn_stage3.py --dataset {args.dataset} --seed {args.seed} "
              f"first (the campaign now saves w_*.pt).", flush=True)
        sys.exit(1)
    state = torch.load(src, map_location="cpu")
    print(f"[load] {src}", flush=True)

    results = {}
    # exact reference, same weights
    torch.manual_seed(args.seed)
    exact = Model(trace_len, nq=args.qubits, conv_layers=args.conv_layers,
                  head="quantum", ansatz="qcnn", shots=None)
    exact.load_state_dict(state)
    ge = C.guessing_entropy(probs_of(exact, Xa), pa, real_key)
    r = C.recov(ge)
    results["exact"] = dict(recov_05=r, ge_min=float(ge.min()))
    np.save(os.path.join(args.out_dir, f"ge_{tag}_exact.npy"), ge)
    print(f"[shots=exact] recov[GE<0.5]={r} minGE={ge.min():.3f}", flush=True)

    for S in args.shots:
        t0 = time.time()
        recs = []
        for rep in range(args.reps):
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            # only the measurement RNG varies across reps: same weights, same
            # traces, so the spread is purely the cost of finite sampling
            m = Model(trace_len, nq=args.qubits, conv_layers=args.conv_layers,
                      head="quantum", ansatz="qcnn", shots=S, rng_seed=1000 + rep)
            m.load_state_dict(state)
            probs = attack_probs_sampled(m, Xa, args.eval_batch)
            ge = C.guessing_entropy(probs, pa, real_key)
            recs.append(C.recov(ge))
            if rep == 0:
                np.save(os.path.join(args.out_dir, f"ge_{tag}_S{S}.npy"), ge)
        ok = [v for v in recs if v is not None]
        med = float(np.median(ok)) if ok else None
        results[str(S)] = dict(recov_05=med, reps=recs, n_fail=len(recs) - len(ok),
                               secs=round(time.time() - t0, 1))
        spread = f" [{min(ok)}--{max(ok)}]" if len(ok) > 1 else ""
        print(f"[shots={S:>5}] median recov[GE<0.5]={med}{spread} "
              f"over {args.reps} reps ({(time.time()-t0)/60:.1f} min)", flush=True)

    with open(os.path.join(args.out_dir, f"res_{tag}.json"), "w") as fh:
        json.dump(dict(tag=tag, dataset=args.dataset, seed=args.seed,
                       source=os.path.basename(src), results=results), fh, indent=1)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
