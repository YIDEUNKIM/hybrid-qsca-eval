"""
qcnn_stage3.py -- Stage 3 of the storyline: learned 1D-conv front-end -> angle
encoding -> quantum block -> classical head, plus its parameter-matched
classical control.

Supersedes qcnn_convfront.py / qcnn_convfront_sel.py / qcnn_scan.py /
qcnn_scan_sel.py, which were four near-identical copies. Model code is verbatim
from those files; what is new is:
  --seed          explicit seed so results are a median over repeats
  --head          quantum | classical  (the parameter-matched ablation)
  --ansatz        qcnn | sel           (conv/pool QCNN vs StronglyEntanglingLayers)
  early stopping  now scored on a PROFILING-side validation split (R5), not on
                  attack traces with the real key as the original scan did.

Data / label / GE / recovery criterion all come from sca_common so that every
model in the campaign provably shares them.
"""
import argparse, copy, json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml

import sca_common as C


# ------------------------------------------------- conv front-end (verbatim) --
class ConvFront(nn.Module):
    """Configurable-depth 1D-conv: full window -> NQ angle features in [0, pi]."""
    def __init__(self, trace_len, nq, n_layers=2, n_filt=16):
        super().__init__()
        blocks = []
        in_ch = 1
        for _ in range(n_layers):
            blocks += [nn.Conv1d(in_ch, n_filt, 11, padding=5), nn.ReLU(), nn.AvgPool1d(4)]
            in_ch = n_filt
        blocks += [nn.Flatten()]
        self.net = nn.Sequential(*blocks)
        with torch.no_grad():
            d = self.net(torch.zeros(1, 1, trace_len)).shape[1]
        self.proj = nn.Linear(d, nq)

    def forward(self, x):
        return torch.sigmoid(self.proj(self.net(x.unsqueeze(1)))) * np.pi


# ------------------------------------- quantum blocks (verbatim from originals) --
def make_qcnn(nq, shots=None, rng_seed=None):
    """conv/pool QCNN ansatz (Cong et al.), as in qcnn_convfront.py.
    shots=None gives exact expectation values; an integer draws that many
    samples per expectation value, as a real device would."""
    dev = qml.device("default.qubit", wires=nq, shots=shots, seed=rng_seed)

    def conv(w, wires):
        P = [(wires[i], wires[i + 1]) for i in range(0, len(wires) - 1, 2)]
        for j, (a, b) in enumerate(P):
            qml.RY(w[j, 0], wires=a); qml.RY(w[j, 1], wires=b)
            qml.CNOT(wires=[a, b]); qml.RY(w[j, 2], wires=b)

    def pool(w, wires):
        P = [(wires[i], wires[i + 1]) for i in range(0, len(wires) - 1, 2)]; k = []
        for j, (a, b) in enumerate(P):
            qml.CRZ(w[j], wires=[b, a]); k.append(a)
        return k

    n1 = nq // 2
    nk = len(range(0, nq - 1, 2))
    n2 = nk // 2

    @qml.qnode(dev, interface="torch",
               diff_method=("backprop" if shots is None else "parameter-shift"))
    def circ(inputs, conv1, pool1, conv2):
        for i in range(nq):
            qml.RY(inputs[..., i], wires=i)
        conv(conv1, list(range(nq)))
        kept = pool(pool1, list(range(nq)))
        conv(conv2, kept)
        return [qml.expval(qml.PauliZ(w)) for w in kept] + \
               [qml.expval(qml.PauliX(w)) for w in kept]

    shapes = {"conv1": (n1, 3), "pool1": (n1,), "conv2": (max(n2, 1), 3)}
    return qml.qnn.TorchLayer(circ, shapes), 2 * nk


def make_sel(nq, q_layers=6, shots=None, rng_seed=None):
    """StronglyEntanglingLayers ansatz, as in qcnn_convfront_sel.py."""
    dev = qml.device("default.qubit", wires=nq, shots=shots, seed=rng_seed)

    @qml.qnode(dev, interface="torch",
               diff_method=("backprop" if shots is None else "parameter-shift"))
    def circ(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(nq), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(nq))
        return [qml.expval(qml.PauliZ(w)) for w in range(nq)] + \
               [qml.expval(qml.PauliX(w)) for w in range(nq)]

    return qml.qnn.TorchLayer(circ, {"weights": (q_layers, nq, 3)}), 2 * nq


class Model(nn.Module):
    def __init__(self, trace_len, nq=8, conv_layers=2, n_filt=16,
                 head="quantum", ansatz="qcnn", control="wide", shots=None,
                 rng_seed=None):
        super().__init__()
        self.front = ConvFront(trace_len, nq, n_layers=conv_layers, n_filt=n_filt)
        if head == "quantum":
            self.mid, feat = (make_qcnn(nq, shots=shots, rng_seed=rng_seed)
                              if ansatz == "qcnn" else make_sel(nq, shots=shots, rng_seed=rng_seed))
        else:
            # Classical control: ONLY this block differs from the quantum run.
            #
            # "wide" is the control as written in the original code. Note it is
            # NOT parameter-matched: for nq=8/qcnn it holds 144 trainable
            # parameters against the circuit's 22. A dense R^nq -> R^feat block
            # cannot in fact be made as small as the circuit while keeping the
            # same input/output width, which is itself worth reporting; the
            # wide control is therefore a CONSERVATIVE control, handing the
            # classical side ~6.5x more capacity.
            #
            # "narrow" adds the opposite bound: a rank-r bottleneck with no
            # biases, r chosen to bring the count as close to the circuit's as
            # a dense block allows. Reporting both brackets the comparison.
            feat = (2 * (nq // 2)) if ansatz == "qcnn" else (2 * nq)
            if control == "wide":
                self.mid = nn.Sequential(nn.Linear(nq, feat), nn.Tanh(),
                                         nn.Linear(feat, feat), nn.Tanh())
            else:
                target = 22 if ansatz == "qcnn" else 6 * nq * 3
                r = max(1, min(range(1, feat + 1),
                               key=lambda r: abs(r * (nq + feat) - target)))
                self.mid = nn.Sequential(nn.Linear(nq, r, bias=False), nn.Tanh(),
                                         nn.Linear(r, feat, bias=False), nn.Tanh())
            # Tanh-aware initialisation. With PyTorch's default (Kaiming-uniform,
            # tuned for ReLU) the stacked tanh saturates and the block can settle
            # at the uniform-prediction loss ln(256)=5.545 with exactly zero
            # gradient -- and whether it does so depends on floating-point
            # summation order, i.e. on the thread count. An untrainable control
            # is not a control, so we start it in the linear regime of the tanh.
            # The quantum block needs no such care: <Z>,<X> readout of a unitary
            # has no saturating regime to fall into.
            for _m in self.mid:
                if isinstance(_m, nn.Linear):
                    nn.init.xavier_uniform_(_m.weight, gain=5.0 / 3.0)
                    if _m.bias is not None:
                        nn.init.zeros_(_m.bias)
        self.bn = nn.BatchNorm1d(feat)
        self.fc = nn.Sequential(nn.Linear(feat, 128), nn.ReLU(), nn.Linear(128, 256))

    def forward(self, x):
        return self.fc(self.bn(self.mid(self.front(x))))


def probs_of(model, F, batch=2048):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(F), batch):
            out.append(torch.softmax(model(torch.tensor(F[i:i + batch])), 1).cpu())
    model.train()
    return torch.cat(out).numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["fixed", "variable", "desync50", "desync100"], default="variable")
    ap.add_argument("--data-dir", default="D:/qml-sca-run/data")
    ap.add_argument("--out-dir", default="D:/qml-sca-run/out")
    ap.add_argument("--head", choices=["quantum", "classical"], default="quantum")
    ap.add_argument("--ansatz", choices=["qcnn", "sel"], default="qcnn")
    ap.add_argument("--control", choices=["wide", "narrow"], default="wide")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--conv-layers", type=int, default=2)
    ap.add_argument("--filters", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=45000)
    ap.add_argument("--n-val", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=250)   # cap; patience stops earlier
    ap.add_argument("--patience", type=int, default=8)   # evals without improvement
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--early-stop", dest="early_stop", action="store_true", default=True)
    ap.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    ctl = "" if args.head == "quantum" else f"_{args.control}"
    tag = args.tag or (f"s3_{args.dataset}_{args.head}{ctl}_{args.ansatz}"
                       f"_q{args.qubits}_c{args.conv_layers}_s{args.seed}")
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()

    db = C.db_path(args.data_dir, args.dataset)
    (Xtr, ytr), (Xva, yva, pva, kva), (Xa, pa), real_key = C.load(
        db, args.n_train, seed=args.seed, n_val=args.n_val if args.early_stop else 0)
    trace_len = Xtr.shape[1]
    print(f"[data] {tag} len={trace_len} train={len(Xtr)} val={0 if Xva is None else len(Xva)} "
          f"attack={len(Xa)} key=0x{real_key:02x} early_stop={args.early_stop}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = Model(trace_len, nq=args.qubits, conv_layers=args.conv_layers,
                  n_filt=args.filters, head=args.head, ansatz=args.ansatz,
                  control=args.control)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nmid = sum(p.numel() for p in model.mid.parameters() if p.requires_grad)
    print(f"[model] {tag} params={npar} (mid-block={nmid})", flush=True)

    g = torch.Generator(); g.manual_seed(args.seed)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                    batch_size=args.batch, shuffle=True, generator=g)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lf = nn.CrossEntropyLoss()

    best = {"score": float("inf"), "ep": 0, "state": None}
    stale = 0
    for ep in range(args.epochs):
        model.train(); tl = 0.0; tc = 0; tn = 0
        for xb, yb in dl:
            opt.zero_grad(); o = model(xb); l = lf(o, yb); l.backward(); opt.step()
            tl += l.item() * len(xb); tn += len(xb); tc += (o.argmax(1) == yb).sum().item()
        line = f"[ep {ep+1}/{args.epochs}] loss={tl/tn:.4f} acc={tc/tn:.4f}"
        if args.early_stop and ((ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs):
            # R5: proxy computed on the PROFILING-side validation split only.
            sc = C.val_ge_proxy(probs_of(model, Xva), pva, kva)
            line += f" val_ge_auc={sc:.2f}"
            if sc < best["score"]:
                best.update(score=sc, ep=ep + 1, state=copy.deepcopy(model.state_dict()))
                stale = 0
                line += " *best*"
            else:
                stale += 1
                line += f" (stale {stale}/{args.patience})"
        print(line, flush=True)
        # Same stopping rule for every arm: the quantum block converges in tens of
        # epochs and then overfits, the classical control needs several times more.
        # A shared fixed budget would be unfair to one of them whichever value we
        # picked, so both simply train until the validation proxy stops improving.
        if args.early_stop and stale >= args.patience:
            print(f"[patience] no improvement for {stale} evals, stopping at ep {ep+1}",
                  flush=True)
            break

    if args.early_stop and best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"[early-stop] best val_ge_auc={best['score']:.2f} at epoch {best['ep']}", flush=True)

    torch.save(model.state_dict(), os.path.join(args.out_dir, f"w_{tag}.pt"))

    # attack traces are touched exactly once, here
    probs = probs_of(model, Xa)
    np.savez(os.path.join(args.out_dir, f"probs_{tag}.npz"),
             probs=probs, pa=pa, real_key=real_key)
    ge = C.guessing_entropy(probs, pa, real_key)
    np.save(os.path.join(args.out_dir, f"ge_{tag}.npy"), ge)

    r05, r1 = C.recov(ge, 0.5), C.recov(ge, 1.0)
    elapsed = time.time() - t_start
    summary = dict(tag=tag, dataset=args.dataset, head=args.head, ansatz=args.ansatz,
                   control=(args.control if args.head == "classical" else None),
                   qubits=args.qubits, conv_layers=args.conv_layers, seed=args.seed,
                   epochs=args.epochs, epochs_run=ep + 1, patience=args.patience,
                   lr=args.lr, params=npar, mid_params=nmid,
                   best_epoch=best["ep"], val_ge_auc=best["score"],
                   ge_min=float(ge.min()), ge_argmin=int(ge.argmin()) + 1,
                   ge_final=float(ge[-1]), recov_05=r05, recov_1=r1,
                   wallclock_s=round(elapsed, 1))
    with open(os.path.join(args.out_dir, f"res_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"[GE] {tag} min={ge.min():.3f}@{int(ge.argmin())+1} final={ge[-1]:.3f} "
          f"recov[GE<0.5]={r05} (GE<1={r1}) time={elapsed/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
