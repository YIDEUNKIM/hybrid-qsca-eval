"""
qcnn_stages12.py -- Storyline stages 1 & 2 (the fixed-encoder configurations),
run under exactly the protocol of sca_common (R1-R5).

  stage 1 (--mode pure):   PCA(->256) -> AmplitudeEmbedding(8q) -> SEL -> probs(256).
                           NO classical NN anywhere.
  stage 2 (--mode hybrid): PCA(->256) -> AmplitudeEmbedding(8q) -> QCNN conv+pool
                           -> <Z>,<X> -> classical head(256).
  stage 3 is qcnn_stage3.py (conv -> angle -> circuit -> head).

1->2 adds the classical head; 2->3 changes ONLY the encoder. PCA is a FIXED
(unlearned) transform -- that is the point of the comparison.

Patched from the original: memory-efficient load via sca_common (the original
read the full 200k profiling set as float64, peaking ~4.5 GB), explicit --seed,
and the same R5-clean early stopping as stage 3 so all three stages are selected
the same way. Circuit/PCA/label code is otherwise verbatim.
"""
import argparse, copy, json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml

import sca_common as C

NQ = 8
AMP_DIM = 2 ** NQ


def pca_fit_transform(Xtr, Xother, k=AMP_DIM):
    """StandardScaler + PCA (numpy covariance eigendecomp), fit on training only.
    Covariance accumulated in float64 for conditioning; data stays float32."""
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
    Ztr = (Xtr - mu) / sd
    cov = (Ztr.T.astype(np.float64) @ Ztr.astype(np.float64)) / len(Ztr)
    evals, evecs = np.linalg.eigh(cov)
    comps = evecs[:, ::-1][:, :k]
    var_kept = float(evals[::-1][:k].sum() / evals.sum())
    out = [(Ztr @ comps).astype(np.float32)]
    for X in Xother:
        out.append((((X - mu) / sd) @ comps).astype(np.float32) if X is not None else None)
    return out, var_kept


def _conv(w, wires):
    P = [(wires[i], wires[i + 1]) for i in range(0, len(wires) - 1, 2)]
    for j, (a, b) in enumerate(P):
        qml.RY(w[j, 0], wires=a); qml.RY(w[j, 1], wires=b)
        qml.CNOT(wires=[a, b]); qml.RY(w[j, 2], wires=b)


def _pool(w, wires):
    P = [(wires[i], wires[i + 1]) for i in range(0, len(wires) - 1, 2)]; k = []
    for j, (a, b) in enumerate(P):
        qml.CRZ(w[j], wires=[b, a]); k.append(a)
    return k


class PureQ(nn.Module):
    """stage 1: amplitude -> VQC -> probs(256), no classical NN."""
    def __init__(self, q_layers=6):
        super().__init__()
        dev = qml.device("default.qubit", wires=NQ)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circ(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(NQ), normalize=True, pad_with=0.0)
            qml.StronglyEntanglingLayers(weights, wires=range(NQ))
            return qml.probs(wires=range(NQ))
        self.q = qml.qnn.TorchLayer(circ, {"weights": (q_layers, NQ, 3)})

    def forward(self, x):
        return torch.log(self.q(x) + 1e-12)


class HybridAmp(nn.Module):
    """stage 2: amplitude -> QCNN conv+pool -> <Z>,<X> -> classical head(256)."""
    def __init__(self):
        super().__init__()
        dev = qml.device("default.qubit", wires=NQ)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circ(inputs, conv1, pool1, conv2):
            qml.AmplitudeEmbedding(inputs, wires=range(NQ), normalize=True, pad_with=0.0)
            _conv(conv1, list(range(NQ)))
            kept = _pool(pool1, list(range(NQ)))
            _conv(conv2, kept)
            return [qml.expval(qml.PauliZ(w)) for w in kept] + \
                   [qml.expval(qml.PauliX(w)) for w in kept]
        n1 = NQ // 2; nk = n1; n2 = nk // 2
        self.q = qml.qnn.TorchLayer(circ, {"conv1": (n1, 3), "pool1": (n1,), "conv2": (n2, 3)})
        feat = 2 * nk
        self.bn = nn.BatchNorm1d(feat)
        self.fc = nn.Sequential(nn.Linear(feat, 128), nn.ReLU(), nn.Linear(128, 256))

    def forward(self, x):
        return self.fc(self.bn(self.q(x)))


def probs_of(model, F, mode, batch=2048):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(F), batch):
            o = model(torch.tensor(F[i:i + batch]))
            out.append((torch.exp(o) if mode == "pure" else torch.softmax(o, 1)).cpu())
    model.train()
    return torch.cat(out).numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["fixed", "variable", "desync50", "desync100"], default="fixed")
    ap.add_argument("--data-dir", default="D:/qml-sca-run/data")
    ap.add_argument("--out-dir", default="D:/qml-sca-run/out")
    ap.add_argument("--mode", choices=["pure", "hybrid"], required=True)
    ap.add_argument("--n-train", type=int, default=45000)
    ap.add_argument("--n-val", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=200)   # cap; patience stops earlier
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    stage = "1pure" if args.mode == "pure" else "2hybrid"
    tag = args.tag or f"s{stage}_{args.dataset}_s{args.seed}"
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()

    db = C.db_path(args.data_dir, args.dataset)
    # standardize=False: this pipeline does its own scaler inside the PCA fit
    (Xtr, ytr), (Xva, yva, pva, kva), (Xa, pa), real_key = C.load(
        db, args.n_train, seed=args.seed, n_val=args.n_val, standardize=False)
    (Ftr, Fva, Fa), var_kept = pca_fit_transform(Xtr, [Xva, Xa], k=AMP_DIM)
    del Xtr, Xva, Xa
    print(f"[data] {tag} mode={args.mode} pca={AMP_DIM} var_kept={var_kept:.3f} "
          f"train={len(Ftr)} val={len(Fva)} attack={len(Fa)} key=0x{real_key:02x}", flush=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = PureQ() if args.mode == "pure" else HybridAmp()
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] stage={stage} qubits={NQ} params={npar}", flush=True)

    g = torch.Generator(); g.manual_seed(args.seed)
    dl = DataLoader(TensorDataset(torch.tensor(Ftr), torch.tensor(ytr)),
                    batch_size=args.batch, shuffle=True, generator=g)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lf = nn.NLLLoss() if args.mode == "pure" else nn.CrossEntropyLoss()

    best = {"score": float("inf"), "ep": 0, "state": None}
    stale = 0
    for ep in range(args.epochs):
        model.train(); tl = 0.0; tc = 0; tn = 0
        for xb, yb in dl:
            opt.zero_grad(); o = model(xb); l = lf(o, yb); l.backward(); opt.step()
            tl += l.item() * len(xb); tn += len(xb); tc += (o.argmax(1) == yb).sum().item()
        line = f"[ep {ep+1}/{args.epochs}] loss={tl/tn:.4f} acc={tc/tn:.4f}"
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            sc = C.val_ge_proxy(probs_of(model, Fva, args.mode), pva, kva)
            line += f" val_ge_auc={sc:.2f}"
            if sc < best["score"]:
                best.update(score=sc, ep=ep + 1, state=copy.deepcopy(model.state_dict()))
                stale = 0
                line += " *best*"
            else:
                stale += 1
                line += f" (stale {stale}/{args.patience})"
        print(line, flush=True)
        if stale >= args.patience:
            print(f"[patience] no improvement for {stale} evals, stopping at ep {ep+1}",
                  flush=True)
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"[early-stop] best val_ge_auc={best['score']:.2f} at epoch {best['ep']}", flush=True)

    torch.save(model.state_dict(), os.path.join(args.out_dir, f"w_{tag}.pt"))

    probs = probs_of(model, Fa, args.mode)
    np.savez(os.path.join(args.out_dir, f"probs_{tag}.npz"),
             probs=probs, pa=pa, real_key=real_key)
    ge = C.guessing_entropy(probs, pa, real_key)
    np.save(os.path.join(args.out_dir, f"ge_{tag}.npy"), ge)

    r05, r1 = C.recov(ge, 0.5), C.recov(ge, 1.0)
    elapsed = time.time() - t_start
    with open(os.path.join(args.out_dir, f"res_{tag}.json"), "w") as fh:
        json.dump(dict(tag=tag, stage=stage, dataset=args.dataset, mode=args.mode,
                       seed=args.seed, epochs=args.epochs, epochs_run=ep + 1, params=npar,
                       var_kept=var_kept, best_epoch=best["ep"],
                       val_ge_auc=best["score"], ge_min=float(ge.min()),
                       ge_argmin=int(ge.argmin()) + 1, ge_final=float(ge[-1]),
                       recov_05=r05, recov_1=r1, wallclock_s=round(elapsed, 1)), fh, indent=1)
    print(f"[GE] {tag} min={ge.min():.3f}@{int(ge.argmin())+1} final={ge[-1]:.3f} "
          f"recov[GE<0.5]={r05} (GE<1={r1}) time={elapsed/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
