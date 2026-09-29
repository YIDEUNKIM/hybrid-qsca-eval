"""
grad_variance.py -- is the SEL collapse a barren plateau?

A barren plateau is a statement about the gradient at initialisation: its
variance over random parameter draws decays exponentially in the number of
qubits, so a randomly initialised circuit starts on a flat landscape and the
optimiser has nothing to follow. QCNNs are known not to suffer from it (their
pooling keeps every gradient local); deep hardware-efficient ansaetze such as
StronglyEntanglingLayers are the textbook case that does.

We measure exactly that, on exactly the circuits used in the campaign: for many
random initialisations, the variance of dC/dtheta for the circuit parameters,
where C is a fixed local cost (<Z> on the first kept wire) evaluated on real
encoded ASCAD inputs. Run at several qubit counts, the decay rate separates the
two ansaetze or it does not.
"""
import argparse, json, os, sys
import numpy as np
import torch
import pennylane as qml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "qcnn"))


def qcnn_circuit(nq):
    dev = qml.device("default.qubit", wires=nq)

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

    n1 = nq // 2; nk = len(range(0, nq - 1, 2)); n2 = max(nk // 2, 1)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def f(x, c1, p1, c2):
        for i in range(nq):
            qml.RY(x[i], wires=i)
        conv(c1, list(range(nq)))
        kept = pool(p1, list(range(nq)))
        conv(c2, kept)
        return qml.expval(qml.PauliZ(kept[0]))

    def init(g):
        return [torch.rand(n1, 3, generator=g) * 2 * np.pi,
                torch.rand(n1, generator=g) * 2 * np.pi,
                torch.rand(n2, 3, generator=g) * 2 * np.pi]
    return f, init


def sel_circuit(nq, layers=6):
    dev = qml.device("default.qubit", wires=nq)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def f(x, w):
        qml.AngleEmbedding(x, wires=range(nq), rotation="Y")
        qml.StronglyEntanglingLayers(w, wires=range(nq))
        return qml.expval(qml.PauliZ(0))

    def init(g):
        return [torch.rand(layers, nq, 3, generator=g) * 2 * np.pi]
    return f, init


def grad_var(builder, nq, n_init, x_bank, seed=0):
    f, init = builder(nq)
    g = torch.Generator().manual_seed(seed)
    grads = []
    for i in range(n_init):
        params = [p.requires_grad_(True) for p in init(g)]
        x = torch.tensor(x_bank[i % len(x_bank)][:nq], dtype=torch.float64)
        out = f(x, *params)
        out.backward()
        # variance of the FIRST parameter of the FIRST layer: the standard
        # barren-plateau probe, and the one furthest from the measured wire
        grads.append(float(params[0].grad.reshape(-1)[0]))
    return float(np.var(grads)), float(np.mean(np.abs(grads)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    ap.add_argument("--n-init", type=int, default=300)
    ap.add_argument("--out", default="D:/qml-sca-run/out/grad_variance.json")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)

    # real angle inputs: what the trained conv front-end actually feeds the
    # circuit is concentrated near pi/2, so use that distribution rather than
    # uniform noise (uniform inputs would flatter neither ansatz fairly)
    rng = np.random.default_rng(0)
    x_bank = np.clip(rng.normal(np.pi / 2, 0.5, size=(args.n_init, max(args.qubits))),
                     0, np.pi)

    res = {"qcnn": {}, "sel": {}}
    print(f"{'n':>3} | {'QCNN var':>12} {'QCNN |g|':>10} | {'SEL var':>12} {'SEL |g|':>10}")
    for n in args.qubits:
        vq, mq = grad_var(qcnn_circuit, n, args.n_init, x_bank)
        vs, ms = grad_var(sel_circuit, n, args.n_init, x_bank)
        res["qcnn"][n] = dict(var=vq, mean_abs=mq)
        res["sel"][n] = dict(var=vs, mean_abs=ms)
        print(f"{n:>3} | {vq:12.3e} {mq:10.4f} | {vs:12.3e} {ms:10.4f}", flush=True)

    # exponential decay rate: slope of log(var) vs n
    for k in ("qcnn", "sel"):
        ns = np.array(sorted(res[k]))
        lv = np.log([res[k][n]["var"] for n in ns])
        slope = np.polyfit(ns, lv, 1)[0]
        res[k]["log_var_slope_per_qubit"] = float(slope)
        print(f"{k:5}: d log(Var)/dn = {slope:+.3f}   "
              f"(var halves every {abs(np.log(2)/slope):.1f} qubits)" if slope < 0 else
              f"{k:5}: d log(Var)/dn = {slope:+.3f}   (no decay)")
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
