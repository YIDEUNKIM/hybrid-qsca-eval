"""
train_cnnbest.py -- ASCAD CNN_best baseline in PyTorch, on the GPU.

Patched from the original torch_ascad.py: paths, --seed, GPU batch sizing for a
3 GB card, and the shared sca_common data/GE code so it obeys R1-R5 like every
other model in the campaign. Architecture and optimiser are verbatim from the
ASCAD paper's CNN_best (VGG-like, 5 conv blocks 64/128/256/512/512 with k=11 and
AvgPool2, then Dense 4096 x2, RMSprop lr=1e-5).
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qcnn"))
import sca_common as C


class ASCAD_CNNbest(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        filt = [64, 128, 256, 512, 512]; blocks = []; c = 1
        for f in filt:
            blocks += [nn.Conv1d(c, f, 11, padding=5), nn.ReLU(), nn.AvgPool1d(2)]; c = f
        self.conv = nn.Sequential(*blocks)
        with torch.no_grad():
            d = self.conv(torch.zeros(1, 1, n_in)).flatten(1).shape[1]
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(d, 4096), nn.ReLU(),
                                nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 256))

    def forward(self, x):
        return self.fc(self.conv(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["fixed", "variable", "desync50", "desync100"], default="variable")
    ap.add_argument("--data-dir", default="D:/qml-sca-run/data")
    ap.add_argument("--out-dir", default="D:/qml-sca-run/out")
    ap.add_argument("--n-train", type=int, default=45000)
    ap.add_argument("--epochs", type=int, default=75)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or f"cnnbest_{args.dataset}_s{args.seed}"
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev} ({torch.cuda.get_device_name(0) if dev.type=='cuda' else 'CPU'})",
          flush=True)
    t_start = time.time()

    db = C.db_path(args.data_dir, args.dataset)
    # no validation split: CNN_best is trained for a fixed epoch count, as published
    (Xtr, ytr), _, (Xa, pa), real_key = C.load(db, args.n_train, seed=args.seed, n_val=0)
    n_in = Xtr.shape[1]
    print(f"[data] {tag} len={n_in} train={len(Xtr)} attack={len(Xa)} key=0x{real_key:02x}",
          flush=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = ASCAD_CNNbest(n_in).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    print(f"[model] {tag} params={npar:,} batch={args.batch}", flush=True)

    g = torch.Generator(); g.manual_seed(args.seed)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr[:, None, :]), torch.tensor(ytr)),
                    batch_size=args.batch, shuffle=True, generator=g)
    opt = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    lf = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        model.train(); tl = 0.0; tn = 0; tc = 0
        for xb, yb in dl:
            xb = xb.to(dev, non_blocking=True); yb = yb.to(dev, non_blocking=True)
            opt.zero_grad(); o = model(xb); l = lf(o, yb); l.backward(); opt.step()
            tl += l.item() * len(xb); tn += len(xb); tc += (o.argmax(1) == yb).sum().item()
        if (ep + 1) % 5 == 0 or ep == 0:
            el = (time.time() - t_start) / 60
            print(f"[ep {ep+1}/{args.epochs}] loss={tl/tn:.4f} acc={tc/tn:.4f} "
                  f"({el:.1f}min elapsed)", flush=True)

    torch.save(model.state_dict(), os.path.join(args.out_dir, f"w_{tag}.pt"))

    model.eval(); outs = []
    with torch.no_grad():
        for i in range(0, len(Xa), 512):
            outs.append(torch.softmax(model(torch.tensor(Xa[i:i+512, None, :]).to(dev)), 1).cpu())
    probs = torch.cat(outs).numpy().astype(np.float64)

    np.savez(os.path.join(args.out_dir, f"probs_{tag}.npz"),
             probs=probs, pa=pa, real_key=real_key)
    ge = C.guessing_entropy(probs, pa, real_key)
    np.save(os.path.join(args.out_dir, f"ge_{tag}.npy"), ge)
    r05, r1 = C.recov(ge, 0.5), C.recov(ge, 1.0)
    elapsed = time.time() - t_start
    with open(os.path.join(args.out_dir, f"res_{tag}.json"), "w") as fh:
        json.dump(dict(tag=tag, model="CNN_best", dataset=args.dataset, seed=args.seed,
                       epochs=args.epochs, params=npar, ge_min=float(ge.min()),
                       ge_argmin=int(ge.argmin()) + 1, ge_final=float(ge[-1]),
                       recov_05=r05, recov_1=r1, wallclock_s=round(elapsed, 1)), fh, indent=1)
    print(f"[GE] {tag} min={ge.min():.3f}@{int(ge.argmin())+1} final={ge[-1]:.3f} "
          f"recov[GE<0.5]={r05} (GE<1={r1}) time={elapsed/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
