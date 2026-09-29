"""srlib.py -- GE + SR estimator for the execution studies (Sec. 4).

Maths identical to sca_common.guessing_entropy (same default_rng(seed) stream,
same strict-rank convention), but the per-ordering ranks are kept so that the
top-1 success rate SR(N) = P[rank(k*) = 0 at N] can be read off next to GE(N).
"""
import os, sys
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
sys.path.insert(0, os.path.join(PKG, "code", "qcnn"))
sys.path.insert(0, os.path.join(PKG, "code"))
import sca_common as C  # noqa: E402

CKPT = os.path.join(PKG, "checkpoint", "w_s3_f_quantum_qcnn_s0_w.pt")
RES = os.path.join(PKG, "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DATA = os.environ.get("ASCAD_H5", os.path.join(PKG, "data", "ASCAD.h5"))


def ge_sr(probs, pa, real_key, nmax=C.GE_NMAX, navg=C.GE_NAVG, seed=0):
    """Returns (ge, sr, ranks); ge is bit-identical to sca_common.guessing_entropy."""
    N = len(probs)
    logp = np.empty((256, N))
    for kg in range(256):
        logp[kg] = np.log(probs[np.arange(N), C.AES_SBOX[(pa ^ kg).astype(np.uint8)]] + 1e-40)
    rng = np.random.default_rng(seed)
    nm = min(nmax, N)
    ranks = np.zeros((navg, nm), dtype=np.int64)
    for j in range(navg):
        pm = rng.permutation(N)[:nm]
        cum = np.cumsum(logp[:, pm], 1)
        ranks[j] = (cum > cum[real_key][None, :]).sum(0)
    return ranks.mean(0), (ranks == 0).mean(0), ranks.astype(np.uint8)   # ranks <= 255


def t_stays(curve, pred):
    """Smallest 1-based N such that pred holds at every N' >= N (R4 'stays' rule)."""
    bad = np.flatnonzero(~pred(np.asarray(curve)))
    if not bad.size:
        return 1
    start = int(bad[-1]) + 2
    return start if start <= len(curve) else None


def metrics(ge, sr):
    t05 = C.recov(ge)
    return dict(
        T_ge05=t05,
        SR_at_T_ge05=(float(sr[t05 - 1]) if t05 else None),
        T_sr50=t_stays(sr, lambda s: s >= 0.5),
        T_sr90=t_stays(sr, lambda s: s >= 0.9),
        T_sr100=t_stays(sr, lambda s: s >= 1.0),
        T_ge0=t_stays(ge, lambda g: g == 0),
        SR_end=float(sr[-1]), GE_end=float(ge[-1]), GE_min=float(ge.min()),
        GE_100=(float(ge[99]) if len(ge) >= 100 else None),
        SR_100=(float(sr[99]) if len(sr) >= 100 else None), H=int(len(ge)),
    )


def fmt(m):
    return (f"T_GE<0.5={m['T_ge05']}  SR@T={m['SR_at_T_ge05']}  "
            f"T_SR>=0.9={m['T_sr90']}  T_SR=1={m['T_sr100']}  "
            f"SR(H={m['H']})={m['SR_end']:.2f}  GE(H)={m['GE_end']:.3f}")


def load_data():
    (Xtr, ytr), _, (Xa, pa), real_key = C.load(DATA, 45000, seed=0, n_val=0)
    trace_len = Xtr.shape[1]
    del Xtr, ytr
    return Xa, pa, real_key, trace_len


def load_state():
    import torch
    return torch.load(CKPT, map_location="cpu", weights_only=True)
