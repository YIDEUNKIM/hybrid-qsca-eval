"""
sca_common.py -- shared pieces for the qml-sca campaign, factored out of the
original scripts so that every model provably uses the SAME data pipeline,
the SAME label, the SAME GE routine and the SAME recovery criterion.

Derived from qcnn_convfront.py / qcnn_scan.py / qcnn_stages12.py (originals kept
untouched under the source tree). Three deliberate changes vs the originals:

  (1) MEMORY. ASCAD traces are stored as int8. The originals did
      np.array(f[...], dtype=np.float32/64) on the FULL 200k profiling set,
      peaking at 3-4.5 GB. We slice the 50k subset inside h5py FIRST and only
      then widen to float32, peaking under 1 GB. Same traces, same subset.

  (2) R5 / TEST-SET LEAK. The original qcnn_scan.py selected the early-stopping
      checkpoint with a GE proxy computed on Fa[:5000] using the REAL KEY, and
      then reported final GE on the full attack set, which contains those very
      traces. That is model selection on the test set. Here the proxy is
      computed on a VALIDATION split carved out of the PROFILING set (disjoint
      from the training subset), whose keys the evaluator legitimately knows.
      Attack traces are touched exactly once, for the final score.

  (3) SEEDS. Everything takes an explicit --seed so results can be reported as
      a median over repeats instead of a single lucky run.

Standardization note: the originals fit mu/sd on the full profiling set; we fit
on the 50k training subset only (statistically equivalent, and strictly the
more correct choice for a profiling model).
"""
import numpy as np
import h5py

AES_SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
], dtype=np.uint8)

TARGET_BYTE = 2

# Unified project criteria (R3 / R4). Nothing may override these.
GE_NAVG = 100      # attack-set orderings averaged
GE_NMAX = 3000     # trace-count window
GE_THR = 0.5       # recovery threshold, "first stays below"


DB_FILES = {
    "fixed":     "ASCAD.h5",           # fixed key, 700 samples, aligned
    "variable":  "ascad-variable.h5",  # random key, 1400 samples, aligned
    "desync50":  "ASCAD_desync50.h5",  # fixed key, random shift up to 50 samples
    "desync100": "ASCAD_desync100.h5", # fixed key, random shift up to 100 samples
}


def db_path(data_dir, dataset):
    """dataset in DB_FILES -> h5 file path."""
    return f"{data_dir}/{DB_FILES[dataset]}"


def load(db, n_train, seed=0, n_val=20000, standardize=True):
    """Mask-free loader (R1). Label = Sbox(p[2] ^ k[2]), unmasked.

    Returns (Xtr, ytr), (Xval, yval, pval, kval), (Xa, pa), real_key.
    The validation split is disjoint from the training subset and comes from the
    PROFILING set, so using it for model selection leaks nothing (R5).
    """
    with h5py.File(db, "r") as f:
        ptr = f["Profiling_traces/traces"]
        Ntot = ptr.shape[0]
        perm = np.random.RandomState(seed).permutation(Ntot)
        tr_idx = np.sort(perm[:n_train])
        va_idx = np.sort(perm[n_train:n_train + n_val])

        # int8 on disk -> slice first, widen after (keeps peak RAM < 1 GB)
        Xtr = ptr[tr_idx].astype(np.float32)
        Xva = ptr[va_idx].astype(np.float32) if n_val > 0 else None

        Mp = f["Profiling_traces/metadata"]
        ptxt = Mp["plaintext"]
        keys = Mp["key"]
        ptr_tr = ptxt[tr_idx][:, TARGET_BYTE]
        key_tr = keys[tr_idx][:, TARGET_BYTE]
        if n_val > 0:
            ptr_va = ptxt[va_idx][:, TARGET_BYTE]
            key_va = keys[va_idx][:, TARGET_BYTE]

        Xa = f["Attack_traces/traces"][:].astype(np.float32)
        Ma = f["Attack_traces/metadata"]
        pa = np.array(Ma["plaintext"][:, TARGET_BYTE], np.uint8)
        real_key = int(np.array(Ma["key"][0])[TARGET_BYTE])

    ytr = AES_SBOX[ptr_tr ^ key_tr].astype(np.int64)

    if standardize:
        mu = Xtr.mean(0, keepdims=True)
        sd = Xtr.std(0, keepdims=True)
        sd[sd == 0] = 1.0
        Xtr -= mu; Xtr /= sd
        Xa -= mu; Xa /= sd
        if n_val > 0:
            Xva -= mu; Xva /= sd

    if n_val > 0:
        yva = AES_SBOX[ptr_va ^ key_va].astype(np.int64)
        val = (Xva, yva, ptr_va.astype(np.uint8), key_va.astype(np.uint8))
    else:
        val = (None, None, None, None)
    return (Xtr, ytr), val, (Xa, pa), real_key


def guessing_entropy(probs, pa, real_key, nmax=GE_NMAX, navg=GE_NAVG, seed=0):
    """Shared GE estimator (R3). Identical maths to the originals."""
    N = len(probs)
    logp = np.empty((256, N))
    for kg in range(256):
        logp[kg] = np.log(probs[np.arange(N), AES_SBOX[(pa ^ kg).astype(np.uint8)]] + 1e-40)
    rng = np.random.default_rng(seed)
    nm = min(nmax, N)
    ge = np.zeros(nm)
    for _ in range(navg):
        pm = rng.permutation(N)[:nm]
        cum = np.cumsum(logp[:, pm], 1)
        ge += (cum > cum[real_key][None, :]).sum(0)
    return ge / navg


def recov(ge, thr=GE_THR):
    """Smallest N such that GE stays below `thr` for every N' >= N (R4)."""
    below = ge < thr
    for i in range(len(ge)):
        if np.all(below[i:]):
            return i + 1
    return None


def val_ge_proxy(probs, pv, kv, navg=50, nmax=512, min_group=8, seed=0):
    """Leak-free early-stopping signal (R5), computed on the PROFILING-side
    validation split.

    The validation traces do not share one key (ASCAD_variable draws a random
    key per trace), so a single GE curve is undefined. We instead group the
    split by its key byte, run the ordinary GE estimator inside each group, and
    average the resulting curves over groups. At the campaign's 5k split this is
    256 groups of ~19 traces each; individually those curves are very noisy, but
    averaging over 256 independent keys makes the mean stable enough to rank
    checkpoints (which is all the proxy is for). For the fixed-key database this
    degenerates to exactly one group of 5000 traces, where nmax is the binding
    limit rather than the group size.

    Resolution matters more than it looks: with nmax too small every checkpoint
    of a not-yet-converged model sits near the random rank of 127.5, the spread
    between them is pure noise, and the argmin lands on whichever early epoch got
    lucky -- which silently returns an untrained model. nmax/navg here are set so
    the proxy separates real differences from that noise.

    Returns the area under the averaged GE curve (mean of the curve). Lower is
    better; AUC discriminates between checkpoints even when several of them
    eventually reach rank 0, which a bare min-GE does not.
    """
    curves = []
    for k in np.unique(kv):
        m = (kv == k)
        if m.sum() < min_group:
            continue
        g = guessing_entropy(probs[m], pv[m], int(k),
                             nmax=min(nmax, int(m.sum())), navg=navg, seed=seed)
        curves.append(g)
    if not curves:
        return float("inf")
    L = min(len(c) for c in curves)
    return float(np.mean([c[:L] for c in curves]))
