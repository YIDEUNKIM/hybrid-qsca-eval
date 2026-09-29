"""validation_groups.py -- key-byte groups behind the R5 checkpoint proxy (Sec. 2.3).

Replays the split of sca_common.load (RandomState(seed).permutation over the
profiling set, positions 45000..49999 as validation) and the grouping of
sca_common.val_ge_proxy: groups with fewer than 8 traces are dropped, each group
curve has min(512, group size) points, and all curves are cut to the shortest
one, whose length is L.

  python analysis/validation_groups.py --db data/ascad-variable.h5
  python analysis/validation_groups.py --db data/ASCAD.h5 --seeds 0   # one group, L = 512
"""
import argparse
import os
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "qcnn"))
import sca_common as C  # noqa: E402

N_TRAIN, N_VAL = 45000, 5000     # qcnn_stage3.py defaults
MIN_GROUP, NMAX = 8, 512         # sca_common.val_ge_proxy defaults


def validation_groups(keys, seed):
    """Group sizes of one seed's validation split, the retained sizes, and L."""
    perm = np.random.RandomState(seed).permutation(len(keys))
    kv = keys[np.sort(perm[N_TRAIN:N_TRAIN + N_VAL])]
    sizes = np.bincount(kv, minlength=256)
    sizes = sizes[sizes > 0]
    kept = sizes[sizes >= MIN_GROUP]
    return sizes, kept, int(np.minimum(NMAX, kept).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(ROOT, "data", "ascad-variable.h5"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = ap.parse_args()
    with h5py.File(args.db, "r") as f:
        keys = f["Profiling_traces/metadata"]["key"][:, C.TARGET_BYTE].astype(np.int64)
    for seed in args.seeds:
        sizes, kept, L = validation_groups(keys, seed)
        print(f"seed {seed}: {len(kept)}/{len(sizes)} groups kept, "
              f"median size {np.median(sizes):g}, smallest {sizes.min()}, L = {L}")


if __name__ == "__main__":
    main()
