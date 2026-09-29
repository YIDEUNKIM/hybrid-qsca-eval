"""
train_tf_baselines.py -- the TensorFlow classical baselines, unified.

  --model rijsdijk : Rijsdijk et al. (TCHES 2021) RL-discovered ID models,
                     retrained at 50k. Uses the authors' OWN OneCycleLR code and
                     their exact layer conventions (SAME-pad conv, average
                     pooling, SeLU, He-uniform init, glorot-uniform softmax,
                     Adam + one-cycle max_lr=5e-3, end%=0.2, scale%=0.1).
                       fixed    (Table 12): C(128,25,1), P(25,25), FC20, FC15, SM256, batch 50
                       variable (Table 12): C(128,3,1),  P(75,75), FC30, FC2,  SM256, batch 400
  --model mlp      : ASCAD paper MLP_best: 6 x Dense(200, ReLU) + Dense(256, softmax),
                     RMSprop lr=1e-5, batch 100, 100 epochs.

Merged from train_rijsdijk_fixed_50k.py / train_rijsdijk_var_50k.py / train_mlp.py.
The OneCycleLR TF2.21/Keras3 patches are carried over verbatim. Data, label, GE
and recovery criterion now come from sca_common (R1-R5), so these baselines are
scored by the same code as the quantum models.
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")  # TF has no GPU on native Windows anyway

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qcnn"))
import sca_common as C

HERE = os.path.dirname(os.path.abspath(__file__))

# --- the repo's OWN OneCycleLR (standalone file load, no package side effects) ---
_spec = importlib.util.spec_from_file_location(
    "one_cycle_lr", os.path.join(HERE, "metaqnn", "training", "one_cycle_lr.py"))
ocl = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ocl)


def _get_lr(opt):
    lr = getattr(opt, "learning_rate", None)
    if lr is None: lr = opt.lr
    try: return float(lr.numpy())
    except Exception: return float(lr)


def _set_lr(opt, v):
    lr = getattr(opt, "learning_rate", None)
    if lr is not None and hasattr(lr, "assign"):
        lr.assign(float(v))
    else:
        opt.learning_rate = float(v)


class OneCycleLR(ocl.OneCycleLR):
    """Repo OneCycleLR (exact schedule math via inherited compute_lr), patched for
    TF2.21/Keras3: (1) callback params lacks 'samples' -> use 'steps';
    (2) optimizer.lr renamed to learning_rate and K.set_value removed -> assign."""
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.batch_size = self.params.get('batch_size')
        self.samples = self.params.get('samples')
        self.steps = self.params.get('steps')
        if self.steps is not None:
            self.num_iterations = self.epochs * self.steps
        else:
            remainder = 0 if (self.samples % self.batch_size) == 0 else 1
            self.num_iterations = (self.epochs + remainder) * self.samples // self.batch_size
        self.mid_cycle_id = int(self.num_iterations * ((1. - self.end_percentage)) / float(2))
        self._reset()
        _set_lr(self.model.optimizer, self.compute_lr())

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.clr_iterations += 1
        new_lr = self.compute_lr()
        self.history.setdefault('lr', []).append(_get_lr(self.model.optimizer))
        _set_lr(self.model.optimizer, new_lr)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


def build_rijsdijk(dataset, n_in):
    """paper Table 12 identity models, per dataset."""
    L = tf.keras.layers
    if dataset != "variable":          # fixed key and its desynchronised variants
        conv_k, pool = 25, 25
        dense = [20, 15]
    else:
        conv_k, pool = 3, 75
        dense = [30, 2]
    layers = [L.InputLayer(shape=(n_in, 1)),
              L.Convolution1D(filters=128, kernel_size=conv_k, strides=1,
                              kernel_initializer='he_uniform', activation='selu',
                              padding='same'),
              L.AveragePooling1D(pool_size=pool, strides=pool),
              L.Flatten()]
    for u in dense:
        layers.append(L.Dense(u, kernel_initializer='he_uniform', activation='selu'))
    layers.append(L.Dense(256, kernel_initializer='glorot_uniform', activation='softmax'))
    m = tf.keras.Sequential(layers)
    m.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m


def build_mlp(n_in):
    L = tf.keras.layers
    layers = [L.InputLayer(shape=(n_in,))]
    for _ in range(6):
        layers.append(L.Dense(200, activation='relu'))
    layers.append(L.Dense(256, activation='softmax'))
    m = tf.keras.Sequential(layers)
    m.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rijsdijk", "mlp"], required=True)
    ap.add_argument("--dataset", choices=["fixed", "variable", "desync50", "desync100"], default="variable")
    ap.add_argument("--data-dir", default="D:/qml-sca-run/data")
    ap.add_argument("--out-dir", default="D:/qml-sca-run/out")
    ap.add_argument("--n-train", type=int, default=45000)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.model == "rijsdijk":
        epochs = args.epochs or 50
        # repo hardcodes 50 for the fixed config; paper Table 9 uses 400 for variable
        batch = args.batch or (400 if args.dataset == "variable" else 50)
    else:
        epochs = args.epochs or 100
        batch = args.batch or 100

    tag = args.tag or f"{args.model}_{args.dataset}_s{args.seed}"
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    print(f"[cfg] {tag} epochs={epochs} batch={batch} n_train={args.n_train}", flush=True)

    db = C.db_path(args.data_dir, args.dataset)
    (Xtr, ytr), _, (Xa, pa), real_key = C.load(db, args.n_train, seed=args.seed, n_val=0)
    n_in = Xtr.shape[1]
    if args.model == "rijsdijk":
        Xtr = Xtr.reshape(-1, n_in, 1); Xa_in = Xa.reshape(-1, n_in, 1)
    else:
        Xa_in = Xa
    ytr_oh = tf.keras.utils.to_categorical(ytr, num_classes=256)
    print(f"[data] {tag} len={n_in} train={len(Xtr)} attack={len(Xa)} key=0x{real_key:02x}",
          flush=True)

    tf.random.set_seed(args.seed); np.random.seed(args.seed)
    model = build_rijsdijk(args.dataset, n_in) if args.model == "rijsdijk" else build_mlp(n_in)
    tp = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
    print(f"[model] {tag} trainable params={tp}", flush=True)

    cbs = []
    if args.model == "rijsdijk":
        cbs = [OneCycleLR(max_lr=5e-3, end_percentage=0.2, scale_percentage=0.1,
                          maximum_momentum=None, minimum_momentum=None, verbose=False)]
    hist = model.fit(Xtr, ytr_oh, epochs=epochs, batch_size=batch, shuffle=True,
                     verbose=2, callbacks=cbs)

    probs = model.predict(Xa_in, batch_size=512, verbose=0).astype(np.float64)
    np.savez(os.path.join(args.out_dir, f"probs_{tag}.npz"),
             probs=probs, pa=pa, real_key=real_key)
    ge = C.guessing_entropy(probs, pa, real_key)
    np.save(os.path.join(args.out_dir, f"ge_{tag}.npy"), ge)
    r05, r1 = C.recov(ge, 0.5), C.recov(ge, 1.0)
    elapsed = time.time() - t0
    h = hist.history
    with open(os.path.join(args.out_dir, f"res_{tag}.json"), "w") as fh:
        json.dump(dict(tag=tag, model=args.model, dataset=args.dataset, seed=args.seed,
                       epochs=epochs, batch=batch, params=tp,
                       final_loss=float(h['loss'][-1]), final_acc=float(h['accuracy'][-1]),
                       ge_min=float(ge.min()), ge_argmin=int(ge.argmin()) + 1,
                       ge_final=float(ge[-1]), recov_05=r05, recov_1=r1,
                       wallclock_s=round(elapsed, 1)), fh, indent=1)
    print(f"[GE] {tag} min={ge.min():.3f}@{int(ge.argmin())+1} final={ge[-1]:.3f} "
          f"recov[GE<0.5]={r05} (GE<1={r1}) time={elapsed/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
