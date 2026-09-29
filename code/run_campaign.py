"""
run_campaign.py -- sequential, resumable driver for the qml-sca campaign.

Replaces orchestrate_stages.sh / orchestrate_scan.sh (bash + nohup + pgrep, which
do not work on Windows). Runs a queue one job at a time, writes a per-job log,
and SKIPS any job whose res_<tag>.json already exists -- so it can be killed and
restarted at any point without losing work.

  python run_campaign.py --phase variable
  python run_campaign.py --phase fixed
  python run_campaign.py --phase variable --dry-run

Epoch policy (deliberate, and stated in the paper):
  * stage 3 runs get 60 epochs with best-checkpoint selection on the validation
    GE proxy;
  * stages 1 & 2 -- the configurations we expect to FAIL -- get 100 epochs, i.e.
    strictly more than the winner, so that "it did not converge" cannot explain
    the negative result.
"""
import argparse, json, os, subprocess, sys, time

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
LOGS = os.path.join(OUT, "logs")
PY_Q = os.path.join(BASE, "env", "Scripts", "python.exe")     # torch + pennylane
PY_TF = os.path.join(BASE, "tfenv", "Scripts", "python.exe")  # tensorflow
QCNN = os.path.join(BASE, "qcnn")
BL = os.path.join(BASE, "baselines")

VAR_SEEDS = list(range(10))   # variance study (headline arms, fixed dataset)
FEW_SEEDS = [0, 1, 2]         # everything else


def s3(ds, tag, seed=0, **kw):
    a = ["--dataset", ds, "--seed", str(seed)]
    for k, v in kw.items():
        a += [f"--{k.replace('_','-')}", str(v)]
    return (tag, PY_Q, os.path.join(QCNN, "qcnn_stage3.py"), a + ["--tag", tag])


def s12(ds, mode, tag, seed=0):
    return (tag, PY_Q, os.path.join(QCNN, "qcnn_stages12.py"),
            ["--dataset", ds, "--mode", mode, "--seed", str(seed), "--tag", tag])


def tfb(model, ds, tag, seed=0):
    return (tag, PY_TF, os.path.join(BL, "train_tf_baselines.py"),
            ["--model", model, "--dataset", ds, "--seed", str(seed), "--tag", tag])


def cnnb(ds, tag, seed=0):
    return (tag, PY_Q, os.path.join(BL, "train_cnnbest.py"),
            ["--dataset", ds, "--seed", str(seed), "--tag", tag])


def queue_for(phase):
    q = []
    if phase == "variance":
        # P1 -- the trainability/variance study the comparison actually needs.
        # ASCAD fixed only: it is the dataset where recovery happens and where we
        # have the 205-trace anchor to validate against.
        for s_ in VAR_SEEDS:
            q.append(s3("fixed", f"s3_f_quantum_qcnn_s{s_}", s_, head="quantum", ansatz="qcnn"))
            q.append(s3("fixed", f"s3_f_clswide_qcnn_s{s_}", s_, head="classical",
                        ansatz="qcnn", control="wide"))
            q.append(s3("fixed", f"s3_f_clsnarrow_qcnn_s{s_}", s_, head="classical",
                        ansatz="qcnn", control="narrow"))
    elif phase == "encoder":
        # P2 -- the paper's central claim: stage 1/2 (fixed encoder) vs stage 3.
        for s_ in [0, 1]:
            q.append(s12("fixed", "pure", f"s1_f_pure_s{s_}", s_))
            q.append(s12("fixed", "hybrid", f"s2_f_hybrid_s{s_}", s_))
        for s_ in FEW_SEEDS:
            q.append(s3("variable", f"s3_v_quantum_qcnn_s{s_}", s_, head="quantum", ansatz="qcnn"))
            q.append(s3("variable", f"s3_v_clswide_qcnn_s{s_}", s_, head="classical",
                        ansatz="qcnn", control="wide"))
        for s_ in [0, 1]:
            q.append(s12("variable", "pure", f"s1_v_pure_s{s_}", s_))
            q.append(s12("variable", "hybrid", f"s2_v_hybrid_s{s_}", s_))
    elif phase == "desync":
        # Does the learned encoder's advantage survive trace misalignment? The
        # desynchronised ASCAD variants share the fixed-key geometry, so this is
        # the same comparison with the only change being alignment.
        for ds, d in (("desync50", "d50"), ("desync100", "d100")):
            for s_ in FEW_SEEDS:
                q.append(s3(ds, f"s3_{d}_quantum_qcnn_s{s_}", s_, head="quantum",
                            ansatz="qcnn"))
                q.append(s3(ds, f"s3_{d}_clswide_qcnn_s{s_}", s_, head="classical",
                            ansatz="qcnn", control="wide"))
            q.append(s12(ds, "hybrid", f"s2_{d}_hybrid_s0", 0))
            q.append(tfb("rijsdijk", ds, f"rijsdijk_{d}_s0"))
    elif phase == "weights":
        # Re-run the variance-study configurations that we most want to inspect,
        # this time persisting the checkpoint. Distinct "_w" tags so the recorded
        # variance results are never overwritten.
        #
        # Why this is not optional: the campaign is NOT bit-reproducible from the
        # seed alone -- thread count changes floating-point summation order, and
        # near the trainability boundary that decides whether a run converges at
        # all. The stored weights are the only durable record of what was
        # actually trained, and the only way to diagnose the seeds that failed.
        for s_ in (1, 4, 5):        # the quantum seeds that never recovered
            q.append(s3("fixed", f"s3_f_quantum_qcnn_s{s_}_w", s_, head="quantum",
                        ansatz="qcnn"))
        q.append(s3("fixed", "s3_f_quantum_qcnn_s9_w", 9, head="quantum",
                    ansatz="qcnn"))       # worst run that did recover (2956)
        q.append(s3("fixed", "s3_f_clswide_qcnn_s2_w", 2, head="classical",
                    ansatz="qcnn", control="wide"))   # the one classical failure
    elif phase == "extras":
        # P3/P4 -- ansatz family, scan, and baselines retrained at our budget.
        # Baselines FIRST. Table III currently quotes these architectures at the
        # original 50k budget, which contradicts R2; retraining them at 45k is
        # what makes the budget claim true, so it must not be the part that gets
        # cut if time runs short. Ansatz family next, qubit/depth scan last.
        for ds, d in (("fixed", "f"), ("variable", "v")):
            q.append(tfb("rijsdijk", ds, f"rijsdijk_{d}_s0"))
            q.append(tfb("mlp", ds, f"mlp_{d}_s0"))
            q.append(cnnb(ds, f"cnnbest_{d}_s0"))
        for ds, d in (("fixed", "f"), ("variable", "v")):
            for s_ in FEW_SEEDS:
                q.append(s3(ds, f"s3_{d}_quantum_sel_s{s_}", s_, head="quantum", ansatz="sel"))
            q.append(s3(ds, f"s3_{d}_clsnarrow_sel_s0", 0, head="classical",
                        ansatz="sel", control="narrow"))
        for ds, d in (("fixed", "f"), ("variable", "v")):
            q.append(s3(ds, f"s3_{d}_quantum_qcnn_q8c3_s0", 0, head="quantum",
                        ansatz="qcnn", conv_layers=3))
            q.append(s3(ds, f"s3_{d}_quantum_qcnn_q10c3_s0", 0, head="quantum",
                        ansatz="qcnn", qubits=10, conv_layers=3))
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["variance", "encoder", "desync", "weights", "extras"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on tag")
    args = ap.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    q = queue_for(args.phase)
    if args.only:
        q = [j for j in q if args.only in j[0]]

    todo = [j for j in q if not os.path.exists(os.path.join(OUT, f"res_{j[0]}.json"))]
    print(f"=== phase {args.phase}: {len(q)} jobs, {len(q)-len(todo)} already done, "
          f"{len(todo)} to run ===", flush=True)
    for tag, py, script, a in q:
        mark = "DONE" if os.path.exists(os.path.join(OUT, f"res_{tag}.json")) else "todo"
        print(f"  [{mark}] {tag}", flush=True)
    if args.dry_run:
        return

    t_all = time.time()
    for i, (tag, py, script, a) in enumerate(todo, 1):
        log = os.path.join(LOGS, f"{tag}.log")
        print(f"\n>>> [{i}/{len(todo)}] {tag}  ({time.strftime('%H:%M:%S')})", flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            r = subprocess.run([py, "-u", script] + a, stdout=fh,
                               stderr=subprocess.STDOUT, cwd=os.path.dirname(script))
        dt = (time.time() - t0) / 60
        if r.returncode != 0:
            print(f"    FAILED rc={r.returncode} after {dt:.1f}min -- see {log}", flush=True)
            with open(log, encoding="utf-8", errors="replace") as fh:
                print("    " + "\n    ".join(fh.read().strip().splitlines()[-12:]), flush=True)
            continue
        rp = os.path.join(OUT, f"res_{tag}.json")
        if os.path.exists(rp):
            d = json.load(open(rp))
            print(f"    ok {dt:.1f}min  recov[GE<0.5]={d.get('recov_05')} "
                  f"min={d.get('ge_min'):.2f}@{d.get('ge_argmin')}", flush=True)
        else:
            print(f"    ok {dt:.1f}min but no res json?", flush=True)
    print(f"\n=== phase {args.phase} finished in {(time.time()-t_all)/3600:.2f} h ===", flush=True)


if __name__ == "__main__":
    main()
