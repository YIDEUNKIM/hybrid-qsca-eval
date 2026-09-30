# hybrid-qsca-eval

Code and result files for *A Controlled Evaluation of Hybrid Quantum–Classical Models
for Profiled Side-Channel Analysis* (ACM SAC 2027 submission).

The repository contains the shared evaluation protocol, every model trained in Sec. 3
(Stage-1/2/3 hybrids, classical controls and classical reference models), the
per-run outputs behind Tables 1–6, the frozen checkpoint used for all execution
studies, and the scripts and result files behind the finite-shot, device-noise and
`ibm_fez` tables (Tables 7–9). Table numbers follow the submitted manuscript.

## Layout

| Path | Contents |
|---|---|
| `code/qcnn/sca_common.py` | Protocol shared by every model: mask-free labels and data split (R1, R2), GE/SR estimator (R3), first-stays-below recovery rule (R4), validation GE proxy (R5) |
| `code/qcnn/qcnn_stages12.py` | Stage-1/2 models on PCA/amplitude inputs: a SEL circuit whose 256-outcome measurement is the posterior (S1), or the QCNN with a classical head (S2); R5 checkpoint selection |
| `code/qcnn/qcnn_stage3.py` | Stage-3 models (conv encoder, conv/pool or SEL circuit, classical head), dense and rank-1 controls, training with R5 checkpoint selection |
| `code/baselines/train_tf_baselines.py` | MLP_best and RL-CNN in TensorFlow |
| `code/baselines/train_cnnbest.py` | CNN_best in PyTorch |
| `code/baselines/metaqnn/` | Unmodified `one_cycle_lr.py` and MIT licence from [Rijsdijk et al.](https://github.com/AISyLab/Reinforcement-Learning-for-SCA), loaded by `train_tf_baselines.py` for the RL-CNN schedule |
| `code/grad_variance.py` | Initialization-time gradient-variance diagnostic (Table 2) |
| `code/run_campaign.py` | Sequential driver that launched the training campaign |
| `code/shot_noise.py` | Original finite-shot sweep of the frozen checkpoint |
| `code/hardware_attack.py` | Submission of the trained circuit to IBM Quantum (local encoder and head, circuit on the QPU) |
| `results/campaign/` | Result file, GE curve and training log of each run behind Tables 1–6, and the Table 2 output |
| `checkpoint/` | Frozen checkpoint used in Sec. 4 |
| `results/` | Outputs of the original runs for this checkpoint; the `sr/` replays are checked against them |
| `sr/` | Seeded replays that produce Tables 7–9 with GE and SR, and the IonQ noise-model study |
| `sr/results/` | JSON summaries behind Tables 7–9; `curves_hw_fez_*.npz` hold the measured device expectation values |
| `hardware/exports/` | IBM Quantum job exports of the three `ibm_fez` jobs (account id removed) |
| `analysis/` | R5 validation-group sizes, and checks that recompute Tables 1–6 and Tables 7–9 |
| `figures/make_fig_recovery_budget.py` | Figure 2 from the ten-seed runs of Table 3, and the counts and McNemar test quoted with it |

## Setup

```bash
pip install -r requirements.txt
mkdir data
```

Place the ASCAD databases in `data/`:

- Fixed key, including the desynchronized variants: `ASCAD.h5`, `ASCAD_desync50.h5` and
  `ASCAD_desync100.h5` from
  [ASCAD_data.zip](https://static.data.gouv.fr/resources/ascad/20180530-163000/ASCAD_data.zip)
  (`ASCAD_data/ASCAD_databases/`).
- Variable key: [ascad-variable.h5](https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190903-083349/ascad-variable.h5).

The `sr/` scripts read the fixed-key file from `data/ASCAD.h5`, or from `ASCAD_H5` if set.

`train_tf_baselines.py` also needs TensorFlow. The campaign ran it in a separate
environment with TensorFlow 2.21 (Keras 3), on the CPU.

## Training (Sec. 3)

Every training script writes `res_<tag>.json` (`recov_05` is T<sub>GE<0.5</sub>), the GE
curve, the attack-set probabilities and the weights to `--out-dir`. The seed fixes the
45k/5k profiling split, the mini-batch order and the model initialization. The Stage-1/2
and Stage-3 scripts use six CPU threads by default. The default `--data-dir` and
`--out-dir` are the campaign's Windows paths, so pass both explicitly.

Stage 3 (`qcnn_stage3.py`):

```bash
C="python code/qcnn/qcnn_stage3.py --data-dir data --out-dir out"
$C --dataset fixed --head quantum --seed 0                      # conv/pool hybrid
$C --dataset fixed --head classical --control wide --seed 0     # dense control, 144 params
$C --dataset fixed --head classical --control narrow --seed 0   # rank-1 control, 16 params
$C --dataset fixed --head quantum --ansatz sel --seed 0         # SEL circuit
$C --dataset variable --head quantum --seed 0                   # variable key
$C --dataset desync50 --head quantum --seed 0                   # also desync100
$C --dataset fixed --head quantum --conv-layers 3 --seed 0      # depth scan; --qubits 10
```

Table 3 repeats the first three commands over seeds 0–9.

Stage 1 and Stage 2 (`qcnn_stages12.py`). Standardization and the 256-component PCA
are fitted on the 45k training split only:

```bash
S="python code/qcnn/qcnn_stages12.py --data-dir data --out-dir out"
$S --dataset fixed --mode pure --seed 0      # S1: PCA/amplitude, SEL, no classifier
$S --dataset fixed --mode hybrid --seed 0    # S2: PCA/amplitude, QCNN, classical head
```

Classical reference models. They train on the same 45k traces with no validation split
and a fixed epoch count (RL-CNN 50, MLP_best 100, CNN_best 75), and are scored with
their last weights:

```bash
B="--dataset fixed --data-dir data --out-dir out"
python code/baselines/train_tf_baselines.py --model mlp $B
python code/baselines/train_tf_baselines.py --model rijsdijk $B
python code/baselines/train_cnnbest.py $B
```

Gradient-variance diagnostic (Table 2), which needs no data:

```bash
python code/grad_variance.py --out out/grad_variance.json
```

`run_campaign.py --phase <variance|encoder|desync|extras|weights>` lists and runs the
campaign queue under the tags used in `results/campaign/`. It passes neither
`--data-dir` nor `--out-dir` and calls the `env/` and `tfenv/` virtual environments of
the campaign machine, so elsewhere run the commands above instead.

### Per-run outputs

`results/campaign/` holds, for each of 81 runs, `res_<tag>.json`, the GE curve
`ge_curves/ge_<tag>.npy` (N = 1…3000) and the training log `logs/<tag>.log`.
`summary_all_runs.csv` lists the result files in one table, and `grad_variance.json`
is the Table 2 output. Attack-set probabilities (5.2 GB) and the weights of the
campaign runs (0.7 GB) are not included because of their size.

```bash
python analysis/check_campaign_tables.py      # prints Tables 1-6 and the Sec. 3 numbers
python figures/make_fig_recovery_budget.py    # Figure 2, written to out/fig_recovery_budget.pdf
```

Figure 2 counts, for each attack-trace budget B, the ten-seed runs of Table 3 whose
T<sub>GE<0.5</sub> is at most B. The script also prints the counts at B = 250, 1000 and
3000, the paired outcomes of S3-QCNN and the dense control, and the two-sided exact
McNemar p-value quoted in Sec. 3.

Tags read as `s3_f_quantum_qcnn_s7`: Stage 3, fixed key, quantum middle block, QCNN
circuit, seed 7.

- Database: `f` fixed key, `v` variable key, `d50`/`d100` desynchronized.
- Middle block: `quantum`, `clswide` (dense control, `--control wide`), `clsnarrow`
  (rank-1 control; with `sel`, a 144-parameter rank-6 block).
- `q8c3`, `q10c3`: 8 or 10 qubits with three encoder blocks.
- `s1_*_pure` is Stage 1 and `s2_*_hybrid` is Stage 2. `rijsdijk` is RL-CNN, `mlp` is
  MLP_best and `cnnbest` is CNN_best.

The manifest below lists the runs behind each table. `check_campaign_tables.py` and
the Figure 2 script read exactly these tags, so the weight-saving rerun
`s3_f_quantum_qcnn_s0_w` enters none of them.

| Table | Rows | Runs |
|---|---|---|
| 1 | S1, S2 | `s1_{f,v}_pure_s0`, `s2_{f,v}_hybrid_s0` |
| | S3-QCNN | `s3_f_quantum_qcnn_s0`–`s9`, `s3_v_quantum_qcnn_s0`–`s2` |
| | S3-SEL | `s3_{f,v}_quantum_sel_s0`–`s2` |
| 2 | QCNN, SEL | `grad_variance.json` |
| 3, Fig. 2 | QCNN, dense, rank-1 | `s3_f_{quantum,clswide,clsnarrow}_qcnn_s0`–`s9` |
| 4 | MLP_best, CNN_best, RL-CNN | `{mlp,cnnbest,rijsdijk}_{f,v}_s0` |
| 5 | n = 8, ℓ = 2 | the S3-QCNN runs of Table 1 |
| | n = 8, ℓ = 3 | `s3_f_quantum_qcnn_q8c3_s0`, `s3_v_quantum_qcnn_q8c3_s0`–`s2` |
| | n = 10, ℓ = 3 | `s3_{f,v}_quantum_qcnn_q10c3_s0` |
| 6 | S3-QCNN, dense control | `s3_{d50,d100}_{quantum,clswide}_qcnn_s0`–`s2` |
| | S2, RL-CNN | `s2_{d50,d100}_hybrid_s0`, `rijsdijk_{d50,d100}_s0` |

Runs that no table uses:

- `s1_{f,v}_pure_s1` and `s2_{f,v}_hybrid_s1`, a second seed of S1 and S2. All four
  fail to recover.
- `s3_{f,v}_clsnarrow_sel_s0`, the rank-6 classical replacement of the SEL circuit. It
  recovers at 2,569 traces on the fixed-key database and fails on the variable-key one.
- `s3_f_quantum_qcnn_s0_w`, the training run of the Sec. 4 checkpoint (see below).

### Notes on the records

- The Time column of Table 3 is the median of `wallclock_s` over the ten runs.
- Training loss and accuracy in the logs are running means over each epoch's
  mini-batches. The values quoted in Sec. 3 are those of the last trained epoch, not
  those of the restored checkpoint.
- Table 5 lists whole-model parameter counts, taken from the `params` field of the
  variable-key (1,400-sample) and fixed-key (700-sample) runs.
- `run_campaign.py` queued only seed 0 of `s3_v_quantum_qcnn_q8c3`. Seeds 1 and 2 of
  that configuration and `s3_f_quantum_qcnn_s0_w` were launched separately with
  `qcnn_stage3.py`, and their logs record the data and model settings. Of the `weights`
  phase, only the seed-0 run above is part of this record.
- `grad_variance.py` draws each input from a bank of 300 synthetic angle vectors,
  N(π/2, 0.5) clipped to [0, π], not from encoded ASCAD traces as its docstring says.
  All circuit parameters are redrawn uniformly from [0, 2π) at each of the 300 draws.
  C is ⟨Z⟩ on wire 0, which is the first kept wire of the QCNN. The derivative is taken
  with respect to the first entry of the first parameter tensor: the first
  R<sub>Y</sub> angle on wire 0 for the QCNN, and the first Rot angle on wire 0, an
  R<sub>Z</sub> phase, for SEL.
- Two docstrings predate the final protocol. `run_campaign.py` describes 60- and
  100-epoch budgets, whereas the scripts stop on the R5 proxy with caps of 250 epochs
  (Stage 3) and 200 epochs (Stages 1/2). `train_tf_baselines.py` mentions retraining
  at 50k, whereas the runs use 45k training traces, as their `[cfg]` log lines show.

## Execution studies (Sec. 4)

All Sec. 4 studies use `checkpoint/w_s3_f_quantum_qcnn_s0_w.pt`. It comes from a
separate seed-0 training run of the default Stage-3 model, `s3_f_quantum_qcnn_s0_w`,
and is not one of the ten runs in Table 3. Under exact evaluation of the full 10k
fixed-key pool with W = 3000, it recovers in 332 traces.

Its training log and result file are in `results/campaign/`. Its GE curve there is
identical to `results/ge_shots_f_s0_exact.npy`, which `sr/exact.py` reproduces from the
checkpoint. The ten-seed run `s3_f_quantum_qcnn_s0` has the same recorded settings, but
the two training losses already differ at the first epoch (5.5551 and 5.5549). Both
runs restore the epoch-15 weights, and they recover in 219 and 332 traces.

Run from `sr/`, starting with `exact.py`, which writes the probabilities that the other
scripts reuse:

```bash
cd sr
python exact.py                                  # exact reference: 332 / 296 / 163
python shots.py                                  # Table 7, 7 shot settings x 5 realizations
python shots.py --pool 1500 --shots 512 2048     # Table 8, sampling only
python devnoise.py                               # Table 8, IBM fake_sherbrooke model
python ionq_devnoise.py --env-file <file>        # Table 8, IonQ forte-enterprise-1 model
python shots.py --pool 650 --shots 256           # Table 9, sampling only
python hw_sr_from_export.py ../hardware/exports/job-*   # Table 9, ibm_fez
cd .. && python analysis/check_tables.py         # prints Tables 7-9 from sr/results
```

`shots.py` and `exact.py` reproduce the original GE curves in `results/` bit for bit.
`devnoise.py` fixes the simulator seed of each realization.

`ionq_devnoise.py` sends jobs to the IonQ cloud simulator, not to a QPU. Its API key is
read from `IONQ_API_KEY` or `--env-file`.

`code/hardware_attack.py` submits to IBM Quantum with a token taken from
`QISKIT_IBM_TOKEN`. It loads the checkpoint from `--out-dir`:

```bash
python code/hardware_attack.py --backend ibm:ibm_fez --n-attack 650 --shots 256 \
    --data-dir data --out-dir checkpoint
```

`hw_sr_from_export.py` rescores the submitted jobs from the exports without a token.

Shot counts are per measurement basis. The four ⟨Z⟩ values share one basis and the
four ⟨X⟩ values share the other, so one attack trace costs 2S shots. The `total_shots`
field that `hardware_attack.py` writes counts n × 8 × S instead, one per observable.
