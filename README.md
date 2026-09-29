# hybrid-qsca-eval

Code and result files for *A Controlled Evaluation of Hybrid Quantum–Classical Models
for Profiled Side-Channel Analysis* (ACM SAC 2027 submission).

The repository contains the shared evaluation protocol, the Stage-3 hybrid models and
their classical controls, the frozen checkpoint used for all execution studies, and
the scripts and result files behind the finite-shot, device-noise and `ibm_fez`
tables.

## Layout

| Path | Contents |
|---|---|
| `code/qcnn/sca_common.py` | Protocol shared by every model: mask-free labels and data split (R1, R2), GE/SR estimator (R3), first-stays-below recovery rule (R4), validation GE proxy (R5) |
| `code/qcnn/qcnn_stage3.py` | Stage-3 models (conv encoder, conv/pool or SEL circuit, classical head), wide and rank-1 controls, training with R5 checkpoint selection |
| `code/shot_noise.py` | Original finite-shot sweep of the frozen checkpoint |
| `code/hardware_attack.py` | Submission of the trained circuit to IBM Quantum (local encoder and head, circuit on the QPU) |
| `checkpoint/` | Frozen checkpoint used in Sec. 4 |
| `results/` | Outputs of the original runs for this checkpoint; the `sr/` replays are checked against them |
| `sr/` | Seeded replays that produce Tables 7–9 with GE and SR, and the IonQ noise-model study |
| `sr/results/` | JSON summaries behind Tables 7–9; `curves_hw_fez_*.npz` hold the measured device expectation values |
| `hardware/exports/` | IBM Quantum job exports of the three `ibm_fez` jobs (account id removed) |
| `analysis/` | R5 validation-group sizes and a check that recomputes Tables 7–9 |

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

## Training (Sec. 3)

`qcnn_stage3.py` trains one run and writes `res_<tag>.json` (`recov_05` is
T<sub>GE<0.5</sub>), the GE curve, the attack-set probabilities and the selected weights
to `--out-dir`. The seed fixes the 45k/5k profiling split, the mini-batch order and the
model initialization. Runs use six CPU threads by default.

```bash
C="python code/qcnn/qcnn_stage3.py --data-dir data --out-dir out"
$C --dataset fixed --head quantum --seed 0                      # conv/pool hybrid
$C --dataset fixed --head classical --control wide --seed 0     # wide control, 144 params
$C --dataset fixed --head classical --control narrow --seed 0   # rank-1 control, 16 params
$C --dataset fixed --head quantum --ansatz sel --seed 0         # SEL circuit
$C --dataset variable --head quantum --seed 0                   # variable key
$C --dataset desync50 --head quantum --seed 0                   # also desync100
$C --dataset fixed --head quantum --conv-layers 3 --seed 0      # depth scan; --qubits 10
```

Table 2 repeats the first three commands over ten seeds.

`analysis/validation_groups.py` prints the key-byte groups of the R5 proxy for each
seed: one group with L = 512 on the fixed-key database, and L = 8–11 over seeds 0–9 on
the variable-key database.

## Execution studies (Sec. 4)

All Sec. 4 studies use `checkpoint/w_s3_f_quantum_qcnn_s0_w.pt`. It comes from a
separate seed-0 training run of the default Stage-3 model and is not one of the ten runs
in Table 2. Under exact evaluation of the full 10k fixed-key pool with W = 3000, it
recovers in 332 traces.

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

## Not yet included

- The Stage-1/2 (PCA/amplitude) training script.
- The scripts for the classical reference models (MLP_best, CNN_best, RL-CNN).
- The initialization-time gradient-variance diagnostic (Table 3).
- The per-run outputs of the ten-seed and three-seed campaigns behind Tables 1, 2, 4, 5
  and 6.
