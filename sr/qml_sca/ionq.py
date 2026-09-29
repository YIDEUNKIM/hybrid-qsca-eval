"""Journaled official IonQ Forte-1 noisy SIMULATOR execution (never a QPU).

Only local QIS serialization is reused from qiskit-ionq. HTTP POST is attempted
once: SDK automatic submission retries can duplicate jobs after a lost response.
Resume with the same journal to poll the saved job. An ambiguous submission is
never resubmitted; use adopt_job_id after locating its unique name in IonQ Cloud.

Primary references, checked 2026-09-10:
https://docs.ionq.com/features/simulation-with-noise-models
https://docs.ionq.com/api-reference/v0.4/jobs/create-job
https://docs.ionq.com/api-reference/v0.4/schemas/results-formats
https://docs.ionq.com/api-reference/v0.4/jobs/get-job-artifact

The public service supplies the proprietary current Forte model and compiler.
Their internal parameters/calibration date are not assumed to be available.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import time
from uuid import UUID

import requests
from qiskit import QuantumCircuit
from qiskit_ionq.helpers import qiskit_circ_to_ionq_circ

API_ROOT = "https://api.ionq.co/v0.4"
BACKEND = "simulator"
NOISE_MODEL = "forte-1"
MAX_CIRCUITS = 5000
MAX_GATES = 150000
MAX_BODY_BYTES = 10000000


class CloudError(RuntimeError):
    """Sanitized service or result error, without credentials/HTTP request data."""


class SubmissionAmbiguous(CloudError):
    """A POST may have reached the server. It must not be repeated."""


class JobPending(CloudError):
    """The saved job is unfinished; invoke again with the same journal."""


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _sha(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temp.open("w") as stream:
            stream.write(_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def _locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CloudError("Another process is using this submission journal.") from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def load_api_key(env_file=None):
    """Read only IONQ_API_KEY; do not source a shell file or export its contents."""
    if env_file is None:
        key = os.environ.get("IONQ_API_KEY", "").strip()
    else:
        key = ""
        for line in Path(env_file).expanduser().read_text().splitlines():
            match = re.fullmatch(r"\s*(?:export\s+)?IONQ_API_KEY\s*=\s*(.*?)\s*", line)
            if match:
                value = match.group(1)
                if value[:1] in ("'", '"'):
                    if len(value) < 2 or value[-1] != value[0]:
                        raise CloudError("Malformed quoted IONQ_API_KEY assignment.")
                    value = value[1:-1]
                else:
                    value = value.split(" #", 1)[0].strip()
                key = value
    if not key or any(ch.isspace() for ch in key):
        raise CloudError("A valid nonempty IONQ_API_KEY is required.")
    return key


def build_payload(circuits, shots, seed):
    """Build a strictly simulator-only v0.4 QIS batch and readout provenance."""
    if not circuits or len(circuits) > MAX_CIRCUITS:
        raise ValueError("Circuit count must lie in [1, 5000].")
    if isinstance(shots, bool) or not isinstance(shots, int) or not 1 <= shots <= 1000000:
        raise ValueError("shots must be an integer in [1, 1000000].")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 1 <= seed < 2**31:
        raise ValueError("noise seed must be an integer in [1, 2**31).")
    entries, readouts = [], []
    for index, circuit in enumerate(circuits):
        if not isinstance(circuit, QuantumCircuit) or circuit.parameters:
            raise ValueError("Every circuit must be a bound QuantumCircuit.")
        if not 1 <= circuit.num_qubits <= 29 or circuit.num_clbits < 1:
            raise ValueError("Measured circuits must have 1 to 29 qubits.")
        mapping = [None] * circuit.num_clbits
        measured = False
        for inst in circuit.data:
            name = inst.operation.name
            if name == "barrier":
                continue
            if name == "measure":
                measured = True
                c = circuit.find_bit(inst.clbits[0]).index
                q = circuit.find_bit(inst.qubits[0]).index
                if mapping[c] is not None or q in mapping:
                    raise ValueError("Readouts must be distinct terminal measurements.")
                mapping[c] = q
            elif (
                measured or name in ("reset", "delay") or getattr(inst.operation, "condition", None)
            ):
                raise ValueError(
                    "Only unitary QIS circuits with terminal measurements are supported."
                )
        if any(value is None for value in mapping):
            raise ValueError("Every classical bit must be measured exactly once.")
        gates, _, sdk_mapping = qiskit_circ_to_ionq_circ(circuit, "qis")
        if list(sdk_mapping) != mapping:
            raise ValueError("SDK measurement mapping differs from explicit circuit mapping.")
        entries.append({"name": f"circuit_{index:06d}", "circuit": gates})
        readouts.append(
            {
                "qubits": circuit.num_qubits,
                "classical_to_qubit": mapping,
                "gate_counts": dict(circuit.count_ops()),
                "depth": circuit.depth(),
            }
        )
    if sum(len(entry["circuit"]) for entry in entries) > MAX_GATES:
        raise ValueError("IonQ batch exceeds 150000 gates.")
    payload = {
        "type": "ionq.multi-circuit.v1",
        "backend": BACKEND,
        "shots": shots,
        "noise": {"model": NOISE_MODEL, "seed": seed},
        "settings": {"error_mitigation": {"debiasing": False}},
        "input": {
            "gateset": "qis",
            "qubits": max(c.num_qubits for c in circuits),
            "circuits": entries,
        },
    }
    fingerprint = _sha({"payload": payload, "readouts": readouts})
    payload["name"] = "qml-sca-forte1-" + fingerprint[:24]
    payload["metadata"] = {
        "qml_sca_request_sha256": fingerprint,
        "measurement_order": "Qiskit classical bit n-1 to 0",
        "sdk_serializer": "qiskit-ionq " + version("qiskit-ionq"),
    }
    if len(_json(payload).encode()) > MAX_BODY_BYTES:
        raise ValueError("IonQ payload exceeds 10 MB.")
    return payload, readouts


def _uuid(value):
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        raise CloudError("Service returned an invalid job/artifact UUID.") from None


def _request(session, key, method, path, payload=None):
    # Fixed origin and allowlisted paths also prevent an artifact URL from
    # forwarding credentials to an unrelated host. No redirect or POST retry.
    if not re.fullmatch(
        r"/jobs(?:/[0-9a-f-]{36}(?:/(?:artifacts/[A-Za-z0-9_-]{1,128}|results/probabilities(?:/aggregated)?))?)?",
        path,
    ):
        raise CloudError("Request path is outside the simulator job allowlist.")
    if method == "POST" and (
        path != "/jobs"
        or payload.get("backend") != BACKEND
        or payload.get("noise", {}).get("model") != NOISE_MODEL
    ):
        raise CloudError("Only Forte-1 noisy simulator submissions are permitted.")
    if method not in ("GET", "POST"):
        raise CloudError("Unsupported HTTP operation.")
    try:
        response = session.request(
            method,
            API_ROOT + path,
            headers={"Authorization": "apiKey " + key, "Content-Type": "application/json"},
            json=payload,
            timeout=30,
            allow_redirects=False,
        )
    except requests.RequestException:
        raise CloudError(
            "IonQ network request failed; credentials and request details omitted."
        ) from None
    if response.status_code not in (200, 201):
        raise CloudError(f"IonQ HTTP {response.status_code}; raw error response omitted.")
    try:
        return response.json()
    except (ValueError, TypeError):
        raise CloudError("IonQ response is not valid JSON.") from None


def _validate_job(job, payload):
    if job.get("backend") != BACKEND:
        raise CloudError("Saved/retrieved job is not the permitted simulator backend.")
    # Older service responses omit noise; record that omission, never infer a
    # reported calibration/model version. A contradictory response is fatal.
    noise = job.get("noise")
    if noise is not None and noise.get("model") != NOISE_MODEL:
        raise CloudError("Retrieved job reports a different noise model.")
    if job.get("shots") is not None and int(job["shots"]) != payload["shots"]:
        raise CloudError("Retrieved shot count differs from the submission.")
    if ((job.get("settings") or {}).get("error_mitigation") or {}).get("debiasing"):
        raise CloudError("Unexpected debiasing would change the shot histogram contract.")


def decode_counts(raw, result_format, readout, shots):
    """Preserve shot-aware histograms, marginalizing only unmeasured wires.

    v1 state keys are decimal integers (qubit 0 is least significant).
    v2 QIS output_all keys are zero-padded binary integers. Never statistically
    resample probabilities: p*shots must already be an integer within 1e-6.
    """
    probability = "probabilities" in result_format
    v2 = result_format.endswith(".v2")
    if v2:
        kind = "probabilities" if probability else "histogram"
        try:
            distribution = raw[kind]["registers"]["output_all"]
        except (KeyError, TypeError):
            raise CloudError("Expected the full QIS output_all register in v2 results.") from None
    else:
        distribution = raw
    if not isinstance(distribution, dict) or not distribution:
        raise CloudError("Result distribution is empty or malformed.")
    result = Counter()
    mapping = readout["classical_to_qubit"]
    for key, value in distribution.items():
        if not isinstance(key, str) or not re.fullmatch(r"[01]+" if v2 else r"[0-9]+", key):
            raise CloudError("Invalid state key in IonQ results.")
        state = int(key, 2 if v2 else 10)
        if state >= 2 ** readout["qubits"]:
            raise CloudError("Result state exceeds circuit width.")
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value < 0
        ):
            raise CloudError("Invalid histogram value.")
        scaled = value * shots if probability else value
        count = round(scaled)
        if abs(scaled - count) > 1e-6:
            raise CloudError(
                "Noisy probabilities do not lie on the requested shot grid; refusing resampling."
            )
        measured = sum(((state >> qubit) & 1) << cbit for cbit, qubit in enumerate(mapping))
        if count:
            result[format(measured, f"0{len(mapping)}b")] += count
    if sum(result.values()) != shots:
        raise CloudError("IonQ counts do not sum to the requested shots.")
    return dict(sorted(result.items()))


def _artifact(session, key, job_id, descriptor):
    if descriptor.get("id"):
        artifact_id = str(descriptor["id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", artifact_id):
            raise CloudError("Unsafe artifact identifier returned.")
        # The live v0.4 service uses opaque IDs as well as UUIDs. The aggregate
        # alias is explicitly exposed via its legacy compatibility endpoint.
        if artifact_id == "probabilities-aggregate":
            return _request(
                session, key, "GET", f"/jobs/{_uuid(job_id)}/results/probabilities/aggregated"
            )
        return _request(session, key, "GET", f"/jobs/{_uuid(job_id)}/artifacts/{artifact_id}")
    # Legacy v0.4 descriptors use a URL. Use the fixed known endpoint instead.
    if descriptor.get("url"):
        expected = f"/v0.4/jobs/{_uuid(job_id)}/results/probabilities"
        returned = descriptor["url"]
        if returned not in (
            expected,
            expected + "/aggregated",
            "https://api.ionq.co" + expected,
            "https://api.ionq.co" + expected + "/aggregated",
        ):
            raise CloudError(
                "Legacy result URL differs from the expected same-origin job endpoint."
            )
        suffix = "/aggregated" if returned.endswith("/aggregated") else ""
        return _request(session, key, "GET", f"/jobs/{_uuid(job_id)}/results/probabilities{suffix}")
    raise CloudError("No supported artifact identifier returned.")


def _fetch_results(session, key, job, journal, path):
    readouts = journal["readouts"]
    shots = journal["payload"]["shots"]
    results = job.get("results") or {}
    children = job.get("child_job_ids") or job.get("children") or []
    if len(children) != len(readouts):
        raise CloudError("Child-job count differs from submitted circuit count.")
    children = [_uuid(child) for child in children]
    aggregate_format = "ionq.result.probabilities-aggregate.json.v1"
    descriptor = results.get(aggregate_format) or results.get("probabilities")
    if descriptor:
        raw = journal.get("raw_aggregate")
        if raw is None:
            raw = _artifact(session, key, job["id"], descriptor)
            journal["raw_aggregate"] = raw
            _save(path, journal)
        if not isinstance(raw, dict) or set(raw) != set(children):
            raise CloudError("Aggregate result child IDs differ from the ordered job record.")
        return [
            decode_counts(raw[child], "ionq.result.probabilities.json.v1", readout, shots)
            for child, readout in zip(children, readouts)
        ]
    counts = []
    saved_children = journal.setdefault("child_results", {})
    formats = [
        "ionq.result.histogram.json.v1",
        "ionq.result.histogram.json.v2",
        "ionq.result.probabilities.json.v1",
        "ionq.result.probabilities.json.v2",
        "probabilities",
    ]
    for child, readout in zip(children, readouts):
        if child not in saved_children:
            child_job = _request(session, key, "GET", f"/jobs/{child}")
            _validate_job(child_job, journal["payload"])
            artifacts = child_job.get("results") or {}
            chosen = next((fmt for fmt in formats if fmt in artifacts), None)
            if chosen is None:
                raise CloudError("Child job has no supported histogram/probability artifact.")
            raw = _artifact(session, key, child, artifacts[chosen])
            saved_children[child] = {"job": child_job, "format": chosen, "raw": raw}
            _save(path, journal)
        saved = saved_children[child]
        counts.append(decode_counts(saved["raw"], saved["format"], readout, shots))
    return counts


def simulate_measurements(
    circuits,
    backend_name,
    shots,
    seed,
    *,
    journal_path,
    env_file=None,
    timeout=900,
    poll_interval=5,
    adopt_job_id=None,
):
    """Return (joint_counts, metadata); resume exactly one saved cloud job.

    backend_name is only 'ionq_forte1_cloud'. timeout=0 submits/polls once and
    raises JobPending for unfinished work. A new call resumes without a POST.
    """
    if backend_name != "ionq_forte1_cloud":
        raise ValueError("The only allowed backend is ionq_forte1_cloud (simulator).")
    if timeout < 0 or not 0 < poll_interval <= 30:
        raise ValueError("timeout must be nonnegative and poll_interval in (0,30].")
    payload, readouts = build_payload(circuits, shots, seed)
    contract = {"payload": payload, "readouts": readouts}
    path = Path(journal_path)
    with _locked(path):
        if path.exists():
            journal = json.loads(path.read_text())
            if journal.get("contract_sha256") != _sha(contract):
                raise CloudError("Journal differs from the circuit/shot/seed contract.")
        else:
            journal = {
                "format_version": 1,
                **contract,
                "contract_sha256": _sha(contract),
                "phase": "prepared",
                "created_utc": _now(),
            }
            _save(path, journal)
        if journal["phase"] == "completed":
            return journal["counts"], journal["backend_metadata"]
        key = load_api_key(env_file)
        with requests.Session() as session:
            # Explicitly disable urllib3 retries, including POST.
            session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
            if adopt_job_id:
                recovered = _request(session, key, "GET", f"/jobs/{_uuid(adopt_job_id)}")
                _validate_job(recovered, payload)
                if (
                    recovered.get("name") != payload["name"]
                    or (recovered.get("metadata") or {}).get("qml_sca_request_sha256")
                    != payload["metadata"]["qml_sca_request_sha256"]
                ):
                    raise CloudError(
                        "Adopted job does not match the unique submission fingerprint."
                    )
                journal.update(
                    job_id=_uuid(adopt_job_id), phase="submitted", recovered_job=recovered
                )
                _save(path, journal)
            if not journal.get("job_id"):
                if journal["phase"] != "prepared":
                    raise SubmissionAmbiguous(
                        "Submission may have succeeded. Locate job named "
                        + payload["name"]
                        + "; resume using adopt_job_id, never a new journal."
                    )
                journal.update(phase="submitting", submit_started_utc=_now())
                _save(path, journal)
                try:
                    response = _request(session, key, "POST", "/jobs", payload)
                    job_id = _uuid(response["id"])
                except (CloudError, KeyError, TypeError):
                    journal.update(phase="submission_ambiguous", submit_error_utc=_now())
                    _save(path, journal)
                    raise SubmissionAmbiguous(
                        "Submission response was not confirmed; do not resubmit. Journal: "
                        + str(path)
                    ) from None
                journal.update(job_id=job_id, phase="submitted", submit_response=response)
                _save(path, journal)
            job_id = _uuid(journal["job_id"])
            deadline = time.monotonic() + timeout
            while True:
                job = _request(session, key, "GET", f"/jobs/{job_id}")
                _validate_job(job, payload)
                journal.update(last_job=job, last_polled_utc=_now())
                _save(path, journal)
                if job.get("status") == "completed":
                    break
                if job.get("status") in ("failed", "canceled"):
                    journal["phase"] = job["status"]
                    _save(path, journal)
                    raise CloudError(
                        f"Simulator job {job_id} {job['status']}; details saved in journal."
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise JobPending(f"Simulator job {job_id} pending; resume journal {path}.")
                time.sleep(min(poll_interval, remaining))
            counts = _fetch_results(session, key, job, journal, path)
        metadata = {
            "provider": "IonQ official cloud",
            "backend": BACKEND,
            "noise_model_requested": NOISE_MODEL,
            "noise_reported": job.get("noise"),
            "noise_seed": seed,
            "job_id": job_id,
            "child_job_ids": job.get("child_job_ids") or job.get("children"),
            "job_name": payload["name"],
            "job_stats": job.get("stats"),
            "settings_requested": payload["settings"],
            "settings_reported": job.get("settings"),
            "execution_duration_ms": job.get("execution_duration_ms"),
            "submitted_at": job.get("submitted_at"),
            "completed_at": job.get("completed_at"),
            "billing_reported": {
                key: val for key, val in job.items() if "cost" in key or "billing" in key
            },
            "shots_per_circuit": shots,
            "circuits": len(circuits),
            "total_shots": shots * len(circuits),
            "gateset": "qis",
            "compiler": "IonQ cloud default QIS compiler; no local native compilation",
            "compiler_seed": None,
            "calibration_date": None,
            "model_version": None,
            "readouts": readouts,
            "sdk_serializer_version": version("qiskit-ionq"),
            "count_conversion": "server shot histogram; p*shots integer-validated; no client sampling",
            "request_sha256": payload["metadata"]["qml_sca_request_sha256"],
            "journal_path": str(path.resolve()),
            "limitations": [
                "Official noisy simulation, not QPU execution.",
                "Model internals and calibration/version are not exposed by the documented API.",
                "Seed controls simulator randomness; cloud compiler reproducibility is not guaranteed across service updates.",
                "The noisy simulator does not implement hardware debiasing.",
            ],
        }
        journal.update(
            phase="completed", counts=counts, backend_metadata=metadata, completed_local_utc=_now()
        )
        _save(path, journal)
        return counts, metadata
