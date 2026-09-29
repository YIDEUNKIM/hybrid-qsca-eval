"""Provider adapters for the existing eight-qubit QCNN quantum block.

No model or training loop is defined here. Weights use the original TorchLayer
names and shapes: conv1, pool1, conv2. Only the source's angle-input QCNN with
8 qubits is supported; amplitude encoding and SEL are not silently mapped
to this circuit. Feature order is Z on surviving wires followed by X on them.

Prefer attaching both measurement groups to the logical circuit *before*
transpiling. Classical bit j always stores surviving logical wire j's result,
so counts remain meaningful after layout/routing. For an already-transpiled
unitary circuit, final_index_layout() resolves original logical wires to the
current physical circuit. Basis rotations appended in that case must themselves
be translated to the target gate set before execution.

Qiskit references:
https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.QuantumCircuit
https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.TranspileLayout
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

WEIGHT_NAMES = ("conv1", "pool1", "conv2")
METADATA_KEY = "qml_sca_qcnn"


def _validate_nq(nq: int) -> int:
    if isinstance(nq, (bool, np.bool_)) or nq != 8:
        raise ValueError("This adapter supports the source QCNN at nq=8 only.")
    return int(nq)


def qcnn_parameter_shapes(nq: int) -> dict[str, tuple[int, ...]]:
    nq = _validate_nq(nq)
    return {"conv1": (nq // 2, 3), "pool1": (nq // 2,), "conv2": (nq // 4, 3)}


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("Angles and circuit weights must be finite.")
    return result


def _weights(weights: Mapping[str, Any], nq: int) -> dict[str, np.ndarray]:
    shapes = qcnn_parameter_shapes(nq)
    if set(weights) != set(shapes):
        raise ValueError(f"Expected exactly the original QCNN weight names {WEIGHT_NAMES}.")
    result = {}
    for name, shape in shapes.items():
        value = _numpy(weights[name])
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, received {value.shape}.")
        result[name] = value.copy()
    return result


def qcnn_weights_from_state_dict(
    state_dict: Mapping[str, Any], nq: int, prefix: str = "mid."
) -> dict[str, np.ndarray]:
    """Copy original model.mid weights; pass prefix='' for mid.state_dict()."""
    required = [prefix + name for name in WEIGHT_NAMES]
    missing = [name for name in required if name not in state_dict]
    if missing:
        raise KeyError(f"Missing original QCNN checkpoint weights: {missing}")
    return _weights({name: state_dict[prefix + name] for name in WEIGHT_NAMES}, nq)


def flatten_qcnn_weights(weights: Mapping[str, Any], nq: int) -> np.ndarray:
    checked = _weights(weights, nq)
    return np.concatenate([checked[name].ravel(order="C") for name in WEIGHT_NAMES])


def unflatten_qcnn_weights(values: Any, nq: int) -> dict[str, np.ndarray]:
    values = _numpy(values)
    shapes = qcnn_parameter_shapes(nq)
    size = sum(int(np.prod(shape)) for shape in shapes.values())
    if values.shape != (size,):
        raise ValueError(f"Flat weights must have shape ({size},), received {values.shape}.")
    result, offset = {}, 0
    for name, shape in shapes.items():
        count = int(np.prod(shape))
        result[name] = values[offset : offset + count].reshape(shape).copy()
        offset += count
    return result


def _operations(angles: Sequence[Any], weights: Mapping[str, np.ndarray]):
    """Preserve source order, including CRZ control=discarded, target=kept."""
    nq = len(angles)
    for wire, angle in enumerate(angles):
        yield "ry", angle, (wire,)
    pairs = [(wire, wire + 1) for wire in range(0, nq, 2)]
    for index, (a, b) in enumerate(pairs):
        w = weights["conv1"][index]
        yield "ry", w[0], (a,)
        yield "ry", w[1], (b,)
        yield "cx", None, (a, b)
        yield "ry", w[2], (b,)
    for index, (a, b) in enumerate(pairs):
        yield "crz", weights["pool1"][index], (b, a)
    kept = list(range(0, nq, 2))
    for index in range(len(kept) // 2):
        a, b = kept[2 * index : 2 * index + 2]
        w = weights["conv2"][index]
        yield "ry", w[0], (a,)
        yield "ry", w[1], (b,)
        yield "cx", None, (a, b)
        yield "ry", w[2], (b,)


def _build(angles: Sequence[Any], weights: Mapping[str, Any]):
    from qiskit import QuantumCircuit

    nq = _validate_nq(len(angles))
    checked = _weights(weights, nq)
    circuit = QuantumCircuit(nq, name=f"source_qcnn_q{nq}")
    for gate, parameter, wires in _operations(angles, checked):
        if gate == "cx":
            circuit.cx(*wires)
        else:
            getattr(circuit, gate)(parameter, *wires)
    kept = list(range(0, nq, 2))
    circuit.metadata = {
        METADATA_KEY: {
            "ansatz": "source_angle_qcnn",
            "nq": nq,
            "logical_readout_wires": kept,
            "feature_order": [f"{basis}{wire}" for basis in ("Z", "X") for wire in kept],
        }
    }
    return circuit


def build_qcnn_circuit(angles: Any, weights: Mapping[str, Any]):
    """Translate one original learned-angle vector and frozen weights to Qiskit."""
    angles = _numpy(angles)
    if angles.ndim != 1:
        raise ValueError("build_qcnn_circuit expects one angle vector, not a batch.")
    return _build(angles, weights)


def build_qcnn_template(nq: int, weights: Mapping[str, Any]):
    """Return (circuit, input_parameters) for compile-once, bind-many inference."""
    from qiskit.circuit import ParameterVector

    inputs = ParameterVector("input_angle", _validate_nq(nq))
    return _build(inputs, weights), tuple(inputs)


def _readout_wires(circuit, logical_wires: Sequence[int] | None) -> tuple[int, ...]:
    if logical_wires is None:
        info = (circuit.metadata or {}).get(METADATA_KEY, {})
        logical_wires = info.get("logical_readout_wires")
        if logical_wires is None:
            raise ValueError(
                "Provide logical_wires for a circuit without adapter readout metadata."
            )
    if len(logical_wires) == 0 or any(
        isinstance(wire, (bool, np.bool_)) or not isinstance(wire, (int, np.integer))
        for wire in logical_wires
    ):
        raise ValueError("logical_wires must be a nonempty sequence of integer wire indices.")
    wires = tuple(int(wire) for wire in logical_wires)
    if len(set(wires)) != len(wires) or min(wires) < 0:
        raise ValueError("logical_wires must be unique nonnegative indices.")
    return wires


def logical_to_physical(circuit, logical_wires: Sequence[int] | None = None) -> tuple[int, ...]:
    """Map source logical readout indices through Qiskit's final routed layout."""
    wires = _readout_wires(circuit, logical_wires)
    layout = circuit.layout
    positions = (
        list(range(circuit.num_qubits))
        if layout is None
        else layout.final_index_layout(filter_ancillas=True)
    )
    if max(wires) >= len(positions):
        raise ValueError("A logical readout wire is outside the transpiler input layout.")
    return tuple(int(positions[wire]) for wire in wires)


def measurement_circuits(circuit, logical_wires: Sequence[int] | None = None) -> dict[str, Any]:
    """Joint all-Z and joint all-X readout; each circuit needs its own shots.

    Each group's classical bit j corresponds to logical_wires[j]. That mapping
    survives subsequent transpilation because measurements retain their cbits.
    Existing classical registers/measurements are rejected rather than guessed.
    """
    from qiskit import ClassicalRegister

    if circuit.num_clbits or any(
        item.operation.name in ("measure", "reset") for item in circuit.data
    ):
        raise ValueError("Supply an unmeasured unitary circuit without classical bits.")
    wires = _readout_wires(circuit, logical_wires)
    physical = logical_to_physical(circuit, wires)
    groups = {}
    for basis in ("Z", "X"):
        measured = circuit.copy(name=f"{circuit.name}_{basis}")
        register = ClassicalRegister(len(wires), "readout")
        measured.add_register(register)
        if basis == "X":
            for wire in physical:
                measured.h(wire)
        for index, wire in enumerate(physical):
            measured.measure(wire, register[index])
        measured.metadata = dict(measured.metadata or {})
        measured.metadata["qml_sca_measurement"] = {
            "basis": basis,
            "logical_wires": list(wires),
            "classical_bits": list(range(len(wires))),
        }
        groups[basis] = measured
    return groups


def counts_to_expectations(counts: Mapping[str | int, float], n_readouts: int) -> np.ndarray:
    """Convert joint counts to marginal expectations in cbit-0-first order.

    Accept Qiskit binary strings (including register separators), hex strings,
    or integer outcome keys. This contract is for exactly n_readouts cbits.
    """
    if (
        isinstance(n_readouts, bool)
        or not isinstance(n_readouts, (int, np.integer))
        or n_readouts <= 0
    ):
        raise ValueError("n_readouts must be a positive integer.")
    total, expectations = 0.0, np.zeros(n_readouts)
    for outcome, count in counts.items():
        count = float(count)
        if not np.isfinite(count) or count < 0:
            raise ValueError("Counts/probability weights must be finite and nonnegative.")
        if isinstance(outcome, (int, np.integer)) and not isinstance(outcome, (bool, np.bool_)):
            integer = int(outcome)
        elif isinstance(outcome, str):
            key = outcome.replace(" ", "")
            integer = int(key, 16 if key.lower().startswith("0x") else 2)
        else:
            raise ValueError("Outcome keys must be bitstrings, hex strings, or integers.")
        if integer < 0 or integer >= 2**n_readouts:
            raise ValueError("Outcome contains bits beyond the declared readout register.")
        bits = (integer >> np.arange(n_readouts)) & 1
        expectations += count * (1 - 2 * bits)
        total += count
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Counts must have positive finite total weight.")
    return expectations / total


def expectations_from_counts(z_counts, x_counts, n_features_per_basis: int) -> np.ndarray:
    return np.concatenate(
        [
            counts_to_expectations(z_counts, n_features_per_basis),
            counts_to_expectations(x_counts, n_features_per_basis),
        ]
    )


def qiskit_exact_expectations(
    circuit, logical_wires: Sequence[int] | None = None, *, max_qubits: int = 20
) -> np.ndarray:
    """Exact logical features from an unmeasured circuit, respecting final layout.

    A full backend-width statevector can be intractable; use the original small
    logical circuit for parity checks instead of materializing idle hardware
    ancillas. The explicit size guard prevents an accidental 100+-qubit vector.
    """
    from qiskit.quantum_info import Pauli, Statevector

    if circuit.num_qubits > max_qubits:
        raise ValueError(
            "Circuit exceeds exact-statevector size guard; evaluate the small logical circuit."
        )
    if circuit.num_clbits or any(
        item.operation.name in ("measure", "reset") for item in circuit.data
    ):
        raise ValueError("Exact expectations require an unmeasured unitary circuit.")
    physical = logical_to_physical(circuit, logical_wires)
    state = Statevector.from_instruction(circuit)
    features = []
    for basis in ("Z", "X"):
        for wire in physical:
            label = ["I"] * circuit.num_qubits
            label[circuit.num_qubits - 1 - wire] = basis
            features.append(float(np.real(state.expectation_value(Pauli("".join(label))))))
    return np.array(features)


def _matrix(gate: str, parameter: float | None) -> np.ndarray:
    if gate == "ry":
        cosine, sine = np.cos(parameter / 2), np.sin(parameter / 2)
        return np.array([[cosine, -sine], [sine, cosine]], dtype=complex)
    if gate == "cx":
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    if gate == "crz":
        return np.diag([1, 1, np.exp(-1j * parameter / 2), np.exp(1j * parameter / 2)])
    raise ValueError(gate)


def _apply(state: np.ndarray, gate: np.ndarray, wires: Sequence[int], nq: int) -> np.ndarray:
    order = list(wires) + [wire for wire in range(nq) if wire not in wires]
    arranged = state.reshape([2] * nq).transpose(order).reshape(2 ** len(wires), -1)
    return (gate @ arranged).reshape([2] * nq).transpose(np.argsort(order)).reshape(-1)


def numpy_qcnn_expectations(angles: Any, weights: Mapping[str, Any]) -> np.ndarray:
    """Small exact reference for one vector or a batch; wire 0 is NumPy's MSB.

    This helper is not a trained-model substitute. Integration tests compare it
    and the Qiskit adapter against the original PennyLane make_qcnn forward.
    """
    angles = _numpy(angles)
    if angles.ndim not in (1, 2):
        raise ValueError("Angles must have shape (nq,) or (batch, nq).")
    nq = _validate_nq(angles.shape[-1])
    checked = _weights(weights, nq)
    batch = angles[None, :] if angles.ndim == 1 else angles
    result = []
    z, x = np.diag([1, -1]), np.array([[0, 1], [1, 0]])
    for sample in batch:
        state = np.zeros(2**nq, dtype=complex)
        state[0] = 1
        for gate, parameter, wires in _operations(sample, checked):
            state = _apply(state, _matrix(gate, parameter), wires, nq)
        result.append(
            [
                float(np.vdot(state, _apply(state, observable, [wire], nq)).real)
                for observable in (z, x)
                for wire in range(0, nq, 2)
            ]
        )
    array = np.asarray(result).reshape(len(batch), nq)
    return array[0] if angles.ndim == 1 else array
