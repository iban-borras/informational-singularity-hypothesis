"""Exact JSON checkpoints for the canonical streaming TE accumulator."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from level2_transfer_entropy import StreamingTEAccumulator


CHECKPOINT_VERSION = 1
ACCUMULATOR_VERSION = 1


def accumulator_to_payload(accumulator: StreamingTEAccumulator) -> dict[str, Any]:
    return {
        "version": ACCUMULATOR_VERSION,
        "scales": [int(value) for value in accumulator.scales],
        "k": int(accumulator.k),
        "n_bins": int(accumulator.n_bins),
        "counters": [
            {
                "pair": [int(i), int(j)],
                "tables": {
                    name: _encode_int_map(tables[name]) for name in sorted(tables)
                },
            }
            for (i, j), tables in sorted(accumulator.counters.items())
        ],
        "total_samples": _encode_pair_scalars(accumulator.total_samples),
        "chunks_processed": int(accumulator.chunks_processed),
        "total_bits_processed": int(accumulator.total_bits_processed),
        "mi_counters": [
            {"index": int(index), "counts": _encode_int_map(counts)}
            for index, counts in sorted(accumulator.mi_counters.items())
        ],
        "mi_joint_counters": [
            {"pair": [int(i), int(j)], "counts": _encode_int_map(counts)}
            for (i, j), counts in sorted(accumulator.mi_joint_counters.items())
        ],
        "mi_samples": [
            [int(index), int(value)]
            for index, value in sorted(accumulator.mi_samples.items())
        ],
        "mi_joint_samples": _encode_pair_scalars(accumulator.mi_joint_samples),
        "corr_accumulators": [
            {
                "pair": [int(i), int(j)],
                "values": {
                    "sum_x": float(values["sum_x"]),
                    "sum_y": float(values["sum_y"]),
                    "sum_xy": float(values["sum_xy"]),
                    "sum_x2": float(values["sum_x2"]),
                    "sum_y2": float(values["sum_y2"]),
                    "n": int(values["n"]),
                },
            }
            for (i, j), values in sorted(accumulator.corr_accumulators.items())
        ],
    }


def accumulator_from_payload(
    payload: dict[str, Any],
    accumulator_factory: Callable[..., StreamingTEAccumulator] | None = None,
) -> StreamingTEAccumulator:
    if int(payload.get("version", -1)) != ACCUMULATOR_VERSION:
        raise ValueError("unsupported TE accumulator checkpoint version")

    if accumulator_factory is None:
        from level2_transfer_entropy import StreamingTEAccumulator

        accumulator_factory = StreamingTEAccumulator
    accumulator = accumulator_factory(
        [int(value) for value in payload["scales"]],
        k=int(payload["k"]),
        n_bins=int(payload["n_bins"]),
    )

    counters: dict[tuple[int, int], dict[str, dict[int, int]]] = {}
    for row in payload["counters"]:
        pair = _decode_pair(row["pair"])
        counters[pair] = {
            name: _decode_int_map(values) for name, values in row["tables"].items()
        }
    _require_keys("counters", accumulator.counters, counters)
    accumulator.counters = counters

    total_samples = _decode_pair_scalars(payload["total_samples"], int)
    _require_keys("total_samples", accumulator.total_samples, total_samples)
    accumulator.total_samples = total_samples
    accumulator.chunks_processed = int(payload["chunks_processed"])
    accumulator.total_bits_processed = int(payload["total_bits_processed"])

    mi_counters = {
        int(row["index"]): _decode_int_map(row["counts"])
        for row in payload["mi_counters"]
    }
    _require_keys("mi_counters", accumulator.mi_counters, mi_counters)
    accumulator.mi_counters = mi_counters

    mi_joint_counters = {
        _decode_pair(row["pair"]): _decode_int_map(row["counts"])
        for row in payload["mi_joint_counters"]
    }
    _require_keys("mi_joint_counters", accumulator.mi_joint_counters, mi_joint_counters)
    accumulator.mi_joint_counters = mi_joint_counters

    mi_samples = {int(index): int(value) for index, value in payload["mi_samples"]}
    _require_keys("mi_samples", accumulator.mi_samples, mi_samples)
    accumulator.mi_samples = mi_samples

    mi_joint_samples = _decode_pair_scalars(payload["mi_joint_samples"], int)
    _require_keys("mi_joint_samples", accumulator.mi_joint_samples, mi_joint_samples)
    accumulator.mi_joint_samples = mi_joint_samples

    corr_accumulators: dict[tuple[int, int], dict[str, float | int]] = {}
    for row in payload["corr_accumulators"]:
        values = row["values"]
        corr_accumulators[_decode_pair(row["pair"])] = {
            "sum_x": float(values["sum_x"]),
            "sum_y": float(values["sum_y"]),
            "sum_xy": float(values["sum_xy"]),
            "sum_x2": float(values["sum_x2"]),
            "sum_y2": float(values["sum_y2"]),
            "n": int(values["n"]),
        }
    _require_keys("corr_accumulators", accumulator.corr_accumulators, corr_accumulators)
    accumulator.corr_accumulators = corr_accumulators
    return accumulator


def build_checkpoint(
    *,
    identity: dict[str, Any],
    observed: StreamingTEAccumulator,
    shuffled: StreamingTEAccumulator,
    rng: np.random.Generator,
    progress: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "status": status,
        "identity": identity,
        "progress": progress,
        "rng_state": _json_safe(rng.bit_generator.state),
        "observed_accumulator": accumulator_to_payload(observed),
        "shuffled_accumulator": accumulator_to_payload(shuffled),
    }


def restore_checkpoint(
    payload: dict[str, Any],
    expected_identity: dict[str, Any],
    accumulator_factory: Callable[..., StreamingTEAccumulator] | None = None,
) -> tuple[
    StreamingTEAccumulator,
    StreamingTEAccumulator,
    np.random.Generator,
    dict[str, Any],
]:
    if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError("unsupported resumable TE checkpoint version")
    if payload.get("identity") != expected_identity:
        raise ValueError("TE checkpoint identity does not match this execution")

    observed = accumulator_from_payload(
        payload["observed_accumulator"], accumulator_factory
    )
    shuffled = accumulator_from_payload(
        payload["shuffled_accumulator"], accumulator_factory
    )
    rng = np.random.default_rng()
    rng.bit_generator.state = payload["rng_state"]
    return observed, shuffled, rng, dict(payload["progress"])


def load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _encode_int_map(mapping: dict[Any, Any]) -> list[list[int]]:
    # Insertion order fixes the floating-point summation order at final readout.
    return [[int(key), int(value)] for key, value in mapping.items()]


def _decode_int_map(rows: list[list[Any]]) -> dict[int, int]:
    return {int(key): int(value) for key, value in rows}


def _encode_pair_scalars(mapping: dict[tuple[int, int], Any]) -> list[list[int]]:
    return [[int(i), int(j), int(value)] for (i, j), value in sorted(mapping.items())]


def _decode_pair_scalars(
    rows: list[list[Any]], converter: type[int] | type[float]
) -> dict[tuple[int, int], Any]:
    return {(int(i), int(j)): converter(value) for i, j, value in rows}


def _decode_pair(raw: list[Any]) -> tuple[int, int]:
    if len(raw) != 2:
        raise ValueError("invalid TE pair in checkpoint")
    return int(raw[0]), int(raw[1])


def _require_keys(
    name: str, expected: dict[Any, Any], observed: dict[Any, Any]
) -> None:
    if set(expected) != set(observed):
        raise ValueError(f"{name} keys do not match the accumulator protocol")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
