from __future__ import annotations

import gzip
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


ALGORITHM_ID = "legacy-v33-observable-box-counting-v1"
CHECKPOINT_VERSION = 1

_BYTE_TO_CODES = np.empty((256, 4), dtype=np.uint8)
_BYTE_VALUES = np.arange(256, dtype=np.uint8)
_BYTE_TO_CODES[:, 0] = (_BYTE_VALUES >> 6) & 0b11
_BYTE_TO_CODES[:, 1] = (_BYTE_VALUES >> 4) & 0b11
_BYTE_TO_CODES[:, 2] = (_BYTE_VALUES >> 2) & 0b11
_BYTE_TO_CODES[:, 3] = _BYTE_VALUES & 0b11


@dataclass
class StreamingBoxCounter:
    """Exact bounded-memory equivalent of the legacy v33 box counter."""

    max_box_size: int = 64

    def __post_init__(self) -> None:
        if self.max_box_size <= 2:
            raise ValueError("max_box_size must be greater than 2")
        self.sizes = np.arange(2, self.max_box_size, dtype=np.int64)
        self.completed_counts = np.zeros(self.sizes.size, dtype=np.uint64)
        self.partial_lengths = np.zeros(self.sizes.size, dtype=np.uint16)
        self.partial_has_one = np.zeros(self.sizes.size, dtype=np.bool_)
        self.observable_bits = 0
        self.observable_ones = 0
        self.structural_chars = 0
        self.packed_bytes = 0

    def update_observable(self, bits: np.ndarray) -> None:
        values = np.asarray(bits, dtype=np.uint8)
        if values.ndim != 1:
            raise ValueError("observable bits must be a one-dimensional array")
        if values.size == 0:
            return
        if np.any(values > 1):
            raise ValueError("observable array contains values outside {0, 1}")

        prefix_dtype = (
            np.uint32 if values.size <= np.iinfo(np.uint32).max else np.uint64
        )
        prefix = np.empty(values.size + 1, dtype=prefix_dtype)
        prefix[0] = 0
        np.cumsum(values, dtype=prefix_dtype, out=prefix[1:])

        n_values = int(values.size)
        for index, raw_size in enumerate(self.sizes):
            size = int(raw_size)
            cursor = 0
            partial_length = int(self.partial_lengths[index])
            has_one = bool(self.partial_has_one[index])

            if partial_length:
                take = min(n_values, size - partial_length)
                if int(prefix[take]) > 0:
                    has_one = True
                partial_length += take
                cursor = take
                if partial_length == size:
                    if has_one:
                        self.completed_counts[index] += 1
                    partial_length = 0
                    has_one = False
                elif cursor == n_values:
                    self.partial_lengths[index] = partial_length
                    self.partial_has_one[index] = has_one
                    continue

            remaining = n_values - cursor
            full_length = (remaining // size) * size
            if full_length:
                endpoints = np.arange(
                    cursor + size,
                    cursor + full_length + 1,
                    size,
                    dtype=np.int64,
                )
                occupied = prefix[endpoints] > prefix[endpoints - size]
                self.completed_counts[index] += np.count_nonzero(occupied)
                cursor += full_length

            if cursor < n_values:
                partial_length = n_values - cursor
                has_one = bool(prefix[n_values] > prefix[cursor])

            self.partial_lengths[index] = partial_length
            self.partial_has_one[index] = has_one

        self.observable_bits += n_values
        self.observable_ones += int(prefix[-1])

    def result(self) -> dict[str, Any]:
        if self.observable_bits < 4:
            return {
                "algorithm": ALGORITHM_ID,
                "max_box_size": self.max_box_size,
                "observable_bits": self.observable_bits,
                "observable_ones": self.observable_ones,
                "fractal_dimension": 1.0,
                "box_counts": [],
            }

        max_size_exclusive = min(
            self.max_box_size,
            max(2, self.observable_bits // 4),
        )
        active = self.sizes < max_size_exclusive
        sizes = self.sizes[active]
        counts = self.completed_counts[active].copy()
        trailing = self.partial_has_one[active] & (self.partial_lengths[active] > 0)
        counts += trailing.astype(np.uint64)

        if sizes.size == 0:
            fractal_dimension = 1.0
        else:
            log_sizes = np.log(sizes.astype(np.float64))
            log_counts = np.log(np.maximum(1, counts).astype(np.float64))
            if sizes.size < 2:
                fractal_dimension = 1.0
            else:
                slope = float(np.polyfit(log_sizes, log_counts, 1)[0])
                fractal_dimension = float(max(0.0, min(2.0, -slope)))

        rows = []
        for size, count in zip(sizes.tolist(), counts.tolist(), strict=True):
            total_boxes = math.ceil(self.observable_bits / size)
            rows.append(
                {
                    "box_size": int(size),
                    "occupied_boxes": int(count),
                    "total_boxes": int(total_boxes),
                    "occupancy_fraction": float(count / total_boxes),
                }
            )

        return {
            "algorithm": ALGORITHM_ID,
            "max_box_size": self.max_box_size,
            "observable_bits": self.observable_bits,
            "observable_ones": self.observable_ones,
            "observable_zeros": self.observable_bits - self.observable_ones,
            "fractal_dimension": fractal_dimension,
            "box_counts": rows,
        }

    def to_checkpoint(
        self,
        *,
        source_sha256: str,
        sequence_length: int,
        expected_observable_bits: int,
        status: str = "running",
    ) -> dict[str, Any]:
        return {
            "version": CHECKPOINT_VERSION,
            "algorithm": ALGORITHM_ID,
            "status": status,
            "source_sha256": source_sha256,
            "sequence_length": int(sequence_length),
            "expected_observable_bits": int(expected_observable_bits),
            "max_box_size": self.max_box_size,
            "packed_bytes_read": self.packed_bytes,
            "structural_chars_seen": self.structural_chars,
            "observable_bits_seen": self.observable_bits,
            "observable_ones_seen": self.observable_ones,
            "completed_counts": [int(value) for value in self.completed_counts],
            "partial_lengths": [int(value) for value in self.partial_lengths],
            "partial_has_one": [bool(value) for value in self.partial_has_one],
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: dict[str, Any],
        *,
        source_sha256: str,
        sequence_length: int,
        expected_observable_bits: int,
        max_box_size: int,
    ) -> "StreamingBoxCounter":
        expected = {
            "version": CHECKPOINT_VERSION,
            "algorithm": ALGORITHM_ID,
            "source_sha256": source_sha256,
            "sequence_length": int(sequence_length),
            "expected_observable_bits": int(expected_observable_bits),
            "max_box_size": int(max_box_size),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"checkpoint mismatch for {key}")

        counter = cls(max_box_size=max_box_size)
        for key in ("completed_counts", "partial_lengths", "partial_has_one"):
            if len(payload.get(key, [])) != counter.sizes.size:
                raise ValueError(f"checkpoint has invalid {key} length")

        counter.completed_counts[:] = np.asarray(
            payload["completed_counts"], dtype=np.uint64
        )
        counter.partial_lengths[:] = np.asarray(
            payload["partial_lengths"], dtype=np.uint16
        )
        counter.partial_has_one[:] = np.asarray(
            payload["partial_has_one"], dtype=np.bool_
        )
        counter.packed_bytes = int(payload["packed_bytes_read"])
        counter.structural_chars = int(payload["structural_chars_seen"])
        counter.observable_bits = int(payload["observable_bits_seen"])
        counter.observable_ones = int(payload["observable_ones_seen"])

        if counter.packed_bytes < 0 or counter.structural_chars < 0:
            raise ValueError("checkpoint positions cannot be negative")
        expected_structural_position = min(counter.packed_bytes * 4, sequence_length)
        if counter.structural_chars != expected_structural_position:
            raise ValueError("checkpoint packed and structural positions disagree")
        if counter.structural_chars > sequence_length:
            raise ValueError("checkpoint structural position exceeds source length")
        if counter.observable_bits > expected_observable_bits:
            raise ValueError("checkpoint observable position exceeds expected length")
        if (
            counter.observable_ones < 0
            or counter.observable_ones > counter.observable_bits
        ):
            raise ValueError("checkpoint observable-one count is invalid")
        if np.any(counter.partial_lengths >= counter.sizes):
            raise ValueError("checkpoint partial box length exceeds its box size")
        return counter


def scan_v33_observable_box_counts(
    source_path: Path,
    *,
    sequence_length: int,
    expected_observable_bits: int,
    source_sha256: str,
    max_box_size: int = 64,
    raw_chunk_bytes: int = 4 * 1024 * 1024,
    checkpoint_interval_bytes: int = 256 * 1024 * 1024,
    resume_state: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if expected_observable_bits <= 0:
        raise ValueError("expected_observable_bits must be positive")
    if raw_chunk_bytes <= 0 or checkpoint_interval_bytes <= 0:
        raise ValueError("chunk and checkpoint sizes must be positive")

    source_path = Path(source_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    if resume_state is None:
        counter = StreamingBoxCounter(max_box_size=max_box_size)
    else:
        counter = StreamingBoxCounter.from_checkpoint(
            resume_state,
            source_sha256=source_sha256,
            sequence_length=sequence_length,
            expected_observable_bits=expected_observable_bits,
            max_box_size=max_box_size,
        )

    expected_packed_bytes = (sequence_length + 3) // 4
    if counter.packed_bytes > expected_packed_bytes:
        raise ValueError("checkpoint packed offset exceeds encoded source length")

    next_checkpoint = (
        (counter.packed_bytes // checkpoint_interval_bytes) + 1
    ) * checkpoint_interval_bytes

    with gzip.open(source_path, "rb") as handle:
        remaining_skip = counter.packed_bytes
        skipped = 0
        while remaining_skip:
            raw = handle.read(min(raw_chunk_bytes, remaining_skip))
            if not raw:
                raise EOFError("source ended while recovering checkpoint position")
            skipped += len(raw)
            remaining_skip -= len(raw)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "recovery",
                        "packed_bytes_read": skipped,
                        "packed_bytes_target": counter.packed_bytes,
                        "structural_chars_seen": min(
                            skipped * 4, counter.structural_chars
                        ),
                        "observable_bits_seen": None,
                    }
                )

        while counter.structural_chars < sequence_length:
            remaining_packed = expected_packed_bytes - counter.packed_bytes
            raw = handle.read(min(raw_chunk_bytes, remaining_packed))
            if not raw:
                raise EOFError(
                    f"source ended after {counter.structural_chars:,}/{sequence_length:,} structural chars"
                )

            codes = _decode_structural_codes(raw)
            valid_chars = min(codes.size, sequence_length - counter.structural_chars)
            codes = codes[:valid_chars]
            observable = np.ascontiguousarray(codes[codes <= 1], dtype=np.uint8)
            counter.update_observable(observable)
            counter.structural_chars += int(valid_chars)
            counter.packed_bytes += len(raw)

            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "scan",
                        "packed_bytes_read": counter.packed_bytes,
                        "packed_bytes_target": expected_packed_bytes,
                        "structural_chars_seen": counter.structural_chars,
                        "observable_bits_seen": counter.observable_bits,
                    }
                )

            if (
                checkpoint_callback is not None
                and counter.packed_bytes >= next_checkpoint
            ):
                checkpoint_callback(
                    counter.to_checkpoint(
                        source_sha256=source_sha256,
                        sequence_length=sequence_length,
                        expected_observable_bits=expected_observable_bits,
                    )
                )
                next_checkpoint = (
                    (counter.packed_bytes // checkpoint_interval_bytes) + 1
                ) * checkpoint_interval_bytes

    if counter.structural_chars != sequence_length:
        raise RuntimeError(
            "structural scan did not finish at the declared source length"
        )
    if counter.packed_bytes != expected_packed_bytes:
        raise RuntimeError("packed scan did not finish at the expected byte boundary")
    if counter.observable_bits != expected_observable_bits:
        raise RuntimeError(
            "observable bit count mismatch: "
            f"expected {expected_observable_bits:,}, observed {counter.observable_bits:,}"
        )

    checkpoint = counter.to_checkpoint(
        source_sha256=source_sha256,
        sequence_length=sequence_length,
        expected_observable_bits=expected_observable_bits,
        status="completed",
    )
    if checkpoint_callback is not None:
        checkpoint_callback(checkpoint)

    result = counter.result()
    result.update(
        {
            "status": "completed",
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "sequence_length": sequence_length,
            "packed_bytes_read": counter.packed_bytes,
            "expected_packed_bytes": expected_packed_bytes,
        }
    )
    return result


def legacy_reference(bits: np.ndarray, max_box_size: int = 64) -> dict[str, Any]:
    """Small-input reference matching the legacy in-memory v33 implementation."""
    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > 1):
        raise ValueError("bits must be a one-dimensional binary array")

    rows = []
    if values.size >= 4:
        max_size_exclusive = min(max_box_size, max(2, values.size // 4))
        for size in range(2, max_size_exclusive):
            count = 0
            for start in range(0, values.size, size):
                if np.any(values[start : start + size] == 1):
                    count += 1
            rows.append((size, count))

    if values.size < 4 or len(rows) < 2:
        dimension = 1.0
    else:
        log_sizes = np.log(np.asarray([row[0] for row in rows], dtype=np.float64))
        log_counts = np.log(
            np.maximum(1, np.asarray([row[1] for row in rows], dtype=np.float64))
        )
        dimension = float(max(0.0, min(2.0, -np.polyfit(log_sizes, log_counts, 1)[0])))

    return {
        "fractal_dimension": dimension,
        "box_counts": [
            {"box_size": int(size), "occupied_boxes": int(count)}
            for size, count in rows
        ],
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _decode_structural_codes(raw: bytes) -> np.ndarray:
    if not raw:
        return np.empty(0, dtype=np.uint8)
    byte_values = np.frombuffer(raw, dtype=np.uint8)
    return np.ascontiguousarray(_BYTE_TO_CODES[byte_values].reshape(-1))
