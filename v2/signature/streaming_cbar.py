"""Exact streaming counter for the canonical Dyck mean-children observable.

The packed v33 alphabet stores four structural symbols per byte. This module
counts tree edges and branching nodes directly from that representation,
without materializing ``TreeNode`` objects or expanding the structural text.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised only in minimal installs
    HAS_NUMBA = False
    njit = None


ProgressCallback = Callable[[dict[str, Any]], None]
CheckpointCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class SegmentSpec:
    """A half-open character interval parsed as an independent Dyck tree."""

    index: int
    start: int
    length: int

    @property
    def end(self) -> int:
        return self.start + self.length

    def to_dict(self) -> dict[str, int]:
        return {
            "segment_index": self.index,
            "start_char": self.start,
            "end_char_exclusive": self.end,
            "length_chars": self.length,
        }


class DyckMeanChildrenCounter:
    """Count canonical mean-children integers with O(maximum depth) memory."""

    def __init__(
        self,
        initial_stack_capacity: int = 256,
        *,
        allow_python_fallback: bool = True,
    ) -> None:
        if initial_stack_capacity < 1:
            raise ValueError("initial_stack_capacity must be positive")
        if not allow_python_fallback and not HAS_NUMBA:
            raise RuntimeError("Numba backend is required but unavailable")
        self.stack = np.zeros(initial_stack_capacity, dtype=np.uint64)
        self.depth = 1
        self.total_edges = 0
        self.closed_branching_nodes = 0
        self.unmatched_closes = 0
        self.max_stack_depth = 1
        self.max_children = 0
        self.processed_chars = 0
        self.allow_python_fallback = allow_python_fallback
        self.backend = "numba" if HAS_NUMBA else "python"

    def scan(self, packed: bytes, symbol_start: int, symbol_end: int) -> None:
        """Consume ``[symbol_start, symbol_end)`` from packed bytes exactly."""
        total_symbols = len(packed) * 4
        if not 0 <= symbol_start <= symbol_end <= total_symbols:
            raise ValueError(
                f"Invalid packed symbol interval [{symbol_start}, {symbol_end}) "
                f"for {total_symbols} symbols"
            )
        if symbol_start == symbol_end:
            return

        values = np.frombuffer(packed, dtype=np.uint8)
        if self.backend == "numba":
            before = self.to_state()
            try:
                self._scan_numba(values, symbol_start, symbol_end)
                return
            except Exception:
                # A cache/compiler failure must not change semantics. The pure
                # Python path is exact and remains suitable for audit fixtures.
                self.restore_state(before)
                if not self.allow_python_fallback:
                    raise
                self.backend = "python"
        self._scan_python(values, symbol_start, symbol_end)

    def _scan_numba(self, values: np.ndarray, start: int, end: int) -> None:
        while True:
            before = self.to_state()
            result = _scan_codes_numba(
                values,
                start,
                end,
                self.stack,
                self.depth,
                self.total_edges,
                self.closed_branching_nodes,
                self.unmatched_closes,
                self.max_stack_depth,
                self.max_children,
            )
            overflow = bool(result[-1])
            if not overflow:
                (
                    self.depth,
                    self.total_edges,
                    self.closed_branching_nodes,
                    self.unmatched_closes,
                    self.max_stack_depth,
                    self.max_children,
                    consumed,
                    _,
                ) = (int(value) for value in result)
                self.processed_chars += consumed
                return

            required = max(len(self.stack) * 2, self.depth + 1)
            self.restore_state(before, minimum_capacity=required)

    def _scan_python(self, values: np.ndarray, start: int, end: int) -> None:
        for symbol_index in range(start, end):
            byte = int(values[symbol_index >> 2])
            shift = 6 - ((symbol_index & 3) << 1)
            code = (byte >> shift) & 0b11
            if code == 2:  # '('
                parent = self.depth - 1
                self.stack[parent] += 1
                child_count = int(self.stack[parent])
                if child_count > self.max_children:
                    self.max_children = child_count
                self._ensure_capacity(self.depth + 1)
                self.stack[self.depth] = 0
                self.depth += 1
                self.total_edges += 1
                if self.depth > self.max_stack_depth:
                    self.max_stack_depth = self.depth
            elif code == 3:  # ')'
                if self.depth > 1:
                    self.depth -= 1
                    if self.stack[self.depth] > 0:
                        self.closed_branching_nodes += 1
                else:
                    self.unmatched_closes += 1
        self.processed_chars += end - start

    def _ensure_capacity(self, required: int) -> None:
        if required <= len(self.stack):
            return
        capacity = len(self.stack)
        while capacity < required:
            capacity *= 2
        expanded = np.zeros(capacity, dtype=np.uint64)
        expanded[: self.depth] = self.stack[: self.depth]
        self.stack = expanded

    def result(self, spec: SegmentSpec | None = None) -> dict[str, Any]:
        open_branching = int(np.count_nonzero(self.stack[: self.depth]))
        branching = self.closed_branching_nodes + open_branching
        result: dict[str, Any] = {
            "total_nodes": self.total_edges + 1,
            "total_edges": self.total_edges,
            "total_branching_nodes": branching,
            "cbar": self.total_edges / branching if branching else None,
            "processed_chars": self.processed_chars,
            "max_stack_depth": self.max_stack_depth,
            "max_children": self.max_children,
            "unmatched_closes": self.unmatched_closes,
            "unclosed_nodes": self.depth - 1,
            "parser_backend": self.backend,
        }
        if spec is not None:
            result = {**spec.to_dict(), **result}
        return result

    def to_state(self) -> dict[str, Any]:
        return {
            "active_child_counts": [int(value) for value in self.stack[: self.depth]],
            "total_edges": self.total_edges,
            "closed_branching_nodes": self.closed_branching_nodes,
            "unmatched_closes": self.unmatched_closes,
            "max_stack_depth": self.max_stack_depth,
            "max_children": self.max_children,
            "processed_chars": self.processed_chars,
            "backend": self.backend,
        }

    def restore_state(self, state: dict[str, Any], minimum_capacity: int = 0) -> None:
        active = state.get("active_child_counts") or [0]
        capacity = max(256, minimum_capacity, len(active))
        self.stack = np.zeros(capacity, dtype=np.uint64)
        self.stack[: len(active)] = np.asarray(active, dtype=np.uint64)
        self.depth = len(active)
        self.total_edges = int(state.get("total_edges", 0))
        self.closed_branching_nodes = int(state.get("closed_branching_nodes", 0))
        self.unmatched_closes = int(state.get("unmatched_closes", 0))
        self.max_stack_depth = int(state.get("max_stack_depth", self.depth))
        self.max_children = int(state.get("max_children", 0))
        self.processed_chars = int(state.get("processed_chars", 0))
        requested_backend = str(state.get("backend", self.backend))
        self.backend = "numba" if requested_backend == "numba" and HAS_NUMBA else "python"

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        *,
        allow_python_fallback: bool = True,
    ) -> "DyckMeanChildrenCounter":
        counter = cls(
            max(256, len(state.get("active_child_counts") or [])),
            allow_python_fallback=allow_python_fallback,
        )
        counter.restore_state(state)
        return counter


def scan_v33_gzip_segments(
    source_path: Path,
    sequence_length: int,
    segments: Iterable[SegmentSpec],
    *,
    source_sha256: str,
    chunk_bytes: int = 16 * 1024 * 1024,
    checkpoint_interval_bytes: int = 256 * 1024 * 1024,
    require_numba: bool = False,
    resume_state: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> dict[str, Any]:
    """Scan sorted, non-overlapping segments in one forward gzip traversal."""
    source_path = Path(source_path).resolve()
    specs = list(segments)
    _validate_scan(source_path, sequence_length, specs, source_sha256, chunk_bytes)
    if require_numba and not HAS_NUMBA:
        raise RuntimeError("Numba is required for contract-scale streaming cbar scans")

    completed: list[dict[str, Any]] = []
    segment_cursor = 0
    active: DyckMeanChildrenCounter | None = None
    byte_offset = 0
    total_uncompressed_bytes_read = 0
    resumed_from_byte = 0
    backend_history: set[str] = set()

    if resume_state:
        _validate_resume(resume_state, source_path, source_sha256, sequence_length, specs)
        completed = list(resume_state.get("completed_segments", []))
        segment_cursor = int(resume_state.get("segment_cursor", len(completed)))
        byte_offset = int(resume_state.get("packed_byte_offset", 0))
        resumed_from_byte = byte_offset
        active_state = resume_state.get("active_counter")
        if active_state is not None:
            active = DyckMeanChildrenCounter.from_state(
                active_state,
                allow_python_fallback=not require_numba,
            )

    started = time.monotonic()
    last_checkpoint_byte = byte_offset
    compressed_position = 0
    with gzip.open(source_path, "rb") as handle:
        if byte_offset:
            _discard_exact(handle, byte_offset, chunk_bytes, progress_callback)

        while segment_cursor < len(specs):
            raw = handle.read(chunk_bytes)
            if not raw:
                break
            chunk_start_byte = byte_offset
            chunk_end_byte = chunk_start_byte + len(raw)
            chunk_start_char = chunk_start_byte * 4
            chunk_end_char = min(chunk_end_byte * 4, sequence_length)
            total_uncompressed_bytes_read += len(raw)

            while segment_cursor < len(specs):
                spec = specs[segment_cursor]
                if spec.start >= chunk_end_char:
                    break
                if spec.end <= chunk_start_char:
                    raise RuntimeError(
                        f"Segment {spec.index} ended before scan offset {chunk_start_char:,}"
                    )
                if active is None:
                    active = DyckMeanChildrenCounter(
                        allow_python_fallback=not require_numba
                    )

                overlap_start = max(spec.start, chunk_start_char)
                overlap_end = min(spec.end, chunk_end_char)
                if overlap_start < overlap_end:
                    active.scan(
                        raw,
                        overlap_start - chunk_start_char,
                        overlap_end - chunk_start_char,
                    )
                    backend_history.add(active.backend)

                if overlap_end < spec.end:
                    break
                result = active.result(spec)
                if result["processed_chars"] != spec.length:
                    raise RuntimeError(
                        f"Segment {spec.index} processed {result['processed_chars']:,} "
                        f"of {spec.length:,} characters"
                    )
                completed.append(result)
                active = None
                segment_cursor += 1

            byte_offset = chunk_end_byte
            compressed_position = _compressed_position(handle)
            elapsed = time.monotonic() - started
            if progress_callback is not None:
                progress_callback(
                    _progress_payload(
                        specs,
                        completed,
                        segment_cursor,
                        active,
                        byte_offset,
                        compressed_position,
                        elapsed,
                        resumed_from_byte,
                    )
                )

            should_checkpoint = (
                checkpoint_callback is not None
                and (
                    byte_offset - last_checkpoint_byte >= checkpoint_interval_bytes
                    or segment_cursor == len(specs)
                )
            )
            if should_checkpoint:
                checkpoint_callback(
                    build_checkpoint_state(
                        source_path,
                        source_sha256,
                        sequence_length,
                        specs,
                        completed,
                        segment_cursor,
                        active,
                        byte_offset,
                    )
                )
                last_checkpoint_byte = byte_offset

    if segment_cursor != len(specs):
        spec = specs[segment_cursor]
        raise EOFError(
            f"Source ended before segment {spec.index}: completed {len(completed)}/{len(specs)}"
        )

    elapsed = time.monotonic() - started
    aggregate_edges = sum(int(row["total_edges"]) for row in completed)
    aggregate_branching = sum(int(row["total_branching_nodes"]) for row in completed)
    return {
        "segments": completed,
        "aggregate_total_edges": aggregate_edges,
        "aggregate_total_nodes": aggregate_edges + len(completed),
        "aggregate_branching_nodes": aggregate_branching,
        "cbar_aggregate": (
            aggregate_edges / aggregate_branching if aggregate_branching else None
        ),
        "uncompressed_bytes_read_this_process": total_uncompressed_bytes_read,
        "packed_byte_offset": byte_offset,
        "compressed_position_rough": compressed_position,
        "structural_chars_traversed": min(byte_offset * 4, sequence_length),
        "resumed_from_packed_byte": resumed_from_byte,
        "elapsed_seconds_this_process": elapsed,
        "parser_backends": sorted(backend_history),
    }


def build_checkpoint_state(
    source_path: Path,
    source_sha256: str,
    sequence_length: int,
    specs: list[SegmentSpec],
    completed: list[dict[str, Any]],
    segment_cursor: int,
    active: DyckMeanChildrenCounter | None,
    packed_byte_offset: int,
) -> dict[str, Any]:
    return {
        "format": "hsi-v2-streaming-cbar-checkpoint-v1",
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256.upper(),
        "sequence_length": sequence_length,
        "segments": [spec.to_dict() for spec in specs],
        "completed_segments": completed,
        "segment_cursor": segment_cursor,
        "active_counter": None if active is None else active.to_state(),
        "packed_byte_offset": packed_byte_offset,
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_checkpoint_atomic(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_scan(
    source_path: Path,
    sequence_length: int,
    specs: list[SegmentSpec],
    source_sha256: str,
    chunk_bytes: int,
) -> None:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if len(source_sha256) != 64:
        raise ValueError("source_sha256 must be a 64-character SHA-256")
    if not specs:
        raise ValueError("At least one segment is required")
    previous_end = -1
    for spec in specs:
        if spec.index < 1 or spec.start < 0 or spec.length <= 0:
            raise ValueError(f"Invalid segment: {spec}")
        if spec.start < previous_end:
            raise ValueError("Segments must be sorted and non-overlapping")
        if spec.end > sequence_length:
            raise ValueError(f"Segment {spec.index} exceeds sequence_length")
        previous_end = spec.end


def _validate_resume(
    state: dict[str, Any],
    source_path: Path,
    source_sha256: str,
    sequence_length: int,
    specs: list[SegmentSpec],
) -> None:
    expected_segments = [spec.to_dict() for spec in specs]
    checks = {
        "format": "hsi-v2-streaming-cbar-checkpoint-v1",
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256.upper(),
        "sequence_length": sequence_length,
        "segments": expected_segments,
    }
    for key, expected in checks.items():
        if state.get(key) != expected:
            raise ValueError(f"Checkpoint {key} mismatch")
    byte_offset = int(state.get("packed_byte_offset", -1))
    if byte_offset < 0:
        raise ValueError("Checkpoint packed_byte_offset is invalid")


def _discard_exact(
    handle: gzip.GzipFile,
    byte_count: int,
    chunk_bytes: int,
    progress_callback: ProgressCallback | None,
) -> None:
    remaining = byte_count
    discarded = 0
    started = time.monotonic()
    while remaining:
        raw = handle.read(min(chunk_bytes, remaining))
        if not raw:
            raise EOFError(f"Could not recover gzip offset {byte_count:,}")
        size = len(raw)
        discarded += size
        remaining -= size
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "recovery-seek",
                    "packed_bytes_discarded": discarded,
                    "packed_bytes_target": byte_count,
                    "elapsed_seconds": time.monotonic() - started,
                }
            )


def _compressed_position(handle: gzip.GzipFile) -> int:
    fileobj = getattr(handle, "fileobj", None)
    tell = getattr(fileobj, "tell", None)
    if tell is None:
        return 0
    try:
        return int(tell())
    except OSError:
        return 0


def _progress_payload(
    specs: list[SegmentSpec],
    completed: list[dict[str, Any]],
    cursor: int,
    active: DyckMeanChildrenCounter | None,
    byte_offset: int,
    compressed_position: int,
    elapsed: float,
    resumed_from_byte: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "phase": "scan",
        "completed_segments": len(completed),
        "total_segments": len(specs),
        "packed_byte_offset": byte_offset,
        "structural_char_offset": byte_offset * 4,
        "compressed_position_rough": compressed_position,
        "elapsed_seconds": elapsed,
        "resumed_from_packed_byte": resumed_from_byte,
    }
    if cursor < len(specs):
        spec = specs[cursor]
        processed = 0 if active is None else active.processed_chars
        payload.update(
            {
                "active_segment": spec.index,
                "active_segment_processed_chars": processed,
                "active_segment_total_chars": spec.length,
                "active_segment_fraction": processed / spec.length,
            }
        )
    return payload


def _scan_codes_python_reference(
    values: np.ndarray,
    start: int,
    end: int,
    stack: np.ndarray,
    depth: int,
    edges: int,
    closed_branching: int,
    unmatched_closes: int,
    max_depth: int,
    max_children: int,
) -> tuple[int, int, int, int, int, int, int, int]:
    for symbol_index in range(start, end):
        byte = int(values[symbol_index >> 2])
        shift = 6 - ((symbol_index & 3) << 1)
        code = (byte >> shift) & 0b11
        if code == 2:
            if depth >= len(stack):
                return (
                    depth,
                    edges,
                    closed_branching,
                    unmatched_closes,
                    max_depth,
                    max_children,
                    symbol_index - start,
                    1,
                )
            stack[depth - 1] += 1
            children = int(stack[depth - 1])
            if children > max_children:
                max_children = children
            stack[depth] = 0
            depth += 1
            edges += 1
            if depth > max_depth:
                max_depth = depth
        elif code == 3:
            if depth > 1:
                depth -= 1
                if stack[depth] > 0:
                    closed_branching += 1
            else:
                unmatched_closes += 1
    return (
        depth,
        edges,
        closed_branching,
        unmatched_closes,
        max_depth,
        max_children,
        end - start,
        0,
    )


if HAS_NUMBA:
    _scan_codes_numba = njit(cache=False)(_scan_codes_python_reference)
else:  # pragma: no cover - normal test environment has numba
    _scan_codes_numba = _scan_codes_python_reference
