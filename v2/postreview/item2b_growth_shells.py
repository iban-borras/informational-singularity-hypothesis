from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover - execution gate rejects this environment.
    psutil = None  # type: ignore[assignment]

from v2.common.io import companion_metadata_path, load_observable_cache


PREREG_SHA256 = "C087C99D2808064ECE68976F2991C120C7C7DAF682F3A6BBB828FA9988C55653"
SOURCE_SPECS = {
    19: ("BBC2C50BE8B9CBFEA0D755C4CB68572623D7B9821EF2321DB1F702B058F12718", 3_975_956_062),
    20: ("B1FE25ACB1729C4862BE843559696206896402E3698E5F2F671F3E37795CA3D8", 11_198_388_932),
    23: ("9BCBCDBB7C3E5D9CAEBB1D4C47403F36FB08DCC5DFFCB0716163BAD9705BD1B7", 246_162_834_941),
    24: ("B766982BDB9AF90975F8E4D0B7D92D413964E402DBB32044C2FC8EBF0AE57702", 686_779_035_011),
}
WINDOW_BITS = 3_000_000
WINDOW_SPECS = (
    (20, "1/10", 4_697_899_349),
    (20, "1/2", 7_585_672_497),
    (20, "9/10", 10_473_445_645),
    (24, "1/10", 290_224_154_948),
    (24, "1/2", 466_469_434_976),
    (24, "9/10", 642_714_715_004),
)
SEEDS = (17, 101, 211, 307, 401, 503)
SCALES = (8, 12, 16, 20, 24, 28, 32, 40, 48)
POLICIES = ("prefix", "suffix")


@dataclass(frozen=True)
class WindowSpec:
    depth: int
    q: str
    start: int
    length: int = WINDOW_BITS

    @property
    def end(self) -> int:
        return self.start + self.length

    @property
    def slug(self) -> str:
        return f"d{self.depth}_q{self.q.replace('/', 'of')}"


FROZEN_WINDOWS = tuple(WindowSpec(*values) for values in WINDOW_SPECS)


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_progress = 0.0

    def emit(self, event: str, *, force: bool = True, **fields: Any) -> None:
        now = time.monotonic()
        if not force and now - self._last_progress < 30.0:
            return
        if event == "progress":
            self._last_progress = now
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            **fields,
        }
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class V33ObservableReader:
    """Bounded vectorized decoder for the packed four-symbol v33 stream."""

    def __init__(
        self,
        path: Path,
        *,
        sequence_length: int,
        expected_observable_bits: int,
        raw_chunk_bytes: int,
        progress: Callable[["V33ObservableReader"], None] | None = None,
    ) -> None:
        if not 1 <= raw_chunk_bytes <= 16 * 1024 * 1024:
            raise ValueError("raw_chunk_bytes must be in [1, 16 MiB].")
        if sequence_length <= 0 or expected_observable_bits <= 0:
            raise ValueError("sequence and observable lengths must be positive.")
        self.path = path
        self.sequence_length = int(sequence_length)
        self.expected_observable_bits = int(expected_observable_bits)
        self.raw_chunk_bytes = int(raw_chunk_bytes)
        self.progress = progress
        self.packed_bytes_expected = (self.sequence_length + 3) // 4
        self.packed_bytes_read = 0
        self.structural_symbols_decoded = 0
        self.observable_bits_decoded = 0
        self.padding_symbols_excluded = (4 - (self.sequence_length % 4)) % 4
        self.peak_rss_bytes = current_rss_bytes()
        self._handle = gzip.open(path, "rb")
        self._closed = False

    def __enter__(self) -> "V33ObservableReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True

    def next_observable(self) -> np.ndarray | None:
        if self.structural_symbols_decoded >= self.sequence_length:
            self._finish_validation()
            return None

        remaining_packed = self.packed_bytes_expected - self.packed_bytes_read
        raw = self._handle.read(min(self.raw_chunk_bytes, remaining_packed))
        if not raw:
            raise RuntimeError(
                f"Unexpected EOF in {self.path}: decoded {self.structural_symbols_decoded}/"
                f"{self.sequence_length} structural symbols."
            )

        self.packed_bytes_read += len(raw)
        source = np.frombuffer(raw, dtype=np.uint8)
        codes = np.empty(source.size * 4, dtype=np.uint8)
        codes[0::4] = source >> 6
        codes[1::4] = (source >> 4) & 0x03
        codes[2::4] = (source >> 2) & 0x03
        codes[3::4] = source & 0x03
        remaining_symbols = self.sequence_length - self.structural_symbols_decoded
        if codes.size > remaining_symbols:
            codes = codes[:remaining_symbols]
        self.structural_symbols_decoded += int(codes.size)
        observable = np.ascontiguousarray(codes[codes <= 1], dtype=np.uint8)
        self.observable_bits_decoded += int(observable.size)
        if self.observable_bits_decoded > self.expected_observable_bits:
            raise RuntimeError(
                f"Observable bit count exceeds frozen length for {self.path}: "
                f"{self.observable_bits_decoded}>{self.expected_observable_bits}."
            )
        self.peak_rss_bytes = max(self.peak_rss_bytes, current_rss_bytes())
        if self.progress is not None:
            self.progress(self)
        return observable

    def _finish_validation(self) -> None:
        if self.observable_bits_decoded != self.expected_observable_bits:
            raise RuntimeError(
                f"Observable bit count mismatch for {self.path}: expected "
                f"{self.expected_observable_bits}, got {self.observable_bits_decoded}."
            )
        if self.packed_bytes_read != self.packed_bytes_expected:
            raise RuntimeError(
                f"Packed-byte count mismatch for {self.path}: expected "
                f"{self.packed_bytes_expected}, got {self.packed_bytes_read}."
            )
        if self._handle.read(1):
            raise RuntimeError(f"Unexpected trailing packed bytes after sequence_length in {self.path}.")


class ObservableCursor:
    def __init__(self, reader: V33ObservableReader) -> None:
        self.reader = reader
        self._chunk = np.empty(0, dtype=np.uint8)
        self._index = 0
        self.position = 0

    def take(self, maximum: int) -> np.ndarray:
        if maximum <= 0:
            return np.empty(0, dtype=np.uint8)
        while self._index >= self._chunk.size:
            next_chunk = self.reader.next_observable()
            if next_chunk is None:
                return np.empty(0, dtype=np.uint8)
            self._chunk = next_chunk
            self._index = 0
            if self._chunk.size == 0:
                continue
        width = min(maximum, int(self._chunk.size - self._index))
        view = self._chunk[self._index : self._index + width]
        self._index += width
        self.position += width
        return view

    def discard(self, count: int) -> None:
        remaining = count
        while remaining:
            chunk = self.take(remaining)
            if chunk.size == 0:
                raise RuntimeError(f"Observable stream ended with {remaining} bits still to discard.")
            remaining -= int(chunk.size)

    def read_exact(self, count: int) -> np.ndarray:
        output = np.empty(count, dtype=np.uint8)
        written = 0
        while written < count:
            chunk = self.take(count - written)
            if chunk.size == 0:
                raise RuntimeError(f"Observable stream ended after {written}/{count} requested bits.")
            output[written : written + chunk.size] = chunk
            written += int(chunk.size)
        return output

    def drain(self) -> None:
        while self.take(64 * 1024 * 1024).size:
            pass


def scan_nested_pair(
    parent_path: Path,
    child_path: Path,
    *,
    parent_sequence_length: int,
    child_sequence_length: int,
    parent_observable_bits: int,
    child_observable_bits: int,
    windows: tuple[WindowSpec, ...],
    raw_chunk_bytes: int,
    logger: RunLogger,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.monotonic()
    windows_frozen = 0

    def progress(reader: V33ObservableReader) -> None:
        elapsed = max(time.monotonic() - started, 1e-9)
        percent = 100.0 * reader.observable_bits_decoded / reader.expected_observable_bits
        rate = reader.observable_bits_decoded / elapsed
        eta = (reader.expected_observable_bits - reader.observable_bits_decoded) / max(rate, 1e-9)
        logger.emit(
            "progress",
            force=False,
            phase="stage0-prefix-and-window-scan",
            snapshot=reader.path.name,
            structural_bytes_read=reader.packed_bytes_read,
            observable_bits_decoded=reader.observable_bits_decoded,
            target_observable_bits=reader.expected_observable_bits,
            percent=round(percent, 6),
            rate_bits_per_second=round(rate, 3),
            elapsed_seconds=round(elapsed, 3),
            eta_seconds=round(eta, 3),
            peak_rss_bytes=reader.peak_rss_bytes,
            windows_frozen=f"{windows_frozen}/{len(windows)}",
        )

    with V33ObservableReader(
        parent_path,
        sequence_length=parent_sequence_length,
        expected_observable_bits=parent_observable_bits,
        raw_chunk_bytes=raw_chunk_bytes,
        progress=progress,
    ) as parent_reader, V33ObservableReader(
        child_path,
        sequence_length=child_sequence_length,
        expected_observable_bits=child_observable_bits,
        raw_chunk_bytes=raw_chunk_bytes,
        progress=progress,
    ) as child_reader:
        parent = ObservableCursor(parent_reader)
        child = ObservableCursor(child_reader)
        compared = 0
        while compared < parent_observable_bits:
            parent_chunk = parent.take(parent_observable_bits - compared)
            child_chunk = child.take(parent_observable_bits - compared)
            width = min(parent_chunk.size, child_chunk.size)
            if width == 0:
                raise RuntimeError("Parent or child ended during exact observable-prefix comparison.")
            equality = parent_chunk[:width] == child_chunk[:width]
            if not bool(np.all(equality)):
                absolute = compared + int(np.argmax(~equality))
                raise RuntimeError(f"Observable-prefix divergence at bit {absolute:,}.")
            if parent_chunk.size != width:
                parent._index -= int(parent_chunk.size - width)
                parent.position -= int(parent_chunk.size - width)
            if child_chunk.size != width:
                child._index -= int(child_chunk.size - width)
                child.position -= int(child_chunk.size - width)
            compared += width

        parent.drain()
        if parent.position != parent_observable_bits:
            raise RuntimeError("Parent observable stream exceeds its frozen length.")

        extracted: dict[str, np.ndarray] = {}
        for spec in sorted(windows, key=lambda item: item.start):
            if child.position > spec.start:
                raise RuntimeError(f"Window {spec.slug} starts behind the streaming cursor.")
            child.discard(spec.start - child.position)
            extracted[spec.slug] = child.read_exact(spec.length)
            windows_frozen += 1
            logger.emit(
                "window-captured",
                depth=spec.depth,
                q=spec.q,
                start=spec.start,
                end=spec.end,
                windows_frozen=f"{windows_frozen}/{len(windows)}",
            )

        child.drain()
        if child.position != child_observable_bits:
            raise RuntimeError("Child observable stream does not match its frozen length.")

        telemetry = {
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "prefix_bits_compared": parent_observable_bits,
            "parent_packed_bytes_read": parent_reader.packed_bytes_read,
            "child_packed_bytes_read": child_reader.packed_bytes_read,
            "parent_padding_symbols_excluded": parent_reader.padding_symbols_excluded,
            "child_padding_symbols_excluded": child_reader.padding_symbols_excluded,
            "peak_rss_bytes": max(parent_reader.peak_rss_bytes, child_reader.peak_rss_bytes),
        }
        return extracted, telemetry


def freeze_cache(
    cache_path: Path,
    bits: np.ndarray,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, np.ascontiguousarray(bits, dtype=np.uint8), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, cache_path)
    cache_sha256 = sha256_file(cache_path)
    sidecar = {
        **metadata,
        "format": "npy",
        "dtype": "uint8",
        "observable_length": int(bits.size),
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": cache_sha256,
    }
    sidecar_path = cache_path.with_suffix(".json")
    write_json_atomic(sidecar_path, sidecar)
    loaded, verified, _ = load_observable_cache(cache_path, expected_bits=int(bits.size))
    if not np.array_equal(loaded, bits) or verified["cache_sha256"] != cache_sha256:
        raise RuntimeError(f"Post-write cache verification failed: {cache_path}")
    return sidecar


def freeze_or_reuse_cache(
    cache_path: Path,
    bits: np.ndarray,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    sidecar_path = cache_path.with_suffix(".json")
    if cache_path.exists() or sidecar_path.exists():
        try:
            loaded, sidecar, _ = load_observable_cache(
                cache_path,
                expected_bits=int(bits.size),
            )
            if np.array_equal(loaded, bits) and all(
                sidecar.get(key) == value for key, value in metadata.items()
            ):
                return {**sidecar, "reused": True}
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            pass

        quarantine = cache_path.parent / "quarantine" / datetime.now().strftime("%Y%m%dT%H%M%S")
        quarantine.mkdir(parents=True, exist_ok=True)
        for path in (cache_path, sidecar_path):
            if path.exists():
                shutil.move(str(path), str(quarantine / path.name))

    return {**freeze_cache(cache_path, bits, metadata=metadata), "reused": False}


def source_sequence_length(path: Path) -> int:
    metadata_path = companion_metadata_path(path)
    if not metadata_path.exists():
        raise RuntimeError(f"Missing structural metadata: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    value = payload.get("sequence_length")
    if not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"Invalid sequence_length in {metadata_path}: {value!r}")
    return value


def verify_compiled_kernels() -> dict[str, Any]:
    import numba
    from v2.common import null_models
    from v2.phase1.tower import rolling_codes_uint64_max64

    if null_models.njit is None:
        raise RuntimeError("Numba is required by the frozen contract.")
    bits = np.asarray([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    compiled_codes = rolling_codes_uint64_max64(bits, 4)
    reference_codes = np.asarray(
        [int("".join(str(int(value)) for value in bits[i : i + 4]), 2) for i in range(5)],
        dtype=np.uint64,
    )
    if not np.array_equal(compiled_codes, reference_codes):
        raise RuntimeError("Compiled rolling-code fixture differs from Python reference.")

    random_values = np.asarray([0.1, 0.8, 0.2, 0.9], dtype=np.float64)
    compiled_chain = null_models._generate_markov1_chunk(random_values, 0, 0.25, 0.75)
    state = 0
    reference_chain = np.empty(random_values.size, dtype=np.uint8)
    for index, value in enumerate(random_values):
        state = 1 if value < (0.75 if state else 0.25) else 0
        reference_chain[index] = state
    if not np.array_equal(compiled_chain, reference_chain):
        raise RuntimeError("Compiled Markov-1 fixture differs from Python reference.")
    if not getattr(rolling_codes_uint64_max64, "signatures", ()):
        raise RuntimeError("rolling_codes_uint64_max64 did not compile with Numba.")
    if not getattr(null_models._generate_markov1_chunk, "signatures", ()):
        raise RuntimeError("Markov-1 chain kernel did not compile with Numba.")
    rolling_hits = sum(getattr(rolling_codes_uint64_max64, "_cache_hits", {}).values())
    rolling_misses = sum(getattr(rolling_codes_uint64_max64, "_cache_misses", {}).values())
    markov_hits = sum(getattr(null_models._generate_markov1_chunk, "_cache_hits", {}).values())
    markov_misses = sum(getattr(null_models._generate_markov1_chunk, "_cache_misses", {}).values())
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "numba": numba.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "rolling_codes_signatures": [str(value) for value in rolling_codes_uint64_max64.signatures],
        "markov1_signatures": [str(value) for value in null_models._generate_markov1_chunk.signatures],
        "rolling_codes_cache_hits": rolling_hits,
        "rolling_codes_cache_misses": rolling_misses,
        "markov1_cache_hits": markov_hits,
        "markov1_cache_misses": markov_misses,
        "compilation_mode": "cached" if rolling_hits and markov_hits and not (rolling_misses or markov_misses) else "fresh-or-mixed",
        "compiled_fixture_match": True,
    }


def gate_point(run_dir: Path, *, tail_edges: int = 2) -> dict[str, Any]:
    from v2.phase1.report import build_report_rows
    from v2.phase1.transport import build_transport_rows

    run = load_phase1_run(run_dir)
    report_row = build_report_rows([run])[0]
    transport_row = build_transport_rows([run], tail_edges=tail_edges)[0]
    return {
        "retention_last": float(report_row["retention_last"]),
        "active_mean_tail": float(transport_row["tail_transport_active_mean"]),
        "run_dir": str(run_dir.resolve()),
    }


def load_phase1_run(run_dir: Path) -> dict[str, Any]:
    dataset_path = run_dir / "dataset.json"
    summary_path = run_dir / "phase1_summary.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    return {
        "dataset": dataset,
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "moments_prefix": load_optional_json(run_dir / "moments" / "moments_prefix.json"),
        "moments_suffix": load_optional_json(run_dir / "moments" / "moments_suffix.json"),
        "_dataset_path": str(dataset_path.resolve()),
        "_summary_path": str(summary_path.resolve()),
        "_run_dir": str(run_dir.resolve()),
        "_generated_at_dt": datetime.fromisoformat(dataset["generated_at"]),
        "generated_at": dataset["generated_at"],
    }


def build_growth_shell_readout(
    jobs: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {(int(item["depth"]), str(item["q"]), item["seed"]): item for item in jobs}
    per_window: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    by_depth_q: dict[tuple[int, str], dict[str, Any]] = {}

    for spec in FROZEN_WINDOWS:
        observed_job = lookup.get((spec.depth, spec.q, None))
        if observed_job is None:
            raise RuntimeError(f"Missing observed Phase 1 job for {spec.slug}.")
        observed = gate_point(Path(observed_job["run_dir"]))
        null_points = []
        for seed in SEEDS:
            null_job = lookup.get((spec.depth, spec.q, seed))
            if null_job is None:
                raise RuntimeError(f"Missing Markov-1 Phase 1 job for {spec.slug}, seed={seed}.")
            point = gate_point(Path(null_job["run_dir"]))
            point["seed"] = seed
            null_points.append(point)
            per_seed.append(
                {
                    "depth": spec.depth,
                    "q": spec.q,
                    "seed": seed,
                    "retention_markov1": point["retention_last"],
                    "active_mean_tail_markov1": point["active_mean_tail"],
                    "run_dir": point["run_dir"],
                }
            )

        max_retention = max(item["retention_last"] for item in null_points)
        distance = min(
            math.hypot(
                observed["retention_last"] - item["retention_last"],
                observed["active_mean_tail"] - item["active_mean_tail"],
            )
            for item in null_points
        )
        gap = observed["retention_last"] - max_retention
        row = {
            "depth": spec.depth,
            "q": spec.q,
            "retention_B": observed["retention_last"],
            "active_mean_tail_B": observed["active_mean_tail"],
            "max_retention_markov1": max_retention,
            "G": gap,
            "D": distance,
            "S": gap >= 0.30 and distance >= 0.20,
            "observed_run_dir": observed["run_dir"],
        }
        by_depth_q[(spec.depth, spec.q)] = row
        per_window.append(row)

    delta_rows = []
    for q in ("1/10", "1/2", "9/10"):
        low = by_depth_q[(20, q)]
        high = by_depth_q[(24, q)]
        delta = {
            "q": q,
            "Delta_G": high["G"] - low["G"],
            "Delta_D": high["D"] - low["D"],
        }
        delta_rows.append(delta)
        low.update(delta)
        high.update(delta)
    outcome_state = classify_growth_shell_rows(per_window)
    c20 = outcome_state["C20"]
    c24 = outcome_state["C24"]
    outcome = outcome_state["outcome"]

    summary = {
        "status": "completed",
        "outcome": outcome,
        "C20": c20,
        "C24": c24,
        "thresholds": {"G": 0.30, "D": 0.20, "Delta_G_band": 0.05},
        "per_window": per_window,
        "depth_deltas": delta_rows,
        "n3_boundary": (
            "The q positions are deterministic structural probes, not independent random samples; "
            "no p-value, population confidence interval, or asymptotic claim is authorized."
        ),
    }
    return summary, per_window, per_seed


def classify_growth_shell_rows(per_window: list[dict[str, Any]]) -> dict[str, Any]:
    by_depth_q = {(int(row["depth"]), str(row["q"])): row for row in per_window}
    required = {(depth, q) for depth in (20, 24) for q in ("1/10", "1/2", "9/10")}
    if set(by_depth_q) != required:
        raise ValueError("Outcome classification requires exactly six depth/q rows.")
    c20 = sum(bool(by_depth_q[(20, q)]["S"]) for q in ("1/10", "1/2", "9/10"))
    c24 = sum(bool(by_depth_q[(24, q)]["S"]) for q in ("1/10", "1/2", "9/10"))
    deltas = [by_depth_q[(24, q)]["G"] - by_depth_q[(20, q)]["G"] for q in ("1/10", "1/2", "9/10")]
    negative = sum(value <= -0.05 for value in deltas)
    positive = sum(value >= 0.05 for value in deltas)
    stable = sum(abs(value) < 0.05 for value in deltas)
    if c20 < 2:
        outcome = "markov1-shell-baseline-not-established"
    elif c24 < c20 or negative >= 2:
        outcome = "markov1-shell-separation-contracts-at-iter24"
    elif c24 >= c20 and positive >= 2 and negative == 0:
        outcome = "markov1-shell-separation-expands-at-iter24"
    elif c24 >= 2 and stable >= 2:
        outcome = "markov1-shell-separation-stable-at-iter24"
    else:
        outcome = "markov1-shell-position-heterogeneous"
    return {"outcome": outcome, "C20": c20, "C24": c24, "Delta_G": deltas}


def load_optional_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def current_rss_bytes() -> int:
    if psutil is None:
        return 0
    return int(psutil.Process(os.getpid()).memory_info().rss)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_file_progress(path: Path, logger: RunLogger, *, label: str) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    read = 0
    started = time.monotonic()
    with open(path, "rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
            read += len(chunk)
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = read / elapsed
            logger.emit(
                "progress",
                force=False,
                phase="stage0-source-hash",
                snapshot=label,
                structural_bytes_read=read,
                target_structural_bytes=size,
                percent=round(100.0 * read / max(size, 1), 6),
                rate_bytes_per_second=round(rate, 3),
                elapsed_seconds=round(elapsed, 3),
                eta_seconds=round((size - read) / max(rate, 1e-9), 3),
                peak_rss_bytes=current_rss_bytes(),
                windows_frozen="0/3",
            )
    return digest.hexdigest().upper()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
