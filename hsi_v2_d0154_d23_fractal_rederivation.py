#!/usr/bin/env python3
"""Exact bounded-memory D_f re-derivation over the clean D@23 snapshot."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from v2.common.cli import resolve_results_dir
from v2.signature.streaming_fractal import (
    ALGORITHM_ID,
    legacy_reference,
    load_checkpoint,
    scan_v33_observable_box_counts,
    write_checkpoint_atomic,
)


REPO_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = REPO_DIR.parent
IMPLEMENTATION_PATH = REPO_DIR / "v2" / "signature" / "streaming_fractal.py"
DECISION_LOG_PATH = REPO_DIR / "Documentation" / "HSI_v2_Decision_Log.md"
SOURCE_AUDIT_PATH = (
    WORKSPACE_DIR / "docs" / "HSI_v2_Audit_D0154_D23_Level0_Sanitation_Closure.md"
)
SOURCE_AUDIT_SHA256 = "D1A277DF9B2D3C063B675D61386C28CB896306E665B345C05590F297181D8C04"

D_STRUCT_SHA256 = "7E7CC9412FB349860698D83C798664BABF9C3232FB167E1F5A3B1DE782BAA0EC"
D_METADATA_SHA256 = "7C2D8B99FFE69A1530C81A6093224071D74CAB0626AF8A01846E7DC5C356F9CD"
D_STATE_SHA256 = "F18CDB8C559029D5A03B01C070B901F54B0304E7ECBB5EDA543AE141A9C5919A"
D_REPORT_SHA256 = "1C2C87458C3E98070E5A56907CF4BE5A4DF95A4B8DB339AE912399C6AE4B53E3"
D_SEQUENCE_LENGTH = 232_475_477_397
D_OBSERVABLE_BITS = 226_533_047_253
D_STRUCT_BYTES = 4_519_094_483
D_CURRENT_STATE_SHA256 = (
    "5FECEB66FFC86F38D952786C6D696C79C2DBC239DD4E91B46729D73A27FB57E9"
)

MAX_BOX_SIZE = 64
PAPER_FRACTAL_DIMENSION = 0.992
DISPLAY_DECIMALS = 3
DEFAULT_OUTPUT_ROOT = "results/hsi_v2/d0154_d23_fractal_rederivation"


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only D-0154 D@23 fractal-dimension re-derivation."
    )
    parser.add_argument("--results-base", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume-run", default=None)
    parser.add_argument("--chunk-bytes", type=positive_int, default=4 * 1024 * 1024)
    parser.add_argument(
        "--checkpoint-bytes", type=positive_int, default=256 * 1024 * 1024
    )
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument("--code-audit-file", default=None)
    parser.add_argument("--code-audit-sha256", default=None)
    parser.add_argument("--decision-log-anchor-confirmed", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    configure_stdout()
    args = build_parser().parse_args()
    results_base = resolve_results_base(args.results_base)
    sources = source_paths(results_base)
    output_root = resolve_results_dir(args.output_dir, anchor_file=__file__)
    code_audit_path = (
        None
        if args.code_audit_file is None
        else str(Path(args.code_audit_file).expanduser().resolve())
    )
    code_hashes = {
        "runner": sha256_file(Path(__file__)),
        "implementation": sha256_file(IMPLEMENTATION_PATH),
    }
    plan = {
        "script": Path(__file__).name,
        "results_base": str(results_base),
        "output_root": str(output_root),
        "source_audit": {
            "path": str(SOURCE_AUDIT_PATH),
            "sha256": SOURCE_AUDIT_SHA256,
        },
        "code_audit": {
            "path": code_audit_path,
            "sha256": normalize_sha(args.code_audit_sha256),
        },
        "decision_log_anchor_confirmed": args.decision_log_anchor_confirmed,
        "sources": {key: str(value) for key, value in sources.items()},
        "code_hashes": code_hashes,
        "protocol": {
            "algorithm": ALGORITHM_ID,
            "observable_filter": "2-bit codes 00/01 retained; 10/11 discarded",
            "box_partition": "non-overlapping from observable offset zero",
            "trailing_partial_box": "counted iff it contains at least one 1",
            "fit": "numpy.polyfit(log(box_size), log(max(1, occupied_boxes)), 1)",
            "max_box_size": MAX_BOX_SIZE,
            "active_box_sizes": list(range(2, MAX_BOX_SIZE)),
            "chunk_bytes": args.chunk_bytes,
            "checkpoint_bytes": args.checkpoint_bytes,
            "expected_sequence_length": D_SEQUENCE_LENGTH,
            "expected_observable_bits": D_OBSERVABLE_BITS,
            "paper_value": PAPER_FRACTAL_DIMENSION,
            "display_decimals": DISPLAY_DECIMALS,
        },
    }

    if not args.execute:
        print(json.dumps({**plan, "status": "planned"}, indent=2))
        print("\nDry run only: no D@23 source file was opened or hashed.")
        return 0

    validate_execution_gate(args)
    run_dir = prepare_run_dir(output_root, args.resume_run)
    log_path = run_dir / "run.log"
    runtime_state_path = run_dir / "runtime_state.json"
    validate_or_write_runtime_state(
        runtime_state_path,
        plan,
        require_existing=args.resume_run is not None,
    )

    started = time.monotonic()
    error: str | None = None
    equivalence: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    peak_rss = peak_rss_bytes()
    log(log_path, "[start] D-0154 clean D@23 streaming D_f re-derivation")

    try:
        verify_static_anchors(log_path)
        equivalence = run_equivalence_gate(run_dir, log_path)
        if not equivalence["passed"]:
            raise RuntimeError("synthetic legacy-equivalence gate failed")

        verify_source_chain(sources, log_path, args.progress_seconds)
        checkpoint_path = run_dir / "checkpoint.json"
        resume_state = (
            load_checkpoint(checkpoint_path) if checkpoint_path.exists() else None
        )
        progress = ProgressLogger(
            log_path,
            interval_seconds=args.progress_seconds,
            total_structural_chars=D_SEQUENCE_LENGTH,
        )
        result = scan_v33_observable_box_counts(
            sources["struct"],
            sequence_length=D_SEQUENCE_LENGTH,
            expected_observable_bits=D_OBSERVABLE_BITS,
            source_sha256=D_STRUCT_SHA256,
            max_box_size=MAX_BOX_SIZE,
            raw_chunk_bytes=args.chunk_bytes,
            checkpoint_interval_bytes=args.checkpoint_bytes,
            resume_state=resume_state,
            progress_callback=progress,
            checkpoint_callback=lambda payload: write_checkpoint_atomic(
                checkpoint_path, payload
            ),
        )
        result.update(classify_reconciliation(float(result["fractal_dimension"])))
        log(
            log_path,
            f"[result] D_f={result['fractal_dimension']:.15f} "
            f"display={result['clean_display']} status={result['reconciliation_status']}",
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log(log_path, f"[failure] {error}")

    peak_rss = max(peak_rss, peak_rss_bytes())
    elapsed = time.monotonic() - started
    summary = build_summary(result, equivalence, error, elapsed, peak_rss)
    write_json(run_dir / "summary.json", summary)
    write_box_counts_csv(run_dir / "box_counts.csv", result)
    write_report(run_dir / "report.md", summary)
    finalize_manifest(run_dir, plan, sources, summary, elapsed, peak_rss)
    manifest_sha = sha256_file(run_dir / "manifest.json")
    (run_dir / "manifest.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii"
    )

    print(f"Status: {summary['status']}")
    if result is not None:
        print(f"D_f: {result['fractal_dimension']:.15f}")
        print(f"Reconciliation: {result['reconciliation_status']}")
    print(f"Output: {run_dir}")
    print(f"Manifest SHA-256: {manifest_sha}")
    return 0 if error is None else 2


def resolve_results_base(raw: str | None) -> Path:
    value = raw or os.environ.get("HSI_RESULTS_BASE_DIR")
    if not value:
        value = str(REPO_DIR / "results")
    return Path(value).expanduser().resolve()


def source_paths(results_base: Path) -> dict[str, Path]:
    var_dir = results_base / "level0" / "phi_snapshots" / "var_D"
    return {
        "struct": var_dir / "phi_iter23.struct.gz",
        "metadata": var_dir / "phi_iter23.json",
        "state": var_dir / "phi_iter23.state.txt.gz",
        "report": results_base
        / "level0"
        / "reports"
        / "variant_D_23_20260818_092824.json",
    }


def validate_execution_gate(args: argparse.Namespace) -> None:
    if not args.decision_log_anchor_confirmed:
        raise SystemExit("--decision-log-anchor-confirmed is required for execution")
    audit_sha = normalize_sha(args.code_audit_sha256)
    if audit_sha is None or args.code_audit_file is None:
        raise SystemExit("code audit file and SHA-256 are required for execution")
    audit_path = Path(args.code_audit_file).expanduser().resolve()
    if not audit_path.exists():
        raise SystemExit(f"code audit file not found: {audit_path}")
    if sha256_file(audit_path) != audit_sha:
        raise SystemExit("code audit SHA-256 does not match the supplied file")


def verify_static_anchors(log_path: Path) -> None:
    verify_file(
        SOURCE_AUDIT_PATH,
        SOURCE_AUDIT_SHA256,
        None,
        log_path,
        "D@23 Level 0 closure audit",
    )
    decision_text = DECISION_LOG_PATH.read_text(encoding="utf-8")
    if SOURCE_AUDIT_SHA256 not in decision_text:
        raise RuntimeError("D@23 source audit is not anchored in the Decision Log")
    if "Amendment 31" not in decision_text:
        raise RuntimeError("D@23 audit closure amendment is missing")
    log(log_path, "[anchor] D@23 source audit and Amendment 31 verified")


def verify_source_chain(
    sources: dict[str, Path], log_path: Path, progress_seconds: float
) -> None:
    verify_file_with_progress(
        sources["struct"],
        D_STRUCT_SHA256,
        D_STRUCT_BYTES,
        log_path,
        "D@23 structural source",
        progress_seconds,
    )
    verify_file(sources["metadata"], D_METADATA_SHA256, None, log_path, "D@23 metadata")
    verify_file(sources["state"], D_STATE_SHA256, None, log_path, "D@23 state")
    verify_file(
        sources["report"], D_REPORT_SHA256, None, log_path, "D@23 Level 0 report"
    )

    metadata = read_json(sources["metadata"])
    expected_metadata = {
        "iteration": 23,
        "sequence_length": D_SEQUENCE_LENGTH,
        "format": "v33_structural_streaming",
        "encoding": "2bit",
        "current_state_sha256": D_CURRENT_STATE_SHA256,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"D@23 metadata mismatch for {key}")

    with gzip.open(sources["state"], "rt", encoding="utf-8") as handle:
        state = handle.read()
    if state != "0" or sha256_bytes(state.encode("utf-8")) != D_CURRENT_STATE_SHA256:
        raise RuntimeError("D@23 state content does not match the frozen state hash")

    report = read_json(sources["report"])
    if report.get("variant") != "D" or int(report.get("iterations", -1)) != 23:
        raise RuntimeError("D@23 report identity mismatch")
    if int(report.get("total_bits", -1)) != D_OBSERVABLE_BITS:
        raise RuntimeError("D@23 observable length differs from the frozen report")
    if report.get("fractal_dimension") is not None:
        raise RuntimeError("D@23 report unexpectedly contains a pre-existing D_f value")
    log(log_path, "[source] clean D@23 source chain verified; report D_f remains null")


def run_equivalence_gate(run_dir: Path, log_path: Path) -> dict[str, Any]:
    fixtures = {
        "mixed-structural": "0(1)001(()1)0(0)11100",
        "sparse-observable": "0000000010000000000000001000000000000000",
        "dense-observable": "1110111111011111110111111101111111",
        "non-byte-aligned": "(01)1(0",
    }
    rows = []
    passed = True
    for name, structural in fixtures.items():
        observable_text = "".join(char for char in structural if char in "01")
        bits = np.frombuffer(observable_text.encode("ascii"), dtype=np.uint8) - ord("0")
        reference = legacy_reference(bits, max_box_size=MAX_BOX_SIZE)
        packed = pack_structural(structural)
        fixture_path = run_dir / f"equivalence_{name}.struct.gz"
        with gzip.open(fixture_path, "wb", compresslevel=1) as handle:
            handle.write(packed)
        source_sha = sha256_file(fixture_path)
        observed = scan_v33_observable_box_counts(
            fixture_path,
            sequence_length=len(structural),
            expected_observable_bits=len(observable_text),
            source_sha256=source_sha,
            max_box_size=MAX_BOX_SIZE,
            raw_chunk_bytes=1,
            checkpoint_interval_bytes=2,
        )
        reference_counts = [
            (row["box_size"], row["occupied_boxes"]) for row in reference["box_counts"]
        ]
        observed_counts = [
            (row["box_size"], row["occupied_boxes"]) for row in observed["box_counts"]
        ]
        exact = (
            reference_counts == observed_counts
            and observed["observable_bits"] == len(observable_text)
            and abs(
                float(reference["fractal_dimension"])
                - float(observed["fractal_dimension"])
            )
            <= 1e-15
        )
        rows.append(
            {
                "fixture": name,
                "structural_chars": len(structural),
                "observable_bits": len(observable_text),
                "reference_dimension": reference["fractal_dimension"],
                "streaming_dimension": observed["fractal_dimension"],
                "exact_match": exact,
            }
        )
        passed = passed and exact
        fixture_path.unlink()

    payload = {"passed": passed, "fixtures": rows}
    write_json(run_dir / "equivalence.json", payload)
    log(log_path, f"[equivalence] synthetic legacy equivalence passed={passed}")
    return payload


def classify_reconciliation(value: float) -> dict[str, Any]:
    paper_display = f"{PAPER_FRACTAL_DIMENSION:.{DISPLAY_DECIMALS}f}"
    clean_display = f"{value:.{DISPLAY_DECIMALS}f}"
    status = "display-match" if clean_display == paper_display else "display-mismatch"
    return {
        "paper_value": PAPER_FRACTAL_DIMENSION,
        "paper_display": paper_display,
        "clean_display": clean_display,
        "absolute_delta": abs(value - PAPER_FRACTAL_DIMENSION),
        "reconciliation_status": status,
        "review_required": status != "display-match",
    }


def build_summary(
    result: dict[str, Any] | None,
    equivalence: dict[str, Any] | None,
    error: str | None,
    elapsed: float,
    peak_rss: int,
) -> dict[str, Any]:
    return {
        "status": "completed" if error is None else "stopped",
        "error": error,
        "variant": "D",
        "iteration": 23,
        "source_sha256": D_STRUCT_SHA256,
        "source_sequence_length": D_SEQUENCE_LENGTH,
        "source_observable_bits": D_OBSERVABLE_BITS,
        "equivalence_gate": equivalence,
        "result": result,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "boundary": (
            "Read-only deterministic reconciliation of the legacy D_f definition; "
            "no Level 0 report or manuscript file is modified."
        ),
    }


def write_box_counts_csv(path: Path, result: dict[str, Any] | None) -> None:
    rows = [] if result is None else result.get("box_counts", [])
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "box_size",
                "occupied_boxes",
                "total_boxes",
                "occupancy_fraction",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    result = summary.get("result")
    lines = [
        "# D-0154 clean D@23 fractal-dimension re-derivation",
        "",
        f"- Status: `{summary['status']}`",
        f"- Source: clean `D@23` (`{D_STRUCT_SHA256}`)",
        f"- Algorithm: `{ALGORITHM_ID}`",
        f"- Synthetic equivalence: `{bool(summary.get('equivalence_gate', {}).get('passed'))}`",
        f"- Elapsed seconds: `{summary['elapsed_seconds']:.3f}`",
        f"- Peak RSS bytes: `{summary['peak_rss_bytes']}`",
        "",
    ]
    if summary.get("error"):
        lines.extend(["## Stop", "", f"`{summary['error']}`", ""])
    if result is not None:
        lines.extend(
            [
                "## Readout",
                "",
                f"- Full-stream D_f: `{result['fractal_dimension']:.15f}`",
                f"- Paper display: `{result['paper_display']}`",
                f"- Clean display: `{result['clean_display']}`",
                f"- Reconciliation: `{result['reconciliation_status']}`",
                f"- Observable bits scanned: `{result['observable_bits']}`",
                f"- Observable ones: `{result['observable_ones']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Method boundary",
            "",
            "This is a deterministic read-only reproduction of the legacy v33 observable box-counting definition with `max_box_size=64`. It does not introduce a new estimator, threshold, source, or scientific claim. The clean Level 0 report and the manuscript remain untouched pending independent artifact audit and editorial ruling.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def finalize_manifest(
    run_dir: Path,
    plan: dict[str, Any],
    sources: dict[str, Path],
    summary: dict[str, Any],
    elapsed: float,
    peak_rss: int,
) -> None:
    artifact_names = (
        "summary.json",
        "report.md",
        "box_counts.csv",
        "equivalence.json",
        "checkpoint.json",
        "runtime_state.json",
        "run.log",
    )
    artifacts = {}
    for name in artifact_names:
        path = run_dir / name
        if path.exists():
            artifacts[name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    payload = {
        **plan,
        "status": summary["status"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "source_chain": {
            "struct": frozen_artifact(sources["struct"], D_STRUCT_SHA256),
            "metadata": frozen_artifact(sources["metadata"], D_METADATA_SHA256),
            "state": frozen_artifact(sources["state"], D_STATE_SHA256),
            "level0_report": frozen_artifact(sources["report"], D_REPORT_SHA256),
        },
        "reconciliation_status": (
            None
            if summary.get("result") is None
            else summary["result"]["reconciliation_status"]
        ),
        "artifacts": artifacts,
    }
    write_json(run_dir / "manifest.json", payload)


def prepare_run_dir(output_root: Path, resume_run: str | None) -> Path:
    if resume_run:
        run_dir = Path(resume_run).expanduser().resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"resume run directory not found: {run_dir}")
        return run_dir
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"d0154-d23-fractal__maxbox-64__{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def validate_or_write_runtime_state(
    path: Path, plan: dict[str, Any], *, require_existing: bool
) -> None:
    identity = {
        "script": plan["script"],
        "sources": plan["sources"],
        "code_hashes": plan["code_hashes"],
        "code_audit": plan["code_audit"],
        "protocol": plan["protocol"],
    }
    if path.exists():
        existing = read_json(path)
        if existing != identity:
            raise SystemExit("resume runtime identity differs from the original run")
        return
    if require_existing:
        raise SystemExit("resume run lacks runtime_state.json")
    write_json(path, identity)


class ProgressLogger:
    def __init__(
        self, log_path: Path, interval_seconds: float, total_structural_chars: int
    ):
        self.log_path = log_path
        self.interval_seconds = max(1.0, interval_seconds)
        self.total_structural_chars = total_structural_chars
        self.started = time.monotonic()
        self.last_emit = 0.0
        self.last_phase = None

    def __call__(self, payload: dict[str, Any]) -> None:
        now = time.monotonic()
        phase = payload["phase"]
        target = int(payload["packed_bytes_target"])
        current = int(payload["packed_bytes_read"])
        if (
            now - self.last_emit < self.interval_seconds
            and current < target
            and phase == self.last_phase
        ):
            return
        self.last_emit = now
        self.last_phase = phase
        elapsed = max(now - self.started, 1e-9)
        pct = 100.0 * current / target if target else 100.0
        if phase == "scan":
            structural = int(payload["structural_chars_seen"])
            rate = structural / elapsed
            eta = (self.total_structural_chars - structural) / rate if rate else 0.0
            message = (
                f"[progress] scan {pct:6.2f}% | structural={structural:,}/"
                f"{self.total_structural_chars:,} | observable="
                f"{int(payload['observable_bits_seen']):,} | rate={rate / 1e6:.2f} Mchar/s | "
                f"eta={format_duration(eta)}"
            )
        else:
            message = (
                f"[progress] checkpoint recovery {pct:6.2f}% | "
                f"packed={current:,}/{target:,} bytes"
            )
        log(self.log_path, message)


def verify_file(
    path: Path,
    expected_sha: str,
    expected_bytes: int | None,
    log_path: Path,
    label: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise RuntimeError(f"{label} byte-size mismatch")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise RuntimeError(f"{label} SHA-256 mismatch")
    log(log_path, f"[hash] {label}: {actual}")


def verify_file_with_progress(
    path: Path,
    expected_sha: str,
    expected_bytes: int,
    log_path: Path,
    label: str,
    progress_seconds: float,
) -> None:
    if not path.exists() or path.stat().st_size != expected_bytes:
        raise RuntimeError(f"{label} missing or byte-size mismatch")
    digest = hashlib.sha256()
    processed = 0
    started = time.monotonic()
    last = started
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
            processed += len(block)
            now = time.monotonic()
            if now - last >= max(1.0, progress_seconds):
                last = now
                pct = 100.0 * processed / expected_bytes
                log(
                    log_path,
                    f"[hash] {label}: {pct:.1f}% ({processed:,}/{expected_bytes:,})",
                )
    actual = digest.hexdigest().upper()
    if actual != expected_sha:
        raise RuntimeError(f"{label} SHA-256 mismatch")
    log(log_path, f"[hash] {label}: {actual}")


def frozen_artifact(path: Path, expected_sha: str) -> dict[str, Any]:
    return {"path": str(path), "sha256": expected_sha, "bytes": path.stat().st_size}


def pack_structural(structural: str) -> bytes:
    code = {"0": 0, "1": 1, "(": 2, ")": 3}
    output = bytearray()
    for start in range(0, len(structural), 4):
        values = [code[value] for value in structural[start : start + 4]]
        values.extend([0] * (4 - len(values)))
        output.append(
            (values[0] << 6) | (values[1] << 4) | (values[2] << 2) | values[3]
        )
    return bytes(output)


def normalize_sha(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip().upper()
    if len(value) != 64 or any(char not in "0123456789ABCDEF" for char in value):
        raise SystemExit(
            "SHA-256 values must contain exactly 64 hexadecimal characters"
        )
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def log(path: Path, message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def peak_rss_bytes() -> int:
    try:
        import psutil

        memory = psutil.Process().memory_info()
        return int(getattr(memory, "peak_wset", memory.rss))
    except Exception:
        return 0


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


if __name__ == "__main__":
    raise SystemExit(main())
