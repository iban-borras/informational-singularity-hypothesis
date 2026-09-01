#!/usr/bin/env python3
"""D-0154 exact streaming cbar re-derivation for clean F@27."""

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

from level1_nesting_tree import build_tree_from_structural, collect_all_metrics_iterative
from v2.common.cli import resolve_results_dir
from v2.signature.streaming_cbar import (
    DyckMeanChildrenCounter,
    SegmentSpec,
    load_checkpoint,
    scan_v33_gzip_segments,
    write_checkpoint_atomic,
)


REPO_DIR = Path(__file__).resolve().parent
PREREG_PATH = REPO_DIR.parent / "prereg_D0154_F27_cbar_streaming_v1.0.md"
PREREG_SHA256 = "B21B5935631D4C4B9CA2D98CC56E1B67C92542235EA4A49675F7190367F01C02"
DESIGN_AUDIT_PATH = REPO_DIR.parent / "docs" / "HSI_v2_Audit_Prereg_D0154_F27_cbar_streaming_v0.1.md"
DESIGN_AUDIT_SHA256 = "21D4493F9DC7FEF6C1B608D553B096048A87CE06E557467EB4C3DB417B48D71D"
IMPLEMENTATION_PATH = REPO_DIR / "v2" / "signature" / "streaming_cbar.py"
DECISION_LOG_PATH = REPO_DIR / "Documentation" / "HSI_v2_Decision_Log.md"

F_STRUCT_SHA256 = "B3D77CBCC3507C96D9E20C71A84A965C6C01CAC127006D51B0250BA65F75F970"
F_METADATA_SHA256 = "B89EC7FA2AC44CBA4ADAF6306D6EB27420424BAE963F431728D1A801A48A8D1F"
F_STATE_SHA256 = "0A41CF52CF0505F6DA289BAD33C3DCD2E774DDD35A4B7458E7FE6BC9FB25A834"
F_SEQUENCE_LENGTH = 1_397_647_527_079
F_STRUCT_BYTES = 20_031_695_910
F_OBSERVABLE_BITS = 1_118_478_867_305
F_CURRENT_STATE_SHA256 = "5FECEB66FFC86F38D952786C6D696C79C2DBC239DD4E91B46729D73A27FB57E9"

E_STRUCT_SHA256 = "8D3BE0CE2159B699D424103C7BD4C02517D2A020ABE2B671EFFFA15CDCAC2DAD"
E_SEQUENCE_LENGTH = 702_184_938_708
E_EXPECTED = {
    "total_nodes": 18_147_523,
    "total_edges": 18_147_522,
    "total_branching_nodes": 6_931_737,
    "cbar": 2.61803383480937,
}

SEGMENT_CHARS = 1_000_000_000
F_SEGMENTS = [
    SegmentSpec(1, 0, SEGMENT_CHARS),
    SegmentSpec(2, 349_161_881_769, SEGMENT_CHARS),
    SegmentSpec(3, 698_323_763_539, SEGMENT_CHARS),
    SegmentSpec(4, 1_047_485_645_309, SEGMENT_CHARS),
]
PHI_PLUS_ONE = 2.618033988749895
AGGREGATE_TOLERANCE = 1e-6
SEGMENT_TOLERANCE = 5e-6
DEFAULT_OUTPUT_ROOT = "results/hsi_v2/d0154_f27_cbar_streaming"

SYNTHETIC_FIXTURES = {
    "single-child": "()",
    "multiple-root-children": "()()",
    "nested-siblings": "(()())",
    "unmatched-closes": "))()",
    "unclosed-opens": "((()",
    "mixed-non-byte-aligned": "0(1)0",
    "nested-content": "1((0)(1(0)))1",
}


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact D-0154 streaming re-derivation of F@27 cbar."
    )
    parser.add_argument("--stage", choices=("stage0", "stage1", "all"), default="all")
    parser.add_argument("--results-base", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume-run", default=None)
    parser.add_argument("--stage0-equivalence", default=None)
    parser.add_argument("--chunk-bytes", type=positive_int, default=16 * 1024 * 1024)
    parser.add_argument("--checkpoint-bytes", type=positive_int, default=256 * 1024 * 1024)
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument("--code-audit-sha256", default=None)
    parser.add_argument("--code-audit-file", default=None)
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
        None if args.code_audit_file is None else str(Path(args.code_audit_file).expanduser().resolve())
    )
    plan = {
        "script": Path(__file__).name,
        "stage": args.stage,
        "results_base": str(results_base),
        "output_root": str(output_root),
        "preregistration": {"path": str(PREREG_PATH), "sha256": PREREG_SHA256},
        "design_audit": {"path": str(DESIGN_AUDIT_PATH), "sha256": DESIGN_AUDIT_SHA256},
        "code_audit": {
            "path": code_audit_path,
            "sha256": normalize_sha(args.code_audit_sha256),
        },
        "decision_log_anchor_confirmed": args.decision_log_anchor_confirmed,
        "sources": {key: str(value) for key, value in sources.items()},
        "segments": [spec.to_dict() for spec in F_SEGMENTS],
        "protocol": {
            "packed_alphabet": {"00": "0", "01": "1", "10": "(", "11": ")"},
            "segment_chars": SEGMENT_CHARS,
            "num_segments": len(F_SEGMENTS),
            "aggregate_tolerance": AGGREGATE_TOLERANCE,
            "segment_tolerance": SEGMENT_TOLERANCE,
            "chunk_bytes": args.chunk_bytes,
            "checkpoint_bytes": args.checkpoint_bytes,
            "f27_structural_sequence_length": F_SEQUENCE_LENGTH,
            "f27_observable_bits": F_OBSERVABLE_BITS,
        },
    }
    if not args.execute:
        print(json.dumps({**plan, "status": "planned"}, indent=2))
        print("\nDry run only: no source file was opened or hashed.")
        return 0

    validate_execution_gate(args, plan)
    run_dir = prepare_run_dir(output_root, args.resume_run)
    log_path = run_dir / "run.log"
    runtime_state_path = run_dir / "runtime_state.json"
    validate_or_write_runtime_state(
        runtime_state_path,
        plan,
        require_existing=args.resume_run is not None,
    )
    log(log_path, f"[start] D-0154 F@27 streaming cbar | stage={args.stage}")

    started = time.monotonic()
    peak_rss = peak_rss_bytes()
    equivalence: dict[str, Any] | None = None
    stage1: dict[str, Any] | None = None
    error: str | None = None

    try:
        verify_static_anchors(log_path)
        if args.stage in ("stage0", "all"):
            equivalence = execute_stage0(
                sources,
                run_dir,
                log_path,
                args.chunk_bytes,
                args.checkpoint_bytes,
                args.progress_seconds,
            )
            write_json(run_dir / "equivalence.json", equivalence)
        else:
            equivalence = load_external_equivalence(args.stage0_equivalence)
            write_json(run_dir / "equivalence.json", equivalence)

        validate_equivalence_payload(equivalence)
        calibration_source = Path(
            equivalence["e24_calibration"]["source_path"]
        ).resolve()
        if calibration_source != sources["e_struct"].resolve():
            raise EquivalenceGateError(
                "Stage 0 calibration source path differs from the canonical E@24 source"
            )
        calibration_metadata = Path(
            equivalence["e24_calibration"]["metadata_path"]
        ).resolve()
        if calibration_metadata != sources["e_metadata"].resolve():
            raise EquivalenceGateError(
                "Stage 0 metadata path differs from the canonical E@24 metadata"
            )

        if args.stage in ("stage1", "all"):
            stage1 = execute_stage1(
                sources,
                run_dir,
                log_path,
                args.chunk_bytes,
                args.checkpoint_bytes,
                args.progress_seconds,
            )
    except EquivalenceGateError as exc:
        error = str(exc)
        log(log_path, f"[stop] {error}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log(log_path, f"[failure] {error}")

    peak_rss = max(peak_rss, peak_rss_bytes())
    elapsed = time.monotonic() - started
    outcome = classify_outcome(equivalence, stage1, error, args.stage)
    summary = build_summary(outcome, equivalence, stage1, error, elapsed, peak_rss)
    write_json(run_dir / "summary.json", summary)
    write_segment_csv(run_dir / "per_segment_cbar.csv", stage1)
    write_report(run_dir / "report.md", summary)
    finalize_manifest(run_dir, plan, sources, summary, elapsed, peak_rss)
    manifest_sha = sha256_file(run_dir / "manifest.json")
    (run_dir / "manifest.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii"
    )

    print(f"Outcome: {outcome if outcome is not None else 'N.A. (Stage 0 only)'}")
    print(f"Output: {run_dir}")
    print(f"Manifest SHA-256: {manifest_sha}")
    return 0 if error is None else 2


class EquivalenceGateError(RuntimeError):
    pass


def execute_stage0(
    sources: dict[str, Path],
    run_dir: Path,
    log_path: Path,
    chunk_bytes: int,
    checkpoint_bytes: int,
    progress_seconds: float,
) -> dict[str, Any]:
    log(log_path, "[stage0] Synthetic equivalence")
    synthetic = run_synthetic_equivalence()
    if not synthetic["passed"]:
        return {"status": "failed", "synthetic": synthetic, "e24_calibration": None}

    e_path = sources["e_struct"]
    verify_file(e_path, E_STRUCT_SHA256, None, log_path, "E@24 structural source")
    metadata = read_json(sources["e_metadata"])
    e_metadata_expected = {
        "iteration": 24,
        "sequence_length": E_SEQUENCE_LENGTH,
        "format": "v33_structural_streaming",
        "encoding": "2bit",
    }
    for key, expected in e_metadata_expected.items():
        if metadata.get(key) != expected:
            raise EquivalenceGateError(f"E@24 metadata mismatch for {key}")

    checkpoint_path = run_dir / "stage0_checkpoint.json"
    resume = load_checkpoint(checkpoint_path) if checkpoint_path.exists() else None
    progress = ProgressLogger(log_path, progress_seconds, e_path.stat().st_size)
    scan = scan_v33_gzip_segments(
        e_path,
        E_SEQUENCE_LENGTH,
        [SegmentSpec(1, 0, SEGMENT_CHARS)],
        source_sha256=E_STRUCT_SHA256,
        chunk_bytes=chunk_bytes,
        checkpoint_interval_bytes=checkpoint_bytes,
        require_numba=True,
        resume_state=resume,
        progress_callback=progress,
        checkpoint_callback=lambda state: write_checkpoint_atomic(checkpoint_path, state),
    )
    row = scan["segments"][0]
    integer_match = all(int(row[key]) == value for key, value in E_EXPECTED.items() if key != "cbar")
    cbar_delta = abs(float(row["cbar"]) - E_EXPECTED["cbar"])
    calibration_passed = integer_match and cbar_delta <= 1e-15
    calibration = {
        "source_path": str(e_path),
        "source_sha256": E_STRUCT_SHA256,
        "metadata_path": str(sources["e_metadata"]),
        "metadata_sha256": sha256_file(sources["e_metadata"]),
        "expected": E_EXPECTED,
        "observed": row,
        "integer_match": integer_match,
        "cbar_absolute_delta": cbar_delta,
        "passed": calibration_passed,
        "scan_runtime": runtime_fields(scan),
    }
    status = "passed" if synthetic["passed"] and calibration_passed else "failed"
    log(log_path, f"[stage0] status={status}")
    return {
        "status": status,
        "preregistration_sha256": PREREG_SHA256,
        "synthetic": synthetic,
        "e24_calibration": calibration,
    }


def execute_stage1(
    sources: dict[str, Path],
    run_dir: Path,
    log_path: Path,
    chunk_bytes: int,
    checkpoint_bytes: int,
    progress_seconds: float,
) -> dict[str, Any]:
    log(log_path, "[stage1] Verifying frozen F@27 source chain")
    verify_file(sources["f_struct"], F_STRUCT_SHA256, F_STRUCT_BYTES, log_path, "F@27 structural source")
    verify_file(sources["f_metadata"], F_METADATA_SHA256, None, log_path, "F@27 metadata")
    verify_file(sources["f_state"], F_STATE_SHA256, None, log_path, "F@27 state")
    metadata = read_json(sources["f_metadata"])
    verify_f_metadata(metadata, sources["f_state"])

    checkpoint_path = run_dir / "stage1_checkpoint.json"
    resume = load_checkpoint(checkpoint_path) if checkpoint_path.exists() else None
    progress = ProgressLogger(log_path, progress_seconds, sources["f_struct"].stat().st_size)
    scan = scan_v33_gzip_segments(
        sources["f_struct"],
        F_SEQUENCE_LENGTH,
        F_SEGMENTS,
        source_sha256=F_STRUCT_SHA256,
        chunk_bytes=chunk_bytes,
        checkpoint_interval_bytes=checkpoint_bytes,
        require_numba=True,
        resume_state=resume,
        progress_callback=progress,
        checkpoint_callback=lambda state: write_checkpoint_atomic(checkpoint_path, state),
    )
    for row in scan["segments"]:
        if not row["total_branching_nodes"] or row["processed_chars"] != SEGMENT_CHARS:
            raise RuntimeError(f"Incomplete or branchless target segment {row['segment_index']}")
        row["phi_plus_one"] = PHI_PLUS_ONE
        row["phi_plus_one_distance"] = abs(float(row["cbar"]) - PHI_PLUS_ONE)

    aggregate = scan["cbar_aggregate"]
    if aggregate is None:
        raise RuntimeError("F@27 aggregate has zero branching nodes")
    scan["phi_plus_one"] = PHI_PLUS_ONE
    scan["phi_plus_one_distance"] = abs(float(aggregate) - PHI_PLUS_ONE)
    scan["maximum_segment_distance"] = max(
        float(row["phi_plus_one_distance"]) for row in scan["segments"]
    )
    scan["source_chain"] = {
        "struct_path": str(sources["f_struct"]),
        "struct_sha256": F_STRUCT_SHA256,
        "metadata_path": str(sources["f_metadata"]),
        "metadata_sha256": F_METADATA_SHA256,
        "state_path": str(sources["f_state"]),
        "state_sha256": F_STATE_SHA256,
    }
    log(
        log_path,
        f"[stage1] aggregate_cbar={aggregate:.15f} "
        f"distance={scan['phi_plus_one_distance']:.3e}",
    )
    return scan


def run_synthetic_equivalence() -> dict[str, Any]:
    rows = []
    passed = True
    for name, structural in SYNTHETIC_FIXTURES.items():
        root, metadata = build_tree_from_structural(
            structural, max_nodes=1_000_000, timeout_seconds=60
        )
        _, _, _, children_counts, complete = collect_all_metrics_iterative(
            root, timeout_seconds=60
        )
        canonical = {
            "total_nodes": int(metadata["nodes_created"]),
            "total_edges": int(metadata["nodes_created"]) - 1,
            "total_branching_nodes": len(children_counts),
        }
        packed = pack_structural(structural)
        counter = DyckMeanChildrenCounter(allow_python_fallback=False)
        counter.scan(packed, 0, len(structural))
        observed = counter.result()
        exact = complete and all(observed[key] == value for key, value in canonical.items())
        passed = passed and exact
        rows.append(
            {
                "fixture": name,
                "structural_length": len(structural),
                "canonical": canonical,
                "streaming": {key: observed[key] for key in canonical},
                "exact_match": exact,
            }
        )
    return {"passed": passed, "fixtures": rows}


def pack_structural(structural: str) -> bytes:
    code = {"0": 0, "1": 1, "(": 2, ")": 3}
    output = bytearray()
    for start in range(0, len(structural), 4):
        values = [code[value] for value in structural[start : start + 4]]
        values.extend([0] * (4 - len(values)))
        output.append((values[0] << 6) | (values[1] << 4) | (values[2] << 2) | values[3])
    return bytes(output)


def classify_outcome(
    equivalence: dict[str, Any] | None,
    stage1: dict[str, Any] | None,
    error: str | None,
    stage: str,
) -> str | None:
    if error or equivalence is None or equivalence.get("status") != "passed":
        return "inputs-or-equivalence-not-derivable"
    if stage == "stage0":
        return None
    if stage1 is None:
        return "inputs-or-equivalence-not-derivable"
    if float(stage1["phi_plus_one_distance"]) > AGGREGATE_TOLERANCE:
        return "f-cbar-not-preserved-at-27"
    if float(stage1["maximum_segment_distance"]) > SEGMENT_TOLERANCE:
        return "f-cbar-aggregate-only-at-27"
    return "f-cbar-preserved-at-27"


def build_summary(
    outcome: str | None,
    equivalence: dict[str, Any] | None,
    stage1: dict[str, Any] | None,
    error: str | None,
    elapsed: float,
    peak_rss: int,
) -> dict[str, Any]:
    return {
        "status": (
            "stage0-completed"
            if error is None and outcome is None
            else "completed" if error is None else "stopped"
        ),
        "outcome": outcome,
        "scientific_readout": {
            "equivalence": equivalence,
            "f27": None if stage1 is None else scientific_scan_fields(stage1),
        },
        "runtime": {
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 3),
            "peak_rss_bytes": peak_rss,
            "target_scan": None if stage1 is None else runtime_fields(stage1),
            "error": error,
        },
    }


def scientific_scan_fields(scan: dict[str, Any]) -> dict[str, Any]:
    return {
        "segments": scan["segments"],
        "aggregate_total_edges": scan["aggregate_total_edges"],
        "aggregate_total_nodes": scan["aggregate_total_nodes"],
        "aggregate_branching_nodes": scan["aggregate_branching_nodes"],
        "cbar_aggregate": scan["cbar_aggregate"],
        "phi_plus_one": scan["phi_plus_one"],
        "phi_plus_one_distance": scan["phi_plus_one_distance"],
        "maximum_segment_distance": scan["maximum_segment_distance"],
        "source_chain": scan["source_chain"],
    }


def runtime_fields(scan: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in scan.items()
        if key
        in {
            "uncompressed_bytes_read_this_process",
            "packed_byte_offset",
            "compressed_position_rough",
            "structural_chars_traversed",
            "resumed_from_packed_byte",
            "elapsed_seconds_this_process",
            "parser_backends",
        }
    }


def load_external_equivalence(raw_path: str | None) -> dict[str, Any]:
    if not raw_path:
        raise EquivalenceGateError("--stage1 requires --stage0-equivalence")
    path = Path(raw_path).expanduser().resolve()
    payload = read_json(path)
    validate_equivalence_payload(payload)
    payload = dict(payload)
    payload["external_artifact"] = {
        "path": str(path),
        "sha256": sha256_file(path),
    }
    return payload


def validate_equivalence_payload(payload: dict[str, Any]) -> None:
    if payload.get("preregistration_sha256") != PREREG_SHA256:
        raise EquivalenceGateError("Stage 0 preregistration hash mismatch")
    if payload.get("status") != "passed":
        raise EquivalenceGateError("Stage 0 did not pass")
    synthetic = payload.get("synthetic") or {}
    fixtures = synthetic.get("fixtures") or []
    if not synthetic.get("passed") or len(fixtures) != len(SYNTHETIC_FIXTURES):
        raise EquivalenceGateError("Stage 0 synthetic fixture set is incomplete")
    if any(not row.get("exact_match") for row in fixtures):
        raise EquivalenceGateError("Stage 0 contains a non-equivalent synthetic fixture")
    calibration = payload.get("e24_calibration") or {}
    observed = calibration.get("observed") or {}
    if calibration.get("source_sha256") != E_STRUCT_SHA256:
        raise EquivalenceGateError("Stage 0 E@24 source hash mismatch")
    if not calibration.get("passed") or not calibration.get("integer_match"):
        raise EquivalenceGateError("Stage 0 E@24 calibration did not pass")
    for key, expected in E_EXPECTED.items():
        if key == "cbar":
            if abs(float(observed.get(key, float("inf"))) - expected) > 1e-15:
                raise EquivalenceGateError("Stage 0 E@24 cbar mismatch")
        elif int(observed.get(key, -1)) != expected:
            raise EquivalenceGateError(f"Stage 0 E@24 {key} mismatch")


def verify_static_anchors(log_path: Path) -> None:
    verify_file(PREREG_PATH, PREREG_SHA256, None, log_path, "preregistration")
    verify_file(DESIGN_AUDIT_PATH, DESIGN_AUDIT_SHA256, None, log_path, "design audit")


def verify_f_metadata(metadata: dict[str, Any], state_path: Path) -> None:
    required = {
        "iteration": 27,
        "sequence_length": F_SEQUENCE_LENGTH,
        "format": "v33_structural_streaming",
        "encoding": "2bit",
        "current_state_sha256": F_CURRENT_STATE_SHA256,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"F@27 metadata mismatch for {key}")
    if int(metadata.get("compressed_size_bytes", -1)) != F_STRUCT_BYTES:
        raise RuntimeError("F@27 compressed size metadata mismatch")
    with gzip.open(state_path, "rt", encoding="ascii") as handle:
        state = handle.read()
    if len(state) != 1 or hashlib.sha256(state.encode("ascii")).hexdigest().upper() != F_CURRENT_STATE_SHA256:
        raise RuntimeError("F@27 current state content mismatch")


def validate_execution_gate(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    if sha256_file(PREREG_PATH) != PREREG_SHA256:
        raise SystemExit("Frozen preregistration hash mismatch; execution blocked")
    if sha256_file(DESIGN_AUDIT_PATH) != DESIGN_AUDIT_SHA256:
        raise SystemExit("Grace design-audit hash mismatch; execution blocked")
    if not args.decision_log_anchor_confirmed:
        raise SystemExit("Execution requires --decision-log-anchor-confirmed")
    decision_log_text = DECISION_LOG_PATH.read_text(encoding="utf-8")
    if PREREG_SHA256 not in decision_log_text:
        raise SystemExit("Frozen preregistration SHA-256 is not anchored in the Decision Log")
    code_audit = plan["code_audit"]
    if code_audit["sha256"] is None or code_audit["path"] is None:
        raise SystemExit(
            "Execution requires --code-audit-file and --code-audit-sha256 from Grace's code audit"
        )
    audit_path = Path(code_audit["path"])
    if not audit_path.is_file():
        raise SystemExit(f"Code-audit file not found: {audit_path}")
    observed_audit_sha = sha256_file(audit_path)
    if observed_audit_sha != code_audit["sha256"]:
        raise SystemExit(
            f"Code-audit SHA-256 mismatch: {observed_audit_sha} != {code_audit['sha256']}"
        )
    if observed_audit_sha == DESIGN_AUDIT_SHA256:
        raise SystemExit("The preregistration audit cannot substitute for the code audit")
    audit_text = audit_path.read_text(encoding="utf-8", errors="replace")
    if "APPROVED" not in audit_text.upper():
        raise SystemExit("Code audit does not contain an APPROVED verdict")
    audited_files = {
        Path(__file__).name: sha256_file(Path(__file__)),
        IMPLEMENTATION_PATH.name: sha256_file(IMPLEMENTATION_PATH),
    }
    for filename, code_sha in audited_files.items():
        if filename not in audit_text or code_sha not in audit_text.upper():
            raise SystemExit(
                f"Code audit does not bind {filename} at current SHA-256 {code_sha}"
            )
    if code_audit["sha256"] not in decision_log_text:
        raise SystemExit("Code-audit SHA-256 is not anchored in the Decision Log")


def prepare_run_dir(output_root: Path, resume_run: str | None) -> Path:
    if resume_run:
        run_dir = Path(resume_run).expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        return run_dir
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"d0154-f27-cbar-streaming__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def validate_or_write_runtime_state(
    path: Path,
    plan: dict[str, Any],
    *,
    require_existing: bool,
) -> None:
    frozen = {
        "format": "hsi-v2-d0154-f27-cbar-runtime-v1",
        "stage": plan["stage"],
        "results_base": plan["results_base"],
        "preregistration_sha256": PREREG_SHA256,
        "code_audit": plan["code_audit"],
        "protocol": plan["protocol"],
    }
    if path.exists():
        if read_json(path) != frozen:
            raise RuntimeError("Resume-run protocol differs from its frozen runtime state")
        return
    if require_existing:
        raise RuntimeError("Resume directory has no frozen runtime_state.json")
    write_json(path, frozen)


def source_paths(results_base: Path) -> dict[str, Path]:
    root = results_base / "level0" / "phi_snapshots"
    return {
        "e_struct": root / "var_E" / "phi_iter24.struct.gz",
        "e_metadata": root / "var_E" / "phi_iter24.json",
        "f_struct": root / "var_F" / "phi_iter27.struct.gz",
        "f_metadata": root / "var_F" / "phi_iter27.json",
        "f_state": root / "var_F" / "phi_iter27.state.txt.gz",
    }


def resolve_results_base(raw_path: str | None) -> Path:
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    configured = os.environ.get("HSI_RESULTS_BASE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (REPO_DIR / "results").resolve()


def finalize_manifest(
    run_dir: Path,
    plan: dict[str, Any],
    sources: dict[str, Path],
    summary: dict[str, Any],
    elapsed: float,
    peak_rss: int,
) -> dict[str, Any]:
    source_records: dict[str, Any] = {}
    equivalence = summary["scientific_readout"].get("equivalence") or {}
    calibration = equivalence.get("e24_calibration") or {}
    if calibration:
        source_records["e_struct"] = {
            "path": calibration["source_path"],
            "bytes": sources["e_struct"].stat().st_size,
            "sha256": calibration["source_sha256"],
        }
        source_records["e_metadata"] = {
            "path": calibration["metadata_path"],
            "bytes": sources["e_metadata"].stat().st_size,
            "sha256": calibration["metadata_sha256"],
        }
    f27 = summary["scientific_readout"].get("f27") or {}
    if f27:
        chain = f27["source_chain"]
        source_records["f_struct"] = {
            "path": chain["struct_path"],
            "bytes": sources["f_struct"].stat().st_size,
            "sha256": chain["struct_sha256"],
        }
        source_records["f_metadata"] = {
            "path": chain["metadata_path"],
            "bytes": sources["f_metadata"].stat().st_size,
            "sha256": chain["metadata_sha256"],
        }
        source_records["f_state"] = {
            "path": chain["state_path"],
            "bytes": sources["f_state"].stat().st_size,
            "sha256": chain["state_sha256"],
        }
    manifest = {
        **plan,
        "status": summary["status"],
        "outcome": summary["outcome"],
        "source_records": source_records,
        "implementation": {
            "wrapper_sha256": sha256_file(Path(__file__)),
            "streaming_counter_sha256": sha256_file(IMPLEMENTATION_PATH),
            "python": sys.version,
        },
        "runtime": {
            "elapsed_seconds": round(elapsed, 3),
            "peak_rss_bytes": peak_rss,
        },
        "outputs": {},
    }
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)
    outputs = {}
    for name in ("summary.json", "report.md", "per_segment_cbar.csv", "equivalence.json", "run.log"):
        path = run_dir / name
        if path.exists():
            outputs[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest["outputs"] = outputs
    write_json(manifest_path, manifest)
    return manifest


class ProgressLogger:
    def __init__(self, log_path: Path, interval: float, compressed_size: int) -> None:
        self.log_path = log_path
        self.interval = max(1.0, interval)
        self.compressed_size = compressed_size
        self.last = 0.0

    def __call__(self, payload: dict[str, Any]) -> None:
        now = time.monotonic()
        if now - self.last < self.interval:
            return
        self.last = now
        rss = peak_rss_bytes()
        if payload.get("phase") == "recovery-seek":
            done = int(payload["packed_bytes_discarded"])
            target = int(payload["packed_bytes_target"])
            log(self.log_path, f"[resume] {done:,}/{target:,} packed bytes ({100.0 * done / target:.2f}%)")
            return
        compressed = int(payload.get("compressed_position_rough", 0))
        compressed_pct = 100.0 * compressed / self.compressed_size if self.compressed_size else 0.0
        segment = payload.get("active_segment", "gap")
        segment_pct = 100.0 * float(payload.get("active_segment_fraction", 0.0))
        log(
            self.log_path,
            f"[progress] segment={segment} segment_progress={segment_pct:.2f}% "
            f"completed={payload['completed_segments']}/{payload['total_segments']} "
            f"structural_offset={payload['structural_char_offset']:,} "
            f"compressed_rough={compressed_pct:.2f}% rss={format_bytes(rss)} "
            f"elapsed={format_duration(float(payload['elapsed_seconds']))}",
        )


def write_segment_csv(path: Path, stage1: dict[str, Any] | None) -> None:
    fields = [
        "segment_index",
        "start_char",
        "end_char_exclusive",
        "length_chars",
        "total_nodes",
        "total_edges",
        "total_branching_nodes",
        "cbar",
        "phi_plus_one_distance",
        "max_stack_depth",
        "max_children",
        "unmatched_closes",
        "unclosed_nodes",
        "parser_backend",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        if stage1 is not None:
            writer.writerows(stage1.get("segments", []))


def write_report(path: Path, summary: dict[str, Any]) -> None:
    readout = summary["scientific_readout"]
    outcome_display = summary["outcome"] or "N.A. (Stage 0 only)"
    lines = [
        "# D-0154 F@27 exact streaming cbar re-derivation",
        "",
        f"- Status: `{summary['status']}`",
        f"- Outcome: `{outcome_display}`",
        f"- Preregistration SHA-256: `{PREREG_SHA256}`",
        f"- Stage 0: `{(readout.get('equivalence') or {}).get('status', 'not-run')}`",
        "",
    ]
    f27 = readout.get("f27")
    if f27:
        lines.extend(
            [
                "## F@27 readout",
                "",
                f"- Aggregate cbar: `{f27['cbar_aggregate']:.15f}`",
                f"- Distance to phi_plus_one: `{f27['phi_plus_one_distance']:.15g}`",
                f"- Maximum segment distance: `{f27['maximum_segment_distance']:.15g}`",
                f"- Aggregate edges: `{f27['aggregate_total_edges']:,}`",
                f"- Aggregate branching nodes: `{f27['aggregate_branching_nodes']:,}`",
                "",
                "| Segment | Nodes | Edges | Branching nodes | cbar | Distance | Max stack |",
                "|---:|---:|---:|---:|---:|---:|---:|",
                *[
                    f"| {row['segment_index']} | {row['total_nodes']:,} | "
                    f"{row['total_edges']:,} | {row['total_branching_nodes']:,} | "
                    f"{row['cbar']:.15f} | {row['phi_plus_one_distance']:.6g} | "
                    f"{row['max_stack_depth']:,} |"
                    for row in f27["segments"]
                ],
                "",
                "This is an exact streaming re-derivation of the canonical mean-children observable, not an independent metric.",
            ]
        )
    runtime = summary["runtime"]
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- Elapsed: `{runtime['elapsed_seconds']:.3f} s`",
            f"- Peak resident memory: `{runtime['peak_rss_bytes']:,} bytes`",
        ]
    )
    scan_runtime = runtime.get("target_scan")
    if scan_runtime:
        lines.extend(
            [
                f"- Packed bytes read in this process: `{scan_runtime['uncompressed_bytes_read_this_process']:,}`",
                f"- Structural characters traversed: `{scan_runtime['structural_chars_traversed']:,}`",
                f"- Resumed from packed byte: `{scan_runtime['resumed_from_packed_byte']:,}`",
            ]
        )
    if summary["runtime"].get("error"):
        lines.extend(["", "## Stop reason", "", f"`{summary['runtime']['error']}`"])
    lines.extend(
        [
            "",
            "No figure or manuscript edit is authorized by this run.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def verify_file(path: Path, expected_sha: str, expected_bytes: int | None, log_path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise RuntimeError(f"{label} size mismatch: {size:,} != {expected_bytes:,}")
    log(log_path, f"[hash] {label}: {path} ({size:,} bytes)")
    observed = sha256_file(path, log_path=log_path, label=label)
    if observed != expected_sha:
        raise RuntimeError(f"{label} SHA-256 mismatch: {observed}")


def sha256_file(path: Path, *, log_path: Path | None = None, label: str = "file") -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    consumed = 0
    last = time.monotonic()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            consumed += len(block)
            now = time.monotonic()
            if log_path is not None and now - last >= 30.0:
                last = now
                log(log_path, f"[hash] {label}: {100.0 * consumed / total:.1f}%")
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def log(path: Path, message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def peak_rss_bytes() -> int:
    try:
        import psutil

        info = psutil.Process().memory_info()
        return int(max(info.rss, getattr(info, "peak_wset", 0)))
    except ImportError:
        return 0


def normalize_sha(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    if len(normalized) != 64 or any(char not in "0123456789ABCDEF" for char in normalized):
        raise SystemExit("--code-audit-sha256 must be a 64-character hexadecimal SHA-256")
    return normalized


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"


def format_bytes(value: int) -> str:
    if value >= 1024**3:
        return f"{value / 1024**3:.2f} GiB"
    if value >= 1024**2:
        return f"{value / 1024**2:.2f} MiB"
    return f"{value:,} B"


def configure_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


if __name__ == "__main__":
    raise SystemExit(main())
