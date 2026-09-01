#!/usr/bin/env python3
"""D-0154 B-only calibration for the reconstructed legacy DET protocol."""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = REPO_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from level1_deep_analysis import recurrence_analysis  # noqa: E402
from utils.streaming_phi_loader import StreamingPhiLoader  # noqa: E402

PREREG_PATH = WORKSPACE_DIR / "prereg_D0154_DET_B_only_calibration_v1.0.md"
PREREG_OTS_PATH = PREREG_PATH.with_name(PREREG_PATH.name + ".ots")
PREREG_SHA256 = "B51F2AF171357ED53CF99E2769ED5A9A079A9B0BBC8010B35CCFF73A7F0B97D7"
DECISION_LOG_PATH = REPO_DIR / "Documentation" / "HSI_v2_Decision_Log.md"
CODE_AUDIT_PATH = WORKSPACE_DIR / "docs" / "HSI_v2_Audit_Code_D0154_DET_B_only_Calibration.md"

SOURCE_RELATIVE = Path("level0/phi_snapshots/var_B/phi_iter23.struct.gz")
METADATA_RELATIVE = Path("level0/phi_snapshots/var_B/phi_iter23.json")
HISTORICAL_RELATIVE = Path("level1/analysis/deep_analysis_var_B_iter23_multiscale.json")
OUTPUT_RELATIVE = Path("hsi_v2/d0154_det_b_calibration")

SOURCE_SHA256 = "9BCBCDBB7C3E5D9CAEBB1D4C47403F36FB08DCC5DFFCB0716163BAD9705BD1B7"
SOURCE_BYTES = 5_264_486_973
METADATA_SHA256 = "1C76FE8DEE4B0DCF0B5B371746136CFA15ADC583FF54717EE6630DD875646C21"
HISTORICAL_SHA256 = "27D26B00866A72E8B5F16A2B8326EDDB8256A9FBCAFC3C2D3953D4D59C9D8D46"

STRUCTURAL_LENGTH = 252_105_209_576
OBSERVABLE_COUNT = 246_162_834_941
MEDIUM_DIVISOR = 1_000_000
MEDIUM_MIN_STEP = 100
MEDIUM_STEP = 252_105
MEDIUM_COUNT = 976_430
DET_BITS = 10_000
REPRESENTATIVE_STEP = 97
OBSERVABLE_INTERVAL = 24_454_185
THRESHOLD_SEED = 42
PAPER_DISPLAY = "0.55"

MEDIUM_SAMPLER_SHA256 = "505E4D4D97D9050C7F847A1A84DE97C457F3597603E3DC2DE60254D9EC2A2D62"
RQA_SHA256 = "1DD6DFC4938F68DDAB4A74B5DA67695B3F58D72F53DCAA92AF1CECBB0EB25B16"
LOADER_SHA256 = "44E5F208755203F3214BEAEB4D88472F3A0468D67726203CC1826BC32E5FE2AE"


class OperationalAbort(RuntimeError):
    """A frozen input or execution gate failed before scientific readout."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="D-0154 B-only calibration of the reconstructed legacy DET protocol."
    )
    parser.add_argument("--results-base", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-run", default=None)
    parser.add_argument("--chunk-bytes", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-every-chunks", type=int, default=20)
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument(
        "--code-audit-sha256",
        default=None,
        help="Grace code-audit SHA-256; required with --execute and verified against docs/ and the Decision Log.",
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    configure_stdout()
    load_dotenv()
    args = build_parser().parse_args()
    validate_cli(args)

    results_base = resolve_results_base(args.results_base)
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (results_base / OUTPUT_RELATIVE).resolve()
    )
    plan = build_plan(args, results_base, output_root)

    if not args.execute:
        print_plan(plan)
        return 0

    code_hashes = current_code_hashes()
    frozen = validate_frozen_contract_and_audit(
        str(args.code_audit_sha256), code_hashes["runner"]
    )
    validate_dependency_hashes(code_hashes)
    source_identity = validate_source_identity(results_base)
    execution_identity = {
        "schema_version": 1,
        "preregistration_sha256": PREREG_SHA256,
        "code_audit_sha256": frozen["code_audit_sha256"],
        "source": source_identity,
        "protocol": plan["protocol"],
        "code_hashes": code_hashes,
    }

    run_dir, generated_at, resume_count = prepare_run_directory(
        args, output_root, execution_identity
    )
    log_path = run_dir / "run.log"
    manifest_path = run_dir / "manifest.json"
    checkpoint_path = run_dir / "checkpoint.json"
    started = time.time()
    peak_rss = current_rss_bytes()

    manifest = {
        **plan,
        "status": "running",
        "generated_at": generated_at,
        "run_dir": str(run_dir),
        "resume_count": resume_count,
        "frozen_gates": frozen,
        "execution_identity": execution_identity,
        "outputs": {},
    }
    write_json(manifest_path, manifest)
    if resume_count:
        log(log_path, f"[resume] Exact sampling resume #{resume_count}: {run_dir}")
    else:
        write_log_header(log_path, manifest)

    summary: dict[str, Any] = {
        "status": "running",
        "scientific_outcome": None,
        "generated_at": generated_at,
        "run_dir": str(run_dir),
        "paper_display": PAPER_DISPLAY,
        "protocol": plan["protocol"],
        "source": source_identity,
    }

    try:
        log(log_path, "[phase] Full B@23 observable scan and frozen medium sampling")
        scan, peak_rss = scan_medium_samples(
            Path(source_identity["struct_path"]),
            execution_identity,
            checkpoint_path,
            args.chunk_bytes,
            args.checkpoint_every_chunks,
            args.progress_seconds,
            log_path,
            peak_rss,
        )
        enforce_scan_gates(scan)

        representative = build_representative(scan["medium_samples"])
        representative_path = run_dir / "representative_bits.bin"
        atomic_write_bytes(representative_path, bytes(representative).translate(
            bytes.maketrans(b"\x00\x01", b"01")
        ))
        sample_sha256 = sha256_file(representative_path)
        log(log_path, f"[gate] Representative sample: {len(representative):,} bits; SHA-256 {sample_sha256}")

        peak_rss = max(peak_rss, current_rss_bytes())
        log(log_path, "[phase] Single frozen RQA call (seed=42)")
        np.random.seed(THRESHOLD_SEED)
        bits = representative_path.read_text(encoding="ascii")
        det_result = recurrence_analysis(bits, max_bits=len(bits), verbose=False)
        peak_rss = max(peak_rss, current_rss_bytes())

        determinism = float(det_result["determinism"])
        calibration_display = f"{determinism:.2f}"
        outcome = classify_outcome(calibration_display)
        finished_at = datetime.now().isoformat(timespec="seconds")
        elapsed = round(time.time() - started, 3)

        details = {
            "status": "completed",
            "variant": "B",
            "iteration": 23,
            "paper_display": PAPER_DISPLAY,
            "calibration_display": calibration_display,
            "determinism_full_precision": determinism,
            "scientific_outcome": outcome,
            "sampling": {
                key: value for key, value in scan.items() if key != "medium_samples"
            },
            "representative_sample": {
                "encoding": "ASCII bytes '0'/'1', one byte per observable bit",
                "bits": len(representative),
                "sha256": sample_sha256,
            },
            "rqa": json_safe(det_result),
            "threshold_seed": THRESHOLD_SEED,
            "elapsed_seconds": elapsed,
            "peak_rss_bytes": peak_rss,
        }
        write_json(run_dir / "b_det_details.json", details)

        summary.update(
            {
                "status": "completed",
                "scientific_outcome": outcome,
                "finished_at": finished_at,
                "determinism_full_precision": determinism,
                "calibration_display": calibration_display,
                "display_reproduces": calibration_display == PAPER_DISPLAY,
                "medium_step": scan["medium_step"],
                "medium_sample_count": scan["medium_sample_count"],
                "representative_step": REPRESENTATIVE_STEP,
                "representative_sample_sha256": sample_sha256,
                "observable_bits_scanned": scan["observable_bits_scanned"],
                "elapsed_seconds": elapsed,
                "peak_rss_bytes": peak_rss,
            }
        )
        write_json(run_dir / "summary.json", summary)
        write_report(run_dir / "report.md", summary)
        log(log_path, f"[result] display={calibration_display} outcome={outcome}")
        log(log_path, "[complete] B-only DET calibration completed; post-execution audit required")

        if checkpoint_path.exists():
            checkpoint_path.unlink()
        manifest.update(
            {
                "status": "completed",
                "finished_at": finished_at,
                "scientific_outcome": outcome,
                "elapsed_seconds": elapsed,
                "peak_rss_bytes": peak_rss,
            }
        )
        manifest["outputs"] = hash_primary_outputs(run_dir)
        write_json(manifest_path, manifest)
        manifest_sha = sha256_file(manifest_path)
        atomic_write_text(run_dir / "manifest.sha256", f"{manifest_sha}  manifest.json\n")

        print("D-0154 DET B-only calibration completed.")
        print(f"Outcome: {outcome}")
        print(f"Output: {run_dir}")
        print(f"Manifest SHA-256: {manifest_sha}")
        return 0
    except OperationalAbort as exc:
        log(log_path, f"[abort] {exc}")
        finalize_abort(manifest_path, run_dir, manifest, summary, exc, started, peak_rss)
        return 2
    except (Exception, KeyboardInterrupt) as exc:
        log(log_path, f"[interrupted] {type(exc).__name__}: {exc}")
        manifest.update(
            {
                "status": "interrupted",
                "interrupted_at": datetime.now().isoformat(timespec="seconds"),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_json(manifest_path, manifest)
        raise


def validate_cli(args: argparse.Namespace) -> None:
    if args.resume_run and not args.execute:
        raise SystemExit("--resume-run requires --execute.")
    if args.execute and not args.code_audit_sha256:
        raise SystemExit("--execute requires --code-audit-sha256 after Grace approval and Decision Log anchoring.")
    if args.chunk_bytes <= 0:
        raise SystemExit("--chunk-bytes must be positive.")
    if args.checkpoint_every_chunks <= 0:
        raise SystemExit("--checkpoint-every-chunks must be positive.")
    if args.progress_seconds <= 0:
        raise SystemExit("--progress-seconds must be positive.")


def build_plan(args: argparse.Namespace, results_base: Path, output_root: Path) -> dict[str, Any]:
    return {
        "script": Path(__file__).name,
        "decision_log_context": "D-0154 Amendments 46-49 DET B-only calibration",
        "results_base": str(results_base),
        "output_root": str(output_root),
        "target": {"variant": "B", "iteration": 23},
        "protocol": {
            "sample_protocol": "full_medium_sample_then_dynamic_thinning",
            "structural_length": STRUCTURAL_LENGTH,
            "expected_observable_count": OBSERVABLE_COUNT,
            "medium_divisor": MEDIUM_DIVISOR,
            "medium_min_step": MEDIUM_MIN_STEP,
            "medium_step": MEDIUM_STEP,
            "expected_medium_count": MEDIUM_COUNT,
            "det_bits": DET_BITS,
            "representative_step": REPRESENTATIVE_STEP,
            "observable_interval": OBSERVABLE_INTERVAL,
            "threshold_seed": THRESHOLD_SEED,
            "paper_display": PAPER_DISPLAY,
            "display_operation": 'format(determinism, ".2f")',
            "chunk_bytes": args.chunk_bytes,
            "checkpoint_every_chunks": args.checkpoint_every_chunks,
            "progress_seconds": args.progress_seconds,
        },
        "non_observation_barrier": {
            "execute_flag_required": True,
            "code_audit_file_required": str(CODE_AUDIT_PATH),
            "code_audit_sha_must_be_in_decision_log": True,
            "preregistration_sha256": PREREG_SHA256,
        },
    }


def validate_frozen_contract_and_audit(
    code_audit_sha256: str, runner_sha256: str
) -> dict[str, Any]:
    normalized_audit = normalize_sha256(code_audit_sha256)
    if sha256_file(PREREG_PATH) != PREREG_SHA256:
        raise OperationalAbort("Frozen preregistration SHA-256 mismatch.")
    if not PREREG_OTS_PATH.is_file():
        raise OperationalAbort(f"Missing OTS sidecar: {PREREG_OTS_PATH}")
    if not CODE_AUDIT_PATH.is_file():
        raise OperationalAbort(f"Missing Grace code audit: {CODE_AUDIT_PATH}")
    if sha256_file(CODE_AUDIT_PATH) != normalized_audit:
        raise OperationalAbort("Grace code-audit SHA-256 mismatch.")
    audit_text = CODE_AUDIT_PATH.read_text(encoding="utf-8")
    if runner_sha256 not in audit_text:
        raise OperationalAbort("Grace code audit does not bind the executing runner SHA-256.")
    decision_text = DECISION_LOG_PATH.read_text(encoding="utf-8")
    if PREREG_SHA256 not in decision_text:
        raise OperationalAbort("Frozen preregistration SHA is not anchored in the Decision Log.")
    if normalized_audit not in decision_text:
        raise OperationalAbort("Grace code-audit SHA is not anchored in the Decision Log.")
    return {
        "preregistration_path": str(PREREG_PATH),
        "preregistration_sha256": PREREG_SHA256,
        "ots_sidecar_path": str(PREREG_OTS_PATH),
        "ots_sidecar_sha256": sha256_file(PREREG_OTS_PATH),
        "code_audit_path": str(CODE_AUDIT_PATH),
        "code_audit_sha256": normalized_audit,
        "decision_log_path": str(DECISION_LOG_PATH),
        "decision_log_sha256_at_execution": sha256_file(DECISION_LOG_PATH),
    }


def current_code_hashes() -> dict[str, str]:
    return {
        "runner": sha256_file(Path(__file__)),
        "medium_sampler": sha256_file(REPO_DIR / "utils" / "streaming_subsample.py"),
        "rqa_metric": sha256_file(REPO_DIR / "level1_deep_analysis.py"),
        "streaming_loader": sha256_file(REPO_DIR / "utils" / "streaming_phi_loader.py"),
    }


def validate_dependency_hashes(code_hashes: dict[str, str]) -> None:
    expected = {
        "medium_sampler": MEDIUM_SAMPLER_SHA256,
        "rqa_metric": RQA_SHA256,
        "streaming_loader": LOADER_SHA256,
    }
    for name, digest in expected.items():
        if code_hashes.get(name) != digest:
            raise OperationalAbort(f"Frozen dependency SHA-256 mismatch: {name}.")


def validate_source_identity(results_base: Path) -> dict[str, Any]:
    struct_path = (results_base / SOURCE_RELATIVE).resolve()
    metadata_path = (results_base / METADATA_RELATIVE).resolve()
    historical_path = (results_base / HISTORICAL_RELATIVE).resolve()
    for path in (struct_path, metadata_path, historical_path):
        if not path.is_file():
            raise OperationalAbort(f"Missing frozen source artifact: {path}")
    if struct_path.stat().st_size != SOURCE_BYTES:
        raise OperationalAbort("B@23 structural source byte-size mismatch.")
    if sha256_file(struct_path) != SOURCE_SHA256:
        raise OperationalAbort("B@23 structural source SHA-256 mismatch.")
    if sha256_file(metadata_path) != METADATA_SHA256:
        raise OperationalAbort("B@23 metadata SHA-256 mismatch.")
    if sha256_file(historical_path) != HISTORICAL_SHA256:
        raise OperationalAbort("Historical B@23 multiscale metadata SHA-256 mismatch.")
    metadata = read_json(metadata_path)
    expected_metadata = {
        "iteration": 23,
        "sequence_length": STRUCTURAL_LENGTH,
        "format": "v33_structural_streaming",
        "encoding": "2bit",
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise OperationalAbort(f"B@23 metadata field mismatch: {key}.")
    return {
        "variant": "B",
        "iteration": 23,
        "struct_path": str(struct_path),
        "struct_bytes": SOURCE_BYTES,
        "struct_sha256": SOURCE_SHA256,
        "metadata_path": str(metadata_path),
        "metadata_sha256": METADATA_SHA256,
        "historical_metadata_path": str(historical_path),
        "historical_metadata_sha256": HISTORICAL_SHA256,
        "sequence_length": STRUCTURAL_LENGTH,
        "format": "v33_structural_streaming",
        "encoding": "2bit",
    }


def prepare_run_directory(
    args: argparse.Namespace,
    output_root: Path,
    execution_identity: dict[str, Any],
) -> tuple[Path, str, int]:
    if args.resume_run:
        run_dir = Path(args.resume_run).expanduser().resolve()
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise OperationalAbort(f"Resume manifest not found: {manifest_path}")
        manifest = read_json(manifest_path)
        if manifest.get("status") == "completed":
            raise OperationalAbort("Completed calibration runs cannot be resumed.")
        if manifest.get("execution_identity") != execution_identity:
            raise OperationalAbort("Resume execution identity mismatch.")
        generated_at = str(manifest.get("generated_at", ""))
        if not generated_at:
            raise OperationalAbort("Resume manifest has no generated_at value.")
        return run_dir, generated_at, int(manifest.get("resume_count", 0)) + 1

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"d0154-det-b-calibration__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, datetime.now().isoformat(timespec="seconds"), 0


def scan_medium_samples(
    struct_path: Path,
    execution_identity: dict[str, Any],
    checkpoint_path: Path,
    chunk_bytes: int,
    checkpoint_every_chunks: int,
    progress_seconds: float,
    log_path: Path,
    peak_rss: int,
) -> tuple[dict[str, Any], int]:
    state = initial_scan_state(execution_identity)
    if checkpoint_path.is_file():
        state = load_checkpoint(checkpoint_path, execution_identity)
        log(
            log_path,
            f"[resume] Restored observable={state['observable_index']:,}, "
            f"medium={len(state['medium_samples']):,}; replaying gzip prefix",
        )

    resume_position = int(state["observable_index"])
    stream_cursor = 0
    processed_chunks = 0
    started = time.time()
    last_progress = started
    loader = StreamingPhiLoader(str(struct_path))

    try:
        for chunk in loader.iter_bits_numpy(chunk_bytes=chunk_bytes):
            chunk_start = stream_cursor
            chunk_end = chunk_start + len(chunk)
            stream_cursor = chunk_end

            if chunk_end <= resume_position:
                now = time.time()
                if now - last_progress >= progress_seconds:
                    replay_fraction = min(chunk_end / max(resume_position, 1), 1.0)
                    replay_rate = chunk_end / max(now - started, 1e-9)
                    replay_eta = (resume_position - chunk_end) / replay_rate if replay_rate > 0 else math.inf
                    log(
                        log_path,
                        f"[resume-replay] {replay_fraction:7.3%} | "
                        f"observable={chunk_end:,}/{resume_position:,} | "
                        f"rate={replay_rate / 1e6:.2f} Mobs/s | eta={format_duration(replay_eta)}",
                    )
                    last_progress = now
                continue
            if chunk_start < resume_position:
                chunk = chunk[resume_position - chunk_start :]
                chunk_start = resume_position

            if chunk_start != int(state["observable_index"]):
                raise OperationalAbort("Resume replay did not reach the exact observable boundary.")
            capture_observable_chunk(state, chunk, chunk_start, MEDIUM_STEP)
            processed_chunks += 1
            peak_rss = max(peak_rss, current_rss_bytes())

            now = time.time()
            if now - last_progress >= progress_seconds:
                elapsed = max(now - started, 1e-9)
                position = int(state["observable_index"])
                fraction = min(position / OBSERVABLE_COUNT, 1.0)
                rate = max(position - resume_position, 0) / elapsed
                eta = (OBSERVABLE_COUNT - position) / rate if rate > 0 else math.inf
                log(
                    log_path,
                    f"[progress] {fraction:7.3%} | observable={position:,}/{OBSERVABLE_COUNT:,} | "
                    f"medium={len(state['medium_samples']):,} | rate={rate / 1e6:.2f} Mobs/s | "
                    f"eta={format_duration(eta)}",
                )
                last_progress = now

            if processed_chunks % checkpoint_every_chunks == 0:
                write_checkpoint(checkpoint_path, state)
    except BaseException:
        write_checkpoint(checkpoint_path, state)
        raise

    write_checkpoint(checkpoint_path, state)
    return {
        "medium_step": MEDIUM_STEP,
        "medium_sample_count": len(state["medium_samples"]),
        "observable_bits_scanned": int(state["observable_index"]),
        "next_capture": int(state["next_capture"]),
        "chunks_completed": int(state["chunks_completed"]),
        "medium_samples": state["medium_samples"],
        "scan_elapsed_seconds": round(time.time() - started, 3),
    }, peak_rss


def initial_scan_state(execution_identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "execution_identity": execution_identity,
        "observable_index": 0,
        "next_capture": 0,
        "chunks_completed": 0,
        "medium_samples": bytearray(),
    }


def capture_observable_chunk(
    state: dict[str, Any],
    chunk: np.ndarray,
    chunk_start: int,
    medium_step: int,
) -> None:
    chunk_end = chunk_start + len(chunk)
    next_capture = int(state["next_capture"])
    if next_capture < chunk_start:
        missed = (chunk_start - next_capture + medium_step - 1) // medium_step
        next_capture += missed * medium_step
    if next_capture < chunk_end:
        start_offset = next_capture - chunk_start
        indices = np.arange(start_offset, len(chunk), medium_step, dtype=np.int64)
        if len(indices):
            state["medium_samples"].extend(chunk[indices].tobytes())
            next_capture = chunk_start + int(indices[-1]) + medium_step
    state["next_capture"] = next_capture
    state["observable_index"] = chunk_end
    state["chunks_completed"] = int(state["chunks_completed"]) + 1


def write_checkpoint(path: Path, state: dict[str, Any]) -> None:
    payload = {
        key: value for key, value in state.items() if key != "medium_samples"
    }
    payload["medium_samples_base64"] = base64.b64encode(
        bytes(state["medium_samples"])
    ).decode("ascii")
    payload["medium_samples_sha256"] = hashlib.sha256(
        bytes(state["medium_samples"])
    ).hexdigest().upper()
    write_json(path, payload)


def load_checkpoint(path: Path, execution_identity: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != 1:
        raise OperationalAbort("Unsupported checkpoint schema.")
    if payload.get("execution_identity") != execution_identity:
        raise OperationalAbort("Checkpoint execution identity mismatch.")
    try:
        medium_samples = bytearray(base64.b64decode(payload["medium_samples_base64"], validate=True))
    except Exception as exc:
        raise OperationalAbort("Checkpoint medium sample is not valid base64.") from exc
    digest = hashlib.sha256(bytes(medium_samples)).hexdigest().upper()
    if digest != payload.get("medium_samples_sha256"):
        raise OperationalAbort("Checkpoint medium-sample SHA-256 mismatch.")
    observable_index = int(payload["observable_index"])
    next_capture = int(payload["next_capture"])
    expected_count = math.ceil(observable_index / MEDIUM_STEP) if observable_index else 0
    if len(medium_samples) != expected_count:
        raise OperationalAbort("Checkpoint medium-sample count is inconsistent with its offset.")
    if next_capture != len(medium_samples) * MEDIUM_STEP:
        raise OperationalAbort("Checkpoint next_capture is inconsistent with its sample count.")
    return {
        "schema_version": 1,
        "execution_identity": execution_identity,
        "observable_index": observable_index,
        "next_capture": next_capture,
        "chunks_completed": int(payload["chunks_completed"]),
        "medium_samples": medium_samples,
    }


def enforce_scan_gates(scan: dict[str, Any]) -> None:
    if scan["observable_bits_scanned"] != OBSERVABLE_COUNT:
        raise OperationalAbort(
            f"Observable count mismatch: {scan['observable_bits_scanned']:,} != {OBSERVABLE_COUNT:,}."
        )
    if scan["medium_step"] != MEDIUM_STEP:
        raise OperationalAbort("Frozen medium_step mismatch.")
    if scan["medium_sample_count"] != MEDIUM_COUNT:
        raise OperationalAbort(
            f"Medium count mismatch: {scan['medium_sample_count']:,} != {MEDIUM_COUNT:,}."
        )
    expected_next = MEDIUM_COUNT * MEDIUM_STEP
    if scan["next_capture"] != expected_next:
        raise OperationalAbort("Final next_capture mismatch.")


def build_representative(medium_samples: bytearray) -> bytearray:
    representative_step = max(1, len(medium_samples) // DET_BITS)
    if representative_step != REPRESENTATIVE_STEP:
        raise OperationalAbort("Frozen representative_step mismatch.")
    representative = medium_samples[::representative_step][:DET_BITS]
    if len(representative) != DET_BITS:
        raise OperationalAbort(
            f"Representative sample underfilled: {len(representative):,} != {DET_BITS:,}."
        )
    if any(bit not in (0, 1) for bit in representative):
        raise OperationalAbort("Representative sample contains non-binary values.")
    return representative


def classify_outcome(calibration_display: str) -> str:
    return (
        "b-calibration-reproduces"
        if calibration_display == PAPER_DISPLAY
        else "b-calibration-fails"
    )


def finalize_abort(
    manifest_path: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    exc: Exception,
    started: float,
    peak_rss: int,
) -> None:
    finished_at = datetime.now().isoformat(timespec="seconds")
    elapsed = round(time.time() - started, 3)
    summary.update(
        {
            "status": "inputs-or-run-not-derivable",
            "scientific_outcome": None,
            "finished_at": finished_at,
            "abort_reason": str(exc),
            "elapsed_seconds": elapsed,
            "peak_rss_bytes": peak_rss,
        }
    )
    write_json(run_dir / "summary.json", summary)
    write_report(run_dir / "report.md", summary)
    manifest.update(
        {
            "status": "inputs-or-run-not-derivable",
            "scientific_outcome": None,
            "finished_at": finished_at,
            "abort_reason": str(exc),
            "elapsed_seconds": elapsed,
            "peak_rss_bytes": peak_rss,
        }
    )
    manifest["outputs"] = hash_primary_outputs(run_dir)
    write_json(manifest_path, manifest)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    status = str(summary["status"])
    lines = [
        "# D-0154 DET B-Only Calibration",
        "",
        f"- Status: `{status}`",
        f"- Scientific outcome: `{summary.get('scientific_outcome')}`",
        f"- Paper display: `{summary.get('paper_display')}`",
        f"- Run directory: `{summary.get('run_dir')}`",
        "",
        "## Frozen Protocol",
        "",
        "- Target: clean unaffected `B@23`.",
        f"- Medium sampling: step `{MEDIUM_STEP:,}`, expected count `{MEDIUM_COUNT:,}`.",
        f"- Representative sample: `medium[::{REPRESENTATIVE_STEP}][:{DET_BITS}]`.",
        f"- RQA threshold seed: `{THRESHOLD_SEED}`.",
        f"- Decisive criterion: `format(determinism, \".2f\") == \"{PAPER_DISPLAY}\"`.",
        "- No prefix control or alternate depth is part of this contract.",
        "",
        "## Readout",
        "",
    ]
    if status == "completed":
        lines.extend(
            [
                f"- Full-precision DET: `{summary['determinism_full_precision']!r}`",
                f"- Calibration display: `{summary['calibration_display']}`",
                f"- Observable bits scanned: `{summary['observable_bits_scanned']:,}`",
                f"- Representative sample SHA-256: `{summary['representative_sample_sha256']}`",
                "",
                "The Amendment 46 interpretation matrix governs this outcome. Post-execution Grace audit is mandatory before any paper-facing use.",
            ]
        )
    else:
        lines.extend(
            [
                f"- Abort reason: `{summary.get('abort_reason')}`",
                "",
                "No scientific outcome is assigned. The five affected DET cells remain blocked.",
            ]
        )
    atomic_write_text(path, "\n".join(lines) + "\n")


def hash_primary_outputs(run_dir: Path) -> dict[str, str]:
    names = (
        "summary.json",
        "report.md",
        "b_det_details.json",
        "representative_bits.bin",
        "run.log",
    )
    return {name: sha256_file(run_dir / name) for name in names if (run_dir / name).is_file()}


def normalize_sha256(value: str) -> str:
    normalized = value.strip().upper()
    if len(normalized) != 64 or any(char not in "0123456789ABCDEF" for char in normalized):
        raise OperationalAbort("Code-audit SHA-256 must be 64 hexadecimal characters.")
    return normalized


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def load_dotenv() -> None:
    env_path = REPO_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_results_base(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    env_value = os.environ.get("HSI_RESULTS_BASE_DIR") or os.environ.get("HSI_V1_RESULTS_BASE_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (REPO_DIR / "results").resolve()


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(json_safe(payload), indent=2) + "\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temp_path, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def log(path: Path, message: str) -> None:
    print(message, flush=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def write_log_header(path: Path, manifest: dict[str, Any]) -> None:
    atomic_write_text(path, "# D-0154 DET B-only calibration\n\n" + json.dumps(json_safe(manifest), indent=2) + "\n\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def current_rss_bytes() -> int:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
    return 0


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m {secs:02d}s"


def print_plan(plan: dict[str, Any]) -> None:
    print("D-0154 DET B-only calibration plan (no target access)")
    print("-" * 76)
    print(f"target: {plan['target']}")
    print(f"results_base: {plan['results_base']}")
    print(f"output_root: {plan['output_root']}")
    print(f"preregistration_sha256: {PREREG_SHA256}")
    print("Execution remains blocked until --execute and an anchored Grace code-audit SHA are supplied.")


if __name__ == "__main__":
    raise SystemExit(main())
