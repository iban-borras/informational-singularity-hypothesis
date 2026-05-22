#!/usr/bin/env python3
"""D-0144 Bxor counterfactual execution.

This script deliberately does not touch the canonical Level 0 generator.  It
implements the frozen Q2b/B' contract as an isolated public runner:

1. Generate a separate structural snapshot for variant BXOR.
2. Run the canonical Phase 1 tower on that snapshot.
3. Derive a compact readout against existing B/F/M/N/O/P and B-null artifacts.

If the frozen counterfactual cannot produce the required iteration-20 snapshot
under the declared rule, the run emits a controlled
``not-executable-under-v0.1`` verdict instead of tuning the rule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from level0.accumulation_manager import AccumulationManager
from level0.hybrid_collapse_engine import HybridCollapseEngine
from level0.phi_snapshot_manager import PhiSnapshotManager
from v2.common.io import load_observable_prefix_bits, load_struct_metadata
from v2.phase1.report import (
    build_report_rows,
    discover_phase1_runs,
    filter_runs,
    select_latest_per_variant,
)
from v2.phase1.transport import build_transport_rows

import hsi_v2_raw_sequence_orthogonal_metrics as raw_metrics


CONTRACT_ID = "D-0144"
PROPOSAL_SHA256 = "1514946F394F75BF32574B177F80B71F2C123E4FCB340DF072BFA21D53496615"
Q1_SUMMARY_SHA256 = "32C3F58631CE9BEEADC6C55E28AFB35FC6F292961AA54B02ECABEF044C856A76"
Q2A_NOTE_SHA256 = "1BABC8BC3B4C54FCFDCE3F61D9520188CCC1E468DEB569464C08ADF2D60BEF4F"
VARIANT = "BXOR"
REFERENCE_VARIANTS = ("B", "F", "M", "N", "O", "P")
REFERENCE_ROOTS = (
    "results/hsi_v2/phase1",
    "results/hsi_v2/phase1_external_recursive_controls",
)
REFERENCE_NULL_ROOTS = (
    "results/hsi_v2/phase1/nulls",
    "results/hsi_v2/phase1_sensitivity_thresholds/phase1-threshold-sensitivity__oat-16__seed-607__20260508T121536/children/baseline/B__matched-lz__seed-607/nulls",
    "results/hsi_v2/phase1_sensitivity_thresholds/phase1-threshold-sensitivity__oat-16__seed-607__20260508T121536/children/baseline/B__phase-matched-lz__seed-607/nulls",
)


class NotExecutable(Exception):
    """Controlled contract stop."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the D-0144 Bxor counterfactual.")
    parser.add_argument("--output-dir", default="results/hsi_v2/q2b_bprime_counterfactual")
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--segment-bits", type=int, default=1_000_000)
    parser.add_argument("--num-segments", type=int, default=3)
    parser.add_argument("--scales", default="8,12,16,20,24,28,32")
    parser.add_argument("--policies", default="prefix,suffix")
    parser.add_argument("--sample-bits", type=int, default=3_000_000)
    parser.add_argument("--max-ram-gb", type=float, default=32.0)
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing run directory if needed.")
    parser.add_argument("--skip-phase1", action="store_true", help="Generate snapshot only.")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    started = time.perf_counter()

    output_root = resolve_project_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = f"q2b-bxor-counterfactual__iter-{args.max_iterations}__seg-{args.num_segments}x{compact_int(args.segment_bits)}__{timestamp}"
    run_dir = output_root / run_slug
    if run_dir.exists() and args.force:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    summary: dict[str, Any] = {
        "generated_at": now_utc(),
        "script": Path(__file__).name,
        "contract": {
            "decision_log_id": CONTRACT_ID,
            "proposal_sha256": PROPOSAL_SHA256,
            "q1_summary_sha256": Q1_SUMMARY_SHA256,
            "q2a_note_sha256": Q2A_NOTE_SHA256,
            "counterfactual": True,
            "variant": VARIANT,
            "canonical_level0_modified": False,
            "local_rule": "Each collapsed bit block maps to sum(bits) mod 2.",
            "no_tuning_after_signature": True,
        },
        "parameters": vars(args),
        "status": "running",
    }

    preserve_temp_on_error = True
    try:
        phase_print("Generating isolated Bxor snapshot", f"run={run_slug}", quiet=args.quiet)
        snapshot_info = generate_bxor_snapshot(run_dir, args)
        summary["snapshot"] = snapshot_info

        if args.skip_phase1:
            summary["status"] = "snapshot-only"
            summary["outcome"] = "not-evaluated"
        else:
            phase_print("Running canonical Phase 1 on Bxor", snapshot_info["struct_path"], quiet=args.quiet)
            phase1_info = run_phase1(snapshot_info, run_dir, args)
            summary["phase1"] = phase1_info

            phase_print("Building Q2b readout", "Phase 1 + raw orthogonal metrics", quiet=args.quiet)
            readout = build_readout(snapshot_info, phase1_info, run_dir, args)
            summary["readout"] = readout
            summary["status"] = "completed"
            summary["outcome"] = classify_outcome(readout)
        preserve_temp_on_error = False

    except NotExecutable as exc:
        summary["status"] = "controlled-stop"
        summary["outcome"] = "not-executable-under-v0.1"
        summary["controlled_stop_reason"] = str(exc)
        phase_print("Controlled stop", str(exc), quiet=args.quiet)
        preserve_temp_on_error = False
    except Exception:
        write_failure_artifacts(run_dir, summary, started)
        raise
    finally:
        if not args.keep_temp and not preserve_temp_on_error:
            cleanup_temp(run_dir)

    write_artifacts(run_dir, summary, started)
    if not args.quiet:
        print("")
        print(render_console_summary(summary))
        print("")
        print(f"Saved summary to: {run_dir / 'summary.json'}")
        print(f"Saved report to: {run_dir / 'report.md'}")
        print(f"Saved manifest to: {run_dir / 'manifest.json'}")
    return 0


def generate_bxor_snapshot(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    temp_dir = run_dir / "temp_generation"
    snapshot_dir = run_dir / "level0" / "phi_snapshots" / f"var_{VARIANT}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    max_ram_bytes = int(args.max_ram_gb * 1_000_000_000)
    engine = HybridCollapseEngine(
        max_ram_bytes=max_ram_bytes,
        simplify_fn=simplify_bxor,
        compress=False,
        compress_level=args.compression_level,
    )
    accumulator = AccumulationManager(
        output_dir=str(temp_dir),
        variant=VARIANT,
        compress=False,
        compress_level=args.compression_level,
    )

    current_state = "0"
    iteration_rows: list[dict[str, Any]] = []
    try:
        for iteration_index in range(args.max_iterations):
            iteration = iteration_index + 1
            accumulator.append(current_state)
            frame_path = accumulator.build_decay_frame("1", compress_output=False, prefix_mode=False)
            frame_size = frame_path.stat().st_size
            if frame_size > max_ram_bytes:
                raise NotExecutable(
                    f"iteration {iteration} decay frame is {frame_size:,} bytes, above "
                    f"--max-ram-gb {args.max_ram_gb:g}; refusing unsafe chunked collapse under D-0144."
                )

            print(
                f"[BXOR] iteration {iteration:02d}/{args.max_iterations}: "
                f"acc={accumulator.get_length():,} chars | frame={frame_size:,} bytes",
                flush=True,
            )
            state, pass_rows = collapse_frame_bxor(
                frame_path=frame_path,
                accumulator=accumulator,
                engine=engine,
                temp_dir=temp_dir,
                iteration=iteration,
            )
            if len(state) > 1:
                state = simplify_bxor(state)
            current_state = state
            iteration_rows.append(
                {
                    "iteration": iteration,
                    "final_state": current_state,
                    "accumulation_chars": accumulator.get_length(),
                    "observable_bits": accumulator.get_clean_bits_count(),
                    "decay_frame_bytes": frame_size,
                    "collapse_passes": len(pass_rows),
                    "passes": pass_rows,
                }
            )

        accumulator.append(current_state)
        source_file = accumulator.file_path
        observable_bits = accumulator.get_clean_bits_count()
        if observable_bits < args.segment_bits * args.num_segments:
            raise NotExecutable(
                f"BXOR produced {observable_bits:,} observable bits; "
                f"need {args.segment_bits * args.num_segments:,}."
            )

        metadata = {
            "variant": VARIANT,
            "variant_code": VARIANT,
            "counterfactual": True,
            "contract": CONTRACT_ID,
            "proposal_sha256": PROPOSAL_SHA256,
            "q2a_note_sha256": Q2A_NOTE_SHA256,
            "canonical_level0_modified": False,
            "local_rule": "parity XOR: collapsed bit block -> sum(bits) mod 2",
            "derivation_strategy": "same B outer accumulation and inside-out collapse; only local simplifier changed",
            "absolute_token": "1",
            "source_temp_accumulation": str(source_file),
            "observable_bits": observable_bits,
            "command": [sys.executable, *sys.argv],
        }
        manager = PhiSnapshotManager(
            data_dir=str(snapshot_dir),
            memory_threshold=1,
            compression_level=args.compression_level,
        )
        save_info = manager.save_phi_state_structural_from_file(
            str(source_file),
            args.max_iterations,
            metadata=metadata,
        )
        struct_path = snapshot_dir / f"phi_iter{args.max_iterations}.struct.gz"
        metadata_path = snapshot_dir / f"phi_iter{args.max_iterations}.json"
        struct_sha = sha256_file(struct_path)
        metadata_payload = load_json(metadata_path)
        metadata_payload.update(
            {
                "sha256_struct_gz": struct_sha,
                "observable_bits": observable_bits,
                "iteration_stats": iteration_rows,
            }
        )
        write_json(metadata_path, metadata_payload)

        return {
            "status": "completed",
            "struct_path": str(struct_path),
            "metadata_path": str(metadata_path),
            "struct_sha256": struct_sha,
            "metadata_sha256": sha256_file(metadata_path),
            "observable_bits": observable_bits,
            "sequence_length": save_info.get("sequence_length"),
            "compressed_size_bytes": save_info.get("compressed_size_bytes"),
            "compression_ratio": save_info.get("compression_ratio"),
            "iteration_stats": iteration_rows,
        }
    finally:
        accumulator._flush()


def collapse_frame_bxor(
    *,
    frame_path: Path,
    accumulator: AccumulationManager,
    engine: HybridCollapseEngine,
    temp_dir: Path,
    iteration: int,
) -> tuple[str, list[dict[str, Any]]]:
    current_file = frame_path
    pass_rows: list[dict[str, Any]] = []
    pass_num = 0
    while True:
        pass_num += 1
        input_size = current_file.stat().st_size
        accumulator.append_from_file(current_file)
        next_file = temp_dir / f"bxor_i{iteration:02d}_p{pass_num:03d}.tmp"
        output_size, had_changes = engine.collapse_one_pass(
            current_file,
            next_file,
            log_progress=(pass_num == 1),
        )
        pass_rows.append(
            {
                "pass": pass_num,
                "input_bytes": input_size,
                "output_bytes": output_size,
                "had_changes": bool(had_changes),
            }
        )
        print(
            f"   [BXOR pass {pass_num:02d}] {input_size:,} -> {output_size:,} bytes"
            f" | changed={had_changes}",
            flush=True,
        )
        if current_file != frame_path and current_file.exists():
            current_file.unlink()
        if not had_changes or output_size <= 1:
            state = next_file.read_text(encoding="utf-8")
            next_file.unlink(missing_ok=True)
            frame_path.unlink(missing_ok=True)
            return state, pass_rows
        current_file = next_file


def simplify_bxor(seq: str) -> str:
    parity = 0
    saw_bit = False
    for char in seq:
        if char == "1":
            parity ^= 1
            saw_bit = True
        elif char == "0":
            saw_bit = True
    if not saw_bit:
        return ""
    return "1" if parity else "0"


def run_phase1(snapshot_info: dict[str, Any], run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    phase1_root = run_dir / "phase1"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "hsi_v2_phase1_run.py"),
        "--input",
        snapshot_info["struct_path"],
        "--variant",
        VARIANT,
        "--iteration",
        str(args.max_iterations),
        "--segment-bits",
        str(args.segment_bits),
        "--num-segments",
        str(args.num_segments),
        "--scales",
        args.scales,
        "--policies",
        args.policies,
        "--output-dir",
        str(phase1_root),
    ]
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    runs = discover_phase1_runs(phase1_root)
    selected = select_latest_per_variant(filter_runs(runs, variants=[VARIANT]), [VARIANT])
    if not selected:
        raise RuntimeError("Phase 1 completed but no BXOR run was discovered.")
    run = selected[0]
    return {
        "command": command,
        "run_dir": run["_run_dir"],
        "dataset_path": run["_dataset_path"],
        "summary_path": run["_summary_path"],
        "dataset_sha256": sha256_file(Path(run["_dataset_path"])),
        "summary_sha256": sha256_file(Path(run["_summary_path"])),
    }


def build_readout(
    snapshot_info: dict[str, Any],
    phase1_info: dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    bxor_run = discover_phase1_runs(Path(phase1_info["run_dir"]).parent)
    bxor_selected = select_latest_per_variant(filter_runs(bxor_run, variants=[VARIANT]), [VARIANT])
    if not bxor_selected:
        raise RuntimeError("Cannot rediscover BXOR Phase 1 run for readout.")

    reference_runs = load_reference_runs()
    all_runs = reference_runs + bxor_selected
    phase1_rows = build_report_rows(all_runs, include_hints=True)
    transport_rows = build_transport_rows(all_runs, tail_edges=2, include_hints=True)
    raw_readout = compute_bxor_raw_metrics(snapshot_info, run_dir, args)

    bxor_phase1 = select_row(phase1_rows, VARIANT)
    bxor_transport = select_row(transport_rows, VARIANT)
    comparison_rows = [
        row
        for row in phase1_rows
        if row["variant"] in set(REFERENCE_VARIANTS) | {VARIANT} or "matched-lz" in row["variant"]
    ]
    transport_comparison_rows = [
        row
        for row in transport_rows
        if row["variant"] in set(REFERENCE_VARIANTS) | {VARIANT} or "matched-lz" in row["variant"]
    ]

    phase1_csv = run_dir / "phase1_comparison.csv"
    transport_csv = run_dir / "transport_comparison.csv"
    raw_csv = run_dir / "raw_metrics_bxor.csv"
    write_csv(phase1_csv, comparison_rows)
    write_csv(transport_csv, transport_comparison_rows)
    write_csv(raw_csv, raw_readout["rows"])

    return {
        "phase1_bxor": bxor_phase1,
        "transport_bxor": bxor_transport,
        "raw_bxor": raw_readout["bxor_row"],
        "lz_phi_fingerprint_bxor": raw_readout["lz_phi_fingerprint_bxor"],
        "reference_rows": comparison_rows,
        "transport_reference_rows": transport_comparison_rows,
        "raw_reference": raw_readout,
        "outputs": {
            "phase1_comparison_csv": str(phase1_csv),
            "transport_comparison_csv": str(transport_csv),
            "raw_metrics_bxor_csv": str(raw_csv),
        },
        "output_hashes": {
            "phase1_comparison_csv_sha256": sha256_file(phase1_csv),
            "transport_comparison_csv_sha256": sha256_file(transport_csv),
            "raw_metrics_bxor_csv_sha256": sha256_file(raw_csv),
        },
    }


def load_reference_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for raw_root in REFERENCE_ROOTS:
        root = resolve_project_path(raw_root)
        runs.extend(discover_phase1_runs(root))
    selected: list[dict[str, Any]] = []
    for variant in REFERENCE_VARIANTS:
        candidates = filter_runs(
            runs,
            variants=[variant],
            segment_bits=1_000_000,
            num_segments=3,
            segment_offset_bits=0,
            scales=[8, 12, 16, 20, 24, 28, 32],
            policies=["prefix", "suffix"],
        )
        if variant != "N":
            candidates = [run for run in candidates if run["dataset"]["config"].get("iteration") == 20]
        selected.extend(select_latest_per_variant(candidates, [variant]))

    null_runs: list[dict[str, Any]] = []
    for raw_root in REFERENCE_NULL_ROOTS:
        root = resolve_project_path(raw_root)
        if root.exists():
            null_runs.extend(discover_phase1_runs(root))
    null_selected = select_latest_per_variant(
        [
            run
            for run in null_runs
            if run["dataset"]["config"].get("source_variant") == "B"
            and run["dataset"]["config"].get("null_model") in {"matched-lz", "phase-matched-lz"}
            and run["dataset"]["config"].get("segment_bits") == 1_000_000
            and run["dataset"]["config"].get("num_segments") == 3
            and run["dataset"]["config"].get("scales") == [8, 12, 16, 20, 24, 28, 32]
        ]
    )
    return selected + null_selected


def compute_bxor_raw_metrics(
    snapshot_info: dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    q1_summary = load_latest_q1_summary()
    if q1_summary["_sha256"] != Q1_SUMMARY_SHA256:
        raise RuntimeError(
            "Latest Q1 raw-sequence summary does not match the D-0144 anchor. "
            f"Expected {Q1_SUMMARY_SHA256}, got {q1_summary['_sha256']}."
        )
    params = q1_summary["parameters"]
    mi_lags = [int(value) for value in params["mi_lags"]]
    raw_args = argparse.Namespace(
        mi_block_bits=int(params["mi_block_bits"]),
        mi_max_pairs=int(params["mi_max_pairs"]),
        fft_window_bits=int(params["fft_window_bits"]),
        fft_max_windows=int(params["fft_max_windows"]),
        block_bits=int(params["block_bits"]),
        pybdm_max_bits=int(params["pybdm_max_bits"]),
    )
    bits = load_observable_prefix_bits(Path(snapshot_info["struct_path"]), args.sample_bits)
    metrics, details = raw_metrics.compute_all_metrics(bits, args=raw_args, mi_lags=mi_lags)
    bxor_row = {
        "label": VARIANT,
        "variant": VARIANT,
        "kind": "observed_counterfactual",
        "source_variant": VARIANT,
        "seed": None,
        "null_model": None,
        "struct_path": snapshot_info["struct_path"],
        "struct_sha256": snapshot_info["struct_sha256"],
        "observable_bits": int(bits.size),
        **metrics,
    }
    rows = list(q1_summary["rows"]) + [bxor_row]
    fingerprint = calculate_lz_phi_fingerprint(bits[: min(int(bits.size), 50_000)])
    payload = {
        "q1_summary_path": str(q1_summary["_path"]),
        "q1_summary_sha256": q1_summary["_sha256"],
        "rows": rows,
        "bxor_row": bxor_row,
        "bxor_metric_details": details,
        "lz_phi_fingerprint_bxor": fingerprint,
    }
    write_json(run_dir / "raw_metrics_bxor_details.json", payload)
    return payload


def classify_outcome(readout: dict[str, Any]) -> str:
    phase = readout["phase1_bxor"]
    transport = readout["transport_bxor"]
    raw = readout["raw_bxor"]
    lz = readout["lz_phi_fingerprint_bxor"]
    retention = as_float(phase.get("retention_last"))
    active = as_float(transport.get("tail_transport_active_mean"))
    lz_phi_distance = as_float(lz.get("phi_distance"))
    zlib_ratio = as_float(raw.get("zlib_ratio"))

    # Gate-1-like here means "in the B/I compact-selective neighborhood",
    # not a hard 0.90 retention threshold: canonical B itself is near 0.86
    # in the frozen 3x1M table used for this contract.
    gate1_like = retention >= 0.80 and active >= 0.30
    if gate1_like and lz_phi_distance > 0.05:
        return "signature-independent-of-phi-candidate"
    if gate1_like and lz_phi_distance <= 0.05:
        return "phi-persists-under-Bxor"
    if not gate1_like:
        return "canonical-rule-dependent"
    if math.isfinite(zlib_ratio):
        return "ambiguous-counterfactual-readout"
    return "not-evaluable"


def calculate_lz_phi_fingerprint(bits: np.ndarray) -> dict[str, Any]:
    text = "".join("1" if value else "0" for value in bits.astype(np.uint8))
    scales = [1000, 2000, 5000, 10000, 20000, 50000]
    scales = [scale for scale in scales if scale <= len(text)]
    by_scale: dict[str, dict[str, float]] = {}
    ratios: list[float] = []
    for scale in scales:
        lz = lempel_ziv_complexity(text[:scale])
        normalized = lz / (scale / math.log2(scale)) if scale > 1 else math.nan
        by_scale[str(scale)] = {"raw_lz": float(lz), "normalized_lz": float(normalized)}
    for left, right in zip(scales, scales[1:]):
        a = by_scale[str(left)]["normalized_lz"]
        b = by_scale[str(right)]["normalized_lz"]
        if a > 0:
            ratios.append(float(b / a))
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    mean_ratio = float(np.mean(ratios)) if ratios else math.nan
    return {
        "method": "small-sample LZ76 ratio over raw observable prefix; secondary only",
        "bits_analyzed": len(text),
        "scales": scales,
        "by_scale": by_scale,
        "lz_ratios": ratios,
        "mean_ratio": mean_ratio,
        "inv_phi": inv_phi,
        "phi_distance": abs(mean_ratio - inv_phi) if math.isfinite(mean_ratio) else math.nan,
    }


def lempel_ziv_complexity(text: str) -> int:
    if not text:
        return 0
    n = len(text)
    complexity = 1
    prefix_len = 1
    while prefix_len < n:
        max_match = 0
        for start in range(prefix_len):
            match_len = 0
            while (
                prefix_len + match_len < n
                and text[start + match_len] == text[prefix_len + match_len]
            ):
                match_len += 1
                if start + match_len >= prefix_len:
                    break
            if match_len > max_match:
                max_match = match_len
        prefix_len += max_match + 1
        complexity += 1
    return complexity


def load_latest_q1_summary() -> dict[str, Any]:
    root = resolve_project_path("results/hsi_v2/raw_sequence_orthogonal_metrics")
    candidates = sorted(root.glob("*/summary.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No Q1 raw-sequence summary found.")
    path = candidates[-1]
    payload = load_json(path)
    payload["_path"] = str(path)
    payload["_sha256"] = sha256_file(path)
    return payload


def write_artifacts(run_dir: Path, summary: dict[str, Any], started: float) -> None:
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.json"
    write_json(summary_path, summary)
    report_path.write_text(render_report(summary), encoding="utf-8")
    manifest = {
        "generated_at": now_utc(),
        "script": Path(__file__).name,
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "contract": summary["contract"],
        "status": summary.get("status"),
        "outcome": summary.get("outcome"),
        "outputs": {
            "summary_json": str(summary_path),
            "report_md": str(report_path),
            "manifest_json": str(manifest_path),
            **summary.get("readout", {}).get("outputs", {}),
        },
        "output_hashes": {
            "summary_json_sha256": sha256_file(summary_path),
            "report_md_sha256": sha256_file(report_path),
            **summary.get("readout", {}).get("output_hashes", {}),
        },
        "snapshot": summary.get("snapshot"),
        "phase1": summary.get("phase1"),
        "elapsed_seconds": summary["elapsed_seconds"],
    }
    write_json(manifest_path, manifest)


def write_failure_artifacts(run_dir: Path, summary: dict[str, Any], started: float) -> None:
    summary["status"] = "failed"
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    write_json(run_dir / "summary.failed.json", summary)


def render_console_summary(summary: dict[str, Any]) -> str:
    lines = [
        "D-0144 Bxor counterfactual",
        "-" * 96,
        f"status: {summary.get('status')}",
        f"outcome: {summary.get('outcome')}",
    ]
    readout = summary.get("readout") or {}
    if readout:
        phase = readout["phase1_bxor"]
        transport = readout["transport_bxor"]
        lz = readout["lz_phi_fingerprint_bxor"]
        lines.extend(
            [
                f"BXOR retention@last: {fmt(phase.get('retention_last'))}",
                f"BXOR active_mean_tail: {fmt(transport.get('tail_transport_active_mean'))}",
                f"BXOR LZ mean-ratio secondary: {fmt(lz.get('mean_ratio'))}",
                f"BXOR distance to 1/phi: {fmt(lz.get('phi_distance'))}",
            ]
        )
    if summary.get("controlled_stop_reason"):
        lines.append(f"controlled stop: {summary['controlled_stop_reason']}")
    return "\n".join(lines)


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# D-0144 Bxor Counterfactual Readout",
        "",
        "## Contract",
        "",
        f"- Decision Log entry: `{CONTRACT_ID}`",
        f"- Proposal SHA256: `{PROPOSAL_SHA256}`",
        "- Canonical Level 0 generator modified: `false`",
        "- Counterfactual rule: B outer accumulation and inside-out collapse, with local simplifier `sum(bits) mod 2`.",
        "- No rule tuning, no upstream object regeneration, and no manuscript claim are authorized by this run.",
        "",
        "## Status",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Outcome: `{summary.get('outcome')}`",
    ]
    if summary.get("controlled_stop_reason"):
        lines.append(f"- Controlled stop reason: `{summary['controlled_stop_reason']}`")
    snapshot = summary.get("snapshot") or {}
    if snapshot:
        lines.extend(
            [
                "",
                "## Snapshot",
                "",
                f"- Structural snapshot: `{snapshot.get('struct_path')}`",
                f"- SHA256: `{snapshot.get('struct_sha256')}`",
                f"- Observable bits: `{snapshot.get('observable_bits'):,}`",
                f"- Compression ratio: `{snapshot.get('compression_ratio')}`",
            ]
        )
    readout = summary.get("readout") or {}
    if readout:
        phase = readout["phase1_bxor"]
        transport = readout["transport_bxor"]
        raw = readout["raw_bxor"]
        lz = readout["lz_phi_fingerprint_bxor"]
        lines.extend(
            [
                "",
                "## Bxor Primary Coordinates",
                "",
                "| Observable | Value |",
                "|---|---:|",
                f"| retention@last | {fmt(phase.get('retention_last'))} |",
                f"| prefix_survival_last | {fmt(phase.get('prefix_survival_last'))} |",
                f"| suffix_survival_last | {fmt(phase.get('suffix_survival_last'))} |",
                f"| active_mean_tail | {fmt(transport.get('tail_transport_active_mean'))} |",
                f"| breadth_ratio_last_to_first | {fmt(phase.get('breadth_ratio_last_to_first'))} |",
                f"| raw MI tail | {fmt(raw.get('long_range_mi_tail_mean_bits'))} |",
                f"| raw spectral entropy | {fmt(raw.get('spectral_entropy_norm'))} |",
                f"| raw block entropy | {fmt(raw.get('block_entropy_norm'))} |",
                f"| raw zlib ratio | {fmt(raw.get('zlib_ratio'))} |",
                f"| secondary LZ mean ratio | {fmt(lz.get('mean_ratio'))} |",
                f"| secondary distance to 1/phi | {fmt(lz.get('phi_distance'))} |",
                "",
                "## Phase 1 Comparison",
                "",
                "| Variant | retention@last | prefix surv | suffix surv | kept@last | candidate@last |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in readout["reference_rows"]:
            lines.append(
                f"| {row['variant']} | {fmt(row.get('retention_last'))} | "
                f"{fmt(row.get('prefix_survival_last'))} | {fmt(row.get('suffix_survival_last'))} | "
                f"{row.get('kept_last')} | {row.get('candidate_last')} |"
            )
        lines.extend(
            [
                "",
                "## Transport Comparison",
                "",
                "| Variant | active_mean_tail | entropy_mean_tail | decay_mean_tail |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in readout["transport_reference_rows"]:
            lines.append(
                f"| {row['variant']} | {fmt(row.get('tail_transport_active_mean'))} | "
                f"{fmt(row.get('tail_transport_entropy_mean'))} | "
                f"{fmt(row.get('tail_transport_decay_mean'))} |"
            )
        lines.extend(
            [
                "",
                "## Interpretive Guard",
                "",
                "The automated outcome is a compact readout, not a manuscript claim. If Bxor preserves the compact-selective regime while moving away from the secondary phi fingerprint, the result supports a future claim that the signature is not reducible to phi. If Bxor loses the Gate 1 regime, the result supports canonical-rule dependence. If Bxor keeps both the regime and the phi fingerprint, the phi-artifact question remains open.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: scalar_for_csv(row.get(key)) for key in keys})


def select_row(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get("variant") == variant]
    if not matches:
        raise RuntimeError(f"Missing row for variant {variant}")
    return matches[-1]


def cleanup_temp(run_dir: Path) -> None:
    temp_dir = run_dir / "temp_generation"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_iterations <= 0:
        raise SystemExit("--max-iterations must be positive")
    if args.segment_bits <= 0 or args.num_segments <= 0:
        raise SystemExit("--segment-bits and --num-segments must be positive")
    if args.sample_bits <= 0:
        raise SystemExit("--sample-bits must be positive")
    if args.max_ram_gb <= 0:
        raise SystemExit("--max-ram-gb must be positive")
    if not (1 <= args.compression_level <= 9):
        raise SystemExit("--compression-level must be in 1..9")


def resolve_project_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compact_int(value: int) -> str:
    if value % 1_000_000 == 0:
        return f"{value // 1_000_000}M"
    if value % 1_000 == 0:
        return f"{value // 1_000}K"
    return str(value)


def scalar_for_csv(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def as_float(value: Any) -> float:
    try:
        if value is None:
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def fmt(value: Any) -> str:
    value = as_float(value)
    if not math.isfinite(value):
        return "-"
    if abs(value) >= 100:
        return f"{value:.3f}"
    if abs(value) >= 1:
        return f"{value:.5f}"
    return f"{value:.6f}"


def phase_print(title: str, detail: str = "", *, quiet: bool = False) -> None:
    if quiet:
        return
    print(f"\n[Phase] {title}", flush=True)
    if detail:
        print(f"        {detail}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
