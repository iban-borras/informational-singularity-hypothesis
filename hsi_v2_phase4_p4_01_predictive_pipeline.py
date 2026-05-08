#!/usr/bin/env python3
"""HSI v2 Phase 4 P4-01 predictive holdout launcher.

This is the clean reproduction entrypoint for the first pre-registered
Phase 4 prediction. It keeps the public command small, splits LZ-family
nulls internally, and avoids using the N2-12b top-k wrapper in a degenerate
top=128 vs top=128 configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hsi_v2_phase2_off_universe_mass_readout import SIGNATURE_FIELDS
from hsi_v2_phase2_parent_survival_revalidation import flush_output, latest_child_file
from hsi_v2_phase2_phase_channel_width_topk_sensitivity import (
    COMPARISON_FIELDS,
    ENVELOPE_FIELDS,
    MISSING_FIELDS,
    RESPONSE_FIELDS,
    band_label,
    build_envelopes,
    build_missing_rows,
    load_response_rows,
    write_csv,
)
from hsi_v2_phase2_transport_defect_strict import parse_int_list, phase_print
from utils.progress import format_time
from v2.common.cli import resolve_dir


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase4/p4_01_predictive_pipeline"
DEFAULT_PHASE1_DIR = "results/hsi_v2/phase1_high_scales"
DEFAULT_BANDS = (
    "696000000,705000000,714000000,723000000,726000000,729000000,"
    "738000000,747000000,756000000"
)
DEFAULT_HOLDOUT_BANDS = "738000000,747000000,756000000"
DEFAULT_LAGS = (
    "-30000000,-29500000,-27000000,-26500000,-26000000,-24000000,"
    "23500000,24500000,26500000,29500000"
)
DEFAULT_PHASE_MATCHED_LZ_SEEDS = "607,709,811,907,1009,1103"
DEFAULT_MATCHED_LZ_REFERENCE_SEEDS = "17,101,211,307,401,503,607,709,811,907,1009,1103"
LZ_CHAIN_WORKER_CAP = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the P4-01 predictive holdout pipeline with the pre-registered "
            "lag grid, calibration bands, holdout bands, and separated LZ-family nulls."
        )
    )
    parser.add_argument("--phase1-dir", default=DEFAULT_PHASE1_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help=(
            "Maximum requested workers. P4-01 caps heavy LZ-family child chains "
            f"to {LZ_CHAIN_WORKER_CAP} workers internally to avoid memory-pressure failures."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    band_starts = parse_int_list(DEFAULT_BANDS, label="DEFAULT_BANDS", allow_zero=True)
    lags = parse_int_list(DEFAULT_LAGS, label="DEFAULT_LAGS", allow_zero=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_slug = f"phase4-p4-01-predictive-pipeline__top-128__bands-9__lags-10__{timestamp}"
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "run_id": run_slug,
        "timestamp_utc": timestamp,
        "status": "dry-run" if args.dry_run else "running",
        "parameters": {
            "phase1_dir": str(phase1_dir),
            "workers": args.workers,
            "bands": DEFAULT_BANDS,
            "holdout_bands": DEFAULT_HOLDOUT_BANDS,
            "lags": DEFAULT_LAGS,
            "top_patterns": 128,
            "window_count": 19,
            "window_step_bits": 500_000,
            "phase_matched_lz_seeds": DEFAULT_PHASE_MATCHED_LZ_SEEDS,
            "matched_lz_reference_seeds": DEFAULT_MATCHED_LZ_REFERENCE_SEEDS,
            "dry_run": args.dry_run,
        },
        "predictions": [
            "P4-01A: holdout bands should not look like a clean universal continuation.",
            "P4-01B: any positive holdout channel must stay inside the predeclared lag grid.",
            "P4-01C: phase-matched-LZ should remain the hardest LZ-family null near late boundaries.",
            "P4-01D: if channel retention survives, the Phase 3 readout should remain thermodynamic rather than topological.",
        ],
        "stages": [],
        "outputs": {},
    }

    started = time.perf_counter()
    try:
        phase_print(
            "Preparing P4-01 predictive holdout launcher",
            (
                "top=128 | bands=696M..756M | holdout=738M,747M,756M | "
                f"lags=10 | workers={args.workers}"
            ),
            quiet=args.quiet,
        )
        flush_output()

        pmlz_lag_run = stage_lag_response(
            args=args,
            state=state,
            run_dir=run_dir,
            phase1_dir=phase1_dir,
            stage_name="phase-matched-lz primary hard-null lag response",
            output_name="pmlz_lag_response",
            null_models="markov1,phase-matched-lz",
            seeds=DEFAULT_PHASE_MATCHED_LZ_SEEDS,
            worker_cap=LZ_CHAIN_WORKER_CAP,
        )
        matched_lag_run = stage_lag_response(
            args=args,
            state=state,
            run_dir=run_dir,
            phase1_dir=phase1_dir,
            stage_name="matched-lz reference lag response",
            output_name="matched_lz_reference_lag_response",
            null_models="markov1,matched-lz",
            seeds=DEFAULT_MATCHED_LZ_REFERENCE_SEEDS,
            worker_cap=LZ_CHAIN_WORKER_CAP,
        )

        if args.dry_run:
            state["outputs"] = {
                "pmlz_lag_response_root": str(run_dir / "pmlz_lag_response"),
                "matched_lz_reference_lag_response_root": str(
                    run_dir / "matched_lz_reference_lag_response"
                ),
            }
            write_pipeline_files(run_dir, state)
            print_dry_run_notice(run_dir)
            return 0

        assert pmlz_lag_run is not None
        assert matched_lag_run is not None
        pmlz_envelope_run = build_single_top_envelope_artifact(
            state=state,
            run_dir=run_dir,
            output_name="pmlz_channel_envelope",
            source_lag_run=pmlz_lag_run,
            band_starts=band_starts,
            lags=lags,
        )
        matched_envelope_run = build_single_top_envelope_artifact(
            state=state,
            run_dir=run_dir,
            output_name="matched_lz_reference_envelope",
            source_lag_run=matched_lag_run,
            band_starts=band_starts,
            lags=lags,
        )
        signature_run = build_empty_signature_artifact(run_dir=run_dir)

        transition_run = stage_simple(
            args=args,
            state=state,
            run_dir=run_dir,
            name="N2-14 phase-channel transition graph",
            output_name="transition_graph",
            command=[
                sys.executable,
                str(SCRIPT_DIR / "hsi_v2_phase2_phase_channel_transition_graph.py"),
                "--envelope-runs",
                str(pmlz_envelope_run),
                "--signature-run",
                str(signature_run),
                "--output-dir",
                str(run_dir / "transition_graph"),
            ],
            required_file="summary.json",
        )
        weighted_run = stage_simple(
            args=args,
            state=state,
            run_dir=run_dir,
            name="N3-05b weighted quotient child-routing graph",
            output_name="weighted_quotient",
            command=[
                sys.executable,
                str(SCRIPT_DIR / "hsi_v2_phase3_weighted_quotient_child_routing_spectral_graph.py"),
                "--transition-run",
                str(transition_run),
                "--child-routing-run",
                str(pmlz_lag_run),
                "--output-dir",
                str(run_dir / "weighted_quotient"),
            ],
            required_file="summary.json",
        )
        component_run = stage_simple(
            args=args,
            state=state,
            run_dir=run_dir,
            name="N3-05c component-normalized quotient spectrum",
            output_name="component_quotient",
            command=[
                sys.executable,
                str(SCRIPT_DIR / "hsi_v2_phase3_component_quotient_child_routing_spectrum.py"),
                "--weighted-quotient-run",
                str(weighted_run),
                "--output-dir",
                str(run_dir / "component_quotient"),
            ],
            required_file="summary.json",
        )
        flow_run = stage_simple(
            args=args,
            state=state,
            run_dir=run_dir,
            name="N3-07 flow-entropy readout",
            output_name="flow_entropy",
            command=[
                sys.executable,
                str(SCRIPT_DIR / "hsi_v2_phase3_flow_entropy_readout.py"),
                "--component-run",
                str(component_run),
                "--output-dir",
                str(run_dir / "flow_entropy"),
            ],
            required_file="summary.json",
        )
        prediction_readout = build_prediction_readout(
            state=state,
            run_dir=run_dir,
            pmlz_envelope_run=pmlz_envelope_run,
            matched_envelope_run=matched_envelope_run,
            flow_run=flow_run,
        )

        state["status"] = "completed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        state["outputs"].update(
            {
                "pmlz_lag_response": str(pmlz_lag_run),
                "matched_lz_reference_lag_response": str(matched_lag_run),
                "pmlz_channel_envelope": str(pmlz_envelope_run),
                "matched_lz_reference_envelope": str(matched_envelope_run),
                "empty_signature": str(signature_run),
                "transition_graph": str(transition_run),
                "weighted_quotient": str(weighted_run),
                "component_quotient": str(component_run),
                "flow_entropy": str(flow_run),
                "prediction_readout": str(prediction_readout),
            }
        )
        write_pipeline_files(run_dir, state)
        print_final_summary(run_dir, state)
        return 0
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        state["status"] = "failed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        state["failure"] = str(exc)
        mark_running_stage_failed(state, str(exc))
        write_pipeline_files(run_dir, state)
        raise SystemExit(
            "P4-01 pipeline failed. Inspect the child log above and the partial "
            f"manifest at: {run_dir / 'manifest.json'}"
        ) from exc


def stage_lag_response(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    run_dir: Path,
    phase1_dir: Path,
    stage_name: str,
    output_name: str,
    null_models: str,
    seeds: str,
    worker_cap: int,
) -> Path | None:
    output_dir = run_dir / output_name
    effective_workers = min(args.workers, worker_cap)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "hsi_v2_phase2_child_routing_lag_response.py"),
        "--phase1-dir",
        str(phase1_dir),
        "--output-dir",
        str(output_dir),
        "--stage",
        "all",
        "--top-patterns",
        "128",
        "--band-starts",
        DEFAULT_BANDS,
        "--window-count",
        "19",
        "--window-step-bits",
        "500000",
        "--lags=" + DEFAULT_LAGS,
        "--null-models",
        null_models,
        "--matched-lz-seeds",
        seeds,
        "--workers",
        str(effective_workers),
        "--quiet-children",
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.quiet:
        command.append("--quiet")

    record_stage(
        state,
        stage_name,
        command,
        "planned" if args.dry_run else "running",
        metadata={
            "requested_workers": args.workers,
            "effective_workers": effective_workers,
            "worker_cap_reason": "LZ-family chains load many cached 765M-bit sources per target.",
        },
    )
    write_pipeline_files(run_dir, state)
    if args.dry_run:
        return None

    phase_print(
        "Running P4-01 lag-response stage",
        (
            f"{stage_name} | null_models={null_models} | seeds={seeds} | "
            f"workers={effective_workers}/{args.workers}"
        ),
        quiet=args.quiet,
    )
    flush_output()
    run_command(command)
    run_path = latest_child_file(output_dir, "summary.json").parent
    finish_stage(state, stage_name, run_path)
    write_pipeline_files(run_dir, state)
    return run_path


def build_single_top_envelope_artifact(
    *,
    state: dict[str, Any],
    run_dir: Path,
    output_name: str,
    source_lag_run: Path,
    band_starts: list[int],
    lags: list[int],
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    envelope_root = run_dir / output_name
    envelope_root.mkdir(parents=True, exist_ok=True)
    artifact_dir = envelope_root / f"phase4-single-top-channel-envelope__src-{source_lag_run.name}__{timestamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    response_rows = load_response_rows(top=128, run_dir=source_lag_run)
    envelope_rows = build_envelopes(response_rows, b_threshold=0.90, margin_threshold=0.30)
    missing_rows = build_missing_rows(
        response_rows,
        top_values=[128],
        band_starts=band_starts,
        lags=lags,
        window_count=19,
        window_step_bits=500_000,
    )
    summary = {
        "run_id": artifact_dir.name,
        "timestamp_utc": timestamp,
        "source_lag_response_run": str(source_lag_run),
        "notes": [
            "P4-01 single-top envelope artifact.",
            "Built directly from a fresh N2-11 lag-response run.",
            "No top-k reference rows are mixed into this artifact.",
        ],
        "channel_envelopes": envelope_rows,
        "missing_cells": missing_rows,
    }
    manifest = {
        "script": Path(__file__).name,
        "run_dir": str(artifact_dir),
        "source_lag_response_run": str(source_lag_run),
        "outputs": {
            "summary": str(artifact_dir / "summary.json"),
            "report": str(artifact_dir / "report.md"),
            "response_rows_csv": str(artifact_dir / "response_rows.csv"),
            "channel_envelopes_csv": str(artifact_dir / "channel_envelopes.csv"),
            "missing_cells_csv": str(artifact_dir / "missing_cells.csv"),
            "topk_comparison_csv": str(artifact_dir / "topk_comparison.csv"),
            "manifest": str(artifact_dir / "manifest.json"),
        },
    }
    (artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (artifact_dir / "report.md").write_text(render_envelope_report(summary), encoding="utf-8")
    write_csv(artifact_dir / "response_rows.csv", response_rows, RESPONSE_FIELDS)
    write_csv(artifact_dir / "channel_envelopes.csv", envelope_rows, ENVELOPE_FIELDS)
    write_csv(artifact_dir / "missing_cells.csv", missing_rows, MISSING_FIELDS)
    write_csv(artifact_dir / "topk_comparison.csv", [], COMPARISON_FIELDS)

    state["stages"].append(
        {
            "name": f"build single-top envelope: {output_name}",
            "status": "completed",
            "run_path": str(artifact_dir),
            "started_at_utc": timestamp,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    write_pipeline_files(run_dir, state)
    return artifact_dir


def build_empty_signature_artifact(*, run_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    signature_root = run_dir / "empty_signature"
    signature_root.mkdir(parents=True, exist_ok=True)
    artifact_dir = signature_root / f"phase4-empty-routing-signatures__{timestamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_id": artifact_dir.name,
        "timestamp_utc": timestamp,
        "notes": [
            "P4-01 transition graph uses fresh single-top envelopes.",
            "No top-k expansion signature is injected here, because P4-01 is not a top-k expansion readout.",
        ],
    }
    manifest = {
        "script": Path(__file__).name,
        "run_dir": str(artifact_dir),
        "outputs": {
            "summary": str(artifact_dir / "summary.json"),
            "routing_signatures_csv": str(artifact_dir / "routing_signatures.csv"),
            "manifest": str(artifact_dir / "manifest.json"),
        },
    }
    (artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (artifact_dir / "routing_signatures.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=SIGNATURE_FIELDS).writeheader()
    return artifact_dir


def build_prediction_readout(
    *,
    state: dict[str, Any],
    run_dir: Path,
    pmlz_envelope_run: Path,
    matched_envelope_run: Path,
    flow_run: Path,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    readout_root = run_dir / "prediction_readout"
    readout_root.mkdir(parents=True, exist_ok=True)
    artifact_dir = readout_root / f"phase4-p4-01-prediction-readout__{timestamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    holdout_labels = {
        band_label(start, window_count=19, window_step_bits=500_000)
        for start in parse_int_list(DEFAULT_HOLDOUT_BANDS, label="DEFAULT_HOLDOUT_BANDS", allow_zero=True)
    }
    pmlz_rows = read_csv_dicts(pmlz_envelope_run / "channel_envelopes.csv")
    matched_rows = read_csv_dicts(matched_envelope_run / "channel_envelopes.csv")
    flow_rows = read_csv_dicts(flow_run / "flow_entropy_summary.csv")

    holdout_pmlz = [
        row for row in pmlz_rows
        if row.get("band") in holdout_labels
    ]
    pmlz_by_cell = {(row["band"], row["channel"]): row for row in pmlz_rows}
    matched_by_cell = {(row["band"], row["channel"]): row for row in matched_rows}

    strong_holdout_cells = [
        row for row in holdout_pmlz
        if int_or_zero(row.get("strong_count")) > 0
    ]
    p4_01c_cells = []
    for key, pmlz in sorted(pmlz_by_cell.items()):
        if key[0] not in holdout_labels:
            continue
        matched = matched_by_cell.get(key)
        if not matched:
            continue
        pmlz_margin = optional_float(pmlz.get("best_margin"))
        matched_margin = optional_float(matched.get("best_margin"))
        p4_01c_cells.append(
            {
                "band": key[0],
                "channel": key[1],
                "phase_matched_lz_margin": pmlz_margin,
                "matched_lz_margin": matched_margin,
                "phase_matched_lz_is_harder_or_equal": (
                    pmlz_margin is not None
                    and matched_margin is not None
                    and pmlz_margin <= matched_margin
                ),
            }
        )
    observed_flow = [
        row for row in flow_rows
        if row.get("source_label") == "observed" and row.get("variant") in {"B", "E"}
    ]
    strict_flow_rows = [
        row for row in observed_flow
        if row.get("verdict") == "strict-low-entropy-flow"
    ]

    p4_01c_supported = sum(
        1 for row in p4_01c_cells if row["phase_matched_lz_is_harder_or_equal"]
    )
    p4_01c_total = len(p4_01c_cells)
    p4_01c_strong_threshold = (4 * p4_01c_total + 4) // 5 if p4_01c_total else 0
    p4_01c_mixed_threshold = (p4_01c_total + 1) // 2 if p4_01c_total else 0
    if not p4_01c_cells:
        p4_01c_status = "not_supported_or_inconclusive"
    elif p4_01c_supported >= p4_01c_strong_threshold:
        p4_01c_status = "supported"
    elif p4_01c_supported >= p4_01c_mixed_threshold:
        p4_01c_status = "mixed_partial"
    else:
        p4_01c_status = "not_supported_or_inconclusive"
    payload = {
        "run_id": artifact_dir.name,
        "timestamp_utc": timestamp,
        "inputs": {
            "pmlz_envelope_run": str(pmlz_envelope_run),
            "matched_envelope_run": str(matched_envelope_run),
            "flow_run": str(flow_run),
        },
        "readout": {
            "P4-01A": {
                "holdout_cells": len(holdout_pmlz),
                "strong_holdout_cells": len(strong_holdout_cells),
                "status": (
                    "supported_boundary_or_discontinuous"
                    if len(strong_holdout_cells) < len(holdout_pmlz)
                    else "clean_continuation_risk"
                ),
            },
            "P4-01B": {
                "registered_lag_grid_only": True,
                "strong_cell_best_lags": [
                    {
                        "band": row["band"],
                        "channel": row["channel"],
                        "best_margin_lag": row.get("best_margin_lag", ""),
                    }
                    for row in strong_holdout_cells
                ],
                "status": "bounded_by_predeclared_grid",
            },
            "P4-01C": {
                "holdout_comparison_cells": p4_01c_total,
                "phase_matched_lz_harder_or_equal_cells": p4_01c_supported,
                "supported_threshold_cells": p4_01c_strong_threshold,
                "mixed_threshold_cells": p4_01c_mixed_threshold,
                "status": p4_01c_status,
                "cells": p4_01c_cells,
            },
            "P4-01D": {
                "observed_flow_rows": len(observed_flow),
                "strict_low_entropy_flow_rows": len(strict_flow_rows),
                "status": (
                    "supported"
                    if observed_flow and len(strict_flow_rows) == len(observed_flow)
                    else "not_supported_or_inconclusive"
                ),
            },
        },
    }
    (artifact_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "script": Path(__file__).name,
                "run_dir": str(artifact_dir),
                "inputs": payload["inputs"],
                "outputs": {
                    "summary": str(artifact_dir / "summary.json"),
                    "report": str(artifact_dir / "report.md"),
                    "manifest": str(artifact_dir / "manifest.json"),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "report.md").write_text(render_prediction_report(payload), encoding="utf-8")
    state["stages"].append(
        {
            "name": "P4-01 prediction readout",
            "status": "completed",
            "run_path": str(artifact_dir),
            "started_at_utc": timestamp,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    write_pipeline_files(run_dir, state)
    return artifact_dir


def stage_simple(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    run_dir: Path,
    name: str,
    output_name: str,
    command: list[str],
    required_file: str,
) -> Path:
    if args.quiet:
        command.append("--quiet")
    record_stage(state, name, command, "running")
    write_pipeline_files(run_dir, state)
    phase_print("Running P4-01 stage", name, quiet=args.quiet)
    flush_output()
    run_command(command)
    run_path = latest_child_file(run_dir / output_name, required_file).parent
    finish_stage(state, name, run_path)
    write_pipeline_files(run_dir, state)
    return run_path


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=str(SCRIPT_DIR), check=True)


def record_stage(
    state: dict[str, Any],
    name: str,
    command: list[str],
    status: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    item = {
        "name": name,
        "status": status,
        "command": command,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if metadata:
        item.update(metadata)
    state["stages"].append(item)


def finish_stage(state: dict[str, Any], name: str, run_path: Path) -> None:
    for stage in reversed(state["stages"]):
        if stage["name"] == name and stage["status"] == "running":
            stage["status"] = "completed"
            stage["run_path"] = str(run_path)
            stage["finished_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            return
    raise RuntimeError(f"Internal launcher error: stage not found: {name}")


def mark_running_stage_failed(state: dict[str, Any], failure: str) -> None:
    for stage in reversed(state["stages"]):
        if stage.get("status") == "running":
            stage["status"] = "failed"
            stage["failure"] = failure
            stage["finished_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            return


def write_pipeline_files(run_dir: Path, state: dict[str, Any]) -> None:
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.json"
    payload = {
        **state,
        "artifacts": {
            "summary": str(summary_path),
            "report": str(report_path),
            "manifest": str(manifest_path),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "script": Path(__file__).name,
                "run_dir": str(run_dir),
                "status": state["status"],
                "parameters": state["parameters"],
                "stages": state["stages"],
                "outputs": state["outputs"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report_path.write_text(render_report(payload), encoding="utf-8")


def render_report(payload: dict[str, Any]) -> str:
    params = payload["parameters"]
    lines = [
        "# Phase 4 P4-01 Predictive Holdout Pipeline",
        "",
        "## Status",
        "",
        f"- Status: `{payload['status']}`",
        f"- Run ID: `{payload['run_id']}`",
        f"- Generated UTC: `{payload['timestamp_utc']}`",
        f"- Workers: `{params['workers']}`",
        "",
        "## Pre-registered Contract",
        "",
        f"- Calibration + holdout band starts: `{params['bands']}`",
        f"- Holdout band starts: `{params['holdout_bands']}`",
        f"- Lag grid: `{params['lags']}`",
        f"- Primary hard null: `phase-matched-lz` seeds `{params['phase_matched_lz_seeds']}`",
        f"- Reference null envelope: `matched-lz` seeds `{params['matched_lz_reference_seeds']}`",
        "",
        "## Predictions",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["predictions"])
    lines.extend(["", "## Stage Ledger", ""])
    for stage in payload["stages"]:
        lines.append(f"- `{stage['status']}` {stage['name']}")
        if stage.get("run_path"):
            lines.append(f"  - Output: `{stage['run_path']}`")
    if payload.get("outputs"):
        lines.extend(["", "## Final Outputs", ""])
        for key, value in sorted(payload["outputs"].items()):
            lines.append(f"- `{key}`: `{value}`")
    if payload.get("failure"):
        lines.extend(["", "## Failure", "", f"```text\n{payload['failure']}\n```"])
    return "\n".join(lines) + "\n"


def render_envelope_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 4 Single-Top Channel Envelope",
        "",
        f"- Source lag-response run: `{summary['source_lag_response_run']}`",
        "- Top-k: `128`",
        "- B threshold: `0.90`",
        "- Margin threshold: `0.30`",
        "",
        "## Envelope Rows",
        "",
        "| Band | Channel | Strong cells | Width | Best lag | Best margin |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["channel_envelopes"]:
        lines.append(
            f"| {row['band']} | {row['channel']} | {row['strong_count']} | "
            f"{blank(row['strong_width_bits'])} | {blank(row['best_margin_lag'])} | "
            f"{blank(row['best_margin'])} |"
        )
    if summary["missing_cells"]:
        lines.extend(["", "## Missing Cells", ""])
        lines.append(f"- Missing cells: `{len(summary['missing_cells'])}`")
    return "\n".join(lines) + "\n"


def render_prediction_report(payload: dict[str, Any]) -> str:
    readout = payload["readout"]
    lines = [
        "# Phase 4 P4-01 Prediction Readout",
        "",
        "This is a compact first-pass readout over the generated P4-01 artifacts.",
        "It is not a substitute for scientific interpretation, Grace audit, or Sofia editorial review.",
        "",
        "## Prediction Status",
        "",
        "| Prediction | Status | Key count |",
        "|---|---|---:|",
    ]
    lines.append(
        "| P4-01A | "
        f"{readout['P4-01A']['status']} | "
        f"{readout['P4-01A']['strong_holdout_cells']}/{readout['P4-01A']['holdout_cells']} strong holdout cells |"
    )
    lines.append(
        "| P4-01B | "
        f"{readout['P4-01B']['status']} | "
        f"{len(readout['P4-01B']['strong_cell_best_lags'])} strong-cell lags |"
    )
    lines.append(
        "| P4-01C | "
        f"{readout['P4-01C']['status']} | "
        f"{readout['P4-01C']['phase_matched_lz_harder_or_equal_cells']}/"
        f"{readout['P4-01C']['holdout_comparison_cells']} harder/equal cells |"
    )
    lines.append(
        "| P4-01D | "
        f"{readout['P4-01D']['status']} | "
        f"{readout['P4-01D']['strict_low_entropy_flow_rows']}/"
        f"{readout['P4-01D']['observed_flow_rows']} strict rows |"
    )
    lines.extend(
        [
            "",
            "## Caution",
            "",
            "- P4-01A is scored only as boundary/discontinuity vs clean continuation, not as a final law.",
            "- P4-01C is `supported` only under strong holdout dominance; half-split outcomes are `mixed_partial`.",
            "- P4-01C compares sampled holdout envelope margins; it is not a universal null dominance claim.",
            "- P4-01D inherits the Phase 3 wording fence: thermodynamic routing concentration, not recovered geometry.",
            "",
        ]
    )
    return "\n".join(lines)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def int_or_zero(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(value)


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def blank(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value)


def print_dry_run_notice(run_dir: Path) -> None:
    print("Phase 4 P4-01 predictive holdout pipeline")
    print("-" * 96)
    print("Dry run only. Inspect manifest for planned delegated commands.")
    print(f"\nSaved dry-run manifest to: {run_dir / 'manifest.json'}")
    print(f"Saved dry-run report to: {run_dir / 'report.md'}")


def print_final_summary(run_dir: Path, state: dict[str, Any]) -> None:
    print("Phase 4 P4-01 predictive holdout pipeline")
    print("-" * 96)
    print(f"status={state['status']} elapsed={format_time(float(state['elapsed_seconds']))}")
    print(f"\nSaved summary to: {run_dir / 'summary.json'}")
    print(f"Saved report to: {run_dir / 'report.md'}")
    print(f"Saved manifest to: {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
