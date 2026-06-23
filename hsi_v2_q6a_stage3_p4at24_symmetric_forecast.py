#!/usr/bin/env python3
"""Q6-A Stage 3 P4@24 symmetric forecast runner.

This wrapper executes the frozen Q6-A Stage 2 v1.0 contract. It delegates the
heavy routing readout to the audited lag-response pipeline and only handles the
sequential band order, tertile aggregation, hard-null classification, stop
rule, and reproducibility manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hsi_v2_phase2_parent_survival_revalidation import latest_child_file
from hsi_v2_phase2_transport_defect_strict import phase_print
from utils.progress import format_time
from v2.common.cli import resolve_dir, resolve_results_dir
from v2.common.naming import compact_int


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent

CONTRACT_PATH = WORKSPACE_ROOT / "prereg_Q6A_stage2_p4at24_symmetric_forecast_v1.0.md"
CONTRACT_SHA256 = "C5D05C2C0D82073191A2CFB8EA1E406ECC04EFDD9B18925FED8DC6DF75DDC602"

DEFAULT_PHASE1_DIR = r"D:\Iban\Projectes\HSI\results\hsi_v2\q6a_stage1_high_scales_symmetric"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/q6a_stage2_p4at24_symmetric_forecast"

STAGE1_HASHES = {
    r"D:\Iban\Projectes\HSI\results\hsi_v2\q6a_stage1_high_scales_symmetric\phase1-tower__var-B__iter-24__seg-3x1M__m-8-12-16-20-24-28-32-33-34-35-36-40-44-48__proj-prefix-suffix__20260617T100722\phase1_summary.json": "BE2F5983979ADB01CCF730B0C38143561B4C0A0EC774CD4FFCDDB6A53BC94647",
    r"D:\Iban\Projectes\HSI\results\hsi_v2\q6a_stage1_high_scales_symmetric\phase1-tower__var-B__iter-24__seg-3x1M__m-8-12-16-20-24-28-32-33-34-35-36-40-44-48__proj-prefix-suffix__20260617T100722\manifest.json": "D81074A755BA443E223D82610045468C0E11084BCB853C64B03DE5BA9BD618DA",
    r"D:\Iban\Projectes\HSI\results\hsi_v2\q6a_stage1_high_scales_symmetric\phase1-tower__var-E__iter-24__seg-3x1M__m-8-12-16-20-24-28-32-33-34-35-36-40-44-48__proj-prefix-suffix__20260617T101010\phase1_summary.json": "BE2F5983979ADB01CCF730B0C38143561B4C0A0EC774CD4FFCDDB6A53BC94647",
    r"D:\Iban\Projectes\HSI\results\hsi_v2\q6a_stage1_high_scales_symmetric\phase1-tower__var-E__iter-24__seg-3x1M__m-8-12-16-20-24-28-32-33-34-35-36-40-44-48__proj-prefix-suffix__20260617T101010\manifest.json": "BCBB327D98B4C21CF7F8AD1E96757E2022992AD6EBDCF4D7625B6BB495DA48D4",
    r"D:\Iban\Projectes\HSI\results\hsi_v2\q6a_stage1_high_scales_symmetric\readout\q6a-stage1-readout__B-E__m48__20260617T101207\summary.json": "2AB4BC493FFB1B15EB78FC72E0B84CB757FC4B67C56794C35EA33BBA4401D1F5",
    r"D:\Iban\Projectes\HSI\results\hsi_v2\q6a_stage1_high_scales_symmetric\readout\q6a-stage1-readout__B-E__m48__20260617T101207\manifest.json": "B455B762B27B3DF26E62BC23F1B2C0CD46FBB5873002FC9E945A2276A231E115",
    r"D:\Iban\Projectes\HSI\results\hsi_v2\q6a_stage1_high_scales_symmetric\readout\q6a-stage1-readout__B-E__m48__20260617T101207\report.md": "41B011D28C39F002EAA8230D9E2565C0FFBCBCE92B57A84AA1BAA692979006E6",
}

SOURCE_HASHES = {
    r"D:\Iban\Projectes\HSI\results\level0\phi_snapshots\var_B\phi_iter24.struct.gz": "B766982BDB9AF90975F8E4D0B7D92D413964E402DBB32044C2FC8EBF0AE57702",
    r"D:\Iban\Projectes\HSI\results\level0\phi_snapshots\var_E\phi_iter24.struct.gz": "8D3BE0CE2159B699D424103C7BD4C02517D2A020ABE2B671EFFFA15CDCAC2DAD",
}

BANDS = [804_000_000, 854_000_000, 904_000_000, 10_000_000_000, 100_000_000_000, 500_000_000_000]
LAGS = [-29_500_000, -27_000_000, -26_500_000, -24_000_000, 24_500_000, 26_500_000]
NULL_SEEDS = [607, 709, 811, 907, 1009, 1103]
TERTILES = [
    ("T1", 0, 33),
    ("T2", 33, 34),
    ("T3", 67, 34),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Q6-A Stage 3 P4@24 symmetric forecast.")
    parser.add_argument("--phase1-dir", default=DEFAULT_PHASE1_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-run-dir", default="")
    parser.add_argument("--skip-large-source-hash", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    started = time.perf_counter()
    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_root = resolve_results_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.resume_run_dir:
        run_dir = resolve_dir(args.resume_run_dir, anchor_file=__file__)
        state = load_json(run_dir / "stage3_state.json")
        timestamp = state["timestamp_utc"]
        reset_derived_state_for_resume(state)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_slug = f"q6a-stage3-p4at24-symmetric-forecast__bands-6__{timestamp}"
        run_dir = output_root / run_slug
        run_dir.mkdir(parents=True, exist_ok=True)
        state = build_initial_state(timestamp, run_dir, args, phase1_dir)

    try:
        verify_inputs(skip_large_source_hash=args.skip_large_source_hash, quiet=args.quiet)
        state["input_verification"] = "passed"
        write_state(run_dir, state)

        if args.dry_run:
            state["status"] = "dry-run"
            state["planned_commands"] = build_all_planned_commands(args, phase1_dir, run_dir)
            state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            write_outputs(run_dir, state)
            print(render_console_summary(state))
            return 0

        execute_stage3(args=args, phase1_dir=phase1_dir, run_dir=run_dir, state=state)
        state["status"] = "completed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_outputs(run_dir, state)
        print(render_console_summary(state))
        print(f"\nSaved summary to: {run_dir / 'summary.json'}")
        print(f"Saved report to: {run_dir / 'report.md'}")
        print(f"Saved manifest to: {run_dir / 'manifest.json'}")
        return 0
    except subprocess.CalledProcessError as exc:
        state["status"] = "failed"
        state["error"] = {"returncode": int(exc.returncode), "command": [str(part) for part in exc.cmd]}
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_outputs(run_dir, state)
        print(f"[x] Q6-A Stage 3 failed after {format_time(time.perf_counter() - started)}")
        print(f"Saved failure summary to: {run_dir / 'summary.json'}")
        return int(exc.returncode) if int(exc.returncode) > 0 else 1
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_outputs(run_dir, state)
        print(f"[x] Q6-A Stage 3 failed after {format_time(time.perf_counter() - started)}: {exc}")
        print(f"Saved failure summary to: {run_dir / 'summary.json'}")
        return 1


def build_initial_state(timestamp: str, run_dir: Path, args: argparse.Namespace, phase1_dir: Path) -> dict[str, Any]:
    return {
        "run_id": run_dir.name,
        "timestamp_utc": timestamp,
        "status": "running",
        "contract": {
            "path": str(CONTRACT_PATH),
            "sha256": CONTRACT_SHA256,
        },
        "parameters": {
            "phase1_dir": str(phase1_dir),
            "output_dir": str(run_dir),
            "anchor_variant": "E",
            "candidate_variant": "B",
            "iteration": 24,
            "top_patterns": 128,
            "window_step_bits": 500_000,
            "effective_band_span_bits": 50_000_000,
            "lags": LAGS,
            "null_models": ["phase-matched-lz", "matched-lz"],
            "null_seeds": NULL_SEEDS,
            "bands": BANDS,
            "workers": args.workers,
            "tertiles": TERTILES,
            "skip_large_source_hash": bool(args.skip_large_source_hash),
        },
        "input_verification": "pending",
        "tertile_runs": [],
        "per_cell_readout": [],
        "per_tertile_readout": [],
        "per_band_readout": [],
        "sequential_stop_trace": [],
        "derivability_failures": [],
        "overall_outcome": None,
        "thermodynamic_stop_point": None,
        "not_run_after_stop": [],
    }


def reset_derived_state_for_resume(state: dict[str, Any]) -> None:
    """Keep completed child summaries, then rebuild derived readouts."""
    state["status"] = "resumed"
    state["per_cell_readout"] = []
    state["per_tertile_readout"] = []
    state["per_band_readout"] = []
    state["sequential_stop_trace"] = []
    state["derivability_failures"] = []
    state["overall_outcome"] = None
    state["thermodynamic_stop_point"] = None
    state["not_run_after_stop"] = []
    state.pop("error", None)


def verify_inputs(*, skip_large_source_hash: bool, quiet: bool) -> None:
    expected = {str(CONTRACT_PATH): CONTRACT_SHA256}
    expected.update(STAGE1_HASHES)
    if not skip_large_source_hash:
        expected.update(SOURCE_HASHES)
    for path_text, expected_hash in expected.items():
        path = Path(path_text)
        phase_print("Verifying input hash", str(path), quiet=quiet)
        actual = sha256_file(path)
        if actual.upper() != expected_hash.upper():
            raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected_hash}, got {actual}")


def execute_stage3(*, args: argparse.Namespace, phase1_dir: Path, run_dir: Path, state: dict[str, Any]) -> None:
    consecutive_catchup = 0
    for band_start in BANDS:
        if state.get("thermodynamic_stop_point") is not None:
            state["not_run_after_stop"].append(
                {"band_start_bits": band_start, "status": "not-run-after-thermodynamic-stop"}
            )
            continue

        band_cells: list[dict[str, Any]] = []
        band_tertiles: list[dict[str, Any]] = []
        for tertile_label, offset_index, count in TERTILES:
            tertile_start = band_start + offset_index * 500_000
            summary_path = run_or_reuse_tertile(
                args=args,
                phase1_dir=phase1_dir,
                run_dir=run_dir,
                band_start=band_start,
                tertile_label=tertile_label,
                tertile_start=tertile_start,
                window_count=count,
                state=state,
            )
            child_summary = load_json(summary_path)
            source_rows = child_summary.get("source_rows", [])
            if not source_rows:
                state["derivability_failures"].append(
                    {
                        "band_start_bits": band_start,
                        "band": band_label(band_start),
                        "tertile": tertile_label,
                        "tertile_start_bits": tertile_start,
                        "summary_path": str(summary_path),
                        "reason": "no-source-rows",
                        "target_statuses": summarize_target_statuses(child_summary.get("targets", [])),
                    }
                )
                state["overall_outcome"] = "inputs-or-run-not-derivable"
                write_state(run_dir, state)
                return
            tertile_cells = build_cells_from_source_rows(
                source_rows,
                band_start=band_start,
                tertile_label=tertile_label,
                tertile_start=tertile_start,
            )
            band_cells.extend(tertile_cells)
            band_tertiles.extend(classify_tertiles(tertile_cells, band_start=band_start, tertile_label=tertile_label))
            state["per_cell_readout"].extend(tertile_cells)
            state["per_tertile_readout"].extend(band_tertiles[-2:])
            write_state(run_dir, state)

        band_rows = classify_band(band_tertiles, band_start=band_start)
        state["per_band_readout"].extend(band_rows)
        negative = next(row for row in band_rows if row["channel"] == "negative")
        if negative["classification"] == "null-mimicry-or-asymptotic-wall":
            consecutive_catchup += 1
        else:
            consecutive_catchup = 0
        state["sequential_stop_trace"].append(
            {
                "band_start_bits": band_start,
                "band": band_label(band_start),
                "negative_classification": negative["classification"],
                "consecutive_null_mimicry": consecutive_catchup,
                "stop_triggered_after_band": consecutive_catchup >= 3,
            }
        )
        if consecutive_catchup >= 3:
            first_index = len(state["sequential_stop_trace"]) - 3
            state["thermodynamic_stop_point"] = state["sequential_stop_trace"][first_index]["band_start_bits"]
        write_state(run_dir, state)

    state["overall_outcome"] = build_overall_outcome(state)


def run_or_reuse_tertile(
    *,
    args: argparse.Namespace,
    phase1_dir: Path,
    run_dir: Path,
    band_start: int,
    tertile_label: str,
    tertile_start: int,
    window_count: int,
    state: dict[str, Any],
) -> Path:
    target_key = f"{band_start}:{tertile_label}"
    for item in state.get("tertile_runs", []):
        if item.get("target_key") == target_key and Path(item["summary_path"]).is_file():
            return Path(item["summary_path"])

    output_dir = run_dir / "lag_response_runs" / f"band-{compact_int(band_start)}" / tertile_label
    command = build_lag_response_command(
        args=args,
        phase1_dir=phase1_dir,
        output_dir=output_dir,
        band_start=tertile_start,
        window_count=window_count,
    )
    phase_print(
        "Q6-A Stage 3 tertile",
        (
            f"band={band_label(band_start)} | {tertile_label} | "
            f"start={compact_int(tertile_start)} | windows={window_count}"
        ),
        quiet=args.quiet,
    )
    subprocess.run(command, cwd=str(SCRIPT_DIR), check=True)
    summary_path = latest_child_file(output_dir, "summary.json")
    state["tertile_runs"].append(
        {
            "target_key": target_key,
            "band_start_bits": band_start,
            "band": band_label(band_start),
            "tertile": tertile_label,
            "tertile_start_bits": tertile_start,
            "window_count": window_count,
            "command": [str(part) for part in command],
            "summary_path": str(summary_path),
            "run_dir": str(summary_path.parent),
        }
    )
    return summary_path


def build_lag_response_command(
    *,
    args: argparse.Namespace,
    phase1_dir: Path,
    output_dir: Path,
    band_start: int,
    window_count: int,
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "hsi_v2_phase2_child_routing_lag_response.py"),
        "--phase1-dir",
        str(phase1_dir),
        "--output-dir",
        str(output_dir),
        "--stage",
        "all",
        "--anchor-variant",
        "E",
        "--candidate-variant",
        "B",
        "--iteration",
        "24",
        "--segment-bits",
        "1000000",
        "--num-segments",
        "3",
        "--scales",
        "8,12,16,20,24,28,32,40,48",
        "--phase1-policies",
        "prefix,suffix",
        "--low-scale",
        "40",
        "--high-scale",
        "48",
        "--top-patterns",
        "128",
        "--pattern-selection",
        "bridge-linked",
        "--band-starts",
        str(band_start),
        "--window-count",
        str(window_count),
        "--window-step-bits",
        "500000",
        "--probe-backward-bits",
        "45000000",
        "--probe-forward-bits",
        "45000000",
        "--scan-step-bits",
        "500000",
        f"--lags={','.join(str(item) for item in LAGS)}",
        "--null-models",
        "phase-matched-lz,matched-lz",
        "--matched-lz-seeds",
        ",".join(str(item) for item in NULL_SEEDS),
        "--workers",
        str(args.workers),
        "--quiet-children",
    ] + (["--quiet"] if args.quiet else [])


def build_all_planned_commands(args: argparse.Namespace, phase1_dir: Path, run_dir: Path) -> list[list[str]]:
    commands: list[list[str]] = []
    for band_start in BANDS:
        for tertile_label, offset_index, count in TERTILES:
            commands.append(
                [
                    str(part)
                    for part in build_lag_response_command(
                        args=args,
                        phase1_dir=phase1_dir,
                        output_dir=run_dir / "lag_response_runs" / f"band-{compact_int(band_start)}" / tertile_label,
                        band_start=band_start + offset_index * 500_000,
                        window_count=count,
                    )
                ]
            )
    return commands


def build_cells_from_source_rows(
    source_rows: list[dict[str, Any]],
    *,
    band_start: int,
    tertile_label: str,
    tertile_start: int,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[int(row["lag_bits"])].append(row)

    cells: list[dict[str, Any]] = []
    for lag_bits in sorted(grouped):
        rows = grouped[lag_bits]
        b_value = observed_retention(rows, "B")
        e_value = observed_retention(rows, "E")
        matched_values = null_values(rows, "matched-lz")
        phase_values = null_values(rows, "phase-matched-lz")
        if b_value is None or e_value is None or not matched_values or not phase_values:
            raise RuntimeError(
                f"Incomplete B/E/null coverage for {band_label(band_start)} {tertile_label} lag={lag_bits}"
            )
        observed_floor = min(b_value, e_value)
        observed_gap = abs(b_value - e_value)
        phase_mean = statistics.mean(phase_values)
        phase_std = statistics.stdev(phase_values) if len(phase_values) > 1 else 0.0
        phase_max = max(phase_values)
        matched_max = max(matched_values)
        hard_null_ceiling = max(matched_max, phase_max)
        phase_margin_to_max = observed_floor - phase_max
        hard_margin = observed_floor - hard_null_ceiling
        if phase_std == 0:
            if observed_floor > phase_mean:
                phase_z = math.inf
            elif observed_floor < phase_mean:
                phase_z = -math.inf
            else:
                phase_z = 0.0
        else:
            phase_z = (observed_floor - phase_mean) / phase_std

        significant = (
            observed_floor >= 0.90
            and phase_z > 3.0
            and phase_margin_to_max >= 0.02
            and hard_margin >= 0.02
        )
        non_significant = phase_z <= 1.96 and phase_margin_to_max <= 0.02
        b_strong = b_value >= 0.90 and (b_value - hard_null_ceiling) >= 0.02
        e_strong = e_value >= 0.90 and (e_value - hard_null_ceiling) >= 0.02
        asymmetric = (b_strong != e_strong) or observed_gap > 0.10

        cells.append(
            {
                "band": band_label(band_start),
                "band_start_bits": band_start,
                "tertile": tertile_label,
                "tertile_start_bits": tertile_start,
                "channel": "negative" if lag_bits < 0 else "positive",
                "lag_bits": lag_bits,
                "lag": compact_int(lag_bits),
                "B_retention": b_value,
                "E_retention": e_value,
                "matched_lz_values": ";".join(f"{value:.12g}" for value in matched_values),
                "phase_matched_lz_values": ";".join(f"{value:.12g}" for value in phase_values),
                "observed_floor": observed_floor,
                "observed_gap": observed_gap,
                "phase_mean": phase_mean,
                "phase_std": phase_std,
                "phase_max": phase_max,
                "matched_max": matched_max,
                "hard_null_ceiling": hard_null_ceiling,
                "phase_margin_to_max": phase_margin_to_max,
                "hard_margin": hard_margin,
                "phase_z": phase_z,
                "significant_separation_cell": significant,
                "non_significant_cell": non_significant,
                "asymmetric_cell": asymmetric,
            }
        )
    return cells


def classify_tertiles(cells: list[dict[str, Any]], *, band_start: int, tertile_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for channel in ("negative", "positive"):
        channel_cells = [cell for cell in cells if cell["channel"] == channel]
        significant = any(bool(cell["significant_separation_cell"]) for cell in channel_cells)
        asymmetric = any(bool(cell["asymmetric_cell"]) for cell in channel_cells)
        best_phase_z = max(float_or_low(cell["phase_z"]) for cell in channel_cells)
        best_phase_margin = max(float_or_low(cell["phase_margin_to_max"]) for cell in channel_cells)
        best_hard_margin = max(float_or_low(cell["hard_margin"]) for cell in channel_cells)
        non_significant = (not significant) and best_phase_z <= 1.96 and best_phase_margin <= 0.02
        rows.append(
            {
                "band": band_label(band_start),
                "band_start_bits": band_start,
                "tertile": tertile_label,
                "channel": channel,
                "significant_separation_tertile": significant,
                "non_significant_tertile": non_significant,
                "asymmetric_tertile": asymmetric,
                "best_phase_z": best_phase_z,
                "best_phase_margin_to_max": best_phase_margin,
                "best_hard_margin": best_hard_margin,
            }
        )
    return rows


def classify_band(tertile_rows: list[dict[str, Any]], *, band_start: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for channel in ("negative", "positive"):
        channel_rows = [row for row in tertile_rows if row["channel"] == channel]
        significant_count = sum(bool(row["significant_separation_tertile"]) for row in channel_rows)
        non_significant_count = sum(bool(row["non_significant_tertile"]) for row in channel_rows)
        asymmetric_count = sum(bool(row["asymmetric_tertile"]) for row in channel_rows)
        if asymmetric_count >= 1:
            classification = "B-E-asymmetric"
        elif significant_count >= 2:
            classification = "symmetric-hard-support"
        elif non_significant_count >= 2:
            classification = "null-mimicry-or-asymptotic-wall"
        else:
            classification = "no-forward-support"
        rows.append(
            {
                "band": band_label(band_start),
                "band_start_bits": band_start,
                "channel": channel,
                "significant_tertiles": significant_count,
                "non_significant_tertiles": non_significant_count,
                "asymmetric_tertiles": asymmetric_count,
                "classification": classification,
            }
        )
    return rows


def build_overall_outcome(state: dict[str, Any]) -> str:
    if state.get("derivability_failures"):
        return "inputs-or-run-not-derivable"
    negative_rows = [row for row in state["per_band_readout"] if row["channel"] == "negative"]
    if state.get("thermodynamic_stop_point") is not None:
        return "q6a-p4at24-null-mimicry-or-asymptotic-wall"
    if any(row["classification"] == "B-E-asymmetric" for row in negative_rows):
        return "q6a-p4at24-BE-asymmetric"
    support_count = sum(row["classification"] == "symmetric-hard-support" for row in negative_rows)
    catchup_count = sum(row["classification"] == "null-mimicry-or-asymptotic-wall" for row in negative_rows)
    local = [row for row in negative_rows if int(row["band_start_bits"]) in BANDS[:3]]
    high = [row for row in negative_rows if int(row["band_start_bits"]) in BANDS[3:]]
    local_support = sum(row["classification"] == "symmetric-hard-support" for row in local)
    high_support = sum(row["classification"] == "symmetric-hard-support" for row in high)
    if support_count >= 4:
        return "q6a-p4at24-symmetric-forward-support"
    if local_support >= 2 and high_support < 2:
        return "q6a-p4at24-local-only-support"
    if catchup_count >= 4:
        return "q6a-p4at24-null-mimicry-or-asymptotic-wall"
    return "q6a-p4at24-mixed-or-low-power"


def summarize_target_statuses(targets: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for target in targets:
        status = str(target.get("status", "missing-status"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def observed_retention(rows: list[dict[str, Any]], variant: str) -> float | None:
    selected = [
        row for row in rows
        if str(row.get("variant", "")).upper() == variant
        and str(row.get("source_label", "")) == "observed"
    ]
    if not selected:
        return None
    return pooled_retention(selected)


def null_values(rows: list[dict[str, Any]], null_model: str) -> list[float]:
    by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = str(row.get("source_label", ""))
        model = str(row.get("null_model", ""))
        if null_model not in source and null_model not in model:
            continue
        seed = str(row.get("null_seed", ""))
        by_seed[seed].append(row)
    return [pooled_retention(seed_rows) for _, seed_rows in sorted(by_seed.items()) if seed_rows]


def pooled_retention(rows: list[dict[str, Any]]) -> float:
    overlap = sum(parse_float(row.get("child_overlap_mass_sum")) for row in rows)
    anchor = sum(parse_float(row.get("anchor_event_total_sum")) for row in rows)
    if anchor <= 0:
        return math.nan
    return overlap / anchor


def write_outputs(run_dir: Path, state: dict[str, Any]) -> None:
    write_state(run_dir, state)
    write_json(run_dir / "summary.json", state)
    write_csv(run_dir / "per_cell_readout.csv", state.get("per_cell_readout", []), PER_CELL_FIELDS)
    write_csv(run_dir / "per_tertile_readout.csv", state.get("per_tertile_readout", []), PER_TERTILE_FIELDS)
    write_csv(run_dir / "per_band_readout.csv", state.get("per_band_readout", []), PER_BAND_FIELDS)
    write_csv(run_dir / "sequential_stop_trace.csv", state.get("sequential_stop_trace", []), STOP_FIELDS)
    write_csv(run_dir / "lag_response.csv", build_lag_response_rows(state), LAG_RESPONSE_FIELDS)
    (run_dir / "report.md").write_text(render_report(state, run_dir), encoding="utf-8")
    manifest = build_manifest(run_dir, state)
    write_json(run_dir / "manifest.json", manifest)


def build_lag_response_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in state.get("per_cell_readout", []):
        grouped[(cell["band"], int(cell["lag_bits"]), cell["channel"])].append(cell)
    for (band, lag_bits, channel), cells in sorted(grouped.items(), key=lambda item: (band_start_from_label(item[0][0]), item[0][1])):
        rows.append(
            {
                "band": band,
                "lag_bits": lag_bits,
                "lag": compact_int(lag_bits),
                "channel": channel,
                "B_retention_avg_tertiles": statistics.mean(float(cell["B_retention"]) for cell in cells),
                "E_retention_avg_tertiles": statistics.mean(float(cell["E_retention"]) for cell in cells),
                "observed_floor_avg_tertiles": statistics.mean(float(cell["observed_floor"]) for cell in cells),
                "phase_max_max_tertiles": max(float(cell["phase_max"]) for cell in cells),
                "matched_max_max_tertiles": max(float(cell["matched_max"]) for cell in cells),
                "hard_margin_min_tertiles": min(float(cell["hard_margin"]) for cell in cells),
                "phase_z_min_tertiles": min(float_or_low(cell["phase_z"]) for cell in cells),
            }
        )
    return rows


def build_manifest(run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    outputs = {
        "summary": run_dir / "summary.json",
        "report": run_dir / "report.md",
        "manifest": run_dir / "manifest.json",
        "lag_response": run_dir / "lag_response.csv",
        "per_cell_readout": run_dir / "per_cell_readout.csv",
        "per_tertile_readout": run_dir / "per_tertile_readout.csv",
        "per_band_readout": run_dir / "per_band_readout.csv",
        "sequential_stop_trace": run_dir / "sequential_stop_trace.csv",
    }
    output_hashes = {
        key: sha256_file(path)
        for key, path in outputs.items()
        if path.name != "manifest.json" and path.exists()
    }
    return {
        "script": Path(__file__).name,
        "run_dir": str(run_dir),
        "contract_sha256": CONTRACT_SHA256,
        "parameters": state.get("parameters", {}),
        "overall_outcome": state.get("overall_outcome"),
        "thermodynamic_stop_point": state.get("thermodynamic_stop_point"),
        "tertile_runs": state.get("tertile_runs", []),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "output_sha256": output_hashes,
    }


def render_report(state: dict[str, Any], run_dir: Path) -> str:
    lines = [
        "# Q6-A Stage 3 P4@24 symmetric forecast",
        "",
        f"- Status: `{state.get('status')}`",
        f"- Overall outcome: `{state.get('overall_outcome')}`",
        f"- Thermodynamic stop point: `{state.get('thermodynamic_stop_point')}`",
        f"- Contract SHA-256: `{CONTRACT_SHA256}`",
        f"- Run directory: `{run_dir}`",
        "",
        "## Band Readout",
        "",
        "| Band | Channel | Significant tertiles | Non-significant tertiles | Asymmetric tertiles | Classification |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in state.get("per_band_readout", []):
        lines.append(
            f"| {row['band']} | {row['channel']} | {row['significant_tertiles']} | "
            f"{row['non_significant_tertiles']} | {row['asymmetric_tertiles']} | "
            f"{row['classification']} |"
        )
    if state.get("derivability_failures"):
        lines.extend(["", "## Derivability Failures", ""])
        for failure in state["derivability_failures"]:
            lines.append(
                "- "
                f"{failure['band']} {failure['tertile']}: `{failure['reason']}`; "
                f"target statuses = `{failure.get('target_statuses', {})}`; "
                f"summary = `{failure['summary_path']}`."
            )
    lines.extend(
        [
            "",
            "## Stop Trace",
            "",
            "| Band | Negative classification | Consecutive null mimicry | Stop triggered |",
            "|---|---|---:|---|",
        ]
    )
    for row in state.get("sequential_stop_trace", []):
        lines.append(
            f"| {row['band']} | {row['negative_classification']} | "
            f"{row['consecutive_null_mimicry']} | {row['stop_triggered_after_band']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a Phase 4 diagnostic extension only. It does not authorize any "
            "manuscript edit, Level 2 claim, or geometric interpretation.",
            "",
        ]
    )
    return "\n".join(lines)


def render_console_summary(state: dict[str, Any]) -> str:
    lines = [
        "Q6-A Stage 3 P4@24 symmetric forecast",
        "-" * 88,
        f"status: {state.get('status')}",
        f"outcome: {state.get('overall_outcome')}",
        f"thermodynamic_stop_point: {state.get('thermodynamic_stop_point')}",
        "",
        f"{'band':<18} {'channel':<9} {'sig':>3} {'nonsig':>6} {'asym':>4} {'class':<34}",
    ]
    for row in state.get("per_band_readout", []):
        lines.append(
            f"{row['band']:<18} {row['channel']:<9} {row['significant_tertiles']:>3} "
            f"{row['non_significant_tertiles']:>6} {row['asymmetric_tertiles']:>4} "
            f"{row['classification']:<34}"
        )
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def parse_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def float_or_low(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return -1e9
    if math.isnan(parsed):
        return -1e9
    return parsed


def band_label(start_bits: int) -> str:
    return f"{start_bits // 1_000_000}M-{(start_bits + 50_000_000) // 1_000_000}M"


def band_start_from_label(label: str) -> int:
    return int(str(label).split("-", 1)[0].removesuffix("M")) * 1_000_000


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(run_dir: Path, state: dict[str, Any]) -> None:
    write_json(run_dir / "stage3_state.json", state)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


PER_CELL_FIELDS = [
    "band",
    "band_start_bits",
    "tertile",
    "tertile_start_bits",
    "channel",
    "lag_bits",
    "lag",
    "B_retention",
    "E_retention",
    "matched_lz_values",
    "phase_matched_lz_values",
    "observed_floor",
    "observed_gap",
    "phase_mean",
    "phase_std",
    "phase_max",
    "matched_max",
    "hard_null_ceiling",
    "phase_margin_to_max",
    "hard_margin",
    "phase_z",
    "significant_separation_cell",
    "non_significant_cell",
    "asymmetric_cell",
]

PER_TERTILE_FIELDS = [
    "band",
    "band_start_bits",
    "tertile",
    "channel",
    "significant_separation_tertile",
    "non_significant_tertile",
    "asymmetric_tertile",
    "best_phase_z",
    "best_phase_margin_to_max",
    "best_hard_margin",
]

PER_BAND_FIELDS = [
    "band",
    "band_start_bits",
    "channel",
    "significant_tertiles",
    "non_significant_tertiles",
    "asymmetric_tertiles",
    "classification",
]

STOP_FIELDS = [
    "band_start_bits",
    "band",
    "negative_classification",
    "consecutive_null_mimicry",
    "stop_triggered_after_band",
]

LAG_RESPONSE_FIELDS = [
    "band",
    "lag_bits",
    "lag",
    "channel",
    "B_retention_avg_tertiles",
    "E_retention_avg_tertiles",
    "observed_floor_avg_tertiles",
    "phase_max_max_tertiles",
    "matched_max_max_tertiles",
    "hard_margin_min_tertiles",
    "phase_z_min_tertiles",
]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
