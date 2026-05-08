#!/usr/bin/env python3
"""HSI v2 Phase 4 P4-06 boundary feature-shift readout.

P4-06 is a read-only diagnostic over the already accepted P4-04/P4-05
artifacts. It compares the last supported forward band with the first
null-envelope catch-up band, decomposing the hard-margin collapse into B,
phase-matched-LZ, and block-entropy components.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.progress import format_time
from v2.common.cli import resolve_dir


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_P4_04_RUN = (
    "results/hsi_v2/phase4/p4_04_forward_regime_forecast/"
    "phase4-p4-04-forward-regime-forecast__bands-3__lags-6__20260507T200920"
)
DEFAULT_P4_05_RUN = (
    "results/hsi_v2/phase4/p4_05_horizon_bracket_forecast/"
    "phase4-p4-05-horizon-bracket-forecast__bands-4__lags-6__20260507T225929"
)
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase4/p4_06_boundary_feature_shift"
SUPPORTED_BAND = "786M-795M"
BOUNDARY_BAND = "795M-804M"
CHANNEL = "negative"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read P4-04/P4-05 artifacts and decompose the transition from "
            "the last supported band to the first null-envelope catch-up band."
        )
    )
    parser.add_argument("--p4-04-run", default=DEFAULT_P4_04_RUN)
    parser.add_argument("--p4-05-run", default=DEFAULT_P4_05_RUN)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    p4_04_run = resolve_dir(args.p4_04_run, anchor_file=__file__)
    p4_05_run = resolve_dir(args.p4_05_run, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_slug = f"phase4-p4-06-boundary-feature-shift__{SUPPORTED_BAND}_to_{BOUNDARY_BAND}__{timestamp}"
    run_dir = output_root / sanitize_slug(run_slug)
    run_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "run_id": run_dir.name,
        "timestamp_utc": timestamp,
        "status": "running",
        "parameters": {
            "p4_04_run": str(p4_04_run),
            "p4_05_run": str(p4_05_run),
            "supported_band": SUPPORTED_BAND,
            "boundary_band": BOUNDARY_BAND,
            "channel": CHANNEL,
            "output_dir": str(output_root),
        },
        "source_files": {},
        "best_row_comparison": {},
        "lag_aligned_rows": [],
        "aggregate_shift": {},
        "verdict": {},
    }

    started = time.perf_counter()
    try:
        phase_print(
            "Preparing P4-06 boundary feature-shift readout",
            f"{SUPPORTED_BAND} -> {BOUNDARY_BAND} | channel={CHANNEL}",
            quiet=args.quiet,
        )

        p4_04_summary = p4_04_run / "summary.json"
        p4_05_summary = p4_05_run / "summary.json"
        p4_04_lag_csv = find_single_file(p4_04_run, "lag_response.csv")
        p4_05_lag_csv = find_single_file(p4_05_run, "lag_response.csv")
        state["source_files"] = {
            "p4_04_summary": str(p4_04_summary.resolve()),
            "p4_05_summary": str(p4_05_summary.resolve()),
            "p4_04_lag_response_csv": str(p4_04_lag_csv.resolve()),
            "p4_05_lag_response_csv": str(p4_05_lag_csv.resolve()),
        }

        p4_04 = load_json(p4_04_summary)
        p4_05 = load_json(p4_05_summary)
        p4_04_rows = load_csv_dicts(p4_04_lag_csv)
        p4_05_rows = load_csv_dicts(p4_05_lag_csv)

        supported_best = find_forecast_row(p4_04, SUPPORTED_BAND, CHANNEL)
        boundary_best = find_forecast_row(p4_05, BOUNDARY_BAND, CHANNEL)
        aligned_rows = build_lag_aligned_rows(p4_04_rows, p4_05_rows)
        aggregate_shift = build_aggregate_shift(aligned_rows)
        best_comparison = compare_best_rows(supported_best, boundary_best)
        verdict = classify_boundary(best_comparison, aggregate_shift)

        state.update(
            {
                "status": "completed",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "best_row_comparison": best_comparison,
                "lag_aligned_rows": aligned_rows,
                "aggregate_shift": aggregate_shift,
                "verdict": verdict,
            }
        )
        write_outputs(run_dir, state)
        print(render_console_summary(state))
        print_saved_paths(run_dir)
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        state["error"] = str(exc)
        write_outputs(run_dir, state)
        print(f"[x] P4-06 failed after {format_time(time.perf_counter() - started)}: {exc}")
        print(f"Saved failure summary to: {run_dir / 'summary.json'}")
        return 1


def find_forecast_row(summary: dict[str, Any], band: str, channel: str) -> dict[str, Any]:
    for row in summary.get("forecast_rows", []):
        if row.get("band") == band and row.get("channel") == channel:
            return row
    raise KeyError(f"Missing forecast row for {band} / {channel}")


def build_lag_aligned_rows(p4_04_rows: list[dict[str, str]], p4_05_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    before = {
        int(row["lag_bits"]): row
        for row in p4_04_rows
        if row.get("band") == SUPPORTED_BAND and int(row["lag_bits"]) < 0
    }
    after = {
        int(row["lag_bits"]): row
        for row in p4_05_rows
        if row.get("band") == BOUNDARY_BAND and int(row["lag_bits"]) < 0
    }
    rows: list[dict[str, Any]] = []
    for lag_bits in sorted(set(before) & set(after)):
        left = before[lag_bits]
        right = after[lag_bits]
        left_metrics = extract_lag_metrics(left)
        right_metrics = extract_lag_metrics(right)
        rows.append(
            {
                "lag_bits": lag_bits,
                "lag": left.get("lag") or right.get("lag"),
                **prefixed("supported", left_metrics),
                **prefixed("boundary", right_metrics),
                "delta_B_retention": right_metrics["B_retention"] - left_metrics["B_retention"],
                "delta_phase_matched_lz_max": right_metrics["phase_matched_lz_max"] - left_metrics["phase_matched_lz_max"],
                "delta_block_entropy_max": right_metrics["block_entropy_max"] - left_metrics["block_entropy_max"],
                "delta_hard_null_max": right_metrics["hard_null_max"] - left_metrics["hard_null_max"],
                "delta_hard_margin": right_metrics["hard_margin"] - left_metrics["hard_margin"],
            }
        )
    if not rows:
        raise ValueError("No overlapping negative lags between supported and boundary bands")
    return rows


def extract_lag_metrics(row: dict[str, str]) -> dict[str, float]:
    b_ret = parse_float(row["B_retention"])
    pmlz = parse_float(row["phase_matched_lz_max"])
    bent = parse_float(row["block_entropy_max"])
    hard_null = max(pmlz, bent)
    return {
        "B_retention": b_ret,
        "phase_matched_lz_max": pmlz,
        "block_entropy_max": bent,
        "hard_null_max": hard_null,
        "hard_margin": b_ret - hard_null,
    }


def compare_best_rows(supported: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    supported_hard_null = max(float(supported["best_phase_matched_lz_max"]), float(supported["best_block_entropy_max"]))
    boundary_hard_null = max(float(boundary["best_phase_matched_lz_max"]), float(boundary["best_block_entropy_max"]))
    return {
        "supported_band": supported["band"],
        "boundary_band": boundary["band"],
        "supported_best_lag": supported["best_lag"],
        "boundary_best_lag": boundary["best_lag"],
        "supported_B_retention": float(supported["best_B_retention"]),
        "boundary_B_retention": float(boundary["best_B_retention"]),
        "delta_B_retention": float(boundary["best_B_retention"]) - float(supported["best_B_retention"]),
        "supported_phase_matched_lz_max": float(supported["best_phase_matched_lz_max"]),
        "boundary_phase_matched_lz_max": float(boundary["best_phase_matched_lz_max"]),
        "delta_phase_matched_lz_max": float(boundary["best_phase_matched_lz_max"]) - float(supported["best_phase_matched_lz_max"]),
        "supported_block_entropy_max": float(supported["best_block_entropy_max"]),
        "boundary_block_entropy_max": float(boundary["best_block_entropy_max"]),
        "delta_block_entropy_max": float(boundary["best_block_entropy_max"]) - float(supported["best_block_entropy_max"]),
        "supported_hard_null_max": supported_hard_null,
        "boundary_hard_null_max": boundary_hard_null,
        "delta_hard_null_max": boundary_hard_null - supported_hard_null,
        "supported_hard_margin": float(supported["best_hard_margin"]),
        "boundary_hard_margin": float(boundary["best_hard_margin"]),
        "delta_hard_margin": float(boundary["best_hard_margin"]) - float(supported["best_hard_margin"]),
        "supported_observed_regime": supported["observed_regime"],
        "boundary_observed_regime": boundary["observed_regime"],
    }


def build_aggregate_shift(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "delta_B_retention",
        "delta_phase_matched_lz_max",
        "delta_block_entropy_max",
        "delta_hard_null_max",
        "delta_hard_margin",
        "supported_hard_margin",
        "boundary_hard_margin",
        "supported_B_retention",
        "boundary_B_retention",
        "supported_phase_matched_lz_max",
        "boundary_phase_matched_lz_max",
        "supported_block_entropy_max",
        "boundary_block_entropy_max",
    ]
    output = {"lag_count": float(len(rows))}
    for key in keys:
        values = [float(row[key]) for row in rows]
        output[f"{key}_mean"] = sum(values) / len(values)
        output[f"{key}_min"] = min(values)
        output[f"{key}_max"] = max(values)
    return output


def classify_boundary(best: dict[str, Any], aggregate: dict[str, float]) -> dict[str, Any]:
    b_drop = -float(best["delta_B_retention"])
    null_rise = float(best["delta_hard_null_max"])
    margin_loss = -float(best["delta_hard_margin"])
    pmlz_rise = float(best["delta_phase_matched_lz_max"])
    bent_rise = float(best["delta_block_entropy_max"])

    if margin_loss <= 0:
        status = "no-boundary-loss"
        driver = "none"
    elif null_rise > max(0.25, b_drop * 2):
        status = "null-envelope-catch-up"
        driver = "hard-null-envelope-dominated-by-phase-matched-lz" if pmlz_rise >= bent_rise else "hard-null-envelope-dominated-by-block-entropy"
    elif b_drop > max(0.25, null_rise):
        status = "B-retention-loss"
        driver = "B"
    else:
        status = "mixed-boundary-shift"
        driver = "mixed"

    return {
        "status": status,
        "primary_driver": driver,
        "margin_loss": margin_loss,
        "best_row_B_drop": b_drop,
        "best_row_hard_null_rise": null_rise,
        "best_row_phase_matched_lz_rise": pmlz_rise,
        "best_row_block_entropy_rise": bent_rise,
        "aggregate_hard_margin_loss_mean": -float(aggregate["delta_hard_margin_mean"]),
        "short_reading": (
            "The P4-04/P4-05 boundary is driven by a surge in the hard-null "
            "envelope, dominated by phase-matched-LZ, not by disappearance "
            "of B retention."
            if status == "null-envelope-catch-up"
            else "The P4-04/P4-05 boundary is mixed or not dominated by null-envelope catch-up."
        ),
    }


def write_outputs(run_dir: Path, state: dict[str, Any]) -> None:
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    rows_path = run_dir / "lag_aligned_shift.csv"
    best_path = run_dir / "best_row_shift.csv"
    manifest_path = run_dir / "manifest.json"

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    write_csv(state.get("lag_aligned_rows", []), rows_path)
    write_csv([state.get("best_row_comparison", {})] if state.get("best_row_comparison") else [], best_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown_report(state) + "\n")

    manifest = {
        "run_id": state["run_id"],
        "timestamp_utc": state["timestamp_utc"],
        "status": state["status"],
        "script": "hsi_v2_phase4_p4_06_boundary_feature_shift.py",
        "parameters": state["parameters"],
        "source_files": state["source_files"],
        "outputs": {
            "summary": str(summary_path.resolve()),
            "report": str(report_path.resolve()),
            "lag_aligned_shift_csv": str(rows_path.resolve()),
            "best_row_shift_csv": str(best_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        },
    }
    if "error" in state:
        manifest["error"] = state["error"]
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def render_console_summary(state: dict[str, Any]) -> str:
    best = state.get("best_row_comparison", {})
    verdict = state.get("verdict", {})
    lines = [
        "Phase 4 P4-06 boundary feature-shift",
        "-" * 108,
        f"status={verdict.get('status', '-')}; driver={verdict.get('primary_driver', '-')}",
    ]
    if best:
        lines.append(
            "best-row shift | "
            f"B {best['supported_B_retention']:.4f}->{best['boundary_B_retention']:.4f} "
            f"| pMLZ {best['supported_phase_matched_lz_max']:.4f}->{best['boundary_phase_matched_lz_max']:.4f} "
            f"| block {best['supported_block_entropy_max']:.4f}->{best['boundary_block_entropy_max']:.4f} "
            f"| margin {best['supported_hard_margin']:.4f}->{best['boundary_hard_margin']:.4f}"
        )
    return "\n".join(lines)


def render_markdown_report(state: dict[str, Any]) -> str:
    best = state.get("best_row_comparison", {})
    aggregate = state.get("aggregate_shift", {})
    verdict = state.get("verdict", {})
    lines = [
        "# HSI v2 Phase 4 P4-06 Boundary Feature-Shift",
        "",
        "P4-06 is a read-only diagnostic over accepted P4-04/P4-05 artifacts.",
        "It does not extend the forecast horizon and does not scan for renewed signal.",
        "",
        "## Verdict",
        "",
        f"- Status: `{verdict.get('status', '-')}`",
        f"- Primary driver: `{verdict.get('primary_driver', '-')}`",
        f"- Reading: {verdict.get('short_reading', '-')}",
        "",
        "## Best-Row Boundary Shift",
        "",
        "| Metric | Last supported band | First boundary band | Delta |",
        "|---|---:|---:|---:|",
    ]
    if best:
        lines.extend(
            [
                f"| Best lag | `{best['supported_best_lag']}` | `{best['boundary_best_lag']}` | - |",
                f"| B retention | {best['supported_B_retention']:.4f} | {best['boundary_B_retention']:.4f} | {best['delta_B_retention']:+.4f} |",
                f"| phase-matched-LZ max | {best['supported_phase_matched_lz_max']:.4f} | {best['boundary_phase_matched_lz_max']:.4f} | {best['delta_phase_matched_lz_max']:+.4f} |",
                f"| block-entropy max | {best['supported_block_entropy_max']:.4f} | {best['boundary_block_entropy_max']:.4f} | {best['delta_block_entropy_max']:+.4f} |",
                f"| hard null max | {best['supported_hard_null_max']:.4f} | {best['boundary_hard_null_max']:.4f} | {best['delta_hard_null_max']:+.4f} |",
                f"| hard margin | {best['supported_hard_margin']:.4f} | {best['boundary_hard_margin']:.4f} | {best['delta_hard_margin']:+.4f} |",
            ]
        )
    lines.extend(["", "## Lag-Aligned Aggregate Shift", ""])
    if aggregate:
        lines.extend(
            [
                f"- Negative lag count compared: `{int(aggregate['lag_count'])}`",
                f"- Mean B-retention shift: `{aggregate['delta_B_retention_mean']:+.4f}`",
                f"- Mean phase-matched-LZ max shift: `{aggregate['delta_phase_matched_lz_max_mean']:+.4f}`",
                f"- Mean block-entropy max shift: `{aggregate['delta_block_entropy_max_mean']:+.4f}`",
                f"- Mean hard-null max shift: `{aggregate['delta_hard_null_max_mean']:+.4f}`",
                f"- Mean hard-margin shift: `{aggregate['delta_hard_margin_mean']:+.4f}`",
            ]
        )
    lines.extend(["", "## Source Artifacts", ""])
    for key, value in state.get("source_files", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines)


def find_single_file(root: Path, name: str) -> Path:
    candidates = sorted(root.rglob(name))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one {name} under {root}, found {len(candidates)}")
    return candidates[0]


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_dicts(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def parse_float(value: str) -> float:
    if value is None or str(value).strip() == "":
        return 0.0
    return float(value)


def sanitize_slug(value: str) -> str:
    return value.replace(">", "to").replace("/", "-").replace(" ", "_")


def phase_print(title: str, detail: str = "", quiet: bool = False) -> None:
    if quiet:
        return
    print(f"\n[Phase] {title}")
    if detail:
        print(f"        {detail}")
    sys.stdout.flush()


def print_saved_paths(run_dir: Path) -> None:
    print(f"\nSaved summary to: {run_dir / 'summary.json'}")
    print(f"Saved report to: {run_dir / 'report.md'}")
    print(f"Saved lag-aligned CSV to: {run_dir / 'lag_aligned_shift.csv'}")
    print(f"Saved best-row CSV to: {run_dir / 'best_row_shift.csv'}")
    print(f"Saved manifest to: {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
