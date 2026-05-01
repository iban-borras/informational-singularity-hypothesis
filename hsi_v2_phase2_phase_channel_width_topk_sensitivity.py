#!/usr/bin/env python3
"""HSI v2 Phase 2 N2-12b top-k sensitivity for phase-channel width."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hsi_v2_phase2_parent_survival_revalidation import (
    append_flag,
    flush_output,
    latest_child_file,
    run_subprocess,
)
from hsi_v2_phase2_transport_defect_strict import parse_int_list, phase_print
from utils.progress import format_time
from v2.common.cli import resolve_dir
from v2.common.naming import compact_int


DEFAULT_BANDS = "696000000,705000000,714000000,723000000,726000000,729000000"
DEFAULT_LAGS = "-30000000,-29500000,-27000000,-26500000,-26000000,-24000000,23500000,24500000,26500000,29500000"
DEFAULT_REFERENCE_ROOT = "results/hsi_v2/phase2/phase_channel_width_n2_12_full_reval"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase2/phase_channel_width_topk_sensitivity"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run N2-12b top-k sensitivity by reusing the audited N2-11/N2-12 "
            "lag-response contract over selected top-k resolutions."
        )
    )
    parser.add_argument("--phase1-dir", default="results/hsi_v2/phase1_high_scales")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage", choices=("observed", "nulls", "all"), default="all")
    parser.add_argument(
        "--top-patterns-list",
        default="64",
        help="Top-k values to run now. The canonical top-128 run is read as reference by default.",
    )
    parser.add_argument("--reference-top", type=int, default=128)
    parser.add_argument("--reference-run", default="")
    parser.add_argument("--reference-root", default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--band-starts", default=DEFAULT_BANDS)
    parser.add_argument("--window-count", type=int, default=19)
    parser.add_argument("--window-step-bits", type=int, default=500_000)
    parser.add_argument("--lags", default=DEFAULT_LAGS)
    parser.add_argument("--b-threshold", type=float, default=0.90)
    parser.add_argument("--margin-threshold", type=float, default=0.30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--verbose-children",
        action="store_true",
        help="Show full logs from delegated child scripts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)

    top_values = parse_int_list(args.top_patterns_list, label="--top-patterns-list", allow_zero=False)
    lags = parse_int_list(args.lags, label="--lags", allow_zero=True)
    band_starts = parse_int_list(args.band_starts, label="--band-starts", allow_zero=True)

    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_run = resolve_reference_run(args)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    top_label_values = [args.reference_top, *top_values]
    run_slug = (
        "phase2-phase-channel-width-topk-sensitivity"
        f"__stage-{args.stage}"
        f"__top-{min(top_label_values)}-{max(top_label_values)}x{len(top_label_values)}"
        f"__lags-{len(lags)}"
        f"__{timestamp}"
    )
    run_dir = output_dir / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing N2-12b phase-channel width top-k sensitivity",
        (
            f"stage={args.stage} | run_top={','.join(str(v) for v in top_values)} | "
            f"reference_top={args.reference_top} | targets_per_top={len(band_starts) * len(lags)}"
        ),
        quiet=args.quiet,
    )
    flush_output()

    commands: list[dict[str, Any]] = []
    run_paths: dict[int, Path] = {args.reference_top: reference_run}
    started_at = time.perf_counter()
    total_top_runs = len(top_values)
    completed_top_runs = 0

    for top in top_values:
        top_output = run_dir / f"top-{top}" / "lag_response"
        command = build_lag_response_command(
            args=args,
            phase1_dir=phase1_dir,
            output_dir=top_output,
            top_patterns=top,
        )
        commands.append({"top_patterns": top, "command": command})
        if args.dry_run:
            continue
        phase_print(
            "Running delegated N2-12 lag grid",
            f"top={top} | lags={len(lags)} | bands={len(band_starts)}",
            quiet=args.quiet,
        )
        flush_output()
        run_subprocess(command, cwd=Path(__file__).resolve().parent)
        run_paths[top] = latest_child_file(top_output, "summary.json").parent
        completed_top_runs += 1
        elapsed = max(0.0, time.perf_counter() - started_at)
        if completed_top_runs < total_top_runs:
            eta_seconds = (elapsed / completed_top_runs) * (total_top_runs - completed_top_runs)
            eta_text = format_time(eta_seconds)
        else:
            eta_text = "0s"
        phase_print(
            "Top-k sensitivity progress",
            (
                f"{completed_top_runs}/{total_top_runs} | "
                f"{(completed_top_runs / total_top_runs) * 100.0:5.1f}% | "
                f"elapsed={format_time(elapsed)} | ETA ~{eta_text} | top={top}"
            ),
            quiet=args.quiet,
        )
        flush_output()

    response_rows: list[dict[str, Any]] = []
    for top, path in sorted(run_paths.items()):
        if path is None:
            continue
        response_rows.extend(load_response_rows(top=top, run_dir=path))

    missing_rows = build_missing_rows(
        response_rows,
        top_values=[args.reference_top, *top_values],
        band_starts=band_starts,
        lags=lags,
        window_count=args.window_count,
        window_step_bits=args.window_step_bits,
    )
    envelope_rows = build_envelopes(
        response_rows,
        b_threshold=args.b_threshold,
        margin_threshold=args.margin_threshold,
    )
    comparison_rows = build_comparison_rows(
        response_rows,
        reference_top=args.reference_top,
        test_tops=top_values,
    )

    summary = {
        "run_id": run_slug,
        "timestamp_utc": timestamp,
        "parameters": {
            "stage": args.stage,
            "phase1_dir": str(phase1_dir),
            "top_patterns_list": top_values,
            "reference_top": args.reference_top,
            "reference_run": str(reference_run),
            "band_starts": args.band_starts,
            "window_count": args.window_count,
            "window_step_bits": args.window_step_bits,
            "lags": args.lags,
            "b_threshold": args.b_threshold,
            "margin_threshold": args.margin_threshold,
            "dry_run": args.dry_run,
        },
        "run_paths": {str(top): str(path) for top, path in sorted(run_paths.items())},
        "channel_envelopes": envelope_rows,
        "topk_comparison": comparison_rows,
        "missing_cells": missing_rows,
        "response_rows": response_rows,
    }
    manifest = {
        "script": Path(__file__).name,
        "run_dir": str(run_dir),
        "reference_run": str(reference_run),
        "commands": commands,
        "outputs": {
            "summary": str(run_dir / "summary.json"),
            "report": str(run_dir / "report.md"),
            "topk_comparison_csv": str(run_dir / "topk_comparison.csv"),
            "channel_envelopes_csv": str(run_dir / "channel_envelopes.csv"),
            "missing_cells_csv": str(run_dir / "missing_cells.csv"),
            "response_csv": str(run_dir / "response_rows.csv"),
            "manifest": str(run_dir / "manifest.json"),
        },
    }

    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "manifest.json", manifest)
    write_csv(run_dir / "response_rows.csv", response_rows, RESPONSE_FIELDS)
    write_csv(run_dir / "channel_envelopes.csv", envelope_rows, ENVELOPE_FIELDS)
    write_csv(run_dir / "topk_comparison.csv", comparison_rows, COMPARISON_FIELDS)
    write_csv(run_dir / "missing_cells.csv", missing_rows, MISSING_FIELDS)
    (run_dir / "report.md").write_text(render_report(summary), encoding="utf-8")

    print(render_console_summary(envelope_rows, comparison_rows, missing_rows, dry_run=args.dry_run))
    print(f"\nSaved summary to: {run_dir / 'summary.json'}")
    print(f"Saved report to: {run_dir / 'report.md'}")
    print(f"Saved comparison CSV to: {run_dir / 'topk_comparison.csv'}")
    print(f"Saved envelopes CSV to: {run_dir / 'channel_envelopes.csv'}")
    print(f"Saved missing cells CSV to: {run_dir / 'missing_cells.csv'}")
    print(f"Saved manifest to: {run_dir / 'manifest.json'}")
    flush_output()
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.window_count <= 0:
        raise SystemExit("--window-count must be positive")
    if args.window_step_bits <= 0:
        raise SystemExit("--window-step-bits must be positive")
    if args.reference_top <= 0:
        raise SystemExit("--reference-top must be positive")


def resolve_reference_run(args: argparse.Namespace) -> Path:
    if args.reference_run.strip():
        path = resolve_dir(args.reference_run, anchor_file=__file__)
        if not (path / "lag_response.csv").is_file():
            raise SystemExit(f"Reference run does not contain lag_response.csv: {path}")
        return path
    root = resolve_dir(args.reference_root, anchor_file=__file__)
    if not root.is_dir():
        raise SystemExit(f"Reference root not found: {root}")
    candidates = [
        child for child in root.iterdir()
        if child.is_dir()
        and child.name.startswith("phase2-child-routing-lag-response__")
        and (child / "lag_response.csv").is_file()
    ]
    if not candidates:
        raise SystemExit(f"No reference lag-response runs found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_lag_response_command(
    *,
    args: argparse.Namespace,
    phase1_dir: Path,
    output_dir: Path,
    top_patterns: int,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "hsi_v2_phase2_child_routing_lag_response.py"),
        "--stage",
        args.stage,
        "--phase1-dir",
        str(phase1_dir),
        "--output-dir",
        str(output_dir),
        "--top-patterns",
        str(top_patterns),
        "--band-starts",
        args.band_starts,
        "--window-count",
        str(args.window_count),
        "--window-step-bits",
        str(args.window_step_bits),
        "--lags=" + args.lags,
    ]
    append_flag(command, "--dry-run", args.dry_run)
    append_flag(command, "--quiet-children", not args.verbose_children)
    append_flag(command, "--quiet", args.quiet)
    return command


def load_response_rows(*, top: int, run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (run_dir / "lag_response.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item = {
                "top_patterns": top,
                "band": row["band"],
                "lag": row["lag"],
                "lag_bits": int(row["lag_bits"]),
                "channel": "negative" if int(row["lag_bits"]) < 0 else "positive",
                "B_retention": to_float(row["B_retention"]),
                "markov1_retention": to_float(row["markov1_retention"]),
                "matched_lz_max": to_float(row["matched_lz_max"]),
                "B_minus_matched_lz_max": to_float(row["B_minus_matched_lz_max"]),
            }
            rows.append(item)
    return sorted(rows, key=lambda row: (row["top_patterns"], band_start(row["band"]), row["lag_bits"]))


def build_envelopes(
    rows: list[dict[str, Any]],
    *,
    b_threshold: float,
    margin_threshold: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for top in sorted({int(row["top_patterns"]) for row in rows}):
        top_rows = [row for row in rows if int(row["top_patterns"]) == top]
        for band in sorted({row["band"] for row in top_rows}, key=band_start):
            for channel in ("negative", "positive"):
                channel_rows = [
                    row for row in top_rows
                    if row["band"] == band and row["channel"] == channel
                ]
                if not channel_rows:
                    continue
                strong = [
                    row for row in channel_rows
                    if row["B_retention"] is not None
                    and row["B_minus_matched_lz_max"] is not None
                    and row["B_retention"] >= b_threshold
                    and row["B_minus_matched_lz_max"] >= margin_threshold
                ]
                best = max(channel_rows, key=lambda row: value_or_low(row["B_minus_matched_lz_max"]))
                output.append(
                    {
                        "top_patterns": top,
                        "band": band,
                        "channel": channel,
                        "strong_count": len(strong),
                        "strong_lag_min_bits": min_lag(strong),
                        "strong_lag_max_bits": max_lag(strong),
                        "strong_width_bits": width_bits(strong),
                        "best_margin_lag_bits": best["lag_bits"],
                        "best_margin_lag": best["lag"],
                        "best_margin_B_retention": best["B_retention"],
                        "best_margin": best["B_minus_matched_lz_max"],
                    }
                )
    return output


def build_comparison_rows(
    rows: list[dict[str, Any]],
    *,
    reference_top: int,
    test_tops: list[int],
) -> list[dict[str, Any]]:
    by_cell = {
        (int(row["top_patterns"]), row["band"], int(row["lag_bits"])): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    reference_rows = [row for row in rows if int(row["top_patterns"]) == reference_top]
    for ref in reference_rows:
        for top in test_tops:
            test = by_cell.get((top, ref["band"], int(ref["lag_bits"])))
            if not test:
                continue
            output.append(
                {
                    "test_top": top,
                    "reference_top": reference_top,
                    "band": ref["band"],
                    "lag": ref["lag"],
                    "lag_bits": ref["lag_bits"],
                    "B_delta": delta(test["B_retention"], ref["B_retention"]),
                    "margin_delta": delta(
                        test["B_minus_matched_lz_max"],
                        ref["B_minus_matched_lz_max"],
                    ),
                    "test_B": test["B_retention"],
                    "reference_B": ref["B_retention"],
                    "test_margin": test["B_minus_matched_lz_max"],
                    "reference_margin": ref["B_minus_matched_lz_max"],
                }
            )
    return output


def build_missing_rows(
    rows: list[dict[str, Any]],
    *,
    top_values: list[int],
    band_starts: list[int],
    lags: list[int],
    window_count: int,
    window_step_bits: int,
) -> list[dict[str, Any]]:
    observed = {
        (int(row["top_patterns"]), row["band"], int(row["lag_bits"]))
        for row in rows
    }
    missing: list[dict[str, Any]] = []
    for top in sorted(set(top_values)):
        for start in band_starts:
            band = band_label(start, window_count=window_count, window_step_bits=window_step_bits)
            for lag in lags:
                key = (top, band, lag)
                if key in observed:
                    continue
                missing.append(
                    {
                        "top_patterns": top,
                        "band": band,
                        "lag": compact_int(lag),
                        "lag_bits": lag,
                        "channel": "negative" if lag < 0 else "positive",
                    }
                )
    return missing


def band_label(start_bits: int, *, window_count: int, window_step_bits: int) -> str:
    last_start = start_bits + (window_count - 1) * window_step_bits
    return f"{compact_int(start_bits)}-{compact_int(last_start)}"


def to_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def band_start(label: str) -> int:
    return int(label.split("M", 1)[0])


def value_or_low(value: float | None) -> float:
    return value if value is not None else -1e9


def min_lag(rows: list[dict[str, Any]]) -> int | None:
    return min((int(row["lag_bits"]) for row in rows), default=None)


def max_lag(rows: list[dict[str, Any]]) -> int | None:
    return max((int(row["lag_bits"]) for row in rows), default=None)


def width_bits(rows: list[dict[str, Any]]) -> int | None:
    if not rows:
        return None
    return max_lag(rows) - min_lag(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_console_summary(
    envelopes: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> str:
    lines = [
        "Phase 2 N2-12b top-k sensitivity",
        "-" * 112,
    ]
    if dry_run:
        lines.append("Dry run only. Inspect manifest for delegated commands.")
        return "\n".join(lines)
    lines.append(f"{'top':>5} {'band':<10} {'ch':<3} {'cells':>5} {'width':>8} {'best_lag':>9} {'margin':>8}")
    for row in envelopes:
        lines.append(
            f"{int(row['top_patterns']):>5} {row['band']:<10} {row['channel'][:3]:<3} "
            f"{int(row['strong_count']):>5} {compact_optional(row['strong_width_bits']):>8} "
            f"{row['best_margin_lag']:>9} {fmt(row['best_margin']):>8}"
        )
    if comparisons:
        margin_deltas = [row["margin_delta"] for row in comparisons if row["margin_delta"] is not None]
        if margin_deltas:
            lines.append("")
            lines.append(
                "margin_delta_vs_reference: "
                f"min={min(margin_deltas):+.4f} avg={sum(margin_deltas)/len(margin_deltas):+.4f} "
                f"max={max(margin_deltas):+.4f}"
            )
    if missing:
        missing_counts: dict[int, int] = {}
        for row in missing:
            top = int(row["top_patterns"])
            missing_counts[top] = missing_counts.get(top, 0) + 1
        lines.append("")
        lines.append(
            "missing_cells: "
            + ", ".join(f"top-{top}={count}" for top, count in sorted(missing_counts.items()))
        )
    return "\n".join(lines)


def render_report(summary: dict[str, Any]) -> str:
    params = summary["parameters"]
    lines = [
        "# Phase 2 N2-12b Top-k Sensitivity",
        "",
        "This run reuses the N2-12 lag grid and compares additional top-k resolutions against the canonical top-128 reference.",
        "",
        "## Parameters",
        "",
        f"- Stage: `{params['stage']}`",
        f"- Top-k values run: `{', '.join(str(v) for v in params['top_patterns_list'])}`",
        f"- Reference top-k: `{params['reference_top']}`",
        f"- Reference run: `{params['reference_run']}`",
        f"- Lags: `{params['lags']}`",
        "",
        "## Channel Envelopes",
        "",
        "| Top-k | Band | Channel | Strong cells | Width | Best margin lag | Best margin |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["channel_envelopes"]:
        lines.append(
            f"| {row['top_patterns']} | {row['band']} | {row['channel']} | "
            f"{row['strong_count']} | {compact_optional(row['strong_width_bits'])} | "
            f"{row['best_margin_lag']} | {fmt(row['best_margin'])} |"
        )
    lines.extend(
        [
            "",
            "## Comparison Against Reference",
            "",
            "| Test top-k | Band | Lag | B delta | Margin delta | Test margin | Reference margin |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["topk_comparison"]:
        lines.append(
            f"| {row['test_top']} | {row['band']} | {row['lag']} | "
            f"{fmt(row['B_delta'])} | {fmt(row['margin_delta'])} | "
            f"{fmt(row['test_margin'])} | {fmt(row['reference_margin'])} |"
        )
    lines.extend(
        [
            "",
            "## Missing Cells",
            "",
        ]
    )
    if not summary["missing_cells"]:
        lines.append("No missing cells under the expected top-k x band x lag grid.")
    else:
        counts: dict[int, int] = {}
        for row in summary["missing_cells"]:
            top = int(row["top_patterns"])
            counts[top] = counts.get(top, 0) + 1
        lines.append(", ".join(f"top-{top}: {count}" for top, count in sorted(counts.items())))
        lines.extend(
            [
                "",
                "| Top-k | Band | Lag | Channel |",
                "| ---: | --- | ---: | --- |",
            ]
        )
        for row in summary["missing_cells"]:
            lines.append(
                f"| {row['top_patterns']} | {row['band']} | {row['lag']} | {row['channel']} |"
            )
    return "\n".join(lines) + "\n"


def compact_optional(value: int | None) -> str:
    return "-" if value is None else compact_int(int(value))


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{float(value):.4f}"


RESPONSE_FIELDS = [
    "top_patterns",
    "band",
    "lag",
    "lag_bits",
    "channel",
    "B_retention",
    "markov1_retention",
    "matched_lz_max",
    "B_minus_matched_lz_max",
]

ENVELOPE_FIELDS = [
    "top_patterns",
    "band",
    "channel",
    "strong_count",
    "strong_lag_min_bits",
    "strong_lag_max_bits",
    "strong_width_bits",
    "best_margin_lag_bits",
    "best_margin_lag",
    "best_margin_B_retention",
    "best_margin",
]

COMPARISON_FIELDS = [
    "test_top",
    "reference_top",
    "band",
    "lag",
    "lag_bits",
    "B_delta",
    "margin_delta",
    "test_B",
    "reference_B",
    "test_margin",
    "reference_margin",
]

MISSING_FIELDS = [
    "top_patterns",
    "band",
    "lag",
    "lag_bits",
    "channel",
]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
