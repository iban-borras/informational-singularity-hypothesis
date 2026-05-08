#!/usr/bin/env python3
"""HSI v2 Phase 2 N2-12 phase-channel width scout.

This launcher does not define a new observable. It expands a fine lag grid
around selected N2-11 phase channels and delegates the measurement to the
audited child-routing lag-response pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hsi_v2_phase2_child_routing_lag_response import (
    DEFAULT_BAND_STARTS,
    fmt_float,
)
from hsi_v2_phase2_parent_survival_revalidation import (
    append_flag,
    flush_output,
    latest_child_file,
    load_json,
    run_subprocess,
)
from hsi_v2_phase2_transport_defect_strict import parse_int_list, phase_print
from v2.common.cli import resolve_dir
from v2.common.naming import compact_int


DEFAULT_CHANNEL_CENTERS = "-27000000,26500000"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase2/phase_channel_width_n2_12"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run N2-12: a fine lag-grid width scout around selected N2-11 "
            "phase channels, using N2-11 as the audited computational unit."
        )
    )
    parser.add_argument(
        "--profile",
        choices=("canonical-scout", "canonical-full"),
        default="canonical-scout",
        help=(
            "canonical-scout defaults to observed-only; canonical-full defaults "
            "to observed plus nulls."
        ),
    )
    parser.add_argument("--phase1-dir", default="results/hsi_v2/phase1_high_scales")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stage",
        choices=("auto", "observed", "nulls", "all"),
        default="auto",
        help="Use auto to let the profile choose observed/all.",
    )
    parser.add_argument("--top-patterns", type=int, default=128)
    parser.add_argument("--band-starts", default=DEFAULT_BAND_STARTS)
    parser.add_argument("--window-count", type=int, default=19)
    parser.add_argument("--window-step-bits", type=int, default=500_000)
    parser.add_argument("--channel-centers", default=DEFAULT_CHANNEL_CENTERS)
    parser.add_argument("--radius-bits", type=int, default=3_000_000)
    parser.add_argument("--lag-step-bits", type=int, default=500_000)
    parser.add_argument("--probe-backward-bits", type=int, default=45_000_000)
    parser.add_argument("--probe-forward-bits", type=int, default=45_000_000)
    parser.add_argument("--scan-step-bits", type=int, default=500_000)
    parser.add_argument("--null-models", default="")
    parser.add_argument("--matched-lz-seeds", default="")
    parser.add_argument("--b-threshold", type=float, default=0.90)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel lag targets per band in the delegated N2-11 lag-response run.",
    )
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verbose-children",
        action="store_true",
        help="Show full logs from delegated N2-11 child scripts.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)

    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage = resolve_stage(args)
    centers = parse_int_list(args.channel_centers, label="--channel-centers", allow_zero=True)
    lags = expand_lags(
        centers=centers,
        radius_bits=args.radius_bits,
        step_bits=args.lag_step_bits,
    )
    band_starts = parse_int_list(args.band_starts, label="--band-starts", allow_zero=True)
    target_count = len(band_starts) * len(lags)
    if args.max_targets:
        target_count = min(target_count, args.max_targets)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(args=args, stage=stage, centers=centers, lags=lags, timestamp=timestamp)
    run_dir = output_dir / run_slug
    n2_11_root = run_dir / "lag_response"
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing N2-12 phase-channel width scout",
        (
            f"profile={args.profile} | stage={stage} | top={args.top_patterns} | "
            f"bands={len(band_starts)} | lags={len(lags)} | targets={target_count}"
        ),
        quiet=args.quiet,
    )
    flush_output()

    command = build_n2_11_command(
        args=args,
        stage=stage,
        phase1_dir=phase1_dir,
        output_dir=n2_11_root,
        lags=lags,
    )

    if args.dry_run:
        n2_11_summary_path = None
        n2_11_run_dir = None
        response_rows: list[dict[str, Any]] = []
    else:
        phase_print(
            "Running delegated N2-11 lag-response grid",
            (
                f"channels={','.join(compact_int(value) for value in centers)} | "
                f"radius={compact_int(args.radius_bits)} | step={compact_int(args.lag_step_bits)}"
            ),
            quiet=args.quiet,
        )
        flush_output()
        run_subprocess(command, cwd=Path(__file__).resolve().parent)
        n2_11_summary_path = latest_child_file(n2_11_root, "summary.json")
        n2_11_run_dir = n2_11_summary_path.parent
        response_rows = load_json(n2_11_summary_path).get("lag_response", [])

    channel_rows = build_channel_rows(
        response_rows=response_rows,
        centers=centers,
        radius_bits=args.radius_bits,
    )
    width_rows = summarize_widths(
        channel_rows=channel_rows,
        centers=centers,
        b_threshold=args.b_threshold,
        margin_threshold=args.margin_threshold,
    )

    summary = {
        "run_id": run_slug,
        "timestamp_utc": timestamp,
        "parameters": {
            "profile": args.profile,
            "stage": stage,
            "phase1_dir": str(phase1_dir),
            "top_patterns": args.top_patterns,
            "band_starts": args.band_starts,
            "window_count": args.window_count,
            "window_step_bits": args.window_step_bits,
            "channel_centers": centers,
            "radius_bits": args.radius_bits,
            "lag_step_bits": args.lag_step_bits,
            "lags": lags,
            "null_models": args.null_models,
            "matched_lz_seeds": args.matched_lz_seeds,
            "b_threshold": args.b_threshold,
            "margin_threshold": args.margin_threshold,
            "workers": args.workers,
            "dry_run": args.dry_run,
            "max_targets": args.max_targets,
        },
        "n2_11_run_dir": str(n2_11_run_dir) if n2_11_run_dir else None,
        "n2_11_summary_path": str(n2_11_summary_path) if n2_11_summary_path else None,
        "channel_width": width_rows,
        "channel_response": channel_rows,
    }
    manifest = {
        "script": Path(__file__).name,
        "run_dir": str(run_dir),
        "delegated_observable": "hsi_v2_phase2_child_routing_lag_response.py",
        "delegated_command": command,
        "outputs": {
            "summary": str(run_dir / "summary.json"),
            "report": str(run_dir / "report.md"),
            "channel_width_csv": str(run_dir / "channel_width.csv"),
            "channel_response_csv": str(run_dir / "channel_response.csv"),
            "manifest": str(run_dir / "manifest.json"),
        },
    }

    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "manifest.json", manifest)
    write_csv(run_dir / "channel_width.csv", width_rows, CHANNEL_WIDTH_FIELDS)
    write_csv(run_dir / "channel_response.csv", channel_rows, CHANNEL_RESPONSE_FIELDS)
    (run_dir / "report.md").write_text(render_report(summary), encoding="utf-8")

    print(render_console_summary(width_rows, dry_run=args.dry_run, target_count=target_count))
    print(f"\nSaved summary to: {run_dir / 'summary.json'}")
    print(f"Saved report to: {run_dir / 'report.md'}")
    print(f"Saved channel width CSV to: {run_dir / 'channel_width.csv'}")
    print(f"Saved channel response CSV to: {run_dir / 'channel_response.csv'}")
    print(f"Saved manifest to: {run_dir / 'manifest.json'}")
    flush_output()
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.top_patterns <= 0:
        raise SystemExit("--top-patterns must be positive")
    if args.window_count <= 0:
        raise SystemExit("--window-count must be positive")
    if args.window_step_bits <= 0:
        raise SystemExit("--window-step-bits must be positive")
    if args.radius_bits < 0:
        raise SystemExit("--radius-bits must be non-negative")
    if args.lag_step_bits <= 0:
        raise SystemExit("--lag-step-bits must be positive")
    if args.scan_step_bits <= 0:
        raise SystemExit("--scan-step-bits must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.max_targets < 0:
        raise SystemExit("--max-targets must be non-negative")


def resolve_stage(args: argparse.Namespace) -> str:
    if args.stage != "auto":
        return args.stage
    return "all" if args.profile == "canonical-full" else "observed"


def expand_lags(*, centers: list[int], radius_bits: int, step_bits: int) -> list[int]:
    values: set[int] = set()
    for center in centers:
        start = center - radius_bits
        stop = center + radius_bits
        current = start
        while current <= stop:
            values.add(current)
            current += step_bits
        values.add(center)
    return sorted(values)


def build_n2_11_command(
    *,
    args: argparse.Namespace,
    stage: str,
    phase1_dir: Path,
    output_dir: Path,
    lags: list[int],
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "hsi_v2_phase2_child_routing_lag_response.py"),
        "--stage",
        stage,
        "--phase1-dir",
        str(phase1_dir),
        "--output-dir",
        str(output_dir),
        "--top-patterns",
        str(args.top_patterns),
        "--band-starts",
        args.band_starts,
        "--window-count",
        str(args.window_count),
        "--window-step-bits",
        str(args.window_step_bits),
        "--lags=" + ",".join(str(value) for value in lags),
        "--probe-backward-bits",
        str(args.probe_backward_bits),
        "--probe-forward-bits",
        str(args.probe_forward_bits),
        "--scan-step-bits",
        str(args.scan_step_bits),
    ]
    if args.null_models:
        command.extend(["--null-models", args.null_models])
    if args.matched_lz_seeds:
        command.extend(["--matched-lz-seeds", args.matched_lz_seeds])
    command.extend(["--workers", str(args.workers)])
    if args.max_targets:
        command.extend(["--max-targets", str(args.max_targets)])
    append_flag(command, "--dry-run", args.dry_run)
    append_flag(command, "--quiet-children", not args.verbose_children)
    append_flag(command, "--quiet", args.quiet)
    return command


def build_channel_rows(
    *,
    response_rows: list[dict[str, Any]],
    centers: list[int],
    radius_bits: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in response_rows:
        lag_bits = int(row["lag_bits"])
        center = nearest_center(lag_bits, centers)
        delta = lag_bits - center
        if abs(delta) > radius_bits:
            continue
        b_retention = as_float(row.get("B_retention"))
        null_family, null_n, matched_max, margin = selected_seeded_lz_null(row)
        rows.append(
            {
                "band": row["band"],
                "channel_center_bits": center,
                "channel_center": compact_int(center),
                "lag_bits": lag_bits,
                "lag": compact_int(lag_bits),
                "delta_from_center_bits": delta,
                "delta_from_center": compact_int(delta),
                "B_retention": b_retention,
                "markov1_retention": as_float(row.get("markov1_retention")),
                "null_family": null_family,
                "null_family_n": null_n,
                "matched_lz_max": matched_max,
                "B_minus_matched_lz_max": margin,
            }
        )
    return sorted(
        rows,
        key=lambda item: (band_start(str(item["band"])), item["channel_center_bits"], item["lag_bits"]),
    )


def selected_seeded_lz_null(row: dict[str, Any]) -> tuple[str, int, float | None, float | None]:
    phase_n = int(as_float(row.get("phase_matched_lz_n")) or 0)
    phase_max = as_float(row.get("phase_matched_lz_max"))
    if phase_n > 0 and phase_max is not None:
        return (
            "phase_matched_lz",
            phase_n,
            phase_max,
            as_float(row.get("B_minus_phase_matched_lz_max")),
        )
    matched_n = int(as_float(row.get("matched_lz_n")) or 0)
    return (
        "matched_lz",
        matched_n,
        as_float(row.get("matched_lz_max")),
        as_float(row.get("B_minus_matched_lz_max")),
    )


def summarize_widths(
    *,
    channel_rows: list[dict[str, Any]],
    centers: list[int],
    b_threshold: float,
    margin_threshold: float,
) -> list[dict[str, Any]]:
    bands = sorted({str(row["band"]) for row in channel_rows}, key=band_start)
    output: list[dict[str, Any]] = []
    for band in bands:
        for center in centers:
            rows = [
                row
                for row in channel_rows
                if row["band"] == band and int(row["channel_center_bits"]) == center
            ]
            if not rows:
                continue
            best_b = max(rows, key=lambda row: value_or_low(row.get("B_retention")))
            margin_rows = [row for row in rows if row.get("B_minus_matched_lz_max") is not None]
            best_margin = (
                max(margin_rows, key=lambda row: value_or_low(row.get("B_minus_matched_lz_max")))
                if margin_rows
                else None
            )
            b_pass = [row for row in rows if value_or_low(row.get("B_retention")) >= b_threshold]
            margin_pass = [
                row
                for row in margin_rows
                if value_or_low(row.get("B_minus_matched_lz_max")) >= margin_threshold
            ]
            output.append(
                {
                    "band": band,
                    "channel_center_bits": center,
                    "channel_center": compact_int(center),
                    "lag_count": len(rows),
                    "best_B_lag_bits": best_b["lag_bits"],
                    "best_B_lag": best_b["lag"],
                    "best_B_retention": best_b["B_retention"],
                    "best_margin_lag_bits": best_margin["lag_bits"] if best_margin else None,
                    "best_margin_lag": best_margin["lag"] if best_margin else None,
                    "best_margin": best_margin["B_minus_matched_lz_max"] if best_margin else None,
                    "B_ge_threshold_lag_min": min_lag(b_pass),
                    "B_ge_threshold_lag_max": max_lag(b_pass),
                    "B_ge_threshold_width_bits": width_bits(b_pass),
                    "margin_ge_threshold_lag_min": min_lag(margin_pass),
                    "margin_ge_threshold_lag_max": max_lag(margin_pass),
                    "margin_ge_threshold_width_bits": width_bits(margin_pass),
                }
            )
    return output


def nearest_center(lag_bits: int, centers: list[int]) -> int:
    return min(centers, key=lambda center: (abs(lag_bits - center), center))


def band_start(label: str) -> int:
    return int(label.split("M", 1)[0])


def as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def value_or_low(value: Any) -> float:
    return float(value) if value is not None else -1e9


def min_lag(rows: list[dict[str, Any]]) -> int | None:
    return min((int(row["lag_bits"]) for row in rows), default=None)


def max_lag(rows: list[dict[str, Any]]) -> int | None:
    return max((int(row["lag_bits"]) for row in rows), default=None)


def width_bits(rows: list[dict[str, Any]]) -> int | None:
    if not rows:
        return None
    return max_lag(rows) - min_lag(rows)


def build_run_slug(
    *,
    args: argparse.Namespace,
    stage: str,
    centers: list[int],
    lags: list[int],
    timestamp: str,
) -> str:
    center_part = "-".join(compact_int(value) for value in centers)
    return (
        f"phase2-phase-channel-width__profile-{args.profile}"
        f"__stage-{stage}"
        f"__top-{args.top_patterns}"
        f"__centers-{center_part}"
        f"__radius-{compact_int(args.radius_bits)}"
        f"__step-{compact_int(args.lag_step_bits)}"
        f"__lags-{len(lags)}"
        f"__{timestamp}"
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_console_summary(rows: list[dict[str, Any]], *, dry_run: bool, target_count: int) -> str:
    lines = [
        "Phase 2 N2-12 phase-channel width scout",
        "-" * 118,
    ]
    if dry_run:
        lines.append(f"Dry run planned {target_count} delegated N2-11 targets.")
        return "\n".join(lines)
    lines.append(
        f"{'band':<11} {'center':>8} {'best_B':>8} {'B_ret':>8} "
        f"{'best_margin':>12} {'margin':>8} {'B_width':>9} {'M_width':>9}"
    )
    if not rows:
        lines.append("No width rows were emitted.")
        return "\n".join(lines)
    for row in rows:
        lines.append(
            f"{row['band']:<11} {row['channel_center']:>8} "
            f"{row['best_B_lag']:>8} {fmt_float(row['best_B_retention']):>8} "
            f"{str(row['best_margin_lag'] or '-'):>12} "
            f"{fmt_float(row['best_margin']):>8} "
            f"{compact_optional(row['B_ge_threshold_width_bits']):>9} "
            f"{compact_optional(row['margin_ge_threshold_width_bits']):>9}"
        )
    return "\n".join(lines)


def render_report(summary: dict[str, Any]) -> str:
    params = summary["parameters"]
    lines = [
        "# HSI v2 Phase 2 N2-12 Phase-Channel Width Scout",
        "",
        "This run expands a fine lag grid around selected N2-11 phase channels and delegates measurement to the audited N2-11 lag-response pipeline.",
        "",
        "## Parameters",
        "",
        f"- Profile: `{params['profile']}`",
        f"- Stage: `{params['stage']}`",
        f"- Top patterns: `{params['top_patterns']}`",
        f"- Channel centers: `{', '.join(compact_int(value) for value in params['channel_centers'])}`",
        f"- Radius: `{compact_int(params['radius_bits'])}`",
        f"- Lag step: `{compact_int(params['lag_step_bits'])}`",
        f"- B threshold: `{params['b_threshold']}`",
        f"- Margin threshold: `{params['margin_threshold']}`",
        f"- Delegated N2-11 run: `{summary['n2_11_run_dir'] or '-'}`",
        "",
        "## Width Summary",
        "",
        "| Band | Center | Best B lag | B retention | Best margin lag | Margin | B width | Margin width |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["channel_width"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["band"]),
                    str(row["channel_center"]),
                    str(row["best_B_lag"]),
                    fmt_float(row["best_B_retention"]),
                    str(row["best_margin_lag"] or "-"),
                    fmt_float(row["best_margin"]),
                    compact_optional(row["B_ge_threshold_width_bits"]),
                    compact_optional(row["margin_ge_threshold_width_bits"]),
                ]
            )
            + " |"
        )
    if not summary["channel_width"]:
        lines.append("| - | - | - | - | - | - | - | - |")
    return "\n".join(lines) + "\n"


def compact_optional(value: int | None) -> str:
    return "-" if value is None else compact_int(int(value))


CHANNEL_WIDTH_FIELDS = [
    "band",
    "channel_center_bits",
    "channel_center",
    "lag_count",
    "best_B_lag_bits",
    "best_B_lag",
    "best_B_retention",
    "best_margin_lag_bits",
    "best_margin_lag",
    "best_margin",
    "B_ge_threshold_lag_min",
    "B_ge_threshold_lag_max",
    "B_ge_threshold_width_bits",
    "margin_ge_threshold_lag_min",
    "margin_ge_threshold_lag_max",
    "margin_ge_threshold_width_bits",
]

CHANNEL_RESPONSE_FIELDS = [
    "band",
    "channel_center_bits",
    "channel_center",
    "lag_bits",
    "lag",
    "delta_from_center_bits",
    "delta_from_center",
    "B_retention",
    "markov1_retention",
    "null_family",
    "null_family_n",
    "matched_lz_max",
    "B_minus_matched_lz_max",
]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
