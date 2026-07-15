#!/usr/bin/env python3
"""HSI v2 Phase 2 N2-11 child-routing lag-response pipeline.

This wrapper does not introduce a new routing observable. It reuses the
audited child-destination routing stack and evaluates it over an explicit
lag grid, so boundary bands can be read as response profiles instead of as
single-lag decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hsi_v2_phase2_child_routing_persistence import (
    build_lagaware_cmd,
    build_offsets,
    build_probe_cmd,
    build_selection,
    write_synthetic_revalidation_summary,
)
from hsi_v2_phase2_parent_shell_atlas_sensitivity import ensure_required_phase1_inputs
from hsi_v2_phase2_parent_survival_revalidation import (
    append_flag,
    flush_output,
    latest_child_file,
    load_json,
    run_subprocess,
)
from hsi_v2_phase2_transport_defect_strict import parse_int_list, phase_print
from utils.progress import HeartbeatProgress, format_time
from v2.common.cli import resolve_dir, resolve_results_dir
from v2.common.naming import compact_int


DEFAULT_BAND_STARTS = (
    "696000000,705000000,714000000,723000000,726000000,729000000"
)
DEFAULT_LAGS = "-88000000,-27000000,-17000000,-14000000,-5500000,0,26500000,87000000"


def normalize_negative_csv_options(argv: list[str]) -> list[str]:
    """Allow natural CLI use such as `--lags -30000000,...`.

    `argparse` treats a value starting with '-' as a new option unless the user
    writes `--lags=-30000000,...`. For reproducible commands, accepting both
    forms is safer and avoids shell-specific surprises.
    """
    normalized: list[str] = []
    index = 0
    csv_options = {"--lags"}
    while index < len(argv):
        current = argv[index]
        if current in csv_options and index + 1 < len(argv):
            value = argv[index + 1]
            if value.startswith("-") and "," in value:
                normalized.append(f"{current}={value}")
                index += 2
                continue
        normalized.append(current)
        index += 1
    return normalized


def parse_lag_list(text: str) -> list[int]:
    values: list[int] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid lag value: {item!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one lag is required")
    return values


def lag_slug(lag_bits: int) -> str:
    if lag_bits < 0:
        return "neg-" + compact_int(abs(lag_bits))
    if lag_bits > 0:
        return "pos-" + compact_int(lag_bits)
    return "zero"


def band_label(start_bits: int, window_count: int, window_step_bits: int) -> str:
    stop_bits = start_bits + (window_count - 1) * window_step_bits
    return f"{compact_int(start_bits)}-{compact_int(stop_bits)}"


def estimated_lagaware_required_bits(target: dict[str, Any], args: argparse.Namespace) -> int:
    window_bits = int(args.segment_bits) * int(args.num_segments)
    max_offset = max(int(value) for value in target["offsets"])
    lag_bits = int(target["lag_bits"])
    required_bits = max_offset + max(window_bits, window_bits + max(lag_bits, 0))
    if lag_bits < 0:
        required_bits = max(required_bits, max_offset + window_bits)
    return int(required_bits)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def source_kind(row: dict[str, Any]) -> str:
    variant = str(row.get("variant", ""))
    source_label = str(row.get("source_label", row.get("source", "")))
    if variant == "B" and source_label == "observed":
        return "B_observed"
    if variant == "E" and source_label == "observed":
        return "E_observed"
    if "markov1" in source_label or variant.endswith("markov1"):
        return "markov1"
    if "phase-matched-lz" in source_label or variant.endswith("phase-matched-lz"):
        return "phase_matched_lz"
    if "block-entropy" in source_label or variant.endswith("block-entropy"):
        return "block_entropy"
    if "matched-lz" in source_label or variant.endswith("matched-lz"):
        return "matched_lz"
    return "other"


def metric_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_response(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[(str(row["band"]), int(row["lag_bits"]))].append(row)

    response_rows: list[dict[str, Any]] = []
    for (band, lag_bits), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        b_value: float | None = None
        e_value: float | None = None
        markov_value: float | None = None
        matched_values: list[float] = []
        phase_matched_values: list[float] = []
        block_entropy_values: list[float] = []
        for row in rows:
            retention = metric_float(row, "child_destination_retention_pooled")
            kind = source_kind(row)
            if retention is None:
                continue
            if kind == "B_observed":
                b_value = retention
            elif kind == "E_observed":
                e_value = retention
            elif kind == "markov1":
                markov_value = retention
            elif kind == "matched_lz":
                matched_values.append(retention)
            elif kind == "phase_matched_lz":
                phase_matched_values.append(retention)
            elif kind == "block_entropy":
                block_entropy_values.append(retention)

        matched_min = min(matched_values) if matched_values else None
        matched_max = max(matched_values) if matched_values else None
        matched_avg = sum(matched_values) / len(matched_values) if matched_values else None
        phase_matched_min = min(phase_matched_values) if phase_matched_values else None
        phase_matched_max = max(phase_matched_values) if phase_matched_values else None
        phase_matched_avg = (
            sum(phase_matched_values) / len(phase_matched_values)
            if phase_matched_values
            else None
        )
        block_entropy_min = min(block_entropy_values) if block_entropy_values else None
        block_entropy_max = max(block_entropy_values) if block_entropy_values else None
        block_entropy_avg = (
            sum(block_entropy_values) / len(block_entropy_values)
            if block_entropy_values
            else None
        )
        response_rows.append(
            {
                "band": band,
                "lag_bits": lag_bits,
                "lag": compact_int(lag_bits),
                "B_retention": b_value,
                "E_retention": e_value,
                "markov1_retention": markov_value,
                "matched_lz_min": matched_min,
                "matched_lz_avg": matched_avg,
                "matched_lz_max": matched_max,
                "matched_lz_n": len(matched_values),
                "B_minus_matched_lz_max": (
                    b_value - matched_max
                    if b_value is not None and matched_max is not None
                    else None
                ),
                "phase_matched_lz_min": phase_matched_min,
                "phase_matched_lz_avg": phase_matched_avg,
                "phase_matched_lz_max": phase_matched_max,
                "phase_matched_lz_n": len(phase_matched_values),
                "B_minus_phase_matched_lz_max": (
                    b_value - phase_matched_max
                    if b_value is not None and phase_matched_max is not None
                    else None
                ),
                "block_entropy_min": block_entropy_min,
                "block_entropy_avg": block_entropy_avg,
                "block_entropy_max": block_entropy_max,
                "block_entropy_n": len(block_entropy_values),
                "B_minus_block_entropy_max": (
                    b_value - block_entropy_max
                    if b_value is not None and block_entropy_max is not None
                    else None
                ),
            }
        )
    return response_rows


def fmt_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def run_subprocess_with_heartbeat(
    command: list[str],
    *,
    cwd: Path,
    quiet_children: bool,
    task_name: str,
    detail: str,
) -> None:
    if quiet_children:
        with HeartbeatProgress(task_name, interval=5.0, message=detail):
            run_subprocess(command, cwd=cwd)
    else:
        run_subprocess(command, cwd=cwd)


def progress_detail(
    *,
    completed: int,
    total: int,
    started_at: float,
    latest_band: str,
    latest_lag_bits: int,
    latest_status: str,
) -> str:
    elapsed = max(0.0, time.perf_counter() - started_at)
    pct = (completed / total) * 100.0 if total > 0 else 100.0
    if completed > 0 and completed < total:
        eta_seconds = (elapsed / completed) * (total - completed)
        eta_text = format_time(eta_seconds)
    elif completed >= total:
        eta_text = "0s"
    else:
        eta_text = "-"
    return (
        f"{completed}/{total} | {pct:5.1f}% | elapsed={format_time(elapsed)} | "
        f"ETA ~{eta_text} | latest={latest_band}:{compact_int(latest_lag_bits)} | "
        f"status={latest_status}"
    )


def render_console_summary(response_rows: list[dict[str, Any]]) -> str:
    lines = [
        "Phase 2 N2-11 child-routing lag-response profile",
        "-" * 144,
        (
            f"{'band':<12} {'lag':>9} {'B':>8} {'markov1':>8} "
            f"{'mlz_max':>8} {'B-mlz':>8} {'pmlz_max':>9} {'B-pmlz':>9} "
            f"{'bent_max':>9} {'B-bent':>9}"
        ),
    ]
    for row in response_rows:
        lines.append(
            f"{row['band']:<12} {row['lag']:>9} "
            f"{fmt_float(row.get('B_retention')):>8} "
            f"{fmt_float(row.get('markov1_retention')):>8} "
            f"{fmt_float(row.get('matched_lz_max')):>8} "
            f"{fmt_float(row.get('B_minus_matched_lz_max')):>8} "
            f"{fmt_float(row.get('phase_matched_lz_max')):>9} "
            f"{fmt_float(row.get('B_minus_phase_matched_lz_max')):>9} "
            f"{fmt_float(row.get('block_entropy_max')):>9} "
            f"{fmt_float(row.get('B_minus_block_entropy_max')):>9}"
        )
    return "\n".join(lines)


def render_report(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    target_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# HSI v2 Phase 2 N2-11 child-routing lag-response profile",
        "",
        "This run reuses the audited child-destination routing contract and sweeps explicit candidate lags.",
        "The goal is to read phase response directly, not to select a single lag by fiat.",
        "",
        "## Parameters",
        "",
        f"- Stage: `{args.stage}`",
        f"- Anchor variant: `{args.anchor_variant}`",
        f"- Candidate variant: `{args.candidate_variant}`",
        f"- Top patterns: `{args.top_patterns}`",
        f"- Window count: `{args.window_count}`",
        f"- Window step bits: `{args.window_step_bits}`",
        f"- Lags: `{args.lags}`",
        f"- Workers: `{args.workers}`",
        "- Execution order: `descending_estimated_required_bits_for_cache_warmth`",
        "",
        "## Response Summary",
        "",
        (
            "| Band | Lag | B retention | markov1 | matched-LZ max | B - max(matched-LZ) | "
            "phase-matched-LZ max | B - max(phase-matched-LZ) | block-entropy max | "
            "B - max(block-entropy) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in response_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["band"]),
                    str(row["lag"]),
                    fmt_float(row.get("B_retention")),
                    fmt_float(row.get("markov1_retention")),
                    fmt_float(row.get("matched_lz_max")),
                    fmt_float(row.get("B_minus_matched_lz_max")),
                    fmt_float(row.get("phase_matched_lz_max")),
                    fmt_float(row.get("B_minus_phase_matched_lz_max")),
                    fmt_float(row.get("block_entropy_max")),
                    fmt_float(row.get("B_minus_block_entropy_max")),
                ]
            )
            + " |"
        )
    if not response_rows:
        lines.append("| - | - | - | - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Targets",
            "",
            "| Band | Lag | Status |",
            "|---|---:|---|",
        ]
    )
    for row in target_rows:
        lines.append(f"| {row['band']} | {row['lag']} | {row['status']} |")

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Run directory: `{run_dir}`",
            "- Summary: `summary.json`",
            "- Response CSV: `lag_response.csv`",
            "- Source CSV: `source_response.csv`",
            "- Targets CSV: `targets.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run N2-11 child-routing lag-response profiles across bands and lags."
    )
    parser.add_argument("--phase1-dir", default="results/hsi_v2/phase1_high_scales")
    parser.add_argument("--output-dir", default="results/hsi_v2/phase2/child_routing_lag_response")
    parser.add_argument("--profile", default="lag-response-boundary")
    parser.add_argument("--stage", choices=["observed", "nulls", "all"], default="all")
    parser.add_argument("--anchor-variant", default="E")
    parser.add_argument("--candidate-variant", default="B")
    parser.add_argument("--iteration", type=int, default=20)
    parser.add_argument("--segment-bits", type=int, default=1_000_000)
    parser.add_argument("--num-segments", type=int, default=3)
    parser.add_argument("--scales", default="8,12,16,20,24,28,32,40,48")
    parser.add_argument("--phase1-policies", default="prefix,suffix")
    parser.add_argument("--low-scale", type=int, default=40)
    parser.add_argument("--high-scale", type=int, default=48)
    parser.add_argument("--top-patterns", type=int, default=128)
    parser.add_argument("--pattern-selection", default="bridge-linked")
    parser.add_argument("--band-starts", default=DEFAULT_BAND_STARTS)
    parser.add_argument("--window-count", type=int, default=19)
    parser.add_argument("--window-step-bits", type=int, default=500_000)
    parser.add_argument("--lags", default=DEFAULT_LAGS)
    parser.add_argument("--probe-backward-bits", type=int, default=45_000_000)
    parser.add_argument("--probe-forward-bits", type=int, default=45_000_000)
    parser.add_argument("--scan-step-bits", type=int, default=500_000)
    parser.add_argument("--null-models", default="")
    parser.add_argument("--matched-lz-seeds", default="")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel lag targets per band after cache warming. Keep low on memory-constrained machines.",
    )
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--quiet-children",
        action="store_true",
        help="Keep N2-11 progress visible while silencing delegated child scripts.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.window_count <= 0:
        raise SystemExit("--window-count must be positive")
    if args.window_step_bits <= 0:
        raise SystemExit("--window-step-bits must be positive")
    if args.top_patterns <= 0:
        raise SystemExit("--top-patterns must be positive")
    if args.scan_step_bits <= 0:
        raise SystemExit("--scan-step-bits must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")


def build_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    band_starts = parse_int_list(args.band_starts, label="--band-starts", allow_zero=True)
    lag_values = parse_lag_list(args.lags)
    targets: list[dict[str, Any]] = []
    for start_bits in band_starts:
        label = band_label(start_bits, args.window_count, args.window_step_bits)
        offsets = build_offsets(
            start_bits=start_bits,
            count=args.window_count,
            step_bits=args.window_step_bits,
        )
        for lag_bits in lag_values:
            targets.append(
                {
                    "band_start_bits": start_bits,
                    "band": label,
                    "offsets": offsets,
                    "lag_bits": lag_bits,
                    "lag": compact_int(lag_bits),
                    "lag_slug": lag_slug(lag_bits),
                }
            )
    for target in targets:
        target["estimated_required_bits"] = estimated_lagaware_required_bits(target, args)
    if args.max_targets:
        targets = targets[: args.max_targets]
    if not targets:
        raise SystemExit("No lag-response targets were built")
    return targets


def run_lag_response_target_pipeline(
    *,
    project_root: Path,
    phase1_dir: Path,
    band_dir: Path,
    selection: dict[str, Any],
    target: dict[str, Any],
    offsets: list[int],
    top_patterns: int,
    probe_summary_path: Path,
    recommended_lag: int | None,
    child_quiet: bool,
    use_heartbeat: bool,
) -> dict[str, Any]:
    label = str(target["band"])
    start_bits = int(target["band_start_bits"])
    lag_bits = int(target["lag_bits"])
    target_dir = band_dir / str(target["lag_slug"])
    commands: list[dict[str, Any]] = []

    def run_step(command: list[str], *, task_name: str, detail: str) -> None:
        if use_heartbeat:
            run_subprocess_with_heartbeat(
                command,
                cwd=project_root,
                quiet_children=child_quiet,
                task_name=task_name,
                detail=detail,
            )
        else:
            run_subprocess(command, cwd=project_root)

    lagaware_dir = target_dir / "lagaware"
    lagaware_cmd = build_lagaware_cmd(
        script_dir=project_root,
        phase1_dir=phase1_dir,
        output_dir=lagaware_dir,
        selection=selection,
        offsets=offsets,
        top_patterns=top_patterns,
        lag_bits=lag_bits,
    )
    append_flag(lagaware_cmd, "--quiet", child_quiet)
    commands.append({"stage": "lagaware", "band": label, "lag_bits": lag_bits, "cmd": lagaware_cmd})
    run_step(
        lagaware_cmd,
        task_name=f"N2-11 lagaware {compact_int(lag_bits)}",
        detail=f"band={label}",
    )
    lagaware_summary_path = latest_child_file(lagaware_dir, "summary.json")

    revalidation_summary_path = write_synthetic_revalidation_summary(
        band_root=target_dir,
        selection=selection,
        phase1_dir=phase1_dir,
        top_patterns=top_patterns,
        offsets=offsets,
        probe_lag_bits=recommended_lag,
        selected_lag_bits=lag_bits,
        lag_mode="lag-response",
        probe_summary_path=probe_summary_path,
        lagaware_summary_path=lagaware_summary_path,
    )

    atlas_dir = target_dir / "atlas"
    atlas_cmd = [
        sys.executable,
        str(project_root / "hsi_v2_phase2_parent_shell_atlas.py"),
        "--revalidation-run",
        str(revalidation_summary_path),
        "--output-dir",
        str(atlas_dir),
    ]
    append_flag(atlas_cmd, "--quiet", child_quiet)
    commands.append({"stage": "atlas", "band": label, "lag_bits": lag_bits, "cmd": atlas_cmd})
    run_step(
        atlas_cmd,
        task_name=f"N2-11 atlas {compact_int(lag_bits)}",
        detail=f"band={label}",
    )
    atlas_summary_path = latest_child_file(atlas_dir, "summary.json")
    atlas_dataset_path = atlas_summary_path.with_name("dataset.json")
    atlas_payload = load_json(atlas_dataset_path)
    if not atlas_payload.get("atlas_rows"):
        return {
            "target_row": {
                "band": label,
                "band_start_bits": start_bits,
                "lag_bits": lag_bits,
                "lag": compact_int(lag_bits),
                "estimated_required_bits": target["estimated_required_bits"],
                "probe_recommended_lag_bits": recommended_lag,
                "probe_recommended_lag": (
                    compact_int(int(recommended_lag))
                    if recommended_lag is not None
                    else None
                ),
                "status": "no-atlas-rows",
                "probe_summary_path": str(probe_summary_path),
                "lagaware_summary_path": str(lagaware_summary_path),
                "atlas_summary_path": str(atlas_summary_path),
                "child_summary_path": None,
            },
            "source_rows": [],
            "commands": commands,
        }

    child_dir = target_dir / "child_routing"
    child_cmd = [
        sys.executable,
        str(project_root / "hsi_v2_phase2_child_destination_routing.py"),
        "--atlas-run",
        str(atlas_dataset_path),
        "--output-dir",
        str(child_dir),
    ]
    append_flag(child_cmd, "--quiet", child_quiet)
    commands.append({"stage": "child_routing", "band": label, "lag_bits": lag_bits, "cmd": child_cmd})
    run_step(
        child_cmd,
        task_name=f"N2-11 child-routing {compact_int(lag_bits)}",
        detail=f"band={label}",
    )
    child_summary_path = latest_child_file(child_dir, "summary.json")
    child_summary = load_json(child_summary_path)

    enriched_source_rows = []
    for row in child_summary.get("source_summary", []):
        enriched = dict(row)
        enriched.update(
            {
                "band": label,
                "band_start_bits": start_bits,
                "lag_bits": lag_bits,
                "lag": compact_int(lag_bits),
                "probe_recommended_lag_bits": recommended_lag,
            }
        )
        enriched_source_rows.append(enriched)

    return {
        "target_row": {
            "band": label,
            "band_start_bits": start_bits,
            "lag_bits": lag_bits,
            "lag": compact_int(lag_bits),
            "estimated_required_bits": target["estimated_required_bits"],
            "probe_recommended_lag_bits": recommended_lag,
            "probe_recommended_lag": (
                compact_int(recommended_lag) if recommended_lag is not None else None
            ),
            "status": "completed",
            "probe_summary_path": str(probe_summary_path),
            "lagaware_summary_path": str(lagaware_summary_path),
            "atlas_summary_path": str(atlas_summary_path),
            "child_summary_path": str(child_summary_path),
        },
        "source_rows": enriched_source_rows,
        "commands": commands,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(normalize_negative_csv_options(argv or []))
    validate_args(args)

    project_root = Path(__file__).resolve().parent
    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_dir = resolve_results_dir(args.output_dir, anchor_file=__file__)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    targets = build_targets(args)
    lags = parse_lag_list(args.lags)
    selection = build_selection(args)
    selection.update(
        {
            "iteration": args.iteration,
            "segment_bits": args.segment_bits,
            "num_segments": args.num_segments,
            "scales": args.scales,
            "phase1_policies": args.phase1_policies,
            "low_scale": args.low_scale,
            "high_scale": args.high_scale,
            "top_patterns": args.top_patterns,
            "pattern_selection": args.pattern_selection,
            "scan_forward_bits": args.probe_forward_bits,
            "scan_backward_bits": args.probe_backward_bits,
        }
    )
    if args.anchor_variant == args.candidate_variant:
        raise SystemExit("--anchor-variant and --candidate-variant must differ")
    selection["anchor_variant"] = args.anchor_variant
    selection["candidate_variant"] = args.candidate_variant
    selection["variants"] = f"{args.anchor_variant},{args.candidate_variant}"
    selection["variants_list"] = [args.anchor_variant, args.candidate_variant]

    first_band = targets[0]["band"]
    last_band = targets[-1]["band"]
    run_slug = (
        "phase2-child-routing-lag-response"
        f"__stage-{args.stage}"
        f"__anchor-{args.anchor_variant}"
        f"__cand-{args.candidate_variant}"
        f"__top-{args.top_patterns}"
        f"__bands-{first_band}-to-{last_band}"
        f"__lags-{len(lags)}"
        f"__{timestamp}"
    )
    run_dir = output_dir / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing N2-11 child-routing lag-response profile",
        (
            f"stage={args.stage} | anchor={args.anchor_variant} | "
            f"candidate={args.candidate_variant} | targets={len(targets)} | "
            f"bands={len(set(t['band'] for t in targets))} | lags={len(lags)}"
        ),
        quiet=args.quiet,
    )
    flush_output()

    target_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    phase1_input_rows: list[dict[str, Any]] = []
    child_quiet = args.quiet or args.quiet_children
    started_at = time.perf_counter()
    total_targets = len(targets)
    completed_targets = 0

    targets_by_band: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for target in targets:
        targets_by_band.setdefault(target["band"], []).append(target)
    execution_bands = sorted(
        targets_by_band.items(),
        key=lambda item: max(int(target["estimated_required_bits"]) for target in item[1]),
        reverse=True,
    )

    if args.dry_run:
        for target in targets:
            target_rows.append(
                {
                    "band": target["band"],
                    "band_start_bits": target["band_start_bits"],
                    "lag_bits": target["lag_bits"],
                    "lag": target["lag"],
                    "estimated_required_bits": target["estimated_required_bits"],
                    "status": "planned",
                }
            )
    else:
        phase1_input_rows = ensure_required_phase1_inputs(
            script_dir=project_root,
            phase1_dir=phase1_dir,
            selection=selection,
            quiet=args.quiet,
        )
        commands.extend(
            {
                "stage": "phase1-input",
                "variant": row["variant"],
                "source_kind": row["source_kind"],
                "null_model": row.get("null_model"),
                "null_seed": row.get("null_seed"),
                "action": row["action"],
                "cmd": row.get("command"),
            }
            for row in phase1_input_rows
        )
        for band_index, (label, natural_band_targets) in enumerate(execution_bands, start=1):
            band_targets = sorted(
                natural_band_targets,
                key=lambda target: int(target["estimated_required_bits"]),
                reverse=True,
            )
            start_bits = int(band_targets[0]["band_start_bits"])
            offsets = band_targets[0]["offsets"]
            band_dir = run_dir / label
            probe_dir = band_dir / "probe"
            probe_scan_start = max(0, start_bits - args.probe_backward_bits)
            probe_scan_stop = start_bits + args.probe_forward_bits

            phase_print(
                "Band probe",
                (
                    f"{band_index}/{len(targets_by_band)} | band={label} | "
                    f"scan={compact_int(probe_scan_start)}..{compact_int(probe_scan_stop)} | "
                    f"cache_warm={compact_int(int(band_targets[0]['estimated_required_bits']))}"
                ),
                quiet=args.quiet,
            )
            flush_output()
            probe_cmd = build_probe_cmd(
                script_dir=project_root,
                phase1_dir=phase1_dir,
                output_dir=probe_dir,
                selection=selection,
                top_patterns=args.top_patterns,
                definition_offsets=offsets,
                scan_start_bits=probe_scan_start,
                scan_stop_bits=probe_scan_stop,
                scan_step_bits=args.scan_step_bits,
            )
            append_flag(probe_cmd, "--quiet", child_quiet)
            commands.append({"stage": "probe", "band": label, "cmd": probe_cmd})
            run_subprocess_with_heartbeat(
                probe_cmd,
                cwd=project_root,
                quiet_children=child_quiet,
                task_name=f"N2-11 probe {label}",
                detail=f"scan={compact_int(probe_scan_start)}..{compact_int(probe_scan_stop)}",
            )
            probe_summary_path = latest_child_file(probe_dir, "summary.json")
            probe_summary = load_json(probe_summary_path)
            recommended_lag = probe_summary.get("summary", {}).get("recommended_lag_bits")

            warm_target = band_targets[0]
            remaining_targets = band_targets[1:]
            effective_workers = min(int(args.workers), max(1, len(remaining_targets)))

            ordered_targets = [warm_target]
            if args.workers <= 1 or not remaining_targets:
                ordered_targets.extend(remaining_targets)
                for lag_index, target in enumerate(ordered_targets, start=1):
                    lag_bits = int(target["lag_bits"])
                    phase_print(
                        "Lag-response target",
                        (
                            f"band {band_index}/{len(targets_by_band)} | "
                            f"lag {lag_index}/{len(ordered_targets)} | band={label} | "
                            f"lag={compact_int(lag_bits)} | "
                            f"probe_rec={compact_int(int(recommended_lag)) if recommended_lag is not None else '-'}"
                        ),
                        quiet=args.quiet,
                    )
                    flush_output()
                    result = run_lag_response_target_pipeline(
                        project_root=project_root,
                        phase1_dir=phase1_dir,
                        band_dir=band_dir,
                        selection=selection,
                        target=target,
                        offsets=offsets,
                        top_patterns=args.top_patterns,
                        probe_summary_path=probe_summary_path,
                        recommended_lag=recommended_lag,
                        child_quiet=child_quiet,
                        use_heartbeat=True,
                    )
                    target_rows.append(result["target_row"])
                    source_rows.extend(result["source_rows"])
                    commands.extend(result["commands"])
                    completed_targets += 1
                    phase_print(
                        "Lag-response progress",
                        progress_detail(
                            completed=completed_targets,
                            total=total_targets,
                            started_at=started_at,
                            latest_band=label,
                            latest_lag_bits=lag_bits,
                            latest_status=result["target_row"]["status"],
                        ),
                        quiet=args.quiet,
                    )
                    flush_output()
            else:
                phase_print(
                    "Lag-response cache warm target",
                    (
                        f"band {band_index}/{len(targets_by_band)} | band={label} | "
                        f"lag={compact_int(int(warm_target['lag_bits']))} | "
                        f"required={compact_int(int(warm_target['estimated_required_bits']))}"
                    ),
                    quiet=args.quiet,
                )
                flush_output()
                result = run_lag_response_target_pipeline(
                    project_root=project_root,
                    phase1_dir=phase1_dir,
                    band_dir=band_dir,
                    selection=selection,
                    target=warm_target,
                    offsets=offsets,
                    top_patterns=args.top_patterns,
                    probe_summary_path=probe_summary_path,
                    recommended_lag=recommended_lag,
                    child_quiet=True,
                    use_heartbeat=True,
                )
                target_rows.append(result["target_row"])
                source_rows.extend(result["source_rows"])
                commands.extend(result["commands"])
                completed_targets += 1
                phase_print(
                    "Lag-response progress",
                    progress_detail(
                        completed=completed_targets,
                        total=total_targets,
                        started_at=started_at,
                        latest_band=label,
                        latest_lag_bits=int(warm_target["lag_bits"]),
                        latest_status=result["target_row"]["status"],
                    ),
                    quiet=args.quiet,
                )
                flush_output()

                phase_print(
                    "Lag-response parallel targets",
                    (
                        f"band {band_index}/{len(targets_by_band)} | band={label} | "
                        f"targets={len(remaining_targets)} | workers={effective_workers}"
                    ),
                    quiet=args.quiet,
                )
                flush_output()
                with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                    futures = {
                        executor.submit(
                            run_lag_response_target_pipeline,
                            project_root=project_root,
                            phase1_dir=phase1_dir,
                            band_dir=band_dir,
                            selection=selection,
                            target=target,
                            offsets=offsets,
                            top_patterns=args.top_patterns,
                            probe_summary_path=probe_summary_path,
                            recommended_lag=recommended_lag,
                            child_quiet=True,
                            use_heartbeat=False,
                        ): target
                        for target in remaining_targets
                    }
                    with HeartbeatProgress(
                        f"N2-11 parallel {label}",
                        interval=5.0,
                        message=f"0/{len(remaining_targets)} lag targets completed",
                    ) as heartbeat:
                        band_completed = 0
                        for future in as_completed(futures):
                            target = futures[future]
                            result = future.result()
                            target_rows.append(result["target_row"])
                            source_rows.extend(result["source_rows"])
                            commands.extend(result["commands"])
                            completed_targets += 1
                            band_completed += 1
                            lag_bits = int(target["lag_bits"])
                            heartbeat.set_message(
                                f"{band_completed}/{len(remaining_targets)} | "
                                f"latest={compact_int(lag_bits)} | "
                                f"status={result['target_row']['status']}"
                            )
                            phase_print(
                                "Lag-response progress",
                                progress_detail(
                                    completed=completed_targets,
                                    total=total_targets,
                                    started_at=started_at,
                                    latest_band=label,
                                    latest_lag_bits=lag_bits,
                                    latest_status=result["target_row"]["status"],
                                ),
                                quiet=args.quiet,
                            )
                            flush_output()

    response_rows = aggregate_response(source_rows)

    summary = {
        "run_id": run_slug,
        "timestamp_utc": timestamp,
        "parameters": {
            "phase1_dir": str(phase1_dir),
            "output_dir": str(output_dir),
            "profile": args.profile,
            "stage": args.stage,
            "anchor_variant": args.anchor_variant,
            "candidate_variant": args.candidate_variant,
            "iteration": args.iteration,
            "segment_bits": args.segment_bits,
            "num_segments": args.num_segments,
            "scales": args.scales,
            "phase1_policies": args.phase1_policies,
            "low_scale": args.low_scale,
            "high_scale": args.high_scale,
            "top_patterns": args.top_patterns,
            "pattern_selection": args.pattern_selection,
            "band_starts": args.band_starts,
            "window_count": args.window_count,
            "window_step_bits": args.window_step_bits,
            "probe_backward_bits": args.probe_backward_bits,
            "probe_forward_bits": args.probe_forward_bits,
            "scan_step_bits": args.scan_step_bits,
            "lags": args.lags,
            "null_models": args.null_models,
            "matched_lz_seeds": args.matched_lz_seeds,
            "workers": args.workers,
            "dry_run": args.dry_run,
            "execution_order": "descending_estimated_required_bits_for_cache_warmth",
        },
        "phase1_inputs": phase1_input_rows,
        "targets": target_rows,
        "lag_response": response_rows,
        "source_rows": source_rows,
    }

    manifest = {
        "script": Path(__file__).name,
        "run_dir": str(run_dir),
        "phase1_dir": str(phase1_dir),
        "arguments": vars(args),
        "selection": selection,
        "commands": commands,
        "outputs": {
            "summary": str(run_dir / "summary.json"),
            "report": str(run_dir / "report.md"),
            "lag_response_csv": str(run_dir / "lag_response.csv"),
            "source_response_csv": str(run_dir / "source_response.csv"),
            "targets_csv": str(run_dir / "targets.csv"),
        },
    }

    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "report.md").write_text(
        render_report(args=args, run_dir=run_dir, target_rows=target_rows, response_rows=response_rows),
        encoding="utf-8",
    )

    write_csv(
        run_dir / "targets.csv",
        target_rows,
        [
            "band",
            "band_start_bits",
            "lag_bits",
            "lag",
            "estimated_required_bits",
            "probe_recommended_lag_bits",
            "probe_recommended_lag",
            "status",
            "probe_summary_path",
            "lagaware_summary_path",
            "atlas_summary_path",
            "child_summary_path",
        ],
    )
    write_csv(
        run_dir / "source_response.csv",
        source_rows,
        [
            "band",
            "band_start_bits",
            "lag_bits",
            "lag",
            "probe_recommended_lag_bits",
            "variant",
            "source_label",
            "child_destination_retention_pooled",
            "window_synchronous_child_destination_retention_pooled",
            "monitored_candidate_reroute_share_pooled",
            "candidate_on_anchor_child_share_pooled",
            "top_child_full_match_mass_fraction",
            "anchor_child_deficit_mass_sum",
            "dominant_routing_class",
            "source_kind",
            "null_model",
            "null_seed",
        ],
    )
    write_csv(
        run_dir / "lag_response.csv",
        response_rows,
        [
            "band",
            "lag_bits",
            "lag",
            "B_retention",
            "E_retention",
            "markov1_retention",
            "matched_lz_min",
            "matched_lz_avg",
            "matched_lz_max",
            "matched_lz_n",
            "B_minus_matched_lz_max",
            "phase_matched_lz_min",
            "phase_matched_lz_avg",
            "phase_matched_lz_max",
            "phase_matched_lz_n",
            "B_minus_phase_matched_lz_max",
            "block_entropy_min",
            "block_entropy_avg",
            "block_entropy_max",
            "block_entropy_n",
            "B_minus_block_entropy_max",
        ],
    )

    if response_rows:
        print(render_console_summary(response_rows))
    else:
        status_counts: dict[str, int] = {}
        for row in target_rows:
            status = str(row.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
        status_text = ", ".join(
            f"{status}={count}" for status, count in sorted(status_counts.items())
        )
        mode_text = "Dry run planned" if args.dry_run else "No response rows produced for"
        print(
            "Phase 2 N2-11 child-routing lag-response profile\n"
            + "-" * 72
            + f"\n{mode_text} {len(target_rows)} targets."
            + (f" Status: {status_text}." if status_text else "")
        )

    print(f"\nSaved summary to: {run_dir / 'summary.json'}")
    print(f"Saved report to: {run_dir / 'report.md'}")
    print(f"Saved lag response CSV to: {run_dir / 'lag_response.csv'}")
    print(f"Saved source response CSV to: {run_dir / 'source_response.csv'}")
    print(f"Saved manifest to: {run_dir / 'manifest.json'}")
    flush_output()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
