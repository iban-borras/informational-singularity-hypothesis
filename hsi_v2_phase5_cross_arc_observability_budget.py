#!/usr/bin/env python3
"""HSI v2 Phase 5 cross-arc observability budget audit.

This is a read-only Stage 0 tool for Phase 5 Candidate B-mini. It does not run
lag-response experiments and does not inspect any response outcome. It only
checks which candidate arcs are materially executable from existing Level-0
snapshot lengths and the declared P4-style band/lag geometry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v2.common.cli import resolve_dir


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_SNAPSHOT_DIR = "results/level0/phi_snapshots"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase5_cross_arc_horizon/observability_budget"
DEFAULT_PREREG = "Documentation/HSI_v2_Phase5_CandidateB_CrossArc_Preregistration_v0_1.md"
DEFAULT_VARIANTS = "B,E"
DEFAULT_NULL_MODELS = "markov1,phase-matched-lz,block-entropy"
DEFAULT_SEEDS = "607,709,811,907,1009,1103"
DEFAULT_LAGS = "-29500000,-27000000,-26500000,-24000000,24500000,26500000"

DEFAULT_SCAN_START = 300_000_000
DEFAULT_SCAN_STOP = 1_350_000_000
DEFAULT_SCAN_STEP = 27_000_000
DEFAULT_BANDS_PER_ARC = 4
DEFAULT_BAND_STEP_BITS = 9_000_000
DEFAULT_WINDOW_COUNT = 19
DEFAULT_WINDOW_STEP_BITS = 500_000
DEFAULT_WINDOW_BITS = 3_000_000

PHASE4_EXCLUDED_RANGES = (
    {
        "label": "P4-04/P4-05 horizon bracket",
        "start_bits": 768_000_000,
        "stop_bits": 831_000_000,
        "reason": "Phase 4 forward forecast and horizon-bracket region; cannot seed independent Phase 5 arcs",
    },
)

REFERENCE_USED_RANGES = (
    {
        "label": "Phase 2/3 central shell-routing diagnostics",
        "start_bits": 696_000_000,
        "stop_bits": 741_000_000,
        "reason": "earlier mechanistic stack region; reported for traceability, not an automatic exclusion",
    },
    *PHASE4_EXCLUDED_RANGES,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit executable candidate arcs for Phase 5 Candidate B-mini without "
            "running any lag-response readout."
        )
    )
    parser.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prereg", default=DEFAULT_PREREG)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--null-models", default=DEFAULT_NULL_MODELS)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--lags", default=DEFAULT_LAGS)
    parser.add_argument("--scan-start-bits", type=int, default=DEFAULT_SCAN_START)
    parser.add_argument("--scan-stop-bits", type=int, default=DEFAULT_SCAN_STOP)
    parser.add_argument("--scan-step-bits", type=int, default=DEFAULT_SCAN_STEP)
    parser.add_argument("--bands-per-arc", type=int, default=DEFAULT_BANDS_PER_ARC)
    parser.add_argument("--band-step-bits", type=int, default=DEFAULT_BAND_STEP_BITS)
    parser.add_argument("--window-count", type=int, default=DEFAULT_WINDOW_COUNT)
    parser.add_argument("--window-step-bits", type=int, default=DEFAULT_WINDOW_STEP_BITS)
    parser.add_argument("--window-bits", type=int, default=DEFAULT_WINDOW_BITS)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)

    snapshot_dir = resolve_dir(args.snapshot_dir, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    prereg_path = resolve_dir(args.prereg, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"phase5-cross-arc-observability-budget__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    if not args.quiet:
        print("Phase 5 Candidate B-mini observability budget")
        print("-" * 96)
        print(
            f"scan={compact(args.scan_start_bits)}-{compact(args.scan_stop_bits)} "
            f"step={compact(args.scan_step_bits)} | bands/arc={args.bands_per_arc}"
        )

    variants = parse_csv(args.variants)
    null_models = parse_csv(args.null_models)
    seeds = [int(item) for item in parse_csv(args.seeds)]
    lags = [int(item) for item in parse_csv(args.lags)]
    source_rows = load_source_inventory(snapshot_dir, variants)
    source_lengths = {row["variant"]: int(row["sequence_length"]) for row in source_rows}
    candidate_rows = build_candidate_rows(args, variants, null_models, seeds, lags, source_lengths)
    summary = build_summary(
        args=args,
        run_dir=run_dir,
        timestamp=timestamp,
        prereg_path=prereg_path,
        variants=variants,
        null_models=null_models,
        seeds=seeds,
        lags=lags,
        source_rows=source_rows,
        candidate_rows=candidate_rows,
        elapsed=time.perf_counter() - started,
    )

    write_json(run_dir / "summary.json", summary)
    write_report(run_dir / "report.md", summary)
    write_csv(run_dir / "candidate_arcs.csv", candidate_rows)
    write_json(run_dir / "manifest.json", build_manifest(summary))

    if not args.quiet:
        print(render_console_summary(summary))
        print(f"\nSaved summary to: {run_dir / 'summary.json'}")
        print(f"Saved report to: {run_dir / 'report.md'}")
        print(f"Saved candidate arcs CSV to: {run_dir / 'candidate_arcs.csv'}")
        print(f"Saved manifest to: {run_dir / 'manifest.json'}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.scan_step_bits <= 0:
        raise SystemExit("--scan-step-bits must be positive")
    if args.scan_stop_bits < args.scan_start_bits:
        raise SystemExit("--scan-stop-bits must be >= --scan-start-bits")
    if args.bands_per_arc <= 0:
        raise SystemExit("--bands-per-arc must be positive")
    if args.band_step_bits <= 0 or args.window_step_bits <= 0 or args.window_bits <= 0:
        raise SystemExit("band/window sizes must be positive")


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def load_source_inventory(snapshot_dir: Path, variants: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        meta_path = snapshot_dir / f"var_{variant}" / "phi_iter20.json"
        struct_path = snapshot_dir / f"var_{variant}" / "phi_iter20.struct.gz"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing snapshot metadata for variant {variant}: {meta_path}")
        if not struct_path.exists():
            raise FileNotFoundError(f"Missing snapshot struct for variant {variant}: {struct_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "variant": variant,
                "metadata_path": str(meta_path.resolve()),
                "struct_path": str(struct_path.resolve()),
                "sequence_length": int(meta["sequence_length"]),
                "metadata_sha256": sha256_file(meta_path),
                "struct_size_bytes": int(struct_path.stat().st_size),
                "struct_mtime_ns": int(struct_path.stat().st_mtime_ns),
            }
        )
    return rows


def build_candidate_rows(
    args: argparse.Namespace,
    variants: list[str],
    null_models: list[str],
    seeds: list[int],
    lags: list[int],
    source_lengths: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_positive_lag = max([0, *[lag for lag in lags if lag > 0]])
    min_length = min(source_lengths[variant] for variant in variants)
    for start in range(args.scan_start_bits, args.scan_stop_bits + 1, args.scan_step_bits):
        band_starts = [start + i * args.band_step_bits for i in range(args.bands_per_arc)]
        band_stop = band_starts[-1] + (args.window_count - 1) * args.window_step_bits
        required_bits = band_stop + args.window_bits + max_positive_lag
        overlaps_phase4 = any(overlaps(start, required_bits, item["start_bits"], item["stop_bits"]) for item in PHASE4_EXCLUDED_RANGES)
        overlaps_reference = [
            item["label"]
            for item in REFERENCE_USED_RANGES
            if overlaps(start, required_bits, item["start_bits"], item["stop_bits"])
        ]
        missing_variants = [variant for variant in variants if source_lengths[variant] < required_bits]
        executable = not missing_variants and not overlaps_phase4
        rows.append(
            {
                "arc_start_bits": start,
                "arc_label": f"{compact(start)}->{compact(band_starts[-1])}",
                "band_starts_bits": ";".join(str(value) for value in band_starts),
                "band_labels": ";".join(band_label(value, args.window_count, args.window_step_bits) for value in band_starts),
                "required_bits": required_bits,
                "min_available_observed_bits": min_length,
                "available_margin_bits": min_length - required_bits,
                "executable": "Y" if executable else "N",
                "exclusion_reason": build_exclusion_reason(missing_variants, overlaps_phase4),
                "overlaps_reference_ranges": ";".join(overlaps_reference) if overlaps_reference else "-",
                "variant_requirements": ";".join(f"{variant}:{source_lengths[variant]}" for variant in variants),
                "null_models_required": ",".join(null_models),
                "null_seeds_required": ",".join(str(seed) for seed in seeds),
                "lag_grid_bits": ",".join(str(lag) for lag in lags),
            }
        )
    return rows


def build_exclusion_reason(missing_variants: list[str], overlaps_phase4: bool) -> str:
    reasons: list[str] = []
    if missing_variants:
        reasons.append("insufficient observed source length for " + ",".join(missing_variants))
    if overlaps_phase4:
        reasons.append("overlaps Phase 4 horizon-bracket exclusion")
    return "; ".join(reasons) if reasons else "-"


def overlaps(left_start: int, left_stop: int, right_start: int, right_stop: int) -> bool:
    return left_start < right_stop and right_start < left_stop


def band_label(start_bits: int, window_count: int, window_step_bits: int) -> str:
    stop_bits = start_bits + (window_count - 1) * window_step_bits
    return f"{compact(start_bits)}-{compact(stop_bits)}"


def build_summary(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    timestamp: str,
    prereg_path: Path,
    variants: list[str],
    null_models: list[str],
    seeds: list[int],
    lags: list[int],
    source_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    elapsed: float,
) -> dict[str, Any]:
    executable = [row for row in candidate_rows if row["executable"] == "Y"]
    non_overlapping = greedy_non_overlapping(executable)
    return {
        "run_id": run_dir.name,
        "timestamp_utc": timestamp,
        "status": "completed",
        "elapsed_seconds": round(elapsed, 3),
        "preregistration": {
            "path": str(prereg_path.resolve()),
            "sha256": sha256_file(prereg_path) if prereg_path.exists() else None,
            "note": "Draft v0.1 only; this audit does not freeze arcs and does not authorize B-mini execution.",
        },
        "parameters": {
            "snapshot_dir": str(resolve_dir(args.snapshot_dir, anchor_file=__file__)),
            "variants": variants,
            "null_models": null_models,
            "seeds": seeds,
            "lags": lags,
            "scan_start_bits": args.scan_start_bits,
            "scan_stop_bits": args.scan_stop_bits,
            "scan_step_bits": args.scan_step_bits,
            "bands_per_arc": args.bands_per_arc,
            "band_step_bits": args.band_step_bits,
            "window_count": args.window_count,
            "window_step_bits": args.window_step_bits,
            "window_bits": args.window_bits,
            "excluded_ranges": list(PHASE4_EXCLUDED_RANGES),
            "reference_used_ranges": list(REFERENCE_USED_RANGES),
        },
        "source_inventory": source_rows,
        "candidate_count": len(candidate_rows),
        "executable_candidate_count": len(executable),
        "greedy_non_overlapping_executable_count": len(non_overlapping),
        "greedy_non_overlapping_executable_arcs": [
            {
                "arc_start_bits": row["arc_start_bits"],
                "arc_label": row["arc_label"],
                "required_bits": row["required_bits"],
                "available_margin_bits": row["available_margin_bits"],
            }
            for row in non_overlapping
        ],
        "verdict": build_verdict(executable, non_overlapping),
    }


def greedy_non_overlapping(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    last_required_stop = -1
    for row in sorted(rows, key=lambda item: int(item["arc_start_bits"])):
        start = int(row["arc_start_bits"])
        if start >= last_required_stop:
            selected.append(row)
            last_required_stop = int(row["required_bits"])
    return selected


def build_verdict(executable: list[dict[str, Any]], non_overlapping: list[dict[str, Any]]) -> dict[str, Any]:
    if len(non_overlapping) >= 3:
        status = "budget-supports-b-mini-3-arc-freeze"
    elif len(non_overlapping) == 2:
        status = "budget-supports-b-mini-2-arc-freeze"
    elif executable:
        status = "budget-insufficient-for-independent-b-mini"
    else:
        status = "no-executable-independent-arcs"
    return {
        "status": status,
        "executable_candidates": len(executable),
        "non_overlapping_executable_arcs": len(non_overlapping),
        "execution_authorized": False,
        "reason": "Stage 0 budget audit only; a signed v0.2 contract must freeze arcs before any lag-response execution.",
    }


def build_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": summary["run_id"],
        "timestamp_utc": summary["timestamp_utc"],
        "script": str(Path(__file__).resolve()),
        "status": summary["status"],
        "parameters": summary["parameters"],
        "preregistration": summary["preregistration"],
        "outputs": {
            "summary": "summary.json",
            "report": "report.md",
            "candidate_arcs": "candidate_arcs.csv",
            "manifest": "manifest.json",
        },
        "execution_authorized": False,
    }


def render_console_summary(summary: dict[str, Any]) -> str:
    verdict = summary["verdict"]
    lines = [
        "",
        "Phase 5 observability budget summary",
        "-" * 96,
        f"candidates={summary['candidate_count']} | executable={summary['executable_candidate_count']} | "
        f"nonoverlap={summary['greedy_non_overlapping_executable_count']} | verdict={verdict['status']}",
    ]
    for row in summary["greedy_non_overlapping_executable_arcs"][:8]:
        lines.append(
            f"  {row['arc_label']:>18} | required={compact(row['required_bits']):>8} | "
            f"margin={compact(row['available_margin_bits']):>8}"
        )
    return "\n".join(lines)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# {summary['run_id']}",
        "",
        "## Verdict",
        "",
        f"- Status: `{summary['verdict']['status']}`",
        f"- Executable candidates: `{summary['executable_candidate_count']}/{summary['candidate_count']}`",
        f"- Greedy non-overlapping executable arcs: `{summary['greedy_non_overlapping_executable_count']}`",
        "- Execution authorized: `false`",
        "",
        "This is a Stage 0 observability-budget audit only. It does not freeze arcs and does not run any lag-response readout.",
        "",
        "## Preregistration",
        "",
        f"- Path: `{summary['preregistration']['path']}`",
        f"- SHA256: `{summary['preregistration']['sha256']}`",
        "",
        "## Source Inventory",
        "",
        "| variant | sequence_length | metadata_sha256 |",
        "|---|---:|---|",
    ]
    for row in summary["source_inventory"]:
        lines.append(f"| {row['variant']} | {row['sequence_length']} | `{row['metadata_sha256']}` |")
    lines.extend(
        [
            "",
            "## Greedy Non-Overlapping Executable Arcs",
            "",
            "| arc | required_bits | available_margin_bits |",
            "|---|---:|---:|",
        ]
    )
    for row in summary["greedy_non_overlapping_executable_arcs"]:
        lines.append(f"| `{row['arc_label']}` | {row['required_bits']} | {row['available_margin_bits']} |")
    lines.extend(
        [
            "",
            "## Exclusion Policy",
            "",
            "The Phase 4 horizon-bracket range is excluded from independent Phase 5 arc selection.",
            "",
            "| label | start | stop | reason |",
            "|---|---:|---:|---|",
        ]
    )
    for row in summary["parameters"]["reference_used_ranges"]:
        lines.append(f"| {row['label']} | {row['start_bits']} | {row['stop_bits']} | {row['reason']} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.json`",
            "- `candidate_arcs.csv`",
            "- `manifest.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def compact(value: int) -> str:
    sign = "-" if value < 0 else ""
    value = abs(int(value))
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{sign}{value // 1_000_000}M"
    if value >= 1_000 and value % 1_000 == 0:
        return f"{sign}{value // 1_000}K"
    return f"{sign}{value}"


if __name__ == "__main__":
    raise SystemExit(main())
