#!/usr/bin/env python3
"""
CAT_24 B iteration-28 preflight.

Read-only planner for extending the canonical B Level-0 snapshot chain from the
latest available iteration to iteration 28. It estimates disk and wall-time cost
from existing B snapshots/reports and emits preregistration artifacts. It does
not generate Phi or run any downstream readout.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.cli import resolve_results_dir


DEFAULT_TARGET_ITERATION = 28
DEFAULT_LEVEL0_ROOT = "results/level0/phi_snapshots"
DEFAULT_REPORTS_DIR = "results/level0/reports"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/cat24_iteration28_preflight"
DEFAULT_RESULTS_BASE = r"D:\Iban\Projectes\HSI\results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only CAT_24 preflight for B@28.")
    parser.add_argument("--variant", default="B", help="Variant to inspect. CAT_24 default is B.")
    parser.add_argument("--target-iteration", type=int, default=DEFAULT_TARGET_ITERATION)
    parser.add_argument(
        "--level0-root",
        default=DEFAULT_LEVEL0_ROOT,
        help="Level-0 phi_snapshots root used as the checkpoint source.",
    )
    parser.add_argument(
        "--reports-dir",
        default=DEFAULT_REPORTS_DIR,
        help="Level-0 reports directory used for historical wall-time estimates.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    variant = args.variant.strip().upper()
    if not variant:
        raise SystemExit("--variant cannot be empty")
    if args.target_iteration < 1:
        raise SystemExit("--target-iteration must be positive")

    level0_root = resolve_results_dir(args.level0_root, anchor_file=__file__)
    reports_dir = resolve_results_dir(args.reports_dir, anchor_file=__file__)
    output_root = resolve_results_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    snapshot_rows = collect_snapshot_rows(level0_root / f"var_{variant}")
    if not snapshot_rows:
        raise SystemExit(f"No Level-0 snapshots found for variant {variant} under {level0_root}")

    time_rows = collect_time_rows(reports_dir, variant)
    latest = max(snapshot_rows, key=lambda row: row["iteration"])
    estimates = estimate_missing_iterations(snapshot_rows, time_rows, args.target_iteration)
    disk = shutil.disk_usage(output_root.anchor or ".")

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / (
        f"cat24-iteration28-preflight__var-{variant}__target-{args.target_iteration}__{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "status": "preflight-only",
        "variant": variant,
        "latest_available_iteration": latest["iteration"],
        "target_iteration": args.target_iteration,
        "level0_root": str(level0_root),
        "reports_dir": str(reports_dir),
        "existing_snapshots": snapshot_rows,
        "historical_time_rows": time_rows,
        "estimates": estimates,
        "disk": {
            "output_root": str(output_root),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "recommended_generation_command": build_generation_command(variant, args.target_iteration),
        "execution_authorized_by_this_script": False,
    }
    manifest = {
        "stage": "cat24_iteration28_preflight",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "hsi_v2_cat24_iteration28_preflight.py",
        "inputs": {
            "level0_root": str(level0_root),
            "reports_dir": str(reports_dir),
            "variant_dir": str(level0_root / f"var_{variant}"),
        },
        "outputs": {
            "summary": str(run_dir / "summary.json"),
            "manifest": str(run_dir / "manifest.json"),
            "report": str(run_dir / "report.md"),
        },
        "contract_intent": {
            "scope": "pre-execution feasibility estimate only",
            "does_not_generate_level0": True,
            "recommended_initial_scope": "B-only to iteration 28",
        },
    }

    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "report.md").write_text(render_report(summary), encoding="utf-8")

    print("CAT_24 B iteration-28 preflight")
    print("-" * 72)
    print("status: preflight-only")
    print(f"latest available: {variant}@{latest['iteration']}")
    print(f"target: {variant}@{args.target_iteration}")
    print(f"output: {run_dir}")
    if estimates:
        last = estimates[-1]
        print(f"estimated target struct.gz: {last['estimated_struct_gz_bytes']:,} bytes")
        print(f"estimated cumulative generation from next iter: {last['estimated_cumulative_time_hours']:.1f} h")
    print(f"free bytes at output root: {disk.free:,}")
    return 0


def collect_snapshot_rows(variant_dir: Path) -> list[dict]:
    rows = []
    pattern = re.compile(r"phi_iter(\d+)\.struct\.gz$")
    if not variant_dir.exists():
        return rows
    for struct_path in variant_dir.glob("phi_iter*.struct.gz"):
        match = pattern.match(struct_path.name)
        if not match:
            continue
        iteration = int(match.group(1))
        metadata_path = variant_dir / f"phi_iter{iteration}.json"
        metadata = load_json(metadata_path) if metadata_path.exists() else {}
        rows.append(
            {
                "iteration": iteration,
                "struct_path": str(struct_path),
                "metadata_path": str(metadata_path) if metadata_path.exists() else None,
                "struct_gz_bytes": struct_path.stat().st_size,
                "format": metadata.get("format"),
                "compression_ratio": metadata.get("compression_ratio"),
            }
        )
    return sorted(rows, key=lambda row: row["iteration"])


def collect_time_rows(reports_dir: Path, variant: str) -> list[dict]:
    best_by_iteration: dict[int, dict] = {}
    if not reports_dir.exists():
        return []
    for report_path in reports_dir.glob(f"variant_{variant}_*.json"):
        if report_path.name.endswith(".enriched.json"):
            continue
        try:
            payload = load_json(report_path)
        except Exception:
            continue
        for row in payload.get("per_iteration", []):
            iteration = int(row.get("iteration") or 0)
            phi_length = row.get("phi_length")
            time_sec = row.get("time_sec")
            if iteration <= 0 or phi_length is None or time_sec is None:
                continue
            candidate = {
                "iteration": iteration,
                "phi_length_bits": int(phi_length),
                "iteration_time_seconds": float(time_sec),
                "source_report": str(report_path),
            }
            previous = best_by_iteration.get(iteration)
            if previous is None or candidate["phi_length_bits"] > previous["phi_length_bits"]:
                best_by_iteration[iteration] = candidate
    return [best_by_iteration[key] for key in sorted(best_by_iteration)]


def estimate_missing_iterations(snapshot_rows: list[dict], time_rows: list[dict], target_iteration: int) -> list[dict]:
    latest_snapshot = max(snapshot_rows, key=lambda row: row["iteration"])
    if latest_snapshot["iteration"] >= target_iteration:
        return []

    compressed_ratios = [
        right["struct_gz_bytes"] / left["struct_gz_bytes"]
        for left, right in zip(snapshot_rows, snapshot_rows[1:])
        if left["iteration"] >= 20 and left["struct_gz_bytes"] > 0
    ]
    compressed_ratio = geometric_mean(compressed_ratios[-3:]) if compressed_ratios else 2.81

    time_by_iter = {row["iteration"]: row for row in time_rows}
    historical_iters = sorted(time_by_iter)
    raw_ratio = None
    time_ratio = None
    if len(historical_iters) >= 4:
        tail = historical_iters[-4:]
        raw_ratios = [
            time_by_iter[right]["phi_length_bits"] / time_by_iter[left]["phi_length_bits"]
            for left, right in zip(tail, tail[1:])
            if time_by_iter[left]["phi_length_bits"] > 0
        ]
        time_ratios = [
            time_by_iter[right]["iteration_time_seconds"] / time_by_iter[left]["iteration_time_seconds"]
            for left, right in zip(tail, tail[1:])
            if time_by_iter[left]["iteration_time_seconds"] > 0
        ]
        raw_ratio = geometric_mean(raw_ratios)
        time_ratio = geometric_mean(time_ratios)

    previous_compressed = latest_snapshot["struct_gz_bytes"]
    previous_raw = time_by_iter.get(latest_snapshot["iteration"], {}).get("phi_length_bits")
    previous_time = time_by_iter.get(latest_snapshot["iteration"], {}).get("iteration_time_seconds")
    cumulative_time = 0.0
    estimates = []

    for iteration in range(latest_snapshot["iteration"] + 1, target_iteration + 1):
        estimated_compressed = round(previous_compressed * compressed_ratio)
        estimated_temp = round(estimated_compressed * 4.0)
        estimated_raw = round(previous_raw * raw_ratio) if previous_raw and raw_ratio else None
        estimated_time = previous_time * time_ratio if previous_time and time_ratio else None
        if estimated_time:
            cumulative_time += estimated_time / 3600.0
        estimates.append(
            {
                "iteration": iteration,
                "compressed_growth_ratio": compressed_ratio,
                "raw_growth_ratio": raw_ratio,
                "time_growth_ratio": time_ratio,
                "estimated_struct_gz_bytes": estimated_compressed,
                "estimated_peak_temp_bytes": estimated_temp,
                "estimated_total_working_bytes": estimated_compressed + estimated_temp,
                "estimated_raw_bits": estimated_raw,
                "estimated_iteration_time_hours": estimated_time / 3600.0 if estimated_time else None,
                "estimated_cumulative_time_hours": cumulative_time if estimated_time else None,
            }
        )
        previous_compressed = estimated_compressed
        previous_raw = estimated_raw
        previous_time = estimated_time
    return estimates


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def build_generation_command(variant: str, target_iteration: int) -> str:
    return (
        f"$env:HSI_RESULTS_BASE_DIR = \"{DEFAULT_RESULTS_BASE}\"; "
        "$env:PYTHONUTF8 = \"1\"; "
        "$env:PYTHONIOENCODING = \"utf-8\"; "
        f"python -m hsi_agents_project.level0.generator --variant {variant} "
        f"--iterations {target_iteration} --log-every 1 --no-generate-plots"
    )


def render_report(summary: dict) -> str:
    existing_rows = "\n".join(
        f"| {row['iteration']} | {row['struct_gz_bytes']} | {row.get('format') or ''} |"
        for row in summary["existing_snapshots"]
        if row["iteration"] >= max(1, summary["latest_available_iteration"] - 6)
    )
    estimate_rows = "\n".join(
        (
            f"| {row['iteration']} | "
            f"{row['estimated_raw_bits'] or ''} | "
            f"{row['estimated_struct_gz_bytes']} | "
            f"{row['estimated_total_working_bytes']} | "
            f"{format_optional(row['estimated_iteration_time_hours'])} | "
            f"{format_optional(row['estimated_cumulative_time_hours'])} |"
        )
        for row in summary["estimates"]
    )
    return f"""# CAT_24 B Iteration-28 Preflight

Status: preflight-only. This artifact does not authorize Level-0 generation or
any downstream Phase-1/Phase-4 readout.

Latest available snapshot: `B@{summary['latest_available_iteration']}`.
Target snapshot: `B@{summary['target_iteration']}`.

## Recent Existing Snapshots

| iteration | struct.gz bytes | format |
|---:|---:|---|
{existing_rows}

## Missing-Iteration Estimate

| iteration | estimated raw bits | estimated struct.gz bytes | estimated working bytes | estimated iter hours | cumulative hours |
|---:|---:|---:|---:|---:|---:|
{estimate_rows}

The compressed-size estimate uses the geometric mean of the recent observed B
`struct.gz` growth ratios. Temporary working storage is conservatively estimated
as `4x` the final compressed snapshot for each new iteration.

The wall-time estimate uses historical B per-iteration timings when available.
It should be treated as a warning bound, not a promise: current code, CPU,
compression settings, and external-drive throughput can move the real runtime.

## Recommended Initial Scope

B-only, resume from the existing checkpoint chain copied under
`HSI_RESULTS_BASE_DIR`, generate through iteration 28, skip plots during the
run, and stop before any downstream readout.

```powershell
{summary['recommended_generation_command']}
```

If the external path differs, only `HSI_RESULTS_BASE_DIR` may be changed.
Variant, target iteration, and generator semantics are not changed by this
preflight.
"""


def format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
