#!/usr/bin/env python3
"""HSI v2 Phase 4 P4-02b selected block-entropy sweep.

This is a wider but still bounded follow-up to P4-02. It tests a fixed set
of informative band/lag pairs instead of running a full band x lag grid.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hsi_v2_phase2_parent_survival_revalidation import flush_output, latest_child_file
from hsi_v2_phase2_transport_defect_strict import phase_print
from utils.progress import format_time
from v2.common.cli import resolve_dir


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_PHASE1_DIR = "results/hsi_v2/phase1_high_scales"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase4/p4_02b_block_entropy_selected_sweep"
DEFAULT_SEEDS = "607,709,811"

SELECTED_TARGETS = (
    {
        "name": "early-negative-channel",
        "band_start_bits": 696_000_000,
        "lag_bits": -27_000_000,
        "channel": "negative",
        "reason": "Early negative-channel pressure point from P4-01.",
    },
    {
        "name": "precentral-negative-channel",
        "band_start_bits": 705_000_000,
        "lag_bits": -27_000_000,
        "channel": "negative",
        "reason": "Strongest precentral negative-channel point.",
    },
    {
        "name": "central-negative-channel",
        "band_start_bits": 714_000_000,
        "lag_bits": -27_000_000,
        "channel": "negative",
        "reason": "Central negative-channel anchor from P4-02.",
    },
    {
        "name": "late-negative-channel",
        "band_start_bits": 723_000_000,
        "lag_bits": -27_000_000,
        "channel": "negative",
        "reason": "Late negative-channel continuation.",
    },
    {
        "name": "late-positive-channel",
        "band_start_bits": 723_000_000,
        "lag_bits": 26_500_000,
        "channel": "positive",
        "reason": "Positive-channel counterpart with strong P4-01 separation.",
    },
    {
        "name": "null-shadow-boundary-a",
        "band_start_bits": 738_000_000,
        "lag_bits": -27_000_000,
        "channel": "negative",
        "reason": "Boundary cell where phase-matched-LZ reached observed retention.",
    },
    {
        "name": "null-shadow-boundary-b",
        "band_start_bits": 747_000_000,
        "lag_bits": -27_000_000,
        "channel": "negative",
        "reason": "Later boundary cell where phase-matched-LZ remained close to observed.",
    },
    {
        "name": "late-holdout-reactivation",
        "band_start_bits": 756_000_000,
        "lag_bits": -24_000_000,
        "channel": "negative",
        "reason": "Late holdout reactivation point from P4-01/P4-02.",
    },
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run P4-02b: a selected, bounded sweep of block-entropy local "
            "transition-entropy null probes over informative band/lag pairs."
        )
    )
    parser.add_argument("--phase1-dir", default=DEFAULT_PHASE1_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--seeds",
        default=DEFAULT_SEEDS,
        help="Seed list for seeded local transition-entropy null realizations.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Worker count passed through to each child lag-response probe.",
    )
    parser.add_argument(
        "--target-workers",
        type=int,
        default=1,
        help=(
            "Parallel P4-02b targets after a sequential cache-warm target. "
            "Use this, not --workers, to saturate multi-core machines."
        ),
    )
    parser.add_argument(
        "--limit-targets",
        type=int,
        default=0,
        help="Optional smoke limit over the fixed target list; 0 runs all targets.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.target_workers <= 0:
        raise SystemExit("--target-workers must be positive")
    if args.limit_targets < 0:
        raise SystemExit("--limit-targets cannot be negative")

    targets = list(SELECTED_TARGETS)
    if args.limit_targets:
        targets = targets[: args.limit_targets]
    if not targets:
        raise SystemExit("No P4-02b targets selected")

    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    seed_count = len([item for item in str(args.seeds).split(",") if item.strip()])
    run_slug = (
        "phase4-p4-02b-block-entropy-selected-sweep"
        f"__targets-{len(targets)}__seeds-{seed_count}__{timestamp}"
    )
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "run_id": run_slug,
        "timestamp_utc": timestamp,
        "status": "dry-run" if args.dry_run else "running",
        "parameters": {
            "phase1_dir": str(phase1_dir),
            "output_dir": str(output_root),
            "seeds": args.seeds,
            "workers": args.workers,
            "target_workers": args.target_workers,
            "null_models": "markov1,block-entropy",
            "top_patterns": 128,
            "targets": targets,
            "dry_run": bool(args.dry_run),
            "limit_targets": args.limit_targets,
        },
        "results": [],
        "commands": [],
    }

    started = time.perf_counter()
    try:
        phase_print(
            "Preparing P4-02b selected block-entropy sweep",
            (
                f"targets={len(targets)} | top=128 | nulls=markov1,block-entropy | "
                f"seeds={args.seeds} | child_workers={args.workers} | "
                f"target_workers={args.target_workers}"
            ),
            quiet=args.quiet,
        )
        flush_output()

        execute_targets(
            args=args,
            state=state,
            run_dir=run_dir,
            phase1_dir=phase1_dir,
            targets=targets,
        )
        state["results"].sort(key=lambda row: int(row.get("target_index", 999999)))
        state["commands"].sort(key=lambda row: int(row.get("target_index", 999999)))

        state["status"] = "dry-run" if args.dry_run else "completed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_outputs(run_dir, state)
        print(render_console_summary(state))
        print(f"\nSaved summary to: {run_dir / 'summary.json'}")
        print(f"Saved report to: {run_dir / 'report.md'}")
        print(f"Saved manifest to: {run_dir / 'manifest.json'}")
        return 0
    except subprocess.CalledProcessError as exc:
        state["status"] = "failed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        state["error"] = {
            "returncode": int(exc.returncode),
            "command": [str(part) for part in exc.cmd],
        }
        write_outputs(run_dir, state)
        print(
            f"[x] P4-02b selected sweep failed after {format_time(time.perf_counter() - started)} "
            f"(returncode={exc.returncode})."
        )
        print(f"Saved failure summary to: {run_dir / 'summary.json'}")
        return int(exc.returncode) if int(exc.returncode) > 0 else 1


def execute_targets(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    run_dir: Path,
    phase1_dir: Path,
    targets: list[dict[str, Any]],
) -> None:
    target_count = len(targets)
    if args.dry_run or args.target_workers <= 1 or target_count <= 1:
        for index, target in enumerate(targets, start=1):
            record = run_target(
                args=args,
                run_dir=run_dir,
                phase1_dir=phase1_dir,
                index=index,
                target_count=target_count,
                target=target,
                quiet_phase=False,
            )
            state["commands"].append(record["command_record"])
            state["results"].append(record["result"])
        return

    warm_target = targets[0]
    phase_print(
        "P4-02b cache warm target",
        f"{warm_target['name']} | generates/reuses shared Phase 1 prerequisites",
        quiet=args.quiet,
    )
    flush_output()
    warm_record = run_target(
        args=args,
        run_dir=run_dir,
        phase1_dir=phase1_dir,
        index=1,
        target_count=target_count,
        target=warm_target,
        quiet_phase=True,
    )
    state["commands"].append(warm_record["command_record"])
    state["results"].append(warm_record["result"])

    remaining = list(enumerate(targets[1:], start=2))
    effective_workers = min(int(args.target_workers), len(remaining))
    phase_print(
        "P4-02b parallel target pool",
        f"targets={len(remaining)} | target_workers={effective_workers} | child_workers={args.workers}",
        quiet=args.quiet,
    )
    flush_output()

    completed = 1
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = [
            executor.submit(
                run_target,
                args=args,
                run_dir=run_dir,
                phase1_dir=phase1_dir,
                index=index,
                target_count=target_count,
                target=target,
                quiet_phase=True,
            )
            for index, target in remaining
        ]
        for future in as_completed(futures):
            record = future.result()
            state["commands"].append(record["command_record"])
            state["results"].append(record["result"])
            completed += 1
            result = record["result"]
            phase_print(
                "P4-02b progress",
                f"{completed}/{target_count} | latest={result['name']} | status={result['status']}",
                quiet=args.quiet,
            )
            flush_output()


def run_target(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    phase1_dir: Path,
    index: int,
    target_count: int,
    target: dict[str, Any],
    quiet_phase: bool,
) -> dict[str, Any]:
    band_start = int(target["band_start_bits"])
    lag_bits = int(target["lag_bits"])
    target_root = run_dir / target["name"]
    command = [
        sys.executable,
        str(SCRIPT_DIR / "hsi_v2_phase2_child_routing_lag_response.py"),
        "--phase1-dir",
        str(phase1_dir),
        "--output-dir",
        str(target_root),
        "--stage",
        "all",
        "--top-patterns",
        "128",
        "--band-starts",
        str(band_start),
        f"--lags={lag_bits}",
        "--null-models",
        "markov1,block-entropy",
        "--matched-lz-seeds",
        str(args.seeds),
        "--workers",
        str(args.workers),
        "--quiet-children",
    ]
    if args.quiet:
        command.append("--quiet")
    command_record = {"target_index": index, "target": target["name"], "command": command}

    if not quiet_phase:
        phase_print(
            f"P4-02b target {index}/{target_count}",
            (
                f"{target['name']} | channel={target['channel']} | "
                f"band_start={band_start} | lag={lag_bits}"
            ),
            quiet=args.quiet,
        )
        flush_output()

    if args.dry_run:
        return {
            "command_record": command_record,
            "result": {
                **target,
                "target_index": index,
                "status": "planned",
                "command": command,
            },
        }

    started = time.perf_counter()
    subprocess.run(command, cwd=str(SCRIPT_DIR), check=True)
    summary_path = latest_child_file(target_root, "summary.json")
    summary = load_json(summary_path)
    rows = summary.get("lag_response", [])
    row = rows[0] if rows else {}
    return {
        "command_record": command_record,
        "result": {
            **target,
            "target_index": index,
            "status": "completed",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "child_summary": str(summary_path),
            "B_retention": row.get("B_retention"),
            "markov1_retention": row.get("markov1_retention"),
            "block_entropy_max": row.get("block_entropy_max"),
            "B_minus_block_entropy_max": row.get("B_minus_block_entropy_max"),
            "block_entropy_n": row.get("block_entropy_n"),
        },
    }


def render_console_summary(state: dict[str, Any]) -> str:
    lines = [
        "Phase 4 P4-02b selected block-entropy transition-entropy sweep",
        "-" * 112,
        (
            f"{'target':<28} {'ch':>4} {'B':>8} {'markov1':>8} "
            f"{'bent_max':>9} {'B-bent':>9} {'n':>4}"
        ),
    ]
    for row in state["results"]:
        lines.append(
            f"{str(row['name']):<28} "
            f"{str(row.get('channel', '-'))[:4]:>4} "
            f"{fmt_float(row.get('B_retention')):>8} "
            f"{fmt_float(row.get('markov1_retention')):>8} "
            f"{fmt_float(row.get('block_entropy_max')):>9} "
            f"{fmt_float(row.get('B_minus_block_entropy_max')):>9} "
            f"{str(row.get('block_entropy_n', '-')):>4}"
        )
    return "\n".join(lines)


def render_report(state: dict[str, Any], run_dir: Path) -> str:
    lines = [
        "# HSI v2 Phase 4 P4-02b selected block-entropy transition-entropy sweep",
        "",
        (
            "This selected sweep widens P4-02 without running a full grid. "
            "Each row is a fixed band/lag pressure point inherited from the "
            "P4-01/P4-02 readout."
        ),
        "It remains an internal Phase 4 probe, not a paper-facing claim.",
        "",
        "## Parameters",
        "",
        f"- Null models: `{state['parameters']['null_models']}`",
        f"- Seeds: `{state['parameters']['seeds']}`",
        f"- Top patterns: `{state['parameters']['top_patterns']}`",
        f"- Child workers: `{state['parameters']['workers']}`",
        f"- Target workers: `{state['parameters']['target_workers']}`",
        f"- Status: `{state['status']}`",
        "",
        "## Selected Sweep Readout",
        "",
        (
            "| Target | Channel | Band start | Lag | B retention | markov1 | "
            "block-entropy max | B - max(block-entropy) | n |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in state["results"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
                    str(row.get("channel", "-")),
                    str(row["band_start_bits"]),
                    str(row["lag_bits"]),
                    fmt_float(row.get("B_retention")),
                    fmt_float(row.get("markov1_retention")),
                    fmt_float(row.get("block_entropy_max")),
                    fmt_float(row.get("B_minus_block_entropy_max")),
                    str(row.get("block_entropy_n", "-")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Run directory: `{run_dir}`",
            "- Summary: `summary.json`",
            "- Manifest: `manifest.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(run_dir: Path, state: dict[str, Any]) -> None:
    write_json(run_dir / "summary.json", state)
    write_json(
        run_dir / "manifest.json",
        {
            "script": Path(__file__).name,
            "run_dir": str(run_dir),
            "commands": state["commands"],
            "outputs": {
                "summary": str(run_dir / "summary.json"),
                "report": str(run_dir / "report.md"),
                "manifest": str(run_dir / "manifest.json"),
            },
        },
    )
    (run_dir / "report.md").write_text(render_report(state, run_dir), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
