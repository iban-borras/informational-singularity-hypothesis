#!/usr/bin/env python3
"""HSI v2 Phase 4 P4-02 block-entropy null scout.

This launcher keeps the public command small while testing a non-LZ local
transition-entropy null against two preregistered-like pressure points:

- 714M-723M at -27M, the central negative channel.
- 756M-765M at -24M, the late holdout/boundary channel from P4-01.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hsi_v2_phase2_parent_survival_revalidation import flush_output, latest_child_file
from hsi_v2_phase2_transport_defect_strict import phase_print
from utils.progress import format_time
from v2.common.cli import resolve_dir


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_PHASE1_DIR = "results/hsi_v2/phase1_high_scales"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase4/p4_02_block_entropy_scout"
DEFAULT_SEEDS = "607,709,811"
SCOUT_TARGETS = (
    {
        "name": "central-negative-channel",
        "band_start_bits": 714_000_000,
        "lag_bits": -27_000_000,
        "reason": "Central N2-11 negative channel pressure point.",
    },
    {
        "name": "late-holdout-boundary",
        "band_start_bits": 756_000_000,
        "lag_bits": -24_000_000,
        "reason": "P4-01 late holdout/boundary pressure point.",
    },
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the P4-02 scout with a block-entropy null calibrated by "
            "local transition entropy. The launcher executes two fixed "
            "pressure-point N2-11 lag-response probes."
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
        help="Worker count passed through to the child lag-response probes.",
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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    seed_count = len([item for item in str(args.seeds).split(",") if item.strip()])
    run_slug = f"phase4-p4-02-block-entropy-scout__targets-2__seeds-{seed_count}__{timestamp}"
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
            "null_models": "markov1,block-entropy",
            "top_patterns": 128,
            "targets": SCOUT_TARGETS,
            "dry_run": bool(args.dry_run),
        },
        "results": [],
        "commands": [],
    }

    started = time.perf_counter()
    try:
        phase_print(
            "Preparing P4-02 block-entropy scout",
            (
                "targets=2 | top=128 | nulls=markov1,block-entropy | "
                f"seeds={args.seeds} | workers={args.workers}"
            ),
            quiet=args.quiet,
        )
        flush_output()

        for index, target in enumerate(SCOUT_TARGETS, start=1):
            run_target(
                args=args,
                state=state,
                run_dir=run_dir,
                phase1_dir=phase1_dir,
                index=index,
                target=target,
            )

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
            f"[x] P4-02 scout failed after {format_time(time.perf_counter() - started)} "
            f"(returncode={exc.returncode})."
        )
        print(f"Saved failure summary to: {run_dir / 'summary.json'}")
        return int(exc.returncode) if int(exc.returncode) > 0 else 1


def run_target(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    run_dir: Path,
    phase1_dir: Path,
    index: int,
    target: dict[str, Any],
) -> None:
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
    state["commands"].append({"target": target["name"], "command": command})

    phase_print(
        f"P4-02 target {index}/{len(SCOUT_TARGETS)}",
        f"{target['name']} | band_start={band_start} | lag={lag_bits}",
        quiet=args.quiet,
    )
    flush_output()

    if args.dry_run:
        state["results"].append(
            {
                **target,
                "status": "planned",
                "command": command,
            }
        )
        return

    started = time.perf_counter()
    subprocess.run(command, cwd=str(SCRIPT_DIR), check=True)
    summary_path = latest_child_file(target_root, "summary.json")
    summary = load_json(summary_path)
    rows = summary.get("lag_response", [])
    row = rows[0] if rows else {}
    state["results"].append(
        {
            **target,
            "status": "completed",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "child_summary": str(summary_path),
            "B_retention": row.get("B_retention"),
            "markov1_retention": row.get("markov1_retention"),
            "block_entropy_max": row.get("block_entropy_max"),
            "B_minus_block_entropy_max": row.get("B_minus_block_entropy_max"),
            "block_entropy_n": row.get("block_entropy_n"),
        }
    )


def render_console_summary(state: dict[str, Any]) -> str:
    lines = [
        "Phase 4 P4-02 block-entropy transition-entropy null scout",
        "-" * 96,
        f"{'target':<28} {'B':>8} {'markov1':>8} {'bent_max':>9} {'B-bent':>9} {'n':>4}",
    ]
    for row in state["results"]:
        lines.append(
            f"{str(row['name']):<28} "
            f"{fmt_float(row.get('B_retention')):>8} "
            f"{fmt_float(row.get('markov1_retention')):>8} "
            f"{fmt_float(row.get('block_entropy_max')):>9} "
            f"{fmt_float(row.get('B_minus_block_entropy_max')):>9} "
            f"{str(row.get('block_entropy_n', '-')):>4}"
        )
    return "\n".join(lines)


def render_report(state: dict[str, Any], run_dir: Path) -> str:
    lines = [
        "# HSI v2 Phase 4 P4-02 block-entropy transition-entropy null scout",
        "",
        (
            "This scout tests a non-LZ seeded block-shuffle null selected to "
            "preserve local transition entropy, not LZ profile."
        ),
        "It is an internal Phase 4 probe, not a paper-facing claim.",
        "",
        "## Parameters",
        "",
        f"- Null models: `{state['parameters']['null_models']}`",
        f"- Seeds: `{state['parameters']['seeds']}`",
        f"- Top patterns: `{state['parameters']['top_patterns']}`",
        f"- Workers: `{state['parameters']['workers']}`",
        f"- Status: `{state['status']}`",
        "",
        "## Scout Readout",
        "",
        "| Target | Band start | Lag | B retention | markov1 | block-entropy max | B - max(block-entropy) | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in state["results"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
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
