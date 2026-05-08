#!/usr/bin/env python3
"""Build a cheap Phase 3 robustness matrix from existing N2-14/N3-07 runs.

This script performs no new heavy measurement. It reads transition-graph
summaries and flow-entropy summaries, then writes a compact matrix that compares
how topology-level phase-channel structure and thermodynamic routing-flow
separation behave across null families.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "hsi_v2" / "phase3" / "robustness_matrix"

DEFAULT_CASES = [
    {
        "label": "matched-lz canonical",
        "null_family": "matched-lz",
        "transition_run": "results/hsi_v2/phase2/phase_channel_transition_graph/phase2-phase-channel-transition-graph__runs-2__20260430T140724",
        "flow_run": "results/hsi_v2/phase3/flow_entropy_readout/phase3-flow-entropy-readout__src-620fcc3412__20260501T140015",
        "note": "Original seeded matched-LZ envelope.",
    },
    {
        "label": "matched-lz disjoint seeds",
        "null_family": "matched-lz-disjoint",
        "transition_run": "results/hsi_v2/phase2/phase_channel_transition_graph/phase2-phase-channel-transition-graph__runs-2__20260430T140724",
        "flow_run": "results/hsi_v2/phase3/flow_entropy_readout_independent_seeds/phase3-flow-entropy-readout__src-e8827e75d7__20260502T153352",
        "note": "Disjoint-seed rerun of N3-07 over the fixed N2-14 transition circuit.",
    },
    {
        "label": "phase-matched-lz",
        "null_family": "phase-matched-lz",
        "transition_run": "results/hsi_v2/phase2/phase_channel_transition_graph_phase_matched_lz_reval/phase2-phase-channel-transition-graph__runs-1__20260503T073426",
        "flow_run": "results/hsi_v2/phase3/flow_entropy_readout_phase_matched_lz_reval/phase3-flow-entropy-readout__src-9de9a95303__20260503T073503",
        "note": "Phase-preserving LZ-family null from D-0075/D-0076.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a derived Phase 3 robustness matrix from existing summaries."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-patterns", type=int, default=128)
    parser.add_argument(
        "--case-file",
        help=(
            "JSON file with a list of case objects. Each object must include "
            "label, null_family, transition_run, and flow_run."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="LABEL|NULL_FAMILY|TRANSITION_RUN|FLOW_RUN",
        help=(
            "Optional case spec. Repeat to override defaults. Use exact run "
            "directories containing summary.json unless --allow-latest is set."
        ),
    )
    parser.add_argument(
        "--allow-latest",
        action="store_true",
        help=(
            "Exploratory mode only: if a case path is a parent directory, resolve "
            "the most recent child containing summary.json. Disabled by default for reproducibility."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def log(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


def as_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_run(root_or_run: Path, required_file: str = "summary.json", allow_latest: bool = False) -> Path:
    root_or_run = root_or_run.resolve()
    if (root_or_run / required_file).exists():
        return root_or_run
    if not root_or_run.exists():
        raise FileNotFoundError(f"Run root not found: {root_or_run}")
    if not allow_latest:
        raise FileNotFoundError(
            f"Expected exact run directory containing {required_file}, got parent: {root_or_run}. "
            "Pass an exact timestamped run path, or use --allow-latest for local exploratory work."
        )

    candidates = [
        item
        for item in root_or_run.iterdir()
        if item.is_dir() and (item / required_file).exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No child run with {required_file}: {root_or_run}")
    return max(candidates, key=lambda item: item.stat().st_mtime).resolve()


def normalize_case(raw: dict[str, Any]) -> dict[str, str]:
    required = {"label", "null_family", "transition_run", "flow_run"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Case is missing required fields: {sorted(missing)}")
    normalized = {
        "label": str(raw["label"]),
        "null_family": str(raw["null_family"]),
        "transition_run": str(raw["transition_run"]),
        "flow_run": str(raw["flow_run"]),
        "note": str(raw.get("note", "")),
    }
    if not all(normalized[field].strip() for field in required):
        raise ValueError(f"Case contains an empty required field: {raw}")
    return normalized


def parse_case_specs(specs: list[str], case_file: str | None) -> list[dict[str, str]]:
    if specs and case_file:
        raise ValueError("Use either --case or --case-file, not both.")

    if case_file:
        payload = load_json(as_path(case_file))
        if isinstance(payload, dict):
            raw_cases = payload.get("cases")
        else:
            raw_cases = payload
        if not isinstance(raw_cases, list):
            raise ValueError("--case-file must contain a list or an object with a 'cases' list.")
        return [normalize_case(item) for item in raw_cases]

    if not specs:
        return [normalize_case(item) for item in DEFAULT_CASES]

    parsed: list[dict[str, str]] = []
    for spec in specs:
        parts = [part.strip() for part in spec.split("|")]
        if len(parts) != 4 or not all(parts):
            raise ValueError(
                "--case must be LABEL|NULL_FAMILY|TRANSITION_ROOT|FLOW_ROOT"
            )
        label, null_family, transition_root, flow_root = parts
        parsed.append(
            normalize_case({
                "label": label,
                "null_family": null_family,
                "transition_run": transition_root,
                "flow_run": flow_root,
                "note": "User-specified case.",
            })
        )
    return parsed


def transition_row(summary: dict[str, Any], top_patterns: int, channel: str) -> dict[str, Any]:
    for row in summary.get("paths", []):
        if int(row.get("top_patterns", -1)) == top_patterns and row.get("channel") == channel:
            return row
    raise KeyError(f"Missing transition path for top={top_patterns}, channel={channel}")


def observed_flow_rows(summary: dict[str, Any], channel: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in summary.get("flow_entropy_rows", []):
        if row.get("channel") != channel:
            continue
        if row.get("source_class") != "observed":
            continue
        variant = str(row.get("variant", ""))
        if variant in {"B", "E"}:
            rows[variant] = row
    missing = {"B", "E"} - set(rows)
    if missing:
        raise KeyError(f"Missing observed flow rows for channel={channel}: {sorted(missing)}")
    return rows


def flow_score(row: dict[str, Any]) -> int:
    value = row.get("flow_entropy_score")
    return int(value) if value is not None else 0


def verdict_is_strict(row: dict[str, Any]) -> bool:
    return row.get("verdict") == "strict-low-entropy-flow" and flow_score(row) >= 4


def classify_case(rows: list[dict[str, Any]]) -> str:
    all_strict = all(
        row["B_verdict"] == "strict-low-entropy-flow"
        and row["E_verdict"] == "strict-low-entropy-flow"
        and row["B_score"] >= 4
        and row["E_score"] >= 4
        for row in rows
    )
    no_stable_transition = any("no-stable" in str(row["transition_verdict"]) for row in rows)
    boundary_marked_transition = any(int(row["boundary_count"]) > 0 for row in rows)
    strong_transition = all(
        "structured" in str(row["transition_verdict"])
        for row in rows
    )

    if all_strict and no_stable_transition:
        return "topology-weakened-flow-robust"
    if all_strict and boundary_marked_transition:
        return "boundary-marked-flow-robust"
    if all_strict and strong_transition:
        return "topology-structured-flow-robust"
    if all_strict:
        return "flow-robust"
    return "not-robust"


def format_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 Robustness Matrix",
        "",
        "Derived readout only: this report does not compute a new observable. It compares already generated N2-14 transition-graph summaries with N3-07 flow-entropy summaries.",
        "",
        "## Case Summary",
        "",
        "| Case | Null family | Interpretation | Strict channels | Boundary channels |",
        "|---|---:|---|---:|---:|",
    ]
    for case in summary["case_summary"]:
        lines.append(
            "| {label} | {null_family} | {interpretation} | {strict_channels}/2 | {boundary_channels} |".format(
                **case
            )
        )

    lines.extend(
        [
            "",
            "## Channel Matrix",
            "",
            "| Case | Channel | Transition | Active | Boundary | B score | E score | B dH | E dH |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["channel_rows"]:
        lines.append(
            "| {case_label} | {channel} | {transition_verdict} | {active_count}/{band_count} | {boundary_count} | {B_score} | {E_score} | {B_entropy_gap} | {E_entropy_gap} |".format(
                **{**row, "B_entropy_gap": format_float(row["B_entropy_gap"]), "E_entropy_gap": format_float(row["E_entropy_gap"])}
            )
        )

    lines.extend(
        [
            "",
            "## Reading",
            "",
        "- `matched-lz canonical` and `matched-lz disjoint seeds` test stochastic robustness of the same fixed circuit.",
        "- `phase-matched-lz` is the stronger phase-preserving LZ-family null; if transition structure weakens but N3-07 stays strict, the correct reading is thermodynamic routing-flow robustness rather than recovered geometry.",
        "- The script uses exact timestamped run directories by default. `--allow-latest` is exploratory-only and is not used for this run.",
        "- This matrix is an internal synthesis artifact. It does not by itself authorize a manuscript upgrade.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    cases = parse_case_specs(args.case, args.case_file)
    output_root = as_path(args.output_dir)

    resolved_cases: list[dict[str, str]] = []
    for case in cases:
        transition_run = resolve_run(as_path(case["transition_run"]), allow_latest=args.allow_latest)
        flow_run = resolve_run(as_path(case["flow_run"]), allow_latest=args.allow_latest)
        resolved_cases.append(
            {
                **case,
                "transition_run": str(transition_run),
                "flow_run": str(flow_run),
            }
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"phase3-robustness-matrix__cases-{len(cases)}__top-{args.top_patterns}__{timestamp}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    log("[Phase] Preparing Phase 3 robustness matrix", args.quiet)
    log(f"        cases={len(cases)} | top={args.top_patterns}", args.quiet)

    channel_rows: list[dict[str, Any]] = []
    case_summary: list[dict[str, Any]] = []

    for index, case in enumerate(resolved_cases, start=1):
        label = case["label"]
        transition_run = Path(case["transition_run"])
        flow_run = Path(case["flow_run"])
        transition_summary = load_json(transition_run / "summary.json")
        flow_summary = load_json(flow_run / "summary.json")

        log(f"[Case] {index}/{len(cases)} | {label}", args.quiet)
        log(f"       transition={transition_run.name}", args.quiet)
        log(f"       flow={flow_run.name}", args.quiet)

        case_rows: list[dict[str, Any]] = []
        for channel in ("negative", "positive"):
            trow = transition_row(transition_summary, args.top_patterns, channel)
            flows = observed_flow_rows(flow_summary, channel)
            brow = flows["B"]
            erow = flows["E"]
            row = {
                "case_label": label,
                "null_family": case["null_family"],
                "channel": channel,
                "top_patterns": args.top_patterns,
                "transition_verdict": trow.get("verdict", ""),
                "active_count": int(trow.get("active_count", 0)),
                "band_count": int(trow.get("band_count", 0)),
                "boundary_count": int(trow.get("boundary_count", 0)),
                "max_active_path_bands": int(trow.get("max_active_path_bands", 0)),
                "B_score": flow_score(brow),
                "B_verdict": brow.get("verdict", ""),
                "B_entropy_gap": brow.get("entropy_gap_vs_best_matched"),
                "B_effective_gap": brow.get("effective_gap_vs_best_matched"),
                "B_top3_gap": brow.get("top3_gap_vs_best_matched"),
                "E_score": flow_score(erow),
                "E_verdict": erow.get("verdict", ""),
                "E_entropy_gap": erow.get("entropy_gap_vs_best_matched"),
                "E_effective_gap": erow.get("effective_gap_vs_best_matched"),
                "E_top3_gap": erow.get("top3_gap_vs_best_matched"),
                "transition_run": str(transition_run),
                "flow_run": str(flow_run),
            }
            case_rows.append(row)
            channel_rows.append(row)

        interpretation = classify_case(case_rows)
        for row in case_rows:
            row["interpretation"] = interpretation

        strict_channels = sum(
            1
            for row in case_rows
            if row["B_verdict"] == "strict-low-entropy-flow"
            and row["E_verdict"] == "strict-low-entropy-flow"
            and row["B_score"] >= 4
            and row["E_score"] >= 4
        )
        boundary_channels = sum(1 for row in case_rows if row["boundary_count"] > 0)
        case_summary.append(
            {
                "label": label,
                "null_family": case["null_family"],
                "interpretation": interpretation,
                "strict_channels": strict_channels,
                "boundary_channels": boundary_channels,
                "transition_run": str(transition_run),
                "flow_run": str(flow_run),
                "note": case.get("note", ""),
            }
        )

    summary_payload = {
        "run_id": run_id,
        "generated_at": timestamp,
        "selection": {
            "top_patterns": args.top_patterns,
            "cases": resolved_cases,
            "allow_latest": args.allow_latest,
        },
        "case_summary": case_summary,
        "channel_rows": channel_rows,
        "notes": [
            "Derived readout only: no new heavy observable is computed.",
            "The disjoint matched-LZ case intentionally reuses the fixed N2-14 transition circuit.",
            "Legacy field names containing 'matched' mean the active seeded LZ-family envelope inside each N3-07 run.",
            "Default cases are exact timestamped run directories; --allow-latest is exploratory-only.",
        ],
    }

    matrix_fields = [
        "case_label",
        "null_family",
        "channel",
        "top_patterns",
        "transition_verdict",
        "active_count",
        "band_count",
        "boundary_count",
        "max_active_path_bands",
        "B_score",
        "B_verdict",
        "B_entropy_gap",
        "B_effective_gap",
        "B_top3_gap",
        "E_score",
        "E_verdict",
        "E_entropy_gap",
        "E_effective_gap",
        "E_top3_gap",
        "interpretation",
        "transition_run",
        "flow_run",
    ]
    case_fields = [
        "label",
        "null_family",
        "interpretation",
        "strict_channels",
        "boundary_channels",
        "transition_run",
        "flow_run",
        "note",
    ]

    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    matrix_path = run_dir / "robustness_matrix.csv"
    case_path = run_dir / "case_summary.csv"
    manifest_path = run_dir / "manifest.json"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
        handle.write("\n")
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(render_report(summary_payload) + "\n")
    write_csv(matrix_path, channel_rows, matrix_fields)
    write_csv(case_path, case_summary, case_fields)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "script": str(Path(__file__).resolve()),
                "generated_at": timestamp,
                "outputs": {
                    "summary": str(summary_path),
                    "report": str(report_path),
                    "robustness_matrix": str(matrix_path),
                    "case_summary": str(case_path),
                },
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    log("Phase 3 robustness matrix", args.quiet)
    log("-" * 112, args.quiet)
    log(f"{'case':30s} {'family':22s} {'interp':34s} {'strict':>7s} {'bound':>6s}", args.quiet)
    for row in case_summary:
        log(
            f"{row['label'][:30]:30s} {row['null_family'][:22]:22s} "
            f"{row['interpretation'][:34]:34s} {row['strict_channels']:7d} {row['boundary_channels']:6d}",
            args.quiet,
        )
    log("", args.quiet)
    log(f"Saved summary to: {summary_path}", args.quiet)
    log(f"Saved report to: {report_path}", args.quiet)
    log(f"Saved robustness CSV to: {matrix_path}", args.quiet)
    log(f"Saved manifest to: {manifest_path}", args.quiet)
    return 0


def cli() -> int:
    try:
        return main()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
