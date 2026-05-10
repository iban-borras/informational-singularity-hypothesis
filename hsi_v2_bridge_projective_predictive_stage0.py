#!/usr/bin/env python3
"""Stage 0 feasibility for projective-density / predictive-channel alignment.

Implements D-0107. This script is intentionally blind: it may collect frozen
predictive scores and raw local Phase 1 metrics, but it never computes an
alignment statistic, rank comparison, plot, or qualitative alignment verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hsi_v2_phase2_transport_defect_strict import phase_print
from utils.progress import format_time
from v2.common.cli import resolve_dir
from v2.phase1.report import discover_phase1_runs, filter_runs
from v2.phase1.transport import build_transport_rows


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_OUTPUT_DIR = "results/hsi_v2/bridge_projective_predictive_alignment/stage0_feasibility"
DEFAULT_PHASE1_LOCAL_DIR = "results/hsi_v2/bridge_projective_predictive_alignment/stage0_feasibility/phase1_local"
DEFAULT_PHASE4_DIR = "results/hsi_v2/phase4"
DEFAULT_BMINI_SUMMARY = (
    "results/hsi_v2/phase5_cross_arc_horizon/bmini_v0_2/"
    "phase5-bmini-cross-arc-horizon__arcs-3__20260508T153520/summary.json"
)
DEFAULT_CONTRACT = "Documentation/HSI_v2_Bridge_ProjectiveDensity_PredictiveAlignment_Preregistration_v0_1.md"

SCALES = "8,12,16,20,24,28,32"
POLICIES = "prefix,suffix"
SEGMENT_BITS = 1_000_000
NUM_SEGMENTS = 3

P4_LAGS = {-29_500_000, -27_000_000, -26_500_000, -24_000_000, 24_500_000, 26_500_000}

FROZEN_BANDS = [
    {"arc_id": "P4", "band": "768M-777M", "start_bits": 768_000_000},
    {"arc_id": "P4", "band": "777M-786M", "start_bits": 777_000_000},
    {"arc_id": "P4", "band": "786M-795M", "start_bits": 786_000_000},
    {"arc_id": "P4", "band": "795M-804M", "start_bits": 795_000_000},
    {"arc_id": "P4", "band": "804M-813M", "start_bits": 804_000_000},
    {"arc_id": "P4", "band": "813M-822M", "start_bits": 813_000_000},
    {"arc_id": "P4", "band": "822M-831M", "start_bits": 822_000_000},
    {"arc_id": "ARC-A", "band": "300M-309M", "start_bits": 300_000_000},
    {"arc_id": "ARC-A", "band": "309M-318M", "start_bits": 309_000_000},
    {"arc_id": "ARC-A", "band": "318M-327M", "start_bits": 318_000_000},
    {"arc_id": "ARC-A", "band": "327M-336M", "start_bits": 327_000_000},
    {"arc_id": "ARC-A", "band": "336M-345M", "start_bits": 336_000_000},
    {"arc_id": "ARC-B", "band": "597M-606M", "start_bits": 597_000_000},
    {"arc_id": "ARC-B", "band": "606M-615M", "start_bits": 606_000_000},
    {"arc_id": "ARC-B", "band": "615M-624M", "start_bits": 615_000_000},
    {"arc_id": "ARC-B", "band": "624M-633M", "start_bits": 624_000_000},
    {"arc_id": "ARC-B", "band": "633M-642M", "start_bits": 633_000_000},
    {"arc_id": "ARC-C", "band": "1110M-1119M", "start_bits": 1_110_000_000},
    {"arc_id": "ARC-C", "band": "1119M-1128M", "start_bits": 1_119_000_000},
    {"arc_id": "ARC-C", "band": "1128M-1137M", "start_bits": 1_128_000_000},
    {"arc_id": "ARC-C", "band": "1137M-1146M", "start_bits": 1_137_000_000},
    {"arc_id": "ARC-C", "band": "1146M-1155M", "start_bits": 1_146_000_000},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D-0107 Stage 0 bridge feasibility audit.")
    parser.add_argument("--phase4-dir", default=DEFAULT_PHASE4_DIR)
    parser.add_argument("--bmini-summary", default=DEFAULT_BMINI_SUMMARY)
    parser.add_argument("--phase1-local-dir", default=DEFAULT_PHASE1_LOCAL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--run-phase1", action="store_true", help="Run missing local Phase 1 towers.")
    parser.add_argument("--workers", type=int, default=1, help="Reserved for future process scheduling; currently sequential.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    started = time.perf_counter()
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    phase1_local_dir = resolve_dir(args.phase1_local_dir, anchor_file=__file__)
    phase4_dir = resolve_dir(args.phase4_dir, anchor_file=__file__)
    bmini_summary = resolve_dir(args.bmini_summary, anchor_file=__file__)
    contract = resolve_dir(args.contract, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)
    phase1_local_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"bridge-projective-predictive-stage0__bands-22__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "run_id": run_dir.name,
        "timestamp_utc": timestamp,
        "contract": {"path": str(contract.resolve()), "sha256": sha256_file(contract)},
        "mode": "run-phase1" if args.run_phase1 else "plan-only",
        "blindness": {
            "alignment_statistics_computed": False,
            "scatter_or_heatmap_created": False,
            "qualitative_alignment_verdict": False,
        },
        "parameters": {
            "phase4_dir": str(phase4_dir),
            "bmini_summary": str(bmini_summary),
            "phase1_local_dir": str(phase1_local_dir),
            "segment_bits": SEGMENT_BITS,
            "num_segments": NUM_SEGMENTS,
            "scales": SCALES,
            "policies": POLICIES,
        },
    }

    try:
        phase_print(
            "Preparing D-0107 Stage 0 bridge feasibility",
            f"bands=22 | mode={state['mode']} | output={run_dir.name}",
            quiet=args.quiet,
        )
        bmini_rows = load_bmini_strict_rows(bmini_summary)
        p4_rows = load_p4_rows(phase4_dir)
        predictive_rows = build_predictive_rows(bmini_rows=bmini_rows, p4_rows=p4_rows)

        local_runs = discover_local_phase1_runs(phase1_local_dir)
        commands = build_phase1_commands(phase1_local_dir)
        missing = [row for row in FROZEN_BANDS if row["start_bits"] not in local_runs]

        if args.run_phase1 and missing:
            for index, band in enumerate(missing, start=1):
                cmd = commands[band["start_bits"]]
                phase_print(
                    f"Local Phase 1 {index}/{len(missing)}",
                    f"{band['arc_id']} {band['band']} | offset={band['start_bits']}",
                    quiet=args.quiet,
                )
                subprocess.run(cmd, cwd=str(SCRIPT_DIR), check=True)
            local_runs = discover_local_phase1_runs(phase1_local_dir)

        phase1_rows = build_phase1_metric_rows(local_runs)
        frozen_rows = []
        for band in FROZEN_BANDS:
            predictive = predictive_rows.get(band["band"], {})
            phase1 = phase1_rows.get(band["start_bits"], {})
            frozen_rows.append(
                {
                    **band,
                    "predictive_available": bool(predictive),
                    "phase1_local_available": bool(phase1),
                    "phase1_command": shell_join(commands[band["start_bits"]]),
                    "phase1_summary_path": phase1.get("summary_path", ""),
                }
            )

        metric_feasibility = build_metric_feasibility(frozen_rows, phase1_rows)
        verdict = build_verdict(metric_feasibility, local_runs_executed=args.run_phase1)
        state.update(
            {
                "status": "completed",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "frozen_bands": frozen_rows,
                "predictive_scores": list(predictive_rows.values()),
                "phase1_local_metrics_raw": list(phase1_rows.values()),
                "metric_feasibility": metric_feasibility,
                "verdict": verdict,
            }
        )
        write_outputs(run_dir, state)
        print(render_console_summary(state))
        print_saved_paths(run_dir)
        return 0
    except subprocess.CalledProcessError as exc:
        state["status"] = "failed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        state["error"] = {"returncode": int(exc.returncode), "command": [str(part) for part in exc.cmd]}
        write_outputs(run_dir, state)
        print(f"[x] Stage 0 failed after {format_time(time.perf_counter() - started)}")
        print_saved_paths(run_dir)
        return int(exc.returncode) if int(exc.returncode) > 0 else 1


def load_bmini_strict_rows(summary_path: Path) -> list[dict[str, Any]]:
    payload = load_json(summary_path)
    return list(payload.get("strict_p4_rows", []))


def load_p4_rows(phase4_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(phase4_dir.glob("p4_04_forward_regime_forecast/**/lag_response.csv")):
        rows.extend(load_csv_dicts(csv_path, source_path=csv_path))
    for csv_path in sorted(phase4_dir.glob("p4_05_horizon_bracket_forecast/**/lag_response.csv")):
        rows.extend(load_csv_dicts(csv_path, source_path=csv_path))
    return rows


def build_predictive_rows(*, bmini_rows: list[dict[str, Any]], p4_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {band["band"]: [] for band in FROZEN_BANDS}
    for row in bmini_rows:
        if row.get("phase") != "strict_p4":
            continue
        if int_or_none(row.get("lag_bits")) not in P4_LAGS:
            continue
        if row.get("band") in grouped:
            grouped[row["band"]].append(row)
    for row in p4_rows:
        if int_or_none(row.get("lag_bits")) not in P4_LAGS:
            continue
        if row.get("band") in grouped:
            grouped[row["band"]].append(row)

    output = {}
    for band in FROZEN_BANDS:
        rows = grouped[band["band"]]
        if not rows:
            continue
        hard_margins = [row_hard_margin(row) for row in rows if row_hard_margin(row) is not None]
        b_retentions = [float_or_none(row.get("B_retention")) for row in rows if float_or_none(row.get("B_retention")) is not None]
        if not hard_margins or not b_retentions:
            continue
        hard_margin_max = max(hard_margins)
        b_retention_max = max(b_retentions)
        strong = any(float_or_none(row.get("B_retention"), 0.0) >= 0.90 and row_hard_margin(row, -999.0) >= 0.30 for row in rows)
        shadow = (not strong) and any(
            float_or_none(row.get("B_retention"), 0.0) >= 0.50 and row_hard_margin(row, 999.0) <= 0.05
            for row in rows
        )
        output[band["band"]] = {
            **band,
            "hard_margin_max": hard_margin_max,
            "B_retention_max": b_retention_max,
            "strict_band_class": "strong" if strong else "shadow" if shadow else "inactive",
            "predictive_row_count": len(rows),
            "source_paths": ";".join(sorted({str(row.get("_source_path", "")) for row in rows if row.get("_source_path")})),
        }
    return output


def discover_local_phase1_runs(phase1_local_dir: Path) -> dict[int, dict[str, Any]]:
    runs = discover_phase1_runs(phase1_local_dir)
    selected = filter_runs(
        runs,
        variants=["B"],
        iteration=20,
        segment_bits=SEGMENT_BITS,
        num_segments=NUM_SEGMENTS,
        scales=[8, 12, 16, 20, 24, 28, 32],
        policies=["prefix", "suffix"],
    )
    by_offset: dict[int, dict[str, Any]] = {}
    for run in selected:
        offset = int(run["dataset"]["config"].get("segment_offset_bits", 0))
        current = by_offset.get(offset)
        if current is None or run["_generated_at_dt"] > current["_generated_at_dt"]:
            by_offset[offset] = run
    return by_offset


def build_phase1_commands(phase1_local_dir: Path) -> dict[int, list[str]]:
    commands = {}
    for band in FROZEN_BANDS:
        commands[band["start_bits"]] = [
            sys.executable,
            str(SCRIPT_DIR / "hsi_v2_phase1_run.py"),
            "--variant",
            "B",
            "--iteration",
            "20",
            "--segment-bits",
            str(SEGMENT_BITS),
            "--num-segments",
            str(NUM_SEGMENTS),
            "--segment-offset-bits",
            str(band["start_bits"]),
            "--scales",
            SCALES,
            "--policies",
            POLICIES,
            "--output-dir",
            str(phase1_local_dir),
            "--quiet",
        ]
    return commands


def build_phase1_metric_rows(local_runs: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    output = {}
    for offset, run in sorted(local_runs.items()):
        summary = run["summary"]
        scale_summary = summary.get("scale_summary", [])
        fiber_summary = summary.get("fiber_summary", {})
        prefix = fiber_summary.get("prefix", [])
        suffix = fiber_summary.get("suffix", [])
        tail_survivals = [item["survival_fraction"] for item in prefix[-2:] + suffix[-2:] if "survival_fraction" in item]
        last_scale = scale_summary[-1] if scale_summary else {}
        retention_tail = mean(tail_survivals)
        pattern_density_tail = safe_div(last_scale.get("kept_pattern_count"), last_scale.get("candidate_pattern_count"))
        transport_row = None
        transport_error = ""
        try:
            transport_rows = build_transport_rows([run], main_policy="prefix", control_policy="suffix", tail_edges=2)
            transport_row = transport_rows[0] if transport_rows else None
        except Exception as exc:  # noqa: BLE001 - report feasibility, do not fail Stage 0.
            transport_error = str(exc)
        output[offset] = {
            "start_bits": offset,
            "projective_retention_tail": retention_tail,
            "pattern_density_tail": pattern_density_tail,
            "transport_active_mean_tail": (
                transport_row.get("tail_transport_active_mean") if transport_row else None
            ),
            "transport_metric_error": transport_error,
            "summary_path": run["_summary_path"],
            "run_dir": run["_run_dir"],
        }
    return output


def build_metric_feasibility(frozen_rows: list[dict[str, Any]], phase1_rows: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(FROZEN_BANDS)
    arcs = sorted({row["arc_id"] for row in FROZEN_BANDS})
    rows = []
    for metric in ("transport_active_mean_tail", "projective_retention_tail", "pattern_density_tail"):
        available = [
            offset for offset, payload in phase1_rows.items()
            if payload.get(metric) is not None
        ]
        per_arc = {}
        for arc in arcs:
            arc_bands = [band for band in FROZEN_BANDS if band["arc_id"] == arc]
            have = sum(1 for band in arc_bands if phase1_rows.get(band["start_bits"], {}).get(metric) is not None)
            per_arc[arc] = f"{have}/{len(arc_bands)}"
        rows.append(
            {
                "metric": metric,
                "available_bands": len(available),
                "required_bands": total,
                "all_bands_available": len(available) == total,
                "per_arc_availability": per_arc,
                "selection_priority": {"transport_active_mean_tail": 1, "projective_retention_tail": 2, "pattern_density_tail": 3}[metric],
            }
        )
    return rows


def build_verdict(metric_rows: list[dict[str, Any]], *, local_runs_executed: bool) -> dict[str, Any]:
    for row in sorted(metric_rows, key=lambda item: item["selection_priority"]):
        if row["all_bands_available"]:
            return {
                "status": "stage1-executable",
                "recommended_metric": row["metric"],
                "reason": "highest-priority candidate metric available for every frozen band",
            }
    any_local = any(row["available_bands"] > 0 for row in metric_rows)
    if not local_runs_executed and not any_local:
        return {
            "status": "local-phase1-runs-required",
            "recommended_metric": "",
            "reason": "no existing local Phase 1 runs found; rerun with --run-phase1 to compute raw Stage 0 metrics",
        }
    return {
        "status": "requires-new-contract-for-local-phase1-runs" if not any_local else "not-executable-with-current-artifacts",
        "recommended_metric": "",
        "reason": "no candidate metric is available for every frozen band",
    }


def write_outputs(run_dir: Path, state: dict[str, Any]) -> None:
    write_json(run_dir / "summary.json", state)
    write_csv(run_dir / "frozen_bands.csv", state.get("frozen_bands", []))
    write_csv(run_dir / "predictive_scores.csv", state.get("predictive_scores", []))
    write_csv(run_dir / "phase1_local_metrics_raw.csv", state.get("phase1_local_metrics_raw", []))
    write_csv(run_dir / "metric_feasibility.csv", flatten_metric_rows(state.get("metric_feasibility", [])))
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": state.get("run_id"),
            "timestamp_utc": state.get("timestamp_utc"),
            "status": state.get("status"),
            "contract": state.get("contract"),
            "mode": state.get("mode"),
            "blindness": state.get("blindness"),
            "parameters": state.get("parameters"),
            "outputs": [
                "summary.json",
                "report.md",
                "frozen_bands.csv",
                "predictive_scores.csv",
                "phase1_local_metrics_raw.csv",
                "metric_feasibility.csv",
                "manifest.json",
            ],
        },
    )
    (run_dir / "report.md").write_text(render_report(state), encoding="utf-8")


def render_report(state: dict[str, Any]) -> str:
    verdict = state.get("verdict", {})
    lines = [
        "# D-0107 Stage 0 Bridge Feasibility",
        "",
        "## Blindness",
        "",
        "- Alignment statistics computed: `false`",
        "- Scatter / heatmap overlay created: `false`",
        "- Qualitative alignment verdict: `false`",
        "",
        "## Verdict",
        "",
        f"- Status: `{verdict.get('status', '-')}`",
        f"- Recommended metric: `{verdict.get('recommended_metric') or '-'}`",
        f"- Reason: {verdict.get('reason', '-')}",
        "",
        "## Metric Feasibility",
        "",
        "| metric | available | all bands | P4 | ARC-A | ARC-B | ARC-C |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in state.get("metric_feasibility", []):
        per_arc = row["per_arc_availability"]
        lines.append(
            f"| `{row['metric']}` | {row['available_bands']}/{row['required_bands']} | "
            f"`{row['all_bands_available']}` | {per_arc.get('P4', '-')} | {per_arc.get('ARC-A', '-')} | "
            f"{per_arc.get('ARC-B', '-')} | {per_arc.get('ARC-C', '-')} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This report intentionally does not compare Phase 1 local metrics to predictive-channel scores.",
            "- `predictive_scores.csv` and `phase1_local_metrics_raw.csv` are separate outputs.",
            "- Stage 1 remains unauthorized unless a future v0.2 contract is accepted.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_console_summary(state: dict[str, Any]) -> str:
    verdict = state.get("verdict", {})
    lines = [
        "D-0107 Stage 0 bridge feasibility",
        "-" * 88,
        f"mode: {state.get('mode')}",
        f"status: {state.get('status')}",
        f"verdict: {verdict.get('status', '-')}",
        f"recommended_metric: {verdict.get('recommended_metric') or '-'}",
        "",
        f"{'metric':32} {'available':>10} {'P4':>8} {'ARC-A':>8} {'ARC-B':>8} {'ARC-C':>8}",
    ]
    for row in state.get("metric_feasibility", []):
        per_arc = row["per_arc_availability"]
        lines.append(
            f"{row['metric']:32} {row['available_bands']}/{row['required_bands']:>7} "
            f"{per_arc.get('P4', '-'):>8} {per_arc.get('ARC-A', '-'):>8} "
            f"{per_arc.get('ARC-B', '-'):>8} {per_arc.get('ARC-C', '-'):>8}"
        )
    return "\n".join(lines)


def print_saved_paths(run_dir: Path) -> None:
    print(f"\nSaved summary to: {run_dir / 'summary.json'}")
    print(f"Saved report to: {run_dir / 'report.md'}")
    print(f"Saved frozen bands CSV to: {run_dir / 'frozen_bands.csv'}")
    print(f"Saved metric feasibility CSV to: {run_dir / 'metric_feasibility.csv'}")
    print(f"Saved manifest to: {run_dir / 'manifest.json'}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_csv_dicts(path: Path, *, source_path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["_source_path"] = str(source_path)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        per_arc = row.get("per_arc_availability", {})
        flattened.append(
            {
                "metric": row["metric"],
                "available_bands": row["available_bands"],
                "required_bands": row["required_bands"],
                "all_bands_available": row["all_bands_available"],
                "P4": per_arc.get("P4", ""),
                "ARC-A": per_arc.get("ARC-A", ""),
                "ARC-B": per_arc.get("ARC-B", ""),
                "ARC-C": per_arc.get("ARC-C", ""),
                "selection_priority": row["selection_priority"],
            }
        )
    return flattened


def row_hard_margin(row: dict[str, Any], default: float | None = None) -> float | None:
    direct = float_or_none(row.get("hard_margin"))
    if direct is not None:
        return direct
    candidates = [
        float_or_none(row.get("B_minus_matched_lz_max")),
        float_or_none(row.get("B_minus_phase_matched_lz_max")),
        float_or_none(row.get("B_minus_block_entropy_max")),
    ]
    candidates = [value for value in candidates if value is not None]
    if not candidates:
        return default
    return min(candidates)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def safe_div(numerator: Any, denominator: Any) -> float | None:
    numerator_f = float_or_none(numerator)
    denominator_f = float_or_none(denominator)
    if numerator_f is None or denominator_f in (None, 0.0):
        return None
    return numerator_f / denominator_f


def float_or_none(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def shell_join(parts: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in parts)


if __name__ == "__main__":
    raise SystemExit(main())
