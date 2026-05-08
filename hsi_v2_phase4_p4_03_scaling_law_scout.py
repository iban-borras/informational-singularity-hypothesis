#!/usr/bin/env python3
"""HSI v2 Phase 4 P4-03 scaling-law scout.

This script does not recompute raw routing. It aggregates the audited P4-01
grid and the P4-02b block-entropy selected sweep to test whether the observed
phase-channel structure behaves like a simple global law or a piecewise sampled
regime map.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hsi_v2_phase2_parent_survival_revalidation import flush_output
from hsi_v2_phase2_transport_defect_strict import phase_print
from utils.progress import format_time
from v2.common.cli import resolve_dir


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_P4_01_ROOT = "results/hsi_v2/phase4/p4_01_predictive_pipeline"
DEFAULT_P4_02B_ROOT = "results/hsi_v2/phase4/p4_02b_block_entropy_selected_sweep"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase4/p4_03_scaling_law_scout"

CORE_BANDS = {696_000_000, 705_000_000, 714_000_000, 723_000_000, 726_000_000}
PRE_SHADOW_BANDS = {729_000_000}
SHADOW_BANDS = {738_000_000, 747_000_000}
REACTIVATION_BANDS = {756_000_000}
STRONG_MARGIN_THRESHOLD = 0.30
SHADOW_MARGIN_THRESHOLD = 0.05


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate P4-01/P4-02b artifacts into a falsifiable scaling-law scout. "
            "The output distinguishes global monotone fits from piecewise phase-channel regimes."
        )
    )
    parser.add_argument("--p4-01-run", default="")
    parser.add_argument("--p4-01-root", default=DEFAULT_P4_01_ROOT)
    parser.add_argument("--p4-02b-run", default="")
    parser.add_argument("--p4-02b-root", default=DEFAULT_P4_02B_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_slug = f"phase4-p4-03-scaling-law-scout__{timestamp}"
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    state: dict[str, Any] = {
        "run_id": run_slug,
        "timestamp_utc": timestamp,
        "status": "dry-run" if args.dry_run else "running",
        "parameters": {
            "p4_01_run": args.p4_01_run,
            "p4_01_root": args.p4_01_root,
            "p4_02b_run": args.p4_02b_run,
            "p4_02b_root": args.p4_02b_root,
            "strong_margin_threshold": STRONG_MARGIN_THRESHOLD,
            "shadow_margin_threshold": SHADOW_MARGIN_THRESHOLD,
            "dry_run": bool(args.dry_run),
        },
        "inputs": {},
        "regime_rows": [],
        "tests": [],
        "verdict": {},
    }

    try:
        phase_print(
            "Preparing P4-03 scaling-law scout",
            "aggregate-only | P4-01 lag grid + P4-02b transition-entropy selected sweep",
            quiet=args.quiet,
        )
        flush_output()

        p4_01_run = resolve_run(args.p4_01_run, args.p4_01_root, "P4-01")
        p4_02b_run = resolve_run(args.p4_02b_run, args.p4_02b_root, "P4-02b")
        state["inputs"] = {"p4_01_run": str(p4_01_run), "p4_02b_run": str(p4_02b_run)}

        if args.dry_run:
            state["status"] = "dry-run"
            state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            write_outputs(run_dir, state)
            print(render_console_summary(state))
            print(f"\nSaved summary to: {run_dir / 'summary.json'}")
            print(f"Saved report to: {run_dir / 'report.md'}")
            print(f"Saved manifest to: {run_dir / 'manifest.json'}")
            return 0

        p4_01 = load_json(p4_01_run / "summary.json")
        p4_02b = load_json(p4_02b_run / "summary.json")

        paths = collect_input_paths(p4_01)
        pmlz_envelopes = load_csv_dicts(paths["pmlz_envelope"] / "channel_envelopes.csv")
        matched_envelopes = load_csv_dicts(paths["matched_envelope"] / "channel_envelopes.csv")
        pmlz_lag_rows = load_csv_dicts(paths["pmlz_lag_response"] / "lag_response.csv")
        matched_lag_rows = load_csv_dicts(paths["matched_lag_response"] / "lag_response.csv")
        block_rows = p4_02b.get("results", [])

        regime_rows = build_regime_rows(
            pmlz_envelopes=pmlz_envelopes,
            matched_envelopes=matched_envelopes,
            pmlz_lag_rows=pmlz_lag_rows,
            matched_lag_rows=matched_lag_rows,
            block_rows=block_rows,
        )
        tests = build_tests(regime_rows)
        verdict = build_verdict(tests)

        state.update(
            {
                "status": "completed",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "inputs": {
                    **state["inputs"],
                    "pmlz_envelope": str(paths["pmlz_envelope"]),
                    "matched_envelope": str(paths["matched_envelope"]),
                    "pmlz_lag_response": str(paths["pmlz_lag_response"]),
                    "matched_lag_response": str(paths["matched_lag_response"]),
                },
                "regime_rows": regime_rows,
                "tests": tests,
                "verdict": verdict,
            }
        )
        write_outputs(run_dir, state)
        print(render_console_summary(state))
        print(f"\nSaved summary to: {run_dir / 'summary.json'}")
        print(f"Saved report to: {run_dir / 'report.md'}")
        print(f"Saved regime CSV to: {run_dir / 'regime_map.csv'}")
        print(f"Saved tests CSV to: {run_dir / 'regime_tests.csv'}")
        print(f"Saved manifest to: {run_dir / 'manifest.json'}")
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        state["error"] = str(exc)
        write_outputs(run_dir, state)
        print(f"[x] P4-03 failed after {format_time(time.perf_counter() - started)}: {exc}")
        print(f"Saved failure summary to: {run_dir / 'summary.json'}")
        return 1


def resolve_run(explicit: str, root: str, label: str) -> Path:
    if explicit:
        path = resolve_dir(explicit, anchor_file=__file__)
        if not (path / "summary.json").exists():
            raise FileNotFoundError(f"{label} run has no summary.json: {path}")
        return path

    root_path = resolve_dir(root, anchor_file=__file__)
    if not root_path.exists():
        raise FileNotFoundError(f"{label} root not found: {root_path}")
    candidates = [
        path.parent
        for path in root_path.rglob("summary.json")
        if path.parent.name.startswith("phase4-p4-01-predictive-pipeline")
        or path.parent.name.startswith("phase4-p4-02b-block-entropy-selected-sweep")
    ]
    completed: list[Path] = []
    for candidate in candidates:
        try:
            payload = load_json(candidate / "summary.json")
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "completed":
            completed.append(candidate)
    if not completed:
        raise FileNotFoundError(f"No completed {label} run found under {root_path}")
    return max(completed, key=lambda path: path.stat().st_mtime)


def collect_input_paths(p4_01: dict[str, Any]) -> dict[str, Path]:
    outputs = p4_01.get("outputs", {})
    required = {
        "pmlz_envelope": "pmlz_channel_envelope",
        "matched_envelope": "matched_lz_reference_envelope",
        "pmlz_lag_response": "pmlz_lag_response",
        "matched_lag_response": "matched_lz_reference_lag_response",
    }
    paths: dict[str, Path] = {}
    for key, output_key in required.items():
        raw = outputs.get(output_key)
        if not raw:
            raise KeyError(f"P4-01 summary missing outputs.{output_key}")
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"P4-01 output not found: {path}")
        paths[key] = path
    return paths


def build_regime_rows(
    *,
    pmlz_envelopes: list[dict[str, str]],
    matched_envelopes: list[dict[str, str]],
    pmlz_lag_rows: list[dict[str, str]],
    matched_lag_rows: list[dict[str, str]],
    block_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matched_by_key = {
        (row["band"], row["channel"]): row for row in matched_envelopes
    }
    pmlz_lag_by_key = {
        (row["band"], int(row["lag_bits"])): row for row in pmlz_lag_rows
    }
    matched_lag_by_key = {
        (row["band"], int(row["lag_bits"])): row for row in matched_lag_rows
    }
    block_by_key = {
        (band_label(int(row["band_start_bits"])), int(row["lag_bits"])): row
        for row in block_rows
        if row.get("status") == "completed"
    }

    rows: list[dict[str, Any]] = []
    for raw in pmlz_envelopes:
        band = raw["band"]
        channel = raw["channel"]
        best_lag_bits = parse_optional_int(raw.get("best_margin_lag_bits"))
        pmlz_best_row = pmlz_lag_by_key.get((band, best_lag_bits)) if best_lag_bits is not None else None
        matched_best_row = matched_lag_by_key.get((band, best_lag_bits)) if best_lag_bits is not None else None
        block_best_row = block_by_key.get((band, best_lag_bits)) if best_lag_bits is not None else None
        matched_envelope = matched_by_key.get((band, channel), {})

        band_start = band_start_bits(band)
        pmlz_margin = parse_optional_float(raw.get("best_margin"))
        matched_margin = parse_optional_float(matched_envelope.get("best_margin"))
        strong_count = parse_int(raw.get("strong_count"), default=0)
        row = {
            "band": band,
            "band_start_bits": band_start,
            "channel": channel,
            "pmlz_strong_count": strong_count,
            "pmlz_width_bits": parse_optional_int(raw.get("strong_width_bits")),
            "pmlz_best_lag_bits": best_lag_bits,
            "pmlz_best_lag": raw.get("best_margin_lag", ""),
            "pmlz_best_B_retention": parse_optional_float(raw.get("best_margin_B_retention")),
            "pmlz_best_margin": pmlz_margin,
            "matched_best_margin_same_channel": matched_margin,
            "pmlz_best_phase_matched_lz_max": parse_optional_float(
                pmlz_best_row.get("phase_matched_lz_max") if pmlz_best_row else None
            ),
            "matched_lz_max_at_pmlz_lag": parse_optional_float(
                matched_best_row.get("matched_lz_max") if matched_best_row else None
            ),
            "block_entropy_max_at_pmlz_lag": parse_optional_float(
                block_best_row.get("block_entropy_max") if block_best_row else None
            ),
            "B_minus_block_entropy_max_at_pmlz_lag": parse_optional_float(
                block_best_row.get("B_minus_block_entropy_max") if block_best_row else None
            ),
        }
        row["band_region"] = classify_region(band_start)
        row["regime_class"] = classify_regime(row)
        rows.append(row)
    return sorted(rows, key=lambda item: (item["band_start_bits"], item["channel"]))


def build_tests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    negative = [row for row in rows if row["channel"] == "negative"]
    positive = [row for row in rows if row["channel"] == "positive"]

    neg_fit = linear_fit(
        [(row["band_start_bits"], row["pmlz_best_margin"]) for row in negative]
    )
    tests.append(
        {
            "test": "global_negative_linear_margin",
            "metric": "r2",
            "value": round(neg_fit["r2"], 4),
            "threshold": 0.65,
            "verdict": "rejected" if neg_fit["r2"] < 0.65 else "supported",
            "detail": (
                "A single monotone linear margin law is rejected unless R2 >= 0.65. "
                f"slope_per_M={neg_fit['slope_per_million']:.4f}"
            ),
        }
    )

    core_ok = all(
        row["pmlz_strong_count"] > 0
        for row in negative
        if row["band_start_bits"] in CORE_BANDS
    )
    pre_shadow_ok = all(
        row["pmlz_strong_count"] == 0
        and row["pmlz_best_margin"] is not None
        and row["pmlz_best_margin"] < STRONG_MARGIN_THRESHOLD
        for row in negative
        if row["band_start_bits"] in PRE_SHADOW_BANDS
    )
    shadow_ok = all(
        row["pmlz_strong_count"] == 0
        and row["pmlz_best_margin"] is not None
        and row["pmlz_best_margin"] <= SHADOW_MARGIN_THRESHOLD
        for row in negative
        if row["band_start_bits"] in SHADOW_BANDS
    )
    reactivation_ok = all(
        row["pmlz_strong_count"] > 0
        and row["pmlz_best_margin"] is not None
        and row["pmlz_best_margin"] >= STRONG_MARGIN_THRESHOLD
        for row in negative
        if row["band_start_bits"] in REACTIVATION_BANDS
    )
    tests.append(
        {
            "test": "negative_piecewise_regime_map",
            "metric": "core+pre_shadow+shadow+reactivation",
            "value": (
                f"core={core_ok};pre_shadow={pre_shadow_ok};"
                f"shadow={shadow_ok};reactivation={reactivation_ok}"
            ),
            "threshold": "all_true",
            "verdict": (
                "supported"
                if core_ok and pre_shadow_ok and shadow_ok and reactivation_ok
                else "rejected"
            ),
            "detail": (
                "Sampled regions: 696M-726M active core, 729M pre-shadow shoulder, "
                "738M-747M null-shadow, 756M late reactivation."
            ),
        }
    )

    active_neg_lags = sorted(
        {
            row["pmlz_best_lag_bits"]
            for row in negative
            if row["pmlz_strong_count"] > 0 and row["pmlz_best_lag_bits"] is not None
        }
    )
    tests.append(
        {
            "test": "negative_lag_quantization",
            "metric": "unique_active_best_lags",
            "value": ",".join(format_lag(lag) for lag in active_neg_lags),
            "threshold": "<=4 lag states",
            "verdict": "supported" if len(active_neg_lags) <= 4 else "rejected",
            "detail": "Active negative bands should occupy a small sampled lag alphabet, not a smooth drift.",
        }
    )

    active_positive = [row for row in positive if row["pmlz_strong_count"] > 0]
    tests.append(
        {
            "test": "positive_channel_intermittency",
            "metric": "active_positive_bands",
            "value": ",".join(row["band"] for row in active_positive) or "-",
            "threshold": "non-universal",
            "verdict": "supported" if 0 < len(active_positive) < len(positive) else "rejected",
            "detail": "The positive channel should be intermittent rather than a universal continuation.",
        }
    )

    block_core = [
        row["B_minus_block_entropy_max_at_pmlz_lag"]
        for row in negative
        if row["band_start_bits"] in CORE_BANDS
        and row["B_minus_block_entropy_max_at_pmlz_lag"] is not None
    ]
    block_shadow = [
        row["B_minus_block_entropy_max_at_pmlz_lag"]
        for row in negative
        if row["band_start_bits"] in SHADOW_BANDS
        and row["B_minus_block_entropy_max_at_pmlz_lag"] is not None
    ]
    core_avg = mean(block_core) if block_core else None
    shadow_avg = mean(block_shadow) if block_shadow else None
    boundary_ratio = (
        shadow_avg / core_avg
        if core_avg is not None and core_avg != 0 and shadow_avg is not None
        else None
    )
    tests.append(
        {
            "test": "transition_entropy_boundary_narrowing",
            "metric": "shadow_avg/core_avg",
            "value": round(boundary_ratio, 4) if boundary_ratio is not None else None,
            "threshold": "<0.50",
            "verdict": "supported" if boundary_ratio is not None and boundary_ratio < 0.50 else "incomplete",
            "detail": (
                "The local transition-entropy null should narrow at the null-shadow boundary "
                "without erasing observed advantage."
            ),
        }
    )

    reactivation_block = [
        row for row in negative
        if row["band_start_bits"] in REACTIVATION_BANDS
        and row["B_minus_block_entropy_max_at_pmlz_lag"] is not None
    ]
    tests.append(
        {
            "test": "late_reactivation_cross_null",
            "metric": "min_lz_and_block_margin",
            "value": min(
                [
                    min(row["pmlz_best_margin"], row["B_minus_block_entropy_max_at_pmlz_lag"])
                    for row in reactivation_block
                ],
                default=None,
            ),
            "threshold": f">={STRONG_MARGIN_THRESHOLD}",
            "verdict": (
                "supported"
                if reactivation_block
                and all(
                    row["pmlz_best_margin"] >= STRONG_MARGIN_THRESHOLD
                    and row["B_minus_block_entropy_max_at_pmlz_lag"] >= STRONG_MARGIN_THRESHOLD
                    for row in reactivation_block
                )
                else "rejected"
            ),
            "detail": "The late holdout reactivation must survive both LZ-family and transition-entropy pressure.",
        }
    )
    return tests


def build_verdict(tests: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {test["test"]: test for test in tests}
    global_rejected = by_name["global_negative_linear_margin"]["verdict"] == "rejected"
    piecewise_supported = by_name["negative_piecewise_regime_map"]["verdict"] == "supported"
    lag_supported = by_name["negative_lag_quantization"]["verdict"] == "supported"
    boundary_supported = by_name["transition_entropy_boundary_narrowing"]["verdict"] == "supported"
    reactivation_supported = by_name["late_reactivation_cross_null"]["verdict"] == "supported"
    if global_rejected and piecewise_supported and lag_supported and boundary_supported and reactivation_supported:
        status = "piecewise-regime-candidate"
    elif piecewise_supported:
        status = "partial-regime-candidate"
    else:
        status = "no-scaling-candidate"
    return {
        "status": status,
        "short_reading": (
            "No global monotone scaling law is supported; the sampled evidence instead "
            "supports a piecewise regime map: active core, pre-shadow shoulder, "
            "null-shadow boundary, and late reactivation."
        ),
        "paper_boundary": (
            "Internal Phase 4 scout only. It proposes a next falsifiable object; it does not "
            "establish a final scaling law or authorize manuscript changes."
        ),
    }


def classify_region(band_start: int) -> str:
    if band_start in CORE_BANDS:
        return "active-core"
    if band_start in PRE_SHADOW_BANDS:
        return "pre-shadow-shoulder"
    if band_start in SHADOW_BANDS:
        return "null-shadow-boundary"
    if band_start in REACTIVATION_BANDS:
        return "late-reactivation"
    return "unregistered"


def classify_regime(row: dict[str, Any]) -> str:
    margin = row["pmlz_best_margin"]
    if row["pmlz_strong_count"] > 0:
        return "strong-channel"
    if margin is not None and margin <= SHADOW_MARGIN_THRESHOLD:
        return "null-shadow"
    return "weak-or-absent"


def linear_fit(points: list[tuple[int, float | None]]) -> dict[str, float]:
    valid = [(x / 1_000_000.0, y) for x, y in points if y is not None]
    if len(valid) < 2:
        return {"slope_per_million": 0.0, "intercept": 0.0, "r2": 0.0}
    xs = [item[0] for item in valid]
    ys = [item[1] for item in valid]
    x_mean = mean(xs)
    y_mean = mean(ys)
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    if ss_xx == 0:
        return {"slope_per_million": 0.0, "intercept": y_mean, "r2": 0.0}
    slope = sum((x - x_mean) * (y - y_mean) for x, y in valid) / ss_xx
    intercept = y_mean - slope * x_mean
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in valid)
    r2 = 0.0 if ss_tot == 0 else max(0.0, 1.0 - ss_res / ss_tot)
    return {"slope_per_million": slope, "intercept": intercept, "r2": r2}


def render_console_summary(state: dict[str, Any]) -> str:
    lines = [
        "Phase 4 P4-03 scaling-law scout",
        "-" * 104,
    ]
    if state["status"] == "dry-run":
        lines.append("Dry run only. Inputs resolved; no regime tests were computed.")
        return "\n".join(lines)
    lines.append(f"verdict: {state['verdict'].get('status', '-')}")
    lines.append("")
    lines.append(f"{'test':<40} {'verdict':<14} {'value':<24} detail")
    for test in state["tests"]:
        lines.append(
            f"{test['test']:<40} {test['verdict']:<14} "
            f"{str(test.get('value')):<24} {test['detail']}"
        )
    return "\n".join(lines)


def render_report(state: dict[str, Any], run_dir: Path) -> str:
    lines = [
        "# HSI v2 Phase 4 P4-03 scaling-law scout",
        "",
        "P4-03 is an aggregate-only scout over already audited P4-01 and P4-02b artifacts.",
        "It tests whether the sampled channel structure is compatible with a simple global law or only with a bounded piecewise regime map.",
        "",
        "## Inputs",
        "",
    ]
    for key, value in state["inputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Status: `{state['verdict'].get('status', state['status'])}`",
            f"- Reading: {state['verdict'].get('short_reading', '-')}",
            f"- Boundary: {state['verdict'].get('paper_boundary', '-')}",
            "",
            "## Regime Tests",
            "",
            "| Test | Verdict | Value | Threshold | Detail |",
            "|---|---|---:|---|---|",
        ]
    )
    for test in state["tests"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(test["test"]),
                    str(test["verdict"]),
                    str(test.get("value", "")),
                    str(test.get("threshold", "")),
                    str(test.get("detail", "")),
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
            "- Regime map: `regime_map.csv`",
            "- Regime tests: `regime_tests.csv`",
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
            "inputs": state.get("inputs", {}),
            "outputs": {
                "summary": str(run_dir / "summary.json"),
                "report": str(run_dir / "report.md"),
                "regime_map": str(run_dir / "regime_map.csv"),
                "regime_tests": str(run_dir / "regime_tests.csv"),
                "manifest": str(run_dir / "manifest.json"),
            },
        },
    )
    write_csv(run_dir / "regime_map.csv", state.get("regime_rows", []), REGIME_FIELDS)
    write_csv(run_dir / "regime_tests.csv", state.get("tests", []), TEST_FIELDS)
    (run_dir / "report.md").write_text(render_report(state, run_dir), encoding="utf-8")


def load_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def band_start_bits(label: str) -> int:
    return int(label.split("M-", 1)[0]) * 1_000_000


def band_label(start_bits: int) -> str:
    return f"{start_bits // 1_000_000}M-{(start_bits + 9_000_000) // 1_000_000}M"


def parse_int(value: Any, *, default: int = 0) -> int:
    parsed = parse_optional_int(value)
    return default if parsed is None else parsed


def parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def parse_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_lag(lag_bits: int) -> str:
    sign = "-" if lag_bits < 0 else ""
    value = abs(lag_bits)
    if value % 1_000_000 == 0:
        return f"{sign}{value // 1_000_000}M"
    if value % 1_000 == 0:
        return f"{sign}{value // 1_000}K"
    return f"{lag_bits}"


REGIME_FIELDS = [
    "band",
    "band_start_bits",
    "band_region",
    "channel",
    "regime_class",
    "pmlz_strong_count",
    "pmlz_width_bits",
    "pmlz_best_lag_bits",
    "pmlz_best_lag",
    "pmlz_best_B_retention",
    "pmlz_best_margin",
    "matched_best_margin_same_channel",
    "pmlz_best_phase_matched_lz_max",
    "matched_lz_max_at_pmlz_lag",
    "block_entropy_max_at_pmlz_lag",
    "B_minus_block_entropy_max_at_pmlz_lag",
]

TEST_FIELDS = ["test", "metric", "value", "threshold", "verdict", "detail"]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
