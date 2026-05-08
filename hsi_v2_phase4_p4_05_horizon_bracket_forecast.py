#!/usr/bin/env python3
"""HSI v2 Phase 4 P4-05 horizon-bracket forecast.

This launcher extends P4-04 by testing a finite forward horizon bracket. The
contract is deliberately strict: four consecutive untouched bands are tested
under the same hard null envelope, and at least three must preserve the
negative late-reactivation channel.
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

from hsi_v2_phase2_parent_survival_revalidation import flush_output, latest_child_file
from hsi_v2_phase2_transport_defect_strict import phase_print
from utils.progress import format_time
from v2.common.cli import resolve_dir


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_PHASE1_DIR = "results/hsi_v2/phase1_high_scales"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase4/p4_05_horizon_bracket_forecast"
DEFAULT_SEEDS = "607,709,811,907,1009,1103"
DEFAULT_LAGS = "-29500000,-27000000,-26500000,-24000000,24500000,26500000"

FORECAST_BANDS = (
    {
        "band_start_bits": 795_000_000,
        "expected_negative_regime": "late-reactivation",
        "reason": "first band after the P4-04 supported horizon",
    },
    {
        "band_start_bits": 804_000_000,
        "expected_negative_regime": "late-reactivation",
        "reason": "second forward band tests whether P4-04 continues past 795M",
    },
    {
        "band_start_bits": 813_000_000,
        "expected_negative_regime": "late-reactivation",
        "reason": "third forward band makes the bracket longer than the original P4-04 forecast",
    },
    {
        "band_start_bits": 822_000_000,
        "expected_negative_regime": "late-reactivation",
        "reason": "fourth forward band probes the finite-horizon edge without boundary micro-mapping",
    },
)

STRONG_B_THRESHOLD = 0.90
STRONG_MARGIN_THRESHOLD = 0.30
SHADOW_MARGIN_THRESHOLD = 0.05
NEGATIVE_SUPPORT_THRESHOLD = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run P4-05: a predeclared horizon-bracket forecast extending "
            "the P4-04 late-reactivation result."
        )
    )
    parser.add_argument("--phase1-dir", default=DEFAULT_PHASE1_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--workers", type=int, default=5)
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
    run_slug = f"phase4-p4-05-horizon-bracket-forecast__bands-4__lags-6__{timestamp}"
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
            "top_patterns": 128,
            "lags": DEFAULT_LAGS,
            "null_models": "markov1,phase-matched-lz,block-entropy",
            "strong_b_threshold": STRONG_B_THRESHOLD,
            "strong_margin_threshold": STRONG_MARGIN_THRESHOLD,
            "shadow_margin_threshold": SHADOW_MARGIN_THRESHOLD,
            "negative_support_threshold": f"{NEGATIVE_SUPPORT_THRESHOLD}/{len(FORECAST_BANDS)}",
            "forecast_bands": FORECAST_BANDS,
            "dry_run": bool(args.dry_run),
        },
        "command": [],
        "lag_response_run": None,
        "forecast_rows": [],
        "tests": [],
        "verdict": {},
    }

    started = time.perf_counter()
    try:
        phase_print(
            "Preparing P4-05 horizon-bracket forecast",
            (
                "bands=795M,804M,813M,822M | expected_negative=late-reactivation | "
                "lags=6 | nulls=phase-matched-lz,block-entropy"
            ),
            quiet=args.quiet,
        )
        flush_output()

        command = build_lag_response_command(args, phase1_dir, run_dir)
        state["command"] = [str(part) for part in command]

        if args.dry_run:
            state["status"] = "dry-run"
            state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            write_outputs(run_dir, state)
            print(render_console_summary(state))
            print(f"\nSaved summary to: {run_dir / 'summary.json'}")
            print(f"Saved report to: {run_dir / 'report.md'}")
            print(f"Saved manifest to: {run_dir / 'manifest.json'}")
            return 0

        subprocess.run(command, cwd=str(SCRIPT_DIR), check=True)
        lag_summary = latest_child_file(run_dir / "lag_response", "summary.json")
        lag_run = lag_summary.parent
        lag_csv = lag_run / "lag_response.csv"
        if not lag_csv.exists():
            raise FileNotFoundError(f"Missing lag_response.csv: {lag_csv}")

        lag_rows = load_csv_dicts(lag_csv)
        forecast_rows = classify_forecast(lag_rows)
        tests = build_tests(forecast_rows)
        verdict = build_verdict(tests)

        state.update(
            {
                "status": "completed",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "lag_response_run": str(lag_run),
                "forecast_rows": forecast_rows,
                "tests": tests,
                "verdict": verdict,
            }
        )
        write_outputs(run_dir, state)
        print(render_console_summary(state))
        print(f"\nSaved summary to: {run_dir / 'summary.json'}")
        print(f"Saved report to: {run_dir / 'report.md'}")
        print(f"Saved forecast CSV to: {run_dir / 'forecast_regimes.csv'}")
        print(f"Saved tests CSV to: {run_dir / 'forecast_tests.csv'}")
        print(f"Saved manifest to: {run_dir / 'manifest.json'}")
        return 0
    except subprocess.CalledProcessError as exc:
        state["status"] = "failed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        state["error"] = {"returncode": int(exc.returncode), "command": [str(part) for part in exc.cmd]}
        write_outputs(run_dir, state)
        print(f"[x] P4-05 failed after {format_time(time.perf_counter() - started)}")
        print(f"Saved failure summary to: {run_dir / 'summary.json'}")
        return int(exc.returncode) if int(exc.returncode) > 0 else 1
    except Exception as exc:
        state["status"] = "failed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        state["error"] = str(exc)
        write_outputs(run_dir, state)
        print(f"[x] P4-05 failed after {format_time(time.perf_counter() - started)}: {exc}")
        print(f"Saved failure summary to: {run_dir / 'summary.json'}")
        return 1


def build_lag_response_command(args: argparse.Namespace, phase1_dir: Path, run_dir: Path) -> list[str]:
    band_starts = ",".join(str(row["band_start_bits"]) for row in FORECAST_BANDS)
    return [
        sys.executable,
        str(SCRIPT_DIR / "hsi_v2_phase2_child_routing_lag_response.py"),
        "--phase1-dir",
        str(phase1_dir),
        "--output-dir",
        str(run_dir / "lag_response"),
        "--stage",
        "all",
        "--top-patterns",
        "128",
        "--band-starts",
        band_starts,
        f"--lags={DEFAULT_LAGS}",
        "--null-models",
        "markov1,phase-matched-lz,block-entropy",
        "--matched-lz-seeds",
        str(args.seeds),
        "--workers",
        str(args.workers),
        "--quiet-children",
    ] + (["--quiet"] if args.quiet else [])


def classify_forecast(lag_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_band: dict[str, list[dict[str, str]]] = {}
    for row in lag_rows:
        by_band.setdefault(row["band"], []).append(row)

    rows: list[dict[str, Any]] = []
    for band_spec in FORECAST_BANDS:
        band = band_label(int(band_spec["band_start_bits"]))
        band_rows = by_band.get(band, [])
        if not band_rows:
            rows.append(
                {
                    **band_spec,
                    "band": band,
                    "channel": "negative",
                    "status": "missing",
                    "observed_regime": "missing",
                    "prediction_hit": False,
                }
            )
            continue
        for channel in ("negative", "positive"):
            channel_rows = [
                row for row in band_rows
                if lag_channel(parse_int(row["lag_bits"])) == channel
            ]
            rows.append(classify_band_channel(band_spec, band, channel, channel_rows))
    return rows


def classify_band_channel(
    band_spec: dict[str, Any],
    band: str,
    channel: str,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    ranked: list[dict[str, Any]] = []
    for row in rows:
        b_ret = parse_optional_float(row.get("B_retention"))
        pmlz_max = parse_optional_float(row.get("phase_matched_lz_max"))
        bent_max = parse_optional_float(row.get("block_entropy_max"))
        margins = [
            b_ret - value
            for value in (pmlz_max, bent_max)
            if b_ret is not None and value is not None
        ]
        hard_margin = min(margins) if margins else None
        ranked.append(
            {
                "lag_bits": parse_int(row["lag_bits"]),
                "lag": row.get("lag", ""),
                "B_retention": b_ret,
                "phase_matched_lz_max": pmlz_max,
                "block_entropy_max": bent_max,
                "hard_margin": hard_margin,
                "strong": (
                    b_ret is not None
                    and b_ret >= STRONG_B_THRESHOLD
                    and hard_margin is not None
                    and hard_margin >= STRONG_MARGIN_THRESHOLD
                ),
            }
        )

    best = max(ranked, key=lambda item: value_or_low(item["hard_margin"])) if ranked else {}
    strong_count = sum(1 for item in ranked if item.get("strong"))
    best_margin = best.get("hard_margin")
    observed_regime = classify_regime(strong_count, best_margin)
    expected = (
        band_spec["expected_negative_regime"]
        if channel == "negative"
        else "intermittent-or-absent"
    )
    prediction_hit = expected_hit(channel, expected, observed_regime, strong_count)
    return {
        **band_spec,
        "band": band,
        "channel": channel,
        "status": "completed",
        "expected_regime": expected,
        "observed_regime": observed_regime,
        "prediction_hit": prediction_hit,
        "strong_count": strong_count,
        "best_lag_bits": best.get("lag_bits"),
        "best_lag": best.get("lag"),
        "best_B_retention": best.get("B_retention"),
        "best_phase_matched_lz_max": best.get("phase_matched_lz_max"),
        "best_block_entropy_max": best.get("block_entropy_max"),
        "best_hard_margin": best_margin,
    }


def classify_regime(strong_count: int, best_margin: float | None) -> str:
    if strong_count > 0:
        return "strong-channel"
    if best_margin is None:
        return "unresolved"
    if best_margin <= SHADOW_MARGIN_THRESHOLD:
        return "null-shadow-boundary"
    if best_margin < STRONG_MARGIN_THRESHOLD:
        return "pre-shadow-shoulder"
    return "weak-or-absent"


def expected_hit(channel: str, expected: str, observed: str, strong_count: int) -> bool:
    if channel == "positive":
        return observed != "strong-channel" or strong_count <= 2
    if expected == "late-reactivation":
        return observed == "strong-channel"
    return False


def build_tests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    negative = [row for row in rows if row["channel"] == "negative"]
    positive = [row for row in rows if row["channel"] == "positive"]
    negative_hits = sum(1 for row in negative if row["prediction_hit"])
    positive_strong = sum(1 for row in positive if row["observed_regime"] == "strong-channel")
    tests = [
        {
            "test": "negative_horizon_bracket",
            "metric": "hits/total",
            "value": f"{negative_hits}/{len(negative)}",
            "threshold": f">={NEGATIVE_SUPPORT_THRESHOLD}/{len(negative)}",
            "verdict": "supported" if negative_hits >= NEGATIVE_SUPPORT_THRESHOLD else "rejected",
            "detail": (
                "At least three of four forward negative bands must remain strong "
                "against the phase-matched-LZ/block-entropy envelope."
            ),
        },
        {
            "test": "positive_channel_nonuniversality",
            "metric": "nonuniversal_positive",
            "value": f"{positive_strong}/{len(positive)} strong",
            "threshold": "not all strong",
            "verdict": "supported" if 0 <= positive_strong < len(positive) else "rejected",
            "detail": "Positive forecast channels should remain intermittent or absent, not universal.",
        },
    ]
    return tests


def build_verdict(tests: list[dict[str, Any]]) -> dict[str, str]:
    supported = sum(1 for test in tests if test["verdict"] == "supported")
    if supported == len(tests):
        status = "horizon-bracket-forecast-supported"
    elif supported:
        status = "horizon-bracket-forecast-mixed"
    else:
        status = "horizon-bracket-forecast-rejected"
    return {
        "status": status,
        "short_reading": (
            "P4-05 tests whether the P4-04 late negative-channel reactivation "
            "extends across a four-band forward horizon under the same hard null envelope."
        ),
        "paper_boundary": "Internal Phase 4 forecast only; no manuscript edit is authorized.",
    }


def render_console_summary(state: dict[str, Any]) -> str:
    lines = [
        "Phase 4 P4-05 horizon-bracket forecast",
        "-" * 108,
    ]
    if state["status"] == "dry-run":
        lines.append("Dry run only. Inspect manifest for delegated command.")
        return "\n".join(lines)
    lines.append(f"verdict: {state['verdict'].get('status', '-')}")
    lines.append("")
    lines.append(f"{'band':<10} {'ch':<3} {'expected':<22} {'observed':<22} {'hit':>4} {'best_lag':>9} {'margin':>8}")
    for row in state["forecast_rows"]:
        lines.append(
            f"{row['band']:<10} {row['channel'][:3]:<3} "
            f"{row.get('expected_regime', '-'):<22} {row.get('observed_regime', '-'):<22} "
            f"{str(row.get('prediction_hit', '-')):>4} {str(row.get('best_lag', '-')):>9} "
            f"{fmt_float(row.get('best_hard_margin')):>8}"
        )
    return "\n".join(lines)


def render_report(state: dict[str, Any], run_dir: Path) -> str:
    lines = [
        "# HSI v2 Phase 4 P4-05 horizon-bracket forecast",
        "",
        "P4-05 is a predeclared finite-horizon bracket, not open-ended signal chasing.",
        "It tests whether the P4-04 negative-channel reactivation survives four further bands.",
        "",
        "## Parameters",
        "",
        f"- Null models: `{state['parameters']['null_models']}`",
        f"- Seeds: `{state['parameters']['seeds']}`",
        f"- Lags: `{state['parameters']['lags']}`",
        f"- Workers: `{state['parameters']['workers']}`",
        f"- Status: `{state['status']}`",
        "",
        "## Forecast Contract",
        "",
        f"- Negative-channel support threshold: `{state['parameters']['negative_support_threshold']}`",
        "- Negative expected regime: `late-reactivation` for all forecast bands.",
        "- Positive expected regime: `intermittent-or-absent`.",
        "- Strong-cell threshold: B retention >= 0.90 and hard margin >= 0.30.",
        "",
        "## Verdict",
        "",
        f"- Status: `{state.get('verdict', {}).get('status', state['status'])}`",
        f"- Reading: {state.get('verdict', {}).get('short_reading', '-')}",
        "",
        "## Forecast Regimes",
        "",
        "| Band | Channel | Expected | Observed | Hit | Best lag | Hard margin |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in state["forecast_rows"]:
        lines.append(
            f"| {row['band']} | {row['channel']} | {row.get('expected_regime', '-')} | "
            f"{row.get('observed_regime', '-')} | {row.get('prediction_hit', '-')} | "
            f"{row.get('best_lag', '-')} | {fmt_float(row.get('best_hard_margin'))} |"
        )
    lines.extend(
        [
            "",
            "## Tests",
            "",
            "| Test | Verdict | Value | Threshold | Detail |",
            "|---|---|---:|---|---|",
        ]
    )
    for test in state["tests"]:
        lines.append(
            f"| {test['test']} | {test['verdict']} | {test['value']} | "
            f"{test['threshold']} | {test['detail']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Run directory: `{run_dir}`",
            "- Summary: `summary.json`",
            "- Forecast regimes: `forecast_regimes.csv`",
            "- Forecast tests: `forecast_tests.csv`",
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
            "command": state.get("command", []),
            "outputs": {
                "summary": str(run_dir / "summary.json"),
                "report": str(run_dir / "report.md"),
                "forecast_regimes": str(run_dir / "forecast_regimes.csv"),
                "forecast_tests": str(run_dir / "forecast_tests.csv"),
                "manifest": str(run_dir / "manifest.json"),
            },
        },
    )
    write_csv(run_dir / "forecast_regimes.csv", state.get("forecast_rows", []), FORECAST_FIELDS)
    write_csv(run_dir / "forecast_tests.csv", state.get("tests", []), TEST_FIELDS)
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def lag_channel(lag_bits: int) -> str:
    return "negative" if lag_bits < 0 else "positive"


def band_label(start_bits: int) -> str:
    return f"{start_bits // 1_000_000}M-{(start_bits + 9_000_000) // 1_000_000}M"


def parse_int(value: Any) -> int:
    return int(float(value))


def parse_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def value_or_low(value: Any) -> float:
    parsed = parse_optional_float(value)
    return -1e9 if parsed is None else parsed


def fmt_float(value: Any) -> str:
    parsed = parse_optional_float(value)
    return "-" if parsed is None else f"{parsed:.4f}"


FORECAST_FIELDS = [
    "band",
    "band_start_bits",
    "channel",
    "expected_regime",
    "observed_regime",
    "prediction_hit",
    "strong_count",
    "best_lag_bits",
    "best_lag",
    "best_B_retention",
    "best_phase_matched_lz_max",
    "best_block_entropy_max",
    "best_hard_margin",
    "reason",
]

TEST_FIELDS = ["test", "metric", "value", "threshold", "verdict", "detail"]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
