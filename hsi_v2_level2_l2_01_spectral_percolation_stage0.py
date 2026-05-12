#!/usr/bin/env python3
"""L2-01 Stage 0 observability budget for spectral-percolation windows.

Implements D-0115. This script inventories multi-band windows and routing-row
availability only. It does not build quotient graphs or compute spectral
invariants.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hsi_v2_phase2_transport_defect_strict import phase_print
from v2.common.cli import resolve_dir


DEFAULT_OUTPUT_DIR = "results/hsi_v2/level2/l2_01_spectral_percolation/stage0_observability"
DEFAULT_CONTRACT = "Documentation/HSI_v2_Level2_L2_01_SpectralPercolation_Preregistration_v0_2.md"
DEFAULT_PHASE4_ROOT = "results/hsi_v2/phase4"
DEFAULT_BMINI_RUN = (
    "results/hsi_v2/phase5_cross_arc_horizon/bmini_v0_2/"
    "phase5-bmini-cross-arc-horizon__arcs-3__20260508T153520"
)
DEFAULT_Q2_RUN = (
    "results/hsi_v2/q2_internal_routing_entropy/stage1_entropy/"
    "q2-internal-routing-entropy-stage1__20260509T171750"
)
CONTRACT_SHA256 = "A548C01770DAD757151D62D8ED15575D85E811074922AFC57C062E21A8D003CC"
NEGATIVE_LAGS = [-29_500_000, -27_000_000, -26_500_000, -24_000_000]
PREFERRED_WINDOW_SIZE = 3
MIN_WINDOW_SIZE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D-0115 L2-01 spectral-percolation Stage 0.")
    parser.add_argument("--phase4-root", default=DEFAULT_PHASE4_ROOT)
    parser.add_argument("--bmini-run", default=DEFAULT_BMINI_RUN)
    parser.add_argument("--q2-run", default=DEFAULT_Q2_RUN)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    phase4_root = resolve_dir(args.phase4_root, anchor_file=__file__)
    bmini_run = resolve_dir(args.bmini_run, anchor_file=__file__)
    q2_run = resolve_dir(args.q2_run, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    contract = resolve_dir(args.contract, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    verify_contract(contract)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"l2-01-spectral-percolation-stage0__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing D-0115 L2-01 spectral-percolation Stage 0",
        "multi-band observability only | no spectral invariants computed",
        quiet=args.quiet,
    )

    q2_bands = load_q2_bands(q2_run)
    lag_rows = load_lag_rows(phase4_root, bmini_run)
    routing_index = index_routing_rows(phase4_root, bmini_run)
    band_rows = build_band_rows(lag_rows, routing_index, q2_bands)
    window_rows = build_window_rows(band_rows)
    verdict = build_verdict(window_rows)
    stage1_candidates = propose_stage1_candidates(window_rows) if verdict["status"] == "stage1-executable" else []

    summary = {
        "run_id": run_dir.name,
        "timestamp_utc": timestamp,
        "status": "completed",
        "contract": {"path": str(contract.resolve()), "sha256": sha256_file(contract)},
        "parameters": {
            "preferred_window_size": PREFERRED_WINDOW_SIZE,
            "minimum_window_size": MIN_WINDOW_SIZE,
            "negative_lags": NEGATIVE_LAGS,
            "phase4_root": str(phase4_root),
            "bmini_run": str(bmini_run),
            "q2_run": str(q2_run),
        },
        "counts": {
            "bands": len(band_rows),
            "windows": len(window_rows),
            "stage1_candidate_windows": len(stage1_candidates),
        },
        "verdict": verdict,
        "boundary": [
            "No spectral invariants are computed in Stage 0.",
            "No quotient graph is built in Stage 0.",
            "No figure is emitted.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "band_inventory.csv", band_rows)
    write_csv(run_dir / "window_inventory.csv", window_rows)
    write_csv(run_dir / "stage1_candidate_windows.csv", stage1_candidates)
    write_json(run_dir / "manifest.json", build_manifest(summary))
    (run_dir / "report.md").write_text(render_report(summary, window_rows, stage1_candidates), encoding="utf-8")

    print(render_console_summary(summary, window_rows, stage1_candidates))
    print_saved_paths(run_dir)
    return 0


def load_lag_rows(phase4_root: Path, bmini_run: Path) -> list[dict[str, Any]]:
    rows = []
    sources = [
        ("P4", "p4_04_forward_regime_forecast", "P4"),
        ("P4", "p4_05_horizon_bracket_forecast", "P4"),
    ]
    for source_group, dirname, arc_id in sources:
        for path in sorted((phase4_root / dirname).glob("**/lag_response.csv")):
            rows.extend(load_lag_csv(path, source_group=source_group, arc_id=arc_id))
    for path in sorted((bmini_run / "strict_p4").glob("**/lag_response.csv")):
        rows.extend(load_lag_csv(path, source_group="B-mini", arc_id="B-mini"))
    return rows


def load_lag_csv(path: Path, *, source_group: str, arc_id: str) -> list[dict[str, Any]]:
    output = []
    for row in load_csv(path):
        lag_bits = int(float(row["lag_bits"]))
        if lag_bits not in NEGATIVE_LAGS:
            continue
        band = row["band"]
        output.append(
            {
                "source_group": source_group,
                "arc_id": infer_arc_id(source_group, arc_id, band),
                "band": band,
                "band_start_bits": band_start(band),
                "lag_bits": lag_bits,
                "B_retention": to_float(row.get("B_retention")),
                "hard_margin": hard_margin(row),
                "lag_response_csv": str(path),
            }
        )
    return output


def infer_arc_id(source_group: str, default_arc: str, band: str) -> str:
    if source_group != "B-mini":
        return default_arc
    start = band_start(band)
    if 300_000_000 <= start < 345_000_000:
        return "ARC-A"
    if 597_000_000 <= start < 642_000_000:
        return "ARC-B"
    if 1_110_000_000 <= start < 1_155_000_000:
        return "ARC-C"
    return "B-mini"


def build_band_rows(
    lag_rows: list[dict[str, Any]],
    routing_index: dict[tuple[str, int], list[Path]],
    q2_bands: set[str],
) -> list[dict[str, Any]]:
    by_band = defaultdict(list)
    for row in lag_rows:
        by_band[(row["arc_id"], row["band"])].append(row)

    output = []
    for (arc_id, band), rows in sorted(by_band.items(), key=lambda item: (item[0][0], band_start(item[0][1]))):
        best = max(rows, key=lambda row: row["hard_margin"])
        regime = classify_regime(arc_id, band, rows)
        available_lags = []
        routing_files = 0
        for row in rows:
            paths = routing_index.get((band, row["lag_bits"]), [])
            if paths:
                available_lags.append(row["lag_bits"])
                routing_files += len(paths)
        output.append(
            {
                "arc_id": arc_id,
                "band": band,
                "band_start_bits": band_start(band),
                "regime_label": regime,
                "negative_lags_total": len(rows),
                "negative_lags_with_routing": len(set(available_lags)),
                "routing_rows_files": routing_files,
                "best_hard_margin": best["hard_margin"],
                "best_B_retention": max(row["B_retention"] for row in rows),
                "q2_overlap": band in q2_bands,
                "lag_response_csv": best["lag_response_csv"],
            }
        )
    return output


def classify_regime(arc_id: str, band: str, rows: list[dict[str, Any]]) -> str:
    best_margin = max(row["hard_margin"] for row in rows)
    best_retention = max(row["B_retention"] for row in rows)
    if best_retention >= 0.90 and best_margin >= 0.30:
        return "interior"
    if best_retention >= 0.50 and best_margin <= 0.05:
        return "boundary"
    if arc_id in {"ARC-A", "ARC-C"} and best_margin <= 0.05:
        return "inactive"
    return "unlabelled"


def build_window_rows(band_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arc_regime = defaultdict(list)
    for row in band_rows:
        if row["regime_label"] == "unlabelled":
            continue
        by_arc_regime[(row["arc_id"], row["regime_label"])].append(row)

    output = []
    for (arc_id, regime), rows in sorted(by_arc_regime.items()):
        sorted_rows = sorted(rows, key=lambda row: int(row["band_start_bits"]))
        for size in (PREFERRED_WINDOW_SIZE, MIN_WINDOW_SIZE):
            for start in range(0, len(sorted_rows) - size + 1):
                chunk = sorted_rows[start : start + size]
                if not adjacent(chunk):
                    continue
                output.append(window_from_rows(arc_id, regime, chunk, preferred=size == PREFERRED_WINDOW_SIZE))
    return output


def window_from_rows(arc_id: str, regime: str, rows: list[dict[str, Any]], *, preferred: bool) -> dict[str, Any]:
    full_lag_coverage = all(int(row["negative_lags_with_routing"]) >= len(NEGATIVE_LAGS) for row in rows)
    has_cross_band_continuity = len(rows) >= 2 and full_lag_coverage
    q2_overlap_count = sum(1 for row in rows if str(row["q2_overlap"]) == "True" or row["q2_overlap"] is True)
    return {
        "window_id": f"{arc_id}__{regime}__{rows[0]['band']}_to_{rows[-1]['band']}__n{len(rows)}",
        "arc_id": arc_id,
        "regime_label": regime,
        "bands": ";".join(row["band"] for row in rows),
        "band_count": len(rows),
        "preferred_size": preferred,
        "all_bands_have_full_lag_routing": full_lag_coverage,
        "has_cross_band_continuity_potential": has_cross_band_continuity,
        "q2_overlap_bands": q2_overlap_count,
        "q2_overlap_fraction": q2_overlap_count / len(rows),
        "routing_rows_files_total": sum(int(row["routing_rows_files"]) for row in rows),
        "stage1_eligible": full_lag_coverage and has_cross_band_continuity,
        "holdout_class": holdout_class(regime, q2_overlap_count, len(rows)),
        "dependency_note": dependency_note(regime, q2_overlap_count, len(rows)),
    }


def build_verdict(window_rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in window_rows if truthy(row["stage1_eligible"])]
    if not eligible:
        return {"status": "inputs-not-derivable", "reason": "no eligible multi-band windows with routing coverage"}

    by_regime = defaultdict(list)
    for row in eligible:
        by_regime[row["regime_label"]].append(row)
    for regime in ("interior", "boundary", "inactive"):
        if len(by_regime.get(regime, [])) < 2:
            return {
                "status": "insufficient-level2-window-budget",
                "reason": f"fewer than two eligible windows for regime {regime}",
            }
    boundary = by_regime["boundary"]
    non_q2_boundary = [row for row in boundary if row["holdout_class"] != "q2-overlap"]
    if not non_q2_boundary:
        return {
            "status": "insufficient-level2-holdout",
            "reason": "no non-Q2 boundary or boundary-adjacent eligible window available",
        }
    return {
        "status": "stage1-executable",
        "reason": "eligible multi-band windows exist for all regimes with at least one non-Q2 boundary window",
    }


def propose_stage1_candidates(window_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in window_rows if truthy(row["stage1_eligible"])]
    selected = []
    for regime in ("interior", "boundary", "inactive"):
        rows = sorted(
            [row for row in eligible if row["regime_label"] == regime],
            key=lambda row: (
                1 if row["holdout_class"] == "q2-overlap" else 0,
                -int(row["band_count"]),
                row["arc_id"],
                row["bands"],
            ),
        )
        selected.extend(rows[:2])
    return selected


def render_report(
    summary: dict[str, Any],
    window_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> str:
    by_regime = defaultdict(list)
    for row in window_rows:
        by_regime[row["regime_label"]].append(row)
    lines = [
        "# D-0115 L2-01 Spectral-Percolation Stage 0",
        "",
        "## Boundary",
        "",
        "- Stage 0 inventories observability only.",
        "- No spectral invariant, graph distance, comparison, or figure is computed.",
        "- Any downgrade to single-band or per-cell diagnostics requires a separate future Decision Log entry.",
        "",
        "## Verdict",
        "",
        f"- Status: `{summary['verdict']['status']}`",
        f"- Reason: {summary['verdict']['reason']}",
        "",
        "## Window Inventory By Regime",
        "",
        "| Regime | Windows | Eligible | Non-Q2 eligible |",
        "|---|---:|---:|---:|",
    ]
    for regime in ("interior", "boundary", "inactive"):
        rows = by_regime.get(regime, [])
        eligible = [row for row in rows if truthy(row["stage1_eligible"])]
        non_q2 = [row for row in eligible if row["holdout_class"] != "q2-overlap"]
        lines.append(f"| `{regime}` | {len(rows)} | {len(eligible)} | {len(non_q2)} |")
    lines.extend(
        [
            "",
            "## Proposed Stage 1 Candidate Windows",
            "",
            "| Window | Regime | Arc | Bands | Q2 overlap | Holdout |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for row in candidates:
        lines.append(
            f"| `{row['window_id']}` | `{row['regime_label']}` | `{row['arc_id']}` | "
            f"`{row['bands']}` | {row['q2_overlap_bands']} | `{row['holdout_class']}` |"
        )
    return "\n".join(lines) + "\n"


def render_console_summary(
    summary: dict[str, Any],
    window_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> str:
    by_regime = defaultdict(list)
    for row in window_rows:
        by_regime[row["regime_label"]].append(row)
    lines = [
        "D-0115 L2-01 spectral-percolation Stage 0",
        "-" * 96,
        f"verdict: {summary['verdict']['status']}",
        f"reason: {summary['verdict']['reason']}",
        "",
        f"{'regime':<10} {'windows':>8} {'eligible':>9} {'nonQ2':>7}",
    ]
    for regime in ("interior", "boundary", "inactive"):
        rows = by_regime.get(regime, [])
        eligible = [row for row in rows if truthy(row["stage1_eligible"])]
        non_q2 = [row for row in eligible if row["holdout_class"] != "q2-overlap"]
        lines.append(f"{regime:<10} {len(rows):>8} {len(eligible):>9} {len(non_q2):>7}")
    lines.append("")
    lines.append(f"stage1_candidate_windows: {len(candidates)}")
    return "\n".join(lines)


def build_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": summary["run_id"],
        "timestamp_utc": summary["timestamp_utc"],
        "status": summary["status"],
        "contract": summary["contract"],
        "parameters": summary["parameters"],
        "outputs": [
            "summary.json",
            "report.md",
            "band_inventory.csv",
            "window_inventory.csv",
            "stage1_candidate_windows.csv",
            "manifest.json",
        ],
        "stage0_boundary": "No spectral invariants, graph distances, comparisons, or figures computed.",
    }


def print_saved_paths(run_dir: Path) -> None:
    print(f"\nSaved summary to: {run_dir / 'summary.json'}")
    print(f"Saved report to: {run_dir / 'report.md'}")
    print(f"Saved band inventory to: {run_dir / 'band_inventory.csv'}")
    print(f"Saved window inventory to: {run_dir / 'window_inventory.csv'}")
    print(f"Saved manifest to: {run_dir / 'manifest.json'}")


def index_routing_rows(phase4_root: Path, bmini_run: Path) -> dict[tuple[str, int], list[Path]]:
    roots = [
        phase4_root / "p4_04_forward_regime_forecast",
        phase4_root / "p4_05_horizon_bracket_forecast",
        bmini_run / "strict_p4",
    ]
    index = defaultdict(list)
    for root in roots:
        for path in root.glob("**/routing_rows.csv"):
            band = find_band(path)
            lag = find_lag(path)
            if band and lag in NEGATIVE_LAGS:
                index[(band, lag)].append(path)
    return index


def load_q2_bands(q2_run: Path) -> set[str]:
    path = q2_run / "cell_entropy_envelope.csv"
    if not path.exists():
        return set()
    return {row["band"] for row in load_csv(path)}


def holdout_class(regime: str, q2_overlap: int, total: int) -> str:
    if q2_overlap == 0:
        return "non-q2"
    if q2_overlap == total:
        return "q2-overlap"
    return "mixed-q2-overlap"


def dependency_note(regime: str, q2_overlap: int, total: int) -> str:
    if q2_overlap == 0:
        return ""
    return f"{q2_overlap}/{total} bands overlap Q2/D-0112"


def adjacent(rows: list[dict[str, Any]]) -> bool:
    starts = [int(row["band_start_bits"]) for row in rows]
    return all(right - left == 9_000_000 for left, right in zip(starts, starts[1:]))


def verify_contract(contract: Path) -> None:
    actual = sha256_file(contract)
    if actual.upper() != CONTRACT_SHA256:
        raise SystemExit(f"Contract hash mismatch: expected {CONTRACT_SHA256}, got {actual}")


def hard_margin(row: dict[str, str]) -> float:
    margins = []
    for field in ("B_minus_phase_matched_lz_max", "B_minus_block_entropy_max", "B_minus_matched_lz_max"):
        value = row.get(field)
        if value not in (None, ""):
            margins.append(float(value))
    return min(margins) if margins else 0.0


def find_band(path: Path) -> str | None:
    for part in path.parts:
        if "M-" in part and part.endswith("M"):
            return part
    return None


def find_lag(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("neg-") or part.startswith("pos-"):
            sign = -1 if part.startswith("neg-") else 1
            raw = part.split("-", 1)[1]
            if raw.endswith("K"):
                return sign * int(raw[:-1]) * 1_000
            if raw.endswith("M"):
                return sign * int(raw[:-1]) * 1_000_000
            return sign * int(raw)
    return None


def band_start(label: str) -> int:
    return int(label.split("-", 1)[0].replace("M", "")) * 1_000_000


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def truthy(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


if __name__ == "__main__":
    raise SystemExit(main())
