#!/usr/bin/env python3
"""Stage 1 alignment readout for D-0108.

Reads only sealed Stage 0 CSVs and emits the preregistered descriptive
alignment readout. No figures and no manuscript-facing claims are produced.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hsi_v2_phase2_transport_defect_strict import phase_print
from v2.common.cli import resolve_dir


DEFAULT_STAGE0_RUN = (
    "results/hsi_v2/bridge_projective_predictive_alignment/stage0_feasibility/"
    "bridge-projective-predictive-stage0__bands-22__20260509T160812"
)
DEFAULT_OUTPUT_DIR = "results/hsi_v2/bridge_projective_predictive_alignment/stage1_alignment"
DEFAULT_CONTRACT = "Documentation/HSI_v2_Bridge_ProjectiveDensity_PredictiveAlignment_Preregistration_v0_2.md"

EXPECTED_HASHES = {
    "phase1_local_metrics_raw.csv": "7C4E27580D3C526C82AA9CB09BDCC915B71F41BBD7B19C2F2B689A00CB0EF43D",
    "predictive_scores.csv": "20534DB529DB130C9D7C3709EE4AC215945E4389F3DE8AA0C92DDAFF4D2155D7",
    "summary.json": "7A53ABD1B7F5EE9392338B4C760BAD814665F5F0329BF57AC615D8BC7E8D1E83",
}
ARC_ORDER = ("P4", "ARC-A", "ARC-B", "ARC-C")
P_FLOOR = 1.0 / 24.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D-0108 Stage 1 bridge alignment readout.")
    parser.add_argument("--stage0-run", default=DEFAULT_STAGE0_RUN)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    stage0_run = resolve_dir(args.stage0_run, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    contract = resolve_dir(args.contract, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"bridge-projective-predictive-stage1__bands-22__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing D-0108 Stage 1 alignment readout",
        f"stage0={stage0_run.name} | output={run_dir.name}",
        quiet=args.quiet,
    )

    verify_inputs(stage0_run)
    phase1_rows = load_csv(stage0_run / "phase1_local_metrics_raw.csv")
    predictive_rows = load_csv(stage0_run / "predictive_scores.csv")
    joined = join_rows(phase1_rows, predictive_rows)
    if len(joined) != 22:
        raise SystemExit(f"Expected 22 joined frozen bands, found {len(joined)}")

    band_spearman = spearman(
        [row["transport_active_mean_tail"] for row in joined],
        [row["hard_margin_max"] for row in joined],
    )
    band_pearson = pearson(
        [row["transport_active_mean_tail"] for row in joined],
        [row["hard_margin_max"] for row in joined],
    )
    ordinal = build_ordinal_summary(joined)
    arc_rows = build_arc_summary(joined)
    permutation_rows = build_permutation_table(arc_rows)
    observed_arc = next(row for row in permutation_rows if row["permutation"] == "observed")
    category = classify_result(band_spearman, observed_arc["spearman"])

    summary = {
        "run_id": run_dir.name,
        "timestamp_utc": timestamp,
        "status": "completed",
        "contract": {"path": str(contract.resolve()), "sha256": sha256_file(contract)},
        "stage0_run": str(stage0_run.resolve()),
        "sealed_inputs": EXPECTED_HASHES,
        "readout": {
            "band_n": len(joined),
            "arc_n": len(arc_rows),
            "band_spearman": band_spearman,
            "band_pearson": band_pearson,
            "arc_spearman_observed": observed_arc["spearman"],
            "arc_permutation_extreme_count": observed_arc["extreme_count"],
            "arc_permutation_fraction": observed_arc["permutation_fraction"],
            "p_floor": P_FLOOR,
            "category": category,
        },
        "power_boundary": {
            "band_level": "N=22 bands are adjacent/structured and must not be read as independent samples.",
            "arc_level": "N=4 arcs; permutation calibration is descriptive and low-power.",
        },
        "manuscript_boundary": "Interpretive categories do not authorize manuscript changes by themselves.",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "joined_band_table.csv", joined)
    write_csv(run_dir / "arc_summary.csv", arc_rows)
    write_csv(run_dir / "ordinal_summary.csv", ordinal)
    write_csv(run_dir / "permutation_table.csv", permutation_rows)
    write_json(run_dir / "manifest.json", build_manifest(summary))
    (run_dir / "report.md").write_text(render_report(summary, ordinal, arc_rows), encoding="utf-8")

    print(render_console_summary(summary))
    print_saved_paths(run_dir)
    return 0


def verify_inputs(stage0_run: Path) -> None:
    for filename, expected in EXPECTED_HASHES.items():
        path = stage0_run / filename
        if not path.exists():
            raise SystemExit(f"Missing sealed input: {path}")
        actual = sha256_file(path)
        if actual.upper() != expected.upper():
            raise SystemExit(f"Hash mismatch for {path}: expected {expected}, got {actual}")


def join_rows(phase1_rows: list[dict[str, str]], predictive_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    phase_by_start = keyed_by_start(phase1_rows, "phase1")
    pred_by_start = keyed_by_start(predictive_rows, "predictive")
    if set(phase_by_start) != set(pred_by_start):
        missing_phase = sorted(set(pred_by_start) - set(phase_by_start))
        missing_pred = sorted(set(phase_by_start) - set(pred_by_start))
        raise SystemExit(f"Input key mismatch. missing_phase={missing_phase} missing_predictive={missing_pred}")

    joined = []
    for start in sorted(phase_by_start):
        phase = phase_by_start[start]
        pred = pred_by_start[start]
        joined.append(
            {
                "start_bits": start,
                "arc_id": pred["arc_id"],
                "band": pred["band"],
                "transport_active_mean_tail": require_float(phase["transport_active_mean_tail"], "transport_active_mean_tail"),
                "projective_retention_tail": require_float(phase["projective_retention_tail"], "projective_retention_tail"),
                "pattern_density_tail": require_float(phase["pattern_density_tail"], "pattern_density_tail"),
                "hard_margin_max": require_float(pred["hard_margin_max"], "hard_margin_max"),
                "B_retention_max": require_float(pred["B_retention_max"], "B_retention_max"),
                "strict_band_class": pred["strict_band_class"],
            }
        )
    return joined


def build_ordinal_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for label in ("strong", "shadow", "inactive"):
        values = [row["transport_active_mean_tail"] for row in rows if row["strict_band_class"] == label]
        output.append(
            {
                "strict_band_class": label,
                "n": len(values),
                "transport_active_mean_tail_mean": mean(values),
                "transport_active_mean_tail_median": median(values),
                "transport_active_mean_tail_min": min(values) if values else None,
                "transport_active_mean_tail_max": max(values) if values else None,
            }
        )
    return output


def build_arc_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for arc in ARC_ORDER:
        arc_rows = [row for row in rows if row["arc_id"] == arc]
        if not arc_rows:
            raise SystemExit(f"Missing arc rows for {arc}")
        output.append(
            {
                "arc_id": arc,
                "band_count": len(arc_rows),
                "transport_active_mean_tail_mean": mean([row["transport_active_mean_tail"] for row in arc_rows]),
                "transport_active_mean_tail_median": median([row["transport_active_mean_tail"] for row in arc_rows]),
                "hard_margin_max": max(row["hard_margin_max"] for row in arc_rows),
                "strong_count": sum(1 for row in arc_rows if row["strict_band_class"] == "strong"),
                "shadow_count": sum(1 for row in arc_rows if row["strict_band_class"] == "shadow"),
                "inactive_count": sum(1 for row in arc_rows if row["strict_band_class"] == "inactive"),
            }
        )
    return output


def build_permutation_table(arc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phase_values = [row["transport_active_mean_tail_mean"] for row in arc_rows]
    predictive_values = [row["hard_margin_max"] for row in arc_rows]
    observed = spearman(phase_values, predictive_values)
    permutations = []
    for perm_index, perm in enumerate(itertools.permutations(range(len(ARC_ORDER))), start=1):
        perm_values = [predictive_values[index] for index in perm]
        stat = spearman(phase_values, perm_values)
        permutations.append(
            {
                "permutation": ",".join(ARC_ORDER[index] for index in perm),
                "spearman": stat,
                "abs_spearman": abs(stat),
                "is_extreme_vs_observed": abs(stat) >= abs(observed),
            }
        )
    extreme_count = sum(1 for row in permutations if row["is_extreme_vs_observed"])
    fraction = extreme_count / len(permutations)
    rows = [
        {
            "permutation": "observed",
            "spearman": observed,
            "abs_spearman": abs(observed),
            "is_extreme_vs_observed": True,
            "extreme_count": extreme_count,
            "permutation_count": len(permutations),
            "permutation_fraction": fraction,
            "p_floor": P_FLOOR,
        }
    ]
    for row in permutations:
        rows.append(
            {
                **row,
                "extreme_count": extreme_count,
                "permutation_count": len(permutations),
                "permutation_fraction": fraction,
                "p_floor": P_FLOOR,
            }
        )
    return rows


def classify_result(band_spearman: float, arc_spearman: float) -> str:
    if band_spearman >= 0.50 and arc_spearman > 0:
        return "aligned-bridge-support"
    if band_spearman <= 0.10:
        return "parallel-emergence-outcome"
    return "weak-or-mixed-alignment"


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rankdata(xs), rankdata(ys))


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("Pearson requires paired vectors with at least two values.")
    mean_x = mean(xs)
    mean_y = mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


def rankdata(values: list[float]) -> list[float]:
    order = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and order[end][1] == order[index][1]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        for original_index, _ in order[index:end]:
            ranks[original_index] = avg_rank
        index = end
    return ranks


def render_report(summary: dict[str, Any], ordinal: list[dict[str, Any]], arc_rows: list[dict[str, Any]]) -> str:
    readout = summary["readout"]
    lines = [
        "# D-0108 Stage 1 Bridge Alignment Readout",
        "",
        "## Contract Boundary",
        "",
        "- No figure is authorized by v0.2.",
        "- Interpretive categories do not authorize manuscript changes by themselves.",
        "- Band-level `N=22` is structured and adjacent; it must not be read as 22 independent samples.",
        "- Arc-level `N=4` is low-power; permutation calibration is descriptive.",
        f"- Permutation `p_floor = 1/24 = {P_FLOOR:.4f}`.",
        "",
        "## Primary Readout",
        "",
        f"- Band-level Spearman: `{readout['band_spearman']:.6f}`",
        f"- Band-level Pearson: `{readout['band_pearson']:.6f}`",
        f"- Arc-level observed Spearman: `{readout['arc_spearman_observed']:.6f}`",
        f"- Arc-level exact permutation fraction: `{readout['arc_permutation_fraction']:.6f}`",
        f"- Arc-level extreme permutations: `{readout['arc_permutation_extreme_count']}/24`",
        f"- Category: `{readout['category']}`",
        "",
        "## Ordinal Summary",
        "",
        "| class | n | mean Phase 1 metric | median Phase 1 metric |",
        "|---|---:|---:|---:|",
    ]
    for row in ordinal:
        lines.append(
            f"| `{row['strict_band_class']}` | {row['n']} | {_fmt(row['transport_active_mean_tail_mean'])} | "
            f"{_fmt(row['transport_active_mean_tail_median'])} |"
        )
    lines.extend(["", "## Arc Summary", "", "| arc | bands | mean Phase 1 metric | max hard margin | strong | shadow | inactive |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in arc_rows:
        lines.append(
            f"| `{row['arc_id']}` | {row['band_count']} | {_fmt(row['transport_active_mean_tail_mean'])} | "
            f"{_fmt(row['hard_margin_max'])} | {row['strong_count']} | {row['shadow_count']} | {row['inactive_count']} |"
        )
    return "\n".join(lines) + "\n"


def render_console_summary(summary: dict[str, Any]) -> str:
    r = summary["readout"]
    return "\n".join(
        [
            "D-0108 Stage 1 bridge alignment",
            "-" * 88,
            f"band_spearman: {r['band_spearman']:.6f}",
            f"band_pearson: {r['band_pearson']:.6f}",
            f"arc_spearman: {r['arc_spearman_observed']:.6f}",
            f"arc_perm: {r['arc_permutation_extreme_count']}/24 = {r['arc_permutation_fraction']:.6f} (floor {P_FLOOR:.4f})",
            f"category: {r['category']}",
        ]
    )


def build_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": summary["run_id"],
        "timestamp_utc": summary["timestamp_utc"],
        "status": summary["status"],
        "contract": summary["contract"],
        "stage0_run": summary["stage0_run"],
        "sealed_inputs": summary["sealed_inputs"],
        "outputs": [
            "summary.json",
            "report.md",
            "joined_band_table.csv",
            "arc_summary.csv",
            "ordinal_summary.csv",
            "permutation_table.csv",
            "manifest.json",
        ],
        "manuscript_boundary": summary["manuscript_boundary"],
    }


def print_saved_paths(run_dir: Path) -> None:
    print(f"\nSaved summary to: {run_dir / 'summary.json'}")
    print(f"Saved report to: {run_dir / 'report.md'}")
    print(f"Saved joined band CSV to: {run_dir / 'joined_band_table.csv'}")
    print(f"Saved arc summary CSV to: {run_dir / 'arc_summary.csv'}")
    print(f"Saved permutation CSV to: {run_dir / 'permutation_table.csv'}")
    print(f"Saved manifest to: {run_dir / 'manifest.json'}")


def keyed_by_start(rows: list[dict[str, str]], label: str) -> dict[int, dict[str, str]]:
    output = {}
    for row in rows:
        start = int(row["start_bits"])
        if start in output:
            raise SystemExit(f"Duplicate {label} start_bits key: {start}")
        output[start] = row
    return output


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
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_float(value: str, label: str) -> float:
    if value in ("", None):
        raise SystemExit(f"Missing required numeric value: {label}")
    return float(value)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
