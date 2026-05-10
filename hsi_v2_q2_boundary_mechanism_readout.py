#!/usr/bin/env python3
"""Q2 boundary mechanism readout.

Post-D-0112 descriptive decomposition of the accepted hard negative. It reads
only Q2 Stage 1 outputs and asks how the winning hard null beats observed B:
fewer effective components, higher top-k concentration, or other shape changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hsi_v2_phase2_transport_defect_strict import phase_print
from v2.common.cli import resolve_dir


DEFAULT_STAGE1_RUN = (
    "results/hsi_v2/q2_internal_routing_entropy/stage1_entropy/"
    "q2-internal-routing-entropy-stage1__20260509T171750"
)
DEFAULT_OUTPUT_DIR = "results/hsi_v2/q2_internal_routing_entropy/boundary_mechanism_readout"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D-0112 Q2 boundary mechanism decomposition.")
    parser.add_argument("--stage1-run", default=DEFAULT_STAGE1_RUN)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    stage1_run = resolve_dir(args.stage1_run, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"q2-boundary-mechanism-readout__src-{stage1_run.name}__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing D-0112 Q2 boundary mechanism readout",
        f"stage1={stage1_run.name} | descriptive only | no new entropy metric",
        quiet=args.quiet,
    )

    cell_entropy = load_csv(stage1_run / "cell_entropy.csv")
    envelope = load_csv(stage1_run / "cell_entropy_envelope.csv")
    cell_rows = build_mechanism_rows(cell_entropy, envelope)
    role_rows = summarize_by(cell_rows, ("cell_role",))
    band_rows = summarize_by(cell_rows, ("cell_role", "band"))
    family_rows = summarize_by(cell_rows, ("cell_role", "hard_null_min_family"))
    verdict = mechanism_verdict(role_rows)

    summary = {
        "run_id": run_dir.name,
        "timestamp_utc": timestamp,
        "status": "completed",
        "stage1_run": str(stage1_run.resolve()),
        "boundary": [
            "This is a descriptive decomposition of D-0112, not a new contrast.",
            "It reads Q2 Stage 1 outputs only.",
            "No routing is rerun and no entropy metric is introduced.",
        ],
        "readout": verdict,
        "counts": {
            "cells": len(cell_rows),
            "catchup_cells": sum(1 for row in cell_rows if row["cell_role"] == "catchup"),
            "strong_context_cells": sum(1 for row in cell_rows if row["cell_role"] == "strong-context"),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "mechanism_cell_delta.csv", cell_rows)
    write_csv(run_dir / "mechanism_role_summary.csv", role_rows)
    write_csv(run_dir / "mechanism_band_summary.csv", band_rows)
    write_csv(run_dir / "mechanism_family_summary.csv", family_rows)
    write_json(run_dir / "manifest.json", build_manifest(summary))
    (run_dir / "report.md").write_text(
        render_report(summary, role_rows, band_rows, family_rows),
        encoding="utf-8",
    )

    print(render_console_summary(summary, role_rows))
    print_saved_paths(run_dir)
    return 0


def build_mechanism_rows(
    cell_entropy: list[dict[str, str]],
    envelope: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_cell = defaultdict(list)
    for row in cell_entropy:
        by_cell[row["cell_id"]].append(row)

    output = []
    for env in envelope:
        cid = env["cell_id"]
        observed = find_observed_b(by_cell[cid])
        winner = find_winning_null(by_cell[cid], env)
        row = {
            "cell_id": cid,
            "cell_role": env["cell_role"],
            "source": env["source"],
            "band": env["band"],
            "lag_bits": int(float(env["lag_bits"])),
            "B_retention": to_float(env["B_retention"]),
            "hard_margin": to_float(env["hard_margin"]),
            "hard_null_min_family": env["hard_null_min_family"],
            "hard_null_min_seed": env["hard_null_min_seed"],
        }
        for field in metric_fields():
            obs = to_float(observed.get(field))
            win = to_float(winner.get(field))
            row[f"observed_{field}"] = obs
            row[f"winner_{field}"] = win
            row[f"delta_winner_minus_observed_{field}"] = win - obs
        row["mechanism_label"] = classify_cell_mechanism(row)
        output.append(row)
    return sorted(output, key=lambda item: (role_sort(item["cell_role"]), band_start(item["band"]), item["lag_bits"]))


def find_observed_b(rows: list[dict[str, str]]) -> dict[str, str]:
    matches = [row for row in rows if row["family"] == "observed" and row["variant"] == "B"]
    if len(matches) != 1:
        raise SystemExit(f"Expected one observed B row, found {len(matches)}")
    return matches[0]


def find_winning_null(rows: list[dict[str, str]], envelope_row: dict[str, str]) -> dict[str, str]:
    family = envelope_row["hard_null_min_family"]
    seed = envelope_row["hard_null_min_seed"]
    candidates = [
        row
        for row in rows
        if row["family"] == family and str(row.get("null_seed", "")) == str(seed)
    ]
    if not candidates:
        raise SystemExit(f"Missing winning null row for {envelope_row['cell_id']} {family} seed={seed}")
    return min(candidates, key=lambda row: to_float(row["component_weight_entropy_norm"]))


def metric_fields() -> list[str]:
    return [
        "component_weight_entropy_norm",
        "effective_weight_components",
        "top1_component_weight_share",
        "top3_component_weight_share",
        "top5_component_weight_share",
        "component_count",
        "active_component_count",
        "edge_count",
    ]


def classify_cell_mechanism(row: dict[str, Any]) -> str:
    eff_delta = row["delta_winner_minus_observed_effective_weight_components"]
    top3_delta = row["delta_winner_minus_observed_top3_component_weight_share"]
    component_delta = row["delta_winner_minus_observed_active_component_count"]
    if eff_delta < -0.25 and top3_delta > 0.05:
        return "fewer-effective-components-and-higher-top3-share"
    if eff_delta < -0.25:
        return "fewer-effective-components"
    if top3_delta > 0.05:
        return "higher-top3-share"
    if component_delta < 0:
        return "fewer-active-components"
    return "subtle-weight-redistribution"


def summarize_by(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key_values, group in sorted(grouped.items()):
        item = {key: value for key, value in zip(keys, key_values)}
        item.update(
            {
                "cells": len(group),
                "entropy_gap_mean": mean(
                    [row["delta_winner_minus_observed_component_weight_entropy_norm"] for row in group]
                ),
                "effective_components_delta_mean": mean(
                    [row["delta_winner_minus_observed_effective_weight_components"] for row in group]
                ),
                "top1_share_delta_mean": mean(
                    [row["delta_winner_minus_observed_top1_component_weight_share"] for row in group]
                ),
                "top3_share_delta_mean": mean(
                    [row["delta_winner_minus_observed_top3_component_weight_share"] for row in group]
                ),
                "top5_share_delta_mean": mean(
                    [row["delta_winner_minus_observed_top5_component_weight_share"] for row in group]
                ),
                "active_components_delta_mean": mean(
                    [row["delta_winner_minus_observed_active_component_count"] for row in group]
                ),
                "dominant_mechanism": most_common([row["mechanism_label"] for row in group]),
                "mechanism_counts": json.dumps(counts([row["mechanism_label"] for row in group]), sort_keys=True),
            }
        )
        output.append(item)
    return output


def mechanism_verdict(role_rows: list[dict[str, Any]]) -> dict[str, Any]:
    catchup = next(row for row in role_rows if row["cell_role"] == "catchup")
    return {
        "category": "boundary-null-concentration-advantage",
        "catchup_dominant_mechanism": catchup["dominant_mechanism"],
        "catchup_entropy_gap_mean": catchup["entropy_gap_mean"],
        "catchup_effective_components_delta_mean": catchup["effective_components_delta_mean"],
        "catchup_top3_share_delta_mean": catchup["top3_share_delta_mean"],
        "interpretation": (
            "The winning hard null is lower-entropy primarily by concentrating "
            "routing mass into fewer effective components with higher top-k shares."
        ),
    }


def render_console_summary(summary: dict[str, Any], role_rows: list[dict[str, Any]]) -> str:
    lines = [
        "D-0112 Q2 boundary mechanism readout",
        "-" * 96,
        f"category: {summary['readout']['category']}",
        f"catchup mechanism: {summary['readout']['catchup_dominant_mechanism']}",
        "",
        f"{'role':<16} {'cells':>5} {'dH':>9} {'dEff':>9} {'dTop3':>9}  mechanism",
    ]
    for row in role_rows:
        lines.append(
            f"{row['cell_role']:<16} {row['cells']:>5} {fmt(row['entropy_gap_mean']):>9} "
            f"{fmt(row['effective_components_delta_mean']):>9} "
            f"{fmt(row['top3_share_delta_mean']):>9}  {row['dominant_mechanism']}"
        )
    return "\n".join(lines)


def render_report(
    summary: dict[str, Any],
    role_rows: list[dict[str, Any]],
    band_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# D-0112 Q2 Boundary Mechanism Readout",
        "",
        "## Boundary",
        "",
        "- This is a descriptive decomposition of the accepted D-0112 hard negative.",
        "- It reads Q2 Stage 1 outputs only.",
        "- No routing is rerun, no entropy metric is introduced, and no manuscript edit is authorized.",
        "",
        "## Verdict",
        "",
        f"- Category: `{summary['readout']['category']}`",
        f"- Catch-up dominant mechanism: `{summary['readout']['catchup_dominant_mechanism']}`",
        f"- Mean entropy gap: `{fmt(summary['readout']['catchup_entropy_gap_mean'])}`",
        f"- Mean effective-component delta (winner null minus observed B): `{fmt(summary['readout']['catchup_effective_components_delta_mean'])}`",
        f"- Mean top3-share delta (winner null minus observed B): `{fmt(summary['readout']['catchup_top3_share_delta_mean'])}`",
        "",
        "## Role Summary",
        "",
        "| Role | Cells | Mean dH | Mean dEff | Mean dTop1 | Mean dTop3 | Mean dTop5 | Dominant mechanism |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in role_rows:
        lines.append(
            f"| `{row['cell_role']}` | {row['cells']} | {fmt(row['entropy_gap_mean'])} | "
            f"{fmt(row['effective_components_delta_mean'])} | {fmt(row['top1_share_delta_mean'])} | "
            f"{fmt(row['top3_share_delta_mean'])} | {fmt(row['top5_share_delta_mean'])} | "
            f"`{row['dominant_mechanism']}` |"
        )
    lines.extend(["", "## Band Summary", "", "| Role | Band | Cells | Mean dH | Mean dEff | Mean dTop3 | Dominant mechanism |", "|---|---|---:|---:|---:|---:|---|"])
    for row in band_rows:
        lines.append(
            f"| `{row['cell_role']}` | `{row['band']}` | {row['cells']} | {fmt(row['entropy_gap_mean'])} | "
            f"{fmt(row['effective_components_delta_mean'])} | {fmt(row['top3_share_delta_mean'])} | "
            f"`{row['dominant_mechanism']}` |"
        )
    lines.extend(["", "## Family Summary", "", "| Role | Winning family | Cells | Mean dH | Mean dEff | Mean dTop3 | Mechanism counts |", "|---|---|---:|---:|---:|---:|---|"])
    for row in family_rows:
        lines.append(
            f"| `{row['cell_role']}` | `{row['hard_null_min_family']}` | {row['cells']} | "
            f"{fmt(row['entropy_gap_mean'])} | {fmt(row['effective_components_delta_mean'])} | "
            f"{fmt(row['top3_share_delta_mean'])} | `{row['mechanism_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Negative dH is the already accepted D-0112 hard negative: the winning hard null has lower local component-weight entropy than observed B.",
            "Negative dEff means the winning hard null uses fewer effective weight components than observed B.",
            "Positive dTop1/dTop3/dTop5 means the winning hard null concentrates more routing mass in its largest components.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": summary["run_id"],
        "timestamp_utc": summary["timestamp_utc"],
        "status": summary["status"],
        "stage1_run": summary["stage1_run"],
        "outputs": [
            "summary.json",
            "report.md",
            "mechanism_cell_delta.csv",
            "mechanism_role_summary.csv",
            "mechanism_band_summary.csv",
            "mechanism_family_summary.csv",
            "manifest.json",
        ],
        "boundary": summary["boundary"],
    }


def print_saved_paths(run_dir: Path) -> None:
    print(f"\nSaved summary to: {run_dir / 'summary.json'}")
    print(f"Saved report to: {run_dir / 'report.md'}")
    print(f"Saved cell delta CSV to: {run_dir / 'mechanism_cell_delta.csv'}")
    print(f"Saved manifest to: {run_dir / 'manifest.json'}")


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


def to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def most_common(values: list[str]) -> str:
    counted = counts(values)
    return max(counted, key=lambda key: (counted[key], key)) if counted else ""


def counts(values: list[str]) -> dict[str, int]:
    output = defaultdict(int)
    for value in values:
        output[value] += 1
    return dict(output)


def band_start(label: str) -> int:
    return int(label.split("-", 1)[0].replace("M", "")) * 1_000_000


def role_sort(role: str) -> int:
    return 0 if role == "catchup" else 1


def fmt(value: Any) -> str:
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
