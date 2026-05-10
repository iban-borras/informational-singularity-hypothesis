#!/usr/bin/env python3
"""Q2 Stage 0 feasibility for internal routing entropy at catch-up.

Implements D-0110. This script selects frozen catch-up/context cells from
surface strict-grid readouts and checks routing-row coverage for the existing
N3-05b/N3-05c/N3-07 path. It does not compute entropy gaps.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hsi_v2_bridge_projective_predictive_stage0 import row_hard_margin
from hsi_v2_phase2_transport_defect_strict import phase_print
from v2.common.cli import resolve_dir


DEFAULT_OUTPUT_DIR = "results/hsi_v2/q2_internal_routing_entropy/stage0_feasibility"
DEFAULT_CONTRACT = "Documentation/HSI_v2_Q2_InternalRoutingEntropy_Preregistration_v0_1.md"
DEFAULT_PHASE4_DIR = "results/hsi_v2/phase4"
DEFAULT_BMINI_RUN = (
    "results/hsi_v2/phase5_cross_arc_horizon/bmini_v0_2/"
    "phase5-bmini-cross-arc-horizon__arcs-3__20260508T153520"
)

NEGATIVE_LAGS = {-29_500_000, -27_000_000, -26_500_000, -24_000_000}
PRIMARY_CATCHUP_BANDS = {
    "795M-804M": "P4",
    "804M-813M": "P4",
    "813M-822M": "P4",
    "822M-831M": "P4",
    "624M-633M": "ARC-B",
    "633M-642M": "ARC-B",
}
CONTEXT_STRONG_BANDS = {
    "768M-777M": "P4 strong",
    "777M-786M": "P4 strong",
    "786M-795M": "P4 strong",
    "597M-606M": "ARC-B strong",
    "606M-615M": "ARC-B strong",
    "615M-624M": "ARC-B strong",
}
REQUIRED_HARD_NULLS = {"phase-matched-lz", "block-entropy"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D-0110 Q2 internal routing entropy Stage 0 feasibility.")
    parser.add_argument("--phase4-dir", default=DEFAULT_PHASE4_DIR)
    parser.add_argument("--bmini-run", default=DEFAULT_BMINI_RUN)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    phase4_dir = resolve_dir(args.phase4_dir, anchor_file=__file__)
    bmini_run = resolve_dir(args.bmini_run, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    contract = resolve_dir(args.contract, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"q2-internal-routing-entropy-stage0__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing D-0110 Q2 internal-routing entropy Stage 0",
        "selecting frozen catch-up/context cells; no entropy gaps computed",
        quiet=args.quiet,
    )

    surface_rows = load_surface_rows(phase4_dir, bmini_run)
    routing_index = index_routing_rows(phase4_dir, bmini_run)
    catchup_rows = select_cells(surface_rows, PRIMARY_CATCHUP_BANDS, mode="catchup")
    context_rows = select_cells(surface_rows, CONTEXT_STRONG_BANDS, mode="strong-context")
    band_status_rows = build_band_status(catchup_rows)
    coverage_rows = build_coverage_rows(catchup_rows + context_rows, routing_index)
    feasibility_rows = build_entropy_path_feasibility(coverage_rows)
    verdict = build_verdict(feasibility_rows, band_status_rows)

    summary = {
        "run_id": run_dir.name,
        "timestamp_utc": timestamp,
        "status": "completed",
        "contract": {"path": str(contract.resolve()), "sha256": sha256_file(contract)},
        "parameters": {
            "phase4_dir": str(phase4_dir),
            "bmini_run": str(bmini_run),
            "negative_lags": sorted(NEGATIVE_LAGS),
        },
        "notes": [
            "Stage 0 selects cells and checks routing-row feasibility only.",
            "No component entropy, entropy_gap, or observed-vs-null entropy contrast is computed.",
            "Bands with zero qualifying catch-up cells are flagged explicitly.",
        ],
        "counts": {
            "catchup_cells": len(catchup_rows),
            "context_cells": len(context_rows),
            "coverage_rows": len(coverage_rows),
        },
        "verdict": verdict,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "catchup_cells.csv", catchup_rows)
    write_csv(run_dir / "context_cells.csv", context_rows)
    write_csv(run_dir / "band_status.csv", band_status_rows)
    write_csv(run_dir / "routing_coverage.csv", coverage_rows)
    write_csv(run_dir / "entropy_path_feasibility.csv", feasibility_rows)
    write_json(run_dir / "manifest.json", build_manifest(summary))
    (run_dir / "report.md").write_text(render_report(summary, band_status_rows, feasibility_rows), encoding="utf-8")

    print(render_console_summary(summary, band_status_rows, feasibility_rows))
    print_saved_paths(run_dir)
    return 0


def load_surface_rows(phase4_dir: Path, bmini_run: Path) -> list[dict[str, Any]]:
    rows = []
    for csv_path in sorted(phase4_dir.glob("p4_04_forward_regime_forecast/**/lag_response.csv")):
        rows.extend(load_csv(csv_path, source_path=csv_path, source_group="P4"))
    for csv_path in sorted(phase4_dir.glob("p4_05_horizon_bracket_forecast/**/lag_response.csv")):
        rows.extend(load_csv(csv_path, source_path=csv_path, source_group="P4"))
    for csv_path in sorted((bmini_run / "strict_p4").glob("**/lag_response.csv")):
        rows.extend(load_csv(csv_path, source_path=csv_path, source_group="B-mini"))
    return [normalize_surface_row(row) for row in rows]


def normalize_surface_row(row: dict[str, Any]) -> dict[str, Any]:
    lag_bits = int(row["lag_bits"])
    return {
        "source_group": row["_source_group"],
        "surface_csv": row["_source_path"],
        "band": row["band"],
        "lag": row.get("lag", ""),
        "lag_bits": lag_bits,
        "channel": "negative" if lag_bits < 0 else "positive" if lag_bits > 0 else "zero",
        "B_retention": to_float(row.get("B_retention")),
        "hard_margin": row_hard_margin(row),
        "phase_matched_lz_max": to_optional_float(row.get("phase_matched_lz_max")),
        "block_entropy_max": to_optional_float(row.get("block_entropy_max")),
    }


def select_cells(rows: list[dict[str, Any]], frozen_bands: dict[str, str], *, mode: str) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row["band"] not in frozen_bands:
            continue
        if row["lag_bits"] not in NEGATIVE_LAGS:
            continue
        if mode == "catchup":
            qualifies = row["B_retention"] >= 0.50 and row["hard_margin"] <= 0.05
            cell_role = "catchup" if qualifies else "not-catchup-cell"
        else:
            qualifies = row["B_retention"] >= 0.90 and row["hard_margin"] >= 0.30
            cell_role = "strong-context" if qualifies else "not-strong-context-cell"
        if not qualifies:
            continue
        selected.append(
            {
                "source": frozen_bands[row["band"]],
                "band": row["band"],
                "lag": row["lag"],
                "lag_bits": row["lag_bits"],
                "channel": "negative",
                "cell_role": cell_role,
                "B_retention": row["B_retention"],
                "hard_margin": row["hard_margin"],
                "phase_matched_lz_max": row["phase_matched_lz_max"],
                "block_entropy_max": row["block_entropy_max"],
                "surface_csv": row["surface_csv"],
            }
        )
    return dedupe_cells(selected)


def dedupe_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = {}
    for row in rows:
        key = (row["band"], int(row["lag_bits"]), row["cell_role"])
        current = deduped.get(key)
        if current is None:
            deduped[key] = row
    return sorted(deduped.values(), key=lambda row: (row["band"], int(row["lag_bits"]), row["cell_role"]))


def build_band_status(catchup_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_band = defaultdict(list)
    for row in catchup_rows:
        rows_by_band[row["band"]].append(row)
    output = []
    for band, source in PRIMARY_CATCHUP_BANDS.items():
        rows = rows_by_band.get(band, [])
        output.append(
            {
                "source": source,
                "band": band,
                "qualifying_catchup_cells": len(rows),
                "status": "catchup-cells-present" if rows else "not-catchup-under-cell-rule",
                "cause_if_zero": "" if rows else "no negative strict-grid cell with B_retention >= 0.50 and hard_margin <= 0.05",
                "positive_cell_fraction_placeholder": "stage1-only",
            }
        )
    return output


def index_routing_rows(phase4_dir: Path, bmini_run: Path) -> dict[tuple[str, int], list[Path]]:
    roots = [
        phase4_dir / "p4_04_forward_regime_forecast",
        phase4_dir / "p4_05_horizon_bracket_forecast",
        bmini_run / "strict_p4",
    ]
    index: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for root in roots:
        for path in root.glob("**/routing_rows.csv"):
            band = find_band_part(path)
            lag = find_lag_part(path)
            if band and lag is not None:
                index[(band, lag)].append(path)
    return index


def build_coverage_rows(cells: list[dict[str, Any]], routing_index: dict[tuple[str, int], list[Path]]) -> list[dict[str, Any]]:
    rows = []
    for cell in cells:
        paths = routing_index.get((cell["band"], int(cell["lag_bits"])), [])
        family_counts = Counter()
        source_counts = Counter()
        total_rows = 0
        for path in paths:
            loaded = load_plain_csv(path)
            total_rows += len(loaded)
            for row in loaded:
                family = row.get("null_model") or "observed"
                family_counts[family] += 1
                source_counts[row.get("source_kind", "")] += 1
        hard_available = all(family_counts.get(family, 0) > 0 for family in REQUIRED_HARD_NULLS)
        rows.append(
            {
                "source": cell["source"],
                "band": cell["band"],
                "lag_bits": cell["lag_bits"],
                "cell_role": cell["cell_role"],
                "routing_rows_files": len(paths),
                "routing_rows_total": total_rows,
                "observed_rows": family_counts.get("observed", 0),
                "phase_matched_lz_rows": family_counts.get("phase-matched-lz", 0),
                "block_entropy_rows": family_counts.get("block-entropy", 0),
                "markov1_rows": family_counts.get("markov1", 0),
                "hard_null_families_available": hard_available,
                "routing_paths": ";".join(str(path) for path in paths),
            }
        )
    return rows


def build_entropy_path_feasibility(coverage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_role = defaultdict(list)
    for row in coverage_rows:
        by_role[row["cell_role"]].append(row)
    output = []
    for role in ("catchup", "strong-context"):
        rows = by_role.get(role, [])
        output.append(
            {
                "cell_role": role,
                "cell_count": len(rows),
                "cells_with_routing_rows": sum(1 for row in rows if int(row["routing_rows_files"]) > 0),
                "cells_with_observed": sum(1 for row in rows if int(row["observed_rows"]) > 0),
                "cells_with_phase_matched_lz": sum(1 for row in rows if int(row["phase_matched_lz_rows"]) > 0),
                "cells_with_block_entropy": sum(1 for row in rows if int(row["block_entropy_rows"]) > 0),
                "cells_with_full_hard_envelope": sum(1 for row in rows if str(row["hard_null_families_available"]) == "True"),
                "entropy_path_status": "n3-route-inputs-present" if rows and all(str(row["hard_null_families_available"]) == "True" and int(row["observed_rows"]) > 0 for row in rows) else "incomplete-inputs",
                "transition_adapter_required": True,
                "adapter_reason": "N3-05b requires a transition nodes/edges run for these frozen cells; lag_response/routing_rows inputs are present but no Q2-specific transition_run exists yet.",
            }
        )
    return output


def build_verdict(feasibility_rows: list[dict[str, Any]], band_status_rows: list[dict[str, Any]]) -> dict[str, Any]:
    catchup = next(row for row in feasibility_rows if row["cell_role"] == "catchup")
    if int(catchup["cell_count"]) == 0:
        return {"status": "not-executable-no-catchup-cells", "reason": "no qualifying catch-up cells"}
    if catchup["entropy_path_status"] != "n3-route-inputs-present":
        return {"status": "not-executable-missing-routing-inputs", "reason": "observed or hard-null routing rows are missing"}
    zero_bands = [row["band"] for row in band_status_rows if row["status"] != "catchup-cells-present"]
    return {
        "status": "stage1-contract-needed-with-transition-adapter",
        "reason": "routing inputs are present; v0.2 must freeze a Q2 transition adapter before entropy contrast",
        "zero_cell_bands": zero_bands,
    }


def render_report(summary: dict[str, Any], band_status: list[dict[str, Any]], feasibility: list[dict[str, Any]]) -> str:
    lines = [
        "# D-0110 Q2 Internal Routing Entropy Stage 0",
        "",
        "## Boundary",
        "",
        "- No entropy gap is computed in Stage 0.",
        "- No observed-vs-null internal entropy comparison is made in Stage 0.",
        "- Hard-null family coverage is reported separately before any min-envelope contrast.",
        "",
        "## Verdict",
        "",
        f"- Status: `{summary['verdict']['status']}`",
        f"- Reason: {summary['verdict']['reason']}",
        "",
        "## Catch-Up Band Status",
        "",
        "| source | band | cells | status | cause if zero |",
        "|---|---|---:|---|---|",
    ]
    for row in band_status:
        lines.append(f"| {row['source']} | `{row['band']}` | {row['qualifying_catchup_cells']} | `{row['status']}` | {row['cause_if_zero']} |")
    lines.extend(["", "## Entropy Path Feasibility", "", "| role | cells | observed | pMLZ | block | status | adapter |", "|---|---:|---:|---:|---:|---|---|"])
    for row in feasibility:
        lines.append(
            f"| `{row['cell_role']}` | {row['cell_count']} | {row['cells_with_observed']} | "
            f"{row['cells_with_phase_matched_lz']} | {row['cells_with_block_entropy']} | "
            f"`{row['entropy_path_status']}` | `{row['transition_adapter_required']}` |"
        )
    lines.extend(
        [
            "",
            "## v0.2 Preview",
            "",
            "- The natural contrast is sign-test style over catch-up bands/cells, not Spearman/Pearson.",
            "- v0.2 must freeze the Q2 transition adapter before computing `component_weight_entropy_norm`.",
            "- Per-family distributions must be reported separately before the hard-null min envelope is interpreted.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_console_summary(summary: dict[str, Any], band_status: list[dict[str, Any]], feasibility: list[dict[str, Any]]) -> str:
    lines = [
        "D-0110 Q2 internal routing entropy Stage 0",
        "-" * 96,
        f"verdict: {summary['verdict']['status']}",
        f"catchup_cells: {summary['counts']['catchup_cells']}",
        f"context_cells: {summary['counts']['context_cells']}",
        "",
        f"{'role':16} {'cells':>6} {'obs':>6} {'pMLZ':>6} {'block':>6} {'status'}",
    ]
    for row in feasibility:
        lines.append(
            f"{row['cell_role']:16} {row['cell_count']:>6} {row['cells_with_observed']:>6} "
            f"{row['cells_with_phase_matched_lz']:>6} {row['cells_with_block_entropy']:>6} "
            f"{row['entropy_path_status']}"
        )
    zero = [row["band"] for row in band_status if row["status"] != "catchup-cells-present"]
    if zero:
        lines.append("")
        lines.append("zero-cell catch-up bands: " + ", ".join(zero))
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
            "catchup_cells.csv",
            "context_cells.csv",
            "band_status.csv",
            "routing_coverage.csv",
            "entropy_path_feasibility.csv",
            "manifest.json",
        ],
        "stage0_boundary": "No entropy_gap or observed-vs-null entropy contrast computed.",
    }


def print_saved_paths(run_dir: Path) -> None:
    print(f"\nSaved summary to: {run_dir / 'summary.json'}")
    print(f"Saved report to: {run_dir / 'report.md'}")
    print(f"Saved catch-up cells CSV to: {run_dir / 'catchup_cells.csv'}")
    print(f"Saved feasibility CSV to: {run_dir / 'entropy_path_feasibility.csv'}")
    print(f"Saved manifest to: {run_dir / 'manifest.json'}")


def find_band_part(path: Path) -> str | None:
    for part in path.parts:
        if "M-" in part and part.endswith("M"):
            return part
    return None


def find_lag_part(path: Path) -> int | None:
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


def load_csv(path: Path, *, source_path: Path, source_group: str) -> list[dict[str, Any]]:
    rows = load_plain_csv(path)
    for row in rows:
        row["_source_path"] = str(source_path)
        row["_source_group"] = source_group
    return rows


def load_plain_csv(path: Path) -> list[dict[str, str]]:
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


def to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def to_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
