#!/usr/bin/env python3
"""Q2 Stage 1 internal routing entropy readout.

Implements the accepted D-0111/v0.2 contract: build per-cell adapters,
reuse the existing N3-05b/N3-05c component quotient path, and report a
band-level sign readout. The script does not modify N3 definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hsi_v2_phase2_transport_defect_strict import phase_print
from hsi_v2_phase3_component_quotient_child_routing_spectrum import (
    SOURCE_FIELDS as N3_05C_SOURCE_FIELDS,
)
from hsi_v2_phase3_quotient_child_routing_spectral_graph import (
    SUMMARY_FIELDS as N3_05B_SUMMARY_FIELDS,
    write_csv,
    write_csv_dynamic,
    write_json,
)
from v2.common.cli import resolve_dir
from v2.phase3.component_quotient_spectrum import build_component_quotient_readout
from v2.phase3.quotient_child_routing_spectrum import build_quotient_child_routing_spectrum


DEFAULT_STAGE0_RUN = (
    "results/hsi_v2/q2_internal_routing_entropy/stage0_feasibility/"
    "q2-internal-routing-entropy-stage0__20260509T165423"
)
DEFAULT_OUTPUT_DIR = "results/hsi_v2/q2_internal_routing_entropy/stage1_entropy"
DEFAULT_CONTRACT = "Documentation/HSI_v2_Q2_InternalRoutingEntropy_Preregistration_v0_2.md"

EXPECTED_HASHES = {
    "catchup_cells.csv": "25174BAB9BE0469061D827C1F36FFBB7D83AEF7CB4B20E816D078458AFE994E4",
    "context_cells.csv": "85177E96E7674D7B4D3BE637EE85427FEBF90B9838D02BC68F6DCEB5A08D57FF",
    "routing_coverage.csv": "080AD771D826533763D37D42DE1E1A4DF1337BB583B7C824D890311AEA5D9605",
    "summary.json": "05A0A05A2E768AB61591938A0C7735E4FD8754253B4355F6E0C0B112ED71A8B0",
}
CONTRACT_SHA256 = "8DAD57F54BB70687AB2B4FBC85398BF543CC5A639D55E487EC9879B96521F011"
HARD_NULLS = ("phase-matched-lz", "block-entropy")
TOP_PATTERNS = 128
PARENT_SCALE = 40


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D-0111 Q2 internal-routing entropy Stage 1.")
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
    run_dir = output_root / f"q2-internal-routing-entropy-stage1__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing D-0111 Q2 internal-routing entropy Stage 1",
        f"stage0={stage0_run.name} | per-cell adapters | no new entropy metric",
        quiet=args.quiet,
    )

    verify_inputs(stage0_run, contract)
    catchup_cells = load_csv(stage0_run / "catchup_cells.csv")
    context_cells = load_csv(stage0_run / "context_cells.csv")
    coverage_rows = load_csv(stage0_run / "routing_coverage.csv")
    coverage_index = index_coverage(coverage_rows)

    adapter_root = run_dir / "adapter"
    n3_root = run_dir / "n3_artifacts"
    adapter_root.mkdir(parents=True, exist_ok=True)
    n3_root.mkdir(parents=True, exist_ok=True)

    selected_cells = tag_cells(catchup_cells, "catchup") + tag_cells(context_cells, "strong-context")
    cell_entropy_rows: list[dict[str, Any]] = []
    adapter_rows: list[dict[str, Any]] = []

    for index, cell in enumerate(selected_cells, start=1):
        if not args.quiet:
            print(f"[cell {index:02d}/{len(selected_cells):02d}] {cell_id(cell)}")
        cell_result = process_cell(cell, coverage_index, adapter_root, n3_root)
        cell_entropy_rows.extend(cell_result["entropy_rows"])
        adapter_rows.append(cell_result["adapter_row"])

    envelope_rows = build_cell_envelope_rows(cell_entropy_rows)
    band_rows = build_band_summary(envelope_rows)
    family_rows = build_family_summary(cell_entropy_rows)
    category = classify_outcome(band_rows)
    summary = {
        "run_id": run_dir.name,
        "timestamp_utc": timestamp,
        "status": "completed",
        "contract": {"path": str(contract.resolve()), "sha256": sha256_file(contract)},
        "stage0_run": str(stage0_run.resolve()),
        "sealed_inputs": EXPECTED_HASHES,
        "counts": {
            "catchup_cells": len(catchup_cells),
            "strong_context_cells": len(context_cells),
            "cell_entropy_rows": len(cell_entropy_rows),
            "catchup_bands": len([row for row in band_rows if row["cell_role"] == "catchup"]),
        },
        "readout": {
            "category": category,
            "catchup_positive_bands": sum(
                1
                for row in band_rows
                if row["cell_role"] == "catchup" and row["band_sign"] == "B-lower-entropy"
            ),
            "catchup_negative_bands": sum(
                1
                for row in band_rows
                if row["cell_role"] == "catchup" and row["band_sign"] == "null-lower-entropy"
            ),
            "catchup_zero_bands": sum(
                1
                for row in band_rows
                if row["cell_role"] == "catchup" and row["band_sign"] == "no-separation"
            ),
            "sign_test_p_floor_all_positive": 0.03125,
            "observed_B_entropy_std_catchup": stddev(
                [
                    row["component_weight_entropy_norm"]
                    for row in cell_entropy_rows
                    if row["cell_role"] == "catchup" and row["family"] == "observed"
                ]
            ),
        },
        "notes": [
            "Per-cell adapters change N3 granularity from full transition-domain entropy to local cell entropy, without changing N3 formulas.",
            "The empty transition edges.csv is a compatibility artifact and contributes no structure.",
            "Strong-context cells are descriptive calibration only and cannot affect the catch-up verdict.",
            "No figure is authorized or emitted by this run.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    write_json(run_dir / "summary.json", summary)
    write_csv_dynamic(run_dir / "cell_entropy.csv", cell_entropy_rows)
    write_csv_dynamic(run_dir / "cell_entropy_envelope.csv", envelope_rows)
    write_csv_dynamic(run_dir / "band_entropy_summary.csv", band_rows)
    write_csv_dynamic(run_dir / "family_entropy_summary.csv", family_rows)
    write_csv_dynamic(run_dir / "adapter_index.csv", adapter_rows)
    write_json(run_dir / "adapter_manifest.json", build_adapter_manifest(adapter_rows))
    write_json(run_dir / "manifest.json", build_manifest(summary))
    (run_dir / "report.md").write_text(
        render_report(summary, band_rows, family_rows),
        encoding="utf-8",
    )

    print(render_console_summary(summary, band_rows))
    print_saved_paths(run_dir)
    return 0


def process_cell(
    cell: dict[str, Any],
    coverage_index: dict[tuple[str, int, str], dict[str, str]],
    adapter_root: Path,
    n3_root: Path,
) -> dict[str, Any]:
    cid = cell_id(cell)
    coverage = coverage_index.get((cell["band"], int(cell["lag_bits"]), cell["cell_role"]))
    if coverage is None:
        raise SystemExit(f"Missing routing coverage for selected cell: {cid}")
    source_rows = load_required_routing_rows(coverage, cid)

    cell_root = adapter_root / "cells" / cid
    transition_run = cell_root / "transition_run"
    child_run = cell_root / "child_routing_run"
    transition_run.mkdir(parents=True, exist_ok=True)
    child_run.mkdir(parents=True, exist_ok=True)

    write_transition_adapter(cell, transition_run)
    write_child_routing_adapter(cell, child_run, source_rows)

    weighted_run = n3_root / cid / "n3_05b_weighted_quotient"
    component_run = n3_root / cid / "n3_05c_component_quotient"
    weighted_run.mkdir(parents=True, exist_ok=True)
    component_run.mkdir(parents=True, exist_ok=True)

    quotient = build_quotient_child_routing_spectrum(
        transition_run=transition_run,
        child_routing_run=child_run,
        top_patterns=TOP_PATTERNS,
        parent_scale=PARENT_SCALE,
        edge_weight_mode="pooled-retained-mass",
    )
    if quotient["missing_cells"]:
        raise SystemExit(f"N3-05b missing cell after adapter build: {cid}")
    write_json(weighted_run / "summary.json", quotient)
    write_csv(weighted_run / "graph_summary.csv", quotient["graph_summary"], N3_05B_SUMMARY_FIELDS)
    write_csv_dynamic(weighted_run / "graph_nodes.csv", quotient["graph_nodes"])
    write_csv_dynamic(weighted_run / "graph_edges.csv", quotient["graph_edges"])
    write_json(weighted_run / "manifest.json", {"cell_id": cid, "adapter_cell": str(cell_root)})

    component = build_component_quotient_readout(weighted_run)
    write_json(component_run / "summary.json", component)
    write_csv(component_run / "source_summary.csv", component["source_summary"], N3_05C_SOURCE_FIELDS)
    write_csv_dynamic(component_run / "component_summary.csv", component["component_summary"])
    write_json(component_run / "manifest.json", {"cell_id": cid, "weighted_quotient_run": str(weighted_run)})

    entropy_rows = extract_entropy_rows(cell, cid, component["source_summary"])
    ensure_cell_has_required_entropy(entropy_rows, cid)
    return {
        "entropy_rows": entropy_rows,
        "adapter_row": {
            "cell_id": cid,
            "cell_role": cell["cell_role"],
            "source": cell["source"],
            "band": cell["band"],
            "lag_bits": cell["lag_bits"],
            "transition_run": str(transition_run),
            "child_routing_run": str(child_run),
            "n3_05b_run": str(weighted_run),
            "n3_05c_run": str(component_run),
            "routing_paths": coverage.get("routing_paths", ""),
        },
    }


def verify_inputs(stage0_run: Path, contract: Path) -> None:
    for filename, expected in EXPECTED_HASHES.items():
        path = stage0_run / filename
        if not path.exists():
            raise SystemExit(f"Missing sealed Stage 0 input: {path}")
        actual = sha256_file(path)
        if actual.upper() != expected.upper():
            raise SystemExit(f"Hash mismatch for {path}: expected {expected}, got {actual}")
    actual_contract = sha256_file(contract)
    if actual_contract.upper() != CONTRACT_SHA256:
        raise SystemExit(
            f"Contract hash mismatch for {contract}: expected {CONTRACT_SHA256}, got {actual_contract}"
        )


def tag_cells(rows: list[dict[str, str]], role: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = dict(row)
        item["cell_role"] = role
        item["lag_bits"] = int(float(item["lag_bits"]))
        item["B_retention"] = float(item["B_retention"])
        item["hard_margin"] = float(item["hard_margin"])
        output.append(item)
    return output


def index_coverage(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], dict[str, str]]:
    index = {}
    for row in rows:
        index[(row["band"], int(float(row["lag_bits"])), row["cell_role"])] = row
    return index


def load_required_routing_rows(coverage: dict[str, str], cid: str) -> list[dict[str, str]]:
    paths = [Path(item) for item in coverage.get("routing_paths", "").split(";") if item.strip()]
    rows = []
    for path in paths:
        rows.extend(load_csv(path))
    allowed = []
    family_counts = Counter()
    for row in rows:
        family = row.get("null_model") or "observed"
        if family == "observed" and row.get("variant") == "B":
            allowed.append(row)
            family_counts["observed"] += 1
        elif family in HARD_NULLS:
            allowed.append(row)
            family_counts[family] += 1
    missing = [family for family in ("observed", *HARD_NULLS) if family_counts[family] <= 0]
    if missing:
        raise SystemExit(f"Missing required routing families for {cid}: {', '.join(missing)}")
    return allowed


def write_transition_adapter(cell: dict[str, Any], transition_run: Path) -> None:
    node = {
        "top_patterns": TOP_PATTERNS,
        "band": cell["band"],
        "band_start_bits": band_start(cell["band"]),
        "band_stop_bits": band_stop(cell["band"]),
        "channel": "negative",
        "node_class": "q2-catchup" if cell["cell_role"] == "catchup" else "q2-strong-context",
        "strong_count": 1,
        "best_margin_lag_bits": cell["lag_bits"],
        "best_margin": cell["hard_margin"],
        "cell_role": cell["cell_role"],
        "source": cell["source"],
    }
    write_csv_dynamic(transition_run / "nodes.csv", [node])
    write_header_only_csv(transition_run / "edges.csv", ["id", "source", "target", "edge_type"])
    write_json(transition_run / "manifest.json", {"adapter": "q2-per-cell-transition", "cell": node})


def write_child_routing_adapter(
    cell: dict[str, Any],
    child_run: Path,
    rows: list[dict[str, str]],
) -> None:
    lag_dir = child_run / cell["band"] / str(cell["lag_bits"])
    lag_dir.mkdir(parents=True, exist_ok=True)
    write_csv_dynamic(lag_dir / "routing_rows.csv", rows)
    summary = {
        "adapter": "q2-per-cell-child-routing",
        "band": cell["band"],
        "lag_bits": cell["lag_bits"],
        "cell_role": cell["cell_role"],
        "row_count": len(rows),
        "families": sorted({row.get("null_model") or "observed" for row in rows}),
    }
    write_json(child_run / "summary.json", summary)
    write_json(child_run / "manifest.json", summary)


def extract_entropy_rows(
    cell: dict[str, Any],
    cid: str,
    source_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for row in source_summary:
        family = family_from_component_row(row)
        rows.append(
            {
                "cell_id": cid,
                "cell_role": cell["cell_role"],
                "source": cell["source"],
                "band": cell["band"],
                "lag_bits": cell["lag_bits"],
                "B_retention": cell["B_retention"],
                "hard_margin": cell["hard_margin"],
                "variant": row.get("variant", ""),
                "source_label": row.get("source_label", ""),
                "family": family,
                "null_seed": row.get("source_label", "").split("seed ", 1)[-1].rstrip("]") if "seed " in row.get("source_label", "") else "",
                "component_weight_entropy_norm": to_optional_float(row.get("component_weight_entropy_norm")),
                "effective_weight_components": to_optional_float(row.get("effective_weight_components")),
                "top1_component_weight_share": to_optional_float(row.get("top1_component_weight_share")),
                "top3_component_weight_share": to_optional_float(row.get("top3_component_weight_share")),
                "top5_component_weight_share": to_optional_float(row.get("top5_component_weight_share")),
                "component_count": to_optional_int(row.get("component_count")),
                "active_component_count": to_optional_int(row.get("active_component_count")),
                "edge_count": to_optional_int(row.get("edge_count")),
                "verdict": row.get("verdict", ""),
            }
        )
    return rows


def family_from_component_row(row: dict[str, Any]) -> str:
    variant = str(row.get("variant", ""))
    label = str(row.get("source_label", ""))
    text = f"{variant} {label}"
    if "phase-matched-lz" in text:
        return "phase-matched-lz"
    if "block-entropy" in text:
        return "block-entropy"
    return "observed"


def ensure_cell_has_required_entropy(rows: list[dict[str, Any]], cid: str) -> None:
    observed = [row for row in rows if row["family"] == "observed" and row["variant"] == "B"]
    if len(observed) != 1 or observed[0]["component_weight_entropy_norm"] is None:
        raise SystemExit(f"Missing observed B entropy for {cid}")
    for family in HARD_NULLS:
        family_rows = [row for row in rows if row["family"] == family]
        if not family_rows:
            raise SystemExit(f"Missing {family} entropy rows for {cid}")
        if any(row["component_weight_entropy_norm"] is None for row in family_rows):
            raise SystemExit(f"Null entropy contains empty values for {cid}: {family}")


def build_cell_envelope_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["cell_id"]].append(row)
    output = []
    for cid, cell_rows in grouped.items():
        observed = next(row for row in cell_rows if row["family"] == "observed" and row["variant"] == "B")
        family_min = {}
        for family in HARD_NULLS:
            candidates = [row for row in cell_rows if row["family"] == family]
            family_min[family] = min(candidates, key=lambda row: row["component_weight_entropy_norm"])
        best_family = min(HARD_NULLS, key=lambda family: family_min[family]["component_weight_entropy_norm"])
        best = family_min[best_family]
        gap = best["component_weight_entropy_norm"] - observed["component_weight_entropy_norm"]
        output.append(
            {
                "cell_id": cid,
                "cell_role": observed["cell_role"],
                "source": observed["source"],
                "band": observed["band"],
                "lag_bits": observed["lag_bits"],
                "B_retention": observed["B_retention"],
                "hard_margin": observed["hard_margin"],
                "observed_B_entropy": observed["component_weight_entropy_norm"],
                "phase_matched_lz_entropy_min": family_min["phase-matched-lz"]["component_weight_entropy_norm"],
                "phase_matched_lz_min_seed": family_min["phase-matched-lz"]["null_seed"],
                "block_entropy_entropy_min": family_min["block-entropy"]["component_weight_entropy_norm"],
                "block_entropy_min_seed": family_min["block-entropy"]["null_seed"],
                "hard_null_entropy_min": best["component_weight_entropy_norm"],
                "hard_null_min_family": best_family,
                "hard_null_min_seed": best["null_seed"],
                "entropy_gap": gap,
                "gap_sign": sign_label(gap),
            }
        )
    return sorted(output, key=lambda row: (role_sort(row["cell_role"]), band_start(row["band"]), int(row["lag_bits"])))


def build_band_summary(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_band = defaultdict(list)
    for row in cell_rows:
        by_band[(row["cell_role"], row["source"], row["band"])].append(row)
    output = []
    for (role, source, band), rows in sorted(by_band.items(), key=lambda item: (role_sort(item[0][0]), band_start(item[0][2]))):
        gaps = [row["entropy_gap"] for row in rows]
        med = median(gaps)
        output.append(
            {
                "cell_role": role,
                "source": source,
                "band": band,
                "selected_cells": len(rows),
                "median_entropy_gap": med,
                "mean_entropy_gap": mean(gaps),
                "positive_cell_fraction": sum(1 for gap in gaps if gap > 0.0) / len(gaps),
                "negative_cell_fraction": sum(1 for gap in gaps if gap < 0.0) / len(gaps),
                "zero_cell_fraction": sum(1 for gap in gaps if gap == 0.0) / len(gaps),
                "band_sign": band_sign(med),
                "support_note": "single-cell-band" if len(rows) == 1 else "",
                "min_family_counts": json.dumps(dict(Counter(row["hard_null_min_family"] for row in rows)), sort_keys=True),
            }
        )
    return output


def build_family_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family = defaultdict(list)
    for row in rows:
        by_family[(row["cell_role"], row["family"])].append(row)
    output = []
    for (role, family), items in sorted(by_family.items()):
        values = [row["component_weight_entropy_norm"] for row in items if row["component_weight_entropy_norm"] is not None]
        output.append(
            {
                "cell_role": role,
                "family": family,
                "rows": len(values),
                "mean_entropy": mean(values),
                "median_entropy": median(values),
                "min_entropy": min(values) if values else None,
                "max_entropy": max(values) if values else None,
                "std_entropy": stddev(values),
            }
        )
    return output


def classify_outcome(band_rows: list[dict[str, Any]]) -> str:
    catchup = [row for row in band_rows if row["cell_role"] == "catchup"]
    if len(catchup) != 6:
        return "incomplete-inputs-stop"
    positive = sum(1 for row in catchup if row["band_sign"] == "B-lower-entropy")
    negative = sum(1 for row in catchup if row["band_sign"] == "null-lower-entropy")
    if positive >= 5 and negative == 0:
        return "internal-asymmetry-supported"
    if negative >= 4:
        return "null-internally-lower-entropy"
    if positive <= 2 and negative == 0:
        return "no-internal-separation"
    return "mixed-internal-asymmetry"


def render_console_summary(summary: dict[str, Any], band_rows: list[dict[str, Any]]) -> str:
    lines = [
        "D-0111 Q2 internal routing entropy Stage 1",
        "-" * 104,
        f"category: {summary['readout']['category']}",
        (
            "catchup bands: "
            f"+{summary['readout']['catchup_positive_bands']} "
            f"-{summary['readout']['catchup_negative_bands']} "
            f"0={summary['readout']['catchup_zero_bands']}"
        ),
        f"observed_B_entropy_std_catchup: {fmt(summary['readout']['observed_B_entropy_std_catchup'])}",
        "",
        f"{'role':<16} {'band':<11} {'cells':>5} {'median_gap':>12} {'pos':>7}  sign",
    ]
    for row in band_rows:
        lines.append(
            f"{row['cell_role']:<16} {row['band']:<11} {int(row['selected_cells']):>5} "
            f"{fmt(row['median_entropy_gap']):>12} {fmt(row['positive_cell_fraction']):>7}  "
            f"{row['band_sign']}"
        )
    return "\n".join(lines)


def render_report(
    summary: dict[str, Any],
    band_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# D-0111 Q2 Internal Routing Entropy Stage 1",
        "",
        "## Boundary",
        "",
        "- This run reuses N3-05b/N3-05c and introduces no new entropy metric.",
        "- Per-cell adapters restrict the input volume; they do not change N3 formulas.",
        "- Under the one-cell adapter, `component_weight_entropy_norm` is a local version of the N3 component-weight entropy object. It measures internal routing entropy at a specific surface-tied cell, not entropy of the original full transition domain.",
        "- `transition_run/edges.csv` is empty by design and contributes no structure.",
        "- Strong-context cells are descriptive calibration only.",
        "- No figure is authorized or emitted.",
        "",
        "## Verdict",
        "",
        f"- Category: `{summary['readout']['category']}`",
        f"- Catch-up band signs: +{summary['readout']['catchup_positive_bands']} / -{summary['readout']['catchup_negative_bands']} / 0={summary['readout']['catchup_zero_bands']}",
        f"- Sign-test p-floor if all six bands were positive: `{summary['readout']['sign_test_p_floor_all_positive']:.5f}`",
        f"- Observed B entropy std across catch-up cells: `{fmt(summary['readout']['observed_B_entropy_std_catchup'])}`",
        "",
        "## Band Readout",
        "",
        "| Role | Source | Band | Cells | Median gap | Positive fraction | Sign | Support note | Min-family counts |",
        "|---|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in band_rows:
        lines.append(
            f"| `{row['cell_role']}` | `{row['source']}` | `{row['band']}` | {row['selected_cells']} | "
            f"{fmt(row['median_entropy_gap'])} | {fmt(row['positive_cell_fraction'])} | "
            f"`{row['band_sign']}` | {row['support_note']} | `{row['min_family_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Family Entropy Summary",
            "",
            "| Role | Family | Rows | Mean | Median | Min | Max | Std |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in family_rows:
        lines.append(
            f"| `{row['cell_role']}` | `{row['family']}` | {row['rows']} | "
            f"{fmt(row['mean_entropy'])} | {fmt(row['median_entropy'])} | "
            f"{fmt(row['min_entropy'])} | {fmt(row['max_entropy'])} | {fmt(row['std_entropy'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretive Notes",
            "",
            "- A positive `entropy_gap` means observed B has lower local component-weight entropy than the best hard null for that cell.",
            "- The hard-null minimum is intentionally hostile: it chooses the lower-entropy seed/family among `phase-matched-lz` and `block-entropy`.",
            "- The band `813M-822M` has one catch-up cell, so its band median and positive fraction are single-cell summaries.",
            "- These Q2 labels do not authorize manuscript changes by themselves.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_adapter_manifest(adapter_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "adapter": "q2-per-cell-n3-format-bridge",
        "cell_count": len(adapter_rows),
        "contract_sha256": CONTRACT_SHA256,
        "rows": adapter_rows,
    }


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
            "cell_entropy.csv",
            "cell_entropy_envelope.csv",
            "band_entropy_summary.csv",
            "family_entropy_summary.csv",
            "adapter_index.csv",
            "adapter_manifest.json",
            "manifest.json",
        ],
        "manuscript_boundary": "No manuscript edit is authorized by this run.",
    }


def print_saved_paths(run_dir: Path) -> None:
    print(f"\nSaved summary to: {run_dir / 'summary.json'}")
    print(f"Saved report to: {run_dir / 'report.md'}")
    print(f"Saved cell entropy CSV to: {run_dir / 'cell_entropy.csv'}")
    print(f"Saved band summary CSV to: {run_dir / 'band_entropy_summary.csv'}")
    print(f"Saved manifest to: {run_dir / 'manifest.json'}")


def cell_id(cell: dict[str, Any]) -> str:
    return f"{cell['cell_role']}__{slug(cell['source'])}__{cell['band']}__lag-{int(cell['lag_bits'])}"


def slug(value: Any) -> str:
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "source"


def band_start(label: str) -> int:
    return int(label.split("-", 1)[0].replace("M", "")) * 1_000_000


def band_stop(label: str) -> int:
    return int(label.split("-", 1)[1].replace("M", "")) * 1_000_000


def band_sign(value: float) -> str:
    if value > 0.0:
        return "B-lower-entropy"
    if value < 0.0:
        return "null-lower-entropy"
    return "no-separation"


def sign_label(value: float) -> str:
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def role_sort(role: str) -> int:
    return 0 if role == "catchup" else 1


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_header_only_csv(path: Path, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def to_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def to_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0


def stddev(values: list[float]) -> float | None:
    if not values:
        return None
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
