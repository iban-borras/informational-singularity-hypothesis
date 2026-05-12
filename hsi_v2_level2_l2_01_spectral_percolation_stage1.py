#!/usr/bin/env python3
"""L2-01 Stage 1 spectral-percolation readout.

Implements D-0116. The script builds only the frozen multi-band N3-05b
quotient graphs and emits the preregistered tier readouts. It does not create
figures or edit manuscript-facing files.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hsi_v2_level2_l2_01_spectral_percolation_stage0 import band_start, hard_margin, sha256_file
from hsi_v2_phase2_transport_defect_strict import phase_print
from v2.common.cli import resolve_dir
from v2.phase3.quotient_child_routing_spectrum import (
    build_quotient_graphs,
    flatten_edges,
    flatten_nodes,
    summarize_graphs,
)


CONTRACT_SHA256 = "CE903E412CD0AD9CFB6E9F5D4B2B599D2EC70AEB2EE05E8A1CB2E11F93DA836C"
STAGE0_HASHES = {
    "summary.json": "BDEB00CE4962038E7131736D529CC623C2FE374C329A176B358339950EC6924C",
    "band_inventory.csv": "4ECD7EBB2C75570D04D2B57AA900E1799B233924C1015FA70D086C4600805E2F",
    "window_inventory.csv": "929C756A56693541B9B44F85CB2B7D7AB5AFB71319D52485ADE334DB5D24165A",
    "stage1_candidate_windows.csv": "42E2C13DC7EB4F2492C50834DF2D1FEA25CE1858B95942E2501A4EAB20B28638",
    "manifest.json": "7C7D8DC15F8710A77999847745F488F12845EECCDC2AFA95F935D26CCDA87D07",
}
DEFAULT_CONTRACT = "Documentation/HSI_v2_Level2_L2_01_SpectralPercolation_Preregistration_v0_3.md"
DEFAULT_STAGE0_RUN = (
    "results/hsi_v2/level2/l2_01_spectral_percolation/stage0_observability/"
    "l2-01-spectral-percolation-stage0__20260510T091237"
)
DEFAULT_PHASE4_ROOT = "results/hsi_v2/phase4"
DEFAULT_BMINI_RUN = (
    "results/hsi_v2/phase5_cross_arc_horizon/bmini_v0_2/"
    "phase5-bmini-cross-arc-horizon__arcs-3__20260508T153520"
)
DEFAULT_OUTPUT_DIR = "results/hsi_v2/level2/l2_01_spectral_percolation/stage1"
NEGATIVE_LAGS = {-29_500_000, -27_000_000, -26_500_000, -24_000_000}
PRIMARY_INVARIANTS = [
    "beta0",
    "zero_eigenvalue_count",
    "lambda2_norm_laplacian",
    "largest_component_fraction",
    "edge_density",
]
HARD_NULL_FAMILIES = ["phase-matched-lz", "block-entropy", "matched-lz"]

FROZEN_WINDOWS = [
    {
        "tier": "tier1",
        "pair_id": "T1-ARC-B",
        "role": "interior",
        "arc_id": "ARC-B",
        "window_id": "ARC-B__interior__606M-615M_to_615M-624M__n2",
        "bands": ["606M-615M", "615M-624M"],
    },
    {
        "tier": "tier1",
        "pair_id": "T1-ARC-B",
        "role": "boundary",
        "arc_id": "ARC-B",
        "window_id": "ARC-B__boundary__624M-633M_to_633M-642M__n2",
        "bands": ["624M-633M", "633M-642M"],
    },
    {
        "tier": "tier1",
        "pair_id": "T1-P4",
        "role": "interior",
        "arc_id": "P4",
        "window_id": "P4__interior__777M-786M_to_786M-795M__n2",
        "bands": ["777M-786M", "786M-795M"],
    },
    {
        "tier": "tier1",
        "pair_id": "T1-P4",
        "role": "boundary",
        "arc_id": "P4",
        "window_id": "P4__boundary__795M-804M_to_804M-813M__n2",
        "bands": ["795M-804M", "804M-813M"],
    },
    {
        "tier": "tier2",
        "role": "interior",
        "arc_id": "ARC-B",
        "window_id": "ARC-B__interior__597M-606M_to_615M-624M__n3",
        "bands": ["597M-606M", "606M-615M", "615M-624M"],
    },
    {
        "tier": "tier2",
        "role": "interior",
        "arc_id": "P4",
        "window_id": "P4__interior__768M-777M_to_786M-795M__n3",
        "bands": ["768M-777M", "777M-786M", "786M-795M"],
    },
    {
        "tier": "tier2",
        "role": "boundary",
        "arc_id": "ARC-A",
        "window_id": "ARC-A__boundary__300M-309M_to_318M-327M__n3",
        "bands": ["300M-309M", "309M-318M", "318M-327M"],
    },
    {
        "tier": "tier2",
        "role": "boundary",
        "arc_id": "ARC-A",
        "window_id": "ARC-A__boundary__309M-318M_to_327M-336M__n3",
        "bands": ["309M-318M", "318M-327M", "327M-336M"],
    },
    {
        "tier": "tier2",
        "role": "inactive",
        "arc_id": "ARC-C",
        "window_id": "ARC-C__inactive__1110M-1119M_to_1128M-1137M__n3",
        "bands": ["1110M-1119M", "1119M-1128M", "1128M-1137M"],
    },
    {
        "tier": "tier2",
        "role": "inactive",
        "arc_id": "ARC-C",
        "window_id": "ARC-C__inactive__1119M-1128M_to_1137M-1146M__n3",
        "bands": ["1119M-1128M", "1128M-1137M", "1137M-1146M"],
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D-0116 L2-01 spectral-percolation Stage 1.")
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--stage0-run", default=DEFAULT_STAGE0_RUN)
    parser.add_argument("--phase4-root", default=DEFAULT_PHASE4_ROOT)
    parser.add_argument("--bmini-run", default=DEFAULT_BMINI_RUN)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    contract = resolve_dir(args.contract, anchor_file=__file__)
    stage0_run = resolve_dir(args.stage0_run, anchor_file=__file__)
    phase4_root = resolve_dir(args.phase4_root, anchor_file=__file__)
    bmini_run = resolve_dir(args.bmini_run, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    verify_inputs(contract, stage0_run)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"l2-01-spectral-percolation-stage1__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing D-0116 L2-01 spectral-percolation Stage 1",
        "frozen windows | N3-05b pooled-retained-mass | no figures",
        quiet=args.quiet,
    )

    band_rows = load_stage0_band_rows(stage0_run)
    routing_index, routing_paths = index_routing_rows([phase4_root, bmini_run])
    window_rows, node_rows, edge_rows = build_all_window_graphs(
        band_rows=band_rows,
        routing_index=routing_index,
        routing_paths=routing_paths,
    )
    if not observed_b_complete(window_rows):
        return write_stop(
            run_dir,
            status="inputs-not-derivable",
            reason="at least one frozen observed-B window is missing",
            started=started,
            args=args,
        )

    tier1_rows, tier1_verdict = build_tier1(window_rows, source_filter=observed_b_filter)
    tier2_rows, tier2_verdict = build_tier2(window_rows, source_filter=observed_b_filter)
    tier3_rows, tier3_verdict, coverage_flag = build_tier3(window_rows)
    if tier3_verdict == "hard-null-inputs-not-derivable":
        overall = "hard-null-inputs-not-derivable"
    else:
        overall = overall_verdict(tier1_verdict, tier2_verdict, tier3_verdict)

    summary = {
        "run_id": run_dir.name,
        "timestamp_utc": timestamp,
        "status": "completed",
        "overall_verdict": overall,
        "tier1_verdict": tier1_verdict,
        "tier2_verdict": tier2_verdict,
        "tier3_verdict": tier3_verdict,
        "coverage_flag": coverage_flag,
        "contract": {"path": str(contract.resolve()), "sha256": sha256_file(contract)},
        "stage0_run": str(stage0_run.resolve()),
        "stage0_hashes": STAGE0_HASHES,
        "parameters": {
            "edge_weight_mode": "pooled-retained-mass",
            "parent_scale": 40,
            "top_patterns": 128,
            "channel": "negative",
            "zero_tolerance": 1e-9,
            "node_count_imbalance_threshold": 2.0,
        },
        "counts": {
            "windows": len(FROZEN_WINDOWS),
            "window_graph_rows": len(window_rows),
            "graph_nodes": len(node_rows),
            "graph_edges": len(edge_rows),
        },
        "boundary": [
            "No figure is emitted.",
            "No manuscript edit is authorized.",
            "Readouts are descriptive directional results under low-N preregistered constraints.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    write_json(run_dir / "summary.json", summary)
    write_csv_dynamic(run_dir / "window_graph_summary.csv", window_rows)
    write_csv_dynamic(run_dir / "tier1_within_arc_contrasts.csv", tier1_rows)
    write_csv_dynamic(run_dir / "tier2_cross_arc_medians.csv", tier2_rows)
    write_csv_dynamic(run_dir / "tier3_hard_null_mimicry.csv", tier3_rows)
    write_csv_dynamic(run_dir / "graph_nodes.csv", node_rows)
    write_csv_dynamic(run_dir / "graph_edges.csv", edge_rows)
    write_json(run_dir / "manifest.json", build_manifest(summary, args))
    (run_dir / "report.md").write_text(
        render_report(summary, tier1_rows, tier2_rows, tier3_rows),
        encoding="utf-8",
    )

    if not args.quiet:
        print(render_console_summary(summary))
        print_saved_paths(run_dir)
    return 0


def verify_inputs(contract: Path, stage0_run: Path) -> None:
    contract_hash = sha256_file(contract)
    if contract_hash != CONTRACT_SHA256:
        raise SystemExit(f"Contract SHA mismatch: expected {CONTRACT_SHA256}, got {contract_hash}")
    for name, expected in STAGE0_HASHES.items():
        path = stage0_run / name
        if not path.is_file():
            raise SystemExit(f"Missing Stage 0 artifact: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(f"Stage 0 hash mismatch for {name}: expected {expected}, got {actual}")


def load_stage0_band_rows(stage0_run: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in load_csv(stage0_run / "band_inventory.csv"):
        rows[row["band"]] = row
    return rows


def index_routing_rows(roots: list[Path]) -> tuple[dict[tuple[str, int], list[dict[str, str]]], dict[tuple[str, int], list[str]]]:
    index: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    paths: dict[tuple[str, int], list[str]] = defaultdict(list)
    for root in roots:
        for path in root.rglob("routing_rows.csv"):
            rows = load_csv(path)
            if not rows:
                continue
            lag_bits = int(float(rows[0]["candidate_lag_bits"]))
            if lag_bits not in NEGATIVE_LAGS:
                continue
            band = find_band(path)
            if not band:
                continue
            key = (band, lag_bits)
            index[key].extend(rows)
            paths[key].append(str(path))
    return dict(index), dict(paths)


def build_all_window_graphs(
    *,
    band_rows: dict[str, dict[str, Any]],
    routing_index: dict[tuple[str, int], list[dict[str, str]]],
    routing_paths: dict[tuple[str, int], list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    window_graph_rows = []
    all_nodes = []
    all_edges = []
    for window in FROZEN_WINDOWS:
        cells = []
        source_paths = []
        for band in window["bands"]:
            if band not in band_rows:
                raise SystemExit(f"Frozen band missing from Stage 0 inventory: {band}")
            lag_bits, best_margin = best_lag_for_band(Path(band_rows[band]["lag_response_csv"]), band)
            key = (band, lag_bits)
            if key not in routing_index:
                raise SystemExit(f"Missing routing rows for frozen cell {band} lag={lag_bits}")
            source_paths.extend(routing_paths.get(key, []))
            start = band_start(band)
            cells.append(
                {
                    "top_patterns": 128,
                    "band": band,
                    "band_start_bits": start,
                    "band_stop_bits": start + 9_000_000,
                    "channel": "negative",
                    "node_class": window["role"],
                    "strong_count": None,
                    "best_margin_lag_bits": lag_bits,
                    "best_margin": best_margin,
                }
            )
        graphs = build_quotient_graphs(
            cells,
            routing_index=routing_index,
            parent_scale=40,
            edge_weight_mode="pooled-retained-mass",
        )
        summary_rows = summarize_graphs(graphs)
        graph_by_key = {(graph["source_key"], graph["channel"]): graph for graph in graphs.values()}
        for row in summary_rows:
            graph = graph_by_key[(row["source_key"], row["channel"])]
            enriched = enrich_graph_row(row, graph)
            enriched.update(window_metadata(window, source_paths))
            window_graph_rows.append(enriched)
        for row in flatten_nodes(graphs):
            row.update({"window_id": window["window_id"], "tier": window["tier"], "role": window["role"], "arc_id": window["arc_id"]})
            all_nodes.append(row)
        for row in flatten_edges(graphs):
            row.update({"window_id": window["window_id"], "tier": window["tier"], "role": window["role"], "arc_id": window["arc_id"]})
            all_edges.append(row)
    return window_graph_rows, all_nodes, all_edges


def enrich_graph_row(row: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    node_count = int(row["node_count"])
    edge_count = int(row["edge_count"])
    beta0 = component_count(list(graph["nodes"].values()), list(graph["edges"].values()))
    edge_density = 0.0 if node_count < 2 else (2.0 * edge_count) / (node_count * (node_count - 1))
    beta1 = edge_count - node_count + beta0
    out = dict(row)
    out.update(
        {
            "beta0": beta0,
            "beta1": beta1,
            "edge_density": edge_density,
            "beta0_zero_mismatch": beta0 != int(row["zero_eigenvalue_count"]),
        }
    )
    return out


def window_metadata(window: dict[str, Any], source_paths: list[str]) -> dict[str, Any]:
    return {
        "tier": window["tier"],
        "pair_id": window.get("pair_id", ""),
        "role": window["role"],
        "arc_id": window["arc_id"],
        "window_id": window["window_id"],
        "bands": ";".join(window["bands"]),
        "source_routing_paths": json.dumps(sorted(set(source_paths))),
    }


def build_tier1(
    rows: list[dict[str, Any]],
    *,
    source_filter,
) -> tuple[list[dict[str, Any]], str]:
    by_pair_role = {}
    for row in rows:
        if row["tier"] == "tier1" and source_filter(row):
            by_pair_role[(row["pair_id"], row["role"])] = row
    out = []
    statuses = []
    for pair_id in ("T1-ARC-B", "T1-P4"):
        interior = by_pair_role.get((pair_id, "interior"))
        boundary = by_pair_role.get((pair_id, "boundary"))
        if not interior or not boundary:
            statuses.append("transition-missing")
            out.append({"pair_id": pair_id, "transition_status": "transition-missing"})
            continue
        contrast = contrast_rows(pair_id, interior, boundary)
        statuses.append(contrast["transition_status"])
        out.append(contrast)
    if all(status == "transition-positive" for status in statuses):
        verdict = "within-arc-transition-support"
    elif all(status == "transition-negative" for status in statuses):
        verdict = "within-arc-no-support"
    else:
        verdict = "within-arc-mixed"
    for row in out:
        row["tier1_verdict"] = verdict
    return out, verdict


def build_tier2(
    rows: list[dict[str, Any]],
    *,
    source_filter,
) -> tuple[list[dict[str, Any]], str]:
    by_role = defaultdict(list)
    for row in rows:
        if row["tier"] == "tier2" and source_filter(row):
            by_role[row["role"]].append(row)
    medians = {}
    out = []
    for role in ("interior", "boundary", "inactive"):
        role_rows = by_role.get(role, [])
        record = {"role": role, "window_count": len(role_rows)}
        for invariant in PRIMARY_INVARIANTS:
            values = [float(row[invariant]) for row in role_rows]
            record[f"median_{invariant}"] = statistics.median(values) if values else ""
        record["node_count_min"] = min((int(row["node_count"]) for row in role_rows), default=0)
        record["node_count_max"] = max((int(row["node_count"]) for row in role_rows), default=0)
        record["node_count_imbalance"] = node_imbalance(role_rows)
        medians[role] = record
        out.append(record)
    status = median_transition_status(medians["interior"], medians["boundary"])
    for row in out:
        row["tier2_verdict"] = {
            "transition-positive": "cross-arc-transition-support",
            "transition-negative": "cross-arc-no-support",
        }.get(status, "cross-arc-mixed")
        row["transition_status"] = status
    return out, out[0]["tier2_verdict"]


def build_tier3(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str, str]:
    out = []
    family_categories = []
    coverage_flag = ""
    for family in HARD_NULL_FAMILIES:
        family_rows = [row for row in rows if row.get("null_model") == family]
        if not family_rows:
            continue
        seeds = sorted({str(row.get("null_seed", "")) for row in family_rows})
        seed_categories = []
        complete_count = 0
        for seed in seeds:
            seed_filter = lambda row, family=family, seed=seed: row.get("null_model") == family and str(row.get("null_seed", "")) == seed
            complete = seed_complete(rows, seed_filter)
            if not complete:
                out.append({"null_model": family, "null_seed": seed, "coverage": "partial"})
                continue
            complete_count += 1
            tier1_rows, tier1_verdict = build_tier1(rows, source_filter=seed_filter)
            tier2_rows, tier2_verdict = build_tier2(rows, source_filter=seed_filter)
            category = seed_mimicry_category(tier1_verdict, tier2_verdict)
            seed_categories.append(category)
            out.append(
                {
                    "null_model": family,
                    "null_seed": seed,
                    "coverage": "complete",
                    "tier1_verdict": tier1_verdict,
                    "tier2_verdict": tier2_verdict,
                    "seed_mimicry_category": category,
                    "tier1_detail": json.dumps(tier1_rows, sort_keys=True),
                    "tier2_detail": json.dumps(tier2_rows, sort_keys=True),
                }
            )
        if complete_count == 0:
            family_category = "hard-null-inputs-not-derivable"
        elif "hard-null-mimicry" in seed_categories:
            family_category = "hard-null-mimicry"
        elif "partial-hard-null-mimicry" in seed_categories:
            family_category = "partial-hard-null-mimicry"
        else:
            family_category = "no-hard-null-mimicry"
        if complete_count < len(seeds):
            coverage_flag = "hard-null-inputs-partial"
        family_categories.append(family_category)
        out.append(
            {
                "null_model": family,
                "null_seed": "__family__",
                "coverage": "complete" if complete_count == len(seeds) else "partial",
                "complete_seed_count": complete_count,
                "total_seed_count": len(seeds),
                "family_mimicry_category": family_category,
            }
        )
    if not family_categories or all(category == "hard-null-inputs-not-derivable" for category in family_categories):
        return out, "hard-null-inputs-not-derivable", coverage_flag
    if "hard-null-mimicry" in family_categories:
        return out, "hard-null-mimicry", coverage_flag
    if "partial-hard-null-mimicry" in family_categories:
        return out, "partial-hard-null-mimicry", coverage_flag
    return out, "no-hard-null-mimicry", coverage_flag


def contrast_rows(pair_id: str, interior: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "beta0_direction": float(boundary["beta0"]) > float(interior["beta0"]),
        "zero_eigenvalue_count_direction": float(boundary["zero_eigenvalue_count"]) > float(interior["zero_eigenvalue_count"]),
        "lambda2_norm_laplacian_direction": float(boundary["lambda2_norm_laplacian"]) < float(interior["lambda2_norm_laplacian"]),
        "largest_component_fraction_direction": float(boundary["largest_component_fraction"]) < float(interior["largest_component_fraction"]),
        "edge_density_direction": float(boundary["edge_density"]) < float(interior["edge_density"]),
    }
    passed = sum(1 for value in checks.values() if value)
    if passed >= 4:
        status = "transition-positive"
    elif passed <= 1:
        status = "transition-negative"
    else:
        status = "transition-mixed"
    node_ratio = safe_ratio(
        max(float(interior["node_count"]), float(boundary["node_count"])),
        min(float(interior["node_count"]), float(boundary["node_count"])),
    )
    return {
        "pair_id": pair_id,
        "interior_window": interior["window_id"],
        "boundary_window": boundary["window_id"],
        "passed_primary_directions": passed,
        "transition_status": status,
        "node_count_ratio": node_ratio,
        "node_count_imbalance": node_ratio > 2.0,
        "beta0_zero_mismatch": truthy(interior["beta0_zero_mismatch"]) or truthy(boundary["beta0_zero_mismatch"]),
        **checks,
    }


def median_transition_status(interior: dict[str, Any], boundary: dict[str, Any]) -> str:
    checks = [
        float(boundary["median_beta0"]) > float(interior["median_beta0"]),
        float(boundary["median_zero_eigenvalue_count"]) > float(interior["median_zero_eigenvalue_count"]),
        float(boundary["median_lambda2_norm_laplacian"]) < float(interior["median_lambda2_norm_laplacian"]),
        float(boundary["median_largest_component_fraction"]) < float(interior["median_largest_component_fraction"]),
        float(boundary["median_edge_density"]) < float(interior["median_edge_density"]),
    ]
    passed = sum(1 for value in checks if value)
    if passed >= 4:
        return "transition-positive"
    if passed <= 1:
        return "transition-negative"
    return "transition-mixed"


def seed_mimicry_category(tier1: str, tier2: str) -> str:
    t1_support = tier1 == "within-arc-transition-support"
    t2_support = tier2 == "cross-arc-transition-support"
    if t1_support and t2_support:
        return "hard-null-mimicry"
    if t1_support or t2_support:
        return "partial-hard-null-mimicry"
    return "no-hard-null-mimicry"


def overall_verdict(tier1: str, tier2: str, tier3: str) -> str:
    if tier3 == "hard-null-mimicry":
        return "hard-null-mimicry"
    if tier3 == "partial-hard-null-mimicry":
        return "mixed-or-low-power"
    t1_support = tier1 == "within-arc-transition-support"
    t2_support = tier2 == "cross-arc-transition-support"
    if t1_support and t2_support:
        return "spectral-percolation-candidate"
    if t1_support and not t2_support:
        return "within-arc-only-support"
    if tier1 == "within-arc-no-support" and t2_support:
        return "cross-arc-only-support"
    if tier1 == "within-arc-mixed" or tier2 == "cross-arc-mixed":
        return "mixed-or-low-power"
    return "no-level2-support"


def observed_b_complete(rows: list[dict[str, Any]]) -> bool:
    windows = {window["window_id"] for window in FROZEN_WINDOWS}
    have = {row["window_id"] for row in rows if observed_b_filter(row)}
    return windows <= have


def seed_complete(rows: list[dict[str, Any]], source_filter) -> bool:
    windows = {window["window_id"] for window in FROZEN_WINDOWS}
    have = {row["window_id"] for row in rows if source_filter(row)}
    return windows <= have


def observed_b_filter(row: dict[str, Any]) -> bool:
    return row.get("source_kind") == "observed" and row.get("variant") == "B"


def best_lag_for_band(lag_csv: Path, band: str) -> tuple[int, float]:
    candidates = []
    for row in load_csv(lag_csv):
        if row.get("band") != band:
            continue
        lag_bits = int(float(row["lag_bits"]))
        if lag_bits not in NEGATIVE_LAGS:
            continue
        candidates.append((lag_bits, hard_margin(row)))
    if not candidates:
        raise SystemExit(f"No frozen negative lag rows for {band} in {lag_csv}")
    return max(candidates, key=lambda item: item[1])


def component_count(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> int:
    adjacency = {node["id"]: set() for node in nodes}
    for edge in edges:
        left, right = edge["nodes"]
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)
    seen = set()
    count = 0
    for node_id in adjacency:
        if node_id in seen:
            continue
        count += 1
        frontier = [node_id]
        seen.add(node_id)
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append(neighbor)
    return count


def node_imbalance(rows: list[dict[str, Any]]) -> bool:
    counts = [float(row["node_count"]) for row in rows if float(row["node_count"]) > 0]
    if len(counts) < 2:
        return False
    return max(counts) / min(counts) > 2.0


def safe_ratio(top: float, bottom: float) -> float:
    if bottom <= 0:
        return float("inf")
    return top / bottom


def find_band(path: Path) -> str | None:
    for part in path.parts:
        if "M-" in part and part.endswith("M"):
            return part
    return None


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv_dynamic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def truthy(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def build_manifest(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run_id": summary["run_id"],
        "script": Path(__file__).name,
        "generated_at": summary["timestamp_utc"],
        "contract": summary["contract"],
        "stage0_run": summary["stage0_run"],
        "stage0_hashes": summary["stage0_hashes"],
        "arguments": vars(args),
        "parameters": summary["parameters"],
        "outputs": [
            "summary.json",
            "report.md",
            "window_graph_summary.csv",
            "tier1_within_arc_contrasts.csv",
            "tier2_cross_arc_medians.csv",
            "tier3_hard_null_mimicry.csv",
            "graph_nodes.csv",
            "graph_edges.csv",
            "manifest.json",
        ],
    }


def write_stop(run_dir: Path, *, status: str, reason: str, started: float, args: argparse.Namespace) -> int:
    summary = {
        "run_id": run_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
        "status": status,
        "reason": reason,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "manifest.json", {"script": Path(__file__).name, "arguments": vars(args), **summary})
    (run_dir / "report.md").write_text(f"# L2-01 Stage 1 Stop\n\n- Status: `{status}`\n- Reason: {reason}\n", encoding="utf-8")
    print(f"L2-01 Stage 1 stopped: {status} | {reason}")
    return 0


def render_report(
    summary: dict[str, Any],
    tier1_rows: list[dict[str, Any]],
    tier2_rows: list[dict[str, Any]],
    tier3_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# D-0116 L2-01 Spectral-Percolation Stage 1",
        "",
        "## Boundary",
        "",
        "- No figure is authorized.",
        "- No manuscript edit is authorized.",
        "- This is a descriptive directional readout under low-N pre-registered constraints, not an inferential test.",
        "",
        "## Verdict",
        "",
        f"- Overall: `{summary['overall_verdict']}`",
        f"- Tier 1: `{summary['tier1_verdict']}`",
        f"- Tier 2: `{summary['tier2_verdict']}`",
        f"- Tier 3: `{summary['tier3_verdict']}`",
        f"- Coverage flag: `{summary['coverage_flag'] or '-'}`",
        "",
        "## Tier 1 Within-Arc Contrasts",
        "",
        "| Pair | Status | Passed | Node ratio | Node imbalance | Beta0/zero mismatch |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in tier1_rows:
        lines.append(
            f"| `{row.get('pair_id', '')}` | `{row.get('transition_status', '')}` | "
            f"{row.get('passed_primary_directions', '')} | {fmt(row.get('node_count_ratio', ''))} | "
            f"`{row.get('node_count_imbalance', '')}` | `{row.get('beta0_zero_mismatch', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Tier 2 Cross-Arc Medians",
            "",
            "| Role | Windows | Median beta0 | Median zero | Median lambda2 | Median LCC | Median density | Node imbalance |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in tier2_rows:
        lines.append(
            f"| `{row['role']}` | {row['window_count']} | {fmt(row.get('median_beta0'))} | "
            f"{fmt(row.get('median_zero_eigenvalue_count'))} | {fmt(row.get('median_lambda2_norm_laplacian'))} | "
            f"{fmt(row.get('median_largest_component_fraction'))} | {fmt(row.get('median_edge_density'))} | "
            f"`{row.get('node_count_imbalance')}` |"
        )
    lines.extend(
        [
            "",
            "## Tier 3 Hard-Null Mimicry",
            "",
            "| Null model | Seed | Coverage | Category | Tier 1 | Tier 2 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in tier3_rows:
        lines.append(
            f"| `{row.get('null_model', '')}` | `{row.get('null_seed', '')}` | `{row.get('coverage', '')}` | "
            f"`{row.get('seed_mimicry_category', row.get('family_mimicry_category', ''))}` | "
            f"`{row.get('tier1_verdict', '')}` | `{row.get('tier2_verdict', '')}` |"
        )
    return "\n".join(lines) + "\n"


def render_console_summary(summary: dict[str, Any]) -> str:
    lines = [
        "D-0116 L2-01 spectral-percolation Stage 1",
        "-" * 88,
        f"overall: {summary['overall_verdict']}",
        f"tier1:   {summary['tier1_verdict']}",
        f"tier2:   {summary['tier2_verdict']}",
        f"tier3:   {summary['tier3_verdict']}",
    ]
    if summary["coverage_flag"]:
        lines.append(f"coverage: {summary['coverage_flag']}")
    return "\n".join(lines)


def print_saved_paths(run_dir: Path) -> None:
    print(f"\nSaved summary to: {run_dir / 'summary.json'}")
    print(f"Saved report to: {run_dir / 'report.md'}")
    print(f"Saved window graph summary to: {run_dir / 'window_graph_summary.csv'}")
    print(f"Saved manifest to: {run_dir / 'manifest.json'}")


def fmt(value: Any) -> str:
    if value in (None, ""):
        return "-"
    return f"{float(value):.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
