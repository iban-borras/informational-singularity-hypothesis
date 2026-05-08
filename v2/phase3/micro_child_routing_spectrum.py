from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .defect_spectrum import compute_graph_spectrum


BAND_RE = re.compile(r"^\d+M-\d+M$")


def build_micro_child_routing_spectrum(
    *,
    transition_run: Path,
    child_routing_run: Path,
    top_patterns: int,
    retention_threshold: float = 0.75,
) -> dict[str, Any]:
    """Build unit-weighted micro-routing graphs from existing Phase 2 artifacts."""
    cells = load_transition_cells(transition_run, top_patterns=top_patterns)
    routing_index = index_routing_row_files(child_routing_run)
    graphs = build_graphs_from_cells(
        cells,
        routing_index=routing_index,
        retention_threshold=retention_threshold,
    )
    graph_rows = summarize_graphs(graphs)
    return {
        "selection": {
            "transition_run": str(transition_run),
            "child_routing_run": str(child_routing_run),
            "top_patterns": top_patterns,
            "retention_threshold": retention_threshold,
        },
        "notes": [
            "N3-04 is a micro-scale spectral readout over existing child-routing artifacts.",
            "Nodes are parent patterns, retained child destinations, and explicit dropout/reroute terminals inside sampled phase-channel cells.",
            "Edges are unit-weighted to avoid metric tuning: parent-child retention, parent-dropout/reroute defect, and same-parent continuity across adjacent bands.",
            "The object is pre-geometric and internal: it tests graph connectivity of child-routing events, not curvature or a final transport law.",
        ],
        "transition_cells": cells,
        "graph_summary": graph_rows,
        "graph_nodes": flatten_graph_nodes(graphs),
        "graph_edges": flatten_graph_edges(graphs),
        "missing_cells": missing_cells(cells, routing_index),
    }


def load_transition_cells(transition_run: Path, *, top_patterns: int) -> list[dict[str, Any]]:
    rows = read_csv(transition_run / "nodes.csv")
    cells = []
    for row in rows:
        if to_int(row.get("top_patterns")) != top_patterns:
            continue
        lag_bits = optional_int(row.get("best_margin_lag_bits"))
        if lag_bits is None:
            continue
        cells.append(
            {
                "top_patterns": top_patterns,
                "band": row["band"],
                "band_start_bits": to_int(row["band_start_bits"]),
                "band_stop_bits": to_int(row["band_stop_bits"]),
                "channel": row["channel"],
                "node_class": row["node_class"],
                "strong_count": to_int(row.get("strong_count")),
                "best_margin_lag_bits": lag_bits,
                "best_margin": optional_float(row.get("best_margin")),
            }
        )
    cells.sort(key=lambda item: (item["channel"], item["band_start_bits"]))
    return cells


def index_routing_row_files(child_routing_run: Path) -> dict[tuple[str, int], list[dict[str, str]]]:
    index: dict[tuple[str, int], list[dict[str, str]]] = {}
    for path in child_routing_run.rglob("routing_rows.csv"):
        band = find_band_label(path.relative_to(child_routing_run))
        if band is None:
            continue
        rows = read_csv(path)
        if not rows:
            continue
        lag_bits = infer_lag_bits(rows)
        if lag_bits is None:
            continue
        index[(band, lag_bits)] = rows
    return index


def build_graphs_from_cells(
    cells: list[dict[str, Any]],
    *,
    routing_index: dict[tuple[str, int], list[dict[str, str]]],
    retention_threshold: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    graphs: dict[tuple[str, str], dict[str, Any]] = {}
    parent_by_graph_channel_band_pattern: dict[tuple[str, str, str, str], str] = {}
    band_order_by_channel: dict[str, list[str]] = defaultdict(list)

    for cell in cells:
        band_order_by_channel[cell["channel"]].append(cell["band"])
        raw_rows = routing_index.get((cell["band"], cell["best_margin_lag_bits"]), [])
        for row in aggregate_routing_rows(raw_rows):
            if float_or_zero(row.get("anchor_event_total_sum")) <= 0:
                continue
            source_key = source_key_for(row)
            graph_key = (source_key, cell["channel"])
            graph = graphs.setdefault(
                graph_key,
                {
                    "source_key": source_key,
                    "variant": row["variant"],
                    "source_label": row["source_label"],
                    "source_kind": row["source_kind"],
                    "null_model": row.get("null_model", ""),
                    "null_seed": row.get("null_seed", ""),
                    "channel": cell["channel"],
                    "nodes": {},
                    "edges": {},
                    "cell_count": 0,
                    "cell_labels": set(),
                    "routing_class_counts": Counter(),
                    "conserved_parent_count": 0,
                    "parent_count": 0,
                    "retention_sum": 0.0,
                },
            )
            if cell["band"] not in graph["cell_labels"]:
                graph["cell_labels"].add(cell["band"])
                graph["cell_count"] += 1

            pattern = row["pattern"]
            retention = optional_float(row.get("child_destination_retention_pooled")) or 0.0
            graph["parent_count"] += 1
            graph["retention_sum"] += retention
            if retention >= retention_threshold:
                graph["conserved_parent_count"] += 1
            graph["routing_class_counts"][row.get("routing_class", "")] += 1

            parent_id = f"P::{cell['channel']}::{cell['band']}::{pattern}"
            add_node(
                graph,
                parent_id,
                node_type="parent",
                band=cell["band"],
                channel=cell["channel"],
                pattern=pattern,
                routing_class=row.get("routing_class", ""),
                retention=retention,
                node_class=cell["node_class"],
            )
            parent_by_graph_channel_band_pattern[(source_key, cell["channel"], cell["band"], pattern)] = parent_id

            anchor_children = parse_json_map(row.get("anchor_child_mass_map", "{}"))
            candidate_children = parse_json_map(row.get("candidate_child_mass_map", "{}"))
            for child in sorted(set(anchor_children) & set(candidate_children)):
                if float(candidate_children.get(child, 0.0)) <= 0:
                    continue
                child_id = f"C::{cell['channel']}::{cell['band']}::{child}"
                add_node(
                    graph,
                    child_id,
                    node_type="child",
                    band=cell["band"],
                    channel=cell["channel"],
                    pattern=child,
                    routing_class="retained-child",
                    retention=1.0,
                    node_class=cell["node_class"],
                )
                add_edge(graph, parent_id, child_id, edge_type="retained-child")

            if float_or_zero(row.get("anchor_child_deficit_mass_sum")) > 0:
                dropout_id = f"D::{cell['channel']}::{cell['band']}::{pattern}"
                add_node(
                    graph,
                    dropout_id,
                    node_type="dropout",
                    band=cell["band"],
                    channel=cell["channel"],
                    pattern="__dropout__",
                    routing_class="dropout",
                    retention=0.0,
                    node_class=cell["node_class"],
                )
                add_edge(graph, parent_id, dropout_id, edge_type="dropout")

            if float_or_zero(row.get("candidate_reroute_mass_sum")) > 0:
                reroute_id = f"R::{cell['channel']}::{cell['band']}::{pattern}"
                add_node(
                    graph,
                    reroute_id,
                    node_type="reroute",
                    band=cell["band"],
                    channel=cell["channel"],
                    pattern="__reroute__",
                    routing_class="reroute",
                    retention=0.0,
                    node_class=cell["node_class"],
                )
                add_edge(graph, parent_id, reroute_id, edge_type="reroute")

    for (source_key, channel), graph in graphs.items():
        bands = sorted(set(band_order_by_channel[channel]), key=band_start_key)
        for left, right in zip(bands, bands[1:]):
            left_patterns = {
                key[3]: node_id
                for key, node_id in parent_by_graph_channel_band_pattern.items()
                if key[0] == source_key and key[1] == channel and key[2] == left
            }
            right_patterns = {
                key[3]: node_id
                for key, node_id in parent_by_graph_channel_band_pattern.items()
                if key[0] == source_key and key[1] == channel and key[2] == right
            }
            for pattern in sorted(set(left_patterns) & set(right_patterns)):
                add_edge(
                    graph,
                    left_patterns[pattern],
                    right_patterns[pattern],
                    edge_type="parent-continuity",
                )

    return graphs


def aggregate_routing_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if float_or_zero(row.get("anchor_local_event_total")) <= 0:
            continue
        key = (source_key_for(row), row["pattern"])
        current = grouped.setdefault(
            key,
            {
                "variant": row["variant"],
                "source_label": row["source_label"],
                "source_kind": row["source_kind"],
                "null_model": row.get("null_model", ""),
                "null_seed": row.get("null_seed", ""),
                "anchor_variant": row["anchor_variant"],
                "pattern": row["pattern"],
                "anchor_event_total_sum": 0.0,
                "candidate_event_total_sum": 0.0,
                "anchor_child_deficit_mass_sum": 0.0,
                "candidate_reroute_mass_sum": 0.0,
                "anchor_child_mass_map": {},
                "candidate_child_mass_map": {},
            },
        )
        current["anchor_event_total_sum"] += float_or_zero(row.get("anchor_local_event_total"))
        current["candidate_event_total_sum"] += float_or_zero(row.get("candidate_local_event_total"))
        current["anchor_child_deficit_mass_sum"] += float_or_zero(
            row.get("anchor_child_deficit_mass")
            or row.get("window_anchor_child_deficit_mass")
        )
        current["candidate_reroute_mass_sum"] += float_or_zero(row.get("candidate_reroute_mass"))
        merge_mass_map(current["anchor_child_mass_map"], parse_json_map(row.get("anchor_child_mass_map", "{}")))
        merge_mass_map(
            current["candidate_child_mass_map"],
            parse_json_map(row.get("candidate_child_mass_map", "{}")),
        )

    aggregated = []
    for item in grouped.values():
        anchor_map = item["anchor_child_mass_map"]
        candidate_map = item["candidate_child_mass_map"]
        common = sorted(set(anchor_map) & set(candidate_map))
        overlap = sum(min(anchor_map[child], candidate_map[child]) for child in common)
        retention = (
            overlap / item["anchor_event_total_sum"]
            if item["anchor_event_total_sum"] > 0
            else 0.0
        )
        reroute_share = (
            item["candidate_reroute_mass_sum"] / item["candidate_event_total_sum"]
            if item["candidate_event_total_sum"] > 0
            else 0.0
        )
        item["child_destination_retention_pooled"] = retention
        item["candidate_reroute_share_pooled"] = reroute_share
        item["routing_class"] = classify_routing(
            child_retention=retention,
            reroute_share=reroute_share,
            candidate_total=item["candidate_event_total_sum"],
        )
        aggregated.append(item)
    return aggregated


def summarize_graphs(graphs: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for graph in graphs.values():
        nodes = list(graph["nodes"].values())
        edges = list(graph["edges"].values())
        spectrum = compute_graph_spectrum(
            {
                "node_records": [{"id": node["id"]} for node in nodes],
                "edge_records": [{"nodes": edge["nodes"], "edge_weight": 1.0} for edge in edges],
            }
        )
        edge_counts = Counter(edge["edge_type"] for edge in edges)
        defect_edges = edge_counts.get("dropout", 0) + edge_counts.get("reroute", 0)
        routing_edges = (
            edge_counts.get("retained-child", 0)
            + edge_counts.get("dropout", 0)
            + edge_counts.get("reroute", 0)
        )
        parent_count = int(graph["parent_count"])
        conserved_fraction = (
            graph["conserved_parent_count"] / parent_count if parent_count else None
        )
        defect_fraction = defect_edges / routing_edges if routing_edges else None
        row = {
            "source_key": graph["source_key"],
            "variant": graph["variant"],
            "source_label": graph["source_label"],
            "source_kind": graph["source_kind"],
            "null_model": graph["null_model"],
            "null_seed": graph["null_seed"],
            "channel": graph["channel"],
            "cell_count": graph["cell_count"],
            "parent_count": parent_count,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "retained_child_edges": edge_counts.get("retained-child", 0),
            "dropout_edges": edge_counts.get("dropout", 0),
            "reroute_edges": edge_counts.get("reroute", 0),
            "continuity_edges": edge_counts.get("parent-continuity", 0),
            "defect_edge_fraction": defect_fraction,
            "conserved_parent_fraction": conserved_fraction,
            "mean_child_retention": (
                graph["retention_sum"] / parent_count if parent_count else None
            ),
            "largest_component_fraction": largest_component_fraction(nodes, edges),
            "lambda2_norm_laplacian": spectrum["lambda2_norm_laplacian"],
            "zero_eigenvalue_count": spectrum["zero_eigenvalue_count"],
            "spectral_radius_adjacency": spectrum["spectral_radius_adjacency"],
            "routing_class_counts": dict(sorted(graph["routing_class_counts"].items())),
            "verdict": micro_verdict(
                parent_count=parent_count,
                retained_edges=edge_counts.get("retained-child", 0),
                defect_fraction=defect_fraction,
                conserved_fraction=conserved_fraction,
            ),
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["channel"],
            source_sort_key(row["source_kind"], row["null_model"], row["null_seed"]),
            row["variant"],
            row["source_label"],
        )
    )
    return rows


def flatten_graph_nodes(graphs: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for graph in graphs.values():
        for node in graph["nodes"].values():
            rows.append(
                {
                    "source_key": graph["source_key"],
                    "variant": graph["variant"],
                    "source_label": graph["source_label"],
                    "channel": graph["channel"],
                    **node,
                }
            )
    rows.sort(key=lambda row: (row["channel"], row["source_key"], row["id"]))
    return rows


def flatten_graph_edges(graphs: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for graph in graphs.values():
        for edge in graph["edges"].values():
            rows.append(
                {
                    "source_key": graph["source_key"],
                    "variant": graph["variant"],
                    "source_label": graph["source_label"],
                    "channel": graph["channel"],
                    **edge,
                }
            )
    rows.sort(key=lambda row: (row["channel"], row["source_key"], row["id"]))
    return rows


def missing_cells(
    cells: list[dict[str, Any]],
    routing_index: dict[tuple[str, int], list[dict[str, str]]],
) -> list[dict[str, Any]]:
    return [
        cell
        for cell in cells
        if (cell["band"], cell["best_margin_lag_bits"]) not in routing_index
    ]


def add_node(graph: dict[str, Any], node_id: str, **fields: Any) -> None:
    graph["nodes"][node_id] = {"id": node_id, **fields}


def add_edge(graph: dict[str, Any], left: str, right: str, *, edge_type: str) -> None:
    if left == right:
        return
    a, b = sorted([left, right])
    edge_id = f"{edge_type}::{a}::{b}"
    graph["edges"].setdefault(
        edge_id,
        {"id": edge_id, "nodes": [a, b], "edge_type": edge_type, "edge_weight": 1.0},
    )


def graph_key_label(row: dict[str, str]) -> str:
    seed = row.get("null_seed", "")
    if seed:
        return f"{row['variant']}|{row['source_label']}|seed={seed}"
    return f"{row['variant']}|{row['source_label']}"


def source_key_for(row: dict[str, str]) -> str:
    return graph_key_label(row)


def source_sort_key(source_kind: str, null_model: str, null_seed: str) -> tuple[int, int]:
    if source_kind == "observed":
        return (0, 0)
    if null_model == "markov1":
        return (1, to_int(null_seed))
    if null_model == "matched-lz":
        return (2, to_int(null_seed))
    if null_model == "phase-matched-lz":
        return (3, to_int(null_seed))
    return (9, to_int(null_seed))


def micro_verdict(
    *,
    parent_count: int,
    retained_edges: int,
    defect_fraction: float | None,
    conserved_fraction: float | None,
) -> str:
    if parent_count == 0 or retained_edges == 0:
        return "no-micro-routing-support"
    defect = defect_fraction if defect_fraction is not None else 1.0
    conserved = conserved_fraction if conserved_fraction is not None else 0.0
    if defect <= 0.05 and conserved >= 0.90:
        return "coherent-micro-routing"
    if defect >= 0.35:
        return "defect-heavy-micro-routing"
    return "partial-micro-routing"


def classify_routing(
    *,
    child_retention: float,
    reroute_share: float,
    candidate_total: float,
) -> str:
    if candidate_total <= 0:
        return "destination-collapse"
    if reroute_share >= 0.25:
        return "rerouted"
    if reroute_share <= 0.10:
        if child_retention >= 0.75:
            return "same-destination-conserved"
        if child_retention >= 0.50:
            return "same-destination-partial"
        if child_retention > 0.0:
            return "same-destination-deficit"
        return "destination-collapse"
    return "mixed-destination"


def largest_component_fraction(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> float:
    if not nodes:
        return 0.0
    adjacency = {node["id"]: set() for node in nodes}
    for edge in edges:
        left, right = edge["nodes"]
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)
    seen = set()
    largest = 0
    for node_id in adjacency:
        if node_id in seen:
            continue
        frontier = [node_id]
        component = set()
        while frontier:
            current = frontier.pop()
            if current in component:
                continue
            component.add(current)
            frontier.extend(adjacency[current] - component)
        seen.update(component)
        largest = max(largest, len(component))
    return largest / len(nodes)


def find_band_label(path: Path) -> str | None:
    for part in path.parts:
        if BAND_RE.match(part):
            return part
    return None


def infer_lag_bits(rows: list[dict[str, str]]) -> int | None:
    values = [
        to_int(row.get("candidate_lag_bits"))
        for row in rows
        if row.get("variant") != row.get("anchor_variant") or row.get("source_kind") != "observed"
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def parse_json_map(value: str | None) -> dict[str, float]:
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(key): float(item) for key, item in value.items()}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return {str(key): float(item) for key, item in dict(payload).items()}


def merge_mass_map(target: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        if value <= 0:
            continue
        target[key] = target.get(key, 0.0) + float(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def band_start_key(label: str) -> int:
    return to_int(label.split("-", 1)[0].replace("M", "")) * 1_000_000


def to_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(float(str(value)))


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(float(str(value)))


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(str(value))


def float_or_zero(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(str(value))
