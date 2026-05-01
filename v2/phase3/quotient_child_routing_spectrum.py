from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .defect_spectrum import compute_graph_spectrum
from .micro_child_routing_spectrum import (
    aggregate_routing_rows,
    index_routing_row_files,
    load_transition_cells,
    optional_float,
    parse_json_map,
    source_sort_key,
)


def build_quotient_child_routing_spectrum(
    *,
    transition_run: Path,
    child_routing_run: Path,
    top_patterns: int,
    parent_scale: int = 40,
    edge_weight_mode: str = "unit",
) -> dict[str, Any]:
    """Project child-routing events onto prefix/suffix parent-family quotients."""
    if edge_weight_mode not in {"unit", "pooled-retained-mass"}:
        raise ValueError(f"Unsupported edge_weight_mode: {edge_weight_mode}")
    cells = load_transition_cells(transition_run, top_patterns=top_patterns)
    routing_index = index_routing_row_files(child_routing_run)
    graphs = build_quotient_graphs(
        cells,
        routing_index=routing_index,
        parent_scale=parent_scale,
        edge_weight_mode=edge_weight_mode,
    )
    graph_rows = summarize_graphs(graphs)
    object_name = (
        "N3-05b weighted quotient spectral readout"
        if edge_weight_mode == "pooled-retained-mass"
        else "N3-05 unit quotient spectral readout"
    )
    return {
        "selection": {
            "transition_run": str(transition_run),
            "child_routing_run": str(child_routing_run),
            "top_patterns": top_patterns,
            "parent_scale": parent_scale,
            "edge_weight_mode": edge_weight_mode,
        },
        "notes": [
            f"{object_name} over projected child-routing families.",
            "Child destinations are projected back to parent-scale families by prefix/suffix only.",
            "No Hamming threshold, similarity clustering, or fitted metric is used.",
            "The graph contains retained projected flow only; dropout/reroute remain pressure metrics, not connectivity terminals.",
            "The weighted mode uses pooled retained mass min(anchor_child_mass, candidate_child_mass), split evenly across prefix/suffix projections.",
            "This is pre-geometric and internal: it tests quotient connectivity, not curvature or a final transport law.",
        ],
        "transition_cells": cells,
        "missing_cells": missing_cells(cells, routing_index),
        "graph_summary": graph_rows,
        "graph_nodes": flatten_nodes(graphs),
        "graph_edges": flatten_edges(graphs),
    }


def build_quotient_graphs(
    cells: list[dict[str, Any]],
    *,
    routing_index: dict[tuple[str, int], list[dict[str, str]]],
    parent_scale: int,
    edge_weight_mode: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    graphs: dict[tuple[str, str], dict[str, Any]] = {}
    node_by_source_channel_band_family: dict[tuple[str, str, str, str], str] = {}
    band_order_by_channel: dict[str, list[str]] = defaultdict(list)

    for cell in cells:
        band_order_by_channel[cell["channel"]].append(cell["band"])
        rows = aggregate_routing_rows(
            routing_index.get((cell["band"], cell["best_margin_lag_bits"]), [])
        )
        for row in rows:
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
                    "cell_labels": set(),
                    "parent_count": 0,
                    "retention_sum": 0.0,
                    "deficit_sum": 0.0,
                    "candidate_total_sum": 0.0,
                    "reroute_sum": 0.0,
                    "routing_class_counts": Counter(),
                },
            )
            graph["cell_labels"].add(cell["band"])
            graph["parent_count"] += 1
            graph["retention_sum"] += optional_float(
                row.get("child_destination_retention_pooled")
            ) or 0.0
            graph["deficit_sum"] += float(row.get("anchor_child_deficit_mass_sum", 0.0))
            graph["candidate_total_sum"] += float(row.get("candidate_event_total_sum", 0.0))
            graph["reroute_sum"] += float(row.get("candidate_reroute_mass_sum", 0.0))
            graph["routing_class_counts"][str(row.get("routing_class", ""))] += 1

            parent_family = str(row["pattern"])
            parent_id = quotient_node_id(cell["channel"], cell["band"], parent_family)
            add_node(
                graph,
                parent_id,
                family=parent_family,
                band=cell["band"],
                channel=cell["channel"],
                node_type="family",
                node_class=cell["node_class"],
            )
            add_node_mass(
                graph,
                parent_id,
                parent_mass=float(row.get("anchor_event_total_sum", 0.0)),
            )
            node_by_source_channel_band_family[
                (source_key, cell["channel"], cell["band"], parent_family)
            ] = parent_id

            anchor_children = parse_json_map(row.get("anchor_child_mass_map", {}))
            candidate_children = parse_json_map(row.get("candidate_child_mass_map", {}))
            for child in sorted(set(anchor_children) & set(candidate_children)):
                retained_mass = min(
                    float(anchor_children.get(child, 0.0)),
                    float(candidate_children.get(child, 0.0)),
                )
                if retained_mass <= 0.0:
                    continue
                projections = project_child_family(child, parent_scale)
                if not projections:
                    continue
                edge_weight = (
                    1.0
                    if edge_weight_mode == "unit"
                    else retained_mass / len(projections)
                )
                for projection in projections:
                    target_id = quotient_node_id(cell["channel"], cell["band"], projection)
                    add_node(
                        graph,
                        target_id,
                        family=projection,
                        band=cell["band"],
                        channel=cell["channel"],
                        node_type="family",
                        node_class=cell["node_class"],
                    )
                    add_node_mass(graph, parent_id, flow_mass=edge_weight)
                    add_node_mass(graph, target_id, projected_mass=edge_weight)
                    add_edge(
                        graph,
                        parent_id,
                        target_id,
                        edge_type="projected-flow",
                        edge_weight=edge_weight,
                    )

    for (source_key, channel), graph in graphs.items():
        bands = sorted(set(band_order_by_channel[channel]), key=band_start_key)
        for left, right in zip(bands, bands[1:]):
            left_families = {
                key[3]: node_id
                for key, node_id in node_by_source_channel_band_family.items()
                if key[0] == source_key and key[1] == channel and key[2] == left
            }
            right_families = {
                key[3]: node_id
                for key, node_id in node_by_source_channel_band_family.items()
                if key[0] == source_key and key[1] == channel and key[2] == right
            }
            for family in sorted(set(left_families) & set(right_families)):
                edge_weight = 1.0
                if edge_weight_mode == "pooled-retained-mass":
                    edge_weight = min(
                        graph["nodes"][left_families[family]].get("flow_mass", 0.0),
                        graph["nodes"][right_families[family]].get("flow_mass", 0.0),
                    )
                    if edge_weight <= 0.0:
                        continue
                add_edge(
                    graph,
                    left_families[family],
                    right_families[family],
                    edge_type="family-continuity",
                    edge_weight=edge_weight,
                )
    return graphs


def summarize_graphs(graphs: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for graph in graphs.values():
        nodes = list(graph["nodes"].values())
        edges = list(graph["edges"].values())
        spectrum = compute_graph_spectrum(
            {
                "node_records": [{"id": node["id"]} for node in nodes],
                "edge_records": [
                    {
                        "nodes": edge["nodes"],
                        "edge_weight": float(edge.get("edge_weight", 1.0)),
                    }
                    for edge in edges
                ],
            }
        )
        edge_counts = Counter(edge["edge_type"] for edge in edges)
        edge_weight_sums = defaultdict(float)
        for edge in edges:
            edge_weight_sums[edge["edge_type"]] += float(edge.get("edge_weight", 1.0))
        degree_entropy = normalized_degree_entropy(nodes, edges)
        mean_retention = (
            graph["retention_sum"] / graph["parent_count"]
            if graph["parent_count"]
            else None
        )
        deficit_per_parent = (
            graph["deficit_sum"] / graph["parent_count"]
            if graph["parent_count"]
            else None
        )
        reroute_share = (
            graph["reroute_sum"] / graph["candidate_total_sum"]
            if graph["candidate_total_sum"] > 0
            else None
        )
        row = {
            "source_key": graph["source_key"],
            "variant": graph["variant"],
            "source_label": graph["source_label"],
            "source_kind": graph["source_kind"],
            "null_model": graph["null_model"],
            "null_seed": graph["null_seed"],
            "channel": graph["channel"],
            "cell_count": len(graph["cell_labels"]),
            "parent_count": graph["parent_count"],
            "node_count": len(nodes),
            "edge_count": len(edges),
            "projected_flow_edges": edge_counts.get("projected-flow", 0),
            "family_continuity_edges": edge_counts.get("family-continuity", 0),
            "projected_flow_weight": edge_weight_sums.get("projected-flow", 0.0),
            "family_continuity_weight": edge_weight_sums.get("family-continuity", 0.0),
            "total_edge_weight": sum(edge_weight_sums.values()),
            "flow_continuity_ratio": ratio_or_none(
                edge_weight_sums.get("projected-flow", 0.0),
                edge_weight_sums.get("family-continuity", 0.0),
            ),
            "mean_child_retention": mean_retention,
            "deficit_per_parent": deficit_per_parent,
            "reroute_share": reroute_share,
            "largest_component_fraction": largest_component_fraction(nodes, edges),
            "lambda2_norm_laplacian": spectrum["lambda2_norm_laplacian"],
            "zero_eigenvalue_count": spectrum["zero_eigenvalue_count"],
            "spectral_radius_adjacency": spectrum["spectral_radius_adjacency"],
            "degree_entropy_norm": degree_entropy,
            "routing_class_counts": dict(sorted(graph["routing_class_counts"].items())),
            "verdict": quotient_verdict(
                edge_count=len(edges),
                zero_count=spectrum["zero_eigenvalue_count"],
                mean_retention=mean_retention,
                largest_component_fraction=largest_component_fraction(nodes, edges),
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


def flatten_nodes(graphs: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
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
    return sorted(rows, key=lambda row: (row["channel"], row["source_key"], row["id"]))


def flatten_edges(graphs: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
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
    return sorted(rows, key=lambda row: (row["channel"], row["source_key"], row["id"]))


def project_child_family(child: str, parent_scale: int) -> list[str]:
    if len(child) < parent_scale:
        return []
    projections = {child[:parent_scale], child[-parent_scale:]}
    return sorted(projections)


def quotient_node_id(channel: str, band: str, family: str) -> str:
    return f"Q::{channel}::{band}::{family}"


def add_node(graph: dict[str, Any], node_id: str, **fields: Any) -> None:
    current = graph["nodes"].setdefault(node_id, {"id": node_id, **fields})
    for key, value in fields.items():
        current.setdefault(key, value)


def add_node_mass(
    graph: dict[str, Any],
    node_id: str,
    *,
    parent_mass: float = 0.0,
    projected_mass: float = 0.0,
    flow_mass: float = 0.0,
) -> None:
    node = graph["nodes"][node_id]
    node["parent_mass"] = float(node.get("parent_mass", 0.0)) + float(parent_mass)
    node["projected_mass"] = float(node.get("projected_mass", 0.0)) + float(projected_mass)
    node["flow_mass"] = float(node.get("flow_mass", 0.0)) + float(flow_mass)


def add_edge(
    graph: dict[str, Any],
    left: str,
    right: str,
    *,
    edge_type: str,
    edge_weight: float,
) -> None:
    if left == right:
        return
    a, b = sorted([left, right])
    edge_id = f"{edge_type}::{a}::{b}"
    current = graph["edges"].setdefault(
        edge_id,
        {"id": edge_id, "nodes": [a, b], "edge_type": edge_type, "edge_weight": 0.0},
    )
    current["edge_weight"] = float(current.get("edge_weight", 0.0)) + float(edge_weight)


def normalized_degree_entropy(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> float | None:
    if not nodes or not edges:
        return None
    degree = Counter()
    for edge in edges:
        left, right = edge["nodes"]
        degree[left] += 1
        degree[right] += 1
    total = sum(degree.values())
    if total <= 0:
        return None
    entropy = 0.0
    for node in nodes:
        p = degree[node["id"]] / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy / math.log(len(nodes)) if len(nodes) > 1 else None


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


def quotient_verdict(
    *,
    edge_count: int,
    zero_count: int,
    mean_retention: float | None,
    largest_component_fraction: float,
) -> str:
    if edge_count == 0:
        return "no-quotient-flow"
    retention = mean_retention if mean_retention is not None else 0.0
    if retention >= 0.75 and largest_component_fraction >= 0.25:
        return "structured-quotient-flow"
    if retention < 0.50:
        return "diffuse-or-deficit-quotient-flow"
    if zero_count > 1:
        return "fragmented-quotient-flow"
    return "partial-quotient-flow"


def missing_cells(
    cells: list[dict[str, Any]],
    routing_index: dict[tuple[str, int], list[dict[str, str]]],
) -> list[dict[str, Any]]:
    return [
        cell
        for cell in cells
        if (cell["band"], cell["best_margin_lag_bits"]) not in routing_index
    ]


def source_key_for(row: dict[str, Any]) -> str:
    seed = row.get("null_seed", "")
    if seed:
        return f"{row['variant']}|{row['source_label']}|seed={seed}"
    return f"{row['variant']}|{row['source_label']}"


def ratio_or_none(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def band_start_key(label: str) -> int:
    return int(label.split("-", 1)[0].replace("M", "")) * 1_000_000
