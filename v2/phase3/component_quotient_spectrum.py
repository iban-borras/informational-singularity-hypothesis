from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .defect_spectrum import compute_graph_spectrum


def build_component_quotient_readout(weighted_quotient_run: Path) -> dict[str, Any]:
    """Read N3-05b and normalize the quotient analysis component by component."""
    nodes = load_csv(weighted_quotient_run / "graph_nodes.csv")
    edges = load_csv(weighted_quotient_run / "graph_edges.csv")
    grouped_nodes = group_by_source_channel(nodes)
    grouped_edges = group_by_source_channel(edges)
    component_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for key in sorted(grouped_nodes):
        source_nodes = grouped_nodes[key]
        source_edges = grouped_edges.get(key, [])
        components = connected_components(source_nodes, source_edges)
        source_component_rows = summarize_components(key, components, source_edges)
        component_rows.extend(source_component_rows)
        summary_rows.append(summarize_source(key, source_nodes, source_edges, source_component_rows))

    return {
        "selection": {
            "weighted_quotient_run": str(weighted_quotient_run),
        },
        "notes": [
            "N3-05c is a component-wise normalized readout over the N3-05b weighted prefix/suffix quotient.",
            "It does not introduce Hamming balls, fitted clustering, or a new metric.",
            "It asks whether normalized shape inside the fixed quotient components separates observed B/E from matched-LZ after removing absolute mass scale.",
            "This remains internal and pre-geometric: component normalization is a falsification check, not a final geometry.",
        ],
        "source_summary": summary_rows,
        "component_summary": component_rows,
    }


def load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def group_by_source_channel(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source_key"], row["channel"])].append(row)
    return grouped


def connected_components(
    nodes: list[dict[str, str]],
    edges: list[dict[str, str]],
) -> list[set[str]]:
    adjacency = {node["id"]: set() for node in nodes}
    for edge in edges:
        left, right = parse_edge_nodes(edge)
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)

    components = []
    seen: set[str] = set()
    for node_id in sorted(adjacency):
        if node_id in seen:
            continue
        stack = [node_id]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        seen.update(component)
        components.append(component)
    components.sort(key=lambda item: (-len(item), sorted(item)[0]))
    return components


def summarize_components(
    key: tuple[str, str],
    components: list[set[str]],
    edges: list[dict[str, str]],
) -> list[dict[str, Any]]:
    source_key, channel = key
    edges_by_component = []
    for component in components:
        current_edges = [
            edge
            for edge in edges
            if set(parse_edge_nodes(edge)).issubset(component)
        ]
        edges_by_component.append(current_edges)

    total_weight = sum(edge_weight(edge) for edge in edges)
    rows = []
    for index, (component, component_edges) in enumerate(zip(components, edges_by_component), start=1):
        weight_by_type = defaultdict(float)
        for edge in component_edges:
            weight_by_type[edge["edge_type"]] += edge_weight(edge)
        component_weight = sum(weight_by_type.values())
        row = {
            "source_key": source_key,
            "channel": channel,
            "component_id": index,
            "node_count": len(component),
            "edge_count": len(component_edges),
            "projected_flow_weight": weight_by_type.get("projected-flow", 0.0),
            "family_continuity_weight": weight_by_type.get("family-continuity", 0.0),
            "total_edge_weight": component_weight,
            "weight_share": component_weight / total_weight if total_weight > 0 else 0.0,
            "flow_continuity_ratio": ratio_or_none(
                weight_by_type.get("projected-flow", 0.0),
                weight_by_type.get("family-continuity", 0.0),
            ),
            **component_spectral_shape(component, component_edges),
        }
        rows.append(row)
    return rows


def component_spectral_shape(component: set[str], edges: list[dict[str, str]]) -> dict[str, Any]:
    total_weight = sum(edge_weight(edge) for edge in edges)
    if not component:
        return {
            "lambda2_norm_laplacian": None,
            "mass_normalized_rho": None,
            "degree_entropy_norm": None,
        }
    edge_records = []
    for edge in edges:
        left, right = parse_edge_nodes(edge)
        weight = edge_weight(edge)
        edge_records.append(
            {
                "nodes": [left, right],
                "edge_weight": weight / total_weight if total_weight > 0 else 0.0,
            }
        )
    spectrum = compute_graph_spectrum(
        {
            "node_records": [{"id": node_id} for node_id in sorted(component)],
            "edge_records": edge_records,
        }
    )
    return {
        "lambda2_norm_laplacian": spectrum["lambda2_norm_laplacian"],
        "mass_normalized_rho": spectrum["spectral_radius_adjacency"],
        "degree_entropy_norm": normalized_degree_entropy(component, edges),
    }


def summarize_source(
    key: tuple[str, str],
    nodes: list[dict[str, str]],
    edges: list[dict[str, str]],
    component_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_key, channel = key
    first_node = nodes[0] if nodes else {}
    weights = [float(row["total_edge_weight"]) for row in component_rows]
    positive_rows = [row for row in component_rows if float(row["total_edge_weight"]) > 0.0]
    positive_weights = [float(row["total_edge_weight"]) for row in positive_rows]
    total_weight = sum(positive_weights)
    shares = [weight / total_weight for weight in positive_weights] if total_weight > 0 else []
    edge_type_weights = defaultdict(float)
    for edge in edges:
        edge_type_weights[edge["edge_type"]] += edge_weight(edge)
    weighted_lambda2 = weighted_mean(positive_rows, "lambda2_norm_laplacian")
    weighted_norm_rho = weighted_mean(positive_rows, "mass_normalized_rho")
    weighted_degree_entropy = weighted_mean(positive_rows, "degree_entropy_norm")
    return {
        "source_key": source_key,
        "variant": first_node.get("variant", ""),
        "source_label": first_node.get("source_label", ""),
        "channel": channel,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "component_count": len(component_rows),
        "active_component_count": len(positive_rows),
        "total_edge_weight": total_weight,
        "projected_flow_weight": edge_type_weights.get("projected-flow", 0.0),
        "family_continuity_weight": edge_type_weights.get("family-continuity", 0.0),
        "flow_continuity_ratio": ratio_or_none(
            edge_type_weights.get("projected-flow", 0.0),
            edge_type_weights.get("family-continuity", 0.0),
        ),
        "component_weight_entropy_norm": entropy_norm(shares),
        "effective_weight_components": math.exp(entropy(shares)) if shares else 0.0,
        "top1_component_weight_share": top_k_share(shares, 1),
        "top3_component_weight_share": top_k_share(shares, 3),
        "top5_component_weight_share": top_k_share(shares, 5),
        "weighted_component_lambda2": weighted_lambda2,
        "weighted_mass_normalized_rho": weighted_norm_rho,
        "weighted_degree_entropy_norm": weighted_degree_entropy,
        "verdict": component_verdict(
            edge_count=len(edges),
            component_count=len(component_rows),
            active_count=len(positive_rows),
            entropy_value=entropy_norm(shares),
        ),
    }


def weighted_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = row.get(field)
        weight = float(row.get("weight_share", 0.0))
        if value is None or weight <= 0:
            continue
        numerator += float(value) * weight
        denominator += weight
    return numerator / denominator if denominator > 0 else None


def component_verdict(
    *,
    edge_count: int,
    component_count: int,
    active_count: int,
    entropy_value: float | None,
) -> str:
    if edge_count == 0:
        return "no-component-flow"
    if active_count == component_count and (entropy_value or 0.0) >= 0.85:
        return "distributed-component-flow"
    if (entropy_value or 0.0) < 0.65:
        return "concentrated-component-flow"
    return "partial-component-flow"


def parse_edge_nodes(edge: dict[str, str]) -> tuple[str, str]:
    nodes = json.loads(edge["nodes"])
    return str(nodes[0]), str(nodes[1])


def edge_weight(edge: dict[str, str]) -> float:
    return float(edge.get("edge_weight") or 0.0)


def entropy(values: list[float]) -> float:
    total = sum(values)
    if total <= 0:
        return 0.0
    result = 0.0
    for value in values:
        p = value / total
        if p > 0:
            result -= p * math.log(p)
    return result


def entropy_norm(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return entropy(values) / math.log(len(values))


def top_k_share(values: list[float], k: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values, reverse=True)
    return sum(ordered[:k]) / sum(ordered) if sum(ordered) > 0 else None


def normalized_degree_entropy(component: set[str], edges: list[dict[str, str]]) -> float | None:
    if not component or not edges:
        return None
    degree = Counter()
    for edge in edges:
        left, right = parse_edge_nodes(edge)
        weight = edge_weight(edge)
        degree[left] += weight
        degree[right] += weight
    total = sum(degree.values())
    if total <= 0:
        return None
    result = 0.0
    for node_id in component:
        p = degree[node_id] / total
        if p > 0:
            result -= p * math.log(p)
    return result / math.log(len(component)) if len(component) > 1 else None


def ratio_or_none(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)
