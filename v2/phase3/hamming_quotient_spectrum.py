from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .defect_spectrum import compute_graph_spectrum
from .quotient_child_routing_spectrum import largest_component_fraction, normalized_degree_entropy


def build_hamming_quotient_readout(
    weighted_quotient_run: Path,
    *,
    radii: list[int],
) -> dict[str, Any]:
    nodes = load_csv(weighted_quotient_run / "graph_nodes.csv")
    edges = load_csv(weighted_quotient_run / "graph_edges.csv")
    grouped_nodes = group_by_source_channel(nodes)
    grouped_edges = group_by_source_channel(edges)

    rows = []
    node_rows = []
    edge_rows = []
    for radius in radii:
        if radius < 0:
            raise ValueError(f"Hamming radius must be non-negative: {radius}")
        for key in sorted(grouped_nodes):
            source_nodes = grouped_nodes[key]
            source_edges = grouped_edges.get(key, [])
            graph = build_hamming_graph(key, source_nodes, source_edges, radius=radius)
            rows.append(summarize_hamming_graph(graph))
            node_rows.extend(flatten_nodes(graph))
            edge_rows.extend(flatten_edges(graph))

    return {
        "selection": {
            "weighted_quotient_run": str(weighted_quotient_run),
            "radii": radii,
        },
        "notes": [
            "N3-06 is a Hamming quotient over the existing N3-05b weighted prefix/suffix graph.",
            "Families whose Hamming distance is <= radius are identified before aggregating the existing retained-flow edges.",
            "No new transport edge is invented: the quotient only merges nearby family labels, then sums inherited edge weights.",
            "The default radius sweep is predeclared and should be read as sensitivity, not tuned evidence.",
            "This is an internal pre-geometric test, not curvature or a final geometry.",
        ],
        "graph_summary": rows,
        "graph_nodes": node_rows,
        "graph_edges": edge_rows,
    }


def build_hamming_graph(
    key: tuple[str, str],
    nodes: list[dict[str, str]],
    edges: list[dict[str, str]],
    *,
    radius: int,
) -> dict[str, Any]:
    source_key, channel = key
    source_label = nodes[0].get("source_label", "") if nodes else ""
    variant = nodes[0].get("variant", "") if nodes else ""
    families = sorted({node["family"] for node in nodes})
    family_to_cluster = hamming_clusters(families, radius=radius)
    original_node_by_id = {node["id"]: node for node in nodes}
    quotient_nodes: dict[str, dict[str, Any]] = {}
    original_to_quotient: dict[str, str] = {}

    for node in nodes:
        cluster = family_to_cluster[node["family"]]
        quotient_id = f"HQ::r{radius}::{channel}::{node['band']}::{cluster}"
        original_to_quotient[node["id"]] = quotient_id
        current = quotient_nodes.setdefault(
            quotient_id,
            {
                "id": quotient_id,
                "source_key": source_key,
                "source_label": source_label,
                "variant": variant,
                "channel": channel,
                "radius": radius,
                "band": node["band"],
                "cluster": cluster,
                "family_count": 0,
                "parent_mass": 0.0,
                "projected_mass": 0.0,
                "flow_mass": 0.0,
            },
        )
        current["family_count"] += 1
        current["parent_mass"] += float(node.get("parent_mass") or 0.0)
        current["projected_mass"] += float(node.get("projected_mass") or 0.0)
        current["flow_mass"] += float(node.get("flow_mass") or 0.0)

    quotient_edges: dict[str, dict[str, Any]] = {}
    collapsed_self_weight = 0.0
    collapsed_self_edges = 0
    inherited_edge_count = 0
    for edge in edges:
        left, right = parse_edge_nodes(edge)
        if left not in original_to_quotient or right not in original_to_quotient:
            continue
        q_left = original_to_quotient[left]
        q_right = original_to_quotient[right]
        weight = float(edge.get("edge_weight") or 0.0)
        inherited_edge_count += 1
        if q_left == q_right:
            collapsed_self_weight += weight
            collapsed_self_edges += 1
            continue
        a, b = sorted([q_left, q_right])
        edge_id = f"{edge['edge_type']}::{a}::{b}"
        current = quotient_edges.setdefault(
            edge_id,
            {
                "id": edge_id,
                "source_key": source_key,
                "source_label": source_label,
                "variant": variant,
                "channel": channel,
                "radius": radius,
                "edge_type": edge["edge_type"],
                "nodes": [a, b],
                "edge_weight": 0.0,
            },
        )
        current["edge_weight"] += weight

    return {
        "source_key": source_key,
        "source_label": source_label,
        "variant": variant,
        "channel": channel,
        "radius": radius,
        "family_count": len(families),
        "hamming_cluster_count": len(set(family_to_cluster.values())),
        "nodes": quotient_nodes,
        "edges": quotient_edges,
        "inherited_edge_count": inherited_edge_count,
        "collapsed_self_edges": collapsed_self_edges,
        "collapsed_self_weight": collapsed_self_weight,
    }


def summarize_hamming_graph(graph: dict[str, Any]) -> dict[str, Any]:
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
    edge_weight_by_type = defaultdict(float)
    for edge in edges:
        edge_weight_by_type[edge["edge_type"]] += float(edge.get("edge_weight", 0.0))
    total_edge_weight = sum(edge_weight_by_type.values())
    return {
        "variant": graph["variant"],
        "source_label": graph["source_label"],
        "source_key": graph["source_key"],
        "channel": graph["channel"],
        "radius": graph["radius"],
        "family_count": graph["family_count"],
        "hamming_cluster_count": graph["hamming_cluster_count"],
        "node_count": len(nodes),
        "edge_count": len(edges),
        "inherited_edge_count": graph["inherited_edge_count"],
        "collapsed_self_edges": graph["collapsed_self_edges"],
        "collapsed_self_weight": graph["collapsed_self_weight"],
        "projected_flow_weight": edge_weight_by_type.get("projected-flow", 0.0),
        "family_continuity_weight": edge_weight_by_type.get("family-continuity", 0.0),
        "total_edge_weight": total_edge_weight,
        "largest_component_fraction": largest_component_fraction(nodes, edges),
        "lambda2_norm_laplacian": spectrum["lambda2_norm_laplacian"],
        "zero_eigenvalue_count": spectrum["zero_eigenvalue_count"],
        "spectral_radius_adjacency": spectrum["spectral_radius_adjacency"],
        "degree_entropy_norm": normalized_degree_entropy(nodes, edges),
        "verdict": hamming_verdict(edge_count=len(edges), zero_count=spectrum["zero_eigenvalue_count"]),
    }


def hamming_clusters(families: list[str], *, radius: int) -> dict[str, str]:
    parent = {family: family for family in families}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if root_right < root_left:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left

    for index, left in enumerate(families):
        for right in families[index + 1 :]:
            if hamming_distance(left, right) <= radius:
                union(left, right)

    clusters = defaultdict(list)
    for family in families:
        clusters[find(family)].append(family)

    root_to_label = {}
    for cluster_index, root in enumerate(sorted(clusters), start=1):
        root_to_label[root] = f"C{cluster_index:04d}_{clusters[root][0]}"
    return {family: root_to_label[find(family)] for family in families}


def flatten_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(graph["nodes"].values(), key=lambda row: row["id"])


def flatten_edges(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(graph["edges"].values(), key=lambda row: row["id"])


def load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def group_by_source_channel(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source_key"], row["channel"])].append(row)
    return grouped


def parse_edge_nodes(edge: dict[str, str]) -> tuple[str, str]:
    nodes = json.loads(edge["nodes"])
    return str(nodes[0]), str(nodes[1])


def hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right))


def hamming_verdict(*, edge_count: int, zero_count: int) -> str:
    if edge_count == 0:
        return "no-hamming-quotient-flow"
    if zero_count == 1:
        return "connected-hamming-quotient"
    if zero_count < 24:
        return "partially-fused-hamming-quotient"
    return "fragmented-hamming-quotient"
