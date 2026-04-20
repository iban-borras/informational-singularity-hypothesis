from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

from .defects import (
    _infer_segment_count,
    _pairwise_jaccard_mean,
    _require_edge,
    _safe_ratio,
    _weight_entropy,
    classify_variant_role,
    load_policy_fibers,
)
from utils.progress import ProgressIndicator

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def build_sparse_graph_rows(
    runs: list[dict],
    *,
    window_scales: tuple[int, int, int, int],
    left_policy: str,
    right_policy: str,
    min_segment_recurrence: int,
    top_hub_k: int,
    show_progress: bool = False,
) -> list[dict]:
    rows = []
    for run in runs:
        config = run["dataset"]["config"]
        run_dir = _run_dir(run)
        left_fibers = load_policy_fibers(run_dir, left_policy)
        right_fibers = load_policy_fibers(run_dir, right_policy)
        graph_result = compute_sparse_graph_result(
            left_fibers,
            right_fibers,
            window_scales=window_scales,
            left_policy=left_policy,
            right_policy=right_policy,
            min_segment_recurrence=min_segment_recurrence,
            top_hub_k=top_hub_k,
            show_progress=show_progress,
            progress_label=config["variant"],
        )
        row = {
            "variant": config["variant"],
            "variant_role": classify_variant_role(config["variant"]),
            "iteration": config.get("iteration"),
            "segment_bits": config.get("segment_bits"),
            "num_segments": config.get("num_segments"),
            "segment_offset_bits": config.get("segment_offset_bits", 0),
            "scales": list(config.get("scales", [])),
            "phase1_policies": list(config.get("policies", [])),
            "window_scales": list(window_scales),
            "left_policy": left_policy,
            "right_policy": right_policy,
            "min_segment_recurrence": min_segment_recurrence,
            "top_hub_k": top_hub_k,
            "run_dir": str(run_dir),
            "dataset_path": run["_dataset_path"],
            "summary_path": run["_summary_path"],
            "generated_at": run["generated_at"],
            "graph_result": graph_result,
            "node_count": graph_result["node_count"],
            "edge_count": graph_result["edge_count"],
            "largest_component_fraction": graph_result["largest_component_fraction"],
            "recurrent_edge_fraction": graph_result["recurrent_edge_fraction"],
            "recurrent_edge_mass_fraction": graph_result["recurrent_edge_mass_fraction"],
            "core_node_fraction": graph_result["core_node_fraction"],
            "degree_entropy": graph_result["degree_entropy"],
            "hub_mass_fraction": graph_result["hub_mass_fraction"],
            "segment_core_jaccard": graph_result["segment_core_jaccard"],
            "null_conditioned_edge_rarity": graph_result["null_conditioned_edge_rarity"],
        }
        rows.append(row)
    return rows


def compute_sparse_graph_result(
    left_fibers: list[dict],
    right_fibers: list[dict],
    *,
    window_scales: tuple[int, int, int, int],
    left_policy: str,
    right_policy: str,
    min_segment_recurrence: int,
    top_hub_k: int,
    show_progress: bool = False,
    progress_label: str | None = None,
) -> dict:
    base_scale, parent_scale, bridge_scale, high_scale = window_scales
    if not (base_scale < parent_scale < bridge_scale < high_scale):
        raise ValueError("Sparse graph pilot requires four strictly ascending scales.")

    left_low = _require_edge(left_fibers, target_m=parent_scale, source_m=bridge_scale)
    right_low = _require_edge(right_fibers, target_m=parent_scale, source_m=bridge_scale)
    left_high = _require_edge(left_fibers, target_m=bridge_scale, source_m=high_scale)
    right_high = _require_edge(right_fibers, target_m=bridge_scale, source_m=high_scale)

    left_high_index = {item["pattern"]: item for item in left_high.get("parents", [])}
    right_high_index = {item["pattern"]: item for item in right_high.get("parents", [])}
    bridge_items = _build_bridge_items(left_low, right_low, base_scale=base_scale)
    segment_count = min(_infer_segment_count(left_low), _infer_segment_count(right_low))

    nodes = []
    segment_node_maps = [dict() for _ in range(segment_count)]
    node_iter = _wrap_progress(
        bridge_items,
        total=len(bridge_items),
        show_progress=show_progress,
        desc=_progress_desc("Bridge nodes", progress_label),
        unit="bridge",
    )
    for bridge in node_iter:
        bridge_pattern = bridge["bridge_pattern"]
        if not _bridge_future_bearing(bridge_pattern, left_high_index, right_high_index, segment_index=None):
            continue

        left_dominant = _dominant_descendant(left_high_index.get(bridge_pattern), segment_index=None)
        right_dominant = _dominant_descendant(right_high_index.get(bridge_pattern), segment_index=None)
        dominant_descendants = sorted({pattern for pattern in (left_dominant, right_dominant) if pattern is not None})
        segment_presence = []
        segment_descendants = []
        for segment_index in range(segment_count):
            active = _bridge_future_bearing(
                bridge_pattern,
                left_high_index,
                right_high_index,
                segment_index=segment_index,
            ) and _bridge_weight(bridge, segment_index) > 0
            segment_presence.append(active)
            left_segment_desc = _dominant_descendant(left_high_index.get(bridge_pattern), segment_index=segment_index)
            right_segment_desc = _dominant_descendant(right_high_index.get(bridge_pattern), segment_index=segment_index)
            descendants = sorted({pattern for pattern in (left_segment_desc, right_segment_desc) if pattern is not None})
            segment_descendants.append(descendants)
            if active:
                segment_node_maps[segment_index][bridge_pattern] = {
                    "left_parent": bridge["left_parent"],
                    "right_parent": bridge["right_parent"],
                    "dominant_descendants": descendants,
                }

        nodes.append(
            {
                "id": bridge_pattern,
                "bridge_pattern": bridge_pattern,
                "left_parent": bridge["left_parent"],
                "right_parent": bridge["right_parent"],
                "node_weight": _bridge_weight(bridge, segment_index=None),
                "raw_count": int(bridge.get("raw_count", 0)),
                "dominant_descendants": dominant_descendants,
                "segment_counts": list(bridge.get("segment_counts", [])),
                "segment_presence": segment_presence,
                "segment_descendants": segment_descendants,
            }
        )

    _close_progress(node_iter)

    edge_records = []
    segment_edge_sets = [set() for _ in range(segment_count)]
    total_edge_weight = 0.0
    weighted_degrees: dict[str, float] = {node["id"]: 0.0 for node in nodes}

    total_pairs = (len(nodes) * (len(nodes) - 1)) // 2
    edge_iter = _wrap_progress(
        combinations(nodes, 2),
        total=total_pairs,
        show_progress=show_progress,
        desc=_progress_desc("Bridge edges", progress_label),
        unit="pair",
    )
    for left_node, right_node in edge_iter:
        edge = _build_edge_record(left_node, right_node, segment_node_maps=segment_node_maps)
        if edge is None:
            continue
        edge_records.append(edge)
        total_edge_weight += edge["edge_weight"]
        weighted_degrees[left_node["id"]] += edge["edge_weight"]
        weighted_degrees[right_node["id"]] += edge["edge_weight"]
        for segment_index in edge["segment_indices"]:
            segment_edge_sets[segment_index].add(edge["id"])
    _close_progress(edge_iter)

    recurrent_edge_ids = {
        edge["id"]
        for edge in edge_records
        if edge["segment_recurrence_count"] >= min_segment_recurrence
    }
    recurrent_edge_weight = sum(edge["edge_weight"] for edge in edge_records if edge["id"] in recurrent_edge_ids)
    recurrent_nodes = {
        node_id
        for edge in edge_records
        if edge["id"] in recurrent_edge_ids
        for node_id in edge["nodes"]
    }
    segment_core_sets = [edge_set & recurrent_edge_ids for edge_set in segment_edge_sets]

    top_edges = sorted(
        edge_records,
        key=lambda item: (-item["edge_weight"], -item["segment_recurrence_count"], item["id"]),
    )[:10]
    top_nodes = [
        {"node": node_id, "weighted_degree": degree}
        for node_id, degree in sorted(weighted_degrees.items(), key=lambda item: (-item[1], item[0]))[: max(1, top_hub_k)]
    ]
    degree_values = [value for value in weighted_degrees.values() if value > 0]
    total_degree_mass = sum(weighted_degrees.values())
    hub_mass = sum(item["weighted_degree"] for item in top_nodes)

    return {
        "window_scales": [base_scale, parent_scale, bridge_scale, high_scale],
        "left_policy": left_policy,
        "right_policy": right_policy,
        "min_segment_recurrence": min_segment_recurrence,
        "top_hub_k": top_hub_k,
        "node_count": len(nodes),
        "edge_count": len(edge_records),
        "largest_component_fraction": _largest_component_fraction(nodes, edge_records),
        "recurrent_edge_fraction": _safe_ratio(len(recurrent_edge_ids), len(edge_records)),
        "recurrent_edge_mass_fraction": _safe_ratio(recurrent_edge_weight, total_edge_weight),
        "core_node_fraction": _safe_ratio(len(recurrent_nodes), len(nodes)),
        "degree_entropy": _weight_entropy(degree_values),
        "hub_mass_fraction": _safe_ratio(hub_mass, total_degree_mass),
        "segment_core_jaccard": _pairwise_jaccard_mean(segment_core_sets),
        "null_conditioned_edge_rarity": None,
        "node_records": nodes,
        "edge_records": edge_records,
        "top_edges": top_edges,
        "top_nodes": top_nodes,
        "segment_results": [
            {
                "segment_index": segment_index,
                "active_node_count": len(segment_node_maps[segment_index]),
                "edge_count": len(segment_edge_sets[segment_index]),
                "core_edge_count": len(segment_core_sets[segment_index]),
            }
            for segment_index in range(segment_count)
        ],
        "notes": [
            "node = future-bearing shared bridge at m=28 present under both policies and still transportable to m=32",
            "edge = sparse reuse relation through shared left/right parent at m=24 or shared dominant descendant at m=32",
            "edge weight = shared-corner count + shared-dominant-descendant count + segment recurrence count",
            "largest_component_fraction is computed on node count, not node mass, in this first executable skeleton",
            "null_conditioned_edge_rarity is intentionally left unset here; it belongs to a later observed-vs-null comparison layer",
        ],
    }


def render_console_summary(rows: list[dict]) -> str:
    lines = [
        "Phase 2 sparse multiscale graph pilot",
        "-" * 122,
        f"{'variant':<14}{'iter':>6}{'nodes':>8}{'edges':>8}{'lcc':>10}{'rec_e':>10}{'rec_m':>10}{'core_n':>10}{'deg_H':>10}{'seg_J':>10}",
    ]
    for row in rows:
        lines.append(
            f"{row['variant']:<14}"
            f"{_fmt_int(row['iteration']):>6}"
            f"{row.get('node_count', 0):>8}"
            f"{row.get('edge_count', 0):>8}"
            f"{_fmt(row.get('largest_component_fraction')):>10}"
            f"{_fmt(row.get('recurrent_edge_fraction')):>10}"
            f"{_fmt(row.get('recurrent_edge_mass_fraction')):>10}"
            f"{_fmt(row.get('core_node_fraction')):>10}"
            f"{_fmt(row.get('degree_entropy')):>10}"
            f"{_fmt(row.get('segment_core_jaccard')):>10}"
        )
    return "\n".join(lines)


def render_markdown_report(rows: list[dict], selection: dict) -> str:
    lines = [
        "# Phase 2 Sparse Multiscale Graph Pilot",
        "",
        "## Selection",
        "",
        f"- Variants: {', '.join(selection['variants'])}",
        f"- Segment protocol: {selection['num_segments']} x {selection['segment_bits']}",
        f"- Segment offset bits: {selection['segment_offset_bits']}",
        f"- Phase 1 scales: {', '.join(str(value) for value in selection['phase1_scales'])}",
        f"- Graph window: {', '.join(str(value) for value in selection['window_scales'])}",
        f"- Left policy: {selection['left_policy']}",
        f"- Right policy: {selection['right_policy']}",
        f"- Min segment recurrence: {selection['min_segment_recurrence']}",
        f"- Top hub k: {selection['top_hub_k']}",
        "",
        "## Compact Readout",
        "",
        "| Variant | Iter | Nodes | Edges | Largest component frac | Recurrent edge frac | Recurrent edge mass frac | Core node frac | Degree entropy | Segment core Jaccard |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['iteration']} | {row.get('node_count', 0)} | {row.get('edge_count', 0)} | "
            f"{_fmt(row.get('largest_component_fraction'))} | {_fmt(row.get('recurrent_edge_fraction'))} | "
            f"{_fmt(row.get('recurrent_edge_mass_fraction'))} | {_fmt(row.get('core_node_fraction'))} | "
            f"{_fmt(row.get('degree_entropy'))} | {_fmt(row.get('segment_core_jaccard'))} |"
        )

    for row in rows:
        result = row["graph_result"]
        lines.extend(
            [
                "",
                f"## {row['variant']}",
                "",
                f"- Node count: {result['node_count']}",
                f"- Edge count: {result['edge_count']}",
                f"- Largest component fraction: {_fmt(result['largest_component_fraction'])}",
                f"- Recurrent edge fraction: {_fmt(result['recurrent_edge_fraction'])}",
                f"- Recurrent edge mass fraction: {_fmt(result['recurrent_edge_mass_fraction'])}",
                f"- Core node fraction: {_fmt(result['core_node_fraction'])}",
                f"- Degree entropy: {_fmt(result['degree_entropy'])}",
                f"- Hub mass fraction: {_fmt(result['hub_mass_fraction'])}",
                f"- Segment core Jaccard: {_fmt(result['segment_core_jaccard'])}",
                "",
                "Top nodes:",
            ]
        )
        if result["top_nodes"]:
            for node in result["top_nodes"]:
                lines.append(f"- `{node['node']}` | weighted_degree={_fmt(node['weighted_degree'])}")
        else:
            lines.append("- none")
        lines.append("")
        lines.append("Top edges:")
        if result["top_edges"]:
            for edge in result["top_edges"][:5]:
                relation_labels = ", ".join(edge["relation_labels"])
                lines.append(
                    f"- `{edge['nodes'][0]} <-> {edge['nodes'][1]}` | weight={_fmt(edge['edge_weight'])} | "
                    f"recurrence={edge['segment_recurrence_count']} | relations={relation_labels}"
                )
        else:
            lines.append("- none")
    return "\n".join(lines)


def _build_edge_record(left_node: dict, right_node: dict, *, segment_node_maps: list[dict]) -> dict | None:
    shared_left = left_node["left_parent"] == right_node["left_parent"]
    shared_right = left_node["right_parent"] == right_node["right_parent"]
    shared_descendants = sorted(set(left_node["dominant_descendants"]) & set(right_node["dominant_descendants"]))
    if not shared_left and not shared_right and not shared_descendants:
        return None

    segment_indices = []
    for segment_index, segment_nodes in enumerate(segment_node_maps):
        left_segment = segment_nodes.get(left_node["id"])
        right_segment = segment_nodes.get(right_node["id"])
        if left_segment is None or right_segment is None:
            continue
        shared_segment_descendants = set(left_segment["dominant_descendants"]) & set(right_segment["dominant_descendants"])
        if (
            left_segment["left_parent"] == right_segment["left_parent"]
            or left_segment["right_parent"] == right_segment["right_parent"]
            or shared_segment_descendants
        ):
            segment_indices.append(segment_index)

    relation_labels = []
    if shared_left:
        relation_labels.append("shared_left_parent")
    if shared_right:
        relation_labels.append("shared_right_parent")
    if shared_descendants:
        relation_labels.append("shared_dominant_descendant")
    relation_score = int(shared_left) + int(shared_right) + len(shared_descendants)
    edge_nodes = tuple(sorted((left_node["id"], right_node["id"])))
    return {
        "id": f"{edge_nodes[0]}||{edge_nodes[1]}",
        "nodes": [edge_nodes[0], edge_nodes[1]],
        "relation_labels": relation_labels,
        "shared_dominant_descendants": shared_descendants,
        "segment_indices": segment_indices,
        "segment_recurrence_count": len(segment_indices),
        "edge_weight": float(relation_score + len(segment_indices)),
    }


def _build_bridge_items(left_low: dict, right_low: dict, *, base_scale: int) -> list[dict]:
    left_child_to_parent = {}
    for parent in left_low.get("parents", []):
        for child in parent.get("children", []):
            left_child_to_parent[child["pattern"]] = parent
    right_child_to_parent = {}
    for parent in right_low.get("parents", []):
        for child in parent.get("children", []):
            right_child_to_parent[child["pattern"]] = parent

    bridge_patterns = sorted(set(left_child_to_parent) & set(right_child_to_parent))
    items = []
    for bridge_pattern in bridge_patterns:
        left_parent = left_child_to_parent[bridge_pattern]
        right_parent = right_child_to_parent[bridge_pattern]
        if left_parent["pattern"][-base_scale:] != right_parent["pattern"][:base_scale]:
            continue
        child_meta = _child_meta(left_parent, bridge_pattern)
        items.append(
            {
                "bridge_pattern": bridge_pattern,
                "left_parent": left_parent["pattern"],
                "right_parent": right_parent["pattern"],
                "raw_count": child_meta.get("raw_count", 0),
                "normalized_frequency": child_meta.get("normalized_frequency", 0.0),
                "segment_counts": child_meta.get("segment_counts", []),
            }
        )
    return items


def _bridge_future_bearing(
    bridge_pattern: str,
    left_high_index: dict[str, dict],
    right_high_index: dict[str, dict],
    *,
    segment_index: int | None,
) -> bool:
    return _has_positive_child(left_high_index.get(bridge_pattern), segment_index) and _has_positive_child(
        right_high_index.get(bridge_pattern),
        segment_index,
    )


def _has_positive_child(parent: dict | None, segment_index: int | None) -> bool:
    if parent is None:
        return False
    return any(_child_weight(child, segment_index) > 0 for child in parent.get("children", []))


def _dominant_descendant(parent: dict | None, *, segment_index: int | None) -> str | None:
    if parent is None:
        return None
    best_pattern = None
    best_score = 0.0
    best_count = 0
    for child in parent.get("children", []):
        score = _child_weight(child, segment_index)
        raw_count = int(child.get("raw_count", 0))
        pattern = child["pattern"]
        if score > best_score or (score == best_score and raw_count > best_count) or (
            score == best_score and raw_count == best_count and best_pattern is not None and pattern < best_pattern
        ):
            best_pattern = pattern
            best_score = score
            best_count = raw_count
    return best_pattern if best_score > 0 else None


def _largest_component_fraction(nodes: list[dict], edge_records: list[dict]) -> float:
    node_ids = [node["id"] for node in nodes]
    if not node_ids:
        return 0.0
    adjacency = {node_id: set() for node_id in node_ids}
    for edge in edge_records:
        left, right = edge["nodes"]
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen = set()
    largest = 0
    for node_id in node_ids:
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
        if len(component) > largest:
            largest = len(component)
    return _safe_ratio(largest, len(node_ids))


def _child_meta(parent: dict, bridge_pattern: str) -> dict:
    for child in parent.get("children", []):
        if child["pattern"] == bridge_pattern:
            return child
    raise KeyError(bridge_pattern)


def _child_weight(child: dict, segment_index: int | None) -> float:
    if segment_index is None:
        return float(child.get("normalized_frequency", 0.0))
    counts = child.get("segment_counts", [])
    if segment_index >= len(counts):
        return 0.0
    return float(counts[segment_index])


def _bridge_weight(bridge: dict, segment_index: int | None) -> float:
    if segment_index is None:
        return float(bridge.get("normalized_frequency", 0.0))
    counts = bridge.get("segment_counts", [])
    if segment_index >= len(counts):
        return 0.0
    return float(counts[segment_index])


def _run_dir(run: dict) -> Path:
    return Path(run["_run_dir"])


def _wrap_progress(iterable, *, total: int, show_progress: bool, desc: str, unit: str):
    if not show_progress or total <= 0:
        return iterable
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit=unit, leave=False, file=sys.stdout)
    progress = ProgressIndicator(desc, total=total)
    progress.__enter__()

    def _generator():
        try:
            for index, item in enumerate(iterable, start=1):
                yield item
                progress.update(index)
        finally:
            progress.__exit__(None, None, None)

    return _generator()


def _close_progress(progress_iter) -> None:
    close = getattr(progress_iter, "close", None)
    if callable(close):
        close()


def _progress_desc(base: str, label: str | None) -> str:
    return base if not label else f"{base} {label}"


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "-"
    return str(value)
