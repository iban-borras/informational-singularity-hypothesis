from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from v2.phase1.tower import rolling_codes_uint64_max64
from v2.phase2.defects import _require_edge, _safe_ratio, _weight_entropy, load_policy_fibers
from v2.phase2.square import _build_bridge_items, _normalized_hamming
from v2.phase3.defect_spectrum import compute_graph_spectrum


DROP_NODE = "__dropout__"


def compute_lag_conditioned_defect_result(
    *,
    anchor_run_dir: Path | str,
    candidate_run_dir: Path | str,
    anchor_window_bits: np.ndarray,
    candidate_window_bits: np.ndarray,
    window_scales: tuple[int, int, int, int],
    left_policy: str,
    right_policy: str,
    top_bridges: int,
) -> dict[str, Any]:
    base_scale, parent_scale, bridge_scale, high_scale = window_scales
    if not (base_scale < parent_scale < bridge_scale < high_scale):
        raise ValueError("Lag-conditioned defect requires four strictly ascending scales.")
    if bridge_scale > 64 or high_scale > 64:
        raise ValueError("Lag-conditioned pilot currently supports scales up to 64.")
    if anchor_window_bits.size < high_scale or candidate_window_bits.size < high_scale:
        raise ValueError("Window bits are shorter than the requested high scale.")
    if top_bridges < 1:
        raise ValueError("top_bridges must be positive.")

    anchor_path = Path(anchor_run_dir).resolve()
    candidate_path = Path(candidate_run_dir).resolve()

    anchor_left = load_policy_fibers(anchor_path, left_policy)
    anchor_right = load_policy_fibers(anchor_path, right_policy)
    candidate_left = load_policy_fibers(candidate_path, left_policy)
    candidate_right = load_policy_fibers(candidate_path, right_policy)

    anchor_left_low = _require_edge(anchor_left, target_m=parent_scale, source_m=bridge_scale)
    anchor_right_low = _require_edge(anchor_right, target_m=parent_scale, source_m=bridge_scale)
    anchor_left_high = _require_edge(anchor_left, target_m=bridge_scale, source_m=high_scale)
    anchor_right_high = _require_edge(anchor_right, target_m=bridge_scale, source_m=high_scale)
    candidate_left_low = _require_edge(candidate_left, target_m=parent_scale, source_m=bridge_scale)
    candidate_right_low = _require_edge(candidate_right, target_m=parent_scale, source_m=bridge_scale)
    candidate_left_high = _require_edge(candidate_left, target_m=bridge_scale, source_m=high_scale)
    candidate_right_high = _require_edge(candidate_right, target_m=bridge_scale, source_m=high_scale)

    bridge_items = _build_bridge_items(anchor_left_low, anchor_right_low, base_scale=base_scale)
    anchor_bridge_counts = _window_code_count_map(anchor_window_bits, bridge_scale)
    candidate_bridge_counts = _window_code_count_map(candidate_window_bits, bridge_scale)
    anchor_high_counts = _window_code_count_map(anchor_window_bits, high_scale)
    candidate_high_counts = _window_code_count_map(candidate_window_bits, high_scale)

    candidate_left_parent_index = _index_parents(candidate_left_low)
    candidate_right_parent_index = _index_parents(candidate_right_low)
    anchor_left_high_index = _index_parents(anchor_left_high)
    anchor_right_high_index = _index_parents(anchor_right_high)
    candidate_left_high_index = _index_parents(candidate_left_high)
    candidate_right_high_index = _index_parents(candidate_right_high)

    monitored = _select_active_anchor_bridges(
        bridge_items,
        count_map=anchor_bridge_counts,
        pattern_scale=bridge_scale,
        top_bridges=top_bridges,
    )

    details: list[dict[str, Any]] = []
    future_weight = 0.0
    defect_weight = 0.0
    dropout_weight = 0.0
    anchored_weight = 0.0
    disagreement_weight = 0.0
    family_weights: dict[str, float] = {}
    distances = []

    for bridge in monitored:
        actual = bridge["bridge_pattern"]
        weight = float(bridge["anchor_local_count"])
        anchor_future = _future_bearing_local(
            actual,
            anchor_left_high_index,
            anchor_right_high_index,
            anchor_high_counts,
        )
        candidate_future = _future_bearing_local(
            actual,
            candidate_left_high_index,
            candidate_right_high_index,
            candidate_high_counts,
        )
        left_candidate = _dominant_local_child(
            candidate_left_parent_index.get(bridge["left_parent"]),
            candidate_bridge_counts,
        )
        right_candidate = _dominant_local_child(
            candidate_right_parent_index.get(bridge["right_parent"]),
            candidate_bridge_counts,
        )
        squareable = bool(anchor_future and candidate_future and left_candidate and right_candidate)
        anchored = bool(squareable and left_candidate == actual and right_candidate == actual)
        dropout = bool(anchor_future and not candidate_future)
        disagreement = bool(squareable and left_candidate != right_candidate)
        defect = bool(anchor_future and not anchored)
        distance = None
        if squareable:
            distance = (
                _normalized_hamming(actual, left_candidate)
                + _normalized_hamming(actual, right_candidate)
                + _normalized_hamming(left_candidate, right_candidate)
            ) / 3.0

        if anchor_future:
            future_weight += weight
            if anchored:
                anchored_weight += weight
            if defect:
                defect_weight += weight
                family = _defect_family(actual, left_candidate, right_candidate, dropout=dropout)
                family_weights[family] = family_weights.get(family, 0.0) + weight
            if dropout:
                dropout_weight += weight
            if disagreement:
                disagreement_weight += weight
            if distance is not None:
                distances.append((distance, weight))

        details.append(
            {
                "bridge_pattern": actual,
                "left_parent": bridge["left_parent"],
                "right_parent": bridge["right_parent"],
                "anchor_local_count": int(bridge["anchor_local_count"]),
                "candidate_local_count": int(candidate_bridge_counts.get(_pattern_code(actual), 0)),
                "anchor_future": anchor_future,
                "candidate_future": candidate_future,
                "left_candidate": left_candidate,
                "right_candidate": right_candidate,
                "squareable": squareable,
                "anchored_square": anchored,
                "dropout_flag": dropout,
                "square_defect_flag": defect,
                "bridge_disagreement_flag": disagreement,
                "square_distance": distance,
            }
        )

    graph_result = _build_defect_transition_graph(details)
    spectrum = compute_graph_spectrum(graph_result)
    top_families = [
        {"family": family, "weight": weight}
        for family, weight in sorted(family_weights.items(), key=lambda item: (-item[1], item[0]))[:10]
    ]
    return {
        "anchor_run_dir": str(anchor_path),
        "candidate_run_dir": str(candidate_path),
        "window_scales": list(window_scales),
        "left_policy": left_policy,
        "right_policy": right_policy,
        "top_bridges": top_bridges,
        "anchor_bridge_universe_count": len(bridge_items),
        "monitored_bridge_count": len(monitored),
        "anchor_future_bridge_count": sum(1 for item in details if item["anchor_future"]),
        "candidate_future_bridge_count": sum(1 for item in details if item["candidate_future"]),
        "squareable_bridge_count": sum(1 for item in details if item["squareable"]),
        "anchored_square_count": sum(1 for item in details if item["anchored_square"]),
        "square_defect_count": sum(1 for item in details if item["square_defect_flag"]),
        "dropout_count": sum(1 for item in details if item["dropout_flag"]),
        "bridge_disagreement_count": sum(1 for item in details if item["bridge_disagreement_flag"]),
        "anchor_future_bridge_fraction": _safe_ratio(sum(1 for item in details if item["anchor_future"]), len(details)),
        "candidate_future_bridge_fraction": _safe_ratio(
            sum(1 for item in details if item["candidate_future"]),
            len(details),
        ),
        "squareable_bridge_fraction": _safe_ratio(sum(1 for item in details if item["squareable"]), len(details)),
        "anchored_square_mass_rate": _safe_ratio(anchored_weight, future_weight),
        "square_defect_mass_rate": _safe_ratio(defect_weight, future_weight),
        "dropout_mass_rate": _safe_ratio(dropout_weight, future_weight),
        "bridge_disagreement_mass_rate": _safe_ratio(disagreement_weight, future_weight),
        "mean_square_distance": _weighted_mean(distances),
        "square_entropy": _weight_entropy(list(family_weights.values())),
        "top_square_families": top_families,
        "graph_result": graph_result,
        "spectrum": spectrum,
        "bridge_details": details,
        "notes": [
            "The monitored bridge universe is anchor-defined and window-local.",
            "Candidate bridge decisions are evaluated in the lag-shifted candidate window.",
            "A dropout is counted as a square defect when the anchor bridge is future-bearing but the candidate bridge is not.",
            "The defect graph is a sparse transition graph from anchor bridges to candidate corner choices or dropout.",
            "This remains pre-geometric; no curvature or transport law is claimed.",
        ],
    }


def compact_lag_conditioned_row(row: dict[str, Any]) -> dict[str, Any]:
    result = row["result"]
    spectrum = result["spectrum"]
    graph = result["graph_result"]
    return {
        "variant": row["variant"],
        "source_label": row["source_label"],
        "source_kind": row["source_kind"],
        "null_model": row.get("null_model"),
        "null_seed": row.get("null_seed"),
        "anchor_offset_bits": row["anchor_offset_bits"],
        "candidate_offset_bits": row["candidate_offset_bits"],
        "candidate_lag_bits": row["candidate_lag_bits"],
        "monitored_bridge_count": result["monitored_bridge_count"],
        "anchor_future_bridge_fraction": result["anchor_future_bridge_fraction"],
        "candidate_future_bridge_fraction": result["candidate_future_bridge_fraction"],
        "squareable_bridge_fraction": result["squareable_bridge_fraction"],
        "anchored_square_mass_rate": result["anchored_square_mass_rate"],
        "square_defect_mass_rate": result["square_defect_mass_rate"],
        "dropout_mass_rate": result["dropout_mass_rate"],
        "bridge_disagreement_mass_rate": result["bridge_disagreement_mass_rate"],
        "mean_square_distance": result["mean_square_distance"],
        "square_entropy": result["square_entropy"],
        "defect_node_count": graph["node_count"],
        "defect_edge_count": graph["edge_count"],
        "lambda2_norm_laplacian": spectrum["lambda2_norm_laplacian"],
        "zero_eigenvalue_count": spectrum["zero_eigenvalue_count"],
        "spectral_radius_adjacency": spectrum["spectral_radius_adjacency"],
        "spectral_status": _spectral_status(graph, spectrum),
    }


def summarize_lag_conditioned_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int | None], list[dict[str, Any]]] = {}
    for row in rows:
        compact = compact_lag_conditioned_row(row)
        groups.setdefault(
            (compact["variant"], compact["source_label"], compact["null_seed"]),
            [],
        ).append(compact)
    summary = []
    for (variant, source_label, null_seed), group in sorted(groups.items(), key=lambda item: _summary_key(item[0])):
        summary.append(
            {
                "variant": variant,
                "source_label": source_label,
                "source_kind": group[0]["source_kind"],
                "null_model": group[0]["null_model"],
                "null_seed": null_seed,
                "window_count": len(group),
                "anchor_future_avg": _mean(item["anchor_future_bridge_fraction"] for item in group),
                "candidate_future_avg": _mean(item["candidate_future_bridge_fraction"] for item in group),
                "anchored_mass_avg": _mean(item["anchored_square_mass_rate"] for item in group),
                "defect_mass_avg": _mean(item["square_defect_mass_rate"] for item in group),
                "dropout_mass_avg": _mean(item["dropout_mass_rate"] for item in group),
                "distance_avg": _mean(item["mean_square_distance"] for item in group),
                "edge_count_avg": _mean(item["defect_edge_count"] for item in group),
                "spectral_radius_avg": _mean(item["spectral_radius_adjacency"] for item in group),
            }
        )
    return summary


def _select_active_anchor_bridges(
    bridge_items: list[dict[str, Any]],
    *,
    count_map: dict[int, int],
    pattern_scale: int,
    top_bridges: int,
) -> list[dict[str, Any]]:
    active = []
    for bridge in bridge_items:
        count = int(count_map.get(_pattern_code(bridge["bridge_pattern"]), 0))
        if count <= 0:
            continue
        active.append({**bridge, "anchor_local_count": count})
    return sorted(
        active,
        key=lambda item: (-int(item["anchor_local_count"]), -int(item.get("raw_count", 0)), item["bridge_pattern"]),
    )[:top_bridges]


def _future_bearing_local(
    bridge_pattern: str,
    left_high_index: dict[str, dict],
    right_high_index: dict[str, dict],
    high_count_map: dict[int, int],
) -> bool:
    return _parent_has_local_child(left_high_index.get(bridge_pattern), high_count_map) and _parent_has_local_child(
        right_high_index.get(bridge_pattern),
        high_count_map,
    )


def _parent_has_local_child(parent: dict | None, count_map: dict[int, int]) -> bool:
    if parent is None:
        return False
    for child in parent.get("children", []):
        if int(count_map.get(_pattern_code(child["pattern"]), 0)) > 0:
            return True
    return False


def _dominant_local_child(parent: dict | None, count_map: dict[int, int]) -> str | None:
    if parent is None:
        return None
    best_pattern = None
    best_score = 0
    best_raw = 0
    for child in parent.get("children", []):
        pattern = str(child["pattern"])
        score = int(count_map.get(_pattern_code(pattern), 0))
        raw = int(child.get("raw_count", 0))
        if score <= 0:
            continue
        if score > best_score or (score == best_score and raw > best_raw) or (
            score == best_score and raw == best_raw and (best_pattern is None or pattern < best_pattern)
        ):
            best_pattern = pattern
            best_score = score
            best_raw = raw
    return best_pattern


def _build_defect_transition_graph(details: list[dict[str, Any]]) -> dict[str, Any]:
    node_weights: dict[str, float] = {}
    edge_weights: dict[tuple[str, str], float] = {}
    for detail in details:
        if not detail["anchor_future"]:
            continue
        actual = detail["bridge_pattern"]
        weight = float(detail["anchor_local_count"])
        node_weights[actual] = node_weights.get(actual, 0.0) + weight
        targets = []
        if detail["dropout_flag"]:
            targets.append(DROP_NODE)
        else:
            for candidate in (detail["left_candidate"], detail["right_candidate"]):
                if candidate and candidate != actual:
                    targets.append(candidate)
        for target in sorted(set(targets)):
            node_weights[target] = node_weights.get(target, 0.0) + weight
            edge = tuple(sorted((actual, target)))
            if edge[0] == edge[1]:
                continue
            edge_weights[edge] = edge_weights.get(edge, 0.0) + weight

    nodes = [
        {"id": node_id, "node_weight": weight}
        for node_id, weight in sorted(node_weights.items(), key=lambda item: (-item[1], item[0]))
    ]
    edges = [
        {
            "id": f"{left}||{right}",
            "nodes": [left, right],
            "edge_weight": weight,
        }
        for (left, right), weight in sorted(edge_weights.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_records": nodes,
        "edge_records": edges,
    }


def _window_code_count_map(window_bits: np.ndarray, pattern_scale: int) -> dict[int, int]:
    if int(window_bits.size) < pattern_scale:
        return {}
    codes = rolling_codes_uint64_max64(window_bits, pattern_scale)
    unique, counts = np.unique(codes, return_counts=True)
    return {
        int(code): int(count)
        for code, count in zip(unique.tolist(), counts.tolist())
    }


def _index_parents(edge: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(parent["pattern"]): parent
        for parent in edge.get("parents", [])
    }


def _defect_family(actual: str, left: str | None, right: str | None, *, dropout: bool) -> str:
    if dropout:
        return f"{actual}=>{DROP_NODE}"
    return f"{actual}=>{left or '-'}|{right or '-'}"


def _pattern_code(pattern: str) -> int:
    return int(pattern, 2)


def _weighted_mean(values: list[tuple[float, float]]) -> float | None:
    total = sum(weight for _, weight in values)
    if total <= 0.0:
        return None
    return sum(value * weight for value, weight in values) / total


def _mean(values) -> float | None:
    vals = [float(value) for value in values if value is not None and not _bad_float(value)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _bad_float(value: Any) -> bool:
    return isinstance(value, float) and (math.isnan(value) or math.isinf(value))


def _spectral_status(graph: dict[str, Any], spectrum: dict[str, Any]) -> str:
    if graph["node_count"] == 0:
        return "untestable-no-nodes"
    if graph["edge_count"] == 0:
        return "no-defect-edges"
    if spectrum["zero_eigenvalue_count"] == 1:
        return "connected-defect-graph"
    return "fragmented-defect-graph"


def _summary_key(key: tuple[str, str, int | None]) -> tuple[int, str, int]:
    variant, source_label, seed = key
    if source_label == "observed":
        return (0, variant, -1)
    if "markov1" in source_label:
        return (1, variant, -1)
    return (2, variant, int(seed or -1))
