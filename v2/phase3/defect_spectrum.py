from __future__ import annotations

import math
from typing import Any

import numpy as np

from v2.phase2.sparse_graph import build_sparse_graph_rows
from v2.phase2.square import build_square_rows


def build_defect_spectrum_rows(
    runs: list[dict[str, Any]],
    *,
    window_scales: tuple[int, int, int, int],
    left_policy: str,
    right_policy: str,
    min_segment_recurrence: int,
    top_hub_k: int,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    config_by_run_dir = {str(run["_run_dir"]): run["dataset"]["config"] for run in runs}
    square_rows = build_square_rows(
        runs,
        window_scales=window_scales,
        left_policy=left_policy,
        right_policy=right_policy,
    )
    graph_rows = build_sparse_graph_rows(
        runs,
        window_scales=window_scales,
        left_policy=left_policy,
        right_policy=right_policy,
        min_segment_recurrence=min_segment_recurrence,
        top_hub_k=top_hub_k,
        show_progress=show_progress,
    )
    graph_by_run = {row["run_dir"]: row for row in graph_rows}
    rows: list[dict[str, Any]] = []
    for square_row in square_rows:
        graph_row = graph_by_run[square_row["run_dir"]]
        config = config_by_run_dir[square_row["run_dir"]]
        spectrum = compute_graph_spectrum(graph_row["graph_result"])
        row = {
            "variant": square_row["variant"],
            "source_kind": _source_kind(config),
            "null_model": config.get("null_model"),
            "null_seed": config.get("null_seed"),
            "iteration": square_row["iteration"],
            "segment_bits": square_row["segment_bits"],
            "num_segments": square_row["num_segments"],
            "segment_offset_bits": square_row["segment_offset_bits"],
            "scales": square_row["scales"],
            "window_scales": square_row["window_scales"],
            "left_policy": left_policy,
            "right_policy": right_policy,
            "run_dir": square_row["run_dir"],
            "dataset_path": square_row["dataset_path"],
            "summary_path": square_row["summary_path"],
            "generated_at": square_row["generated_at"],
            "future_bridge_fraction": square_row["future_bridge_fraction"],
            "squareable_bridge_fraction": square_row["squareable_bridge_fraction"],
            "anchored_square_rate": square_row["anchored_square_rate"],
            "square_defect_rate": square_row["square_defect_rate"],
            "square_defect_mass_rate": square_row["square_result"]["square_defect_mass_rate"],
            "bridge_disagreement_rate": square_row["bridge_disagreement_rate"],
            "mean_square_distance": square_row["mean_square_distance"],
            "square_entropy": square_row["square_entropy"],
            "square_segment_consistency": square_row["segment_consistency"],
            "node_count": graph_row["node_count"],
            "edge_count": graph_row["edge_count"],
            "largest_component_fraction": graph_row["largest_component_fraction"],
            "recurrent_edge_fraction": graph_row["recurrent_edge_fraction"],
            "recurrent_edge_mass_fraction": graph_row["recurrent_edge_mass_fraction"],
            "core_node_fraction": graph_row["core_node_fraction"],
            "degree_entropy": graph_row["degree_entropy"],
            "hub_mass_fraction": graph_row["hub_mass_fraction"],
            "graph_segment_core_jaccard": graph_row["segment_core_jaccard"],
            "lambda2_norm_laplacian": spectrum["lambda2_norm_laplacian"],
            "zero_eigenvalue_count": spectrum["zero_eigenvalue_count"],
            "spectral_radius_adjacency": spectrum["spectral_radius_adjacency"],
            "spectral_status": spectral_status(graph_row, spectrum),
            "square_result": square_row["square_result"],
            "graph_result": graph_row["graph_result"],
            "spectrum": spectrum,
        }
        rows.append(row)
    return rows


def compute_graph_spectrum(graph_result: dict[str, Any], *, zero_tol: float = 1e-9) -> dict[str, Any]:
    nodes = graph_result.get("node_records", [])
    node_ids = [node["id"] for node in nodes]
    n = len(node_ids)
    if n == 0:
        return {
            "node_count": 0,
            "edge_count": 0,
            "lambda2_norm_laplacian": None,
            "zero_eigenvalue_count": 0,
            "spectral_radius_adjacency": None,
            "norm_laplacian_eigenvalues": [],
            "adjacency_eigenvalues": [],
        }

    index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    adjacency = np.zeros((n, n), dtype=float)
    for edge in graph_result.get("edge_records", []):
        left, right = edge["nodes"]
        if left not in index or right not in index:
            continue
        weight = float(edge.get("edge_weight", 1.0))
        i = index[left]
        j = index[right]
        adjacency[i, j] += weight
        adjacency[j, i] += weight

    degree = adjacency.sum(axis=1)
    positive = degree > 0
    laplacian = np.zeros((n, n), dtype=float)
    laplacian[np.diag_indices(n)] = positive.astype(float)
    if positive.any():
        inv_sqrt = np.zeros_like(degree)
        inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
        laplacian -= (inv_sqrt[:, None] * adjacency) * inv_sqrt[None, :]

    norm_eigs = np.linalg.eigvalsh(laplacian)
    norm_eigs = np.clip(norm_eigs, 0.0, None)
    adjacency_eigs = np.linalg.eigvalsh(adjacency)
    zero_count = int(np.sum(norm_eigs <= zero_tol))
    lambda2 = float(norm_eigs[1]) if n >= 2 else None
    spectral_radius = float(np.max(np.abs(adjacency_eigs))) if adjacency_eigs.size else None
    return {
        "node_count": n,
        "edge_count": len(graph_result.get("edge_records", [])),
        "lambda2_norm_laplacian": lambda2,
        "zero_eigenvalue_count": zero_count,
        "spectral_radius_adjacency": spectral_radius,
        "norm_laplacian_eigenvalues": [float(value) for value in norm_eigs[: min(12, len(norm_eigs))]],
        "adjacency_eigenvalues": [float(value) for value in adjacency_eigs[-min(12, len(adjacency_eigs)) :]],
    }


def spectral_status(graph_row: dict[str, Any], spectrum: dict[str, Any]) -> str:
    if graph_row["node_count"] == 0:
        return "untestable-no-nodes"
    if graph_row["edge_count"] == 0:
        return "untestable-no-edges"
    if spectrum["zero_eigenvalue_count"] == 1:
        return "connected-pre-spectral"
    return "fragmented-pre-spectral"


def render_console_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "Phase 3 minimal defect/spectrum pilot",
        "-" * 132,
        (
            f"{'variant':<16}{'src':<10}{'sq_def':>8}{'sq_mass':>9}{'dist':>8}"
            f"{'nodes':>8}{'edges':>8}{'rec_m':>8}{'lcc':>8}{'lam2':>9}{'zero':>6}  status"
        ),
    ]
    for row in rows:
        lines.append(
            f"{row['variant']:<16}"
            f"{row['source_kind']:<10}"
            f"{_fmt(row['square_defect_rate']):>8}"
            f"{_fmt(row['square_defect_mass_rate']):>9}"
            f"{_fmt(row['mean_square_distance']):>8}"
            f"{row['node_count']:>8}"
            f"{row['edge_count']:>8}"
            f"{_fmt(row['recurrent_edge_mass_fraction']):>8}"
            f"{_fmt(row['largest_component_fraction']):>8}"
            f"{_fmt(row['lambda2_norm_laplacian']):>9}"
            f"{row['zero_eigenvalue_count']:>6}  "
            f"{row['spectral_status']}"
        )
    return "\n".join(lines)


def render_markdown_report(rows: list[dict[str, Any]], selection: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 Minimal Defect/Spectrum Pilot",
        "",
        "## Contract",
        "",
        "- This is a pre-geometric pilot over existing Phase 1 artifacts.",
        "- `delta_square` is the monitored commuting-square defect.",
        "- `G_m` is the sampled sparse bridge graph.",
        "- The spectrum is the normalized Laplacian spectrum of `G_m`.",
        "- No geometry, curvature, or transport law is claimed by this run.",
        "",
        "## Selection",
        "",
        f"- Variants/sources: {', '.join(selection['variants'])}",
        f"- Segment protocol: {selection['num_segments']} x {selection['segment_bits']}",
        f"- Segment offset bits: {selection['segment_offset_bits']}",
        f"- Phase 1 scales: {', '.join(str(value) for value in selection['phase1_scales'])}",
        f"- Window scales: {', '.join(str(value) for value in selection['window_scales'])}",
        f"- Policies: {selection['left_policy']} vs {selection['right_policy']}",
        f"- Min segment recurrence: {selection['min_segment_recurrence']}",
        "",
        "## Compact Readout",
        "",
        "| Variant | Source | Square defect | Defect mass | Mean distance | Nodes | Edges | Recurrent edge mass | LCC | Lambda2 | Zero eigs | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['source_kind']} | {_fmt(row['square_defect_rate'])} | "
            f"{_fmt(row['square_defect_mass_rate'])} | {_fmt(row['mean_square_distance'])} | "
            f"{row['node_count']} | {row['edge_count']} | {_fmt(row['recurrent_edge_mass_fraction'])} | "
            f"{_fmt(row['largest_component_fraction'])} | {_fmt(row['lambda2_norm_laplacian'])} | "
            f"{row['zero_eigenvalue_count']} | {row['spectral_status']} |"
        )

    lines.extend(
        [
            "",
            "## Falsification Reading",
            "",
            "A positive-looking pilot is not enough. The next step is only justified if observed `B/E` remain testable and differ from strong nulls across both the square-defect and graph/spectral readouts.",
            "",
            "Immediate failure modes:",
            "",
            "- observed runs are untestable while nulls are testable;",
            "- all sources share the same defect/spectral profile;",
            "- the signal exists only in one source, one seed, or one scale window;",
            "- the sampled graph has no recurrent core.",
        ]
    )
    return "\n".join(lines)


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "variant",
        "source_kind",
        "null_model",
        "null_seed",
        "iteration",
        "future_bridge_fraction",
        "squareable_bridge_fraction",
        "anchored_square_rate",
        "square_defect_rate",
        "square_defect_mass_rate",
        "bridge_disagreement_rate",
        "mean_square_distance",
        "square_entropy",
        "square_segment_consistency",
        "node_count",
        "edge_count",
        "largest_component_fraction",
        "recurrent_edge_fraction",
        "recurrent_edge_mass_fraction",
        "core_node_fraction",
        "degree_entropy",
        "hub_mass_fraction",
        "graph_segment_core_jaccard",
        "lambda2_norm_laplacian",
        "zero_eigenvalue_count",
        "spectral_radius_adjacency",
        "spectral_status",
    ]
    return [{key: row.get(key) for key in keys} for row in rows]


def _source_kind(config: dict[str, Any]) -> str:
    null_model = config.get("null_model")
    if null_model:
        return str(null_model)
    return "observed"


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "-"
    return f"{float(value):.4f}"
