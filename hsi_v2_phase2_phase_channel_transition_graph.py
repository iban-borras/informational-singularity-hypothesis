#!/usr/bin/env python3
"""HSI v2 Phase 2 N2-14 phase-channel transition graph readout."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v2.common.cli import resolve_dir


DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase2/phase_channel_transition_graph"
DEFAULT_ENVELOPE_ROOTS = (
    "results/hsi_v2/phase2/phase_channel_width_topk_sensitivity,"
    "results/hsi_v2/phase2/phase_channel_width_topk_sensitivity_frontier_top96"
)
DEFAULT_SIGNATURE_ROOT = "results/hsi_v2/phase2/off_universe_mass_readout"

NODE_FIELDS = [
    "top_patterns",
    "band",
    "band_start_bits",
    "band_stop_bits",
    "channel",
    "strong_count",
    "strong_lag_min_bits",
    "strong_lag_max_bits",
    "strong_width_bits",
    "best_margin_lag_bits",
    "best_margin",
    "node_class",
    "mechanistic_signatures",
    "source_run",
]

EDGE_FIELDS = [
    "top_patterns",
    "channel",
    "source_band",
    "target_band",
    "source_class",
    "target_class",
    "transition_type",
    "lag_delta_bits",
    "margin_delta",
    "width_delta_bits",
    "source_best_lag_bits",
    "target_best_lag_bits",
]

PATH_FIELDS = [
    "top_patterns",
    "channel",
    "band_count",
    "active_count",
    "boundary_count",
    "max_active_path_bands",
    "max_active_path",
    "transition_types",
    "verdict",
]

COUNT_FIELDS = ["top_patterns", "channel", "transition_type", "count"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small falsifiable transition graph from N2-12b phase-channel "
            "envelopes and N2-13a routing signatures."
        )
    )
    parser.add_argument(
        "--envelope-runs",
        default="",
        help="Comma-separated N2-12/N2-12b run directories containing channel_envelopes.csv.",
    )
    parser.add_argument(
        "--envelope-roots",
        default=DEFAULT_ENVELOPE_ROOTS,
        help="Comma-separated roots used when --envelope-runs is omitted.",
    )
    parser.add_argument(
        "--signature-run",
        default="",
        help="N2-13a run directory containing routing_signatures.csv. Defaults to latest under --signature-root.",
    )
    parser.add_argument("--signature-root", default=DEFAULT_SIGNATURE_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-top", type=int, default=128)
    parser.add_argument("--width-threshold-bits", type=int, default=2_000_000)
    parser.add_argument("--lag-tolerance-bits", type=int, default=3_000_000)
    parser.add_argument("--min-path-bands", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.reference_top <= 0:
        raise SystemExit("--reference-top must be positive")
    if args.width_threshold_bits < 0:
        raise SystemExit("--width-threshold-bits must be non-negative")
    if args.lag_tolerance_bits < 0:
        raise SystemExit("--lag-tolerance-bits must be non-negative")
    if args.min_path_bands <= 0:
        raise SystemExit("--min-path-bands must be positive")

    envelope_runs = resolve_envelope_runs(args)
    signature_run = resolve_signature_run(args)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_slug = f"phase2-phase-channel-transition-graph__runs-{len(envelope_runs)}__{timestamp}"
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    envelope_rows = load_envelopes(envelope_runs)
    signature_rows = read_csv(signature_run / "routing_signatures.csv")
    nodes = build_nodes(
        envelope_rows,
        signature_rows,
        width_threshold_bits=args.width_threshold_bits,
    )
    edges = build_edges(nodes, lag_tolerance_bits=args.lag_tolerance_bits)
    path_rows = build_path_rows(
        nodes,
        edges,
        reference_top=args.reference_top,
        min_path_bands=args.min_path_bands,
    )
    count_rows = build_transition_counts(edges)

    summary = {
        "run_id": run_slug,
        "timestamp_utc": timestamp,
        "parameters": vars(args),
        "envelope_runs": [str(path) for path in envelope_runs],
        "signature_run": str(signature_run),
        "notes": [
            "N2-14 is an internal readout over existing N2-12b and N2-13a artifacts.",
            "Nodes are band x channel envelope states; edges are adjacent-band transitions within the same top-k and channel.",
            "The readout is pre-spectral: it does not build a Laplacian, geometry, or final transport law.",
            "The falsification target is simple: if active channel states do not form paths across adjacent bands, the scaffold reading fails.",
        ],
        "nodes": nodes,
        "edges": edges,
        "paths": path_rows,
        "transition_counts": count_rows,
    }
    manifest = {
        "script": Path(__file__).name,
        "run_dir": str(run_dir),
        "inputs": {
            "envelope_runs": [str(path) for path in envelope_runs],
            "signature_run": str(signature_run),
        },
        "outputs": {
            "summary": str(run_dir / "summary.json"),
            "report": str(run_dir / "report.md"),
            "nodes_csv": str(run_dir / "nodes.csv"),
            "edges_csv": str(run_dir / "edges.csv"),
            "paths_csv": str(run_dir / "paths.csv"),
            "transition_counts_csv": str(run_dir / "transition_counts.csv"),
            "manifest": str(run_dir / "manifest.json"),
        },
    }

    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "manifest.json", manifest)
    write_csv(run_dir / "nodes.csv", nodes, NODE_FIELDS)
    write_csv(run_dir / "edges.csv", edges, EDGE_FIELDS)
    write_csv(run_dir / "paths.csv", path_rows, PATH_FIELDS)
    write_csv(run_dir / "transition_counts.csv", count_rows, COUNT_FIELDS)
    (run_dir / "report.md").write_text(render_report(summary), encoding="utf-8")

    if not args.quiet:
        print(render_console_summary(path_rows, count_rows, reference_top=args.reference_top))
        print(f"\nSaved summary to: {run_dir / 'summary.json'}")
        print(f"Saved report to: {run_dir / 'report.md'}")
        print(f"Saved nodes CSV to: {run_dir / 'nodes.csv'}")
        print(f"Saved edges CSV to: {run_dir / 'edges.csv'}")
        print(f"Saved manifest to: {run_dir / 'manifest.json'}")
    return 0


def resolve_envelope_runs(args: argparse.Namespace) -> list[Path]:
    if args.envelope_runs.strip():
        runs = [
            resolve_dir(raw.strip(), anchor_file=__file__)
            for raw in args.envelope_runs.split(",")
            if raw.strip()
        ]
    else:
        runs = []
        for raw_root in args.envelope_roots.split(","):
            root_text = raw_root.strip()
            if not root_text:
                continue
            root = resolve_dir(root_text, anchor_file=__file__)
            runs.append(latest_run_with_file(root, "channel_envelopes.csv"))
    if not runs:
        raise SystemExit("No envelope runs resolved.")
    for run in runs:
        if not (run / "channel_envelopes.csv").is_file():
            raise SystemExit(f"Missing channel_envelopes.csv: {run}")
    return sorted(set(runs), key=lambda path: str(path))


def resolve_signature_run(args: argparse.Namespace) -> Path:
    if args.signature_run.strip():
        run = resolve_dir(args.signature_run, anchor_file=__file__)
    else:
        root = resolve_dir(args.signature_root, anchor_file=__file__)
        run = latest_run_with_file(root, "routing_signatures.csv")
    if not (run / "routing_signatures.csv").is_file():
        raise SystemExit(f"Missing routing_signatures.csv: {run}")
    return run


def latest_run_with_file(root: Path, filename: str) -> Path:
    if not root.is_dir():
        raise SystemExit(f"Root not found: {root}")
    candidates = [child for child in root.iterdir() if child.is_dir() and (child / filename).is_file()]
    if not candidates:
        raise SystemExit(f"No run with {filename} found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_envelopes(runs: list[Path]) -> list[dict[str, Any]]:
    deduped: dict[tuple[int, str, str], dict[str, Any]] = {}
    for run in runs:
        for row in read_csv(run / "channel_envelopes.csv"):
            parsed = {
                "top_patterns": to_int(row["top_patterns"]),
                "band": row["band"],
                "channel": row["channel"],
                "strong_count": to_int(row["strong_count"]),
                "strong_lag_min_bits": to_optional_int(row.get("strong_lag_min_bits", "")),
                "strong_lag_max_bits": to_optional_int(row.get("strong_lag_max_bits", "")),
                "strong_width_bits": to_optional_int(row.get("strong_width_bits", "")),
                "best_margin_lag_bits": to_optional_int(row.get("best_margin_lag_bits", "")),
                "best_margin_lag": row.get("best_margin_lag", ""),
                "best_margin_B_retention": to_optional_float(row.get("best_margin_B_retention", "")),
                "best_margin": to_optional_float(row.get("best_margin", "")),
                "source_run": run.name,
                "source_mtime": run.stat().st_mtime,
            }
            key = (parsed["top_patterns"], parsed["band"], parsed["channel"])
            current = deduped.get(key)
            if current is None or parsed["source_mtime"] >= current["source_mtime"]:
                deduped[key] = parsed
    return sorted(
        deduped.values(),
        key=lambda row: (row["top_patterns"], parse_band(row["band"])[0], row["channel"]),
    )


def build_nodes(
    envelope_rows: list[dict[str, Any]],
    signature_rows: list[dict[str, str]],
    *,
    width_threshold_bits: int,
) -> list[dict[str, Any]]:
    nodes = []
    for row in envelope_rows:
        band_start, band_stop = parse_band(row["band"])
        signatures = matching_signatures(row, signature_rows)
        signature_names = sorted({item["mechanistic_signature"] for item in signatures if item.get("mechanistic_signature")})
        node_class = classify_node(row, signature_names, width_threshold_bits=width_threshold_bits)
        nodes.append(
            {
                "top_patterns": row["top_patterns"],
                "band": row["band"],
                "band_start_bits": band_start,
                "band_stop_bits": band_stop,
                "channel": row["channel"],
                "strong_count": row["strong_count"],
                "strong_lag_min_bits": blank_if_none(row["strong_lag_min_bits"]),
                "strong_lag_max_bits": blank_if_none(row["strong_lag_max_bits"]),
                "strong_width_bits": blank_if_none(row["strong_width_bits"]),
                "best_margin_lag_bits": blank_if_none(row["best_margin_lag_bits"]),
                "best_margin": blank_if_none(row["best_margin"]),
                "node_class": node_class,
                "mechanistic_signatures": ";".join(signature_names),
                "source_run": row["source_run"],
            }
        )
    return nodes


def matching_signatures(row: dict[str, Any], signature_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    top = row["top_patterns"]
    band = row["band"]
    channel = row["channel"]
    matches = []
    for sig in signature_rows:
        if sig.get("band") != band or sig.get("channel") != channel:
            continue
        test_top = to_optional_int(sig.get("test_top", ""))
        reference_top = to_optional_int(sig.get("reference_top", ""))
        if top == test_top or top == reference_top:
            matches.append(sig)
    return matches


def classify_node(row: dict[str, Any], signature_names: list[str], *, width_threshold_bits: int) -> str:
    if "reference-null-shadowing" in signature_names or "reference-null-shadow-relief" in signature_names:
        return "null-shadow-boundary"
    strong_count = int(row["strong_count"])
    width = row["strong_width_bits"] if row["strong_width_bits"] is not None else 0
    if strong_count <= 0:
        return "inactive"
    if width >= width_threshold_bits:
        return "wide-active"
    return "narrow-active"


def build_edges(nodes: list[dict[str, Any]], *, lag_tolerance_bits: int) -> list[dict[str, Any]]:
    by_series: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for node in nodes:
        by_series.setdefault((int(node["top_patterns"]), node["channel"]), []).append(node)

    edges = []
    for (top, channel), series in sorted(by_series.items()):
        ordered = sorted(series, key=lambda row: int(row["band_start_bits"]))
        for source, target in zip(ordered, ordered[1:]):
            source_lag = optional_int_from_value(source["best_margin_lag_bits"])
            target_lag = optional_int_from_value(target["best_margin_lag_bits"])
            source_width = optional_int_from_value(source["strong_width_bits"]) or 0
            target_width = optional_int_from_value(target["strong_width_bits"]) or 0
            source_margin = optional_float_from_value(source["best_margin"])
            target_margin = optional_float_from_value(target["best_margin"])
            edges.append(
                {
                    "top_patterns": top,
                    "channel": channel,
                    "source_band": source["band"],
                    "target_band": target["band"],
                    "source_class": source["node_class"],
                    "target_class": target["node_class"],
                    "transition_type": classify_transition(source, target, lag_tolerance_bits=lag_tolerance_bits),
                    "lag_delta_bits": blank_if_none(delta(target_lag, source_lag)),
                    "margin_delta": blank_if_none(delta(target_margin, source_margin)),
                    "width_delta_bits": target_width - source_width,
                    "source_best_lag_bits": blank_if_none(source_lag),
                    "target_best_lag_bits": blank_if_none(target_lag),
                }
            )
    return edges


def classify_transition(
    source: dict[str, Any],
    target: dict[str, Any],
    *,
    lag_tolerance_bits: int,
) -> str:
    if source["node_class"] == "null-shadow-boundary" or target["node_class"] == "null-shadow-boundary":
        return "null-shadow-boundary"

    source_active = is_active_node(source)
    target_active = is_active_node(target)
    if source_active and target_active:
        if intervals_overlap(source, target):
            return "continuous-envelope"
        source_lag = optional_int_from_value(source["best_margin_lag_bits"])
        target_lag = optional_int_from_value(target["best_margin_lag_bits"])
        if source_lag is not None and target_lag is not None and abs(target_lag - source_lag) <= lag_tolerance_bits:
            return "drift-envelope"
        return "phase-jump"
    if not source_active and target_active:
        return "activation"
    if source_active and not target_active:
        return "attenuation"
    return "inactive-continuity"


def build_path_rows(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    reference_top: int,
    min_path_bands: int,
) -> list[dict[str, Any]]:
    edge_key = {
        (edge["top_patterns"], edge["channel"], edge["source_band"], edge["target_band"]): edge
        for edge in edges
    }
    by_series: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for node in nodes:
        by_series.setdefault((int(node["top_patterns"]), node["channel"]), []).append(node)

    rows = []
    for (top, channel), series in sorted(by_series.items()):
        ordered = sorted(series, key=lambda row: int(row["band_start_bits"]))
        active_count = sum(1 for node in ordered if is_active_node(node))
        boundary_count = sum(1 for node in ordered if node["node_class"] == "null-shadow-boundary")
        max_path_nodes = longest_active_path(ordered)
        transition_types = []
        for source, target in zip(ordered, ordered[1:]):
            edge = edge_key[(top, channel, source["band"], target["band"])]
            transition_types.append(edge["transition_type"])
        rows.append(
            {
                "top_patterns": top,
                "channel": channel,
                "band_count": len(ordered),
                "active_count": active_count,
                "boundary_count": boundary_count,
                "max_active_path_bands": len(max_path_nodes),
                "max_active_path": " -> ".join(node["band"] for node in max_path_nodes),
                "transition_types": ";".join(transition_types),
                "verdict": classify_path_verdict(
                    top=top,
                    reference_top=reference_top,
                    active_path_len=len(max_path_nodes),
                    boundary_count=boundary_count,
                    min_path_bands=min_path_bands,
                ),
            }
        )
    return rows


def build_transition_counts(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter = Counter((edge["top_patterns"], edge["channel"], edge["transition_type"]) for edge in edges)
    return [
        {"top_patterns": top, "channel": channel, "transition_type": transition_type, "count": count}
        for (top, channel, transition_type), count in sorted(counter.items())
    ]


def classify_path_verdict(
    *,
    top: int,
    reference_top: int,
    active_path_len: int,
    boundary_count: int,
    min_path_bands: int,
) -> str:
    if active_path_len < min_path_bands:
        return "no-stable-channel-path"
    if top == reference_top and boundary_count:
        return "structured-channel-with-boundary"
    return "structured-channel-path"


def is_active_node(node: dict[str, Any]) -> bool:
    return int(node["strong_count"]) > 0


def intervals_overlap(source: dict[str, Any], target: dict[str, Any]) -> bool:
    a0 = optional_int_from_value(source["strong_lag_min_bits"])
    a1 = optional_int_from_value(source["strong_lag_max_bits"])
    b0 = optional_int_from_value(target["strong_lag_min_bits"])
    b1 = optional_int_from_value(target["strong_lag_max_bits"])
    if None in (a0, a1, b0, b1):
        return False
    return max(a0, b0) <= min(a1, b1)


def longest_active_path(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for node in nodes:
        if is_active_node(node):
            current.append(node)
            if len(current) > len(best):
                best = list(current)
        else:
            current = []
    return best


def render_console_summary(path_rows: list[dict[str, Any]], count_rows: list[dict[str, Any]], *, reference_top: int) -> str:
    lines = [
        "Phase 2 N2-14 phase-channel transition graph",
        "-" * 104,
        f"{'top':>5} {'channel':<8} {'active':>8} {'boundary':>9} {'max_path':>9}  verdict",
    ]
    for row in path_rows:
        marker = "*" if int(row["top_patterns"]) == reference_top else " "
        lines.append(
            f"{marker}{int(row['top_patterns']):>4} {row['channel']:<8} "
            f"{int(row['active_count']):>8}/{int(row['band_count']):<2} "
            f"{int(row['boundary_count']):>9} "
            f"{int(row['max_active_path_bands']):>9}  {row['verdict']}"
        )
    lines.append("")
    lines.append("Transition counts")
    lines.append("-" * 104)
    for row in count_rows:
        lines.append(
            f"top={row['top_patterns']} | {row['channel']:<8} | "
            f"{row['transition_type']:<24} {row['count']}"
        )
    return "\n".join(lines)


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 2 N2-14 Phase-Channel Transition Graph",
        "",
        "## Status",
        "",
        "Internal diagnostic readout. This is a pre-spectral graph over existing N2-12b/N2-13a artifacts.",
        "",
        "## Falsification Contract",
        "",
        "- If active channel states do not form paths across adjacent bands, the phase-channel scaffold reading fails.",
        "- If the late boundary cannot be separated from the central active path, the null-shadowing boundary reading weakens.",
        "- If lower top-k and reference top-k disagree everywhere, the graph is only a top-k artifact.",
        "",
        "## Path Summary",
        "",
        "| Top-k | Channel | Active bands | Boundary nodes | Max active path | Verdict |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary["paths"]:
        lines.append(
            f"| {row['top_patterns']} | {row['channel']} | "
            f"{row['active_count']}/{row['band_count']} | {row['boundary_count']} | "
            f"{row['max_active_path_bands']} | {row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Transition Counts",
            "",
            "| Top-k | Channel | Transition | Count |",
            "| ---: | --- | --- | ---: |",
        ]
    )
    for row in summary["transition_counts"]:
        lines.append(
            f"| {row['top_patterns']} | {row['channel']} | {row['transition_type']} | {row['count']} |"
        )
    lines.extend(
        [
            "",
            "## Reading Boundary",
            "",
            "N2-14 does not introduce geometry, spectra, or a final transport law. It only tests whether the already measured phase-channel cells assemble into adjacent-band paths with explicit boundary states.",
        ]
    )
    return "\n".join(lines)


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def parse_band(value: str) -> tuple[int, int]:
    left, right = value.split("-", 1)
    return parse_m_bits(left), parse_m_bits(right)


def parse_m_bits(value: str) -> int:
    value = value.strip()
    if value.endswith("M"):
        return int(float(value[:-1]) * 1_000_000)
    if value.endswith("K"):
        return int(float(value[:-1]) * 1_000)
    return int(value)


def to_int(value: str) -> int:
    return int(float(str(value).strip()))


def to_optional_int(value: str | None) -> int | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    return int(float(text))


def to_optional_float(value: str | None) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    return float(text)


def optional_int_from_value(value: Any) -> int | None:
    if value == "" or value is None:
        return None
    return int(float(value))


def optional_float_from_value(value: Any) -> float | None:
    if value == "" or value is None:
        return None
    return float(value)


def blank_if_none(value: Any) -> Any:
    return "" if value is None else value


def delta(left: float | int | None, right: float | int | None) -> float | int | None:
    if left is None or right is None:
        return None
    return left - right


if __name__ == "__main__":
    raise SystemExit(main())
