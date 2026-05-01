#!/usr/bin/env python3
"""HSI v2 Phase 2 N2-13 monitored-universe expansion readout."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v2.common.cli import resolve_dir


DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase2/off_universe_mass_readout"
DEFAULT_RUN_ROOTS = (
    "results/hsi_v2/phase2/phase_channel_width_topk_sensitivity,"
    "results/hsi_v2/phase2/phase_channel_width_topk_sensitivity_frontier_top96"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify N2-12b top-k sensitivity cells by what changes when the "
            "monitored child universe is expanded to the canonical reference top-k."
        )
    )
    parser.add_argument(
        "--topk-runs",
        default="",
        help="Comma-separated N2-12b run directories. Defaults to latest run under each root.",
    )
    parser.add_argument(
        "--topk-roots",
        default=DEFAULT_RUN_ROOTS,
        help="Comma-separated roots used when --topk-runs is omitted.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--b-threshold", type=float, default=0.90)
    parser.add_argument("--margin-threshold", type=float, default=0.30)
    parser.add_argument(
        "--no-routing-signatures",
        action="store_true",
        help="Skip N2-13a child-routing signature extraction from existing subruns.",
    )
    parser.add_argument(
        "--large-expansion-threshold",
        type=float,
        default=0.10,
        help="Retention delta used only to label large top-k expansion effects.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dirs = resolve_topk_runs(args)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_slug = f"phase2-off-universe-mass-readout__runs-{len(run_dirs)}__{timestamp}"
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    cell_rows: list[dict[str, Any]] = []
    for source_run in run_dirs:
        cell_rows.extend(
            classify_cells(
                source_run,
                b_threshold=args.b_threshold,
                margin_threshold=args.margin_threshold,
                large_expansion_threshold=args.large_expansion_threshold,
            )
        )

    cell_rows = dedupe_cells(cell_rows)
    signature_rows = []
    if not args.no_routing_signatures:
        signature_rows = build_routing_signature_rows(cell_rows, run_dirs)
    band_rows = summarize_by_band(cell_rows)
    source_rows = summarize_by_source(cell_rows)
    signature_summary_rows = summarize_signatures(signature_rows)

    summary = {
        "run_id": run_slug,
        "timestamp_utc": timestamp,
        "parameters": vars(args),
        "topk_runs": [str(path) for path in run_dirs],
        "notes": [
            "N2-13 is a readout over existing N2-12b top-k sensitivity outputs.",
            "It does not classify the full infinite/off-top-k universe.",
            "It classifies what is recovered, lost, or flipped when a lower monitored top-k is expanded to the canonical reference top-k.",
            "Classification uses the inherited N2-12 strong-cell convention: B retention threshold plus margin threshold.",
            "N2-13a routing signatures mine existing child-routing subruns; they do not trigger new heavy computation.",
        ],
        "source_summary": source_rows,
        "band_summary": band_rows,
        "signature_summary": signature_summary_rows,
        "cell_classification": cell_rows,
        "routing_signatures": signature_rows,
    }
    manifest = {
        "script": Path(__file__).name,
        "run_dir": str(run_dir),
        "inputs": [str(path) for path in run_dirs],
        "outputs": {
            "summary": str(run_dir / "summary.json"),
            "report": str(run_dir / "report.md"),
            "cell_classification_csv": str(run_dir / "cell_classification.csv"),
            "band_summary_csv": str(run_dir / "band_summary.csv"),
            "source_summary_csv": str(run_dir / "source_summary.csv"),
            "routing_signatures_csv": str(run_dir / "routing_signatures.csv"),
            "signature_summary_csv": str(run_dir / "signature_summary.csv"),
            "manifest": str(run_dir / "manifest.json"),
        },
    }

    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "manifest.json", manifest)
    write_csv(run_dir / "cell_classification.csv", cell_rows, CELL_FIELDS)
    write_csv(run_dir / "band_summary.csv", band_rows, BAND_FIELDS)
    write_csv(run_dir / "source_summary.csv", source_rows, SOURCE_FIELDS)
    write_csv(run_dir / "routing_signatures.csv", signature_rows, SIGNATURE_FIELDS)
    write_csv(run_dir / "signature_summary.csv", signature_summary_rows, SIGNATURE_SUMMARY_FIELDS)
    (run_dir / "report.md").write_text(render_report(summary), encoding="utf-8")

    if not args.quiet:
        print(render_console_summary(source_rows, band_rows, signature_summary_rows))
        print(f"\nSaved summary to: {run_dir / 'summary.json'}")
        print(f"Saved report to: {run_dir / 'report.md'}")
        print(f"Saved cell CSV to: {run_dir / 'cell_classification.csv'}")
        print(f"Saved band CSV to: {run_dir / 'band_summary.csv'}")
        print(f"Saved routing signatures CSV to: {run_dir / 'routing_signatures.csv'}")
        print(f"Saved manifest to: {run_dir / 'manifest.json'}")
    return 0


def resolve_topk_runs(args: argparse.Namespace) -> list[Path]:
    if args.topk_runs.strip():
        runs = [
            resolve_dir(raw.strip(), anchor_file=__file__)
            for raw in args.topk_runs.split(",")
            if raw.strip()
        ]
    else:
        candidates: list[Path] = []
        for raw_root in args.topk_roots.split(","):
            root_text = raw_root.strip()
            if not root_text:
                continue
            root = resolve_dir(root_text, anchor_file=__file__)
            candidates.extend(topk_runs_under(root))
        runs = latest_run_per_test_top(candidates)
    if not runs:
        raise SystemExit("No top-k sensitivity runs resolved.")
    for run in runs:
        if not (run / "topk_comparison.csv").is_file():
            raise SystemExit(f"Missing topk_comparison.csv: {run}")
    return sorted(set(runs), key=lambda path: str(path))


def topk_runs_under(root: Path) -> list[Path]:
    if not root.is_dir():
        raise SystemExit(f"Top-k root not found: {root}")
    candidates = [
        child for child in root.iterdir()
        if child.is_dir()
        and child.name.startswith("phase2-phase-channel-width-topk-sensitivity__")
        and (child / "topk_comparison.csv").is_file()
    ]
    if not candidates:
        raise SystemExit(f"No N2-12b top-k runs found under {root}")
    return candidates


def latest_run_per_test_top(candidates: list[Path]) -> list[Path]:
    by_top: dict[int, Path] = {}
    for run in candidates:
        for top in test_tops_in_run(run):
            current = by_top.get(top)
            if current is None or run.stat().st_mtime > current.stat().st_mtime:
                by_top[top] = run
    return [by_top[top] for top in sorted(by_top)]


def test_tops_in_run(run: Path) -> set[int]:
    rows = read_csv(run / "topk_comparison.csv")
    return {int(row["test_top"]) for row in rows if row.get("test_top")}


def classify_cells(
    source_run: Path,
    *,
    b_threshold: float,
    margin_threshold: float,
    large_expansion_threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(source_run / "topk_comparison.csv"):
        test_b = to_float(row["test_B"])
        ref_b = to_float(row["reference_B"])
        test_margin = to_float(row["test_margin"])
        ref_margin = to_float(row["reference_margin"])
        test_mlz = value_or_none(test_b, test_margin)
        ref_mlz = value_or_none(ref_b, ref_margin)
        b_gain = none_if_missing(ref_b, test_b)
        mlz_gain = none_if_missing(ref_mlz, test_mlz)
        margin_gain = none_if_missing(ref_margin, test_margin)
        test_strong = is_strong(test_b, test_margin, b_threshold, margin_threshold)
        ref_strong = is_strong(ref_b, ref_margin, b_threshold, margin_threshold)
        rows.append(
            {
                "source_run": source_run.name,
                "test_top": int(row["test_top"]),
                "reference_top": int(row["reference_top"]),
                "band": row["band"],
                "lag": row["lag"],
                "lag_bits": int(row["lag_bits"]),
                "channel": "negative" if int(row["lag_bits"]) < 0 else "positive",
                "test_B": test_b,
                "reference_B": ref_b,
                "B_expansion_gain": b_gain,
                "test_matched_lz_max": test_mlz,
                "reference_matched_lz_max": ref_mlz,
                "matched_lz_expansion_gain": mlz_gain,
                "test_margin": test_margin,
                "reference_margin": ref_margin,
                "margin_expansion_gain": margin_gain,
                "test_strong": test_strong,
                "reference_strong": ref_strong,
                "expansion_class": classify_expansion(
                    test_strong=test_strong,
                    reference_strong=ref_strong,
                    b_gain=b_gain,
                    mlz_gain=mlz_gain,
                    margin_gain=margin_gain,
                    large_expansion_threshold=large_expansion_threshold,
                ),
            }
        )
    return rows


def classify_expansion(
    *,
    test_strong: bool,
    reference_strong: bool,
    b_gain: float | None,
    mlz_gain: float | None,
    margin_gain: float | None,
    large_expansion_threshold: float,
) -> str:
    b_gain = b_gain if b_gain is not None else 0.0
    mlz_gain = mlz_gain if mlz_gain is not None else 0.0
    margin_gain = margin_gain if margin_gain is not None else 0.0
    if test_strong and reference_strong:
        if abs(b_gain) >= large_expansion_threshold:
            return "strong-but-retention-shifted"
        return "core-stable"
    if not test_strong and reference_strong:
        if b_gain >= large_expansion_threshold:
            return "high-k-recovered-B"
        if margin_gain >= large_expansion_threshold:
            return "high-k-recovered-margin"
        return "reference-only-strong"
    if test_strong and not reference_strong:
        if margin_gain <= -large_expansion_threshold:
            return "low-k-only-shadowed-at-reference"
        return "low-k-only-strong"
    if b_gain >= large_expansion_threshold:
        return "subthreshold-B-expansion"
    if mlz_gain >= large_expansion_threshold:
        return "subthreshold-null-shadow-expansion"
    return "inactive-or-stable-subthreshold"


def dedupe_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        key = (
            row["test_top"],
            row["reference_top"],
            row["band"],
            row["lag_bits"],
        )
        by_key[key] = row
    return sorted(
        by_key.values(),
        key=lambda row: (
            int(row["test_top"]),
            int(row["reference_top"]),
            band_start(row["band"]),
            int(row["lag_bits"]),
        ),
    )


def build_routing_signature_rows(
    cell_rows: list[dict[str, Any]],
    run_dirs: list[Path],
) -> list[dict[str, Any]]:
    runs_by_name = {run.name: run for run in run_dirs}
    lag_run_cache: dict[tuple[str, int], Path | None] = {}
    metrics_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
    output: list[dict[str, Any]] = []

    for row in cell_rows:
        source_run = runs_by_name.get(str(row["source_run"]))
        if source_run is None:
            continue
        test_top = int(row["test_top"])
        reference_top = int(row["reference_top"])
        test_metrics = routing_metrics_for_cell(
            source_run=source_run,
            top=test_top,
            band=str(row["band"]),
            lag=str(row["lag"]),
            lag_bits=int(row["lag_bits"]),
            lag_run_cache=lag_run_cache,
            metrics_cache=metrics_cache,
        )
        ref_metrics = routing_metrics_for_cell(
            source_run=source_run,
            top=reference_top,
            band=str(row["band"]),
            lag=str(row["lag"]),
            lag_bits=int(row["lag_bits"]),
            lag_run_cache=lag_run_cache,
            metrics_cache=metrics_cache,
        )
        output.append(build_signature_row(row, test_metrics, ref_metrics))

    return sorted(
        output,
        key=lambda row: (
            int(row["test_top"]),
            int(row["reference_top"]),
            band_start(row["band"]),
            int(row["lag_bits"]),
        ),
    )


def routing_metrics_for_cell(
    *,
    source_run: Path,
    top: int,
    band: str,
    lag: str,
    lag_bits: int,
    lag_run_cache: dict[tuple[str, int], Path | None],
    metrics_cache: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    lag_run = lag_response_run_for_top(source_run, top, lag_run_cache)
    if lag_run is None:
        return {}
    cache_key = (str(lag_run), band, lag_bits)
    if cache_key in metrics_cache:
        return metrics_cache[cache_key]

    child_run = child_routing_run_for_cell(lag_run, band, lag, lag_bits)
    if child_run is None:
        metrics_cache[cache_key] = {}
        return {}

    source_rows = read_csv(child_run / "source_routing.csv")
    pattern_rows = read_csv(child_run / "pattern_routing.csv")
    b_row = first_row(source_rows, variant="B", source_kind="observed")
    markov_row = first_row(source_rows, variant="B-markov1")
    mlz_rows = [row for row in source_rows if row.get("variant") == "B-matched-lz"]
    mlz_shadow = max(
        mlz_rows,
        key=lambda item: parse_float(item.get("child_destination_retention_pooled")),
        default=None,
    )
    b_patterns = matching_pattern_rows(pattern_rows, b_row)
    mlz_patterns = matching_pattern_rows(pattern_rows, mlz_shadow)

    metrics = {
        "routing_run": str(child_run),
        "pattern_count": int(parse_float(b_row.get("pattern_count")) or 0) if b_row else None,
        "B_retention": parse_float(b_row.get("child_destination_retention_pooled")) if b_row else None,
        "B_sync_retention": parse_float(b_row.get("window_synchronous_child_destination_retention_pooled")) if b_row else None,
        "B_candidate_on_anchor_share": parse_float(b_row.get("candidate_on_anchor_child_share_pooled")) if b_row else None,
        "B_reroute_share": parse_float(b_row.get("candidate_reroute_share_pooled")) if b_row else None,
        "B_monitored_reroute_share": parse_float(b_row.get("monitored_candidate_reroute_share_pooled")) if b_row else None,
        "B_deficit_entropy_norm": normalized_entropy_from_rows(b_patterns, "anchor_child_deficit_mass_sum"),
        "B_deficit_top1_share": top_share_from_rows(b_patterns, "anchor_child_deficit_mass_sum", top_n=1),
        "markov_retention": parse_float(markov_row.get("child_destination_retention_pooled")) if markov_row else None,
        "matched_lz_shadow_source": mlz_shadow.get("source_label") if mlz_shadow else "",
        "matched_lz_shadow_seed": mlz_shadow.get("null_seed") if mlz_shadow else "",
        "matched_lz_shadow_retention": (
            parse_float(mlz_shadow.get("child_destination_retention_pooled"))
            if mlz_shadow
            else None
        ),
        "matched_lz_shadow_sync_retention": (
            parse_float(mlz_shadow.get("window_synchronous_child_destination_retention_pooled"))
            if mlz_shadow
            else None
        ),
        "matched_lz_shadow_candidate_on_anchor_share": (
            parse_float(mlz_shadow.get("candidate_on_anchor_child_share_pooled"))
            if mlz_shadow
            else None
        ),
        "matched_lz_shadow_reroute_share": (
            parse_float(mlz_shadow.get("candidate_reroute_share_pooled"))
            if mlz_shadow
            else None
        ),
        "matched_lz_shadow_monitored_reroute_share": (
            parse_float(mlz_shadow.get("monitored_candidate_reroute_share_pooled"))
            if mlz_shadow
            else None
        ),
        "matched_lz_shadow_deficit_entropy_norm": normalized_entropy_from_rows(
            mlz_patterns,
            "anchor_child_deficit_mass_sum",
        ),
        "matched_lz_shadow_deficit_top1_share": top_share_from_rows(
            mlz_patterns,
            "anchor_child_deficit_mass_sum",
            top_n=1,
        ),
        "matched_lz_shadow_deficit_top3_share": top_share_from_rows(
            mlz_patterns,
            "anchor_child_deficit_mass_sum",
            top_n=3,
        ),
        "matched_lz_shadow_routing_class_counts": (
            mlz_shadow.get("pattern_routing_class_counts")
            if mlz_shadow
            else ""
        ),
    }
    metrics_cache[cache_key] = metrics
    return metrics


def lag_response_run_for_top(
    source_run: Path,
    top: int,
    cache: dict[tuple[str, int], Path | None],
) -> Path | None:
    key = (str(source_run), top)
    if key in cache:
        return cache[key]
    summary_path = source_run / "summary.json"
    if not summary_path.is_file():
        cache[key] = None
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_path = summary.get("run_paths", {}).get(str(top))
    resolved = Path(run_path) if run_path else None
    if resolved is not None and not resolved.is_absolute():
        resolved = (source_run / resolved).resolve()
    if resolved is not None and not (resolved / "lag_response.csv").is_file():
        resolved = None
    cache[key] = resolved
    return resolved


def child_routing_run_for_cell(
    lag_run: Path,
    band: str,
    lag: str,
    lag_bits: int,
) -> Path | None:
    lag_slug = ("neg-" if lag_bits < 0 else "pos-") + lag.lstrip("-")
    base = lag_run / band / lag_slug / "child_routing"
    if not base.is_dir():
        return None
    candidates = [
        child for child in base.iterdir()
        if child.is_dir()
        and (child / "source_routing.csv").is_file()
        and (child / "pattern_routing.csv").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_signature_row(
    cell: dict[str, Any],
    test: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    signature = infer_mechanistic_signature(cell, test, reference)
    return {
        "source_run": cell["source_run"],
        "test_top": cell["test_top"],
        "reference_top": cell["reference_top"],
        "band": cell["band"],
        "lag": cell["lag"],
        "lag_bits": cell["lag_bits"],
        "channel": cell["channel"],
        "expansion_class": cell["expansion_class"],
        "mechanistic_signature": signature,
        "test_strong": cell["test_strong"],
        "reference_strong": cell["reference_strong"],
        "B_expansion_gain": cell["B_expansion_gain"],
        "matched_lz_expansion_gain": cell["matched_lz_expansion_gain"],
        "margin_expansion_gain": cell["margin_expansion_gain"],
        "test_B_retention": test.get("B_retention"),
        "reference_B_retention": reference.get("B_retention"),
        "test_B_reroute_share": test.get("B_reroute_share"),
        "reference_B_reroute_share": reference.get("B_reroute_share"),
        "test_matched_lz_shadow_seed": test.get("matched_lz_shadow_seed"),
        "reference_matched_lz_shadow_seed": reference.get("matched_lz_shadow_seed"),
        "test_matched_lz_shadow_retention": test.get("matched_lz_shadow_retention"),
        "reference_matched_lz_shadow_retention": reference.get("matched_lz_shadow_retention"),
        "matched_lz_shadow_retention_gain": none_if_missing(
            reference.get("matched_lz_shadow_retention"),
            test.get("matched_lz_shadow_retention"),
        ),
        "test_matched_lz_deficit_entropy_norm": test.get("matched_lz_shadow_deficit_entropy_norm"),
        "reference_matched_lz_deficit_entropy_norm": reference.get("matched_lz_shadow_deficit_entropy_norm"),
        "test_matched_lz_deficit_top1_share": test.get("matched_lz_shadow_deficit_top1_share"),
        "reference_matched_lz_deficit_top1_share": reference.get("matched_lz_shadow_deficit_top1_share"),
        "reference_deficit_shape": deficit_shape_label(
            reference.get("matched_lz_shadow_deficit_entropy_norm"),
            reference.get("matched_lz_shadow_deficit_top1_share"),
        ),
        "test_pattern_count": test.get("pattern_count"),
        "reference_pattern_count": reference.get("pattern_count"),
        "pattern_count_gain": none_if_missing(reference.get("pattern_count"), test.get("pattern_count")),
        "routing_evidence": routing_evidence_label(test, reference),
    }


def infer_mechanistic_signature(
    cell: dict[str, Any],
    test: dict[str, Any],
    reference: dict[str, Any],
) -> str:
    b_gain = float(cell["B_expansion_gain"] or 0.0)
    mlz_gain = float(cell["matched_lz_expansion_gain"] or 0.0)
    margin_gain = float(cell["margin_expansion_gain"] or 0.0)
    test_strong = bool(cell["test_strong"])
    reference_strong = bool(cell["reference_strong"])
    reroute = max(
        float_or_zero(test.get("B_reroute_share")),
        float_or_zero(reference.get("B_reroute_share")),
        float_or_zero(test.get("matched_lz_shadow_reroute_share")),
        float_or_zero(reference.get("matched_lz_shadow_reroute_share")),
    )
    if reroute >= 0.05:
        return "monitored-rerouting-present"
    if test_strong and reference_strong:
        return "stable-monitored-channel"
    if test_strong and not reference_strong and mlz_gain >= 0.10 and margin_gain <= -0.10:
        return "reference-null-shadowing"
    if not test_strong and reference_strong and mlz_gain <= -0.05:
        return "reference-null-shadow-relief"
    if b_gain >= 0.10:
        return "expanded-rank-shell-B-gain"
    if mlz_gain >= 0.10:
        return "expanded-rank-shell-null-gain"
    return "subthreshold-or-inactive"


def routing_evidence_label(test: dict[str, Any], reference: dict[str, Any]) -> str:
    shares = [
        float_or_zero(test.get("B_reroute_share")),
        float_or_zero(reference.get("B_reroute_share")),
        float_or_zero(test.get("matched_lz_shadow_reroute_share")),
        float_or_zero(reference.get("matched_lz_shadow_reroute_share")),
    ]
    if max(shares, default=0.0) >= 0.05:
        return "monitored-rerouting"
    if test or reference:
        return "same-destination-or-deficit"
    return "routing-artifact-missing"


def summarize_signatures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["test_top"], row["mechanistic_signature"])
        grouped.setdefault(key, []).append(row)
    output = []
    for group in grouped.values():
        first = group[0]
        output.append(
            {
                "test_top": first["test_top"],
                "reference_top": first["reference_top"],
                "mechanistic_signature": first["mechanistic_signature"],
                "cells": len(group),
                "bands": ",".join(sorted({str(row["band"]) for row in group}, key=band_start)),
                "strong_test_cells": sum(1 for row in group if row["test_strong"]),
                "strong_reference_cells": sum(1 for row in group if row["reference_strong"]),
                "max_B_expansion_gain": max(
                    [float(row["B_expansion_gain"]) for row in group if row["B_expansion_gain"] is not None],
                    default=None,
                ),
                "max_matched_lz_expansion_gain": max(
                    [
                        float(row["matched_lz_expansion_gain"])
                        for row in group
                        if row["matched_lz_expansion_gain"] is not None
                    ],
                    default=None,
                ),
                "dominant_reference_deficit_shape": dominant_text(
                    [str(row["reference_deficit_shape"]) for row in group]
                ),
            }
        )
    return sorted(output, key=lambda row: (int(row["test_top"]), row["mechanistic_signature"]))


def summarize_by_band(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["test_top"], row["reference_top"], row["band"], row["channel"])
        grouped.setdefault(key, []).append(row)
    output = []
    for group in grouped.values():
        first = group[0]
        classes = count_classes(group)
        b_gains = [float(row["B_expansion_gain"]) for row in group if row["B_expansion_gain"] is not None]
        mlz_gains = [
            float(row["matched_lz_expansion_gain"])
            for row in group
            if row["matched_lz_expansion_gain"] is not None
        ]
        output.append(
            {
                "test_top": first["test_top"],
                "reference_top": first["reference_top"],
                "band": first["band"],
                "channel": first["channel"],
                "cells": len(group),
                "test_strong_cells": sum(1 for row in group if row["test_strong"]),
                "reference_strong_cells": sum(1 for row in group if row["reference_strong"]),
                "max_B_expansion_gain": max(b_gains, default=None),
                "avg_B_expansion_gain": average(b_gains),
                "max_matched_lz_expansion_gain": max(mlz_gains, default=None),
                "avg_matched_lz_expansion_gain": average(mlz_gains),
                "class_counts": classes,
                "dominant_class": dominant_class(classes),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            int(row["test_top"]),
            band_start(row["band"]),
            0 if row["channel"] == "negative" else 1,
        ),
    )


def summarize_by_source(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["test_top"], row["reference_top"])
        grouped.setdefault(key, []).append(row)
    output = []
    for group in grouped.values():
        first = group[0]
        classes = count_classes(group)
        b_gains = [float(row["B_expansion_gain"]) for row in group if row["B_expansion_gain"] is not None]
        mlz_gains = [
            float(row["matched_lz_expansion_gain"])
            for row in group
            if row["matched_lz_expansion_gain"] is not None
        ]
        output.append(
            {
                "test_top": first["test_top"],
                "reference_top": first["reference_top"],
                "cells": len(group),
                "test_strong_cells": sum(1 for row in group if row["test_strong"]),
                "reference_strong_cells": sum(1 for row in group if row["reference_strong"]),
                "max_B_expansion_gain": max(b_gains, default=None),
                "avg_B_expansion_gain": average(b_gains),
                "max_matched_lz_expansion_gain": max(mlz_gains, default=None),
                "avg_matched_lz_expansion_gain": average(mlz_gains),
                "class_counts": classes,
                "dominant_class": dominant_class(classes),
            }
        )
    return sorted(output, key=lambda row: int(row["test_top"]))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def first_row(
    rows: list[dict[str, str]],
    *,
    variant: str,
    source_kind: str | None = None,
) -> dict[str, str] | None:
    for row in rows:
        if row.get("variant") != variant:
            continue
        if source_kind is not None and row.get("source_kind") != source_kind:
            continue
        return row
    return None


def matching_pattern_rows(
    rows: list[dict[str, str]],
    source_row: dict[str, str] | None,
) -> list[dict[str, str]]:
    if source_row is None:
        return []
    source_label = source_row.get("source_label", "")
    variant = source_row.get("variant", "")
    return [
        row for row in rows
        if row.get("variant") == variant and row.get("source_label") == source_label
    ]


def normalized_entropy_from_rows(rows: list[dict[str, str]], field: str) -> float | None:
    values = positive_values(rows, field)
    if not values:
        return None
    total = sum(values)
    if total <= 0:
        return None
    entropy = 0.0
    for value in values:
        probability = value / total
        entropy -= probability * math.log(probability)
    if len(values) <= 1:
        return 0.0
    return entropy / math.log(len(values))


def top_share_from_rows(rows: list[dict[str, str]], field: str, *, top_n: int) -> float | None:
    values = sorted(positive_values(rows, field), reverse=True)
    if not values:
        return None
    total = sum(values)
    if total <= 0:
        return None
    return sum(values[:top_n]) / total


def positive_values(rows: list[dict[str, str]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = parse_float(row.get(field))
        if value is not None and value > 0:
            values.append(value)
    return values


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def render_console_summary(
    source_rows: list[dict[str, Any]],
    band_rows: list[dict[str, Any]],
    signature_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "Phase 2 N2-13 monitored-universe expansion readout",
        "-" * 122,
        f"{'top':>5} {'ref':>5} {'cells':>5} {'testS':>6} {'refS':>5} {'maxBg':>8} {'maxNg':>8} {'dominant':>34}",
    ]
    for row in source_rows:
        lines.append(
            f"{int(row['test_top']):>5} {int(row['reference_top']):>5} "
            f"{int(row['cells']):>5} {int(row['test_strong_cells']):>6} "
            f"{int(row['reference_strong_cells']):>5} "
            f"{fmt(row['max_B_expansion_gain']):>8} "
            f"{fmt(row['max_matched_lz_expansion_gain']):>8} "
            f"{row['dominant_class']:>34}"
        )
    lines.extend(["", "Boundary-sensitive bands:"])
    for row in band_rows:
        if "low-k-only" in str(row["class_counts"]) or "reference-only" in str(row["class_counts"]):
            lines.append(
                f"- top={row['test_top']} band={row['band']} {row['channel']}: "
                f"test strong={row['test_strong_cells']}, ref strong={row['reference_strong_cells']}, "
                f"classes={row['class_counts']}"
            )
    if signature_rows:
        lines.extend(["", "N2-13a routing signatures:"])
        for row in signature_rows:
            lines.append(
                f"- top={row['test_top']} {row['mechanistic_signature']}: "
                f"cells={row['cells']} | bands={row['bands']} | "
                f"shape={row['dominant_reference_deficit_shape']}"
            )
    return "\n".join(lines)


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 2 N2-13 Monitored-Universe Expansion Readout",
        "",
        "This readout classifies N2-12b top-k sensitivity cells by comparing lower monitored top-k runs against the canonical reference top-k.",
        "",
        "It does not classify the full off-universe mass. It only classifies what changes when the monitored universe is expanded to the reference top-k.",
        "",
        "## Source Summary",
        "",
        "| Test top-k | Reference top-k | Cells | Test strong | Reference strong | Max B gain | Max matched-LZ gain | Dominant class |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["source_summary"]:
        lines.append(
            f"| {row['test_top']} | {row['reference_top']} | {row['cells']} | "
            f"{row['test_strong_cells']} | {row['reference_strong_cells']} | "
            f"{fmt(row['max_B_expansion_gain'])} | {fmt(row['max_matched_lz_expansion_gain'])} | "
            f"{row['dominant_class']} |"
        )
    lines.extend(
        [
            "",
            "## Band Summary",
            "",
            "| Test top-k | Band | Channel | Test strong | Reference strong | Max B gain | Max matched-LZ gain | Dominant class |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["band_summary"]:
        lines.append(
            f"| {row['test_top']} | {row['band']} | {row['channel']} | "
            f"{row['test_strong_cells']} | {row['reference_strong_cells']} | "
            f"{fmt(row['max_B_expansion_gain'])} | {fmt(row['max_matched_lz_expansion_gain'])} | "
            f"{row['dominant_class']} |"
        )
    if summary.get("signature_summary"):
        lines.extend(
            [
                "",
                "## N2-13a Routing Signatures",
                "",
                "These signatures reuse existing child-routing subruns. They are discriminative proxies over the monitored child universe, not a full classification of the unbounded off-top-k space.",
                "",
                "| Test top-k | Signature | Cells | Bands | Strong test | Strong reference | Max B gain | Max matched-LZ gain | Reference deficit shape |",
                "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in summary["signature_summary"]:
            lines.append(
                f"| {row['test_top']} | {row['mechanistic_signature']} | {row['cells']} | "
                f"{row['bands']} | {row['strong_test_cells']} | {row['strong_reference_cells']} | "
                f"{fmt(row['max_B_expansion_gain'])} | {fmt(row['max_matched_lz_expansion_gain'])} | "
                f"{row['dominant_reference_deficit_shape']} |"
            )
    lines.extend(
        [
            "",
            "## Class Vocabulary",
            "",
            "- `core-stable`: the cell is strong at both top-k resolutions.",
            "- `reference-only-strong`: the cell is strong only after expansion to the reference top-k, without a large B-retention jump.",
            "- `high-k-recovered-B`: expansion recovers enough observed B retention to cross the strong criterion.",
            "- `low-k-only-strong`: the cell is strong only at the lower top-k.",
            "- `low-k-only-shadowed-at-reference`: the lower-top strong cell loses margin after expansion, usually because the strongest matched-LZ background rises.",
            "- `subthreshold-B-expansion`: observed B grows under expansion but remains below the strong-cell criterion.",
            "- `subthreshold-null-shadow-expansion`: matched-LZ grows under expansion while the cell remains subthreshold.",
            "",
            "## Signature Vocabulary",
            "",
            "- `stable-monitored-channel`: the cell is strong both before and after expansion.",
            "- `reference-null-shadowing`: expansion raises the matched-LZ background enough to erase a lower-top strong cell.",
            "- `reference-null-shadow-relief`: expansion lowers the matched-LZ background enough for the reference cell to become strong.",
            "- `expanded-rank-shell-B-gain`: observed B grows in the expanded monitored rank shell but does not necessarily become strong.",
            "- `expanded-rank-shell-null-gain`: matched-LZ grows in the expanded monitored rank shell while the cell remains subthreshold.",
            "- `monitored-rerouting-present`: monitored reroute mass is non-negligible inside the existing child-routing readout.",
            "",
        ]
    )
    return "\n".join(lines)


def to_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def float_or_zero(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def value_or_none(left: float | None, margin: float | None) -> float | None:
    if left is None or margin is None:
        return None
    return left - margin


def none_if_missing(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def is_strong(
    b_retention: float | None,
    margin: float | None,
    b_threshold: float,
    margin_threshold: float,
) -> bool:
    return (
        b_retention is not None
        and margin is not None
        and b_retention >= b_threshold
        and margin >= margin_threshold
    )


def count_classes(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["expansion_class"])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def dominant_class(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def dominant_text(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        if not value or value == "-":
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return "-"
    return dominant_class(counts)


def deficit_shape_label(entropy_norm: Any, top1_share: Any) -> str:
    entropy = parse_optional_number(entropy_norm)
    top1 = parse_optional_number(top1_share)
    if entropy is None or top1 is None:
        return "no-deficit-or-unavailable"
    if entropy >= 0.80 and top1 <= 0.20:
        return "diffuse-deficit"
    if entropy <= 0.55 or top1 >= 0.35:
        return "concentrated-deficit"
    return "mixed-deficit"


def parse_optional_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def band_start(label: str) -> int:
    return int(label.split("M", 1)[0])


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


CELL_FIELDS = [
    "source_run",
    "test_top",
    "reference_top",
    "band",
    "lag",
    "lag_bits",
    "channel",
    "test_B",
    "reference_B",
    "B_expansion_gain",
    "test_matched_lz_max",
    "reference_matched_lz_max",
    "matched_lz_expansion_gain",
    "test_margin",
    "reference_margin",
    "margin_expansion_gain",
    "test_strong",
    "reference_strong",
    "expansion_class",
]

BAND_FIELDS = [
    "test_top",
    "reference_top",
    "band",
    "channel",
    "cells",
    "test_strong_cells",
    "reference_strong_cells",
    "max_B_expansion_gain",
    "avg_B_expansion_gain",
    "max_matched_lz_expansion_gain",
    "avg_matched_lz_expansion_gain",
    "class_counts",
    "dominant_class",
]

SOURCE_FIELDS = [
    "test_top",
    "reference_top",
    "cells",
    "test_strong_cells",
    "reference_strong_cells",
    "max_B_expansion_gain",
    "avg_B_expansion_gain",
    "max_matched_lz_expansion_gain",
    "avg_matched_lz_expansion_gain",
    "class_counts",
    "dominant_class",
]


SIGNATURE_FIELDS = [
    "source_run",
    "test_top",
    "reference_top",
    "band",
    "lag",
    "lag_bits",
    "channel",
    "expansion_class",
    "mechanistic_signature",
    "test_strong",
    "reference_strong",
    "B_expansion_gain",
    "matched_lz_expansion_gain",
    "margin_expansion_gain",
    "test_B_retention",
    "reference_B_retention",
    "test_B_reroute_share",
    "reference_B_reroute_share",
    "test_matched_lz_shadow_seed",
    "reference_matched_lz_shadow_seed",
    "test_matched_lz_shadow_retention",
    "reference_matched_lz_shadow_retention",
    "matched_lz_shadow_retention_gain",
    "test_matched_lz_deficit_entropy_norm",
    "reference_matched_lz_deficit_entropy_norm",
    "test_matched_lz_deficit_top1_share",
    "reference_matched_lz_deficit_top1_share",
    "reference_deficit_shape",
    "test_pattern_count",
    "reference_pattern_count",
    "pattern_count_gain",
    "routing_evidence",
]


SIGNATURE_SUMMARY_FIELDS = [
    "test_top",
    "reference_top",
    "mechanistic_signature",
    "cells",
    "bands",
    "strong_test_cells",
    "strong_reference_cells",
    "max_B_expansion_gain",
    "max_matched_lz_expansion_gain",
    "dominant_reference_deficit_shape",
]


if __name__ == "__main__":
    raise SystemExit(main())
