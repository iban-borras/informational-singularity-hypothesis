#!/usr/bin/env python3
"""
D-0154 Priority-A signature rederivation batch.

This script is a conservative wrapper around legacy Level 0/Level 1 metrics
used by the paper signature table. It does not change metric definitions.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from level1_deep_analysis import lempel_ziv_complexity, recurrence_analysis, test_multiple_constants
from level1_nesting_tree import analyze_nesting_tree
from utils.streaming_phi_loader import StreamingPhiLoader
from v2.signature.resumable_te import (
    build_checkpoint as build_te_checkpoint,
    load_checkpoint as load_te_checkpoint,
    restore_checkpoint as restore_te_checkpoint,
    write_checkpoint_atomic as write_te_checkpoint_atomic,
)


DEFAULT_TARGETS = "E:24,I:23,D:20,G:20,F:20"
DEFAULT_METRICS = "lz,cbar,te,det,fractal"
DEFAULT_LZ_SCALES = "1000,2000,5000,10000,20000,50000"
DEFAULT_TE_SCALES = "4,8,16,32,64,128"
DEFAULT_OUTPUT_ROOT = "results/hsi_v2/d0154_priority_a_signature"
RESUMABLE_TE_MODULE = REPO_DIR / "v2" / "signature" / "resumable_te.py"
PHI_PLUS_ONE = (1.0 + math.sqrt(5.0)) / 2.0 + 1.0

PAPER_VALUES: dict[str, dict[str, float | None]] = {
    "E": {"lz": 0.6149, "cbar": 2.618034, "te": 4.10, "det": 0.55, "fractal": None},
    "I": {"lz": 0.6270, "cbar": 2.618034, "te": 27.46, "det": 0.55, "fractal": None},
    "F": {"lz": 0.5729, "cbar": 2.618034, "te": 0.92, "det": 0.56, "fractal": None},
    "D": {"lz": 0.6052, "cbar": 2.618034, "te": None, "det": 0.55, "fractal": 0.992},
    "G": {"lz": 0.6149, "cbar": 2.618034, "te": None, "det": 0.55, "fractal": None},
}

DISPLAY_DECIMALS = {
    "lz": 4,
    "cbar": 6,
    "te": 2,
    "det": 2,
    "fractal": 3,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rederive D-0154 Priority-A signature metrics from clean snapshots."
    )
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--results-base", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume-run", default=None)
    parser.add_argument("--lz-scales", default=DEFAULT_LZ_SCALES)
    parser.add_argument("--det-bits", type=int, default=10_000)
    parser.add_argument("--cbar-segment-chars", type=int, default=1_000_000_000)
    parser.add_argument("--cbar-segments", type=int, default=4)
    parser.add_argument("--cbar-tree-timeout", type=int, default=600)
    parser.add_argument("--cbar-max-nodes", type=int, default=50_000_000)
    parser.add_argument("--cbar-bootstrap", type=int, default=0)
    parser.add_argument("--te-scales", default=DEFAULT_TE_SCALES)
    parser.add_argument("--te-history", type=int, default=3)
    parser.add_argument("--te-chunk-bits", type=int, default=10_000_000)
    parser.add_argument("--te-shuffle-seed", type=int, default=42)
    parser.add_argument("--te-max-bits", type=int, default=0, help="0 means full observable stream.")
    parser.add_argument("--te-progress-seconds", type=int, default=60)
    parser.add_argument("--te-checkpoint-chunks", type=int, default=25)
    parser.add_argument(
        "--te-reference-run",
        default=None,
        help=(
            "Optional prior TE run directory or summary.json to tabulate as a "
            "protocol-diagnostic reference, e.g. the D-0154 2M-chunk run."
        ),
    )
    parser.add_argument("--skip-source-hash", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    configure_stdout()
    load_dotenv()
    args = build_parser().parse_args()

    targets = parse_targets(args.targets)
    metrics = parse_tokens(args.metrics)
    if "all" in metrics:
        metrics = parse_tokens(DEFAULT_METRICS)

    results_base = resolve_results_base(args.results_base)
    output_root = resolve_results_path(args.output_dir, results_base)
    if args.resume_run:
        run_dir = Path(args.resume_run).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        run_dir = output_root / f"d0154-signature-priority-a__{timestamp}"

    if args.te_checkpoint_chunks <= 0:
        raise SystemExit("--te-checkpoint-chunks must be positive")
    if "te" in metrics and args.skip_source_hash:
        raise SystemExit("Resumable TE requires source hashing; remove --skip-source-hash")

    plan = {
        "status": "planned" if not args.execute else "running",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "decision_log_context": "D-0154 Priority A signature rederivation",
        "results_base": str(results_base),
        "run_dir": str(run_dir),
        "targets": targets,
        "metrics": metrics,
        "code_hashes": {
            "runner": sha256_file(Path(__file__)),
            "resumable_te": sha256_file(RESUMABLE_TE_MODULE),
        },
        "protocol": {
            "lz_scales": parse_ints(args.lz_scales),
            "det_bits": args.det_bits,
            "cbar_segment_chars": args.cbar_segment_chars,
            "cbar_segments": args.cbar_segments,
            "cbar_tree_timeout": args.cbar_tree_timeout,
            "cbar_max_nodes": args.cbar_max_nodes,
            "cbar_bootstrap": args.cbar_bootstrap,
            "te_scales": parse_ints(args.te_scales),
            "te_history": args.te_history,
            "te_chunk_bits": args.te_chunk_bits,
            "te_shuffle_seed": args.te_shuffle_seed,
            "te_max_bits": args.te_max_bits,
            "te_progress_seconds": args.te_progress_seconds,
            "te_checkpoint_chunks": args.te_checkpoint_chunks,
            "te_reference_run": args.te_reference_run,
            "source_hash_policy": "skipped" if args.skip_source_hash else "sha256",
        },
    }

    if not args.execute:
        print_plan(plan)
        return 0

    log_path = run_dir / "batch.log"
    manifest_path = run_dir / "manifest.json"
    resuming = args.resume_run is not None
    prior_manifest: dict[str, Any] | None = None
    if resuming:
        if not run_dir.is_dir() or not manifest_path.exists():
            raise SystemExit(f"Resume run is missing its manifest: {run_dir}")
        prior_manifest = read_json(manifest_path)
        if prior_manifest.get("generated_at"):
            plan["generated_at"] = prior_manifest["generated_at"]
        validate_resume_plan(prior_manifest, plan)
        log(log_path, f"[resume] Reopening interrupted run: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(manifest_path, plan)
        write_log_header(log_path, plan)

    log(log_path, "[phase] Discovering clean sources and hashing declared inputs")
    source_records = discover_sources(targets, results_base, args.skip_source_hash)
    if prior_manifest is not None and "sources" in prior_manifest and prior_manifest["sources"] != source_records:
        raise RuntimeError("Resume source records do not match the interrupted run")
    write_json(manifest_path, {**plan, "status": "running", "sources": source_records})
    log(log_path, "[phase] Source discovery complete")
    summary: dict[str, Any] = {
        "status": "running",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "targets": targets,
        "metrics": metrics,
        "sources": source_records,
        "rows": [],
        "display_comparison": [],
        "review_required": False,
        "notes": [
            "This is a read-only rederivation over clean D-0154 snapshots.",
            "Metric definitions are legacy-compatible; no new signature threshold is introduced.",
            "DET provenance is treated conservatively because the current paper value is a rounded legacy display cell.",
        ],
    }
    te_reference = load_te_reference_run(args.te_reference_run) if args.te_reference_run else None
    if te_reference is not None:
        summary["te_reference_run"] = te_reference

    all_details: dict[str, Any] = {}
    for variant, iteration in targets.items():
        record = source_records[variant]
        log(log_path, f"[variant] {variant}@{iteration}")
        variant_details: dict[str, Any] = {
            "variant": variant,
            "iteration": iteration,
            "source": record,
            "metrics": {},
        }

        if "lz" in metrics:
            cached = load_completed_metric(run_dir, variant, "lz", resuming, log_path)
            if cached is None:
                log(log_path, f"[metric] {variant}: LZ")
                cached = compute_lz_metric(
                    Path(record["struct_path"]),
                    parse_ints(args.lz_scales),
                    log_path,
                    args.quiet,
                )
                write_json(run_dir / f"{variant}_lz.json", cached)
            variant_details["metrics"]["lz"] = cached

        if "det" in metrics:
            log(log_path, f"[metric] {variant}: DET")
            variant_details["metrics"]["det"] = compute_det_metric(
                Path(record["struct_path"]),
                args.det_bits,
                log_path,
                args.quiet,
            )
            write_json(run_dir / f"{variant}_det.json", variant_details["metrics"]["det"])

        if "fractal" in metrics:
            log(log_path, f"[metric] {variant}: fractal")
            variant_details["metrics"]["fractal"] = compute_fractal_metric(Path(record["report_path"]))
            write_json(run_dir / f"{variant}_fractal.json", variant_details["metrics"]["fractal"])

        if "cbar" in metrics:
            log(log_path, f"[metric] {variant}: cbar")
            variant_details["metrics"]["cbar"] = compute_cbar_metric(
                Path(record["struct_path"]),
                args.cbar_segment_chars,
                args.cbar_segments,
                args.cbar_tree_timeout,
                args.cbar_max_nodes,
                args.cbar_bootstrap,
                log_path,
                args.quiet,
            )
            write_json(run_dir / f"{variant}_cbar.json", variant_details["metrics"]["cbar"])

        if "te" in metrics:
            te_checkpoint_path = run_dir / f"{variant}_te_checkpoint.json"
            te_identity = build_te_identity(
                Path(record["struct_path"]),
                parse_ints(args.te_scales),
                args.te_history,
                args.te_chunk_bits,
                args.te_shuffle_seed,
                args.te_max_bits,
                str(record["struct_sha256"]),
                int(record["struct_bytes"]),
                plan["code_hashes"],
            )
            cached = load_completed_metric(
                run_dir,
                variant,
                "te",
                resuming,
                log_path,
                completion_checkpoint=te_checkpoint_path,
                completion_identity=te_identity,
            )
            if cached is None:
                log(log_path, f"[metric] {variant}: TE")
                cached = compute_te_metric(
                    Path(record["struct_path"]),
                    parse_ints(args.te_scales),
                    args.te_history,
                    args.te_chunk_bits,
                    args.te_shuffle_seed,
                    args.te_max_bits,
                    args.te_progress_seconds,
                    log_path,
                    args.quiet,
                    checkpoint_path=te_checkpoint_path,
                    checkpoint_chunks=args.te_checkpoint_chunks,
                    source_sha256=str(record["struct_sha256"]),
                    source_size=int(record["struct_bytes"]),
                    code_hashes=plan["code_hashes"],
                    resume=resuming,
                )
                write_json(run_dir / f"{variant}_te.json", cached)
            variant_details["metrics"]["te"] = cached

        row = build_summary_row(variant_details)
        comparisons = compare_to_paper(row)
        summary["rows"].append(row)
        summary["display_comparison"].extend(comparisons)
        all_details[variant] = variant_details
        write_json(run_dir / "summary.json", summary)
        write_csv(run_dir / "signature_metric_rows.csv", summary["rows"])
        write_csv(run_dir / "signature_display_comparison.csv", summary["display_comparison"])

    review_statuses = {"mismatch", "missing-new-value", "new-value-not-in-paper-cell"}
    summary["review_required"] = any(item["status"] in review_statuses for item in summary["display_comparison"])
    if "te" in metrics:
        summary["te_protocol_reconciliation"] = build_te_protocol_reconciliation(
            summary,
            te_reference,
            int(args.te_chunk_bits),
        )
    summary["status"] = "completed"
    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_dir / "raw_metric_details.json", all_details)
    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "signature_metric_rows.csv", summary["rows"])
    write_csv(run_dir / "signature_display_comparison.csv", summary["display_comparison"])
    write_report(run_dir / "report.md", summary)

    manifest = {
        **plan,
        "status": "completed",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "script_sha256": sha256_file(Path(__file__)),
        "sources": source_records,
        "outputs": {},
    }
    write_json(manifest_path, manifest)
    manifest["outputs"] = hash_outputs(run_dir)
    write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    (run_dir / "manifest.sha256").write_text(f"{manifest_sha}  manifest.json\n", encoding="utf-8")

    print("D-0154 Priority-A signature batch completed.")
    print(f"Output: {run_dir}")
    print(f"Manifest SHA-256: {manifest_sha}")
    print(f"Review required: {summary['review_required']}")
    return 0


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def load_dotenv() -> None:
    env_path = REPO_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_results_base(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    env_value = os.environ.get("HSI_RESULTS_BASE_DIR") or os.environ.get("HSI_V1_RESULTS_BASE_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (REPO_DIR / "results").resolve()


def resolve_results_path(raw: str, results_base: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    parts = path.parts
    if parts and parts[0].lower() == "results":
        return (results_base / Path(*parts[1:])).resolve()
    return (REPO_DIR / path).resolve()


def parse_targets(raw: str) -> dict[str, int]:
    targets: dict[str, int] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        variant, iteration = token.split(":", 1)
        targets[variant.strip().upper()] = int(iteration)
    if not targets:
        raise SystemExit("No targets selected.")
    return targets


def parse_tokens(raw: str) -> list[str]:
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def validate_resume_plan(prior: dict[str, Any], current: dict[str, Any]) -> None:
    if prior.get("status") == "completed":
        raise RuntimeError("Cannot resume a completed signature batch")
    for key in ("script", "results_base", "run_dir", "targets", "metrics", "code_hashes", "protocol"):
        if prior.get(key) != current.get(key):
            raise RuntimeError(f"Resume plan mismatch in {key}")


def load_completed_metric(
    run_dir: Path,
    variant: str,
    metric: str,
    resuming: bool,
    log_path: Path,
    *,
    completion_checkpoint: Path | None = None,
    completion_identity: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    path = run_dir / f"{variant}_{metric}.json"
    if not resuming or not path.exists():
        return None
    if completion_checkpoint is not None:
        if not completion_checkpoint.exists():
            raise RuntimeError(
                f"Completed {metric} metric exists without its completion checkpoint: {path.name}"
            )
        checkpoint = load_te_checkpoint(completion_checkpoint)
        if checkpoint.get("status") != "completed":
            raise RuntimeError(
                f"Completed {metric} metric has a non-completed checkpoint: "
                f"{completion_checkpoint.name}"
            )
        if completion_identity is not None and checkpoint.get("identity") != completion_identity:
            raise RuntimeError(
                f"Completed {metric} checkpoint identity does not match this execution"
            )
    payload = read_json(path)
    log(log_path, f"[resume] Reusing completed metric: {path.name}")
    return payload


def build_te_identity(
    struct_path: Path,
    scales: list[int],
    history: int,
    chunk_bits: int,
    shuffle_seed: int,
    max_bits: int,
    source_sha256: str,
    source_size: int,
    code_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "source_path": str(struct_path.resolve()),
        "source_sha256": source_sha256.upper(),
        "source_size": int(source_size),
        "scales": [int(value) for value in scales],
        "history": int(history),
        "n_bins": 2,
        "method": "density",
        "chunk_bits": int(chunk_bits),
        "raw_chunk_bytes": int(max(1, chunk_bits // 2)),
        "shuffle_seed": int(shuffle_seed),
        "max_bits": int(max_bits),
        "code_hashes": dict(code_hashes),
    }


def discover_sources(targets: dict[str, int], results_base: Path, skip_hash: bool) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    snapshot_root = results_base / "level0" / "phi_snapshots"
    report_root = results_base / "level0" / "reports"
    for variant, iteration in targets.items():
        struct_path = snapshot_root / f"var_{variant}" / f"phi_iter{iteration}.struct.gz"
        metadata_path = snapshot_root / f"var_{variant}" / f"phi_iter{iteration}.json"
        report_path = latest_report(report_root, variant, iteration)
        if not struct_path.exists():
            raise FileNotFoundError(f"Missing clean snapshot for {variant}@{iteration}: {struct_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata for {variant}@{iteration}: {metadata_path}")
        if report_path is None:
            raise FileNotFoundError(f"Missing Level 0 report for {variant}@{iteration} under {report_root}")
        metadata = read_json(metadata_path)
        report = read_json(report_path)
        sequence_length = int(
            metadata.get("sequence_length")
            or metadata.get("final_length")
            or report.get("sequence_length")
            or report.get("final_length")
            or 0
        )
        records[variant] = {
            "variant": variant,
            "iteration": iteration,
            "struct_path": str(struct_path),
            "metadata_path": str(metadata_path),
            "report_path": str(report_path),
            "struct_bytes": struct_path.stat().st_size,
            "sequence_length": sequence_length,
            "metadata_sha256": sha256_file(metadata_path),
            "report_sha256": sha256_file(report_path),
            "struct_sha256": None if skip_hash else sha256_file(struct_path),
        }
    return records


def latest_report(report_root: Path, variant: str, iteration: int) -> Path | None:
    candidates = sorted(
        report_root.glob(f"variant_{variant}_{iteration}_*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def compute_lz_metric(struct_path: Path, scales: list[int], log_path: Path, quiet: bool) -> dict[str, Any]:
    max_bits = max(scales)
    bits, load_stats = load_observable_bits_prefix(struct_path, max_bits)
    by_scale: dict[str, dict[str, float | int]] = {}
    for index, scale in enumerate(scales, start=1):
        log(log_path, f"  LZ scale {index}/{len(scales)}: {scale:,} bits")
        segment = bits[:scale]
        raw_lz = lempel_ziv_complexity(segment)
        normalized = raw_lz / (scale / math.log2(scale)) if scale > 1 else 0.0
        by_scale[str(scale)] = {"raw_lz": int(raw_lz), "normalized_lz": float(normalized)}
    lz_ratios: list[float] = []
    for left, right in zip(scales, scales[1:]):
        left_norm = float(by_scale[str(left)]["normalized_lz"])
        right_norm = float(by_scale[str(right)]["normalized_lz"])
        if left_norm > 0:
            lz_ratios.append(right_norm / left_norm)
    constants = test_multiple_constants(lz_ratios, verbose=False)
    return {
        "metric": "legacy_lz_arbitrary_scales",
        "scales": scales,
        "load_stats": load_stats,
        "by_scale": by_scale,
        "lz_ratios": lz_ratios,
        "constants_test": constants,
        "value": float(constants["mean_ratio"]) if "mean_ratio" in constants else None,
    }


def compute_det_metric(struct_path: Path, det_bits: int, log_path: Path, quiet: bool) -> dict[str, Any]:
    bits, load_stats = load_observable_bits_prefix(struct_path, det_bits)
    result = recurrence_analysis(bits, max_bits=det_bits, verbose=not quiet)
    return {
        "metric": "legacy_recurrence_determinism_prefix",
        "det_bits": det_bits,
        "load_stats": load_stats,
        "recurrence": result,
        "value": float(result["determinism"]) if "determinism" in result else None,
        "provenance_note": "This reuses the legacy recurrence_analysis determinism readout; paper DET display provenance is checked conservatively.",
    }


def compute_fractal_metric(report_path: Path) -> dict[str, Any]:
    report = read_json(report_path)
    value = report.get("fractal_dimension")
    if value is None:
        value = report.get("fractal_dimension_streaming")
    return {
        "metric": "level0_report_fractal_dimension",
        "source_report": str(report_path),
        "value": None if value is None else float(value),
    }


def compute_cbar_metric(
    struct_path: Path,
    segment_chars: int,
    num_segments: int,
    tree_timeout: int,
    max_nodes: int,
    n_bootstrap: int,
    log_path: Path,
    quiet: bool,
) -> dict[str, Any]:
    metadata_path = struct_path.with_suffix("").with_suffix(".json")
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    total_chars = int(metadata.get("sequence_length") or metadata.get("final_length") or 0)
    segment_records: list[dict[str, Any]] = []
    for index, (position, segment) in enumerate(
        iter_structural_segments(struct_path, segment_chars, num_segments, total_chars, log_path),
        start=1,
    ):
        log(log_path, f"  cbar segment {index}/{num_segments}: pos={position:,}, chars={len(segment):,}")
        result = analyze_nesting_tree(
            segment,
            max_chars=None,
            n_bootstrap=n_bootstrap,
            verbose=not quiet,
            tree_timeout=tree_timeout,
            max_nodes=max_nodes,
        )
        value = None
        children_analysis = result.get("children_analysis", {})
        content_analysis = result.get("content_analysis", {})
        tree_stats = result.get("tree_stats", {})
        total_branching_nodes = children_analysis.get("total_branching_nodes")
        total_nodes = tree_stats.get("total_nodes")
        total_contents = content_analysis.get("total_contents")
        edge_to_branching_ratio = None
        content_to_branching_ratio = None
        if result.get("analysis_complete") and "children_analysis" in result:
            value = float(result["children_analysis"]["mean_children"])
            if total_branching_nodes:
                if total_nodes is not None:
                    edge_to_branching_ratio = (float(total_nodes) - 1.0) / float(total_branching_nodes)
                if total_contents is not None:
                    content_to_branching_ratio = float(total_contents) / float(total_branching_nodes)
        segment_records.append(
            {
                "segment_index": index,
                "position": position,
                "chars": len(segment),
                "analysis_complete": bool(result.get("analysis_complete", False)),
                "interruption_reason": result.get("interruption_reason"),
                "mean_children": value,
                "total_branching_nodes": total_branching_nodes,
                "total_nodes": total_nodes,
                "total_contents": total_contents,
                "edge_to_branching_ratio": edge_to_branching_ratio,
                "content_to_branching_ratio": content_to_branching_ratio,
                "phi_plus_one_distance": None if value is None else abs(value - PHI_PLUS_ONE),
                "children_analysis": children_analysis,
                "content_analysis": content_analysis,
                "tree_stats": tree_stats,
                "build_metadata": result.get("build_metadata", {}),
            }
        )
        del segment
    weighted_numerator = 0.0
    weighted_denominator = 0.0
    total_contents = 0.0
    total_nodes = 0.0
    for row in segment_records:
        if row["mean_children"] is None or not row["total_branching_nodes"]:
            continue
        branching = float(row["total_branching_nodes"])
        weighted_numerator += float(row["mean_children"]) * branching
        weighted_denominator += branching
        if row["total_contents"] is not None:
            total_contents += float(row["total_contents"])
        if row["total_nodes"] is not None:
            total_nodes += float(row["total_nodes"])
    weighted_value = weighted_numerator / weighted_denominator if weighted_denominator else None
    content_ratio = total_contents / weighted_denominator if weighted_denominator and total_contents else None
    edge_ratio = (total_nodes - len(segment_records)) / weighted_denominator if weighted_denominator and total_nodes else None
    return {
        "metric": "legacy_nesting_tree_mean_children",
        "status": "completed" if weighted_value is not None else "inputs-not-derivable",
        "segment_chars": segment_chars,
        "num_segments": num_segments,
        "segments": segment_records,
        "value": weighted_value,
        "unweighted_segment_mean": (
            float(np.mean([row["mean_children"] for row in segment_records if row["mean_children"] is not None]))
            if any(row["mean_children"] is not None for row in segment_records)
            else None
        ),
        "aggregate_branching_nodes": int(weighted_denominator) if weighted_denominator else 0,
        "aggregate_content_nodes": int(total_contents) if total_contents else None,
        "aggregate_nodes": int(total_nodes) if total_nodes else None,
        "aggregate_content_to_branching_ratio": content_ratio,
        "aggregate_edge_to_branching_ratio": edge_ratio,
        "phi_plus_one": PHI_PLUS_ONE,
        "phi_plus_one_distance": None if weighted_value is None else abs(weighted_value - PHI_PLUS_ONE),
        "completed_segments": sum(1 for row in segment_records if row["mean_children"] is not None),
    }


def compute_te_metric(
    struct_path: Path,
    scales: list[int],
    history: int,
    chunk_bits: int,
    shuffle_seed: int,
    max_bits: int,
    progress_seconds: int,
    log_path: Path,
    quiet: bool,
    *,
    checkpoint_path: Path,
    checkpoint_chunks: int,
    source_sha256: str,
    source_size: int,
    code_hashes: dict[str, str],
    resume: bool,
) -> dict[str, Any]:
    from level2_transfer_entropy import StreamingTEAccumulator

    chunk_bytes = max(1, chunk_bits // 2)
    target = max_bits if max_bits > 0 else None
    compressed_size = struct_path.stat().st_size
    if compressed_size != source_size:
        raise RuntimeError("TE source size changed after source discovery")
    identity = build_te_identity(
        struct_path,
        scales,
        history,
        chunk_bits,
        shuffle_seed,
        max_bits,
        source_sha256,
        source_size,
        code_hashes,
    )

    previous_elapsed = 0.0
    checkpoint_writes = 0
    resume_seek_seconds = 0.0
    checkpoint_restored = False
    if resume and checkpoint_path.exists():
        payload = load_te_checkpoint(checkpoint_path)
        accumulator, shuffle_accumulator, rng, progress = restore_te_checkpoint(
            payload, identity, StreamingTEAccumulator
        )
        consumed = int(progress["observable_bits_consumed"])
        yielded = int(progress["observable_bits_yielded"])
        chunk_count = int(progress["chunks_processed"])
        encoded_bytes_read = int(progress["encoded_bytes_read"])
        compressed_bytes_read = int(progress.get("compressed_bytes_read", 0))
        previous_elapsed = float(progress.get("elapsed_seconds", 0.0))
        checkpoint_writes = int(progress.get("checkpoint_writes", 0))
        checkpoint_restored = True
        log(
            log_path,
            f"  TE checkpoint restored: chunk={chunk_count:,}, "
            f"obs_bits={consumed:,}, encoded_bytes={encoded_bytes_read:,}",
        )
    else:
        if checkpoint_path.exists():
            raise RuntimeError(f"TE checkpoint exists but --resume-run was not supplied: {checkpoint_path}")
        accumulator = StreamingTEAccumulator(scales, k=history, n_bins=2)
        shuffle_accumulator = StreamingTEAccumulator(scales, k=history, n_bins=2)
        rng = np.random.default_rng(seed=shuffle_seed)
        consumed = 0
        yielded = 0
        chunk_count = 0
        encoded_bytes_read = 0
        compressed_bytes_read = 0

    started = time.time()
    last_progress = started
    log(
        log_path,
        f"  TE stream start: compressed={compressed_size:,} bytes, "
        f"target={'full stream' if target is None else f'{target:,} observable bits'}, "
        f"chunk_bytes={chunk_bytes:,}, checkpoint_every={checkpoint_chunks:,} chunks",
    )
    if encoded_bytes_read:
        log(log_path, f"  TE recovery seek: uncompressed encoded offset={encoded_bytes_read:,}")

    first_resumed_chunk = encoded_bytes_read > 0
    for bits, encoded_bytes_read, compressed_bytes_read in iter_bits_numpy_with_progress(
        struct_path, chunk_bytes, start_encoded_bytes=encoded_bytes_read
    ):
        if first_resumed_chunk:
            resume_seek_seconds = time.time() - started
            first_resumed_chunk = False
            log(log_path, f"  TE recovery seek complete in {format_duration(resume_seek_seconds)}")
        chunk_count += 1
        yielded += int(len(bits))
        if target is not None:
            remaining = target - consumed
            if remaining <= 0:
                break
            if len(bits) > remaining:
                bits = bits[:remaining]
        consumed += int(len(bits))
        accumulator.process_chunk(bits, method="density")
        shuffled = bits.copy()
        rng.shuffle(shuffled)
        shuffle_accumulator.process_chunk(shuffled, method="density")
        now = time.time()
        elapsed = previous_elapsed + (now - started)
        if chunk_count % checkpoint_chunks == 0:
            checkpoint_writes += 1
            write_te_checkpoint_atomic(
                checkpoint_path,
                build_te_checkpoint(
                    identity=identity,
                    observed=accumulator,
                    shuffled=shuffle_accumulator,
                    rng=rng,
                    progress={
                        "encoded_bytes_read": int(encoded_bytes_read),
                        "compressed_bytes_read": int(compressed_bytes_read),
                        "observable_bits_consumed": int(consumed),
                        "observable_bits_yielded": int(yielded),
                        "chunks_processed": int(chunk_count),
                        "elapsed_seconds": float(elapsed),
                        "checkpoint_writes": int(checkpoint_writes),
                    },
                    status="running",
                ),
            )
            log(
                log_path,
                f"  TE checkpoint {checkpoint_writes:,}: chunk={chunk_count:,}, obs_bits={consumed:,}",
            )
        if not quiet and (chunk_count == 1 or now - last_progress >= progress_seconds):
            last_progress = now
            progress_fraction = min(1.0, consumed / target) if target is not None and target > 0 else None
            compressed_fraction = min(1.0, compressed_bytes_read / compressed_size) if compressed_size > 0 else 0.0
            rate = consumed / elapsed if elapsed > 0 else 0.0
            eta = (
                (elapsed * (1.0 - progress_fraction) / progress_fraction)
                if progress_fraction is not None and progress_fraction > 0
                else None
            )
            log(
                log_path,
                "  TE progress: "
                f"chunk={chunk_count:,}, "
                f"obs_bits={consumed:,}, "
                f"encoded_bytes={encoded_bytes_read:,}, "
                f"compressed_pos={compressed_bytes_read:,}/{compressed_size:,} "
                f"({100.0 * compressed_fraction:.2f}% rough), "
                f"rate={rate / 1_000_000:.2f} Mobs/s, "
                f"elapsed={format_duration(elapsed)}, "
                f"eta={format_duration(eta) if eta is not None else 'unknown'}",
            )
        del shuffled
        if chunk_count % 200 == 0:
            gc.collect()

    observed = accumulator.compute_all_metrics()
    shuffled_metrics = shuffle_accumulator.compute_all_metrics()
    ratio_info = compute_te_ratio(observed["te_matrix"], shuffled_metrics["te_matrix"], scales)
    total_elapsed = previous_elapsed + (time.time() - started)
    checkpoint_writes += 1
    write_te_checkpoint_atomic(
        checkpoint_path,
        build_te_checkpoint(
            identity=identity,
            observed=accumulator,
            shuffled=shuffle_accumulator,
            rng=rng,
            progress={
                "encoded_bytes_read": int(encoded_bytes_read),
                "compressed_bytes_read": int(compressed_bytes_read),
                "observable_bits_consumed": int(consumed),
                "observable_bits_yielded": int(yielded),
                "chunks_processed": int(chunk_count),
                "elapsed_seconds": float(total_elapsed),
                "checkpoint_writes": int(checkpoint_writes),
            },
            status="completed",
        ),
    )
    return {
        "metric": "legacy_streaming_te_density_shuffle",
        "scales": scales,
        "history": history,
        "chunk_bits": chunk_bits,
        "shuffle_seed": shuffle_seed,
        "max_bits": max_bits,
        "observable_bits_consumed": consumed,
        "observable_bits_yielded": yielded,
        "encoded_bytes_read": encoded_bytes_read,
        "compressed_bytes_read": compressed_bytes_read,
        "compressed_size_bytes": compressed_size,
        "chunks_processed": chunk_count,
        "elapsed_seconds": round(total_elapsed, 3),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_writes": checkpoint_writes,
        "resumed": checkpoint_restored,
        "resume_seek_seconds": round(resume_seek_seconds, 3),
        "te_matrix": observed["te_matrix"].tolist(),
        "shuffle_te_matrix": shuffled_metrics["te_matrix"].tolist(),
        **ratio_info,
        "value": ratio_info["te_ratio_observed_vs_shuffle"],
    }


def iter_bits_numpy_with_progress(struct_path: Path, chunk_bytes: int, start_encoded_bytes: int = 0):
    with gzip.open(struct_path, "rb") as handle:
        if start_encoded_bytes:
            restored = int(handle.seek(start_encoded_bytes))
            if restored != start_encoded_bytes:
                raise RuntimeError(
                    f"TE recovery seek mismatch: expected {start_encoded_bytes}, got {restored}"
                )
        encoded_bytes_read = start_encoded_bytes
        while True:
            raw = handle.read(chunk_bytes)
            if not raw:
                break
            encoded_bytes_read += len(raw)
            compressed_position = 0
            fileobj = getattr(handle, "fileobj", None)
            tell = getattr(fileobj, "tell", None)
            if tell is not None:
                try:
                    compressed_position = int(tell())
                except OSError:
                    compressed_position = 0
            byte_array = np.frombuffer(raw, dtype=np.uint8)
            pair0 = (byte_array >> 6) & 0b11
            pair1 = (byte_array >> 4) & 0b11
            pair2 = (byte_array >> 2) & 0b11
            pair3 = byte_array & 0b11
            all_pairs = np.column_stack([pair0, pair1, pair2, pair3]).ravel()
            mask = all_pairs <= 1
            bits = all_pairs[mask].astype(np.int8)
            if len(bits) > 0:
                yield bits, encoded_bytes_read, compressed_position


def compute_te_ratio(te_matrix: np.ndarray, shuffle_te_matrix: np.ndarray, scales: list[int]) -> dict[str, float]:
    n_scales = len(scales)
    bottom_up: list[float] = []
    top_down: list[float] = []
    shuffle_bottom_up: list[float] = []
    shuffle_top_down: list[float] = []
    for i in range(n_scales):
        for j in range(n_scales):
            if i < j:
                bottom_up.append(float(te_matrix[i, j]))
                shuffle_bottom_up.append(float(shuffle_te_matrix[i, j]))
            elif i > j:
                top_down.append(float(te_matrix[i, j]))
                shuffle_top_down.append(float(shuffle_te_matrix[i, j]))
    avg_bottom_up = float(np.mean(bottom_up)) if bottom_up else 0.0
    avg_top_down = float(np.mean(top_down)) if top_down else 0.0
    avg_observed_te = (avg_bottom_up + avg_top_down) / 2
    avg_shuffle_bottom_up = float(np.mean(shuffle_bottom_up)) if shuffle_bottom_up else 0.0
    avg_shuffle_top_down = float(np.mean(shuffle_top_down)) if shuffle_top_down else 0.0
    avg_shuffle_te = (avg_shuffle_bottom_up + avg_shuffle_top_down) / 2
    if avg_shuffle_te > 1e-10:
        ratio = avg_observed_te / avg_shuffle_te
    else:
        ratio = float("inf") if avg_observed_te > 1e-10 else 1.0
    return {
        "avg_bottom_up": avg_bottom_up,
        "avg_top_down": avg_top_down,
        "avg_observed_te": float(avg_observed_te),
        "avg_shuffle_bottom_up": avg_shuffle_bottom_up,
        "avg_shuffle_top_down": avg_shuffle_top_down,
        "avg_shuffle_te": float(avg_shuffle_te),
        "te_ratio_observed_vs_shuffle": float(ratio),
    }


def load_observable_bits_prefix(struct_path: Path, max_bits: int) -> tuple[str, dict[str, Any]]:
    loader = StreamingPhiLoader(str(struct_path))
    chunks: list[bytes] = []
    consumed = 0
    yielded = 0
    chunk_count = 0
    for bits in loader.iter_bits_numpy():
        chunk_count += 1
        yielded += int(len(bits))
        remaining = max_bits - consumed
        if remaining <= 0:
            break
        if len(bits) > remaining:
            bits = bits[:remaining]
        chunks.append((bits + ord("0")).astype(np.uint8).tobytes())
        consumed += int(len(bits))
        if consumed >= max_bits:
            break
    if consumed < max_bits:
        raise RuntimeError(f"{struct_path}: expected {max_bits} observable bits, got {consumed}")
    return b"".join(chunks).decode("ascii"), {
        "observable_bits_consumed": consumed,
        "observable_bits_yielded_before_trim": yielded,
        "chunks_read": chunk_count,
    }


def iter_structural_segments(
    struct_path: Path,
    segment_size: int,
    num_segments: int,
    total_chars: int,
    log_path: Path,
):
    if total_chars <= 0:
        compressed_size = struct_path.stat().st_size
        total_chars = compressed_size * 16
    capture_positions = [
        max(0, int((i / num_segments) * max(0, total_chars - segment_size)))
        for i in range(num_segments)
    ]
    chunk_size = 4_194_304
    char_position = 0
    segment_index = 0
    buffer = bytearray()
    capturing = False
    progress_step = 25_000_000
    scan_step = 250_000_000
    next_scan_log = scan_step
    next_capture_log = progress_step
    with gzip.open(struct_path, "rb") as handle:
        while segment_index < num_segments:
            raw = handle.read(chunk_size)
            if not raw:
                break
            byte_array = np.frombuffer(raw, dtype=np.uint8)
            pairs = np.column_stack(
                [
                    (byte_array >> 6) & 0b11,
                    (byte_array >> 4) & 0b11,
                    (byte_array >> 2) & 0b11,
                    byte_array & 0b11,
                ]
            ).ravel()
            for pair in pairs:
                if not capturing and char_position >= capture_positions[segment_index]:
                    capturing = True
                    next_capture_log = progress_step
                    log(log_path, f"  capturing cbar segment {segment_index + 1}/{num_segments} at {char_position:,}")
                elif not capturing and char_position >= next_scan_log:
                    log(
                        log_path,
                        f"  scanned {char_position:,}/{total_chars:,} chars; "
                        f"next segment {segment_index + 1}/{num_segments} starts at {capture_positions[segment_index]:,}",
                    )
                    next_scan_log += scan_step
                if capturing:
                    if pair == 0:
                        buffer.append(48)
                    elif pair == 1:
                        buffer.append(49)
                    elif pair == 2:
                        buffer.append(40)
                    else:
                        buffer.append(41)
                    if len(buffer) >= next_capture_log:
                        log(
                            log_path,
                            f"  segment {segment_index + 1}/{num_segments}: "
                            f"{len(buffer):,}/{segment_size:,} chars captured "
                            f"({100.0 * len(buffer) / segment_size:.1f}%)",
                        )
                        next_capture_log += progress_step
                    if len(buffer) >= segment_size:
                        yield capture_positions[segment_index], buffer.decode("ascii")
                        buffer = bytearray()
                        capturing = False
                        segment_index += 1
                        if segment_index >= num_segments:
                            break
                char_position += 1
    if segment_index < num_segments and buffer:
        yield capture_positions[segment_index], buffer.decode("ascii")


def build_summary_row(details: dict[str, Any]) -> dict[str, Any]:
    metrics = details["metrics"]
    row: dict[str, Any] = {
        "variant": details["variant"],
        "iteration": details["iteration"],
    }
    for key in ["lz", "cbar", "te", "det", "fractal"]:
        metric = metrics.get(key)
        value = None if metric is None else metric.get("value")
        row[key] = value
        row[f"{key}_display"] = display_value(key, value)
        row[f"{key}_run_status"] = "not-run" if metric is None else metric.get("status", "completed")
    return row


def compare_to_paper(row: dict[str, Any]) -> list[dict[str, Any]]:
    variant = row["variant"]
    expected = PAPER_VALUES.get(variant, {})
    comparisons: list[dict[str, Any]] = []
    for metric in ["lz", "cbar", "te", "det", "fractal"]:
        old = expected.get(metric)
        new = row.get(metric)
        if row.get(f"{metric}_run_status") == "not-run":
            status = "not-run"
        elif old is None and new is None:
            status = "not-applicable"
        elif old is None and new is not None:
            status = "new-value-not-in-paper-cell"
        elif old is not None and new is None:
            status = "missing-new-value"
        else:
            decimals = DISPLAY_DECIMALS[metric]
            status = "match" if round(float(old), decimals) == round(float(new), decimals) else "mismatch"
        comparisons.append(
            {
                "variant": variant,
                "metric": metric,
                "paper_value": old,
                "new_value": new,
                "paper_display": display_value(metric, old),
                "new_display": display_value(metric, new),
                "status": status,
            }
        )
    return comparisons


def load_te_reference_run(raw: str) -> dict[str, Any]:
    reference_path = Path(raw).expanduser().resolve()
    summary_path = reference_path / "summary.json" if reference_path.is_dir() else reference_path
    if not summary_path.exists():
        raise FileNotFoundError(f"TE reference summary not found: {summary_path}")
    run_dir = summary_path.parent
    manifest_path = run_dir / "manifest.json"
    reference_summary = read_json(summary_path)
    reference_manifest = read_json(manifest_path) if manifest_path.exists() else {}
    by_variant: dict[str, dict[str, Any]] = {}
    for row in reference_summary.get("rows", []):
        variant = str(row.get("variant", "")).upper()
        if not variant:
            continue
        by_variant[variant] = {
            "iteration": row.get("iteration"),
            "te": row.get("te"),
            "te_display": row.get("te_display"),
            "te_run_status": row.get("te_run_status"),
        }
    protocol = reference_manifest.get("protocol", {})
    return {
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        "te_chunk_bits": protocol.get("te_chunk_bits"),
        "te_shuffle_seed": protocol.get("te_shuffle_seed"),
        "te_history": protocol.get("te_history"),
        "te_scales": protocol.get("te_scales"),
        "by_variant": by_variant,
    }


def build_te_protocol_reconciliation(
    summary: dict[str, Any],
    reference: dict[str, Any] | None,
    current_chunk_bits: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    reference_by_variant = {} if reference is None else reference.get("by_variant", {})
    for row in summary.get("rows", []):
        variant = str(row.get("variant", "")).upper()
        if not variant or row.get("te_run_status") == "not-run":
            continue
        paper_te = PAPER_VALUES.get(variant, {}).get("te")
        reference_row = reference_by_variant.get(variant, {})
        reference_te = reference_row.get("te")
        rows.append(
            {
                "variant": variant,
                "paper_display_value": paper_te,
                "paper_display": display_value("te", paper_te),
                "clean_iteration": row.get("iteration"),
                "clean_chunk_bits": current_chunk_bits,
                "clean_te": row.get("te"),
                "clean_display": display_value("te", row.get("te")),
                "reference_chunk_bits": None if reference is None else reference.get("te_chunk_bits"),
                "reference_iteration": reference_row.get("iteration"),
                "reference_te": reference_te,
                "reference_display": display_value("te", reference_te),
                "paper_vs_clean_delta": (
                    None if paper_te is None or row.get("te") is None else float(row.get("te")) - float(paper_te)
                ),
                "reference_vs_clean_delta": (
                    None if reference_te is None or row.get("te") is None else float(reference_te) - float(row.get("te"))
                ),
            }
        )
    return {
        "purpose": "D-0154 TE provenance reconciliation: paper display vs clean legacy-chunk rerun vs diagnostic reference run.",
        "paper_chunk_bits_assumption": 10_000_000,
        "current_chunk_bits": current_chunk_bits,
        "reference_run_dir": None if reference is None else reference.get("run_dir"),
        "reference_summary_sha256": None if reference is None else reference.get("summary_sha256"),
        "rows": rows,
        "provenance_note": (
            "The manuscript table stores display values, not full TE provenance. "
            "Historical appendix material labels the E TE cell as E@23; this "
            "reconciliation reports the actual clean source iteration separately "
            "so Sofia/Grace can separate depth effects from chunk-size effects."
        ),
    }


def display_value(metric: str, value: Any) -> str:
    if value is None:
        return "---"
    return f"{float(value):.{DISPLAY_DECIMALS[metric]}f}"


def format_duration(seconds: float | None) -> str:
    if seconds is None or math.isnan(float(seconds)) or math.isinf(float(seconds)):
        return "unknown"
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def write_report(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    comparisons = summary["display_comparison"]
    review_statuses = {"mismatch", "missing-new-value", "new-value-not-in-paper-cell"}
    mismatch_count = sum(1 for item in comparisons if item["status"] in review_statuses)
    lines = [
        "# D-0154 Priority-A signature batch",
        "",
        f"- Status: `{summary['status']}`",
        f"- Run directory: `{summary['run_dir']}`",
        f"- Review required: `{summary['review_required']}`",
        f"- Display items requiring review: `{mismatch_count}`",
        "",
        "## Protocol",
        "",
        "- Read-only over clean D-0154 Level 0 snapshots.",
        "- Legacy metric definitions only; no new signature threshold is introduced.",
        "- Display comparison uses the manuscript precision for each metric.",
        "- `new-value-not-in-paper-cell` means the clean value exists but the current table intentionally prints `---` or only mentions it in prose.",
        "",
        "## Metric Rows",
        "",
        "| Variant | Iter | LZ | cbar | TE | D_f | DET |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['iteration']} | {row['lz_display']} | "
            f"{row['cbar_display']} | {row['te_display']} | {row['fractal_display']} | {row['det_display']} |"
        )
    lines.extend(["", "## Display Comparison", ""])
    lines.append("| Variant | Metric | Paper | Clean | Status |")
    lines.append("|---|---|---:|---:|---|")
    for item in comparisons:
        lines.append(
            f"| {item['variant']} | {item['metric']} | {item['paper_display']} | "
            f"{item['new_display']} | `{item['status']}` |"
        )
    reconciliation = summary.get("te_protocol_reconciliation")
    if reconciliation:
        lines.extend(["", "## TE Protocol Reconciliation", ""])
        lines.append(reconciliation["provenance_note"])
        lines.extend(
            [
                "",
                "| Variant | Paper@10M display | Clean iter | Clean chunk | Clean TE | Reference chunk | Reference TE | Paper-clean delta | Ref-clean delta |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in reconciliation.get("rows", []):
            reference_chunk = row.get("reference_chunk_bits")
            reference_chunk_display = "---" if reference_chunk is None else f"{int(reference_chunk):,}"
            paper_delta = row.get("paper_vs_clean_delta")
            paper_delta_display = "---" if paper_delta is None else f"{float(paper_delta):.6g}"
            reference_delta = row.get("reference_vs_clean_delta")
            reference_delta_display = "---" if reference_delta is None else f"{float(reference_delta):.6g}"
            lines.append(
                f"| {row['variant']} | {row['paper_display']} | {row['clean_iteration']} | "
                f"{row['clean_chunk_bits']:,} | {display_value('te', row['clean_te'])} | "
                f"{reference_chunk_display} | "
                f"{display_value('te', row['reference_te'])} | "
                f"{paper_delta_display} | "
                f"{reference_delta_display} |"
            )
        if reconciliation.get("reference_run_dir"):
            lines.extend(
                [
                    "",
                    f"- Reference run: `{reconciliation['reference_run_dir']}`",
                    f"- Reference summary SHA-256: `{reconciliation['reference_summary_sha256']}`",
                ]
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- If any visible table cell changes, editorial freeze remains active until Sofia/Grace review.",
            "- DET is intentionally conservative: a mismatch here indicates provenance review, not automatic scientific change.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)
        handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def log(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def write_log_header(log_path: Path, plan: dict[str, Any]) -> None:
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("# D-0154 Priority-A signature batch\n\n")
        handle.write(json.dumps(json_safe(plan), indent=2))
        handle.write("\n\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def hash_outputs(run_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "manifest.sha256", "batch.log"}:
            continue
        hashes[path.relative_to(run_dir).as_posix()] = sha256_file(path)
    return hashes


def print_plan(plan: dict[str, Any]) -> None:
    print("D-0154 Priority-A signature batch plan (dry-run)")
    print("-" * 76)
    print(f"run_dir: {plan['run_dir']}")
    print(f"targets: {plan['targets']}")
    print(f"metrics: {plan['metrics']}")
    print("\nAdd --execute to run.")


if __name__ == "__main__":
    raise SystemExit(main())
