#!/usr/bin/env python3
"""
Q6-A Stage 0 closure for the clean E@24 Level-0 snapshot.

This is a read-only audit artifact generator. It does not run Phase 1, Phase 4,
or any downstream readout. Its job is to freeze the post-patch E@24 source and
quantify the pre-patch E legacy divergence that motivated the clean rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.cli import resolve_results_dir


DEFAULT_OUTPUT_DIR = "results/hsi_v2/q6a_stage0_e24_closure"
DEFAULT_LEVEL0_ROOT = "results/level0/phi_snapshots"
DEFAULT_REPORTS_DIR = "results/level0/reports"
DEFAULT_LEGACY_DIR = "results/level0/legacy_invalidated/q6a_prepatch_E_legacy_20260607T172555"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze and summarize clean E@24 for Q6-A Stage 0.")
    parser.add_argument("--variant", default="E")
    parser.add_argument("--reference-variant", default="B")
    parser.add_argument("--iteration", type=int, default=24)
    parser.add_argument("--level0-root", default=DEFAULT_LEVEL0_ROOT)
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--legacy-dir", default=DEFAULT_LEGACY_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-large-hash",
        action="store_true",
        help="Skip SHA-256 over the large phi_iterN.struct.gz snapshot.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    variant = args.variant.strip().upper()
    ref_variant = args.reference_variant.strip().upper()
    iteration = args.iteration

    level0_root = resolve_results_dir(args.level0_root, anchor_file=__file__)
    reports_dir = resolve_results_dir(args.reports_dir, anchor_file=__file__)
    legacy_dir = resolve_results_dir(args.legacy_dir, anchor_file=__file__)
    output_root = resolve_results_dir(args.output_dir, anchor_file=__file__)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"q6a-stage0-e24-closure__var-{variant}__iter-{iteration}__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    current = collect_variant_snapshot(level0_root, reports_dir, variant, iteration, hash_large=not args.skip_large_hash)
    reference = collect_variant_snapshot(level0_root, reports_dir, ref_variant, iteration, hash_large=False)
    legacy_rows = collect_legacy_comparison(level0_root, legacy_dir, variant, iteration)

    summary = {
        "status": "stage0-source-frozen",
        "cycle": "Q6-A",
        "stage": "0",
        "variant": variant,
        "iteration": iteration,
        "source_current": current,
        "reference_variant": reference,
        "current_vs_reference": compare_current_to_reference(current, reference),
        "legacy_invalidated_dir": str(legacy_dir),
        "legacy_comparison": legacy_rows,
        "scientific_reading": {
            "clean_e24_available": True,
            "legacy_e_invalidated": bool(legacy_rows),
            "legacy_divergence_reading": (
                "Pre-patch E legacy is not a near-neighbor of clean E at high iterations; "
                "it follows a much smaller large-state trajectory and must remain quarantined."
            ),
            "no_downstream_readout_performed": True,
        },
    }

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "hsi_v2_q6a_stage0_e24_closure.py",
        "contract": "D-0152/D-0153 Stage 0 closure artifact",
        "inputs": {
            "level0_root": str(level0_root),
            "reports_dir": str(reports_dir),
            "legacy_dir": str(legacy_dir),
        },
        "outputs": {
            "summary": str(run_dir / "summary.json"),
            "manifest": str(run_dir / "manifest.json"),
            "report": str(run_dir / "report.md"),
            "legacy_comparison_csv": str(run_dir / "legacy_comparison.csv"),
        },
        "constraints": {
            "read_only": True,
            "phase1_not_run": True,
            "phase4_not_observed": True,
            "no_p4_above_804m_observed": True,
        },
    }

    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "manifest.json", manifest)
    write_legacy_csv(run_dir / "legacy_comparison.csv", legacy_rows)
    (run_dir / "report.md").write_text(render_report(summary), encoding="utf-8")

    print("Q6-A Stage 0 E@24 closure")
    print("-" * 72)
    print(f"status: {summary['status']}")
    print(f"E@24 observable bits: {current.get('observable_bits'):,}")
    print(f"E@24 structural length: {current.get('structural_length'):,}")
    print(f"E@24 struct.gz bytes: {current.get('struct_gz_bytes'):,}")
    print(f"E@24 struct sha256: {current.get('struct_sha256') or 'SKIPPED'}")
    print(f"B/E observable delta: {summary['current_vs_reference'].get('observable_bits_delta'):,}")
    print(f"output: {run_dir}")
    return 0


def collect_variant_snapshot(
    level0_root: Path,
    reports_dir: Path,
    variant: str,
    iteration: int,
    *,
    hash_large: bool,
) -> dict[str, Any]:
    variant_dir = level0_root / f"var_{variant}"
    struct_path = variant_dir / f"phi_iter{iteration}.struct.gz"
    metadata_path = variant_dir / f"phi_iter{iteration}.json"
    state_path = variant_dir / f"phi_iter{iteration}.state.txt.gz"
    report_path = latest_report_path(reports_dir, variant, iteration)
    log_path = reports_dir / f"variant_{variant}_execution.log.md"

    if not struct_path.exists():
        raise FileNotFoundError(struct_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if report_path is None:
        raise FileNotFoundError(f"No variant_{variant}_{iteration}_*.json under {reports_dir}")

    metadata = load_json(metadata_path)
    report = load_json(report_path)
    return {
        "variant": variant,
        "iteration": iteration,
        "struct_path": str(struct_path),
        "metadata_path": str(metadata_path),
        "report_path": str(report_path),
        "execution_log_path": str(log_path) if log_path.exists() else None,
        "structural_length": int(metadata.get("sequence_length") or 0),
        "observable_bits": int(report.get("total_bits") or 0),
        "parenthesis_symbols": int(metadata.get("sequence_length") or 0) - int(report.get("total_bits") or 0),
        "format": metadata.get("format"),
        "compression_level": metadata.get("compression_level"),
        "compression_ratio": metadata.get("compression_ratio"),
        "struct_gz_bytes": struct_path.stat().st_size,
        "state_checkpoint_exists": state_path.exists(),
        "current_state_length": metadata.get("current_state_length"),
        "current_state_sha256": metadata.get("current_state_sha256"),
        "struct_sha256": sha256_file(struct_path) if hash_large else None,
        "metadata_sha256": sha256_file(metadata_path),
        "state_sha256": sha256_file(state_path) if state_path.exists() else None,
        "report_sha256": sha256_file(report_path),
        "execution_log_sha256": sha256_file(log_path) if log_path.exists() else None,
    }


def collect_legacy_comparison(level0_root: Path, legacy_dir: Path, variant: str, iteration: int) -> list[dict[str, Any]]:
    current_dir = level0_root / f"var_{variant}"
    legacy_variant_dir = legacy_dir / "phi_snapshots" / f"var_{variant}"
    rows: list[dict[str, Any]] = []
    if not legacy_variant_dir.exists():
        return rows

    for i in range(1, iteration):
        current_meta = current_dir / f"phi_iter{i}.json"
        legacy_meta = legacy_variant_dir / f"phi_iter{i}.json"
        if not current_meta.exists() or not legacy_meta.exists():
            continue
        current_payload = load_json(current_meta)
        legacy_payload = load_json(legacy_meta)
        current_struct = current_dir / f"phi_iter{i}.struct.gz"
        legacy_struct = legacy_variant_dir / f"phi_iter{i}.struct.gz"
        current_len = int(current_payload.get("sequence_length") or current_payload.get("total_bits") or 0)
        legacy_len = int(legacy_payload.get("sequence_length") or legacy_payload.get("total_bits") or 0)
        rows.append(
            {
                "iteration": i,
                "legacy_structural_length": legacy_len,
                "clean_structural_length": current_len,
                "clean_over_legacy": (current_len / legacy_len) if legacy_len else None,
                "legacy_struct_gz_bytes": legacy_struct.stat().st_size if legacy_struct.exists() else None,
                "clean_struct_gz_bytes": current_struct.stat().st_size if current_struct.exists() else None,
            }
        )
    return rows


def compare_current_to_reference(current: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    def ratio(key: str) -> float | None:
        denominator = reference.get(key)
        return (current.get(key) / denominator) if denominator else None

    return {
        "observable_bits_ratio": ratio("observable_bits"),
        "observable_bits_delta": current.get("observable_bits") - reference.get("observable_bits"),
        "structural_length_ratio": ratio("structural_length"),
        "structural_length_delta": current.get("structural_length") - reference.get("structural_length"),
        "struct_gz_bytes_ratio": ratio("struct_gz_bytes"),
        "struct_gz_bytes_delta": current.get("struct_gz_bytes") - reference.get("struct_gz_bytes"),
    }


def render_report(summary: dict[str, Any]) -> str:
    current = summary["source_current"]
    reference = summary["reference_variant"]
    comparison = summary["current_vs_reference"]
    legacy_tail = [row for row in summary["legacy_comparison"] if row["iteration"] >= 18]
    legacy_rows = "\n".join(
        (
            f"| {row['iteration']} | {row['legacy_structural_length']:,} | "
            f"{row['clean_structural_length']:,} | {row['clean_over_legacy']:.3f} |"
        )
        for row in legacy_tail
    )
    return f"""# Q6-A Stage 0 E@24 Closure

Status: `{summary['status']}`.

This artifact freezes the clean post-patch E@24 Level-0 snapshot and compares it
against the quarantined pre-patch E legacy chain. No Phase 1, Phase 4, or
predictive readout is executed here.

## Clean E@24 Source

- Structural snapshot: `{current['struct_path']}`
- Observable bits: `{current['observable_bits']:,}`
- Structural length: `{current['structural_length']:,}`
- `struct.gz` bytes: `{current['struct_gz_bytes']:,}`
- `struct.gz` SHA-256: `{current['struct_sha256'] or 'SKIPPED'}`
- Metadata SHA-256: `{current['metadata_sha256']}`
- Report SHA-256: `{current['report_sha256']}`
- State checkpoint present: `{current['state_checkpoint_exists']}`
- Current-state length: `{current['current_state_length']}`

## B@24 Reference Comparison

- B@24 observable bits: `{reference['observable_bits']:,}`
- E@24 observable bits: `{current['observable_bits']:,}`
- E/B observable ratio: `{comparison['observable_bits_ratio']:.12f}`
- E-B observable delta: `{comparison['observable_bits_delta']:,}`
- E/B structural ratio: `{comparison['structural_length_ratio']:.12f}`
- E-B structural delta: `{comparison['structural_length_delta']:,}`

## Quarantined E Legacy Divergence

| iter | legacy structural | clean structural | clean / legacy |
|---:|---:|---:|---:|
{legacy_rows}

The legacy chain is not a near-neighbor of the clean post-patch E trajectory at
large iterations. It must remain quarantined and must not be used as Q6-A
infrastructure.

## Next Gate

Stage 1 requires a separate frozen contract before any high-scale B@24/E@24
projective tower readout is executed.
"""


def latest_report_path(reports_dir: Path, variant: str, iteration: int) -> Path | None:
    reports = sorted(
        reports_dir.glob(f"variant_{variant}_{iteration}_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return reports[0] if reports else None


def sha256_file(path: Path, chunk_size: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_legacy_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "iteration",
        "legacy_structural_length",
        "clean_structural_length",
        "clean_over_legacy",
        "legacy_struct_gz_bytes",
        "clean_struct_gz_bytes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(headers) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(header) or "") for header in headers) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
