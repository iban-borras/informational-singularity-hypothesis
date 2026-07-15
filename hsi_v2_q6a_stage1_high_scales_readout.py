#!/usr/bin/env python3
"""
Q6-A Stage 1 readout over B@24/E@24 high-scale Phase 1 towers.

Read-only classifier for the frozen Stage 1 contract. It consumes completed
`hsi_v2_phase1_run.py` outputs and applies the pre-registered outcome rules.
No Phase 4 data is read or generated here.
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


DEFAULT_INPUT_ROOT = "results/hsi_v2/q6a_stage1_high_scales_symmetric"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/q6a_stage1_high_scales_symmetric/readout"
EXPECTED_SCALES = (8, 12, 16, 20, 24, 28, 32, 33, 34, 35, 36, 40, 44, 48)
HIGH_SCALES = (36, 40, 44, 48)
HIGH_EDGES = ((35, 36), (36, 40), (40, 44), (44, 48))
CONTRACT_SHA256 = "E3E1DFBCFD55E44C8963CE2FDDD267FD0477ACAA9E8AE438B6BE54126B09537F"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify Q6-A Stage 1 B/E high-scale tower readout.")
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variants", default="B,E")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_root = resolve_results_dir(args.input_root, anchor_file=__file__)
    output_root = resolve_results_dir(args.output_dir, anchor_file=__file__)
    variants = [item.strip().upper() for item in args.variants.split(",") if item.strip()]
    if variants != ["B", "E"]:
        raise SystemExit("This readout is frozen for variants B,E in that order.")

    runs = {variant: find_latest_variant_run(input_root, variant) for variant in variants}
    missing = [variant for variant, run_dir in runs.items() if run_dir is None]
    if missing:
        print(f"inputs-or-run-not-derivable: missing runs for {', '.join(missing)}")
        return 2

    payloads = {variant: load_run_payload(run_dir) for variant, run_dir in runs.items() if run_dir is not None}
    readout = classify(payloads)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"q6a-stage1-readout__B-E__m48__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "status": "completed",
        "contract_sha256": CONTRACT_SHA256,
        "input_root": str(input_root),
        "runs": {variant: str(run_dir_) for variant, run_dir_ in runs.items()},
        "outcome": readout["outcome"],
        "violations": readout["violations"],
        "variant_readouts": readout["variant_readouts"],
        "cross_variant_readout": readout["cross_variant_readout"],
        "no_phase4_data_observed": True,
    }
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "hsi_v2_q6a_stage1_high_scales_readout.py",
        "inputs": {
            variant: {
                "run_dir": str(run_dir_),
                "summary_sha256": sha256_file(run_dir_ / "phase1_summary.json"),
                "manifest_sha256": sha256_file(run_dir_ / "manifest.json"),
            }
            for variant, run_dir_ in runs.items()
            if run_dir_ is not None
        },
        "outputs": {
            "summary": str(run_dir / "summary.json"),
            "manifest": str(run_dir / "manifest.json"),
            "report": str(run_dir / "report.md"),
        },
        "constraints": {
            "read_only": True,
            "phase4_not_observed": True,
            "contract_sha256": CONTRACT_SHA256,
        },
    }

    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "report.md").write_text(render_report(summary), encoding="utf-8")

    print("Q6-A Stage 1 high-scale readout")
    print("-" * 72)
    print(f"status: completed")
    print(f"outcome: {summary['outcome']}")
    print(f"violations: {len(summary['violations'])}")
    print(f"output: {run_dir}")
    return 0


def find_latest_variant_run(input_root: Path, variant: str) -> Path | None:
    candidates = []
    pattern = f"phase1-tower__var-{variant}__iter-24__seg-3x1M__m-*__proj-prefix-suffix__*"
    for run_dir in input_root.glob(pattern):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "phase1_summary.json"
        manifest_path = run_dir / "manifest.json"
        if summary_path.exists() and manifest_path.exists():
            candidates.append(run_dir)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_run_payload(run_dir: Path) -> dict[str, Any]:
    summary = load_json(run_dir / "phase1_summary.json")
    manifest = load_json(run_dir / "manifest.json")
    return {"run_dir": str(run_dir), "summary": summary, "manifest": manifest}


def classify(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    variant_readouts = {}
    violations: list[dict[str, Any]] = []

    for variant, payload in payloads.items():
        readout = classify_variant(variant, payload)
        variant_readouts[variant] = readout
        violations.extend(readout["violations"])

    cross = classify_cross_variant(variant_readouts)
    violations.extend(cross["violations"])

    if any(item["kind"] == "collapse" for item in violations):
        outcome = "q6a-stage1-collapses-before-m48"
    elif violations:
        outcome = "q6a-stage1-high-scale-degraded-or-asymmetric"
    else:
        outcome = "q6a-stage1-symmetric-high-scale-derivable"

    return {
        "outcome": outcome,
        "violations": violations,
        "variant_readouts": variant_readouts,
        "cross_variant_readout": cross,
    }


def classify_variant(variant: str, payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    scale_rows = {int(row["m"]): row for row in summary.get("scale_summary", [])}
    fiber_rows = {
        policy: {(int(row["target_m"]), int(row["source_m"])): row for row in rows}
        for policy, rows in summary.get("fiber_summary", {}).items()
    }
    violations: list[dict[str, Any]] = []

    for scale in EXPECTED_SCALES:
        row = scale_rows.get(scale)
        if row is None:
            violations.append({"variant": variant, "kind": "collapse", "reason": "missing_scale", "m": scale})
            continue
        kept = int(row.get("kept_pattern_count") or 0)
        if kept <= 0:
            violations.append({"variant": variant, "kind": "collapse", "reason": "empty_pattern_space", "m": scale})
        if scale in HIGH_SCALES and kept <= 16:
            violations.append(
                {"variant": variant, "kind": "degradation", "reason": "active_support_floor", "m": scale, "kept": kept}
            )

    high_ratios = {}
    for previous_m, next_m in HIGH_EDGES:
        previous_count = int(scale_rows.get(previous_m, {}).get("kept_pattern_count") or 0)
        next_count = int(scale_rows.get(next_m, {}).get("kept_pattern_count") or 0)
        ratio = (next_count / previous_count) if previous_count else None
        high_ratios[f"{previous_m}->{next_m}"] = ratio
        if ratio is None:
            violations.append({"variant": variant, "kind": "collapse", "reason": "ratio_not_derivable", "edge": f"{previous_m}->{next_m}"})
        elif ratio < 0.5:
            violations.append(
                {"variant": variant, "kind": "degradation", "reason": "cardinality_ratio", "edge": f"{previous_m}->{next_m}", "ratio": ratio}
            )

    high_survival = {}
    for policy in ("prefix", "suffix"):
        policy_rows = fiber_rows.get(policy, {})
        for previous_m, next_m in HIGH_EDGES:
            key = (previous_m, next_m)
            row = policy_rows.get(key)
            edge_label = f"{previous_m}->{next_m}"
            if row is None:
                violations.append({"variant": variant, "kind": "collapse", "reason": "missing_fiber_edge", "policy": policy, "edge": edge_label})
                continue
            survival = float(row.get("survival_fraction") or 0.0)
            high_survival[f"{policy}:{edge_label}"] = survival
            if survival < 0.3:
                violations.append(
                    {"variant": variant, "kind": "degradation", "reason": "survival_fraction", "policy": policy, "edge": edge_label, "survival": survival}
                )

    return {
        "run_dir": payload["run_dir"],
        "kept_by_scale": {str(m): int(scale_rows[m]["kept_pattern_count"]) for m in sorted(scale_rows)},
        "candidate_by_scale": {str(m): int(scale_rows[m]["candidate_pattern_count"]) for m in sorted(scale_rows)},
        "high_cardinality_ratios": high_ratios,
        "high_survival": high_survival,
        "violations": violations,
    }


def classify_cross_variant(variant_readouts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    ratios = {}
    b_counts = variant_readouts["B"]["kept_by_scale"]
    e_counts = variant_readouts["E"]["kept_by_scale"]
    for scale in HIGH_SCALES:
        b = int(b_counts.get(str(scale), 0))
        e = int(e_counts.get(str(scale), 0))
        if b <= 0 or e <= 0:
            violations.append({"kind": "collapse", "reason": "cross_variant_count_missing", "m": scale, "B": b, "E": e})
            continue
        ratio = max(b, e) / min(b, e)
        ratios[str(scale)] = ratio
        if ratio > 4.0:
            violations.append({"kind": "degradation", "reason": "cross_variant_kept_count_divergence", "m": scale, "ratio": ratio, "B": b, "E": e})
    return {"kept_count_max_over_min_by_scale": ratios, "violations": violations}


def render_report(summary: dict[str, Any]) -> str:
    variant_sections = []
    for variant, readout in summary["variant_readouts"].items():
        high_rows = "\n".join(
            f"| {scale} | {readout['kept_by_scale'][str(scale)]} | {readout['candidate_by_scale'][str(scale)]} |"
            for scale in HIGH_SCALES
        )
        ratio_rows = "\n".join(
            f"| {edge} | {ratio:.4f} |"
            for edge, ratio in readout["high_cardinality_ratios"].items()
            if ratio is not None
        )
        variant_sections.append(
            f"""## Variant {variant}

Run: `{readout['run_dir']}`

| m | kept | candidate |
|---:|---:|---:|
{high_rows}

| edge | kept ratio |
|---|---:|
{ratio_rows}
"""
        )
    violations = "\n".join(f"- `{item}`" for item in summary["violations"]) or "- none"
    return f"""# Q6-A Stage 1 High-Scale Symmetric Readout

Outcome: `{summary['outcome']}`.

Contract SHA-256: `{summary['contract_sha256']}`.

No Phase 4 data was observed or generated by this readout.

{chr(10).join(variant_sections)}

## Cross-Variant Readout

Kept-count max/min ratios by high scale:

```json
{json.dumps(summary['cross_variant_readout']['kept_count_max_over_min_by_scale'], indent=2)}
```

## Violations

{violations}
"""


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
