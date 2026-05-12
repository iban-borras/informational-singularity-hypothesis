#!/usr/bin/env python3
"""D-0118 L2-02 quotient-contraction mimicry probe.

This is a read-only derivation over the frozen L2-01 Stage 1 graph summary.
It does not rebuild graphs, create figures, or edit manuscript-facing files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hsi_v2_phase2_transport_defect_strict import phase_print
from v2.common.cli import resolve_dir


CONTRACT_SHA256 = "3F201C6CD66955721C262AF74CCEE806773A129ECC9DB0B09ADCD42FF432BB9A"
L2_01_WINDOW_GRAPH_SHA256 = "9DC815C3A80BEF180043C8726CEC402C514BF9CC27902F34A21547913BC2EFE2"
DEFAULT_CONTRACT = "Documentation/HSI_v2_Level2_L2_02_QuotientContraction_Preregistration_v0_2.md"
DEFAULT_L2_01_STAGE1_RUN = (
    "results/hsi_v2/level2/l2_01_spectral_percolation/stage1/"
    "l2-01-spectral-percolation-stage1__20260510T101433"
)
DEFAULT_OUTPUT_DIR = "results/hsi_v2/level2/l2_02_quotient_contraction/mimicry_probe"

PAIR_IDS = ["T1-ARC-B", "T1-P4"]
HARD_NULL_FAMILIES = ["phase-matched-lz", "block-entropy", "matched-lz"]
INDICATORS = [
    "node_count_contraction",
    "beta1_cycle_loss",
    "total_edge_weight_loss",
    "relative_density_rise",
    "largest_component_nonincrease",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D-0118 L2-02 quotient-contraction mimicry probe.")
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--l2-01-stage1-run", default=DEFAULT_L2_01_STAGE1_RUN)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    contract = resolve_dir(args.contract, anchor_file=__file__)
    l2_01_run = resolve_dir(args.l2_01_stage1_run, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    verify_hash(contract, CONTRACT_SHA256, "L2-02 contract")
    window_graph_csv = l2_01_run / "window_graph_summary.csv"
    verify_hash(window_graph_csv, L2_01_WINDOW_GRAPH_SHA256, "L2-01 window_graph_summary.csv")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"l2-02-quotient-contraction-mimicry__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing D-0118 L2-02 quotient-contraction mimicry probe",
        "read-only derivation from L2-01 Stage 1 | mimicry-as-primary | no figures",
        quiet=args.quiet,
    )

    rows = load_csv(window_graph_csv)
    observed_rows = select_observed_b_tier1(rows)
    if not complete_pairs(observed_rows):
        return write_stop(
            run_dir,
            "no contraction",
            "observed-B frozen within-arc windows are incomplete",
            started,
            args,
            contract,
            l2_01_run,
        )

    observed_contrasts = build_pair_contrasts(observed_rows, source_group="B|observed")
    observed_category = observed_b_category(observed_contrasts)
    hard_null_rows = select_hard_null_tier1(rows)
    null_contrasts, null_seed_rows = build_hard_null_readout(hard_null_rows)
    complete_nulls = [row for row in null_seed_rows if row["coverage"] == "complete"]
    overall = overall_verdict(observed_category, complete_nulls)
    cross_arc_context = select_observed_b_cross_arc_context(rows)

    summary = {
        "run_id": run_dir.name,
        "timestamp_utc": timestamp,
        "status": "completed",
        "overall_verdict": overall,
        "observed_b_category": observed_category,
        "complete_hard_null_seed_count": len(complete_nulls),
        "complete_hard_null_families": sorted({row["null_model"] for row in complete_nulls}),
        "any_null_mimicry_or_partial": any(
            row["seed_category"] in {"null-contraction-mimicry", "partial-null-contraction-mimicry"}
            for row in complete_nulls
        ),
        "contract": {"path": str(contract.resolve()), "sha256": sha256_file(contract)},
        "source": {
            "l2_01_stage1_run": str(l2_01_run.resolve()),
            "window_graph_summary_csv": str(window_graph_csv.resolve()),
            "window_graph_summary_sha256": sha256_file(window_graph_csv),
        },
        "counts": {
            "source_rows": len(rows),
            "observed_b_pairs": len(observed_contrasts),
            "hard_null_pair_contrasts": len(null_contrasts),
            "hard_null_seed_rows": len(null_seed_rows),
            "cross_arc_context_rows": len(cross_arc_context),
        },
        "boundary": [
            "No virgin holdout windows are used for observed-B.",
            "Observed-B classification is post-hoc descriptive confirmation, not independent evidence.",
            "Hard-null mimicry is the deciding component whenever observed-B contraction is present.",
            "No figure is emitted.",
            "No manuscript edit is authorized.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    write_json(run_dir / "summary.json", summary)
    write_csv_dynamic(run_dir / "observed_b_contraction.csv", observed_contrasts)
    write_csv_dynamic(run_dir / "hard_null_contraction_mimicry.csv", null_seed_rows)
    write_csv_dynamic(run_dir / "hard_null_pair_contrasts.csv", null_contrasts)
    write_csv_dynamic(run_dir / "cross_arc_context.csv", cross_arc_context)
    write_json(run_dir / "manifest.json", build_manifest(summary, args, run_dir))
    (run_dir / "report.md").write_text(
        render_report(summary, observed_contrasts, null_seed_rows),
        encoding="utf-8",
    )

    print_summary(summary)
    print(f"\nSaved summary to: {run_dir / 'summary.json'}")
    print(f"Saved report to: {run_dir / 'report.md'}")
    print(f"Saved observed-B CSV to: {run_dir / 'observed_b_contraction.csv'}")
    print(f"Saved hard-null CSV to: {run_dir / 'hard_null_contraction_mimicry.csv'}")
    print(f"Saved manifest to: {run_dir / 'manifest.json'}")
    return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"{label} SHA256 mismatch: expected {expected}, got {actual}")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_observed_b_tier1(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["tier"] == "tier1" and row["source_key"] == "B|observed"]


def select_hard_null_tier1(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["tier"] == "tier1" and row["null_model"] in HARD_NULL_FAMILIES
    ]


def select_observed_b_cross_arc_context(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    context = []
    for row in rows:
        if row["tier"] != "tier2" or row["source_key"] != "B|observed":
            continue
        context.append(
            {
                "arc_id": row["arc_id"],
                "role": row["role"],
                "window_id": row["window_id"],
                "bands": row["bands"],
                "node_count": row["node_count"],
                "beta1": row["beta1"],
                "edge_density": row["edge_density"],
                "largest_component_fraction": row["largest_component_fraction"],
                "total_edge_weight": row["total_edge_weight"],
                "context_only": True,
            }
        )
    return context


def complete_pairs(rows: list[dict[str, str]]) -> bool:
    seen = {(row["pair_id"], row["role"]) for row in rows}
    return seen == {(pair_id, role) for pair_id in PAIR_IDS for role in ["interior", "boundary"]}


def build_pair_contrasts(rows: list[dict[str, str]], source_group: str) -> list[dict[str, Any]]:
    by_pair_role = {(row["pair_id"], row["role"]): row for row in rows}
    contrasts = []
    for pair_id in PAIR_IDS:
        interior = by_pair_role.get((pair_id, "interior"))
        boundary = by_pair_role.get((pair_id, "boundary"))
        if interior is None or boundary is None:
            continue
        indicator_values = evaluate_indicators(interior, boundary)
        pass_count = sum(1 for value in indicator_values.values() if value)
        node_ratio = safe_ratio(as_float(boundary["node_count"]), as_float(interior["node_count"]))
        contrasts.append(
            {
                "source_group": source_group,
                "pair_id": pair_id,
                "interior_window_id": interior["window_id"],
                "boundary_window_id": boundary["window_id"],
                "interior_node_count": as_int(interior["node_count"]),
                "boundary_node_count": as_int(boundary["node_count"]),
                "node_count_ratio": node_ratio,
                "node_count_guard": node_count_guard(node_ratio),
                "interior_beta1": as_int(interior["beta1"]),
                "boundary_beta1": as_int(boundary["beta1"]),
                "interior_total_edge_weight": as_float(interior["total_edge_weight"]),
                "boundary_total_edge_weight": as_float(boundary["total_edge_weight"]),
                "interior_edge_density": as_float(interior["edge_density"]),
                "boundary_edge_density": as_float(boundary["edge_density"]),
                "interior_largest_component_fraction": as_float(interior["largest_component_fraction"]),
                "boundary_largest_component_fraction": as_float(boundary["largest_component_fraction"]),
                **indicator_values,
                "indicator_pass_count": pass_count,
                "contraction_positive": pass_count >= 4,
            }
        )
    return contrasts


def evaluate_indicators(interior: dict[str, str], boundary: dict[str, str]) -> dict[str, bool]:
    return {
        "node_count_contraction": as_float(boundary["node_count"]) < as_float(interior["node_count"]),
        "beta1_cycle_loss": as_float(boundary["beta1"]) < as_float(interior["beta1"]),
        "total_edge_weight_loss": as_float(boundary["total_edge_weight"]) < as_float(interior["total_edge_weight"]),
        "relative_density_rise": as_float(boundary["edge_density"]) > as_float(interior["edge_density"]),
        "largest_component_nonincrease": as_float(boundary["largest_component_fraction"])
        <= as_float(interior["largest_component_fraction"]),
    }


def observed_b_category(contrasts: list[dict[str, Any]]) -> str:
    positives = sum(1 for row in contrasts if row["contraction_positive"])
    if positives == 2:
        return "B-contraction-descriptively-present"
    if positives == 1:
        return "B-contraction-descriptively-partial"
    return "B-contraction-not-recovered"


def build_hard_null_readout(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["null_model"], row["null_seed"], row["source_key"])].append(row)

    pair_contrasts: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for (null_model, null_seed, source_key), group_rows in sorted(groups.items()):
        coverage = "complete" if complete_pairs(group_rows) else ("partial" if group_rows else "absent")
        contrasts = build_pair_contrasts(group_rows, source_group=source_key) if coverage == "complete" else []
        positive_pairs = sum(1 for row in contrasts if row["contraction_positive"])
        if coverage != "complete":
            seed_category = "inputs-incomplete"
        elif positive_pairs == 2:
            seed_category = "null-contraction-mimicry"
        elif positive_pairs == 1:
            seed_category = "partial-null-contraction-mimicry"
        else:
            seed_category = "no-null-contraction-mimicry"
        for contrast in contrasts:
            contrast.update(
                {
                    "null_model": null_model,
                    "null_seed": null_seed,
                    "source_key": source_key,
                    "seed_category": seed_category,
                }
            )
        pair_contrasts.extend(contrasts)
        seed_rows.append(
            {
                "null_model": null_model,
                "null_seed": null_seed,
                "source_key": source_key,
                "coverage": coverage,
                "positive_pair_count": positive_pairs if coverage == "complete" else "",
                "seed_category": seed_category,
            }
        )
    return pair_contrasts, seed_rows


def overall_verdict(observed_category: str, complete_nulls: list[dict[str, Any]]) -> str:
    if not complete_nulls:
        return "hard-null-inputs-not-derivable"
    if observed_category != "B-contraction-descriptively-present":
        return "no contraction"
    for row in complete_nulls:
        if row["seed_category"] in {"null-contraction-mimicry", "partial-null-contraction-mimicry"}:
            return "non-B-specific contraction"
    return "B-specific contraction candidate"


def node_count_guard(node_ratio: float | None) -> str:
    if node_ratio is None:
        return "undefined"
    if node_ratio < 0.25:
        return "severe-size-contraction"
    if node_ratio < 0.75:
        return "moderate-size-contraction"
    return "weak-or-no-size-contraction"


def as_float(raw: str) -> float:
    return float(raw) if raw not in {"", "-"} else 0.0


def as_int(raw: str) -> int:
    return int(float(raw)) if raw not in {"", "-"} else 0


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def write_stop(
    run_dir: Path,
    verdict: str,
    reason: str,
    started: float,
    args: argparse.Namespace,
    contract: Path,
    l2_01_run: Path,
) -> int:
    summary = {
        "run_id": run_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
        "status": "stopped",
        "overall_verdict": verdict,
        "stop_reason": reason,
        "contract": {"path": str(contract.resolve()), "sha256": sha256_file(contract)},
        "source": {"l2_01_stage1_run": str(l2_01_run.resolve())},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "manifest.json", build_manifest(summary, args, run_dir))
    (run_dir / "report.md").write_text(render_report(summary, [], []), encoding="utf-8")
    print_summary(summary)
    return 0


def build_manifest(summary: dict[str, Any], args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    return {
        "script": Path(__file__).name,
        "command": " ".join(["python", Path(__file__).name, *cli_args(args)]),
        "summary": summary,
        "artifacts": {
            path.name: sha256_file(path)
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }


def cli_args(args: argparse.Namespace) -> list[str]:
    out = [
        "--contract",
        str(args.contract),
        "--l2-01-stage1-run",
        str(args.l2_01_stage1_run),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.quiet:
        out.append("--quiet")
    return out


def render_report(
    summary: dict[str, Any],
    observed_contrasts: list[dict[str, Any]],
    null_seed_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# L2-02 Quotient-Contraction Mimicry Probe",
        "",
        f"- Status: `{summary['status']}`",
        f"- Overall verdict: `{summary['overall_verdict']}`",
        f"- Observed-B category: `{summary.get('observed_b_category', '-')}`",
        f"- Complete hard-null seeds: `{summary.get('complete_hard_null_seed_count', 0)}`",
        f"- Complete hard-null families: `{', '.join(summary.get('complete_hard_null_families', [])) or '-'}`",
        "",
        "## Contract Boundary",
        "",
        "- No virgin holdout windows are used for observed-B.",
        "- Observed-B classification is post-hoc descriptive confirmation, not independent evidence.",
        "- The primary new test is hard-null mimicry.",
        "- `hard-null-inputs-not-derivable` means no complete hard-null seed is available in any hard-null family.",
        "- Any complete hard-null seed with full or partial mimicry blocks `B-specific contraction candidate`.",
        "- No figure is emitted and no manuscript edit is authorized.",
        "",
        "## Observed-B Pair Readout",
        "",
    ]
    if observed_contrasts:
        lines.append("| Pair | Passes | Positive | Node ratio | Guard |")
        lines.append("|---|---:|---|---:|---|")
        for row in observed_contrasts:
            lines.append(
                f"| `{row['pair_id']}` | {row['indicator_pass_count']}/5 | "
                f"`{row['contraction_positive']}` | {format_float(row['node_count_ratio'])} | "
                f"`{row['node_count_guard']}` |"
            )
    else:
        lines.append("No observed-B contrasts were emitted.")
    lines.extend(["", "## Hard-Null Mimicry Readout", ""])
    if null_seed_rows:
        lines.append("| Null model | Seed | Coverage | Positive pairs | Category |")
        lines.append("|---|---:|---|---:|---|")
        for row in null_seed_rows:
            positive_pairs = row["positive_pair_count"]
            positive_pairs_text = "-" if positive_pairs == "" else str(positive_pairs)
            lines.append(
                f"| `{row['null_model']}` | `{row['null_seed']}` | `{row['coverage']}` | "
                f"{positive_pairs_text} | `{row['seed_category']}` |"
            )
    else:
        lines.append("No hard-null rows were emitted.")
    return "\n".join(lines) + "\n"


def format_float(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def print_summary(summary: dict[str, Any]) -> None:
    print("D-0118 L2-02 quotient-contraction mimicry probe")
    print("-" * 80)
    print(f"status: {summary['status']}")
    print(f"overall: {summary['overall_verdict']}")
    if "observed_b_category" in summary:
        print(f"observed_B: {summary['observed_b_category']}")
        print(f"complete_hard_null_seed_count: {summary['complete_hard_null_seed_count']}")
        print(f"complete_hard_null_families: {', '.join(summary['complete_hard_null_families']) or '-'}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_dynamic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
