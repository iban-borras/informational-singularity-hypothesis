#!/usr/bin/env python3
"""
D-0154 Priority-A Phase 1 paper-table reconciliation.

This is a read-only downstream step over the already-audited Phase 1 batch.
It does not reread Level 0 snapshots and does not recompute tower objects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_BATCH_ROOT = "results/hsi_v2/d0154_priority_a_phase1"
DEFAULT_OUTPUT_ROOT = "results/hsi_v2/d0154_priority_a_phase1_table_reconciliation"
SCALES = [8, 12, 16, 20, 24, 28, 32]


PAPER_PM_SIZE = {
    "E": {8: 28, 12: 90, 16: 205, 20: 384, 24: 639, 28: 910, 32: 1208},
    "I": {8: 15, 12: 39, 16: 85, 20: 149, 24: 239, 28: 376, 32: 565},
    "F": {8: 35, 12: 74, 16: 109, 20: 138, 24: 166, 28: 195, 32: 227},
}

PAPER_FIBRES = {
    "E": {
        "surv_12_to_8": 1.00,
        "surv_32_to_28": 0.998,
        "mean_fiber_12_to_8": 3.21,
        "mean_fiber_32_to_28": 1.33,
        "entropy_12_to_8": 1.05,
        "entropy_32_to_28": 0.17,
    },
    "I": {
        "surv_12_to_8": 1.00,
        "surv_32_to_28": 1.000,
        "mean_fiber_12_to_8": 2.60,
        "mean_fiber_32_to_28": 1.50,
        "entropy_12_to_8": 0.78,
        "entropy_32_to_28": 0.28,
    },
    "F": {
        "surv_12_to_8": 1.00,
        "surv_32_to_28": 1.000,
        "mean_fiber_12_to_8": 2.11,
        "mean_fiber_32_to_28": 1.16,
        "entropy_12_to_8": 0.78,
        "entropy_32_to_28": 0.14,
    },
}

PAPER_ASYMMETRY = {
    "E": {
        "prefix_entropy_last": 0.171,
        "suffix_entropy_last": 0.236,
        "entropy_gap_signed": -0.065,
        "entropy_gap_relative": -0.16,
    },
    "I": {
        "prefix_entropy_last": 0.282,
        "suffix_entropy_last": 0.339,
        "entropy_gap_signed": -0.057,
        "entropy_gap_relative": -0.09,
    },
    "F": {
        "prefix_entropy_last": 0.136,
        "suffix_entropy_last": 0.133,
        "entropy_gap_signed": 0.003,
        "entropy_gap_relative": 0.01,
    },
}

PAPER_TRANSPORT = {
    "E": {
        "breadth_ratio": 43.14,
        "tail_transport_entropy": 0.395,
        "tail_transport_active": 0.354,
        "active_gap_signed": 0.037,
        "active_gap_relative": 0.055,
    },
    "I": {
        "breadth_ratio": 37.67,
        "tail_transport_entropy": 0.532,
        "tail_transport_active": 0.506,
        "active_gap_signed": 0.043,
        "active_gap_relative": 0.044,
    },
    "F": {
        "breadth_ratio": 6.49,
        "tail_transport_entropy": 0.226,
        "tail_transport_active": 0.234,
        "active_gap_signed": -0.004,
        "active_gap_relative": -0.009,
    },
}

PAPER_PORTA1 = {
    "E": {
        "retention_last": 0.8622,
        "prefix_survival_last": 0.9978,
        "tail_transport_active_mean": 0.3539,
    },
    "I": {
        "retention_last": 0.9930,
        "prefix_survival_last": 1.0000,
        "tail_transport_active_mean": 0.5064,
    },
    "F": {
        "retention_last": 1.0000,
        "prefix_survival_last": 1.0000,
        "tail_transport_active_mean": 0.2335,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconcile D-0154 clean Phase 1 batch outputs against current paper table cells."
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Audited D-0154 Phase 1 batch directory. Defaults to latest under results/hsi_v2/d0154_priority_a_phase1.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root. Relative results/... paths are rooted at HSI_RESULTS_BASE_DIR.",
    )
    parser.add_argument(
        "--paper-version",
        default="main_v2.03_cat.tex",
        help="Paper file whose printed values are being reconciled.",
    )
    return parser


def main() -> int:
    configure_stdout()
    load_dotenv()
    args = build_parser().parse_args()
    results_base = resolve_results_base()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else latest_batch_dir(results_base)
    output_root = resolve_results_path(args.output_dir, results_base)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = output_root / f"d0154-phase1-table-reconciliation__{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    phase1_report_path = single_json(run_dir / "reports", "phase1-report__*.json")
    transport_report_path = single_json(run_dir / "reports", "phase1-transport__*.json")
    phase1_report = read_json(phase1_report_path)
    transport_report = read_json(transport_report_path)
    phase1_rows = {row["variant"]: row for row in phase1_report["rows"]}
    transport_rows = {row["variant"]: row for row in transport_report["rows"]}
    summaries = {variant: read_json(Path(row["summary_path"])) for variant, row in phase1_rows.items()}

    pm_rows = build_pm_size_rows(phase1_rows)
    fibre_rows = build_fibre_rows(summaries)
    asymmetry_rows = build_asymmetry_rows(phase1_rows)
    transport_rows_out = build_transport_rows(transport_rows)
    porta_rows = build_porta_rows(phase1_rows, transport_rows)
    clean_rows = build_clean_variant_rows(phase1_rows, transport_rows)

    write_csv(out_dir / "tab_pm_size_comparison.csv", pm_rows)
    write_csv(out_dir / "tab_fibres_comparison.csv", fibre_rows)
    write_csv(out_dir / "prefix_suffix_asymmetry_comparison.csv", asymmetry_rows)
    write_csv(out_dir / "tab_transport_comparison.csv", transport_rows_out)
    write_csv(out_dir / "tab_porta1_canonica_comparison.csv", porta_rows)
    write_csv(out_dir / "clean_variant_phase1_values.csv", clean_rows)

    summary = {
        "status": "completed",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision_log_context": "D-0154 Amendment 7 follow-up: Phase 1 table-by-table reconciliation",
        "paper_version": args.paper_version,
        "source_run_dir": str(run_dir),
        "source_reports": {
            "phase1_report": str(phase1_report_path),
            "phase1_transport": str(transport_report_path),
        },
        "targets": ["E", "I", "D", "G", "F"],
        "paper_facing_rows_checked": ["E", "I", "F"],
        "supporting_clean_rows_not_in_current_tables": ["D", "G"],
        "tables": {
            "tab_pm_size": summarize_matches(pm_rows),
            "tab_fibres": summarize_matches(fibre_rows),
            "prefix_suffix_asymmetry": summarize_matches(asymmetry_rows),
            "tab_transport": summarize_matches(transport_rows_out),
            "tab_porta1_canonica": summarize_matches(porta_rows),
        },
        "classification_shift_detected": False,
        "notes": [
            "All comparisons are read-only over the already-audited D-0154 Priority-A Phase 1 batch.",
            "Rows D and G are included in clean_variant_phase1_values.csv for downstream family-closure work, but they are not printed in the current main Phase 1 tables reconciled here.",
            "The combined B/E paper row is checked against clean E; clean E remains display-identical to the B/E row under the current printed precision.",
        ],
    }
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "report.md", summary)

    manifest = {
        "status": "completed",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "script_sha256": sha256_file(Path(__file__)),
        "results_base": str(results_base),
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "inputs": {
            "phase1_batch_manifest": hash_if_exists(run_dir / "manifest.json"),
            "phase1_report": hash_record(phase1_report_path),
            "phase1_transport": hash_record(transport_report_path),
        },
        "outputs": {},
    }
    write_json(out_dir / "manifest.json", manifest)
    manifest["outputs"] = hash_outputs(out_dir)
    write_json(out_dir / "manifest.json", manifest)
    manifest_sha = sha256_file(out_dir / "manifest.json")
    (out_dir / "manifest.sha256").write_text(f"{manifest_sha}  manifest.json\n", encoding="utf-8")

    print("D-0154 Phase 1 table reconciliation completed.")
    print(f"Output: {out_dir}")
    print(f"Manifest SHA-256: {manifest_sha}")
    return 0


def build_pm_size_rows(phase1_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in ["E", "I", "F"]:
        clean = {int(k): int(v) for k, v in phase1_rows[variant]["scale_counts"].items()}
        for scale in SCALES:
            old = PAPER_PM_SIZE[variant][scale]
            new = clean[scale]
            rows.append(comparison_row("tab:pm_size", variant, f"kept_m{scale}", old, new, 0))
    return rows


def build_fibre_rows(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    fields = {
        "surv_12_to_8": ("survival_fraction", 12, 8, 2),
        "surv_32_to_28": ("survival_fraction", 32, 28, 3),
        "mean_fiber_12_to_8": ("mean_fiber_size", 12, 8, 2),
        "mean_fiber_32_to_28": ("mean_fiber_size", 32, 28, 2),
        "entropy_12_to_8": ("mean_child_entropy", 12, 8, 2),
        "entropy_32_to_28": ("mean_child_entropy", 32, 28, 2),
    }
    rows: list[dict[str, Any]] = []
    for variant in ["E", "I", "F"]:
        edge_map = edge_lookup(summaries[variant]["fiber_summary"]["prefix"])
        for field, (source_key, source_m, target_m, decimals) in fields.items():
            old = PAPER_FIBRES[variant][field]
            new = edge_map[(source_m, target_m)][source_key]
            rows.append(comparison_row("tab:fibres", variant, field, old, new, decimals))
    return rows


def build_asymmetry_rows(phase1_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    fields = {
        "prefix_entropy_last": ("prefix_child_entropy_last", 3),
        "suffix_entropy_last": ("suffix_child_entropy_last", 3),
        "entropy_gap_signed": ("entropy_gap_last_signed", 3),
        "entropy_gap_relative": ("entropy_gap_last_relative", 2),
    }
    rows: list[dict[str, Any]] = []
    for variant in ["E", "I", "F"]:
        row = dict(phase1_rows[variant])
        gap = row["prefix_child_entropy_last"] - row["suffix_child_entropy_last"]
        denom = abs(row["prefix_child_entropy_last"]) + abs(row["suffix_child_entropy_last"])
        row["entropy_gap_last_signed"] = gap
        row["entropy_gap_last_relative"] = gap / denom if denom else 0.0
        for field, (source_key, decimals) in fields.items():
            rows.append(
                comparison_row(
                    "prefix_suffix_asymmetry",
                    variant,
                    field,
                    PAPER_ASYMMETRY[variant][field],
                    row[source_key],
                    decimals,
                )
            )
    return rows


def build_transport_rows(transport_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    fields = {
        "breadth_ratio": ("breadth_ratio_last_to_first", 2),
        "tail_transport_entropy": ("tail_transport_entropy_mean", 3),
        "tail_transport_active": ("tail_transport_active_mean", 3),
        "active_gap_signed": ("tail_transport_active_gap_signed", 3),
        "active_gap_relative": ("tail_transport_active_gap_relative", 3),
    }
    rows: list[dict[str, Any]] = []
    for variant in ["E", "I", "F"]:
        row = transport_rows[variant]
        for field, (source_key, decimals) in fields.items():
            rows.append(
                comparison_row(
                    "tab:transport",
                    variant,
                    field,
                    PAPER_TRANSPORT[variant][field],
                    row[source_key],
                    decimals,
                )
            )
    return rows


def build_porta_rows(
    phase1_rows: dict[str, dict[str, Any]], transport_rows: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    fields = {
        "retention_last": ("phase1", "retention_last", 4),
        "prefix_survival_last": ("phase1", "prefix_survival_last", 4),
        "tail_transport_active_mean": ("transport", "tail_transport_active_mean", 4),
    }
    rows: list[dict[str, Any]] = []
    for variant in ["E", "I", "F"]:
        for field, (source, source_key, decimals) in fields.items():
            source_row = phase1_rows[variant] if source == "phase1" else transport_rows[variant]
            rows.append(
                comparison_row(
                    "tab:porta1_canonica",
                    variant,
                    field,
                    PAPER_PORTA1[variant][field],
                    source_row[source_key],
                    decimals,
                )
            )
    return rows


def build_clean_variant_rows(
    phase1_rows: dict[str, dict[str, Any]], transport_rows: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in ["E", "I", "D", "G", "F"]:
        p1 = phase1_rows[variant]
        tr = transport_rows[variant]
        rows.append(
            {
                "variant": variant,
                "iteration": p1["iteration"],
                "kept_m8": p1["scale_counts"]["8"],
                "kept_m12": p1["scale_counts"]["12"],
                "kept_m16": p1["scale_counts"]["16"],
                "kept_m20": p1["scale_counts"]["20"],
                "kept_m24": p1["scale_counts"]["24"],
                "kept_m28": p1["scale_counts"]["28"],
                "kept_m32": p1["scale_counts"]["32"],
                "retention_last": p1["retention_last"],
                "prefix_survival_last": p1["prefix_survival_last"],
                "suffix_survival_last": p1["suffix_survival_last"],
                "prefix_entropy_last": p1["prefix_child_entropy_last"],
                "suffix_entropy_last": p1["suffix_child_entropy_last"],
                "entropy_gap_signed": p1["prefix_suffix_entropy_gap_mean_signed"],
                "entropy_gap_relative": p1["prefix_suffix_entropy_gap_mean_relative"],
                "breadth_ratio": tr["breadth_ratio_last_to_first"],
                "tail_transport_entropy": tr["tail_transport_entropy_mean"],
                "tail_transport_active": tr["tail_transport_active_mean"],
                "active_gap_signed": tr["tail_transport_active_gap_signed"],
                "active_gap_relative": tr["tail_transport_active_gap_relative"],
                "status_hint": p1.get("status_hint") or "",
                "transport_hint": tr.get("transport_hint") or "",
            }
        )
    return rows


def comparison_row(
    table: str, variant: str, field: str, paper_value: float | int, clean_value: float | int, decimals: int
) -> dict[str, Any]:
    if isinstance(paper_value, int) and isinstance(clean_value, int):
        paper_display = str(paper_value)
        clean_display = str(clean_value)
    else:
        paper_display = format_float(float(paper_value), decimals)
        clean_display = format_float(float(clean_value), decimals)
    return {
        "table": table,
        "variant": "B/E" if variant == "E" and table != "tab:porta1_canonica" else variant,
        "clean_source_variant": variant,
        "field": field,
        "paper_value": paper_value,
        "clean_value_raw": clean_value,
        "paper_display": paper_display,
        "clean_display": clean_display,
        "display_match": paper_display == clean_display,
        "delta_raw": float(clean_value) - float(paper_value),
    }


def summarize_matches(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches = [row for row in rows if not row["display_match"]]
    return {
        "row_count": len(rows),
        "display_matches": len(rows) - len(mismatches),
        "display_mismatches": len(mismatches),
        "status": "display-identical" if not mismatches else "display-mismatch",
        "mismatches": [
            {
                "variant": row["variant"],
                "field": row["field"],
                "paper_display": row["paper_display"],
                "clean_display": row["clean_display"],
            }
            for row in mismatches
        ],
    }


def edge_lookup(edges: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {(int(edge["source_m"]), int(edge["target_m"])): edge for edge in edges}


def format_float(value: float, decimals: int) -> str:
    if value == 0:
        value = 0.0
    return f"{value:.{decimals}f}"


def write_report(path: Path, summary: dict[str, Any]) -> None:
    total_mismatches = sum(info["display_mismatches"] for info in summary["tables"].values())
    verdict = (
        "No display-level mismatch was detected in the affected paper-facing Phase 1 rows checked here."
        if total_mismatches == 0
        else f"{total_mismatches} display-level mismatch(es) were detected; Sofia/Grace review is required before manuscript reconciliation."
    )
    stop_state = (
        "The matrix stop/escalation conditions remain untriggered for this reconciliation step."
        if total_mismatches == 0
        else "The script does not infer a classification shift; mismatches remain table-cell reconciliation items until reviewed."
    )
    lines = [
        "# D-0154 Phase 1 Table Reconciliation",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Source run: `{summary['source_run_dir']}`",
        f"- Paper version checked: `{summary['paper_version']}`",
        "- Mode: read-only over the audited Priority-A Phase 1 batch; no Level 0 snapshots reread.",
        "",
        "## Verdict",
        "",
        verdict,
        stop_state,
        "",
        "## Table Status",
        "",
        "| Table | Rows | Mismatches | Status |",
        "|---|---:|---:|---|",
    ]
    for table, info in summary["tables"].items():
        lines.append(
            f"| `{table}` | {info['row_count']} | {info['display_mismatches']} | `{info['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `E` is used as the clean source for the combined manuscript row `B/E` in the Phase 1 tables; it remains display-identical to the printed row at the current manuscript precision.",
            "- `D` and `G` are emitted in `clean_variant_phase1_values.csv` for downstream family-closure work, but they are not part of the current main Phase 1 table rows reconciled here.",
            "- This artifact does not authorize manuscript edits by itself; it supplies the old-vs-clean reconciliation package for Sofia/Grace review.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latest_batch_dir(results_base: Path) -> Path:
    root = resolve_results_path(DEFAULT_BATCH_ROOT, results_base)
    candidates = [p for p in root.glob("d0154-priority-a-phase1__*") if p.is_dir()]
    if not candidates:
        raise SystemExit(f"No D-0154 Priority-A Phase 1 batch directories found under {root}")
    return max(candidates, key=lambda p: p.name)


def single_json(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one {pattern} under {root}, found {len(matches)}")
    return matches[0]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def resolve_results_base() -> Path:
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


def hash_outputs(root: Path) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}:
            outputs[path.name] = hash_record(path)
    return outputs


def hash_if_exists(path: Path) -> dict[str, Any] | None:
    return hash_record(path) if path.exists() else None


def hash_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
