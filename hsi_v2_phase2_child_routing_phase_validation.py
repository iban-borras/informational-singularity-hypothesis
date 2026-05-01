#!/usr/bin/env python3
"""
HSI v2 Phase 2 Child-Routing Phase Validation

N2-10c targeted validator. It reads the N2-10b tri-estimator audit, selects
strong/transition phase bands, and runs the audited N2-09 child-routing
persistence wrapper once per selected lag. Transition bands can be validated
against both the statistical consensus lag and the first-full topological lag.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hsi_v2_phase2_parent_survival_revalidation import (
    append_flag,
    flush_output,
    latest_child_file,
    load_json,
    run_subprocess,
)
from hsi_v2_phase2_transport_defect_strict import phase_print
from v2.common.cli import resolve_dir
from v2.common.naming import compact_int


DEFAULT_TRI_SUMMARY = "results/hsi_v2/phase2/phase_lag_tri_estimator/summary_latest.json"
DEFAULT_ATLAS_SUMMARY = "results/hsi_v2/phase2/phase_lag_atlas/summary_latest.json"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase2/child_routing_phase_validation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate N2-10 phase bands by running child-routing persistence "
            "only for tri-estimator strong/transition bands."
        )
    )
    parser.add_argument("--tri-summary", type=str, default=DEFAULT_TRI_SUMMARY)
    parser.add_argument("--atlas-summary", type=str, default=DEFAULT_ATLAS_SUMMARY)
    parser.add_argument("--phase1-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--confidences",
        type=str,
        default="strong,transition",
        help="Comma-separated N2-10b confidence classes to validate.",
    )
    parser.add_argument(
        "--lag-policy",
        type=str,
        default="consensus",
        choices=("consensus", "first-full", "dual-transition"),
        help=(
            "Lag selection policy. 'dual-transition' validates discrepant transition "
            "bands against both consensus and first-full lags."
        ),
    )
    parser.add_argument("--stage", type=str, default="all", choices=("observed", "nulls", "all"))
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_targets is not None and args.max_targets <= 0:
        parser.error("--max-targets must be positive when provided.")

    root = Path(__file__).parent.resolve()
    tri_summary_path = resolve_dir(args.tri_summary, anchor_file=__file__)
    atlas_summary_path = resolve_dir(args.atlas_summary, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    tri_payload = load_json(tri_summary_path)
    atlas_payload = load_json(atlas_summary_path)
    confidence_filter = parse_confidences(args.confidences)
    targets = build_targets(
        tri_payload=tri_payload,
        atlas_payload=atlas_payload,
        confidence_filter=confidence_filter,
        lag_policy=args.lag_policy,
    )
    if args.max_targets is not None:
        targets = targets[: args.max_targets]
    if not targets:
        parser.error("No validation targets matched the requested confidence filter.")

    atlas_selection = atlas_payload.get("selection", {})
    phase1_dir = (
        resolve_dir(args.phase1_dir, anchor_file=__file__)
        if args.phase1_dir
        else Path(atlas_selection["phase1_dir"])
    )

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(args, targets, timestamp)
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing N2-10c child-routing phase validation",
        f"stage={args.stage} | targets={len(targets)} | confidences={','.join(confidence_filter)}",
        quiet=args.quiet,
    )
    flush_output()

    commands: list[dict] = []
    target_rows: list[dict] = []
    band_rows: list[dict] = []
    source_rows: list[dict] = []

    for index, target in enumerate(targets, start=1):
        target_root = run_dir / target["target_label"]
        command = build_validation_command(
            root=root,
            phase1_dir=phase1_dir,
            output_dir=target_root,
            stage=args.stage,
            target=target,
            atlas_selection=atlas_selection,
            quiet=args.quiet,
        )
        commands.append(
            {
                "target_label": target["target_label"],
                "band_label": target["band_label"],
                "lag_source": target["selected_lag_source"],
                "selected_lag_bits": target["selected_lag_bits"],
                "command": command,
            }
        )
        target_rows.append({**target, "validation_root": str(target_root)})
        phase_print(
            "Validation target",
            (
                f"{index}/{len(targets)} | {target['band_label']} | "
                f"lag={target['selected_lag_bits']} ({target['selected_lag_source']}) | "
                f"conf={target['confidence']}"
            ),
            quiet=args.quiet,
        )
        flush_output()
        if args.dry_run:
            continue

        run_subprocess(command, cwd=root)
        child_summary_path = latest_child_file(target_root, "summary.json")
        child_payload = load_json(child_summary_path)
        collect_validation_rows(
            target=target,
            child_summary_path=child_summary_path,
            child_payload=child_payload,
            band_rows=band_rows,
            source_rows=source_rows,
        )

    generated_at = datetime.now().isoformat(timespec="seconds")
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    target_csv_path = run_dir / "targets.csv"
    band_csv_path = run_dir / "band_validation.csv"
    source_csv_path = run_dir / "source_validation.csv"
    manifest_path = run_dir / "manifest.json"

    summary_payload = {
        "generated_at": generated_at,
        "stage": args.stage,
        "dry_run": bool(args.dry_run),
        "lag_policy": args.lag_policy,
        "tri_summary": str(tri_summary_path),
        "atlas_summary": str(atlas_summary_path),
        "phase1_dir": str(phase1_dir),
        "confidence_filter": confidence_filter,
        "targets": target_rows,
        "band_validation": band_rows,
        "source_validation": source_rows,
    }
    manifest_payload = {
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_child_routing_phase_validation.py",
        "cwd": str(Path.cwd()),
        "arguments": vars(args),
        "commands": commands,
        "outputs": {
            "summary": str(summary_path),
            "report": str(report_path),
            "targets_csv": str(target_csv_path),
            "band_csv": str(band_csv_path),
            "source_csv": str(source_csv_path),
            "manifest": str(manifest_path),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_report(summary_payload) + "\n")
    write_csv(target_rows, target_csv_path)
    write_csv(band_rows, band_csv_path)
    write_csv(source_rows, source_csv_path)

    if not args.quiet:
        print(render_console_summary(summary_payload))
        print("")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved target CSV to: {target_csv_path}")
        print(f"Saved band CSV to: {band_csv_path}")
        print(f"Saved source CSV to: {source_csv_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def parse_confidences(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_targets(
    *,
    tri_payload: dict,
    atlas_payload: dict,
    confidence_filter: list[str],
    lag_policy: str,
) -> list[dict]:
    atlas_rows = {row["band_label"]: row for row in atlas_payload.get("rows", [])}
    tolerance_bits = int(tri_payload.get("tolerance_bits", 0))
    targets = []
    for row in tri_payload.get("rows", []):
        confidence = row.get("confidence")
        if confidence not in confidence_filter:
            continue
        band_label = row["band_label"]
        atlas_row = atlas_rows.get(band_label)
        if atlas_row is None:
            raise ValueError(f"Missing atlas row for {band_label}")
        base = {
            "band_index": row["band_index"],
            "band_label": band_label,
            "band_start_bits": row["band_start_bits"],
            "band_end_bits": row["band_end_bits"],
            "confidence": confidence,
            "consensus_lag_bits": row.get("consensus_lag_bits"),
            "first_full_lag_bits": row.get("first_full_lag_bits"),
            "hit_corr_lag_bits": row.get("hit_corr_lag_bits"),
            "unique_corr_lag_bits": row.get("unique_corr_lag_bits"),
            "lag_spread_bits": row.get("lag_spread_bits"),
            "probe_scan_start_bits": atlas_row["scan_start_bits"],
            "probe_scan_stop_bits": atlas_row["scan_stop_bits"],
        }
        for selected in selected_lags(base, lag_policy, tolerance_bits):
            targets.append({**base, **selected})
    return targets


def selected_lags(row: dict, lag_policy: str, tolerance_bits: int) -> list[dict]:
    consensus = as_optional_int(row.get("consensus_lag_bits"))
    first_full = as_optional_int(row.get("first_full_lag_bits"))
    confidence = row.get("confidence")
    band_label = row["band_label"]

    if lag_policy == "first-full":
        return lag_target(band_label, "first_full", first_full)
    if lag_policy == "dual-transition":
        targets = lag_target(band_label, "consensus", consensus)
        if (
            confidence == "transition"
            and first_full is not None
            and consensus is not None
            and abs(first_full - consensus) > tolerance_bits
        ):
            targets += lag_target(band_label, "first_full", first_full)
        return targets
    return lag_target(band_label, "consensus", consensus)


def lag_target(band_label: str, source: str, lag: int | None) -> list[dict]:
    if lag is None:
        return []
    label = band_label if source == "consensus" else f"{band_label}__{source}"
    return [
        {
            "target_label": label,
            "selected_lag_source": source,
            "selected_lag_bits": lag,
        }
    ]


def as_optional_int(value) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def build_validation_command(
    *,
    root: Path,
    phase1_dir: Path,
    output_dir: Path,
    stage: str,
    target: dict,
    atlas_selection: dict,
    quiet: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(root / "hsi_v2_phase2_child_routing_persistence.py"),
        "--phase1-dir",
        str(phase1_dir),
        "--output-dir",
        str(output_dir),
        "--stage",
        stage,
        "--band-starts",
        str(target["band_start_bits"]),
        "--window-count",
        str(atlas_selection["window_count"]),
        "--window-step-bits",
        str(atlas_selection["window_step_bits"]),
        "--lag-mode",
        "fixed",
        "--candidate-lag-bits",
        str(target["selected_lag_bits"]),
        "--top-patterns",
        str(atlas_selection["top_patterns"]),
        "--probe-scan-start-bits",
        str(target["probe_scan_start_bits"]),
        "--probe-scan-stop-bits",
        str(target["probe_scan_stop_bits"]),
    ]
    append_flag(command, "--quiet", quiet)
    return command


def collect_validation_rows(
    *,
    target: dict,
    child_summary_path: Path,
    child_payload: dict,
    band_rows: list[dict],
    source_rows: list[dict],
) -> None:
    common = {
        "phase_target_label": target["target_label"],
        "phase_band_label": target["band_label"],
        "phase_confidence": target["confidence"],
        "phase_selected_lag_source": target["selected_lag_source"],
        "phase_selected_lag_bits": target["selected_lag_bits"],
        "phase_consensus_lag_bits": target["consensus_lag_bits"],
        "phase_first_full_lag_bits": target["first_full_lag_bits"],
        "phase_lag_spread_bits": target["lag_spread_bits"],
        "validation_summary_path": str(child_summary_path),
    }
    for row in child_payload.get("band_summary", []):
        band_rows.append({**common, **row})
    for row in child_payload.get("source_persistence", []):
        source_rows.append({**common, **row})


def render_console_summary(payload: dict) -> str:
    source_rows = payload["source_validation"]
    target_rows = payload["targets"]
    lines = [
        "Phase 2 N2-10c child-routing phase validation",
        "-" * 132,
        f"targets={len(target_rows)} | stage={payload['stage']} | dry_run={payload['dry_run']}",
    ]
    if payload["dry_run"]:
        lines.append("")
        lines.append(f"{'target':<28} {'conf':<11} {'lag_src':<10} {'lag':>9} {'scan':>22}")
        for row in target_rows:
            scan = f"{compact_int(row['probe_scan_start_bits'])}-{compact_int(row['probe_scan_stop_bits'])}"
            lines.append(
                f"{short_target(row['target_label']):<28} {row['confidence']:<11} "
                f"{row['selected_lag_source']:<10} {lag_label(row['selected_lag_bits']):>9} {scan:>22}"
            )
        return "\n".join(lines)

    lines.append("")
    lines.append(
        f"{'target':<28} {'conf':<11} {'lag_src':<10} {'lag':>9} {'variant':<14} {'source':<28} {'childRet':>9} {'class':<24}"
    )
    for row in source_rows:
        lines.append(
            f"{short_target(row['phase_target_label']):<28} "
            f"{row['phase_confidence']:<11} "
            f"{row['phase_selected_lag_source']:<10} "
            f"{lag_label(row['phase_selected_lag_bits']):>9} "
            f"{row.get('variant', '-'):<14} "
            f"{clip(row.get('source_label', '-'), 28):<28} "
            f"{fmt_float(row.get('child_destination_retention_pooled')):>9} "
            f"{row.get('dominant_routing_class', '-'):<24}"
        )
    return "\n".join(lines)


def render_report(payload: dict) -> str:
    lines = [
        "# Phase 2 N2-10c Child-Routing Phase Validation",
        "",
        "## Scope",
        "",
        "- This is a targeted validation layer over N2-10b.",
        "- Each selected band is validated with the audited N2-09/N2-08 child-routing stack.",
        "- The selected lag follows the requested lag policy; dual-transition validates discrepant transition bands twice.",
        "",
        "## Targets",
        "",
        "| Target | Band | Confidence | Lag source | Selected lag | Consensus | First full | Hit corr | Unique corr | Scan |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["targets"]:
        scan = f"{compact_int(row['probe_scan_start_bits'])}-{compact_int(row['probe_scan_stop_bits'])}"
        lines.append(
            f"| `{row['target_label']}` | `{row['band_label']}` | {row['confidence']} | "
            f"{row['selected_lag_source']} | {lag_label(row['selected_lag_bits'])} | "
            f"{lag_label(row['consensus_lag_bits'])} | "
            f"{lag_label(row['first_full_lag_bits'])} | "
            f"{lag_label(row['hit_corr_lag_bits'])} | "
            f"{lag_label(row['unique_corr_lag_bits'])} | {scan} |"
        )
    if payload["dry_run"]:
        lines.extend(["", "_Dry run only: no validation rows generated._"])
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "## Source Validation",
            "",
            "| Target | Band | Confidence | Lag source | Lag | Variant | Source | ChildRet | Routing class |",
            "| --- | --- | --- | --- | ---: | --- | --- | ---: | --- |",
        ]
    )
    for row in payload["source_validation"]:
        lines.append(
            f"| `{row['phase_target_label']}` | `{row['phase_band_label']}` | {row['phase_confidence']} | "
            f"{row['phase_selected_lag_source']} | {lag_label(row['phase_selected_lag_bits'])} | "
            f"{row.get('variant', '-')} | {row.get('source_label', '-')} | "
            f"{fmt_float(row.get('child_destination_retention_pooled'))} | "
            f"{row.get('dominant_routing_class', '-')} |"
        )
    return "\n".join(lines)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_run_slug(args, targets: list[dict], timestamp: str) -> str:
    start_part = f"{compact_int(targets[0]['band_start_bits'])}-{compact_int(targets[-1]['band_start_bits'])}x{len(targets)}"
    conf_part = "-".join(parse_confidences(args.confidences))
    return (
        "phase2-child-routing-phase-validation"
        f"__stage-{args.stage}"
        f"__conf-{conf_part}"
        f"__lag-{args.lag_policy}"
        f"__bands-{start_part}"
        f"__{timestamp}"
    )


def lag_label(value) -> str:
    if value is None or value == "":
        return "-"
    value = int(value)
    sign = "-" if value < 0 else ""
    abs_value = abs(value)
    if abs_value == 0:
        return "0"
    if abs_value % 1_000_000 == 0:
        return f"{sign}{abs_value // 1_000_000}M"
    if abs_value % 500_000 == 0:
        return f"{sign}{abs_value / 1_000_000:g}M"
    if abs_value % 1_000 == 0:
        return f"{sign}{abs_value // 1_000}K"
    return str(value)


def fmt_float(value) -> str:
    if value is None or value == "":
        return "-"
    return f"{float(value):.4f}"


def clip(value: str, size: int) -> str:
    text = str(value)
    if len(text) <= size:
        return text
    return text[: size - 3] + "..."


def short_band(label: str) -> str:
    return label.replace("band-", "")


def short_target(label: str) -> str:
    return label.replace("band-", "")


if __name__ == "__main__":
    raise SystemExit(main())
