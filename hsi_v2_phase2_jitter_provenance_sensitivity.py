#!/usr/bin/env python3
"""
HSI v2 Phase 2 Jitter Provenance Sensitivity

N2-07b wrapper over a completed parent-shell atlas sensitivity sweep. It
recomputes the same-parent jitter-provenance readout for each selected
top-k/lag atlas and summarizes whether the provenance class is stable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hsi_v2_phase2_transport_defect_strict import parse_int_list, phase_print
from v2.common.cli import resolve_dir
from v2.common.naming import compact_int
from v2.phase2.jitter_provenance import build_jitter_provenance_readout


DEFAULT_SENSITIVITY_ROOT = "results/hsi_v2/phase2/parent_shell_atlas_sensitivity"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase2/jitter_provenance_sensitivity"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run N2-07b jitter-provenance sensitivity over atlas artifacts "
            "from a completed N2-06 parent-shell atlas sensitivity sweep."
        )
    )
    parser.add_argument(
        "--sensitivity-run",
        type=str,
        default="",
        help="Path to one phase2-parent-shell-atlas-sensitivity__... run. Defaults to latest stage-all run.",
    )
    parser.add_argument(
        "--sensitivity-root",
        type=str,
        default=DEFAULT_SENSITIVITY_ROOT,
        help="Root used to discover the latest sensitivity run when --sensitivity-run is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where N2-07b artifacts will be written.",
    )
    parser.add_argument(
        "--top-patterns",
        type=str,
        default="64,128",
        help="Comma-separated top-k values to include. Defaults to the non-blind regimes.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        top_values = parse_int_list(
            args.top_patterns,
            label="--top-patterns",
            allow_zero=False,
        )
    except ValueError as exc:
        parser.error(str(exc))

    sensitivity_run = resolve_sensitivity_run(args)
    atlas_datasets = find_atlas_datasets(sensitivity_run, top_values=set(top_values))
    if not atlas_datasets:
        parser.error(f"No matching atlas datasets found under {sensitivity_run}")

    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(sensitivity_run, top_values, timestamp)
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing N2-07b jitter-provenance sensitivity",
        (
            f"run={sensitivity_run.name} | top={','.join(str(value) for value in top_values)} | "
            f"atlas={len(atlas_datasets)}"
        ),
        quiet=args.quiet,
    )

    source_rows: list[dict] = []
    pattern_rows: list[dict] = []
    skipped_rows: list[dict] = []
    for dataset_path in atlas_datasets:
        payload = load_json(dataset_path)
        selection = payload.get("selection", {})
        top_patterns = int(selection.get("top_patterns", 0))
        lag_bits = int(selection.get("candidate_lag_bits", 0))
        try:
            readout = build_jitter_provenance_readout(payload)
        except ValueError as exc:
            skipped_rows.append(
                {
                    "top_patterns": top_patterns,
                    "lag_bits": lag_bits,
                    "atlas_dataset_path": str(dataset_path),
                    "reason": str(exc),
                }
            )
            continue

        for row in readout["source_summary"]:
            source_rows.append(
                {
                    **flatten_source_row(row),
                    "top_patterns": top_patterns,
                    "lag_bits": lag_bits,
                    "lag_label": compact_int(lag_bits),
                    "atlas_dataset_path": str(dataset_path),
                }
            )
        for row in readout["pattern_summary"]:
            pattern_rows.append(
                {
                    **flatten_pattern_row(row),
                    "top_patterns": top_patterns,
                    "lag_bits": lag_bits,
                    "lag_label": compact_int(lag_bits),
                    "atlas_dataset_path": str(dataset_path),
                }
            )

    envelope_rows = summarize_source_envelope(source_rows)

    generated_at = datetime.now().isoformat(timespec="seconds")
    dataset_path = run_dir / "dataset.json"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    source_csv_path = run_dir / "source_provenance_sensitivity.csv"
    pattern_csv_path = run_dir / "pattern_provenance_sensitivity.csv"
    envelope_csv_path = run_dir / "source_envelope.csv"
    manifest_path = run_dir / "manifest.json"

    dataset_payload = {
        "stage": "phase2_jitter_provenance_sensitivity",
        "generated_at": generated_at,
        "selection": {
            "sensitivity_run": str(sensitivity_run),
            "top_patterns": top_values,
            "atlas_dataset_count": len(atlas_datasets),
        },
        "source_envelope": envelope_rows,
        "source_rows": source_rows,
        "pattern_rows": pattern_rows,
        "skipped_rows": skipped_rows,
        "notes": [
            "N2-07b repeats the N2-07 same-parent provenance readout across N2-06 top-k/lag atlas artifacts.",
            "Default top-k excludes top=32 because D-0050 classifies it as a blind resolution control for the monitored lateral shell.",
            "The readout tracks temporal compensation within the same parent pattern; it is not yet cross-parent routing.",
        ],
    }
    summary_payload = {
        "generated_at": generated_at,
        "selection": dataset_payload["selection"],
        "source_envelope": envelope_rows,
        "skipped_rows": skipped_rows,
    }
    manifest_payload = {
        "generated_at": generated_at,
        "run_slug": run_slug,
        "script": "hsi_v2_phase2_jitter_provenance_sensitivity.py",
        "cwd": str(Path.cwd()),
        "inputs": {
            "sensitivity_run": str(sensitivity_run),
            "atlas_datasets": [str(path) for path in atlas_datasets],
        },
        "outputs": {
            "dataset": str(dataset_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "source_csv": str(source_csv_path),
            "pattern_csv": str(pattern_csv_path),
            "envelope_csv": str(envelope_csv_path),
            "manifest": str(manifest_path),
        },
        "arguments": vars(args),
    }

    phase_print("Writing N2-07b artifacts", str(run_dir), quiet=args.quiet)
    with open(dataset_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_payload, handle, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_report(summary_payload) + "\n")
    write_csv(source_rows, source_csv_path, SOURCE_FIELDNAMES)
    write_csv(pattern_rows, pattern_csv_path, PATTERN_FIELDNAMES)
    write_csv(envelope_rows, envelope_csv_path, ENVELOPE_FIELDNAMES)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    if not args.quiet:
        print(render_console_summary(envelope_rows))
        if skipped_rows:
            print("")
            print(f"Skipped atlas datasets: {len(skipped_rows)}")
        print("")
        print(f"Saved dataset to: {dataset_path}")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved source CSV to: {source_csv_path}")
        print(f"Saved pattern CSV to: {pattern_csv_path}")
        print(f"Saved envelope CSV to: {envelope_csv_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


SOURCE_FIELDNAMES = [
    "top_patterns",
    "lag_bits",
    "lag_label",
    "variant",
    "source_label",
    "source_kind",
    "null_model",
    "null_seed",
    "pattern_count",
    "gross_loss_mass_sum",
    "gross_surplus_mass_sum",
    "compensated_loss_mass",
    "net_deficit_mass",
    "net_surplus_mass",
    "compensated_loss_share_of_gross",
    "net_deficit_share_of_gross",
    "loss_centroid_bits",
    "surplus_centroid_bits",
    "provenance_shift_bits",
    "dominant_provenance_class",
    "atlas_dataset_path",
]

PATTERN_FIELDNAMES = [
    "top_patterns",
    "lag_bits",
    "lag_label",
    "variant",
    "source_label",
    "source_kind",
    "null_model",
    "null_seed",
    "shell_subtype",
    "shell_subtype_label",
    "pattern",
    "gross_loss_mass_sum",
    "gross_surplus_mass_sum",
    "compensated_loss_mass",
    "net_deficit_mass",
    "net_surplus_mass",
    "compensated_loss_share_of_gross",
    "net_deficit_share_of_gross",
    "provenance_shift_bits",
    "provenance_class",
    "atlas_dataset_path",
]

ENVELOPE_FIELDNAMES = [
    "variant",
    "source_label",
    "source_kind",
    "null_model",
    "null_seed",
    "combo_count",
    "top_patterns",
    "lag_bits",
    "dominant_class_counts",
    "stable_dominant_class",
    "compensated_share_min",
    "compensated_share_avg",
    "compensated_share_max",
    "net_deficit_share_min",
    "net_deficit_share_avg",
    "net_deficit_share_max",
    "net_deficit_mass_min",
    "net_deficit_mass_avg",
    "net_deficit_mass_max",
    "provenance_shift_min",
    "provenance_shift_avg",
    "provenance_shift_max",
]


def resolve_sensitivity_run(args) -> Path:
    if args.sensitivity_run.strip():
        path = resolve_dir(args.sensitivity_run, anchor_file=__file__)
        if not path.is_dir():
            raise SystemExit(f"Sensitivity run not found: {path}")
        return path

    root = resolve_dir(args.sensitivity_root, anchor_file=__file__)
    if not root.is_dir():
        raise SystemExit(f"Sensitivity root not found: {root}")
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("phase2-parent-shell-atlas-sensitivity__")
        and "__stage-all__" in path.name
        and (path / "summary.json").is_file()
    ]
    if not candidates:
        raise SystemExit(f"No stage-all sensitivity runs found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_atlas_datasets(root: Path, *, top_values: set[int]) -> list[Path]:
    datasets: list[Path] = []
    for path in root.rglob("dataset.json"):
        if not path.parent.name.startswith("phase2-parent-shell-atlas__"):
            continue
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("stage") != "phase2_parent_shell_atlas":
            continue
        top_patterns = int(payload.get("selection", {}).get("top_patterns", 0))
        if top_patterns not in top_values:
            continue
        datasets.append(path)
    return sorted(
        datasets,
        key=lambda path: (
            int(load_json(path).get("selection", {}).get("top_patterns", 0)),
            int(load_json(path).get("selection", {}).get("candidate_lag_bits", 0)),
        ),
    )


def flatten_source_row(row: dict) -> dict:
    return {field: row.get(field) for field in SOURCE_FIELDNAMES if field not in {"top_patterns", "lag_bits", "lag_label", "atlas_dataset_path"}}


def flatten_pattern_row(row: dict) -> dict:
    return {field: row.get(field) for field in PATTERN_FIELDNAMES if field not in {"top_patterns", "lag_bits", "lag_label", "atlas_dataset_path"}}


def summarize_source_envelope(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["variant"],
            row["source_label"],
            row["source_kind"],
            row.get("null_model"),
            row.get("null_seed"),
        )
        grouped.setdefault(key, []).append(row)

    envelope_rows = []
    for group in grouped.values():
        first = group[0]
        class_counts = Counter(str(row["dominant_provenance_class"]) for row in group)
        stable_class = None
        if len(class_counts) == 1:
            stable_class = next(iter(class_counts))
        envelope_rows.append(
            {
                "variant": first["variant"],
                "source_label": first["source_label"],
                "source_kind": first["source_kind"],
                "null_model": first.get("null_model"),
                "null_seed": first.get("null_seed"),
                "combo_count": len(group),
                "top_patterns": ",".join(str(value) for value in sorted({int(row["top_patterns"]) for row in group})),
                "lag_bits": ",".join(str(value) for value in sorted({int(row["lag_bits"]) for row in group})),
                "dominant_class_counts": json.dumps(dict(sorted(class_counts.items()))),
                "stable_dominant_class": stable_class,
                "compensated_share_min": _min_present(group, "compensated_loss_share_of_gross"),
                "compensated_share_avg": _mean_present(group, "compensated_loss_share_of_gross"),
                "compensated_share_max": _max_present(group, "compensated_loss_share_of_gross"),
                "net_deficit_share_min": _min_present(group, "net_deficit_share_of_gross"),
                "net_deficit_share_avg": _mean_present(group, "net_deficit_share_of_gross"),
                "net_deficit_share_max": _max_present(group, "net_deficit_share_of_gross"),
                "net_deficit_mass_min": _min_present(group, "net_deficit_mass"),
                "net_deficit_mass_avg": _mean_present(group, "net_deficit_mass"),
                "net_deficit_mass_max": _max_present(group, "net_deficit_mass"),
                "provenance_shift_min": _min_present(group, "provenance_shift_bits"),
                "provenance_shift_avg": _mean_present(group, "provenance_shift_bits"),
                "provenance_shift_max": _max_present(group, "provenance_shift_bits"),
            }
        )

    envelope_rows.sort(
        key=lambda item: (
            item["variant"],
            _source_kind_sort_key(item["source_kind"], item.get("null_model")),
            item.get("null_seed") if item.get("null_seed") is not None else -1,
        )
    )
    return envelope_rows


def render_console_summary(envelope_rows: list[dict]) -> str:
    lines = [
        "Phase 2 jitter provenance sensitivity",
        "-" * 142,
        (
            f"{'variant':<12} {'source':<30} {'n':>3} {'class':>24} "
            f"{'compShare':>21} {'netShare':>21} {'netDef':>21}"
        ),
    ]
    for row in envelope_rows:
        lines.append(
            f"{_clip(row['variant'], 12):<12} "
            f"{_clip(row['source_label'], 30):<30} "
            f"{int(row['combo_count']):>3} "
            f"{_clip(row['stable_dominant_class'] or row['dominant_class_counts'], 24):>24} "
            f"{_fmt_range(row['compensated_share_min'], row['compensated_share_max']):>21} "
            f"{_fmt_range(row['net_deficit_share_min'], row['net_deficit_share_max']):>21} "
            f"{_fmt_range(row['net_deficit_mass_min'], row['net_deficit_mass_max'], decimals=0):>21}"
        )
    return "\n".join(lines)


def render_report(summary_payload: dict) -> str:
    selection = summary_payload["selection"]
    lines = [
        "# Phase 2 Jitter Provenance Sensitivity",
        "",
        "## Selection",
        "",
        f"- Sensitivity run: `{selection['sensitivity_run']}`",
        f"- Top patterns: {', '.join(str(value) for value in selection['top_patterns'])}",
        f"- Atlas datasets: {selection['atlas_dataset_count']}",
        "",
        "## Method Boundary",
        "",
        "This is a sensitivity wrapper over N2-07. It repeats the same-parent provenance readout over selected N2-06 atlas artifacts.",
        "By default it excludes `top = 32`, because that slice is a blind resolution control for the monitored lateral shell.",
        "The object does not yet infer cross-parent routing.",
        "",
        "## Source Envelope",
        "",
        "| Variant | Source | Combos | Stable class | Class counts | Comp/Loss rng | Net/Loss rng | NetDef rng |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in summary_payload["source_envelope"]:
        lines.append(
            f"| {row['variant']} | {row['source_label']} | {row['combo_count']} | "
            f"{row['stable_dominant_class'] or '-'} | `{row['dominant_class_counts']}` | "
            f"{_fmt_range(row['compensated_share_min'], row['compensated_share_max'])} | "
            f"{_fmt_range(row['net_deficit_share_min'], row['net_deficit_share_max'])} | "
            f"{_fmt_range(row['net_deficit_mass_min'], row['net_deficit_mass_max'], decimals=0)} |"
        )

    skipped = summary_payload.get("skipped_rows", [])
    if skipped:
        lines.extend(["", "## Skipped Atlas Datasets", ""])
        for row in skipped:
            lines.append(
                f"- top={row['top_patterns']} lag={row['lag_bits']}: {row['reason']}"
            )
    return "\n".join(lines)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def build_run_slug(sensitivity_run: Path, top_values: list[int], timestamp: str) -> str:
    source_stamp = sensitivity_run.name.split("__")[-1]
    return (
        f"phase2-jitter-provenance-sensitivity__top-"
        f"{'-'.join(str(value) for value in top_values)}"
        f"__src-{source_stamp}__{timestamp}"
    )


def _min_present(rows: list[dict], field: str):
    values = _present_values(rows, field)
    return min(values) if values else None


def _max_present(rows: list[dict], field: str):
    values = _present_values(rows, field)
    return max(values) if values else None


def _mean_present(rows: list[dict], field: str):
    values = _present_values(rows, field)
    if not values:
        return None
    return sum(values) / len(values)


def _present_values(rows: list[dict], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
            continue
        values.append(float(value))
    return values


def _fmt_range(low, high, *, decimals: int = 4) -> str:
    if low is None or high is None:
        return "-"
    if decimals == 0:
        return f"{low:.0f}-{high:.0f}"
    return f"{low:.{decimals}f}-{high:.{decimals}f}"


def _source_kind_sort_key(source_kind: str, null_model: str | None) -> int:
    if source_kind == "observed":
        return 0
    if null_model == "markov1":
        return 1
    if null_model == "matched-lz":
        return 2
    return 9


def _clip(value, width: int) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
