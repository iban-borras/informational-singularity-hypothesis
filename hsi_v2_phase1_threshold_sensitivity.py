#!/usr/bin/env python3
"""HSI v2 Phase 1 threshold sensitivity launcher.

This script tests whether the Gate 1 compact-selective readout survives
one-at-a-time perturbations of the Phase 1 stability thresholds. It does not
change the Phase 1 tower implementation; it only calls hsi_v2_phase1_run.py
with predeclared CLI parameters and summarizes the resulting null envelope.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.progress import format_time
from v2.common.cli import resolve_dir


SCRIPT_DIR = Path(__file__).resolve().parent
PREREG_PATH = SCRIPT_DIR / "Documentation" / "HSI_v2_Phase1_Threshold_Sensitivity_Preregistration.md"

DEFAULT_PHASE1_DIR = "results/hsi_v2/phase1_sensitivity_thresholds"
DEFAULT_LEVEL0_ROOT = "results/level0/phi_snapshots"
DEFAULT_VARIANTS = ("B", "E", "I")
DEFAULT_NULLS = ("matched-lz", "phase-matched-lz")
DEFAULT_INITIAL_SEED = 607
DEFAULT_ESCALATION_SEED = 709
DEFAULT_SCALES = "8,12,16,20,24,28,32"
DEFAULT_POLICIES = "prefix,suffix"

BASELINE = {
    "cv_max": 1.5,
    "min_count_floor": 16,
    "min_count_rate": 1e-6,
    "min_segment_support": None,
}

CV_VALUES = (1.0, 1.5, 2.0, 2.5, 3.0, math.inf)
FLOOR_VALUES = (4, 8, 16, 32, 64)
RATE_VALUES = (0.0, 1e-7, 1e-6, 1e-5, 1e-4)
SUPPORT_VALUES = (1, 2, 3)

SURVIVE_SCORE_MIN = 0.75
SURVIVE_SCORE_MARGIN = 0.25
SURVIVE_RETENTION_MIN = 0.70
SURVIVE_RETENTION_MARGIN = 0.25
WEAK_SCORE_MIN = 0.60
WEAK_SCORE_MARGIN = 0.10
WEAK_RETENTION_MIN = 0.55
WEAK_RETENTION_MARGIN = 0.10
ESCALATE_SCORE_MARGIN = 0.30
ESCALATE_RETENTION_MARGIN = 0.30


@dataclass(frozen=True)
class Combo:
    combo_id: str
    varied_parameter: str
    varied_value: str
    cv_max: float
    min_count_floor: int
    min_count_rate: float
    min_segment_support: int | None


@dataclass(frozen=True)
class Target:
    combo: Combo
    variant: str
    sequence_kind: str
    null_model: str | None
    null_seed: int | None

    @property
    def label(self) -> str:
        if self.sequence_kind == "observed":
            return f"{self.combo.combo_id}:{self.variant}:observed"
        return f"{self.combo.combo_id}:{self.variant}:{self.null_model}:seed-{self.null_seed}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the HSI v2 Phase 1 OAT threshold sensitivity analysis."
    )
    parser.add_argument("--level0-root", default=DEFAULT_LEVEL0_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_PHASE1_DIR)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--iteration", type=int, default=20)
    parser.add_argument("--segment-bits", type=int, default=1_000_000)
    parser.add_argument("--num-segments", type=int, default=3)
    parser.add_argument("--scales", default=DEFAULT_SCALES)
    parser.add_argument("--policies", default=DEFAULT_POLICIES)
    parser.add_argument("--initial-seed", type=int, default=DEFAULT_INITIAL_SEED)
    parser.add_argument("--escalation-seed", type=int, default=DEFAULT_ESCALATION_SEED)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-combos", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose-children", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.segment_bits <= 0 or args.num_segments <= 0:
        raise SystemExit("--segment-bits and --num-segments must be positive")

    variants = parse_csv(args.variants)
    if not variants:
        raise SystemExit("--variants cannot be empty")

    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    level0_root = resolve_dir(args.level0_root, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    combos = build_oat_combos()
    if args.limit_combos is not None:
        combos = combos[: max(0, args.limit_combos)]
    run_slug = f"phase1-threshold-sensitivity__oat-{len(combos)}__seed-{args.initial_seed}__{timestamp}"
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "run_id": run_slug,
        "timestamp_utc": timestamp,
        "status": "dry-run" if args.dry_run else "running",
        "preregistration": {
            "path": str(PREREG_PATH),
            "sha256": sha256_file(PREREG_PATH) if PREREG_PATH.exists() else None,
        },
        "parameters": {
            "level0_root": str(level0_root),
            "output_root": str(output_root),
            "variants": variants,
            "iteration": args.iteration,
            "segment_bits": args.segment_bits,
            "num_segments": args.num_segments,
            "scales": args.scales,
            "policies": args.policies,
            "null_models": list(DEFAULT_NULLS),
            "initial_seed": args.initial_seed,
            "escalation_seed": args.escalation_seed,
            "workers": args.workers,
            "grid": [combo_to_dict(combo) for combo in combos],
            "verdict_thresholds": verdict_thresholds(),
        },
        "initial_targets": [],
        "escalation_targets": [],
        "child_runs": [],
        "row_summary": [],
        "combo_summary": [],
    }

    started = time.perf_counter()
    try:
        phase_print(
            "Preparing Phase 1 threshold sensitivity",
            f"combos={len(combos)} | variants={','.join(variants)} | initial_seed={args.initial_seed}",
            quiet=args.quiet,
        )
        initial_targets = build_targets(combos, variants, seed=args.initial_seed, include_observed=True)
        state["initial_targets"] = [target_to_dict(target) for target in initial_targets]

        if args.dry_run:
            state["status"] = "dry-run"
            state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            write_outputs(run_dir, state)
            print(render_console_summary(state))
            print_saved_paths(run_dir)
            return 0

        child_rows = run_targets(
            targets=initial_targets,
            args=args,
            run_dir=run_dir,
            level0_root=level0_root,
            quiet=args.quiet,
        )
        combo_summary = summarize_combos(combos, child_rows)
        escalation_combo_ids = [
            row["combo_id"]
            for row in combo_summary
            if should_escalate(row)
        ]

        escalation_targets: list[Target] = []
        if escalation_combo_ids:
            phase_print(
                "Escalating fragile combinations",
                f"combos={','.join(escalation_combo_ids)} | extra_seed={args.escalation_seed}",
                quiet=args.quiet,
            )
            escalation_combos = [combo for combo in combos if combo.combo_id in set(escalation_combo_ids)]
            escalation_targets = build_targets(
                escalation_combos,
                variants,
                seed=args.escalation_seed,
                include_observed=False,
            )
            child_rows.extend(
                run_targets(
                    targets=escalation_targets,
                    args=args,
                    run_dir=run_dir,
                    level0_root=level0_root,
                    quiet=args.quiet,
                )
            )
            combo_summary = summarize_combos(combos, child_rows)

        state["status"] = "completed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        state["escalation_targets"] = [target_to_dict(target) for target in escalation_targets]
        state["child_runs"] = [
            {
                "label": row["target_label"],
                "run_dir": row["run_dir"],
                "summary_path": row["summary_path"],
                "manifest_path": row["manifest_path"],
            }
            for row in child_rows
        ]
        state["row_summary"] = child_rows
        state["combo_summary"] = combo_summary
        write_outputs(run_dir, state)
        print(render_console_summary(state))
        print_saved_paths(run_dir)
        return 0
    except subprocess.CalledProcessError as exc:
        state["status"] = "failed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        state["error"] = {"returncode": int(exc.returncode), "command": [str(part) for part in exc.cmd]}
        write_outputs(run_dir, state)
        print(f"[x] Phase 1 threshold sensitivity failed after {format_time(time.perf_counter() - started)}")
        print(f"Saved failure summary to: {run_dir / 'summary.json'}")
        return int(exc.returncode) if int(exc.returncode) > 0 else 1
    except Exception as exc:
        state["status"] = "failed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        state["error"] = str(exc)
        write_outputs(run_dir, state)
        print(f"[x] Phase 1 threshold sensitivity failed after {format_time(time.perf_counter() - started)}: {exc}")
        print(f"Saved failure summary to: {run_dir / 'summary.json'}")
        return 1


def build_oat_combos() -> list[Combo]:
    combos: list[Combo] = [
        Combo(
            combo_id="baseline",
            varied_parameter="baseline",
            varied_value="canonical",
            cv_max=BASELINE["cv_max"],
            min_count_floor=BASELINE["min_count_floor"],
            min_count_rate=BASELINE["min_count_rate"],
            min_segment_support=BASELINE["min_segment_support"],
        )
    ]
    for value in CV_VALUES:
        if value == BASELINE["cv_max"]:
            continue
        combos.append(make_combo("cv_max", format_cv(value), cv_max=value))
    for value in FLOOR_VALUES:
        if value == BASELINE["min_count_floor"]:
            continue
        combos.append(make_combo("min_count_floor", str(value), min_count_floor=value))
    for value in RATE_VALUES:
        if value == BASELINE["min_count_rate"]:
            continue
        combos.append(make_combo("min_count_rate", format_rate(value), min_count_rate=value))
    for value in SUPPORT_VALUES:
        if value == 2:
            continue
        combos.append(make_combo("min_segment_support", str(value), min_segment_support=value))
    return combos


def make_combo(varied_parameter: str, varied_value: str, **updates: Any) -> Combo:
    values = dict(BASELINE)
    values.update(updates)
    combo_id = f"{short_param(varied_parameter)}-{slug_value(varied_value)}"
    return Combo(
        combo_id=combo_id,
        varied_parameter=varied_parameter,
        varied_value=varied_value,
        cv_max=float(values["cv_max"]),
        min_count_floor=int(values["min_count_floor"]),
        min_count_rate=float(values["min_count_rate"]),
        min_segment_support=values["min_segment_support"],
    )


def build_targets(combos: list[Combo], variants: list[str], *, seed: int, include_observed: bool) -> list[Target]:
    targets: list[Target] = []
    for combo in combos:
        for variant in variants:
            if include_observed:
                targets.append(Target(combo, variant, "observed", None, None))
            for null_model in DEFAULT_NULLS:
                targets.append(Target(combo, variant, "null_surrogate", null_model, seed))
    return targets


def run_targets(
    *,
    targets: list[Target],
    args: argparse.Namespace,
    run_dir: Path,
    level0_root: Path,
    quiet: bool,
) -> list[dict[str, Any]]:
    if not targets:
        return []
    rows: list[dict[str, Any]] = []
    completed = 0
    total = len(targets)
    phase_print("Running Phase 1 child targets", f"targets={total} | workers={args.workers}", quiet=quiet)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(run_one_target, target, args, run_dir, level0_root): target
            for target in targets
        }
        for future in as_completed(future_map):
            target = future_map[future]
            row = future.result()
            rows.append(row)
            completed += 1
            if not quiet:
                print(
                    f"[Target] {completed}/{total} | {target.label} | "
                    f"score={row['compact_selective_score']:.4f} | ret={row['retention_last']:.4f}"
                )
                sys.stdout.flush()
    rows.sort(key=lambda item: (item["combo_id"], item["variant"], item["sequence_kind"], item.get("null_model") or "", item.get("null_seed") or 0))
    return rows


def run_one_target(target: Target, args: argparse.Namespace, run_dir: Path, level0_root: Path) -> dict[str, Any]:
    target_dir = run_dir / "children" / target.combo.combo_id / target_dir_name(target)
    target_dir.mkdir(parents=True, exist_ok=True)
    command = build_phase1_command(target, args, target_dir, level0_root)
    if not args.verbose_children:
        command.append("--quiet")
    subprocess.run(command, cwd=str(SCRIPT_DIR), check=True)
    child_run_dir = latest_run_dir(target_dir, target.sequence_kind)
    return load_target_row(target, child_run_dir, command)


def build_phase1_command(target: Target, args: argparse.Namespace, output_dir: Path, level0_root: Path) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "hsi_v2_phase1_run.py"),
        "--variant",
        target.variant,
        "--iteration",
        str(args.iteration),
        "--level0-root",
        str(level0_root),
        "--segment-bits",
        str(args.segment_bits),
        "--num-segments",
        str(args.num_segments),
        "--scales",
        str(args.scales),
        "--policies",
        str(args.policies),
        "--min-count-floor",
        str(target.combo.min_count_floor),
        "--min-count-rate",
        format_rate(target.combo.min_count_rate),
        "--cv-max",
        format_cv(target.combo.cv_max),
        "--output-dir",
        str(output_dir),
    ]
    if target.combo.min_segment_support is not None:
        command.extend(["--min-segment-support", str(target.combo.min_segment_support)])
    if target.sequence_kind == "null_surrogate":
        command.extend(["--null-model", str(target.null_model), "--null-seed", str(target.null_seed)])
    return command


def latest_run_dir(output_dir: Path, sequence_kind: str) -> Path:
    run_root = output_dir / "nulls" if sequence_kind == "null_surrogate" else output_dir
    candidates = sorted(run_root.glob("*/dataset.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Could not find child dataset under {run_root}")
    return candidates[0].parent


def load_target_row(target: Target, child_run_dir: Path, command: list[str]) -> dict[str, Any]:
    dataset_path = child_run_dir / "dataset.json"
    summary_path = child_run_dir / "phase1_summary.json"
    manifest_path = child_run_dir / "manifest.json"
    with open(dataset_path, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    last = summary["scale_summary"][-1]
    prefix = summary["fiber_summary"]["prefix"][-1]
    suffix = summary["fiber_summary"]["suffix"][-1]
    retention = safe_ratio(last["kept_pattern_count"], last["candidate_pattern_count"])
    min_survival = min(float(prefix["survival_fraction"]), float(suffix["survival_fraction"]))
    score = retention * min_survival
    return {
        **combo_to_dict(target.combo),
        "target_label": target.label,
        "variant": target.variant,
        "sequence_kind": target.sequence_kind,
        "null_model": target.null_model,
        "null_seed": target.null_seed,
        "kept_last": int(last["kept_pattern_count"]),
        "candidate_last": int(last["candidate_pattern_count"]),
        "retention_last": retention,
        "prefix_survival_last": float(prefix["survival_fraction"]),
        "suffix_survival_last": float(suffix["survival_fraction"]),
        "prefix_mean_fiber_last": float(prefix["mean_fiber_size"]),
        "suffix_mean_fiber_last": float(suffix["mean_fiber_size"]),
        "prefix_child_entropy_last": float(prefix["mean_child_entropy"]),
        "suffix_child_entropy_last": float(suffix["mean_child_entropy"]),
        "compact_selective_score": score,
        "run_dir": str(child_run_dir.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "summary_sha256": sha256_file(summary_path),
        "child_command": [str(part) for part in command],
        "child_config": dataset.get("config", {}),
    }


def summarize_combos(combos: list[Combo], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_combo: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_combo.setdefault(row["combo_id"], []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for combo in combos:
        combo_rows = by_combo.get(combo.combo_id, [])
        observed = [row for row in combo_rows if row["sequence_kind"] == "observed"]
        nulls = [row for row in combo_rows if row["sequence_kind"] == "null_surrogate"]
        if not observed or not nulls:
            summary_rows.append({**combo_to_dict(combo), "status": "incomplete", "verdict": "missing"})
            continue
        hsi_score_min = min(row["compact_selective_score"] for row in observed)
        null_score_max = max(row["compact_selective_score"] for row in nulls)
        hsi_retention_min = min(row["retention_last"] for row in observed)
        null_retention_max = max(row["retention_last"] for row in nulls)
        score_margin = hsi_score_min - null_score_max
        retention_margin = hsi_retention_min - null_retention_max
        verdict = classify_combo(hsi_score_min, score_margin, hsi_retention_min, retention_margin)
        summary_rows.append(
            {
                **combo_to_dict(combo),
                "status": "complete",
                "active_seed_count": len({row["null_seed"] for row in nulls if row.get("null_seed") is not None}),
                "observed_rows": len(observed),
                "null_rows": len(nulls),
                "hsi_score_min": hsi_score_min,
                "null_score_max": null_score_max,
                "score_margin": score_margin,
                "hsi_retention_min": hsi_retention_min,
                "null_retention_max": null_retention_max,
                "retention_margin": retention_margin,
                "hsi_prefix_survival_min": min(row["prefix_survival_last"] for row in observed),
                "hsi_suffix_survival_min": min(row["suffix_survival_last"] for row in observed),
                "null_prefix_survival_max": max(row["prefix_survival_last"] for row in nulls),
                "null_suffix_survival_max": max(row["suffix_survival_last"] for row in nulls),
                "verdict": verdict,
            }
        )
    return summary_rows


def classify_combo(hsi_score_min: float, score_margin: float, hsi_retention_min: float, retention_margin: float) -> str:
    if (
        hsi_score_min >= SURVIVE_SCORE_MIN
        and score_margin >= SURVIVE_SCORE_MARGIN
        and hsi_retention_min >= SURVIVE_RETENTION_MIN
        and retention_margin >= SURVIVE_RETENTION_MARGIN
    ):
        return "survives"
    if (
        hsi_score_min >= WEAK_SCORE_MIN
        and score_margin >= WEAK_SCORE_MARGIN
        and hsi_retention_min >= WEAK_RETENTION_MIN
        and retention_margin >= WEAK_RETENTION_MARGIN
    ):
        return "weakens"
    return "collapses"


def should_escalate(combo_row: dict[str, Any]) -> bool:
    if combo_row.get("verdict") != "survives":
        return True
    return (
        float(combo_row.get("score_margin", 0.0)) < ESCALATE_SCORE_MARGIN
        or float(combo_row.get("retention_margin", 0.0)) < ESCALATE_RETENTION_MARGIN
    )


def write_outputs(run_dir: Path, state: dict[str, Any]) -> None:
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    combo_csv_path = run_dir / "combo_summary.csv"
    row_csv_path = run_dir / "row_summary.csv"
    heatmap_csv_path = run_dir / "oat_heatmap.csv"
    heatmap_svg_path = run_dir / "oat_heatmap.svg"
    manifest_path = run_dir / "manifest.json"

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    write_csv(state.get("combo_summary", []), combo_csv_path)
    write_csv(state.get("row_summary", []), row_csv_path)
    write_heatmap_csv(state.get("combo_summary", []), heatmap_csv_path)
    write_heatmap_svg(state.get("combo_summary", []), heatmap_svg_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown_report(state) + "\n")

    manifest = {
        "run_id": state["run_id"],
        "timestamp_utc": state["timestamp_utc"],
        "status": state["status"],
        "script": "hsi_v2_phase1_threshold_sensitivity.py",
        "cwd": str(Path.cwd()),
        "preregistration": state["preregistration"],
        "parameters": state["parameters"],
        "initial_targets": state["initial_targets"],
        "escalation_targets": state["escalation_targets"],
        "child_runs": state["child_runs"],
        "outputs": {
            "summary": str(summary_path.resolve()),
            "report": str(report_path.resolve()),
            "combo_summary_csv": str(combo_csv_path.resolve()),
            "row_summary_csv": str(row_csv_path.resolve()),
            "oat_heatmap_csv": str(heatmap_csv_path.resolve()),
            "oat_heatmap_svg": str(heatmap_svg_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        },
    }
    if "error" in state:
        manifest["error"] = state["error"]
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys() if key != "child_config" and key != "child_command"})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_heatmap_csv(combo_rows: list[dict[str, Any]], path: Path) -> None:
    rows = [
        {
            "varied_parameter": row.get("varied_parameter"),
            "varied_value": row.get("varied_value"),
            "score_margin": row.get("score_margin"),
            "retention_margin": row.get("retention_margin"),
            "verdict": row.get("verdict"),
        }
        for row in combo_rows
    ]
    write_csv(rows, path)


def write_heatmap_svg(combo_rows: list[dict[str, Any]], path: Path) -> None:
    if not combo_rows:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"300\" height=\"80\"></svg>\n", encoding="utf-8")
        return
    cell_w = 92
    cell_h = 34
    left = 150
    top = 50
    rows_by_param: dict[str, list[dict[str, Any]]] = {}
    for row in combo_rows:
        rows_by_param.setdefault(str(row.get("varied_parameter")), []).append(row)
    params = list(rows_by_param)
    max_cols = max(len(items) for items in rows_by_param.values())
    width = left + max_cols * cell_w + 40
    height = top + len(params) * cell_h + 50
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8f5ed"/>',
        '<text x="20" y="28" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#172033">Phase 1 threshold sensitivity</text>',
    ]
    for r, param in enumerate(params):
        y = top + r * cell_h
        parts.append(f'<text x="20" y="{y + 22}" font-family="Arial, sans-serif" font-size="13" fill="#172033">{escape_xml(param)}</text>')
        for c, row in enumerate(rows_by_param[param]):
            x = left + c * cell_w
            margin = row.get("score_margin")
            verdict = str(row.get("verdict"))
            fill = verdict_color(verdict, margin)
            text_color = "#101827" if fill in ("#f4d35e", "#cdeac0") else "#ffffff"
            label = str(row.get("varied_value"))
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" rx="4" fill="{fill}"/>')
            parts.append(f'<text x="{x + 8}" y="{y + 14}" font-family="Arial, sans-serif" font-size="11" fill="{text_color}">{escape_xml(label)}</text>')
            parts.append(f'<text x="{x + 8}" y="{y + 27}" font-family="Arial, sans-serif" font-size="11" fill="{text_color}">{format_float(margin)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def verdict_color(verdict: str, margin: Any) -> str:
    if verdict == "survives":
        return "#16837a"
    if verdict == "weakens":
        return "#f4d35e"
    if verdict == "collapses":
        return "#b23a48"
    if margin is None:
        return "#9aa4b2"
    return "#9aa4b2"


def render_console_summary(state: dict[str, Any]) -> str:
    lines = [
        "Phase 1 threshold sensitivity",
        "-" * 116,
        f"{'combo':<18} {'param':<20} {'value':>10} {'hsi':>8} {'null':>8} {'margin':>8} {'retGap':>8} {'verdict':>10}",
    ]
    for row in state.get("combo_summary", []):
        lines.append(
            f"{row.get('combo_id',''):<18} "
            f"{row.get('varied_parameter',''):<20} "
            f"{row.get('varied_value',''):>10} "
            f"{format_float(row.get('hsi_score_min')):>8} "
            f"{format_float(row.get('null_score_max')):>8} "
            f"{format_float(row.get('score_margin')):>8} "
            f"{format_float(row.get('retention_margin')):>8} "
            f"{row.get('verdict',''):>10}"
        )
    if not state.get("combo_summary"):
        lines.append(f"Dry run targets: {len(state.get('initial_targets', []))}")
    return "\n".join(lines)


def render_markdown_report(state: dict[str, Any]) -> str:
    lines = [
        "# HSI v2 Phase 1 Threshold Sensitivity",
        "",
        "## Contract",
        "",
        f"- Preregistration: `{state['preregistration']['path']}`",
        f"- Preregistration SHA256: `{state['preregistration']['sha256']}`",
        f"- Status: `{state['status']}`",
        "",
        "## Verdict Table",
        "",
        "| Combo | Parameter | Value | HSI score min | Null score max | Score margin | Retention margin | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in state.get("combo_summary", []):
        lines.append(
            f"| `{row.get('combo_id')}` | `{row.get('varied_parameter')}` | `{row.get('varied_value')}` | "
            f"{format_float(row.get('hsi_score_min'))} | {format_float(row.get('null_score_max'))} | "
            f"{format_float(row.get('score_margin'))} | {format_float(row.get('retention_margin'))} | "
            f"`{row.get('verdict')}` |"
        )
    if not state.get("combo_summary"):
        lines.append(f"| dry-run | - | - | - | - | - | - | `{len(state.get('initial_targets', []))} targets` |")
    if state.get("escalation_targets"):
        lines.extend(["", "## Escalation", ""])
        lines.append(f"- Escalation targets run: `{len(state['escalation_targets'])}`")
    if state.get("error"):
        lines.extend(["", "## Error", "", f"`{state['error']}`"])
    return "\n".join(lines)


def print_saved_paths(run_dir: Path) -> None:
    print(f"\nSaved summary to: {run_dir / 'summary.json'}")
    print(f"Saved report to: {run_dir / 'report.md'}")
    print(f"Saved combo CSV to: {run_dir / 'combo_summary.csv'}")
    print(f"Saved row CSV to: {run_dir / 'row_summary.csv'}")
    print(f"Saved heatmap CSV to: {run_dir / 'oat_heatmap.csv'}")
    print(f"Saved heatmap SVG to: {run_dir / 'oat_heatmap.svg'}")
    print(f"Saved manifest to: {run_dir / 'manifest.json'}")


def parse_csv(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def combo_to_dict(combo: Combo) -> dict[str, Any]:
    return {
        "combo_id": combo.combo_id,
        "varied_parameter": combo.varied_parameter,
        "varied_value": combo.varied_value,
        "cv_max": format_cv(combo.cv_max),
        "min_count_floor": combo.min_count_floor,
        "min_count_rate": format_rate(combo.min_count_rate),
        "min_segment_support": combo.min_segment_support,
    }


def target_to_dict(target: Target) -> dict[str, Any]:
    return {
        **combo_to_dict(target.combo),
        "variant": target.variant,
        "sequence_kind": target.sequence_kind,
        "null_model": target.null_model,
        "null_seed": target.null_seed,
        "label": target.label,
    }


def verdict_thresholds() -> dict[str, float]:
    return {
        "survive_score_min": SURVIVE_SCORE_MIN,
        "survive_score_margin": SURVIVE_SCORE_MARGIN,
        "survive_retention_min": SURVIVE_RETENTION_MIN,
        "survive_retention_margin": SURVIVE_RETENTION_MARGIN,
        "weak_score_min": WEAK_SCORE_MIN,
        "weak_score_margin": WEAK_SCORE_MARGIN,
        "weak_retention_min": WEAK_RETENTION_MIN,
        "weak_retention_margin": WEAK_RETENTION_MARGIN,
        "escalate_score_margin": ESCALATE_SCORE_MARGIN,
        "escalate_retention_margin": ESCALATE_RETENTION_MARGIN,
    }


def target_dir_name(target: Target) -> str:
    if target.sequence_kind == "observed":
        return f"{target.variant}__observed"
    return f"{target.variant}__{target.null_model}__seed-{target.null_seed}"


def short_param(parameter: str) -> str:
    return {
        "cv_max": "cv",
        "min_count_floor": "floor",
        "min_count_rate": "rate",
        "min_segment_support": "support",
    }.get(parameter, parameter)


def slug_value(value: str) -> str:
    return value.replace(".", "p").replace("+", "").replace("-", "neg").replace("e", "e")


def format_cv(value: float) -> str:
    return "inf" if math.isinf(float(value)) else f"{float(value):g}"


def format_rate(value: float) -> str:
    return f"{float(value):g}"


def format_float(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def escape_xml(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def phase_print(title: str, detail: str = "", quiet: bool = False) -> None:
    if quiet:
        return
    print(f"\n[Phase] {title}")
    if detail:
        print(f"        {detail}")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
