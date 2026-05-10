#!/usr/bin/env python3
"""HSI v2 Phase 5 Candidate B-mini cross-arc horizon launcher.

This implements D-0104. It runs the strict P4-object check, the broad scout,
and post-scout local analogue brackets, while keeping the B-mini continue/stop
decision tied only to strict P4-grid replication.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hsi_v2_phase2_parent_survival_revalidation import latest_child_file
from hsi_v2_phase2_transport_defect_strict import phase_print
from utils.progress import format_time
from v2.common.cli import resolve_dir
from v2.common.naming import compact_int


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_PHASE1_DIR = "results/hsi_v2/phase1_high_scales"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase5_cross_arc_horizon/bmini_v0_2"
DEFAULT_CONTRACT = "Documentation/HSI_v2_Phase5_CandidateB_CrossArc_Preregistration_v0_2.md"
DEFAULT_SEEDS = "607,709,811,907,1009,1103"
P4_LAGS = (-29_500_000, -27_000_000, -26_500_000, -24_000_000, 24_500_000, 26_500_000)
SCOUT_LAGS = tuple(range(-60_000_000, 60_000_000 + 1, 2_500_000))
NULL_MODELS = "markov1,phase-matched-lz,block-entropy"

STRONG_B_THRESHOLD = 0.90
STRONG_MARGIN_THRESHOLD = 0.30
SHADOW_MARGIN_THRESHOLD = 0.05
SHADOW_B_MIN = 0.50
SCOUT_B_THRESHOLD = 0.70
SCOUT_MARGIN_THRESHOLD = 0.10
SCOUT_CLUSTER_MIN = 3
SCOUT_ANCHOR_LAG = -27_000_000

ARCS = (
    {"arc_id": "ARC-A", "scout_start_bits": 300_000_000},
    {"arc_id": "ARC-B", "scout_start_bits": 597_000_000},
    {"arc_id": "ARC-C", "scout_start_bits": 1_110_000_000},
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 5 B-mini cross-arc finite-horizon test.")
    parser.add_argument("--phase1-dir", default=DEFAULT_PHASE1_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    contract_path = resolve_dir(args.contract, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"phase5-bmini-cross-arc-horizon__arcs-3__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    state: dict[str, Any] = {
        "run_id": run_dir.name,
        "timestamp_utc": timestamp,
        "status": "dry-run" if args.dry_run else "running",
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": sha256_file(contract_path),
        },
        "parameters": {
            "phase1_dir": str(phase1_dir),
            "output_dir": str(output_root),
            "seeds": args.seeds,
            "workers": args.workers,
            "null_models": NULL_MODELS,
            "p4_lags": list(P4_LAGS),
            "scout_lags": list(SCOUT_LAGS),
            "arcs": build_arc_specs(),
            "thresholds": {
                "strong_b": STRONG_B_THRESHOLD,
                "strong_margin": STRONG_MARGIN_THRESHOLD,
                "shadow_margin": SHADOW_MARGIN_THRESHOLD,
                "shadow_b_min": SHADOW_B_MIN,
                "scout_b": SCOUT_B_THRESHOLD,
                "scout_margin": SCOUT_MARGIN_THRESHOLD,
                "scout_cluster_min": SCOUT_CLUSTER_MIN,
            },
        },
        "commands": {},
        "strict_p4_rows": [],
        "scout_rows": [],
        "bracket_rows": [],
        "arc_verdicts": [],
        "verdict": {},
    }

    try:
        phase_print(
            "Preparing Phase 5 B-mini cross-arc horizon test",
            "arcs=300M,597M,1110M | strict=P4-grid | scout=-60M..60M step=2.5M",
            quiet=args.quiet,
        )

        strict_cmd = build_lag_response_command(
            phase1_dir=phase1_dir,
            output_dir=run_dir / "strict_p4",
            band_starts=[start for arc in build_arc_specs() for start in arc["all_band_starts_bits"]],
            lags=P4_LAGS,
            seeds=args.seeds,
            workers=args.workers,
            quiet=args.quiet,
        )
        scout_cmd = build_lag_response_command(
            phase1_dir=phase1_dir,
            output_dir=run_dir / "scout",
            band_starts=[arc["scout_start_bits"] for arc in build_arc_specs()],
            lags=SCOUT_LAGS,
            seeds=args.seeds,
            workers=args.workers,
            quiet=args.quiet,
        )
        state["commands"]["strict_p4"] = [str(part) for part in strict_cmd]
        state["commands"]["scout"] = [str(part) for part in scout_cmd]

        if args.dry_run:
            state["status"] = "dry-run"
            state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            write_all_outputs(run_dir, state)
            print(render_console_summary(state))
            print_saved_paths(run_dir)
            return 0

        run_step("Strict P4-object check", strict_cmd, quiet=args.quiet)
        strict_csv = latest_child_file(run_dir / "strict_p4", "lag_response.csv")
        strict_rows = annotate_rows(load_csv_dicts(strict_csv), phase="strict_p4")

        run_step("Broad lag scout", scout_cmd, quiet=args.quiet)
        scout_csv = latest_child_file(run_dir / "scout", "lag_response.csv")
        scout_rows = annotate_rows(load_csv_dicts(scout_csv), phase="scout")
        scout_decisions = build_scout_decisions(scout_rows)

        bracket_rows: list[dict[str, Any]] = []
        bracket_commands: dict[str, list[str]] = {}
        for arc in build_arc_specs():
            decision = scout_decisions[arc["arc_id"]]
            if not decision["scout_active"]:
                continue
            cmd = build_lag_response_command(
                phase1_dir=phase1_dir,
                output_dir=run_dir / "brackets" / arc["arc_id"],
                band_starts=arc["bracket_band_starts_bits"],
                lags=tuple(int(value) for value in decision["local_lag_core_bits"]),
                seeds=args.seeds,
                workers=args.workers,
                quiet=args.quiet,
            )
            bracket_commands[arc["arc_id"]] = [str(part) for part in cmd]
            run_step(f"Local analogue bracket {arc['arc_id']}", cmd, quiet=args.quiet)
            csv_path = latest_child_file(run_dir / "brackets" / arc["arc_id"], "lag_response.csv")
            bracket_rows.extend(annotate_rows(load_csv_dicts(csv_path), phase="local_bracket", arc_id=arc["arc_id"]))

        state["commands"]["brackets"] = bracket_commands
        strict_rows = annotate_arc_ids(strict_rows)
        scout_rows = annotate_arc_ids(scout_rows)
        bracket_rows = annotate_arc_ids(bracket_rows)
        arc_verdicts = build_arc_verdicts(strict_rows, scout_rows, bracket_rows, scout_decisions)
        state.update(
            {
                "status": "completed",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "strict_p4_rows": strict_rows,
                "scout_rows": scout_rows,
                "bracket_rows": bracket_rows,
                "scout_decisions": scout_decisions,
                "arc_verdicts": arc_verdicts,
                "verdict": build_overall_verdict(arc_verdicts),
            }
        )
        write_all_outputs(run_dir, state)
        print(render_console_summary(state))
        print_saved_paths(run_dir)
        return 0
    except subprocess.CalledProcessError as exc:
        state["status"] = "failed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        state["error"] = {"returncode": int(exc.returncode), "command": [str(part) for part in exc.cmd]}
        write_all_outputs(run_dir, state)
        print(f"[x] Phase 5 B-mini failed after {format_time(time.perf_counter() - started)}")
        print_saved_paths(run_dir)
        return int(exc.returncode) if int(exc.returncode) > 0 else 1
    except Exception as exc:
        state["status"] = "failed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        state["error"] = str(exc)
        write_all_outputs(run_dir, state)
        print(f"[x] Phase 5 B-mini failed after {format_time(time.perf_counter() - started)}: {exc}")
        print_saved_paths(run_dir)
        return 1


def build_arc_specs() -> list[dict[str, Any]]:
    specs = []
    for arc in ARCS:
        scout = int(arc["scout_start_bits"])
        bracket = [scout + step * 9_000_000 for step in range(1, 5)]
        all_starts = [scout, *bracket]
        specs.append(
            {
                "arc_id": arc["arc_id"],
                "scout_start_bits": scout,
                "scout_band": band_label(scout),
                "bracket_band_starts_bits": bracket,
                "bracket_bands": [band_label(value) for value in bracket],
                "all_band_starts_bits": all_starts,
                "all_bands": [band_label(value) for value in all_starts],
            }
        )
    return specs


def build_lag_response_command(
    *,
    phase1_dir: Path,
    output_dir: Path,
    band_starts: list[int],
    lags: tuple[int, ...],
    seeds: str,
    workers: int,
    quiet: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "hsi_v2_phase2_child_routing_lag_response.py"),
        "--phase1-dir",
        str(phase1_dir),
        "--output-dir",
        str(output_dir),
        "--stage",
        "all",
        "--top-patterns",
        "128",
        "--band-starts",
        ",".join(str(value) for value in band_starts),
        f"--lags={','.join(str(value) for value in lags)}",
        "--null-models",
        NULL_MODELS,
        "--matched-lz-seeds",
        seeds,
        "--workers",
        str(workers),
        "--quiet-children",
    ]
    if quiet:
        cmd.append("--quiet")
    return cmd


def run_step(label: str, command: list[str], *, quiet: bool) -> None:
    phase_print(label, command[-3] if len(command) > 3 else "", quiet=quiet)
    subprocess.run(command, cwd=str(SCRIPT_DIR), check=True)


def annotate_rows(rows: list[dict[str, str]], *, phase: str, arc_id: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        item["phase"] = phase
        if arc_id:
            item["arc_id"] = arc_id
        metrics = extract_metrics(row)
        item.update(metrics)
        out.append(item)
    return out


def annotate_arc_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    band_to_arc = {}
    for arc in build_arc_specs():
        for band in arc["all_bands"]:
            band_to_arc[band] = arc["arc_id"]
    for row in rows:
        row.setdefault("arc_id", band_to_arc.get(str(row.get("band", "")), "unknown"))
    return rows


def extract_metrics(row: dict[str, Any]) -> dict[str, Any]:
    b_ret = parse_float(row.get("B_retention"))
    pmlz = parse_float(row.get("phase_matched_lz_max"))
    bent = parse_float(row.get("block_entropy_max"))
    markov = parse_float(row.get("markov1_retention"))
    hard_null = max(value for value in [pmlz, bent, markov] if value is not None)
    return {
        "B_retention_float": b_ret,
        "phase_matched_lz_max_float": pmlz,
        "block_entropy_max_float": bent,
        "markov1_retention_float": markov,
        "hard_null_max": hard_null,
        "hard_margin": None if b_ret is None else b_ret - hard_null,
    }


def build_scout_decisions(scout_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for arc in build_arc_specs():
        rows = [
            row
            for row in scout_rows
            if row.get("arc_id") == arc["arc_id"] and int(row["lag_bits"]) < 0
        ]
        eligible = [
            row
            for row in rows
            if float(row["B_retention_float"]) >= SCOUT_B_THRESHOLD
            and float(row["hard_margin"]) >= SCOUT_MARGIN_THRESHOLD
        ]
        clusters = build_connected_clusters(eligible)
        active_clusters = [cluster for cluster in clusters if len(cluster) >= SCOUT_CLUSTER_MIN]
        if not active_clusters:
            decisions[arc["arc_id"]] = {
                "arc_id": arc["arc_id"],
                "scout_active": False,
                "eligible_lags": [int(row["lag_bits"]) for row in eligible],
                "clusters": [[int(row["lag_bits"]) for row in cluster] for cluster in clusters],
                "local_lag_core_bits": [],
                "p4_anchor_tiebreak_used": False,
            }
            continue
        selected, used_anchor = choose_cluster(active_clusters)
        core = choose_core(selected)
        decisions[arc["arc_id"]] = {
            "arc_id": arc["arc_id"],
            "scout_active": True,
            "eligible_lags": [int(row["lag_bits"]) for row in eligible],
            "clusters": [[int(row["lag_bits"]) for row in cluster] for cluster in clusters],
            "selected_cluster_bits": [int(row["lag_bits"]) for row in selected],
            "local_lag_core_bits": [int(row["lag_bits"]) for row in core],
            "p4_anchor_tiebreak_used": used_anchor,
        }
    return decisions


def build_connected_clusters(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    rows = sorted(rows, key=lambda row: int(row["lag_bits"]))
    clusters: list[list[dict[str, Any]]] = []
    for row in rows:
        if not clusters or int(row["lag_bits"]) - int(clusters[-1][-1]["lag_bits"]) != 2_500_000:
            clusters.append([row])
        else:
            clusters[-1].append(row)
    return clusters


def choose_cluster(clusters: list[list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], bool]:
    scored = []
    for cluster in clusters:
        margins = [float(row["hard_margin"]) for row in cluster]
        lags = [int(row["lag_bits"]) for row in cluster]
        median = lags[len(lags) // 2]
        scored.append(
            (
                len(cluster),
                sum(margins) / len(margins),
                -abs(median - SCOUT_ANCHOR_LAG),
                -median,
                cluster,
            )
        )
    scored.sort(reverse=True, key=lambda item: item[:4])
    winner = scored[0]
    anchor_used = len(scored) > 1 and winner[:2] == scored[1][:2]
    return winner[4], anchor_used


def choose_core(cluster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(cluster) <= 3:
        return cluster
    best = None
    for index in range(0, len(cluster) - 2):
        candidate = cluster[index : index + 3]
        score = sum(float(row["hard_margin"]) for row in candidate)
        if best is None or score > best[0]:
            best = (score, candidate)
    assert best is not None
    return best[1]


def build_arc_verdicts(
    strict_rows: list[dict[str, Any]],
    scout_rows: list[dict[str, Any]],
    bracket_rows: list[dict[str, Any]],
    scout_decisions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    verdicts = []
    for arc in build_arc_specs():
        arc_id = arc["arc_id"]
        strict = classify_finite_horizon([row for row in strict_rows if row["arc_id"] == arc_id])
        local = classify_finite_horizon([row for row in bracket_rows if row["arc_id"] == arc_id])
        scout = scout_decisions[arc_id]
        if strict["finite_horizon_like"]:
            category = "strict-replication"
        elif local["finite_horizon_like"]:
            category = "local-analogue-only"
        elif not scout["scout_active"]:
            category = "scout-inactive"
        else:
            category = "no-replication"
        verdicts.append(
            {
                "arc_id": arc_id,
                "scout_band": arc["scout_band"],
                "bracket_bands": ";".join(arc["bracket_bands"]),
                "category": category,
                "strict_finite_horizon_like": strict["finite_horizon_like"],
                "strict_strong_bands": strict["strong_bands"],
                "strict_shadow_bands": strict["shadow_bands"],
                "scout_active": scout["scout_active"],
                "local_lag_core_bits": ",".join(str(value) for value in scout.get("local_lag_core_bits", [])) or "-",
                "p4_anchor_tiebreak_used": scout.get("p4_anchor_tiebreak_used", False),
                "local_finite_horizon_like": local["finite_horizon_like"],
                "local_strong_bands": local["strong_bands"],
                "local_shadow_bands": local["shadow_bands"],
            }
        )
    return verdicts


def classify_finite_horizon(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_band: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_band.setdefault(str(row["band"]), []).append(row)
    strong_bands: list[str] = []
    shadow_bands: list[str] = []
    for band, band_rows in by_band.items():
        if any(is_strong(row) for row in band_rows):
            strong_bands.append(band)
        if any(is_shadow(row) for row in band_rows):
            shadow_bands.append(band)
    ordered_bands = list(by_band)
    finite = False
    if strong_bands and shadow_bands:
        first_strong = min(ordered_bands.index(band) for band in strong_bands if band in ordered_bands)
        last_shadow = max(ordered_bands.index(band) for band in shadow_bands if band in ordered_bands)
        finite = first_strong < last_shadow
    return {
        "finite_horizon_like": finite,
        "strong_bands": ";".join(strong_bands) if strong_bands else "-",
        "shadow_bands": ";".join(shadow_bands) if shadow_bands else "-",
    }


def is_strong(row: dict[str, Any]) -> bool:
    return float(row["B_retention_float"]) >= STRONG_B_THRESHOLD and float(row["hard_margin"]) >= STRONG_MARGIN_THRESHOLD


def is_shadow(row: dict[str, Any]) -> bool:
    return float(row["hard_margin"]) <= SHADOW_MARGIN_THRESHOLD and float(row["B_retention_float"]) >= SHADOW_B_MIN


def build_overall_verdict(arc_verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    strict_count = sum(1 for row in arc_verdicts if row["category"] == "strict-replication")
    local_count = sum(1 for row in arc_verdicts if row["category"] == "local-analogue-only")
    if strict_count >= 2:
        status = "supports-full-candidate-b-contract"
    elif strict_count == 1:
        status = "ambiguous-stop-and-review"
    else:
        status = "negative-bmini-no-full-b"
    return {
        "status": status,
        "strict_replication_count": strict_count,
        "local_analogue_only_count": local_count,
        "decision_basis": "strict P4-object check only; local analogue cannot rescue strict failure",
    }


def render_console_summary(state: dict[str, Any]) -> str:
    lines = ["", "Phase 5 B-mini cross-arc horizon", "-" * 112]
    if state.get("status") == "dry-run":
        lines.append("dry-run only | commands written, no lag-response execution")
        return "\n".join(lines)
    verdict = state.get("verdict", {})
    lines.append(
        f"status={verdict.get('status','-')} | strict={verdict.get('strict_replication_count','-')}/3 | "
        f"local-only={verdict.get('local_analogue_only_count','-')}/3"
    )
    for row in state.get("arc_verdicts", []):
        lines.append(
            f"{row['arc_id']:>5} | {row['category']:<20} | strict strong={row['strict_strong_bands']} "
            f"shadow={row['strict_shadow_bands']} | core={row['local_lag_core_bits']}"
        )
    return "\n".join(lines)


def write_all_outputs(run_dir: Path, state: dict[str, Any]) -> None:
    write_json(run_dir / "summary.json", state)
    write_report(run_dir, state)
    write_csv(run_dir / "strict_p4_results.csv", state.get("strict_p4_rows", []))
    write_csv(run_dir / "scout_results.csv", state.get("scout_rows", []))
    write_csv(run_dir / "bracket_results.csv", state.get("bracket_rows", []))
    write_csv(run_dir / "arc_verdicts.csv", state.get("arc_verdicts", []))
    source_rows = []
    for key in ("strict_p4_rows", "scout_rows", "bracket_rows"):
        source_rows.extend(state.get(key, []))
    write_csv(run_dir / "source_response.csv", source_rows)
    manifest = {
        "run_id": state["run_id"],
        "timestamp_utc": state["timestamp_utc"],
        "script": str(Path(__file__).resolve()),
        "status": state["status"],
        "contract": state["contract"],
        "parameters": state["parameters"],
        "outputs": {
            "summary": "summary.json",
            "report": "report.md",
            "strict_p4_results": "strict_p4_results.csv",
            "scout_results": "scout_results.csv",
            "bracket_results": "bracket_results.csv",
            "arc_verdicts": "arc_verdicts.csv",
            "source_response": "source_response.csv",
            "manifest": "manifest.json",
        },
    }
    write_json(run_dir / "manifest.json", manifest)


def write_report(run_dir: Path, state: dict[str, Any]) -> None:
    lines = [
        f"# {state['run_id']}",
        "",
        "## Contract",
        "",
        f"- Path: `{state['contract']['path']}`",
        f"- SHA256: `{state['contract']['sha256']}`",
        "",
        "## Verdict",
        "",
    ]
    if state.get("status") == "dry-run":
        lines.append("- Status: `dry-run`")
        lines.append("- No lag-response execution was run.")
    else:
        verdict = state.get("verdict", {})
        lines.append(f"- Status: `{verdict.get('status','-')}`")
        lines.append(f"- Strict replications: `{verdict.get('strict_replication_count','-')}/3`")
        lines.append(f"- Local-analogue-only arcs: `{verdict.get('local_analogue_only_count','-')}/3`")
        lines.append("")
        lines.append("The B-mini continue/stop decision is based only on the strict P4-object check.")
        lines.append("")
        lines.append("## Arc Verdicts")
        lines.append("")
        lines.append("| arc | category | strict strong | strict shadow | scout core | P4 tie-break used |")
        lines.append("|---|---|---|---|---|---|")
        for row in state.get("arc_verdicts", []):
            lines.append(
                f"| {row['arc_id']} | `{row['category']}` | `{row['strict_strong_bands']}` | "
                f"`{row['strict_shadow_bands']}` | `{row['local_lag_core_bits']}` | "
                f"`{row['p4_anchor_tiebreak_used']}` |"
            )
    lines.extend(["", "## Files", "", "- `summary.json`", "- `manifest.json`"])
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def print_saved_paths(run_dir: Path) -> None:
    print(f"\nSaved summary to: {run_dir / 'summary.json'}")
    print(f"Saved report to: {run_dir / 'report.md'}")
    print(f"Saved manifest to: {run_dir / 'manifest.json'}")


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def band_label(start_bits: int) -> str:
    return f"{compact_int(start_bits)}-{compact_int(start_bits + 9_000_000)}"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
