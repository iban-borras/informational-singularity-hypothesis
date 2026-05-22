#!/usr/bin/env python3
"""
Q3 empirical Dyck-randomized controls for mean_children.

This script tests whether non-HSI parenthesized controls can reproduce the
canonical nesting-tree mean_children value phi + 1 under the audited Dyck parser.

It is intentionally a harness around level1_nesting_tree.py. It does not modify
Level 0 generators or canonical parser semantics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "hsi_v2" / "q3_empirical_dyck_randomized"
REFERENCE_JSON = PROJECT_ROOT / "results" / "level1" / "analysis" / "nesting_tree_var_B_iter20_1G.json"
DEFAULT_SEEDS = (607, 709, 811)
DEFAULT_FAMILIES = ("UD", "MD", "DM")
PHI_PLUS_ONE = (1.0 + math.sqrt(5.0)) / 2.0 + 1.0


sys.path.insert(0, str(PROJECT_ROOT))
from level1_nesting_tree import build_tree_from_structural, collect_all_metrics_iterative  # noqa: E402


@dataclass(frozen=True)
class ReferenceStats:
    variant: str
    iteration: int
    segment_index: int
    segment_position: int
    segment_size: int
    reference_total_bits: int
    parenthesis_count: int
    parenthesis_density: float
    reference_mean_children: float
    reference_branching_nodes: int
    depth_distribution: Dict[int, int]
    source_json: str
    source_json_sha256: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def load_reference_stats(path: Path, segment_index: int = 0) -> ReferenceStats:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    if segment_index >= len(segments):
        raise ValueError(f"Reference segment {segment_index} not found in {path}")

    segment = segments[segment_index]
    segment_size = int(data["segment_size"])
    total_bits = int(segment["tree_stats"]["total_bits"])
    parens = segment_size - total_bits
    if parens <= 0:
        raise ValueError("Reference segment has no parentheses; cannot run Q3 controls")
    if parens % 2:
        # Keep the run balanced for Dyck controls by dropping one parenthesis slot.
        parens -= 1

    depth_raw = segment["tree_stats"].get("depth_distribution", {})
    depth_distribution = {int(k): int(v) for k, v in depth_raw.items()}

    return ReferenceStats(
        variant=str(data["variant"]),
        iteration=int(data["iteration"]),
        segment_index=int(segment.get("segment_index", segment_index)),
        segment_position=int(segment.get("segment_position", 0)),
        segment_size=segment_size,
        reference_total_bits=total_bits,
        parenthesis_count=parens,
        parenthesis_density=parens / segment_size,
        reference_mean_children=float(segment["children_analysis"]["mean_children"]),
        reference_branching_nodes=int(segment["children_analysis"]["total_branching_nodes"]),
        depth_distribution=depth_distribution,
        source_json=str(path),
        source_json_sha256=sha256_file(path),
    )


def _constrained_open_probability(
    opens_left: int,
    closes_left: int,
    depth: int,
    remaining_slots: int,
    proposed_p_open: float,
) -> Optional[bool]:
    """Return forced choice if constraints require it, else None."""
    if opens_left <= 0:
        return False
    if closes_left <= 0:
        return True
    if depth <= 0:
        return True
    # If all remaining slots are needed to close the current stack, close now.
    if depth >= remaining_slots:
        return False
    # Keep enough future closes to close all future opens.
    if closes_left <= opens_left - 1:
        return False
    return None


def make_ud_skeleton(parenthesis_count: int, rng: np.random.Generator) -> np.ndarray:
    """Reflected random-walk Dyck skeleton with exact balance."""
    pairs = parenthesis_count // 2
    opens_left = pairs
    closes_left = pairs
    depth = 0
    out = np.empty(parenthesis_count, dtype=np.uint8)

    for i in range(parenthesis_count):
        forced = _constrained_open_probability(
            opens_left, closes_left, depth, parenthesis_count - i, 0.5
        )
        if forced is None:
            take_open = bool(rng.random() < 0.5)
        else:
            take_open = forced

        if take_open:
            out[i] = ord("(")
            opens_left -= 1
            depth += 1
        else:
            out[i] = ord(")")
            closes_left -= 1
            depth -= 1

    return out


def make_md_probabilities(depth_distribution: Dict[int, int]) -> Dict[int, float]:
    """Birth-death probabilities whose local bias follows the B depth profile."""
    if not depth_distribution:
        return {0: 1.0}

    max_depth = max(depth_distribution)
    probs: Dict[int, float] = {}
    for depth in range(max_depth + 1):
        here = max(1, depth_distribution.get(depth, 0))
        nxt = max(1, depth_distribution.get(depth + 1, 0))
        # If the next layer is rarer, opening becomes less likely.
        p = nxt / (here + nxt)
        probs[depth] = min(0.92, max(0.08, p))
    probs[0] = 1.0
    probs[max_depth] = min(probs.get(max_depth, 0.08), 0.08)
    return probs


def make_md_skeleton(
    parenthesis_count: int,
    rng: np.random.Generator,
    depth_distribution: Dict[int, int],
) -> np.ndarray:
    """Balanced Dyck skeleton using B-derived depth-biased Markov transitions."""
    pairs = parenthesis_count // 2
    opens_left = pairs
    closes_left = pairs
    depth = 0
    probs = make_md_probabilities(depth_distribution)
    max_known_depth = max(probs) if probs else 0
    out = np.empty(parenthesis_count, dtype=np.uint8)

    for i in range(parenthesis_count):
        p_open = probs.get(min(depth, max_known_depth), 0.08)
        forced = _constrained_open_probability(
            opens_left, closes_left, depth, parenthesis_count - i, p_open
        )
        if forced is None:
            take_open = bool(rng.random() < p_open)
        else:
            take_open = forced

        if take_open:
            out[i] = ord("(")
            opens_left -= 1
            depth += 1
        else:
            out[i] = ord(")")
            closes_left -= 1
            depth -= 1

    return out


def make_dm_skeleton(parenthesis_count: int, rng: np.random.Generator) -> np.ndarray:
    """Malformed density-matched insertion skeleton, not balanced by construction."""
    draws = rng.integers(0, 2, size=parenthesis_count, dtype=np.uint8)
    return np.where(draws == 0, ord("("), ord(")")).astype(np.uint8)


def generate_control_string(
    family: str,
    seed: int,
    ref: ReferenceStats,
    output_chars: int,
    parenthesis_count: int,
) -> Tuple[str, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    started = time.time()

    print(
        f"[generate] {family} seed={seed} chars={output_chars:,} "
        f"parens={parenthesis_count:,}",
        flush=True,
    )

    # Fill terminals first; parentheses overwrite a fixed-size random subset.
    arr = rng.integers(0, 2, size=output_chars, dtype=np.uint8)
    arr += ord("0")

    pos_started = time.time()
    positions = rng.choice(output_chars, size=parenthesis_count, replace=False)
    positions.sort()
    print(f"[generate] {family} seed={seed} positions in {time.time() - pos_started:.1f}s", flush=True)

    sk_started = time.time()
    if family == "UD":
        skeleton = make_ud_skeleton(parenthesis_count, rng)
    elif family == "MD":
        skeleton = make_md_skeleton(parenthesis_count, rng, ref.depth_distribution)
    elif family == "DM":
        skeleton = make_dm_skeleton(parenthesis_count, rng)
    else:
        raise ValueError(f"Unknown family: {family}")
    print(f"[generate] {family} seed={seed} skeleton in {time.time() - sk_started:.1f}s", flush=True)

    arr[positions] = skeleton
    del positions
    del skeleton
    gc.collect()

    raw = arr.tobytes()
    structural_sha256 = hashlib.sha256(raw).hexdigest().upper()
    structural = raw.decode("ascii")
    del raw
    del arr
    gc.collect()

    open_count = structural.count("(")
    close_count = structural.count(")")
    bit_count = output_chars - open_count - close_count
    meta = {
        "family": family,
        "seed": seed,
        "chars": output_chars,
        "bit_count": bit_count,
        "open_count": open_count,
        "close_count": close_count,
        "parenthesis_count": open_count + close_count,
        "parenthesis_density": (open_count + close_count) / output_chars if output_chars else 0.0,
        "balance_delta": open_count - close_count,
        "structural_sha256": structural_sha256,
        "elapsed_seconds": time.time() - started,
    }
    return structural, meta


def analyze_control(
    family: str,
    seed: int,
    structural: str,
    generation_meta: Dict[str, Any],
    max_nodes: int,
    timeout_seconds: int,
) -> Dict[str, Any]:
    print(f"[analyze] {family} seed={seed} parser start", flush=True)
    started = time.time()
    tree, build = build_tree_from_structural(
        structural,
        max_chars=None,
        # The canonical parser prints emoji/status glyphs in verbose mode, which
        # can crash Windows cp1252 consoles before analysis starts. The parser
        # semantics are unchanged with verbose disabled.
        verbose=False,
        max_nodes=max_nodes,
        timeout_seconds=timeout_seconds,
    )

    if not build.get("completed"):
        elapsed = time.time() - started
        row = {
            "family": family,
            "seed": seed,
            "analysis_complete": False,
            "interruption_reason": build.get("reason"),
            "mean_children": None,
            "phi_plus_one": PHI_PLUS_ONE,
            "phi_diff": None,
            "rounded6_matches_phi_plus_one": False,
            "branching_nodes": None,
            "max_children": None,
            "tree_nodes": build.get("nodes_created"),
            "max_depth": build.get("max_depth"),
            "chars_processed": build.get("chars_processed"),
            "build_completed": build.get("completed"),
            "build_reason": build.get("reason"),
            "generation": generation_meta,
            "elapsed_seconds": elapsed,
        }
        print(
            f"[analyze] {family} seed={seed} build incomplete "
            f"reason={row['interruption_reason']}",
            flush=True,
        )
        return row

    branching_ratios, depth_dist, content_lengths, children_counts, collection_complete = (
        collect_all_metrics_iterative(tree, verbose=False, timeout_seconds=timeout_seconds)
    )
    elapsed = time.time() - started
    complete = bool(collection_complete)
    mean_children = float(np.mean(children_counts)) if children_counts else None
    phi_diff = None if mean_children is None else float(mean_children - PHI_PLUS_ONE)
    rounded_phi_match = (
        mean_children is not None and round(float(mean_children), 6) == round(PHI_PLUS_ONE, 6)
    )

    row = {
        "family": family,
        "seed": seed,
        "analysis_complete": complete,
        "interruption_reason": None if collection_complete else "collection_timeout",
        "mean_children": None if mean_children is None else float(mean_children),
        "phi_plus_one": PHI_PLUS_ONE,
        "phi_diff": phi_diff,
        "rounded6_matches_phi_plus_one": rounded_phi_match,
        "branching_nodes": len(children_counts),
        "max_children": max(children_counts) if children_counts else None,
        # Avoid TreeNode.subtree_size()/max_depth() recursion on deep random
        # Dyck controls; build metadata is the iterative parser count.
        "tree_nodes": build.get("nodes_created"),
        "max_depth": build.get("max_depth"),
        "chars_processed": build.get("chars_processed"),
        "build_completed": build.get("completed"),
        "build_reason": build.get("reason"),
        "generation": generation_meta,
        "elapsed_seconds": elapsed,
    }
    del tree
    del branching_ratios
    del depth_dist
    del content_lengths
    del children_counts
    gc.collect()
    print(
        f"[analyze] {family} seed={seed} complete={complete} "
        f"mean_children={row['mean_children']} reason={row['interruption_reason']}",
        flush=True,
    )
    return row


def classify_family(rows: List[Dict[str, Any]], stability_tolerance: float) -> Dict[str, Any]:
    successes = [r for r in rows if r.get("analysis_complete") and r.get("mean_children") is not None]
    if not successes:
        outcome = "c-bar-undefined-or-fails"
    elif all(r["rounded6_matches_phi_plus_one"] for r in successes) and len(successes) == len(rows):
        outcome = "c-bar-converges-to-phi"
    elif len(successes) == len(rows):
        vals = [float(r["mean_children"]) for r in successes]
        stable = (max(vals) - min(vals)) <= stability_tolerance
        outcome = "c-bar-converges-to-not-phi" if stable else "c-bar-mixed"
    else:
        outcome = "c-bar-mixed"

    vals = [float(r["mean_children"]) for r in successes]
    mean = float(np.mean(vals)) if vals else None
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if vals else None
    ci95 = None
    if len(vals) > 1:
        ci_half = 1.96 * float(np.std(vals, ddof=1)) / math.sqrt(len(vals))
        ci95 = [mean - ci_half, mean + ci_half]

    return {
        "family": rows[0]["family"] if rows else None,
        "outcome": outcome,
        "successful_runs": len(successes),
        "total_runs": len(rows),
        "mean_children_mean": mean,
        "mean_children_std": std,
        "mean_children_min": min(vals) if vals else None,
        "mean_children_max": max(vals) if vals else None,
        "mean_children_ci95_normal": ci95,
        "all_successes_match_phi_rounded6": bool(successes)
        and all(r["rounded6_matches_phi_plus_one"] for r in successes),
    }


def classify_global(family_summaries: List[Dict[str, Any]]) -> str:
    outcomes = [f["outcome"] for f in family_summaries]
    if any(o == "c-bar-converges-to-phi" for o in outcomes):
        return "c-bar-converges-to-phi"
    if all(o == "c-bar-undefined-or-fails" for o in outcomes):
        return "c-bar-undefined-or-fails"
    if all(o == "c-bar-converges-to-not-phi" for o in outcomes):
        return "c-bar-converges-to-not-phi"
    return "c-bar-mixed"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv

    fields = [
        "family",
        "seed",
        "analysis_complete",
        "interruption_reason",
        "mean_children",
        "phi_diff",
        "rounded6_matches_phi_plus_one",
        "branching_nodes",
        "tree_nodes",
        "max_depth",
        "chars_processed",
        "elapsed_seconds",
        "structural_sha256",
        "parenthesis_density",
        "balance_delta",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            gen = row.get("generation", {})
            flat = dict(row)
            flat["structural_sha256"] = gen.get("structural_sha256")
            flat["parenthesis_density"] = gen.get("parenthesis_density")
            flat["balance_delta"] = gen.get("balance_delta")
            writer.writerow({k: flat.get(k) for k in fields})


def write_report(
    path: Path,
    run_name: str,
    ref: ReferenceStats,
    family_summaries: List[Dict[str, Any]],
    global_outcome: str,
    rows: List[Dict[str, Any]],
    summary_sha: str,
) -> None:
    lines = [
        f"# Q3 empirical Dyck-randomized controls",
        "",
        f"Run: `{run_name}`",
        f"Outcome: `{global_outcome}`",
        "",
        "## Reference",
        "",
        (
            f"The run uses the canonical B@{ref.iteration} Level-1 segment "
            f"{ref.segment_index} (`{ref.segment_size:,}` structural chars). "
            f"This is the lowest available B reference with `mean_children` rounded "
            f"to `2.618034` across the publication-scale 1G segments."
        ),
        "",
        f"- Reference mean_children: `{ref.reference_mean_children:.12f}`",
        f"- Reference parenthesis density: `{ref.parenthesis_density:.8f}`",
        f"- Reference JSON SHA256: `{ref.source_json_sha256}`",
        "",
        "## Family outcomes",
        "",
        "| family | outcome | success | mean_children mean | 95% CI | rounded-6 phi match |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for fs in family_summaries:
        ci = fs.get("mean_children_ci95_normal")
        ci_text = "---" if not ci else f"[{ci[0]:.6f}, {ci[1]:.6f}]"
        mean = fs.get("mean_children_mean")
        mean_text = "---" if mean is None else f"{mean:.6f}"
        lines.append(
            f"| {fs['family']} | `{fs['outcome']}` | "
            f"{fs['successful_runs']}/{fs['total_runs']} | {mean_text} | {ci_text} | "
            f"{fs['all_successes_match_phi_rounded6']} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            (
                "These are pre-registered random-Dyck controls: they add parenthesis "
                "syntax without the HSI accumulation plus stratified inside-to-out "
                "collapse architecture. The readout tests whether parenthesis syntax "
                "alone determines `mean_children = phi + 1`."
            ),
            "",
            (
                "Allowed wording remains bounded: this run can say whether random "
                "Dyck-like syntax reproduces the six-decimal mean-children signature; "
                "it does not prove exclusivity, universality, geometry, or definitive "
                "closure."
            ),
            "",
            f"Summary SHA256: `{summary_sha}`",
            "",
            "## Per-run details",
            "",
            "| family | seed | complete | mean_children | phi_diff | branching_nodes | nodes | reason |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        mc = row.get("mean_children")
        pd = row.get("phi_diff")
        lines.append(
            f"| {row['family']} | {row['seed']} | {row['analysis_complete']} | "
            f"{'---' if mc is None else f'{mc:.9f}'} | "
            f"{'---' if pd is None else f'{pd:.3e}'} | "
            f"{row.get('branching_nodes') or '---'} | {row.get('tree_nodes') or '---'} | "
            f"{row.get('interruption_reason') or row.get('build_reason') or '---'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--reference-json", type=Path, default=REFERENCE_JSON)
    parser.add_argument("--reference-segment", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=None, help="Override reference segment size.")
    parser.add_argument("--parenthesis-count", type=int, default=None, help="Override reference paren count.")
    parser.add_argument("--max-nodes", type=int, default=50_000_000)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--stability-tolerance", type=float, default=1e-3)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write manifest skeleton without executing.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    families = [x.strip().upper() for x in args.families.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    for family in families:
        if family not in DEFAULT_FAMILIES:
            raise ValueError(f"Unsupported family {family}; expected one of {DEFAULT_FAMILIES}")

    ref = load_reference_stats(args.reference_json, args.reference_segment)
    output_chars = int(args.max_chars or ref.segment_size)
    parenthesis_count = int(args.parenthesis_count or round(output_chars * ref.parenthesis_density))
    if parenthesis_count % 2:
        parenthesis_count -= 1
    if parenthesis_count <= 0 or parenthesis_count >= output_chars:
        raise ValueError("Invalid parenthesis_count for requested output_chars")

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_name = args.run_tag or (
        f"q3-empirical-dyck-randomized__chars-{output_chars}__"
        f"families-{'-'.join(families)}__seeds-{'-'.join(map(str, seeds))}__{timestamp}"
    )
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    code_sha = sha256_file(Path(__file__))
    pre_manifest = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": code_sha,
        "families": families,
        "seeds": seeds,
        "reference": ref.__dict__,
        "output_chars": output_chars,
        "parenthesis_count": parenthesis_count,
        "parenthesis_density": parenthesis_count / output_chars,
        "max_nodes": args.max_nodes,
        "timeout_seconds": args.timeout_seconds,
        "stability_tolerance": args.stability_tolerance,
        "dry_run": bool(args.dry_run),
    }
    (run_dir / "pre_manifest.json").write_text(json.dumps(pre_manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"Dry-run manifest written to {run_dir / 'pre_manifest.json'}")
        return 0

    rows: List[Dict[str, Any]] = []
    started = time.time()
    for family in families:
        for seed in seeds:
            structural, gen_meta = generate_control_string(
                family, seed, ref, output_chars, parenthesis_count
            )
            try:
                row = analyze_control(
                    family,
                    seed,
                    structural,
                    gen_meta,
                    max_nodes=args.max_nodes,
                    timeout_seconds=args.timeout_seconds,
                )
                rows.append(row)
            finally:
                del structural
                gc.collect()

            partial_path = run_dir / "partial_results.json"
            partial_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    family_summaries = [
        classify_family([r for r in rows if r["family"] == family], args.stability_tolerance)
        for family in families
    ]
    global_outcome = classify_global(family_summaries)

    summary = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "global_outcome": global_outcome,
        "phi_plus_one": PHI_PLUS_ONE,
        "reference": ref.__dict__,
        "output_chars": output_chars,
        "parenthesis_count": parenthesis_count,
        "families": family_summaries,
        "runs": rows,
        "elapsed_seconds": time.time() - started,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_sha = sha256_file(summary_path)

    manifest = dict(pre_manifest)
    manifest.update(
        {
            "completed_at": datetime.now().isoformat(),
            "elapsed_seconds": time.time() - started,
            "global_outcome": global_outcome,
            "summary_sha256": summary_sha,
            "artifacts": {
                "summary_json": str(summary_path),
                "results_csv": str(run_dir / "results.csv"),
                "report_md": str(run_dir / "report.md"),
                "manifest_json": str(run_dir / "manifest.json"),
            },
        }
    )
    write_csv(run_dir / "results.csv", rows)
    write_report(run_dir / "report.md", run_name, ref, family_summaries, global_outcome, rows, summary_sha)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nQ3 empirical Dyck-randomized controls")
    print("-" * 80)
    print(f"status: completed")
    print(f"outcome: {global_outcome}")
    for fs in family_summaries:
        mean = fs.get("mean_children_mean")
        mean_text = "-" if mean is None else f"{mean:.6f}"
        print(f"{fs['family']:>3}  {fs['outcome']:<30} {fs['successful_runs']}/{fs['total_runs']}  {mean_text}")
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved report to: {run_dir / 'report.md'}")
    print(f"Saved manifest to: {run_dir / 'manifest.json'}")
    print(f"Summary SHA256: {summary_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
