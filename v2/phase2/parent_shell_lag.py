from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from v2.phase1.tower import rolling_codes_uint64_max64


def build_window_pattern_counts(
    bits: np.ndarray,
    offsets: list[int],
    *,
    window_bits: int,
    pattern_scale: int,
    selected_patterns: list[dict],
) -> dict[int, dict[str, int]]:
    pattern_labels = [str(item["pattern"]) for item in selected_patterns]
    pattern_codes = np.array([int(label, 2) for label in pattern_labels], dtype=np.uint64)
    code_to_label = {
        int(code): label
        for code, label in zip(pattern_codes.tolist(), pattern_labels)
    }

    counts_by_offset: dict[int, dict[str, int]] = {}
    for offset in offsets:
        window = bits[offset : offset + window_bits]
        if int(window.size) < pattern_scale:
            counts_by_offset[offset] = {}
            continue
        codes = rolling_codes_uint64_max64(window, pattern_scale)
        if codes.size == 0:
            counts_by_offset[offset] = {}
            continue
        mask = np.isin(codes, pattern_codes)
        if not np.any(mask):
            counts_by_offset[offset] = {}
            continue
        unique_codes, raw_counts = np.unique(codes[mask], return_counts=True)
        counts_by_offset[offset] = {
            code_to_label[int(code)]: int(count)
            for code, count in zip(unique_codes.tolist(), raw_counts.tolist())
        }
    return counts_by_offset


def summarize_shell_lag_probe(
    *,
    anchor_counts: dict[int, dict[str, int]],
    candidate_counts: dict[int, dict[str, int]],
    definition_offsets: list[int],
    scan_offsets: list[int],
    scan_step_bits: int,
) -> tuple[dict, list[dict]]:
    anchor_active_sets = [
        {pattern for pattern, count in anchor_counts.get(offset, {}).items() if int(count) > 0}
        for offset in definition_offsets
    ]
    core_patterns = _set_intersection(anchor_active_sets)
    shell_patterns = _set_union(anchor_active_sets) - core_patterns

    anchor_shell_hits = []
    anchor_shell_unique = []
    candidate_shell_hits = []
    candidate_shell_unique = []

    rows = []
    for offset in scan_offsets:
        anchor_row = anchor_counts.get(offset, {})
        candidate_row = candidate_counts.get(offset, {})

        anchor_core_hit_count = sum(anchor_row.get(pattern, 0) for pattern in core_patterns)
        anchor_core_unique_count = sum(1 for pattern in core_patterns if anchor_row.get(pattern, 0) > 0)
        anchor_shell_hit_count = sum(anchor_row.get(pattern, 0) for pattern in shell_patterns)
        anchor_shell_unique_count = sum(1 for pattern in shell_patterns if anchor_row.get(pattern, 0) > 0)

        candidate_core_hit_count = sum(candidate_row.get(pattern, 0) for pattern in core_patterns)
        candidate_core_unique_count = sum(1 for pattern in core_patterns if candidate_row.get(pattern, 0) > 0)
        candidate_shell_hit_count = sum(candidate_row.get(pattern, 0) for pattern in shell_patterns)
        candidate_shell_unique_count = sum(1 for pattern in shell_patterns if candidate_row.get(pattern, 0) > 0)

        anchor_shell_hits.append(float(anchor_shell_hit_count))
        anchor_shell_unique.append(float(anchor_shell_unique_count))
        candidate_shell_hits.append(float(candidate_shell_hit_count))
        candidate_shell_unique.append(float(candidate_shell_unique_count))

        rows.append(
            {
                "offset_bits": offset,
                "anchor_core_hit_count": anchor_core_hit_count,
                "anchor_core_unique_count": anchor_core_unique_count,
                "anchor_shell_hit_count": anchor_shell_hit_count,
                "anchor_shell_unique_count": anchor_shell_unique_count,
                "candidate_core_hit_count": candidate_core_hit_count,
                "candidate_core_unique_count": candidate_core_unique_count,
                "candidate_shell_hit_count": candidate_shell_hit_count,
                "candidate_shell_unique_count": candidate_shell_unique_count,
                "candidate_full_shell_flag": candidate_shell_unique_count == len(shell_patterns)
                and len(shell_patterns) > 0,
            }
        )

    first_anchor_shell_offset = _first_offset(
        scan_offsets,
        anchor_shell_unique,
        lambda value: value > 0,
    )
    first_candidate_shell_offset = _first_offset(
        scan_offsets,
        candidate_shell_unique,
        lambda value: value > 0,
    )
    first_candidate_full_shell_offset = _first_offset(
        scan_offsets,
        candidate_shell_unique,
        lambda value: value >= len(shell_patterns) and len(shell_patterns) > 0,
    )

    hit_corr = _best_profile_shift(
        anchor_shell_hits,
        candidate_shell_hits,
        step_bits=scan_step_bits,
    )
    unique_corr = _best_profile_shift(
        anchor_shell_unique,
        candidate_shell_unique,
        step_bits=scan_step_bits,
    )

    recommended_lag_bits = (
        first_candidate_full_shell_offset - first_anchor_shell_offset
        if first_candidate_full_shell_offset is not None and first_anchor_shell_offset is not None
        else hit_corr.get("best_shift_bits")
    )

    summary = {
        "definition_offset_count": len(definition_offsets),
        "scan_offset_count": len(scan_offsets),
        "scan_step_bits": scan_step_bits,
        "anchor_core_pattern_count": len(core_patterns),
        "anchor_shell_pattern_count": len(shell_patterns),
        "anchor_core_patterns": sorted(core_patterns),
        "anchor_shell_patterns": sorted(shell_patterns),
        "first_anchor_shell_offset_bits": first_anchor_shell_offset,
        "first_candidate_shell_offset_bits": first_candidate_shell_offset,
        "first_candidate_full_shell_offset_bits": first_candidate_full_shell_offset,
        "first_full_shell_lag_bits": (
            first_candidate_full_shell_offset - first_anchor_shell_offset
            if first_candidate_full_shell_offset is not None and first_anchor_shell_offset is not None
            else None
        ),
        "recommended_lag_bits": recommended_lag_bits,
        "shell_hits_correlation": hit_corr,
        "shell_unique_correlation": unique_corr,
    }
    return summary, rows


def render_parent_shell_lag_report(selection: dict, summary: dict, rows: list[dict]) -> str:
    lines = [
        "# Phase 2 Parent Shell Lag Probe",
        "",
        "## Selection",
        "",
        f"- Anchor variant: {selection['anchor_variant']}",
        f"- Candidate variant: {selection['candidate_variant']}",
        f"- Segment protocol: {selection['num_segments']} x {selection['segment_bits']}",
        f"- Definition offsets: {', '.join(str(value) for value in selection['definition_offsets'])}",
        f"- Scan offsets: {selection['scan_start_bits']} -> {selection['scan_stop_bits']} step {selection['scan_step_bits']}",
        f"- Source freeze mode: {selection['source_freeze_mode']}",
        f"- Frozen total bits: {selection['frozen_total_bits']}",
        f"- Pattern scale: {selection['pattern_scale']}",
        f"- Pattern selection: {selection['pattern_selection']}",
        f"- Top patterns: {selection['top_patterns']}",
        f"- Window bits: {selection['window_bits']}",
        "",
        "## Summary",
        "",
        f"- Anchor core size: {summary['anchor_core_pattern_count']}",
        f"- Anchor shell size: {summary['anchor_shell_pattern_count']}",
        f"- First anchor shell offset: {_fmt_int(summary['first_anchor_shell_offset_bits'])}",
        f"- First candidate shell offset: {_fmt_int(summary['first_candidate_shell_offset_bits'])}",
        f"- First full candidate shell offset: {_fmt_int(summary['first_candidate_full_shell_offset_bits'])}",
        f"- First-full lag estimate: {_fmt_int(summary['first_full_shell_lag_bits'])}",
        f"- Hit-profile correlation best shift: {_fmt_int(summary['shell_hits_correlation']['best_shift_bits'])} "
        f"(score {_fmt_float(summary['shell_hits_correlation']['best_score'])})",
        f"- Unique-profile correlation best shift: {_fmt_int(summary['shell_unique_correlation']['best_shift_bits'])} "
        f"(score {_fmt_float(summary['shell_unique_correlation']['best_score'])})",
        f"- Recommended lag bits: {_fmt_int(summary['recommended_lag_bits'])}",
        "",
        "## Scan Readout",
        "",
        "| Offset | A core hits | A shell hits | A shell uniq | B core hits | B shell hits | B shell uniq | B full shell |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['offset_bits']} | {row['anchor_core_hit_count']} | {row['anchor_shell_hit_count']} | "
            f"{row['anchor_shell_unique_count']} | {row['candidate_core_hit_count']} | {row['candidate_shell_hit_count']} | "
            f"{row['candidate_shell_unique_count']} | {str(bool(row['candidate_full_shell_flag']))} |"
        )

    return "\n".join(lines)


def write_parent_shell_lag_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "offset_bits",
        "anchor_core_hit_count",
        "anchor_core_unique_count",
        "anchor_shell_hit_count",
        "anchor_shell_unique_count",
        "candidate_core_hit_count",
        "candidate_core_unique_count",
        "candidate_shell_hit_count",
        "candidate_shell_unique_count",
        "candidate_full_shell_flag",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def render_parent_shell_lag_console_summary(selection: dict, summary: dict) -> str:
    lines = [
        "Phase 2 parent shell lag probe",
        "-" * 118,
        f"anchor={selection['anchor_variant']} candidate={selection['candidate_variant']} shell={summary['anchor_shell_pattern_count']} "
        f"first_full_lag={_fmt_int(summary['first_full_shell_lag_bits'])} "
        f"hit_corr_lag={_fmt_int(summary['shell_hits_correlation']['best_shift_bits'])} "
        f"uniq_corr_lag={_fmt_int(summary['shell_unique_correlation']['best_shift_bits'])} "
        f"recommended={_fmt_int(summary['recommended_lag_bits'])}",
    ]
    return "\n".join(lines)


def _best_profile_shift(anchor_values: list[float], candidate_values: list[float], *, step_bits: int) -> dict:
    if not anchor_values or not candidate_values:
        return {"best_shift_steps": None, "best_shift_bits": None, "best_score": None}

    anchor = np.array(anchor_values, dtype=np.float64)
    candidate = np.array(candidate_values, dtype=np.float64)
    n = min(anchor.size, candidate.size)
    if n == 0:
        return {"best_shift_steps": None, "best_shift_bits": None, "best_score": None}

    max_shift = n - 1
    best_shift = None
    best_score = -math.inf
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            left = anchor[: n - shift]
            right = candidate[shift:n]
        else:
            left = anchor[-shift:n]
            right = candidate[: n + shift]
        if left.size < 2 or right.size < 2:
            continue
        if np.allclose(left, left[0]) or np.allclose(right, right[0]):
            continue
        score = float(np.corrcoef(left, right)[0, 1])
        if math.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_shift = shift

    if best_shift is None:
        return {"best_shift_steps": None, "best_shift_bits": None, "best_score": None}
    return {
        "best_shift_steps": int(best_shift),
        "best_shift_bits": int(best_shift * step_bits),
        "best_score": float(best_score),
    }


def _first_offset(offsets: list[int], values: list[float], predicate) -> int | None:
    for offset, value in zip(offsets, values):
        if predicate(value):
            return int(offset)
    return None


def _set_intersection(sets: list[set[str]]) -> set[str]:
    if not sets:
        return set()
    result = set(sets[0])
    for item in sets[1:]:
        result &= item
    return result


def _set_union(sets: list[set[str]]) -> set[str]:
    result: set[str] = set()
    for item in sets:
        result |= item
    return result


def _fmt_int(value: int | None) -> str:
    return "-" if value is None else str(int(value))


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"
