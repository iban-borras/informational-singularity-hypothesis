#!/usr/bin/env python3
"""
Apendix: Deep Divergence Analysis — B vs E vs I
=================================================

Investigates WHY variant E maintains identical trajectory to B for ~91M bytes
while variant I diverges at byte 2 (immediately).

Hypothesis: The order of annihilation (01→0 vs 10→0) combined with the
initial asymmetry of the sequence determines convergence duration.

Analyses performed:
  1. Initial asymmetry: count 01 vs 10 in early iterations (1-10)
  2. Divergence evolution: at which byte do B/E and B/I diverge per iteration?
  3. Pattern density around divergence: 01/10 frequency near divergence point
  4. Structural snapshot: decode chars around divergence for visual inspection
  5. Multi-scale LZ comparison: complexity at windows before/after divergence

Runs WITHOUT parameters — auto-detects data files and iterations.

Author: Sophia (Augment Agent) & Iban Borràs
Date: 2026-02-06
"""

import gzip
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.progress import ProgressIndicator, format_time

try:
    import numpy as np
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    import numpy as np
    NUMBA_AVAILABLE = False

# ============================================================================
# Constants
# ============================================================================
BASE_PATH = Path(__file__).parent / "results" / "level0" / "phi_snapshots"
OUTPUT_DIR = Path(__file__).parent / "results" / "level1" / "divergence_analysis"
PHI = 1.618033988749895
INV_PHI = 0.6180339887498948
VARIANTS = ["B", "E", "I"]

CHUNK_BYTES = 10_000_000  # 10MB per chunk for streaming

# Decoding map (v33 structural format: 2 bits per char)
DECODE = {0b00: '0', 0b01: '1', 0b10: '(', 0b11: ')'}


# ============================================================================
# Numba-compiled kernels
# ============================================================================
if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _count_01_10_numba(arr):
        """Count consecutive 01 and 10 patterns in a bit array."""
        n01 = 0
        n10 = 0
        for i in range(len(arr) - 1):
            if arr[i] == 0 and arr[i + 1] == 1:
                n01 += 1
            elif arr[i] == 1 and arr[i + 1] == 0:
                n10 += 1
        return n01, n10

    @njit(cache=True)
    def _lempel_ziv_numba(arr):
        """LZ76 complexity — Numba JIT."""
        n = len(arr)
        if n == 0:
            return 0
        complexity = 1
        prefix_len = 1
        while prefix_len < n:
            max_match = 0
            for start in range(prefix_len):
                match_len = 0
                while (prefix_len + match_len < n and
                       arr[start + match_len] == arr[prefix_len + match_len]):
                    match_len += 1
                    if start + match_len >= prefix_len:
                        break
                if match_len > max_match:
                    max_match = match_len
            prefix_len += max(1, max_match + 1)
            complexity += 1
        return complexity
else:
    def _count_01_10_numba(arr):
        n01 = int(np.sum((arr[:-1] == 0) & (arr[1:] == 1)))
        n10 = int(np.sum((arr[:-1] == 1) & (arr[1:] == 0)))
        return n01, n10

    _lempel_ziv_numba = None


# ============================================================================
# Utility functions
# ============================================================================
def get_struct_path(variant: str, iteration: int) -> Path:
    return BASE_PATH / f"var_{variant}" / f"phi_iter{iteration}.struct.gz"


def get_highest_iteration(variant: str) -> Optional[int]:
    var_dir = BASE_PATH / f"var_{variant}"
    if not var_dir.exists():
        return None
    max_iter = None
    for f in var_dir.glob("phi_iter*.struct.gz"):
        try:
            n = int(f.stem.replace("phi_iter", "").replace(".struct", ""))
            if max_iter is None or n > max_iter:
                max_iter = n
        except ValueError:
            continue
    return max_iter


def get_common_iterations() -> List[int]:
    """Return sorted list of iterations available for ALL three variants."""
    sets = []
    for v in VARIANTS:
        var_dir = BASE_PATH / f"var_{v}"
        if not var_dir.exists():
            return []
        iters = set()
        for f in var_dir.glob("phi_iter*.struct.gz"):
            try:
                n = int(f.stem.replace("phi_iter", "").replace(".struct", ""))
                iters.add(n)
            except ValueError:
                continue
        sets.append(iters)
    common = sets[0].intersection(*sets[1:])
    return sorted(common)


def decode_bytes_to_bits(raw: bytes) -> np.ndarray:
    """Decode raw bytes from struct.gz into observable bits (0/1) as numpy array."""
    byte_array = np.frombuffer(raw, dtype=np.uint8)
    p0 = (byte_array >> 6) & 0b11
    p1 = (byte_array >> 4) & 0b11
    p2 = (byte_array >> 2) & 0b11
    p3 = byte_array & 0b11
    all_pairs = np.column_stack([p0, p1, p2, p3]).ravel()
    mask = all_pairs <= 1  # Keep only 0 and 1
    return all_pairs[mask].astype(np.int8)


def decode_bytes_to_chars(raw: bytes) -> str:
    """Decode raw bytes from struct.gz into full char sequence (0,1,(,))."""
    chars = []
    for b in raw:
        chars.append(DECODE[(b >> 6) & 0b11])
        chars.append(DECODE[(b >> 4) & 0b11])
        chars.append(DECODE[(b >> 2) & 0b11])
        chars.append(DECODE[b & 0b11])
    return ''.join(chars)


# ============================================================================
# Analysis 1: Initial Asymmetry (01 vs 10 counts per iteration)
# ============================================================================
def analyze_initial_asymmetry(iterations: List[int], max_bits: int = 10_000_000) -> Dict[str, Any]:
    """
    Count 01 vs 10 occurrences in the first max_bits of each variant/iteration.
    This tests whether the initial sequence has more 01 or 10 patterns.
    """
    print(f"\n{'='*70}")
    print("  ANALYSIS 1: Initial Asymmetry (01 vs 10 counts)")
    print(f"{'='*70}")

    results = {}
    total_tasks = len(VARIANTS) * len(iterations)

    with ProgressIndicator("Asymmetry Analysis", total=total_tasks) as prog:
        task_idx = 0
        for variant in VARIANTS:
            results[variant] = {}
            for it in iterations:
                path = get_struct_path(variant, it)
                if not path.exists():
                    task_idx += 1
                    prog.update(task_idx)
                    continue

                # Stream and count
                n01, n10, total_bits = 0, 0, 0
                with gzip.open(path, 'rb') as f:
                    while total_bits < max_bits:
                        raw = f.read(CHUNK_BYTES)
                        if not raw:
                            break
                        bits = decode_bytes_to_bits(raw)
                        if len(bits) == 0:
                            continue
                        c01, c10 = _count_01_10_numba(bits)
                        n01 += c01
                        n10 += c10
                        total_bits += len(bits)

                ratio = n01 / n10 if n10 > 0 else float('inf')
                results[variant][it] = {
                    "n01": int(n01), "n10": int(n10),
                    "ratio_01_10": round(ratio, 6),
                    "bits_analyzed": int(min(total_bits, max_bits)),
                    "asymmetry": "01>10" if n01 > n10 else ("10>01" if n10 > n01 else "symmetric")
                }
                task_idx += 1
                prog.update(task_idx, f"{variant}@{it}: 01/10={ratio:.4f}")

    # Print summary
    print("\n📊 Summary of 01/10 ratios:")
    for v in VARIANTS:
        if v in results:
            for it in sorted(results[v].keys()):
                r = results[v][it]
                print(f"   {v}@{it}: 01={r['n01']:>12,}  10={r['n10']:>12,}  "
                      f"ratio={r['ratio_01_10']:.4f}  ({r['asymmetry']})")
    return results


# ============================================================================
# Analysis 2: Divergence Evolution (per iteration)
# ============================================================================
def find_divergence_byte(path1: Path, path2: Path) -> Tuple[Optional[int], int, int]:
    """Find the first byte where two struct.gz files diverge (decompressed stream).

    Returns:
        (divergence_byte, decompressed_size_1, decompressed_size_2)
        divergence_byte is None if files are identical.
        Sizes are exact decompressed sizes (read to EOF for both files).
    """
    with gzip.open(path1, 'rb') as f1, gzip.open(path2, 'rb') as f2:
        offset = 0
        div_byte = None
        size1, size2 = 0, 0
        while True:
            c1 = f1.read(CHUNK_BYTES)
            c2 = f2.read(CHUNK_BYTES)
            size1 += len(c1) if c1 else 0
            size2 += len(c2) if c2 else 0
            if not c1 and not c2:
                break
            if div_byte is None:
                if not c1 or not c2:
                    div_byte = offset
                else:
                    min_len = min(len(c1), len(c2))
                    for i in range(min_len):
                        if c1[i] != c2[i]:
                            div_byte = offset + i
                            break
                    offset += min_len
            # Keep reading to measure full decompressed sizes
            # (only the shorter file needs draining; the longer keeps going)
            if not c1 and not c2:
                break
            if div_byte is not None:
                # Drain remaining bytes to get accurate sizes
                while True:
                    r1 = f1.read(CHUNK_BYTES)
                    r2 = f2.read(CHUNK_BYTES)
                    size1 += len(r1) if r1 else 0
                    size2 += len(r2) if r2 else 0
                    if not r1 and not r2:
                        break
                break
    return div_byte, size1, size2



def analyze_divergence_evolution(iterations: List[int]) -> Dict[str, Any]:
    """Track at which byte B/E and B/I diverge for each iteration."""
    print(f"\n{'='*70}")
    print("  ANALYSIS 2: Divergence Evolution per Iteration")
    print(f"{'='*70}")

    results = {"B_vs_E": {}, "B_vs_I": {}}
    total_tasks = len(iterations) * 2

    with ProgressIndicator("Divergence Evolution", total=total_tasks) as prog:
        task_idx = 0
        for it in iterations:
            path_b = get_struct_path("B", it)
            path_e = get_struct_path("E", it)
            path_i = get_struct_path("I", it)

            # B vs E
            if path_b.exists() and path_e.exists():
                div_byte, dec_b, dec_e = find_divergence_byte(path_b, path_e)
                results["B_vs_E"][it] = {
                    "divergence_byte": div_byte,
                    "decompressed_size_B": dec_b,
                    "decompressed_size_E": dec_e,
                    "identical_fraction": div_byte / max(dec_b, dec_e) if div_byte else 1.0
                }
            task_idx += 1
            prog.update(task_idx, f"B vs E @{it}")

            # B vs I
            if path_b.exists() and path_i.exists():
                div_byte, dec_b, dec_i = find_divergence_byte(path_b, path_i)
                results["B_vs_I"][it] = {
                    "divergence_byte": div_byte,
                    "decompressed_size_B": dec_b,
                    "decompressed_size_I": dec_i,
                    "identical_fraction": div_byte / max(dec_b, dec_i) if div_byte else 1.0
                }
            task_idx += 1
            prog.update(task_idx, f"B vs I @{it}")

    return results


# ============================================================================
# Analysis 3: Pattern Density Around Divergence
# ============================================================================
def analyze_pattern_density_at_divergence(
    variant: str, iteration: int, divergence_byte: int,
    window_bytes: int = 1_000_000
) -> Dict[str, Any]:
    """Count 01/10 patterns in windows before and after divergence point."""
    path = get_struct_path(variant, iteration)
    if not path.exists():
        return {}

    before_01, before_10, before_bits = 0, 0, 0
    after_01, after_10, after_bits = 0, 0, 0
    start_byte = max(0, divergence_byte - window_bytes)

    with gzip.open(path, 'rb') as f:
        # Skip to start
        if start_byte > 0:
            f.read(start_byte)

        # Read window before divergence
        bytes_to_read = divergence_byte - start_byte
        raw = f.read(bytes_to_read)
        if raw:
            bits = decode_bytes_to_bits(raw)
            if len(bits) > 1:
                before_01, before_10 = _count_01_10_numba(bits)
                before_bits = len(bits)

        # Read window after divergence
        raw = f.read(window_bytes)
        if raw:
            bits = decode_bytes_to_bits(raw)
            if len(bits) > 1:
                after_01, after_10 = _count_01_10_numba(bits)
                after_bits = len(bits)

    return {
        "before": {
            "n01": int(before_01), "n10": int(before_10), "bits": before_bits,
            "ratio": round(before_01 / before_10, 6) if before_10 > 0 else None
        },
        "after": {
            "n01": int(after_01), "n10": int(after_10), "bits": after_bits,
            "ratio": round(after_01 / after_10, 6) if after_10 > 0 else None
        }
    }


def analyze_all_pattern_densities(iterations: List[int],
                                  divergence_data: Dict) -> Dict[str, Any]:
    """Run pattern density analysis for B, E, I around known divergence points."""
    print(f"\n{'='*70}")
    print("  ANALYSIS 3: Pattern Density Around Divergence Points")
    print(f"{'='*70}")

    results = {}
    # Use highest iteration with known divergence
    highest = max(iterations)
    be_div = divergence_data.get("B_vs_E", {}).get(highest, {}).get("divergence_byte")
    bi_div = divergence_data.get("B_vs_I", {}).get(highest, {}).get("divergence_byte")

    with ProgressIndicator("Pattern Density", total=6) as prog:
        idx = 0
        for variant in VARIANTS:
            results[variant] = {}
            # Use B vs E divergence for all (91M bytes)
            if be_div and be_div > 10:
                results[variant]["at_BE_divergence"] = analyze_pattern_density_at_divergence(
                    variant, highest, be_div)
            idx += 1
            prog.update(idx, f"{variant} @ B/E div")

            # Use B vs I divergence (byte 2 — too small for window, skip)
            if bi_div and bi_div > 1000:
                results[variant]["at_BI_divergence"] = analyze_pattern_density_at_divergence(
                    variant, highest, bi_div)
            idx += 1
            prog.update(idx, f"{variant} @ B/I div")

    return results


# ============================================================================
# Analysis 4: Structural Snapshot at Divergence
# ============================================================================
def extract_divergence_snapshot(variant: str, iteration: int,
                                divergence_byte: int,
                                context_bytes: int = 50) -> Dict[str, Any]:
    """Extract decoded chars around divergence point for visual inspection."""
    path = get_struct_path(variant, iteration)
    if not path.exists():
        return {}

    start = max(0, divergence_byte - context_bytes)
    with gzip.open(path, 'rb') as f:
        if start > 0:
            f.read(start)
        raw = f.read(context_bytes * 2 + 10)

    chars = decode_bytes_to_chars(raw)
    # Calculate char offset for divergence
    div_char_offset = (divergence_byte - start) * 4  # approx: 4 chars per byte

    return {
        "variant": variant, "iteration": iteration,
        "divergence_byte": divergence_byte,
        "context_start_byte": start,
        "chars_before": chars[:div_char_offset][-80:] if div_char_offset > 0 else "",
        "chars_after": chars[div_char_offset:div_char_offset + 80],
        "total_chars_extracted": len(chars)
    }


def analyze_structural_snapshots(iterations: List[int],
                                 divergence_data: Dict) -> Dict[str, Any]:
    """Extract structural snapshots for all variants at divergence points."""
    print(f"\n{'='*70}")
    print("  ANALYSIS 4: Structural Snapshots at Divergence")
    print(f"{'='*70}")

    results = {}
    highest = max(iterations)
    be_div = divergence_data.get("B_vs_E", {}).get(highest, {}).get("divergence_byte")
    bi_div = divergence_data.get("B_vs_I", {}).get(highest, {}).get("divergence_byte")

    with ProgressIndicator("Structural Snapshots", total=len(VARIANTS) * 2) as prog:
        idx = 0
        for variant in VARIANTS:
            results[variant] = {}
            if be_div:
                snap = extract_divergence_snapshot(variant, highest, be_div)
                results[variant]["at_BE_divergence"] = snap
            idx += 1
            prog.update(idx, f"{variant} @ B/E")

            if bi_div and bi_div > 0:
                snap = extract_divergence_snapshot(variant, highest, bi_div)
                results[variant]["at_BI_divergence"] = snap
            idx += 1
            prog.update(idx, f"{variant} @ B/I")

    return results


# ============================================================================
# Analysis 5: Multi-scale LZ Comparison
# ============================================================================
def calculate_lz_ratio(bits: np.ndarray) -> float:
    """Calculate LZ complexity ratio for a numpy bit array."""
    if _lempel_ziv_numba is None or len(bits) == 0:
        return 0.0
    n = len(bits)
    c = _lempel_ziv_numba(bits)
    h_max = n / np.log2(n) if n > 1 else 1
    return c / h_max if h_max > 0 else 0.0


def analyze_lz_around_divergence(iterations: List[int],
                                 divergence_data: Dict,
                                 window_bits: int = 500_000) -> Dict[str, Any]:
    """Calculate LZ complexity in windows before and after B/E divergence."""
    print(f"\n{'='*70}")
    print("  ANALYSIS 5: Multi-scale LZ Around Divergence")
    print(f"{'='*70}")

    if not NUMBA_AVAILABLE:
        print("   ⚠️ Numba not available — skipping LZ analysis")
        return {"skipped": "numba_not_available"}

    results = {}
    highest = max(iterations)
    be_div = divergence_data.get("B_vs_E", {}).get(highest, {}).get("divergence_byte")

    if not be_div or be_div < 1000:
        print("   ⚠️ No valid B/E divergence point found — skipping LZ")
        return {"skipped": "no_divergence_point"}

    # Convert byte offset to approximate bits needed
    # Each byte holds 4 pairs of 2 bits; ~50% are observable bits
    approx_bits_per_byte = 2  # conservative: 4 chars/byte, ~50% observable
    window_bytes = window_bits // approx_bits_per_byte

    with ProgressIndicator("LZ Analysis", total=len(VARIANTS) * 2) as prog:
        idx = 0
        for variant in VARIANTS:
            results[variant] = {}
            path = get_struct_path(variant, highest)
            if not path.exists():
                idx += 2
                prog.update(idx)
                continue

            # Read window BEFORE divergence
            start_before = max(0, be_div - window_bytes)
            with gzip.open(path, 'rb') as f:
                if start_before > 0:
                    f.read(start_before)
                raw = f.read(be_div - start_before)
            bits = decode_bytes_to_bits(raw)
            if len(bits) > window_bits:
                bits = bits[-window_bits:]  # Take last window_bits
            lz_before = calculate_lz_ratio(bits)
            results[variant]["before"] = {
                "lz_ratio": round(lz_before, 6),
                "bits_analyzed": len(bits),
                "phi_distance": round(abs(lz_before - INV_PHI), 6)
            }
            idx += 1
            prog.update(idx, f"{variant} before: LZ={lz_before:.4f}")

            # Read window AFTER divergence
            with gzip.open(path, 'rb') as f:
                f.read(be_div)  # skip to divergence
                raw = f.read(window_bytes)
            bits = decode_bytes_to_bits(raw)
            if len(bits) > window_bits:
                bits = bits[:window_bits]
            lz_after = calculate_lz_ratio(bits)
            results[variant]["after"] = {
                "lz_ratio": round(lz_after, 6),
                "bits_analyzed": len(bits),
                "phi_distance": round(abs(lz_after - INV_PHI), 6)
            }
            idx += 1
            prog.update(idx, f"{variant} after: LZ={lz_after:.4f}")

    return results


# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("  🔬 DEEP DIVERGENCE ANALYSIS: B vs E vs I")
    print(f"{'='*70}")
    print(f"   Base path:  {BASE_PATH}")
    print(f"   Output dir: {OUTPUT_DIR}")

    # 1. Auto-detect common iterations
    iterations = get_common_iterations()
    if not iterations:
        print("   ❌ No common iterations found for B, E, I — aborting.")
        sys.exit(1)
    print(f"   Common iterations: {iterations}")

    # 2. Run all analyses
    all_results: Dict[str, Any] = {
        "metadata": {
            "script": "apendix_deep_divergence_analysis_B_E.py",
            "generated_at": datetime.now().isoformat(),
            "variants": VARIANTS,
            "iterations_analyzed": iterations,
            "numba_available": NUMBA_AVAILABLE
        }
    }

    # Analysis 1: Initial asymmetry
    all_results["asymmetry"] = analyze_initial_asymmetry(iterations)

    # Analysis 2: Divergence evolution
    all_results["divergence_evolution"] = analyze_divergence_evolution(iterations)

    # Analysis 3: Pattern density around divergence
    all_results["pattern_density"] = analyze_all_pattern_densities(
        iterations, all_results["divergence_evolution"])

    # Analysis 4: Structural snapshots
    all_results["structural_snapshots"] = analyze_structural_snapshots(
        iterations, all_results["divergence_evolution"])

    # Analysis 5: LZ around divergence
    all_results["lz_around_divergence"] = analyze_lz_around_divergence(
        iterations, all_results["divergence_evolution"])

    # 3. Save results
    out_path = OUTPUT_DIR / "deep_divergence_B_E_I.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n💾 Results saved to: {out_path}")

    # 4. Print summary
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print("  📊 SUMMARY")
    print(f"{'='*70}")

    # Asymmetry summary
    asym = all_results.get("asymmetry", {})
    for v in VARIANTS:
        if v in asym:
            highest_it = max(asym[v].keys()) if asym[v] else None
            if highest_it is not None:
                r = asym[v][highest_it]
                print(f"   {v}@{highest_it} asymmetry: 01/10 ratio = {r['ratio_01_10']:.4f} "
                      f"({r['asymmetry']})")

    # Divergence summary
    div = all_results.get("divergence_evolution", {})
    for pair in ["B_vs_E", "B_vs_I"]:
        if pair in div:
            for it in sorted(div[pair].keys()):
                d = div[pair][it]
                db = d.get("divergence_byte")
                if db is not None:
                    print(f"   {pair} @{it}: diverge at byte {db:,} "
                          f"({d['identical_fraction']:.2%} identical)")

    # LZ summary
    lz = all_results.get("lz_around_divergence", {})
    if "skipped" not in lz:
        for v in VARIANTS:
            if v in lz:
                bef = lz[v].get("before", {}).get("lz_ratio", "?")
                aft = lz[v].get("after", {}).get("lz_ratio", "?")
                print(f"   {v} LZ: before={bef}, after={aft}")

    print(f"\n⏱️  Total time: {format_time(elapsed)}")
    print(f"📂 Results: {out_path}")


if __name__ == "__main__":
    main()