#!/usr/bin/env python3
"""
Apendix: Commutativity Analysis for HSI Variants
=================================================

This script analyzes the divergence between HSI variants to verify the
commutativity property of the binary annihilation process.

Key Question: Does the order of applying annihilation rules (01→0, 10→0)
affect the final result?

Variants compared:
- B (Gold Standard): Simultaneous annihilation
- E (Two-Phase): Sequential (01→0 first, then 10→0)
- I (Inverse Two-Phase): Sequential (10→0 first, then 01→0)

If B ≡ E ≡ I, the process is fully commutative.
If they diverge, the divergence point reveals important properties.

Usage:
    # Full automatic analysis (B vs E and B vs I)
    python apendix_commutativity_analysis.py --full-analysis

    # Full analysis with specific variants
    python apendix_commutativity_analysis.py --full-analysis --variants E I F

    # Full analysis with different reference variant
    python apendix_commutativity_analysis.py --full-analysis --reference B --variants E I

    # Manual mode (single comparison)
    python apendix_commutativity_analysis.py --var1 B --iter1 23 --var2 E --iter2 23

    # Manual post-divergence analysis
    python apendix_commutativity_analysis.py --var1 B --iter1 23 --var2 E --iter2 23 --post-divergence

Author: Sophia (Augment Agent) & Iban Borràs
Date: 2026-01-25
"""

import argparse
import gzip
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.streaming_phi_loader import StreamingPhiLoader

# Optional imports for LZ analysis
try:
    import numpy as np
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    np = None

import math

# Constants
BASE_PATH = Path(__file__).parent / "results" / "level0" / "phi_snapshots"
OUTPUT_DIR = Path(__file__).parent / "results" / "level1" / "commutativity"
PHI = 1.618033988749895
INV_PHI = 0.6180339887498948


def get_struct_path(variant: str, iteration: int) -> Path:
    """Get path to structural file for a variant/iteration."""
    return BASE_PATH / f"var_{variant}" / f"phi_iter{iteration}.struct.gz"


def get_highest_iteration(variant: str) -> Optional[int]:
    """
    Find the highest iteration available for a variant.

    Args:
        variant: Variant letter (e.g., 'B', 'E', 'I')

    Returns:
        Highest iteration number, or None if no files found
    """
    var_dir = BASE_PATH / f"var_{variant}"
    if not var_dir.exists():
        return None

    max_iter = None
    for f in var_dir.glob("phi_iter*.struct.gz"):
        try:
            # Extract iteration number from filename
            iter_str = f.stem.replace("phi_iter", "").replace(".struct", "")
            iter_num = int(iter_str)
            if max_iter is None or iter_num > max_iter:
                max_iter = iter_num
        except ValueError:
            continue

    return max_iter


def get_available_variants() -> Dict[str, int]:
    """
    Get all available variants and their highest iterations.

    Returns:
        Dictionary mapping variant letter to highest iteration
    """
    variants = {}
    if not BASE_PATH.exists():
        return variants

    for var_dir in BASE_PATH.glob("var_*"):
        variant = var_dir.name.replace("var_", "")
        highest = get_highest_iteration(variant)
        if highest is not None:
            variants[variant] = highest

    return variants


# ============================================================================
# LZ Complexity Functions (for post-divergence analysis)
# ============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _lempel_ziv_complexity_numba(arr):
        """Numba-accelerated LZ76 complexity calculation."""
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
            prefix_len += max_match + 1
            complexity += 1
        return complexity


def lempel_ziv_complexity(s: str) -> int:
    """Calculate Lempel-Ziv complexity (LZ76)."""
    if len(s) == 0:
        return 0
    if NUMBA_AVAILABLE:
        arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8) - ord('0')
        return _lempel_ziv_complexity_numba(arr)
    else:
        # Pure Python fallback (slower)
        n = len(s)
        complexity = 1
        prefix_len = 1
        while prefix_len < n:
            max_match = 0
            for start in range(prefix_len):
                match_len = 0
                while (prefix_len + match_len < n and
                       s[start + match_len] == s[prefix_len + match_len]):
                    match_len += 1
                    if start + match_len >= prefix_len:
                        break
                if match_len > max_match:
                    max_match = match_len
            prefix_len += max_match + 1
            complexity += 1
        return complexity


def calculate_lz_ratio(bits: str, scales: list = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Calculate LZ complexity ratio at multiple scales.

    Args:
        bits: Binary string (only '0' and '1')
        scales: List of scales to analyze
        verbose: Print progress

    Returns:
        Dictionary with LZ analysis results
    """
    if scales is None:
        scales = [1000, 2000, 5000, 10000, 20000, 50000]

    scales = [s for s in scales if s <= len(bits)]
    if not scales:
        return {"error": "Not enough data for analysis", "bits_available": len(bits)}

    results = {}
    for scale in scales:
        segment = bits[:scale]
        lz = lempel_ziv_complexity(segment)
        normalized = lz / (scale / math.log2(scale)) if scale > 1 else 0
        results[scale] = {'raw_lz': lz, 'normalized_lz': float(normalized)}

    # Calculate ratios between consecutive scales
    sorted_scales = sorted(results.keys())
    lz_ratios = []
    for i in range(len(sorted_scales) - 1):
        s1, s2 = sorted_scales[i], sorted_scales[i + 1]
        if results[s1]['normalized_lz'] > 0:
            lz_ratios.append(results[s2]['normalized_lz'] / results[s1]['normalized_lz'])

    mean_ratio = float(np.mean(lz_ratios)) if lz_ratios and NUMBA_AVAILABLE else (
        sum(lz_ratios) / len(lz_ratios) if lz_ratios else 0)

    # Check proximity to mathematical constants
    phi_distance = abs(mean_ratio - INV_PHI)
    sqrt3_distance = abs(mean_ratio - 0.5773502691896258)  # 1/√3

    return {
        'by_scale': results,
        'lz_ratios': lz_ratios,
        'mean_ratio': mean_ratio,
        'phi_distance': phi_distance,
        'sqrt3_distance': sqrt3_distance,
        'best_match': '1/φ' if phi_distance < sqrt3_distance else '1/√3',
        'bits_analyzed': len(bits)
    }


def compare_streams(
    path1: Path,
    path2: Path,
    chunk_size: int = 10_000_000,
    max_bytes: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare two structural files byte-by-byte using streaming.
    
    Args:
        path1: Path to first .struct.gz file
        path2: Path to second .struct.gz file
        chunk_size: Bytes to read per chunk
        max_bytes: Maximum bytes to compare (None = all)
        verbose: Print progress
    
    Returns:
        Dictionary with comparison results
    """
    results = {
        "file1": str(path1),
        "file2": str(path2),
        "file1_size": path1.stat().st_size,
        "file2_size": path2.stat().st_size,
        "bytes_compared": 0,
        "divergence_found": False,
        "divergence_byte": None,
        "divergence_char": None,
        "char1_at_divergence": None,
        "char2_at_divergence": None,
        "identical_bytes": 0,
        "identical_chars": 0,
    }
    
    if verbose:
        print(f"\n🔍 Comparing files:")
        print(f"   File 1: {path1.name} ({results['file1_size']:,} bytes compressed)")
        print(f"   File 2: {path2.name} ({results['file2_size']:,} bytes compressed)")
    
    # Open both files for streaming comparison
    with gzip.open(path1, 'rb') as f1, gzip.open(path2, 'rb') as f2:
        byte_offset = 0
        char_offset = 0
        
        while True:
            # Check max_bytes limit
            if max_bytes and byte_offset >= max_bytes:
                if verbose:
                    print(f"\n   ⏹️  Reached max_bytes limit: {max_bytes:,}")
                break
            
            # Read chunks
            remaining = (max_bytes - byte_offset) if max_bytes else chunk_size
            read_size = min(chunk_size, remaining)
            
            chunk1 = f1.read(read_size)
            chunk2 = f2.read(read_size)
            
            # Check for EOF
            if not chunk1 and not chunk2:
                break
            
            if not chunk1 or not chunk2:
                # One file ended before the other
                results["divergence_found"] = True
                results["divergence_byte"] = byte_offset
                results["divergence_char"] = char_offset
                results["note"] = "One file ended before the other"
                if verbose:
                    print(f"\n   ⚠️  File length mismatch at byte {byte_offset:,}")
                break
            
            # Compare byte by byte
            min_len = min(len(chunk1), len(chunk2))
            for i in range(min_len):
                if chunk1[i] != chunk2[i]:
                    # Found divergence!
                    results["divergence_found"] = True
                    results["divergence_byte"] = byte_offset + i
                    results["divergence_char"] = char_offset + (i * 4)  # 4 chars per byte
                    
                    # Decode the divergent bytes to see actual characters
                    # Each byte encodes 4 characters (2 bits each)
                    DECODE = {0b00: '0', 0b01: '1', 0b10: '(', 0b11: ')'}
                    chars1 = [DECODE[(chunk1[i] >> s) & 0b11] for s in [6, 4, 2, 0]]
                    chars2 = [DECODE[(chunk2[i] >> s) & 0b11] for s in [6, 4, 2, 0]]
                    
                    results["char1_at_divergence"] = ''.join(chars1)
                    results["char2_at_divergence"] = ''.join(chars2)
                    
                    if verbose:
                        print(f"\n   🔴 DIVERGENCE FOUND!")
                        print(f"      Byte: {results['divergence_byte']:,}")
                        print(f"      Char: ~{results['divergence_char']:,}")
                        print(f"      File1 chars: {results['char1_at_divergence']}")
                        print(f"      File2 chars: {results['char2_at_divergence']}")
                    break
            
            if results["divergence_found"]:
                break
            
            byte_offset += min_len
            char_offset += min_len * 4
            results["bytes_compared"] = byte_offset
            results["identical_bytes"] = byte_offset
            results["identical_chars"] = char_offset
            
            if verbose and byte_offset % (100_000_000) < chunk_size:
                print(f"   ✅ {byte_offset:,} bytes identical...", end='\r')
    
    if not results["divergence_found"]:
        if verbose:
            print(f"\n   ✅ Files are IDENTICAL ({results['identical_bytes']:,} bytes)")

    return results


def analyze_commutativity(
    var1: str, iter1: int,
    var2: str, iter2: int,
    max_bytes: Optional[int] = None,
    save_results: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze commutativity between two variants.

    Args:
        var1, iter1: First variant and iteration
        var2, iter2: Second variant and iteration
        max_bytes: Maximum bytes to compare
        save_results: Save results to JSON
        verbose: Print progress

    Returns:
        Analysis results dictionary
    """
    path1 = get_struct_path(var1, iter1)
    path2 = get_struct_path(var2, iter2)

    # Validate paths
    if not path1.exists():
        raise FileNotFoundError(f"File not found: {path1}")
    if not path2.exists():
        raise FileNotFoundError(f"File not found: {path2}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  COMMUTATIVITY ANALYSIS: {var1}@{iter1} vs {var2}@{iter2}")
        print(f"{'='*60}")

    # Compare streams
    comparison = compare_streams(path1, path2, max_bytes=max_bytes, verbose=verbose)

    # Build results
    results = {
        "metadata": {
            "script": "apendix_commutativity_analysis.py",
            "generated_at": datetime.now().isoformat(),
            "variant1": var1,
            "iteration1": iter1,
            "variant2": var2,
            "iteration2": iter2,
            "max_bytes_limit": max_bytes,
        },
        "comparison": comparison,
        "conclusion": {
            "is_commutative": not comparison["divergence_found"],
            "divergence_point": comparison.get("divergence_byte"),
            "identical_region_bytes": comparison["identical_bytes"],
            "identical_region_chars": comparison["identical_chars"],
        }
    }

    # Interpretation
    if comparison["divergence_found"]:
        results["interpretation"] = (
            f"Variants {var1}@{iter1} and {var2}@{iter2} DIVERGE at byte "
            f"{comparison['divergence_byte']:,} (~char {comparison['divergence_char']:,}). "
            f"The annihilation process is LOCALLY commutative but diverges after this point."
        )
    else:
        results["interpretation"] = (
            f"Variants {var1}@{iter1} and {var2}@{iter2} are IDENTICAL "
            f"({comparison['identical_bytes']:,} bytes). "
            f"The annihilation process appears to be FULLY commutative."
        )

    if verbose:
        print(f"\n📋 CONCLUSION:")
        print(f"   {results['interpretation']}")

    # Save results
    if save_results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_DIR / f"commutativity_{var1}{iter1}_vs_{var2}{iter2}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"\n💾 Results saved to: {output_file}")

    return results


def extract_post_divergence_bits(
    path: Path,
    start_byte: int,
    max_bits: int = 1_000_000,
    verbose: bool = True
) -> str:
    """
    Extract binary bits from a region starting at a specific byte offset.

    Args:
        path: Path to .struct.gz file
        start_byte: Byte offset to start reading from
        max_bits: Maximum bits to extract
        verbose: Print progress

    Returns:
        Binary string (only '0' and '1', no parentheses)
    """
    DECODE = {0b00: '0', 0b01: '1', 0b10: None, 0b11: None}  # Skip parentheses

    bits = []
    bytes_needed = (max_bits // 4) + 1000  # 4 chars per byte, with margin

    with gzip.open(path, 'rb') as f:
        # Skip to start position
        f.seek(start_byte)

        # Read bytes
        raw = f.read(bytes_needed)

        # Decode to bits (skip parentheses)
        for byte in raw:
            for shift in [6, 4, 2, 0]:
                char = DECODE.get((byte >> shift) & 0b11)
                if char:
                    bits.append(char)
                    if len(bits) >= max_bits:
                        break
            if len(bits) >= max_bits:
                break

    return ''.join(bits)


def analyze_post_divergence(
    var1: str, iter1: int,
    var2: str, iter2: int,
    divergence_byte: int = 90_727_609,
    max_bits: int = 1_000_000,
    save_results: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze LZ complexity in the post-divergence region of two variants.

    Args:
        var1, iter1: First variant and iteration
        var2, iter2: Second variant and iteration
        divergence_byte: Byte offset where divergence starts
        max_bits: Maximum bits to analyze from each variant
        save_results: Save results to JSON
        verbose: Print progress

    Returns:
        Analysis results dictionary
    """
    path1 = get_struct_path(var1, iter1)
    path2 = get_struct_path(var2, iter2)

    # Validate paths
    if not path1.exists():
        raise FileNotFoundError(f"File not found: {path1}")
    if not path2.exists():
        raise FileNotFoundError(f"File not found: {path2}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  POST-DIVERGENCE ANALYSIS: {var1}@{iter1} vs {var2}@{iter2}")
        print(f"  Starting at byte: {divergence_byte:,}")
        print(f"{'='*60}")

    # Extract bits from post-divergence region
    if verbose:
        print(f"\n📥 Extracting bits from {var1}@{iter1}...")
    bits1 = extract_post_divergence_bits(path1, divergence_byte, max_bits, verbose)

    if verbose:
        print(f"📥 Extracting bits from {var2}@{iter2}...")
    bits2 = extract_post_divergence_bits(path2, divergence_byte, max_bits, verbose)

    if verbose:
        print(f"\n   {var1}@{iter1}: {len(bits1):,} bits extracted")
        print(f"   {var2}@{iter2}: {len(bits2):,} bits extracted")

    # Calculate LZ for each
    if verbose:
        print(f"\n🧬 Calculating LZ complexity...")

    lz1 = calculate_lz_ratio(bits1, verbose=verbose)
    lz2 = calculate_lz_ratio(bits2, verbose=verbose)

    # Build results
    results = {
        "metadata": {
            "script": "apendix_commutativity_analysis.py",
            "analysis_type": "post_divergence",
            "generated_at": datetime.now().isoformat(),
            "variant1": var1,
            "iteration1": iter1,
            "variant2": var2,
            "iteration2": iter2,
            "divergence_byte": divergence_byte,
            "max_bits_requested": max_bits,
        },
        "variant1_analysis": {
            "bits_extracted": len(bits1),
            "lz_analysis": lz1,
        },
        "variant2_analysis": {
            "bits_extracted": len(bits2),
            "lz_analysis": lz2,
        },
        "comparison": {
            "mean_ratio_diff": abs(lz1.get('mean_ratio', 0) - lz2.get('mean_ratio', 0)),
            "same_best_match": lz1.get('best_match') == lz2.get('best_match'),
        }
    }

    # Interpretation
    r1 = lz1.get('mean_ratio', 0)
    r2 = lz2.get('mean_ratio', 0)
    m1 = lz1.get('best_match', 'unknown')
    m2 = lz2.get('best_match', 'unknown')

    if verbose:
        print(f"\n📊 RESULTS:")
        print(f"   {var1}@{iter1}: LZ ratio = {r1:.4f} (best match: {m1})")
        print(f"   {var2}@{iter2}: LZ ratio = {r2:.4f} (best match: {m2})")

        if abs(r1 - INV_PHI) < 0.05 and abs(r2 - INV_PHI) < 0.05:
            print(f"\n   ✅ Both variants maintain 1/φ signature after divergence!")
        elif m1 == m2:
            print(f"\n   ⚠️ Both match {m1}, but values differ by {abs(r1-r2):.4f}")
        else:
            print(f"\n   🔴 Signatures DIFFER: {var1}→{m1}, {var2}→{m2}")

    results["interpretation"] = (
        f"Post-divergence LZ analysis: {var1}@{iter1} ratio={r1:.4f} ({m1}), "
        f"{var2}@{iter2} ratio={r2:.4f} ({m2}). "
        f"{'Signatures match.' if m1 == m2 else 'Signatures DIFFER!'}"
    )

    # Save results
    if save_results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_DIR / f"post_divergence_{var1}{iter1}_vs_{var2}{iter2}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"\n💾 Results saved to: {output_file}")

    return results


def full_analysis(
    reference: str = "B",
    variants: list = None,
    ref_iter: int = None,
    max_bits: int = 1_000_000,
    save_results: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform full commutativity analysis: find divergence points and analyze post-divergence.

    Args:
        reference: Reference variant (default: B)
        variants: List of variants to compare against reference (default: E, I)
        ref_iter: Reference iteration (default: highest available)
        max_bits: Maximum bits for post-divergence analysis
        save_results: Save results to JSON
        verbose: Print progress

    Returns:
        Complete analysis results
    """
    if variants is None:
        variants = ["E", "I"]

    # Get available variants
    available = get_available_variants()

    if verbose:
        print(f"\n{'='*70}")
        print(f"  FULL COMMUTATIVITY ANALYSIS")
        print(f"{'='*70}")
        print(f"\n📂 Available variants: {available}")

    # Validate reference
    if reference not in available:
        raise ValueError(f"Reference variant {reference} not found. Available: {list(available.keys())}")

    # Get reference iteration
    if ref_iter is None:
        ref_iter = available[reference]

    if verbose:
        print(f"\n🎯 Reference: {reference}@{ref_iter}")
        print(f"   Comparing against: {variants}")

    # Filter variants that exist
    valid_variants = []
    for v in variants:
        v = v.upper()
        if v == reference:
            continue
        if v in available:
            valid_variants.append((v, available[v]))
        else:
            if verbose:
                print(f"   ⚠️ Variant {v} not found, skipping")

    if not valid_variants:
        raise ValueError(f"No valid variants to compare. Available: {list(available.keys())}")

    # Results container
    results = {
        "metadata": {
            "script": "apendix_commutativity_analysis.py",
            "analysis_type": "full_analysis",
            "generated_at": datetime.now().isoformat(),
            "reference": reference,
            "reference_iteration": ref_iter,
            "variants_analyzed": [v for v, _ in valid_variants],
        },
        "divergence_analysis": {},
        "post_divergence_analysis": {},
        "summary": {}
    }

    # Phase 1: Find divergence points
    if verbose:
        print(f"\n{'─'*70}")
        print(f"  PHASE 1: DIVERGENCE DETECTION")
        print(f"{'─'*70}")

    divergence_points = {}
    for var, var_iter in valid_variants:
        if verbose:
            print(f"\n▶ Analyzing {reference}@{ref_iter} vs {var}@{var_iter}...")

        try:
            comm_result = analyze_commutativity(
                var1=reference, iter1=ref_iter,
                var2=var, iter2=var_iter,
                save_results=save_results,
                verbose=verbose
            )

            div_byte = comm_result["conclusion"]["divergence_point"]
            divergence_points[var] = {
                "iteration": var_iter,
                "divergence_byte": div_byte,
                "is_commutative": comm_result["conclusion"]["is_commutative"]
            }
            results["divergence_analysis"][f"{reference}_vs_{var}"] = comm_result

        except Exception as e:
            if verbose:
                print(f"   ❌ Error: {e}")
            divergence_points[var] = {"error": str(e)}

    # Phase 2: Post-divergence analysis
    if verbose:
        print(f"\n{'─'*70}")
        print(f"  PHASE 2: POST-DIVERGENCE LZ ANALYSIS")
        print(f"{'─'*70}")

    for var, var_iter in valid_variants:
        div_info = divergence_points.get(var, {})
        div_byte = div_info.get("divergence_byte")

        if div_byte is None:
            if verbose:
                print(f"\n▶ Skipping {var} (no divergence point or error)")
            continue

        if div_info.get("is_commutative", False):
            if verbose:
                print(f"\n▶ Skipping {var} (fully commutative, no divergence)")
            continue

        if verbose:
            print(f"\n▶ Post-divergence: {reference}@{ref_iter} vs {var}@{var_iter}")
            print(f"   Starting at byte: {div_byte:,}")

        try:
            post_result = analyze_post_divergence(
                var1=reference, iter1=ref_iter,
                var2=var, iter2=var_iter,
                divergence_byte=div_byte,
                max_bits=max_bits,
                save_results=save_results,
                verbose=verbose
            )
            results["post_divergence_analysis"][f"{reference}_vs_{var}"] = post_result

        except Exception as e:
            if verbose:
                print(f"   ❌ Error: {e}")

    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")

    summary_lines = []
    all_same_point = True
    first_div_byte = None

    for var, info in divergence_points.items():
        div_byte = info.get("divergence_byte")
        if div_byte is not None:
            if first_div_byte is None:
                first_div_byte = div_byte
            elif div_byte != first_div_byte:
                all_same_point = False

            summary_lines.append(f"   {reference} vs {var}: diverges at byte {div_byte:,}")
        elif info.get("is_commutative"):
            summary_lines.append(f"   {reference} vs {var}: FULLY COMMUTATIVE ✅")
        else:
            summary_lines.append(f"   {reference} vs {var}: ERROR - {info.get('error', 'unknown')}")

    results["summary"]["divergence_points"] = divergence_points
    results["summary"]["all_diverge_at_same_point"] = all_same_point
    results["summary"]["common_divergence_byte"] = first_div_byte if all_same_point else None

    if verbose:
        print("\n📊 Divergence Points:")
        for line in summary_lines:
            print(line)

        if all_same_point and first_div_byte:
            print(f"\n   ✅ All variants diverge at the SAME point: byte {first_div_byte:,}")
        elif first_div_byte:
            print(f"\n   ⚠️ Variants diverge at DIFFERENT points!")

    # Save full results
    if save_results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        variants_str = "_".join([v for v, _ in valid_variants])
        output_file = OUTPUT_DIR / f"full_analysis_{reference}{ref_iter}_vs_{variants_str}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"\n💾 Full results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze commutativity between HSI variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full automatic analysis (B vs E and I, auto-detect iterations)
  python apendix_commutativity_analysis.py --full-analysis

  # Full analysis with specific variants
  python apendix_commutativity_analysis.py --full-analysis --variants E I F

  # Full analysis with different reference
  python apendix_commutativity_analysis.py --full-analysis --reference B --variants E

  # Manual commutativity check
  python apendix_commutativity_analysis.py --var1 B --iter1 23 --var2 E --iter2 23

  # Manual post-divergence analysis
  python apendix_commutativity_analysis.py --var1 B --iter1 23 --var2 E --iter2 23 --post-divergence
        """
    )

    # Full analysis mode
    parser.add_argument("--full-analysis", action="store_true",
                        help="Run full analysis: detect divergence + post-divergence LZ")
    parser.add_argument("--reference", type=str, default="B",
                        help="Reference variant for full analysis (default: B)")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Variants to compare (default: E I)")

    # Manual mode
    parser.add_argument("--var1", type=str, default=None, help="First variant (e.g., B)")
    parser.add_argument("--iter1", type=int, default=None, help="First iteration")
    parser.add_argument("--var2", type=str, default=None, help="Second variant (e.g., E)")
    parser.add_argument("--iter2", type=int, default=None, help="Second iteration")

    # Options
    parser.add_argument("--max-bytes", type=int, default=None,
                        help="Maximum bytes to compare (default: all)")
    parser.add_argument("--post-divergence", action="store_true",
                        help="Analyze LZ complexity in post-divergence region")
    parser.add_argument("--divergence-byte", type=int, default=None,
                        help="Byte offset where divergence starts (auto-detected if not specified)")
    parser.add_argument("--max-bits", type=int, default=1_000_000,
                        help="Maximum bits to analyze for post-divergence (default: 1000000)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to JSON")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    try:
        if args.full_analysis:
            # Full automatic analysis
            results = full_analysis(
                reference=args.reference.upper(),
                variants=args.variants,
                ref_iter=args.iter1,  # Use iter1 as reference iteration if specified
                max_bits=args.max_bits,
                save_results=not args.no_save,
                verbose=not args.quiet
            )
            sys.exit(0)

        elif args.post_divergence:
            # Manual post-divergence analysis
            if not all([args.var1, args.iter1, args.var2, args.iter2]):
                print("❌ ERROR: --post-divergence requires --var1, --iter1, --var2, --iter2")
                sys.exit(1)

            divergence_byte = args.divergence_byte
            if divergence_byte is None:
                # Auto-detect divergence point first
                print("🔍 Auto-detecting divergence point...")
                comm_result = analyze_commutativity(
                    var1=args.var1.upper(), iter1=args.iter1,
                    var2=args.var2.upper(), iter2=args.iter2,
                    save_results=False, verbose=not args.quiet
                )
                divergence_byte = comm_result["conclusion"]["divergence_point"]
                if divergence_byte is None:
                    print("✅ No divergence found - variants are identical!")
                    sys.exit(0)

            results = analyze_post_divergence(
                var1=args.var1.upper(),
                iter1=args.iter1,
                var2=args.var2.upper(),
                iter2=args.iter2,
                divergence_byte=divergence_byte,
                max_bits=args.max_bits,
                save_results=not args.no_save,
                verbose=not args.quiet
            )
            sys.exit(0)

        else:
            # Standard commutativity analysis
            if not all([args.var1, args.iter1, args.var2, args.iter2]):
                print("❌ ERROR: Manual mode requires --var1, --iter1, --var2, --iter2")
                print("   Or use --full-analysis for automatic mode")
                sys.exit(1)

            results = analyze_commutativity(
                var1=args.var1.upper(),
                iter1=args.iter1,
                var2=args.var2.upper(),
                iter2=args.iter2,
                max_bytes=args.max_bytes,
                save_results=not args.no_save,
                verbose=not args.quiet
            )
            # Exit code based on result
            sys.exit(0 if results["conclusion"]["is_commutative"] else 1)

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()

