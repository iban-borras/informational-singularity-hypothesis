#!/usr/bin/env python3
"""
Level 1 — Deep Analysis Tools (Phase 2)

Unified script for advanced pattern discovery including:
1. Wavelet φ-Band Detection — Find φ signatures in frequency domain
2. Recurrence Plot Analysis — Detect temporal dynamics and periodicity
3. Lempel-Ziv Complexity — Measure algorithmic complexity at multiple scales

Usage:
    python level1_deep_analysis.py --variant B --iteration 15
    python level1_deep_analysis.py --variant B --iteration 15 --analysis wavelet
    python level1_deep_analysis.py --variant B --iteration 15 --analysis recurrence
    python level1_deep_analysis.py --variant B --iteration 15 --compare F
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from level1.data_loader import load_phi_for_level1

PHI = (1 + math.sqrt(5)) / 2


# =============================================================================
# Wavelet φ-Band Detection
# =============================================================================

def wavelet_analysis(bits: str, max_bits: int = 100000, verbose: bool = True) -> Dict:
    """
    Detect φ-related frequency bands using wavelet-like multi-scale analysis.
    
    Instead of full CWT (computationally expensive), we use a simplified
    multi-scale approach: analyze entropy at scales that form φ-ratios.
    
    Args:
        bits: Binary string
        max_bits: Maximum bits to analyze
        verbose: Print progress
    
    Returns:
        Analysis results with φ-band detection
    """
    if verbose:
        print(f"\n📊 Wavelet φ-Band Detection")
        print(f"   Data: {len(bits):,} bits (analyzing {min(len(bits), max_bits):,})")
    
    data = bits[:max_bits]
    data_arr = np.array([int(b) for b in data], dtype=np.float32)
    
    # Generate φ-related scales: 2, 2φ, 2φ², 2φ³, ...
    phi_scales = []
    scale = 2
    while scale < len(data) // 10:
        phi_scales.append(int(scale))
        scale *= PHI
    
    # Also generate power-of-2 scales for comparison
    pow2_scales = [2**i for i in range(1, 15) if 2**i < len(data) // 10]
    
    if verbose:
        print(f"   φ-scales: {phi_scales[:8]}...")
        print(f"   2^n-scales: {pow2_scales[:8]}...")
    
    def calc_scale_entropy(arr: np.ndarray, scale: int) -> float:
        """Calculate entropy at given scale by averaging blocks."""
        n_blocks = len(arr) // scale
        if n_blocks < 2:
            return 0.0
        blocks = arr[:n_blocks * scale].reshape(n_blocks, scale)
        densities = blocks.mean(axis=1)
        # Bin densities and calculate entropy
        hist, _ = np.histogram(densities, bins=10, range=(0, 1))
        hist = hist[hist > 0] / hist.sum()
        return float(-np.sum(hist * np.log2(hist + 1e-10)))
    
    phi_entropies = []
    pow2_entropies = []
    
    if verbose:
        print(f"   Calculating φ-scale entropies...", end=" ", flush=True)
    
    for scale in phi_scales:
        phi_entropies.append(calc_scale_entropy(data_arr, scale))
    
    if verbose:
        print(f"✓")
        print(f"   Calculating 2^n-scale entropies...", end=" ", flush=True)
    
    for scale in pow2_scales:
        pow2_entropies.append(calc_scale_entropy(data_arr, scale))
    
    if verbose:
        print(f"✓")
    
    # Calculate smoothness (lower = more structured at those scales)
    phi_variance = float(np.var(phi_entropies)) if phi_entropies else 0
    pow2_variance = float(np.var(pow2_entropies)) if pow2_entropies else 0
    
    # φ-band strength: if φ-scales show more consistent entropy, φ is present
    phi_band_strength = 1.0 / (1.0 + phi_variance) if phi_variance > 0 else 0
    pow2_band_strength = 1.0 / (1.0 + pow2_variance) if pow2_variance > 0 else 0
    
    # Ratio indicates φ-preference
    phi_preference = phi_band_strength / (pow2_band_strength + 0.001)
    
    return {
        'phi_scales': phi_scales,
        'pow2_scales': pow2_scales,
        'phi_entropies': phi_entropies,
        'pow2_entropies': pow2_entropies,
        'phi_variance': phi_variance,
        'pow2_variance': pow2_variance,
        'phi_band_strength': phi_band_strength,
        'pow2_band_strength': pow2_band_strength,
        'phi_preference': phi_preference,
        'phi_detected': phi_preference > 1.2  # φ-scales more consistent than 2^n
    }


# =============================================================================
# Recurrence Plot Analysis
# =============================================================================

def recurrence_analysis(bits: str, max_bits: int = 10000, 
                        embedding_dim: int = 3, delay: int = 1,
                        threshold: float = 0.1, verbose: bool = True) -> Dict:
    """
    Recurrence Plot Analysis — detect temporal dynamics.
    
    Args:
        bits: Binary string
        max_bits: Maximum bits (recurrence is O(n²), keep small!)
        embedding_dim: Embedding dimension for phase space
        delay: Time delay for embedding
        threshold: Distance threshold for recurrence
        verbose: Print progress
    
    Returns:
        Recurrence metrics including determinism and laminarity
    """
    if verbose:
        print(f"\n🔄 Recurrence Plot Analysis")
        print(f"   Data: {len(bits):,} bits (analyzing {min(len(bits), max_bits):,})")
        print(f"   Embedding: dim={embedding_dim}, delay={delay}")
    
    data = bits[:max_bits]
    data_arr = np.array([int(b) for b in data], dtype=np.float32)
    
    # Create embedded vectors
    n = len(data_arr) - (embedding_dim - 1) * delay
    if n < 10:
        return {'error': 'Insufficient data for embedding'}
    
    if verbose:
        print(f"   Creating {n} embedded vectors...", end=" ", flush=True)
    
    embedded = np.zeros((n, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = data_arr[i * delay:i * delay + n]
    
    if verbose:
        print(f"✓")
        print(f"   Building recurrence matrix...", end=" ", flush=True)
    
    # Calculate recurrence matrix (O(n²) - that's why we limit max_bits)
    # Use vectorized approach for speed
    diff = embedded[:, np.newaxis, :] - embedded[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    recurrence_matrix = (distances < threshold).astype(np.int8)
    
    if verbose:
        print(f"✓ ({n}×{n} matrix)")
    
    # Calculate RQA metrics
    total_points = n * n
    recurrence_rate = float(np.sum(recurrence_matrix)) / total_points
    
    # Determinism: ratio of recurrence points forming diagonal lines
    if verbose:
        print(f"   Calculating determinism...", end=" ", flush=True)
    
    diag_lengths = []
    for k in range(-n + 1, n):
        diag = np.diag(recurrence_matrix, k)
        # Find runs of 1s
        runs = np.diff(np.where(np.concatenate(([0], diag, [0])) != 1)[0])[::2]
        diag_lengths.extend(runs[runs >= 2].tolist())
    
    if diag_lengths:
        determinism = sum(diag_lengths) / max(1, np.sum(recurrence_matrix))
        avg_diag_length = float(np.mean(diag_lengths))
        max_diag_length = int(max(diag_lengths))
    else:
        determinism = 0
        avg_diag_length = 0
        max_diag_length = 0
    
    if verbose:
        print(f"✓")
    
    return {
        'recurrence_rate': recurrence_rate,
        'determinism': determinism,
        'avg_diagonal_length': avg_diag_length,
        'max_diagonal_length': max_diag_length,
        'embedding_dim': embedding_dim,
        'delay': delay,
        'threshold': threshold,
        'matrix_size': n,
        'is_deterministic': determinism > 0.5
    }


# =============================================================================
# Multi-Scale Lempel-Ziv Complexity
# =============================================================================

def lempel_ziv_complexity(s: str) -> int:
    """Calculate Lempel-Ziv complexity (LZ76)."""
    n = len(s)
    if n == 0:
        return 0

    complexity = 1
    prefix_len = 1

    while prefix_len < n:
        # Find longest match
        max_match = 0
        for start in range(prefix_len):
            match_len = 0
            while (prefix_len + match_len < n and
                   s[start + match_len] == s[prefix_len + match_len]):
                match_len += 1
                if start + match_len >= prefix_len:
                    break
            max_match = max(max_match, match_len)

        prefix_len += max(1, max_match + 1)
        complexity += 1

    return complexity


def multiscale_lz_analysis(bits: str, max_bits: int = 100000,
                           verbose: bool = True) -> Dict:
    """
    Analyze Lempel-Ziv complexity at multiple scales.

    If φ is hidden in the structure, LZ complexity ratios between
    φ-related scales might show special properties.

    Args:
        bits: Binary string
        max_bits: Maximum bits to analyze
        verbose: Print progress

    Returns:
        Multi-scale LZ analysis results
    """
    if verbose:
        print(f"\n🧬 Multi-Scale Lempel-Ziv Complexity")
        print(f"   Data: {len(bits):,} bits (analyzing {min(len(bits), max_bits):,})")

    data = bits[:max_bits]

    # Scales to analyze
    scales = [1000, 2000, 5000, 10000, 20000, 50000]
    scales = [s for s in scales if s <= len(data)]

    if verbose:
        print(f"   Scales: {scales}")

    results = {}
    total_scales = len(scales)

    for idx, scale in enumerate(scales):
        if verbose:
            progress = (idx + 1) / total_scales * 100
            print(f"   [{idx+1}/{total_scales}] Scale {scale} ({progress:.0f}%)...", end=" ", flush=True)

        segment = data[:scale]
        lz = lempel_ziv_complexity(segment)

        # Normalized LZ: LZ / (n / log2(n))
        normalized = lz / (scale / math.log2(scale)) if scale > 1 else 0

        results[scale] = {
            'raw_lz': lz,
            'normalized_lz': float(normalized)
        }

        if verbose:
            print(f"LZ={lz}, norm={normalized:.4f}")

    # Calculate scale ratios
    sorted_scales = sorted(results.keys())
    scale_ratios = []
    lz_ratios = []

    for i in range(len(sorted_scales) - 1):
        s1, s2 = sorted_scales[i], sorted_scales[i + 1]
        scale_ratios.append(s2 / s1)
        if results[s1]['normalized_lz'] > 0:
            lz_ratios.append(results[s2]['normalized_lz'] / results[s1]['normalized_lz'])

    # Check if LZ ratios are close to φ or 1/φ
    phi_distances = [min(abs(r - PHI), abs(r - 1/PHI)) for r in lz_ratios]
    mean_phi_distance = float(np.mean(phi_distances)) if phi_distances else float('nan')

    return {
        'by_scale': results,
        'scale_ratios': scale_ratios,
        'lz_ratios': lz_ratios,
        'phi_distances': phi_distances,
        'mean_phi_distance': mean_phi_distance,
        'phi_detected': mean_phi_distance < 0.3 if not math.isnan(mean_phi_distance) else False
    }


# =============================================================================
# Main Analysis Runner
# =============================================================================

def run_deep_analysis(
    variant: str,
    iteration: int,
    analyses: Optional[List[str]] = None,
    max_bits: int = 100000,
    verbose: bool = True,
    output_path: Optional[str] = None
) -> Dict:
    """
    Run deep analysis on a variant.

    Args:
        variant: Variant code (B, F, etc.)
        iteration: Iteration number
        analyses: List of analyses to run ('wavelet', 'recurrence', 'lz', 'all')
        max_bits: Maximum bits to analyze
        verbose: Print progress
        output_path: Optional path to save JSON results

    Returns:
        Combined analysis results
    """
    if analyses is None or 'all' in analyses:
        analyses = ['wavelet', 'recurrence', 'lz']

    # Load data
    data_dir = f"results/level0/phi_snapshots/var_{variant}"
    if not os.path.exists(data_dir):
        data_dir = f"results/level0/var_{variant}"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found for variant {variant}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"🔬 DEEP ANALYSIS — Variant {variant}, Iteration {iteration}")
        print(f"{'='*60}")

    _, phi_observable, metadata = load_phi_for_level1(
        data_dir, iteration,
        return_structural=False,
        return_observable=True,
        return_metadata=True
    )

    if verbose:
        print(f"   Loaded {len(phi_observable):,} bits")

    results = {
        'variant': variant,
        'iteration': iteration,
        'total_bits': len(phi_observable),
        'max_bits_analyzed': min(max_bits, len(phi_observable)),
        'phi_target': PHI
    }

    # Run requested analyses
    if 'wavelet' in analyses:
        results['wavelet'] = wavelet_analysis(phi_observable, max_bits, verbose)

    if 'recurrence' in analyses:
        # Recurrence is O(n²), use smaller sample
        recurrence_bits = min(10000, max_bits // 10)
        results['recurrence'] = recurrence_analysis(phi_observable, recurrence_bits, verbose=verbose)

    if 'lz' in analyses:
        results['lz'] = multiscale_lz_analysis(phi_observable, max_bits, verbose)

    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"📊 SUMMARY")
        print(f"{'='*60}")

        if 'wavelet' in results:
            w = results['wavelet']
            status = "✅ DETECTED" if w.get('phi_detected') else "❌ Not detected"
            print(f"   Wavelet φ-band: {status} (preference={w['phi_preference']:.3f})")

        if 'recurrence' in results:
            r = results['recurrence']
            status = "✅ DETERMINISTIC" if r.get('is_deterministic') else "❌ Stochastic"
            print(f"   Recurrence: {status} (det={r['determinism']:.3f})")

        if 'lz' in results:
            lz = results['lz']
            status = "✅ DETECTED" if lz.get('phi_detected') else "❌ Not detected"
            print(f"   LZ φ-scaling: {status} (φ-dist={lz['mean_phi_distance']:.3f})")

    if output_path:
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(x) for x in obj]
            return obj

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=2)
        if verbose:
            print(f"\n📁 Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Deep Analysis Tools — Wavelet, Recurrence, LZ complexity"
    )
    parser.add_argument("--variant", "-v", default="B", help="Variant (B, D, E, F, G, H, I)")
    parser.add_argument("--iteration", "-i", type=int, default=15, help="Iteration number")
    parser.add_argument("--max-bits", "-m", type=int, default=100000, help="Max bits to analyze")
    parser.add_argument("--analysis", "-a", type=str, default="all",
                        help="Analysis type: wavelet, recurrence, lz, all (comma-separated)")
    parser.add_argument("--compare", "-c", type=str, default=None,
                        help="Compare with another variant (e.g., --compare F)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    analyses = args.analysis.split(",") if args.analysis != "all" else ["all"]

    try:
        results = run_deep_analysis(
            args.variant, args.iteration,
            analyses=analyses,
            max_bits=args.max_bits,
            output_path=args.output
        )

        if args.compare:
            print(f"\n{'='*60}")
            print(f"📊 COMPARISON: {args.variant} vs {args.compare}")
            print(f"{'='*60}")

            results_compare = run_deep_analysis(
                args.compare, args.iteration,
                analyses=analyses,
                max_bits=args.max_bits
            )

            # Compare key metrics
            print(f"\n   Metric                    {args.variant:>10}  {args.compare:>10}")
            print(f"   {'-'*45}")

            if 'wavelet' in results and 'wavelet' in results_compare:
                print(f"   φ-preference              {results['wavelet']['phi_preference']:>10.3f}  {results_compare['wavelet']['phi_preference']:>10.3f}")

            if 'recurrence' in results and 'recurrence' in results_compare:
                print(f"   Determinism               {results['recurrence']['determinism']:>10.3f}  {results_compare['recurrence']['determinism']:>10.3f}")

            if 'lz' in results and 'lz' in results_compare:
                print(f"   LZ φ-distance             {results['lz']['mean_phi_distance']:>10.3f}  {results_compare['lz']['mean_phi_distance']:>10.3f}")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

