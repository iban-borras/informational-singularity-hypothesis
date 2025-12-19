#!/usr/bin/env python3
"""
Level 1 — Multi-Scale Ratio Analyzer (MSRA)

Discovers hidden φ in PROPORTIONS between structures at different scales.

The hypothesis: φ may exist in the RATIOS between pattern sizes, not in
the patterns themselves — like how human body bone ratios approach φ.

Usage:
    python level1_scale_ratios.py --variant B --iteration 15
    python level1_scale_ratios.py --variant B --iteration 15 --max-bits 1000000
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from level1.data_loader import load_phi_for_level1

PHI = (1 + math.sqrt(5)) / 2  # 1.6180339887...


def segment_into_blocks(bits: str, block_size: int) -> List[str]:
    """Segment bit string into blocks of given size."""
    n_blocks = len(bits) // block_size
    return [bits[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]


def block_entropy(block: str) -> float:
    """Calculate Shannon entropy of a block."""
    if not block:
        return 0.0
    counts = Counter(block)
    total = len(block)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def ones_density(block: str) -> float:
    """Calculate density of 1s in a block."""
    if not block:
        return 0.0
    return block.count('1') / len(block)


def unique_pattern_ratio(blocks: List[str]) -> float:
    """Calculate ratio of unique patterns to total patterns."""
    if not blocks:
        return 0.0
    unique = len(set(blocks))
    return unique / len(blocks)


def run_length_stats(bits: str) -> Tuple[float, float]:
    """Calculate mean and std of run lengths (consecutive 0s or 1s)."""
    if not bits:
        return 0.0, 0.0
    runs = []
    current_char = bits[0]
    current_len = 1
    for char in bits[1:]:
        if char == current_char:
            current_len += 1
        else:
            runs.append(current_len)
            current_char = char
            current_len = 1
    runs.append(current_len)
    return float(np.mean(runs)), float(np.std(runs))


def analyze_scale(bits: str, scale: int, verbose: bool = False) -> Dict[str, float]:
    """Analyze metrics at a given scale (block size)."""
    blocks = segment_into_blocks(bits, scale)
    if not blocks:
        return {}

    n_blocks = len(blocks)
    if verbose and n_blocks > 1000:
        print(f"      Processing {n_blocks:,} blocks...", end=" ", flush=True)

    entropies = [block_entropy(b) for b in blocks]
    densities = [ones_density(b) for b in blocks]
    run_mean, run_std = run_length_stats(bits[:n_blocks * scale])

    if verbose and n_blocks > 1000:
        print("done", flush=True)

    return {
        'entropy_mean': float(np.mean(entropies)),
        'entropy_std': float(np.std(entropies)),
        'density_mean': float(np.mean(densities)),
        'density_std': float(np.std(densities)),
        'unique_ratio': unique_pattern_ratio(blocks),
        'run_length_mean': run_mean,
        'run_length_std': run_std,
        'n_blocks': n_blocks
    }


def calculate_scale_ratios(
    metrics_by_scale: Dict[int, Dict[str, float]],
    scales: List[int]
) -> Dict[str, List[float]]:
    """Calculate ratios between consecutive scales for each metric."""
    ratios = {
        'entropy_mean': [],
        'density_mean': [],
        'unique_ratio': [],
        'run_length_mean': [],
        'scale_pairs': []
    }
    
    for i in range(len(scales) - 1):
        s1, s2 = scales[i], scales[i + 1]
        m1, m2 = metrics_by_scale.get(s1), metrics_by_scale.get(s2)
        
        if not m1 or not m2:
            continue
        
        ratios['scale_pairs'].append(f"{s2}/{s1}")
        
        for metric in ['entropy_mean', 'density_mean', 'unique_ratio', 'run_length_mean']:
            v1, v2 = m1.get(metric, 0), m2.get(metric, 0)
            if v1 > 0:
                ratios[metric].append(v2 / v1)
            else:
                ratios[metric].append(float('nan'))
    
    return ratios


def phi_proximity(ratios: List[float]) -> Tuple[float, float]:
    """Calculate how close ratios are to φ. Returns (mean_distance, std_distance)."""
    valid = [r for r in ratios if not math.isnan(r) and r > 0]
    if not valid:
        return float('nan'), float('nan')
    distances = [abs(r - PHI) for r in valid]
    return float(np.mean(distances)), float(np.std(distances))


def analyze_multi_scale(
    bits: str,
    scales: Optional[List[int]] = None,
    verbose: bool = True
) -> Dict:
    """
    Main analysis: find φ in proportions between scales.
    
    Args:
        bits: Binary string (observable Φ)
        scales: List of block sizes (default: powers of 2 from 16 to 4096)
        verbose: Print progress
    
    Returns:
        Dictionary with analysis results
    """
    if scales is None:
        # Default scales: powers of 2
        max_scale = min(4096, len(bits) // 10)
        scales = [2**k for k in range(4, 13) if 2**k <= max_scale]
    
    if verbose:
        print(f"\n📊 Multi-Scale Ratio Analysis (MSRA)")
        print(f"   Data size: {len(bits):,} bits")
        print(f"   Scales: {scales}")
        print(f"   Target: φ = {PHI:.6f}")
    
    # Analyze each scale
    metrics_by_scale = {}
    total_scales = len(scales)
    for idx, scale in enumerate(scales):
        progress = (idx + 1) / total_scales * 100
        if verbose:
            print(f"   [{idx+1}/{total_scales}] Scale {scale} ({progress:.0f}%)...", end=" ", flush=True)
        metrics = analyze_scale(bits, scale, verbose=verbose)
        if metrics:
            metrics_by_scale[scale] = metrics
            if verbose:
                print(f"✓ ({metrics['n_blocks']:,} blocks)")
        else:
            if verbose:
                print("✗ (insufficient data)")
    
    # Calculate inter-scale ratios
    ratios = calculate_scale_ratios(metrics_by_scale, scales)
    
    # Calculate φ proximity for each metric
    phi_results = {}
    for metric in ['entropy_mean', 'density_mean', 'unique_ratio', 'run_length_mean']:
        mean_dist, std_dist = phi_proximity(ratios[metric])
        phi_results[metric] = {
            'ratios': ratios[metric],
            'phi_distance_mean': mean_dist,
            'phi_distance_std': std_dist
        }
    
    return {
        'scales': scales,
        'scale_pairs': ratios['scale_pairs'],
        'metrics_by_scale': metrics_by_scale,
        'phi_analysis': phi_results,
        'phi_target': PHI,
        'data_size': len(bits)
    }


def print_results(results: Dict) -> None:
    """Print analysis results in a readable format."""
    print(f"\n{'='*60}")
    print(f"🔮 MULTI-SCALE RATIO ANALYSIS — Hidden φ Discovery")
    print(f"{'='*60}")
    print(f"   φ target: {results['phi_target']:.6f}")
    print(f"   Data: {results['data_size']:,} bits")
    print(f"   Scales analyzed: {results['scales']}")

    print(f"\n📈 Scale-to-Scale Ratios (looking for φ ≈ 1.618):")
    print(f"   {'Metric':<20} {'Ratios':<40} {'φ-Distance'}")
    print(f"   {'-'*20} {'-'*40} {'-'*15}")

    for metric, data in results['phi_analysis'].items():
        ratios_str = ", ".join([f"{r:.3f}" for r in data['ratios'][:5]])
        if len(data['ratios']) > 5:
            ratios_str += "..."
        dist = data['phi_distance_mean']
        dist_str = f"{dist:.4f}" if not math.isnan(dist) else "N/A"
        print(f"   {metric:<20} {ratios_str:<40} {dist_str}")

    # Find best metric (closest to φ)
    best_metric = None
    best_dist = float('inf')
    for metric, data in results['phi_analysis'].items():
        dist = data['phi_distance_mean']
        if not math.isnan(dist) and dist < best_dist:
            best_dist = dist
            best_metric = metric

    if best_metric:
        print(f"\n🎯 Best φ-alignment: {best_metric} (distance: {best_dist:.4f})")
        if best_dist < 0.1:
            print(f"   ✅ STRONG φ signal detected!")
        elif best_dist < 0.3:
            print(f"   ⚠️  Moderate φ signal")
        else:
            print(f"   ❌ Weak φ signal")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Scale Ratio Analyzer — Find hidden φ in proportions"
    )
    parser.add_argument("--variant", "-v", default="B", help="Variant (B, D, E, F, G, H, I)")
    parser.add_argument("--iteration", "-i", type=int, default=15, help="Iteration number")
    parser.add_argument("--max-bits", "-m", type=int, default=None, help="Max bits to analyze")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file")
    parser.add_argument("--scales", "-s", type=str, default=None,
                        help="Comma-separated scales (e.g., '16,32,64,128')")

    args = parser.parse_args()

    # Determine data directory (try both locations)
    data_dir = f"results/level0/phi_snapshots/var_{args.variant}"
    if not os.path.exists(data_dir):
        data_dir = f"results/level0/var_{args.variant}"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found for variant {args.variant}")
        sys.exit(1)

    print(f"\n🔬 Loading Variant {args.variant}, iteration {args.iteration}...")

    try:
        _, phi_observable, metadata = load_phi_for_level1(
            data_dir, args.iteration,
            return_structural=False,
            return_observable=True,
            return_metadata=True
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    print(f"   Loaded {len(phi_observable):,} bits")

    # Limit bits if requested
    if args.max_bits and args.max_bits < len(phi_observable):
        phi_observable = phi_observable[:args.max_bits]
        print(f"   Truncated to {args.max_bits:,} bits")

    # Parse custom scales
    scales = None
    if args.scales:
        scales = [int(s) for s in args.scales.split(",")]

    # Run analysis
    results = analyze_multi_scale(phi_observable, scales=scales)
    results['variant'] = args.variant
    results['iteration'] = args.iteration

    # Print results
    print_results(results)

    # Save to JSON if requested
    if args.output:
        # Convert numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(x) for x in obj]
            return obj

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(convert_for_json(results), f, indent=2)
        print(f"\n📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main()

