#!/usr/bin/env python3
"""
Level 1 — Topological Persistence Analysis (TPA)

Find structural "holes" and connectivity patterns that may encode φ.
Uses simplified persistent homology without heavy TDA libraries.

The key insight: If φ is encoded in the TOPOLOGY of Φ, we should find
φ-like ratios in the birth/death times of topological features.

Requirements:
    - phi_snapshots data (structural or observable format)
    - numpy (standard dependency)

Usage:
    python level1_topological.py --variant B --iteration 15
    python level1_topological.py --variant B --iteration 15 --max-bits 100000
    python level1_topological.py --variant B --iteration 15 --compare F

Author: Sophia (Augment Agent) for Iban's HSI Project
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
INV_PHI = 1 / PHI


def log_step(msg: str, level: int = 0):
    """Print a log step with proper indentation."""
    indent = "   " * level
    print(f"{indent}{msg}")


def log_progress(current: int, total: int, prefix: str = ""):
    """Print progress percentage."""
    pct = current / total * 100
    print(f"\r   {prefix} [{current}/{total}] {pct:.1f}%", end="", flush=True)
    if current == total:
        print(" ✓")


# =============================================================================
# Simplified Persistent Homology
# =============================================================================

def compute_run_lengths(bits: str) -> Tuple[List[int], List[int]]:
    """
    Compute run lengths of 0s and 1s.
    
    These represent "holes" (0-runs) and "matter" (1-runs) in the sequence.
    """
    runs_0 = []
    runs_1 = []
    
    if not bits:
        return runs_0, runs_1
    
    current_char = bits[0]
    current_len = 1
    
    for i in range(1, len(bits)):
        if bits[i] == current_char:
            current_len += 1
        else:
            if current_char == '0':
                runs_0.append(current_len)
            else:
                runs_1.append(current_len)
            current_char = bits[i]
            current_len = 1
    
    # Don't forget the last run
    if current_char == '0':
        runs_0.append(current_len)
    else:
        runs_1.append(current_len)
    
    return runs_0, runs_1


def compute_persistence_diagram(runs: List[int], feature_type: str = "H0") -> List[Tuple[float, float]]:
    """
    Create a simplified persistence diagram from run lengths.
    
    Interpretation:
    - Each run "is born" at position 0
    - Each run "dies" at its length
    - Longer runs = more persistent features
    
    This is a simplification of true persistent homology, but captures
    the essential idea: which structures persist across scales?
    """
    if not runs:
        return []
    
    # Normalize runs to [0, 1] range
    max_run = max(runs)
    if max_run == 0:
        return []
    
    diagram = []
    for run in runs:
        birth = 0.0
        death = run / max_run
        diagram.append((birth, death))
    
    return diagram


def analyze_persistence_ratios(diagram: List[Tuple[float, float]], verbose: bool = True) -> Dict:
    """
    Analyze the persistence diagram for φ-like ratios.
    
    Key metric: Do the lifespans (death - birth) show φ ratios?
    """
    if len(diagram) < 2:
        return {'error': 'Insufficient features for ratio analysis'}
    
    lifespans = sorted([d[1] - d[0] for d in diagram], reverse=True)
    
    # Calculate consecutive ratios
    ratios = []
    for i in range(len(lifespans) - 1):
        if lifespans[i + 1] > 0.001:  # Avoid division by tiny numbers
            ratios.append(lifespans[i] / lifespans[i + 1])
    
    if not ratios:
        return {'error': 'No valid ratios computed'}
    
    # Check proximity to φ and 1/φ
    phi_distances = []
    for r in ratios:
        dist_phi = abs(r - PHI)
        dist_inv = abs(r - INV_PHI)
        phi_distances.append(min(dist_phi, dist_inv))
    
    mean_phi_dist = float(np.mean(phi_distances))
    std_phi_dist = float(np.std(phi_distances))
    
    # Count ratios close to φ (within 0.1)
    close_to_phi = sum(1 for d in phi_distances if d < 0.1)
    phi_ratio_fraction = close_to_phi / len(ratios)
    
    return {
        'n_features': len(lifespans),
        'lifespans_top10': lifespans[:10],
        'ratios_top10': ratios[:10],
        'mean_ratio': float(np.mean(ratios)),
        'std_ratio': float(np.std(ratios)),
        'phi_distances_top10': phi_distances[:10],
        'mean_phi_distance': mean_phi_dist,
        'std_phi_distance': std_phi_dist,
        'phi_ratio_fraction': phi_ratio_fraction,
        'phi_detected': mean_phi_dist < 0.3 and phi_ratio_fraction > 0.2
    }


# =============================================================================
# Betti Numbers Analysis
# =============================================================================

def compute_betti_at_scales(bits: str, scales: List[int], verbose: bool = True) -> Dict:
    """
    Compute Betti-like numbers at multiple scales.

    At each scale s, we coarse-grain the sequence by blocks of size s,
    then count connected components (H0) and cycles (H1-proxy).

    Args:
        bits: Binary string
        scales: List of scales to analyze
        verbose: Print progress

    Returns:
        Betti numbers and φ-ratio analysis
    """
    if verbose:
        log_step("Computing Betti numbers at multiple scales...")

    betti_0 = []  # Connected components (1-runs after coarse-graining)
    betti_1 = []  # Cycles proxy (alternation frequency)

    total_scales = len(scales)
    for idx, scale in enumerate(scales):
        if verbose:
            log_progress(idx + 1, total_scales, "Scales")

        # Coarse-grain: each block becomes 1 if majority is 1, else 0
        n_blocks = len(bits) // scale
        if n_blocks < 2:
            betti_0.append(0)
            betti_1.append(0)
            continue

        coarse = []
        for i in range(n_blocks):
            block = bits[i * scale:(i + 1) * scale]
            ones = block.count('1')
            coarse.append('1' if ones > scale // 2 else '0')

        coarse_str = ''.join(coarse)

        # H0: count connected components (1-runs)
        runs_0, runs_1 = compute_run_lengths(coarse_str)
        betti_0.append(len(runs_1))

        # H1 proxy: count alternations (transitions between 0 and 1)
        alternations = sum(1 for i in range(len(coarse_str) - 1)
                          if coarse_str[i] != coarse_str[i + 1])
        betti_1.append(alternations)

    # Analyze ratios between consecutive scales
    b0_ratios = []
    b1_ratios = []

    for i in range(len(betti_0) - 1):
        if betti_0[i + 1] > 0:
            b0_ratios.append(betti_0[i] / betti_0[i + 1])
        if betti_1[i + 1] > 0:
            b1_ratios.append(betti_1[i] / betti_1[i + 1])

    # φ proximity
    b0_phi_dist = [min(abs(r - PHI), abs(r - INV_PHI)) for r in b0_ratios]
    b1_phi_dist = [min(abs(r - PHI), abs(r - INV_PHI)) for r in b1_ratios]

    return {
        'scales': scales,
        'betti_0': betti_0,
        'betti_1': betti_1,
        'b0_ratios': b0_ratios,
        'b1_ratios': b1_ratios,
        'b0_mean_phi_distance': float(np.mean(b0_phi_dist)) if b0_phi_dist else float('nan'),
        'b1_mean_phi_distance': float(np.mean(b1_phi_dist)) if b1_phi_dist else float('nan'),
        'b0_phi_detected': np.mean(b0_phi_dist) < 0.3 if b0_phi_dist else False,
        'b1_phi_detected': np.mean(b1_phi_dist) < 0.3 if b1_phi_dist else False
    }


# =============================================================================
# Main Analysis
# =============================================================================

def run_topological_analysis(
    variant: str,
    iteration: int,
    max_bits: int = 100000,
    verbose: bool = True,
    output_path: Optional[str] = None
) -> Dict:
    """
    Run complete topological persistence analysis.

    Args:
        variant: Variant code (B, F, etc.)
        iteration: Iteration number
        max_bits: Maximum bits to analyze
        verbose: Print progress
        output_path: Optional path to save JSON results

    Returns:
        Complete topological analysis results
    """
    # Data requirements
    if verbose:
        print(f"\n{'='*65}")
        print(f"🔮 TOPOLOGICAL PERSISTENCE ANALYSIS")
        print(f"{'='*65}")
        print(f"\n📋 REQUIREMENTS:")
        print(f"   - phi_snapshots data for variant {variant}, iteration {iteration}")
        print(f"   - Observable or structural format supported")
        print(f"   - numpy (standard dependency)")
        print()

    # Load data
    data_dir = f"results/level0/phi_snapshots/var_{variant}"
    if not os.path.exists(data_dir):
        data_dir = f"results/level0/var_{variant}"

    if not os.path.exists(data_dir):
        print(f"\n❌ DATA NOT FOUND")
        print(f"   Required: {data_dir}")
        print(f"\n   📋 Generate data first:")
        print(f"      python level0_generate.py --variant {variant} --iterations {iteration}")
        return {'error': f'Data directory not found: {data_dir}'}

    if verbose:
        log_step(f"Loading variant {variant}, iteration {iteration}...")

    try:
        _, phi_observable, metadata = load_phi_for_level1(
            data_dir, iteration,
            return_structural=False,
            return_observable=True,
            return_metadata=True
        )
    except Exception as e:
        print(f"\n❌ FAILED TO LOAD DATA: {e}")
        return {'error': str(e)}

    bits = phi_observable[:max_bits]

    if verbose:
        log_step(f"Loaded {len(phi_observable):,} bits, analyzing {len(bits):,}")

    results = {
        'variant': variant,
        'iteration': iteration,
        'total_bits': len(phi_observable),
        'analyzed_bits': len(bits),
        'phi_target': PHI
    }

    # 1. Run length persistence
    if verbose:
        print(f"\n{'─'*50}")
        log_step("📊 STEP 1: Run Length Persistence Analysis")

    runs_0, runs_1 = compute_run_lengths(bits)

    if verbose:
        log_step(f"Found {len(runs_0):,} hole-runs (0s), {len(runs_1):,} matter-runs (1s)", 1)

    # Persistence diagrams
    diagram_0 = compute_persistence_diagram(runs_0, "H0_holes")
    diagram_1 = compute_persistence_diagram(runs_1, "H0_matter")

    # Analyze for φ
    if verbose:
        log_step("Analyzing hole-run persistence for φ...", 1)
    results['holes_persistence'] = analyze_persistence_ratios(diagram_0, verbose)

    if verbose:
        log_step("Analyzing matter-run persistence for φ...", 1)
    results['matter_persistence'] = analyze_persistence_ratios(diagram_1, verbose)

    # 2. Multi-scale Betti analysis
    if verbose:
        print(f"\n{'─'*50}")
        log_step("📊 STEP 2: Multi-Scale Betti Numbers")

    # Use φ-related scales
    phi_scales = []
    scale = 2
    while scale < len(bits) // 20:
        phi_scales.append(int(scale))
        scale = int(scale * PHI)

    if verbose:
        log_step(f"φ-scales: {phi_scales[:10]}...", 1)

    results['betti_analysis'] = compute_betti_at_scales(bits, phi_scales, verbose)

    # 3. Summary
    if verbose:
        print(f"\n{'='*65}")
        print(f"📊 SUMMARY")
        print(f"{'='*65}")

        hp = results.get('holes_persistence', {})
        mp = results.get('matter_persistence', {})
        ba = results.get('betti_analysis', {})

        holes_status = "✅ DETECTED" if hp.get('phi_detected') else "❌ Not detected"
        matter_status = "✅ DETECTED" if mp.get('phi_detected') else "❌ Not detected"
        b0_status = "✅ DETECTED" if ba.get('b0_phi_detected') else "❌ Not detected"
        b1_status = "✅ DETECTED" if ba.get('b1_phi_detected') else "❌ Not detected"

        print(f"   Holes persistence φ:  {holes_status} (dist={hp.get('mean_phi_distance', 'N/A'):.3f})")
        print(f"   Matter persistence φ: {matter_status} (dist={mp.get('mean_phi_distance', 'N/A'):.3f})")
        print(f"   Betti-0 φ-scaling:    {b0_status} (dist={ba.get('b0_mean_phi_distance', float('nan')):.3f})")
        print(f"   Betti-1 φ-scaling:    {b1_status} (dist={ba.get('b1_mean_phi_distance', float('nan')):.3f})")

    # Save results
    if output_path:
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(x) for x in obj]
            return obj

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=2)
        if verbose:
            print(f"\n📁 Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Topological Persistence Analysis — Find φ in structural holes"
    )
    parser.add_argument("--variant", "-v", default="B", help="Variant (B, D, E, F, G, H, I)")
    parser.add_argument("--iteration", "-i", type=int, default=15, help="Iteration number")
    parser.add_argument("--max-bits", "-m", type=int, default=100000, help="Max bits to analyze")
    parser.add_argument("--compare", "-c", type=str, default=None, help="Compare with another variant")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    results = run_topological_analysis(
        args.variant, args.iteration,
        max_bits=args.max_bits,
        output_path=args.output
    )

    if args.compare and 'error' not in results:
        print(f"\n{'='*65}")
        print(f"📊 COMPARISON: {args.variant} vs {args.compare}")
        print(f"{'='*65}")

        results_cmp = run_topological_analysis(
            args.compare, args.iteration,
            max_bits=args.max_bits
        )

        if 'error' not in results_cmp:
            print(f"\n   Metric                         {args.variant:>10}  {args.compare:>10}")
            print(f"   {'-'*50}")

            r1_hp = results.get('holes_persistence', {})
            r2_hp = results_cmp.get('holes_persistence', {})
            r1_ba = results.get('betti_analysis', {})
            r2_ba = results_cmp.get('betti_analysis', {})

            print(f"   Holes φ-distance              {r1_hp.get('mean_phi_distance', float('nan')):>10.3f}  {r2_hp.get('mean_phi_distance', float('nan')):>10.3f}")
            print(f"   Betti-0 φ-distance            {r1_ba.get('b0_mean_phi_distance', float('nan')):>10.3f}  {r2_ba.get('b0_mean_phi_distance', float('nan')):>10.3f}")


if __name__ == "__main__":
    main()

