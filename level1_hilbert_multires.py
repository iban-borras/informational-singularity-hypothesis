#!/usr/bin/env python3
"""
Level 1 — Hilbert Multi-Resolution Comparison (HMRC)

Compares Hilbert curve visualizations at multiple resolutions to find
self-similar φ patterns across scales.

The hypothesis: Feature sizes may scale by φ across resolutions,
revealing hidden order in the apparent turbulence.

Usage:
    python level1_hilbert_multires.py --variant B --iteration 15
    python level1_hilbert_multires.py --variant B --iteration 15 --orders 10,11,12
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


def hilbert_d2xy(n: int, d: int) -> Tuple[int, int]:
    """Convert Hilbert index d to (x, y) coordinates for n×n grid."""
    x = y = 0
    s = 1
    while s < n:
        rx = 1 & (d // 2)
        ry = 1 & (d ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d //= 4
        s *= 2
    return x, y


def bits_to_hilbert_grid(bits: str, order: int) -> np.ndarray:
    """
    Convert bit string to Hilbert curve grid.
    
    Args:
        bits: Binary string
        order: Hilbert curve order (grid will be 2^order × 2^order)
    
    Returns:
        2D numpy array with density values
    """
    n = 2 ** order
    required_bits = n * n
    
    if len(bits) < required_bits:
        bits = bits + '0' * (required_bits - len(bits))
    else:
        bits = bits[:required_bits]
    
    grid = np.zeros((n, n), dtype=np.float32)
    
    for d, bit in enumerate(bits):
        x, y = hilbert_d2xy(n, d)
        grid[y, x] = int(bit)
    
    return grid


def calculate_grid_features(grid: np.ndarray, verbose: bool = False) -> Dict:
    """Calculate features from a Hilbert grid."""
    n = grid.shape[0]
    
    if verbose:
        print(f"      Calculating features for {n}×{n} grid...", end=" ", flush=True)
    
    # Basic stats
    density = float(np.mean(grid))
    
    # Block densities at different sub-scales
    block_sizes = [2, 4, 8, 16] if n >= 16 else [2, 4]
    block_densities = {}
    
    for bs in block_sizes:
        if n % bs != 0:
            continue
        n_blocks = n // bs
        blocks = grid.reshape(n_blocks, bs, n_blocks, bs).mean(axis=(1, 3))
        block_densities[bs] = {
            'mean': float(np.mean(blocks)),
            'std': float(np.std(blocks)),
            'max': float(np.max(blocks)),
            'min': float(np.min(blocks))
        }
    
    # Spatial autocorrelation (Moran's I approximation)
    grid_shifted_h = np.roll(grid, 1, axis=1)
    grid_shifted_v = np.roll(grid, 1, axis=0)
    spatial_corr_h = float(np.corrcoef(grid.flatten(), grid_shifted_h.flatten())[0, 1])
    spatial_corr_v = float(np.corrcoef(grid.flatten(), grid_shifted_v.flatten())[0, 1])
    
    if verbose:
        print("✓")
    
    return {
        'grid_size': n,
        'total_bits': n * n,
        'density': density,
        'block_densities': block_densities,
        'spatial_correlation': {
            'horizontal': spatial_corr_h,
            'vertical': spatial_corr_v,
            'mean': (spatial_corr_h + spatial_corr_v) / 2
        }
    }


def analyze_multi_resolution(
    bits: str,
    orders: Optional[List[int]] = None,
    verbose: bool = True
) -> Dict:
    """
    Analyze Hilbert representations at multiple resolutions.
    
    Args:
        bits: Observable bit string
        orders: List of Hilbert curve orders to analyze
        verbose: Print progress
    
    Returns:
        Analysis results including cross-scale ratios
    """
    if orders is None:
        # Default: orders that fit in available data
        max_order = int(math.log2(math.sqrt(len(bits))))
        orders = list(range(max(4, max_order - 3), max_order + 1))
    
    if verbose:
        print(f"\n🗺️  Hilbert Multi-Resolution Comparison (HMRC)")
        print(f"   Data size: {len(bits):,} bits")
        print(f"   Orders to analyze: {orders}")
        print(f"   Target: φ = {PHI:.6f}")
    
    results_by_order = {}
    total_orders = len(orders)
    
    for idx, order in enumerate(orders):
        required = 4 ** order
        if required > len(bits):
            if verbose:
                print(f"   [{idx+1}/{total_orders}] Order {order}: ✗ (needs {required:,} bits)")
            continue
        
        if verbose:
            print(f"   [{idx+1}/{total_orders}] Order {order} ({4**order:,} bits)...")
        
        grid = bits_to_hilbert_grid(bits, order)
        features = calculate_grid_features(grid, verbose=verbose)
        results_by_order[order] = features
    
    # Calculate cross-scale ratios
    if verbose:
        print(f"\n   Calculating cross-scale ratios...")
    
    cross_scale = calculate_cross_scale_ratios(results_by_order, orders)
    
    return {
        'orders': orders,
        'results_by_order': results_by_order,
        'cross_scale': cross_scale,
        'phi_target': PHI,
        'data_size': len(bits)
    }


def calculate_cross_scale_ratios(
    results_by_order: Dict[int, Dict],
    orders: List[int]
) -> Dict:
    """Calculate ratios between consecutive resolutions."""
    valid_orders = [o for o in orders if o in results_by_order]

    ratios = {
        'density_ratios': [],
        'std_ratios': [],
        'spatial_corr_ratios': [],
        'order_pairs': []
    }

    for i in range(len(valid_orders) - 1):
        o1, o2 = valid_orders[i], valid_orders[i + 1]
        r1, r2 = results_by_order[o1], results_by_order[o2]

        ratios['order_pairs'].append(f"{o2}/{o1}")

        # Density ratio (should be ~1 for balanced data)
        if r1['density'] > 0:
            ratios['density_ratios'].append(r2['density'] / r1['density'])

        # Block std ratio at common block size
        common_bs = 4  # 4×4 blocks
        if common_bs in r1.get('block_densities', {}) and common_bs in r2.get('block_densities', {}):
            std1 = r1['block_densities'][common_bs]['std']
            std2 = r2['block_densities'][common_bs]['std']
            if std1 > 0:
                ratios['std_ratios'].append(std2 / std1)

        # Spatial correlation ratio
        sc1 = r1['spatial_correlation']['mean']
        sc2 = r2['spatial_correlation']['mean']
        if abs(sc1) > 0.001:
            ratios['spatial_corr_ratios'].append(sc2 / sc1)

    # Calculate φ proximity
    all_ratios = ratios['density_ratios'] + ratios['std_ratios'] + ratios['spatial_corr_ratios']
    valid_ratios = [r for r in all_ratios if not math.isnan(r) and r > 0]

    if valid_ratios:
        distances = [abs(r - PHI) for r in valid_ratios]
        ratios['phi_distance_mean'] = float(np.mean(distances))
        ratios['phi_distance_min'] = float(min(distances))
        ratios['closest_ratio'] = valid_ratios[distances.index(min(distances))]
    else:
        ratios['phi_distance_mean'] = float('nan')
        ratios['phi_distance_min'] = float('nan')
        ratios['closest_ratio'] = float('nan')

    return ratios


def print_results(results: Dict) -> None:
    """Print analysis results."""
    print(f"\n{'='*60}")
    print(f"🗺️  HILBERT MULTI-RESOLUTION — Hidden φ Discovery")
    print(f"{'='*60}")
    print(f"   φ target: {results['phi_target']:.6f}")
    print(f"   Data: {results['data_size']:,} bits")

    print(f"\n📐 Resolution Analysis:")
    for order, data in results['results_by_order'].items():
        n = data['grid_size']
        print(f"   Order {order} ({n}×{n} = {n*n:,} bits):")
        print(f"      Density: {data['density']:.4f}")
        print(f"      Spatial corr: {data['spatial_correlation']['mean']:.4f}")

    cs = results['cross_scale']
    print(f"\n📈 Cross-Scale Ratios (looking for φ ≈ 1.618):")
    print(f"   Order pairs: {cs['order_pairs']}")

    if cs['density_ratios']:
        print(f"   Density ratios: {[f'{r:.3f}' for r in cs['density_ratios']]}")
    if cs['std_ratios']:
        print(f"   Std ratios: {[f'{r:.3f}' for r in cs['std_ratios']]}")
    if cs['spatial_corr_ratios']:
        print(f"   Spatial corr ratios: {[f'{r:.3f}' for r in cs['spatial_corr_ratios']]}")

    print(f"\n🎯 φ-Alignment Assessment:")
    dist = cs['phi_distance_mean']
    if not math.isnan(dist):
        print(f"   Mean φ-distance: {dist:.4f}")
        print(f"   Min φ-distance: {cs['phi_distance_min']:.4f}")
        print(f"   Closest ratio to φ: {cs['closest_ratio']:.4f}")

        if dist < 0.1:
            print(f"   ✅ STRONG φ signal in multi-resolution structure!")
        elif dist < 0.3:
            print(f"   ⚠️  Moderate φ signal")
        else:
            print(f"   ❌ Weak φ signal")
    else:
        print(f"   ❌ Insufficient data for φ analysis")


def main():
    parser = argparse.ArgumentParser(
        description="Hilbert Multi-Resolution Comparison — Find hidden φ across scales"
    )
    parser.add_argument("--variant", "-v", default="B", help="Variant (B, D, E, F, G, H, I)")
    parser.add_argument("--iteration", "-i", type=int, default=15, help="Iteration number")
    parser.add_argument("--max-bits", "-m", type=int, default=None, help="Max bits to analyze")
    parser.add_argument("--orders", "-r", type=str, default=None,
                        help="Comma-separated Hilbert orders (e.g., '8,9,10,11')")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    # Determine data directory
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

    if args.max_bits and args.max_bits < len(phi_observable):
        phi_observable = phi_observable[:args.max_bits]
        print(f"   Truncated to {args.max_bits:,} bits")

    # Parse orders
    orders = None
    if args.orders:
        orders = [int(o) for o in args.orders.split(",")]

    # Run analysis
    results = analyze_multi_resolution(phi_observable, orders=orders)
    results['variant'] = args.variant
    results['iteration'] = args.iteration

    print_results(results)

    if args.output:
        def convert_for_json(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(x) for x in obj]
            return obj

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(convert_for_json(results), f, indent=2)
        print(f"\n📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main()

