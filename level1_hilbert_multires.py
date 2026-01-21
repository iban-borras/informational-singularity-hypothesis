#!/usr/bin/env python3
"""
Level 1 — Hilbert Multi-Resolution Comparison (HMRC)

Compares Hilbert curve visualizations at multiple resolutions to find
self-similar φ patterns across scales.

The hypothesis: Feature sizes may scale by φ across resolutions,
revealing hidden order in the apparent turbulence.

OPTIMIZED VERSION:
- Numba JIT compilation for 50-100x speedup on core functions
- Streaming data loading (only reads bits needed for each order)
- Multiprocessing for parallel order analysis
- Publication-quality figure generation

Usage:
    python level1_hilbert_multires.py --variant B --iteration 15
    python level1_hilbert_multires.py --variant B --iteration 15 --orders 10,11,12
    python level1_hilbert_multires.py -v B -i 23 --plot --max-cpu 50
"""

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Numba JIT compilation (with fallback)
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

sys.path.insert(0, str(Path(__file__).parent))

PHI = (1 + math.sqrt(5)) / 2


# =============================================================================
# NUMBA-OPTIMIZED HILBERT FUNCTIONS
# =============================================================================

@njit(cache=True)
def hilbert_d2xy_fast(n: int, d: int) -> Tuple[int, int]:
    """Convert Hilbert index d to (x, y) coordinates for n×n grid. Numba-optimized."""
    x = 0
    y = 0
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


@njit(parallel=True, cache=True)
def bits_to_hilbert_grid_fast(bits_array: np.ndarray, order: int) -> np.ndarray:
    """
    Convert bit array to Hilbert curve grid. Numba-optimized with parallelization.

    Args:
        bits_array: 1D numpy array of uint8 (0 or 1)
        order: Hilbert curve order (grid will be 2^order × 2^order)

    Returns:
        2D numpy array with bit values mapped via Hilbert curve
    """
    n = 2 ** order
    grid = np.zeros((n, n), dtype=np.float32)
    num_bits = min(len(bits_array), n * n)

    for d in prange(num_bits):
        x, y = hilbert_d2xy_fast(n, d)
        grid[y, x] = bits_array[d]

    return grid


# Legacy function for compatibility
def hilbert_d2xy(n: int, d: int) -> Tuple[int, int]:
    """Convert Hilbert index d to (x, y) coordinates for n×n grid."""
    return hilbert_d2xy_fast(n, d)


def bits_to_hilbert_grid(bits: str, order: int) -> np.ndarray:
    """
    Convert bit string to Hilbert curve grid. Uses Numba-optimized version.

    Args:
        bits: Binary string
        order: Hilbert curve order (grid will be 2^order × 2^order)

    Returns:
        2D numpy array with density values
    """
    n = 2 ** order
    required_bits = n * n

    # Convert string to numpy array for Numba
    if len(bits) < required_bits:
        # Pad with zeros
        bits_array = np.zeros(required_bits, dtype=np.uint8)
        for i, c in enumerate(bits):
            bits_array[i] = ord(c) - ord('0')
    else:
        # Use frombuffer for efficiency (works for ASCII 0/1)
        bits_array = np.frombuffer(bits[:required_bits].encode('ascii'), dtype=np.uint8) - ord('0')

    # Use Numba-optimized function
    return bits_to_hilbert_grid_fast(bits_array, order)


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


# =============================================================================
# STREAMING DATA LOADER
# =============================================================================

def stream_bits_for_order(data_dir: str, iteration: int, order: int) -> Optional[str]:
    """
    Stream only the bits needed for a specific Hilbert order.

    Args:
        data_dir: Path to variant data directory
        iteration: Iteration number
        order: Hilbert curve order (needs 4^order bits)

    Returns:
        Bit string of exactly 4^order bits, or None if not enough data
    """
    required_bits = 4 ** order

    # Try to find the structural file
    struct_path = Path(data_dir) / f"phi_iter{iteration}.struct.gz"

    if struct_path.exists():
        try:
            from utils.bitarray_encoder import stream_phi_prefix_gz
            bits = stream_phi_prefix_gz(str(struct_path), required_bits, clean=True)
            if len(bits) >= required_bits:
                return bits[:required_bits]
        except Exception as e:
            print(f"      Warning: streaming failed: {e}")

    return None


def analyze_single_order(args: Tuple) -> Tuple[int, Optional[Dict]]:
    """
    Analyze a single Hilbert order. Designed for multiprocessing.

    Args:
        args: Tuple of (order, bits_or_path, is_path, data_dir, iteration)

    Returns:
        Tuple of (order, features_dict or None)
    """
    order, bits_or_path, is_streaming, data_dir, iteration = args

    try:
        if is_streaming:
            # Stream bits for this specific order
            bits = stream_bits_for_order(data_dir, iteration, order)
            if bits is None:
                return order, None
        else:
            bits = bits_or_path

        required = 4 ** order
        if len(bits) < required:
            return order, None

        grid = bits_to_hilbert_grid(bits, order)
        features = calculate_grid_features(grid, verbose=False)
        features['grid'] = grid  # Keep for visualization
        return order, features

    except Exception as e:
        print(f"      Error processing order {order}: {e}")
        return order, None


def analyze_multi_resolution_parallel(
    data_dir: str,
    iteration: int,
    orders: Optional[List[int]] = None,
    max_workers: int = 4,
    max_bits: Optional[int] = None,
    verbose: bool = True
) -> Dict:
    """
    Analyze Hilbert representations at multiple resolutions using parallel processing.

    Args:
        data_dir: Path to variant data directory
        iteration: Iteration number
        orders: List of Hilbert curve orders to analyze
        max_workers: Maximum parallel workers
        max_bits: Maximum bits to load (for non-streaming mode)
        verbose: Print progress

    Returns:
        Analysis results including cross-scale ratios and grids
    """
    # Determine available data size first
    struct_path = Path(data_dir) / f"phi_iter{iteration}.struct.gz"

    if not struct_path.exists():
        raise FileNotFoundError(f"Data file not found: {struct_path}")

    # Get approximate file size to estimate bits
    file_size = struct_path.stat().st_size
    # Structural format: ~2 bits per char, gzip ~10x compression
    estimated_bits = file_size * 10 * 4  # Very rough estimate

    if max_bits:
        available_bits = min(max_bits, estimated_bits)
    else:
        available_bits = estimated_bits

    # Determine orders to analyze
    if orders is None:
        max_order = int(math.log2(math.sqrt(available_bits)))
        orders = list(range(max(6, max_order - 4), max_order + 1))

    if verbose:
        print(f"\n🗺️  Hilbert Multi-Resolution Comparison (HMRC) — PARALLEL")
        print(f"   {'Numba JIT:':16} {'✅ ENABLED' if HAS_NUMBA else '❌ DISABLED (install numba for 50x speedup)'}")
        print(f"   {'Workers:':16} {max_workers}")
        print(f"   {'Orders:':16} {orders}")
        print(f"   {'Target φ:':16} {PHI:.6f}")

    results_by_order = {}
    grids = {}

    # Use streaming mode: each worker loads only its needed bits
    if verbose:
        print(f"\n   Processing orders in parallel (streaming mode)...")

    # Prepare tasks
    tasks = [(order, None, True, data_dir, iteration) for order in orders]

    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_single_order, task): task[0] for task in tasks}

        for future in as_completed(futures):
            order = futures[future]
            try:
                result_order, features = future.result()
                if features:
                    grid = features.pop('grid', None)
                    results_by_order[result_order] = features
                    if grid is not None:
                        grids[result_order] = grid
                    if verbose:
                        print(f"      Order {result_order}: ✅ ({4**result_order:,} bits)")
                else:
                    if verbose:
                        print(f"      Order {result_order}: ❌ (insufficient data)")
            except Exception as e:
                if verbose:
                    print(f"      Order {order}: ❌ Error: {e}")

    # Calculate cross-scale ratios
    if verbose:
        print(f"\n   Calculating cross-scale ratios...")

    cross_scale = calculate_cross_scale_ratios(results_by_order, orders)

    return {
        'orders': orders,
        'results_by_order': results_by_order,
        'cross_scale': cross_scale,
        'phi_target': PHI,
        'grids': grids  # For visualization
    }


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def generate_hilbert_figures(
    results: Dict,
    variant: str,
    iteration: int,
    output_dir: Path,
    fmt: str = 'png',
    dpi: int = 300
) -> List[str]:
    """
    Generate publication-quality Hilbert visualization figures.

    Args:
        results: Analysis results with grids
        variant: Variant code
        iteration: Iteration number
        output_dir: Output directory
        fmt: Image format (png, pdf, svg)
        dpi: Resolution

    Returns:
        List of generated filenames
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("      ⚠️ matplotlib not available, skipping figures")
        return []

    generated = []
    grids = results.get('grids', {})

    if not grids:
        print("      ⚠️ No grids available for visualization")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Color scheme
    cmap = plt.cm.viridis

    # 1. Individual Hilbert curves for each order
    for order, grid in sorted(grids.items()):
        fig, ax = plt.subplots(figsize=(8, 8))

        im = ax.imshow(grid, cmap=cmap, interpolation='nearest')
        ax.set_title(f'Hilbert Curve — Variant {variant}, Iter {iteration}\n'
                     f'Order {order} ({grid.shape[0]}×{grid.shape[0]} = {grid.size:,} bits)',
                     fontsize=12)
        ax.axis('off')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Bit Value', fontsize=10)

        filename = f'hilbert_{variant}_iter{iteration}_order{order}.{fmt}'
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        generated.append(filename)

    # 2. Multi-resolution comparison (all orders side by side)
    if len(grids) >= 2:
        n_grids = len(grids)
        fig, axes = plt.subplots(1, n_grids, figsize=(4 * n_grids, 4))

        if n_grids == 1:
            axes = [axes]

        for ax, (order, grid) in zip(axes, sorted(grids.items())):
            ax.imshow(grid, cmap=cmap, interpolation='nearest')
            ax.set_title(f'Order {order}\n{grid.shape[0]}×{grid.shape[0]}', fontsize=10)
            ax.axis('off')

        fig.suptitle(f'Hilbert Multi-Resolution — Variant {variant}, Iter {iteration}\n'
                     f'Looking for φ ≈ {PHI:.3f} scaling', fontsize=12)
        plt.tight_layout()

        filename = f'hilbert_multires_{variant}_iter{iteration}.{fmt}'
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        generated.append(filename)

    # 3. φ-distance analysis chart
    cross_scale = results.get('cross_scale', {})
    if cross_scale.get('order_pairs'):
        fig, ax = plt.subplots(figsize=(10, 6))

        pairs = cross_scale['order_pairs']
        x = range(len(pairs))

        # Plot different ratio types
        if cross_scale.get('std_ratios'):
            ax.plot(x[:len(cross_scale['std_ratios'])], cross_scale['std_ratios'],
                   'o-', label='Std Ratios', markersize=8)
        if cross_scale.get('spatial_corr_ratios'):
            ax.plot(x[:len(cross_scale['spatial_corr_ratios'])], cross_scale['spatial_corr_ratios'],
                   's-', label='Spatial Corr Ratios', markersize=8)

        # φ reference line
        ax.axhline(y=PHI, color='gold', linestyle='--', linewidth=2, label=f'φ = {PHI:.3f}')
        ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, label='1.0')

        ax.set_xlabel('Order Pair', fontsize=12)
        ax.set_ylabel('Ratio', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(pairs)
        ax.set_title(f'Cross-Scale Ratios — Variant {variant}, Iter {iteration}\n'
                     f'φ-distance: {cross_scale.get("phi_distance_mean", 0):.4f}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = f'hilbert_phi_ratios_{variant}_iter{iteration}.{fmt}'
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        generated.append(filename)

    return generated


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
        description="Hilbert Multi-Resolution Comparison — Find hidden φ across scales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python level1_hilbert_multires.py -v B -i 23 --plot
    python level1_hilbert_multires.py -v B -i 23 --orders 10,11,12,13 --plot
    python level1_hilbert_multires.py -v B -i 23 --parallel --max-cpu 50 --plot
        """
    )
    parser.add_argument("--variant", "-v", default="B", help="Variant (B, D, E, F, G, H, I)")
    parser.add_argument("--iteration", "-i", type=int, default=15, help="Iteration number")
    parser.add_argument("--max-bits", "-m", type=int, default=None,
                        help="Max bits to analyze (default: auto based on orders)")
    parser.add_argument("--orders", "-r", type=str, default=None,
                        help="Comma-separated Hilbert orders (e.g., '12,13,14,15'). "
                             "Order N creates a 2^N × 2^N grid requiring 4^N bits. "
                             "Recommended: 12-16 for large iterations (>100M bits), "
                             "10-14 for medium, 8-12 for small. Use consecutive orders "
                             "to detect φ-scaling between resolutions.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file")
    parser.add_argument("--plot", "-p", action="store_true", help="Generate visualization figures")
    parser.add_argument("--parallel", action="store_true",
                        help="Use parallel processing (recommended for large data)")
    parser.add_argument("--max-cpu", type=int, default=50,
                        help="Maximum CPU usage %% for parallel mode (default: 50)")
    parser.add_argument("--format", "-f", choices=['png', 'pdf', 'svg'], default='png',
                        help="Image format for plots (default: png)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for plots (default: 300)")
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy non-parallel mode (loads all data to RAM)")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"🗺️  HILBERT MULTI-RESOLUTION COMPARISON")
    print(f"{'='*60}")
    print(f"   Variant: {args.variant}, Iteration: {args.iteration}")

    # Determine data directory
    data_dir = f"results/level0/phi_snapshots/var_{args.variant}"
    if not os.path.exists(data_dir):
        data_dir = f"results/level0/var_{args.variant}"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found for variant {args.variant}")
        sys.exit(1)

    # Parse orders
    orders = None
    if args.orders:
        orders = [int(o) for o in args.orders.split(",")]

    # Determine output directory for figures
    from utils.file_saver import get_output_path, relative_path
    figures_dir = get_output_path(1, "figures", "hilbert")

    # Choose processing mode
    if args.legacy:
        # Legacy mode: load all data first
        print(f"\n🔬 Loading data (legacy mode)...")
        try:
            from level1.data_loader import load_phi_for_level1
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

        results = analyze_multi_resolution(phi_observable, orders=orders)
    else:
        # Optimized parallel mode with streaming
        max_workers = max(1, int(os.cpu_count() * args.max_cpu / 100))
        print(f"\n🔬 Analyzing with parallel streaming...")

        try:
            results = analyze_multi_resolution_parallel(
                data_dir, args.iteration,
                orders=orders,
                max_workers=max_workers,
                max_bits=args.max_bits,
                verbose=True
            )
        except FileNotFoundError as e:
            print(f"❌ {e}")
            sys.exit(1)

    results['variant'] = args.variant
    results['iteration'] = args.iteration

    print_results(results)

    # Generate figures if requested
    if args.plot:
        print(f"\n📊 Generating figures...")
        generated = generate_hilbert_figures(
            results, args.variant, args.iteration,
            figures_dir, fmt=args.format, dpi=args.dpi
        )
        if generated:
            print(f"   📁 Output: {relative_path(figures_dir)}")
            for fname in generated:
                print(f"      • {fname}")
        else:
            print(f"   ⚠️ No figures generated")

    # Save JSON results
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

        # Remove grids from JSON (too large)
        results_for_json = {k: v for k, v in results.items() if k != 'grids'}

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(convert_for_json(results_for_json), f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")

    print(f"\n{'='*60}")
    print(f"✅ Analysis complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

