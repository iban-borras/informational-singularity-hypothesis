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
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

# Numba for JIT compilation
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

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

@njit(cache=True)
def _compute_run_lengths_numba(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Numba-optimized run length computation.

    Returns: (runs_0_arr, runs_1_arr, n_runs_0, n_runs_1)
    Pre-allocates arrays and returns actual counts.
    """
    n = len(arr)
    if n == 0:
        return np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32), 0, 0

    # Pre-allocate (worst case: alternating bits = n/2 runs each)
    max_runs = n // 2 + 1
    runs_0 = np.zeros(max_runs, dtype=np.int32)
    runs_1 = np.zeros(max_runs, dtype=np.int32)
    n_runs_0 = 0
    n_runs_1 = 0

    current_val = arr[0]
    current_len = 1

    for i in range(1, n):
        if arr[i] == current_val:
            current_len += 1
        else:
            if current_val == 0:
                runs_0[n_runs_0] = current_len
                n_runs_0 += 1
            else:
                runs_1[n_runs_1] = current_len
                n_runs_1 += 1
            current_val = arr[i]
            current_len = 1

    # Last run
    if current_val == 0:
        runs_0[n_runs_0] = current_len
        n_runs_0 += 1
    else:
        runs_1[n_runs_1] = current_len
        n_runs_1 += 1

    return runs_0, runs_1, n_runs_0, n_runs_1


def compute_run_lengths(bits: str) -> Tuple[List[int], List[int]]:
    """
    Compute run lengths of 0s and 1s.

    These represent "holes" (0-runs) and "matter" (1-runs) in the sequence.
    Uses Numba JIT for performance on large sequences.
    """
    if not bits:
        return [], []

    # Convert to numpy array for Numba
    arr = np.frombuffer(bits.encode('ascii'), dtype=np.uint8) - ord('0')

    # Use Numba-optimized version
    runs_0_arr, runs_1_arr, n0, n1 = _compute_run_lengths_numba(arr)

    return runs_0_arr[:n0].tolist(), runs_1_arr[:n1].tolist()


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

@njit(cache=True)
def _coarse_grain_and_count_numba(arr: np.ndarray, scale: int) -> Tuple[int, int]:
    """
    Numba-optimized coarse-graining and counting.

    Returns: (n_components, n_alternations)
    """
    n = len(arr)
    n_blocks = n // scale

    if n_blocks < 2:
        return 0, 0

    # Coarse-grain: majority vote per block
    prev_val = -1
    n_components = 0
    n_alternations = 0
    in_component = False

    for i in range(n_blocks):
        # Count ones in block
        ones = 0
        start = i * scale
        end = start + scale
        for j in range(start, end):
            ones += arr[j]

        # Majority vote
        current_val = 1 if ones > scale // 2 else 0

        # Count alternations
        if prev_val >= 0 and current_val != prev_val:
            n_alternations += 1

        # Count components (1-runs)
        if current_val == 1:
            if not in_component:
                n_components += 1
                in_component = True
        else:
            in_component = False

        prev_val = current_val

    return n_components, n_alternations


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

    # Convert to numpy array once
    arr = np.frombuffer(bits.encode('ascii'), dtype=np.uint8) - ord('0')

    betti_0 = []  # Connected components (1-runs after coarse-graining)
    betti_1 = []  # Cycles proxy (alternation frequency)

    # Use tqdm if available
    if verbose and TQDM_AVAILABLE:
        scale_iter = tqdm(scales, desc="   Betti scales", unit="scale",
                          bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        scale_iter = scales

    for idx, scale in enumerate(scale_iter):
        if not TQDM_AVAILABLE and verbose:
            log_progress(idx + 1, len(scales), "Scales")

        # Use Numba-optimized version
        n_comp, n_alt = _coarse_grain_and_count_numba(arr, scale)
        betti_0.append(n_comp)
        betti_1.append(n_alt)

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
    start_time = time.time()

    # Data requirements
    if verbose:
        print(f"\n{'='*65}")
        print(f"🔮 TOPOLOGICAL PERSISTENCE ANALYSIS")
        print(f"{'='*65}")
        print(f"\n📋 CONFIGURATION:")
        print(f"   - Variant: {variant}, Iteration: {iteration}")
        print(f"   - Max bits: {max_bits:,}")
        print(f"   - JIT acceleration: {'Numba ✅' if NUMBA_AVAILABLE else 'Not available ❌'}")
        print(f"   - Progress bars: {'tqdm ✅' if TQDM_AVAILABLE else 'Basic'}")
        print()

    # Load data using streaming loader if available
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
        log_step(f"📥 Loading variant {variant}, iteration {iteration}...")

    # Try streaming loader first (memory efficient)
    struct_path = Path(data_dir) / f"phi_iter{iteration}.struct.gz"

    try:
        if struct_path.exists():
            from utils.streaming_phi_loader import load_phi_for_agents
            phi_observable, _ = load_phi_for_agents(
                str(struct_path),
                max_chars=max_bits * 2,  # Safety margin for parentheses
                observable_only=True,
                show_progress=verbose
            )
            bits = phi_observable[:max_bits]
        else:
            # Fallback to standard loader
            _, phi_observable, metadata = load_phi_for_level1(
                data_dir, iteration,
                return_structural=False,
                return_observable=True,
                return_metadata=True
            )
            bits = phi_observable[:max_bits]
    except Exception as e:
        print(f"\n❌ FAILED TO LOAD DATA: {e}")
        return {'error': str(e)}

    load_time = time.time() - start_time
    if verbose:
        log_step(f"✅ Loaded {len(bits):,} bits in {load_time:.1f}s")

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

        total_time = time.time() - start_time
        print(f"\n   ⏱️ Total time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # Save results
    if output_path:
        def convert(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(x) for x in obj]
            return obj

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file (auto-generated if not specified)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")

    args = parser.parse_args()

    # Auto-generate output path if not specified
    output_path = args.output
    if not args.no_save and output_path is None:
        output_dir = Path("results/level1/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        bits_suffix = f"{args.max_bits // 1_000_000}M" if args.max_bits >= 1_000_000 else f"{args.max_bits // 1000}K"
        output_path = str(output_dir / f"topological_var_{args.variant}_iter{args.iteration}_{bits_suffix}.json")

    results = run_topological_analysis(
        args.variant, args.iteration,
        max_bits=args.max_bits,
        output_path=output_path
    )

    if args.compare and 'error' not in results:
        print(f"\n{'='*65}")
        print(f"📊 COMPARISON: {args.variant} vs {args.compare}")
        print(f"{'='*65}")

        # Auto-generate output path for comparison variant
        cmp_output_path = None
        if not args.no_save:
            output_dir = Path("results/level1/analysis")
            bits_suffix = f"{args.max_bits // 1_000_000}M" if args.max_bits >= 1_000_000 else f"{args.max_bits // 1000}K"
            cmp_output_path = str(output_dir / f"topological_var_{args.compare}_iter{args.iteration}_{bits_suffix}.json")

        results_cmp = run_topological_analysis(
            args.compare, args.iteration,
            max_bits=args.max_bits,
            output_path=cmp_output_path
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

