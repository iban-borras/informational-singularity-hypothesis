#!/usr/bin/env python3
"""
🌀 HSI Level 2 - Takens Embedding & Topological Data Analysis (TDA)

Wrapper script that orchestrates local and global TDA analysis.

Modes:
- global (default): Landmark subsampling for large-scale topology (β₂ surfaces)
- local: Windowed analysis with parallelization (β₁ consistency)
- both: Run both analyses

Method:
1. Takens' Embedding: Convert 1D sequence to 3D point cloud using time delay
2. Persistent Homology: Calculate Betti numbers (β₀, β₁, β₂)

Scientific Value:
If we find persistent β₂ > 0, we've demonstrated that Level 1 generates
a REAL geometric surface (emergence of space).

Usage:
    python level2_takens_tda.py -v B -i 23 --mode global
    python level2_takens_tda.py -v B -i 23 --mode local --windows 10
    python level2_takens_tda.py -v B -i 23 --mode both
    python level2_takens_tda.py -v B F -i 23 --compare

Author: Sophia (for Iban & Antigravity)
Date: January 2026
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np

# Local imports
from utils.streaming_phi_loader import load_phi_for_agents
from level2 import (
    run_local_tda, run_global_tda, run_global_tda_blocks,
    run_witness_tda, run_cubical_tda, TDA_DIR
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Default parameters
DEFAULT_EMBEDDING_DIM = 3
DEFAULT_TIME_DELAY = 1
DEFAULT_MAX_LANDMARKS = 10_000  # For global TDA
DEFAULT_WINDOWS = 10  # For local TDA
DEFAULT_POINTS_PER_WINDOW = 5_000


# ═══════════════════════════════════════════════════════════════════════════════
# INTERPRETATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def interpret_global_betti(betti: Dict[str, int], verbose: bool = True) -> str:
    """Interpret Betti numbers from global TDA analysis."""
    log = print if verbose else lambda *a, **k: None

    log(f"\n   🔮 TOPOLOGICAL INTERPRETATION:")
    log(f"   {'─'*50}")

    beta_0 = betti.get('beta_0', 0)
    beta_1 = betti.get('beta_1', 0)
    beta_2 = betti.get('beta_2', 0)

    interpretations = []

    if beta_0 == 1:
        interp = "✓ Single connected component (unified structure)"
        interpretations.append(interp)
        log(f"      β₀=1: {interp}")
    elif beta_0 > 1:
        interp = f"⚠️ {beta_0} disconnected clusters (fragmented)"
        interpretations.append(interp)
        log(f"      β₀={beta_0}: {interp}")

    if beta_1 > 0:
        interp = f"🌀 {beta_1} persistent loop(s) detected (ORBITAL STRUCTURE)"
        interpretations.append(interp)
        log(f"      β₁={beta_1}: {interp}")
    else:
        interp = "No persistent 1-cycles (no recurring orbits)"
        interpretations.append(interp)
        log(f"      β₁=0: {interp}")

    if beta_2 > 0:
        interp = f"🎈 {beta_2} 2D cavity(ies) detected (EMERGENT SURFACE!)"
        interpretations.append(interp)
        log(f"      β₂={beta_2}: {interp}")
        log(f"\n      ⭐ THIS IS SIGNIFICANT: A closed surface has emerged!")
        log(f"         This could represent a 'holographic membrane'.")
    else:
        interp = "No 2D cavities (no enclosed surfaces)"
        interpretations.append(interp)
        log(f"      β₂=0: {interp}")

    # Overall assessment
    log(f"\n   📋 SUMMARY:")
    if beta_2 > 0:
        assessment = "🌟 STRONG GEOMETRIC EMERGENCE: Persistent 2D surface detected"
    elif beta_1 > 2:
        assessment = "🔶 MODERATE EMERGENCE: Multiple orbital structures detected"
    elif beta_1 > 0:
        assessment = "🔸 WEAK EMERGENCE: Some cyclic structure detected"
    else:
        assessment = "⚪ NO CLEAR GEOMETRY: Point cloud appears unstructured"

    log(f"      {assessment}")
    interpretations.append(f"ASSESSMENT: {assessment}")

    return "; ".join(interpretations)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_variant_tda(variant: str,
                        iteration: int,
                        mode: str = 'global',
                        method: str = 'rips',
                        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                        time_delay: int = DEFAULT_TIME_DELAY,
                        max_landmarks: int = DEFAULT_MAX_LANDMARKS,
                        landmark_method: str = 'uniform',
                        n_windows: int = DEFAULT_WINDOWS,
                        points_per_window: int = DEFAULT_POINTS_PER_WINDOW,
                        n_blocks: int = 0,
                        landmarks_per_block: int = 5000,
                        n_witnesses: int = 50000,
                        cubical_mode: str = 'hilbert',
                        hilbert_block_size: int = 64,
                        save_hilbert: bool = False,
                        backend: str = 'auto',
                        max_bits: int = 50_000_000,
                        visualize: bool = False,
                        verbose: bool = True) -> Dict[str, Any]:
    """
    Perform TDA analysis on a variant using the specified mode and method.

    Args:
        variant: Variant letter (B, D, E, etc.)
        iteration: Iteration number
        mode: Analysis mode ('global', 'local', or 'both')
        method: TDA method ('rips', 'witness', 'cubical')
        embedding_dim: Dimension for Takens embedding
        time_delay: Time delay for embedding
        max_landmarks: Maximum landmarks for global analysis
        landmark_method: 'uniform' or 'random' landmark selection
        n_windows: Number of windows for local analysis
        points_per_window: Points per window for local analysis
        n_blocks: Number of blocks for block processing (0=disabled)
        landmarks_per_block: Landmarks per block
        n_witnesses: Number of witnesses for witness complex
        cubical_mode: 'hilbert' (density gradients) or 'binary' (raw 0/1)
        hilbert_block_size: Bits per Hilbert cell (larger = less noise)
        save_hilbert: Save Hilbert image to PNG file
        backend: TDA backend ('auto', 'ripser', 'giotto', 'gudhi')
        max_bits: Maximum bits to load from sequence
        visualize: Generate 3D visualization
        verbose: Print progress

    Returns:
        Complete analysis results dictionary
    """
    log = print if verbose else lambda *a, **k: None

    print(f"\n{'═'*70}")
    print(f"🌀 HSI TAKENS/TDA ANALYSIS")
    print(f"{'═'*70}")
    print(f"\n📋 CONFIGURATION:")
    print(f"   Variant: {variant}")
    print(f"   Iteration: {iteration}")
    print(f"   Method: {method.upper()}")
    if method == 'rips':
        print(f"   Mode: {mode}")
        print(f"   Backend: {backend}")
    print(f"   Embedding dim: {embedding_dim}")
    print(f"   Time delay (τ): {time_delay}")
    if method == 'witness':
        print(f"   Landmarks: {max_landmarks:,}")
        print(f"   Witnesses: {n_witnesses:,}")
    elif method == 'cubical':
        print(f"   (Sequence converted to 2D image)")
    elif n_blocks > 0:
        print(f"   Block mode: {n_blocks} blocks x {landmarks_per_block:,} landmarks")
    elif mode in ('global', 'both'):
        print(f"   Max landmarks: {max_landmarks:,} ({landmark_method})")
    if method == 'rips' and mode in ('local', 'both'):
        print(f"   Windows: {n_windows} x {points_per_window:,} points")
    print(f"   Max bits: {max_bits:,}")
    print(f"{'─'*70}")

    t_start = time.perf_counter()

    # Find the struct file
    struct_path = Path(f"results/level0/phi_snapshots/var_{variant}/phi_iter{iteration}.struct.gz")

    if not struct_path.exists():
        log(f"   ❌ File not found: {struct_path}")
        return {'error': f'file not found: {struct_path}'}

    # Load sequence
    log(f"\n   📂 Loading Φ sequence (streaming)...")
    t_load = time.perf_counter()
    phi_observable, _ = load_phi_for_agents(
        str(struct_path),
        max_chars=max_bits * 2,  # Safety margin for parentheses
        observable_only=True,
        show_progress=verbose
    )
    load_time = time.perf_counter() - t_load

    if not phi_observable:
        log(f"   ❌ Failed to load sequence")
        return {'error': 'failed to load sequence'}

    # Limit to max_bits and convert to numpy array
    sequence = phi_observable[:max_bits]
    sequence_array = np.array([int(b) for b in sequence], dtype=np.float32)
    log(f"      ✓ Loaded {len(sequence_array):,} bits in {load_time:.2f}s")

    # Initialize results
    results = {
        'variant': variant,
        'iteration': iteration,
        'mode': mode,
        'timestamp': datetime.now().isoformat(),
        'sequence_length': len(sequence_array),
        'config': {
            'embedding_dim': embedding_dim,
            'time_delay': time_delay,
            'max_bits': max_bits
        }
    }

    # Run analyses based on method
    point_cloud = None  # For visualization

    # ═══════════════════════════════════════════════════════════════════════
    # SCALABLE METHODS (witness, cubical) - take precedence
    # ═══════════════════════════════════════════════════════════════════════

    if method == 'witness':
        witness_results = run_witness_tda(
            sequence=sequence_array,
            n_landmarks=max_landmarks,
            n_witnesses=n_witnesses,
            embedding_dim=embedding_dim,
            time_delay=time_delay,
            verbose=verbose
        )
        results['witness'] = witness_results

        # Interpret results
        if witness_results.get('betti'):
            interp = interpret_global_betti(witness_results['betti'], verbose=verbose)
            results['interpretation'] = interp

    elif method == 'cubical':
        cubical_results = run_cubical_tda(
            sequence=sequence_array,
            cubical_mode=cubical_mode,
            hilbert_block_size=hilbert_block_size,
            save_hilbert=save_hilbert,
            variant=variant,
            iteration=iteration,
            verbose=verbose
        )
        results['cubical'] = cubical_results

    # ═══════════════════════════════════════════════════════════════════════
    # RIPS-BASED METHODS (original)
    # ═══════════════════════════════════════════════════════════════════════

    elif n_blocks > 0:
        log(f"\n   🧱 Running BLOCK TDA analysis...")
        block_results = run_global_tda_blocks(
            sequence=sequence_array,
            n_blocks=n_blocks,
            landmarks_per_block=landmarks_per_block,
            embedding_dim=embedding_dim,
            time_delay=time_delay,
            backend=backend,
            verbose=verbose
        )
        results['blocks'] = block_results

        # Interpret aggregated results
        if block_results.get('beta_1'):
            log(f"\n   🔮 BLOCK ANALYSIS INTERPRETATION:")
            log(f"      β₁ consistency: {block_results['beta_1']['std']:.2f} std")
            if block_results['beta_1']['mean'] > 0:
                log(f"      ✓ Persistent cycles detected across blocks")

    elif mode in ('global', 'both'):
        log(f"\n   🌐 Running GLOBAL TDA analysis...")
        global_results = run_global_tda(
            sequence=sequence_array,
            n_landmarks=max_landmarks,
            embedding_dim=embedding_dim,
            time_delay=time_delay,
            landmark_method=landmark_method,
            backend=backend,
            return_point_cloud=visualize,  # Only if visualization requested
            verbose=verbose
        )

        # Extract point cloud for visualization (don't save to JSON)
        if visualize and 'point_cloud' in global_results:
            point_cloud = global_results.pop('point_cloud')

        results['global'] = global_results

        # Interpret global results
        if global_results.get('betti'):
            interp = interpret_global_betti(global_results['betti'], verbose=verbose)
            results['global_interpretation'] = interp

    if n_blocks == 0 and mode in ('local', 'both'):
        log(f"\n   🔬 Running LOCAL TDA analysis...")
        local_results = run_local_tda(
            sequence=sequence_array,
            n_windows=n_windows,
            embedding_dim=embedding_dim,
            time_delay=time_delay,
            max_points_per_window=points_per_window,
            verbose=verbose
        )
        results['local'] = local_results

    total_time = time.perf_counter() - t_start
    results['total_time_s'] = total_time

    # Save results
    TDA_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{mode}" if mode != 'global' else ""
    output_path = TDA_DIR / f"tda_{variant}_iter{iteration}{suffix}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    log(f"\n   💾 Results saved to: {output_path}")

    # Generate 3D visualization if requested
    if visualize and point_cloud is not None and embedding_dim >= 3:
        plot_path = generate_3d_visualization(point_cloud, variant, iteration, TDA_DIR)
        if plot_path:
            results['visualization'] = str(plot_path)
            log(f"   📊 3D plot saved to: {plot_path}")

    log(f"   ⏱️ Total time: {total_time:.2f}s")
    print(f"{'═'*70}\n")

    return results


def generate_3d_visualization(point_cloud: np.ndarray,
                              variant: str,
                              iteration: int,
                              output_dir: Path) -> Optional[Path]:
    """Generate interactive 3D scatter plot of the point cloud."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        return None

    if point_cloud.shape[1] < 3:
        return None

    # Subsample for visualization if too many points
    if len(point_cloud) > 5000:
        indices = np.random.choice(len(point_cloud), 5000, replace=False)
        vis_points = point_cloud[indices]
    else:
        vis_points = point_cloud

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Color by position in sequence (shows temporal evolution)
    colors = np.arange(len(vis_points))

    scatter = ax.scatter(vis_points[:, 0], vis_points[:, 1], vis_points[:, 2],
                        c=colors, cmap='viridis', s=1, alpha=0.6)

    ax.set_xlabel('x(t)')
    ax.set_ylabel('x(t+τ)')
    ax.set_zlabel('x(t+2τ)')
    ax.set_title(f'Takens Embedding - Variant {variant} (Iter {iteration})\n'
                 f'{len(vis_points):,} points shown')

    plt.colorbar(scatter, label='Sequence position', shrink=0.6)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"takens_3d_{variant}_iter{iteration}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🌀 HSI Level 2 - Takens Embedding & TDA Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Global analysis (default)
  python level2_takens_tda.py -v B -i 23 --mode global

  # Local windowed analysis
  python level2_takens_tda.py -v B -i 23 --mode local --windows 10

  # Both analyses
  python level2_takens_tda.py -v B -i 23 --mode both

  # Compare variants
  python level2_takens_tda.py --variants B F -i 23 --compare
        """
    )

    # Variant selection
    parser.add_argument('--variant', '-v', type=str,
                       help='Single variant to analyze (A-M)')
    parser.add_argument('--variants', type=str, nargs='+',
                       help='Multiple variants to compare')
    parser.add_argument('--iteration', '-i', type=int, required=True,
                       help='Iteration number')

    # Mode selection
    parser.add_argument('--mode', type=str, default='global',
                       choices=['global', 'local', 'both'],
                       help='Analysis mode (default: global)')

    # Embedding parameters
    parser.add_argument('--dim', '-d', type=int, default=DEFAULT_EMBEDDING_DIM,
                       help=f'Embedding dimension (default: {DEFAULT_EMBEDDING_DIM})')
    parser.add_argument('--delay', type=int, default=DEFAULT_TIME_DELAY,
                       help=f'Time delay τ (default: {DEFAULT_TIME_DELAY})')

    # Global mode parameters
    parser.add_argument('--landmarks', type=int, default=DEFAULT_MAX_LANDMARKS,
                       help=f'Max landmarks for global TDA (default: {DEFAULT_MAX_LANDMARKS:,})')
    parser.add_argument('--landmark-method', type=str, default='uniform',
                       choices=['uniform', 'random'],
                       help='Landmark selection method (default: uniform)')

    # Local mode parameters
    parser.add_argument('--windows', type=int, default=DEFAULT_WINDOWS,
                       help=f'Number of windows for local TDA (default: {DEFAULT_WINDOWS})')
    parser.add_argument('--points-per-window', type=int, default=DEFAULT_POINTS_PER_WINDOW,
                       help=f'Points per window (default: {DEFAULT_POINTS_PER_WINDOW:,})')

    # Block mode parameters
    parser.add_argument('--blocks', type=int, default=0,
                       help='Number of blocks for block processing (0=disabled)')
    parser.add_argument('--landmarks-per-block', type=int, default=5000,
                       help='Landmarks per block (default: 5000)')

    # Backend selection
    parser.add_argument('--backend', type=str, default='auto',
                       choices=['auto', 'ripser', 'giotto', 'gudhi'],
                       help='TDA backend (default: auto - uses best available)')

    # Scalable methods (witness/cubical)
    parser.add_argument('--method', type=str, default='cubical',
                       choices=['rips', 'witness', 'cubical'],
                       help='TDA method: cubical (fast, default), witness (scalable), rips (exact but slow)')
    parser.add_argument('--cubical-mode', type=str, default='hilbert',
                       choices=['hilbert', 'binary'],
                       help='Cubical mode: hilbert (density gradients, default) or binary (raw 0/1)')
    parser.add_argument('--hilbert-block-size', type=int, default=64,
                       help='Bits per Hilbert cell (default: 64). Larger = less noise, global structure')
    parser.add_argument('--save-hilbert', action='store_true',
                       help='Save Hilbert image used for cubical TDA (PNG)')
    parser.add_argument('--witnesses', type=int, default=50000,
                       help='Number of witnesses for witness complex (default: 50000)')

    # General options
    parser.add_argument('--max-bits', type=int, default=50_000_000,
                       help='Max bits to load from sequence (default: 50M)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate 3D visualization')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple variants')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')

    args = parser.parse_args()

    variants = args.variants or ([args.variant] if args.variant else ['B'])
    verbose = not args.quiet

    if args.compare or len(variants) > 1:
        # Multi-variant comparison
        print(f"\n{'═'*70}")
        print(f"🌀 MULTI-VARIANT TDA COMPARISON")
        print(f"   Variants: {', '.join(variants)}")
        print(f"   Iteration: {args.iteration}")
        print(f"   Mode: {args.mode}")
        print(f"{'═'*70}")

        all_results = {}
        for var in variants:
            result = analyze_variant_tda(
                var, args.iteration,
                mode=args.mode,
                method=args.method,
                embedding_dim=args.dim,
                time_delay=args.delay,
                max_landmarks=args.landmarks,
                landmark_method=args.landmark_method,
                n_windows=args.windows,
                points_per_window=args.points_per_window,
                n_blocks=args.blocks,
                landmarks_per_block=args.landmarks_per_block,
                n_witnesses=args.witnesses,
                cubical_mode=args.cubical_mode,
                hilbert_block_size=args.hilbert_block_size,
                save_hilbert=args.save_hilbert,
                backend=args.backend,
                max_bits=args.max_bits,
                visualize=args.visualize,
                verbose=verbose
            )
            all_results[var] = result

        # Summary comparison (for global mode)
        if args.mode in ('global', 'both'):
            print(f"\n📊 GLOBAL COMPARISON SUMMARY")
            print(f"{'═'*70}")
            print(f"   {'Variant':<10} {'β₀':<8} {'β₁':<8} {'β₂':<8} {'Assessment':<30}")
            print(f"   {'─'*66}")

            for var, res in all_results.items():
                global_res = res.get('global', {})
                betti = global_res.get('betti', {})
                b0 = betti.get('beta_0', '?')
                b1 = betti.get('beta_1', '?')
                b2 = betti.get('beta_2', '?')

                interp = res.get('global_interpretation', '')
                if 'STRONG' in interp:
                    assess = '🌟 STRONG'
                elif 'MODERATE' in interp:
                    assess = '🔶 MODERATE'
                elif 'WEAK' in interp:
                    assess = '🔸 WEAK'
                else:
                    assess = '⚪ NONE'

                print(f"   {var:<10} {b0:<8} {b1:<8} {b2:<8} {assess:<30}")

            print(f"{'═'*70}\n")
    else:
        # Single variant
        analyze_variant_tda(
            variants[0], args.iteration,
            mode=args.mode,
            method=args.method,
            embedding_dim=args.dim,
            time_delay=args.delay,
            max_landmarks=args.landmarks,
            landmark_method=args.landmark_method,
            n_windows=args.windows,
            points_per_window=args.points_per_window,
            n_blocks=args.blocks,
            landmarks_per_block=args.landmarks_per_block,
            n_witnesses=args.witnesses,
            cubical_mode=args.cubical_mode,
            hilbert_block_size=args.hilbert_block_size,
            save_hilbert=args.save_hilbert,
            backend=args.backend,
            max_bits=args.max_bits,
            visualize=args.visualize,
            verbose=verbose
        )


if __name__ == "__main__":
    main()
