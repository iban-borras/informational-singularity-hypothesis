#!/usr/bin/env python3
"""
🌀 HSI Level 2 - Takens Embedding & Topological Data Analysis (TDA)

Implements Antigravity's Proposal 1: Discover if the Φ sequence draws a
geometric shape (torus, sphere, strange attractor) when projected into phase space.

Method:
1. Takens' Embedding: Convert 1D sequence to 3D/nD point clouds using time delay
   (x_t, x_{t+τ}, x_{t+2τ})
2. Persistent Homology (TDA): Calculate Betti numbers (β₀, β₁, β₂)
   - β₀: Connected components
   - β₁: 1D cycles/tunnels (indicates recurring orbits)
   - β₂: 2D cavities (indicates closed surface, like holographic membrane)

Scientific Value:
If we find persistent β₂ > 0, we've demonstrated that Level 1 generates
a REAL geometric surface (emergence of space).

Requirements:
    pip install ripser giotto-tda plotly

Usage:
    python level2_takens_tda.py --variant B --iteration 18 --dim 3
    python level2_takens_tda.py --variants B D E --iteration 18 --compare
    python level2_takens_tda.py --variant B --iteration 18 --visualize

Author: Sophia (for Iban & Antigravity)
Date: December 2025
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# TDA libraries
try:
    import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False

try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceEntropy, Amplitude
    HAS_GIOTTO = True
except ImportError:
    HAS_GIOTTO = False

# Local imports
from metrics.emergence_index import load_phi_sequence

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")
TDA_DIR = RESULTS_DIR / "level2" / "tda_analysis"

# Default parameters
DEFAULT_EMBEDDING_DIM = 3
DEFAULT_TIME_DELAY = 1
DEFAULT_MAX_POINTS = 10_000  # TDA is O(n³) so we must limit points

# ═══════════════════════════════════════════════════════════════════════════════
# TAKENS EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════

def takens_embedding(sequence: str,
                     embedding_dim: int = 3,
                     time_delay: int = 1,
                     max_points: int = 10_000,
                     verbose: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert a binary sequence to a point cloud using Takens' embedding.

    For a sequence s = [s₀, s₁, s₂, ...], we create vectors:
    v_t = (s_t, s_{t+τ}, s_{t+2τ}, ..., s_{t+(d-1)τ})

    where d = embedding_dim and τ = time_delay.

    Args:
        sequence: Binary string
        embedding_dim: Dimension of the embedding space (default 3)
        time_delay: Time delay τ between coordinates (default 1)
        max_points: Maximum number of points to generate
        verbose: Print progress

    Returns:
        Tuple of (point_cloud as numpy array, metadata dict)
    """
    log = print if verbose else lambda *a, **k: None

    log(f"\n   🌀 Creating Takens embedding...")
    log(f"      Dimension: {embedding_dim}")
    log(f"      Time delay (τ): {time_delay}")

    n = len(sequence)

    # Maximum number of vectors we can create
    max_vectors = n - (embedding_dim - 1) * time_delay

    if max_vectors <= 0:
        log(f"   ❌ Sequence too short for embedding")
        return np.array([]), {'error': 'sequence too short'}

    # Subsample if too many points
    if max_vectors > max_points:
        step = max_vectors // max_points
        indices = range(0, max_vectors, step)[:max_points]
        log(f"      Subsampling: {max_vectors:,} → {len(indices):,} points (step={step})")
    else:
        indices = range(max_vectors)
        log(f"      Points: {len(indices):,}")

    # Convert sequence to numeric array
    log(f"      Converting sequence to numeric...")
    seq_array = np.array([int(b) for b in sequence], dtype=np.float32)

    # Create embedding
    log(f"      Building point cloud...")
    t0 = time.perf_counter()

    points = []
    for i in tqdm(indices, desc="Embedding", disable=not HAS_TQDM or not verbose):
        vector = [seq_array[i + j * time_delay] for j in range(embedding_dim)]
        points.append(vector)

    point_cloud = np.array(points, dtype=np.float32)
    embed_time = time.perf_counter() - t0

    log(f"      ✓ Created {point_cloud.shape[0]:,} points in {embed_time:.2f}s")
    log(f"      Shape: {point_cloud.shape}")

    metadata = {
        'embedding_dim': embedding_dim,
        'time_delay': time_delay,
        'original_length': n,
        'max_possible_points': max_vectors,
        'actual_points': point_cloud.shape[0],
        'subsampled': max_vectors > max_points,
        'embedding_time_s': embed_time
    }

    return point_cloud, metadata


# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENT HOMOLOGY (TDA)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_persistent_homology(point_cloud: np.ndarray,
                                 max_dim: int = 2,
                                 verbose: bool = True) -> Dict[str, Any]:
    """
    Compute persistent homology of the point cloud.

    Returns Betti numbers:
    - β₀: Connected components (how many separate clusters)
    - β₁: 1D cycles/loops (recurring orbits, like a donut hole)
    - β₂: 2D cavities (enclosed surfaces, like a sphere)

    Args:
        point_cloud: Numpy array of shape (n_points, n_dim)
        max_dim: Maximum homology dimension to compute
        verbose: Print progress

    Returns:
        Dictionary with Betti numbers, persistence diagrams, etc.
    """
    log = print if verbose else lambda *a, **k: None

    log(f"\n   📊 Computing Persistent Homology...")
    log(f"      Max dimension: {max_dim}")
    log(f"      Points: {point_cloud.shape[0]:,}")

    if not HAS_RIPSER and not HAS_GIOTTO:
        log(f"   ❌ Neither ripser nor giotto-tda installed!")
        log(f"      Install with: pip install ripser giotto-tda")
        return {'error': 'TDA libraries not available', 'betti': None}

    t0 = time.perf_counter()

    result = {
        'n_points': point_cloud.shape[0],
        'embedding_dim': point_cloud.shape[1],
        'max_homology_dim': max_dim,
        'library': None,
        'betti': {},
        'persistence_diagrams': {},
        'persistence_entropy': None,
        'compute_time_s': 0
    }

    # Use ripser (faster for small point clouds)
    if HAS_RIPSER:
        log(f"      Using: ripser")
        result['library'] = 'ripser'

        try:
            # Compute persistence diagrams
            log(f"      Computing Vietoris-Rips complex...")
            diagrams = ripser.ripser(point_cloud, maxdim=max_dim)['dgms']

            # Extract Betti numbers (count persistent features)
            for dim in range(max_dim + 1):
                if dim < len(diagrams):
                    dgm = diagrams[dim]
                    # Filter out infinite persistence (usually β₀ = 1 connected component)
                    finite_dgm = dgm[dgm[:, 1] < np.inf] if len(dgm) > 0 else np.array([])

                    # Count significant features (persistence > 0.1 of max)
                    if len(finite_dgm) > 0:
                        lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]
                        threshold = np.max(lifetimes) * 0.1
                        significant = np.sum(lifetimes > threshold)
                        total = len(lifetimes)
                        max_persistence = np.max(lifetimes)
                    else:
                        significant = 0
                        total = 0
                        max_persistence = 0

                    result['betti'][f'beta_{dim}'] = {
                        'total_features': total,
                        'significant_features': int(significant),
                        'max_persistence': float(max_persistence)
                    }

                    # Store diagram for later visualization
                    result['persistence_diagrams'][f'H{dim}'] = dgm.tolist()

                    symbol = "🟢" if significant > 0 else "⚪"
                    log(f"      {symbol} β_{dim} = {significant} significant ({total} total, max={max_persistence:.4f})")

        except Exception as e:
            log(f"   ❌ ripser error: {e}")
            result['error'] = str(e)

    elif HAS_GIOTTO:
        log(f"      Using: giotto-tda")
        result['library'] = 'giotto-tda'

        try:
            # Reshape for giotto-tda (expects 3D array: n_samples x n_points x n_dim)
            point_cloud_3d = point_cloud.reshape(1, *point_cloud.shape)

            # Compute persistence
            VR = VietorisRipsPersistence(homology_dimensions=list(range(max_dim + 1)))
            diagrams = VR.fit_transform(point_cloud_3d)

            # Compute persistence entropy
            PE = PersistenceEntropy()
            entropy = PE.fit_transform(diagrams)
            result['persistence_entropy'] = float(entropy[0, 0]) if entropy.size > 0 else None

            log(f"      ✓ Computed via giotto-tda")
            # Note: giotto-tda Betti extraction is more complex, simplified here
            result['betti']['note'] = 'See persistence_diagrams for details'

        except Exception as e:
            log(f"   ❌ giotto-tda error: {e}")
            result['error'] = str(e)

    compute_time = time.perf_counter() - t0
    result['compute_time_s'] = compute_time
    log(f"      ⏱️ Computation time: {compute_time:.2f}s")

    return result


def interpret_betti_numbers(betti: Dict[str, Any], verbose: bool = True) -> str:
    """
    Interpret Betti numbers in terms of emergent geometry.

    Returns interpretation string.
    """
    log = print if verbose else lambda *a, **k: None

    log(f"\n   🔮 TOPOLOGICAL INTERPRETATION:")
    log(f"   {'─'*50}")

    beta_0 = betti.get('beta_0', {}).get('significant_features', 0)
    beta_1 = betti.get('beta_1', {}).get('significant_features', 0)
    beta_2 = betti.get('beta_2', {}).get('significant_features', 0)

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
        log(f"         This could represent a 'holographic membrane' or 'spacetime horizon'.")
    else:
        interp = "No 2D cavities (no enclosed surfaces)"
        interpretations.append(interp)
        log(f"      β₂=0: {interp}")

    # Overall assessment
    log(f"\n   📋 SUMMARY:")
    if beta_2 > 0:
        assessment = "🌟 STRONG GEOMETRIC EMERGENCE: Persistent 2D surface detected"
        log(f"      {assessment}")
    elif beta_1 > 2:
        assessment = "🔶 MODERATE EMERGENCE: Multiple orbital structures detected"
        log(f"      {assessment}")
    elif beta_1 > 0:
        assessment = "🔸 WEAK EMERGENCE: Some cyclic structure detected"
        log(f"      {assessment}")
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
                        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                        time_delay: int = DEFAULT_TIME_DELAY,
                        max_points: int = DEFAULT_MAX_POINTS,
                        visualize: bool = False,
                        verbose: bool = True) -> Dict[str, Any]:
    """
    Perform complete TDA analysis on a variant.

    Args:
        variant: Variant letter (B, D, E, etc.)
        iteration: Iteration number
        embedding_dim: Dimension for Takens embedding
        time_delay: Time delay for embedding
        max_points: Maximum points (TDA is O(n³))
        visualize: Generate 3D visualization
        verbose: Print progress

    Returns:
        Complete analysis results dictionary
    """
    log = print if verbose else lambda *a, **k: None

    print(f"\n{'═'*70}")
    print(f"🌀 HSI TAKENS/TDA ANALYSIS")
    print(f"{'═'*70}")
    print(f"   Variant: {variant}")
    print(f"   Iteration: {iteration}")
    print(f"   Embedding dim: {embedding_dim}")
    print(f"   Time delay (τ): {time_delay}")
    print(f"   Max points: {max_points:,}")
    print(f"{'─'*70}")

    t_start = time.perf_counter()

    # Load sequence
    log(f"\n   📂 Loading Φ sequence...")
    sequence = load_phi_sequence(variant, iteration, verbose=verbose)

    if not sequence:
        log(f"   ❌ Failed to load sequence")
        return {'error': 'failed to load sequence'}

    log(f"      Loaded {len(sequence):,} bits")

    # Takens embedding
    point_cloud, embed_meta = takens_embedding(
        sequence,
        embedding_dim=embedding_dim,
        time_delay=time_delay,
        max_points=max_points,
        verbose=verbose
    )

    if point_cloud.size == 0:
        return {'error': 'embedding failed', 'embedding': embed_meta}

    # Persistent homology
    tda_results = compute_persistent_homology(point_cloud, max_dim=2, verbose=verbose)

    # Interpretation
    interpretation = ""
    if tda_results.get('betti'):
        interpretation = interpret_betti_numbers(tda_results['betti'], verbose=verbose)

    total_time = time.perf_counter() - t_start

    # Compile results
    results = {
        'variant': variant,
        'iteration': iteration,
        'timestamp': datetime.now().isoformat(),
        'sequence_length': len(sequence),
        'embedding': embed_meta,
        'tda': tda_results,
        'interpretation': interpretation,
        'total_time_s': total_time
    }

    # Optional 3D visualization
    if visualize and point_cloud.shape[1] >= 3:
        plot_path = generate_3d_visualization(point_cloud, variant, iteration, TDA_DIR)
        if plot_path:
            results['visualization'] = str(plot_path)
            log(f"\n   📊 3D plot saved to: {plot_path}")

    # Save results
    TDA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TDA_DIR / f"tda_{variant}_iter{iteration}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        # Convert non-serializable items
        save_results = results.copy()
        if 'persistence_diagrams' in save_results.get('tda', {}):
            # Keep diagrams as they are (already lists)
            pass
        json.dump(save_results, f, indent=2, default=str)

    log(f"\n   💾 Results saved to: {output_path}")
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
  python level2_takens_tda.py --variant B --iteration 18
  python level2_takens_tda.py --variant B --iteration 18 --dim 4 --delay 2
  python level2_takens_tda.py --variant B --iteration 18 --visualize
  python level2_takens_tda.py --variants B D E --iteration 18 --compare
        """
    )

    parser.add_argument('--variant', '-v', type=str,
                       help='Single variant to analyze (A-I)')
    parser.add_argument('--variants', type=str, nargs='+',
                       help='Multiple variants to compare')
    parser.add_argument('--iteration', '-i', type=int, required=True,
                       help='Iteration number')
    parser.add_argument('--dim', '-d', type=int, default=DEFAULT_EMBEDDING_DIM,
                       help=f'Embedding dimension (default: {DEFAULT_EMBEDDING_DIM})')
    parser.add_argument('--delay', type=int, default=DEFAULT_TIME_DELAY,
                       help=f'Time delay τ (default: {DEFAULT_TIME_DELAY})')
    parser.add_argument('--max-points', type=int, default=DEFAULT_MAX_POINTS,
                       help=f'Max points for TDA (default: {DEFAULT_MAX_POINTS:,})')
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
        print(f"{'═'*70}")

        all_results = {}
        for var in variants:
            result = analyze_variant_tda(
                var, args.iteration,
                embedding_dim=args.dim,
                time_delay=args.delay,
                max_points=args.max_points,
                visualize=args.visualize,
                verbose=verbose
            )
            all_results[var] = result

        # Summary comparison
        print(f"\n📊 COMPARISON SUMMARY")
        print(f"{'═'*70}")
        print(f"   {'Variant':<10} {'β₀':<8} {'β₁':<8} {'β₂':<8} {'Assessment':<30}")
        print(f"   {'─'*66}")

        for var, res in all_results.items():
            betti = res.get('tda', {}).get('betti', {})
            b0 = betti.get('beta_0', {}).get('significant_features', '?')
            b1 = betti.get('beta_1', {}).get('significant_features', '?')
            b2 = betti.get('beta_2', {}).get('significant_features', '?')

            # Extract assessment from interpretation
            interp = res.get('interpretation', '')
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
            embedding_dim=args.dim,
            time_delay=args.delay,
            max_points=args.max_points,
            visualize=args.visualize,
            verbose=verbose
        )


if __name__ == "__main__":
    main()
