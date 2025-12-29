#!/usr/bin/env python3
"""
🔀 HSI Level 2 - Transfer Entropy Multi-Scale Analysis

Implements Antigravity's Proposal 2: Measure information flow between
different scales of the Φ sequence to detect causal relationships.

Transfer Entropy (TE) measures the directed information flow from X → Y:
    TE(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

If TE(X→Y) > 0, then knowing X's past helps predict Y's future beyond
knowing Y's past alone.

Multi-Scale Application:
- Coarse-grain the sequence at different scales (2^k blocks)
- Measure TE between scales: Does micro-structure predict macro-structure?
- If TE(micro→macro) > TE(macro→micro): Bottom-up emergence
- If TE(macro→micro) > TE(micro→macro): Top-down causation

Scientific Value:
This can reveal if HSI has genuine multi-scale causal structure
(like living systems) vs pure noise.

Requirements:
    pip install pyinform  # or use our simplified implementation

Usage:
    python level2_transfer_entropy.py --variant B --iteration 18
    python level2_transfer_entropy.py --variants B D E --iteration 18 --compare
    python level2_transfer_entropy.py --variant B --iteration 18 --scales 1 2 4 8 16

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

# Try to import pyinform (specialized TE library)
try:
    import pyinform
    from pyinform import transfer_entropy as te
    HAS_PYINFORM = True
except ImportError:
    HAS_PYINFORM = False

# Local imports
from metrics.emergence_index import load_phi_sequence

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")
TE_DIR = RESULTS_DIR / "transfer_entropy"

# Default parameters
DEFAULT_SCALES = [1, 2, 4, 8, 16, 32]  # Block sizes for coarse-graining
DEFAULT_HISTORY = 3  # k parameter (history length)
DEFAULT_MAX_SAMPLES = 100_000  # Limit for TE computation

# ═══════════════════════════════════════════════════════════════════════════════
# COARSE-GRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def coarse_grain(sequence: np.ndarray, block_size: int) -> np.ndarray:
    """
    Coarse-grain a binary sequence by majority vote in blocks.

    For block_size=4: [0,1,1,0, 1,1,1,0, ...] → [0, 1, ...]
    (0+1+1+0=2 < 2 → 0, 1+1+1+0=3 > 2 → 1)

    Args:
        sequence: Binary sequence as numpy array
        block_size: Size of blocks to aggregate

    Returns:
        Coarse-grained binary sequence
    """
    n = len(sequence)
    n_blocks = n // block_size

    if n_blocks == 0:
        return sequence

    # Reshape into blocks and compute majority
    truncated = sequence[:n_blocks * block_size]
    blocks = truncated.reshape(n_blocks, block_size)

    # Majority vote: 1 if sum > half, else 0
    coarse = (blocks.sum(axis=1) > block_size / 2).astype(np.int8)

    return coarse


def create_multiscale_representation(sequence: str,
                                      scales: List[int],
                                      max_samples: int,
                                      verbose: bool = True) -> Dict[int, np.ndarray]:
    """
    Create coarse-grained representations at multiple scales.

    Args:
        sequence: Binary string
        scales: List of block sizes
        max_samples: Maximum samples per scale
        verbose: Print progress

    Returns:
        Dictionary mapping scale → coarse-grained array
    """
    log = print if verbose else lambda *a, **k: None

    log(f"\n   📏 Creating multi-scale representations...")
    log(f"      Scales: {scales}")

    # Convert to numpy
    seq_array = np.array([int(b) for b in sequence], dtype=np.int8)

    representations = {}

    for scale in tqdm(scales, desc="Coarse-graining", disable=not HAS_TQDM or not verbose):
        coarse = coarse_grain(seq_array, scale)

        # Subsample if too long
        if len(coarse) > max_samples:
            step = len(coarse) // max_samples
            coarse = coarse[::step][:max_samples]

        representations[scale] = coarse
        log(f"      Scale {scale:3d}: {len(coarse):,} samples")

    return representations


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFER ENTROPY COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_transfer_entropy_simple(source: np.ndarray,
                                     target: np.ndarray,
                                     k: int = 3) -> float:
    """
    Compute Transfer Entropy from source → target using histogram method.

    TE(X→Y) = H(Y_t | Y_{t-k:t-1}) - H(Y_t | Y_{t-k:t-1}, X_{t-k:t-1})

    Simplified implementation when pyinform is not available.

    Args:
        source: Source time series (X)
        target: Target time series (Y)
        k: History length

    Returns:
        Transfer entropy in bits
    """
    n = min(len(source), len(target))
    if n <= k + 1:
        return 0.0

    # Align sequences
    source = source[:n]
    target = target[:n]

    # Build joint states
    # State = (Y_{t-k}, ..., Y_{t-1}, X_{t-k}, ..., X_{t-1}, Y_t)

    from collections import Counter

    # Count joint occurrences
    joint_counts = Counter()  # (Y_past, X_past, Y_future)
    y_past_counts = Counter()  # (Y_past)
    yx_past_counts = Counter()  # (Y_past, X_past)
    y_future_given_past = Counter()  # (Y_past, Y_future)

    for t in range(k, n):
        y_past = tuple(target[t-k:t])
        x_past = tuple(source[t-k:t])
        y_future = target[t]

        joint_counts[(y_past, x_past, y_future)] += 1
        y_past_counts[y_past] += 1
        yx_past_counts[(y_past, x_past)] += 1
        y_future_given_past[(y_past, y_future)] += 1

    total = n - k

    # Compute TE using the formula:
    # TE = sum p(y_past, x_past, y_future) * log2(p(y_future|y_past,x_past) / p(y_future|y_past))

    te_value = 0.0

    for (y_past, x_past, y_future), count in joint_counts.items():
        p_joint = count / total

        # p(y_future | y_past, x_past)
        p_future_given_both = count / yx_past_counts[(y_past, x_past)]

        # p(y_future | y_past)
        p_future_given_y = y_future_given_past[(y_past, y_future)] / y_past_counts[y_past]

        if p_future_given_y > 0:
            te_value += p_joint * np.log2(p_future_given_both / p_future_given_y)

    return max(0.0, te_value)  # TE should be non-negative


def compute_transfer_entropy(source: np.ndarray,
                              target: np.ndarray,
                              k: int = 3,
                              verbose: bool = False) -> Dict[str, float]:
    """
    Compute Transfer Entropy using best available method.

    Args:
        source: Source time series
        target: Target time series
        k: History length
        verbose: Print details

    Returns:
        Dictionary with TE value and metadata
    """
    log = print if verbose else lambda *a, **k: None

    result = {
        'method': 'simple_histogram',
        'k': k,
        'source_length': len(source),
        'target_length': len(target),
        'te_value': 0.0
    }

    if HAS_PYINFORM:
        log(f"      Using: pyinform")
        result['method'] = 'pyinform'
        try:
            te_value = te.transfer_entropy(source.tolist(), target.tolist(), k)
            result['te_value'] = float(te_value)
        except Exception as e:
            log(f"      ⚠️ pyinform error: {e}, falling back to simple method")
            result['te_value'] = compute_transfer_entropy_simple(source, target, k)
    else:
        log(f"      Using: histogram method")
        result['te_value'] = compute_transfer_entropy_simple(source, target, k)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-SCALE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_multiscale_te(representations: Dict[int, np.ndarray],
                          k: int = 3,
                          verbose: bool = True) -> Dict[str, Any]:
    """
    Compute Transfer Entropy between all pairs of scales.

    Key questions:
    - TE(fine → coarse) vs TE(coarse → fine): Direction of information flow
    - If TE(micro → macro) > TE(macro → micro): Bottom-up emergence

    Args:
        representations: Scale → array mapping
        k: History length for TE
        verbose: Print progress

    Returns:
        Dictionary with TE matrix and interpretations
    """
    log = print if verbose else lambda *a, **k: None

    scales = sorted(representations.keys())
    n_scales = len(scales)

    log(f"\n   🔀 Computing Transfer Entropy between scales...")
    log(f"      History length k={k}")

    # Create TE matrix
    te_matrix = np.zeros((n_scales, n_scales))

    # Compute pairwise TE
    pairs = [(i, j) for i in range(n_scales) for j in range(n_scales) if i != j]

    for i, j in tqdm(pairs, desc="Computing TE", disable=not HAS_TQDM or not verbose):
        scale_i = scales[i]
        scale_j = scales[j]

        source = representations[scale_i]
        target = representations[scale_j]

        # Align lengths
        min_len = min(len(source), len(target))
        source = source[:min_len]
        target = target[:min_len]

        te_result = compute_transfer_entropy(source, target, k, verbose=False)
        te_matrix[i, j] = te_result['te_value']

    # Analyze results
    log(f"\n   📊 TRANSFER ENTROPY MATRIX (bits):")
    log(f"      {'':>8}", end='')
    for s in scales:
        log(f" →{s:>5}", end='')
    log()

    for i, s_from in enumerate(scales):
        log(f"      {s_from:>5} →", end='')
        for j in range(n_scales):
            if i == j:
                log(f"   ---", end='')
            else:
                log(f" {te_matrix[i,j]:>5.3f}", end='')
        log()

    # Compute directional asymmetries
    bottom_up = 0.0  # TE from fine to coarse
    top_down = 0.0   # TE from coarse to fine
    n_pairs = 0

    for i in range(n_scales):
        for j in range(i + 1, n_scales):
            # i is finer scale, j is coarser
            bottom_up += te_matrix[i, j]  # fine → coarse
            top_down += te_matrix[j, i]   # coarse → fine
            n_pairs += 1

    if n_pairs > 0:
        avg_bottom_up = bottom_up / n_pairs
        avg_top_down = top_down / n_pairs
    else:
        avg_bottom_up = avg_top_down = 0.0

    # Interpretation
    asymmetry = avg_bottom_up - avg_top_down

    log(f"\n   🔮 DIRECTIONAL ANALYSIS:")
    log(f"      Avg TE (fine → coarse): {avg_bottom_up:.4f} bits")
    log(f"      Avg TE (coarse → fine): {avg_top_down:.4f} bits")
    log(f"      Asymmetry: {asymmetry:+.4f} bits")

    if asymmetry > 0.01:
        interpretation = "🔼 BOTTOM-UP EMERGENCE: Fine scales causally influence coarse scales"
        log(f"\n      {interpretation}")
        log(f"         This suggests genuine emergent behavior (micro → macro causation)")
    elif asymmetry < -0.01:
        interpretation = "🔽 TOP-DOWN CAUSATION: Coarse scales causally influence fine scales"
        log(f"\n      {interpretation}")
        log(f"         This suggests constraint-driven dynamics (macro → micro)")
    else:
        interpretation = "↔️ SYMMETRIC: No clear directional information flow"
        log(f"\n      {interpretation}")
        log(f"         Information flows equally in both directions")

    results = {
        'scales': scales,
        'te_matrix': te_matrix.tolist(),
        'avg_bottom_up': avg_bottom_up,
        'avg_top_down': avg_top_down,
        'asymmetry': asymmetry,
        'interpretation': interpretation,
        'k': k
    }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_variant_te(variant: str,
                       iteration: int,
                       scales: List[int] = None,
                       k: int = DEFAULT_HISTORY,
                       max_samples: int = DEFAULT_MAX_SAMPLES,
                       verbose: bool = True) -> Dict[str, Any]:
    """
    Perform complete multi-scale Transfer Entropy analysis on a variant.

    Args:
        variant: Variant letter
        iteration: Iteration number
        scales: List of coarse-graining scales
        k: History length for TE
        max_samples: Maximum samples per scale
        verbose: Print progress

    Returns:
        Complete analysis results
    """
    if scales is None:
        scales = DEFAULT_SCALES.copy()

    print(f"\n{'═'*70}")
    print(f"🔀 HSI TRANSFER ENTROPY ANALYSIS")
    print(f"{'═'*70}")
    print(f"   Variant: {variant}")
    print(f"   Iteration: {iteration}")
    print(f"   Scales: {scales}")
    print(f"   History k: {k}")
    print(f"   Max samples: {max_samples:,}")
    print(f"   Library: {'pyinform' if HAS_PYINFORM else 'simple histogram'}")
    print(f"{'─'*70}")

    t_start = time.perf_counter()

    # Load sequence
    print(f"\n   📂 Loading Φ sequence...")
    sequence = load_phi_sequence(variant, iteration)

    if not sequence:
        print(f"   ❌ Failed to load sequence")
        return {'error': 'failed to load sequence'}

    print(f"      Loaded {len(sequence):,} bits")

    # Create multi-scale representations
    representations = create_multiscale_representation(
        sequence, scales, max_samples, verbose=verbose
    )

    # Compute TE between scales
    te_results = analyze_multiscale_te(representations, k=k, verbose=verbose)

    total_time = time.perf_counter() - t_start

    # Compile results
    results = {
        'variant': variant,
        'iteration': iteration,
        'timestamp': datetime.now().isoformat(),
        'sequence_length': len(sequence),
        'te_analysis': te_results,
        'total_time_s': total_time
    }

    # Save results
    TE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TE_DIR / f"te_{variant}_iter{iteration}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n   💾 Results saved to: {output_path}")
    print(f"   ⏱️ Total time: {total_time:.2f}s")
    print(f"{'═'*70}\n")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_te_heatmap(results: Dict[str, Any], output_dir: Path) -> Optional[Path]:
    """Generate heatmap of the TE matrix."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    te_analysis = results.get('te_analysis', {})
    te_matrix = np.array(te_analysis.get('te_matrix', []))
    scales = te_analysis.get('scales', [])

    if te_matrix.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(te_matrix, cmap='YlOrRd', aspect='auto')

    # Labels
    ax.set_xticks(range(len(scales)))
    ax.set_yticks(range(len(scales)))
    ax.set_xticklabels([f'→{s}' for s in scales])
    ax.set_yticklabels([f'{s}→' for s in scales])

    ax.set_xlabel('Target Scale', fontsize=12)
    ax.set_ylabel('Source Scale', fontsize=12)
    ax.set_title(f"Transfer Entropy Matrix - Variant {results.get('variant', '?')}\n"
                 f"(Iteration {results.get('iteration', '?')})", fontsize=14)

    # Add values
    for i in range(len(scales)):
        for j in range(len(scales)):
            if i != j:
                text = ax.text(j, i, f'{te_matrix[i, j]:.3f}',
                              ha='center', va='center', fontsize=9)

    plt.colorbar(im, label='Transfer Entropy (bits)')

    # Add asymmetry annotation
    asymmetry = te_analysis.get('asymmetry', 0)
    interpretation = te_analysis.get('interpretation', '')
    ax.text(0.02, -0.12, f"Asymmetry: {asymmetry:+.4f} bits\n{interpretation}",
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"te_heatmap_{results.get('variant', 'X')}_iter{results.get('iteration', 0)}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🔀 HSI Level 2 - Transfer Entropy Multi-Scale Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python level2_transfer_entropy.py --variant B --iteration 18
  python level2_transfer_entropy.py --variant B --iteration 18 --scales 1 2 4 8 16 32
  python level2_transfer_entropy.py --variants B D E --iteration 18 --compare
  python level2_transfer_entropy.py --variant B --iteration 18 --plot
        """
    )

    parser.add_argument('--variant', '-v', type=str,
                       help='Single variant to analyze (A-I)')
    parser.add_argument('--variants', type=str, nargs='+',
                       help='Multiple variants to compare')
    parser.add_argument('--iteration', '-i', type=int, required=True,
                       help='Iteration number')
    parser.add_argument('--scales', type=int, nargs='+', default=DEFAULT_SCALES,
                       help=f'Coarse-graining scales (default: {DEFAULT_SCALES})')
    parser.add_argument('--k', type=int, default=DEFAULT_HISTORY,
                       help=f'History length for TE (default: {DEFAULT_HISTORY})')
    parser.add_argument('--max-samples', type=int, default=DEFAULT_MAX_SAMPLES,
                       help=f'Max samples per scale (default: {DEFAULT_MAX_SAMPLES:,})')
    parser.add_argument('--plot', action='store_true',
                       help='Generate TE heatmap')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple variants')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')

    args = parser.parse_args()

    variants = args.variants or ([args.variant] if args.variant else ['B'])
    verbose = not args.quiet

    all_results = {}

    for var in variants:
        result = analyze_variant_te(
            var, args.iteration,
            scales=args.scales,
            k=args.k,
            max_samples=args.max_samples,
            verbose=verbose
        )
        all_results[var] = result

        if args.plot:
            plot_path = generate_te_heatmap(result, TE_DIR)
            if plot_path:
                print(f"   📊 Heatmap saved to: {plot_path}")

    # Summary comparison
    if len(variants) > 1:
        print(f"\n📊 COMPARISON SUMMARY")
        print(f"{'═'*70}")
        print(f"   {'Variant':<10} {'Bottom-Up':<12} {'Top-Down':<12} {'Asymmetry':<12} {'Direction':<20}")
        print(f"   {'─'*66}")

        for var, res in all_results.items():
            te = res.get('te_analysis', {})
            bu = te.get('avg_bottom_up', 0)
            td = te.get('avg_top_down', 0)
            asym = te.get('asymmetry', 0)

            if asym > 0.01:
                direction = '🔼 Bottom-up'
            elif asym < -0.01:
                direction = '🔽 Top-down'
            else:
                direction = '↔️ Symmetric'

            print(f"   {var:<10} {bu:<12.4f} {td:<12.4f} {asym:<+12.4f} {direction:<20}")

        print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
