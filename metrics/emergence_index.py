"""
🌌 Emergence Index — Metric for Level 2 Potential in HSI

This module calculates the "Emergence Index", a composite metric that estimates
the potential of a Φ sequence to exhibit emergent physical-like behavior.

The index combines three fundamental indicators:
1. Criticality (1/f spectrum): Systems at the "edge of chaos" show 1/f noise
2. Lempel-Ziv Complexity: Neither pure order nor pure chaos = maximum potential
3. Long-Range Coherence: Mutual Information between distant regions

A higher Emergence Index suggests the variant has more potential for generating
complex emergent structures at Level 2 (physical law-like behavior).

Theoretical basis:
- 1/f noise indicates self-organized criticality (Bak et al., 1987)
- Intermediate complexity indicates optimal computational capacity (Langton, 1990)
- Long-range correlations indicate coherent global structure

Author: Iban Borràs with Augment Agent collaboration
Date: December 2024
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.fft import fft
from scipy.stats import linregress
import warnings


def calculate_power_spectrum_slope(sequence: str, min_freq_idx: int = 1) -> Tuple[float, float]:
    """
    Calculate the power spectrum slope to detect 1/f criticality.
    
    A slope of -1 indicates perfect 1/f noise (criticality).
    A slope of 0 indicates white noise (pure randomness).
    A slope of -2 indicates Brownian motion (over-correlated).
    
    Args:
        sequence: Binary string (e.g., "01101001...")
        min_freq_idx: Minimum frequency index to avoid DC component
        
    Returns:
        Tuple of (slope, r_squared) where slope ~ -1 indicates criticality
    """
    # Convert to numpy array
    bits = np.array([int(b) for b in sequence], dtype=np.float64)
    
    # Subtract mean to remove DC component
    bits = bits - np.mean(bits)
    
    # Apply window to reduce spectral leakage
    window = np.hanning(len(bits))
    bits_windowed = bits * window
    
    # Compute FFT
    spectrum = np.abs(fft(bits_windowed)[:len(bits)//2])**2
    
    # Frequency bins (skip DC)
    freqs = np.arange(min_freq_idx, len(spectrum))
    power = spectrum[min_freq_idx:]
    
    # Filter zero/negative values for log
    mask = power > 0
    if np.sum(mask) < 10:
        return 0.0, 0.0
    
    log_freq = np.log10(freqs[mask])
    log_power = np.log10(power[mask])
    
    # Linear regression in log-log space
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slope, intercept, r_value, p_value, std_err = linregress(log_freq, log_power)
    
    return slope, r_value**2


def calculate_lempel_ziv_complexity(sequence: str, max_sample: int = 100_000, verbose: bool = True) -> float:
    """
    Calculate normalized Lempel-Ziv complexity.

    Returns a value between 0 and 1:
    - 0 = perfectly predictable (e.g., "000000...")
    - 1 = maximally complex (incompressible random)
    - ~0.5 = optimal complexity for emergence

    Args:
        sequence: Binary string
        max_sample: Maximum bits to analyze (LZ76 is O(n²), so we sample)
        verbose: Print progress updates

    Returns:
        Normalized LZ complexity (0-1)
    """
    n = len(sequence)
    if n == 0:
        return 0.0

    # Sample if sequence is too long (LZ76 is O(n²))
    if n > max_sample:
        # Take sample from middle for representativeness
        start = (n - max_sample) // 2
        sequence = sequence[start:start + max_sample]
        n = max_sample
        if verbose:
            print(f"      (Sampling {max_sample:,} bits for LZ complexity)")

    # LZ76 algorithm with progress reporting
    complexity = 1
    i = 0
    k = 1
    k_max = 1
    last_progress = 0
    progress_interval = n // 10  # Report every 10%

    while i + k <= n:
        # Progress reporting
        if verbose and progress_interval > 0 and i - last_progress >= progress_interval:
            pct = (i / n) * 100
            print(f"      LZ progress: {pct:.0f}%")
            last_progress = i

        # Check if sequence[i:i+k] appears in sequence[0:i+k-1]
        if sequence[i:i+k] in sequence[0:i+k-1]:
            k += 1
            if i + k > n:
                complexity += 1
                break
        else:
            complexity += 1
            i += k_max if k > k_max else k
            k = 1
            k_max = max(k_max, k)

    # Normalize by theoretical maximum for random sequence
    # For random binary: C_max ≈ n / log2(n)
    if n > 1:
        c_max = n / np.log2(n)
        normalized = complexity / c_max
    else:
        normalized = 0.0
    
    return min(1.0, normalized)


def calculate_long_range_mutual_info(sequence: str, 
                                      block_size: int = 100,
                                      max_distance: int = 10000) -> float:
    """
    Calculate mutual information between distant blocks.
    
    High MI at long distances indicates global coherence.
    
    Args:
        sequence: Binary string
        block_size: Size of blocks to compare
        max_distance: Maximum distance between blocks
        
    Returns:
        Average MI ratio (actual MI / max possible MI)
    """
    n = len(sequence)
    if n < block_size * 2 + max_distance:
        max_distance = (n - block_size * 2) // 2
    
    if max_distance < block_size:
        return 0.0
    
    # Sample multiple block pairs at various distances
    distances = np.linspace(block_size, max_distance, min(20, max_distance // block_size))
    mi_values = []
    
    for dist in distances:
        dist = int(dist)
        # Sample multiple pairs at this distance
        n_samples = min(10, (n - block_size - dist) // (block_size * 2))
        
        for _ in range(max(1, n_samples)):
            start1 = np.random.randint(0, n - block_size - dist)
            start2 = start1 + dist
            
            block1 = sequence[start1:start1 + block_size]
            block2 = sequence[start2:start2 + block_size]
            
            # Calculate MI via joint and marginal probabilities
            mi = _calculate_block_mi(block1, block2)
            mi_values.append(mi)
    
    return np.mean(mi_values) if mi_values else 0.0


def _calculate_block_mi(block1: str, block2: str) -> float:
    """Calculate mutual information between two binary blocks."""
    # Count joint occurrences of bit pairs
    joint = np.zeros((2, 2))
    for b1, b2 in zip(block1, block2):
        joint[int(b1), int(b2)] += 1

    joint /= len(block1)  # Normalize to probabilities

    # Marginal probabilities
    p1 = joint.sum(axis=1)
    p2 = joint.sum(axis=0)

    # Mutual Information: I(X;Y) = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for i in range(2):
        for j in range(2):
            if joint[i, j] > 0 and p1[i] > 0 and p2[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (p1[i] * p2[j]))

    # Normalize by maximum possible MI (1 bit for binary)
    return max(0.0, mi)


def calculate_emergence_index(sequence: str,
                               sample_size: int = 1_000_000,
                               verbose: bool = True) -> Dict[str, Any]:
    """
    Calculate the composite Emergence Index.

    The index combines:
    - Criticality score (how close to 1/f spectrum)
    - Complexity score (optimal LZ complexity around 0.5)
    - Coherence score (long-range mutual information)

    Args:
        sequence: Binary Φ sequence
        sample_size: Maximum sample size for analysis (for large sequences)
        verbose: Print progress messages

    Returns:
        Dictionary with:
        - emergence_index: Composite score (0-1, higher = more potential)
        - criticality: Dict with slope, r_squared, score
        - complexity: Dict with lz_complexity, score
        - coherence: Dict with mi_ratio, score
        - interpretation: Human-readable interpretation
    """
    import time

    def log(msg):
        if verbose:
            print(f"   {msg}", flush=True)

    # Sample if sequence is too large
    if len(sequence) > sample_size:
        # Take middle portion for representative sample
        start = (len(sequence) - sample_size) // 2
        seq = sequence[start:start + sample_size]
    else:
        seq = sequence

    results = {
        'sequence_length': len(sequence),
        'sample_size': len(seq)
    }

    # 1. Criticality (1/f spectrum)
    log("⚡ [1/3] Calculating power spectrum (FFT)...")
    t0 = time.time()
    slope, r_sq = calculate_power_spectrum_slope(seq)
    log(f"   Done in {time.time()-t0:.1f}s - slope={slope:.3f}")
    # Perfect 1/f has slope = -1; score based on distance from -1
    criticality_score = max(0, 1 - abs(slope + 1) / 2)  # 0-1 scale
    results['criticality'] = {
        'slope': slope,
        'r_squared': r_sq,
        'target': -1.0,
        'score': criticality_score,
        'interpretation': _interpret_slope(slope)
    }

    # 2. Lempel-Ziv Complexity
    log("🧩 [2/3] Calculating Lempel-Ziv complexity...")
    t0 = time.time()
    lz = calculate_lempel_ziv_complexity(seq, verbose=verbose)
    log(f"   Done in {time.time()-t0:.1f}s - LZ={lz:.3f}")
    # Optimal complexity is around 0.5 (between order and chaos)
    complexity_score = 1 - 2 * abs(lz - 0.5)  # 0-1 scale, peak at 0.5
    results['complexity'] = {
        'lz_normalized': lz,
        'optimal': 0.5,
        'score': complexity_score,
        'interpretation': _interpret_lz(lz)
    }

    # 3. Long-Range Coherence
    log("🔗 [3/3] Calculating long-range mutual information...")
    t0 = time.time()
    mi_ratio = calculate_long_range_mutual_info(seq)
    log(f"   Done in {time.time()-t0:.1f}s - MI={mi_ratio:.4f}")
    # Higher MI = more coherence = better
    coherence_score = min(1.0, mi_ratio * 10)  # Scale to 0-1
    results['coherence'] = {
        'mi_ratio': mi_ratio,
        'score': coherence_score,
        'interpretation': _interpret_mi(mi_ratio)
    }

    # Composite Emergence Index (weighted average)
    # Criticality weighted slightly higher as it's the strongest indicator
    weights = {'criticality': 0.4, 'complexity': 0.3, 'coherence': 0.3}
    emergence_index = (
        weights['criticality'] * criticality_score +
        weights['complexity'] * complexity_score +
        weights['coherence'] * coherence_score
    )

    results['emergence_index'] = emergence_index
    results['component_weights'] = weights
    results['interpretation'] = _interpret_emergence(emergence_index)

    log(f"✅ Emergence Index: {emergence_index:.4f}")

    return results


def _interpret_slope(slope: float) -> str:
    """Interpret power spectrum slope."""
    if -1.2 <= slope <= -0.8:
        return "1/f noise (critical, high emergence potential)"
    elif slope > -0.3:
        return "White noise (random, low structure)"
    elif slope < -1.5:
        return "Brownian (over-correlated, low dynamics)"
    else:
        return f"Intermediate ({slope:.2f})"


def _interpret_lz(lz: float) -> str:
    """Interpret Lempel-Ziv complexity."""
    if lz < 0.3:
        return "Low complexity (highly ordered, predictable)"
    elif lz > 0.7:
        return "High complexity (near-random, chaotic)"
    elif 0.4 <= lz <= 0.6:
        return "Optimal complexity (edge of chaos, high potential)"
    else:
        return f"Moderate complexity ({lz:.2f})"


def _interpret_mi(mi: float) -> str:
    """Interpret mutual information ratio."""
    if mi < 0.01:
        return "No long-range correlation (local structure only)"
    elif mi < 0.05:
        return "Weak coherence (some global structure)"
    elif mi < 0.1:
        return "Moderate coherence (significant global patterns)"
    else:
        return "Strong coherence (highly correlated distant regions)"


def _interpret_emergence(index: float) -> str:
    """Interpret composite emergence index."""
    if index < 0.3:
        return "Low emergence potential (random or trivially ordered)"
    elif index < 0.5:
        return "Moderate emergence potential"
    elif index < 0.7:
        return "High emergence potential (good candidate for Level 2)"
    else:
        return "Very high emergence potential (strong Level 2 candidate)"


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def load_phi_sequence(variant: str, iteration: int,
                       max_bits: int = 2_000_000) -> Optional[str]:
    """
    Load a SAMPLE of Φ sequence from phi_snapshots for a given variant and iteration.

    To avoid memory issues with large sequences (2GB+), we only load a sample
    from the middle portion of the sequence.

    Args:
        variant: Variant code (A, B, D, E, F, G, H)
        iteration: Iteration number
        max_bits: Maximum bits to load (default 2M for fast analysis)

    Returns:
        Binary string sample or None if not found
    """
    import gzip
    from pathlib import Path

    # Try v33 format first (bitarray in .struct.gz)
    base_path = Path(__file__).parent.parent / "results" / "phi_snapshots" / f"var_{variant}"
    struct_path = base_path / f"phi_iter{iteration}.struct.gz"

    if struct_path.exists():
        try:
            from bitarray import bitarray
            with gzip.open(struct_path, 'rb') as f:
                # Read file to get total size first
                data = f.read()
            ba = bitarray()
            ba.frombytes(data)
            total_bits = len(ba)

            # Sample from middle for representative data
            if total_bits > max_bits:
                start = (total_bits - max_bits) // 2
                sample = ba[start:start + max_bits].to01()
                print(f"   Sampled {max_bits:,} bits from middle of {total_bits:,} total bits")
                return sample
            else:
                return ba.to01()
        except Exception as e:
            print(f"Warning: Could not load {struct_path}: {e}")

    # Try bin.gz format
    bin_path = base_path / f"phi_iter{iteration}.bin.gz"
    if bin_path.exists():
        try:
            from bitarray import bitarray
            with gzip.open(bin_path, 'rb') as f:
                data = f.read()
            ba = bitarray()
            ba.frombytes(data)
            total_bits = len(ba)

            if total_bits > max_bits:
                start = (total_bits - max_bits) // 2
                sample = ba[start:start + max_bits].to01()
                print(f"   Sampled {max_bits:,} bits from middle of {total_bits:,} total bits")
                return sample
            else:
                return ba.to01()
        except Exception as e:
            print(f"Warning: Could not load {bin_path}: {e}")

    return None


def compare_variants_emergence(variants: list, iteration: int) -> Dict[str, Any]:
    """
    Compare emergence index across multiple variants.

    Args:
        variants: List of variant codes (e.g., ['A', 'B', 'D'])
        iteration: Iteration number to compare

    Returns:
        Dictionary with comparison results and ranking
    """
    results = {}

    for variant in variants:
        print(f"   Analyzing variant {variant}...", flush=True)
        seq = load_phi_sequence(variant, iteration)

        if seq is None:
            print(f"   ⚠️ Could not load variant {variant} iter {iteration}")
            results[variant] = None
            continue

        emergence = calculate_emergence_index(seq)
        results[variant] = emergence

        print(f"   ✓ Variant {variant}: EI = {emergence['emergence_index']:.3f}")

    # Rank variants by emergence index
    valid_results = {k: v for k, v in results.items() if v is not None}
    ranking = sorted(valid_results.keys(),
                     key=lambda v: valid_results[v]['emergence_index'],
                     reverse=True)

    return {
        'iteration': iteration,
        'variants': results,
        'ranking': ranking,
        'best_variant': ranking[0] if ranking else None,
        'best_score': valid_results[ranking[0]]['emergence_index'] if ranking else 0
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for Emergence Index calculation."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='Calculate Emergence Index for HSI Φ sequences.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python emergence_index.py --variant B --iteration 18
  python emergence_index.py --variants A B D E F --iteration 18 --compare
  python emergence_index.py --variant B --iteration 18 --output results/emergence_B.json
        """
    )

    parser.add_argument('--variant', '-v', type=str,
                        help='Single variant to analyze (A, B, D, E, F, G, H)')
    parser.add_argument('--variants', nargs='+',
                        help='Multiple variants for comparison')
    parser.add_argument('--iteration', '-i', type=int, required=True,
                        help='Iteration number')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple variants and rank them')
    parser.add_argument('--output', '-o', type=str,
                        help='Output JSON file path')
    parser.add_argument('--sample-size', type=int, default=1_000_000,
                        help='Sample size for large sequences (default: 1M)')

    args = parser.parse_args()

    # Validate arguments
    if not args.variant and not args.variants:
        parser.error("Must specify --variant or --variants")

    print("\n🌌 HSI Emergence Index Calculator")
    print("=" * 50)

    if args.compare or args.variants:
        # Multi-variant comparison mode
        variants = args.variants or [args.variant]
        print(f"\n📊 Comparing variants: {', '.join(variants)}")
        print(f"   Iteration: {args.iteration}\n")

        results = compare_variants_emergence(variants, args.iteration)

        # Print summary
        print("\n" + "=" * 50)
        print("📈 EMERGENCE INDEX RANKING")
        print("=" * 50)

        for rank, variant in enumerate(results['ranking'], 1):
            data = results['variants'][variant]
            ei = data['emergence_index']
            interp = data['interpretation']
            print(f"   {rank}. Variant {variant}: {ei:.4f} - {interp}")

        if results['best_variant']:
            print(f"\n🏆 Best candidate for Level 2: Variant {results['best_variant']}")

    else:
        # Single variant mode
        print(f"\n📊 Analyzing variant {args.variant}, iteration {args.iteration}\n")

        seq = load_phi_sequence(args.variant, args.iteration)
        if seq is None:
            print(f"❌ Could not load variant {args.variant} iteration {args.iteration}")
            return

        print(f"   Loaded {len(seq):,} bits")
        results = calculate_emergence_index(seq, args.sample_size)

        # Print detailed results
        print("\n" + "=" * 50)
        print("🌌 EMERGENCE INDEX ANALYSIS")
        print("=" * 50)

        print(f"\n📏 Sample: {results['sample_size']:,} / {results['sequence_length']:,} bits")

        c = results['criticality']
        print(f"\n⚡ CRITICALITY (1/f spectrum)")
        print(f"   Slope: {c['slope']:.3f} (target: {c['target']})")
        print(f"   R²: {c['r_squared']:.3f}")
        print(f"   Score: {c['score']:.3f}")
        print(f"   → {c['interpretation']}")

        x = results['complexity']
        print(f"\n🧩 COMPLEXITY (Lempel-Ziv)")
        print(f"   LZ normalized: {x['lz_normalized']:.3f} (optimal: {x['optimal']})")
        print(f"   Score: {x['score']:.3f}")
        print(f"   → {x['interpretation']}")

        h = results['coherence']
        print(f"\n🔗 COHERENCE (Long-range MI)")
        print(f"   MI ratio: {h['mi_ratio']:.4f}")
        print(f"   Score: {h['score']:.3f}")
        print(f"   → {h['interpretation']}")

        print(f"\n{'=' * 50}")
        print(f"🌌 EMERGENCE INDEX: {results['emergence_index']:.4f}")
        print(f"   → {results['interpretation']}")
        print("=" * 50)

    # Save results (auto-generate filename if not specified)
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate filename based on variant(s) and iteration
        script_dir = Path(__file__).parent.parent
        results_dir = script_dir / "results"
        if args.compare and args.variants:
            variants_str = "_".join(args.variants)
            filename = f"emergence_vars_{variants_str}_iter{args.iteration}.json"
        else:
            variant = args.variant or (args.variants[0] if args.variants else "unknown")
            filename = f"emergence_var{variant}_iter{args.iteration}.json"
        output_path = results_dir / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {output_path}")


if __name__ == '__main__':
    main()

