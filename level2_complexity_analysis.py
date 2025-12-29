#!/usr/bin/env python3
"""
🧬 HSI Level 2 - Algorithmic Complexity Analysis

Implements Antigravity's Proposal 3: Compare variants using complexity metrics
to distinguish "noise" (high entropy, low logical depth) from "music" 
(balanced entropy, high logical depth).

Metrics calculated:
- Lempel-Ziv Complexity (LZ) - Already implemented
- Shannon Entropy - Standard information measure
- Logical Depth Proxy - Generation time / compressed size
- BDM (Block Decomposition Method) - If pybdm available

Expected results:
- Variant A: High Entropy + Low Logical Depth (noise)
- Variant B: Balanced Entropy + High Logical Depth (organized structure)

Usage:
    python level2_complexity_analysis.py --variants A B D E --iteration 18
    python level2_complexity_analysis.py --variant B --iteration 18 --verbose
    
Author: Sophia (for Iban & Antigravity)
Date: December 2025
"""

import argparse
import json
import time
import zlib
import gzip
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

# Try to import BDM library
try:
    from pybdm import BDM
    HAS_BDM = True
except ImportError:
    HAS_BDM = False

# Local imports
from metrics.emergence_index import (
    calculate_lempel_ziv_complexity,
    load_phi_sequence
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")
COMPLEXITY_DIR = RESULTS_DIR / "complexity_analysis"

# ═══════════════════════════════════════════════════════════════════════════════
# SHANNON ENTROPY
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_shannon_entropy(sequence: str, block_size: int = 8) -> Dict[str, float]:
    """
    Calculate Shannon entropy at multiple scales.
    
    Args:
        sequence: Binary string
        block_size: Size of blocks to analyze (default 8 = byte level)
    
    Returns:
        Dictionary with entropy values at different scales
    """
    n = len(sequence)
    if n == 0:
        return {'bit_entropy': 0.0, 'block_entropy': 0.0, 'normalized': 0.0}
    
    # Bit-level entropy (0/1 distribution)
    ones = sequence.count('1')
    zeros = n - ones
    p1 = ones / n if n > 0 else 0
    p0 = zeros / n if n > 0 else 0
    
    bit_entropy = 0.0
    if p1 > 0:
        bit_entropy -= p1 * np.log2(p1)
    if p0 > 0:
        bit_entropy -= p0 * np.log2(p0)
    
    # Block-level entropy
    if n >= block_size:
        num_blocks = n // block_size
        block_counts: Dict[str, int] = {}
        
        for i in range(num_blocks):
            block = sequence[i*block_size:(i+1)*block_size]
            block_counts[block] = block_counts.get(block, 0) + 1
        
        block_entropy = 0.0
        for count in block_counts.values():
            p = count / num_blocks
            if p > 0:
                block_entropy -= p * np.log2(p)
        
        # Normalize to 0-1 (max entropy for block_size bits is block_size)
        normalized_block = block_entropy / block_size if block_size > 0 else 0
    else:
        block_entropy = bit_entropy
        normalized_block = bit_entropy
    
    return {
        'bit_entropy': bit_entropy,
        'block_entropy': block_entropy,
        'normalized': normalized_block,
        'block_size': block_size,
        'unique_blocks': len(block_counts) if n >= block_size else 1
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOGICAL DEPTH PROXY
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_logical_depth_proxy(sequence: str,
                                   generation_time_ms: Optional[float] = None) -> Dict[str, Any]:
    """
    Calculate a proxy for Bennett's Logical Depth.

    Logical Depth = Time to generate from shortest description / Size of shortest description

    We approximate this as:
    - Shortest description ≈ compressed size (gzip)
    - Generation time = actual time if available, or estimate from sequence properties

    High logical depth = "meaningful complexity" (not trivial, not random)

    Args:
        sequence: Binary string
        generation_time_ms: Optional actual generation time in milliseconds

    Returns:
        Dictionary with logical depth metrics
    """
    n = len(sequence)
    if n == 0:
        return {'logical_depth': 0.0, 'compressed_ratio': 1.0, 'interpretation': 'empty'}

    # Compress the sequence to get "shortest description" approximation
    seq_bytes = sequence.encode('utf-8')
    original_size = len(seq_bytes)

    # Use multiple compression levels for comparison
    t0 = time.perf_counter()
    compressed_fast = gzip.compress(seq_bytes, compresslevel=1)
    compress_time_fast = (time.perf_counter() - t0) * 1000  # ms

    t0 = time.perf_counter()
    compressed_best = gzip.compress(seq_bytes, compresslevel=9)
    compress_time_best = (time.perf_counter() - t0) * 1000  # ms

    compressed_size_fast = len(compressed_fast)
    compressed_size_best = len(compressed_best)

    # Compression ratios
    ratio_fast = compressed_size_fast / original_size
    ratio_best = compressed_size_best / original_size

    # Logical Depth Proxy:
    # - If we have actual generation time, use it
    # - Otherwise, use compression time as a very rough proxy
    #
    # The idea: A sequence with high logical depth is:
    # - Compressible (has structure, so compressed_size < original)
    # - But takes time to generate (complexity is in the process)

    if generation_time_ms is not None:
        effective_gen_time = generation_time_ms
    else:
        # Estimate: sequences with more structure take longer to compress well
        effective_gen_time = compress_time_best

    # Logical Depth = generation_time / compressed_size
    # Normalize to make it comparable across sequence sizes
    if compressed_size_best > 0:
        logical_depth_raw = effective_gen_time / compressed_size_best
        # Normalize by sequence length to compare across different sizes
        logical_depth = logical_depth_raw * (n / 1000)  # Scale factor
    else:
        logical_depth = 0.0

    # Alternative metric: Compression Difficulty Ratio
    # How much harder is best compression vs fast compression?
    # High ratio = more structure to discover (not trivial)
    if compressed_size_fast > 0:
        compression_difficulty = compress_time_best / compress_time_fast if compress_time_fast > 0 else 1
    else:
        compression_difficulty = 1.0

    # Classification using both metrics
    if ratio_best > 0.95:
        interpretation = "incompressible (random-like)"
    elif ratio_best < 0.1 and compression_difficulty < 2.0:
        interpretation = "highly compressible (trivial pattern)"
    elif ratio_best < 0.3 and compression_difficulty >= 2.0:
        interpretation = "STRUCTURED (compressible but complex)"
    elif logical_depth > 1.0:
        interpretation = "HIGH LOGICAL DEPTH (organized complexity)"
    else:
        interpretation = "moderate logical depth"

    return {
        'logical_depth': logical_depth,
        'logical_depth_raw': logical_depth_raw if compressed_size_best > 0 else 0,
        'compression_ratio_fast': ratio_fast,
        'compression_ratio_best': ratio_best,
        'compression_difficulty': compression_difficulty,
        'original_bytes': original_size,
        'compressed_bytes_fast': compressed_size_fast,
        'compressed_bytes_best': compressed_size_best,
        'compress_time_fast_ms': compress_time_fast,
        'compress_time_best_ms': compress_time_best,
        'generation_time_ms': generation_time_ms,
        'interpretation': interpretation
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BDM (Block Decomposition Method)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_bdm_complexity(sequence: str, max_length: int = 100_000) -> Dict[str, Any]:
    """
    Calculate BDM (Block Decomposition Method) complexity.

    BDM approximates Kolmogorov complexity better than LZ or compression.
    Uses CTM (Coding Theorem Method) lookup tables for small blocks.

    Requires pybdm library: pip install pybdm

    Args:
        sequence: Binary string
        max_length: Maximum sequence length to analyze

    Returns:
        Dictionary with BDM metrics
    """
    if not HAS_BDM:
        return {
            'bdm': None,
            'normalized_bdm': None,
            'available': False,
            'note': 'pybdm not installed. Install with: pip install pybdm'
        }

    # Sample if too long
    if len(sequence) > max_length:
        start = (len(sequence) - max_length) // 2
        sequence = sequence[start:start + max_length]

    # Convert to numpy array
    arr = np.array([int(b) for b in sequence], dtype=np.int32)

    try:
        bdm = BDM(ndim=1)
        bdm_value = bdm.bdm(arr)

        # Normalize: BDM for random sequence ≈ length
        normalized = bdm_value / len(arr) if len(arr) > 0 else 0

        return {
            'bdm': bdm_value,
            'normalized_bdm': normalized,
            'length_analyzed': len(arr),
            'available': True
        }
    except Exception as e:
        return {
            'bdm': None,
            'normalized_bdm': None,
            'available': False,
            'error': str(e)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FULL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_variant_complexity(variant: str,
                                iteration: int,
                                verbose: bool = True,
                                max_bits: int = 500_000) -> Optional[Dict[str, Any]]:
    """
    Run full complexity analysis on a single variant.

    Args:
        variant: Variant code (A, B, D, E, etc.)
        iteration: Iteration number
        verbose: Print detailed progress
        max_bits: Maximum bits to analyze

    Returns:
        Dictionary with all complexity metrics
    """
    log = print if verbose else lambda *a, **k: None

    log(f"\n{'='*60}")
    log(f"📊 COMPLEXITY ANALYSIS: Variant {variant} @ Iteration {iteration}")
    log(f"{'='*60}")

    # Load sequence
    log(f"\n   [1/5] Loading Φ sequence...")
    t0 = time.perf_counter()

    seq = load_phi_sequence(variant, iteration)
    if seq is None:
        log(f"   ❌ ERROR: Could not load variant {variant} iteration {iteration}")
        return None

    load_time = (time.perf_counter() - t0) * 1000
    log(f"   ✓ Loaded {len(seq):,} bits in {load_time:.1f}ms")

    # Sample if needed
    original_length = len(seq)
    if len(seq) > max_bits:
        start = (len(seq) - max_bits) // 2
        seq = seq[start:start + max_bits]
        log(f"   ⚠ Sampled {max_bits:,} bits from center (original: {original_length:,})")

    results = {
        'variant': variant,
        'iteration': iteration,
        'sequence_length': original_length,
        'analyzed_length': len(seq),
        'timestamp': datetime.now().isoformat()
    }

    # Shannon Entropy
    log(f"\n   [2/5] Calculating Shannon Entropy...")
    t0 = time.perf_counter()
    entropy_result = calculate_shannon_entropy(seq)
    entropy_time = (time.perf_counter() - t0) * 1000
    results['shannon_entropy'] = entropy_result
    log(f"   ✓ Bit entropy: {entropy_result['bit_entropy']:.4f}")
    log(f"   ✓ Block entropy (8-bit): {entropy_result['normalized']:.4f}")
    log(f"   ✓ Unique 8-bit blocks: {entropy_result['unique_blocks']:,}/256")
    log(f"   ⏱ Time: {entropy_time:.1f}ms")

    # Lempel-Ziv Complexity
    log(f"\n   [3/5] Calculating Lempel-Ziv Complexity...")
    t0 = time.perf_counter()
    lz_value = calculate_lempel_ziv_complexity(seq, verbose=False)
    lz_time = (time.perf_counter() - t0) * 1000
    results['lempel_ziv'] = {
        'normalized': lz_value,
        'time_ms': lz_time
    }
    log(f"   ✓ LZ Complexity: {lz_value:.4f}")
    log(f"   ⏱ Time: {lz_time:.1f}ms")

    # Logical Depth
    log(f"\n   [4/5] Calculating Logical Depth Proxy...")
    t0 = time.perf_counter()
    depth_result = calculate_logical_depth_proxy(seq, generation_time_ms=None)
    depth_time = (time.perf_counter() - t0) * 1000
    results['logical_depth'] = depth_result
    log(f"   ✓ Logical Depth: {depth_result['logical_depth']:.4f}")
    log(f"   ✓ Compression ratio: {depth_result['compression_ratio_best']:.4f}")
    log(f"   ✓ Interpretation: {depth_result['interpretation']}")
    log(f"   ⏱ Time: {depth_time:.1f}ms")

    # BDM (if available)
    log(f"\n   [5/5] Calculating BDM Complexity...")
    t0 = time.perf_counter()
    bdm_result = calculate_bdm_complexity(seq)
    bdm_time = (time.perf_counter() - t0) * 1000
    results['bdm'] = bdm_result

    if bdm_result['available']:
        log(f"   ✓ BDM: {bdm_result['bdm']:.4f}")
        log(f"   ✓ Normalized BDM: {bdm_result['normalized_bdm']:.4f}")
    else:
        log(f"   ⚠ BDM not available: {bdm_result.get('note', bdm_result.get('error', 'unknown'))}")
    log(f"   ⏱ Time: {bdm_time:.1f}ms")

    # Summary classification
    log(f"\n{'─'*60}")
    log(f"   📋 SUMMARY FOR VARIANT {variant}:")

    # Classify based on metrics
    entropy_norm = entropy_result['normalized']
    lz_norm = lz_value
    depth = depth_result['logical_depth']

    if entropy_norm > 0.9 and depth < 0.5:
        classification = "NOISE (high entropy, low depth)"
        emoji = "📻"
    elif entropy_norm < 0.3 and depth < 0.5:
        classification = "TRIVIAL (low entropy, low depth)"
        emoji = "🔄"
    elif 0.4 <= entropy_norm <= 0.8 and depth >= 0.5:
        classification = "ORGANIZED COMPLEXITY (balanced entropy, high depth)"
        emoji = "🎵"
    else:
        classification = "INTERMEDIATE"
        emoji = "🔍"

    results['classification'] = classification
    log(f"   {emoji} Classification: {classification}")
    log(f"{'='*60}\n")

    return results


def compare_variants_complexity(variants: List[str],
                                 iteration: int,
                                 verbose: bool = True,
                                 max_bits: int = 500_000,
                                 save_results: bool = True) -> Dict[str, Any]:
    """
    Compare complexity metrics across multiple variants.

    Args:
        variants: List of variant codes
        iteration: Iteration number
        verbose: Print detailed progress
        max_bits: Maximum bits per variant
        save_results: Save results to JSON

    Returns:
        Dictionary with comparison results
    """
    print(f"\n{'═'*70}")
    print(f"🔬 MULTI-VARIANT COMPLEXITY COMPARISON")
    print(f"   Variants: {', '.join(variants)}")
    print(f"   Iteration: {iteration}")
    print(f"   Max bits per variant: {max_bits:,}")
    print(f"{'═'*70}")

    results = {
        'variants': variants,
        'iteration': iteration,
        'timestamp': datetime.now().isoformat(),
        'analyses': {}
    }

    # Analyze each variant with progress bar
    variant_iter = tqdm(variants, desc="Analyzing variants", unit="var") if HAS_TQDM else variants

    for variant in variant_iter:
        if HAS_TQDM:
            variant_iter.set_description(f"Analyzing {variant}")

        analysis = analyze_variant_complexity(
            variant, iteration,
            verbose=verbose,
            max_bits=max_bits
        )

        if analysis is not None:
            results['analyses'][variant] = analysis

    # Generate comparison summary
    print(f"\n{'═'*70}")
    print(f"📊 COMPARISON SUMMARY")
    print(f"{'═'*70}")

    # Table header
    print(f"\n   {'Variant':<8} {'Entropy':<10} {'LZ':<10} {'Depth':<12} {'Compress':<10} {'Class'}")
    print(f"   {'─'*8} {'─'*10} {'─'*10} {'─'*12} {'─'*10} {'─'*25}")

    rankings = []
    for var in variants:
        if var not in results['analyses']:
            print(f"   {var:<8} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<10} N/A")
            continue

        a = results['analyses'][var]
        entropy = a['shannon_entropy']['normalized']
        lz = a['lempel_ziv']['normalized']
        depth = a['logical_depth']['logical_depth']
        compress = a['logical_depth']['compression_ratio_best']
        classification = a['classification'].split('(')[0].strip()

        print(f"   {var:<8} {entropy:<10.4f} {lz:<10.4f} {depth:<12.4f} {compress:<10.4f} {classification}")

        # HSI-SPECIFIC SCORING:
        # For HSI, we want:
        # - Low LZ (high structure/compressibility) but NOT trivial
        # - Balanced entropy (not too high = noise, not too low = trivial)
        # - High depth relative to compression (structure takes work to find)
        #
        # A truly "interesting" HSI variant should be:
        # - Highly compressible (low compress ratio)
        # - But with moderate entropy (has variety)
        # - And meaningful depth (structure is deep, not surface-level)

        # Compressibility bonus: lower is better (more structure)
        compress_score = 1 - min(compress, 1.0)  # 0.01 → 0.99

        # Entropy: penalize extremes (want ~0.3-0.5 for structured data)
        if 0.25 <= entropy <= 0.55:
            entropy_score = 1.0  # Sweet spot for structured complexity
        else:
            entropy_score = 1 - abs(entropy - 0.4) * 2
        entropy_score = max(0, entropy_score)

        # LZ: for HSI, LOW LZ is good (structure!) but not zero
        if 0.01 <= lz <= 0.1:
            lz_score = 1.0  # Ideal: structured but not trivial
        elif lz < 0.01:
            lz_score = 0.5  # Too trivial
        else:
            lz_score = max(0, 1 - (lz - 0.1) * 2)  # High LZ = noise

        # Combined HSI score
        combined_score = compress_score * 0.4 + entropy_score * 0.3 + lz_score * 0.3

        rankings.append({
            'variant': var,
            'entropy': entropy,
            'lz': lz,
            'depth': depth,
            'score': combined_score
        })

    # Sort and show ranking
    rankings.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n   🏆 RANKING (by Organized Complexity Score):")
    print(f"   {'─'*45}")
    for i, r in enumerate(rankings, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"   {medal} {i}. Variant {r['variant']}: Score = {r['score']:.4f}")

    results['ranking'] = rankings

    # Save results
    if save_results:
        COMPLEXITY_DIR.mkdir(parents=True, exist_ok=True)
        output_path = COMPLEXITY_DIR / f"complexity_{'_'.join(variants)}_iter{iteration}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n   💾 Results saved to: {output_path}")

        # Generate visualization
        plot_path = generate_complexity_plot(results, COMPLEXITY_DIR)
        if plot_path:
            print(f"   📊 Plot saved to: {plot_path}")

    print(f"{'═'*70}\n")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_complexity_plot(results: Dict[str, Any], output_dir: Path) -> Optional[Path]:
    """
    Generate a scatter plot comparing variants on Entropy vs LZ axes.

    Creates a 2x2 subplot:
    1. Entropy vs LZ (main comparison)
    2. Entropy vs Compression Ratio
    3. LZ vs HSI Score
    4. Bar chart of HSI Scores

    Args:
        results: Results from compare_variants_complexity()
        output_dir: Directory to save the plot

    Returns:
        Path to saved plot or None if failed
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("   ⚠️ matplotlib not available for plotting")
        return None

    analyses = results.get('analyses', {})
    if len(analyses) < 2:
        print("   ⚠️ Need at least 2 variants to generate comparison plot")
        return None

    # Extract data
    variants = []
    entropies = []
    lzs = []
    depths = []
    compresses = []
    scores = []

    # Color map for variants
    colors = {
        'A': '#e74c3c',  # Red (chaotic)
        'B': '#3498db',  # Blue (candidate)
        'C': '#9b59b6',  # Purple
        'D': '#2ecc71',  # Green (candidate)
        'E': '#f39c12',  # Orange (candidate)
        'F': '#1abc9c',  # Teal
        'G': '#34495e',  # Dark gray
        'H': '#e91e63',  # Pink
        'I': '#00bcd4',  # Cyan
    }

    for var, data in analyses.items():
        variants.append(var)
        entropies.append(data['shannon_entropy']['normalized'])
        lzs.append(data['lempel_ziv']['normalized'])
        depths.append(data['logical_depth']['logical_depth'])
        compresses.append(data['logical_depth']['compression_ratio_best'])

        # Find score in ranking
        for r in results.get('ranking', []):
            if r['variant'] == var:
                scores.append(r['score'])
                break
        else:
            scores.append(0)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"HSI Algorithmic Complexity Analysis (Iteration {results.get('iteration', '?')})",
                 fontsize=14, fontweight='bold')

    # Plot 1: Entropy vs LZ (main comparison)
    ax1 = axes[0, 0]
    for i, var in enumerate(variants):
        ax1.scatter(entropies[i], lzs[i], s=200, c=colors.get(var, '#999999'),
                   label=f'Variant {var}', edgecolors='black', linewidths=1.5, zorder=5)
        ax1.annotate(var, (entropies[i], lzs[i]), fontsize=12, fontweight='bold',
                    ha='center', va='center', color='white')

    # Add "ideal zone" rectangle (low LZ, moderate entropy)
    ideal_rect = mpatches.Rectangle((0.25, 0.01), 0.3, 0.09,
                                      linewidth=2, edgecolor='green',
                                      facecolor='lightgreen', alpha=0.3, zorder=1)
    ax1.add_patch(ideal_rect)
    ax1.text(0.4, 0.055, 'HSI\nIdeal', fontsize=10, ha='center', va='center',
             color='darkgreen', fontweight='bold')

    ax1.set_xlabel('Shannon Entropy (8-bit normalized)', fontsize=11)
    ax1.set_ylabel('Lempel-Ziv Complexity', fontsize=11)
    ax1.set_title('Entropy vs LZ Complexity', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, max(lzs) * 1.2 if max(lzs) > 0.1 else 0.7)

    # Plot 2: Entropy vs Compression Ratio
    ax2 = axes[0, 1]
    for i, var in enumerate(variants):
        ax2.scatter(entropies[i], compresses[i], s=200, c=colors.get(var, '#999999'),
                   label=f'Variant {var}', edgecolors='black', linewidths=1.5)
        ax2.annotate(var, (entropies[i], compresses[i]), fontsize=12, fontweight='bold',
                    ha='center', va='center', color='white')

    ax2.set_xlabel('Shannon Entropy (8-bit normalized)', fontsize=11)
    ax2.set_ylabel('Compression Ratio (lower = more structure)', fontsize=11)
    ax2.set_title('Entropy vs Compressibility', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Plot 3: LZ vs HSI Score
    ax3 = axes[1, 0]
    for i, var in enumerate(variants):
        ax3.scatter(lzs[i], scores[i], s=200, c=colors.get(var, '#999999'),
                   label=f'Variant {var}', edgecolors='black', linewidths=1.5)
        ax3.annotate(var, (lzs[i], scores[i]), fontsize=12, fontweight='bold',
                    ha='center', va='center', color='white')

    ax3.set_xlabel('Lempel-Ziv Complexity', fontsize=11)
    ax3.set_ylabel('HSI Score', fontsize=11)
    ax3.set_title('LZ Complexity vs HSI Score', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Threshold')

    # Plot 4: Bar chart of HSI Scores
    ax4 = axes[1, 1]
    bar_colors = [colors.get(v, '#999999') for v in variants]
    bars = ax4.bar(variants, scores, color=bar_colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax4.annotate(f'{score:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax4.set_xlabel('Variant', fontsize=11)
    ax4.set_ylabel('HSI Score', fontsize=11)
    ax4.set_title('HSI Score Ranking', fontsize=12)
    ax4.set_ylim(0, 1.1)
    ax4.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7)
    ax4.grid(True, alpha=0.3, axis='y')

    # Adjust layout
    plt.tight_layout()

    # Save
    iteration = results.get('iteration', 0)
    variant_str = '_'.join(sorted(variants))
    plot_path = output_dir / f"complexity_plot_{variant_str}_iter{iteration}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return plot_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🧬 HSI Level 2 - Algorithmic Complexity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python level2_complexity_analysis.py --variant B --iteration 18
    python level2_complexity_analysis.py --variants A B D E --iteration 18
    python level2_complexity_analysis.py --variants A B --iteration 15 --max-bits 100000
        """
    )

    parser.add_argument('-v', '--variant', type=str,
                        help='Single variant to analyze')
    parser.add_argument('--variants', nargs='+', type=str,
                        help='Multiple variants to compare')
    parser.add_argument('-i', '--iteration', type=int, default=18,
                        help='Iteration to analyze (default: 18)')
    parser.add_argument('--max-bits', type=int, default=500_000,
                        help='Maximum bits to analyze per variant (default: 500000)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed progress for each variant')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to JSON')

    args = parser.parse_args()

    if args.variant:
        # Single variant analysis
        analyze_variant_complexity(
            args.variant,
            args.iteration,
            verbose=True,
            max_bits=args.max_bits
        )
    elif args.variants:
        # Multi-variant comparison
        compare_variants_complexity(
            args.variants,
            args.iteration,
            verbose=args.verbose,
            max_bits=args.max_bits,
            save_results=not args.no_save
        )
    else:
        # Default: compare main candidates
        print("No variants specified. Running default comparison (A, B, D, E)...")
        compare_variants_complexity(
            ['A', 'B', 'D', 'E'],
            args.iteration,
            verbose=args.verbose,
            max_bits=args.max_bits,
            save_results=not args.no_save
        )


if __name__ == '__main__':
    main()

