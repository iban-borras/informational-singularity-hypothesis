#!/usr/bin/env python3
"""
Level 1 — Geometric Operators Analysis (Γ, 𝓡, 𝓣, 𝓔)

Implements informational geometry operators from ISH paper Section 4.3:
- Γ (Connection): Pattern transport between iterations
- 𝓡 (Curvature): Rule interference / non-commutativity
- 𝓣 (Torsion): Positional asymmetries within single iteration
- 𝓔 (Energy): Global quality functional

Requirements:
    For FULL analysis (Γ, 𝓡, 𝓔):
        - Level 1 analysis results for 3 consecutive iterations (N-2, N-1, N)
        - Files: results/level1_analysis_var{X}_iter{Y}_min*_max*.json

    For TORSION-only (𝓣):
        - phi_snapshots data for single iteration
        - No Level 1 pre-processing required

Usage:
    # Full analysis (requires Level 1 results)
    python level1_geometric.py --variant B --iteration 17

    # Torsion-only (works with raw phi data)
    python level1_geometric.py --variant B --iteration 15 --torsion-only

    # Check requirements
    python level1_geometric.py --variant B --iteration 17 --check-only

Author: Sophia (Augment Agent) for Iban's HSI Project
"""

import argparse
import glob
import json
import math
import os
import re
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

PHI = (1 + math.sqrt(5)) / 2


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
# Requirement Checking
# =============================================================================

def find_level1_result(variant: str, iteration: int, results_dir: str = "results") -> Optional[str]:
    """Find Level 1 analysis result file for given variant/iteration."""
    pattern = f"{results_dir}/level1_analysis_var{variant}_iter{iteration}_min*_max*.json"
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def check_requirements(variant: str, iteration: int, torsion_only: bool = False,
                       verbose: bool = True) -> Dict:
    """
    Check data requirements for geometric analysis.
    
    Returns dict with:
        - 'ready': bool - all requirements met
        - 'level1_files': dict mapping iteration to file path or None
        - 'phi_data_exists': bool - raw phi data available
        - 'missing': list of missing items
    """
    result = {
        'ready': False,
        'level1_files': {},
        'phi_data_exists': False,
        'missing': [],
        'torsion_only': torsion_only
    }
    
    if verbose:
        print(f"\n{'='*65}")
        print(f"📋 REQUIREMENT CHECK — Variant {variant}, Iteration {iteration}")
        print(f"{'='*65}")
    
    # Check raw phi data
    phi_dirs = [
        f"results/level0/phi_snapshots/var_{variant}",
        f"results/level0/var_{variant}"
    ]
    for phi_dir in phi_dirs:
        if os.path.exists(phi_dir):
            result['phi_data_exists'] = True
            result['phi_data_dir'] = phi_dir
            break
    
    if verbose:
        status = "✅" if result['phi_data_exists'] else "❌"
        print(f"\n   {status} Raw phi_snapshots data: {'Found' if result['phi_data_exists'] else 'NOT FOUND'}")
        if result['phi_data_exists']:
            print(f"      Location: {result.get('phi_data_dir', 'N/A')}")
    
    if torsion_only:
        # Only need raw phi data
        result['ready'] = result['phi_data_exists']
        if not result['ready']:
            result['missing'].append(f"phi_snapshots for variant {variant}")
    else:
        # Need Level 1 results for 3 consecutive iterations
        iterations_needed = [iteration - 2, iteration - 1, iteration]
        
        if verbose:
            print(f"\n   Level 1 analysis results (for Γ, 𝓡, 𝓔):")
        
        all_found = True
        for it in iterations_needed:
            path = find_level1_result(variant, it)
            result['level1_files'][it] = path
            
            if verbose:
                status = "✅" if path else "❌"
                if path:
                    print(f"      {status} Iteration {it}: {os.path.basename(path)}")
                else:
                    print(f"      {status} Iteration {it}: NOT FOUND")
            
            if not path:
                all_found = False
                result['missing'].append(f"level1_analysis_var{variant}_iter{it}_min*_max*.json")
        
        result['ready'] = all_found and result['phi_data_exists']
    
    # Summary
    if verbose:
        print(f"\n{'─'*50}")
        if result['ready']:
            print(f"   ✅ READY for {'Torsion-only' if torsion_only else 'Full'} analysis")
        else:
            print(f"   ❌ NOT READY — Missing requirements:")
            for m in result['missing']:
                print(f"      • {m}")
            
            print(f"\n   📋 TO GENERATE MISSING DATA:")
            if not result['phi_data_exists']:
                print(f"      python level0_generate.py --variant {variant} --iterations {iteration}")
            
            if not torsion_only:
                for it in [iteration - 2, iteration - 1, iteration]:
                    if not result['level1_files'].get(it):
                        print(f"      python level1_analyze_patterns.py -v {variant} -i {it} --report")
            
            print(f"\n   💡 ALTERNATIVE: Run Torsion-only analysis (single iteration):")
            print(f"      python level1_geometric.py -v {variant} -i {iteration} --torsion-only")

    return result


# =============================================================================
# Torsion 𝓣 — Positional Asymmetries (Single Iteration)
# =============================================================================

@njit(cache=True)
def _count_pattern_in_segment_numba(arr: np.ndarray, pattern: np.ndarray) -> int:
    """Numba-optimized pattern counting in a segment."""
    n = len(arr)
    p_len = len(pattern)
    if n < p_len:
        return 0

    count = 0
    for i in range(n - p_len + 1):
        match = True
        for j in range(p_len):
            if arr[i + j] != pattern[j]:
                match = False
                break
        if match:
            count += 1
    return count


def compute_torsion(bits: str, n_segments: int = 5, verbose: bool = True) -> Dict:
    """
    Compute Torsion 𝓣 — measures positional asymmetries.

    From paper: 𝓣_α^(k)(P) = Γ_α^(k)(P) - Γ_α^(k)(P^shift)

    Operational interpretation:
    - Divide Φ into n segments
    - Count pattern frequencies in each segment
    - Torsion = variance of pattern distribution across segments

    High torsion = patterns prefer certain positions (asymmetric)
    Low torsion = patterns uniformly distributed (symmetric)

    Args:
        bits: Binary string
        n_segments: Number of segments to divide data
        verbose: Print progress

    Returns:
        Torsion analysis results
    """
    if verbose:
        log_step(f"Computing Torsion 𝓣 ({n_segments} segments)...")

    n = len(bits)
    segment_size = n // n_segments

    if segment_size < 100:
        return {'error': 'Insufficient data for torsion analysis'}

    # Convert to numpy array once
    arr = np.frombuffer(bits.encode('ascii'), dtype=np.uint8) - ord('0')

    # Analyze several pattern types
    patterns = ['00', '11', '01', '10', '000', '111', '010', '101',
                '0000', '1111', '0101', '1010']

    results = {
        'n_segments': n_segments,
        'segment_size': segment_size,
        'patterns': {}
    }

    torsions = []

    # Use tqdm if available
    if verbose and TQDM_AVAILABLE:
        pattern_iter = tqdm(patterns, desc="   Patterns", unit="pat",
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        pattern_iter = patterns

    for idx, pattern in enumerate(pattern_iter):
        if not TQDM_AVAILABLE and verbose:
            log_progress(idx + 1, len(patterns), "Patterns")

        # Convert pattern to numpy array
        pattern_arr = np.array([int(c) for c in pattern], dtype=np.uint8)

        counts = []
        for seg in range(n_segments):
            start = seg * segment_size
            end = (seg + 1) * segment_size
            segment_arr = arr[start:end]

            # Use Numba-optimized counting
            count = _count_pattern_in_segment_numba(segment_arr, pattern_arr)
            counts.append(count)

        mean_count = np.mean(counts)
        var_count = np.var(counts)

        # Torsion = coefficient of variation (normalized variance)
        torsion = var_count / (mean_count + 1e-10)
        torsions.append(torsion)

        results['patterns'][pattern] = {
            'counts_by_segment': counts,
            'mean': float(mean_count),
            'variance': float(var_count),
            'torsion': float(torsion)
        }

    # Aggregate torsion
    results['mean_torsion'] = float(np.mean(torsions))
    results['max_torsion'] = float(np.max(torsions))
    results['min_torsion'] = float(np.min(torsions))

    # Interpretation
    if results['mean_torsion'] < 0.1:
        results['interpretation'] = "LOW — Symmetric, patterns uniformly distributed"
    elif results['mean_torsion'] < 0.5:
        results['interpretation'] = "MODERATE — Some positional preferences"
    else:
        results['interpretation'] = "HIGH — Strong positional asymmetries"

    # Check for φ in torsion ratios between patterns
    sorted_torsions = sorted(torsions, reverse=True)
    torsion_ratios = []
    for i in range(len(sorted_torsions) - 1):
        if sorted_torsions[i + 1] > 0.001:
            torsion_ratios.append(sorted_torsions[i] / sorted_torsions[i + 1])

    phi_distances = [min(abs(r - PHI), abs(r - 1/PHI)) for r in torsion_ratios]
    results['torsion_ratios'] = torsion_ratios[:5]
    results['phi_distances'] = phi_distances[:5]
    results['mean_phi_distance'] = float(np.mean(phi_distances)) if phi_distances else float('nan')
    results['phi_detected'] = results['mean_phi_distance'] < 0.3 if phi_distances else False

    return results


# =============================================================================
# Connection Γ — Pattern Transport Between Iterations
# =============================================================================

def load_level1_patterns(filepath: str) -> Dict:
    """Load patterns from Level 1 analysis result."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('patterns', data.get('significant_patterns', {}))


def compute_connection(level1_files: Dict[int, str], verbose: bool = True) -> Dict:
    """
    Compute Connection Γ — pattern transport between iterations.

    From paper: Γ_α^(k) : S(R_α) → S(R_{α+1})

    Measures how patterns "travel" from one state to the next:
    - Survival rate: what fraction of patterns persist
    - Frequency change: how pattern frequencies evolve
    - New patterns: what emerges at each iteration

    Args:
        level1_files: Dict mapping iteration to Level 1 result file path
        verbose: Print progress

    Returns:
        Connection analysis results
    """
    if verbose:
        log_step("Computing Connection Γ (pattern transport)...")

    iterations = sorted(level1_files.keys())
    if len(iterations) < 2:
        return {'error': 'Need at least 2 iterations for connection analysis'}

    # Load patterns from each iteration
    patterns_by_iter = {}
    for it in iterations:
        if verbose:
            log_step(f"Loading iteration {it}...", 1)
        patterns_by_iter[it] = load_level1_patterns(level1_files[it])

    results = {
        'iterations': iterations,
        'transitions': []
    }

    # Analyze transitions between consecutive iterations
    for i in range(len(iterations) - 1):
        it_from = iterations[i]
        it_to = iterations[i + 1]

        if verbose:
            log_step(f"Analyzing transition {it_from} → {it_to}...", 1)

        patterns_from = set(patterns_by_iter[it_from].keys())
        patterns_to = set(patterns_by_iter[it_to].keys())

        # Survival: patterns that exist in both
        survived = patterns_from & patterns_to
        survival_rate = len(survived) / len(patterns_from) if patterns_from else 0

        # New patterns
        new_patterns = patterns_to - patterns_from

        # Lost patterns
        lost_patterns = patterns_from - patterns_to

        # Frequency transport for survived patterns
        freq_ratios = []
        for p in survived:
            freq_from = patterns_by_iter[it_from][p].get('frequency',
                        patterns_by_iter[it_from][p].get('count', 1))
            freq_to = patterns_by_iter[it_to][p].get('frequency',
                      patterns_by_iter[it_to][p].get('count', 1))
            if freq_from > 0:
                freq_ratios.append(freq_to / freq_from)

        transition = {
            'from': it_from,
            'to': it_to,
            'patterns_from': len(patterns_from),
            'patterns_to': len(patterns_to),
            'survived': len(survived),
            'survival_rate': float(survival_rate),
            'new_patterns': len(new_patterns),
            'lost_patterns': len(lost_patterns),
            'mean_freq_ratio': float(np.mean(freq_ratios)) if freq_ratios else float('nan')
        }

        results['transitions'].append(transition)

    # Aggregate metrics
    survival_rates = [t['survival_rate'] for t in results['transitions']]
    results['mean_survival_rate'] = float(np.mean(survival_rates))

    # Check for φ in survival rates
    phi_dist = min(abs(results['mean_survival_rate'] - 1/PHI),
                   abs(results['mean_survival_rate'] - (1 - 1/PHI)))
    results['survival_phi_distance'] = float(phi_dist)
    results['phi_in_survival'] = phi_dist < 0.1

    return results


# =============================================================================
# Curvature 𝓡 — Rule Interference / Non-commutativity
# =============================================================================

def compute_curvature(level1_files: Dict[int, str], verbose: bool = True) -> Dict:
    """
    Compute Curvature 𝓡 — measures rule interference.

    From paper: 𝓡_α^(i,j)(P) = Γ_{α+1}^(j)(Γ_α^(i)(P)) - Γ_{α+1}^(i)(Γ_α^(j)(P))

    Interpretation:
    - If 𝓡 ≈ 0: Transport is path-independent (flat geometry)
    - If 𝓡 ≠ 0: Order matters, curvature exists

    Operational: Compare how patterns evolve through different rule orderings.

    Args:
        level1_files: Dict mapping iteration to Level 1 result file path
        verbose: Print progress

    Returns:
        Curvature analysis results
    """
    if verbose:
        log_step("Computing Curvature 𝓡 (rule interference)...")

    iterations = sorted(level1_files.keys())
    if len(iterations) < 3:
        return {'error': 'Need at least 3 iterations for curvature analysis'}

    # Load patterns and rules from each iteration
    data_by_iter = {}
    for it in iterations:
        with open(level1_files[it], 'r', encoding='utf-8') as f:
            data_by_iter[it] = json.load(f)

    it0, it1, it2 = iterations[0], iterations[1], iterations[2]

    # Get patterns at each iteration
    def get_pattern_set(data):
        patterns = data.get('patterns', data.get('significant_patterns', {}))
        return set(patterns.keys())

    p0 = get_pattern_set(data_by_iter[it0])
    p1 = get_pattern_set(data_by_iter[it1])
    p2 = get_pattern_set(data_by_iter[it2])

    # Curvature metric: how differently patterns evolve through iterations
    # Path A: patterns that survive 0→1→2
    path_a = p0 & p1 & p2

    # Alternative: patterns that appear/disappear asymmetrically
    gained_at_1 = p1 - p0
    gained_at_2 = p2 - p1
    lost_at_1 = p0 - p1
    lost_at_2 = p1 - p2

    # Curvature = asymmetry in pattern flow
    total_changes = len(gained_at_1) + len(gained_at_2) + len(lost_at_1) + len(lost_at_2)
    if total_changes == 0:
        curvature = 0.0
    else:
        # Non-commutativity: difference between "gain then lose" vs "lose then gain"
        gain_momentum = len(gained_at_1) - len(lost_at_1)
        later_momentum = len(gained_at_2) - len(lost_at_2)
        curvature = abs(gain_momentum - later_momentum) / total_changes

    results = {
        'iterations': iterations,
        'patterns_at_0': len(p0),
        'patterns_at_1': len(p1),
        'patterns_at_2': len(p2),
        'survived_all': len(path_a),
        'gained_at_1': len(gained_at_1),
        'gained_at_2': len(gained_at_2),
        'lost_at_1': len(lost_at_1),
        'lost_at_2': len(lost_at_2),
        'curvature': float(curvature)
    }

    # Interpretation
    if curvature < 0.1:
        results['interpretation'] = "FLAT — Path-independent, symmetric evolution"
    elif curvature < 0.3:
        results['interpretation'] = "CURVED — Moderate path-dependence"
    else:
        results['interpretation'] = "HIGHLY CURVED — Strong rule interference"

    # φ proximity
    phi_dist = min(abs(curvature - 1/PHI), abs(curvature - (PHI - 1)))
    results['phi_distance'] = float(phi_dist)
    results['phi_detected'] = phi_dist < 0.1

    return results


# =============================================================================
# Energy 𝓔 — Global Quality Functional
# =============================================================================

def compute_energy(torsion: Dict, connection: Dict, curvature: Dict,
                   alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0,
                   verbose: bool = True) -> Dict:
    """
    Compute Energy Functional 𝓔 — global quality metric.

    From paper: 𝓔[ω] = α|𝓡[ω]|² + β|𝓣[ω]|² + γ·Φ_incoh[ω]

    Lower energy = more stable, coherent configuration
    Higher energy = unstable, incoherent

    Args:
        torsion: Torsion analysis results
        connection: Connection analysis results
        curvature: Curvature analysis results
        alpha, beta, gamma: Weighting coefficients
        verbose: Print progress

    Returns:
        Energy analysis results
    """
    if verbose:
        log_step("Computing Energy Functional 𝓔...")

    # Extract values (with defaults for errors)
    R = curvature.get('curvature', 0) if 'error' not in curvature else 0
    T = torsion.get('mean_torsion', 0) if 'error' not in torsion else 0

    # Incoherence: inverse of survival rate (low survival = high incoherence)
    survival = connection.get('mean_survival_rate', 0.5) if 'error' not in connection else 0.5
    incoherence = 1 - survival

    # Energy functional
    curvature_term = alpha * (R ** 2)
    torsion_term = beta * (T ** 2)
    incoherence_term = gamma * incoherence

    total_energy = curvature_term + torsion_term + incoherence_term

    results = {
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'curvature_term': float(curvature_term),
        'torsion_term': float(torsion_term),
        'incoherence_term': float(incoherence_term),
        'total_energy': float(total_energy)
    }

    # Interpretation
    if total_energy < 0.5:
        results['stability'] = "STABLE — Low energy, coherent configuration"
    elif total_energy < 1.0:
        results['stability'] = "MODERATE — Balanced energy"
    else:
        results['stability'] = "UNSTABLE — High energy, incoherent"

    # φ proximity in energy
    phi_dist = min(abs(total_energy - PHI), abs(total_energy - 1/PHI))
    results['phi_distance'] = float(phi_dist)

    return results


# =============================================================================
# Main Analysis Runner
# =============================================================================

def run_geometric_analysis(
    variant: str,
    iteration: int,
    torsion_only: bool = False,
    max_bits: int = 100000,
    verbose: bool = True,
    output_path: Optional[str] = None
) -> Dict:
    """
    Run geometric operators analysis.

    Args:
        variant: Variant code (B, F, etc.)
        iteration: Target iteration
        torsion_only: If True, only compute Torsion (single iteration)
        max_bits: Max bits for Torsion analysis
        verbose: Print progress
        output_path: Optional path to save JSON results

    Returns:
        Complete geometric analysis results
    """
    start_time = time.time()

    if verbose:
        print(f"\n{'='*65}")
        print(f"📐 GEOMETRIC OPERATORS ANALYSIS")
        print(f"{'='*65}")
        print(f"\n📋 CONFIGURATION:")
        print(f"   - Variant: {variant}, Iteration: {iteration}")
        print(f"   - Mode: {'Torsion-only (𝓣)' if torsion_only else 'Full (Γ, 𝓡, 𝓣, 𝓔)'}")
        print(f"   - Max bits: {max_bits:,}")
        print(f"   - JIT acceleration: {'Numba ✅' if NUMBA_AVAILABLE else 'Not available ❌'}")
        print(f"   - Progress bars: {'tqdm ✅' if TQDM_AVAILABLE else 'Basic'}")

    # Check requirements
    req = check_requirements(variant, iteration, torsion_only, verbose)

    if not req['ready']:
        return {'error': 'Requirements not met', 'requirements': req}

    results = {
        'variant': variant,
        'iteration': iteration,
        'mode': 'torsion_only' if torsion_only else 'full',
        'phi_target': PHI
    }

    # Load raw phi data for Torsion using streaming loader
    if verbose:
        print(f"\n{'─'*50}")
        log_step("📥 Loading phi data...")

    # Try streaming loader first (memory efficient)
    struct_path = Path(req['phi_data_dir']) / f"phi_iter{iteration}.struct.gz"

    try:
        if struct_path.exists():
            from utils.streaming_phi_loader import load_phi_for_agents
            phi_bits, _ = load_phi_for_agents(
                str(struct_path),
                max_chars=max_bits * 2,  # Safety margin for parentheses
                observable_only=True,
                show_progress=verbose
            )
            phi_bits = phi_bits[:max_bits]
        else:
            # Fallback to standard loader
            from level1.data_loader import load_phi_for_level1
            _, phi_bits, _ = load_phi_for_level1(
                req['phi_data_dir'], iteration,
                return_structural=False,
                return_observable=True,
                return_metadata=True
            )
            phi_bits = phi_bits[:max_bits]
    except Exception as e:
        print(f"\n❌ FAILED TO LOAD DATA: {e}")
        return {'error': str(e)}

    load_time = time.time() - start_time

    if verbose:
        log_step(f"✅ Loaded {len(phi_bits):,} bits in {load_time:.1f}s")

    # Torsion (always computed)
    if verbose:
        print(f"\n{'─'*50}")
        log_step("📐 STEP 1: Torsion 𝓣 (positional asymmetries)")

    results['torsion'] = compute_torsion(phi_bits, verbose=verbose)

    if not torsion_only:
        # Connection
        if verbose:
            print(f"\n{'─'*50}")
            log_step("📐 STEP 2: Connection Γ (pattern transport)")

        results['connection'] = compute_connection(req['level1_files'], verbose)

        # Curvature
        if verbose:
            print(f"\n{'─'*50}")
            log_step("📐 STEP 3: Curvature 𝓡 (rule interference)")

        results['curvature'] = compute_curvature(req['level1_files'], verbose)

        # Energy
        if verbose:
            print(f"\n{'─'*50}")
            log_step("📐 STEP 4: Energy 𝓔 (global functional)")

        results['energy'] = compute_energy(
            results['torsion'],
            results['connection'],
            results['curvature'],
            verbose=verbose
        )

    # Summary
    if verbose:
        print(f"\n{'='*65}")
        print(f"📊 SUMMARY — Geometric Operators")
        print(f"{'='*65}")

        t = results.get('torsion', {})
        print(f"   𝓣 (Torsion):    {t.get('mean_torsion', 'N/A'):.4f} — {t.get('interpretation', 'N/A')}")

        if not torsion_only:
            c = results.get('connection', {})
            r = results.get('curvature', {})
            e = results.get('energy', {})

            print(f"   Γ (Connection): survival={c.get('mean_survival_rate', 'N/A'):.3f}")
            print(f"   𝓡 (Curvature):  {r.get('curvature', 'N/A'):.4f} — {r.get('interpretation', 'N/A')}")
            print(f"   𝓔 (Energy):     {e.get('total_energy', 'N/A'):.4f} — {e.get('stability', 'N/A')}")

        # φ detection summary
        print(f"\n   φ Detection:")
        t_phi = "✅" if t.get('phi_detected') else "❌"
        print(f"      Torsion φ: {t_phi} (dist={t.get('mean_phi_distance', float('nan')):.3f})")

        if not torsion_only:
            c_phi = "✅" if c.get('phi_in_survival') else "❌"
            r_phi = "✅" if r.get('phi_detected') else "❌"
            print(f"      Survival φ: {c_phi} (dist={c.get('survival_phi_distance', float('nan')):.3f})")
            print(f"      Curvature φ: {r_phi} (dist={r.get('phi_distance', float('nan')):.3f})")

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


def run_scaling_analysis(variant: str, iteration: int, scales_str: str, output_path: Optional[str] = None) -> Dict:
    """
    Run torsion at multiple scales to compute power-law exponent α.

    If Torsion ∝ N^α:
      - α = 1.0: Linear scaling (random noise)
      - α > 1.0: Superlinear (long-range correlations, fractal structure)
      - α < 1.0: Sublinear (self-correction)
    """
    # Parse scales (e.g., "10M,50M,100M" -> [10000000, 50000000, 100000000])
    scales = []
    for s in scales_str.split(','):
        s = s.strip().upper()
        if s.endswith('M'):
            scales.append(int(float(s[:-1]) * 1_000_000))
        elif s.endswith('K'):
            scales.append(int(float(s[:-1]) * 1_000))
        else:
            scales.append(int(s))
    scales = sorted(scales)

    print(f"\n{'='*70}")
    print(f"📐 SCALING ANALYSIS — Torsion vs Data Size")
    print(f"{'='*70}")
    print(f"   Variant: {variant}, Iteration: {iteration}")
    print(f"   Scales: {', '.join(f'{s//1_000_000}M' for s in scales)}")
    print(f"{'='*70}\n")

    # Check requirements
    req = check_requirements(variant, iteration, torsion_only=True, verbose=False)
    if not req['ready']:
        print("   ❌ Requirements not met. Run with --check-only to see details.")
        return {}

    # Load max data needed (largest scale)
    max_scale = max(scales)
    print(f"   📥 Loading {max_scale:,} bits...")

    struct_path = Path(req['phi_data_dir']) / f"phi_iter{iteration}.struct.gz"
    try:
        if struct_path.exists():
            from utils.streaming_phi_loader import load_phi_for_agents
            bits, _ = load_phi_for_agents(
                str(struct_path),
                max_chars=max_scale * 2,
                observable_only=True,
                show_progress=True
            )
            bits = bits[:max_scale]
        else:
            from level1.data_loader import load_phi_for_level1
            _, bits, _ = load_phi_for_level1(
                req['phi_data_dir'], iteration,
                return_structural=False,
                return_observable=True,
                return_metadata=True
            )
            bits = bits[:max_scale]
    except Exception as e:
        print(f"   ❌ Failed to load data: {e}")
        return {}

    print(f"   ✅ Loaded {len(bits):,} bits\n")

    # Measure torsion at each scale
    results = {'variant': variant, 'iteration': iteration, 'measurements': []}

    for scale in scales:
        print(f"\n   📏 Scale: {scale:,} bits ({scale//1_000_000}M)")
        subset = bits[:scale]
        torsion_result = compute_torsion(subset, n_segments=5, verbose=False)
        torsion = torsion_result.get('mean_torsion', 0)
        results['measurements'].append({'bits': scale, 'torsion': torsion})
        print(f"      𝓣 = {torsion:.4f}")

    # Fit power law: log(T) = α * log(N) + c
    if len(results['measurements']) >= 2:
        log_n = np.array([np.log(m['bits']) for m in results['measurements']])
        log_t = np.array([np.log(m['torsion']) if m['torsion'] > 0 else 0 for m in results['measurements']])

        # Simple least squares: α = cov(log_n, log_t) / var(log_n)
        alpha = np.cov(log_n, log_t)[0, 1] / np.var(log_n)
        c = np.mean(log_t) - alpha * np.mean(log_n)

        # R² for fit quality
        predicted = alpha * log_n + c
        ss_res = np.sum((log_t - predicted) ** 2)
        ss_tot = np.sum((log_t - np.mean(log_t)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        results['power_law'] = {'alpha': float(alpha), 'c': float(c), 'r_squared': float(r_squared)}

        print(f"\n{'='*70}")
        print(f"📊 POWER LAW FIT: Torsion ∝ N^α")
        print(f"{'='*70}")
        print(f"   α (exponent) = {alpha:.4f}")
        print(f"   R² (fit)     = {r_squared:.4f}")
        print(f"\n   Interpretation:")
        if alpha > 1.1:
            print(f"   🔥 SUPERLINEAR (α > 1): Long-range correlations detected!")
            print(f"      Torsion grows faster than data size → emergent structure")
        elif alpha < 0.9:
            print(f"   📉 SUBLINEAR (α < 1): Self-correcting behavior")
        else:
            print(f"   ➡️  LINEAR (α ≈ 1): Consistent with random/independent structure")
        print(f"{'='*70}\n")

    # Save results
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"📁 Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Geometric Operators Analysis — Γ, 𝓡, 𝓣, 𝓔"
    )
    parser.add_argument("--variant", "-v", default="B", help="Variant (B, D, E, F, G, H, I)")
    parser.add_argument("--iteration", "-i", type=int, default=17, help="Iteration number")
    parser.add_argument("--torsion-only", "-t", action="store_true",
                        help="Only compute Torsion (single iteration, no Level 1 needed)")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check requirements, don't run analysis")
    parser.add_argument("--max-bits", "-m", type=int, default=100000,
                        help="Max bits for Torsion analysis")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file (auto-generated if not specified)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--scaling-analysis", "-s", action="store_true",
                        help="Run torsion at multiple scales to compute power-law exponent α")
    parser.add_argument("--scales", type=str, default="10M,50M,100M,200M,500M",
                        help="Comma-separated scales for scaling analysis (e.g., '10M,50M,100M,500M')")

    args = parser.parse_args()

    if args.check_only:
        check_requirements(args.variant, args.iteration, args.torsion_only, verbose=True)
        return

    # Scaling analysis mode
    if args.scaling_analysis:
        output_path = args.output
        if not args.no_save and output_path is None:
            output_dir = Path("results/level1/analysis")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"scaling_analysis_var_{args.variant}_iter{args.iteration}.json")

        run_scaling_analysis(args.variant, args.iteration, args.scales, output_path)
        return

    # Auto-generate output path if not specified
    output_path = args.output
    if not args.no_save and output_path is None:
        output_dir = Path("results/level1/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        bits_suffix = f"{args.max_bits // 1_000_000}M" if args.max_bits >= 1_000_000 else f"{args.max_bits // 1000}K"
        mode_suffix = "torsion" if args.torsion_only else "full"
        output_path = str(output_dir / f"geometric_{mode_suffix}_var_{args.variant}_iter{args.iteration}_{bits_suffix}.json")

    run_geometric_analysis(
        args.variant,
        args.iteration,
        torsion_only=args.torsion_only,
        max_bits=args.max_bits,
        output_path=output_path
    )


if __name__ == "__main__":
    main()

