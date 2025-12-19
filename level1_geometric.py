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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

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

    # Analyze several pattern types
    patterns = ['00', '11', '01', '10', '000', '111', '010', '101',
                '0000', '1111', '0101', '1010']

    results = {
        'n_segments': n_segments,
        'segment_size': segment_size,
        'patterns': {}
    }

    torsions = []

    total_patterns = len(patterns)
    for idx, pattern in enumerate(patterns):
        if verbose:
            log_progress(idx + 1, total_patterns, "Patterns")

        counts = []
        for seg in range(n_segments):
            start = seg * segment_size
            end = (seg + 1) * segment_size
            segment = bits[start:end]

            count = 0
            for i in range(len(segment) - len(pattern) + 1):
                if segment[i:i+len(pattern)] == pattern:
                    count += 1
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
    if verbose:
        print(f"\n{'='*65}")
        print(f"📐 GEOMETRIC OPERATORS ANALYSIS")
        print(f"   Variant {variant}, Iteration {iteration}")
        print(f"   Mode: {'Torsion-only (𝓣)' if torsion_only else 'Full (Γ, 𝓡, 𝓣, 𝓔)'}")
        print(f"{'='*65}")

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

    # Load raw phi data for Torsion
    if verbose:
        print(f"\n{'─'*50}")
        log_step("📊 Loading phi data...")

    from level1.data_loader import load_phi_for_level1

    _, phi_bits, _ = load_phi_for_level1(
        req['phi_data_dir'], iteration,
        return_structural=False,
        return_observable=True,
        return_metadata=True
    )

    if verbose:
        log_step(f"Loaded {len(phi_bits):,} bits", 1)

    # Torsion (always computed)
    if verbose:
        print(f"\n{'─'*50}")
        log_step("📐 STEP 1: Torsion 𝓣 (positional asymmetries)")

    results['torsion'] = compute_torsion(phi_bits[:max_bits], verbose=verbose)

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
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    if args.check_only:
        check_requirements(args.variant, args.iteration, args.torsion_only, verbose=True)
        return

    run_geometric_analysis(
        args.variant,
        args.iteration,
        torsion_only=args.torsion_only,
        max_bits=args.max_bits,
        output_path=args.output
    )


if __name__ == "__main__":
    main()

