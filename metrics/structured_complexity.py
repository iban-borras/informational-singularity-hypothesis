"""
🌌 Structured Complexity Index (SCI) & Cosmological Coherence Index (ICC)

Advanced metrics for evaluating HSI variants' potential to generate
universe-like structures with physical law emergence.

SCI (Structured Complexity Index):
- Measures whether a sequence has "meaningful" complexity vs empty randomness
- Combines: Hierarchy + Causality (Transfer Entropy) + NonTriviality

ICC (Índex de Coherència Cosmològica / Cosmological Coherence Index):
- Evaluates compatibility with physical universe generation
- Combines: Compressibility + Coherence + Hierarchy + Criticality + φ-Tendency

Key insight: Physical systems have COMPRESSIBLE LAWS but COMPLEX STATES.
Random noise is incompressible. Trivial order has no complexity.
The "sweet spot" for universe generation is structured complexity.

Author: Iban Borràs with Augment Agent (Sophia) collaboration
Date: December 2024
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from scipy.stats import linregress
import warnings


# =============================================================================
# TRANSFER ENTROPY (Causality Measure)
# =============================================================================

def calculate_transfer_entropy(sequence: str, 
                                history_length: int = 5,
                                max_samples: int = 100_000) -> Dict[str, Any]:
    """
    Calculate Transfer Entropy to measure temporal causality.
    
    Transfer Entropy measures how much the past helps predict the future.
    High TE = strong causal structure (past determines future)
    Low TE = memoryless sequence (past doesn't help)
    
    For binary sequences, we measure:
    TE = H(X_t | X_{t-1}^{t-k}) - H(X_t | X_{t-1}^{t-k}, context)
    
    Simplified as: How predictable is the next bit given history?
    
    Args:
        sequence: Binary string
        history_length: Number of past bits to consider (k)
        max_samples: Maximum samples for large sequences
        
    Returns:
        Dictionary with transfer_entropy, predictability, interpretation
    """
    n = len(sequence)
    if n < history_length + 2:
        return {
            'transfer_entropy': 0.0,
            'predictability': 0.5,
            'conditional_entropy': 1.0,
            'interpretation': 'Sequence too short'
        }
    
    # Sample if too long
    if n > max_samples:
        start = (n - max_samples) // 2
        seq = sequence[start:start + max_samples]
        n = max_samples
    else:
        seq = sequence
    
    # Count transitions: P(X_t | history)
    # history -> {0: count, 1: count}
    history_counts = {}
    
    for i in range(history_length, n):
        history = seq[i - history_length:i]
        next_bit = seq[i]
        
        if history not in history_counts:
            history_counts[history] = {'0': 0, '1': 0, 'total': 0}
        
        history_counts[history][next_bit] += 1
        history_counts[history]['total'] += 1
    
    # Calculate conditional entropy H(X_t | history)
    # H = -Σ P(history) * Σ P(x|history) * log2(P(x|history))
    total_samples = n - history_length
    conditional_entropy = 0.0
    
    for history, counts in history_counts.items():
        p_history = counts['total'] / total_samples
        
        for bit in ['0', '1']:
            if counts[bit] > 0:
                p_bit_given_history = counts[bit] / counts['total']
                conditional_entropy -= p_history * p_bit_given_history * np.log2(p_bit_given_history)
    
    # Unconditional entropy (baseline)
    count_1 = seq.count('1')
    p_1 = count_1 / n if n > 0 else 0.5
    p_0 = 1 - p_1
    
    if p_0 > 0 and p_1 > 0:
        unconditional_entropy = -p_0 * np.log2(p_0) - p_1 * np.log2(p_1)
    else:
        unconditional_entropy = 0.0
    
    # Transfer Entropy = reduction in uncertainty due to history
    transfer_entropy = max(0, unconditional_entropy - conditional_entropy)
    
    # Predictability: 1 - conditional_entropy (normalized)
    predictability = 1 - conditional_entropy if conditional_entropy <= 1 else 0
    
    return {
        'transfer_entropy': transfer_entropy,
        'conditional_entropy': conditional_entropy,
        'unconditional_entropy': unconditional_entropy,
        'predictability': predictability,
        'history_length': history_length,
        'unique_histories': len(history_counts),
        'interpretation': _interpret_transfer_entropy(transfer_entropy, predictability)
    }


def _interpret_transfer_entropy(te: float, pred: float) -> str:
    """Interpret Transfer Entropy results."""
    if te < 0.1:
        causality = "weak causality (nearly memoryless)"
    elif te < 0.3:
        causality = "moderate causality"
    elif te < 0.6:
        causality = "strong causality (past predicts future)"
    else:
        causality = "very strong causality (highly deterministic)"
    
    if pred > 0.8:
        predict = "highly predictable"
    elif pred > 0.5:
        predict = "moderately predictable"
    else:
        predict = "low predictability"
    
    return f"{causality}, {predict}"


# =============================================================================
# NON-TRIVIALITY SCORE
# =============================================================================

def calculate_nontriviality(lz_normalized: float,
                            pattern_diversity: float = None,
                            n_unique_patterns: int = None,
                            n_possible_patterns: int = None) -> Dict[str, Any]:
    """
    Calculate Non-Triviality Score.
    
    NonTriviality requires BOTH:
    1. Intermediate LZ complexity (not trivial order, not pure noise)
    2. Pattern diversity (many different patterns, not repetition)
    
    Formula:
    NonTriviality = LZ_factor × Diversity_factor
    
    Where:
    - LZ_factor = 1 - |2*LZ - 1| (peaks at LZ=0.5)
    - Diversity_factor = unique_patterns / max_possible (if available)
    
    Args:
        lz_normalized: Lempel-Ziv complexity (0-1)
        pattern_diversity: Pre-computed diversity (0-1) or None
        n_unique_patterns: Number of unique patterns found
        n_possible_patterns: Theoretical maximum patterns
        
    Returns:
        Dictionary with nontriviality score and components
    """
    # LZ factor: peaks at 0.5, zero at 0 and 1
    lz_factor = 1 - abs(2 * lz_normalized - 1)
    
    # Diversity factor
    if pattern_diversity is not None:
        diversity_factor = pattern_diversity
    elif n_unique_patterns is not None and n_possible_patterns is not None:
        diversity_factor = n_unique_patterns / max(1, n_possible_patterns)
    else:
        # Default: assume moderate diversity if not provided
        diversity_factor = 0.5
    
    # Combined score
    nontriviality = lz_factor * diversity_factor
    
    return {
        'nontriviality': nontriviality,
        'lz_factor': lz_factor,
        'diversity_factor': diversity_factor,
        'lz_normalized': lz_normalized,
        'interpretation': _interpret_nontriviality(nontriviality, lz_factor, diversity_factor)
    }


def _interpret_nontriviality(nt: float, lz_f: float, div_f: float) -> str:
    """Interpret NonTriviality score."""
    if nt < 0.2:
        return "trivial (too ordered or too random)"
    elif nt < 0.4:
        return "low non-triviality"
    elif nt < 0.6:
        return "moderate non-triviality (interesting structure)"
    elif nt < 0.8:
        return "high non-triviality (complex and diverse)"
    else:
        return "very high non-triviality (optimal for emergence)"


# =============================================================================
# φ-TENDENCY (Golden Ratio Exploration)
# =============================================================================

def calculate_phi_tendency(sequence: str,
                           max_samples: int = 100_000) -> Dict[str, Any]:
    """
    Explore whether φ (golden ratio) appears in the sequence structure.

    This is EXPLORATORY - we don't assume φ should appear.
    We're testing if it does, which would be significant.

    Methods:
    1. Ratio of consecutive segment lengths
    2. Spectral peaks at φ-related frequencies
    3. Bit density ratios in nested regions

    Args:
        sequence: Binary string
        max_samples: Maximum samples for analysis

    Returns:
        Dictionary with phi-related measurements
    """
    PHI = 1.618033988749895
    PHI_INV = 0.618033988749895  # 1/φ = φ-1

    n = len(sequence)
    if n < 100:
        return {
            'phi_detected': False,
            'phi_strength': 0.0,
            'interpretation': 'Sequence too short for φ analysis'
        }

    # Sample if too long
    if n > max_samples:
        start = (n - max_samples) // 2
        seq = sequence[start:start + max_samples]
        n = max_samples
    else:
        seq = sequence

    phi_signals = []

    # Method 1: Run-length ratios
    # Find runs of 0s and 1s, check if consecutive run ratios ≈ φ
    runs = []
    current_bit = seq[0]
    run_length = 1

    for i in range(1, n):
        if seq[i] == current_bit:
            run_length += 1
        else:
            runs.append(run_length)
            current_bit = seq[i]
            run_length = 1
    runs.append(run_length)

    if len(runs) >= 10:
        ratios = []
        for i in range(len(runs) - 1):
            if runs[i] > 0:
                ratio = runs[i + 1] / runs[i]
                ratios.append(ratio)

        if ratios:
            # Check proximity to φ or 1/φ
            phi_matches = sum(1 for r in ratios if abs(r - PHI) < 0.2 or abs(r - PHI_INV) < 0.2)
            phi_ratio_score = phi_matches / len(ratios)
            phi_signals.append(('run_ratios', phi_ratio_score))

    # Method 2: Segment density ratios
    # Split sequence into φ-proportioned segments and compare densities
    split_point = int(n * PHI_INV)  # φ-proportioned split
    seg1 = seq[:split_point]
    seg2 = seq[split_point:]

    if len(seg1) > 0 and len(seg2) > 0:
        density1 = seg1.count('1') / len(seg1)
        density2 = seg2.count('1') / len(seg2)

        if density1 > 0 and density2 > 0:
            density_ratio = max(density1, density2) / min(density1, density2)
            phi_density_score = max(0, 1 - abs(density_ratio - PHI) / PHI)
            phi_signals.append(('density_ratio', phi_density_score))

    # Method 3: Block entropy at φ-scaled sizes
    # Check if entropy changes at φ-related scales
    phi_scales = [int(10 * (PHI ** k)) for k in range(5) if int(10 * (PHI ** k)) < n // 4]

    if len(phi_scales) >= 2:
        scale_entropies = []
        for block_size in phi_scales:
            n_blocks = n // block_size
            if n_blocks < 4:
                continue
            block_counts = {}
            for i in range(n_blocks):
                block = seq[i * block_size:(i + 1) * block_size]
                block_counts[block] = block_counts.get(block, 0) + 1

            total = sum(block_counts.values())
            entropy = 0.0
            for count in block_counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)
            scale_entropies.append(entropy)

        if len(scale_entropies) >= 2:
            # Check if entropy follows φ-scaling
            entropy_ratios = [scale_entropies[i+1] / scale_entropies[i]
                             for i in range(len(scale_entropies)-1)
                             if scale_entropies[i] > 0]
            if entropy_ratios:
                phi_entropy_score = np.mean([max(0, 1 - abs(r - PHI_INV) / PHI_INV)
                                            for r in entropy_ratios])
                phi_signals.append(('entropy_scaling', phi_entropy_score))

    # Combine signals
    if phi_signals:
        phi_strength = np.mean([s[1] for s in phi_signals])
        phi_detected = phi_strength > 0.3
    else:
        phi_strength = 0.0
        phi_detected = False

    return {
        'phi_detected': phi_detected,
        'phi_strength': phi_strength,
        'phi_signals': {name: score for name, score in phi_signals},
        'phi_target': PHI,
        'interpretation': _interpret_phi_tendency(phi_detected, phi_strength)
    }


def _interpret_phi_tendency(detected: bool, strength: float) -> str:
    """Interpret φ-tendency results."""
    if not detected:
        if strength < 0.1:
            return "no φ signature detected (may emerge at higher levels)"
        else:
            return "weak φ signal (inconclusive)"
    else:
        if strength > 0.6:
            return "strong φ signature (golden ratio present in structure)"
        else:
            return "moderate φ signature (possible φ influence)"


# =============================================================================
# STRUCTURED COMPLEXITY INDEX (SCI)
# =============================================================================

def calculate_sci(hierarchy_score: float,
                  transfer_entropy: float,
                  nontriviality: float,
                  weights: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Calculate Structured Complexity Index (SCI).

    SCI distinguishes "meaningful" complexity from empty randomness.
    Physical systems need structure at multiple scales (hierarchy),
    causal relationships (transfer entropy), and non-trivial patterns.

    SCI = α·Hierarchy + β·Causality + γ·NonTriviality

    Args:
        hierarchy_score: Variance/Mean of multi-scale entropy (0-1)
        transfer_entropy: Temporal causality measure (0-1)
        nontriviality: Pattern diversity × LZ factor (0-1)
        weights: Custom weights {hierarchy, causality, nontriviality}

    Returns:
        Dictionary with SCI score and components
    """
    if weights is None:
        weights = {
            'hierarchy': 0.35,      # Structure at multiple scales
            'causality': 0.40,      # Past determines future (physics needs causality!)
            'nontriviality': 0.25   # Not trivial, not random
        }

    sci = (weights['hierarchy'] * hierarchy_score +
           weights['causality'] * transfer_entropy +
           weights['nontriviality'] * nontriviality)

    return {
        'sci': sci,
        'hierarchy_score': hierarchy_score,
        'causality_score': transfer_entropy,
        'nontriviality_score': nontriviality,
        'weights': weights,
        'interpretation': _interpret_sci(sci)
    }


def _interpret_sci(sci: float) -> str:
    """Interpret SCI score."""
    if sci < 0.2:
        return "low structured complexity (trivial or random)"
    elif sci < 0.4:
        return "moderate structured complexity"
    elif sci < 0.6:
        return "good structured complexity (candidate for emergence)"
    elif sci < 0.8:
        return "high structured complexity (strong emergence potential)"
    else:
        return "very high structured complexity (optimal for physical laws)"


# =============================================================================
# COSMOLOGICAL COHERENCE INDEX (ICC)
# =============================================================================

def calculate_icc(compressibility: float,
                  coherence: float,
                  hierarchy: float,
                  criticality: float,
                  phi_tendency: float = 0.0,
                  include_phi: bool = True) -> Dict[str, Any]:
    """
    Calculate Cosmological Coherence Index (ICC).

    ICC evaluates compatibility with physical universe generation.
    A high ICC suggests the variant could generate physics-like structures.

    Components (inspired by known physical universe properties):
    1. Compressibility: Physical laws are compressible (F=ma, E=mc²)
    2. Coherence: Non-local correlations (quantum entanglement)
    3. Hierarchy: Multi-scale structure (quarks → atoms → galaxies)
    4. Criticality: Edge of chaos (phase transitions, complexity)
    5. φ-Tendency: Golden ratio presence (optional, exploratory)

    Formula:
    ICC_base = √(Compressibility × Coherence) × Hierarchy × Criticality_factor
    ICC = ICC_base × (1 + φ_bonus) if include_phi else ICC_base

    Args:
        compressibility: 1 - LZ_normalized (high = more compressible)
        coherence: MI_ratio scaled (0-1)
        hierarchy: Hierarchy score (0-1)
        criticality: Combined Hurst/Slope factor (0-1)
        phi_tendency: φ strength (0-1), optional
        include_phi: Whether to include φ bonus in calculation

    Returns:
        Dictionary with ICC score and components
    """
    # Geometric mean of compressibility and coherence
    # (both are necessary - can't have one without the other)
    comp_coh_factor = np.sqrt(compressibility * coherence)

    # Criticality factor: penalize extremes
    crit_factor = criticality

    # Base ICC
    icc_base = comp_coh_factor * hierarchy * crit_factor

    # φ bonus (optional, exploratory)
    if include_phi and phi_tendency > 0:
        phi_bonus = phi_tendency * 0.2  # Max 20% bonus
        icc = icc_base * (1 + phi_bonus)
    else:
        phi_bonus = 0.0
        icc = icc_base

    # Normalize to 0-1 range (theoretical max is ~1.2 with φ bonus)
    icc = min(1.0, icc)

    return {
        'icc': icc,
        'icc_base': icc_base,
        'compressibility': compressibility,
        'coherence': coherence,
        'hierarchy': hierarchy,
        'criticality': criticality,
        'phi_tendency': phi_tendency,
        'phi_bonus': phi_bonus,
        'include_phi': include_phi,
        'comp_coh_factor': comp_coh_factor,
        'interpretation': _interpret_icc(icc, phi_tendency)
    }


def _interpret_icc(icc: float, phi: float) -> str:
    """Interpret ICC score."""
    if icc < 0.1:
        base = "very low cosmological coherence (unlikely to generate physics)"
    elif icc < 0.25:
        base = "low cosmological coherence"
    elif icc < 0.4:
        base = "moderate cosmological coherence"
    elif icc < 0.6:
        base = "good cosmological coherence (potential for physics emergence)"
    elif icc < 0.8:
        base = "high cosmological coherence (strong universe-generation potential)"
    else:
        base = "very high cosmological coherence (optimal for universe generation)"

    if phi > 0.5:
        base += " [φ signature detected]"

    return base


# =============================================================================
# COMPLETE ANALYSIS FUNCTION
# =============================================================================

def analyze_structured_complexity(sequence: str,
                                   emergence_data: Dict[str, Any] = None,
                                   verbose: bool = True) -> Dict[str, Any]:
    """
    Complete SCI + ICC analysis for a sequence.

    Can use pre-computed emergence_index data or compute from scratch.

    Args:
        sequence: Binary string (Φ sequence)
        emergence_data: Pre-computed data from emergence_index.py (optional)
        verbose: Print progress

    Returns:
        Complete analysis with SCI, ICC, and all components
    """
    import time

    def log(msg):
        if verbose:
            print(f"   {msg}", flush=True)

    results = {'sequence_length': len(sequence)}

    # 1. Transfer Entropy (Causality)
    log("⏳ [1/4] Calculating Transfer Entropy...")
    t0 = time.time()
    te_result = calculate_transfer_entropy(sequence)
    log(f"   Done in {time.time()-t0:.1f}s - TE={te_result['transfer_entropy']:.3f}")
    results['transfer_entropy'] = te_result

    # 2. Get or compute other components
    if emergence_data:
        # Use pre-computed values
        lz = emergence_data.get('complexity', {}).get('lz_normalized', 0.5)
        hierarchy = emergence_data.get('hierarchy', {}).get('raw_hierarchy_score', 0.0)
        coherence = emergence_data.get('coherence', {}).get('mi_ratio', 0.0)
        hurst = emergence_data.get('dfa', {}).get('hurst_exponent', 0.5)
        slope = emergence_data.get('criticality', {}).get('slope', 0.0)
        log("   Using pre-computed emergence data")
    else:
        # Would need to import and compute - for now use defaults
        log("   ⚠️ No emergence data provided, using estimation")
        lz = 0.5
        hierarchy = 0.3
        coherence = 0.1
        hurst = 0.5
        slope = 0.0

    # 3. φ-Tendency (Exploratory)
    log("🔮 [2/4] Exploring φ-tendency...")
    t0 = time.time()
    phi_result = calculate_phi_tendency(sequence)
    log(f"   Done in {time.time()-t0:.1f}s - φ={phi_result['phi_strength']:.3f}")
    results['phi_tendency'] = phi_result

    # 4. NonTriviality
    log("🧩 [3/4] Calculating NonTriviality...")
    nt_result = calculate_nontriviality(lz)
    results['nontriviality'] = nt_result

    # 5. Calculate SCI
    log("📊 [4/4] Computing SCI and ICC...")

    # Normalize transfer entropy to 0-1
    te_normalized = min(1.0, te_result['transfer_entropy'] / 0.5)

    sci_result = calculate_sci(
        hierarchy_score=min(1.0, hierarchy * 2),  # Scale hierarchy
        transfer_entropy=te_normalized,
        nontriviality=nt_result['nontriviality']
    )
    results['sci'] = sci_result

    # 6. Calculate ICC
    # Compressibility: higher for more compressible (lower LZ)
    compressibility = 1 - lz

    # Coherence: scale MI to 0-1
    coherence_scaled = min(1.0, coherence * 5)

    # Criticality factor from Hurst and Slope
    # Optimal: Hurst ≈ 0.7-1.0, Slope ≈ -1
    hurst_factor = 1 - abs(hurst - 0.85) / 0.85 if hurst <= 1.1 else 0.5
    slope_factor = 1 - abs(slope + 1) / 2
    criticality = (hurst_factor + slope_factor) / 2

    icc_result = calculate_icc(
        compressibility=compressibility,
        coherence=coherence_scaled,
        hierarchy=min(1.0, hierarchy * 2),
        criticality=criticality,
        phi_tendency=phi_result['phi_strength'],
        include_phi=True
    )
    results['icc'] = icc_result

    log(f"✅ SCI: {sci_result['sci']:.4f} | ICC: {icc_result['icc']:.4f}")

    return results

