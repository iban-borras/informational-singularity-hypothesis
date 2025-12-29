"""
📊 Order Metrics — HSI Structural Analysis

This module implements metrics for analyzing emergent order in v33 structural data.
Unlike pattern metrics (which measure pattern quality), order metrics measure
the intrinsic order encoded in the stratified containment structure.

Key metrics:
1. Nesting Depth Distribution: How Absolutes nest within each other
2. Stratified Entropy: Information entropy at each depth level
3. Order Emergence Index: Overall measure of emergent order
4. Containment Complexity: Complexity of what Absolutes contain
5. Depth Correlation: How depth levels correlate with each other

Author: Iban Borràs with Augment Agent collaboration
Date: January 2025
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter, defaultdict
from scipy.stats import entropy, pearsonr
from scipy.signal import find_peaks
import json
from pathlib import Path


def _load_oei_weights() -> Dict[str, float]:
    """
    Load Order Emergence Index component weights from config.json if available.

    Returns:
        Dictionary with weights for each component, summing to 1.0
    """
    # Defaults with scientific justification (see order_metrics_explained.md)
    defaults = {
        'depth_organization': 0.3,     # Core: hierarchical structure
        'entropy_gradient': 0.3,       # Core: stratification quality
        'containment_regularity': 0.2, # Support: content regularity
        'balance_score': 0.2           # Gate: structural sanity
    }

    try:
        config_path = Path(__file__).parent.parent / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            metrics_config = config.get('metrics', {})
            custom_weights = metrics_config.get('oei_weights', {})

            if custom_weights:
                merged = {**defaults, **custom_weights}
                total = sum(merged.values())
                return {k: v/total for k, v in merged.items()}
    except Exception:
        pass

    return defaults


class OrderMetrics:
    """
    Calculate metrics for emergent order in structural Φ sequences.
    
    These metrics are designed to detect order emergence (not φ-convergence),
    which is the fundamental property we're looking for in HSI v32.
    """
    
    @staticmethod
    def calculate_nesting_depth_distribution(phi_structural: str) -> Dict[str, Any]:
        """
        Calculate the distribution of nesting depths.
        
        Args:
            phi_structural: Φ with parentheses structure
            
        Returns:
            Dictionary with nesting depth metrics:
            - max_depth: Maximum nesting depth
            - depth_counts: Count of Absolutes at each depth
            - depth_distribution: Normalized distribution
            - depth_entropy: Shannon entropy of depth distribution
            - depth_gini: Gini coefficient (inequality measure)
        """
        depth = 0
        max_depth = 0
        depth_counts = defaultdict(int)
        
        for char in phi_structural:
            if char == '(':
                depth += 1
                max_depth = max(max_depth, depth)
                depth_counts[depth] += 1
            elif char == ')':
                depth -= 1
        
        if not depth_counts:
            return {
                'max_depth': 0,
                'depth_counts': {},
                'depth_distribution': {},
                'depth_entropy': 0.0,
                'depth_gini': 0.0
            }
        
        # Normalize distribution
        total_absolutes = sum(depth_counts.values())
        depth_distribution = {d: count / total_absolutes for d, count in depth_counts.items()}
        
        # Calculate entropy
        probs = np.array(list(depth_distribution.values()))
        depth_entropy = entropy(probs, base=2)
        
        # Calculate Gini coefficient (measure of inequality)
        sorted_counts = np.sort(list(depth_counts.values()))
        n = len(sorted_counts)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
        
        return {
            'max_depth': max_depth,
            'depth_counts': dict(depth_counts),
            'depth_distribution': depth_distribution,
            'depth_entropy': float(depth_entropy),
            'depth_gini': float(gini),
            'total_absolutes': total_absolutes
        }
    
    @staticmethod
    def calculate_stratified_entropy(phi_structural: str) -> Dict[str, Any]:
        """
        Calculate entropy at each stratification level.
        
        This measures how information is distributed across depth levels,
        which is key to understanding emergent order.
        
        Args:
            phi_structural: Φ with parentheses structure
            
        Returns:
            Dictionary with stratified entropy metrics:
            - entropy_by_depth: Entropy at each depth level
            - mean_entropy: Average entropy across depths
            - entropy_variance: Variance of entropy across depths
            - entropy_trend: Linear trend coefficient (increasing/decreasing)
        """
        # Extract sequences at each depth
        depth_sequences = OrderMetrics._extract_sequences_by_depth(phi_structural)
        
        if not depth_sequences:
            return {
                'entropy_by_depth': {},
                'mean_entropy': 0.0,
                'entropy_variance': 0.0,
                'entropy_trend': 0.0
            }
        
        entropy_by_depth = {}
        
        for depth, sequences in depth_sequences.items():
            # Concatenate all sequences at this depth
            full_sequence = ''.join(sequences)
            
            if len(full_sequence) == 0:
                entropy_by_depth[depth] = 0.0
                continue
            
            # Calculate Shannon entropy
            char_counts = Counter(full_sequence)
            total_chars = len(full_sequence)
            probs = np.array([char_counts[c] / total_chars for c in char_counts])
            ent = entropy(probs, base=2)
            
            entropy_by_depth[depth] = float(ent)
        
        # Calculate statistics
        entropy_values = list(entropy_by_depth.values())
        mean_entropy = np.mean(entropy_values) if entropy_values else 0.0
        entropy_variance = np.var(entropy_values) if entropy_values else 0.0
        
        # Calculate trend (linear regression slope)
        if len(entropy_values) > 1:
            depths = np.array(list(entropy_by_depth.keys()))
            entropies = np.array(list(entropy_by_depth.values()))
            trend = np.polyfit(depths, entropies, 1)[0]  # Slope of linear fit
        else:
            trend = 0.0
        
        return {
            'entropy_by_depth': entropy_by_depth,
            'mean_entropy': float(mean_entropy),
            'entropy_variance': float(entropy_variance),
            'entropy_trend': float(trend)
        }

    @staticmethod
    def calculate_order_emergence_index(phi_structural: str) -> Dict[str, Any]:
        """
        Calculate an overall index of emergent order.

        This is a composite metric that combines multiple signals:
        - Depth organization (how well-structured the nesting is)
        - Entropy gradient (how entropy changes with depth)
        - Containment regularity (how regular the contents are)
        - Balance (how balanced the parentheses structure is)

        Higher values indicate more emergent order.

        Args:
            phi_structural: Φ with parentheses structure

        Returns:
            Dictionary with order emergence metrics:
            - order_index: Overall order emergence index (0-1)
            - depth_organization: Depth structure quality (0-1)
            - entropy_gradient: Entropy change quality (0-1)
            - containment_regularity: Content regularity (0-1)
            - balance_score: Parentheses balance quality (0-1)
        """
        # Component 1: Depth organization
        depth_metrics = OrderMetrics.calculate_nesting_depth_distribution(phi_structural)

        # Normalize depth entropy (lower is more organized)
        max_possible_entropy = np.log2(depth_metrics['max_depth']) if depth_metrics['max_depth'] > 1 else 1.0
        depth_organization = 1.0 - (depth_metrics['depth_entropy'] / max_possible_entropy) if max_possible_entropy > 0 else 0.0
        depth_organization = max(0.0, min(1.0, depth_organization))

        # Component 2: Entropy gradient
        entropy_metrics = OrderMetrics.calculate_stratified_entropy(phi_structural)

        # Prefer negative trend (entropy decreases with depth = more order)
        entropy_gradient = 1.0 / (1.0 + abs(entropy_metrics['entropy_trend']))

        # Component 3: Containment regularity
        containment_metrics = OrderMetrics.calculate_containment_complexity(phi_structural)

        # Lower variance in content lengths = more regular
        if containment_metrics['length_variance'] > 0:
            containment_regularity = 1.0 / (1.0 + containment_metrics['length_variance'])
        else:
            containment_regularity = 1.0

        # Component 4: Balance score
        balance_score = 1.0 if OrderMetrics._check_balance(phi_structural) else 0.0

        # Combine components (weighted average)
        #
        # SCIENTIFIC JUSTIFICATION OF WEIGHTS (Dec 2025 review):
        #
        # 1. Depth Organization (30%): Core signal - hierarchical structure is
        #    fundamental to emergent order. Low depth entropy means Absolutes
        #    concentrate at specific depths, indicating organized hierarchy.
        #
        # 2. Entropy Gradient (30%): Core signal - stratification quality shows
        #    whether entropy changes systematically with depth. A negative trend
        #    (entropy decreasing inward) indicates order increasing toward the core.
        #
        # 3. Containment Regularity (20%): Supporting signal - regular content
        #    lengths suggest pattern formation, but regularity without hierarchy
        #    is not true emergent order.
        #
        # 4. Balance Score (20%): Structural sanity check - unbalanced parentheses
        #    indicate corrupted data. This is binary (0 or 1), so it acts as a
        #    gate rather than a gradient.
        #
        # These weights can be customized via config.json → metrics.oei_weights
        weights = _load_oei_weights()

        order_index = (
            weights['depth_organization'] * depth_organization +
            weights['entropy_gradient'] * entropy_gradient +
            weights['containment_regularity'] * containment_regularity +
            weights['balance_score'] * balance_score
        )

        return {
            'order_index': float(order_index),
            'depth_organization': float(depth_organization),
            'entropy_gradient': float(entropy_gradient),
            'containment_regularity': float(containment_regularity),
            'balance_score': float(balance_score),
            'weights': weights
        }

    @staticmethod
    def sensitivity_analysis_oei(
        phi_structural: str,
        variations: List[float] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze OEI sensitivity to weight variations.

        Tests how robust the OEI is when weights are perturbed ±10%, ±20%, etc.
        Useful for demonstrating that results don't depend critically on
        arbitrary weight choices.

        Args:
            phi_structural: Φ with parentheses structure
            variations: List of weight deltas to test (default: [-0.1, 0, +0.1])
            verbose: Print detailed output

        Returns:
            Dictionary with:
            - base_oei: OEI with default weights
            - components: The 4 component values (fixed, don't depend on weights)
            - variations: List of {weights, oei} for each combination
            - oei_range: (min, max) OEI across all variations
            - oei_std: Standard deviation of OEI across variations
            - robustness: 1 - (std / mean), measure of stability
        """
        if variations is None:
            variations = [-0.1, 0.0, +0.1]

        # Calculate components once (they don't depend on weights)
        base_result = OrderMetrics.calculate_order_emergence_index(phi_structural)
        components = {
            'depth_organization': base_result['depth_organization'],
            'entropy_gradient': base_result['entropy_gradient'],
            'containment_regularity': base_result['containment_regularity'],
            'balance_score': base_result['balance_score']
        }
        base_weights = base_result['weights']
        base_oei = base_result['order_index']

        # Generate all weight combinations
        weight_keys = ['depth_organization', 'entropy_gradient',
                       'containment_regularity', 'balance_score']

        all_oeis = []
        variation_results = []

        # Test each weight independently
        for key in weight_keys:
            for delta in variations:
                if delta == 0:
                    continue  # Skip base case (already calculated)

                # Create modified weights
                modified = base_weights.copy()
                modified[key] = max(0.05, modified[key] + delta)  # Min 5% weight

                # Renormalize to sum to 1.0
                total = sum(modified.values())
                modified = {k: v/total for k, v in modified.items()}

                # Calculate OEI with modified weights
                oei = (
                    modified['depth_organization'] * components['depth_organization'] +
                    modified['entropy_gradient'] * components['entropy_gradient'] +
                    modified['containment_regularity'] * components['containment_regularity'] +
                    modified['balance_score'] * components['balance_score']
                )

                all_oeis.append(oei)
                variation_results.append({
                    'varied_component': key,
                    'delta': delta,
                    'weights': modified,
                    'oei': oei,
                    'diff_from_base': oei - base_oei
                })

        # Add base case
        all_oeis.append(base_oei)

        # Statistics
        oei_array = np.array(all_oeis)
        oei_min, oei_max = float(oei_array.min()), float(oei_array.max())
        oei_mean = float(oei_array.mean())
        oei_std = float(oei_array.std())
        robustness = 1.0 - (oei_std / oei_mean) if oei_mean > 0 else 0.0

        if verbose:
            print(f"\n📊 OEI Sensitivity Analysis")
            print(f"   Base OEI: {base_oei:.4f}")
            print(f"   Range: [{oei_min:.4f}, {oei_max:.4f}]")
            print(f"   Std: {oei_std:.4f}")
            print(f"   Robustness: {robustness:.1%}")

        return {
            'base_oei': base_oei,
            'base_weights': base_weights,
            'components': components,
            'variations': variation_results,
            'oei_range': (oei_min, oei_max),
            'oei_mean': oei_mean,
            'oei_std': oei_std,
            'robustness': robustness,
            'num_variations_tested': len(all_oeis)
        }

    @staticmethod
    def calculate_containment_complexity(phi_structural: str) -> Dict[str, Any]:
        """
        Calculate complexity metrics for what Absolutes contain.

        Args:
            phi_structural: Φ with parentheses structure

        Returns:
            Dictionary with containment complexity metrics:
            - num_absolutes: Total number of Absolutes
            - content_lengths: List of content lengths
            - mean_length: Average content length
            - length_variance: Variance in content lengths
            - content_entropy: Average entropy of contents
            - unique_contents: Number of unique contents
            - repetition_rate: Rate of content repetition
        """
        absolutes = OrderMetrics._extract_absolutes(phi_structural)

        if not absolutes:
            return {
                'num_absolutes': 0,
                'content_lengths': [],
                'mean_length': 0.0,
                'length_variance': 0.0,
                'content_entropy': 0.0,
                'unique_contents': 0,
                'repetition_rate': 0.0
            }

        # Extract observable contents
        contents = [abs_info['observable_content'] for abs_info in absolutes]
        content_lengths = [len(c) for c in contents]

        # Calculate length statistics
        mean_length = np.mean(content_lengths)
        length_variance = np.var(content_lengths)

        # Calculate content entropy
        content_entropies = []
        for content in contents:
            if len(content) > 0:
                char_counts = Counter(content)
                total = len(content)
                probs = np.array([char_counts[c] / total for c in char_counts])
                ent = entropy(probs, base=2)
                content_entropies.append(ent)

        mean_content_entropy = np.mean(content_entropies) if content_entropies else 0.0

        # Calculate uniqueness
        unique_contents = len(set(contents))
        repetition_rate = 1.0 - (unique_contents / len(contents)) if len(contents) > 0 else 0.0

        return {
            'num_absolutes': len(absolutes),
            'content_lengths': content_lengths,
            'mean_length': float(mean_length),
            'length_variance': float(length_variance),
            'content_entropy': float(mean_content_entropy),
            'unique_contents': unique_contents,
            'repetition_rate': float(repetition_rate)
        }

    @staticmethod
    def calculate_depth_correlation(phi_structural: str) -> Dict[str, Any]:
        """
        Calculate correlation between adjacent depth levels.

        This measures how information at one depth level relates to
        information at adjacent levels, which can indicate hierarchical
        organization.

        Args:
            phi_structural: Φ with parentheses structure

        Returns:
            Dictionary with depth correlation metrics:
            - correlations: Pearson correlation between adjacent depths
            - mean_correlation: Average correlation
            - correlation_variance: Variance in correlations
        """
        depth_sequences = OrderMetrics._extract_sequences_by_depth(phi_structural)

        if len(depth_sequences) < 2:
            return {
                'correlations': {},
                'mean_correlation': 0.0,
                'correlation_variance': 0.0
            }

        correlations = {}

        # Calculate correlation between adjacent depths
        sorted_depths = sorted(depth_sequences.keys())

        for i in range(len(sorted_depths) - 1):
            depth1 = sorted_depths[i]
            depth2 = sorted_depths[i + 1]

            seq1 = ''.join(depth_sequences[depth1])
            seq2 = ''.join(depth_sequences[depth2])

            # Convert to binary arrays
            arr1 = np.array([1 if c == '1' else 0 if c == '0' else -1 for c in seq1])
            arr2 = np.array([1 if c == '1' else 0 if c == '0' else -1 for c in seq2])

            # Truncate to same length
            min_len = min(len(arr1), len(arr2))
            if min_len > 1:
                arr1 = arr1[:min_len]
                arr2 = arr2[:min_len]

                # Calculate Pearson correlation
                try:
                    corr, _ = pearsonr(arr1, arr2)
                    correlations[f"{depth1}-{depth2}"] = float(corr) if not np.isnan(corr) else 0.0
                except:
                    correlations[f"{depth1}-{depth2}"] = 0.0

        # Calculate statistics
        corr_values = list(correlations.values())
        mean_correlation = np.mean(corr_values) if corr_values else 0.0
        correlation_variance = np.var(corr_values) if corr_values else 0.0

        return {
            'correlations': correlations,
            'mean_correlation': float(mean_correlation),
            'correlation_variance': float(correlation_variance)
        }

    @staticmethod
    def calculate_all_metrics(phi_structural: str) -> Dict[str, Any]:
        """
        Calculate all order metrics for a structural Φ sequence.

        Args:
            phi_structural: Φ with parentheses structure

        Returns:
            Dictionary with all order metrics
        """
        return {
            'nesting_depth': OrderMetrics.calculate_nesting_depth_distribution(phi_structural),
            'stratified_entropy': OrderMetrics.calculate_stratified_entropy(phi_structural),
            'order_emergence': OrderMetrics.calculate_order_emergence_index(phi_structural),
            'containment_complexity': OrderMetrics.calculate_containment_complexity(phi_structural),
            'depth_correlation': OrderMetrics.calculate_depth_correlation(phi_structural)
        }

    # ========== HELPER METHODS ==========

    @staticmethod
    def _extract_sequences_by_depth(phi_structural: str) -> Dict[int, List[str]]:
        """Extract all character sequences at each nesting depth."""
        depth_sequences = defaultdict(list)
        depth = 0
        current_sequence = ''

        for char in phi_structural:
            if char == '(':
                if current_sequence and depth > 0:
                    depth_sequences[depth].append(current_sequence)
                    current_sequence = ''
                depth += 1
            elif char == ')':
                if current_sequence:
                    depth_sequences[depth].append(current_sequence)
                    current_sequence = ''
                depth -= 1
            else:
                current_sequence += char

        # Add final sequence if any
        if current_sequence and depth > 0:
            depth_sequences[depth].append(current_sequence)

        return dict(depth_sequences)

    @staticmethod
    def _extract_absolutes(phi_structural: str) -> List[Dict[str, Any]]:
        """Extract all parenthesized groups (Absolutes) from Φ."""
        absolutes = []
        stack = []
        depth = 0

        for i, char in enumerate(phi_structural):
            if char == '(':
                depth += 1
                stack.append({'start': i, 'depth': depth, 'content': ''})
            elif char == ')':
                if stack:
                    absolute = stack.pop()
                    full_content = phi_structural[absolute['start']:i+1]
                    observable_content = full_content.replace('(', '').replace(')', '')

                    absolutes.append({
                        'full_content': full_content,
                        'observable_content': observable_content,
                        'start_position': absolute['start'],
                        'end_position': i,
                        'depth': absolute['depth']
                    })
                depth -= 1
            else:
                # Add character to all open absolutes
                for abs_info in stack:
                    abs_info['content'] += char

        return absolutes

    @staticmethod
    def _check_balance(phi_structural: str) -> bool:
        """Check if parentheses are balanced."""
        depth = 0
        for char in phi_structural:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0


def create_order_report(phi_structural: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a comprehensive order metrics report.

    Args:
        phi_structural: Φ with parentheses structure
        output_path: Optional path to save JSON report

    Returns:
        Dictionary with complete order analysis
    """
    metrics = OrderMetrics.calculate_all_metrics(phi_structural)

    # Add summary
    report = {
        'summary': {
            'total_length': len(phi_structural),
            'observable_length': len(phi_structural.replace('(', '').replace(')', '')),
            'max_depth': metrics['nesting_depth']['max_depth'],
            'total_absolutes': metrics['nesting_depth']['total_absolutes'],
            'order_index': metrics['order_emergence']['order_index'],
            'is_balanced': OrderMetrics._check_balance(phi_structural)
        },
        'metrics': metrics
    }

    # Save if path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"✅ Order report saved to {output_path}")

    return report


def compare_variants_oei_sensitivity(
    variant_data: Dict[str, str],
    variations: List[float] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare OEI across variants with sensitivity analysis.

    Tests whether the ranking of variants is robust to weight changes.
    This is critical for scientific validity: if Variant B beats Control A
    only with specific weights, the result is fragile.

    Args:
        variant_data: Dict mapping variant name to phi_structural string
        variations: Weight deltas to test (default: [-0.1, 0, +0.1])
        verbose: Print detailed output

    Returns:
        Dictionary with:
        - variant_results: Per-variant sensitivity results
        - ranking_stability: % of variations where ranking is preserved
        - base_ranking: Ranking with default weights
        - robustness_summary: Overall assessment
    """
    if variations is None:
        variations = [-0.1, 0.0, +0.1]

    # Calculate sensitivity for each variant
    variant_results = {}
    for name, phi_s in variant_data.items():
        if phi_s:
            variant_results[name] = OrderMetrics.sensitivity_analysis_oei(
                phi_s, variations, verbose=False
            )

    if len(variant_results) < 2:
        return {'error': 'Need at least 2 variants to compare'}

    # Base ranking (with default weights)
    base_ranking = sorted(
        variant_results.keys(),
        key=lambda v: variant_results[v]['base_oei'],
        reverse=True
    )

    # Count how many weight variations preserve the ranking
    # We check if the order A > B > C is preserved
    num_variations = len(variant_results[base_ranking[0]]['variations'])
    rankings_preserved = 0
    total_tests = num_variations + 1  # +1 for base case

    # Check base case (always preserved by definition)
    rankings_preserved += 1

    # For each variation, reconstruct OEIs and check ranking
    for var_idx in range(num_variations):
        var_ranking = sorted(
            variant_results.keys(),
            key=lambda v: variant_results[v]['variations'][var_idx]['oei']
                          if var_idx < len(variant_results[v]['variations'])
                          else variant_results[v]['base_oei'],
            reverse=True
        )
        if var_ranking == base_ranking:
            rankings_preserved += 1

    ranking_stability = rankings_preserved / total_tests

    # Summary statistics
    summary = {
        'ranking_stability': ranking_stability,
        'base_ranking': base_ranking,
        'variants_compared': len(variant_results),
        'variations_tested': total_tests
    }

    if verbose:
        print(f"\n📊 OEI Cross-Variant Sensitivity Analysis")
        print(f"   Variants: {', '.join(base_ranking)}")
        print(f"   Base ranking (by OEI): {' > '.join(base_ranking)}")
        print(f"\n   Per-variant OEI:")
        for v in base_ranking:
            r = variant_results[v]
            print(f"     {v}: {r['base_oei']:.4f} (range: {r['oei_range'][0]:.4f}-{r['oei_range'][1]:.4f})")
        print(f"\n   Ranking stability: {ranking_stability:.1%} ({rankings_preserved}/{total_tests} variations)")

        if ranking_stability >= 0.9:
            print(f"   ✅ ROBUST: Ranking preserved in ≥90% of weight variations")
        elif ranking_stability >= 0.7:
            print(f"   ⚠️ MODERATE: Ranking preserved in 70-90% of variations")
        else:
            print(f"   ❌ FRAGILE: Ranking changes frequently with weight variations")

    return {
        'variant_results': variant_results,
        'summary': summary,
        'is_robust': ranking_stability >= 0.9
    }


if __name__ == "__main__":
    # Example usage
    print("📊 Order Metrics — HSI")
    print("=" * 60)

    # Example structural Φ
    phi_structural = "(((01)1)0((10)0)1)"

    print(f"\nInput: {phi_structural}")
    print(f"Length: {len(phi_structural)} characters")

    # Calculate all metrics
    metrics = OrderMetrics.calculate_all_metrics(phi_structural)

    print("\n📊 NESTING DEPTH METRICS:")
    print(f"  Max depth: {metrics['nesting_depth']['max_depth']}")
    print(f"  Total Absolutes: {metrics['nesting_depth']['total_absolutes']}")
    print(f"  Depth entropy: {metrics['nesting_depth']['depth_entropy']:.3f}")
    print(f"  Depth Gini: {metrics['nesting_depth']['depth_gini']:.3f}")

    print("\n📊 STRATIFIED ENTROPY METRICS:")
    print(f"  Mean entropy: {metrics['stratified_entropy']['mean_entropy']:.3f}")
    print(f"  Entropy variance: {metrics['stratified_entropy']['entropy_variance']:.3f}")
    print(f"  Entropy trend: {metrics['stratified_entropy']['entropy_trend']:.3f}")

    print("\n📊 ORDER EMERGENCE INDEX:")
    print(f"  Overall order index: {metrics['order_emergence']['order_index']:.3f}")
    print(f"  Depth organization: {metrics['order_emergence']['depth_organization']:.3f}")
    print(f"  Entropy gradient: {metrics['order_emergence']['entropy_gradient']:.3f}")
    print(f"  Containment regularity: {metrics['order_emergence']['containment_regularity']:.3f}")
    print(f"  Balance score: {metrics['order_emergence']['balance_score']:.3f}")

    print("\n📊 CONTAINMENT COMPLEXITY:")
    print(f"  Mean content length: {metrics['containment_complexity']['mean_length']:.2f}")
    print(f"  Length variance: {metrics['containment_complexity']['length_variance']:.2f}")
    print(f"  Content entropy: {metrics['containment_complexity']['content_entropy']:.3f}")
    print(f"  Repetition rate: {metrics['containment_complexity']['repetition_rate']:.3f}")

    print("\n📊 DEPTH CORRELATION:")
    print(f"  Mean correlation: {metrics['depth_correlation']['mean_correlation']:.3f}")
    print(f"  Correlation variance: {metrics['depth_correlation']['correlation_variance']:.3f}")

    print("\n✅ Example completed!")


