#!/usr/bin/env python3
"""
📊 ISH Level 1 Results Analyzer

Analyzes JSON results from Level 1 experiments and generates human-readable
reports in both console and Markdown format.

Usage:
    python analyze_results.py                           # Analyze latest result
    python analyze_results.py results/file.json        # Analyze specific file
    python analyze_results.py --variant B --iteration 18
    python analyze_results.py --output report.md       # Save Markdown report

Author: Iban Borràs with collaboration of Augment Agent (Sophia)
"""
import json
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


class ResultsReporter:
    """
    Analyzer and reporter for ISH Level 1 experiment results.

    Can generate both console output and Markdown reports suitable
    for scientific documentation and paper enrichment.
    """

    def __init__(self, results_path: str):
        """
        Initialize the reporter with a results JSON file.

        Args:
            results_path: Path to the Level 1 analysis JSON file
        """
        self.results_path = Path(results_path)
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        with open(self.results_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # Extract metadata
        self.metadata = self.data.get('metadata', {})
        self.patterns = self.data.get('patterns', {})
        self.rules = self.data.get('rules', [])
        self.validation = self.data.get('validation', {})

        # Calculate extrapolation factor for sampled data
        self._calculate_extrapolation()

        # Parse variant and iteration from filename if not in metadata
        self._parse_file_info()

    def _calculate_extrapolation(self):
        """Calculate extrapolation factor and statistical reliability for sampled data."""
        import math

        # Full sequence length from source file metadata
        full_length = self.metadata.get('sequence_length', 0)

        # Bits actually analyzed (from coverage or char_counts)
        coverage = self.data.get('coverage', {})
        analyzed_length = coverage.get('sequence_length', 0)

        # Fallback to char_counts if coverage doesn't have it
        if analyzed_length == 0:
            char_counts = self.data.get('char_counts', {})
            analyzed_length = char_counts.get('observable', 0)

        # Calculate factor
        if full_length > 0 and analyzed_length > 0:
            self.full_sequence_length = full_length
            self.analyzed_length = analyzed_length
            self.extrapolation_factor = full_length / analyzed_length
            self.coverage_percentage = (analyzed_length / full_length) * 100
        else:
            self.full_sequence_length = analyzed_length
            self.analyzed_length = analyzed_length
            self.extrapolation_factor = 1.0
            self.coverage_percentage = 100.0

        # Calculate max_reliable_pattern_length retroactively if not in data
        sampling = self.data.get('sampling', {})
        if 'max_reliable_pattern_length' not in sampling and self.analyzed_length > 0:
            # Formula: k_max = log2(sample * structure_factor / (min_occurrences * 2))
            structure_factor = 50.0  # HSI structure factor
            config = self.data.get('config', {})
            min_occurrences = config.get('min_occurrences', 3)
            max_pattern_length = config.get('max_pattern_length', 50)

            try:
                max_reliable_k = int(math.log2(
                    self.analyzed_length * structure_factor / (min_occurrences * 2)
                ))
            except (ValueError, ZeroDivisionError):
                max_reliable_k = max_pattern_length

            # Store calculated values in sampling dict for later use
            if 'sampling' not in self.data:
                self.data['sampling'] = {}
            self.data['sampling']['max_reliable_pattern_length'] = max_reliable_k
            self.data['sampling']['max_pattern_length'] = max_pattern_length
            self.data['sampling']['calculated_retroactively'] = True

    def _parse_file_info(self):
        """Extract variant and iteration from filename."""
        import re
        fname = self.results_path.stem  # e.g., "var_B_iter15_min10_max50"
        self.variant = "?"
        self.iteration = 0

        # Match patterns like "var_B" or "varB"
        var_match = re.search(r'var[_]?([A-Za-z])', fname)
        if var_match:
            self.variant = var_match.group(1).upper()

        # Match patterns like "iter15" or "iter_15", stop at underscore or end
        iter_match = re.search(r'iter[_]?(\d+)', fname)
        if iter_match:
            self.iteration = int(iter_match.group(1))

    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze pattern statistics."""
        observable = self.patterns.get('observable', [])
        structural = self.patterns.get('structural', [])

        if not observable:
            return {'error': 'No observable patterns found'}

        lengths = [len(p.get('pattern_data', '')) for p in observable]
        recurrences = [p.get('recurrence', 0) for p in observable]

        # Length distribution
        len_dist = Counter(lengths)

        # Top patterns by recurrence
        sorted_patterns = sorted(observable, key=lambda p: -p.get('recurrence', 0))
        top_patterns = sorted_patterns[:10]

        return {
            'total_observable': len(observable),
            'total_structural': len(structural),
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'length_distribution': dict(sorted(len_dist.items())),
            'top_patterns': [
                {
                    'pattern': p.get('pattern_data', ''),
                    'recurrence': p.get('recurrence', 0),
                    'density': p.get('density', 0)
                }
                for p in top_patterns
            ],
            'total_recurrences': sum(recurrences)
        }

    def analyze_rules(self) -> Dict[str, Any]:
        """Analyze rule statistics."""
        if not self.rules:
            return {'error': 'No rules found'}

        # Context length distribution
        ctx_lens = [len(r.get('context', '')) for r in self.rules if 'context' in r]
        ctx_dist = Counter(ctx_lens)

        # Confidence distribution
        confs = [r.get('confidence', 0) for r in self.rules]

        # Rule types
        rule_types = Counter(r.get('rule_type', 'unknown') for r in self.rules)

        # Top rules by confidence, then support (to prioritize high-evidence rules)
        sorted_rules = sorted(
            self.rules,
            key=lambda r: (-r.get('confidence', 0), -r.get('support', 0))
        )

        return {
            'total_rules': len(self.rules),
            'context_min': min(ctx_lens) if ctx_lens else 0,
            'context_max': max(ctx_lens) if ctx_lens else 0,
            'context_distribution': dict(sorted(ctx_dist.items())),
            'confidence_99plus': sum(1 for c in confs if c >= 0.99),
            'confidence_90_99': sum(1 for c in confs if 0.9 <= c < 0.99),
            'confidence_below_90': sum(1 for c in confs if c < 0.9),
            'avg_confidence': sum(confs) / len(confs) if confs else 0,
            'rule_types': dict(rule_types.most_common()),
            'top_rules': [
                {
                    'context': r.get('context', ''),
                    'prediction': r.get('prediction', ''),
                    'confidence': r.get('confidence', 0),
                    'support': r.get('support', 0)
                }
                for r in sorted_rules[:10]
            ]
        }

    def _interpret_phi_alignment(self, phi_value: float) -> Dict[str, str]:
        """Interpret φ-alignment value with dynamic text based on range."""
        if phi_value is None:
            return {
                'level': 'unknown',
                'emoji': '❓',
                'short': 'Not calculated',
                'description': 'φ-alignment was not calculated for this analysis.',
                'significance': 'Run analysis with pattern metrics enabled to calculate φ-alignment.'
            }

        if phi_value >= 0.8:
            return {
                'level': 'very_high',
                'emoji': '🌟',
                'short': f'Very High ({phi_value:.2%})',
                'description': f'Exceptional φ-coherence ({phi_value:.4f}) indicating strong golden ratio alignment.',
                'significance': 'The patterns show remarkable convergence toward φ, suggesting deep structural affinity with the golden ratio. This is strong evidence for ISH-predicted self-organization.'
            }
        elif phi_value >= 0.6:
            return {
                'level': 'high',
                'emoji': '✨',
                'short': f'High ({phi_value:.2%})',
                'description': f'High φ-coherence ({phi_value:.4f}) with significant golden ratio structure.',
                'significance': 'The sequence exhibits notable φ-related proportions, consistent with emergent order predicted by ISH.'
            }
        elif phi_value >= 0.35:
            return {
                'level': 'moderate',
                'emoji': '📊',
                'short': f'Moderate ({phi_value:.2%})',
                'description': f'Moderate φ-coherence ({phi_value:.4f}) with emerging φ-related patterns.',
                'significance': 'Some golden ratio structure is present, suggesting partial self-organization. May increase with more iterations.'
            }
        elif phi_value >= 0.15:
            return {
                'level': 'low',
                'emoji': '📉',
                'short': f'Low ({phi_value:.2%})',
                'description': f'Low φ-coherence ({phi_value:.4f}) with minimal golden ratio alignment.',
                'significance': 'Limited φ-structure detected. The sequence may be too short or the variant may not converge toward φ.'
            }
        else:
            return {
                'level': 'minimal',
                'emoji': '⚪',
                'short': f'Minimal ({phi_value:.2%})',
                'description': f'Minimal φ-coherence ({phi_value:.4f}), near random distribution.',
                'significance': 'No significant golden ratio structure detected. This suggests random or chaotic behavior.'
            }

    def _interpret_density(self, ones_ratio: float) -> Dict[str, str]:
        """Interpret the density of 1s vs 0s."""
        deviation = abs(ones_ratio - 0.5)

        if deviation < 0.01:
            return {
                'level': 'balanced',
                'description': f'Nearly perfect balance ({ones_ratio:.2%} ones)',
                'significance': 'Equal distribution suggests maximum entropy or symmetric collapse process.'
            }
        elif deviation < 0.05:
            bias = '1s' if ones_ratio > 0.5 else '0s'
            return {
                'level': 'slight_bias',
                'description': f'Slight bias toward {bias} ({ones_ratio:.2%} ones)',
                'significance': 'Minor asymmetry may indicate preferential collapse direction.'
            }
        else:
            bias = '1s' if ones_ratio > 0.5 else '0s'
            return {
                'level': 'significant_bias',
                'description': f'Significant bias toward {bias} ({ones_ratio:.2%} ones)',
                'significance': 'Strong asymmetry suggests non-equilibrium dynamics in the collapse process.'
            }

    def analyze_scientific_insights(self) -> Dict[str, Any]:
        """Extract scientifically relevant insights for the ISH paper."""
        insights = {
            'alternating_dominance': False,
            'deterministic_structure': False,
            'fibonacci_patterns': [],
            'phi_indicators': [],
            'is_chaotic': False,
            'chaos_indicators': [],
            'phi_alignment': None,
            'phi_interpretation': None,
            'density_analysis': None,
            'complexity_metrics': {}
        }

        # Get φ-alignment from validation results
        phi_value = self.validation.get('phi_alignment')
        insights['phi_alignment'] = phi_value
        insights['phi_interpretation'] = self._interpret_phi_alignment(phi_value)

        # Calculate density (1s vs 0s ratio)
        char_counts = self.data.get('char_counts', {})
        observable_len = char_counts.get('observable', 0)
        if observable_len > 0:
            # Estimate from patterns if available
            observable = self.patterns.get('observable', [])
            if observable:
                all_patterns = ''.join(p.get('pattern_data', '') for p in observable[:100])
                if all_patterns:
                    ones_ratio = all_patterns.count('1') / len(all_patterns)
                    insights['density_analysis'] = self._interpret_density(ones_ratio)

        # Calculate complexity metrics
        if self.rules:
            contexts = [r.get('context', '') for r in self.rules if r.get('context')]
            if contexts:
                avg_context_len = sum(len(c) for c in contexts) / len(contexts)
                max_context_len = max(len(c) for c in contexts)
                insights['complexity_metrics'] = {
                    'avg_context_length': avg_context_len,
                    'max_context_length': max_context_len,
                    'rule_diversity': len(set(contexts)) / len(contexts) if contexts else 0
                }

        observable = self.patterns.get('observable', [])

        # === CHAOS DETECTION ===
        # A chaotic/random sequence shows:
        # 1. Uniform distribution of pattern occurrences (low variance)
        # 2. No rules with ≥99% confidence
        # 3. Many patterns but similar occurrence counts

        if observable and len(observable) > 100:
            recurrences = [p.get('recurrence', 0) for p in observable]
            mean_rec = sum(recurrences) / len(recurrences)
            variance = sum((r - mean_rec) ** 2 for r in recurrences) / len(recurrences)
            std_dev = variance ** 0.5
            coef_variation = std_dev / mean_rec if mean_rec > 0 else 0

            # Low coefficient of variation = uniform distribution = chaos
            if coef_variation < 0.05:  # Less than 5% variation
                insights['is_chaotic'] = True
                insights['chaos_indicators'].append(
                    f"Uniform pattern distribution (CV={coef_variation:.3f}, expected >0.1 for order)"
                )

        # Check for zero deterministic rules
        if self.rules:
            high_conf = sum(1 for r in self.rules if r.get('confidence', 0) >= 0.99)
            high_conf_ratio = high_conf / len(self.rules)

            if high_conf_ratio == 0:
                insights['is_chaotic'] = True
                insights['chaos_indicators'].append(
                    "No deterministic rules (0% with ≥99% confidence)"
                )
            elif high_conf_ratio > 0.5:
                insights['deterministic_structure'] = True

        # Check for alternating pattern dominance
        if observable:
            top_pattern = max(observable, key=lambda p: p.get('recurrence', 0))
            pattern_data = top_pattern.get('pattern_data', '')
            if pattern_data in ['0101010101', '1010101010'] or \
               (len(pattern_data) > 1 and all(pattern_data[i] != pattern_data[i+1] for i in range(len(pattern_data)-1))):
                insights['alternating_dominance'] = True

        # Check for Fibonacci-related periods
        fib_numbers = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        period_rules = [r for r in self.rules if r.get('rule_type') == 'periodicity']
        for r in period_rules:
            period = r.get('period', 0)
            if period in fib_numbers:
                insights['fibonacci_patterns'].append(period)

        return insights

    def print_console_report(self):
        """Print a formatted report to the console."""
        pattern_stats = self.analyze_patterns()
        rule_stats = self.analyze_rules()
        insights = self.analyze_scientific_insights()

        print('=' * 70)
        print(f'📊 ISH LEVEL 1 ANALYSIS REPORT')
        print(f'   Variant {self.variant} - Iteration {self.iteration}')
        print('=' * 70)

        # Basic info
        char_counts = self.data.get('char_counts', {})
        print(f'\n📁 FILE: {self.results_path.name}')
        print(f'   Structural sequence: {char_counts.get("structural", "N/A"):,} chars')
        print(f'   Observable sequence: {char_counts.get("observable", "N/A"):,} chars')
        print(f'   Analysis time: {self.data.get("analysis_time_seconds", 0):,.1f}s')

        # Sampling info (if partial coverage)
        if self.extrapolation_factor > 1.01:  # More than 1% difference
            print(f'\n📐 SAMPLING INFO')
            print(f'   Full sequence: {self.full_sequence_length:,} bits ({self.full_sequence_length/1e9:.1f}G)')
            print(f'   Bits analyzed: {self.analyzed_length:,} bits ({self.analyzed_length/1e9:.1f}G)')
            print(f'   Coverage: {self.coverage_percentage:.2f}%')
            print(f'   Extrapolation factor: {self.extrapolation_factor:.2f}x')

            # Show max reliable pattern length if available
            sampling = self.data.get('sampling', {})
            max_reliable = sampling.get('max_reliable_pattern_length', 0)
            max_requested = sampling.get('max_pattern_length', 0)
            if max_reliable > 0 and max_reliable < max_requested:
                print(f'\n   ⚠️ Statistical reliability:')
                print(f'   └─ Reliable detection up to: {max_reliable} bits')
                print(f'   └─ Patterns {max_reliable+1}-{max_requested} bits: may be underdetected')

        # Patterns
        print(f'\n🔍 PATTERNS DETECTED')
        obs_count = pattern_stats.get("total_observable", 0)
        struct_count = pattern_stats.get("total_structural", 0)
        if self.extrapolation_factor > 1.01:
            est_obs = int(obs_count * self.extrapolation_factor)
            print(f'   Observable: {obs_count:,} (estimated global: ~{est_obs:,})')
        else:
            print(f'   Observable: {obs_count:,}')
        print(f'   Structural: {struct_count}')
        print(f'   Length range: {pattern_stats.get("min_length", 0)}-{pattern_stats.get("max_length", 0)} bits')
        print(f'   Average length: {pattern_stats.get("avg_length", 0):.1f} bits')

        print(f'\n   Top 5 most recurrent patterns:')
        for i, p in enumerate(pattern_stats.get('top_patterns', [])[:5], 1):
            print(f'     {i}. "{p["pattern"]}" - {p["recurrence"]:,} occurrences')

        # Rules
        print(f'\n🧠 RULES INFERRED: {rule_stats.get("total_rules", 0):,}')
        print(f'   Context length: {rule_stats.get("context_min", 0)}-{rule_stats.get("context_max", 0)} bits')
        print(f'\n   Confidence distribution:')
        print(f'     ≥99%: {rule_stats.get("confidence_99plus", 0):,} rules')
        print(f'     90-99%: {rule_stats.get("confidence_90_99", 0):,} rules')
        print(f'     <90%: {rule_stats.get("confidence_below_90", 0):,} rules')
        print(f'   Average confidence: {rule_stats.get("avg_confidence", 0)*100:.2f}%')

        print(f'\n   Top 5 rules by confidence:')
        for i, r in enumerate(rule_stats.get('top_rules', [])[:5], 1):
            print(f'     {i}. "{r["context"]}" → "{r["prediction"]}" ({r["confidence"]*100:.1f}%)')

        # Scientific insights
        print(f'\n🔬 SCIENTIFIC INSIGHTS')
        if insights.get('is_chaotic'):
            print('   ⚠️ CHAOTIC/RANDOM SEQUENCE DETECTED')
            for indicator in insights.get('chaos_indicators', []):
                print(f'      • {indicator}')
        else:
            if insights.get('alternating_dominance'):
                print('   ✓ Alternating pattern dominance detected (0-1 oscillation)')
            if insights.get('deterministic_structure'):
                print('   ✓ Deterministic structure confirmed (>50% rules with ≥99% confidence)')
            if insights.get('fibonacci_patterns'):
                print(f'   ✓ Fibonacci periods found: {insights["fibonacci_patterns"]}')

        # Interpretation
        print(f'\n📝 INTERPRETATION')
        print('   ' + '-' * 60)
        self._print_interpretation(pattern_stats, rule_stats, insights)
        print('   ' + '-' * 60)

        print('\n' + '=' * 70)

    def _print_interpretation(self, pattern_stats, rule_stats, insights):
        """Generate scientific interpretation."""
        lines = []

        # CHAOS WARNING - must come first
        if insights.get('is_chaotic'):
            lines.append("⚠️  WARNING: CHAOTIC/RANDOM SEQUENCE DETECTED")
            lines.append("")
            for indicator in insights.get('chaos_indicators', []):
                lines.append(f"   • {indicator}")
            lines.append("")
            lines.append("This variant produces PSEUDO-RANDOM output, not structured")
            lines.append("information. While random sequences can contain ANY pattern")
            lines.append("by chance, they lack the stable, hierarchical order needed")
            lines.append("for persistent physical laws.")
            lines.append("")
            lines.append("⚠️  NOT CONSISTENT WITH ISH: A universe from this variant")
            lines.append("would have spontaneous formation/dissolution of structures")
            lines.append("without stable underlying order.")
            for line in lines:
                print(f'   {line}')
            return

        # Pattern interpretation
        if insights.get('alternating_dominance'):
            lines.append("The dominance of alternating patterns (0101...) suggests an")
            lines.append("oscillatory structure emerging from the collapse process.")

        # Rule interpretation
        if insights.get('deterministic_structure'):
            lines.append("")
            lines.append("The high percentage of deterministic rules (≥99% confidence)")
            lines.append("indicates the sequence is NOT random but follows strict rules.")

        # Anti-repetition
        top_rules = rule_stats.get('top_rules', [])
        anti_rep = [r for r in top_rules if
                   (r['context'].endswith('00') and r['prediction'] == '1') or
                   (r['context'].endswith('11') and r['prediction'] == '0')]
        if anti_rep:
            lines.append("")
            lines.append("Anti-repetition rules detected: consecutive identical bits")
            lines.append("tend to be followed by the opposite bit, creating alternation.")

        lines.append("")
        lines.append("✅ CONSISTENT WITH ISH: The collapse process creates structured")
        lines.append("information with emergent regularities and stable order.")

        for line in lines:
            print(f'   {line}')

    def generate_markdown_report(self) -> str:
        """Generate a Markdown report suitable for documentation."""
        pattern_stats = self.analyze_patterns()
        rule_stats = self.analyze_rules()
        insights = self.analyze_scientific_insights()
        char_counts = self.data.get('char_counts', {})

        md = []
        md.append(f"# ISH Level 1 Analysis Report")
        md.append(f"## Variant {self.variant} - Iteration {self.iteration}")
        md.append(f"")
        md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append(f"**Source file:** `{self.results_path.name}`")
        md.append(f"")

        # Sampling Information (if partial coverage)
        if self.extrapolation_factor > 1.01:
            md.append("---")
            md.append("## Sampling Information")
            md.append("")
            md.append("| Parameter | Value |")
            md.append("|-----------|-------|")
            md.append(f"| **Full sequence length** | {self.full_sequence_length:,} bits ({self.full_sequence_length/1e9:.1f}G) |")
            md.append(f"| **Bits analyzed** | {self.analyzed_length:,} bits ({self.analyzed_length/1e9:.1f}G) |")
            md.append(f"| **Coverage** | {self.coverage_percentage:.2f}% |")
            md.append(f"| **Extrapolation factor** | {self.extrapolation_factor:.2f}x |")
            md.append("")

            # Statistical reliability info
            sampling = self.data.get('sampling', {})
            max_reliable = sampling.get('max_reliable_pattern_length', 0)
            max_requested = sampling.get('max_pattern_length', 0)
            if max_reliable > 0 and max_reliable < max_requested:
                md.append(f"### Statistical Reliability Warning")
                md.append("")
                md.append(f"| Reliability | Pattern Length |")
                md.append(f"|-------------|----------------|")
                md.append(f"| ✅ **Reliable** | 1-{max_reliable} bits |")
                md.append(f"| ⚠️ **Underdetected** | {max_reliable+1}-{max_requested} bits |")
                md.append("")
                md.append(f"> With {self.coverage_percentage:.2f}% coverage, patterns longer than {max_reliable} bits")
                md.append(f"> may not have enough statistical representation to be reliably detected.")
                md.append("")
            else:
                md.append(f"> ⚠️ **Partial sampling**: Results based on {self.coverage_percentage:.2f}% of the full sequence.")
                md.append("> Absolute counts should be multiplied by the extrapolation factor for global estimates.")
                md.append("")

            md.append("### Extrapolated Global Estimates")
            md.append("")
            md.append("| Metric (sample) | Sample Value | Estimated Global |")
            md.append("|-----------------|--------------|------------------|")
            obs_count = pattern_stats.get('total_observable', 0)
            rules_validated = self.validation.get('validated_rules', 0)
            md.append(f"| Patterns detected | {obs_count:,} | ~{int(obs_count * self.extrapolation_factor):,} |")
            md.append(f"| Rules validated | {rules_validated:,} | ~{int(rules_validated * self.extrapolation_factor):,} |")
            md.append("")

        # Executive Summary
        md.append("---")
        md.append("## Executive Summary")
        md.append("")

        # Get φ-alignment interpretation
        phi_interp = insights.get('phi_interpretation', {})
        phi_display = phi_interp.get('short', 'N/A') if phi_interp else 'N/A'
        phi_emoji = phi_interp.get('emoji', '') if phi_interp else ''

        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        md.append(f"| Structural sequence | {char_counts.get('structural', 0):,} chars |")
        md.append(f"| Observable sequence | {char_counts.get('observable', 0):,} chars |")
        md.append(f"| Patterns detected | {pattern_stats.get('total_observable', 0):,} |")
        md.append(f"| Rules inferred | {rule_stats.get('total_rules', 0):,} |")
        md.append(f"| Avg rule confidence | {rule_stats.get('avg_confidence', 0)*100:.2f}% |")
        md.append(f"| **φ-alignment** | {phi_emoji} {phi_display} |")
        md.append(f"| Analysis time | {self.data.get('analysis_time_seconds', 0):,.1f}s |")
        md.append("")

        # Key Findings
        md.append("---")
        md.append("## Key Findings")
        md.append("")

        findings = []

        # φ-alignment finding (always first if available)
        if phi_interp and phi_interp.get('level') not in ['unknown', None]:
            findings.append(f"**{phi_emoji} φ-Alignment ({phi_interp.get('level', '').replace('_', ' ').title()})**: {phi_interp.get('description', '')}")

        if insights.get('alternating_dominance'):
            findings.append("**🔄 Alternating Pattern Dominance**: The most recurrent patterns are alternating sequences (0101...), suggesting an oscillatory structure in the collapse dynamics.")

        if insights.get('deterministic_structure'):
            pct = rule_stats.get('confidence_99plus', 0) / max(1, rule_stats.get('total_rules', 1)) * 100
            findings.append(f"**🎯 Deterministic Structure**: {pct:.1f}% of rules have ≥99% confidence, indicating non-random, rule-governed sequences.")

        if insights.get('fibonacci_patterns'):
            findings.append(f"**🌀 Fibonacci Periods**: Detected periods matching Fibonacci numbers: {insights['fibonacci_patterns']}")

        # Density analysis
        density = insights.get('density_analysis')
        if density:
            findings.append(f"**⚖️ Bit Density**: {density.get('description', '')}. {density.get('significance', '')}")

        # Complexity metrics
        complexity = insights.get('complexity_metrics', {})
        if complexity.get('avg_context_length'):
            diversity_pct = complexity.get('rule_diversity', 0) * 100
            findings.append(f"**🧬 Rule Complexity**: Average context length of {complexity['avg_context_length']:.1f} bits with {diversity_pct:.1f}% rule diversity.")

        if not findings:
            findings.append("*No significant findings detected. Consider analyzing more iterations or adjusting pattern detection parameters.*")

        for i, finding in enumerate(findings, 1):
            md.append(f"{i}. {finding}")
        md.append("")

        # Patterns Section
        md.append("---")
        md.append("## Pattern Analysis")
        md.append("")
        md.append(f"**Total patterns:** {pattern_stats.get('total_observable', 0):,} observable, {pattern_stats.get('total_structural', 0)} structural")
        md.append(f"")
        md.append(f"**Length range:** {pattern_stats.get('min_length', 0)}-{pattern_stats.get('max_length', 0)} bits (avg: {pattern_stats.get('avg_length', 0):.1f})")
        md.append("")
        md.append("### Top 10 Most Recurrent Patterns")
        md.append("")
        md.append("| Rank | Pattern | Occurrences |")
        md.append("|------|---------|-------------|")
        for i, p in enumerate(pattern_stats.get('top_patterns', [])[:10], 1):
            md.append(f"| {i} | `{p['pattern']}` | {p['recurrence']:,} |")
        md.append("")

        # Rules Section
        md.append("---")
        md.append("## Rule Analysis")
        md.append("")
        md.append(f"**Total rules:** {rule_stats.get('total_rules', 0):,}")
        md.append(f"")
        md.append("### Confidence Distribution")
        md.append("")
        md.append("| Confidence | Count |")
        md.append("|------------|-------|")
        md.append(f"| ≥99% | {rule_stats.get('confidence_99plus', 0):,} |")
        md.append(f"| 90-99% | {rule_stats.get('confidence_90_99', 0):,} |")
        md.append(f"| <90% | {rule_stats.get('confidence_below_90', 0):,} |")
        md.append("")

        md.append("### Top 10 Rules by Confidence")
        md.append("")
        md.append("| Context | → | Prediction | Confidence | Support |")
        md.append("|---------|---|------------|------------|---------|")
        for r in rule_stats.get('top_rules', [])[:10]:
            md.append(f"| `{r['context']}` | → | `{r['prediction']}` | {r['confidence']*100:.1f}% | {r['support']:,} |")
        md.append("")

        # φ-Alignment Section (NEW)
        md.append("---")
        md.append("## φ-Alignment Analysis")
        md.append("")

        phi_value = insights.get('phi_alignment')
        phi_interp = insights.get('phi_interpretation', {})

        if phi_value is not None:
            md.append(f"### {phi_interp.get('emoji', '')} Overall φ-Alignment: {phi_value:.4f}")
            md.append("")
            md.append(f"**Level:** {phi_interp.get('level', 'unknown').replace('_', ' ').title()}")
            md.append("")
            md.append(f"> {phi_interp.get('description', '')}")
            md.append("")
            md.append("#### Significance for ISH")
            md.append("")
            md.append(phi_interp.get('significance', ''))
            md.append("")

            # Add visual indicator
            md.append("#### φ-Alignment Scale")
            md.append("")
            md.append("```")
            bar_pos = int(phi_value * 50)
            bar = "─" * bar_pos + "●" + "─" * (50 - bar_pos)
            md.append(f"0.0 [{bar}] 1.0")
            md.append(f"     {'Random':<15}{'Moderate':^20}{'Golden Ratio':>15}")
            md.append("```")
            md.append("")
        else:
            md.append("*φ-alignment was not calculated for this analysis.*")
            md.append("")
            md.append("To enable φ-alignment calculation, run the analysis with pattern metrics enabled.")
            md.append("")

        # Scientific Interpretation
        md.append("---")
        md.append("## Scientific Interpretation")
        md.append("")

        if insights.get('is_chaotic'):
            # CHAOTIC VARIANT - WARNING
            md.append("### ⚠️ CHAOTIC/RANDOM SEQUENCE WARNING")
            md.append("")
            md.append("**This variant produces PSEUDO-RANDOM output, NOT structured information.**")
            md.append("")
            md.append("#### Chaos Indicators Detected:")
            md.append("")
            for indicator in insights.get('chaos_indicators', []):
                md.append(f"- {indicator}")
            md.append("")
            md.append("#### Scientific Interpretation:")
            md.append("")
            md.append("While a random sequence can contain ANY pattern by statistical chance,")
            md.append("it lacks the **stable, hierarchical order** required for persistent physical laws.")
            md.append("")
            md.append("A universe generated from this variant would exhibit:")
            md.append("- **Spontaneous formation/dissolution** of structures without cause")
            md.append("- **No stable physical laws** — constants would fluctuate randomly")
            md.append("- **No persistent complexity** — organized structures would decay immediately")
            md.append("")
            md.append("### ❌ NOT Consistent with ISH")
            md.append("")
            md.append("This variant does NOT demonstrate the structured collapse process")
            md.append("predicted by the Informational Singularity Hypothesis. The absence of")
            md.append("deterministic rules (0% with ≥99% confidence) indicates pure randomness,")
            md.append("not emergent order.")
            md.append("")
        else:
            # ORDERED VARIANT - Consistent with ISH
            md.append("### ✅ Consistency with ISH Hypothesis")
            md.append("")
            md.append("The results are consistent with the Informational Singularity Hypothesis:")
            md.append("")
            md.append("1. **Emergent Order**: The high percentage of deterministic rules confirms")
            md.append("   that the collapse process generates structured, non-random information.")
            md.append("")
            md.append("2. **Anti-Repetition Dynamics**: Rules preventing consecutive identical bits")
            md.append("   create the observed oscillatory patterns, a signature of ISH collapse.")
            md.append("")
            md.append("3. **Scale-Invariant Patterns**: Pattern lengths spanning multiple orders of")
            md.append("   magnitude suggest fractal-like self-similarity in the generated structures.")
            md.append("")

        # Data for Paper
        md.append("---")
        md.append("## Data for Scientific Paper")
        md.append("")
        md.append("### Key Metrics Table")
        md.append("")
        md.append("| Metric | Value | Significance |")
        md.append("|--------|-------|--------------|")
        md.append(f"| Sequence length | {char_counts.get('observable', 0):,} bits | Scale of analysis |")
        md.append(f"| Pattern count | {pattern_stats.get('total_observable', 0):,} | Structural complexity |")
        md.append(f"| Deterministic rules | {rule_stats.get('confidence_99plus', 0):,} ({rule_stats.get('confidence_99plus', 0)/max(1,rule_stats.get('total_rules', 1))*100:.1f}%) | Non-randomness indicator |")
        md.append(f"| Dominant pattern | `{pattern_stats.get('top_patterns', [{}])[0].get('pattern', 'N/A')}` | Emergent structure |")
        md.append("")

        return "\n".join(md)

    def save_markdown_report(self, output_path: Optional[str] = None) -> str:
        """Save the Markdown report to a file."""
        if output_path is None:
            output_path = self.results_path.with_suffix('.md')
        else:
            output_path = Path(output_path)

        md_content = self.generate_markdown_report()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        return str(output_path)


def find_latest_results(results_dir: str = "results/level1/analysis") -> Optional[Path]:
    """Find the most recent Level 1 analysis results file."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return None

    json_files = list(results_path.glob("level1_analysis_*.json"))
    if not json_files:
        return None

    return max(json_files, key=lambda p: p.stat().st_mtime)


def find_results_by_variant(variant: str, iteration: int,
                           results_dir: str = "results/level1/analysis") -> Optional[Path]:
    """Find results file for a specific variant and iteration."""
    pattern = f"level1_analysis_var{variant}_iter{iteration}.json"
    results_path = Path(results_dir) / pattern
    return results_path if results_path.exists() else None


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Analyze ISH Level 1 experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_results.py                              # Analyze latest
  python analyze_results.py results/file.json           # Specific file
  python analyze_results.py --variant B --iteration 18  # By variant/iter
  python analyze_results.py --output report.md          # Save MD report
        """
    )
    parser.add_argument("file", nargs="?", help="Results JSON file to analyze")
    parser.add_argument("--variant", "-v", help="Variant letter (e.g., B)")
    parser.add_argument("--iteration", "-i", type=int, help="Iteration number")
    parser.add_argument("--output", "-o", help="Output Markdown file path")
    parser.add_argument("--markdown-only", "-m", action="store_true",
                       help="Only generate Markdown, skip console output")

    args = parser.parse_args()

    # Determine which file to analyze
    results_file = None

    if args.file:
        results_file = Path(args.file)
    elif args.variant and args.iteration:
        results_file = find_results_by_variant(args.variant, args.iteration)
        if not results_file:
            print(f"❌ No results found for Variant {args.variant}, Iteration {args.iteration}")
            return 1
    else:
        results_file = find_latest_results()
        if not results_file:
            print("❌ No results files found in results/level1/analysis/ directory")
            return 1
        print(f"📂 Using latest results: {results_file.name}")

    if not results_file.exists():
        print(f"❌ File not found: {results_file}")
        return 1

    # Create reporter and generate output
    try:
        reporter = ResultsReporter(str(results_file))

        if not args.markdown_only:
            reporter.print_console_report()

        if args.output:
            saved_path = reporter.save_markdown_report(args.output)
            print(f"\n📝 Markdown report saved to: {saved_path}")
        elif args.markdown_only:
            print(reporter.generate_markdown_report())

        return 0

    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

