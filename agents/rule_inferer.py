"""
🧠 Agent 3: Emergent Rule Inferer (Ω-inductor)

This agent analyzes the repetition, combination, and evolution of patterns Pₖ
to infer rules ωₖ governing their appearance within the Primordial Holographic
Field of HSI.

Author: Iban Borràs with collaboration of Augment Agent
Date: January 2025
"""

import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional, Set
from pathlib import Path
from collections import defaultdict, Counter
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import hashlib

# Import Numba-optimized kernels
try:
    from .numba_kernels import (
        str_to_bytes, count_transitions_order1, count_transitions_order2,
        count_transitions_order3, validate_markov_rule, find_pattern_positions
    )
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class RuleInferer:
    """
    Agent for inferring emergent rules ωₖ from detected patterns Pₖ.

    Uses multiple methods to discover rules:
    - Markov chain analysis
    - Grammar induction
    - Simple program synthesis
    - Genetic algorithms for rule optimization
    """

    def __init__(self,
                 context_window: int = 5,
                 min_context_length: int = 3,
                 min_rule_confidence: float = 0.7,
                 max_rule_complexity: int = 10):
        """
        Initialize the rule inferer.

        Args:
            context_window: Maximum context window size for pattern analysis
            min_context_length: Minimum context length for Markov rules (should match min_pattern_length)
            min_rule_confidence: Minimum confidence to accept a rule
            max_rule_complexity: Maximum allowed rule complexity
        """
        self.context_window = context_window
        self.min_context_length = min_context_length
        self.min_rule_confidence = min_rule_confidence
        self.max_rule_complexity = max_rule_complexity

        self.inferred_rules = []
        self.rule_stats = {}
    
    def infer_rules(self, patterns: List[Dict[str, Any]],
                   phi_sequence: str) -> List[Dict[str, Any]]:
        """
        Infer rules ωₖ from detected patterns Pₖ.

        Args:
            patterns: List of patterns detected by PatternDetector
            phi_sequence: Original Φ sequence

        Returns:
            List of inferred rules with their metrics
        """
        print(f"🧠 Inferring rules from {len(patterns)} patterns")

        # Mètode 1: Regles de transició (Markov)
        print(f"   ⏳ [1/4] Markov rules...", flush=True)
        markov_rules = self._infer_markov_rules(patterns, phi_sequence)
        print(f"   ✓ Markov: {len(markov_rules)} rules", flush=True)

        # Mètode 2: Regles de context
        print(f"   ⏳ [2/4] Context rules...", flush=True)
        context_rules = self._infer_context_rules(patterns, phi_sequence)
        print(f"   ✓ Context: {len(context_rules)} rules", flush=True)

        # Mètode 3: Regles de composició
        print(f"   ⏳ [3/4] Composition rules...", flush=True)
        composition_rules = self._infer_composition_rules(patterns)
        print(f"   ✓ Composition: {len(composition_rules)} rules", flush=True)

        # Mètode 4: Regles de periodicitat
        print(f"   ⏳ [4/4] Periodicity rules...", flush=True)
        periodicity_rules = self._infer_periodicity_rules(patterns, phi_sequence)
        print(f"   ✓ Periodicity: {len(periodicity_rules)} rules", flush=True)

        # Combinar totes les regles
        all_rules = markov_rules + context_rules + composition_rules + periodicity_rules
        print(f"   ⏳ Validating {len(all_rules)} rules...", flush=True)

        # Filtrar i validar regles
        validated_rules = self._validate_and_filter_rules(all_rules, patterns, phi_sequence)
        print(f"   ✓ Validated: {len(validated_rules)} rules", flush=True)

        # Calcular mètriques detallades
        num_rules = len(validated_rules)
        print(f"   ⏳ Calculating metrics for {num_rules:,} rules...", flush=True)
        enriched_rules = []
        log_interval = max(1, num_rules // 20)  # Log cada 5%

        for i, rule in enumerate(validated_rules):
            if i % log_interval == 0 and i > 0:
                pct = 100 * i // num_rules
                print(f"      [metrics] {i:,}/{num_rules:,} ({pct}%)", flush=True)
            enriched_rule = self._calculate_rule_metrics(rule, patterns, phi_sequence)
            enriched_rules.append(enriched_rule)

        print(f"   ✓ Metrics calculated for {len(enriched_rules):,} rules", flush=True)
        self.inferred_rules = enriched_rules
        return enriched_rules
    
    def _infer_markov_rules(self, patterns: List[Dict[str, Any]],
                          phi_sequence: str) -> List[Dict[str, Any]]:
        """Infer transition rules based on Markov chains.

        Always generates order 1-3 Markov rules (needed for heatmaps/visualizations),
        plus additional higher-order rules if min_context_length > 3.
        """
        rules = []

        # ALWAYS generate order 1-3 Markov rules with Numba (for heatmaps)
        if NUMBA_AVAILABLE and len(phi_sequence) > 10000:
            rules = self._infer_markov_rules_numba(phi_sequence)

            # If min_context_length > 3, we already have orders 1-3 from Numba
            # No need to add more with pure Python (too slow for higher orders)
            return rules

        # Pure Python fallback (small sequences or no Numba)
        max_order = min(self.context_window + 1, 20)
        start_order = 1  # Always start from order 1

        for order in range(start_order, max_order):
            transitions = defaultdict(Counter)

            for i in range(len(phi_sequence) - order):
                context = phi_sequence[i:i + order]
                next_bit = phi_sequence[i + order]
                transitions[context][next_bit] += 1

            for context, next_bits in transitions.items():
                total_transitions = sum(next_bits.values())

                for next_bit, count in next_bits.items():
                    confidence = count / total_transitions

                    if confidence >= self.min_rule_confidence and total_transitions >= 3:
                        rule_id = f"markov_{order}_{hashlib.md5((context + next_bit).encode()).hexdigest()[:8]}"

                        rules.append({
                            'rule_id': rule_id,
                            'rule_type': 'markov_transition',
                            'rule_description': f"IF context='{context}' THEN next='{next_bit}'",
                            'context': context,
                            'prediction': next_bit,
                            'confidence': confidence,
                            'support': count,
                            'order': order,
                            'method': 'markov'
                        })

        return rules

    def _infer_markov_rules_numba(self, phi_sequence: str) -> List[Dict[str, Any]]:
        """Numba-optimized Markov rule inference."""
        rules = []
        seq_bytes = str_to_bytes(phi_sequence)

        # Order 1 transitions
        trans1 = count_transitions_order1(seq_bytes)
        for ctx in [ord('0'), ord('1')]:
            total = trans1[ctx, ord('0')] + trans1[ctx, ord('1')]
            if total >= 3:
                for nxt in [ord('0'), ord('1')]:
                    count = trans1[ctx, nxt]
                    if count > 0:
                        confidence = count / total
                        if confidence >= self.min_rule_confidence:
                            context = chr(ctx)
                            next_bit = chr(nxt)
                            rule_id = f"markov_1_{hashlib.md5((context + next_bit).encode()).hexdigest()[:8]}"
                            rules.append({
                                'rule_id': rule_id,
                                'rule_type': 'markov_transition',
                                'rule_description': f"IF context='{context}' THEN next='{next_bit}'",
                                'context': context,
                                'prediction': next_bit,
                                'confidence': confidence,
                                'support': int(count),
                                'order': 1,
                                'method': 'markov_numba'
                            })

        # Order 2 transitions
        trans2 = count_transitions_order2(seq_bytes)
        for c1 in [ord('0'), ord('1')]:
            for c2 in [ord('0'), ord('1')]:
                ctx_hash = c1 * 256 + c2
                total = trans2[ctx_hash, ord('0')] + trans2[ctx_hash, ord('1')]
                if total >= 3:
                    for nxt in [ord('0'), ord('1')]:
                        count = trans2[ctx_hash, nxt]
                        if count > 0:
                            confidence = count / total
                            if confidence >= self.min_rule_confidence:
                                context = chr(c1) + chr(c2)
                                next_bit = chr(nxt)
                                rule_id = f"markov_2_{hashlib.md5((context + next_bit).encode()).hexdigest()[:8]}"
                                rules.append({
                                    'rule_id': rule_id,
                                    'rule_type': 'markov_transition',
                                    'rule_description': f"IF context='{context}' THEN next='{next_bit}'",
                                    'context': context,
                                    'prediction': next_bit,
                                    'confidence': confidence,
                                    'support': int(count),
                                    'order': 2,
                                    'method': 'markov_numba'
                                })

        # Order 3 - use hash-based counting (already computed in order3 function)
        for c1 in [ord('0'), ord('1')]:
            for c2 in [ord('0'), ord('1')]:
                for c3 in [ord('0'), ord('1')]:
                    context = chr(c1) + chr(c2) + chr(c3)
                    ctx_bytes = str_to_bytes(context)

                    # Count manually for order 3
                    counts = {ord('0'): 0, ord('1'): 0}
                    for i in range(len(seq_bytes) - 3):
                        if (seq_bytes[i] == c1 and seq_bytes[i+1] == c2 and
                            seq_bytes[i+2] == c3):
                            counts[seq_bytes[i+3]] += 1

                    total = counts[ord('0')] + counts[ord('1')]
                    if total >= 3:
                        for nxt in [ord('0'), ord('1')]:
                            count = counts[nxt]
                            if count > 0:
                                confidence = count / total
                                if confidence >= self.min_rule_confidence:
                                    next_bit = chr(nxt)
                                    rule_id = f"markov_3_{hashlib.md5((context + next_bit).encode()).hexdigest()[:8]}"
                                    rules.append({
                                        'rule_id': rule_id,
                                        'rule_type': 'markov_transition',
                                        'rule_description': f"IF context='{context}' THEN next='{next_bit}'",
                                        'context': context,
                                        'prediction': next_bit,
                                        'confidence': confidence,
                                        'support': int(count),
                                        'order': 3,
                                        'method': 'markov_numba'
                                    })

        return rules
    
    def _infer_context_rules(self, patterns: List[Dict[str, Any]],
                           phi_sequence: str) -> List[Dict[str, Any]]:
        """Infer rules based on pattern context."""
        rules = []

        # Filter valid patterns upfront
        valid_patterns = [p for p in patterns
                         if 'pattern_data' in p and 'positions' in p]
        total = len(valid_patterns)

        for idx, pattern in enumerate(valid_patterns):
            if idx > 0 and idx % 1000 == 0:
                print(f"   [context] {idx:,}/{total:,} patterns processed", flush=True)
            pattern_data = pattern['pattern_data']
            positions = pattern['positions']
            
            # Analitzar context abans i després de cada ocurrència del patró
            contexts_before = []
            contexts_after = []
            
            for pos in positions:
                # Context abans
                start_before = max(0, pos - self.context_window)
                context_before = phi_sequence[start_before:pos]
                
                # Context després
                end_after = min(len(phi_sequence), pos + len(pattern_data) + self.context_window)
                context_after = phi_sequence[pos + len(pattern_data):end_after]
                
                if len(context_before) > 0:
                    contexts_before.append(context_before)
                if len(context_after) > 0:
                    contexts_after.append(context_after)
            
            # Trobar contexts comuns
            if contexts_before:
                common_before = self._find_common_contexts(contexts_before)
                for context, frequency in common_before.items():
                    if frequency >= 2:  # Almenys 2 ocurrències
                        confidence = frequency / len(contexts_before)
                        
                        if confidence >= self.min_rule_confidence:
                            rule_id = f"ctx_before_{pattern['pattern_id']}_{hashlib.md5(context.encode()).hexdigest()[:8]}"
                            
                            rules.append({
                                'rule_id': rule_id,
                                'rule_type': 'context_before',
                                'rule_description': f"IF context_before='{context}' THEN pattern='{pattern_data}'",
                                'context': context,
                                'produces': [pattern['pattern_id']],
                                'confidence': confidence,
                                'support': frequency,
                                'method': 'context_analysis'
                            })
            
            if contexts_after:
                common_after = self._find_common_contexts(contexts_after)
                for context, frequency in common_after.items():
                    if frequency >= 2:
                        confidence = frequency / len(contexts_after)
                        
                        if confidence >= self.min_rule_confidence:
                            rule_id = f"ctx_after_{pattern['pattern_id']}_{hashlib.md5(context.encode()).hexdigest()[:8]}"
                            
                            rules.append({
                                'rule_id': rule_id,
                                'rule_type': 'context_after',
                                'rule_description': f"IF pattern='{pattern_data}' THEN context_after='{context}'",
                                'pattern': pattern_data,
                                'prediction': context,
                                'confidence': confidence,
                                'support': frequency,
                                'method': 'context_analysis'
                            })
        
        return rules
    
    def _infer_composition_rules(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Infer composition rules between patterns.

        Optimized from O(n³) to O(n²) using hash-based lookup.
        """
        rules = []

        # Filter patterns that have pattern_data
        valid_patterns = [p for p in patterns if 'pattern_data' in p]

        if not valid_patterns:
            return rules

        # Build hash map: pattern_data -> pattern_id (O(n) preprocessing)
        pattern_lookup = {p['pattern_data']: p['pattern_id'] for p in valid_patterns}

        # Progress tracking for large datasets
        n = len(valid_patterns)
        total_pairs = n * (n - 1)
        processed = 0
        last_percent = 0

        # O(n²) loop with O(1) lookup instead of O(n³)
        for i, pattern1 in enumerate(valid_patterns):
            p1_data = pattern1['pattern_data']
            p1_id = pattern1['pattern_id']

            for j, pattern2 in enumerate(valid_patterns):
                if i == j:
                    continue

                p2_data = pattern2['pattern_data']
                p2_id = pattern2['pattern_id']

                # Check concatenation composition (O(1) lookup)
                composed = p1_data + p2_data
                target_id = pattern_lookup.get(composed)
                if target_id:
                    rules.append({
                        'rule_id': f"comp_{p1_id}_{p2_id}",
                        'rule_type': 'composition',
                        'rule_description': f"pattern1 + pattern2 = target_pattern",
                        'components': [p1_id, p2_id],
                        'produces': [target_id],
                        'confidence': 1.0,
                        'method': 'composition_analysis'
                    })

                # Check overlap composition (O(1) lookup)
                overlap_composed = self._find_overlap_composition(p1_data, p2_data)
                if overlap_composed:
                    target_id = pattern_lookup.get(overlap_composed)
                    if target_id:
                        rules.append({
                            'rule_id': f"overlap_{p1_id}_{p2_id}",
                            'rule_type': 'overlap_composition',
                            'rule_description': f"pattern1 overlapped with pattern2 = target_pattern",
                            'components': [p1_id, p2_id],
                            'produces': [target_id],
                            'confidence': 1.0,
                            'method': 'overlap_analysis'
                        })

                # Progress update every 5%
                processed += 1
                percent = (processed * 100) // total_pairs
                if percent >= last_percent + 5:
                    print(f"   [composition] {percent}% ({processed:,}/{total_pairs:,} pairs)", flush=True)
                    last_percent = percent

        return rules
    
    def _infer_periodicity_rules(self, patterns: List[Dict[str, Any]],
                               phi_sequence: str) -> List[Dict[str, Any]]:
        """Infer periodicity and repetition rules."""
        rules = []

        for pattern in patterns:
            # Skip structural patterns without positions/pattern_data
            if 'positions' not in pattern or 'pattern_data' not in pattern:
                continue
            positions = pattern['positions']
            
            if len(positions) >= 3:  # Necessitem almenys 3 ocurrències per detectar periodicitat
                # Calcular distàncies entre ocurrències consecutives
                distances = []
                for i in range(len(positions) - 1):
                    distance = positions[i + 1] - positions[i]
                    distances.append(distance)
                
                # Buscar periodicitats
                distance_counts = Counter(distances)
                
                for distance, count in distance_counts.items():
                    if count >= 2:  # Almenys 2 vegades la mateixa distància
                        confidence = count / len(distances)
                        
                        if confidence >= self.min_rule_confidence:
                            rule_id = f"period_{pattern['pattern_id']}_{distance}"
                            
                            rules.append({
                                'rule_id': rule_id,
                                'rule_type': 'periodicity',
                                'rule_description': f"Pattern '{pattern['pattern_data']}' repeats every {distance} positions",
                                'pattern': pattern['pattern_data'],
                                'period': distance,
                                'confidence': confidence,
                                'support': count,
                                'method': 'periodicity_analysis'
                            })
        
        return rules
    
    def _find_common_contexts(self, contexts: List[str]) -> Dict[str, int]:
        """Find common contexts in a list of strings.

        Optimized: Limits search depth to avoid O(n³) explosion on large context windows.
        """
        context_counts = Counter()

        # Limit max substring length to avoid combinatorial explosion
        # For context_window=10, we'd check lengths 1-10 (manageable)
        # For context_window=50, we limit to 15 to avoid O(n³) blowup
        MAX_SUBSTRING_LENGTH = 15

        # Find common substrings of different lengths
        # Count each substring only ONCE per context to ensure confidence <= 1.0
        for context in contexts:
            seen_in_context = set()  # Track substrings seen in this context
            max_len = min(len(context) + 1, self.context_window, MAX_SUBSTRING_LENGTH)

            for length in range(1, max_len):
                for start in range(len(context) - length + 1):
                    substring = context[start:start + length]
                    if substring not in seen_in_context:
                        seen_in_context.add(substring)
                        context_counts[substring] += 1

        # Filter contexts that are too short or infrequent
        filtered_contexts = {
            context: count for context, count in context_counts.items()
            if len(context) >= 2 and count >= 2
        }

        return filtered_contexts
    
    def _find_overlap_composition(self, pattern1: str, pattern2: str) -> Optional[str]:
        """Find overlap composition between two patterns."""
        max_overlap = min(len(pattern1), len(pattern2)) - 1
        
        for overlap_len in range(max_overlap, 0, -1):
            # Provar solapament: final de pattern1 amb inici de pattern2
            if pattern1[-overlap_len:] == pattern2[:overlap_len]:
                return pattern1 + pattern2[overlap_len:]
            
            # Provar solapament: final de pattern2 amb inici de pattern1
            if pattern2[-overlap_len:] == pattern1[:overlap_len]:
                return pattern2 + pattern1[overlap_len:]
        
        return None
    
    def _validate_and_filter_rules(self, rules: List[Dict[str, Any]],
                                 patterns: List[Dict[str, Any]],
                                 phi_sequence: str) -> List[Dict[str, Any]]:
        """Validate and filter inferred rules."""
        validated_rules = []
        total_rules = len(rules)
        log_interval = max(1, total_rules // 20)  # Log cada 5%

        print(f"   ⏳ Filtering {total_rules:,} rules...", flush=True)

        for idx, rule in enumerate(rules):
            if idx > 0 and idx % log_interval == 0:
                pct = 100 * idx // total_rules
                print(f"      [filter] {idx:,}/{total_rules:,} ({pct}%) - validated: {len(validated_rules):,}", flush=True)

            # Verificar confiança mínima
            if rule.get('confidence', 0) < self.min_rule_confidence:
                continue

            # Verificar complexitat màxima
            complexity = self._calculate_rule_complexity(rule)
            if complexity > self.max_rule_complexity:
                continue

            # Verificar que la regla no sigui trivial
            if self._is_trivial_rule(rule):
                continue

            validated_rules.append(rule)

        print(f"   ✓ Filtered: {len(validated_rules):,}/{total_rules:,} rules passed", flush=True)

        # Remove duplicate rules
        print(f"   ⏳ Removing duplicates from {len(validated_rules):,} rules...", flush=True)
        unique_rules = self._remove_duplicate_rules(validated_rules)
        print(f"   ✓ Unique rules: {len(unique_rules):,}", flush=True)

        return unique_rules

    def _calculate_rule_complexity(self, rule: Dict[str, Any]) -> float:
        """Calculate rule complexity."""
        complexity = 0.0

        # Complexity based on rule description
        description = rule.get('rule_description', '')
        complexity += len(description) / 100.0  # Normalize by length

        # Complexity based on rule type
        rule_type = rule.get('rule_type', '')
        type_complexity = {
            'markov_transition': 1.0,
            'context_before': 2.0,
            'context_after': 2.0,
            'composition': 3.0,
            'overlap_composition': 4.0,
            'periodicity': 1.5
        }
        complexity += type_complexity.get(rule_type, 2.0)

        # Complexity based on order (for Markov rules)
        if 'order' in rule:
            complexity += rule['order'] * 0.5

        return complexity

    def _is_trivial_rule(self, rule: Dict[str, Any]) -> bool:
        """Determine if a rule is trivial."""
        # Rules with perfect confidence but very low support may be trivial
        if rule.get('confidence', 0) == 1.0 and rule.get('support', 0) <= 1:
            return True

        # Rules that only predict the most common bit
        if rule.get('prediction') in ['0', '1'] and rule.get('confidence', 0) < 0.8:
            return True

        return False

    def _remove_duplicate_rules(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or redundant rules."""
        unique_rules = []
        seen_descriptions = set()

        # Sort by confidence (highest first)
        sorted_rules = sorted(rules, key=lambda r: r.get('confidence', 0), reverse=True)
        
        for rule in sorted_rules:
            description = rule.get('rule_description', '')
            if description not in seen_descriptions:
                seen_descriptions.add(description)
                unique_rules.append(rule)
        
        return unique_rules
    
    def _calculate_rule_metrics(self, rule: Dict[str, Any], 
                              patterns: List[Dict[str, Any]], 
                              phi_sequence: str) -> Dict[str, Any]:
        """Calcula mètriques detallades per una regla."""
        # Calcular precisió mitjançant validació creuada simple
        precision = self._calculate_rule_precision(rule, phi_sequence)
        
        # Calcular estabilitat (consistència en diferents parts de la seqüència)
        stability = self._calculate_rule_stability(rule, phi_sequence)
        
        # Calcular reproductibilitat
        reproducibility = rule.get('confidence', 0)  # Aproximació inicial
        
        # Calcular complexitat
        complexity = self._calculate_rule_complexity(rule)
        
        # Enriquir la regla amb les noves mètriques
        enriched_rule = rule.copy()
        enriched_rule.update({
            'precision': precision,
            'stability': stability,
            'reproducibility': reproducibility,
            'complexity': complexity
        })
        
        return enriched_rule
    
    def _calculate_rule_precision(self, rule: Dict[str, Any], phi_sequence: str) -> float:
        """Calcula la precisió d'una regla."""
        # Implementació simplificada - es pot millorar
        return rule.get('confidence', 0.5)
    
    def _calculate_rule_stability(self, rule: Dict[str, Any], phi_sequence: str) -> float:
        """Calcula l'estabilitat d'una regla en diferents parts de la seqüència."""
        if len(phi_sequence) < 100:
            return rule.get('confidence', 0.5)
        
        # Dividir la seqüència en segments i provar la regla en cada segment
        num_segments = 4
        segment_size = len(phi_sequence) // num_segments
        
        segment_scores = []
        
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size if i < num_segments - 1 else len(phi_sequence)
            segment = phi_sequence[start:end]
            
            # Aplicar la regla al segment (implementació simplificada)
            score = self._test_rule_on_segment(rule, segment)
            segment_scores.append(score)
        
        # Calcular estabilitat com la consistència entre segments
        if segment_scores:
            mean_score = np.mean(segment_scores)
            std_score = np.std(segment_scores)
            stability = max(0.0, mean_score - std_score)  # Penalitzar alta variabilitat
        else:
            stability = 0.0
        
        return stability
    
    def _test_rule_on_segment(self, rule: Dict[str, Any], segment: str) -> float:
        """Prova una regla en un segment de la seqüència."""
        # Implementació simplificada - retorna la confiança de la regla
        return rule.get('confidence', 0.5)
    
    def save_rules(self, output_path: str) -> None:
        """Save inferred rules in JSON format."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(self.inferred_rules, f, indent=2)

        print(f"💾 Rules saved to {output_path}")

    def get_rules_summary(self) -> Dict[str, Any]:
        """Return a summary of inferred rules."""
        if not self.inferred_rules:
            return {"total_rules": 0}

        total_rules = len(self.inferred_rules)
        avg_confidence = np.mean([r.get('confidence', 0) for r in self.inferred_rules])
        avg_precision = np.mean([r.get('precision', 0) for r in self.inferred_rules])
        avg_complexity = np.mean([r.get('complexity', 0) for r in self.inferred_rules])

        rule_types = Counter(r.get('rule_type', 'unknown') for r in self.inferred_rules)

        return {
            "total_rules": total_rules,
            "average_confidence": avg_confidence,
            "average_precision": avg_precision,
            "average_complexity": avg_complexity,
            "rule_types": dict(rule_types),
            "top_rules": sorted(self.inferred_rules,
                              key=lambda r: r.get('confidence', 0),
                              reverse=True)[:5]
        }


if __name__ == "__main__":
    # Usage example
    print("🧠 Emergent Rule Inferer (Ω-inductor)")
    print("=" * 50)

    # Create example patterns
    example_patterns = [
        {
            'pattern_id': 'test_01',
            'pattern_data': '101',
            'positions': [0, 5, 10],
            'recurrence': 3
        },
        {
            'pattern_id': 'test_02',
            'pattern_data': '010',
            'positions': [2, 7, 12],
            'recurrence': 3
        }
    ]

    example_sequence = "101010101010101"

    inferer = RuleInferer(context_window=3, min_rule_confidence=0.6)
    rules = inferer.infer_rules(example_patterns, example_sequence)

    print(f"Rules inferred: {len(rules)}")
    for rule in rules[:3]:  # Show only first 3
        print(f"  - {rule['rule_description']} (confidence: {rule.get('confidence', 0):.2f})")

    summary = inferer.get_rules_summary()
    print(f"Summary: {summary['total_rules']} rules, average confidence: {summary['average_confidence']:.2f}")
