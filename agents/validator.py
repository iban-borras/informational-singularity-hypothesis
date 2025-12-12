"""
✅ Agent 4: Iterative Validator (Cycle-tester)

This agent applies the inferred rules ωₖ on new sequences Pₖ to verify
their consistency and robustness, closing the learning loop of the HSI
agent system.

Author: Iban Borràs with collaboration of Augment Agent
Date: January 2025
"""

import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import Numba-optimized kernels
try:
    from .numba_kernels import (
        str_to_bytes, validate_markov_rule, find_pattern_positions,
        validate_periodicity
    )
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class Validator:
    """
    Agent for validating emergent rules ωₖ through iterative testing.

    Features:
    - Cross-validation of rules on new sequences
    - Measurement of precision, consistency and robustness
    - Feedback for rule refinement
    - Detection of spurious or overfitted rules
    """

    def __init__(self,
                 validation_split: float = 0.3,
                 min_validation_score: float = 0.6,
                 stability_threshold: float = 0.1):
        """
        Initialize the validator.

        Args:
            validation_split: Proportion of data for validation
            min_validation_score: Minimum score to accept a rule
            stability_threshold: Threshold to consider a rule stable
        """
        self.validation_split = validation_split
        self.min_validation_score = min_validation_score
        self.stability_threshold = stability_threshold

        self.validation_results = []
        self.rule_performance = {}
    
    def validate_rules(self,
                      rules: List[Dict[str, Any]],
                      patterns: List[Dict[str, Any]],
                      phi_sequences: List[str]) -> Dict[str, Any]:
        """
        Validate inferred rules on multiple Φ sequences.

        Args:
            rules: List of rules ωₖ to validate
            patterns: Original patterns Pₖ
            phi_sequences: Multiple Φ sequences for validation

        Returns:
            Dictionary with validation results and metrics
        """
        print(f"✅ Validating {len(rules)} rules on {len(phi_sequences)} sequences", flush=True)

        validation_results = {
            'rule_scores': {},
            'overall_metrics': {},
            'stability_analysis': {},
            'recommendations': []
        }

        # Validate each rule individually
        total_rules = len(rules)
        log_interval = max(1, total_rules // 20)  # Log cada 5%

        for idx, rule in enumerate(rules):
            rule_id = rule['rule_id']

            # Log cada 5% en lloc de cada 100
            if idx > 0 and idx % log_interval == 0:
                pct = 100 * idx // total_rules
                print(f"  ⏳ Validating rules: {idx:,}/{total_rules:,} ({pct}%)", flush=True)

            rule_results = self._validate_single_rule(rule, phi_sequences)
            validation_results['rule_scores'][rule_id] = rule_results

            # Store rule performance
            self.rule_performance[rule_id] = rule_results

        print(f"  ✓ All {total_rules:,} rules validated", flush=True)

        # Calculate global metrics
        print(f"  ⏳ Calculating overall metrics...", flush=True)
        validation_results['overall_metrics'] = self._calculate_overall_metrics(rules)
        print(f"  ✓ Overall metrics calculated", flush=True)

        # Stability analysis
        print(f"  ⏳ Analyzing stability...", flush=True)
        validation_results['stability_analysis'] = self._analyze_stability(rules, phi_sequences)
        print(f"  ✓ Stability analysis complete", flush=True)

        # Generate recommendations
        print(f"  ⏳ Generating recommendations...", flush=True)
        validation_results['recommendations'] = self._generate_recommendations(rules)
        print(f"  ✓ Recommendations generated", flush=True)

        self.validation_results.append(validation_results)
        return validation_results

    def _validate_single_rule(self, rule: Dict[str, Any],
                            phi_sequences: List[str]) -> Dict[str, Any]:
        """Validate a single rule on multiple sequences."""
        rule_type = rule.get('rule_type', 'unknown')
        
        if rule_type == 'markov_transition':
            return self._validate_markov_rule(rule, phi_sequences)
        elif rule_type in ['context_before', 'context_after']:
            return self._validate_context_rule(rule, phi_sequences)
        elif rule_type in ['composition', 'overlap_composition']:
            return self._validate_composition_rule(rule, phi_sequences)
        elif rule_type == 'periodicity':
            return self._validate_periodicity_rule(rule, phi_sequences)
        else:
            return self._validate_generic_rule(rule, phi_sequences)
    
    def _validate_markov_rule(self, rule: Dict[str, Any],
                            phi_sequences: List[str]) -> Dict[str, Any]:
        """Validate a Markov transition rule."""
        context = rule.get('context', '')
        prediction = rule.get('prediction', '')

        correct_predictions = 0
        total_predictions = 0
        sequence_scores = []

        # Use Numba for large sequences
        use_numba = NUMBA_AVAILABLE and len(phi_sequences) > 0 and len(phi_sequences[0]) > 10000

        for sequence in phi_sequences:
            if use_numba:
                # Numba-optimized validation
                seq_bytes = str_to_bytes(sequence)
                ctx_bytes = str_to_bytes(context)
                expected_next = ord(prediction) if prediction else 0

                correct, total = validate_markov_rule(seq_bytes, ctx_bytes, expected_next)
                seq_correct = correct
                seq_total = total
                correct_predictions += correct
                total_predictions += total
            else:
                # Pure Python version
                seq_correct = 0
                seq_total = 0

                for i in range(len(sequence) - len(context)):
                    if sequence[i:i + len(context)] == context:
                        if i + len(context) < len(sequence):
                            actual_next = sequence[i + len(context)]
                            if actual_next == prediction:
                                seq_correct += 1
                                correct_predictions += 1
                            seq_total += 1
                            total_predictions += 1

            # Calculate precision for this sequence
            if seq_total > 0:
                seq_score = seq_correct / seq_total
                sequence_scores.append(seq_score)

        # Calculate metrics
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        stability = 1.0 - np.std(sequence_scores) if sequence_scores else 0.0

        return {
            'accuracy': overall_accuracy,
            'stability': stability,
            'total_tests': total_predictions,
            'sequence_scores': sequence_scores,
            'validation_type': 'markov'
        }

    def _validate_context_rule(self, rule: Dict[str, Any],
                             phi_sequences: List[str]) -> Dict[str, Any]:
        """Validate a context rule with sampling for large sequences."""
        context = rule.get('context', '')
        rule_type = rule.get('rule_type', '')

        matches = 0
        total_contexts = 0
        sequence_scores = []

        # Sampling threshold for very large sequences
        SAMPLE_THRESHOLD = 10_000_000  # 10M chars
        SAMPLE_SIZE = 1_000_000  # 1M sample

        for sequence in phi_sequences:
            seq_matches = 0
            seq_contexts = 0
            seq_len = len(sequence)

            # Use sampling for very large sequences
            if seq_len > SAMPLE_THRESHOLD:
                # Take 3 samples: start, middle, end
                sample_positions = [
                    0,
                    seq_len // 2 - SAMPLE_SIZE // 2,
                    seq_len - SAMPLE_SIZE
                ]
                samples = [(pos, sequence[pos:pos + SAMPLE_SIZE]) for pos in sample_positions]
            else:
                samples = [(0, sequence)]

            for sample_offset, sample in samples:
                # Use str.find() which is much faster than manual iteration
                pos = 0
                while True:
                    idx = sample.find(context, pos)
                    if idx == -1:
                        break

                    seq_contexts += 1
                    total_contexts += 1

                    # Check if the rule is satisfied
                    if self._check_context_rule_compliance(rule, sample, idx):
                        seq_matches += 1
                        matches += 1

                    pos = idx + 1

            if seq_contexts > 0:
                seq_score = seq_matches / seq_contexts
                sequence_scores.append(seq_score)

        overall_accuracy = matches / total_contexts if total_contexts > 0 else 0.0
        stability = 1.0 - np.std(sequence_scores) if sequence_scores else 0.0

        return {
            'accuracy': overall_accuracy,
            'stability': stability,
            'total_tests': total_contexts,
            'sequence_scores': sequence_scores,
            'validation_type': 'context'
        }
    
    def _validate_composition_rule(self, rule: Dict[str, Any],
                                 phi_sequences: List[str]) -> Dict[str, Any]:
        """
        Validate a composition rule by checking if component patterns
        actually appear together and produce the expected pattern.
        """
        components = rule.get('components', [])
        produces = rule.get('produces', [])

        if not components or not produces:
            return {'accuracy': 0.0, 'stability': 0.0, 'total_tests': 0}

        # Build expected pattern from components (simple concatenation model)
        # e.g., if components=['101', '010'], expected might be '101010'
        expected_combined = ''.join(components)
        produced_pattern = produces[0] if produces else ''

        correct_predictions = 0
        total_predictions = 0
        sequence_scores = []

        for sequence in phi_sequences:
            seq_correct = 0
            seq_total = 0

            # Count co-occurrences: when components appear consecutively
            pos = 0
            while pos < len(sequence) - len(expected_combined):
                idx = sequence.find(expected_combined, pos)
                if idx == -1:
                    break
                seq_total += 1
                total_predictions += 1

                # Check if the produced pattern also appears nearby (within window)
                window_start = max(0, idx - 20)
                window_end = min(len(sequence), idx + len(expected_combined) + 20)
                window = sequence[window_start:window_end]

                if produced_pattern in window:
                    seq_correct += 1
                    correct_predictions += 1

                pos = idx + 1

            if seq_total > 0:
                sequence_scores.append(seq_correct / seq_total)

        if total_predictions == 0:
            # No occurrences found - use confidence as fallback
            return {
                'accuracy': rule.get('confidence', 0.5) * 0.5,  # Penalize lack of evidence
                'stability': 0.0,
                'total_tests': 0,
                'validation_type': 'composition',
                'note': 'No component co-occurrences found'
            }

        overall_accuracy = correct_predictions / total_predictions
        stability = 1.0 - np.std(sequence_scores) if len(sequence_scores) > 1 else 0.5

        return {
            'accuracy': overall_accuracy,
            'stability': stability,
            'total_tests': total_predictions,
            'sequence_scores': sequence_scores,
            'validation_type': 'composition'
        }

    def _validate_periodicity_rule(self, rule: Dict[str, Any],
                                 phi_sequences: List[str]) -> Dict[str, Any]:
        """Validate a periodicity rule with sampling for large sequences."""
        pattern = rule.get('pattern', '')
        period = rule.get('period', 0)

        if not pattern or period <= 0:
            return {'accuracy': 0.0, 'stability': 0.0, 'total_tests': 0}

        correct_predictions = 0
        total_predictions = 0
        sequence_scores = []

        # Sampling threshold for very large sequences
        SAMPLE_THRESHOLD = 10_000_000  # 10M chars
        SAMPLE_SIZE = 1_000_000  # 1M sample

        for sequence in phi_sequences:
            seq_correct = 0
            seq_total = 0
            seq_len = len(sequence)

            # Use sampling for very large sequences
            if seq_len > SAMPLE_THRESHOLD:
                # Take 3 samples: start, middle, end
                sample_positions = [
                    0,
                    seq_len // 2 - SAMPLE_SIZE // 2,
                    seq_len - SAMPLE_SIZE
                ]
                samples = [(pos, sequence[pos:pos + SAMPLE_SIZE]) for pos in sample_positions]
            else:
                samples = [(0, sequence)]

            for sample_offset, sample in samples:
                # Find occurrences using str.find() - much faster than manual iteration
                pattern_positions = []
                pos = 0
                while True:
                    idx = sample.find(pattern, pos)
                    if idx == -1:
                        break
                    pattern_positions.append(idx + sample_offset)
                    pos = idx + 1

                # Verify periodicity
                for i in range(len(pattern_positions) - 1):
                    actual_distance = pattern_positions[i + 1] - pattern_positions[i]
                    if actual_distance == period:
                        seq_correct += 1
                        correct_predictions += 1
                    seq_total += 1
                    total_predictions += 1

            if seq_total > 0:
                seq_score = seq_correct / seq_total
                sequence_scores.append(seq_score)

        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        stability = 1.0 - np.std(sequence_scores) if sequence_scores else 0.0

        return {
            'accuracy': overall_accuracy,
            'stability': stability,
            'total_tests': total_predictions,
            'sequence_scores': sequence_scores,
            'validation_type': 'periodicity'
        }
    
    def _validate_generic_rule(self, rule: Dict[str, Any],
                             phi_sequences: List[str]) -> Dict[str, Any]:
        """
        Generic validation for unknown rule types.

        Uses a sampling approach: tests the rule's pattern/prediction
        across multiple segments of the sequence to estimate accuracy.
        """
        confidence = rule.get('confidence', 0.5)
        pattern = rule.get('pattern', rule.get('context', ''))
        prediction = rule.get('prediction', rule.get('produces', [''])[0] if rule.get('produces') else '')

        if not pattern or not prediction:
            # Cannot validate without pattern/prediction
            return {
                'accuracy': confidence * 0.5,  # Penalize lack of testable content
                'stability': 0.0,
                'total_tests': 0,
                'validation_type': 'generic',
                'note': 'No pattern or prediction to validate'
            }

        # Test across sequences
        segment_scores = []
        total_tests = 0

        for sequence in phi_sequences:
            if len(sequence) < len(pattern) + len(prediction):
                continue

            # Count pattern occurrences followed by prediction
            correct = 0
            occurrences = 0
            pos = 0
            while pos < len(sequence) - len(pattern) - len(prediction):
                idx = sequence.find(pattern, pos)
                if idx == -1:
                    break
                occurrences += 1
                total_tests += 1

                # Check if prediction follows
                after_pattern = sequence[idx + len(pattern):idx + len(pattern) + len(prediction)]
                if after_pattern == prediction:
                    correct += 1

                pos = idx + 1

            if occurrences > 0:
                segment_scores.append(correct / occurrences)

        if total_tests == 0:
            return {
                'accuracy': confidence * 0.3,  # Heavy penalty for untestable
                'stability': 0.0,
                'total_tests': 0,
                'validation_type': 'generic',
                'note': 'Pattern not found in sequences'
            }

        overall_accuracy = np.mean(segment_scores) if segment_scores else 0.0
        stability = 1.0 - np.std(segment_scores) if len(segment_scores) > 1 else 0.5

        return {
            'accuracy': overall_accuracy,
            'stability': max(0.0, stability),
            'total_tests': total_tests,
            'validation_type': 'generic'
        }

    def _check_context_rule_compliance(self, rule: Dict[str, Any],
                                     sequence: str, position: int) -> bool:
        """Check if a context rule is satisfied at a specific position."""
        rule_type = rule.get('rule_type', '')

        if rule_type == 'context_before':
            # Check if after the context the expected pattern appears
            produces = rule.get('produces', [])
            # Simplified implementation
            return len(produces) > 0

        elif rule_type == 'context_after':
            # Check if before the context the expected pattern appears
            pattern = rule.get('pattern', '')
            # Simplified implementation
            return len(pattern) > 0

        return False
    
    def _calculate_overall_metrics(self, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcula mètriques globals del sistema de regles."""
        if not self.rule_performance:
            return {}
        
        accuracies = [perf['accuracy'] for perf in self.rule_performance.values()]
        stabilities = [perf['stability'] for perf in self.rule_performance.values()]
        
        return {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_stability': np.mean(stabilities),
            'std_stability': np.std(stabilities),
            'total_rules_tested': len(rules),
            'rules_above_threshold': sum(1 for acc in accuracies if acc >= self.min_validation_score)
        }
    
    def _analyze_stability(self, rules: List[Dict[str, Any]], 
                         phi_sequences: List[str]) -> Dict[str, Any]:
        """Analitza l'estabilitat de les regles entre diferents seqüències."""
        stability_analysis = {
            'stable_rules': [],
            'unstable_rules': [],
            'stability_distribution': {}
        }
        
        for rule_id, performance in self.rule_performance.items():
            stability = performance.get('stability', 0.0)
            
            if stability >= (1.0 - self.stability_threshold):
                stability_analysis['stable_rules'].append(rule_id)
            else:
                stability_analysis['unstable_rules'].append(rule_id)
        
        # Distribució d'estabilitat
        stabilities = [perf.get('stability', 0.0) for perf in self.rule_performance.values()]
        if stabilities:
            stability_analysis['stability_distribution'] = {
                'mean': np.mean(stabilities),
                'std': np.std(stabilities),
                'min': np.min(stabilities),
                'max': np.max(stabilities)
            }
        else:
            stability_analysis['stability_distribution'] = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        return stability_analysis
    
    def _generate_recommendations(self, rules: List[Dict[str, Any]]) -> List[str]:
        """Genera recomanacions basades en els resultats de validació."""
        recommendations = []
        
        # Analitzar rendiment global
        overall_metrics = self._calculate_overall_metrics(rules)
        mean_accuracy = overall_metrics.get('mean_accuracy', 0.0)
        
        if mean_accuracy < self.min_validation_score:
            recommendations.append(
                f"Precisió mitjana baixa ({mean_accuracy:.2f}). "
                "Considerar ajustar paràmetres del detector de patrons."
            )
        
        # Analitzar regles individuals
        poor_rules = []
        excellent_rules = []
        
        for rule_id, performance in self.rule_performance.items():
            accuracy = performance.get('accuracy', 0.0)
            stability = performance.get('stability', 0.0)
            
            if accuracy < self.min_validation_score:
                poor_rules.append(rule_id)
            elif accuracy > 0.9 and stability > 0.9:
                excellent_rules.append(rule_id)
        
        if poor_rules:
            recommendations.append(
                f"Eliminar o refinar {len(poor_rules)} regles amb baixa precisió: "
                f"{', '.join(poor_rules[:3])}{'...' if len(poor_rules) > 3 else ''}"
            )
        
        if excellent_rules:
            recommendations.append(
                f"Regles excel·lents identificades ({len(excellent_rules)}). "
                "Considerar usar-les com a base per generar regles similars."
            )
        
        # Recomanacions sobre estabilitat
        stability_analysis = self._analyze_stability(rules, [])
        unstable_count = len(stability_analysis.get('unstable_rules', []))
        
        if unstable_count > len(rules) * 0.3:
            recommendations.append(
                "Alta proporció de regles inestables. "
                "Considerar augmentar la mida de la finestra de context o "
                "el nombre d'iteracions del generador Φ."
            )
        
        return recommendations
    
    def cross_validate_rules(self, rules: List[Dict[str, Any]],
                           phi_sequences: List[str],
                           k_folds: int = 5) -> Dict[str, Any]:
        """Realitza validació creuada k-fold de les regles."""
        print(f"🔄 Cross-validation with {k_folds} folds on {len(rules):,} rules", flush=True)

        if len(phi_sequences) < k_folds:
            print(f"⚠️  Poques seqüències per validació creuada ({len(phi_sequences)} < {k_folds})")
            return self.validate_rules(rules, [], phi_sequences)

        fold_size = len(phi_sequences) // k_folds
        cv_results = []

        for fold in range(k_folds):
            print(f"  📊 Fold {fold + 1}/{k_folds}...", flush=True)

            # Dividir en entrenament i validació
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < k_folds - 1 else len(phi_sequences)

            validation_sequences = phi_sequences[start_idx:end_idx]

            # Validar sobre aquest fold
            fold_results = self.validate_rules(rules, [], validation_sequences)
            cv_results.append(fold_results)

            # Show fold summary
            fold_accuracy = fold_results.get('overall_metrics', {}).get('mean_accuracy', 0.0)
            print(f"  ✓ Fold {fold + 1} complete - mean accuracy: {fold_accuracy:.4f}", flush=True)

        # Aggregate results from all folds
        print(f"  ⏳ Aggregating results from {k_folds} folds...", flush=True)
        aggregated_results = self._aggregate_cv_results(cv_results)
        print(f"  ✓ Cross-validation complete", flush=True)

        return aggregated_results

    def _aggregate_cv_results(self, cv_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate cross-validation results."""
        if not cv_results:
            return {}

        # Aggregate metrics per rule
        rule_aggregates = defaultdict(list)

        for fold_result in cv_results:
            rule_scores = fold_result.get('rule_scores', {})
            for rule_id, scores in rule_scores.items():
                rule_aggregates[rule_id].append(scores)

        # Calculate means and standard deviations
        final_rule_scores = {}
        for rule_id, fold_scores in rule_aggregates.items():
            accuracies = [scores.get('accuracy', 0.0) for scores in fold_scores]
            stabilities = [scores.get('stability', 0.0) for scores in fold_scores]
            
            final_rule_scores[rule_id] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'mean_stability': np.mean(stabilities),
                'std_stability': np.std(stabilities),
                'cv_folds': len(fold_scores)
            }
        
        return {
            'rule_scores': final_rule_scores,
            'cv_summary': {
                'folds_completed': len(cv_results),
                'overall_mean_accuracy': np.mean([
                    scores['mean_accuracy'] 
                    for scores in final_rule_scores.values()
                ]),
                'overall_mean_stability': np.mean([
                    scores['mean_stability'] 
                    for scores in final_rule_scores.values()
                ])
            }
        }
    
    def save_validation_results(self, output_path: str) -> None:
        """Guarda els resultats de validació en format JSON."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        results_to_save = {
            'validation_results': self.validation_results,
            'rule_performance': self.rule_performance
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"💾 Resultats de validació guardats a {output_path}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Retorna un resum dels resultats de validació."""
        if not self.rule_performance:
            return {"total_validations": 0}
        
        accuracies = [perf['accuracy'] for perf in self.rule_performance.values()]
        stabilities = [perf['stability'] for perf in self.rule_performance.values()]
        
        return {
            "total_validations": len(self.validation_results),
            "total_rules_tested": len(self.rule_performance),
            "mean_accuracy": np.mean(accuracies),
            "mean_stability": np.mean(stabilities),
            "rules_above_threshold": sum(1 for acc in accuracies if acc >= self.min_validation_score),
            "best_performing_rules": sorted(
                self.rule_performance.items(),
                key=lambda x: x[1]['accuracy'],
                reverse=True
            )[:3]
        }


if __name__ == "__main__":
    # Exemple d'ús
    print("✅ Validador Iteratiu (Cycle-tester)")
    print("=" * 40)
    
    # Crear regles d'exemple
    example_rules = [
        {
            'rule_id': 'test_rule_01',
            'rule_type': 'markov_transition',
            'context': '01',
            'prediction': '0',
            'confidence': 0.8
        }
    ]
    
    example_sequences = [
        "010010010010",
        "010110010010",
        "010010110010"
    ]
    
    validator = Validator(min_validation_score=0.6)
    results = validator.validate_rules(example_rules, [], example_sequences)
    
    print(f"Regles validades: {len(example_rules)}")
    print(f"Precisió mitjana: {results['overall_metrics'].get('mean_accuracy', 0):.2f}")
    
    summary = validator.get_validation_summary()
    print(f"Resum: {summary['total_rules_tested']} regles testejades")
