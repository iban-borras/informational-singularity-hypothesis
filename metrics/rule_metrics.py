"""
📊 Mètriques per a les Regles ωₖ

Aquest mòdul implementa les mètriques definides a validation_metrics_agents.md
per avaluar la qualitat i robustesa de les regles emergents inferides.

Autor: Iban Borràs amb col·laboració d'Augment Agent
Data: Gener 2025
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter, defaultdict
import json


class RuleMetrics:
    """
    Classe per calcular mètriques de qualitat de les regles ωₖ.
    
    Implementa les mètriques definides a l'especificació:
    - Reproductibilitat
    - Estabilitat
    - Precisió
    - Complexitat de la regla
    """
    
    @staticmethod
    def calculate_reproducibility(rule: Dict[str, Any], 
                                test_sequences: List[str],
                                num_trials: int = 10) -> Dict[str, Any]:
        """
        Calcula la reproductibilitat d'una regla.
        
        Args:
            rule: Diccionari amb informació de la regla
            test_sequences: Seqüències per provar la regla
            num_trials: Nombre d'assajos per calcular reproductibilitat
            
        Returns:
            Diccionari amb mètriques de reproductibilitat
        """
        rule_type = rule.get('rule_type', 'unknown')
        successful_reproductions = 0
        total_attempts = 0
        reproduction_scores = []
        
        for sequence in test_sequences:
            for trial in range(num_trials):
                # Aplicar la regla a la seqüència
                success = RuleMetrics._apply_rule_to_sequence(rule, sequence)
                
                if success:
                    successful_reproductions += 1
                total_attempts += 1
                
                reproduction_scores.append(1.0 if success else 0.0)
        
        # Calcular mètriques
        reproducibility_rate = successful_reproductions / total_attempts if total_attempts > 0 else 0.0
        reproducibility_std = np.std(reproduction_scores)
        
        return {
            "reproducibility": reproducibility_rate,
            "successful_reproductions": successful_reproductions,
            "total_attempts": total_attempts,
            "reproducibility_std": reproducibility_std,
            "consistency": 1.0 - reproducibility_std  # Mesura de consistència
        }
    
    @staticmethod
    def calculate_stability(rule: Dict[str, Any], 
                          sequences_over_time: List[str]) -> Dict[str, Any]:
        """
        Calcula l'estabilitat d'una regla en iteracions successives.
        
        Args:
            rule: Diccionari amb informació de la regla
            sequences_over_time: Seqüències de diferents iteracions temporals
            
        Returns:
            Diccionari amb mètriques d'estabilitat
        """
        if len(sequences_over_time) < 2:
            return {"stability": 0.0, "temporal_consistency": 0.0}
        
        # Aplicar la regla a cada seqüència temporal
        temporal_scores = []
        
        for i, sequence in enumerate(sequences_over_time):
            score = RuleMetrics._evaluate_rule_performance(rule, sequence)
            temporal_scores.append(score)
        
        # Calcular estabilitat com a consistència temporal
        mean_score = np.mean(temporal_scores)
        std_score = np.std(temporal_scores)
        
        # Estabilitat: alta si la desviació estàndard és baixa
        stability = max(0.0, 1.0 - (std_score / (mean_score + 1e-10)))
        
        # Tendència temporal (si la regla millora o empitjora amb el temps)
        if len(temporal_scores) > 2:
            trend_slope = np.polyfit(range(len(temporal_scores)), temporal_scores, 1)[0]
        else:
            trend_slope = 0.0
        
        return {
            "stability": stability,
            "temporal_consistency": mean_score,
            "temporal_std": std_score,
            "temporal_scores": temporal_scores,
            "trend_slope": trend_slope,
            "improving_trend": trend_slope > 0.01,
            "degrading_trend": trend_slope < -0.01
        }
    
    @staticmethod
    def calculate_precision(rule: Dict[str, Any], 
                          ground_truth_patterns: List[Dict[str, Any]],
                          test_sequence: str) -> Dict[str, Any]:
        """
        Calcula la precisió d'una regla comparant amb patrons coneguts.
        
        Args:
            rule: Diccionari amb informació de la regla
            ground_truth_patterns: Patrons coneguts per comparar
            test_sequence: Seqüència de test
            
        Returns:
            Diccionari amb mètriques de precisió
        """
        # Aplicar la regla per generar prediccions
        predictions = RuleMetrics._generate_rule_predictions(rule, test_sequence)
        
        # Comparar amb patrons coneguts
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Crear conjunt de posicions conegudes
        known_positions = set()
        for pattern in ground_truth_patterns:
            positions = pattern.get('positions', [])
            known_positions.update(positions)
        
        # Evaluar prediccions
        predicted_positions = set(predictions.get('positions', []))
        
        true_positives = len(predicted_positions.intersection(known_positions))
        false_positives = len(predicted_positions - known_positions)
        false_negatives = len(known_positions - predicted_positions)
        
        # Calcular mètriques de precisió
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "accuracy": true_positives / len(known_positions) if known_positions else 0.0
        }
    
    @staticmethod
    def calculate_complexity(rule: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula la complexitat d'una regla.
        
        Args:
            rule: Diccionari amb informació de la regla
            
        Returns:
            Diccionari amb mètriques de complexitat
        """
        rule_type = rule.get('rule_type', 'unknown')
        rule_description = rule.get('rule_description', '')
        
        # Complexitat basada en el tipus de regla
        type_complexity = {
            'markov_transition': 1.0,
            'context_before': 2.0,
            'context_after': 2.0,
            'composition': 3.0,
            'overlap_composition': 4.0,
            'periodicity': 1.5,
            'unknown': 2.5
        }
        
        base_complexity = type_complexity.get(rule_type, 2.5)
        
        # Complexitat basada en la descripció
        description_complexity = len(rule_description) / 50.0  # Normalitzar
        
        # Complexitat basada en paràmetres específics
        param_complexity = 0.0
        
        # Per regles de Markov: complexitat basada en l'ordre
        if 'order' in rule:
            param_complexity += rule['order'] * 0.5
        
        # Per regles de composició: complexitat basada en el nombre de components
        if 'components' in rule:
            param_complexity += len(rule['components']) * 0.3
        
        # Per regles de context: complexitat basada en la mida del context
        if 'context' in rule:
            param_complexity += len(rule['context']) * 0.1
        
        # Complexitat total
        total_complexity = base_complexity + description_complexity + param_complexity
        
        # Normalitzar a escala 0-10
        normalized_complexity = min(10.0, max(0.0, total_complexity))
        
        # Classificació de complexitat
        if normalized_complexity < 2.0:
            complexity_class = "simple"
        elif normalized_complexity < 5.0:
            complexity_class = "moderate"
        elif normalized_complexity < 8.0:
            complexity_class = "complex"
        else:
            complexity_class = "very_complex"
        
        return {
            "complexity": normalized_complexity,
            "complexity_class": complexity_class,
            "base_complexity": base_complexity,
            "description_complexity": description_complexity,
            "parameter_complexity": param_complexity,
            "interpretability": max(0.0, 1.0 - (normalized_complexity / 10.0))
        }
    
    @staticmethod
    def _apply_rule_to_sequence(rule: Dict[str, Any], sequence: str) -> bool:
        """Aplica una regla a una seqüència i retorna si té èxit."""
        rule_type = rule.get('rule_type', 'unknown')
        
        if rule_type == 'markov_transition':
            context = rule.get('context', '')
            prediction = rule.get('prediction', '')
            
            # Buscar ocurrències del context
            for i in range(len(sequence) - len(context)):
                if sequence[i:i + len(context)] == context:
                    if i + len(context) < len(sequence):
                        actual_next = sequence[i + len(context)]
                        if actual_next == prediction:
                            return True
            return False
        
        elif rule_type == 'periodicity':
            pattern = rule.get('pattern', '')
            period = rule.get('period', 0)
            
            if not pattern or period <= 0:
                return False
            
            # Buscar ocurrències del patró amb la periodicitat esperada
            pattern_positions = []
            for i in range(len(sequence) - len(pattern) + 1):
                if sequence[i:i + len(pattern)] == pattern:
                    pattern_positions.append(i)
            
            # Verificar periodicitat
            for i in range(len(pattern_positions) - 1):
                distance = pattern_positions[i + 1] - pattern_positions[i]
                if distance == period:
                    return True
            
            return False
        
        # Per altres tipus de regles, retornar True per defecte
        return True
    
    @staticmethod
    def _evaluate_rule_performance(rule: Dict[str, Any], sequence: str) -> float:
        """Evalua el rendiment d'una regla en una seqüència."""
        # Implementació simplificada que retorna la confiança de la regla
        confidence = rule.get('confidence', 0.5)
        
        # Ajustar basant-se en l'aplicabilitat a la seqüència
        if RuleMetrics._apply_rule_to_sequence(rule, sequence):
            return confidence
        else:
            return confidence * 0.5  # Penalitzar si la regla no s'aplica
    
    @staticmethod
    def _generate_rule_predictions(rule: Dict[str, Any], sequence: str) -> Dict[str, Any]:
        """Genera prediccions aplicant una regla a una seqüència."""
        rule_type = rule.get('rule_type', 'unknown')
        predictions = {'positions': [], 'patterns': []}
        
        if rule_type == 'markov_transition':
            context = rule.get('context', '')
            prediction = rule.get('prediction', '')
            
            # Buscar posicions on s'aplica la regla
            for i in range(len(sequence) - len(context)):
                if sequence[i:i + len(context)] == context:
                    predictions['positions'].append(i + len(context))
                    predictions['patterns'].append(prediction)
        
        elif rule_type == 'periodicity':
            pattern = rule.get('pattern', '')
            period = rule.get('period', 0)
            
            # Buscar primera ocurrència i predir les següents
            for i in range(len(sequence) - len(pattern) + 1):
                if sequence[i:i + len(pattern)] == pattern:
                    # Predir ocurrències periòdiques
                    next_pos = i + period
                    while next_pos + len(pattern) <= len(sequence):
                        predictions['positions'].append(next_pos)
                        predictions['patterns'].append(pattern)
                        next_pos += period
                    break
        
        return predictions
    
    @staticmethod
    def calculate_phi_alignment_rule(rule: Dict[str, Any],
                                   test_sequences: List[str]) -> Dict[str, Any]:
        """
        Calcula la mètrica φ-alignment per una regla aplicada a seqüències.

        Args:
            rule: Diccionari amb informació de la regla
            test_sequences: Seqüències per avaluar l'alignment

        Returns:
            Diccionari amb mètriques de φ-alignment per la regla
        """
        if not test_sequences:
            return {"rule_phi_alignment": 0.0, "phi_consistency": 0.0}

        phi = 1.618033988749
        phi_alignments = []

        for sequence in test_sequences:
            # Aplicar la regla per generar prediccions
            predictions = RuleMetrics._generate_rule_predictions(rule, sequence)
            predicted_positions = predictions.get('positions', [])

            if len(predicted_positions) >= 2:
                # Analitzar distàncies entre prediccions
                distances = []
                for i in range(len(predicted_positions) - 1):
                    distance = predicted_positions[i + 1] - predicted_positions[i]
                    distances.append(distance)

                if distances:
                    # Calcular si les distàncies segueixen proporcions àuries
                    phi_deviations = []
                    for i in range(len(distances) - 1):
                        if distances[i] > 0:
                            ratio = distances[i + 1] / distances[i]
                            phi_deviation = abs(ratio - phi) / phi
                            phi_deviations.append(phi_deviation)

                    if phi_deviations:
                        avg_deviation = np.mean(phi_deviations)
                        sequence_phi_alignment = max(0.0, 1.0 - avg_deviation)
                        phi_alignments.append(sequence_phi_alignment)

        # Calcular mètriques globals
        if phi_alignments:
            rule_phi_alignment = np.mean(phi_alignments)
            phi_consistency = 1.0 - np.std(phi_alignments)  # Consistència entre seqüències
        else:
            rule_phi_alignment = 0.0
            phi_consistency = 0.0

        return {
            "rule_phi_alignment": rule_phi_alignment,
            "phi_consistency": max(0.0, phi_consistency),
            "sequence_alignments": phi_alignments,
            "phi_coherence_rule": (rule_phi_alignment + phi_consistency) / 2.0
        }

    @staticmethod
    def calculate_all_metrics(rule: Dict[str, Any],
                            test_sequences: List[str],
                            ground_truth_patterns: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calcula totes les mètriques per una regla.

        Args:
            rule: Diccionari amb informació de la regla
            test_sequences: Seqüències per provar la regla
            ground_truth_patterns: Patrons coneguts per comparar (opcional)

        Returns:
            Diccionari amb totes les mètriques calculades
        """
        metrics = {}

        # Reproductibilitat
        if test_sequences:
            reproducibility_metrics = RuleMetrics.calculate_reproducibility(
                rule, test_sequences
            )
            metrics.update(reproducibility_metrics)

        # Estabilitat
        if len(test_sequences) > 1:
            stability_metrics = RuleMetrics.calculate_stability(
                rule, test_sequences
            )
            metrics["stability"] = stability_metrics["stability"]
            metrics["stability_details"] = stability_metrics

        # Precisió (si hi ha ground truth)
        if ground_truth_patterns and test_sequences:
            precision_metrics = RuleMetrics.calculate_precision(
                rule, ground_truth_patterns, test_sequences[0]
            )
            metrics["precision"] = precision_metrics["precision"]
            metrics["precision_details"] = precision_metrics

        # Complexitat
        complexity_metrics = RuleMetrics.calculate_complexity(rule)
        metrics["complexity"] = complexity_metrics["complexity"]
        metrics["complexity_details"] = complexity_metrics

        # φ-alignment (Tendència Daurada) - Mètrica específica HSI per regles
        if test_sequences:
            phi_metrics = RuleMetrics.calculate_phi_alignment_rule(rule, test_sequences)
            metrics["phi_alignment"] = phi_metrics["rule_phi_alignment"]
            metrics["phi_details"] = phi_metrics

        return metrics


def create_rule_report(rule: Dict[str, Any], 
                      test_sequences: List[str],
                      ground_truth_patterns: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Crea un informe complet de mètriques per una regla.
    
    Args:
        rule: Diccionari amb informació de la regla
        test_sequences: Seqüències per provar la regla
        ground_truth_patterns: Patrons coneguts per comparar (opcional)
        
    Returns:
        Informe complet amb totes les mètriques
    """
    # Calcular totes les mètriques
    metrics = RuleMetrics.calculate_all_metrics(
        rule, test_sequences, ground_truth_patterns
    )
    
    # Crear informe estructurat
    report = {
        "rule_id": rule.get('rule_id', 'unknown'),
        "rule_type": rule.get('rule_type', 'unknown'),
        "rule_description": rule.get('rule_description', ''),
        "basic_info": {
            "confidence": rule.get('confidence', 0.0),
            "support": rule.get('support', 0),
            "method": rule.get('method', 'unknown')
        },
        "metrics": metrics,
        "quality_score": _calculate_rule_quality_score(metrics),
        "recommendations": _generate_rule_recommendations(rule, metrics)
    }
    
    return report


def _calculate_rule_quality_score(metrics: Dict[str, Any]) -> float:
    """Calcula una puntuació de qualitat global per una regla."""
    # Ponderar diferents mètriques
    weights = {
        'reproducibility': 0.3,
        'stability': 0.25,
        'precision': 0.25,
        'complexity_penalty': 0.2
    }
    
    reproducibility = metrics.get('reproducibility', 0.0)
    stability = metrics.get('stability', 0.0)
    precision = metrics.get('precision', 0.0)
    complexity = metrics.get('complexity', 5.0)
    
    # Penalització per complexitat (complexitat alta = puntuació baixa)
    complexity_penalty = max(0.0, 1.0 - (complexity / 10.0))
    
    # Calcular puntuació ponderada
    quality_score = (
        weights['reproducibility'] * reproducibility +
        weights['stability'] * stability +
        weights['precision'] * precision +
        weights['complexity_penalty'] * complexity_penalty
    )
    
    return min(1.0, max(0.0, quality_score))


def _generate_rule_recommendations(rule: Dict[str, Any], 
                                 metrics: Dict[str, Any]) -> List[str]:
    """Genera recomanacions basades en les mètriques de la regla."""
    recommendations = []
    
    reproducibility = metrics.get('reproducibility', 0.0)
    stability = metrics.get('stability', 0.0)
    precision = metrics.get('precision', 0.0)
    complexity = metrics.get('complexity', 5.0)
    
    # Recomanacions basades en reproductibilitat
    if reproducibility > 0.8:
        recommendations.append("Alta reproductibilitat: regla robusta i fiable")
    elif reproducibility < 0.5:
        recommendations.append("Baixa reproductibilitat: revisar condicions d'aplicació")
    
    # Recomanacions basades en estabilitat
    if stability > 0.8:
        recommendations.append("Alta estabilitat: regla consistent en el temps")
    elif stability < 0.5:
        recommendations.append("Baixa estabilitat: possible sobreajustament o regla espúria")
    
    # Recomanacions basades en precisió
    if precision > 0.8:
        recommendations.append("Alta precisió: regla predictiva excel·lent")
    elif precision < 0.5:
        recommendations.append("Baixa precisió: molts falsos positius, refinar regla")
    
    # Recomanacions basades en complexitat
    if complexity > 8.0:
        recommendations.append("Complexitat molt alta: considerar simplificar la regla")
    elif complexity < 2.0:
        recommendations.append("Regla simple: fàcil d'interpretar i aplicar")
    
    # Recomanacions específiques per tipus de regla
    rule_type = rule.get('rule_type', 'unknown')
    
    if rule_type == 'markov_transition' and reproducibility > 0.7:
        recommendations.append("Regla Markov fiable: candidata per model predictiu")
    
    if rule_type == 'periodicity' and stability > 0.8:
        recommendations.append("Periodicitat estable: possible llei emergent del sistema")
    
    return recommendations


if __name__ == "__main__":
    # Exemple d'ús
    print("📊 Mètriques per a les Regles ωₖ")
    print("=" * 40)
    
    # Regla d'exemple
    example_rule = {
        'rule_id': 'test_rule',
        'rule_type': 'markov_transition',
        'rule_description': 'IF context="01" THEN next="0"',
        'context': '01',
        'prediction': '0',
        'confidence': 0.8,
        'support': 10
    }
    
    test_sequences = [
        "010010010010",
        "010110010010",
        "010010110010"
    ]
    
    # Generar informe
    report = create_rule_report(example_rule, test_sequences)
    
    print(f"Regla: {report['rule_description']}")
    print(f"Qualitat: {report['quality_score']:.3f}")
    print(f"Reproductibilitat: {report['metrics'].get('reproducibility', 0):.3f}")
    print(f"Complexitat: {report['metrics'].get('complexity', 0):.3f}")
    
    print("\nRecomanacions:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
