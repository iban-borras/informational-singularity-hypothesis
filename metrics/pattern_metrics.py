"""
📊 Mètriques per als Patrons Pₖ

Aquest mòdul implementa les mètriques definides a validation_metrics_agents.md
per avaluar la qualitat i rellevància dels patrons detectats.

Autor: Iban Borràs amb col·laboració d'Augment Agent
Data: Gener 2025
"""

import numpy as np
import gzip
from typing import Dict, Any, List, Optional
from scipy import signal
from collections import Counter


class PatternMetrics:
    """
    Classe per calcular mètriques de qualitat dels patrons Pₖ.
    
    Implementa les mètriques definides a l'especificació:
    - Recurrència
    - Compressibilitat local
    - Autosimilitud
    - Signatura espectral
    """
    
    @staticmethod
    def calculate_recurrence(pattern_data: str, 
                           positions: List[int], 
                           sequence_length: int) -> Dict[str, Any]:
        """
        Calcula la mètrica de recurrència d'un patró.
        
        Args:
            pattern_data: Dades del patró com a string binari
            positions: Llista de posicions on apareix el patró
            sequence_length: Longitud total de la seqüència
            
        Returns:
            Diccionari amb mètriques de recurrència
        """
        occurrences = len(positions)
        pattern_length = len(pattern_data)
        
        # Densitat: proporció de la seqüència coberta pel patró
        total_coverage = occurrences * pattern_length
        density = total_coverage / sequence_length if sequence_length > 0 else 0.0
        
        # Distribució de distàncies entre ocurrències
        distances = []
        if len(positions) > 1:
            for i in range(len(positions) - 1):
                distance = positions[i + 1] - positions[i]
                distances.append(distance)
        
        return {
            "occurrences": occurrences,
            "positions": positions,
            "density": density,
            "distances": distances,
            "mean_distance": np.mean(distances) if distances else 0.0,
            "std_distance": np.std(distances) if distances else 0.0
        }
    
    @staticmethod
    def calculate_compressibility(pattern_data: str, 
                                context_window: int = 10) -> Dict[str, Any]:
        """
        Calcula la compressibilitat local d'un patró.
        
        Args:
            pattern_data: Dades del patró com a string binari
            context_window: Mida de la finestra de context
            
        Returns:
            Diccionari amb mètriques de compressibilitat
        """
        if not pattern_data:
            return {"compressibility": 0.0, "compression_ratio": 1.0}
        
        # Compressibilitat del patró sol
        original_size = len(pattern_data)
        compressed_size = len(gzip.compress(pattern_data.encode()))
        compression_ratio = compressed_size / original_size
        compressibility = 1.0 - compression_ratio
        
        # Compressibilitat amb context (si es proporciona)
        context_compressibility = compressibility  # Per defecte, igual que el patró sol
        
        # Entropia de Shannon com a mesura alternativa
        shannon_entropy = PatternMetrics._calculate_shannon_entropy(pattern_data)
        
        return {
            "compressibility": max(0.0, compressibility),
            "compression_ratio": compression_ratio,
            "context_compressibility": context_compressibility,
            "shannon_entropy": shannon_entropy,
            "original_size": original_size,
            "compressed_size": compressed_size
        }
    
    @staticmethod
    def calculate_autosimilarity(pattern_data: str) -> Dict[str, Any]:
        """
        Calcula l'autosimilitud d'un patró.
        
        Args:
            pattern_data: Dades del patró com a string binari
            
        Returns:
            Diccionari amb mètriques d'autosimilitud
        """
        if len(pattern_data) < 2:
            return {"autosimilarity": 0.0, "autocorrelation": []}
        
        # Convertir a array numèric
        pattern_array = np.array([int(bit) for bit in pattern_data], dtype=float)
        
        # Calcular autocorrelació
        autocorr = np.correlate(pattern_array, pattern_array, mode='full')
        autocorr = autocorr[autocorr.size // 2:]  # Només la part positiva
        
        # Normalitzar
        if autocorr[0] != 0:
            autocorr = autocorr / autocorr[0]
        
        # Mesures d'autosimilitud
        if len(autocorr) > 1:
            # Mitjana de l'autocorrelació (excloent el pic central)
            mean_autocorr = np.mean(np.abs(autocorr[1:]))
            
            # Màxim secundari (indica periodicitat interna)
            max_secondary = np.max(autocorr[1:]) if len(autocorr) > 1 else 0.0
            
            # Posició del màxim secundari
            max_secondary_pos = np.argmax(autocorr[1:]) + 1 if len(autocorr) > 1 else 0
        else:
            mean_autocorr = 0.0
            max_secondary = 0.0
            max_secondary_pos = 0
        
        # Autosimilitud fractal (si el patró és prou llarg)
        fractal_similarity = 0.0
        if len(pattern_data) >= 8:
            fractal_similarity = PatternMetrics._calculate_fractal_similarity(pattern_data)
        
        return {
            "autosimilarity": mean_autocorr,
            "autocorrelation": autocorr.tolist(),
            "max_secondary_peak": max_secondary,
            "max_secondary_position": max_secondary_pos,
            "fractal_similarity": fractal_similarity
        }
    
    @staticmethod
    def calculate_spectral_signature(pattern_data: str) -> Dict[str, Any]:
        """
        Calcula la signatura espectral d'un patró.
        
        Args:
            pattern_data: Dades del patró com a string binari
            
        Returns:
            Diccionari amb la signatura espectral
        """
        if len(pattern_data) < 4:
            return {"spectral_signature": [], "dominant_frequencies": []}
        
        # Convertir a array numèric
        pattern_array = np.array([int(bit) for bit in pattern_data], dtype=float)
        
        # Calcular FFT
        fft_result = np.fft.fft(pattern_array)
        power_spectrum = np.abs(fft_result) ** 2
        
        # Normalitzar
        if np.sum(power_spectrum) > 0:
            power_spectrum = power_spectrum / np.sum(power_spectrum)
        
        # Només freqüències positives
        half_spectrum = power_spectrum[:len(power_spectrum) // 2]
        
        # Trobar freqüències dominants
        peaks, properties = signal.find_peaks(half_spectrum, 
                                            height=np.mean(half_spectrum) * 1.5)
        
        # Calcular freqüències corresponents
        frequencies = np.fft.fftfreq(len(pattern_array))[:len(half_spectrum)]
        dominant_frequencies = frequencies[peaks].tolist()
        
        # Mètriques espectrals addicionals
        spectral_centroid = np.sum(frequencies * half_spectrum) / np.sum(half_spectrum) if np.sum(half_spectrum) > 0 else 0.0
        spectral_spread = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * half_spectrum) / np.sum(half_spectrum)) if np.sum(half_spectrum) > 0 else 0.0
        
        return {
            "spectral_signature": half_spectrum.tolist(),
            "dominant_frequencies": dominant_frequencies,
            "spectral_centroid": spectral_centroid,
            "spectral_spread": spectral_spread,
            "num_peaks": len(peaks),
            "frequencies": frequencies.tolist()
        }
    
    @staticmethod
    def _calculate_shannon_entropy(pattern_data: str) -> float:
        """Calcula l'entropia de Shannon d'un patró."""
        if not pattern_data:
            return 0.0
        
        # Comptar freqüències de cada símbol
        symbol_counts = Counter(pattern_data)
        total_symbols = len(pattern_data)
        
        # Calcular entropia
        entropy = 0.0
        for count in symbol_counts.values():
            probability = count / total_symbols
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    @staticmethod
    def _calculate_fractal_similarity(pattern_data: str) -> float:
        """Calcula la similaritat fractal d'un patró."""
        # Implementació simplificada de dimensió fractal
        pattern_array = np.array([int(bit) for bit in pattern_data])
        
        # Calcular variació a diferents escales
        scales = []
        variations = []
        
        for scale in range(1, min(len(pattern_array) // 2, 8)):
            # Dividir en segments de la escala donada
            segments = []
            for i in range(0, len(pattern_array) - scale + 1, scale):
                segment = pattern_array[i:i + scale]
                if len(segment) == scale:
                    segments.append(segment)
            
            if len(segments) > 1:
                # Calcular variació entre segments
                variation = 0.0
                for i in range(len(segments) - 1):
                    diff = np.sum(np.abs(segments[i] - segments[i + 1]))
                    variation += diff
                
                scales.append(scale)
                variations.append(variation / (len(segments) - 1))
        
        # Calcular pendent (indicador de fractalitat)
        if len(scales) > 1:
            log_scales = np.log(scales)
            log_variations = np.log(np.array(variations) + 1e-10)  # Evitar log(0)
            
            # Regressió lineal
            coefficients = np.polyfit(log_scales, log_variations, 1)
            fractal_dimension = -coefficients[0]  # Pendent negatiu
            
            return max(0.0, min(2.0, fractal_dimension))
        
        return 0.0
    
    @staticmethod
    def calculate_phi_alignment(pattern_data: str) -> Dict[str, Any]:
        """
        Calcula la mètrica φ-alignment (tendència daurada) d'un patró.

        Args:
            pattern_data: Dades del patró com a string binari

        Returns:
            Diccionari amb mètriques de φ-alignment
        """
        if len(pattern_data) < 4:
            return {"phi_alignment": 0.0, "phi_deviation": 1.0, "golden_ratio_proximity": 0.0}

        # Calcular dimensió fractal del patró
        fractal_dim = PatternMetrics._calculate_fractal_similarity(pattern_data)

        # Calcular proximitat a φ (1.618)
        phi = 1.618033988749
        phi_deviation = abs(fractal_dim - phi)
        phi_alignment = max(0.0, 1.0 - (phi_deviation / phi))  # Normalitzar a [0,1]

        # Analitzar proporcions internes del patró
        pattern_length = len(pattern_data)
        golden_ratio_proximity = 0.0

        if pattern_length >= 3:
            # Buscar segments que segueixin la proporció àuria
            for i in range(1, pattern_length):
                segment1_length = i
                segment2_length = pattern_length - i

                if segment2_length > 0:
                    ratio = max(segment1_length, segment2_length) / min(segment1_length, segment2_length)
                    ratio_deviation = abs(ratio - phi)
                    segment_proximity = max(0.0, 1.0 - (ratio_deviation / phi))
                    golden_ratio_proximity = max(golden_ratio_proximity, segment_proximity)

        # Analitzar freqüències espectrals relacionades amb φ
        spectral_phi_alignment = 0.0
        if len(pattern_data) >= 8:
            pattern_array = np.array([int(bit) for bit in pattern_data], dtype=float)
            fft_result = np.fft.fft(pattern_array)
            power_spectrum = np.abs(fft_result) ** 2

            # Buscar pics a freqüències relacionades amb φ
            frequencies = np.fft.fftfreq(len(pattern_array))
            phi_related_freqs = [1/phi, 1/(phi**2), phi/len(pattern_array)]

            for target_freq in phi_related_freqs:
                if target_freq < 0.5:  # Dins del rang de Nyquist
                    closest_idx = np.argmin(np.abs(frequencies - target_freq))
                    if closest_idx < len(power_spectrum):
                        normalized_power = power_spectrum[closest_idx] / np.max(power_spectrum)
                        spectral_phi_alignment = max(spectral_phi_alignment, normalized_power)

        return {
            "phi_alignment": phi_alignment,
            "phi_deviation": phi_deviation,
            "fractal_dimension": fractal_dim,
            "golden_ratio_proximity": golden_ratio_proximity,
            "spectral_phi_alignment": spectral_phi_alignment,
            "phi_coherence": (phi_alignment + golden_ratio_proximity + spectral_phi_alignment) / 3.0
        }

    @staticmethod
    def calculate_all_metrics(pattern_data: str,
                            positions: List[int],
                            sequence_length: int) -> Dict[str, Any]:
        """
        Calcula totes les mètriques per un patró.

        Args:
            pattern_data: Dades del patró com a string binari
            positions: Llista de posicions on apareix el patró
            sequence_length: Longitud total de la seqüència

        Returns:
            Diccionari amb totes les mètriques calculades
        """
        metrics = {}

        # Recurrència
        metrics.update(PatternMetrics.calculate_recurrence(
            pattern_data, positions, sequence_length
        ))

        # Compressibilitat
        compressibility_metrics = PatternMetrics.calculate_compressibility(pattern_data)
        metrics["compressibility"] = compressibility_metrics["compressibility"]
        metrics["compression_details"] = compressibility_metrics

        # Autosimilitud
        autosimilarity_metrics = PatternMetrics.calculate_autosimilarity(pattern_data)
        metrics["autosimilarity"] = autosimilarity_metrics["autosimilarity"]
        metrics["autosimilarity_details"] = autosimilarity_metrics

        # Signatura espectral
        spectral_metrics = PatternMetrics.calculate_spectral_signature(pattern_data)
        metrics["spectral_signature"] = spectral_metrics["spectral_signature"]
        metrics["spectral_details"] = spectral_metrics

        # φ-alignment (Tendència Daurada) - Mètrica específica HSI
        phi_metrics = PatternMetrics.calculate_phi_alignment(pattern_data)
        metrics["phi_alignment"] = phi_metrics["phi_alignment"]
        metrics["phi_details"] = phi_metrics

        return metrics


def create_pattern_report(pattern: Dict[str, Any], 
                        sequence_length: int) -> Dict[str, Any]:
    """
    Crea un informe complet de mètriques per un patró.
    
    Args:
        pattern: Diccionari amb informació del patró
        sequence_length: Longitud de la seqüència original
        
    Returns:
        Informe complet amb totes les mètriques
    """
    pattern_data = pattern.get('pattern_data', '')
    positions = pattern.get('positions', [])
    
    # Calcular totes les mètriques
    metrics = PatternMetrics.calculate_all_metrics(
        pattern_data, positions, sequence_length
    )
    
    # Crear informe estructurat
    report = {
        "pattern_id": pattern.get('pattern_id', 'unknown'),
        "pattern_data": pattern_data,
        "basic_info": {
            "length": len(pattern_data),
            "occurrences": len(positions),
            "method": pattern.get('method', 'unknown')
        },
        "metrics": metrics,
        "quality_score": _calculate_quality_score(metrics),
        "recommendations": _generate_pattern_recommendations(metrics)
    }
    
    return report


def _calculate_quality_score(metrics: Dict[str, Any]) -> float:
    """Calcula una puntuació de qualitat global per un patró incloent φ-alignment."""
    # Ponderar diferents mètriques (incloent φ-alignment com a mètrica clau HSI)
    weights = {
        'density': 0.15,
        'compressibility': 0.2,
        'autosimilarity': 0.15,
        'spectral_complexity': 0.2,
        'phi_alignment': 0.3  # Mètrica més important per HSI
    }

    density = metrics.get('density', 0.0)
    compressibility = metrics.get('compressibility', 0.0)
    autosimilarity = metrics.get('autosimilarity', 0.0)
    phi_alignment = metrics.get('phi_alignment', 0.0)

    # Complexitat espectral (nombre de pics normalitzat)
    spectral_details = metrics.get('spectral_details', {})
    num_peaks = spectral_details.get('num_peaks', 0)
    spectral_complexity = min(1.0, num_peaks / 5.0)  # Normalitzar a [0,1]

    # Calcular puntuació ponderada
    quality_score = (
        weights['density'] * density +
        weights['compressibility'] * compressibility +
        weights['autosimilarity'] * autosimilarity +
        weights['spectral_complexity'] * spectral_complexity +
        weights['phi_alignment'] * phi_alignment
    )

    return min(1.0, max(0.0, quality_score))


def _generate_pattern_recommendations(metrics: Dict[str, Any]) -> List[str]:
    """Genera recomanacions basades en les mètriques del patró incloent φ-alignment."""
    recommendations = []

    density = metrics.get('density', 0.0)
    compressibility = metrics.get('compressibility', 0.0)
    autosimilarity = metrics.get('autosimilarity', 0.0)
    phi_alignment = metrics.get('phi_alignment', 0.0)

    # Recomanacions específiques HSI per φ-alignment
    if phi_alignment > 0.8:
        recommendations.append("🌟 Excellent φ-alignment: patró altament coherent amb la HSI, candidat prioritari per regla emergent")
    elif phi_alignment > 0.6:
        recommendations.append("✨ Bon φ-alignment: patró amb tendència daurada, rellevant per la hipòtesi")
    elif phi_alignment > 0.3:
        recommendations.append("⚡ φ-alignment moderat: patró amb alguna coherència amb proporcions àuries")
    else:
        recommendations.append("⚠️ Baix φ-alignment: patró poc coherent amb la tendència daurada de la HSI")

    # Recomanacions tradicionals
    if density > 0.1:
        recommendations.append("Alta densitat: patró molt recurrent, candidat per regla emergent")
    elif density < 0.01:
        recommendations.append("Baixa densitat: considerar si és soroll o patró genuí")

    if compressibility > 0.7:
        recommendations.append("Alta compressibilitat: estructura interna regular")
    elif compressibility < 0.2:
        recommendations.append("Baixa compressibilitat: possible soroll aleatori")

    if autosimilarity > 0.5:
        recommendations.append("Alta autosimilitud: patró amb estructura fractal o periòdica")

    spectral_details = metrics.get('spectral_details', {})
    num_peaks = spectral_details.get('num_peaks', 0)

    if num_peaks > 3:
        recommendations.append("Múltiples pics espectrals: patró complex amb diverses periodicitats")
    elif num_peaks == 0:
        recommendations.append("Sense pics espectrals: patró aperiòdic o soroll")

    # Recomanació combinada HSI
    phi_details = metrics.get('phi_details', {})
    phi_coherence = phi_details.get('phi_coherence', 0.0)

    if phi_coherence > 0.7:
        recommendations.append("🎯 Alta coherència φ global: patró excel·lent per validar la HSI")

    return recommendations


if __name__ == "__main__":
    # Exemple d'ús
    print("📊 Mètriques per als Patrons Pₖ")
    print("=" * 40)
    
    # Patró d'exemple
    example_pattern = {
        'pattern_id': 'test_pattern',
        'pattern_data': '101010',
        'positions': [0, 10, 20, 30],
        'method': 'sliding_window'
    }
    
    sequence_length = 100
    
    # Generar informe
    report = create_pattern_report(example_pattern, sequence_length)
    
    print(f"Patró: {report['pattern_data']}")
    print(f"Qualitat: {report['quality_score']:.3f}")
    print(f"Densitat: {report['metrics']['density']:.3f}")
    print(f"Compressibilitat: {report['metrics']['compressibility']:.3f}")
    print(f"Autosimilitud: {report['metrics']['autosimilarity']:.3f}")
    print(f"φ-alignment: {report['metrics']['phi_alignment']:.3f}")

    phi_details = report['metrics'].get('phi_details', {})
    if phi_details:
        print(f"Coherència φ: {phi_details.get('phi_coherence', 0):.3f}")
        print(f"Proximitat àuria: {phi_details.get('golden_ratio_proximity', 0):.3f}")

    print("\nRecomanacions:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
