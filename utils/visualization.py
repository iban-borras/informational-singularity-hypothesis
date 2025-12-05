"""
🎨 Utilitats de Visualització per la HSI

Aquest mòdul proporciona funcions per visualitzar seqüències Φ, patrons Pₖ
i regles ωₖ generades pel sistema d'agents.

Autor: Iban Borràs amb col·laboració d'Augment Agent
Data: Gener 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import seaborn as sns


def plot_phi_sequence(phi_sequence: str, 
                     title: str = "Seqüència Φ",
                     save_path: Optional[str] = None,
                     max_length: int = 1000) -> None:
    """
    Visualitza una seqüència Φ com a gràfic de barres binari.
    
    Args:
        phi_sequence: Seqüència binària com a string
        title: Títol del gràfic
        save_path: Camí per guardar la imatge (opcional)
        max_length: Longitud màxima a mostrar
    """
    # Limitar longitud per visualització
    sequence_to_plot = phi_sequence[:max_length]
    binary_array = np.array([int(bit) for bit in sequence_to_plot])
    
    plt.figure(figsize=(15, 4))
    plt.bar(range(len(binary_array)), binary_array, width=1.0, 
            color=['white' if bit == 0 else 'black' for bit in binary_array],
            edgecolor='gray', linewidth=0.1)
    
    plt.title(f"{title} (primers {len(sequence_to_plot)} bits)")
    plt.xlabel("Posició")
    plt.ylabel("Valor")
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gràfic guardat a {save_path}")
    
    plt.show()


def plot_phi_hilbert(phi_sequence: str,
                    grid_size: Optional[int] = None,
                    title: str = "Seqüència Φ - Corba de Hilbert",
                    save_path: Optional[str] = None) -> None:
    """
    Visualitza una seqüència Φ mapejada en una corba de Hilbert.
    
    Args:
        phi_sequence: Seqüència binària com a string
        grid_size: Mida de la graella (si None, s'estima automàticament)
        title: Títol del gràfic
        save_path: Camí per guardar la imatge (opcional)
    """
    binary_array = np.array([int(bit) for bit in phi_sequence])
    
    if grid_size is None:
        grid_size = int(np.ceil(np.sqrt(len(binary_array))))
        # Assegurar que sigui una potència de 2
        grid_size = 2 ** int(np.ceil(np.log2(grid_size)))
    
    # Crear graella
    grid = np.zeros((grid_size, grid_size))
    
    # Mapear seqüència a graella (implementació simplificada)
    for i, bit in enumerate(binary_array):
        if i >= grid_size * grid_size:
            break
        row = i // grid_size
        col = i % grid_size
        grid[row, col] = bit
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='binary', interpolation='nearest')
    plt.title(f"{title} ({grid_size}x{grid_size})")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Mapa de Hilbert guardat a {save_path}")
    
    plt.show()


def plot_pattern_distribution(patterns: List[Dict[str, Any]],
                            title: str = "Distribució de Patrons Pₖ",
                            save_path: Optional[str] = None) -> None:
    """
    Visualitza la distribució de patrons detectats.
    
    Args:
        patterns: Llista de patrons detectats
        title: Títol del gràfic
        save_path: Camí per guardar la imatge (opcional)
    """
    if not patterns:
        print("⚠️  No hi ha patrons per visualitzar")
        return
    
    # Extreure dades per visualització
    pattern_lengths = [len(p['pattern_data']) for p in patterns]
    recurrences = [p.get('recurrence', 0) for p in patterns]
    densities = [p.get('density', 0) for p in patterns]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Histograma de longituds de patrons
    axes[0, 0].hist(pattern_lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title("Distribució de Longituds de Patrons")
    axes[0, 0].set_xlabel("Longitud del Patró")
    axes[0, 0].set_ylabel("Freqüència")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histograma de recurrències
    axes[0, 1].hist(recurrences, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title("Distribució de Recurrències")
    axes[0, 1].set_xlabel("Nombre d'Ocurrències")
    axes[0, 1].set_ylabel("Freqüència")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter plot: longitud vs recurrència
    axes[1, 0].scatter(pattern_lengths, recurrences, alpha=0.6, color='coral')
    axes[1, 0].set_title("Longitud vs Recurrència")
    axes[1, 0].set_xlabel("Longitud del Patró")
    axes[1, 0].set_ylabel("Recurrència")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histograma de densitats
    if any(d > 0 for d in densities):
        axes[1, 1].hist(densities, bins=20, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].set_title("Distribució de Densitats")
        axes[1, 1].set_xlabel("Densitat")
        axes[1, 1].set_ylabel("Freqüència")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "Dades de densitat\nno disponibles", 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Densitats")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Distribució de patrons guardada a {save_path}")
    
    plt.show()


def plot_rule_performance(rules: List[Dict[str, Any]],
                         title: str = "Rendiment de Regles ωₖ",
                         save_path: Optional[str] = None) -> None:
    """
    Visualitza el rendiment de les regles inferides.
    
    Args:
        rules: Llista de regles amb mètriques
        title: Títol del gràfic
        save_path: Camí per guardar la imatge (opcional)
    """
    if not rules:
        print("⚠️  No hi ha regles per visualitzar")
        return
    
    # Extreure mètriques
    confidences = [r.get('confidence', 0) for r in rules]
    precisions = [r.get('precision', 0) for r in rules]
    stabilities = [r.get('stability', 0) for r in rules]
    complexities = [r.get('complexity', 0) for r in rules]
    
    # Tipus de regles
    rule_types = [r.get('rule_type', 'unknown') for r in rules]
    unique_types = list(set(rule_types))
    type_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Scatter plot: confiança vs precisió
    for i, rule_type in enumerate(unique_types):
        mask = [rt == rule_type for rt in rule_types]
        conf_subset = [confidences[j] for j, m in enumerate(mask) if m]
        prec_subset = [precisions[j] for j, m in enumerate(mask) if m]
        
        axes[0, 0].scatter(conf_subset, prec_subset, 
                          label=rule_type, alpha=0.7, color=type_colors[i])
    
    axes[0, 0].set_title("Confiança vs Precisió")
    axes[0, 0].set_xlabel("Confiança")
    axes[0, 0].set_ylabel("Precisió")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histograma d'estabilitat
    axes[0, 1].hist(stabilities, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 1].set_title("Distribució d'Estabilitat")
    axes[0, 1].set_xlabel("Estabilitat")
    axes[0, 1].set_ylabel("Freqüència")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot per tipus de regla
    type_data = []
    type_labels = []
    for rule_type in unique_types:
        type_confidences = [confidences[i] for i, rt in enumerate(rule_types) if rt == rule_type]
        if type_confidences:
            type_data.append(type_confidences)
            type_labels.append(rule_type)
    
    if type_data:
        axes[1, 0].boxplot(type_data, labels=type_labels)
        axes[1, 0].set_title("Confiança per Tipus de Regla")
        axes[1, 0].set_ylabel("Confiança")
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter plot: complexitat vs precisió
    axes[1, 1].scatter(complexities, precisions, alpha=0.6, color='orange')
    axes[1, 1].set_title("Complexitat vs Precisió")
    axes[1, 1].set_xlabel("Complexitat")
    axes[1, 1].set_ylabel("Precisió")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Rendiment de regles guardat a {save_path}")
    
    plt.show()


def plot_validation_results(validation_results: Dict[str, Any],
                          title: str = "Resultats de Validació",
                          save_path: Optional[str] = None) -> None:
    """
    Visualitza els resultats de validació del sistema.
    
    Args:
        validation_results: Resultats de validació del Validator
        title: Títol del gràfic
        save_path: Camí per guardar la imatge (opcional)
    """
    rule_scores = validation_results.get('rule_scores', {})
    overall_metrics = validation_results.get('overall_metrics', {})
    
    if not rule_scores:
        print("⚠️  No hi ha resultats de validació per visualitzar")
        return
    
    # Extreure dades
    rule_ids = list(rule_scores.keys())
    accuracies = [rule_scores[rid].get('accuracy', 0) for rid in rule_ids]
    stabilities = [rule_scores[rid].get('stability', 0) for rid in rule_ids]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Gràfic de barres d'accuracies
    axes[0, 0].bar(range(len(accuracies)), accuracies, alpha=0.7, color='lightgreen')
    axes[0, 0].set_title("Precisió per Regla")
    axes[0, 0].set_xlabel("Índex de Regla")
    axes[0, 0].set_ylabel("Precisió")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gràfic de barres d'estabilitats
    axes[0, 1].bar(range(len(stabilities)), stabilities, alpha=0.7, color='lightcoral')
    axes[0, 1].set_title("Estabilitat per Regla")
    axes[0, 1].set_xlabel("Índex de Regla")
    axes[0, 1].set_ylabel("Estabilitat")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter plot: precisió vs estabilitat
    axes[1, 0].scatter(accuracies, stabilities, alpha=0.6, color='purple')
    axes[1, 0].set_title("Precisió vs Estabilitat")
    axes[1, 0].set_xlabel("Precisió")
    axes[1, 0].set_ylabel("Estabilitat")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Mètriques globals
    if overall_metrics:
        metrics_names = list(overall_metrics.keys())
        metrics_values = list(overall_metrics.values())
        
        # Filtrar només valors numèrics
        numeric_metrics = [(name, val) for name, val in zip(metrics_names, metrics_values) 
                          if isinstance(val, (int, float))]
        
        if numeric_metrics:
            names, values = zip(*numeric_metrics)
            axes[1, 1].bar(range(len(values)), values, alpha=0.7, color='gold')
            axes[1, 1].set_title("Mètriques Globals")
            axes[1, 1].set_xticks(range(len(names)))
            axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
            axes[1, 1].set_ylabel("Valor")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, "Mètriques globals\nno disponibles", 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Resultats de validació guardats a {save_path}")
    
    plt.show()


def create_summary_dashboard(phi_sequence: str,
                           patterns: List[Dict[str, Any]],
                           rules: List[Dict[str, Any]],
                           validation_results: Dict[str, Any],
                           save_path: Optional[str] = None) -> None:
    """
    Crea un dashboard resum amb totes les visualitzacions principals.
    
    Args:
        phi_sequence: Seqüència Φ generada
        patterns: Patrons detectats
        rules: Regles inferides
        validation_results: Resultats de validació
        save_path: Camí per guardar la imatge (opcional)
    """
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("Dashboard HSI - Hipòtesi de Singularitat Informacional", fontsize=20)
    
    # Crear grid de subplots
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Seqüència Φ (primera fila, columnes 0-1)
    ax1 = fig.add_subplot(gs[0, :2])
    binary_array = np.array([int(bit) for bit in phi_sequence[:500]])  # Primers 500 bits
    ax1.bar(range(len(binary_array)), binary_array, width=1.0, 
           color=['white' if bit == 0 else 'black' for bit in binary_array],
           edgecolor='gray', linewidth=0.1)
    ax1.set_title("Seqüència Φ (primers 500 bits)")
    ax1.set_xlabel("Posició")
    ax1.set_ylabel("Valor")
    
    # 2. Mapa de Hilbert (primera fila, columnes 2-3)
    ax2 = fig.add_subplot(gs[0, 2:])
    grid_size = min(32, int(np.ceil(np.sqrt(len(phi_sequence)))))
    grid = np.zeros((grid_size, grid_size))
    for i, bit in enumerate(phi_sequence[:grid_size*grid_size]):
        row, col = i // grid_size, i % grid_size
        grid[row, col] = int(bit)
    ax2.imshow(grid, cmap='binary', interpolation='nearest')
    ax2.set_title(f"Mapa de Hilbert ({grid_size}x{grid_size})")
    ax2.axis('off')
    
    # 3. Distribució de patrons (segona fila, columnes 0-1)
    ax3 = fig.add_subplot(gs[1, :2])
    if patterns:
        pattern_lengths = [len(p['pattern_data']) for p in patterns]
        ax3.hist(pattern_lengths, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title(f"Distribució de Longituds de Patrons (n={len(patterns)})")
        ax3.set_xlabel("Longitud")
        ax3.set_ylabel("Freqüència")
    else:
        ax3.text(0.5, 0.5, "No hi ha patrons\ndetectats", ha='center', va='center')
        ax3.set_title("Patrons Detectats")
    
    # 4. Rendiment de regles (segona fila, columnes 2-3)
    ax4 = fig.add_subplot(gs[1, 2:])
    if rules:
        confidences = [r.get('confidence', 0) for r in rules]
        precisions = [r.get('precision', 0) for r in rules]
        ax4.scatter(confidences, precisions, alpha=0.6, color='coral')
        ax4.set_title(f"Confiança vs Precisió de Regles (n={len(rules)})")
        ax4.set_xlabel("Confiança")
        ax4.set_ylabel("Precisió")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "No hi ha regles\ninferides", ha='center', va='center')
        ax4.set_title("Regles Inferides")
    
    # 5. Resultats de validació (tercera fila, totes les columnes)
    ax5 = fig.add_subplot(gs[2, :])
    rule_scores = validation_results.get('rule_scores', {})
    if rule_scores:
        rule_ids = list(rule_scores.keys())
        accuracies = [rule_scores[rid].get('accuracy', 0) for rid in rule_ids]
        ax5.bar(range(len(accuracies)), accuracies, alpha=0.7, color='lightgreen')
        ax5.set_title("Precisió de Validació per Regla")
        ax5.set_xlabel("Índex de Regla")
        ax5.set_ylabel("Precisió")
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, "No hi ha resultats\nde validació", ha='center', va='center')
        ax5.set_title("Resultats de Validació")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Dashboard guardat a {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Exemple d'ús
    print("🎨 Utilitats de Visualització HSI")
    print("=" * 40)
    
    # Generar dades d'exemple
    example_phi = "0101001101010011010100110101001101010011" * 10
    
    example_patterns = [
        {'pattern_data': '101', 'recurrence': 5, 'density': 0.1},
        {'pattern_data': '010', 'recurrence': 3, 'density': 0.05},
        {'pattern_data': '1010', 'recurrence': 2, 'density': 0.03}
    ]
    
    example_rules = [
        {'confidence': 0.8, 'precision': 0.7, 'stability': 0.9, 'complexity': 2.5, 'rule_type': 'markov'},
        {'confidence': 0.6, 'precision': 0.8, 'stability': 0.7, 'complexity': 3.0, 'rule_type': 'context'}
    ]
    
    # Mostrar visualitzacions d'exemple
    plot_phi_sequence(example_phi[:100], title="Exemple Seqüència Φ")
    plot_pattern_distribution(example_patterns, title="Exemple Distribució Patrons")
    plot_rule_performance(example_rules, title="Exemple Rendiment Regles")
