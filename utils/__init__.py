"""
Mòdul Utils - Utilitats per la HSI

Aquest mòdul conté funcions auxiliars per al sistema d'agents de la
Hipòtesi de Singularitat Informacional.
"""

# Lazy imports to avoid circular dependencies and heavy dependencies like matplotlib
# Import visualization functions directly when needed:
# from hsi_agents_project.utils.visualization import plot_phi_sequence, etc.

__all__ = [
    'plot_phi_sequence',
    'plot_phi_hilbert',
    'plot_pattern_distribution',
    'plot_rule_performance',
    'plot_validation_results',
    'create_summary_dashboard'
]
