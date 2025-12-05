"""
Mòdul Metrics - Mètriques de Validació per la HSI

Aquest mòdul conté les implementacions de les mètriques definides a
validation_metrics_agents.md per avaluar la qualitat dels patrons Pₖ
i regles ωₖ detectades pels agents.

Mòduls disponibles:
- pattern_metrics: Mètriques per avaluar patrons Pₖ
- rule_metrics: Mètriques per avaluar regles ωₖ
- order_metrics: Mètriques per avaluar ordre emergent (v33 structural)
"""

from .pattern_metrics import PatternMetrics, create_pattern_report
from .rule_metrics import RuleMetrics, create_rule_report
from .order_metrics import OrderMetrics, create_order_report

__all__ = [
    'PatternMetrics', 'RuleMetrics', 'OrderMetrics',
    'create_pattern_report', 'create_rule_report', 'create_order_report'
]
