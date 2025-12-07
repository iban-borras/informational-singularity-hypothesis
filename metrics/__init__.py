"""
Metrics Module - Validation Metrics for HSI

This module contains implementations of metrics defined in
validation_metrics_agents.md to evaluate the quality of detected
patterns Pₖ and rules ωₖ.

Available modules:
- pattern_metrics: Metrics to evaluate patterns Pₖ
- rule_metrics: Metrics to evaluate rules ωₖ
- order_metrics: Metrics to evaluate emergent order (v33 structural)
- emergence_index: Emergence Index for Level 2 potential estimation
"""

from .pattern_metrics import PatternMetrics, create_pattern_report
from .rule_metrics import RuleMetrics, create_rule_report
from .order_metrics import OrderMetrics, create_order_report
from .emergence_index import (
    calculate_emergence_index,
    compare_variants_emergence,
    calculate_power_spectrum_slope,
    calculate_lempel_ziv_complexity,
    calculate_long_range_mutual_info
)

__all__ = [
    'PatternMetrics', 'RuleMetrics', 'OrderMetrics',
    'create_pattern_report', 'create_rule_report', 'create_order_report',
    'calculate_emergence_index', 'compare_variants_emergence',
    'calculate_power_spectrum_slope', 'calculate_lempel_ziv_complexity',
    'calculate_long_range_mutual_info'
]
