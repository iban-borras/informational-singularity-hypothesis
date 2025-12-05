"""
Level 1 module — Informational Singularity Hypothesis (HSI)

This package implements Level 1 analysis: Pattern detection, order metrics,
and structural analysis of Level 0 data.

Level 1 focuses on:
- Loading Level 0 data (both v32 and v33 formats)
- Detecting patterns in Φ sequences
- Analyzing structural order (nesting, containment, stratification)
- Computing order metrics (not φ-convergence, but emergent order)
"""

from .data_loader import (
    load_phi_for_level1,
    load_metadata,
    extract_observable_phi,
    extract_structural_info,
)

__all__ = [
    'load_phi_for_level1',
    'load_metadata',
    'extract_observable_phi',
    'extract_structural_info',
]

