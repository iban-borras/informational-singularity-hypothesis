"""
Level 0 module — Informational Singularity Hypothesis (HSI)

This package exposes the Level 0 generator (basal‑pure dynamics) and
snapshot/compression utilities.

Note: We avoid importing the generator at package import time to prevent
Python runpy double‑import warnings when executing
"python -m hsi_agents_project.level0.generator".
"""

# Lazy wrappers to avoid importing generator at package import time

def simulate_phi(*args, **kwargs):
    from .generator import simulate_phi as _impl
    return _impl(*args, **kwargs)


def get_phi_array(*args, **kwargs):
    from .generator import get_phi_array as _impl
    return _impl(*args, **kwargs)


def phi_to_hilbert_coords(*args, **kwargs):
    from .generator import phi_to_hilbert_coords as _impl
    return _impl(*args, **kwargs)

from .phi_snapshot_manager import (
    PhiSnapshotManager,
    create_snapshot_manager,
    estimate_iteration_requirements,
)

__all__ = [
    'simulate_phi',
    'get_phi_array',
    'phi_to_hilbert_coords',
    'PhiSnapshotManager',
    'create_snapshot_manager',
    'estimate_iteration_requirements'
]
