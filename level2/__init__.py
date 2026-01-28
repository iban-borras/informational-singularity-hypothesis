"""
HSI Level 2 - Advanced Complexity Analysis

This package contains modules for Level 2 analysis:
- Takens Embedding & Topological Data Analysis (TDA)
- Transfer Entropy
- Complexity Analysis

Submodules:
- _tda_local: Windowed TDA analysis (parallelized)
- _tda_global: Global TDA with landmark subsampling
"""

from pathlib import Path

# Package metadata
__version__ = "1.0.0"
__author__ = "Sophia (AI Assistant)"

# Results directory
RESULTS_DIR = Path("results/level2")
TDA_DIR = RESULTS_DIR / "tda_analysis"

# Public API
from ._tda_local import run_local_tda
from ._tda_global import (
    run_global_tda, run_global_tda_blocks, get_best_backend,
    run_witness_tda, run_cubical_tda, sequence_to_image
)

__all__ = [
    'run_local_tda', 'run_global_tda', 'run_global_tda_blocks',
    'get_best_backend', 'run_witness_tda', 'run_cubical_tda',
    'sequence_to_image', 'RESULTS_DIR', 'TDA_DIR'
]
