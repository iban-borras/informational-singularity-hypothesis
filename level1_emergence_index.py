#!/usr/bin/env python3
"""
🌌 HSI Level 1 Emergence Index Calculator

Entry point for calculating the Emergence Index, a composite metric that
estimates the potential of a Φ sequence to exhibit emergent physical-like behavior.

The index combines:
- Criticality (1/f spectrum): Systems at "edge of chaos" show 1/f noise
- Lempel-Ziv Complexity: Neither pure order nor pure chaos = maximum potential
- Long-Range Coherence: Mutual Information between distant regions

Usage:
    python level1_emergence_index.py --variant B --iteration 18
    python level1_emergence_index.py --variants A B D E F --iteration 18 --compare
    python level1_emergence_index.py --help

Author: Iban Borràs with collaboration from Augment Agent (Sophia)
"""

import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

from metrics.emergence_index import main

if __name__ == '__main__':
    main()

