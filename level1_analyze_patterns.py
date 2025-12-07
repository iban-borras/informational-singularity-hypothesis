#!/usr/bin/env python3
"""
🔍 HSI Level 1 Pattern Analyzer

Entry point for Level 1 pattern detection and rule inference.
This script orchestrates the analysis of Φ sequences to detect emergent patterns.

Usage:
    python level1_analyze_patterns.py --variant B --iteration 18
    python level1_analyze_patterns.py --variant B --iteration 19 --min-len 10 --max-len 50
    python level1_analyze_patterns.py --help

Author: Iban Borràs with collaboration from Augment Agent (Sophia)
"""

import sys
import runpy
from pathlib import Path

if __name__ == '__main__':
    # Run the orchestrator as the main module
    script_path = Path(__file__).parent / "agents" / "level1_orchestrator.py"
    sys.argv[0] = str(script_path)
    runpy.run_path(str(script_path), run_name="__main__")

