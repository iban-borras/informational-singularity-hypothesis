#!/usr/bin/env python3
"""
Level 1 Visualization - Wrapper Script

Generates publication-quality figures from Level 1 analysis results.

Usage:
    python level1_visualize.py --all
    python level1_visualize.py --variants B A --iterations 18
    python level1_visualize.py --format pdf --dpi 600
"""

import sys
import runpy
from pathlib import Path

if __name__ == '__main__':
    script_path = Path(__file__).parent / "level1" / "level1_figures.py"
    sys.argv[0] = str(script_path)
    runpy.run_path(str(script_path), run_name="__main__")

