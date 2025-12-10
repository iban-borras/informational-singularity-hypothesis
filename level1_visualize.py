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
import subprocess
from pathlib import Path

if __name__ == '__main__':
    # Use subprocess to run as module for proper relative imports
    args = [sys.executable, '-m', 'level1.level1_figures'] + sys.argv[1:]
    sys.exit(subprocess.call(args))

