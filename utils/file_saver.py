"""
📁 Unified File Saving Utility for HSI Agents Project

Provides consistent paths and logging for all file operations.
All saves are logged with relative paths from project root.

Structure:
    results/
    ├── level0/
    │   ├── phi_snapshots/var_{X}/
    │   ├── reports/
    │   └── visualizations/
    ├── level1/
    │   ├── analysis/
    │   ├── metrics/
    │   ├── figures/
    │   └── trends/
    ├── cache/
    └── temp/

Author: Iban Borràs with collaboration from Augment Agent (Sophia)
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Base paths (anchored to project root)
BASE_PATH = Path(__file__).resolve().parent.parent  # hsi_agents_project/
RESULTS_DIR = BASE_PATH / "results"

# Standard categories per level
LEVEL0_CATEGORIES = ['phi_snapshots', 'reports', 'visualizations']
LEVEL1_CATEGORIES = ['analysis', 'metrics', 'figures', 'trends']
SHARED_CATEGORIES = ['cache', 'temp']


def get_results_dir() -> Path:
    """Get the base results directory."""
    return RESULTS_DIR


def get_base_path() -> Path:
    """Get the project base path for relative path calculations."""
    return BASE_PATH


def get_output_path(
    level: int,
    category: str,
    filename: str = "",
    variant: Optional[str] = None
) -> Path:
    """
    Get standardized output path.

    Args:
        level: 0 or 1 (use -1 for shared categories like 'cache')
        category: 'analysis', 'metrics', 'figures', 'reports', 'visualizations', etc.
        filename: The filename (e.g., 'emergence_B_iter17.json'). Empty to get directory.
        variant: Optional variant code for phi_snapshots (creates var_{X}/ subdirectory)

    Returns:
        Full path for the output file or directory

    Examples:
        >>> get_output_path(1, "metrics", "emergence_B_iter17.json")
        Path(".../results/level1/metrics/emergence_B_iter17.json")

        >>> get_output_path(0, "phi_snapshots", variant="B")
        Path(".../results/level0/phi_snapshots/var_B")

        >>> get_output_path(-1, "cache", "level1_cache_abc.pkl")
        Path(".../results/cache/level1_cache_abc.pkl")
    """
    if level == -1:
        # Shared category (cache, temp)
        path = RESULTS_DIR / category
    else:
        path = RESULTS_DIR / f"level{level}" / category

    # Add variant subdirectory for phi_snapshots
    if variant and category == "phi_snapshots":
        path = path / f"var_{variant}"

    # Add filename if provided
    if filename:
        path = path / filename

    # Ensure parent directory exists
    if filename:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)

    return path


def relative_path(path: Path) -> str:
    """Get path relative to project root for logging."""
    try:
        return str(path.relative_to(BASE_PATH))
    except ValueError:
        return str(path)


def save_json(
    data: Dict[str, Any],
    path: Union[Path, str],
    log: bool = True,
    indent: int = 2
) -> Path:
    """
    Save JSON with consistent logging.

    Args:
        data: Dictionary to save
        path: Output path
        log: Whether to print save message
        indent: JSON indentation (default: 2)

    Returns:
        The path where data was saved
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=str)

    if log:
        print(f"💾 Saved: {relative_path(path)}")

    return path


def save_figure(
    fig,
    path: Union[Path, str],
    dpi: int = 150,
    log: bool = True,
    close: bool = True
) -> Path:
    """
    Save matplotlib figure with consistent logging.

    Args:
        fig: Matplotlib figure object
        path: Output path
        dpi: Resolution (default: 150)
        log: Whether to print save message
        close: Whether to close figure after saving

    Returns:
        The path where figure was saved
    """
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, dpi=dpi, bbox_inches='tight')

    if close:
        plt.close(fig)

    if log:
        print(f"📊 Saved: {relative_path(path)}")

    return path

