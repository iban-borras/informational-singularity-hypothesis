#!/usr/bin/env python3
"""
📁 Migration Script: Reorganize results/ folder structure

This script moves existing files to the new unified structure:
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

Run with --dry-run to preview changes without moving files.

Author: Iban Borràs with collaboration from Augment Agent (Sophia)
"""

import shutil
import argparse
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def migrate(dry_run: bool = False):
    """Migrate files to new structure."""
    moved = 0
    
    def move_file(src: Path, dst: Path):
        nonlocal moved
        if dry_run:
            print(f"  [DRY-RUN] {src.name} → {dst.relative_to(RESULTS)}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.move(str(src), str(dst))
                print(f"  ✓ {src.name} → {dst.relative_to(RESULTS)}")
                moved += 1
            else:
                print(f"  ⚠️ Skipped (exists): {dst.name}")
    
    print("=" * 60)
    print("📁 HSI Results Migration Script")
    print("=" * 60)
    if dry_run:
        print("⚠️  DRY RUN MODE - No files will be moved\n")
    
    # 1. Create new directory structure
    print("\n📂 Creating new directory structure...")
    new_dirs = [
        RESULTS / "level0" / "phi_snapshots",
        RESULTS / "level0" / "reports",
        RESULTS / "level0" / "visualizations",
        RESULTS / "level1" / "analysis",
        RESULTS / "level1" / "metrics",
        RESULTS / "level1" / "figures",
        RESULTS / "level1" / "trends" / "plots",
    ]
    for d in new_dirs:
        if not dry_run:
            d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d.relative_to(RESULTS)}")
    
    # 2. Move phi_snapshots to level0/phi_snapshots
    print("\n📦 Moving phi_snapshots to level0/...")
    old_phi = RESULTS / "phi_snapshots"
    new_phi = RESULTS / "level0" / "phi_snapshots"
    if old_phi.exists() and old_phi.is_dir():
        for var_dir in old_phi.iterdir():
            if var_dir.is_dir():
                dst = new_phi / var_dir.name
                if dry_run:
                    print(f"  [DRY-RUN] {var_dir.name}/ → level0/phi_snapshots/{var_dir.name}/")
                else:
                    if not dst.exists():
                        shutil.move(str(var_dir), str(dst))
                        print(f"  ✓ {var_dir.name}/ → level0/phi_snapshots/")
                        moved += 1
    
    # 3. Move reports to level0/reports
    print("\n📦 Moving reports to level0/reports/...")
    old_reports = RESULTS / "reports"
    new_reports = RESULTS / "level0" / "reports"
    if old_reports.exists():
        for f in old_reports.glob("*"):
            if f.is_file():
                move_file(f, new_reports / f.name)
    
    # 4. Move visualizations to level0/visualizations
    print("\n📦 Moving visualizations to level0/visualizations/...")
    old_vis = RESULTS / "visualizations"
    new_vis = RESULTS / "level0" / "visualizations"
    if old_vis.exists():
        for f in old_vis.glob("*"):
            if f.is_file():
                move_file(f, new_vis / f.name)
    
    # 5. Move level1_analysis_* files to level1/analysis/
    print("\n📦 Moving Level 1 analysis files...")
    new_analysis = RESULTS / "level1" / "analysis"
    for f in RESULTS.glob("level1_analysis_*.json"):
        move_file(f, new_analysis / f.name)
    for f in RESULTS.glob("level1_analysis_*.md"):
        move_file(f, new_analysis / f.name)
    for f in RESULTS.glob("report_var*.md"):
        move_file(f, new_analysis / f.name)
    
    # 6. Move emergence_*, sci_icc_* to level1/metrics/
    print("\n📦 Moving Level 1 metrics files...")
    new_metrics = RESULTS / "level1" / "metrics"
    for f in RESULTS.glob("emergence_*.json"):
        move_file(f, new_metrics / f.name)
    for f in RESULTS.glob("sci_icc_*.json"):
        move_file(f, new_metrics / f.name)
    
    # 7. Move trend_analysis and trend_plots to level1/trends/
    print("\n📦 Moving trend analysis files...")
    new_trends = RESULTS / "level1" / "trends"
    trend_json = RESULTS / "trend_analysis.json"
    if trend_json.exists():
        move_file(trend_json, new_trends / trend_json.name)
    
    old_trend_plots = RESULTS / "trend_plots"
    new_trend_plots = RESULTS / "level1" / "trends" / "plots"
    if old_trend_plots.exists():
        for f in old_trend_plots.glob("*"):
            if f.is_file():
                move_file(f, new_trend_plots / f.name)
    
    # 8. Move figures to level1/figures/
    print("\n📦 Moving Level 1 figures...")
    old_figs = RESULTS / "figures"
    new_figs = RESULTS / "level1" / "figures"
    if old_figs.exists():
        for item in old_figs.iterdir():
            if item.is_file():
                move_file(item, new_figs / item.name)
            elif item.is_dir():
                # Move subdirectories (like survival_analysis/)
                dst_dir = new_figs / item.name
                if dry_run:
                    print(f"  [DRY-RUN] {item.name}/ → level1/figures/{item.name}/")
                else:
                    if not dst_dir.exists():
                        shutil.move(str(item), str(dst_dir))
                        print(f"  ✓ {item.name}/ → level1/figures/")
                        moved += 1
    
    # 9. Clean up empty old directories
    print("\n🧹 Cleaning up empty directories...")
    old_dirs = ['phi_snapshots', 'reports', 'visualizations', 'figures', 'trend_plots']
    for folder_name in old_dirs:
        old = RESULTS / folder_name
        if old.exists():
            try:
                if not any(old.iterdir()):
                    if not dry_run:
                        old.rmdir()
                    print(f"  ✓ Removed empty: {folder_name}/")
                else:
                    print(f"  ⚠️ Not empty, kept: {folder_name}/")
            except Exception as e:
                print(f"  ⚠️ Could not remove {folder_name}/: {e}")
    
    print("\n" + "=" * 60)
    if dry_run:
        print("🔍 DRY RUN complete. Run without --dry-run to apply changes.")
    else:
        print(f"✅ Migration complete! Moved {moved} items.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate results/ to new structure")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without moving files")
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)

