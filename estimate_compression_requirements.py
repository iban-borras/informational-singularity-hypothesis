#!/usr/bin/env python3
"""
📊 HSI Compression Requirements Estimator

This script estimates memory and disk requirements for high iterations
of the Φ generator, implementing the strategy defined in level0_data_storage.md.

Author: Iban Borràs in collaboration with Augment Agent
Date: January 2025
"""

import argparse
import sys
from level0 import estimate_iteration_requirements, PhiSnapshotManager


def print_header(hardware_limit_gb=None):
    """Print the program header."""
    print("📊 HSI Compression Requirements Estimator")
    print("=" * 60)
    print("Strategy based on level0_data_storage.md")
    if hardware_limit_gb:
        print(f"Hardware limit: {hardware_limit_gb} GB")
    print("=" * 60)


def get_hardware_limits():
    """Get hardware limits from user input."""
    print("\n🖥️  HARDWARE CONFIGURATION")
    print("-" * 30)

    try:
        # Get available storage
        storage_input = input("💾 Available storage space (GB) [default: 1000]: ").strip()
        storage_gb = float(storage_input) if storage_input else 1000.0

        # Get available RAM
        ram_input = input("🧠 Available RAM (GB) [default: 32]: ").strip()
        ram_gb = float(ram_input) if ram_input else 32.0

        # Confirm values
        print(f"\n✅ Configuration:")
        print(f"   Storage: {storage_gb} GB")
        print(f"   RAM: {ram_gb} GB")

        confirm = input("\nProceed with these values? [Y/n]: ").strip().lower()
        if confirm and confirm != 'y' and confirm != 'yes':
            print("❌ Cancelled by user")
            sys.exit(0)

        return storage_gb, ram_gb

    except (ValueError, KeyboardInterrupt):
        print("\n❌ Invalid input or cancelled by user")
        sys.exit(1)


def print_iteration_estimate(iteration: int, estimate: dict):
    """Print estimation for a specific iteration."""
    print(f"\n🔢 Iteration {iteration}:")
    print(f"   Estimated bits: {estimate['estimated_bits']:,}")
    print(f"   Uncompressed memory: {estimate['uncompressed_mb']:.1f} MB")
    print(f"   Compressed memory: {estimate['compressed_mb']:.1f} MB")
    print(f"   Recommend compression: {'✅ Yes' if estimate['recommended_compression'] else '❌ No'}")


def print_summary_table(estimates: list):
    """Print summary table of estimations."""
    print("\n📋 SUMMARY TABLE")
    print("-" * 80)
    print(f"{'Iter':<6} {'Bits':<15} {'RAM (MB)':<12} {'Compressed (MB)':<15} {'Compression':<12}")
    print("-" * 80)

    for est in estimates:
        iter_num = est['iteration']
        bits = f"{est['estimated_bits']:,}"
        ram_mb = f"{est['uncompressed_mb']:.1f}"
        comp_mb = f"{est['compressed_mb']:.1f}"
        compress = "✅" if est['recommended_compression'] else "❌"

        print(f"{iter_num:<6} {bits:<15} {ram_mb:<12} {comp_mb:<15} {compress:<12}")


def print_storage_requirements(requirements: dict, hardware_limit_gb: float = 1000):
    """Print total storage requirements."""
    print(f"\n💾 TOTAL STORAGE REQUIREMENTS")
    print("-" * 50)
    print(f"Maximum iteration: {requirements['max_iteration']}")
    print(f"Total storage: {requirements['total_estimated_storage_gb']:.2f} GB")
    print(f"Compression starts at: Iteration {requirements['compression_starts_at']}")

    # Check if within hardware limits
    if requirements['total_estimated_storage_gb'] <= hardware_limit_gb:
        print(f"✅ Within hardware limits ({hardware_limit_gb} GB available)")
    else:
        print(f"⚠️  Exceeds hardware limits ({hardware_limit_gb} GB available)")
        print(f"   Excess: {requirements['total_estimated_storage_gb'] - hardware_limit_gb:.2f} GB")


def print_recommendations(max_iteration: int, requirements: dict, hardware_limit_gb: float = 1000):
    """Print recommendations based on analysis."""
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 30)

    storage_needed = requirements['total_estimated_storage_gb']

    if max_iteration <= 20:
        print("✅ Iterations <= 20: Feasible with current configuration")
        print("   • Use automatic compression from iteration 18")
        print("   • Keep snapshots of last 5 states")

    elif max_iteration <= 28:
        print("⚠️  Iterations 21-28: Requires optimization")
        print("   • Enable maximum compression (level=9)")
        print("   • Consider automatic cleanup of old snapshots")
        print("   • Monitor disk usage actively")

    elif max_iteration <= 32:
        if storage_needed <= hardware_limit_gb:
            print("✅ Iterations 29-32: Feasible with optimization")
            print("   • Use aggressive compression")
            print("   • Implement automatic cleanup")
            print("   • Monitor storage closely")
        else:
            print("🔥 Iterations 29-32: Hardware limit exceeded")
            print("   • Use aggressive compression")
            print("   • Implement chunked generation")
            print("   • Consider BitPacked encoding")
            print("   • Mandatory automatic cleanup")

    else:
        print("❌ Iterations > 32: Not feasible with current hardware")
        print("   • Requires hardware with more capacity")
        print("   • Consider distributed computing")
        print("   • Implement streaming algorithms")

    # Configuration-specific recommendations
    print(f"\n🔧 Recommended configuration for iteration {max_iteration}:")
    print(f"   memory_threshold: {10**7 if max_iteration <= 20 else 10**6}")
    print(f"   compression_level: {6 if max_iteration <= 24 else 9}")
    print(f"   keep_snapshots: {5 if max_iteration <= 28 else 2}")

    # Hardware-specific advice
    if storage_needed > hardware_limit_gb:
        shortage = storage_needed - hardware_limit_gb
        print(f"\n⚠️  Storage shortage: {shortage:.1f} GB")
        print("   Consider:")
        print("   • External storage drives")
        print("   • Cloud storage solutions")
        print("   • Lower target iteration")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Estimate compression requirements for HSI simulations",
        epilog="💡 The script will ask for your hardware limits interactively"
    )
    parser.add_argument(
        "--max-iteration", "-m",
        type=int,
        default=32,
        help="Maximum iteration to analyze (default: 32)"
    )
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed estimations for each iteration"
    )
    parser.add_argument(
        "--start-iteration", "-s",
        type=int,
        default=16,
        help="Starting iteration for analysis (default: 16)"
    )
    parser.add_argument(
        "--hardware-storage",
        type=float,
        help="Available storage in GB (if not provided, will ask interactively)"
    )
    parser.add_argument(
        "--hardware-ram",
        type=float,
        help="Available RAM in GB (if not provided, will ask interactively)"
    )

    args = parser.parse_args()

    # Get hardware limits
    if args.hardware_storage and args.hardware_ram:
        storage_gb, ram_gb = args.hardware_storage, args.hardware_ram
        print_header(storage_gb)
    else:
        storage_gb, ram_gb = get_hardware_limits()
        print_header(storage_gb)

    # Generate estimations
    print(f"\n🔍 Analyzing iterations {args.start_iteration} to {args.max_iteration}...")
    requirements = estimate_iteration_requirements(args.max_iteration)
    
    # Show detailed estimations if requested
    if args.detailed:
        print(f"\n📊 DETAILED ESTIMATIONS")
        print("=" * 60)

        manager = PhiSnapshotManager()
        for i in range(args.start_iteration, args.max_iteration + 1):
            estimate = manager.estimate_memory_usage(i)
            print_iteration_estimate(i, estimate)

    # Show summary table
    estimates_for_table = []
    manager = PhiSnapshotManager()

    # Select key iterations for table
    key_iterations = list(range(args.start_iteration, min(args.max_iteration + 1, 25), 2))
    if args.max_iteration > 24:
        key_iterations.extend(range(26, args.max_iteration + 1, 2))

    for i in key_iterations:
        estimate = manager.estimate_memory_usage(i)
        estimates_for_table.append(estimate)

    print_summary_table(estimates_for_table)

    # Show total requirements
    print_storage_requirements(requirements, storage_gb)

    # Show recommendations
    print_recommendations(args.max_iteration, requirements, storage_gb)
    
    print(f"\n🎯 CONCLUSION")
    print("-" * 20)
    storage_needed = requirements['total_estimated_storage_gb']

    if storage_needed <= storage_gb:
        print(f"✅ Feasible to reach iteration {args.max_iteration}")
        print(f"   Storage needed: {storage_needed:.1f} GB")
        print(f"   Available: {storage_gb} GB")
        print(f"   Remaining: {storage_gb - storage_needed:.1f} GB")
    else:
        print(f"⚠️  Challenging to reach iteration {args.max_iteration}")
        print(f"   Storage needed: {storage_needed:.1f} GB")
        print(f"   Available: {storage_gb} GB")
        print(f"   Shortage: {storage_needed - storage_gb:.1f} GB")
        print("   Consider additional optimizations or external storage")

    print("\n" + "=" * 60)
    print("📖 For more details, see level0_data_storage.md")


if __name__ == "__main__":
    main()
