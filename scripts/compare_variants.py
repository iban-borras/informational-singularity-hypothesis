#!/usr/bin/env python3
"""Compare all variants from sci_icc_*.json files."""

import json
from pathlib import Path


def main():
    metrics_dir = Path(__file__).parent.parent / "results" / "level1" / "metrics"
    files = list(metrics_dir.glob("sci_icc_*.json"))
    
    if not files:
        print("No sci_icc_*.json files found!")
        return
    
    all_data = []
    
    for f in sorted(files):
        with open(f) as fp:
            data = json.load(fp)
        
        variants = data.get("variants", {})
        for key, v in variants.items():
            all_data.append({
                "variant": v.get("variant", "?"),
                "iteration": v.get("iteration", 0),
                "ei": v.get("emergence_index", 0),
                "sci": v.get("sci", 0),
                "icc": v.get("icc", 0),
                "phi_tend": v.get("phi_tendency", 0),
                "bits": v.get("total_bits", 0)
            })
    
    # Sort by variant then iteration
    all_data.sort(key=lambda x: (x["variant"], x["iteration"]))
    
    # Print header
    print("\n" + "=" * 75)
    print("📊 COMPARATIVE ANALYSIS - ALL VARIANTS")
    print("=" * 75)
    print(f"{'Variant':<8} {'Iter':<5} {'EI':<9} {'SCI':<9} {'ICC':<9} {'φ-Tend':<9} {'Bits':<12}")
    print("-" * 75)
    
    current_var = None
    var_stats = {}
    
    for d in all_data:
        if current_var != d["variant"]:
            if current_var is not None:
                print()  # Blank line between variants
            current_var = d["variant"]
            var_stats[current_var] = {"ei": [], "sci": [], "icc": [], "phi_tend": []}
        
        # Collect stats
        var_stats[current_var]["ei"].append(d["ei"])
        var_stats[current_var]["sci"].append(d["sci"])
        var_stats[current_var]["icc"].append(d["icc"])
        var_stats[current_var]["phi_tend"].append(d["phi_tend"])
        
        bits_str = f"{d['bits']/1e9:.2f}B" if d["bits"] > 1e9 else f"{d['bits']/1e6:.1f}M"
        print(f"{d['variant']:<8} {d['iteration']:<5} {d['ei']:<9.4f} {d['sci']:<9.4f} {d['icc']:<9.4f} {d['phi_tend']:<9.4f} {bits_str:<12}")
    
    # Print summary by variant
    print("\n" + "=" * 75)
    print("📈 SUMMARY BY VARIANT (mean values)")
    print("=" * 75)
    print(f"{'Variant':<10} {'EI':<10} {'SCI':<10} {'ICC':<10} {'φ-Tend':<10} {'Type':<15}")
    print("-" * 75)
    
    # Classify variants
    def classify(var):
        if var.lower() in ['k', 'l']:  # Rule 30 and similar controls
            return "🔴 Control"
        else:
            return "🟢 HSI"
    
    summaries = []
    for var in sorted(var_stats.keys()):
        stats = var_stats[var]
        avg_ei = sum(stats["ei"]) / len(stats["ei"])
        avg_sci = sum(stats["sci"]) / len(stats["sci"])
        avg_icc = sum(stats["icc"]) / len(stats["icc"])
        avg_phi = sum(stats["phi_tend"]) / len(stats["phi_tend"])
        var_type = classify(var)
        
        summaries.append({
            "var": var,
            "ei": avg_ei,
            "sci": avg_sci,
            "icc": avg_icc,
            "phi": avg_phi,
            "type": var_type
        })
        
        print(f"{var:<10} {avg_ei:<10.4f} {avg_sci:<10.4f} {avg_icc:<10.4f} {avg_phi:<10.4f} {var_type:<15}")
    
    # Calculate HSI vs Control averages
    hsi_vars = [s for s in summaries if "HSI" in s["type"]]
    ctrl_vars = [s for s in summaries if "Control" in s["type"]]
    
    if hsi_vars and ctrl_vars:
        print("\n" + "=" * 75)
        print("🔬 HSI vs CONTROL COMPARISON")
        print("=" * 75)
        
        def avg(lst, key):
            return sum(s[key] for s in lst) / len(lst) if lst else 0
        
        print(f"{'Group':<15} {'EI':<10} {'SCI':<10} {'ICC':<10} {'φ-Tend':<10}")
        print("-" * 55)
        print(f"{'HSI (n=' + str(len(hsi_vars)) + ')':<15} {avg(hsi_vars, 'ei'):<10.4f} {avg(hsi_vars, 'sci'):<10.4f} {avg(hsi_vars, 'icc'):<10.4f} {avg(hsi_vars, 'phi'):<10.4f}")
        print(f"{'Control (n=' + str(len(ctrl_vars)) + ')':<15} {avg(ctrl_vars, 'ei'):<10.4f} {avg(ctrl_vars, 'sci'):<10.4f} {avg(ctrl_vars, 'icc'):<10.4f} {avg(ctrl_vars, 'phi'):<10.4f}")
        
        # Ratio
        print("\n📊 Ratio HSI/Control:")
        for metric in ["ei", "sci", "icc", "phi"]:
            hsi_avg = avg(hsi_vars, metric)
            ctrl_avg = avg(ctrl_vars, metric)
            ratio = hsi_avg / ctrl_avg if ctrl_avg > 0.0001 else float('inf')
            label = {"ei": "EI", "sci": "SCI", "icc": "ICC", "phi": "φ-Tend"}[metric]
            if ratio == float('inf'):
                print(f"   {label}: ∞ (control ≈ 0)")
            else:
                print(f"   {label}: {ratio:.1f}x")
    
    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()

