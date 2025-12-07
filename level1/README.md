# 📊 Level 1 — Pattern Analysis & Order Flow

**Purpose:** Load and analyze Level 0 data with structural information preservation

---

## 🎯 Overview

Level 1 implements pattern detection and order analysis for the Informational Singularity Hypothesis (HSI). Level 1 can analyze **structural order** encoded in parentheses, enabling detection of emergent patterns.

---

## 🔧 Components

### **1. Data Loader** (`data_loader.py`)

Loads Level 0 data with automatic format detection.

**Key functions:**
- `load_phi_for_level1()`: Load Φ with both structural and observable versions
- `load_metadata()`: Load iteration metadata
- `extract_observable_phi()`: Remove parentheses from structural Φ
- `extract_structural_info()`: Extract nesting depth and containment info
- `load_phi_range()`: Load multiple iterations efficiently

**Example:**
```python
from level1.data_loader import load_phi_for_level1, extract_structural_info

# Load iteration 10 with structure
phi_structural, phi_observable, metadata = load_phi_for_level1(
    "results/var_B",
    iteration=10,
    return_structural=True,
    return_observable=True,
    return_metadata=True
)

# Analyze structure
info = extract_structural_info(phi_structural)
print(f"Max nesting depth: {info['max_depth']}")
print(f"Number of Absolutes: {info['num_absolutes']}")
```

---

## 📁 Data Format Support

### **Legacy Format**
- **File:** `phi_iter{N}.bin.gz` (text gzip)
- **Content:** Clean binary string "010101..."
- **Limitation:** No structural information

### **Structural Format (Current)**
- **File:** `phi_iter{N}.struct.gz` (bitarray gzip)
- **Content:** 2-bit encoded structure with parentheses
- **Encoding:**
  - `00` → '0' (state 0)
  - `01` → '1' (state 1)
  - `10` → '(' (Absolute start)
  - `11` → ')' (Absolute end)
- **Advantage:** Preserves complete structural information for order analysis

---

## 🔬 Structural Analysis

### **What is Structural Information?**

In ISH, **order is the fundamental principle**, not φ-convergence. The parentheses in Φ encode the **stratified containment structure** of Absolutes, which is critical for understanding emergent order.

**Example:**
```
Structural Φ: ((01)1)
              ││││││
              ││││└┴─ Absolute 1 closes
              │││└─── State 1
              ││└──── Absolute 2 closes  
              │└───── State 1
              └────── Absolute 2 opens (contains Absolute 1)

Observable Φ: 011
              (parentheses removed)
```

### **Structural Metrics**

1. **Nesting Depth**: How deep Absolutes are nested
2. **Containment**: What each Absolute contains
3. **Stratified Order**: Entropy and information density at each depth level

---

## 🚀 Quick Start

### **1. Load Data**

```python
from level1.data_loader import load_phi_for_level1

# Load data with structure
phi_s, phi_o, meta = load_phi_for_level1("results/var_B", 10)

print(f"Format: {meta['format']}")
print(f"Structural: {phi_s[:50]}...")
print(f"Observable: {phi_o[:50]}...")
```

### **2. Analyze Structure**

```python
from level1.data_loader import extract_structural_info

info = extract_structural_info(phi_s)

print(f"Max depth: {info['max_depth']}")
print(f"Absolutes: {info['num_absolutes']}")
print(f"Depth distribution: {info['depth_distribution']}")
```

### **3. Load Multiple Iterations**

```python
from level1.data_loader import load_phi_range

data = load_phi_range("results/var_B", start_iteration=1, end_iteration=10)

for iter_num, (phi_s, phi_o) in data.items():
    print(f"Iteration {iter_num}: {len(phi_o):,} bits")
```

---

## 📚 Examples

See `example_usage.py` for comprehensive examples:
- Loading single iterations
- Analyzing structural information
- Fast loading (observable only)
- Loading ranges

---

## 🧪 Testing

Run the test suite:
```bash
python hsi_agents_project/test_level1_data_loader.py
```

Tests include:
- Extract observable from structural
- Extract structural information
- Load metadata
- Load v33 data
- Verify format detection

---

## ⚙️ Configuration

Edit `config.json` to configure Level 1:

```json
"level1": {
  "data_source": "results/var_B",
  "max_iterations_to_load": 25,
  "enable_structural_analysis": true,
  "structural_metrics": ["nesting", "containment", "stratified_order"],
  "max_nesting_depth": 50
}
```

---

## 🔗 Next Steps

After loading Level 0 data with the data loader:

1. **Pattern Detection**: Use `agents/pattern_detector.py` to detect patterns
2. **Order Metrics**: Compute nesting, containment, stratified order
3. **Connection**: Calculate Γ, 𝓡, 𝓣 (curvature, torsion)
4. **Level 2**: Spectral analysis, Laplacian, mode extraction

---

**End of README**

