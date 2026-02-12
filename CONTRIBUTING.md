# 🤝 Contributing to the ISH Agents Project

Thank you for your interest in contributing to the **ISH Agents** project! This repository is part of an open scientific effort to explore the emergence of order from pure information, based on the *Informational Singularity Hypothesis (ISH)*.

Whether you come from physics, computer science, AI, mathematics, philosophy, or the arts, your input is valuable. This guide will help you get started.

---

## ✨ Types of Contributions Welcome

### Core Algorithm Improvements
* Improvements to the Level 0 collapse engine (streaming, compression)
* New collapse variants aligned with HSI principles (see `Documentation/variants_spec.md`)
* Performance optimizations (Numba JIT — see existing examples in `agents/numba_kernels.py`)

### Metrics & Analysis
* Extensions to the five core metrics: multi-scale Lempel-Ziv (LZ), RQA determinism (DET), cubical TDA persistence (β₁/β₀), nesting-tree mean children (c̄), and multi-scale Transfer Entropy (TE)
* New statistical validation methods or controls
* Commutativity and divergence analysis (see `appendix_*.py` scripts)

### Agent Development
* New pattern detectors (Pₖ) for structural analysis
* Algorithms to infer emergent rules (ωₖ)
* Validation agents for hypothesis testing

### Visualization & Tools
* Hilbert curve and 2D/3D visualizations of emergent structures
* Improvements to existing plots (`level0_refresh_plots.py`, `level1_visualize.py`)
* Tools for comparing variants across metrics

### Documentation & Outreach
* Translations of documentation
* Tutorials, Jupyter notebooks, and usage examples
* Proposals to validate HSI in scientific or artistic contexts

---

## 🚀 Getting Started

1. **Fork** this repository
2. **Set up environment** (choose one):
   - **Windows**: Run `.\Setup-virtual-envelop.ps1`
   - **Docker**: `docker-compose run hsi -v B -i 15` (see README for details)
   - **Manual**: `python -m venv venv && pip install -r requirements.txt`
3. Create a new branch with a clear name: `feature/fractal-detector` or `fix/phi-metric`
4. Make clear, well-documented changes
5. Run existing tests (`python -m pytest tests/`) to ensure nothing breaks
6. Submit a pull request with a description of your proposal

---

## 🧪 Best Practices

* Follow the style of the existing codebase (`snake_case`, docstrings, modularity)
* Use type hints for function signatures
* Accompany contributions with sample data or basic tests if possible
* If you propose a new metric or agent, include your rationale or a short explanation
* For performance-critical code, consider Numba JIT optimization (see `agents/numba_kernels.py` and existing `@njit` functions)
* All 12 variants (7 HSI + 5 controls) should be considered when validating changes

---

## 💡 Suggested First Contributions

**Beginner-friendly:**
* Add a new visualization mode to `utils/visualization.py`
* Improve docstrings or add usage examples to any script
* Create a Jupyter notebook tutorial for running a basic Level 0 → Level 1 pipeline

**Intermediate:**
* Implement a new pattern detector in `agents/`
* Extend `level1_deep_analysis.py` with a new analysis mode
* Add a new control variant with scientific justification

**Advanced:**
* Extend the Transfer Entropy analysis (`level2_transfer_entropy.py`) with new causal metrics
* Implement a new collapse variant and validate it against the five core metrics
* Contribute to the topological or geometric analysis pipeline (`level1_topological.py`, `level1_geometric.py`)

---

## 🧬 License & Philosophy

This project is released under the MIT License and is grounded in the values of **open research, collaborative creativity, and transdisciplinary exploration**.

All contributors will be acknowledged and cited in any derived scientific publication.

---

## 🙋‍♀️ Contact

* **GitHub Issues**: For bugs, feature requests, and technical discussions
* **Project Creator**: Iban Borràs

Thank you for joining this journey from Nothingness to emergent order. 🌀
