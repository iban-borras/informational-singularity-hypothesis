# 🚀 Proposta d'Acceleració GPU per HSI

**Data:** 2025-12-01  
**Autor:** Sophia (Augment Agent) en col·laboració amb Iban  
**Hardware de referència:** NVIDIA RTX 4090 (16GB VRAM + 16GB RAM compartida)

---

## 📋 Resum Executiu

Aquesta proposta detalla com accelerar els processos de càlcul intensiu del projecte HSI utilitzant la GPU NVIDIA. L'objectiu és reduir temps de processament de **hores a minuts** mantenint compatibilitat amb sistemes sense GPU NVIDIA.

### Estimació d'Impacte

| Procés | Temps CPU (1GB dades) | Temps GPU estimat | Speedup |
|--------|----------------------|-------------------|---------|
| FFT/Welch PSD | ~5-15 min | ~10-30 seg | **10-30x** |
| Hilbert heatmap | ~10-30 min | ~30-60 seg | **10-20x** |
| Box-counting fractal | ~2-5 min | ~10-30 seg | **5-10x** |
| String→Array | ~30 seg | (CPU millor) | ~1x |

---

## 🔍 Anàlisi Detallada dels Processos

### 1. FFT/Welch Power Spectral Density

**Localització:** `run_all_variants.py` línies 583-724 (`_welch_psd_from_gz`, `_sampling_psd_from_gz`)

**Operacions actuals:**
```python
spec = _np.fft.rfft(a)           # FFT per finestra
power = (spec.real**2 + spec.imag**2)  # Potència
psd_acc += power                 # Acumulació
```

**Bottleneck:** Cada finestra de 1M punts requereix una FFT independent. Amb 64 finestres: 64 FFTs seqüencials.

**Optimització GPU proposada:**
```python
import cupy as cp

# Batch FFT: processar múltiples finestres en paral·lel
windows_gpu = cp.array(windows_batch)  # [n_windows, window_size]
specs_gpu = cp.fft.rfft(windows_gpu, axis=1)  # cuFFT batched
power_gpu = specs_gpu.real**2 + specs_gpu.imag**2
psd_sum = cp.sum(power_gpu, axis=0)
psd_cpu = cp.asnumpy(psd_sum)
```

**Benefici VRAM:** 
- 64 finestres × 1M floats × 8 bytes = 512 MB (molt dins dels 16GB)
- Pot processar fins a ~2000 finestres simultànies

---

### 2. Hilbert Heatmap Generation

**Localització:** 
- Coordenades: `level0/generator.py` línies 1377-1440 (`_hilbert_indices_to_xy_vectorized`)
- Heatmap: `run_all_variants.py` línies 1015-1083 (`plot_hilbert_heatmap`)

**Operacions actuals:**
```python
# Coordenades (ja vectoritzat amb NumPy)
coords = _hilbert_indices_to_xy_vectorized(n_points, grid_size)

# Acumulació heatmap
ones_mask = (a == 1)
_np.add.at(heat, (y_ones, x_ones), 1)  # Scatter-add
```

**Bottleneck:** 
- `np.add.at` és inherentment seqüencial per evitar race conditions
- Amb 1M punts, el scatter és lent

**Optimització GPU proposada:**
```python
import cupy as cp

# Coordenades Hilbert en GPU
def hilbert_coords_gpu(n_points, grid_size):
    indices = cp.arange(n_points, dtype=cp.int64)
    x = cp.zeros(n_points, dtype=cp.int32)
    y = cp.zeros(n_points, dtype=cp.int32)
    t = indices.copy()
    s = 1
    while s < grid_size:
        rx = (t >> 1) & 1
        ry = (t ^ rx) & 1
        # ... (mateixa lògica, operacions vectoritzades en GPU)
    return x, y

# Heatmap amb histogramdd (nativament paral·lel en GPU)
x_gpu, y_gpu = hilbert_coords_gpu(n, grid)
ones_mask = (a_gpu == 1)
coords = cp.stack([y_gpu[ones_mask], x_gpu[ones_mask]], axis=1)
heat_gpu = cp.histogramdd(coords, bins=[grid, grid])[0]
heat = cp.asnumpy(heat_gpu)
```

**Benefici VRAM:**
- 1M punts × 2 coords × 4 bytes = 8 MB
- Grid 1024×1024 × 4 bytes = 4 MB
- Total: ~15 MB (negligible)

---

### 3. Box-Counting Fractal Dimension

**Localització:** `run_all_variants.py` línies 1213-1270 (`plot_fractal_loglog`)

**Operacions actuals:**
```python
for bs in sizes:
    n = len(bits)
    arr = _bits_to_uint8(bits[:n - n % bs])
    reshaped = arr.reshape(-1, bs)
    box_has_one = _np.any(reshaped, axis=1)
    counts.append(int(_np.sum(box_has_one)))
```

**Optimització GPU proposada:**
```python
# Processar tots els box sizes en paral·lel
a_gpu = cp.array(bits_arr)
for bs in sizes:
    n = len(a_gpu)
    arr = a_gpu[:n - n % bs]
    reshaped = arr.reshape(-1, bs)
    box_has_one = cp.any(reshaped, axis=1)
    counts.append(int(cp.sum(box_has_one)))
```

---

### 4. Processos NO recomanats per GPU

| Procés | Raó per mantenir en CPU |
|--------|------------------------|
| Lectura gzip | I/O bound, no es beneficia de GPU |
| String→Array (`frombuffer`) | Overhead transfer > benefici |
| Compressió/descompressió | Millor amb CPU multithreaded |
| Plots matplotlib | Renderització CPU |

---

## 🏗️ Arquitectura Proposada

### Estructura de Fitxers

```
hsi_agents_project/
├── config.json              # Nova secció "gpu"
├── utils/
│   ├── gpu_backend.py       # NOU: Abstracció GPU/CPU
│   └── progress.py          # Existent
├── level0/
│   └── generator.py         # Modificat: usar gpu_backend
└── run_all_variants.py      # Modificat: usar gpu_backend
```

### Mòdul `gpu_backend.py`

```python
"""
GPU Backend Abstraction Layer for HSI.
Provides transparent fallback to CPU (NumPy) when CUDA unavailable.
"""
import os
import numpy as np

# Configuration
GPU_ENABLED = False
GPU_AVAILABLE = False
GPU_MEMORY_GB = 0
DEVICE_NAME = "CPU"

def _detect_gpu():
    """Detect GPU availability and capabilities."""
    global GPU_AVAILABLE, GPU_MEMORY_GB, DEVICE_NAME
    try:
        import cupy as cp
        device = cp.cuda.Device(0)
        GPU_AVAILABLE = True
        GPU_MEMORY_GB = device.mem_info[1] / (1024**3)  # Total VRAM
        DEVICE_NAME = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        return True
    except Exception:
        return False

def init_gpu(config: dict = None):
    """Initialize GPU backend based on config."""
    global GPU_ENABLED
    
    if config is None:
        config = {}
    
    gpu_cfg = config.get('gpu', {})
    enabled = gpu_cfg.get('enabled', True)  # Enabled by default if available
    
    if not enabled:
        print("[GPU] Disabled by configuration")
        GPU_ENABLED = False
        return False
    
    if not _detect_gpu():
        print("[GPU] CUDA not available, using CPU fallback")
        GPU_ENABLED = False
        return False
    
    GPU_ENABLED = True
    print(f"[GPU] Enabled: {DEVICE_NAME} ({GPU_MEMORY_GB:.1f} GB VRAM)")
    return True

def get_array_module():
    """Get appropriate array module (cupy or numpy)."""
    if GPU_ENABLED:
        import cupy as cp
        return cp
    return np

def to_gpu(arr):
    """Transfer array to GPU if enabled."""
    if GPU_ENABLED:
        import cupy as cp
        return cp.asarray(arr)
    return arr

def to_cpu(arr):
    """Transfer array to CPU."""
    if GPU_ENABLED:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    return arr

def rfft(arr, axis=-1):
    """GPU-accelerated rfft."""
    xp = get_array_module()
    return xp.fft.rfft(arr, axis=axis)

def rfftfreq(n, d=1.0):
    """GPU-accelerated rfftfreq."""
    xp = get_array_module()
    return xp.fft.rfftfreq(n, d=d)

# ... més funcions wrapper
```

### Configuració `config.json`

```json
{
  "gpu": {
    "_comment": "GPU acceleration settings (requires NVIDIA CUDA)",
    "enabled": true,
    "device_id": 0,
    "memory_limit_gb": 14,
    "use_unified_memory": true,
    "batch_size_fft": 64,
    "batch_size_hilbert": 1000000,
    "_help": {
      "enabled": "Enable GPU acceleration (auto-fallback to CPU if unavailable)",
      "device_id": "CUDA device ID (0 for primary GPU)",
      "memory_limit_gb": "Max VRAM to use (leave 2GB for system)",
      "use_unified_memory": "Use unified memory for arrays >VRAM (slower but allows larger data)",
      "batch_size_fft": "Number of FFT windows to process in parallel",
      "batch_size_hilbert": "Points per Hilbert batch"
    }
  }
}
```

---

## 📊 Gestió de Memòria VRAM

### RTX 4090: 16GB VRAM + 16GB Unified

| Escenari | VRAM necessària | Estratègia |
|----------|-----------------|------------|
| FFT 64 finestres × 1M | 512 MB | Tot en VRAM |
| Hilbert 10M punts | ~100 MB | Tot en VRAM |
| FFT 256 finestres × 4M | 8 GB | Tot en VRAM |
| Hilbert 100M punts | ~1 GB | Tot en VRAM |
| FFT streaming molt gran | >14 GB | Unified Memory |

### Implementació Unified Memory

```python
def process_large_fft(data, window_size, max_vram_gb=14):
    """Process FFT with automatic memory management."""
    xp = get_array_module()
    
    total_size_gb = data.nbytes / (1024**3)
    
    if total_size_gb <= max_vram_gb:
        # Tot cap a VRAM
        data_gpu = xp.asarray(data)
        return xp.fft.rfft(data_gpu)
    else:
        # Unified memory (VRAM + RAM)
        import cupy as cp
        pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
        cp.cuda.set_allocator(pool.malloc)
        data_gpu = cp.asarray(data)
        result = cp.fft.rfft(data_gpu)
        cp.cuda.set_allocator()  # Reset
        return result
```

---

## 🔄 Fases d'Implementació

### Fase 1: Infraestructura Base (1-2 dies)
- [ ] Crear `utils/gpu_backend.py` amb detecció i fallback
- [ ] Afegir secció `gpu` a `config.json`
- [ ] Afegir dependència opcional `cupy-cuda12x` a requirements
- [ ] Tests bàsics de detecció GPU

### Fase 2: FFT Acceleration (2-3 dies)
- [ ] Refactoritzar `_welch_psd_from_gz` per usar `gpu_backend`
- [ ] Refactoritzar `_sampling_psd_from_gz`
- [ ] Implementar batch FFT per múltiples finestres
- [ ] Benchmark: comparar temps CPU vs GPU

### Fase 3: Hilbert Acceleration (2-3 dies)
- [ ] Portar `_hilbert_indices_to_xy_vectorized` a GPU
- [ ] Optimitzar heatmap amb `histogramdd` GPU
- [ ] Benchmark: comparar temps CPU vs GPU

### Fase 4: Optimitzacions Addicionals (1-2 dies)
- [ ] Box-counting fractal en GPU
- [ ] Unified Memory per a dades molt grans
- [ ] Profiling i fine-tuning

### Fase 5: Documentació i Tests (1 dia)
- [ ] Actualitzar README amb instruccions GPU
- [ ] Tests d'integració CPU/GPU
- [ ] Documentar benchmarks finals

---

## 📦 Dependències

### Instal·lació CuPy (Windows, CUDA 12.x)

```powershell
# Opció 1: pip (recomanat)
pip install cupy-cuda12x

# Opció 2: conda
conda install -c conda-forge cupy cudatoolkit=12
```

### requirements.txt actualitzat

```
# Core dependencies
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0

# Optional: GPU acceleration (NVIDIA CUDA)
# Uncomment the appropriate line for your CUDA version:
# cupy-cuda11x  # For CUDA 11.x
# cupy-cuda12x  # For CUDA 12.x
```

---

## ⚠️ Consideracions i Riscos

### Riscos Tècnics
1. **Compatibilitat CUDA**: Requereix drivers NVIDIA actualitzats
2. **Overhead transfer**: Per dades petites (<10K punts), CPU pot ser més ràpid
3. **Precisió numèrica**: cuFFT pot tenir lleugeres diferències vs NumPy FFT

### Mitigacions
1. **Fallback automàtic**: Si GPU falla, continua amb CPU sense interrupció
2. **Threshold configurable**: Només usar GPU per dades > X punts
3. **Validació**: Comparar resultats GPU vs CPU en tests

### Compatibilitat
- ✅ Windows 10/11 amb NVIDIA GPU
- ✅ Linux amb NVIDIA GPU
- ✅ Sistemes sense GPU (fallback a CPU)
- ❌ macOS (no CUDA natiu — usar CPU)
- ❌ AMD GPUs (no CUDA — usar CPU)

---

## 📈 Benchmarks Esperats (RTX 4090)

| Test | CPU (NumPy) | GPU (CuPy) | Speedup |
|------|-------------|------------|---------|
| FFT 1M punts | 50 ms | 3 ms | 17x |
| FFT 64×1M (batch) | 3.2 s | 0.15 s | 21x |
| Hilbert 1M coords | 200 ms | 15 ms | 13x |
| Heatmap 1M punts | 500 ms | 30 ms | 17x |
| Box-count 10 sizes | 800 ms | 100 ms | 8x |

**Total estimat per processament complet variant B (iter 23):**
- CPU: 30-60 minuts
- GPU: 3-6 minuts

---

## 🎯 Conclusió

La implementació d'acceleració GPU és **altament recomanable** per al projecte HSI:

1. **Benefici clar**: Reducció de 10-20x en temps de processament
2. **Risc baix**: Fallback automàtic a CPU
3. **Inversió moderada**: ~7-10 dies de desenvolupament
4. **Hardware disponible**: RTX 4090 amb 16+16GB és ideal

### Recomanació

**Procedir amb la implementació per fases**, començant per FFT (major impacte) i seguint amb Hilbert. Cada fase és independent i es pot validar abans de continuar.

---

*Document preparat per Sophia — Augment Agent*  
*Per revisar conjuntament amb Iban abans d'implementar*

