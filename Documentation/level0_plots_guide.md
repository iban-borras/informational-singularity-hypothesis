# Level 0 Plots Guide

Guia completa de les visualitzacions generades per Level 0.

## Comandes ràpides

```powershell
# Regenerar TOTS els gràfics per a una variant
python level0_refresh_plots.py -v B -i 23

# Regenerar només un gràfic específic
python level0_refresh_plots.py -v B -i 23 --only beta

# Regenerar múltiples gràfics
python level0_refresh_plots.py -v B -i 23 --only beta,hilbert,fft

# Amb límit de bits personalitzat
python level0_refresh_plots.py -v B -i 23 --only beta --beta-bits 2000000000
```

---

## Catàleg de gràfics

### 1. `growth` — Corba de creixement

| | |
|---|---|
| **Fitxer** | `growth_curve_{variant}_i{iter}_{timestamp}.png` |
| **Descripció** | Mostra el creixement de bits per iteració (escala log) |
| **Valor científic** | Valida el creixement exponencial γ ≈ 2.9 per variants HSI |
| **Paràmetres** | Cap específic |

---

### 2. `raster` — Visualització 2D

| | |
|---|---|
| **Fitxer** | `raster2d_{variant}_i{iter}_{timestamp}.png` |
| **Descripció** | Seqüència binària desplegada en una graella 2D |
| **Valor científic** | Permet veure patrons visuals, bandes, repeticions |
| **Paràmetres** | `--raster-bits N` (default: 1M) |

---

### 3. `hilbert` — Mapa de Hilbert

| | |
|---|---|
| **Fitxer** | `hilbert_{variant}_i{iter}_{timestamp}.png` |
| **Descripció** | Mapa que preserva localitat espacial (bits propers → píxels propers) |
| **Valor científic** | Revela clusters, estructura fractal, patrons locals |
| **Paràmetres** | `--hilbert-bits N` (default: 1M) |

---

### 4. `fft` — Espectre FFT

| | |
|---|---|
| **Fitxer** | `fft_spectrum_{variant}_i{iter}_{timestamp}.png` |
| **Descripció** | Espectre de potència via FFT (freqüència vs potència) |
| **Valor científic** | Detecta periodicitats, pics espectrals |
| **Paràmetres** | `--fft-bits N` (default: 2M) |

---

### 5. `beta` — Spectrum Enhanced (β-fit segmentat) ⭐

| | |
|---|---|
| **Fitxer** | `spectrum_enhanced_{variant}_abs{N}_i{iter}_{timestamp}.png` |
| **Descripció** | Anàlisi espectral avançat amb binning logarítmic i detecció automàtica de breakpoints |
| **Valor científic** | Caracteritza el comportament 1/f^β, detecta transicions d'escala |
| **Paràmetres** | `--beta-bits N` (default: 1B) |

**Panells del gràfic:**
- **Esquerra**: Espectre raw (blau clar) + binned (blau fosc) amb bandes d'error
- **Dreta**: Fits segmentats β₁, β₂... amb breakpoints (línies discontínues)

**Interpretació de β:**
| β | Interpretació |
|---|---------------|
| β ≈ 0 | Soroll blanc (aleatori, sense correlacions) |
| β ≈ 1 | Soroll rosa (1/f, criticitat) |
| β ≈ 2 | Soroll marró (moviment brownià) |
| 0.5 < β < 1.5 | Zona de complexitat estructurada |

**Variables d'entorn avançades:**
```powershell
$env:HSI_ENHANCED_MAX_WINDOWS = "512"   # Finestres Welch (més = més precisió)
$env:HSI_SPECTRUM_BINS = "80"            # Bins logarítmics
```

---

### 6. `autocorr` — Autocorrelació

| | |
|---|---|
| **Fitxer** | `autocorrelation_{variant}_i{iter}_{timestamp}.png` |
| **Descripció** | Funció d'autocorrelació vs lag |
| **Valor científic** | Detecta correlacions de llarg abast, memòria en la seqüència |
| **Paràmetres** | `--autocorr-bits N` |

---

### 7. `entropy` — Block Entropy

| | |
|---|---|
| **Fitxer** | `block_entropy_{variant}_i{iter}_{timestamp}.png` |
| **Descripció** | H(L)/L vs L (entropia per bit en funció de la mida del bloc) |
| **Valor científic** | Mesura complexitat per escala, detecta estructura jeràrquica |
| **Paràmetres** | `--entropy-bits N` |

---

### 8. `report` — Informe JSON enriquit

| | |
|---|---|
| **Fitxer** | `variant_{V}_{N}_{timestamp}_enriched.json` |
| **Descripció** | JSON amb mètriques actualitzades i metadades |
| **Valor científic** | Dades per anàlisi posterior, input per Level 1+ |

---

## Exemples d'ús

### Generar spectrum_enhanced per totes les variants principals

```powershell
foreach ($v in @('B','D','E','F','H')) {
    python level0_refresh_plots.py -v $v -i 23 --only beta --beta-bits 1000000000
}
```

### Comparar variant B vs control A

```powershell
python level0_refresh_plots.py -v B -i 18 --only beta
python level0_refresh_plots.py -v A -i 18 --only beta
```

### Regenerar tot per a publicació

```powershell
python level0_refresh_plots.py -v B -i 23 --hilbert-bits 2000000 --fft-bits 5000000 --beta-bits 2000000000
```

---

## Ubicació dels fitxers

Tots els gràfics es guarden a:
```
results/level0/visualizations/
```

Els informes JSON a:
```
results/level0/reports/
```

