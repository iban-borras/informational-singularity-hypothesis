#!/usr/bin/env python3
"""
HSI Variants Master Runner (v33)
Runs the 6 surviving collapse variants (B, D, E, F, G, H) and compares results.

v33 Update: Variants A and C eliminated for incompatibility with HSI v32 principles.
- Variant A: Violated stratified order (fundamental to v32)
- Variant C: Violated micro-state reality (ontological commitment of v32)

This script orchestrates experiments to study emergent order patterns
across different collapse strategies aligned with HSI v32 paradigm.

Author: Iban Borràs with collaboration from Augment Agent (Sophia)
Date: Nov 2025
Objective: Detect emergent order through stratified containment
"""

import subprocess
import sys
import json
import time
import os
import math
from datetime import datetime
import numpy as np
from pathlib import Path

# Ensure hsi_agents_project is importable when running script directly
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Progress indicator utilities
try:
    from hsi_agents_project.utils.progress import ProgressIndicator, progress_print, progress_done
    HAVE_PROGRESS = True
except ImportError:
    HAVE_PROGRESS = False
    def progress_print(msg, **kw): print(msg, flush=True)
    def progress_done(msg): print(msg, flush=True)
    class ProgressIndicator:
        def __init__(self, name, total=None): self.name = name
        def __enter__(self): print(f"[⏳ {self.name}] Starting...", flush=True); return self
        def __exit__(self, *a): print(f"[✓ {self.name}] Done", flush=True)
        def update(self, *a, **kw): pass
        def tick(self, *a): pass
try:
    from hsi_agents_project.main import load_config as load_project_config
except Exception:
    def load_project_config(_path: str = "config.json"):
        try:
            import json as _json
            from pathlib import Path as _P
            p = _P(_path)
            if p.exists():
                with open(p, 'r', encoding='utf-8') as f:
                    return _json.load(f)
        except Exception:
            pass
        return {}

# Anchor outputs under project root (this file lives in hsi_agents_project/)
BASE_PATH = Path(__file__).resolve().parent
ROOT_PATH = BASE_PATH.parent  # repository root; parent of the package dir
RESULTS_DIR = BASE_PATH / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Use unified structure: level0/visualizations
VIS_DIR = RESULTS_DIR / "level0" / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

# Fallback to old location if it exists and has files (backwards compatibility)
_old_vis = RESULTS_DIR / "visualizations"
if _old_vis.exists() and any(_old_vis.iterdir()):
    VIS_DIR = _old_vis


# Optional plotting: guard import for environments without matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception as e:
    print(f"[WARN] matplotlib not available or failed to import: {e}")
    print("       Plots will be skipped. Install matplotlib to enable charts.")
    MATPLOTLIB_OK = False

# Lazy import Level0 utilities (Hilbert mapping)
try:
    from hsi_agents_project.level0.generator import phi_to_hilbert_coords
    HAVE_L0_UTILS = True
except Exception:
    try:
        # Fallback when running as a script inside the package dir
        from level0.generator import phi_to_hilbert_coords  # type: ignore
        HAVE_L0_UTILS = True
    except Exception:
        HAVE_L0_UTILS = False


# ============================================================================
# VECTORIZED CONVERSION HELPERS (~100x faster than fromiter with generator)
# ============================================================================
def _bits_to_uint8(bits: str) -> np.ndarray:
    """Convert '01101...' string to uint8 array (0/1). ~100x faster than fromiter."""
    return (np.frombuffer(bits.encode('ascii'), dtype='S1') == b'1').astype(np.uint8)

def _bits_to_float64(bits: str) -> np.ndarray:
    """Convert '01101...' string to float64 array (0.0/1.0). ~100x faster than fromiter."""
    return (np.frombuffer(bits.encode('ascii'), dtype='S1') == b'1').astype(np.float64)


def _load_dotenv_if_present(path: str = ".env"):
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"): continue
                if "=" not in s: continue
                k, v = s.split("=", 1)
                k, v = k.strip(), v.strip().strip('"')
                # do not override existing
                if k not in os.environ:
                    os.environ[k] = v
    except Exception as e:
        print(f"[WARN] Failed to load .env: {e}")

def run_variant_script(python_flag, module_or_script, variant_name, results_file, variant_code, iterations: int | None):
    """Run one generator variant (A–F) and capture results with live progress."""
    print(f"\n🚀 Running {variant_name}...")
    print("=" * 50)

    start_time = time.time()

    try:
        # Execute generator as a module, pass variant via env var (portable)
        env = dict(**os.environ)
        env["HSI_VARIANT_CODE"] = variant_code
        if iterations is not None:
            env["HSI_ITERATIONS"] = str(iterations)
        env["HSI_LOG_EVERY"] = env.get("HSI_LOG_EVERY", "1")  # show per-iter logs during runner
        # Force UTF-8 I/O on Windows consoles to avoid UnicodeEncodeError
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = env.get("PYTHONIOENCODING", "utf-8")
        # Disable Python output buffering for real-time progress display
        env["PYTHONUNBUFFERED"] = "1"

        # Stream output live for progress (no capture_output)
        # Resolve correct Python executable inside our venv when available
        py_exe = sys.executable or "python"
        # Ensure the project root is on PYTHONPATH so hsi_agents_project is importable
        env["PYTHONPATH"] = os.pathsep.join([
            str(ROOT_PATH),  # parent of hsi_agents_project, so package is importable
            env.get("PYTHONPATH", "")
        ])
        proc = subprocess.Popen(
            [py_exe, python_flag, module_or_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=str(ROOT_PATH)
        )

        live_output_lines = []
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                live_output_lines.append(line)
                # Show progress lines and key info
                stripped = line.lstrip()
                if (stripped.startswith("[iter") or
                    stripped.startswith("[loop]") or
                    stripped.startswith("[post]") or
                    stripped.startswith("[snapshot]") or
                    stripped.startswith("[INFO]") or
                    stripped.startswith("[variant_A]") or
                    "Config:" in line or
                    "Compression" in line or
                    "Saved variant" in line or
                    "Generated" in line or
                    "✅" in line or
                    "%" in line):
                    print(line.rstrip())
        proc.wait(timeout=600)
        execution_time = time.time() - start_time

        full_output = "".join(live_output_lines)
        if proc.returncode == 0:
            print(f"✅ {variant_name} finished in {execution_time:.2f}s")
            return True, execution_time, full_output
        else:
            print(f"❌ Error in {variant_name}:")
            print(full_output)
            return False, execution_time, full_output

    except subprocess.TimeoutExpired:
        print(f"⏰ {variant_name} exceeded time limit")
        return False, 300, "Timeout"
    except Exception as e:
        print(f"💥 Exception in {variant_name}: {e}")
        return False, 0, str(e)

def load_variant_results(filename):
    """Carrega els resultats d'una variant."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ File not found: {filename}")
        return None
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None


def find_variant_result_file(variant_code: str, iterations: int | None = None) -> str:
    """Find the JSON report for a given variant.
    If 'iterations' is provided, try to find a report matching that iteration;
    if not found, fall back to the latest report for the variant.
    """
    reports_dir = BASE_PATH / "results" / "level0" / "reports"
    try:
        if iterations is not None:
            # First try exact iteration match
            paths = sorted(reports_dir.glob(f"variant_{variant_code}_{iterations}_*.json"))
            if paths:
                return str(paths[-1])
            # Fallback: use latest report available for this variant
            print(f"   [info] No report for {variant_code} iter {iterations}, using latest available...", flush=True)
            paths = sorted(reports_dir.glob(f"variant_{variant_code}_*_*.json"))
        else:
            paths = sorted(reports_dir.glob(f"variant_{variant_code}_*_*.json"))
        return str(paths[-1]) if paths else str(reports_dir / f"variant_{variant_code}_NOT_FOUND.json")
    except Exception:
        return str(reports_dir / f"variant_{variant_code}_NOT_FOUND.json")

def compare_phi_convergence(results_list):
    """Compare convergence to φ across variants (if metrics available).
    Uses composite φ-alignment score if available; falls back to simple |D-φ|.
    """
    print("\n" + "*" * 20)
    print("CONVERGENCE COMPARISON TOWARD φ (1.618)")
    print("*" * 20)

    convergence_data = []

    for results in results_list:
        if results:
            variant = results.get('variant', 'unknown')
            # Prefer composite score if present
            phi_align_score = results.get('phi_alignment_score')
            phi_alignment = results.get('phi_alignment')
            fractal_dim = results.get('fractal_dimension')
            total_bits = results.get('total_bits')
            quality = results.get('convergence_quality')

            if (phi_align_score is None and phi_alignment is None) or fractal_dim is None or total_bits is None:
                print(f"[WARN] Variant {variant}: incomplete metrics; skipping in comparison.")
                continue

            score = phi_align_score if phi_align_score is not None else phi_alignment
            convergence_data.append({
                'variant': variant,
                'phi_alignment': phi_alignment,
                'phi_alignment_score': score,
                'fractal_dimension': fractal_dim,
                'total_bits': total_bits,
                'quality': quality or 'n/a'
            })

            # Console summary per variant in compare mode
            status = "*" if (quality == "excellent") else "+" if (quality == "good") else "-"
            try:
                s = f" | composite: {phi_align_score:.6f}" if phi_align_score is not None else ""
                print(f"{status} {variant:<25} | φ-align: {phi_alignment:.6f}{s} | Fractal: {fractal_dim:.6f} | Bits: {total_bits:,}")
            except Exception:
                pass

    if convergence_data:
        best_variant = min(convergence_data, key=lambda x: x['phi_alignment_score'])
        print(f"\nBEST CONVERGENCE: {best_variant['variant']}")
        print(f"   φ‑Alignment (composite): {best_variant['phi_alignment_score']:.6f}")
        print(f"   Fractal dimension: {best_variant['fractal_dimension']:.6f}")
        if best_variant['phi_alignment_score'] < 0.01:
            print("POTENTIAL DISCOVERY: Excellent convergence toward φ!")
        return best_variant

    return None


# Removed: _phi_alignment_weights() - no longer needed in v32


def _growth_and_ratio_metrics(result_obj: dict) -> dict:
    """Calculate growth metrics (exponent, R²) without phi-alignment."""
    import numpy as _np
    per_iter = result_obj.get('per_iteration') or []
    xs = [e.get('iteration') for e in per_iter]
    ys = [e.get('phi_length') for e in per_iter]
    out = {'growth_exponent': None, 'growth_r2': None, 'ratio_median': None}
    if len(xs) < 3 or any(y is None or y <= 0 for y in ys):
        return out
    # Exponential growth fit: log(bits) ~ a + gamma * i
    x = _np.array(xs, dtype=float)
    y = _np.array(ys, dtype=float)
    ylog = _np.log(_np.maximum(1.0, y))
    slope, intercept = _np.polyfit(x, ylog, 1)
    yhat = slope * x + intercept
    ss_res = float(_np.sum((ylog - yhat)**2))
    ss_tot = float(_np.sum((ylog - _np.mean(ylog))**2)) or 1.0
    r2 = 1.0 - ss_res/ss_tot
    # Derivative ratios r_i = d_{i}/d_{i-1}
    dy = _np.diff(y)
    dy = dy[dy > 0]
    if dy.size >= 2:
        ri = dy[1:] / _np.maximum(1.0, dy[:-1])
        # robust: use latter half to avoid transients
        k0 = max(0, int(0.5*ri.size))
        ri_sel = ri[k0:]
        r_med = float(_np.median(ri_sel)) if ri_sel.size else float(_np.median(ri))
        out.update({'ratio_median': r_med})
    out.update({'growth_exponent': float(slope), 'growth_r2': float(r2)})
    return out


def _attach_growth_metrics(variant_results: list, beta_stats: list | None):
    """Attach growth and spectral metrics to variant results (v32: no phi-alignment)."""
    for i, r in enumerate(variant_results):
        if not r:
            continue
        # Growth metrics
        gr = _growth_and_ratio_metrics(r)
        r.update(gr)
        # Beta from spectral analysis (if available)
        if beta_stats and i < len(beta_stats) and beta_stats[i] and beta_stats[i].get('valid') and (beta_stats[i].get('beta') is not None):
            beta = float(beta_stats[i]['beta'])
            d_beta = (3.0 - beta)/2.0
            r['beta'] = beta
            r['fractal_from_beta'] = d_beta
        # Print per-variant summary line
        try:
            print(f"[metrics] {r.get('variant','?'):<24} | γ={r.get('growth_exponent'):.4f} R²={r.get('growth_r2'):.3f} | r_med={r.get('ratio_median')} | D_fractal={r.get('fractal_dimension')} | β={r.get('beta')} Dβ={r.get('fractal_from_beta')}")
        except Exception:
            pass


def plot_growth_and_time(results_list):
    if not MATPLOTLIB_OK:
        print("[WARN] Skipping growth/time plots because matplotlib is not available.")
        return
    # Skip if no valid results
    valid_results = [r for r in results_list if r]
    if not valid_results:
        print("[WARN] Skipping growth/time plots: no valid results.")
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes
    variants = []
    for r in valid_results:
        variant = r.get('variant','?')
        variants.append(variant)
        per_iter = r.get('per_iteration', [])
        xs = [e.get('iteration') for e in per_iter]
        ys_bits = [e.get('phi_length') for e in per_iter]
        ys_time = [e.get('time_sec') for e in per_iter if e.get('time_sec') is not None]
        ax1.plot(xs, ys_bits, label=variant)
        if ys_time:
            ax2.plot(xs[:len(ys_time)], ys_time, label=variant)
    ax1.set_title('Growth curve (bits vs iteration)')
    ax1.set_xlabel('Iteration'); ax1.set_ylabel('Bits'); ax1.set_yscale('log'); ax1.legend()
    ax2.set_title('Time per iteration (s)'); ax2.set_xlabel('Iteration'); ax2.set_ylabel('Seconds'); ax2.legend()
    _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        uniq = sorted(set(v for v in variants if v and v != '?'))
        var_tag = (uniq[0] if len(uniq) == 1 else (''.join(uniq) if uniq else 'UNK'))
    except Exception:
        var_tag = 'UNK'
    max_iter = max((r.get('iterations') or 0) for r in valid_results) or 0
    out = VIS_DIR / f"growth_time_{var_tag}{_abs_suffix()}_i{max_iter}_{_ts}.png"
    rel = out.relative_to(BASE_PATH)
    plt.tight_layout(); plt.savefig(out); print(f"[INFO] Growth/Time saved: {rel}"); plt.close()



def _extract_variant_metrics(r: dict) -> dict:
    return {
        'variant': r.get('variant'),
        'iterations': r.get('iterations'),
        'total_bits': r.get('total_bits'),
        'fractal_dimension': r.get('fractal_dimension'),
        'phi_alignment': r.get('phi_alignment'),
        'growth_exponent': r.get('growth_exponent'),
        'growth_r2': r.get('growth_r2'),
        'ratio_median': r.get('ratio_median'),
        'phi_ratio_error': r.get('phi_ratio_error'),
        'beta': r.get('beta'),
        'fractal_from_beta': r.get('fractal_from_beta'),
        'phi_beta_error': r.get('phi_beta_error'),
        'phi_alignment_score': r.get('phi_alignment_score'),
        'report_path': r.get('report_path')
    }


def _write_enriched_variant_report(r: dict):
    try:
        rp = r.get('report_path')
        if not rp:
            return
        from pathlib import Path as _P
        p = _P(rp)
        out_path = p.with_name(p.stem + '.enriched.json')
        # Keep original metrics + enriched fields, drop huge payloads if any
        slim = _extract_variant_metrics(r)
        slim['timestamp'] = datetime.now().isoformat()
        with open(out_path, 'w') as f:
            json.dump(slim, f, indent=2)
        try:
            rel = out_path.resolve().relative_to(BASE_PATH)
            print(f"[INFO] Wrote enriched report: {rel}")
        except Exception:
            print(f"[INFO] Wrote enriched report: {out_path}")
    except Exception as e:
        print(f"[WARN] Could not write enriched report: {e}")



def _get_abs_mode() -> str | None:
    try:
        v = os.environ.get("HSI_ABSOLUTE_TOKEN")
        return v if v in ("1","10","01") else None
    except Exception:
        return None

def _abs_suffix() -> str:
    a = _get_abs_mode()
    return f"_abs{a}" if a else ""

def _project_snapshots_dir(var: str | None = None) -> str:
    # Match Level0 generator snapshot dir structure, include ABS if provided
    # Fallback to var_{VARIANT} if var_{VARIANT}_abs{mode} doesn't exist
    base = BASE_PATH / "results" / "level0" / "phi_snapshots"
    if var:
        abs_m = _get_abs_mode()
        if abs_m:
            abs_path = base / f"var_{var.upper()}_abs{abs_m}"
            if abs_path.exists():
                return str(abs_path)
            # Fallback: try without abs suffix
        plain_path = base / f"var_{var.upper()}"
        if plain_path.exists():
            return str(plain_path)
        # If neither exists, return the expected path (with abs if set)
        if abs_m:
            return str(base / f"var_{var.upper()}_abs{abs_m}")
        return str(plain_path)
    return str(base)

def _get_spectral_config():
    cfg = load_project_config()
    out = {
        'method': 'auto',
        'window_size': 1_048_576,  # 1 Mi samples
        'overlap': 0.5,           # 50% overlap
        'max_windows': 64,        # cap processing time
        'progress': True,
        'one_sided': True,
    }
    try:
        # Legacy keys inside output{}
        oc = (cfg.get('output') or {})
        sm = (oc.get('spectral_method') or out['method']).lower()
        if sm in ('welch', 'prefix', 'sampling', 'auto'):
            out['method'] = sm
        ws = int(oc.get('fft_window_size')) if oc.get('fft_window_size') is not None else out['window_size']
        out['window_size'] = max(1024, ws)
        ov = oc.get('fft_overlap')
        if ov is not None:
            try:
                ov = float(ov)
                out['overlap'] = min(0.9, max(0.0, ov))
            except Exception:
                pass
        mw = int(oc.get('fft_max_windows')) if oc.get('fft_max_windows') is not None else out['max_windows']
        out['max_windows'] = max(1, mw)
        pr = oc.get('progress_logging')
        if isinstance(pr, bool):
            out['progress'] = pr
        # New preferred section fft{}
        f = (cfg.get('fft') or {})
        if f.get('method'): out['method'] = str(f['method']).lower()
        if f.get('window_size') is not None: out['window_size'] = max(1024, int(f['window_size']))
        if f.get('overlap') is not None:
            try:
                out['overlap'] = min(0.9, max(0.0, float(f['overlap'])))
            except Exception:
                pass
        if f.get('max_windows') is not None: out['max_windows'] = max(1, int(f['max_windows']))
        if f.get('one_sided') is not None: out['one_sided'] = bool(f['one_sided'])
        # Pull absolute_token from config to env if unset
        abs_cfg = str(oc.get('absolute_token')) if oc.get('absolute_token') is not None else None
        if abs_cfg in ('1','10','01') and not os.environ.get('HSI_ABSOLUTE_TOKEN'):
            os.environ['HSI_ABSOLUTE_TOKEN'] = abs_cfg
    except Exception:
        pass
    return out


def _get_hilbert_config():
    cfg = load_project_config()
    out = {
        'max_bits': 1_000_000,
        'target_grid': 1024
    }
    try:
        h = (cfg.get('hilbert') or {})
        if h.get('max_bits') is not None:
            out['max_bits'] = max(1, int(h['max_bits']))
        if h.get('target_grid') is not None:
            out['target_grid'] = max(64, int(h['target_grid']))
    except Exception:
        pass
    return out


def _get_raster_iter_cap():
    try:
        oc = (load_project_config().get('output') or {})
        v = oc.get('raster_iter_cap')
        return int(v) if v is not None else 20
    except Exception:
        return 20



def _bits_for_iteration(results: dict, target_iter: int, max_bits: int) -> str:
    # Load optional raster limit from config
    cfg = load_project_config()
    raster_bits_limit = None
    try:
        raster_bits_limit = int((cfg.get('output') or {}).get('raster_bits_limit'))
    except Exception:
        raster_bits_limit = None

    # Try snapshot for exact iteration first
    # Prefer variant-specific snapshot path
    var = (results.get('variant') or '').upper() or None
    s = _stream_phi_prefix_from_gz(target_iter, max_bits, var=var)
    if s:
        # Apply config limit on snapshot path too
        if raster_bits_limit:
            return s[:min(max_bits, raster_bits_limit)]
        return s[:max_bits]
    # Fallback to final file using the exact length at that iteration (from report)
    per_iter = results.get('per_iteration') or []
    if 1 <= target_iter <= len(per_iter):
        it_bits = int(per_iter[target_iter-1].get('phi_length') or 0)
        need = min(max_bits, it_bits)
    else:
        need = max_bits
    # Apply config limit if present
    if raster_bits_limit:
        need = min(need, raster_bits_limit)
    # Stream from .struct.gz files
    var = results.get('variant') or None
    iters = results.get('iterations') or 0
    return _stream_phi_prefix_from_gz(iters, need, var=var) or ""


def plot_raster2d(results_list, max_iter_cap: int | None = None):
    if not MATPLOTLIB_OK:
        print("[WARN] Skipping 2D raster (matplotlib unavailable).")
        return
    import matplotlib.pyplot as plt
    cap = _get_raster_iter_cap() if max_iter_cap is None else int(max_iter_cap)
    for r in results_list:
        if not r: continue
        var = r.get('variant','?'); total_iters = r.get('iterations') or 0
        target_iter = min(total_iters, cap)
        # read bits for the chosen iteration; if missing for high i, fall back to cap
        bits = _bits_for_iteration(r, target_iter, max_bits=5_000_000)
        if not bits and target_iter > cap:
            # explicit fallback to cap iteration to guarantee a raster is produced
            bits = _bits_for_iteration(r, cap, max_bits=5_000_000)
        if not bits:
            # Try to read a smaller prefix with progress so the user sees activity
            bits = _stream_phi_prefix_from_gz(target_iter, 1_000_000, show_progress=True, label="raster2d", var=var)
        if not bits:
            print(f"[WARN] 2D raster skipped for {var}: no data available.")
            continue
        a = _bits_to_uint8(bits)
        n = a.size
        s = int(np.ceil(np.sqrt(n)))
        # pad to fill perfect square
        if s*s != n:
            pad = np.zeros(s*s - n, dtype=np.uint8)
            a = np.concatenate([a, pad])
        img = a.reshape(s, s)
        plt.figure(figsize=(7,7))
        plt.imshow(img, cmap='Greys', interpolation='nearest', origin='upper')
        plt.title(f'2D raster — {var} (i={target_iter}, {n:,} bits; square {s}×{s})')
        plt.axis('off'); plt.tight_layout()
        _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out = VIS_DIR / f"raster2d_{var}{_abs_suffix()}_i{target_iter}_{_ts}.png"
        plt.savefig(out, dpi=200)
        print(f"[INFO] 2D raster saved: {out.relative_to(BASE_PATH)}")
        plt.close()

def _welch_psd_from_gz(iterations: int, cfg: dict, max_bits_cap: int | None = None, label: str = "welch", var: str | None = None):
    import gzip, numpy as _np, time, json
    w = int(cfg.get('window_size', 1_048_576))
    ov = float(cfg.get('overlap', 0.5))
    progress = bool(cfg.get('progress', True))
    max_windows = int(cfg.get('max_windows', 64))
    # Look under variant subdir first, then fallback
    base = _project_snapshots_dir(var)

    # Detect format from metadata
    json_path = os.path.join(base, f"phi_iter{iterations}.json")
    is_v33 = False
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                meta = json.load(f)
                is_v33 = meta.get("format") == "v33_structural"
        except Exception:
            pass

    # Determine which file to use
    if is_v33:
        gz_path = os.path.join(base, f"phi_iter{iterations}.struct.gz")
    else:
        gz_path = os.path.join(base, f"phi_iter{iterations}.bin.gz")

    # Do NOT fallback to flat directory when variant is given; avoid cross-variant contamination
    if not os.path.exists(gz_path):
        return None, None
    # Optional cap on total samples to process (for very large files)
    limit_samples = None
    if max_bits_cap is not None:
        limit_samples = int(max_bits_cap)
    # Effective window cannot exceed cap; ensure at least one window fits
    w_eff = w if limit_samples is None else max(1024, min(w, limit_samples))
    step = max(1, int(w_eff * (1.0 - ov)))
    if step >= w_eff:
        step = max(1, w_eff//2)
    # Iterate streaming
    try:
        rel = Path(gz_path).resolve().relative_to(BASE_PATH)
    except Exception:
        rel = gz_path
    if progress:
        format_label = "v33" if is_v33 else "v32"
        print(f"[plot] {label}: streaming PSD via Welch from {rel} ({format_label}) | window={w_eff}, overlap={ov*100:.0f}%, max_windows={max_windows}")

    # For v33, load and clean first
    if is_v33:
        from hsi_agents_project.level0.phi_snapshot_manager import PhiSnapshotManager
        manager = PhiSnapshotManager(data_dir=base)
        phi_structural, _ = manager.load_phi_state_structural(iterations)
        phi_clean = phi_structural.replace('(', '').replace(')', '')

        # Apply cap if needed
        if limit_samples is not None:
            phi_clean = phi_clean[:limit_samples]

        # Process windows
        buf = []
        psd_acc = None
        nwin = 0
        t0 = time.perf_counter()
        i = 0
        N = len(phi_clean)
        while i + w_eff <= N and nwin < max_windows:
            window_bits = phi_clean[i:i+w_eff]
            a = _bits_to_float64(window_bits)
            a = a - a.mean()
            spec = _np.fft.rfft(a)
            power = (spec.real**2 + spec.imag**2)
            if psd_acc is None:
                psd_acc = power
            else:
                psd_acc += power
            nwin += 1
            i += step
            if progress and nwin % 2 == 0:
                elapsed = time.perf_counter() - t0
                pct = min(1.0, nwin/max_windows)
                eta = (elapsed/pct - elapsed) if pct > 0 else 0.0
                print(f"[plot] {label}: windows {nwin}/{max_windows} ({pct*100:5.1f}%) ETA ~{eta:.1f}s")
    else:
        # v32: stream from gzip text
        buf = []
        psd_acc = None
        nwin = 0
        t0 = time.perf_counter()
        read_total = 0
        next_mark = 0.10
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            while True:
                chunk = f.read(1_000_000)
                if not chunk:
                    break
                # Gather into buffer and process windows
                buf.append(chunk)
                read_total += len(chunk)
                joined = ''.join(buf)
                # If cap, truncate joined for processing
                if limit_samples is not None and len(joined) > limit_samples:
                    joined = joined[:limit_samples]
                # Process as many windows as possible
                i = 0
                N = len(joined)
                while i + w_eff <= N and nwin < max_windows:
                    window_bits = joined[i:i+w_eff]
                    a = _bits_to_float64(window_bits)
                    a = a - a.mean()
                    spec = _np.fft.rfft(a)
                    power = (spec.real**2 + spec.imag**2)
                    if psd_acc is None:
                        psd_acc = power
                    else:
                        psd_acc += power
                    nwin += 1
                    i += step
                    if progress and nwin % 2 == 0:
                        elapsed = time.perf_counter() - t0
                        pct = min(1.0, nwin/max_windows)
                        eta = (elapsed/pct - elapsed) if pct > 0 else 0.0
                        print(f"[plot] {label}: windows {nwin}/{max_windows} ({pct*100:5.1f}%) ETA ~{eta:.1f}s")
                # Keep the tail for next round
                buf = [joined[i:]]
                if nwin >= max_windows:
                    break
                if progress and limit_samples is not None and read_total < limit_samples and read_total >= limit_samples * next_mark:
                    pct_r = min(1.0, read_total/(limit_samples or 1))
                    elapsed = time.perf_counter() - t0
                    eta = (elapsed/pct_r - elapsed) if pct_r > 0 else 0.0
                    print(f"[plot] {label}: read {read_total:,}/{limit_samples:,} ({pct_r*100:5.1f}%) ETA ~{eta:.1f}s")
                    next_mark += 0.10
    if psd_acc is None or nwin == 0:
        return None, None
    psd_acc /= float(max(1, nwin))
    # Guard: if PSD is all zeros (e.g., constant windows), trigger fallback
    if not _np.any(psd_acc > 0):
        if progress:
            print(f"[plot] {label}: PSD all zeros after {nwin} windows — falling back to prefix")
        return None, None
    freqs = _np.fft.rfftfreq(w_eff, d=1.0)
    return freqs, psd_acc

def _sampling_psd_from_gz(iterations: int, cfg: dict, max_bits_cap: int | None = None, label: str = "sampling", var: str | None = None):
    import gzip, numpy as _np, time, json
    w = int(cfg.get('window_size', 1_048_576))
    progress = bool(cfg.get('progress', True))
    max_windows = int(cfg.get('max_windows', 64))
    base = _project_snapshots_dir(var)

    # Detect format from metadata
    json_path = os.path.join(base, f"phi_iter{iterations}.json")
    is_v33 = False
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                meta = json.load(f)
                is_v33 = meta.get("format") == "v33_structural"
        except Exception:
            pass

    # Determine which file to use
    if is_v33:
        gz_path = os.path.join(base, f"phi_iter{iterations}.struct.gz")
    else:
        gz_path = os.path.join(base, f"phi_iter{iterations}.bin.gz")

    # Do NOT fallback to flat directory when variant is given; avoid cross-variant contamination
    if not os.path.exists(gz_path):
        return None, None
    limit_samples = int(max_bits_cap) if max_bits_cap is not None else None
    w_eff = w if limit_samples is None else max(1024, min(w, limit_samples))
    step = w_eff  # non-overlapping sampling

    try:
        rel = Path(gz_path).resolve().relative_to(BASE_PATH)
    except Exception:
        rel = gz_path
    if progress:
        format_label = "v33" if is_v33 else "v32"
        cap_txt = f", cap={limit_samples:,}" if limit_samples else ""
        print(f"[plot] {label}: sampling PSD from {rel} ({format_label}) | window={w_eff}, max_windows={max_windows}{cap_txt}")

    # For v33, load and clean first
    if is_v33:
        from hsi_agents_project.level0.phi_snapshot_manager import PhiSnapshotManager
        manager = PhiSnapshotManager(data_dir=base)
        phi_structural, _ = manager.load_phi_state_structural(iterations)
        phi_clean = phi_structural.replace('(', '').replace(')', '')

        # Apply cap if needed
        if limit_samples is not None:
            phi_clean = phi_clean[:limit_samples]

        # Process windows
        psd_acc = None
        nwin = 0
        t0 = time.perf_counter()
        i = 0
        N = len(phi_clean)
        while i + w_eff <= N and nwin < max_windows:
            window_bits = phi_clean[i:i+w_eff]
            a = _bits_to_float64(window_bits)
            a = a - a.mean()
            spec = _np.fft.rfft(a)
            power = (spec.real**2 + spec.imag**2)
            if psd_acc is None:
                psd_acc = power
            else:
                psd_acc += power
            nwin += 1
            i += step
            if progress and nwin % 2 == 0:
                elapsed = time.perf_counter() - t0
                pct = min(1.0, nwin/max_windows)
                eta = (elapsed/pct - elapsed) if pct > 0 else 0.0
                print(f"[plot] {label}: windows {nwin}/{max_windows} ({pct*100:5.1f}%) ETA ~{eta:.1f}s")

        if psd_acc is None or nwin == 0:
            return None, None
        psd_acc /= nwin
        freqs = _np.fft.rfftfreq(w_eff, d=1.0)
        return freqs, psd_acc

    # v32: stream from gzip text
    buf = []
    psd_acc = None
    nwin = 0
    t0 = time.perf_counter()
    read_total = 0
    next_mark = 0.10
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        while True:
            chunk = f.read(1_000_000)
            if not chunk:
                break
            buf.append(chunk)
            read_total += len(chunk)
            joined = ''.join(buf)
            if limit_samples is not None and len(joined) > limit_samples:
                joined = joined[:limit_samples]
            i = 0
            N = len(joined)
            while i + w_eff <= N and nwin < max_windows:
                window_bits = joined[i:i+w_eff]
                a = _bits_to_float64(window_bits)
                a = a - a.mean()
                spec = _np.fft.rfft(a)
                power = (spec.real**2 + spec.imag**2)
                if psd_acc is None:
                    psd_acc = power
                else:
                    psd_acc += power
                nwin += 1
                i += step
                if progress and nwin % 2 == 0:
                    elapsed = time.perf_counter() - t0
                    pct = min(1.0, nwin/max_windows)
                    eta = (elapsed/pct - elapsed) if pct > 0 else 0.0
                    print(f"[plot] {label}: windows {nwin}/{max_windows} ({pct*100:5.1f}%) ETA ~{eta:.1f}s")
            buf = [joined[i:]]
            if nwin >= max_windows:
                break
            if progress and limit_samples is not None and read_total < limit_samples and read_total >= limit_samples * next_mark:
                pct_r = min(1.0, read_total/(limit_samples or 1))
                elapsed = time.perf_counter() - t0
                eta = (elapsed/pct_r - elapsed) if pct_r > 0 else 0.0
                print(f"[plot] {label}: read {read_total:,}/{limit_samples:,} ({pct_r*100:5.1f}%) ETA ~{eta:.1f}s")
                next_mark += 0.10
    if psd_acc is None:
        return None, None
    psd_acc /= float(max(1, nwin))
    freqs = _np.fft.rfftfreq(w_eff, d=1.0)
    return freqs, psd_acc

# Decide best spectral method based on available data and window
def _decide_spectral_method_auto(iterations: int, spec_cfg: dict, var: str | None) -> str:
    # If variant snapshot exists, estimate how many windows we can get
    base = _project_snapshots_dir(var)

    # Detect format
    json_path = os.path.join(base, f"phi_iter{iterations}.json")
    is_v33 = False
    if os.path.exists(json_path):
        try:
            import json
            with open(json_path, "r") as f:
                meta = json.load(f)
                is_v33 = meta.get("format") == "v33_structural"
        except Exception:
            pass

    if is_v33:
        gz_path = os.path.join(base, f"phi_iter{iterations}.struct.gz")
    else:
        gz_path = os.path.join(base, f"phi_iter{iterations}.bin.gz")

    w = int(spec_cfg.get('window_size', 1_048_576))
    if not os.path.exists(gz_path):
        print(f"[plot] spectral auto: method=prefix (no snapshot) var={var} i={iterations}")
        return 'prefix'  # fall back to prefix directly
    try:
        import gzip
        # Read a small chunk to estimate length; peeking by gzip size is unreliable.
        limit = min(4*w, 5_000_000)
        got = 0

        if is_v33:
            # For v33, load structural and clean
            from hsi_agents_project.level0.phi_snapshot_manager import PhiSnapshotManager
            manager = PhiSnapshotManager(data_dir=base)
            phi_structural, _ = manager.load_phi_state_structural(iterations)
            phi_clean = phi_structural.replace('(', '').replace(')', '')
            got = min(len(phi_clean), limit)
        else:
            # For v32, stream from gzip
            with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
                for chunk in iter(lambda: f.read(min(1_000_000, limit-got)), ""):
                    got += len(chunk)
                    if got >= limit:
                        break

        nwin_possible = max(0, got // w)
        if nwin_possible >= 4:
            print(f"[plot] spectral auto: method=welch, nwin≈{nwin_possible}, w={w:,} var={var} i={iterations}")
            return 'welch'
        elif nwin_possible >= 1:
            print(f"[plot] spectral auto: method=sampling, nwin≈{nwin_possible}, w={w:,} var={var} i={iterations}")
            return 'sampling'
        else:
            print(f"[plot] spectral auto: method=prefix, nwin≈0, w={w:,} var={var} i={iterations}")
            return 'prefix'
    except Exception as e:
        print(f"[plot] spectral auto: fallback sampling due to error: {e}")
        return 'sampling'



def _stream_phi_prefix_from_gz(iterations: int, max_bits: int, data_dir: str | None = None, show_progress: bool = False, label: str = "", var: str | None = None, return_structural: bool = False) -> str:
    """
    Stream Φ prefix from compressed file.

    Args:
        iterations: Iteration number to load
        max_bits: Maximum number of bits to read
        data_dir: Data directory (if None, uses variant subdir)
        show_progress: Show progress during read
        label: Label for progress messages
        var: Variant code
        return_structural: If True, return structural format (with parentheses) for v33 files

    Returns:
        String with Φ prefix (clean bits or structural depending on format and return_structural)
    """
    import gzip, time, json

    # Prefer variant subdir; fallback to flat
    if data_dir is None:
        base = _project_snapshots_dir(var)
        data_dir = base

    # Check for v33 structural format first
    struct_path = os.path.join(data_dir, f"phi_iter{iterations}.struct.gz")
    v32_path = os.path.join(data_dir, f"phi_iter{iterations}.bin.gz")
    json_path = os.path.join(data_dir, f"phi_iter{iterations}.json")

    # Detect format from metadata
    is_v33 = False
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                meta = json.load(f)
                fmt = meta.get("format", "")
                # Accept both v33_structural and v33_structural_streaming
                is_v33 = fmt.startswith("v33_structural")
        except Exception:
            pass

    # Determine which file to read
    if is_v33 and os.path.exists(struct_path):
        gz_path = struct_path
        format_label = "v33"
    elif os.path.exists(v32_path):
        gz_path = v32_path
        format_label = "v32"
    else:
        return ""

    if show_progress:
        try:
            rel = Path(gz_path).resolve().relative_to(BASE_PATH)
        except Exception:
            rel = gz_path
        print(f"[plot] Reading prefix from: {rel} (format={format_label}, target {max_bits:,} bits) {label}")

    # For v33 structural format, use efficient streaming decoder
    if is_v33:
        from hsi_agents_project.utils.bitarray_encoder import stream_phi_prefix_gz
        # stream_phi_prefix_gz reads only the bytes needed, not the entire file
        return stream_phi_prefix_gz(gz_path, max_bits, clean=(not return_structural))

    # For v32 format: read as text
    out = []
    need = max_bits
    t0 = time.perf_counter()
    next_mark = 0.10
    total = max_bits
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for chunk in iter(lambda: f.read(min(1_000_000, need)), ""):
            out.append(chunk)
            need -= len(chunk)
            if show_progress and total > 0:
                done = total - need
                pct = max(0.0, min(1.0, done/total))
                if pct >= next_mark:
                    elapsed = time.perf_counter() - t0
                    eta = (elapsed/pct - elapsed) if pct > 0 else 0.0
                    print(f"[plot] {label} progress: {pct*100:5.1f}% ({done:,}/{total:,}) ETA ~{eta:.1f}s")
                    next_mark += 0.10
            if need <= 0:
                break
    return "".join(out)[:max_bits]


def _compute_local_entropy(counts: "np.ndarray", window: int = 3) -> "np.ndarray":
    """Compute local Shannon entropy using sliding window on count grid."""
    import numpy as _np
    from scipy.ndimage import uniform_filter

    # Total counts per window
    total = uniform_filter(counts.astype(float), size=window, mode='constant')
    total = _np.maximum(total, 1e-10)  # Avoid division by zero

    # Probability of 1s in each window
    p1 = counts.astype(float) / _np.maximum(counts.max(), 1)
    p1_smooth = uniform_filter(p1, size=window, mode='constant')
    p0_smooth = 1.0 - p1_smooth

    # Shannon entropy: -p*log2(p) - (1-p)*log2(1-p)
    # Handle edge cases where p=0 or p=1
    eps = 1e-10
    p1_safe = _np.clip(p1_smooth, eps, 1-eps)
    p0_safe = 1.0 - p1_safe
    entropy = -p1_safe * _np.log2(p1_safe) - p0_safe * _np.log2(p0_safe)

    return entropy


def _adaptive_hilbert_grid(n_bits: int, config_grid: int) -> int:
    """
    Compute optimal grid size based on available bits.

    Goal: Ensure enough bits per cell for meaningful statistics.
    Minimum ~4 bits/cell for differential detection, ~16 for entropy.

    Args:
        n_bits: Number of bits available
        config_grid: Maximum grid from config

    Returns:
        Optimal grid size (power of 2 for Hilbert curve alignment)
    """
    import math
    # Target: at least 8 bits per cell on average for good statistics
    target_bits_per_cell = 8
    max_cells = n_bits / target_bits_per_cell
    # Grid size = sqrt(cells), rounded to nearest power of 2
    ideal_grid = int(math.sqrt(max_cells))
    # Clamp to reasonable range and nearest power of 2
    ideal_grid = max(64, min(ideal_grid, config_grid))
    # Round to nearest power of 2
    power = int(math.log2(ideal_grid))
    return 2 ** power


def plot_hilbert_heatmap(results_list, max_bits: int, skip: bool):
    if skip:
        return
    if not MATPLOTLIB_OK or not HAVE_L0_UTILS:
        print("[WARN] Skipping Hilbert map (matplotlib or Level0 utils unavailable).", flush=True)
        return
    import matplotlib.pyplot as plt
    import numpy as _np
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
    hc = _get_hilbert_config()
    config_grid = max(64, int(hc.get('target_grid', 1024)))
    max_bits_eff = hc['max_bits'] if max_bits is None else int(max_bits)
    for r in results_list:
        if not r: continue
        iters = r.get('iterations'); var = r.get('variant','?')
        with ProgressIndicator(f"Hilbert {var}") as prog:
            prog.update(message="Reading data...")
            bits = _stream_phi_prefix_from_gz(iters, max_bits_eff, show_progress=True, label="hilbert", var=var)
            if not bits:
                print(f"[WARN] Hilbert map skipped for {var}: no data available.", flush=True)
                continue

            # Adaptive grid: adjust resolution based on actual bits available
            # Truncate bits to nearest power of 4 so Hilbert curve fills grid completely
            import math
            n_bits_raw = len(bits)
            # Find largest power of 4 <= n_bits (ensures complete Hilbert grid fill)
            hilbert_order = int(math.log(n_bits_raw, 4)) if n_bits_raw >= 4 else 1
            exact_bits = 4 ** hilbert_order  # e.g., 4^12 = 16,777,216
            next_power = 4 ** (hilbert_order + 1)  # Next level up

            # Log if significant bits are being discarded
            discarded = n_bits_raw - exact_bits
            if discarded > 0:
                pct_used = (exact_bits / n_bits_raw) * 100
                print(f"[INFO] Hilbert requires power-of-4 bits: using {exact_bits:,} of {n_bits_raw:,} ({pct_used:.1f}%)")
                print(f"       → To use next level (4^{hilbert_order+1}), request --hilbert-bits {next_power:,}")

            bits = bits[:exact_bits]  # Truncate to exact power of 4
            n_bits = exact_bits
            grid_size = 2 ** hilbert_order  # Grid is 2^order × 2^order
            target_grid = min(grid_size, config_grid)
            bits_per_cell = n_bits / (target_grid * target_grid)
            prog.update(message=f"Using {n_bits:,} bits → {grid_size}×{grid_size} grid (fills 100%)")

            # 1) Convert to array — VECTORIZED
            prog.update(message="Converting to array...")
            a = _np.frombuffer(bits.encode('ascii'), dtype='S1') == b'1'
            a = a.astype(_np.uint8)
            prog.update(message="Computing Hilbert coords...")
            coords = phi_to_hilbert_coords(a)
            x = coords[:,0]; y = coords[:,1]
            size = int(max(x.max(), y.max()) + 1)

            # 2) Build count grids: ones and total
            prog.update(message=f"Building heatmaps ({len(a):,} points)...")
            ones_mask = (a == 1)
            heat_ones = _np.zeros((size, size), dtype=_np.int32)
            heat_total = _np.zeros((size, size), dtype=_np.int32)
            _np.add.at(heat_ones, (y[ones_mask], x[ones_mask]), 1)
            _np.add.at(heat_total, (y, x), 1)

            # 3) Rebin to adaptive target_grid
            pool = max(1, size // target_grid)
            if pool > 1:
                prog.update(message=f"Rebinning to {target_grid}x{target_grid}...")
                new_h = size // pool; new_w = size // pool
                heat_ones = heat_ones[:new_h*pool, :new_w*pool].reshape(new_h, pool, new_w, pool).sum(axis=(1,3))
                heat_total = heat_total[:new_h*pool, :new_w*pool].reshape(new_h, pool, new_w, pool).sum(axis=(1,3))

            # 4) Compute differential (deviation from 50% random)
            prog.update(message="Computing differential...")
            with _np.errstate(divide='ignore', invalid='ignore'):
                density = _np.where(heat_total > 0, heat_ones / heat_total, 0.5)
            differential = density - 0.5  # Range: [-0.5, +0.5]

            # 5) Compute local entropy
            prog.update(message="Computing entropy...")
            try:
                entropy = _compute_local_entropy(heat_ones, window=5)
            except ImportError:
                entropy = None  # scipy not available

            # 6) Create 3-panel figure
            prog.update(message="Rendering...")
            n_panels = 3 if entropy is not None else 2
            fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 5))
            _ts = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Panel 1: Density (count of 1s per cell)
            ax1 = axes[0]
            vmax = int(heat_ones.max()) if heat_ones.size else 0
            if vmax <= 0:
                im1 = ax1.imshow(heat_ones, cmap='inferno', origin='lower')
            else:
                vmin_pos = int(heat_ones[heat_ones>0].min()) if (heat_ones>0).any() else 1
                norm = LogNorm(vmin=max(1, vmin_pos), vmax=vmax) if vmin_pos < vmax else Normalize(0, vmax)
                im1 = ax1.imshow(heat_ones, cmap='inferno', origin='lower', norm=norm)
            ax1.set_title('Density (1s per cell)', fontsize=10)
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # Panel 2: Differential (deviation from random)
            ax2 = axes[1]
            vabs = max(abs(differential.min()), abs(differential.max()), 0.01)
            norm_diff = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
            im2 = ax2.imshow(differential, cmap='RdBu_r', origin='lower', norm=norm_diff)
            ax2.set_title('Differential (red=excess 1s, blue=excess 0s)', fontsize=10)
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            # Panel 3: Entropy (if available)
            if entropy is not None:
                ax3 = axes[2]
                im3 = ax3.imshow(entropy, cmap='viridis_r', origin='lower', vmin=0, vmax=1)
                ax3.set_title('Local Entropy (dark=ordered, bright=random)', fontsize=10)
                ax3.axis('off')
                plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            fig.suptitle(f'Hilbert Analysis — {var} (i={iters}, {len(bits):,} bits)', fontsize=12)
            plt.tight_layout()
            out = VIS_DIR / f"hilbert_{var}{_abs_suffix()}_i{iters}_{_ts}.png"
            prog.update(message="Saving...")
            plt.savefig(out, dpi=200)
            plt.close()
        print(f"[INFO] Hilbert saved: {out.relative_to(BASE_PATH)}", flush=True)
    print('[INFO] Saved Hilbert heatmaps (where data available).', flush=True)


def plot_fft(results_list, max_bits: int, skip: bool):
    if skip:
        return
    if not MATPLOTLIB_OK:
        print("[WARN] Skipping FFT (matplotlib unavailable).", flush=True)
        return
    import matplotlib.pyplot as plt
    spec_cfg = _get_spectral_config()
    for r in results_list:
        if not r: continue
        iters = r.get('iterations'); var = r.get('variant','?')
        with ProgressIndicator(f"FFT {var}") as prog:
            method = spec_cfg['method']
            if method == 'auto':
                prog.update(message="Auto-detecting method...")
                method_eff = _decide_spectral_method_auto(iters, spec_cfg, var)
            else:
                method_eff = method
            prog.update(message=f"Computing ({method_eff})...")
            if method_eff == 'welch':
                freqs, power = _welch_psd_from_gz(iters, spec_cfg, max_bits_cap=max_bits, label="fft", var=var)
            elif method_eff == 'sampling':
                freqs, power = _sampling_psd_from_gz(iters, spec_cfg, max_bits_cap=max_bits, label="fft-sampling", var=var)
            else:
                freqs = power = None
            if freqs is None:
                prog.update(message="Reading prefix fallback...")
                bits = _stream_phi_prefix_from_gz(iters, max_bits, show_progress=True, label="fft", var=var)
                if not bits:
                    print(f"[WARN] FFT skipped for {var}: no data available.", flush=True)
                    continue
                prog.update(message="Computing FFT...")
                a = _bits_to_float64(bits)
                a = a - a.mean()
                spec = np.fft.rfft(a)
                power = (spec.real**2 + spec.imag**2)
                freqs = np.fft.rfftfreq(a.size, d=1.0)
            prog.update(message="Rendering plot...")
            plt.figure(figsize=(7,4))
            plt.loglog(freqs[1:], power[1:])
            title_src = (
                f"Welch {spec_cfg['window_size']:,}w" if method_eff=='welch' else (
                    f"Sampling {spec_cfg['window_size']:,}w" if method_eff=='sampling' else f"first {len(bits):,} bits"
                )
            )
            if freqs is None or power is None:
                plt.title(f'FFT power spectrum — {var} (i={iters}, prefix {len(bits):,} bits)')
            else:
                plt.title(f'FFT power spectrum — {var} (i={iters}, {title_src})')
            plt.xlabel('Frequency'); plt.ylabel('Power')
            _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            out = VIS_DIR / f"fft_{var}{_abs_suffix()}_i{iters}_{_ts}.png"
            prog.update(message="Saving...")
            plt.tight_layout(); plt.savefig(out, dpi=200)
            plt.close()
        print(f"[INFO] FFT saved: {out.relative_to(BASE_PATH)}", flush=True)
        plt.close()
    print('[INFO] Saved FFT power spectra (where data available).')


def plot_spectrum_beta_fit(results_list, max_bits: int, skip: bool):
    if skip:
        return []
    if not MATPLOTLIB_OK:
        print("[WARN] Skipping spectrum beta plot (matplotlib unavailable).")
        return []
    import matplotlib.pyplot as plt
    import numpy as _np
    spec_cfg = _get_spectral_config()
    out_metrics = []
    for r in results_list:
        if not r: out_metrics.append(None); continue
        iters = r.get('iterations'); var = r.get('variant','?')
        method = spec_cfg['method']
        if method == 'welch':
            freqs, power = _welch_psd_from_gz(iters, spec_cfg, max_bits_cap=max_bits, label="beta", var=var)
        elif method == 'sampling':
            freqs, power = _sampling_psd_from_gz(iters, spec_cfg, max_bits_cap=max_bits, label="beta-sampling", var=var)
        else:
            freqs = power = None
        if freqs is None:
            bits = _stream_phi_prefix_from_gz(iters, max_bits, var=var)
            if not bits:
                print(f"[WARN] Spectrum beta skipped for {var}: no data available.")
                out_metrics.append(None); continue
            a = _bits_to_float64(bits)
            a = a - a.mean()
            spec = _np.fft.rfft(a)
            power = (spec.real**2 + spec.imag**2)
            freqs = _np.fft.rfftfreq(a.size, d=1.0)
        # Frequency-range selection (avoid DC, high-freq tail and dominant peaks)
        # Defaults can be overridden via env: HSI_BETA_FMIN, HSI_BETA_FMAX
        fmin = float(os.environ.get('HSI_BETA_FMIN', '1e-4'))
        fmax = float(os.environ.get('HSI_BETA_FMAX', '1e-2'))
        mask = (freqs > fmin) & (freqs < fmax) & _np.isfinite(power) & (power > 0)
        if not _np.any(mask):
            # Fallback to mid-band percentiles if absolute range yields nothing
            m = len(freqs)
            i0 = max(1, int(0.01*m))
            i1 = min(m-1, max(i0+30, int(0.20*m)))
            mask = _np.zeros_like(freqs, dtype=bool)
            mask[i0:i1] = True
            mask &= _np.isfinite(power) & (power > 0)
        xf = _np.log(freqs[mask])
        yf = _np.log(power[mask])
        # Guard against insufficient points
        valid_points = xf.size >= 50
        if valid_points:
            slope, intercept = _np.polyfit(xf, yf, 1)
            beta = float(-slope)  # power ~ 1/f^beta
            yhat = slope * xf + intercept
            ss_res = float(_np.sum((yf - yhat)**2))
            ss_tot = float(_np.sum((yf - yf.mean())**2)) or 1.0
            r2 = 1.0 - ss_res/ss_tot
        else:
            beta = None
            r2 = 0.0
        # Gating: accept only if beta in [0.2, 2.5] and fit quality high
        BETA_MIN = float(os.environ.get('HSI_BETA_MIN', '0.2'))
        BETA_MAX = float(os.environ.get('HSI_BETA_MAX', '2.5'))
        MIN_R2 = float(os.environ.get('HSI_BETA_MIN_R2', '0.95'))
        beta_valid = (beta is not None) and (BETA_MIN <= beta <= BETA_MAX) and (r2 >= MIN_R2)
        # Plot
        plt.figure(figsize=(7,4))
        plt.loglog(freqs[1:], power[1:], alpha=0.7, label='Power')
        if valid_points:
            xp = _np.linspace(xf.min(), xf.max(), 100)
            yp = slope * xp + intercept
            line_label = f"Fit β={beta:.3f} (R²={r2:.3f})" if beta is not None else "Fit (insufficient points)"
            color = 'orange' if beta_valid else 'gray'
            plt.plot(_np.exp(xp), _np.exp(yp), color=color, label=line_label)
        title_src = (
            f"Welch {spec_cfg['window_size']:,}w" if method=='welch' else (
                f"Sampling {spec_cfg['window_size']:,}w" if method=='sampling' else f'n={len(bits):,}'
            )
        )
        plt.title(f'FFT Power Spectrum β-fit — {var} (i={iters}, {title_src})')
        plt.xlabel('Frequency'); plt.ylabel('Power'); plt.legend()
        _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out = VIS_DIR / f"spectrum_beta_{var}{_abs_suffix()}_i{iters}_{_ts}.png"
        plt.tight_layout(); plt.savefig(out, dpi=200)
        print(f"[INFO] Spectrum beta plot saved: {out.relative_to(BASE_PATH)}")
        plt.close()
        out_metrics.append({
            'beta': float(beta) if beta_valid else None,
            'r2': float(r2),
            'fmin': float(fmin), 'fmax': float(fmax),
            'points': int(xf.size),
            'valid': bool(beta_valid),
        })
    return out_metrics


def plot_autocorrelation(results_list, max_bits: int, skip: bool):
    """
    Compute and plot autocorrelation to detect periodicity in bit sequences.

    This analysis reveals:
    - Dominant periods in the sequence
    - Whether periods relate to Fibonacci numbers or φ
    - The regularity/predictability of the pattern
    """
    if skip:
        return
    if not MATPLOTLIB_OK:
        print("[WARN] Skipping autocorrelation (matplotlib unavailable).", flush=True)
        return
    import matplotlib.pyplot as plt

    # Fibonacci sequence for comparison
    FIB = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]
    PHI = (1 + 5**0.5) / 2  # Golden ratio

    for r in results_list:
        if not r:
            continue
        iters = r.get('iterations')
        var = r.get('variant', '?')

        with ProgressIndicator(f"Autocorr {var}") as prog:
            prog.update(message="Reading bits...")
            bits = _stream_phi_prefix_from_gz(iters, max_bits, show_progress=False, label="autocorr", var=var)
            if not bits or len(bits) < 1000:
                print(f"[WARN] Autocorrelation skipped for {var}: insufficient data.", flush=True)
                continue

            prog.update(message="Converting to array...")
            # Convert bits to +1/-1 for better autocorrelation signal
            arr = np.array([1 if b == '1' else -1 for b in bits], dtype=np.float32)
            n = len(arr)

            # Limit max lag for computational efficiency
            max_lag = min(5000, n // 4)

            prog.update(message=f"Computing autocorrelation (max_lag={max_lag})...")

            # Use FFT-based autocorrelation for efficiency
            # Autocorrelation via FFT: R(τ) = IFFT(|FFT(x)|²)
            arr_centered = arr - arr.mean()
            fft_arr = np.fft.fft(arr_centered, n=2*n)  # Zero-pad for linear correlation
            power = np.abs(fft_arr)**2
            autocorr_full = np.fft.ifft(power).real[:n]
            autocorr_full /= autocorr_full[0]  # Normalize to 1 at lag 0

            autocorr = autocorr_full[1:max_lag+1]  # Skip lag 0 (always 1)
            lags = np.arange(1, max_lag+1)

            prog.update(message="Finding peaks...")
            # Find peaks in autocorrelation
            peaks = []
            for i in range(1, len(autocorr)-1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    if autocorr[i] > 0.05:  # Threshold for significance
                        peaks.append((lags[i], autocorr[i]))

            # Sort by correlation strength
            peaks.sort(key=lambda x: -x[1])
            top_peaks = peaks[:10]  # Top 10 peaks

            # Check if peaks relate to Fibonacci
            fib_matches = []
            for lag, corr in top_peaks:
                for fib in FIB:
                    if abs(lag - fib) <= 2:  # Allow ±2 tolerance
                        fib_matches.append((lag, fib, corr))
                        break

            # Check if ratios between peaks approximate φ
            phi_ratios = []
            if len(top_peaks) >= 2:
                sorted_peaks = sorted(top_peaks, key=lambda x: x[0])
                for i in range(len(sorted_peaks)-1):
                    lag1, _ = sorted_peaks[i]
                    lag2, _ = sorted_peaks[i+1]
                    if lag1 > 0:
                        ratio = lag2 / lag1
                        phi_ratios.append((lag1, lag2, ratio, abs(ratio - PHI)))

            prog.update(message="Rendering plot...")

            # Create figure with 2 subplots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Plot 1: Full autocorrelation
            ax1 = axes[0]
            ax1.plot(lags, autocorr, 'b-', alpha=0.7, linewidth=0.5)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax1.axhline(y=0.05, color='red', linestyle=':', alpha=0.5, label='Significance threshold')

            # Mark Fibonacci lags
            for fib in FIB:
                if fib <= max_lag:
                    ax1.axvline(x=fib, color='gold', alpha=0.3, linestyle='-')

            # Mark top peaks
            for lag, corr in top_peaks[:5]:
                ax1.plot(lag, corr, 'ro', markersize=8)
                ax1.annotate(f'{lag}', (lag, corr), textcoords='offset points',
                           xytext=(0, 10), ha='center', fontsize=8)

            ax1.set_xlabel('Lag (bits)')
            ax1.set_ylabel('Autocorrelation')
            ax1.set_title(f'Autocorrelation — {var} (i={iters}, {n:,} bits)')
            ax1.set_xlim(0, max_lag)
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)

            # Plot 2: Zoomed view (first 500 lags) with annotations
            ax2 = axes[1]
            zoom_max = min(500, max_lag)
            ax2.plot(lags[:zoom_max], autocorr[:zoom_max], 'b-', alpha=0.8, linewidth=1)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

            # Mark Fibonacci lags with labels
            for fib in FIB:
                if fib <= zoom_max:
                    ax2.axvline(x=fib, color='gold', alpha=0.5, linestyle='-')
                    ax2.annotate(f'F={fib}', (fib, ax2.get_ylim()[1]*0.9),
                               rotation=90, fontsize=7, color='goldenrod')

            ax2.set_xlabel('Lag (bits)')
            ax2.set_ylabel('Autocorrelation')
            ax2.set_title(f'Zoomed (0-{zoom_max}) — Gold lines = Fibonacci numbers')
            ax2.set_xlim(0, zoom_max)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Add text box with analysis summary
            analysis_text = f"Top 5 periods: {', '.join([str(p[0]) for p in top_peaks[:5]])}\n"
            if fib_matches:
                analysis_text += f"Fibonacci matches: {', '.join([f'{m[0]}≈F{m[1]}' for m in fib_matches[:3]])}\n"
            if phi_ratios:
                best_phi = min(phi_ratios, key=lambda x: x[3])
                analysis_text += f"Best φ-ratio: {best_phi[1]}/{best_phi[0]}={best_phi[2]:.4f} (φ={PHI:.4f})"

            fig.text(0.02, 0.02, analysis_text, fontsize=9, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            out = VIS_DIR / f"autocorr_{var}{_abs_suffix()}_i{iters}_{_ts}.png"
            prog.update(message="Saving...")
            plt.savefig(out, dpi=200, bbox_inches='tight')
            plt.close()

        print(f"[INFO] Autocorrelation saved: {out.relative_to(BASE_PATH)}", flush=True)

        # Print summary to console
        print(f"       📊 {var}: Top periods = {[p[0] for p in top_peaks[:5]]}", flush=True)
        if fib_matches:
            print(f"       🔢 Fibonacci matches: {[(m[0], m[1]) for m in fib_matches]}", flush=True)

    print('[INFO] Saved autocorrelation plots.', flush=True)


def create_comparison_visualization(results_list):
    """Create comparative visualizations."""
    print("\n📊 Creating comparative visualizations...")

    variants = []
    phi_alignments = []
    phi_composite = []
    fractal_dims = []
    total_bits = []

    for results in results_list:
        if results:
            variants.append(results['variant'].replace('_', '\n'))
            phi_alignments.append(results.get('phi_alignment'))
            phi_composite.append(results.get('phi_alignment_score'))
            fractal_dims.append(results.get('fractal_dimension'))
            total_bits.append(results.get('total_bits'))

    if not variants:
        print("⚠️ No hi ha dades per visualitzar")
        return

    # Crear figura amb subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🌌 HSI Variants Comparison - Convergence to φ (1.618)', fontsize=16, fontweight='bold')

    # 1. φ-Alignment (simple) i composit
    colors1 = ['red' if (x is not None and x > 0.1) else 'orange' if (x is not None and x > 0.01) else 'green' for x in phi_alignments]
    ax1.bar(variants, [x if x is not None else 0.0 for x in phi_alignments], color=colors1, alpha=0.6, label='|D-φ| (report)')
    if any(s is not None for s in phi_composite):
        ax1.plot(variants, [s if s is not None else float('nan') for s in phi_composite], color='black', marker='o', linestyle='--', label='Composite score')
    ax1.set_title('φ-Alignment (Lower = Better)')
    ax1.set_ylabel('Alignment Error')
    ax1.tick_params(axis='x', rotation=45)
    ax1.axhline(y=0.01, color='green', linestyle='--', alpha=0.7, label='Excellent threshold')
    ax1.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Good threshold')
    ax1.legend()

    # 2. Fractal dimension
    ax2.bar(variants, [x if x is not None else 0.0 for x in fractal_dims], color='blue', alpha=0.7)
    ax2.set_title('Fractal Dimension')
    ax2.set_ylabel('Dimension')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=1.618, color='gold', linestyle='-', linewidth=2, label='φ = 1.618')
    ax2.legend()

    # 3. Total information generated
    ax3.bar(variants, [x if x is not None else 0.0 for x in total_bits], color='purple', alpha=0.7)
    ax3.set_title('Total Information Generated')
    ax3.set_ylabel('Bits')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_yscale('log')

    # 4. Scatter plot: Fractal Dim vs φ-Alignment (composite if available)
    y_scatter = [s if s is not None else a for s, a in zip(phi_composite, phi_alignments)]
    colors = ['green' if (y is not None and y < 0.01) else 'orange' if (y is not None and y < 0.1) else 'red' for y in y_scatter]
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run basal‑pure variants and/or generate plots")
    parser.add_argument("--iterations", "-i", type=int, default=None,
                        help="Number of iterations to pass to the generator (env HSI_ITERATIONS otherwise)")
    # Variant selection
    parser.add_argument("--variant", "-v", type=str, default=None,
                        help="Run only this variant (A, B, D, E, F, G, H, I). A is random control, I is inverse of E.")
    parser.add_argument("--include-control", action="store_true",
                        help="Include Variant A (Random Control) in the run")
    # Plot-only focused flags (prefer CLI over envs)
    parser.add_argument("--plot-only", action="store_true", help="Do not run variants; just load latest report and generate plots")
    parser.add_argument("--report-iter", type=int, default=None, help="Target iteration for plot-only")
    parser.add_argument("--report-path", type=str, default=None, help="Explicit path to a report JSON for plot-only")
    parser.add_argument("--hilbert-bits", type=int, default=None, help="Override config.hilbert.max_bits")
    parser.add_argument("--fft-bits", type=int, default=None, help="Override bits for FFT spectrum")
    parser.add_argument("--autocorr-bits", type=int, default=None, help="Override bits for autocorrelation")
    parser.add_argument("--raster-bits", type=int, default=None, help="Override bits for 2D raster plot")
    parser.add_argument("--beta-bits", type=int, default=None, help="Override bits for spectrum β-fit")
    # Skip flags (alternative to env vars HSI_NO_*)
    parser.add_argument("--no-growth", action="store_true", help="Skip growth plots")
    parser.add_argument("--no-raster", action="store_true", help="Skip 2D raster plot")
    parser.add_argument("--no-hilbert", action="store_true", help="Skip Hilbert heatmap")
    parser.add_argument("--no-fft", action="store_true", help="Skip FFT spectrum")
    parser.add_argument("--no-autocorr", action="store_true", help="Skip autocorrelation")
    parser.add_argument("--no-beta", action="store_true", help="Skip spectrum β-fit")
    parser.add_argument("--only", type=str, default=None,
                        help="Generate ONLY this plot type: growth, raster, hilbert, fft, autocorr, beta")
    return parser.parse_args()

    ax4.scatter(fractal_dims, y_scatter, c=colors, s=100, alpha=0.7)
    ax4.set_xlabel('Fractal Dimension')
    ax4.set_ylabel('φ-Alignment Error')
    ax4.set_title('Fractal Dimension vs φ-Alignment (prefers composite)')
    ax4.axvline(x=1.618, color='gold', linestyle='-', alpha=0.7, label='φ = 1.618')
    ax4.axhline(y=0.01, color='green', linestyle='--', alpha=0.7, label='Excellent')
    ax4.legend()

    # Afegir etiquetes als punts
    for i, variant in enumerate([v.replace('\n', '_') for v in variants]):
        ax4.annotate(variant.split('_')[0], (fractal_dims[i], y_scatter[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.tight_layout()
    plt.savefig('hsi_variants_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("📊 Visualitzacions guardades a 'hsi_variants_comparison.png'")

def main():
    """Main entry: run all variants and compare results."""
    args = parse_args()

    print("=" * 60)
    print("*** HSI VARIANTS MASTER RUNNER ***")
    print("*** Exploring convergence toward φ ***")
    print("=" * 60)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Define all available variants
    # Variant A is Random Control (separate script), others use the main generator
    all_variants = [
        ("-m", "hsi_agents_project.level0_random_control", "Variant A (Random Control)", "variant_A_results.json", "A"),
        ("-m", "hsi_agents_project.level0.generator", "Variant B (Stratified Baseline)", "variant_B_results.json", "B"),
        ("-m", "hsi_agents_project.level0.generator", "Variant D (Minimal Asymmetry)", "variant_D_results.json", "D"),
        ("-m", "hsi_agents_project.level0.generator", "Variant I (Inverse Order: 10→01)", "variant_I_results.json", "I"),
        ("-m", "hsi_agents_project.level0.generator", "Variant E (Ordered Passes)", "variant_E_results.json", "E"),
        ("-m", "hsi_agents_project.level0.generator", "Variant F (Hybrid)", "variant_F_results.json", "F"),
        ("-m", "hsi_agents_project.level0.generator", "Variant G (Raw Stratified)", "variant_G_results.json", "G"),
        ("-m", "hsi_agents_project.level0.generator", "Variant H (Continuous Feedback)", "variant_H_results.json", "H")
    ]

    # Filter variants based on CLI args
    if args.variant:
        # Run only the specified variant
        variant_code = args.variant.upper()
        variants = [v for v in all_variants if v[4] == variant_code]
        if not variants:
            print(f"❌ ERROR: Unknown variant '{variant_code}'")
            print(f"   Available variants: A, B, D, E, F, G, H, I")
            return
        print(f"🎯 Running only Variant {variant_code}")
    elif args.include_control:
        # Include all variants (with A)
        variants = all_variants
        print("🎯 Running all variants INCLUDING Variant A (Random Control)")
    else:
        # Default: exclude A (random control)
        variants = [v for v in all_variants if v[4] != 'A']
        print("🎯 Running HSI variants (B, D, E, F, G, H). Use --include-control to add Variant A.")

    execution_results = []
    variant_results = []

    # Executar totes les variants
    total_start_time = time.time()

    for flag, mod, name, results_file, variant_code in variants:
        success, exec_time, output = run_variant_script(flag, mod, name, results_file, variant_code, args.iterations)
        execution_results.append({
            'variant': name,
            'success': success,
            'execution_time': exec_time,
            'output_preview': output[:200] + "..." if len(output) > 200 else output
        })

        # Load results if execution was successful
        if success:
            print(f"\n📂 Loading results for variant {variant_code}...", flush=True)
            results_path = find_variant_result_file(variant_code)
            results = load_variant_results(results_path)
            # Try to keep a back-pointer to the report path for enrichment
            if results is not None:
                results['report_path'] = results_path
                print(f"   ✅ Results loaded from: {Path(results_path).name}", flush=True)
            variant_results.append(results)
        else:
            variant_results.append(None)

    total_execution_time = time.time() - total_start_time

    # Show execution summary
    print(f"\n⏱️ TOTAL EXECUTION TIME: {total_execution_time:.2f}s", flush=True)
    print(f"✅ Successful variants: {sum(1 for r in execution_results if r['success'])}/{len(variants)}", flush=True)

    # Compare convergence
    print("\n📊 Comparing φ-convergence across variants...", flush=True)
    best_variant = compare_phi_convergence(variant_results)

    # Visualizations (post-generation: minimal set, use plot-only for full control)
    print("\n🎨 Starting visualization generation...", flush=True)

    # Read skip flags from environment
    skip_growth = os.environ.get("HSI_NO_GROWTH") == "1"
    skip_raster = os.environ.get("HSI_NO_RASTER") == "1"
    skip_beta = os.environ.get("HSI_NO_BETA") == "1"

    try:
        fft_prefix = int(os.environ.get("HSI_FFT_PREFIX", "2000000"))
    except Exception:
        fft_prefix = 2_000_000

    print("   [1/5] Comparison visualization...", flush=True)
    create_comparison_visualization(variant_results)

    if skip_growth:
        print("   [2/5] Growth plots SKIPPED", flush=True)
    else:
        print("   [2/5] Growth and time plots...", flush=True)
        plot_growth_and_time(variant_results)

    beta_stats = {}
    if skip_beta:
        print("   [3/5] Spectrum β-fit SKIPPED", flush=True)
    else:
        print("   [3/5] Spectrum β-fit analysis...", flush=True)
        beta_stats = plot_spectrum_beta_fit(variant_results, max_bits=fft_prefix, skip=False)

    # Attach growth and spectral metrics
    print("   [4/5] Attaching growth metrics & saving enriched reports...", flush=True)
    _attach_growth_metrics(variant_results, beta_stats)
    for r in variant_results:
        if r:
            _write_enriched_variant_report(r)

    if skip_raster:
        print("   [5/5] 2D raster SKIPPED", flush=True)
    else:
        print("   [5/5] 2D raster plot...", flush=True)
        plot_raster2d(variant_results, max_iter_cap=None)

    # Save master summary with variants in filename to avoid overwrites
    master_results = {
        'execution_summary': execution_results,
        'best_variant': best_variant,
        'total_execution_time': total_execution_time,
        'timestamp': datetime.now().isoformat(),
        'variants_analyzed': len(variants),
        'successful_executions': sum(1 for r in execution_results if r['success']),
        'spectrum_beta': beta_stats
    }

    # Build filename with variant codes (e.g., hsi_master_results_B_E_I.json)
    # variants is a list of tuples; extract the code (element 4) from each
    variant_codes = '_'.join(sorted([v[4] for v in variants]))
    master_results_dir = RESULTS_DIR / "level0" / "reports"
    master_results_dir.mkdir(parents=True, exist_ok=True)
    master_results_path = master_results_dir / f'hsi_master_results_{variant_codes}.json'
    with open(master_results_path, 'w') as f:
        json.dump(master_results, f, indent=2)
    print(f"💾 Saved: {master_results_path.relative_to(BASE_PATH)}")

    # Export metadata and compress for Level 1 (if enabled in config)
    try:
        cfg = load_project_config("config.json")
        level1_cfg = cfg.get('level1', {})

        if level1_cfg.get('export_metadata', True):
            print("\n📦 Exporting metadata for Level 1...")
            for r in variant_results:
                if r:
                    variant_code = r.get('variant', '').replace('Variant_', '')
                    export_metadata_for_level1(variant_code, r)

        if level1_cfg.get('enable_tar_compression', True):
            print("\n🗜️ Compressing iteration snapshots...")
            for r in variant_results:
                if r:
                    variant_code = r.get('variant', '').replace('Variant_', '')
                    compress_iterations_to_tar(variant_code)
    except Exception as e:
        print(f"[WARN] Level 1 export failed: {e}")

def _run_plot_only():
    """Plot-only mode: load report(s) and generate charts for a specific variant.
    Priority of inputs: CLI flags (preferred) > env vars > auto-detect latest.
    """
    import time as _time
    _load_dotenv_if_present()
    args = parse_args()
    # Variant: prefer explicit CLI
    var = (args.variant or os.environ.get("HSI_VARIANT_CODE") or "B").upper()
    # Prefix caps: prefer CLI > env > config
    if args.hilbert_bits is not None:
        hilbert_prefix = int(args.hilbert_bits)
    else:
        env_hilbert = os.environ.get("HSI_HILBERT_PREFIX")
        if env_hilbert:
            try:
                hilbert_prefix = int(env_hilbert)
            except Exception:
                hilbert_prefix = _get_hilbert_config()['max_bits']
        else:
            hilbert_prefix = _get_hilbert_config()['max_bits']
    # FFT prefix
    if args.fft_bits is not None:
        fft_prefix = int(args.fft_bits)
    else:
        try:
            fft_prefix = int(os.environ.get("HSI_FFT_PREFIX", "2000000"))
        except Exception:
            fft_prefix = 2_000_000
    # Autocorrelation prefix (defaults to fft_prefix if not specified)
    if args.autocorr_bits is not None:
        autocorr_prefix = int(args.autocorr_bits)
    else:
        try:
            autocorr_prefix = int(os.environ.get("HSI_AUTOCORR_PREFIX", str(fft_prefix)))
        except Exception:
            autocorr_prefix = fft_prefix
    # Beta-fit prefix (defaults to fft_prefix if not specified)
    if args.beta_bits is not None:
        beta_prefix = int(args.beta_bits)
    else:
        try:
            beta_prefix = int(os.environ.get("HSI_BETA_PREFIX", str(fft_prefix)))
        except Exception:
            beta_prefix = fft_prefix
    # Raster prefix
    if args.raster_bits is not None:
        raster_prefix = int(args.raster_bits)
    else:
        try:
            raster_prefix = int(os.environ.get("HSI_RASTER_PREFIX", "1000000"))
        except Exception:
            raster_prefix = 1_000_000

    # Skip flags: CLI > env
    only_mode = args.only.lower() if args.only else None
    if only_mode:
        # If --only is set, skip everything except that plot
        skip_g = only_mode != "growth"
        skip_r = only_mode != "raster"
        skip_h = only_mode != "hilbert"
        skip_f = only_mode != "fft"
        skip_b = only_mode != "beta"
        skip_a = only_mode != "autocorr"
        skip_rep = True  # Always skip report in --only mode
    else:
        skip_g = args.no_growth or os.environ.get("HSI_NO_GROWTH") == "1"
        skip_r = args.no_raster or os.environ.get("HSI_NO_RASTER") == "1"
        skip_h = args.no_hilbert or os.environ.get("HSI_NO_HILBERT") == "1"
        skip_f = args.no_fft or os.environ.get("HSI_NO_FFT") == "1"
        skip_b = args.no_beta or os.environ.get("HSI_NO_BETA") == "1"
        skip_a = args.no_autocorr or os.environ.get("HSI_NO_AUTOCORR") == "1"
        skip_rep = os.environ.get("HSI_NO_REPORT") == "1"

    print("\n" + "=" * 60, flush=True)
    print("📊 PLOT-ONLY MODE — Generating visualizations", flush=True)
    print("=" * 60, flush=True)

    # Allow direct report path/iteration via CLI or env
    print(f"[1/8] 🔍 Locating report for variant {var}...", flush=True)
    t0 = _time.perf_counter()
    if args.report_path and os.path.exists(args.report_path):
        path = args.report_path
    else:
        report_path_env = os.environ.get("HSI_REPORT_PATH")
        if report_path_env and os.path.exists(report_path_env):
            path = report_path_env
        else:
            target_iter = args.report_iter or (int(os.environ.get("HSI_REPORT_ITER")) if os.environ.get("HSI_REPORT_ITER") else None)
            path = find_variant_result_file(var, iterations=target_iter)

    print(f"[1/8] 📂 Loading report JSON...", flush=True)
    res = load_variant_results(path)
    if res is not None:
        res['report_path'] = path
    if not res:
        try:
            rel = Path(path).resolve().relative_to(BASE_PATH)
        except Exception:
            rel = path
        print(f"[PLOT-ONLY] ❌ No report found for variant {var} at {rel}")
        return
    print(f"[1/8] ✅ Report loaded in {_time.perf_counter()-t0:.1f}s", flush=True)

    # Compact path in log
    try:
        rel = Path(path).resolve().relative_to(BASE_PATH)
    except Exception:
        rel = path

    # Override iteration if user requested a specific one different from report
    target_iter = args.report_iter or (int(os.environ.get("HSI_REPORT_ITER")) if os.environ.get("HSI_REPORT_ITER") else None)
    report_iter = res.get('iterations', 18)

    if target_iter and target_iter != report_iter:
        # Check if snapshot exists for target iteration
        snapshot_dir = BASE_PATH / "results" / "level0" / "phi_snapshots" / f"var_{var}"
        snapshot_file = snapshot_dir / f"phi_iter{target_iter}.struct.gz"
        if snapshot_file.exists():
            print(f"       [override] Using iteration {target_iter} instead of report's {report_iter}", flush=True)
            res['iterations'] = target_iter
            res['_target_iteration'] = target_iter  # Mark for plot functions
        else:
            print(f"       [warn] No snapshot for iter {target_iter}, using report's iter {report_iter}", flush=True)

    iters = res.get('iterations', '?')
    phi_len = res.get('phi_length', 0)
    print(f"       Variant: {var} | Iterations: {iters} | Φ length: {phi_len:,} bits", flush=True)
    print("", flush=True)

    results_list = [res]
    beta_stats = []  # Initialize for report step

    # Each plot step with timing
    if skip_g:
        print(f"[2/8] ⏭️  Growth plots SKIPPED", flush=True)
    else:
        print(f"[2/8] 📈 Generating growth & time plots...", flush=True)
        t0 = _time.perf_counter()
        plot_growth_and_time(results_list)
        print(f"[2/8] ✅ Done in {_time.perf_counter()-t0:.1f}s", flush=True)

    if skip_r:
        print(f"[3/8] ⏭️  2D raster SKIPPED", flush=True)
    else:
        print(f"[3/8] 🖼️  Generating 2D raster...", flush=True)
        t0 = _time.perf_counter()
        plot_raster2d(results_list, max_iter_cap=None)
        print(f"[3/8] ✅ Done in {_time.perf_counter()-t0:.1f}s", flush=True)

    if skip_h:
        print(f"[4/8] ⏭️  Hilbert heatmap SKIPPED", flush=True)
    else:
        print(f"[4/8] 🌀 Generating Hilbert heatmap ({hilbert_prefix:,} bits)...", flush=True)
        t0 = _time.perf_counter()
        plot_hilbert_heatmap(results_list, hilbert_prefix, skip_h)
        print(f"[4/8] ✅ Done in {_time.perf_counter()-t0:.1f}s", flush=True)

    if skip_f:
        print(f"[5/8] ⏭️  FFT spectrum SKIPPED", flush=True)
    else:
        print(f"[5/8] 📡 Computing FFT spectrum ({fft_prefix:,} bits)...", flush=True)
        t0 = _time.perf_counter()
        plot_fft(results_list, fft_prefix, skip_f)
        print(f"[5/8] ✅ Done in {_time.perf_counter()-t0:.1f}s", flush=True)

    if skip_b:
        print(f"[6/8] ⏭️  Spectrum β-fit SKIPPED", flush=True)
    else:
        print(f"[6/8] 📊 Computing spectrum β-fit ({beta_prefix:,} bits)...", flush=True)
        t0 = _time.perf_counter()
        beta_stats = plot_spectrum_beta_fit(results_list, max_bits=beta_prefix, skip=skip_b)
        print(f"[6/8] ✅ Done in {_time.perf_counter()-t0:.1f}s", flush=True)

    if skip_a:
        print(f"[7/8] ⏭️  Autocorrelation SKIPPED", flush=True)
    else:
        print(f"[7/8] 🔄 Computing autocorrelation ({autocorr_prefix:,} bits)...", flush=True)
        t0 = _time.perf_counter()
        plot_autocorrelation(results_list, autocorr_prefix, skip_a)
        print(f"[7/8] ✅ Done in {_time.perf_counter()-t0:.1f}s", flush=True)

    if skip_rep:
        print(f"[8/8] ⏭️  Report SKIPPED", flush=True)
    else:
        print(f"[8/8] 💾 Writing enriched report...", flush=True)
        t0 = _time.perf_counter()
        _attach_growth_metrics(results_list, beta_stats)
        for r in results_list:
            if r:
                _write_enriched_variant_report(r)
        print(f"[8/8] ✅ Done in {_time.perf_counter()-t0:.1f}s", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("[PLOT-ONLY] ✅ All visualizations complete!", flush=True)

    # Missatge final (plot-only: no best_variant context)
    print("\n" + "🌌" * 20)
    print("Plots generated for variant:", var)

    print("🌌" * 20)
    print("📁 All results saved for further analysis")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# Level 1 Data Export Functions
# ============================================================================

def export_metadata_for_level1(variant_code: str, results: dict):
    """
    Export metadata.json to results/level0/reports/ directory.

    Parameters:
    -----------
    variant_code : str
        Variant code (e.g., 'B', 'D')
    results : dict
        Results dictionary from variant execution
    """
    try:
        reports_dir = RESULTS_DIR / "level0" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "variant": variant_code.upper(),
            "iterations": results.get('iterations', 0),
            "total_bits": results.get('phi_length', 0),
            "absolute_token": results.get('absolute_token', '10'),
            "timestamp": datetime.now().isoformat(),
            "phi_lengths": results.get('phi_lengths_per_iteration', []),
            "compression_ratio": results.get('compression_ratio', 1.0),
            "fractal_dimension": results.get('fractal_dimension'),
            "phi_alignment": results.get('phi_alignment'),
            "execution_time_seconds": results.get('execution_time', 0)
        }

        # Save with variant name in filename for clarity
        metadata_path = reports_dir / f"metadata_{variant_code.upper()}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"[INFO] Metadata exported: {metadata_path.relative_to(BASE_PATH)}")
        return True

    except Exception as e:
        print(f"[WARN] Failed to export metadata for variant {variant_code}: {e}")
        return False


def compress_iterations_to_tar(variant_code: str):
    """
    Compress all iteration snapshots into a single tar.gz file.

    Parameters:
    -----------
    variant_code : str
        Variant code (e.g., 'B', 'D')
    """
    try:
        import tarfile
        import gzip

        # v33: snapshots are in level0/phi_snapshots/var_X/
        var_dir = BASE_PATH / "results" / "level0" / "phi_snapshots" / f"var_{variant_code.upper()}"
        if not var_dir.exists():
            print(f"[WARN] Variant directory not found: {var_dir}")
            return False

        # Find all phi_iter*.struct.gz files (v33 format)
        iter_files = sorted(var_dir.glob("phi_iter*.struct.gz"))
        if not iter_files:
            # Fallback: try legacy format
            iter_files = sorted(var_dir.glob("phi_iter_*.txt.gz"))
        if not iter_files:
            print(f"[WARN] No iteration files found in {var_dir}")
            return False

        # Create tar.gz archive
        tar_path = var_dir / "phi_complete.tar.gz"
        print(f"[INFO] Compressing {len(iter_files)} iteration files...")

        with tarfile.open(tar_path, 'w:gz') as tar:
            for iter_file in iter_files:
                tar.add(iter_file, arcname=iter_file.name)

        # Calculate compression stats
        original_size = sum(f.stat().st_size for f in iter_files)
        compressed_size = tar_path.stat().st_size
        ratio = compressed_size / original_size if original_size > 0 else 1.0

        print(f"[INFO] Compressed {len(iter_files)} files:")
        print(f"       Original: {original_size / 1e9:.2f} GB")
        print(f"       Compressed: {compressed_size / 1e9:.2f} GB")
        print(f"       Ratio: {ratio:.2%}")
        print(f"       Saved: {tar_path.relative_to(BASE_PATH)}")

        return True

    except Exception as e:
        print(f"[WARN] Failed to compress iterations for variant {variant_code}: {e}")
        return False


if __name__ == "__main__":
    args = parse_args()
    # Check both CLI flag and environment variable for plot-only mode
    is_plot_only = args.plot_only or os.environ.get("HSI_PLOT_ONLY") == "1"
    if is_plot_only:
        _run_plot_only()
    else:
        main()

    # =========================================================================
    # 💡 POST-EXECUTION ANALYSIS
    # =========================================================================
    # After running experiments, analyze the generated results with:
    #
    #   python analyze_results.py                     # Analyze latest results
    #   python analyze_results.py --variant B -i 18  # Specific variant/iteration
    #   python analyze_results.py --output report.md # Save Markdown report
    #
    # The analyze_results.py tool generates human-readable reports suitable
    # for scientific documentation and paper enrichment.
    # =========================================================================
