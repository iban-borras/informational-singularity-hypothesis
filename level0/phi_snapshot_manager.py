"""
PhiSnapshotManager — Compression and storage manager for Φ sequences

Implements the compression strategy defined in level0_data_storage.md to enable
high-iteration simulations without exceeding RAM limits.

Author: Iban Borràs with collaboration from Augment Agent (Sophia)
Date: Jan 2025
"""

import gzip
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import os


class PhiSnapshotManager:
    """
    Gestor de snapshots comprimits per seqüències Φ.

    Implementa la estratègia de compressió automàtica quan la seqüència
    supera els límits de memòria definits.
    """

    def __init__(self,
                 data_dir: str = None,
                 memory_threshold: int = 10**8,  # 100MB en bits
                 compression_level: int = 9):
        """
        Inicialitza el gestor de snapshots.

        Args:
            data_dir: Directori on guardar els snapshots
            memory_threshold: Llindar per activar compressió (en nombre de bits)
            compression_level: Nivell de compressió gzip (1-9)
        """
        # Default to project results/snapshots dir if not passed
        if data_dir is None:
            from pathlib import Path as _P
            self.data_dir = _P(__file__).resolve().parent.parent / "results" / "snapshots"
        else:
            self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.memory_threshold = memory_threshold
        self.compression_level = compression_level
        # Optional environment-driven tuning
        try:
            self.snapshot_stats_every = max(0, int(os.environ.get("HSI_SNAPSHOT_STATS_EVERY", "1")))
        except Exception:
            self.snapshot_stats_every = 1
        try:
            level = int(os.environ.get("HSI_COMPRESSION_LEVEL", str(compression_level)))
            if 1 <= level <= 9:
                self.compression_level = level
        except Exception:
            pass

        self.use_compression = False

    def should_use_compression(self, sequence_length: int) -> bool:
        """Determina si cal usar compressió basant-se en la longitud."""
        return sequence_length >= self.memory_threshold

    def save_phi_state(self,
                       bits: str,
                       iteration: int,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Guarda l'estat Φ amb compressió automàtica si cal.

        Args:
            bits: Seqüència binària com a string
            iteration: Número d'iteració
            metadata: Metadades addicionals

        Returns:
            Diccionari amb informació del desament
        """
        start_time = time.time()
        original_size = len(bits)

        # Determine compression
        use_compression = self.should_use_compression(original_size)

        # Prepare metadata
        save_metadata = {
            "iteration": iteration,
            "sequence_length": original_size,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "compressed": use_compression,
            "compression_level": self.compression_level if use_compression else None,
            "data_directory": str(self.data_dir),
            **(metadata or {})
        }

        # Save sequence
        if use_compression:
            # Compressed save
            bin_path = self.data_dir / f"phi_iter{iteration}.bin.gz"
            # Simple progress: chunked write
            chunk = 1_000_000  # 1MB chunks approx (in chars)
            written = 0
            with gzip.open(bin_path, "wt", encoding="utf-8", compresslevel=self.compression_level) as f:
                for i in range(0, len(bits), chunk):
                    f.write(bits[i:i+chunk])
                    written += min(chunk, len(bits) - i)
                    if os.environ.get("HSI_COMPRESS_LOG", "1") == "1":
                        percent = 100.0 * written / len(bits)
                        print(f"[compress] iter={iteration} progress {percent:5.1f}% ({written/1e6:.1f}MB)", end="\r")
            if os.environ.get("HSI_COMPRESS_LOG", "1") == "1":
                print()

            # Compression ratio
            compressed_size = bin_path.stat().st_size
            compression_ratio = compressed_size / (original_size / 8)  # Approximate
            save_metadata["compressed_size_bytes"] = compressed_size
            save_metadata["compression_ratio"] = compression_ratio

        else:
            # Before threshold: allow uncompressed small save
            bin_path = self.data_dir / f"phi_iter{iteration}.bin"
            with open(bin_path, "w", encoding="utf-8") as f:
                f.write(bits)
            save_metadata["file_size_bytes"] = bin_path.stat().st_size

        # Guardar metadades
        json_path = self.data_dir / f"phi_iter{iteration}.json"
        with open(json_path, "w") as f:
            json.dump(save_metadata, f, indent=2)

        # Calcular estadístiques de rendiment
        save_time = time.time() - start_time
        save_metadata["save_time_seconds"] = save_time

        return save_metadata

    def save_phi_state_structural(self,
                                   phi_str: str,
                                   iteration: int,
                                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Guarda l'estat Φ amb estructura (parentheses) en format v33 (2-bit encoding).

        Args:
            phi_str: Seqüència amb estructura ('0', '1', '(', ')')
            iteration: Número d'iteració
            metadata: Metadades addicionals

        Returns:
            Diccionari amb informació del desament
        """
        # Import directly to avoid circular dependency
        import sys
        import os
        # Add parent directory to path temporarily
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from utils.bitarray_encoder import save_phi_structural_gz, get_format_info

        start_time = time.time()
        original_length = len(phi_str)

        # Get format info (now memory-optimized, works for any size)
        format_info = get_format_info(phi_str)

        # Prepare metadata
        save_metadata = {
            "iteration": iteration,
            "sequence_length": original_length,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "format": "v33_structural",
            "encoding": "2bit",
            "compressed": True,
            "compression_level": self.compression_level,
            "data_directory": str(self.data_dir),
            "format_info": format_info,
            **(metadata or {})
        }

        # Save with structural encoding
        struct_path = self.data_dir / f"phi_iter{iteration}.struct.gz"
        compressed_size = save_phi_structural_gz(
            phi_str,
            str(struct_path),
            compresslevel=self.compression_level
        )

        save_metadata["compressed_size_bytes"] = compressed_size

        # Calculate compression ratio (estimate text_bytes if format_info not available)
        if format_info and format_info.get('text_bytes', 0) > 0:
            save_metadata["compression_ratio"] = compressed_size / format_info['text_bytes']
        else:
            # Estimate: 1 byte per char for ASCII (conservative estimate)
            estimated_text_bytes = original_length
            save_metadata["compression_ratio"] = compressed_size / estimated_text_bytes if estimated_text_bytes > 0 else 0.0

        # Guardar metadades (with explicit flush for subprocess visibility)
        json_path = self.data_dir / f"phi_iter{iteration}.json"
        with open(json_path, "w") as f:
            json.dump(save_metadata, f, indent=2)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk for Windows compatibility

        # Calcular estadístiques de rendiment
        save_time = time.time() - start_time
        save_metadata["save_time_seconds"] = save_time

        if self.snapshot_stats_every > 0 and iteration % self.snapshot_stats_every == 0:
            print(f"[v33] iter={iteration} saved structural format: {compressed_size/1024/1024:.2f} MB "
                  f"(ratio={save_metadata['compression_ratio']:.4f}, time={save_time:.2f}s)")

        return save_metadata

    def save_phi_state_structural_from_file(
        self,
        input_file_path: str,
        iteration: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        STREAMING VERSION: Guarda l'estat Φ des de fitxer sense carregar a RAM.

        Args:
            input_file_path: Camí al fitxer amb l'acumulació
            iteration: Número d'iteració
            metadata: Metadades addicionals

        Returns:
            Diccionari amb informació del desament
        """
        from pathlib import Path
        import sys
        import os

        parent_dir = os.path.dirname(os.path.dirname(__file__))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from utils.bitarray_encoder import save_phi_structural_gz_from_file

        start_time = time.time()
        input_path = Path(input_file_path)
        original_length = input_path.stat().st_size  # Approximate char count

        # Prepare metadata
        save_metadata = {
            "iteration": iteration,
            "sequence_length": original_length,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "format": "v33_structural_streaming",
            "encoding": "2bit",
            "compressed": True,
            "compression_level": self.compression_level,
            "data_directory": str(self.data_dir),
            "source_file": str(input_path),
            **(metadata or {})
        }

        # Save with streaming structural encoding
        struct_path = self.data_dir / f"phi_iter{iteration}.struct.gz"
        compressed_size = save_phi_structural_gz_from_file(
            str(input_path),
            str(struct_path),
            compresslevel=self.compression_level
        )

        save_metadata["compressed_size_bytes"] = compressed_size
        save_metadata["compression_ratio"] = compressed_size / original_length if original_length > 0 else 0.0

        # Guardar metadades (with explicit flush for subprocess visibility)
        json_path = self.data_dir / f"phi_iter{iteration}.json"
        with open(json_path, "w") as f:
            json.dump(save_metadata, f, indent=2)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk for Windows compatibility

        save_time = time.time() - start_time
        save_metadata["save_time_seconds"] = save_time

        if self.snapshot_stats_every > 0 and iteration % self.snapshot_stats_every == 0:
            print(f"[v33-streaming] iter={iteration} saved: {compressed_size/1024/1024:.2f} MB "
                  f"(ratio={save_metadata['compression_ratio']:.4f}, time={save_time:.2f}s)")

        return save_metadata

    def load_phi_state(self, iteration: int) -> Tuple[str, Dict[str, Any]]:
        """
        Carrega l'estat Φ d'una iteració específica.

        Args:
            iteration: Número d'iteració a carregar

        Returns:
            Tuple amb (seqüència_bits, metadades)
        """
        start_time = time.time()

        # Carregar metadades
        json_path = self.data_dir / f"phi_iter{iteration}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"No s'ha trobat l'iteració {iteration}")

        with open(json_path, "r") as f:
            metadata = json.load(f)

        # Carregar seqüència
        if metadata.get("compressed", False):
            # Carregar comprimit
            bin_path = self.data_dir / f"phi_iter{iteration}.bin.gz"
            with gzip.open(bin_path, "rt", encoding="utf-8") as f:
                bits = f.read()
        else:
            # Carregar sense comprimir
            bin_path = self.data_dir / f"phi_iter{iteration}.bin"
            with open(bin_path, "r", encoding="utf-8") as f:
                bits = f.read()

        # Actualitzar metadades amb temps de càrrega
        load_time = time.time() - start_time
        metadata["load_time_seconds"] = load_time

        return bits, metadata

    def load_phi_state_structural(self, iteration: int) -> Tuple[str, Dict[str, Any]]:
        """
        Carrega l'estat Φ amb estructura (format v33) d'una iteració específica.

        Args:
            iteration: Número d'iteració a carregar

        Returns:
            Tuple amb (seqüència_amb_estructura, metadades)
        """
        # Import directly to avoid circular dependency
        import sys
        import os
        # Add parent directory to path temporarily
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from utils.bitarray_encoder import load_phi_structural_gz

        start_time = time.time()

        # Carregar metadades
        json_path = self.data_dir / f"phi_iter{iteration}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"No s'ha trobat l'iteració {iteration}")

        with open(json_path, "r") as f:
            metadata = json.load(f)

        # Verificar format
        if metadata.get("format") != "v33_structural":
            raise ValueError(f"Iteration {iteration} is not in v33 structural format (found: {metadata.get('format')})")

        # Carregar seqüència estructural
        struct_path = self.data_dir / f"phi_iter{iteration}.struct.gz"
        if not struct_path.exists():
            raise FileNotFoundError(f"Structural file not found: {struct_path}")

        phi_str = load_phi_structural_gz(str(struct_path))

        # Actualitzar metadades amb temps de càrrega
        load_time = time.time() - start_time
        metadata["load_time_seconds"] = load_time

        return phi_str, metadata

    def get_available_iterations(self) -> list:
        """Retorna una llista de les iteracions disponibles."""
        json_files = list(self.data_dir.glob("phi_iter*.json"))
        iterations = []

        for json_file in json_files:
            try:
                iteration_num = int(json_file.stem.replace("phi_iter", ""))
                iterations.append(iteration_num)
            except ValueError:
                continue

        return sorted(iterations)

    def get_storage_summary(self) -> Dict[str, Any]:
        """Retorna un resum de l'emmagatzematge actual."""
        iterations = self.get_available_iterations()

        if not iterations:
            return {"total_iterations": 0, "total_size_mb": 0}

        total_size = 0
        compressed_count = 0
        uncompressed_count = 0

        for iteration in iterations:
            json_path = self.data_dir / f"phi_iter{iteration}.json"
            with open(json_path, "r") as f:
                metadata = json.load(f)

            if metadata.get("compressed", False):
                compressed_count += 1
                total_size += metadata.get("compressed_size_bytes", 0)
            else:
                uncompressed_count += 1
                total_size += metadata.get("file_size_bytes", 0)

        return {
            "total_iterations": len(iterations),
            "iterations_range": f"{min(iterations)}-{max(iterations)}" if iterations else "None",
            "compressed_snapshots": compressed_count,
            "uncompressed_snapshots": uncompressed_count,
            "total_size_mb": total_size / (1024 * 1024),
            "data_directory": str(self.data_dir)
        }

    def cleanup_old_snapshots(self, keep_last_n: int = 5) -> int:
        """
        Neteja snapshots antics, mantenint només els últims N.

        Args:
            keep_last_n: Nombre de snapshots a mantenir

        Returns:
            Nombre de snapshots eliminats
        """
        iterations = self.get_available_iterations()

        if len(iterations) <= keep_last_n:
            return 0

        to_delete = iterations[:-keep_last_n]
        deleted_count = 0

        for iteration in to_delete:
            # Eliminar fitxers binaris
            for ext in [".bin", ".bin.gz"]:
                bin_path = self.data_dir / f"phi_iter{iteration}{ext}"
                if bin_path.exists():
                    bin_path.unlink()
                    deleted_count += 1

            # Eliminar metadades
            json_path = self.data_dir / f"phi_iter{iteration}.json"
            if json_path.exists():
                json_path.unlink()

        return deleted_count

    def estimate_memory_usage(self, target_iteration: int) -> Dict[str, float]:
        """
        Estima l'ús de memòria per una iteració objectiu.

        Args:
            target_iteration: Iteració objectiu

        Returns:
            Diccionari amb estimacions en MB
        """
        # Creixement observat: ~×2.828 per iteració
        growth_factor = 2.828

        # Longitud estimada basada en el creixement observat
        # Iteració 16: ~41 milions de bits
        base_length = 41_000_000
        base_iteration = 16

        if target_iteration <= base_iteration:
            estimated_length = base_length * (growth_factor ** (target_iteration - base_iteration))
        else:
            estimated_length = base_length * (growth_factor ** (target_iteration - base_iteration))

        # Convertir a MB (assumint 1 bit = 1 byte en string)
        uncompressed_mb = estimated_length / (1024 * 1024)
        compressed_mb = uncompressed_mb * 0.15  # ~85% compressió

        return {
            "iteration": target_iteration,
            "estimated_bits": int(estimated_length),
            "uncompressed_mb": uncompressed_mb,
            "compressed_mb": compressed_mb,
            "recommended_compression": uncompressed_mb > 100
        }


# Funcions d'utilitat per compatibilitat
def create_snapshot_manager(data_dir: str = "data/phi_snapshots") -> PhiSnapshotManager:
    """Crea una instància del gestor de snapshots."""
    return PhiSnapshotManager(data_dir)


def estimate_iteration_requirements(max_iteration: int) -> Dict[str, Any]:
    """Estima els requisits per arribar a una iteració màxima."""
    manager = PhiSnapshotManager()

    estimates = []
    total_compressed_mb = 0

    for i in range(16, max_iteration + 1):
        estimate = manager.estimate_memory_usage(i)
        estimates.append(estimate)

        if estimate["recommended_compression"]:
            total_compressed_mb += estimate["compressed_mb"]

    return {
        "max_iteration": max_iteration,
        "total_estimated_storage_mb": total_compressed_mb,
        "total_estimated_storage_gb": total_compressed_mb / 1024,
        "compression_starts_at": next((e["iteration"] for e in estimates if e["recommended_compression"]), None),
        "detailed_estimates": estimates
    }
