#!/usr/bin/env python3
"""
BENCHMARK SCALABILITÀ - Progetto QuantPivot Gruppo 6
Esegue test su dataset di dimensioni crescenti per tutte e 3 le versioni.
Testa sia C puro che ASM per ogni versione (6 configurazioni totali).
Genera dataset al volo e usa direttamente le librerie Python.
"""

import sys
import os
import time
import numpy as np
import pyfftw
import subprocess

# Configurazione
DATASET_SIZES = [1000, 2000, 10000, 100000, 250000, 500000]
D = 256
NQ = 2000
H = 16
K = 8
X = 64

# Risultati: {size: {version_variant: (fit, prd)}}
# version_variant: "32_C", "32_ASM", "64_C", "64_ASM", "64omp_C", "64omp_ASM"
results = {}

def format_size(n):
    """Formatta dimensione in formato leggibile."""
    if n >= 1000000:
        return f"{n//1000000}M"
    elif n >= 1000:
        return f"{n//1000}K"
    return str(n)

def generate_dataset(n, d, dtype, alignment):
    """Genera dataset random allineato."""
    data = pyfftw.empty_aligned((n, d), dtype=dtype, n=alignment)
    data[:] = np.random.randn(n, d).astype(dtype)
    return data

def rebuild_module(version, use_asm):
    """Ricompila modulo con/senza ASM."""
    env = os.environ.copy()
    
    if version == "32":
        env["USE_ASM_32"] = "1" if use_asm else "0"
    elif version == "64":
        env["USE_ASM_64"] = "1" if use_asm else "0"
    elif version == "64omp":
        env["USE_ASM_OMP"] = "1" if use_asm else "0"
    
    # Rimuovi moduli cached
    import sys
    to_remove = [k for k in sys.modules.keys() if 'quantpivot' in k.lower()]
    for key in to_remove:
        del sys.modules[key]
    
    # Ricompila
    os.chdir("ProgettoGruppo6")
    result = subprocess.run(
        ["python3", "setup.py", "build_ext", "--inplace"],
        env=env,
        capture_output=True,
        text=True
    )
    os.chdir("..")
    
    if result.returncode != 0:
        print(f"ERRORE compilazione {version} {'ASM' if use_asm else 'C'}:")
        print(result.stderr)
        return False
    return True

def run_test_32(dataset, query, use_asm):
    """Esegue test versione 32-bit (C o ASM)."""
    try:
        if not rebuild_module("32", use_asm):
            return None
        
        from gruppo6.quantpivot32 import QuantPivot
        
        model = QuantPivot()
        
        start = time.perf_counter()
        model.fit(dataset, H, X, True)  # silent=True
        fit_time = time.perf_counter() - start
        
        start = time.perf_counter()
        ids, dists = model.predict(query, K, True)  # silent=True
        prd_time = time.perf_counter() - start
        
        return (fit_time, prd_time)
    except Exception as e:
        print(f"ERRORE 32 {'ASM' if use_asm else 'C'}: {e}")
        return None

def run_test_64(dataset, query, use_asm):
    """Esegue test versione 64-bit (C o ASM)."""
    try:
        if not rebuild_module("64", use_asm):
            return None
        
        from gruppo6.quantpivot64 import QuantPivot
        
        model = QuantPivot()
        
        start = time.perf_counter()
        model.fit(dataset, H, X, True)  # silent=True
        fit_time = time.perf_counter() - start
        
        start = time.perf_counter()
        ids, dists = model.predict(query, K, True)  # silent=True
        prd_time = time.perf_counter() - start
        
        return (fit_time, prd_time)
    except Exception as e:
        print(f"ERRORE 64 {'ASM' if use_asm else 'C'}: {e}")
        return None

def run_test_64omp(dataset, query, use_asm):
    """Esegue test versione 64-bit OpenMP (C o ASM)."""
    try:
        if not rebuild_module("64omp", use_asm):
            return None
        
        from gruppo6.quantpivot64omp import QuantPivot
        
        model = QuantPivot()
        
        start = time.perf_counter()
        model.fit(dataset, H, X, True)  # silent=True
        fit_time = time.perf_counter() - start
        
        start = time.perf_counter()
        ids, dists = model.predict(query, K, True)  # silent=True
        prd_time = time.perf_counter() - start
        
        return (fit_time, prd_time)
    except Exception as e:
        print(f"ERRORE 64omp {'ASM' if use_asm else 'C'}: {e}")
        return None

def print_table():
    """Stampa tabella riepilogativa: 3 righe (versioni) x 2 colonne (C/ASM) per ogni dataset."""
    
    print("\n" + "="*80)
    print("  BENCHMARK SCALABILITÀ - Progetto QuantPivot Gruppo 6")
    print(f"  Query: {NQ}, h={H}, k={K}, x={X}")
    print("="*80)
    
    for size in DATASET_SIZES:
        if size not in results:
            continue
        
        label = format_size(size)
        print(f"\n{'='*60}")
        print(f"  DATASET: {label} (N={size:,} × D={D})")
        print(f"{'='*60}")
        
        # Header
        print(f"{'Version':<15} | {'C Puro':<20} | {'ASM Optimized':<20}")
        print(f"{'':<15} | {'FIT / PRD (sec)':<20} | {'FIT / PRD (sec)':<20}")
        print("-"*60)
        
        # 32-bit
        row_32 = f"{'32-bit':<15} |"
        for variant in ["32_C", "32_ASM"]:
            if variant in results[size]:
                res = results[size][variant]
                if res is None:
                    row_32 += f" {'ERROR':<20}|"
                elif res == "SKIP":
                    row_32 += f" {'SKIP (OOM)':<20}|"
                else:
                    fit, prd = res
                    row_32 += f" {fit:>6.2f} / {prd:<10.2f} |"
            else:
                row_32 += f" {'-':<20}|"
        print(row_32)
        
        # 64-bit
        row_64 = f"{'64-bit':<15} |"
        for variant in ["64_C", "64_ASM"]:
            if variant in results[size]:
                res = results[size][variant]
                if res is None:
                    row_64 += f" {'ERROR':<20}|"
                elif res == "SKIP":
                    row_64 += f" {'SKIP (OOM)':<20}|"
                else:
                    fit, prd = res
                    row_64 += f" {fit:>6.2f} / {prd:<10.2f} |"
            else:
                row_64 += f" {'-':<20}|"
        print(row_64)
        
        # 64-bit OpenMP
        row_omp = f"{'64-bit OMP':<15} |"
        for variant in ["64omp_C", "64omp_ASM"]:
            if variant in results[size]:
                res = results[size][variant]
                if res is None:
                    row_omp += f" {'ERROR':<20}|"
                elif res == "SKIP":
                    row_omp += f" {'SKIP (OOM)':<20}|"
                else:
                    fit, prd = res
                    row_omp += f" {fit:>6.2f} / {prd:<10.2f} |"
            else:
                row_omp += f" {'-':<20}|"
        print(row_omp)
        
        print("-"*60)
        
        # Speedup per questo dataset
        print("\nSpeedup ASM vs C (predict only):")
        for version, label_v in [("32", "32-bit"), ("64", "64-bit"), ("64omp", "64-bit OMP")]:
            key_c = f"{version}_C"
            key_asm = f"{version}_ASM"
            if key_c in results[size] and key_asm in results[size]:
                res_c = results[size][key_c]
                res_asm = results[size][key_asm]
                if res_c and res_asm and res_c != "SKIP" and res_asm != "SKIP":
                    _, prd_c = res_c
                    _, prd_asm = res_asm
                    if prd_asm > 0:
                        speedup = prd_c / prd_asm
                        print(f"  {label_v:<15}: {speedup:.2f}x")
    
    print("\n" + "="*80)

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("="*80)
    print("BENCHMARK SCALABILITÀ - QuantPivot (C vs ASM)")
    print("="*80)
    print(f"Parametri: D={D}, NQ={NQ}, h={H}, k={K}, x={X}")
    print("="*80)
    
    # Genera query una volta sola
    print("\nGenerazione query allineate...")
    query_32 = generate_dataset(NQ, D, 'float32', 16)
    query_64 = generate_dataset(NQ, D, 'float64', 32)
    print(f"  Query 32-bit: {query_32.shape}, aligned={query_32.ctypes.data % 16 == 0}")
    print(f"  Query 64-bit: {query_64.shape}, aligned={query_64.ctypes.data % 32 == 0}")
    
    total_tests = len(DATASET_SIZES) * 6  # 6 configurazioni per dataset
    current = 0
    
    for size in DATASET_SIZES:
        label = format_size(size)
        results[size] = {}
        
        print(f"\n{'='*60}")
        print(f"DATASET SIZE: {size:,} x {D}")
        print(f"{'='*60}")
        
        # Genera dataset
        print(f"  Creazione dataset {size:,} x {D}...")
        try:
            ds_32 = generate_dataset(size, D, 'float32', 16)
            print(f"    32-bit: {ds_32.shape}, aligned={ds_32.ctypes.data % 16 == 0}")
        except MemoryError:
            print(f"    32-bit: OOM!")
            ds_32 = None
            
        try:
            ds_64 = generate_dataset(size, D, 'float64', 32)
            print(f"    64-bit: {ds_64.shape}, aligned={ds_64.ctypes.data % 32 == 0}")
        except MemoryError:
            print(f"    64-bit: OOM!")
            ds_64 = None
        
        # Test 32-bit C
        current += 1
        print(f"\n  [{current}/{total_tests}] Testing 32-bit C puro...")
        if ds_32 is not None:
            res = run_test_32(ds_32, query_32, use_asm=False)
            if res:
                print(f"    Fit: {res[0]:.4f}s, Predict: {res[1]:.4f}s, Total: {res[0]+res[1]:.4f}s")
            results[size]["32_C"] = res
        else:
            results[size]["32_C"] = "SKIP"
            print(f"    SKIP (OOM)")
        
        # Test 32-bit ASM
        current += 1
        print(f"\n  [{current}/{total_tests}] Testing 32-bit ASM...")
        if ds_32 is not None:
            res = run_test_32(ds_32, query_32, use_asm=True)
            if res:
                print(f"    Fit: {res[0]:.4f}s, Predict: {res[1]:.4f}s, Total: {res[0]+res[1]:.4f}s")
            results[size]["32_ASM"] = res
        else:
            results[size]["32_ASM"] = "SKIP"
            print(f"    SKIP (OOM)")
        
        # Test 64-bit C
        current += 1
        print(f"\n  [{current}/{total_tests}] Testing 64-bit C puro...")
        if ds_64 is not None:
            res = run_test_64(ds_64, query_64, use_asm=False)
            if res:
                print(f"    Fit: {res[0]:.4f}s, Predict: {res[1]:.4f}s, Total: {res[0]+res[1]:.4f}s")
            results[size]["64_C"] = res
        else:
            results[size]["64_C"] = "SKIP"
            print(f"    SKIP (OOM)")
        
        # Test 64-bit ASM
        current += 1
        print(f"\n  [{current}/{total_tests}] Testing 64-bit ASM...")
        if ds_64 is not None:
            res = run_test_64(ds_64, query_64, use_asm=True)
            if res:
                print(f"    Fit: {res[0]:.4f}s, Predict: {res[1]:.4f}s, Total: {res[0]+res[1]:.4f}s")
            results[size]["64_ASM"] = res
        else:
            results[size]["64_ASM"] = "SKIP"
            print(f"    SKIP (OOM)")
        
        # Test 64-bit OpenMP C
        current += 1
        print(f"\n  [{current}/{total_tests}] Testing 64-bit OpenMP C puro...")
        if ds_64 is not None:
            res = run_test_64omp(ds_64, query_64, use_asm=False)
            if res:
                print(f"    Fit: {res[0]:.4f}s, Predict: {res[1]:.4f}s, Total: {res[0]+res[1]:.4f}s")
            results[size]["64omp_C"] = res
        else:
            results[size]["64omp_C"] = "SKIP"
            print(f"    SKIP (OOM)")
        
        # Test 64-bit OpenMP ASM
        current += 1
        print(f"\n  [{current}/{total_tests}] Testing 64-bit OpenMP ASM...")
        if ds_64 is not None:
            res = run_test_64omp(ds_64, query_64, use_asm=True)
            if res:
                print(f"    Fit: {res[0]:.4f}s, Predict: {res[1]:.4f}s, Total: {res[0]+res[1]:.4f}s")
            results[size]["64omp_ASM"] = res
        else:
            results[size]["64omp_ASM"] = "SKIP"
            print(f"    SKIP (OOM)")
        
        # Libera memoria
        del ds_32, ds_64
        import gc
        gc.collect()
    
    # Stampa tabella finale
    print_table()
    
    # Salva su file
    import io
    from contextlib import redirect_stdout
    
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_table()
    
    with open("benchmark_results.txt", "w") as f:
        f.write(buffer.getvalue())
    
    print(f"\nRisultati salvati in: benchmark_results.txt")

if __name__ == "__main__":
    main()
