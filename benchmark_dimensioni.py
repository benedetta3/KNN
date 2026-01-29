#!/usr/bin/env python3
"""
BENCHMARK DIMENSIONI - Progetto QuantPivot Gruppo 6
Esegue test su dataset con diverse dimensioni (N×D) per tutte e 3 le versioni.
Genera dataset al volo e usa direttamente le librerie Python.
"""

import sys
import os
import time
import numpy as np
import pyfftw

# Configurazione - Dimensioni da testare (N, D)
DIMENSION_CONFIGS = [
    (1024, 512),
    (2048, 1024),
    (4096, 2048),
    (8192, 4096),
    (12480, 5120),  # Corretto a 12480 per essere multiplo di 32
]

# Parametri fissi
H = 16
K = 8
X = 64

# Risultati: {(n, d): {version: (fit, prd)}}
results = {}

def format_dim(n, d):
    """Formatta dimensione in formato leggibile."""
    return f"{n}×{d}"

def generate_dataset(n, d, dtype, alignment):
    """Genera dataset random allineato."""
    data = pyfftw.empty_aligned((n, d), dtype=dtype, n=alignment)
    data[:] = np.random.randn(n, d).astype(dtype)
    return data

def run_test_32(dataset, query):
    """Esegue test versione 32-bit."""
    try:
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
        print(f"ERRORE 32: {e}")
        return None

def run_test_64(dataset, query):
    """Esegue test versione 64-bit."""
    try:
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
        print(f"ERRORE 64: {e}")
        return None

def run_test_64omp(dataset, query):
    """Esegue test versione 64-bit OpenMP."""
    try:
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
        print(f"ERRORE 64omp: {e}")
        return None

def print_table():
    """Stampa tabella riepilogativa."""
    
    print("\n" + "="*90)
    print("  BENCHMARK DIMENSIONI - Progetto QuantPivot Gruppo 6")
    print(f"  Parametri: h={H}, k={K}, x={X}")
    print("="*90)
    
    # Header
    print(f"\n{'Dataset':<14} | {'NQ':<6} | {'32-bit SSE':<20} | {'64-bit AVX':<20} | {'64-bit OMP':<20}")
    print(f"{'(N×D)':<14} | {'':<6} | {'FIT / PRD (sec)':<20} | {'FIT / PRD (sec)':<20} | {'FIT / PRD (sec)':<20}")
    print("-"*90)
    
    for (n, d) in DIMENSION_CONFIGS:
        key = (n, d)
        if key not in results:
            continue
        label = format_dim(n, d)
        nq = results[key].get("nq", n)
        row = f"{label:<14} | {nq:<6} |"
        
        for version in ["32", "64", "64omp"]:
            if version in results[key]:
                res = results[key][version]
                if res is None:
                    row += f" {'ERRORE':<20}|"
                elif res == "SKIP":
                    row += f" {'SKIP (OOM)':<20}|"
                else:
                    fit, prd = res
                    row += f" {fit:>6.2f} / {prd:<10.2f} |"
            else:
                row += f" {'-':<20}|"
        print(row)
    
    print("-"*90)
    
    # Speedup OMP vs 32-bit
    print("\nSPEEDUP OpenMP vs 32-bit SSE:")
    for (n, d) in DIMENSION_CONFIGS:
        key = (n, d)
        if key not in results:
            continue
        label = format_dim(n, d)
        if "32" in results[key] and "64omp" in results[key]:
            res32 = results[key]["32"]
            res_omp = results[key]["64omp"]
            if res32 and res_omp and res32 != "SKIP" and res_omp != "SKIP":
                _, prd32 = res32
                _, prd_omp = res_omp
                if prd_omp > 0:
                    speedup = prd32 / prd_omp
                    print(f"   {label}: {speedup:.2f}x")
    
    # Speedup OMP vs 64-bit
    print("\nSPEEDUP OpenMP vs 64-bit AVX:")
    for (n, d) in DIMENSION_CONFIGS:
        key = (n, d)
        if key not in results:
            continue
        label = format_dim(n, d)
        if "64" in results[key] and "64omp" in results[key]:
            res64 = results[key]["64"]
            res_omp = results[key]["64omp"]
            if res64 and res_omp and res64 != "SKIP" and res_omp != "SKIP":
                _, prd64 = res64
                _, prd_omp = res_omp
                if prd_omp > 0:
                    speedup = prd64 / prd_omp
                    print(f"   {label}: {speedup:.2f}x")
    
    print("\n" + "="*90)

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("="*90)
    print("BENCHMARK DIMENSIONI - QuantPivot")
    print("="*90)
    print(f"Parametri: h={H}, k={K}, x={X}")
    print("Dimensioni testate:")
    for (n, d) in DIMENSION_CONFIGS:
        print(f"   - {n}×{d}")
    print("="*90)
    
    total_tests = len(DIMENSION_CONFIGS) * 3
    current = 0
    
    for (n, d) in DIMENSION_CONFIGS:
        key = (n, d)
        results[key] = {}
        
        # NQ = N (usa lo stesso numero di query del dataset)
        nq = n
        results[key]["nq"] = nq
        
        label = format_dim(n, d)
        
        print(f"\n{'='*70}")
        print(f"DATASET SIZE: {n:,} × {d}")
        print(f"{'='*70}")
        
        # Genera query
        print(f"  Generazione query ({nq}×{d})...")
        try:
            query_32 = generate_dataset(nq, d, 'float32', 16)
            query_64 = generate_dataset(nq, d, 'float64', 32)
            print(f"    Query 32-bit: {query_32.shape}, aligned={query_32.ctypes.data % 16 == 0}")
            print(f"    Query 64-bit: {query_64.shape}, aligned={query_64.ctypes.data % 32 == 0}")
        except MemoryError:
            print(f"    ERRORE: Memoria insufficiente per le query!")
            query_32 = None
            query_64 = None
        
        # Genera dataset
        print(f"  Creazione dataset {n:,} × {d}...")
        try:
            ds_32 = generate_dataset(n, d, 'float32', 16)
            print(f"    32-bit: {ds_32.shape}, aligned={ds_32.ctypes.data % 16 == 0}")
        except MemoryError:
            print(f"    32-bit: OOM!")
            ds_32 = None
            
        try:
            ds_64 = generate_dataset(n, d, 'float64', 32)
            print(f"    64-bit: {ds_64.shape}, aligned={ds_64.ctypes.data % 32 == 0}")
        except MemoryError:
            print(f"    64-bit: OOM!")
            ds_64 = None
        
        # Test 32-bit
        current += 1
        print(f"\n  [{current}/{total_tests}] Testing 32-bit SSE...")
        if ds_32 is not None and query_32 is not None:
            res = run_test_32(ds_32, query_32)
            if res:
                print(f"    Fit: {res[0]:.4f}s, Predict: {res[1]:.4f}s, Total: {res[0]+res[1]:.4f}s")
            results[key]["32"] = res
        else:
            results[key]["32"] = "SKIP"
            print(f"    SKIP (OOM)")
        
        # Test 64-bit
        current += 1
        print(f"\n  [{current}/{total_tests}] Testing 64-bit AVX...")
        if ds_64 is not None and query_64 is not None:
            res = run_test_64(ds_64, query_64)
            if res:
                print(f"    Fit: {res[0]:.4f}s, Predict: {res[1]:.4f}s, Total: {res[0]+res[1]:.4f}s")
            results[key]["64"] = res
        else:
            results[key]["64"] = "SKIP"
            print(f"    SKIP (OOM)")
        
        # Test 64-bit OpenMP
        current += 1
        print(f"\n  [{current}/{total_tests}] Testing 64-bit AVX + OpenMP...")
        if ds_64 is not None and query_64 is not None:
            res = run_test_64omp(ds_64, query_64)
            if res:
                print(f"    Fit: {res[0]:.4f}s, Predict: {res[1]:.4f}s, Total: {res[0]+res[1]:.4f}s")
            results[key]["64omp"] = res
        else:
            results[key]["64omp"] = "SKIP"
            print(f"    SKIP (OOM)")
        
        # Libera memoria
        del ds_32, ds_64, query_32, query_64
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
    
    with open("benchmark_dimensioni_results.txt", "w") as f:
        f.write(buffer.getvalue())
    
    print(f"\nRisultati salvati in: benchmark_dimensioni_results.txt")

if __name__ == "__main__":
    main()
