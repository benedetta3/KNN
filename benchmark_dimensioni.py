#!/usr/bin/env python3
"""
BENCHMARK DIMENSIONI - Progetto QuantPivot Gruppo 6
Esegue test su dataset con diverse dimensioni (N×D).
Testa sia C puro che ASM per ogni versione (6 configurazioni totali).
Formato: 3 righe (32, 64, 64omp) × 2 colonne (C, ASM)
"""

import sys
import os
import time
import numpy as np
import pyfftw
import subprocess

# Configurazione - Dimensioni da testare (N, D)
DIMENSION_CONFIGS = [
    (2000, 256),
    (5000, 512),
    (10000, 1024),
]

# Parametri fissi
H = 16
K = 8
X = 64

# Risultati: {(n, d): {version_variant: (fit, prd)}}
# version_variant: "32_C", "32_ASM", "64_C", "64_ASM", "64omp_C", "64omp_ASM"
results = {}

def format_dim(n, d):
    """Formatta dimensione in formato leggibile."""
    if n >= 1000:
        return f"{n//1000}K×{d}"
    return f"{n}×{d}"

def generate_dataset(n, d, dtype, alignment):
    """Genera dataset random allineato."""
    data = pyfftw.empty_aligned((n, d), dtype=dtype, n=alignment)
    data[:] = np.random.randn(n, d).astype(dtype)
    return data

def rebuild_module(version, use_asm):
    """Ricompila modulo con/senza ASM."""
    env = os.environ.copy()
    env["NATIVE"] = "1"  # Ottimizzazioni aggressive
    
    if version == "32":
        env["USE_ASM_32"] = "1" if use_asm else "0"
    elif version == "64":
        env["USE_ASM_64"] = "1" if use_asm else "0"
    elif version == "64omp":
        env["USE_ASM_OMP"] = "1" if use_asm else "0"
    
    # Rimuovi moduli cached
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
        model.fit(dataset, H, X, True)
        fit_time = time.perf_counter() - start
        
        start = time.perf_counter()
        ids, dists = model.predict(query, K, True)
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
        model.fit(dataset, H, X, True)
        fit_time = time.perf_counter() - start
        
        start = time.perf_counter()
        ids, dists = model.predict(query, K, True)
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
        model.fit(dataset, H, X, True)
        fit_time = time.perf_counter() - start
        
        start = time.perf_counter()
        ids, dists = model.predict(query, K, True)
        prd_time = time.perf_counter() - start
        
        return (fit_time, prd_time)
    except Exception as e:
        print(f"ERRORE 64omp {'ASM' if use_asm else 'C'}: {e}")
        return None

def print_table():
    """Stampa tabella riepilogativa 3×2 (versioni × C/ASM)."""
    
    print("\n" + "="*100)
    print("  BENCHMARK DIMENSIONI - Progetto QuantPivot Gruppo 6")
    print(f"  Parametri: h={H}, k={K}, x={X}")
    print("="*100)
    
    for (n, d) in DIMENSION_CONFIGS:
        key = (n, d)
        if key not in results:
            continue
        
        label = format_dim(n, d)
        nq = n  # usa N query come default
        
        print(f"\n{'─'*100}")
        print(f"  DATASET: {label}  (N={n}, D={d}, NQ={nq})")
        print(f"{'─'*100}")
        print(f"{'Versione':<15} | {'C Baseline':<38} | {'ASM Optimized':<38}")
        print(f"{'':<15} | {'FIT (s)':<10} {'PRD (s)':<10} {'TOT (s)':<10} | {'FIT (s)':<10} {'PRD (s)':<10} {'TOT (s)':<10}")
        print(f"{'-'*15}+{'-'*40}+{'-'*40}")
        
        for version, label in [("32", "32-bit SSE"), ("64", "64-bit AVX"), ("64omp", "64-bit OMP")]:
            row = f"{label:<15} |"
            
            # C baseline
            var_c = f"{version}_C"
            if var_c in results[key]:
                res = results[key][var_c]
                if res:
                    fit, prd = res
                    tot = fit + prd
                    row += f" {fit:>8.3f}  {prd:>8.3f}  {tot:>8.3f}  |"
                else:
                    row += f" {'ERRORE':<36} |"
            else:
                row += f" {'-':<36} |"
            
            # ASM
            var_asm = f"{version}_ASM"
            if var_asm in results[key]:
                res = results[key][var_asm]
                if res:
                    fit, prd = res
                    tot = fit + prd
                    row += f" {fit:>8.3f}  {prd:>8.3f}  {tot:>8.3f}  "
                    
                    # Speedup C→ASM
                    if var_c in results[key] and results[key][var_c]:
                        fit_c, prd_c = results[key][var_c]
                        speedup = (fit_c + prd_c) / (fit + prd)
                        row += f"[{speedup:.2f}x]"
                else:
                    row += f" {'ERRORE':<36}"
            else:
                row += f" {'-':<36}"
            
            print(row)
    
    print("="*100)
    
    # Tabella speedup finale
    print("\n" + "="*60)
    print("  SPEEDUP SUMMARY")
    print("="*60)
    print(f"{'Dataset':<12} | {'32 C→ASM':<12} | {'64 C→ASM':<12} | {'OMP C→ASM':<12}")
    print("-"*60)
    
    for (n, d) in DIMENSION_CONFIGS:
        key = (n, d)
        if key not in results:
            continue
        label = format_dim(n, d)
        row = f"{label:<12} |"
        
        for version in ["32", "64", "64omp"]:
            var_c = f"{version}_C"
            var_asm = f"{version}_ASM"
            
            if var_c in results[key] and var_asm in results[key]:
                res_c = results[key][var_c]
                res_asm = results[key][var_asm]
                if res_c and res_asm:
                    time_c = sum(res_c)
                    time_asm = sum(res_asm)
                    speedup = time_c / time_asm
                    row += f" {speedup:>10.2f}x |"
                else:
                    row += f" {'-':>11} |"
            else:
                row += f" {'-':>11} |"
        
        print(row)
    
    print("="*60)

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("="*100)
    print("BENCHMARK DIMENSIONI - QuantPivot (3 versioni × 2 varianti)")
    print("="*100)
    print(f"Parametri: h={H}, k={K}, x={X}")
    print("Dimensioni testate:")
    for (n, d) in DIMENSION_CONFIGS:
        print(f"   - {n}×{d} (NQ={n})")
    print("="*100)
    
    total_tests = len(DIMENSION_CONFIGS) * 6  # 3 versioni × 2 varianti
    current = 0
    
    for (n, d) in DIMENSION_CONFIGS:
        key = (n, d)
        results[key] = {}
        
        nq = n  # Usa stesso numero di query
        label = format_dim(n, d)
        
        print(f"\n{'='*100}")
        print(f"TESTING: {label} (N={n}, D={d}, NQ={nq})")
        print(f"{'='*100}")
        
        # Genera dataset
        print(f"Generazione dataset {n}×{d}...")
        ds_32 = generate_dataset(n, d, 'float32', 16)
        q_32 = generate_dataset(nq, d, 'float32', 16)
        ds_64 = generate_dataset(n, d, 'float64', 32)
        q_64 = generate_dataset(nq, d, 'float64', 32)
        
        # Test 32-bit
        for use_asm, variant in [(False, "C"), (True, "ASM")]:
            current += 1
            print(f"\n[{current}/{total_tests}] 32-bit {variant}...", end=" ", flush=True)
            result = run_test_32(ds_32, q_32, use_asm)
            results[key][f"32_{variant}"] = result
            if result:
                print(f" FIT={result[0]:.2f}s PRD={result[1]:.2f}s")
            else:
                print(" ERRORE")
        
        # Test 64-bit
        for use_asm, variant in [(False, "C"), (True, "ASM")]:
            current += 1
            print(f"[{current}/{total_tests}] 64-bit {variant}...", end=" ", flush=True)
            result = run_test_64(ds_64, q_64, use_asm)
            results[key][f"64_{variant}"] = result
            if result:
                print(f" FIT={result[0]:.2f}s PRD={result[1]:.2f}s")
            else:
                print(" ERRORE")
        
        # Test 64-bit OpenMP
        for use_asm, variant in [(False, "C"), (True, "ASM")]:
            current += 1
            print(f"[{current}/{total_tests}] 64omp {variant}...", end=" ", flush=True)
            result = run_test_64omp(ds_64, q_64, use_asm)
            results[key][f"64omp_{variant}"] = result
            if result:
                print(f" FIT={result[0]:.2f}s PRD={result[1]:.2f}s")
            else:
                print(" ERRORE")
    
    # Stampa risultati
    print_table()
    
    # Salva risultati su file
    import io
    from contextlib import redirect_stdout
    
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_table()
    
    with open("benchmark_dimensioni_results.txt", "w") as f:
        f.write(buffer.getvalue())
    
    print("\nBenchmark completato!")
    print("Risultati salvati in: benchmark_dimensioni_results.txt")

if __name__ == "__main__":
    main()
