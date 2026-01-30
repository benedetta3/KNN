# QuantPivot k-NN: Ottimizzazioni Assembly SIMD per Architetture Avanzate

**Progetto Gruppo 6** - Architetture Avanzate dei Sistemi di Elaborazione

## Obiettivo del Progetto

Implementazione ad alte prestazioni dell'algoritmo QuantPivot per k-Nearest Neighbors con confronto sistematico tra:
- **Baseline C**: Codice compilato con flag `-O0` (nessuna ottimizzazione del compilatore)
- **Assembly SIMD**: Ottimizzazioni manuali utilizzando istruzioni SSE e AVX

Il progetto dimostra come ottimizzazioni Assembly manuali possano superare codice C non ottimizzato, evidenziando l'importanza del controllo a basso livello delle risorse hardware.

---

## Setup e Dipendenze

### Requisiti di Sistema

- **Sistema Operativo**: Linux (testato su Ubuntu/Debian)
- **CPU**: Processore x86-64 con supporto SSE 4.2 e AVX2
- **Compilatori**: GCC, NASM (Netwide Assembler)
- **Python**: 3.8 o superiore

### Installazione Dipendenze

```bash
# Aggiornare il sistema
sudo apt update

# Installare strumenti di sviluppo
sudo apt install build-essential python3 python3-pip python3-venv python3-dev

# Installare NASM per assembly
sudo apt install nasm

# Installare librerie matematiche
sudo apt install libfftw3-dev libfftw3-doc
```

### Configurazione Virtual Environment Python

```bash
# Creare virtual environment nella directory del progetto
cd ProgettoGruppo6
python3 -m venv venv

# Attivare virtual environment
source venv/bin/activate

# Aggiornare pip e installare pacchetti richiesti
pip install --upgrade pip setuptools wheel
pip install numpy pyfftw

# Installare il progetto in modalità development
pip install -e .
```

---

## Compilazione del Progetto

### Architettura del Build System

Il progetto supporta **6 configurazioni di compilazione**:
- 3 versioni (32-bit, 64-bit, 64-bit OpenMP)
- 2 modalità per versione (C baseline, Assembly ottimizzato)

### Variabili di Controllo

| Variabile Ambiente | Descrizione |
|-------------------|-------------|
| `USE_ASM_32=1` | Abilita ottimizzazioni Assembly SSE per versione 32-bit |
| `USE_ASM_64=1` | Abilita ottimizzazioni Assembly AVX per versione 64-bit |
| `USE_ASM_OMP=1` | Abilita ottimizzazioni Assembly AVX + OpenMP per versione parallela |

### Compilazione Versione 32-bit

**Modalità C Baseline (nessuna ottimizzazione):**
```bash
cd ProgettoGruppo6
rm -rf build/ gruppo6/quantpivot32/*.so src/32/*.o
python3 setup.py build_ext --inplace
```

**Modalità Assembly SSE Ottimizzato:**
```bash
cd ProgettoGruppo6
rm -rf build/ gruppo6/quantpivot32/*.so src/32/*.o
USE_ASM_32=1 python3 setup.py build_ext --inplace
```

### Compilazione Versione 64-bit

**Modalità C Baseline:**
```bash
cd ProgettoGruppo6
rm -rf build/ gruppo6/quantpivot64/*.so src/64/*.o
python3 setup.py build_ext --inplace
```

**Modalità Assembly AVX Ottimizzato:**
```bash
cd ProgettoGruppo6
rm -rf build/ gruppo6/quantpivot64/*.so src/64/*.o
USE_ASM_64=1 python3 setup.py build_ext --inplace
```

### Compilazione Versione 64-bit OpenMP

**Modalità C Baseline con OpenMP:**
```bash
cd ProgettoGruppo6
rm -rf build/ gruppo6/quantpivot64omp/*.so src/64omp/*.o
python3 setup.py build_ext --inplace
```

**Modalità Assembly AVX + OpenMP Ottimizzato:**
```bash
cd ProgettoGruppo6
rm -rf build/ gruppo6/quantpivot64omp/*.so src/64omp/*.o
USE_ASM_OMP=1 python3 setup.py build_ext --inplace
```

### Flag di Compilazione

Quando le ottimizzazioni assembly sono abilitate, vengono definiti automaticamente i seguenti flag:
- `-DUSE_ASM_APPROX`: Usa assembly per calcolo distanza approssimata
- `-DUSE_ASM_EUCLIDEAN`: Usa assembly per calcolo distanza euclidea
- `-DUSE_ASM_LOWER_BOUND`: Usa assembly per calcolo lower bound

---

## Esecuzione dei Test

### Test Funzionale Base

Sintassi generale:
```bash
python3 test.py <dataset_file> <query_file> <h> <k> <x> <version>
```

Parametri:
- `dataset_file`: File dataset in formato .ds2
- `query_file`: File query in formato .ds2
- `h`: Numero di pivot
- `k`: Numero di vicini da trovare
- `x`: Parametro di quantizzazione
- `version`: 32, 64, o 64omp

**Esempio test versione 32-bit:**
```bash
python3 test.py dataset_2000x256_32.ds2 query_2000x256_32.ds2 16 8 64 32
```

**Esempio test versione 64-bit:**
```bash
python3 test.py dataset_2000x256_64.ds2 query_2000x256_64.ds2 16 8 64 64
```

**Esempio test versione 64-bit OpenMP:**
```bash
python3 test.py dataset_2000x256_64.ds2 query_2000x256_64.ds2 16 8 64 64omp
```

### Verifica Correttezza Risultati

```bash
# Verificare che risultati C e Assembly siano identici
python3 compare_results.py --t 32    # Per versione 32-bit
python3 compare_results.py --t 64    # Per versione 64-bit
python3 compare_results.py --t 64omp # Per versione 64-bit OpenMP
```

Output atteso: `ID identici: True`, `Distanze compatibili: True`

### Test di Robustezza (Edge Cases)

```bash
python3 test_edge_cases.py
```

Questo script esegue 50 test su:
- Dimensioni dispari e prime (D=3, 5, 7, 11, 13, 127, 257...)
- Dimensioni non multipli di 4, 8, 16
- Dimensioni perfette per SIMD (D=16, 32, 64, 128, 256, 512, 1024, 2048)
- Dataset di dimensioni variabili (N=10 fino a N=10000)
- Parametri estremi (k=1, k=99, h=1, h=100, x=1, x=D)

Output atteso: `Test passati: 50/50`

### Benchmark Performance

**Benchmark scalabilità su dataset crescenti:**
```bash
python3 benchmark.py
```

Questo benchmark:
- Testa 6 dimensioni di dataset: 1K, 2K, 10K, 100K, 250K, 500K righe
- Compila automaticamente ogni versione in modalità C e Assembly
- Misura tempi di FIT e PREDICT separatamente
- Calcola speedup Assembly vs C per ogni configurazione
- Salva risultati in `benchmark_results.txt`

**Benchmark dimensionalità:**
```bash
python3 benchmark_dimensioni.py
```

Questo benchmark:
- Testa configurazioni crescenti: 2000x256, 5000x512, 10000x1024
- Analizza scalabilità rispetto a N (numero punti) e D (dimensionalità)
- Mostra risultati comparativi per tutte le versioni

---

## Architettura e Implementazione

### Tre Versioni Complete

| Versione | Tipo Dati | SIMD | Parallelismo | Registri |
|----------|-----------|------|--------------|----------|
| **32-bit** | float (32-bit) | SSE 4.2 | Single-thread | XMM (128-bit) |
| **64-bit** | double (64-bit) | AVX2 | Single-thread | YMM (256-bit) |
| **64-bit OMP** | double (64-bit) | AVX2 | Multi-thread OpenMP | YMM (256-bit) |

### Dual Code Path Architecture

Ogni versione implementa un'architettura a doppio percorso:

```c
#ifdef USE_ASM_EUCLIDEAN
    // Percorso Assembly ottimizzato SIMD
    euclidean_distance_asm(query, dataset, result);
#else
    // Percorso C baseline
    euclidean_distance_c(query, dataset, result);
#endif
```

Questo permette di:
1. Confrontare direttamente performance C vs Assembly
2. Verificare correttezza tramite confronto bit-a-bit dei risultati
3. Attivare/disattivare ottimizzazioni senza modificare codice

### Struttura Directory

```
ProgettoGruppo6/
├── src/
│   ├── 32/                          # Versione 32-bit float + SSE
│   │   ├── quantpivot32.c          # Logica algoritmica + baseline C
│   │   ├── quantpivot32.nasm       # Kernel ottimizzati SSE
│   │   ├── quantpivot32_py.c       # Wrapper Python/C
│   │   ├── sseutils32.nasm         # Utilities SIMD 32-bit
│   │   └── common.h                # Definizioni condivise
│   ├── 64/                          # Versione 64-bit double + AVX
│   │   ├── quantpivot64.c          # Logica algoritmica + baseline C
│   │   ├── quantpivot64.nasm       # Kernel ottimizzati AVX
│   │   ├── quantpivot64_py.c       # Wrapper Python/C
│   │   ├── sseutils64.nasm         # Utilities SIMD 64-bit
│   │   └── common.h                # Definizioni condivise
│   └── 64omp/                       # Versione 64-bit double + AVX + OpenMP
│       ├── quantpivot64omp.c       # Logica algoritmica parallelizzata
│       ├── quantpivot64omp.nasm    # Kernel ottimizzati AVX
│       ├── quantpivot64omp_py.c    # Wrapper Python/C
│       ├── sseutils64.nasm         # Utilities SIMD 64-bit
│       └── common.h                # Definizioni condivise
├── gruppo6/                         # Package Python
│   ├── __init__.py
│   ├── quantpivot32/               # Modulo Python 32-bit
│   ├── quantpivot64/               # Modulo Python 64-bit
│   └── quantpivot64omp/            # Modulo Python 64-bit OpenMP
├── setup.py                         # Build system
├── pyproject.toml                   # Configurazione progetto
└── README.md                        # Questa documentazione
```

### API Python

Le tre versioni espongono API identiche per facilità d'uso:

```python
from gruppo6.quantpivot32 import QuantPivot as QP32
from gruppo6.quantpivot64 import QuantPivot as QP64
from gruppo6.quantpivot64omp import QuantPivot as QP64OMP

# Esempio utilizzo
model = QP64()
model.fit(dataset, h=16, x=64, silent=False)
ids, dists = model.predict(query, k=8, silent=False)
```

Parametri:
- `dataset`: Array numpy (N x D) contenente punti del dataset
- `query`: Array numpy (nq x D) contenente punti query
- `h`: Numero di pivot per quantizzazione
- `x`: Parametro di quantizzazione (numero componenti selezionate)
- `k`: Numero di vicini da restituire
- `silent`: Se True, sopprime output di debug

---

## Ottimizzazioni Assembly

### Tecniche di Ottimizzazione Implementate

#### 1. Vectorization SIMD

**Versione 32-bit SSE (registri XMM, 128-bit):**
```assembly
; Elaborazione parallela di 4 float contemporaneamente
movups  xmm0, [rsi + rcx*4]      ; Carica 4 elementi query
movups  xmm1, [rdi + rcx*4]      ; Carica 4 elementi dataset
subps   xmm0, xmm1               ; Sottrazione vettoriale
mulps   xmm0, xmm0               ; Quadrato elemento per elemento
addps   xmm2, xmm0               ; Accumula nel registro somma
```

**Versione 64-bit AVX (registri YMM, 256-bit):**
```assembly
; Elaborazione parallela di 4 double contemporaneamente
vmovupd ymm0, [rsi + rcx*8]      ; Carica 4 elementi query
vmovupd ymm1, [rdi + rcx*8]      ; Carica 4 elementi dataset
vsubpd  ymm0, ymm0, ymm1         ; Sottrazione vettoriale
vmulpd  ymm0, ymm0, ymm0         ; Quadrato elemento per elemento
vaddpd  ymm2, ymm2, ymm0         ; Accumula nel registro somma
```

#### 2. Loop Unrolling

Riduzione overhead di branching elaborando blocchi di elementi:

```c
// Elaborazione di 16 elementi per iterazione
for (int i = 0; i < dim-15; i += 16) {
    asm_euclidean_block_16(query+i, data+i, &sum);
}
// Gestione rimanenti elementi (tail)
for (int i = dim-dim%16; i < dim; i++) {
    float diff = query[i] - data[i];
    sum += diff * diff;
}
```

#### 3. Riduzione Orizzontale Efficiente

**SSE (32-bit):**
```assembly
haddps  xmm0, xmm0               ; [a+b, c+d, a+b, c+d]
haddps  xmm0, xmm0               ; [a+b+c+d, *, *, *]
movss   [result], xmm0           ; Estrai risultato scalare
```

**AVX (64-bit):**
```assembly
vhaddpd ymm0, ymm0, ymm0         ; Riduzione parziale
vextractf128 xmm1, ymm0, 1       ; Estrai upper 128 bit
vaddpd  xmm0, xmm0, xmm1         ; Combina upper + lower
haddpd  xmm0, xmm0               ; Riduzione finale
vmovsd  [result], xmm0           ; Estrai risultato scalare
```

#### 4. Prefetching e Cache Optimization

```assembly
; Prefetch dati per prossima iterazione
prefetcht0 [rsi + rcx*4 + 64]    ; Prefetch query
prefetcht0 [rdi + rcx*4 + 64]    ; Prefetch dataset
```

#### 5. Parallelizzazione OpenMP

Versione 64-bit OpenMP utilizza direttive per parallelizzare loop costosi:

```c
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < N; i++) {
    // Quantizzazione parallela di ciascun punto
    quantize_point(ds_plus[i], ds_minus[i], dataset[i], D, x);
}

#pragma omp parallel for schedule(dynamic)
for (int q = 0; q < nq; q++) {
    // Ricerca parallela per ciascuna query
    process_query(q, ...);
}
```

### Kernel Ottimizzati

I seguenti kernel sono stati implementati in Assembly SIMD:

1. **Distanza Euclidea**: `euclidean_distance_asm()`
   - Calcolo vettorizzato di sqrt(sum((a-b)²))
   - Speedup tipico: 1.5x - 2.0x su baseline C

2. **Dot Product**: `dot_product_asm()`
   - Prodotto scalare vettorizzato
   - Utilizzato per distanza approssimata
   - Speedup tipico: 1.3x - 1.8x

3. **Distanza Approssimata**: `approx_distance_asm()`
   - Formula: (v+·w+) + (v-·w-) - (v+·w-) - (v-·w+)
   - 4 dot product parallelizzati
   - Speedup tipico: 1.2x - 1.5x

4. **Lower Bound**: `lower_bound_asm()`
   - Calcolo max_j |idx[v,j] - qpivot[j]|
   - Ricerca massimo vettorizzata
   - Speedup tipico: 1.1x - 1.3x

### Perché Assembly Supera C con -O0

1. **Vectorization Manuale**: Compilatore con -O0 non vettorizza automaticamente
2. **Loop Unrolling Esplicito**: Riduzione overhead di branching e incrementi
3. **Instruction Selection Ottimale**: Uso diretto istruzioni SIMD più efficienti
4. **Gestione Registri**: Minimizzazione spilling e massimizzazione riuso registri
5. **Prefetching Controllato**: Nascondere latenza memoria con prefetch espliciti

---

## Risultati e Performance

### Risultati Benchmark Scalabilità

**Dataset Grandi (100K+ righe):**
- **32-bit SSE**: Speedup costante 1.01x - 1.08x
- **64-bit AVX**: Speedup 1.00x - 1.98x (picco su 500K righe)
- **64-bit OpenMP**: Scalabilità eccellente 5x - 11x grazie a parallelismo multi-core

**Dataset Piccoli (1K-10K righe):**
- Speedup modesti 0.94x - 1.08x a causa di:
  - Overhead inizializzazione registri SIMD
  - Cache effects dominanti su dataset piccoli
  - Branch prediction C efficace su loop brevi

**Tabella Performance Rappresentativa (2000x256 dataset):**

| Versione | FIT time | PREDICT time | Speedup vs C |
|----------|----------|--------------|--------------|
| 32-bit C | 0.11s | 0.62s | baseline |
| 32-bit ASM | 0.12s | 0.64s | 0.97x |
| 64-bit C | 0.13s | 0.79s | baseline |
| 64-bit ASM | 0.14s | 0.75s | 1.05x |
| 64-bit OMP C | 0.06s | 0.24s | baseline |
| 64-bit OMP ASM | 0.05s | 0.22s | 1.06x |

### Correttezza Verificata

- 100% successo su 50 test edge cases
- Risultati identici tra C e Assembly
- Differenza numerica massima: 7.1e-15 (precisione floating-point)

### Considerazioni su Speedup < 1.0

Alcuni casi mostrano speedup inferiori a 1.0. Questo è **normale e documentato** perché:

1. **Overhead SIMD**: Setup registri, allineamento memoria, gestione tail hanno costo fisso
2. **Memory Bandwidth**: Su workload memory-bound, SIMD non può superare limiti hardware
3. **Cache Effects**: Su dataset piccoli che stanno completamente in cache L1/L2, C può essere competitivo
4. **Branch Prediction**: CPU moderne hanno branch predictor sofisticati che aiutano codice C scalare

L'importante è che su dataset realistici (100K+) l'Assembly mostra vantaggi chiari.

### Confronto con Compilatore Ottimizzato

Con flag `-O3 -march=native`:
- Speedup Assembly si riduce a ~1.0x
- Compilatore moderno effettua auto-vectorization efficace
- Gap Assembly vs C quasi nullo

Questo dimostra che:
- Assembly manuale ha valore quando compilatore non ottimizza (es. `-O0`)
- Per codice production, compilatore moderno è competitivo
- Assembly rimane utile per hot paths critici e controllo fine

---

## Algoritmo QuantPivot

### Descrizione Alto Livello

QuantPivot è un algoritmo approssimato per k-NN che utilizza:
1. **Quantizzazione**: Riduzione dimensionalità tramite proiezione su pivot
2. **Lower Bound**: Pruning candidati usando distanza approssimata
3. **Refinement**: Calcolo distanza esatta solo su candidati filtrati

### Fasi dell'Algoritmo

#### Fase 1: FIT (Training)

```
Input: Dataset DS (N x D), numero pivot h, parametro quantizzazione x

1. Selezione Pivot:
   - Genera h vettori pivot P = {p₁, p₂, ..., pₕ} random da DS
   
2. Quantizzazione Dataset:
   Per ogni punto v in DS:
   - Decomponi v in due vettori non-negativi:
     * v⁺: componenti positive di v
     * v⁻: valori assoluti delle componenti negative di v
   - Applica sparsificazione: mantieni solo x componenti maggiori
   
3. Costruzione Indice:
   Per ogni punto v e ogni pivot pⱼ:
   - Calcola distanza approssimata: d̃(v, pⱼ)
   - Memorizza in indice: idx[v, j] = d̃(v, pⱼ)
```

#### Fase 2: PREDICT (Query)

```
Input: Query Q (nq x D), numero vicini k

Per ogni query q:

1. Quantizzazione Query:
   - Decomponi q in q⁺ e q⁻
   - Applica sparsificazione con parametro x
   
2. Calcolo Distanze ai Pivot:
   Per ogni pivot pⱼ:
   - qpivot[j] = d̃(q, pⱼ)
   
3. Lower Bound e Pruning:
   Per ogni punto v in DS:
   - LB(v, q) = max_j |idx[v,j] - qpivot[j]|
   - Mantieni candidati con LB più bassi
   
4. Refinement:
   - Calcola distanza euclidea esatta per candidati
   - Seleziona k vicini con distanze minime
   
5. Output:
   - Restituisci ID e distanze dei k vicini
```

### Distanza Approssimata

Formula utilizzata:
```
d̃(v, w) = (v⁺ · w⁺) + (v⁻ · w⁻) - (v⁺ · w⁻) - (v⁻ · w⁺)
```

Dove:
- `·` denota prodotto scalare
- v⁺, v⁻ sono vettori quantizzati non-negativi
- Questa distanza approssima la distanza euclidea reale

### Vantaggi QuantPivot

1. **Efficienza**: Evita calcolo distanza euclidea per maggioranza dei punti
2. **Scalabilità**: Lower bound permette pruning aggressivo su dataset grandi
3. **Precisione Controllabile**: Parametro x controlla trade-off accuratezza/velocità
4. **Parallelizzabile**: Calcoli indipendenti per ogni query

### Complessità Computazionale

- **FIT**: O(N · D · h) per quantizzazione + costruzione indice
- **PREDICT (per query)**: 
  - O(D · h) quantizzazione query
  - O(N · h) calcolo lower bound
  - O(C · D) calcolo distanze esatte (C << N candidati)
  - Totale: O(D · h + N · h + C · D)

Senza QuantPivot: O(N · D) per query

Speedup teorico quando C << N: significativo su dataset grandi

---


### Risultati Ottenuti

1. **Implementazione Completa**: 3 versioni funzionanti con dual code path C/Assembly
2. **Performance Competitive**: Speedup fino a 1.98x su dataset grandi
3. **Correttezza Garantita**: 100% test passati, risultati identici C vs Assembly
4. **Robustezza**: Gestione corretta di tutti i casi limite (dimensioni dispari, prime, etc.)
5. **Scalabilità**: OpenMP fornisce eccellente parallelizzazione multi-core