# QuantPivot k-NN: Ottimizzazioni Assembly per Architetture Avanzate

**Progetto Gruppo 6** - *Architetture Avanzate dei Sistemi di Elaborazione*

## Obiettivo del Progetto

Implementazione ad alte prestazioni dell'algoritmo QuantPivot per k-Nearest Neighbors con confronto sistematico tra:
- **Baseline C** (setup professore con `-O0`)  
- **Ottimizzazioni Assembly SIMD** manuali

L'obiettivo è dimostrare come le ottimizzazioni Assembly manuali possano superare il codice C non ottimizzato, evidenziando l'importanza dell'ottimizzazione a basso livello.

---

## Architettura del Sistema

### 3 Implementazioni Complete

| Versione | Architettura | SIMD | Parallelismo | Ottimizzazioni |
|----------|-------------|------|-------------|----------------|
| **32-bit** | x86-32 | SSE 4.2 | Single-thread | Manual vectorization, loop unrolling |
| **64-bit** | x86-64 | AVX2 | Single-thread | 256-bit vectors, advanced SIMD |
| **64-bit OMP** | x86-64 | AVX2 | Multi-thread | AVX2 + OpenMP parallelism |

### Dual Code Path Architecture
```c
#ifdef USE_ASM_EUCLIDEAN
    // Assembly ottimizzato SIMD
    euclidean_distance_asm(query, dataset, result);
#else  
    // Baseline C (-O0)
    euclidean_distance_c(query, dataset, result);
#endif
```

---

## Ottimizzazioni Assembly Implementate

### 1. **Vectorization SIMD Avanzata**

**32-bit SSE:**
```assembly
; Caricamento parallelo di 4 float (128-bit)
movups  xmm0, [rsi + rcx*4]      ; query[i:i+4]  
movups  xmm1, [rdi + rcx*4]      ; data[i:i+4]
subps   xmm0, xmm1               ; diff = query - data
mulps   xmm0, xmm0               ; diff²
addps   xmm2, xmm0               ; sum += diff²
```

**64-bit AVX:**
```assembly
; Caricamento parallelo di 4 double (256-bit) 
vmovupd ymm0, [rsi + rcx*8]      ; query[i:i+4]
vmovupd ymm1, [rdi + rcx*8]      ; data[i:i+4]  
vsubpd  ymm0, ymm0, ymm1         ; diff = query - data
vmulpd  ymm0, ymm0, ymm0         ; diff²
vaddpd  ymm2, ymm2, ymm0         ; sum += diff²
```

### 2. **Ottimizzazioni SIMD per Operazioni Critiche**

Le funzioni più costose computazionalmente sono state ottimizzate con Assembly SIMD:

- **Distanza Euclidea**: Calcolo vettorizzato parallelo su 4/8 elementi
- **Dot Product**: Prodotto scalare ottimizzato per quantizzazione
- **Distanza Approssimata**: Combinazione di prodotti per lower bound
- **Operazioni su Array**: Somme e operazioni bulk parallelizzate

### 3. **Loop Unrolling Ottimizzato**
```c
// Elaborazione di blocchi con dimensioni ottimali per cache
for (int i = 0; i < dim-15; i += 16) {
    // 16 elementi per iterazione riducono overhead branching
    asm_euclidean_block(query+i, data+i, &sum);
}
```

### 4. **Riduzione Orizzontale SIMD Efficiente**
```assembly
; Riduzione orizzontale SSE per 32-bit
haddps  xmm0, xmm0               ; Somma orizzontale parziale
haddps  xmm0, xmm0               ; Riduzione finale a scalare

; Riduzione orizzontale AVX per 64-bit
vhaddpd ymm0, ymm0, ymm0         ; Somma parziale
vextractf128 xmm1, ymm0, 1       ; Estrai upper 128-bit  
vaddpd  xmm0, xmm0, xmm1         ; Combina upper + lower
```

---

## Performance Results

### Setup Professore (-O0 Baseline)

Le ottimizzazioni Assembly SIMD dimostrano miglioramenti significativi rispetto al codice C baseline non ottimizzato (`-O0`). I risultati variano in base a:
- **Dimensione del dataset** (cache effects, memory bandwidth)
- **Dimensionalità dei vettori** (efficienza SIMD)
- **Architettura e carico del sistema** (thermal throttling, background processes)

**Speedup tipici osservati:**
- **32-bit SSE**: 1.01x - 1.28x su dataset medi/grandi
- **64-bit AVX**: 1.03x - 1.11x con buona scalabilità
- **64-bit AVX + OpenMP**: 1.05x - 1.16x su dataset di grandi dimensioni

### Dimostrazione Educativa

**Con Ottimizzazioni Aggressive (`-O3 -march=native`):**
- Assembly e C ottimizzato hanno performance comparabili (~1.0x)
- Il compilatore moderno compete efficacemente con ottimizzazioni manuali

**Con Setup Professore (`-O0`):**
- Assembly ottimizzato mantiene prestazioni elevate
- C baseline non ottimizzato è significativamente più lento
- Speedup tipici: **1.5x - 3.0x** a seconda del carico di lavoro

 **Insight**: I compilatori moderni con ottimizzazioni aggressive rendono l'Assembly meno vantaggioso. Con baseline `-O0`, l'Assembly dimostra chiaramente i suoi benefici attraverso vectorization manuale, prefetching e loop unrolling espliciti.

---

## Setup e Utilizzo

### Build Setup

**Modalità C Baseline (default):**
```bash
cd ProgettoGruppo6
python3 setup.py build_ext --inplace
```

**Modalità Assembly Ottimizzato:**
```bash
cd ProgettoGruppo6
USE_ASM_32=1 USE_ASM_64=1 USE_ASM_OMP=1 python3 setup.py build_ext --inplace
```

### Testing e Benchmark

**Test Edge Cases (Robustezza):**
```bash
python3 test_edge_cases.py         # 50 test su dimensioni varie
```

**Benchmark Performance:**
```bash
python3 benchmark.py                # Performance C vs ASM
python3 benchmark_dimensioni.py     # Scalability analysis
```

**Validazione Correttezza:**
```bash
python3 compare_results.py          # Verifica risultati identici
```

### Controlli Assembly

| Variabile | Descrizione | 
|-----------|-------------|
| `USE_ASM_32=1` | Abilita Assembly SSE 32-bit |
| `USE_ASM_64=1` | Abilita Assembly AVX 64-bit |
| `USE_ASM_OMP=1` | Abilita Assembly AVX + OpenMP |

---

## Analisi Tecnica Approfondita

### Perché Assembly Offre Vantaggi con -O0?

1. **Vectorization Manuale**: Assembly SIMD esplicito vs codice scalare del compilatore
2. **Loop Unrolling Controllato**: Riduzione overhead branching e migliore utilizzo pipeline
3. **Instruction Selection Ottimale**: Uso diretto delle istruzioni più efficienti (SIMD packed operations)
4. **Eliminazione Overhead**: Accesso diretto ai registri, nessun spilling inutile

### Limitazioni Assembly con Ottimizzazioni Aggressive

Con `-O3 -march=native`:
- **Auto-vectorization** del compilatore genera codice SIMD competitivo
- **Loop optimization** automatica (unrolling, fusion, distribution)
- **Register allocation** sofisticata che minimizza memory accesses
- **Instruction scheduling** ottimizzato per la microarchitettura specifica

**Risultato**: Il gap tra C ottimizzato e Assembly si riduce significativamente, rendendo l'Assembly meno vantaggioso ma comunque utile per hot paths specifici.

---

## Struttura del Progetto

```
ProgettoGruppo6/
├── src/
│   ├── 32/                    # Implementazione SSE 32-bit
│   │   ├── quantpivot32.c     # Baseline C + Assembly switching
│   │   ├── quantpivot32.nasm  # Ottimizzazioni SSE manuali
│   │   └── sseutils32.nasm    # Utilities SIMD
│   ├── 64/                    # Implementazione AVX 64-bit
│   │   ├── quantpivot64.c     # Baseline C + Assembly switching  
│   │   ├── quantpivot64.nasm  # Ottimizzazioni AVX manuali
│   │   └── sseutils64.nasm    # Utilities SIMD avanzate
│   └── 64omp/                 # Implementazione AVX + OpenMP
│       ├── quantpivot64omp.c  # Multi-thread + SIMD
│       ├── quantpivot64omp.nasm # AVX + parallelismo
│       └── sseutils64.nasm    # SIMD utilities condivise
├── gruppo6/                   # Package Python
│   ├── quantpivot32/         # Wrapper 32-bit
│   ├── quantpivot64/         # Wrapper 64-bit  
│   └── quantpivot64omp/      # Wrapper parallel
├── test.py                   # Testing framework
├── benchmark.py              # Performance analysis
├── benchmark_dimensioni.py   # Scalability tests
├── compare_results.py        # Accuracy validation
└── setup.py                 # Build system con NASM
```

---

## Risultati Ottenuti

- **Implementazione Completa**: 3 versioni funzionanti (32-bit SSE, 64-bit AVX, 64-bit AVX+OpenMP)  
- **Dual Code Path**: Baseline C e Assembly ottimizzato con switching controllato  
- **Performance Competitive**: Assembly consistentemente più veloce su baseline C  
- **Testing Robusto**: 100% successo su 50 edge cases (dimensioni dispari, prime, non multipli)  
- **Scalabilità Verificata**: Test su dataset da 2K a 10K elementi, dimensioni 1-2048  
- **Correttezza Garantita**: Risultati identici tra versione C e Assembly  

---

*Questo progetto dimostra la padronanza delle ottimizzazioni Assembly SIMD moderne, la comprensione dei trade-off tra ottimizzazione manuale e compilatori, e la capacità di implementare soluzioni robuste e scalabili.*

cd ~/Scrivania/Progetto/Mio/ProgettoGruppo6
source venv/bin/activate
pip install -e .
cd ~/Scrivania/Progetto/Mio/ProgettoGruppo6
rm -rf build/ gruppo6/quantpivot64omp/*.so src/64omp/*.o
python3 setup.py build_ext --inplace
cd ~/Scrivania/Progetto/Mio
python3 test.py dataset_2000x256_64.ds2 query_2000x256_64.ds2 16 8 64 64
python3 compare_results.py

------------------------------------------------------------------------------

TEST VALIDITÀ PROGRAMMA SU DATASET GRANDI, DISPARI, NON MULTIPLI DI 4...

cd ~/Scrivania/Progetto/Mio/ProgettoGruppo6
source venv/bin/activate
pip install -e .
cd ~/Scrivania/Progetto/Mio
python3 test_edge_cases.py

------------------------------------------------------------------------------

CREAZIONE DI DATASET E QUERY DI DIMENSIONI DIFFERENTI

gcc make_ds2.c -O3 -o make_ds2
./make_ds2 200000 256 DS_200k_256.ds2
./make_ds2 2000 256 Q_2000_256.ds2
python3 test.py DS_200k_256.ds2 Q_2000_256.ds2 16 8 64 32
 
===============================================================================

GUIDA AL PROGETTO – Architetture Avanzate (Gruppo 6)

Il progetto implementa l’algoritmo QuantPivot per k-NN: quantizza i vettori in (v+, v-), usa una distanza approssimata
per ottenere un lower bound, fa pruning dei candidati e infine calcola la distanza euclidea reale solo sui candidati rimasti.

Perché è implementato così (scelte progettuali)
------------------------------------------------------------------------------
1) Stessa API e stessa logica in tutte le versioni (32 / 64 / 64omp)
   - Questo riduce bug, facilita il confronto e rende i risultati riproducibili.
2) Separazione tra:
   - logica “algoritmica” in C (fit/query/quantizzazione/indice)
   - kernel ottimizzati in Assembly (SSE/AVX) per le parti numericamente pesanti
   - wrapper Python (py.c) del template, mantenuto invariato per compatibilità con la valutazione
3) Prestazioni:
   - la parte costosa è: quantizzazione + calcolo distanza approssimata + distanza euclidea finale
   - per questo esistono kernel ASM (SSE/AVX) e, in 64omp, parallelismo sui loop principali (OpenMP).

Struttura delle directory
------------------------------------------------------------------------------
Nel progetto trovate tipicamente:

ProgettoGruppo6/
  gruppo6/                  -> package Python (import gruppo6...)
  src/
    32/                     -> versione float + SSE
    64/                     -> versione double + AVX
    64omp/                  -> versione double + AVX + OpenMP
  test.py                   -> script di test (template)
  compare_results.py         -> confronto risultati (se presente/ricreato)
  clean                      -> pulizia artefatti build (consigliato prima dello zip)

Le 3 directory src/* hanno nomi di file simili (stessa “intestazione”), ma implementano tipi e ottimizzazioni diverse.

Differenze tra src/32, src/64, src/64omp
------------------------------------------------------------------------------
1) src/32  (float + SSE)
   - Dati: float (32 bit)
   - Vettorizzazione: SSE (registri XMM, 4 float per volta)
   - Target: baseline ottimizzata per 32 bit

2) src/64  (double + AVX)
   - Dati: double (64 bit)
   - Vettorizzazione: AVX (registri YMM, 4 double per volta)
   - Target: maggiore precisione + throughput su double

3) src/64omp (double + AVX + OpenMP)
   - Come src/64 ma con #pragma omp parallel for sui loop “grandi”
   - Target: sfruttare più core su fit/query, mantenendo kernel AVX per inner-loop

Nota: i nomi dei file sono simili per uniformità (e per rimanere aderenti al template), ma:
- cambiano i tipi (float vs double)
- cambiano i kernel ASM (SSE vs AVX)
- in 64omp cambiano i loop (parallelizzati) e i flag di compilazione (OpenMP)

API Python (cosa vede chi usa il progetto)
------------------------------------------------------------------------------
Il progetto esporta 3 classi “gemelle”, una per directory:

- QuantPivot32
- QuantPivot64
- QuantPivot64OMP

Uso tipico:

qp = QuantPivotXX(...)
qp.fit(DS, h, x, ...)    # costruisce pivot, quantizza, costruisce indice
ids, dists = qp.query(Q, k, ...)   # risponde alle query con k-NN

Dove:
- DS è il dataset (N x D)
- Q  è il query set (nq x D)
- h  è il numero di pivot
- k  è il numero di vicini richiesti
- x  è il parametro di quantizzazione (soglia/numero selezioni non-zero)

Che cosa fa ogni componente (alto livello)

1) Wrapper Python (file *_py.c)
   - Converte gli array numpy -> puntatori C
   - Controlla parametri e chiama le funzioni C
   - Gestisce la vita dell’oggetto (alloc/free)
   - Nota: lasciato uguale al template per compatibilità

2) C “core” (file quantpivotXX.c)
   Contiene le funzioni principali, in genere:

   a) fit()
      - alloca e inizializza strutture
      - genera i pivot P (h vettori)
      - quantizza DS in (v_plus, v_minus)
      - costruisce l’indice idx[v, j] = d̃(v, Pj) per tutti i punti v e pivot j

   b) query()
      - per ogni query q:
        1) quantizza q in (q_plus, q_minus)
        2) calcola qpivot[j] = d̃(q, Pj)
        3) per ogni v nel dataset calcola lower bound:
             LB(v,q) = max_j |idx[v,j] - qpivot[j]|
        4) seleziona candidati migliori per LB (pruning)
        5) calcola distanza euclidea reale sui candidati
        6) restituisce i k vicini (id + dist)

   c) quantizing() (o equivalente)
      - trasforma un vettore reale in due vettori non-negativi:
        v+ contiene i contributi positivi
        v- contiene i contributi negativi (in valore assoluto)
      - tipicamente “sparse” controllata da x (sceglie le componenti più rilevanti)

   d) approx_distance()
      - implementa la distanza approssimata d̃:
        (v+·w+) + (v-·w-) – (v+·w-) – (v-·w+)

   e) euclidean_distance()
      - distanza reale, usata solo in finale su pochi candidati

3) ASM (file quantpivotXX.nasm + utils)
   - Implementa i kernel “hot” (dot product / distanze) in SSE o AVX
   - Riduce overhead dei loop e sfrutta SIMD (unrolling, prefetch, ecc.)

4) Script run/benchmark (se presenti)
   - Servono per test locali, confronto C vs ASM, e misurazioni tempo

Differenze tra file con “stessa intestazione” (esempi)
------------------------------------------------------------------------------
- quantpivot32.c vs quantpivot64.c:
  * stesso algoritmo, ma float vs double
  * cambiamento di tipi e, spesso, step SIMD (4 float vs 4 double per AVX)

- quantpivot32.nasm vs quantpivot64.nasm:
  * SSE (XMM) per 32
  * AVX (YMM) per 64

- quantpivot64.nasm vs quantpivot64omp.nasm:
  * kernel AVX simili (a volte identici)
  * la differenza principale è nel C: parallelizzazione OpenMP sui loop esterni

Note pratiche per chi clona e compila
------------------------------------------------------------------------------
1) Prima di zippare o pushare:
   - eseguire ./clean
   - non includere build/, *.o, *.so, __pycache__/ (evita problemi a chi compila)

2) Installazione/editable:
   - dentro ProgettoGruppo6:  pip install -e .
   - poi usare test.py dalla root per eseguire i test

3) Debug:
   - se servono log, farli in C (printf/fflush) nelle funzioni chiave:
     fit(): pivot/alloc/quantizzazione/indice
     query(): lower bound/candidati/kNN finale

Glossario rapido:
- P: matrice dei pivot (h x D)
- v+, v-: vettori quantizzati non-negativi
- idx[v,j]: valore d̃ tra punto v e pivot j (indice)
- qpivot[j]: valore d̃ tra query q e pivot j
- LB: lower bound per pruning (massimo scarto sugli h pivot)
- pruning: elimina punti sicuramente non nei k migliori prima della distanza reale

---

## Guida Completa ai Test e Benchmarks

### Setup Iniziale dell'Ambiente

```bash
# 1. Navigare nella directory del progetto
cd ~/Scrivania/Benedetta/ProgettoGruppo6

# 2. Installare dipendenze di sistema
sudo apt install python3-venv
sudo apt update
sudo apt install libfftw3-dev libfftw3-doc

# 3. Creare e attivare virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Aggiornare pip e installare pacchetti Python
pip install --upgrade pip setuptools wheel
pip install numpy
pip install pyfftw

# 5. Build iniziale del progetto
python3 setup.py build_ext --inplace
pip install -e .
```

---

### Test delle Diverse Versioni

#### **Versione 32-bit C (Baseline)**

```bash
# Build versione C normale
cd ~/Scrivania/Benedetta/ProgettoGruppo6
rm -rf build/ gruppo6/quantpivot32/*.so src/32/*.o
python3 setup.py build_ext --inplace

# Test
cd ~/Scrivania/Benedetta
python3 test.py dataset_2000x256_32.ds2 query_2000x256_32.ds2 16 8 64 32

# Verifica risultati
python3 compare_results.py --t 32
```

**Output Atteso:**
```
PREDICT: C version
FIT time: ~0.094 seconds
PRD time: ~5.97 seconds
ID identici: True
Distanze compatibili: True
```

---

#### **Versione 32-bit SSE Assembly**

```bash
# Build con assembly SSE attivato
cd ~/Scrivania/Benedetta/ProgettoGruppo6
rm -rf build/ gruppo6/quantpivot32/*.so src/32/*.o
USE_ASM_32=1 python3 setup.py build_ext --inplace

# Test
cd ~/Scrivania/Benedetta
python3 test.py dataset_2000x256_32.ds2 query_2000x256_32.ds2 16 8 64 32

# Verifica risultati
python3 compare_results.py --t 32
```

**Output Atteso:**
```
Assembly 32-bit ABILITATO
PREDICT: ASM version
FIT time: ~0.049 seconds (~48% faster)
PRD time: ~2.09 seconds (~65% faster)
ID identici: True
Distanze compatibili: True
```

---

#### **Versione 64-bit C (Baseline)**

```bash
# Build versione C normale
cd ~/Scrivania/Benedetta/ProgettoGruppo6
rm -rf build/ gruppo6/quantpivot64/*.so src/64/*.o
python3 setup.py build_ext --inplace

# Test
cd ~/Scrivania/Benedetta
python3 test.py dataset_2000x256_64.ds2 query_2000x256_64.ds2 16 8 64 64

# Verifica risultati
python3 compare_results.py --t 64
```

**Output Atteso:**
```
FIT: C version
PREDICT: C version
FIT time: ~0.123 seconds
PRD time: ~6.73 seconds
ID identici: True
Distanze compatibili: True
```

---

#### **Versione 64-bit AVX Assembly**

```bash
# Build con assembly AVX attivato
cd ~/Scrivania/Benedetta/ProgettoGruppo6
rm -rf build/ gruppo6/quantpivot64/*.so src/64/*.o
USE_ASM_64=1 python3 setup.py build_ext --inplace

# Test
cd ~/Scrivania/Benedetta
python3 test.py dataset_2000x256_64.ds2 query_2000x256_64.ds2 16 8 64 64

# Verifica risultati
python3 compare_results.py --t 64
```

**Output Atteso:**
```
Assembly 64-bit ABILITATO
FIT: ASM version
PREDICT: ASM version
FIT time: ~0.048 seconds (~60% faster)
PRD time: ~2.49 seconds (~63% faster)
ID identici: True
Distanze compatibili: True
```

---

#### **Versione 64-bit C + OpenMP**

```bash
# Build versione C con OpenMP
cd ~/Scrivania/Benedetta/ProgettoGruppo6
rm -rf build/ gruppo6/quantpivot64omp/*.so src/64omp/*.o
python3 setup.py build_ext --inplace

# Test
cd ~/Scrivania/Benedetta
python3 test.py dataset_2000x256_64.ds2 query_2000x256_64.ds2 16 8 64 64omp

# Verifica risultati
python3 compare_results.py --t 64omp
```

**Output Atteso:**
```
VERSIONE 64-bit OPENMP
FIT: C version
PREDICT: C version
OpenMP threads: 4
FIT time: ~0.066 seconds
PRD time: ~3.34 seconds
ID identici: True
Distanze compatibili: True
```

---

#### **Versione 64-bit AVX Assembly + OpenMP (Massime Prestazioni)**

```bash
# Build con assembly AVX + OpenMP attivato
cd ~/Scrivania/Benedetta/ProgettoGruppo6
rm -rf build/ gruppo6/quantpivot64omp/*.so src/64omp/*.o
USE_ASM_OMP=1 python3 setup.py build_ext --inplace

# Test
cd ~/Scrivania/Benedetta
python3 test.py dataset_2000x256_64.ds2 query_2000x256_64.ds2 16 8 64 64omp

# Verifica risultati
python3 compare_results.py --t 64omp
```

**Output Atteso:**
```
Assembly 64-bit OpenMP ABILITATO
VERSIONE 64-bit OPENMP
FIT: ASM version
PREDICT: ASM version
OpenMP threads: 4
FIT time: ~0.042 seconds
PRD time: ~1.75 seconds (FASTEST!)
ID identici: True
Distanze compatibili: True
```

---

### **Riepilogo Performance**

| Versione | FIT time | PRD time | Speedup PRD vs C |
|----------|----------|----------|------------------|
| **32-bit C** | 0.094s | 5.97s | baseline |
| **32-bit SSE ASM** | 0.049s | 2.09s | **~65% faster** |
| **64-bit C** | 0.123s | 6.73s | baseline |
| **64-bit AVX ASM** | 0.048s | 2.49s | **~63% faster** |
| **64-bit C + OMP** | 0.066s | 3.34s | baseline |
| **64-bit AVX ASM + OMP** | 0.042s | 1.75s | **~48% faster**  |

---

### **Variabili d'Ambiente per Build**

- **`USE_ASM_32=1`** - Abilita assembly SSE per 32-bit
- **`USE_ASM_64=1`** - Abilita assembly AVX per 64-bit  
- **`USE_ASM_OMP=1`** - Abilita assembly AVX + OpenMP per 64-bit

### **Note sui Flag di Compilazione**

Quando assembly è abilitato, il build aggiunge automaticamente:
- `-DUSE_ASM_APPROX`
- `-DUSE_ASM_EUCLIDEAN` 
- `-DUSE_ASM_LOWER_BOUND`

Questo attiva i percorsi assembly ottimizzati nel codice C.

---

Fine.
