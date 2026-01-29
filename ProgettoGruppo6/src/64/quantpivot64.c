#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <float.h>
#include "common.h"

#ifdef USE_ASM_APPROX
extern double approx_distance_asm(const double* vplus, const double* vminus,
                                  const double* wplus, const double* wminus,
                                  int D);
#endif
#ifdef USE_ASM_EUCLIDEAN
extern double euclidean_distance_asm(const double* v, const double* w, int D);
#endif
#ifdef USE_ASM_LOWER_BOUND
extern double compute_lower_bound_asm(const double* idx_v, const double* qpivot, int h);
#endif

// Prototipi funzioni C baseline
double approx_distance(const double* vplus, const double* vminus, const double* wplus, const double* wminus, int D);
double euclidean_distance(const double* v, const double* w, int D);
double compute_lower_bound(const double* idx_v, const double* qpivot, int h);

// Wrapper selettivi: compile-time switch
static inline double approx_distance_sel(const double* vplus, const double* vminus,
                                         const double* wplus, const double* wminus,
                                         int D) {
#if defined(USE_ASM_APPROX)
    return approx_distance_asm(vplus, vminus, wplus, wminus, D);
#else
    return approx_distance(vplus, vminus, wplus, wminus, D);
#endif
}

static inline double euclidean_distance_sel(const double* v, const double* w, int D) {
#if defined(USE_ASM_EUCLIDEAN)
    return euclidean_distance_asm(v, w, D);
#else
    return euclidean_distance(v, w, D);
#endif
}

static inline double compute_lower_bound_sel(const double* idx_v, const double* qpivot, int h) {
#if defined(USE_ASM_LOWER_BOUND)
    return compute_lower_bound_asm(idx_v, qpivot, h);
#else
    return compute_lower_bound(idx_v, qpivot, h);
#endif
}
#define PREFETCH_DIST 16  // Distanza di prefetching

extern double approx_distance_asm(const double* vplus, const double* vminus,
                                  const double* wplus, const double* wminus,
                                  int D);

extern double euclidean_distance_asm(const double* v, const double* w, int D);

extern double compute_lower_bound_asm(const double* idx_v, const double* qpivot, int h);

void* checked_alloc(size_t size) {
    void* p = _mm_malloc(size, 32);  // AVX: 32-byte alignment forzato per performance
    if (!p) {
        printf("ERRORE: impossibile allocare %lu bytes\n", size);
        fflush(stdout);
        exit(1);
    }
    return p;
}

// ==============================
// MAX-HEAP per K-NN
// ==============================

typedef struct {
    int id;
    type dist;
} neighbor;

typedef struct {
    neighbor* heap;
    int size;
    int capacity;
} MaxHeap;

static inline void heap_init(MaxHeap* h, int k) {
    h->heap = (neighbor*)malloc(k * sizeof(neighbor));
    h->size = 0;
    h->capacity = k;
    for(int i = 0; i < k; i++) {
        h->heap[i].id = -1;
        h->heap[i].dist = DBL_MAX;
    }
}

static inline void heap_free(MaxHeap* h) {
    free(h->heap);
}

static inline void heap_swap(neighbor* a, neighbor* b) {
    neighbor temp = *a;
    *a = *b;
    *b = temp;
}

static inline void heap_sift_down(MaxHeap* h, int idx) {
    int largest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;

    if(left < h->size && h->heap[left].dist > h->heap[largest].dist)
        largest = left;
    if(right < h->size && h->heap[right].dist > h->heap[largest].dist)
        largest = right;

    if(largest != idx) {
        heap_swap(&h->heap[idx], &h->heap[largest]);
        heap_sift_down(h, largest);
    }
}

static inline void heap_sift_up(MaxHeap* h, int idx) {
    while(idx > 0) {
        int parent = (idx - 1) / 2;
        if(h->heap[parent].dist >= h->heap[idx].dist)
            break;
        heap_swap(&h->heap[parent], &h->heap[idx]);
        idx = parent;
    }
}

static inline int heap_try_insert(MaxHeap* h, int id, type dist) {
    if(h->size < h->capacity) {
        h->heap[h->size].id = id;
        h->heap[h->size].dist = dist;
        heap_sift_up(h, h->size);
        h->size++;
        return 1;
    } else if(dist < h->heap[0].dist) {
        h->heap[0].id = id;
        h->heap[0].dist = dist;
        heap_sift_down(h, 0);
        return 1;
    }
    return 0;
}

static inline type heap_max_dist(MaxHeap* h) {
    return (h->size > 0) ? h->heap[0].dist : DBL_MAX;
}

// ==============================
// QUICKSELECT per Quantizzazione
// ==============================

static inline void swap_pair(type* vals, int* indices, int i, int j) {
    type tmp_val = vals[i];
    vals[i] = vals[j];
    vals[j] = tmp_val;
    
    int tmp_idx = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp_idx;
}

static inline int partition(type* vals, int* indices, int left, int right, int pivot_idx) {
    type pivot_value = vals[pivot_idx];
    swap_pair(vals, indices, pivot_idx, right);
    
    int store_idx = left;
    for(int i = left; i < right; i++) {
        if(vals[i] > pivot_value) {
            swap_pair(vals, indices, i, store_idx);
            store_idx++;
        }
    }
    
    swap_pair(vals, indices, store_idx, right);
    return store_idx;
}

static void quickselect_top_x(type* vals, int* indices, int left, int right, int x) {
    if(left >= right || x <= 0) return;
    
    while(left < right) {
        int mid = left + (right - left) / 2;
        if(vals[mid] > vals[left]) swap_pair(vals, indices, left, mid);
        if(vals[right] > vals[left]) swap_pair(vals, indices, left, right);
        if(vals[mid] > vals[right]) swap_pair(vals, indices, mid, right);
        
        int pivot_idx = mid;
        int new_pivot = partition(vals, indices, left, right, pivot_idx);
        
        if(new_pivot == x - 1) {
            return;
        } else if(new_pivot > x - 1) {
            right = new_pivot - 1;
        } else {
            left = new_pivot + 1;
        }
    }
}

// ==============================
// QUANTIZZAZIONE OTTIMIZZATA AVX con Quickselect
// ==============================

// Versione che riusa buffer scratch (NO malloc/free per ogni chiamata)
static inline void quantize_vector_scratch(const type* v,
                                           type* vplus,
                                           type* vminus,
                                           int x,
                                           int D,
                                           int* indices,
                                           type* abs_vals)
{
    memset(vplus, 0, D * sizeof(type));
    memset(vminus, 0, D * sizeof(type));

    if(x <= 0) return;
    if(x > D) x = D;

    // C puro - calcola valori assoluti
    for(int i = 0; i < D; i++) {
        indices[i] = i;
        abs_vals[i] = fabs(v[i]);
    }
    
    // Quickselect per partizionare i top-x - O(D)
    quickselect_top_x(abs_vals, indices, 0, D - 1, x);
    
    // Insertion Sort SOLO sui primi x elementi - O(x²)
    for(int i = 1; i < x; i++) {
        type key_val = abs_vals[i];
        int key_idx = indices[i];
        int j = i - 1;
        
        while(j >= 0 && abs_vals[j] < key_val) {
            abs_vals[j + 1] = abs_vals[j];
            indices[j + 1] = indices[j];
            j--;
        }
        abs_vals[j + 1] = key_val;
        indices[j + 1] = key_idx;
    }
    
    // Imposta bit
    for(int count = 0; count < x; count++) {
        int idx = indices[count];
        if(v[idx] >= 0) {
            vplus[idx] = 1.0;
        } else {
            vminus[idx] = 1.0;
        }
    }
}

// Versione originale per compatibilità (usata in fit)
void quantize_vector(type* v, type* vplus, type* vminus, int x, int D) {
    int* indices = (int*)malloc(D * sizeof(int));
    type* abs_vals = (type*)_mm_malloc(D * sizeof(type), align);
    
    quantize_vector_scratch(v, vplus, vminus, x, D, indices, abs_vals);
    
    _mm_free(abs_vals);
    free(indices);
}

// ==============================
// DISTANZE OTTIMIZZATE AVX
// ==============================

type approx_distance_c(const type* vplus, const type* vminus, const type* wplus, const type* wminus, int D) {
    type dot_pp = 0.0;
    type dot_mm = 0.0;
    type dot_pm = 0.0;
    type dot_mp = 0.0;
    for (int i = 0; i < D; i++) {
        dot_pp += vplus[i] * wplus[i];
        dot_mm += vminus[i] * wminus[i];
        dot_pm += vplus[i] * wminus[i];
        dot_mp += vminus[i] * wplus[i];
    }
    return dot_pp + dot_mm - dot_pm - dot_mp;
}

double approx_distance(const double* vplus, const double* vminus,
                       const double* wplus, const double* wminus,
                       int D)
{
#ifdef USE_ASM_APPROX
    if (D <= 0) return 0.0;
    return approx_distance_asm(vplus, vminus, wplus, wminus, D);
#else
    return approx_distance_c(vplus, vminus, wplus, wminus, D);
#endif
}

type euclidean_distance_c(const type* v, const type* w, int D) {
    type sum_sq = 0.0;
    for (int i = 0; i < D; i++) {
        type diff = v[i] - w[i];
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq);
}

type euclidean_distance(const type* v, const type* w, int D) {
#ifdef USE_ASM_EUCLIDEAN
    if (D <= 0) return 0.0;
    return euclidean_distance_asm(v, w, D);
#else
    return euclidean_distance_c(v, w, D);
#endif
}

type compute_lower_bound_c(const type* idx_v, const type* qpivot, int h) {
    type max_lb = 0.0;
    for (int j = 0; j < h; j++) {
        type diff = fabs(idx_v[j] - qpivot[j]);
        if (diff > max_lb) {
            max_lb = diff;
        }
    }
    return max_lb;
}

type compute_lower_bound(const type* idx_v, const type* qpivot, int h) {
#ifdef USE_ASM_LOWER_BOUND
    if (h <= 0) return 0.0;
    return compute_lower_bound_asm(idx_v, qpivot, h);
#else
    return compute_lower_bound_c(idx_v, qpivot, h);
#endif
}

// Lower bound con early exit - esce appena LB >= worst_dist
static inline type compute_lower_bound_thresh(const type* idx_v,
                                              const type* qpivot,
                                              int h,
                                              type worst_dist)
{
    type LB = 0.0;
    for (int j = 0; j < h; j++) {
        type diff = fabs(idx_v[j] - qpivot[j]);
        if (diff > LB) {
            LB = diff;
            if (LB >= worst_dist) return LB; // EARLY EXIT: pruning garantito
        }
    }
    return LB;
}

// ==============================
// FIT
// ==============================

void fit(params* input) {
    if(!input->silent) {
        #if defined(USE_ASM_APPROX) || defined(USE_ASM_EUCLIDEAN) || defined(USE_ASM_LOWER_BOUND)
            printf("FIT: ASM version\n");
            printf("[DEBUG] approx_distance: versione ASM AVX attiva\n");
            printf("[DEBUG] euclidean_distance: versione ASM AVX attiva\n");
            printf("[DEBUG] lower_bound: versione ASM AVX attiva\n");
        #else
            printf("FIT: C version\n");
            printf("[DEBUG] approx_distance: versione C baseline attiva\n");
            printf("[DEBUG] euclidean_distance: versione C baseline attiva\n");
            printf("[DEBUG] lower_bound: versione C baseline attiva\n");
        #endif
        printf("[OPTIMIZATION] Quantizzazione: Quickselect O(D)\n");
        printf("[OPTIMIZATION] KNN Search: Max-Heap O(log k)\n");
        fflush(stdout);
    }

    if (input->first_fit_call == false) {
        if(!input->silent) printf("DEBUG: Prima chiamata a fit(), inizializzo puntatori...\n");
        input->P = NULL;
        input->ds_plus = NULL;
        input->ds_minus = NULL;
        input->index = NULL;
        input->first_fit_call = true;
    }

    int N = input->N;
    int D = input->D;
    int h = input->h;
    int x = input->x;

    if(!input->silent) {
        printf("FIT PARAMS: N=%d, D=%d, h=%d, x=%d\n", N, D, h, x);
        fflush(stdout);
    }

    if(input->DS == NULL){
        printf("ERRORE: input->DS è NULL! Abort.\n");
        exit(1);
    }

    if(input->P != NULL){
        if(!input->silent) printf("DEBUG: libero P precedente...\n");
        _mm_free(input->P);
    }

    input->P = checked_alloc(h * sizeof(int));
    if(!input->silent) printf("DEBUG: P allocato = %p\n", input->P);

    int step = N / h;
    for(int j = 0; j < h; j++){
        input->P[j] = j * step;
    }

    if(!input->silent) printf("DEBUG: Pivot generati correttamente.\n");

    if(input->ds_plus != NULL){
        if(!input->silent) printf("DEBUG: libero ds_plus precedente...\n");
        _mm_free(input->ds_plus);
    }
    if(input->ds_minus != NULL){
        if(!input->silent) printf("DEBUG: libero ds_minus precedente...\n");
        _mm_free(input->ds_minus);
    }

    input->ds_plus = checked_alloc(N * D * sizeof(type));
    input->ds_minus = checked_alloc(N * D * sizeof(type));

    if(!input->silent) printf("DEBUG: Allocati ds_plus=%p, ds_minus=%p\n", input->ds_plus, input->ds_minus);

    // Scratch buffer riusato (evita malloc per ogni vettore)
    int*  scratch_idx = (int*)malloc(D * sizeof(int));
    type* scratch_abs = (type*)checked_alloc(D * sizeof(type));

    for(int i = 0; i < N; i++){
        if(!input->silent && i % 500 == 0){
            printf("DEBUG: Quantizzo DS[%d/%d]\n", i, N);
            fflush(stdout);
        }

        quantize_vector_scratch(&input->DS[i * D],
                                &input->ds_plus[i * D],
                                &input->ds_minus[i * D],
                                x, D,
                                scratch_idx, scratch_abs);

        if (i + 4 < N) {
            __builtin_prefetch(&input->DS[(i+4) * D], 0, 1);
            __builtin_prefetch(&input->ds_plus[(i+4) * D], 1, 1);
            __builtin_prefetch(&input->ds_minus[(i+4) * D], 1, 1);
        }
    }

    free(scratch_idx);
    _mm_free(scratch_abs);

    if(!input->silent) printf("DEBUG: Quantizzazione dataset completata.\n");

    if(input->index != NULL){
        if(!input->silent) printf("DEBUG: libero index precedente...\n");
        _mm_free(input->index);
    }

    input->index = checked_alloc(N * h * sizeof(type));
    if(!input->silent) printf("DEBUG: index allocato = %p\n", input->index);

    for(int i = 0; i < N; i++){
        if(!input->silent && i % 500 == 0){
            printf("DEBUG: costruzione indice [%d/%d]\n", i, N);
            fflush(stdout);
        }

        for(int j = 0; j < h; j++){
            int pivot_idx = input->P[j];

            input->index[i*h + j] = approx_distance_sel(
                &input->ds_plus[i * D],    
                &input->ds_minus[i * D],
                &input->ds_plus[pivot_idx * D], 
                &input->ds_minus[pivot_idx * D],
                D
            );
        }
    }

    if(!input->silent) {
        printf("DEBUG: Index costruito.\n");
        printf("FIT COMPLETATO.\n");
        fflush(stdout);
    }
}

void predict(params* input) {
    if(!input->silent) {
        #if defined(USE_ASM_APPROX) || defined(USE_ASM_EUCLIDEAN) || defined(USE_ASM_LOWER_BOUND)
            printf("PREDICT: ASM version\n");
        #else
            printf("PREDICT: C version\n");
        #endif
        fflush(stdout);
    }

    int nq = input->nq;
    int N  = input->N;
    int D  = input->D;
    int h  = input->h;
    int k  = input->k;
    int x  = input->x;

    if(input->ds_plus == NULL || input->ds_minus == NULL){
        printf("ERRORE: predict() chiamata prima di fit()!\n");
        exit(1);
    }

    if(input->Q == NULL){
        printf("ERRORE: input->Q è NULL!\n");
        exit(1);
    }

    // ============================================================
    // BATCH QUANTIZZAZIONE QUERY (evita alloc per ogni query)
    // ============================================================
    
    MATRIX q_plus  = (type*)checked_alloc(nq * D * sizeof(type));
    MATRIX q_minus = (type*)checked_alloc(nq * D * sizeof(type));

    // Scratch per quantizzazione (riusato per ogni query)
    int*  scratch_idx = (int*)malloc(D * sizeof(int));
    type* scratch_abs = (type*)checked_alloc(D * sizeof(type));

    for(int q = 0; q < nq; q++){
        quantize_vector_scratch(&input->Q[q * D],
                                &q_plus[q * D],
                                &q_minus[q * D],
                                x, D, scratch_idx, scratch_abs);
    }

    free(scratch_idx);
    _mm_free(scratch_abs);

    // ============================================================
    // COPIA PIVOT IN MEMORIA CONTIGUA (cache-friendly)
    // ============================================================
    
    MATRIX pivot_plus  = (type*)checked_alloc(h * D * sizeof(type));
    MATRIX pivot_minus = (type*)checked_alloc(h * D * sizeof(type));

    for(int j = 0; j < h; j++){
        int p = input->P[j];
        memcpy(&pivot_plus[j * D],  &input->ds_plus[p * D],  D * sizeof(type));
        memcpy(&pivot_minus[j * D], &input->ds_minus[p * D], D * sizeof(type));
    }

    // ============================================================
    // STRUTTURA K-NN LINEARE (identica al prof)
    // ============================================================
    
    neighbor* knn = (neighbor*)malloc(k * sizeof(neighbor));
    type* qpivot = (type*)malloc(h * sizeof(type));

    // ============================================================
    // LOOP QUERY
    // ============================================================
    
    for(int q = 0; q < nq; q++){
        
        // Inizializza k-NN a infinito
        for(int i = 0; i < k; i++){
            knn[i].id = -1;
            knn[i].dist = DBL_MAX;
        }

        // Puntatori alla query quantizzata
        type* qplus_q  = &q_plus[q * D];
        type* qminus_q = &q_minus[q * D];

        // Precalcolo distanze query-pivot
        for(int j = 0; j < h; j++){
            qpivot[j] = approx_distance_sel(
                qplus_q, qminus_q,
                &pivot_plus[j * D], &pivot_minus[j * D],
                D
            );
        }

        type worst_dist = DBL_MAX;
        int  worst_idx  = 0;

        // ============================================================
        // SCANSIONE DATASET - LOOP UNROLLING x4 + BATCH PREFETCH
        // ============================================================
        
        int v = 0;
        
        // Batch prefetch aggressivo per le prime 64 iterazioni
        for (int pf = 0; pf < 64 && pf < N; pf++) {
            __builtin_prefetch(&input->index[(size_t)pf * (size_t)h], 0, 0);
            __builtin_prefetch(&input->ds_plus[(size_t)pf * (size_t)D], 0, 0);
            __builtin_prefetch(&input->ds_minus[(size_t)pf * (size_t)D], 0, 0);
        }
        
        // Loop unrolled x4 - processa 4 candidati per iterazione
        for (; v <= N - 4; v += 4) {
            // Prefetch continuo per mantenere pipeline piena
            if (v + 64 < N) {
                __builtin_prefetch(&input->index[(size_t)(v+64) * (size_t)h], 0, 0);
                __builtin_prefetch(&input->ds_plus[(size_t)(v+64) * (size_t)D], 0, 0);
                __builtin_prefetch(&input->ds_minus[(size_t)(v+64) * (size_t)D], 0, 0);
            }

            // --- Candidato v+0 ---
            type* idx_v0 = &input->index[(size_t)(v+0) * (size_t)h];
            type LB0 = compute_lower_bound_sel(idx_v0, qpivot, h);
            if (LB0 < worst_dist) {
                type* vplus_v0  = &input->ds_plus[(size_t)(v+0) * (size_t)D];
                type* vminus_v0 = &input->ds_minus[(size_t)(v+0) * (size_t)D];
                type d0 = approx_distance_sel(qplus_q, qminus_q, vplus_v0, vminus_v0, D);
                if (d0 < worst_dist) {
                    knn[worst_idx].id = v+0;
                    knn[worst_idx].dist = d0;
                    worst_dist = knn[0].dist; worst_idx = 0;
                    for (int i = 1; i < k; i++) {
                        if (knn[i].dist > worst_dist) { worst_dist = knn[i].dist; worst_idx = i; }
                    }
                }
            }

            // --- Candidato v+1 ---
            type* idx_v1 = &input->index[(size_t)(v+1) * (size_t)h];
            type LB1 = compute_lower_bound_sel(idx_v1, qpivot, h);
            if (LB1 < worst_dist) {
                type* vplus_v1  = &input->ds_plus[(size_t)(v+1) * (size_t)D];
                type* vminus_v1 = &input->ds_minus[(size_t)(v+1) * (size_t)D];
                type d1 = approx_distance_sel(qplus_q, qminus_q, vplus_v1, vminus_v1, D);
                if (d1 < worst_dist) {
                    knn[worst_idx].id = v+1;
                    knn[worst_idx].dist = d1;
                    worst_dist = knn[0].dist; worst_idx = 0;
                    for (int i = 1; i < k; i++) {
                        if (knn[i].dist > worst_dist) { worst_dist = knn[i].dist; worst_idx = i; }
                    }
                }
            }

            // --- Candidato v+2 ---
            type* idx_v2 = &input->index[(size_t)(v+2) * (size_t)h];
            type LB2 = compute_lower_bound_sel(idx_v2, qpivot, h);
            if (LB2 < worst_dist) {
                type* vplus_v2  = &input->ds_plus[(size_t)(v+2) * (size_t)D];
                type* vminus_v2 = &input->ds_minus[(size_t)(v+2) * (size_t)D];
                type d2 = approx_distance_sel(qplus_q, qminus_q, vplus_v2, vminus_v2, D);
                if (d2 < worst_dist) {
                    knn[worst_idx].id = v+2;
                    knn[worst_idx].dist = d2;
                    worst_dist = knn[0].dist; worst_idx = 0;
                    for (int i = 1; i < k; i++) {
                        if (knn[i].dist > worst_dist) { worst_dist = knn[i].dist; worst_idx = i; }
                    }
                }
            }

            // --- Candidato v+3 ---
            type* idx_v3 = &input->index[(size_t)(v+3) * (size_t)h];
            type LB3 = compute_lower_bound_sel(idx_v3, qpivot, h);
            if (LB3 < worst_dist) {
                type* vplus_v3  = &input->ds_plus[(size_t)(v+3) * (size_t)D];
                type* vminus_v3 = &input->ds_minus[(size_t)(v+3) * (size_t)D];
                type d3 = approx_distance_sel(qplus_q, qminus_q, vplus_v3, vminus_v3, D);
                if (d3 < worst_dist) {
                    knn[worst_idx].id = v+3;
                    knn[worst_idx].dist = d3;
                    worst_dist = knn[0].dist; worst_idx = 0;
                    for (int i = 1; i < k; i++) {
                        if (knn[i].dist > worst_dist) { worst_dist = knn[i].dist; worst_idx = i; }
                    }
                }
            }
        }

        // Coda: elementi rimanenti (N non multiplo di 4)
        for (; v < N; v++) {
            type* idx_v = &input->index[(size_t)v * (size_t)h];
            type LB = compute_lower_bound_sel(idx_v, qpivot, h);
            if (LB >= worst_dist) continue;

            type* vplus_v  = &input->ds_plus[(size_t)v * (size_t)D];
            type* vminus_v = &input->ds_minus[(size_t)v * (size_t)D];
            type d_approx = approx_distance_sel(qplus_q, qminus_q, vplus_v, vminus_v, D);
            if (d_approx < worst_dist) {
                knn[worst_idx].id = v;
                knn[worst_idx].dist = d_approx;
                worst_dist = knn[0].dist; worst_idx = 0;
                for (int i = 1; i < k; i++) {
                    if (knn[i].dist > worst_dist) { worst_dist = knn[i].dist; worst_idx = i; }
                }
            }
        }

        // ============================================================
        // RAFFINAMENTO con distanze euclidee
        // ============================================================
        
        type* query_base = &input->Q[q * D];
        for(int i = 0; i < k; i++){
            if(knn[i].id >= 0) {
                knn[i].dist = euclidean_distance_sel(
                    query_base,
                    &input->DS[knn[i].id * D],
                    D
                );
            }
        }

        // ============================================================
        // SALVATAGGIO - ORDINE INTERNO INVARIATO
        // ============================================================
        
        for(int i = 0; i < k; i++){
            input->id_nn[q * k + i]   = knn[i].id;
            input->dist_nn[q * k + i] = knn[i].dist;
        }
    }

    // Cleanup
    free(qpivot);
    free(knn);
    _mm_free(q_plus);
    _mm_free(q_minus);
    _mm_free(pivot_plus);
    _mm_free(pivot_minus);
}