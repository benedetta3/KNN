#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <float.h>
#include <omp.h>
#include "common.h"

// ============================================================================
// VERSIONE 64-bit OPENMP - Selezionabile C/ASM con parallelizzazione OpenMP
// ============================================================================

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
double approx_distance_c(const double* vplus, const double* vminus, const double* wplus, const double* wminus, int D);
double euclidean_distance_c(const double* v, const double* w, int D);
double compute_lower_bound_c(const double* idx_v, const double* qpivot, int h);

// Wrapper selettivi
static inline double approx_distance_sel(const double* vplus, const double* vminus,
                                         const double* wplus, const double* wminus,
                                         int D) {
#if defined(USE_ASM_APPROX)
    return approx_distance_asm(vplus, vminus, wplus, wminus, D);
#else
    return approx_distance_c(vplus, vminus, wplus, wminus, D);
#endif
}

static inline double euclidean_distance_sel(const double* v, const double* w, int D) {
#if defined(USE_ASM_EUCLIDEAN)
    return euclidean_distance_asm(v, w, D);
#else
    return euclidean_distance_c(v, w, D);
#endif
}

static inline double compute_lower_bound_sel(const double* idx_v, const double* qpivot, int h) {
#if defined(USE_ASM_LOWER_BOUND)
    return compute_lower_bound_asm(idx_v, qpivot, h);
#else
    return compute_lower_bound_c(idx_v, qpivot, h);
#endif
}

void* checked_alloc(size_t size) {
    void* p = _mm_malloc(size, 32);
    if (!p) {
        printf("ERRORE: impossibile allocare %lu bytes\n", size);
        fflush(stdout);
        exit(1);
    }
    return p;
}

// ==============================
// QUICKSELECT
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
// QUANTIZZAZIONE
// ==============================

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

    for(int i = 0; i < D; i++) {
        indices[i] = i;
        abs_vals[i] = fabs(v[i]);
    }
    
    quickselect_top_x(abs_vals, indices, 0, D - 1, x);
    
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
    
    for(int count = 0; count < x; count++) {
        int idx = indices[count];
        if(v[idx] >= 0) {
            vplus[idx] = 1.0;
        } else {
            vminus[idx] = 1.0;
        }
    }
}

// ==============================
// DISTANZE C BASELINE
// ==============================

double approx_distance_c(const double* vplus, const double* vminus, 
                        const double* wplus, const double* wminus, int D) {
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

double euclidean_distance_c(const double* v, const double* w, int D) {
    type sum_sq = 0.0;
    for (int i = 0; i < D; i++) {
        type diff = v[i] - w[i];
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq);
}

double compute_lower_bound_c(const double* idx_v, const double* qpivot, int h) {
    type max_lb = 0.0;
    for (int j = 0; j < h; j++) {
        type diff = fabs(idx_v[j] - qpivot[j]);
        if (diff > max_lb) {
            max_lb = diff;
        }
    }
    return max_lb;
}

// ==============================
// FIT - PARALLELIZZATO
// ==============================

void fit(params* input) {
    if(!input->silent) {
        printf("=======================================================\n");
        printf("  VERSIONE 64-bit OPENMP\n");
        #if defined(USE_ASM_APPROX) || defined(USE_ASM_EUCLIDEAN) || defined(USE_ASM_LOWER_BOUND)
            printf("  FIT: ASM version\n");
        #else
            printf("  FIT: C version\n");
        #endif
        printf("  OpenMP threads: %d\n", omp_get_max_threads());
        printf("=======================================================\n");
        fflush(stdout);
    }

    if (input->first_fit_call == false) {
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

    if(input->DS == NULL){
        printf("ERRORE: input->DS è NULL!\n");
        exit(1);
    }

    if(input->P != NULL) _mm_free(input->P);
    input->P = checked_alloc(h * sizeof(int));

    int step = N / h;
    for(int j = 0; j < h; j++){
        input->P[j] = j * step;
    }

    if(input->ds_plus != NULL) _mm_free(input->ds_plus);
    if(input->ds_minus != NULL) _mm_free(input->ds_minus);

    input->ds_plus = checked_alloc(N * D * sizeof(type));
    input->ds_minus = checked_alloc(N * D * sizeof(type));

    // PARALLELIZZAZIONE: ogni thread ha scratch buffer proprio
    #pragma omp parallel
    {
        int*  scratch_idx = (int*)malloc(D * sizeof(int));
        type* scratch_abs = (type*)_mm_malloc(D * sizeof(type), 32);
        
        if (!scratch_idx || !scratch_abs) {
            printf("ERRORE: scratch alloc (thread %d)\n", omp_get_thread_num());
            exit(1);
        }

        #pragma omp for schedule(dynamic, 100)
        for(int i = 0; i < N; i++){
            quantize_vector_scratch(&input->DS[i * D],
                                    &input->ds_plus[i * D],
                                    &input->ds_minus[i * D],
                                    x, D,
                                    scratch_idx, scratch_abs);
        }

        _mm_free(scratch_abs);
        free(scratch_idx);
    }

    if(input->index != NULL) _mm_free(input->index);
    input->index = checked_alloc(N * h * sizeof(type));

    #pragma omp parallel for schedule(dynamic, 100)
    for(int i = 0; i < N; i++){
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
        printf("FIT COMPLETATO.\n");
        fflush(stdout);
    }
}

// ==============================
// PREDICT - PARALLELIZZATO
// ==============================

typedef struct {
    int id;
    type dist;
} neighbor;

// Funzione inline per update efficiente del kNN  
static inline void update_knn_fast(neighbor* knn, int k, int id, type dist, type* worst_dist, int* worst_idx) {
    knn[*worst_idx].id = id;
    knn[*worst_idx].dist = dist;
    
    // Update worst con loop unrolling per k=8 (caso comune)
    *worst_dist = knn[0].dist; 
    *worst_idx = 0;
    
    if (k >= 8) {
        if (knn[1].dist > *worst_dist) { *worst_dist = knn[1].dist; *worst_idx = 1; }
        if (knn[2].dist > *worst_dist) { *worst_dist = knn[2].dist; *worst_idx = 2; }
        if (knn[3].dist > *worst_dist) { *worst_dist = knn[3].dist; *worst_idx = 3; }
        if (knn[4].dist > *worst_dist) { *worst_dist = knn[4].dist; *worst_idx = 4; }
        if (knn[5].dist > *worst_dist) { *worst_dist = knn[5].dist; *worst_idx = 5; }
        if (knn[6].dist > *worst_dist) { *worst_dist = knn[6].dist; *worst_idx = 6; }
        if (knn[7].dist > *worst_dist) { *worst_dist = knn[7].dist; *worst_idx = 7; }
        for (int i = 8; i < k; i++) {
            if (knn[i].dist > *worst_dist) { *worst_dist = knn[i].dist; *worst_idx = i; }
        }
    } else {
        for (int i = 1; i < k; i++) {
            if (knn[i].dist > *worst_dist) { *worst_dist = knn[i].dist; *worst_idx = i; }
        }
    }
}

void predict(params* input) {
    if(!input->silent) {
        printf("=======================================================\n");
        #if defined(USE_ASM_APPROX) || defined(USE_ASM_EUCLIDEAN) || defined(USE_ASM_LOWER_BOUND)
            printf("  PREDICT: ASM version\n");
        #else
            printf("  PREDICT: C version\n");
        #endif
        printf("  OpenMP threads: %d\n", omp_get_max_threads());
        printf("=======================================================\n");
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

    MATRIX q_plus  = (type*)checked_alloc(nq * D * sizeof(type));
    MATRIX q_minus = (type*)checked_alloc(nq * D * sizeof(type));

    // PARALLELIZZAZIONE quantizzazione query
    #pragma omp parallel
    {
        int*  scratch_idx = (int*)malloc(D * sizeof(int));
        type* scratch_abs = (type*)_mm_malloc(D * sizeof(type), 32);
        
        if (!scratch_idx || !scratch_abs) {
            printf("ERRORE: scratch alloc (thread %d)\n", omp_get_thread_num());
            exit(1);
        }

        #pragma omp for
        for(int q = 0; q < nq; q++){
            quantize_vector_scratch(&input->Q[q * D],
                                    &q_plus[q * D],
                                    &q_minus[q * D],
                                    x, D, scratch_idx, scratch_abs);
        }

        _mm_free(scratch_abs);
        free(scratch_idx);
    }

    MATRIX pivot_plus  = (type*)checked_alloc(h * D * sizeof(type));
    MATRIX pivot_minus = (type*)checked_alloc(h * D * sizeof(type));

    for(int j = 0; j < h; j++){
        int p = input->P[j];
        memcpy(&pivot_plus[j * D],  &input->ds_plus[p * D],  D * sizeof(type));
        memcpy(&pivot_minus[j * D], &input->ds_minus[p * D], D * sizeof(type));
    }

    // PARALLELIZZAZIONE K-NN: ogni query è indipendente
    #pragma omp parallel
    {
        // Thread-local structures
        neighbor* knn = (neighbor*)malloc(k * sizeof(neighbor));
        type* qpivot = (type*)malloc(h * sizeof(type));
        
        if (!knn || !qpivot) {
            printf("ERRORE: alloc knn/qpivot (thread %d)\n", omp_get_thread_num());
            exit(1);
        }

        #pragma omp for schedule(dynamic, 10)
        for(int q = 0; q < nq; q++){
            
            for(int i = 0; i < k; i++){
                knn[i].id = -1;
                knn[i].dist = DBL_MAX;
            }

            type* qplus_q  = &q_plus[q * D];
            type* qminus_q = &q_minus[q * D];

            for(int j = 0; j < h; j++){
                qpivot[j] = approx_distance_sel(
                    qplus_q, qminus_q,
                    &pivot_plus[j * D], &pivot_minus[j * D],
                    D
                );
            }

            type worst_dist = DBL_MAX;
            int  worst_idx  = 0;

            int v = 0;
            
            // Loop unrolled x8 per massime prestazioni + prefetching
            for (; v <= N - 8; v += 8) {
                // Prefetch prossimi cache lines per ridurre latency
                __builtin_prefetch(&input->index[(size_t)(v+16) * (size_t)h], 0, 3);
                __builtin_prefetch(&input->ds_plus[(size_t)(v+16) * (size_t)D], 0, 3);
                
                // Process 8 vectors unrolled manually
                for (int offset = 0; offset < 8; offset++) {
                    type* idx_v = &input->index[(size_t)(v+offset) * (size_t)h];
                    type LB = compute_lower_bound_sel(idx_v, qpivot, h);
                    if (LB < worst_dist) {
                        type* vplus_v  = &input->ds_plus[(size_t)(v+offset) * (size_t)D];
                        type* vminus_v = &input->ds_minus[(size_t)(v+offset) * (size_t)D];
                        type d = approx_distance_sel(qplus_q, qminus_q, vplus_v, vminus_v, D);
                        if (d < worst_dist) {
                            update_knn_fast(knn, k, v+offset, d, &worst_dist, &worst_idx);
                        }
                    }
                }
            }

            // Remainder con ottimizzazione
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
                    worst_dist = knn[0].dist; 
                    worst_idx = 0;
                    for (int i = 1; i < k; i++) {
                        if (knn[i].dist > worst_dist) { 
                            worst_dist = knn[i].dist; 
                            worst_idx = i; 
                        }
                    }
                }
            }

            // ========== FALLBACK SCAN: garantisce sempre k vicini ==========
            int valid_neighbors = 0;
            for(int i = 0; i < k; i++) {
                if(knn[i].id >= 0) valid_neighbors++;
            }

            if(valid_neighbors < k) {
                for(int v = 0; v < N; v++) {
                    // Salta se già presente nei knn
                    bool already_in = false;
                    for(int i = 0; i < k; i++) {
                        if(knn[i].id == v) {
                            already_in = true;
                            break;
                        }
                    }
                    if(already_in) continue;
                    
                    type* vplus_v  = &input->ds_plus[(size_t)v * (size_t)D];
                    type* vminus_v = &input->ds_minus[(size_t)v * (size_t)D];
                    type d_approx = approx_distance_sel(qplus_q, qminus_q, vplus_v, vminus_v, D);
                    
                    // Inserisci se migliore del peggiore O se ci sono slot vuoti
                    if(d_approx < worst_dist || knn[worst_idx].id < 0) {
                        knn[worst_idx].id = v;
                        knn[worst_idx].dist = d_approx;
                        
                        // Aggiorna worst
                        worst_dist = knn[0].dist; 
                        worst_idx = 0;
                        for (int i = 1; i < k; i++) {
                            if (knn[i].id < 0 || knn[i].dist > worst_dist) { 
                                worst_dist = knn[i].dist; 
                                worst_idx = i; 
                            }
                        }
                        
                        // Conta valid neighbors e exit se completo
                        valid_neighbors = 0;
                        for(int i = 0; i < k; i++) {
                            if(knn[i].id >= 0) valid_neighbors++;
                        }
                        if(valid_neighbors >= k) break;
                    }
                }
            }
            // ========== FINE FALLBACK SCAN ==========

            // Refinement
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

            // Salvataggio (thread-safe: ogni query scrive in zone diverse)
            for(int i = 0; i < k; i++){
                input->id_nn[q * k + i]   = knn[i].id;
                input->dist_nn[q * k + i] = knn[i].dist;
            }
        }

        free(qpivot);
        free(knn);
        
    } // fine parallel

    _mm_free(q_plus);
    _mm_free(q_minus);
    _mm_free(pivot_plus);
    _mm_free(pivot_minus);

    if(!input->silent) {
        printf("PREDICT COMPLETATO.\n");
        fflush(stdout);
    }
}