#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "common.h"

// ============================================================================
// VERSIONE SELEZIONABILE: C PURO o ASM (32-bit float)
// Se USE_ASM_APPROX/EUCLIDEAN/LOWER_BOUND è definito, usa ASM, altrimenti C puro
// Tutte le funzioni hanno la stessa firma: type (...)
// ============================================================================

#ifdef USE_ASM_APPROX
extern type approx_distance_asm(const type* vplus, const type* vminus,
                                const type* wplus, const type* wminus,
                                int D);
#endif
#ifdef USE_ASM_EUCLIDEAN
extern type euclidean_distance_asm(const type* v, const type* w, int D);
#endif
#ifdef USE_ASM_LOWER_BOUND
extern type compute_lower_bound_asm(const type* idx_v, const type* qpivot, int h);
#endif

// ============================================================================
// DISTANZA APPROSSIMATA - C PURO
// ============================================================================
static inline type approx_distance(const type* vplus, const type* vminus,
                                  const type* wplus, const type* wminus,
                                  int D)
{
    type dot_pp = 0.0f;
    type dot_mm = 0.0f;
    type dot_pm = 0.0f;
    type dot_mp = 0.0f;
    for (int i = 0; i < D; i++) {
        dot_pp += vplus[i] * wplus[i];
        dot_mm += vminus[i] * wminus[i];
        dot_pm += vplus[i] * wminus[i];
        dot_mp += vminus[i] * wplus[i];
    }
    return dot_pp + dot_mm - dot_pm - dot_mp;
}

// ============================================================================
// DISTANZA EUCLIDEA - C PURO
// ============================================================================
static inline type euclidean_distance(const type* v, const type* w, int D) {
    type sum_sq = 0.0f;
    for (int i = 0; i < D; i++) {
        type diff = v[i] - w[i];
        sum_sq += diff * diff;
    }
    return sqrtf(sum_sq);
}

// ============================================================================
// CALCOLO LOWER BOUND - C PURO
// ============================================================================
static inline type compute_lower_bound(const type* idx_v, const type* qpivot, int h) {
    type max_lb = 0.0f;
    for (int j = 0; j < h; j++) {
        type diff = fabsf(idx_v[j] - qpivot[j]);
        if (diff > max_lb) {
            max_lb = diff;
        }
    }
    return max_lb;
}

// ============================================================================
// WRAPPER SELETTIVO: usa ASM se macro attiva, altrimenti C puro
// ============================================================================
static inline type approx_distance_sel(const type* vplus, const type* vminus,
                                       const type* wplus, const type* wminus,
                                       int D) {
#ifdef USE_ASM_APPROX
    return approx_distance_asm(vplus, vminus, wplus, wminus, D);
#else
    return approx_distance(vplus, vminus, wplus, wminus, D);
#endif
}

static inline type euclidean_distance_sel(const type* v, const type* w, int D) {
#ifdef USE_ASM_EUCLIDEAN
    return euclidean_distance_asm(v, w, D);
#else
    return euclidean_distance(v, w, D);
#endif
}

static inline type compute_lower_bound_sel(const type* idx_v, const type* qpivot, int h) {
#ifdef USE_ASM_LOWER_BOUND
    return compute_lower_bound_asm(idx_v, qpivot, h);
#else
    return compute_lower_bound(idx_v, qpivot, h);
#endif
}

// Allocazione memoria con controllo errori
static inline void* checked_alloc(size_t size) {
    void* p = malloc(size);
    if (!p) {
        printf("ERRORE: impossibile allocare %lu bytes\n", (unsigned long)size);
        fflush(stdout);
        exit(1);
    }
    return p;
}

// Funzione di swap per quickselect
static inline void swap_pair(type* vals, int* indices, int i, int j) {
    type tmp_val = vals[i];
    vals[i] = vals[j];
    vals[j] = tmp_val;
    
    int tmp_idx = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp_idx;
}

// Partizione per quickselect
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

// Quickselect per trovare i top-x elementi (O(D))
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

// ============================================================================
// QUANTIZZAZIONE VETTORE - C PURO
// Algoritmo: Quickselect O(D) + Insertion Sort O(x^2) solo sui primi x
// ============================================================================
static inline void quantize_vector(const type* v,
                                   type* vplus,
                                   type* vminus,
                                   int x,
                                   int D,
                                   int* scratch_indices,
                                   type* scratch_abs_vals)
{
    // Reset vettori output
    memset(vplus,  0, (size_t)D * sizeof(type));
    memset(vminus, 0, (size_t)D * sizeof(type));

    if (x <= 0) return;
    if (x > D) x = D;

    // Calcolo valori assoluti - C PURO
    for (int i = 0; i < D; i++) {
        scratch_indices[i] = i;
        scratch_abs_vals[i] = fabsf(v[i]);
    }

    // Quickselect per partizionare i top-x - O(D)
    quickselect_top_x(scratch_abs_vals, scratch_indices, 0, D - 1, x);
    
    // Insertion Sort SOLO sui primi x elementi - O(x^2) ma x << D
    for (int j = 1; j < x; j++) {
        type key_val = scratch_abs_vals[j];
        int key_idx = scratch_indices[j];
        int k = j - 1;
        
        while (k >= 0 && scratch_abs_vals[k] < key_val) {
            scratch_abs_vals[k + 1] = scratch_abs_vals[k];
            scratch_indices[k + 1] = scratch_indices[k];
            k--;
        }
        scratch_abs_vals[k + 1] = key_val;
        scratch_indices[k + 1] = key_idx;
    }

    // Imposta i bit nei vettori quantizzati
    for (int count = 0; count < x; count++) {
        int idx = scratch_indices[count];
        if (v[idx] >= 0.0f) {
            vplus[idx] = 1.0f;
        } else {
            vminus[idx] = 1.0f;
        }
    }
}


// ============================================================================
// FIT: Costruzione indice pivot-based
// ============================================================================
void fit(params* input) {


    // Inizializzazione prima chiamata
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



    if (input->DS == NULL) {
        printf("ERRORE: input->DS è NULL!\n");
        exit(1);
    }

    // ========================================================================
    // 1. SELEZIONE PIVOT
    // ========================================================================
    if (input->P != NULL) {
        free(input->P);
    }

    input->P = (int*)checked_alloc((size_t)h * sizeof(int));
    
    // Selezione pivot: i = floor(n/h) * j, con j = 0..h-1
    int step = (h > 0) ? (N / h) : 0;
    for (int j = 0; j < h; j++) {
        input->P[j] = j * step;
    }



    // ========================================================================
    // 2. QUANTIZZAZIONE DATASET
    // ========================================================================
    if (input->ds_plus != NULL) {
        free(input->ds_plus);
    }
    if (input->ds_minus != NULL) {
        free(input->ds_minus);
    }

    input->ds_plus  = (type*)checked_alloc((size_t)N * (size_t)D * sizeof(type));
    input->ds_minus = (type*)checked_alloc((size_t)N * (size_t)D * sizeof(type));

    // Allocazione buffer scratch riutilizzabili
    int*  scratch_idx = (int*)checked_alloc((size_t)D * sizeof(int));
    type* scratch_abs = (type*)checked_alloc((size_t)D * sizeof(type));



    for (int i = 0; i < N; i++) {
        quantize_vector(&input->DS[(size_t)i * (size_t)D],
                       &input->ds_plus[(size_t)i * (size_t)D],
                       &input->ds_minus[(size_t)i * (size_t)D],
                       x, D,
                       scratch_idx, scratch_abs);
    }

    free(scratch_idx);
    free(scratch_abs);



    // ========================================================================
    // 3. COSTRUZIONE INDICE
    // ========================================================================
    if (input->index != NULL) {
        free(input->index);
    }

    input->index = (type*)checked_alloc((size_t)N * (size_t)h * sizeof(type));



    for (int i = 0; i < N; i++) {
        const type* vplus_i  = &input->ds_plus[(size_t)i * (size_t)D];
        const type* vminus_i = &input->ds_minus[(size_t)i * (size_t)D];
        type* index_row = &input->index[(size_t)i * (size_t)h];
        for (int j = 0; j < h; j++) {
            int pivot_idx = input->P[j];
            const type* pplus  = &input->ds_plus[(size_t)pivot_idx * (size_t)D];
            const type* pminus = &input->ds_minus[(size_t)pivot_idx * (size_t)D];
            index_row[j] = approx_distance_sel(vplus_i, vminus_i, pplus, pminus, D);
        }
    }


}

// ============================================================================
// PREDICT: Ricerca K-NN con pruning basato su pivot
// ============================================================================

typedef struct {
    int id;
    type dist;
} neighbor;

void predict(params* input) {
    if (!input->silent) {
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

    if (input->ds_plus == NULL || input->ds_minus == NULL) {
        printf("ERRORE: predict() chiamata prima di fit()!\n");
        exit(1);
    }
    if (input->Q == NULL) {
        printf("ERRORE: input->Q è NULL!\n");
        exit(1);
    }

    // ========================================================================
    // 1. QUANTIZZAZIONE BATCH DELLE QUERY
    // ========================================================================
    type* all_q_plus  = (type*)checked_alloc((size_t)nq * (size_t)D * sizeof(type));
    type* all_q_minus = (type*)checked_alloc((size_t)nq * (size_t)D * sizeof(type));

    int*  scratch_idx = (int*)checked_alloc((size_t)D * sizeof(int));
    type* scratch_abs = (type*)checked_alloc((size_t)D * sizeof(type));



    for (int q = 0; q < nq; q++) {
        quantize_vector(&input->Q[(size_t)q * (size_t)D],
                       &all_q_plus[(size_t)q * (size_t)D],
                       &all_q_minus[(size_t)q * (size_t)D],
                       x, D,
                       scratch_idx, scratch_abs);
    }

    free(scratch_idx);
    free(scratch_abs);



    // ========================================================================
    // 2. PREPARAZIONE PIVOT CONTIGUI (ottimizzazione cache)
    // ========================================================================
    type* pivot_plus_contig  = (type*)checked_alloc((size_t)h * (size_t)D * sizeof(type));
    type* pivot_minus_contig = (type*)checked_alloc((size_t)h * (size_t)D * sizeof(type));

    for (int j = 0; j < h; j++) {
        int p = input->P[j];
        memcpy(&pivot_plus_contig[(size_t)j * (size_t)D],
               &input->ds_plus[(size_t)p * (size_t)D],
               (size_t)D * sizeof(type));
        memcpy(&pivot_minus_contig[(size_t)j * (size_t)D],
               &input->ds_minus[(size_t)p * (size_t)D],
               (size_t)D * sizeof(type));
    }



    // ========================================================================
    // 3. RICERCA K-NN PER OGNI QUERY
    // ========================================================================
    neighbor* knn = (neighbor*)checked_alloc((size_t)k * sizeof(neighbor));
    type* qpivot  = (type*)checked_alloc((size_t)h * sizeof(type));

    for (int q = 0; q < nq; q++) {
        const type* q_plus  = &all_q_plus[(size_t)q * (size_t)D];
        const type* q_minus = &all_q_minus[(size_t)q * (size_t)D];
        for (int i = 0; i < k; i++) {
            knn[i].id = -1;
            knn[i].dist = FLT_MAX;
        }
        for (int j = 0; j < h; j++) {
            const type* pplus  = &pivot_plus_contig[(size_t)j * (size_t)D];
            const type* pminus = &pivot_minus_contig[(size_t)j * (size_t)D];
            qpivot[j] = approx_distance_sel(q_plus, q_minus, pplus, pminus, D);
        }
        type worst_dist = FLT_MAX;
        int  worst_idx  = 0;
        for (int v = 0; v < N; v++) {
            const type* idx_v = &input->index[(size_t)v * (size_t)h];
            type LB = compute_lower_bound_sel(idx_v, qpivot, h);
            if (LB >= worst_dist) continue;
            const type* vplus_v  = &input->ds_plus[(size_t)v * (size_t)D];
            const type* vminus_v = &input->ds_minus[(size_t)v * (size_t)D];
            type d_approx = approx_distance_sel(q_plus, q_minus, vplus_v, vminus_v, D);
            if (d_approx < worst_dist) {
                knn[worst_idx].id = v;
                knn[worst_idx].dist = d_approx;
                worst_dist = knn[0].dist; worst_idx = 0;
                for (int i = 1; i < k; i++) {
                    if (knn[i].dist > worst_dist) { worst_dist = knn[i].dist; worst_idx = i; }
                }
            }
        }
        const type* query_base = &input->Q[(size_t)q * (size_t)D];
        for (int i = 0; i < k; i++) {
            if (knn[i].id >= 0) {
                knn[i].dist = euclidean_distance_sel(query_base,
                                                    &input->DS[(size_t)knn[i].id * (size_t)D],
                                                    D);
            }
        }
        for (int i = 0; i < k; i++) {
            input->id_nn[(size_t)q * (size_t)k + (size_t)i]   = knn[i].id;
            input->dist_nn[(size_t)q * (size_t)k + (size_t)i] = knn[i].dist;
        }
    }

    // Cleanup
    free(qpivot);
    free(knn);
    free(pivot_plus_contig);
    free(pivot_minus_contig);
    free(all_q_plus);
    free(all_q_minus);


}