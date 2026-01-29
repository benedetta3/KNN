#ifndef QUANTPIVOT_COMMON64
#define QUANTPIVOT_COMMON64
#include <stdbool.h>

#define	type	double
#define	align	32

#define	MATRIX		type*
#define	VECTOR		type*

typedef struct{
	// Variabili
	MATRIX DS; 					// dataset
	int* P;						// vettore contenente gli indici dei pivot
	MATRIX index;				// indice
	MATRIX Q;					// query
	int* id_nn;					// per ogni query point gli ID dei K-NN
	MATRIX dist_nn;				// per ogni query point le distanze dai K-NN
	int h;						// numero di pivot
	int k;						// numero di vicini
	int x;						// parametro x per la quantizzazione
	int N;						// numero di righe del dataset
	int D;						// numero di colonne/feature del dataset
	int nq;						// numero delle query
	int silent;					// modalità silenziosa
	MATRIX ds_plus;   // vettori quantizzati positivi del dataset
    MATRIX ds_minus;  // vettori quantizzati negativi del dataset
	bool first_fit_call;
} params;

#endif