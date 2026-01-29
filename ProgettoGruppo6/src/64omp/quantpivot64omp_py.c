#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>
#include <immintrin.h>

#include "common.h"

#include "quantpivot64omp.c"

// Struttura per l'oggetto QuantPivot
typedef struct {
	PyObject_HEAD
	params* input;
	PyArrayObject* DS_array;
	PyArrayObject* Q_array;
} QuantPivot64OMPObject;

static void mm_free_destructor(PyObject* capsule) {
    void* ptr = PyCapsule_GetPointer(capsule, NULL);
    if (ptr != NULL) {
        _mm_free(ptr);
    }
}

// Deallocazione
static void QuantPivot64OMP_dealloc(QuantPivot64OMPObject *self) {
	if (self->input->P != NULL)
		_mm_free(self->input->P);
	if (self->input->index != NULL)
		_mm_free(self->input->index);

    if (self->input->ds_plus != NULL) _mm_free(self->input->ds_plus);
    if (self->input->ds_minus != NULL) _mm_free(self->input->ds_minus);

	Py_XDECREF(self->DS_array);
	Py_XDECREF(self->Q_array);

	free(self->input);

	Py_TYPE(self)->tp_free((PyObject *)self);
}

static int QuantPivot64OMP_init(QuantPivot64OMPObject *self, PyObject *args, PyObject *kwargs) {
    self->DS_array = NULL;
    self->Q_array = NULL;

    self->input = (params*)calloc(1, sizeof(params));
    if (!self->input) {
        PyErr_NoMemory();
        return -1;
    }

    self->input->DS = NULL;
    self->input->P = NULL;
    self->input->h = -1;
    self->input->k = -1;
    self->input->x = -1;
    self->input->N = -1;
    self->input->D = -1;
    self->input->index = NULL;
    self->input->Q = NULL;
    self->input->nq = -1;
    self->input->id_nn = NULL;
    self->input->dist_nn = NULL;
    self->input->silent = 0;

    self->input->ds_plus = NULL;
    self->input->ds_minus = NULL;
    self->input->first_fit_call = false;

    return 0;
}

// Metodo fit
static PyObject* QuantPivot64OMP_fit(QuantPivot64OMPObject *self, PyObject *args, PyObject *kwargs) {
	PyArrayObject *ds_array;

	int h, x, silent = 1;

	static char *kwlist[] = {"dataset", "n_pivots", "quant_level", "silent", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!ii|i", kwlist,
									&PyArray_Type, &ds_array,
									&h, &x, &silent)) {
		return NULL;
	}

	if (PyArray_NDIM(ds_array) != 2) {
		PyErr_SetString(PyExc_ValueError, "Data must be a 2D array");
		return NULL;
	}

	if (PyArray_TYPE(ds_array) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Data must be float64");
		return NULL;
	}

	type* dataset = (type*)(PyArrayObject*)PyArray_DATA(ds_array);

	uintptr_t addr = (uintptr_t)dataset;
	int is_aligned = (addr % align == 0);

	if(!is_aligned){
		PyErr_SetString(PyExc_ValueError, "Input array (DS) not aligned");
		return NULL;
	}

	self->input->N = (int)PyArray_DIM(ds_array, 0);
	self->input->D = (int)PyArray_DIM(ds_array, 1);

	if (h <= 0 || h > self->input->N) {
		PyErr_SetString(PyExc_ValueError, "n_pivots (h) must be in [1..N]");
		return NULL;
	}
	if (x <= 0 || x > self->input->D) {
		PyErr_SetString(PyExc_ValueError, "quant_level (x) must be in [1..D]");
		return NULL;
	}

	self->input->h = h;
	self->input->x = x;
	self->input->silent = silent;

	Py_INCREF(ds_array);
	Py_XDECREF(self->DS_array);
	self->DS_array = ds_array;

	self->input->DS = dataset;

	// Release GIL per permettere parallelizzazione OpenMP
	Py_BEGIN_ALLOW_THREADS
	fit(self->input);
	Py_END_ALLOW_THREADS

	Py_INCREF(self);
	return (PyObject *)self;
}

// Metodo predict
static PyObject* QuantPivot64OMP_predict(QuantPivot64OMPObject *self, PyObject *args, PyObject *kwargs) {
	PyArrayObject* query_array;
	int k, silent = 0;

	static char* kwlist[] = {"query", "k", "silent", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!i|i", kwlist,
									&PyArray_Type, &query_array,
									&k, &silent))
		return NULL;

	if (self->input->index == NULL) {
		PyErr_SetString(PyExc_RuntimeError,
					"Model not fitted, call fit() before predict()");
		return NULL;
	}

	if (PyArray_NDIM(query_array) != 2) {
		PyErr_SetString(PyExc_ValueError, "Data must be a 2D array");
		return NULL;
	}

	if (PyArray_TYPE(query_array) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Data must be float64");
		return NULL;
	}

	type* query = (type*)(PyArrayObject*)PyArray_DATA(query_array);
	uintptr_t addr = (uintptr_t)query;
	int is_aligned = (addr % align == 0);

	if(!is_aligned){
		PyErr_SetString(PyExc_ValueError, "Query array (Q) not aligned");
		return NULL;
	}

	self->input->Q = query;

	self->input->nq = (int)PyArray_DIM(query_array, 0);

	int qD = (int)PyArray_DIM(query_array, 1);
	if (qD != self->input->D) {
		PyErr_SetString(PyExc_ValueError, "Query dimensionality must match dataset D");
		return NULL;
	}

	if (k <= 0 || k > self->input->N) {
		PyErr_SetString(PyExc_ValueError, "k must be in [1..N]");
		return NULL;
	}

	self->input->k = k;
	self->input->silent = silent;

	self->input->id_nn = (int*) _mm_malloc(self->input->nq * self->input->k * sizeof(int), align);
	self->input->dist_nn = (type*) _mm_malloc(self->input->nq * self->input->k * sizeof(type), align);

	// Release GIL per permettere parallelizzazione OpenMP
	Py_BEGIN_ALLOW_THREADS
	predict(self->input);
	Py_END_ALLOW_THREADS

	npy_intp dims[2] = {self->input->nq, self->input->k};

	PyArrayObject* id_nn_array = (PyArrayObject*)PyArray_SimpleNewFromData(
		2, dims, NPY_INT32, self->input->id_nn
	);
	PyObject* capsule_id = PyCapsule_New(self->input->id_nn, NULL, mm_free_destructor);
	PyArray_SetBaseObject(id_nn_array, capsule_id);

	PyArrayObject* dist_nn_array = (PyArrayObject*)PyArray_SimpleNewFromData(
		2, dims, NPY_FLOAT64, self->input->dist_nn
	);
	PyObject* capsule_dist = PyCapsule_New(self->input->dist_nn, NULL, mm_free_destructor);
	PyArray_SetBaseObject(dist_nn_array, capsule_dist);

	PyObject* result = PyTuple_Pack(2,
									(PyObject*)id_nn_array,
									(PyObject*)dist_nn_array);

	Py_DECREF(id_nn_array);
	Py_DECREF(dist_nn_array);

    return result;
}

// Tabella dei metodi
static PyMethodDef QuantPivot64OMP_methods[] = {
	{
		"fit",
		(PyCFunction)QuantPivot64OMP_fit,
		METH_VARARGS | METH_KEYWORDS,
		"Build the index using data (OpenMP parallelized)\n\n"
		"Parameters:\n"
		"  data: numpy array of shape (N, D), dtype=float64\n"
		"  n_pivots: number of pivots\n"
		"  x: quantization level\n"
		"  s: silent (default=False)\n"
		"\n"
		"Returns:\n"
		"  self"
	},
	{
		"predict",
		(PyCFunction)QuantPivot64OMP_predict,
		METH_VARARGS | METH_KEYWORDS,
		"Query the index (OpenMP parallelized)\n\n"
		"Parameters:\n"
		"  query: numpy array of shape (nq, D), dtype=float64\n"
		"  k: number of neighbors\n"
		"  s: silent (default=False)\n"
		"\n"
		"Returns:\n"
		"  tuple (indices, distances)"
	},
	{NULL, NULL, 0, NULL}
};

// Definizione del tipo Python
static PyTypeObject QuantPivot64OMPType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name = "gruppo6.quantpivot64omp.QuantPivot",
	.tp_doc = "QuantPivot 64-bit indexing and querying with AVX + OpenMP",
	.tp_basicsize = sizeof(QuantPivot64OMPObject),
	.tp_itemsize = 0,
	.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
	.tp_new = PyType_GenericNew,
	.tp_init = (initproc)QuantPivot64OMP_init,
	.tp_dealloc = (destructor)QuantPivot64OMP_dealloc,
	.tp_methods = QuantPivot64OMP_methods,
};


static struct PyModuleDef quantpivot64omp_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_quantpivot64omp",
    .m_doc = "Quantized Pivot Indexing and Querying (64bit AVX + OpenMP)",
    .m_size = -1,
};

// Inizializzazione del modulo
PyMODINIT_FUNC PyInit__quantpivot64omp(void) {
	PyObject *m;

	if (PyType_Ready(&QuantPivot64OMPType) < 0)
		return NULL;

	m = PyModule_Create(&quantpivot64omp_module);
	if (m == NULL)
		return NULL;

	Py_INCREF(&QuantPivot64OMPType);
	if (PyModule_AddObject(m, "QuantPivot", (PyObject *)&QuantPivot64OMPType) < 0) {
		Py_DECREF(&QuantPivot64OMPType);
		Py_DECREF(m);
		return NULL;
	}

	import_array();

	return m;
}