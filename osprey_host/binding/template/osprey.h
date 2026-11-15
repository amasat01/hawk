/* Shared declarations of the extension: types and tables, no function bodies. */
#ifndef OSPREY_H
#define OSPREY_H
#define PY_SSIZE_T_CLEAN
/* The floor is Python 3.9: the stable ABI of that version, nothing newer. */
#define Py_LIMITED_API 0x03090000
#include "osprey_host.h"
#include "raptor_ir.h"
#include <Python.h>

/* The two type flags that make a type unpatchable and uninstantiable exist from Python 3.10. Their
 * bits are free in 3.9, so setting them there does nothing; from 3.10 on they take effect, with one
 * build. */
#ifndef Py_TPFLAGS_IMMUTABLETYPE
#define Py_TPFLAGS_IMMUTABLETYPE (1UL << 8)
#endif
#ifndef Py_TPFLAGS_DISALLOW_INSTANTIATION
#define Py_TPFLAGS_DISALLOW_INSTANTIATION (1UL << 7)
#endif

typedef struct {
    PyObject_HEAD rx_ctx* c;
} Ctx;
typedef struct {
    PyObject_HEAD Ctx* ctx;
    rx_node h;
} Node;

extern PyTypeObject* osprey_node_type;
/* Makes the exception classes of the three kinds of refusal and adds them to the module (defined in
 * the generated code). */
int osprey_add_errors(PyObject* module);
extern PyMethodDef osprey_ctx_methods[];
extern PyMethodDef osprey_node_methods[];
extern PyGetSetDef osprey_node_getset[];
#endif
