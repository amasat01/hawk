/* Shared declarations of the extension: types and tables, no function bodies. */
#ifndef OSPREY_H
#define OSPREY_H
#define PY_SSIZE_T_CLEAN
#define Py_LIMITED_API 0x030A0000
#include "osprey_host.h"
#include "raptor_ir.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD rx_ctx* c;
} Ctx;
typedef struct {
    PyObject_HEAD Ctx* ctx;
    rx_node h;
} Node;

extern PyTypeObject* osprey_node_type;
extern PyMethodDef osprey_ctx_methods[];
extern PyMethodDef osprey_node_methods[];
extern PyGetSetDef osprey_node_getset[];
#endif
