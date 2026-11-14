// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* The lifecycle of the two types and the module init: the only hand-written C of the extension. */
#include "osprey.h"

PyTypeObject* osprey_node_type = NULL;

static PyObject* Ctx_new(PyTypeObject* t, PyObject* a, PyObject* k) {
  (void)a; (void)k;
  Ctx* s = (Ctx*)PyType_GenericAlloc(t, 0);
  if (!s) return NULL;
  s->c = rx_ctx_new();
  if (!s->c) { Py_DECREF((PyObject*)s); PyErr_SetString(PyExc_RuntimeError, "cannot create a context"); return NULL; }
  return (PyObject*)s;
}

static void Ctx_dealloc(PyObject* o) {
  PyTypeObject* t = Py_TYPE(o);
  rx_ctx_free(((Ctx*)o)->c);
  ((freefunc)PyType_GetSlot(t, Py_tp_free))(o);
  Py_DECREF((PyObject*)t);
}

static void Node_dealloc(PyObject* o) {
  PyTypeObject* t = Py_TYPE(o);
  Py_DECREF((PyObject*)((Node*)o)->ctx);
  ((freefunc)PyType_GetSlot(t, Py_tp_free))(o);
  Py_DECREF((PyObject*)t);
}

/* no Py_TPFLAGS_BASETYPE: not subclassable; no dict offset: no attribute can be added; IMMUTABLETYPE: the type cannot be patched */
static PyType_Slot ctx_slots[] = {{Py_tp_new, (void*)Ctx_new}, {Py_tp_dealloc, (void*)Ctx_dealloc}, {Py_tp_methods, osprey_ctx_methods}, {0, 0}};
static PyType_Spec ctx_spec = {"_osprey.Ctx", sizeof(Ctx), 0, Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE, ctx_slots};
static PyType_Slot node_slots[] = {{Py_tp_dealloc, (void*)Node_dealloc}, {Py_tp_methods, osprey_node_methods}, {Py_tp_getset, osprey_node_getset}, {0, 0}};
static PyType_Spec node_spec = {"_osprey.Node", sizeof(Node), 0, Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DISALLOW_INSTANTIATION, node_slots};
static struct PyModuleDef osprey_module = {PyModuleDef_HEAD_INIT, "_osprey", NULL, -1, NULL, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__osprey(void) {
  PyObject* m = PyModule_Create(&osprey_module);
  if (!m) return NULL;
  PyObject* ct = PyType_FromSpec(&ctx_spec);
  PyObject* nt = PyType_FromSpec(&node_spec);
  if (!ct || !nt) { Py_XDECREF(ct); Py_XDECREF(nt); Py_DECREF(m); return NULL; }
  osprey_node_type = (PyTypeObject*)nt;
  Py_INCREF(nt);
  if (PyModule_AddObject(m, "Ctx", ct) < 0 || PyModule_AddObject(m, "Node", nt) < 0 || PyModule_AddIntConstant(m, "ABI_VERSION", (long)rx_abi_version()) < 0) {
    Py_DECREF(m);
    return NULL;
  }
  return m;
}
