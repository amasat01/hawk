/* The lifecycle of the two types and the module init: the only hand-written C of the extension. */
#include "module.h"

PyTypeObject* osprey_node_type = NULL;

/* Makes a Ctx: a new IR context owned by the Python object. */
static PyObject* Ctx_new(PyTypeObject* t, PyObject* a, PyObject* k)
{
    (void)a;
    (void)k;
    Ctx* s = (Ctx*)PyType_GenericAlloc(t, 0);
    if (!s)
        return NULL;
    s->c = rx_ctx_new();
    if (!s->c) {
        Py_DECREF((PyObject*)s);
        PyErr_SetString(PyExc_RuntimeError, "cannot create a context");
        return NULL;
    }
    return (PyObject*)s;
}

/* Frees the IR context with its Ctx object. */
static void Ctx_dealloc(PyObject* o)
{
    PyTypeObject* t = Py_TYPE(o);
    rx_ctx_free(((Ctx*)o)->c);
    ((freefunc)PyType_GetSlot(t, Py_tp_free))(o);
    Py_DECREF((PyObject*)t);
}

/* Releases the Ctx a Node proxy holds. */
static void Node_dealloc(PyObject* o)
{
    PyTypeObject* t = Py_TYPE(o);
    Py_DECREF((PyObject*)((Node*)o)->ctx);
    ((freefunc)PyType_GetSlot(t, Py_tp_free))(o);
    Py_DECREF((PyObject*)t);
}

/* A Node is made by the door only: before Python 3.10 the flag that forbids instantiation does not
 * exist, and the type would inherit object's constructor. */
static PyObject* Node_new(PyTypeObject* t, PyObject* a, PyObject* k)
{
    (void)t;
    (void)a;
    (void)k;
    PyErr_SetString(PyExc_TypeError, "cannot create '_osprey.Node' instances");
    return NULL;
}

/* Two nodes are equal when they are the same node of the same context: a front end keys its tables
 * by node. */
static PyObject* Node_richcompare(PyObject* a, PyObject* b, int op)
{
    if ((op != Py_EQ && op != Py_NE) || !PyObject_TypeCheck(a, osprey_node_type)
        || !PyObject_TypeCheck(b, osprey_node_type))
        Py_RETURN_NOTIMPLEMENTED;
    int same = ((Node*)a)->ctx == ((Node*)b)->ctx && ((Node*)a)->h == ((Node*)b)->h;
    return PyBool_FromLong(op == Py_EQ ? same : !same);
}

/* The hash of a node: its handle and its context together, as the equality sees them. */
static Py_hash_t Node_hash(PyObject* o)
{
    Py_hash_t h
        = ((Py_hash_t)((Node*)o)->h * 1000003) ^ (Py_hash_t)((uintptr_t)((Node*)o)->ctx >> 4);
    return h == -1 ? -2 : h;
}

/* `node.ctx`: the context the node belongs to. The node owns a strong reference to it (taken where
 * the node is made, released in Node_dealloc), so the context outlives every node that reads it. */
static PyMemberDef node_members[] = { { "ctx", Py_T_OBJECT_EX, offsetof(Node, ctx), Py_READONLY,
                                          "the context this node belongs to" },
    { NULL, 0, 0, 0, NULL } };

/* no Py_TPFLAGS_BASETYPE: not subclassable; no dict offset: no attribute can be added;
 * IMMUTABLETYPE: the type cannot be patched */
static PyType_Slot ctx_slots[] = { { Py_tp_new, (void*)Ctx_new },
    { Py_tp_dealloc, (void*)Ctx_dealloc }, { Py_tp_methods, osprey_ctx_methods }, { 0, 0 } };
static PyType_Spec ctx_spec
    = { "_osprey.Ctx", sizeof(Ctx), 0, Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE, ctx_slots };
static PyType_Slot node_slots[] = { { Py_tp_new, (void*)Node_new },
    { Py_tp_dealloc, (void*)Node_dealloc }, { Py_tp_richcompare, (void*)Node_richcompare },
    { Py_tp_hash, (void*)Node_hash }, { Py_tp_methods, osprey_node_methods },
    { Py_tp_getset, osprey_node_getset }, { Py_tp_members, node_members }, { 0, 0 } };
static PyType_Spec node_spec    = { "_osprey.Node", sizeof(Node), 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DISALLOW_INSTANTIATION, node_slots };
static struct PyModuleDef osprey_module
    = { PyModuleDef_HEAD_INIT, "_osprey", NULL, -1, NULL, NULL, NULL, NULL, NULL };

/* The module: the Ctx and Node types and the abi version of the door. */
PyMODINIT_FUNC PyInit__osprey(void)
{
    PyObject* m = PyModule_Create(&osprey_module);
    if (!m)
        return NULL;
    PyObject* ct = PyType_FromSpec(&ctx_spec);
    PyObject* nt = PyType_FromSpec(&node_spec);
    if (!ct || !nt) {
        Py_XDECREF(ct);
        Py_XDECREF(nt);
        Py_DECREF(m);
        return NULL;
    }
    osprey_node_type = (PyTypeObject*)nt;
    Py_INCREF(nt);
    if (PyModule_AddObject(m, "Ctx", ct) < 0 || PyModule_AddObject(m, "Node", nt) < 0
        || PyModule_AddIntConstant(m, "ABI_VERSION", (long)rx_abi_version()) < 0
        || PyModule_AddIntConstant(m, "CONTRACT_VERSION", (long)OB_CONTRACT) < 0
        || osprey_add_errors(m) < 0 || osprey_add_door(m) < 0) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
