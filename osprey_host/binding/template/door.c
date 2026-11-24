/* The runtime's C door (eagle-door/1, eg_door.h) as Python: one Door object per context and one
 * method per door function, named as the function without its eg_ prefix.
 *
 * The binding holds no policy. It converts arguments, calls the function, turns a status and the
 * door's reason text into an exception and keeps a handle's lifetime as the header states it:
 * the caller frees every handle (a context frees what is left), a plane's DLPack tensor belongs
 * to the door once a wrap succeeded, and a forked child may only free the context.
 *
 * A Door is made from an opened backend library (the handle backend_open answers): the door's
 * functions are looked up by name in that library, so this extension links no runtime, and a
 * library that states no runtime (it exports no eg_ function) refuses with NotWritten. The backend
 * must stay open while its Doors and their handles live. The interpreter lock stays held during
 * every call: the deleter of a DLPack tensor, which the door runs on the calling thread, may need
 * it, and so does a host node's Python callable, which the door also runs on the calling thread. */
#include "module.h"

#include "eagle/door/eg_door.h"
#include "eagle/door/eg_unit_meta.h"
#include "plugin/dlpack.h"
#include <string.h>

/* A function taking keywords, as the method table wants it. */
#define KEYWORD_CALL(f) ((PyCFunction)(void (*)(void))(f))

/* Every function of eg_door.h, once: the check of the generator counts them against the header. */
#define DOOR_CALLS(X)                                                                              \
    X(eg_ctx_new)                                                                                  \
    X(eg_ctx_free)                                                                                 \
    X(eg_error)                                                                                    \
    X(eg_ctx_info)                                                                                 \
    X(eg_host_threads)                                                                             \
    X(eg_unit_load)                                                                                \
    X(eg_unit_free)                                                                                \
    X(eg_plane_wrap)                                                                               \
    X(eg_plane_free)                                                                               \
    X(eg_plan_new)                                                                                 \
    X(eg_plan_free)                                                                                \
    X(eg_bind)                                                                                     \
    X(eg_clear)                                                                                    \
    X(eg_set_count)                                                                                \
    X(eg_launch)                                                                                   \
    X(eg_runner_new)                                                                               \
    X(eg_runner_free)                                                                              \
    X(eg_run)                                                                                      \
    X(eg_run_events)                                                                               \
    X(eg_reverse)                                                                                  \
    X(eg_reduce)                                                                                   \
    X(eg_scan)                                                                                     \
    X(eg_compact)                                                                                  \
    X(eg_order_by_key)                                                                             \
    X(eg_reorder)                                                                                  \
    X(eg_copy)                                                                                     \
    X(eg_record_begin)                                                                             \
    X(eg_record_abort)                                                                             \
    X(eg_record_cond_begin)                                                                        \
    X(eg_record_cond_arm)                                                                          \
    X(eg_record_cond_end)                                                                          \
    X(eg_record_fork_begin)                                                                        \
    X(eg_record_fork_branch)                                                                       \
    X(eg_record_fork_end)                                                                          \
    X(eg_record_wait)                                                                              \
    X(eg_record_end)                                                                               \
    X(eg_graph_launch)                                                                             \
    X(eg_graph_reverse)                                                                            \
    X(eg_graph_describe)                                                                           \
    X(eg_graph_free)                                                                               \
    X(eg_device_count)                                                                             \
    X(eg_device_props)                                                                             \
    X(eg_stream_use)                                                                               \
    X(eg_stream_new)                                                                               \
    X(eg_stream_free)                                                                              \
    X(eg_stream_sync)                                                                              \
    X(eg_event_new)                                                                                \
    X(eg_event_free)                                                                               \
    X(eg_event_record)                                                                             \
    X(eg_event_wait)                                                                               \
    X(eg_event_elapsed)                                                                            \
    X(eg_sync)                                                                                     \
    X(eg_plane_new)                                                                                \
    X(eg_plane_dlpack)                                                                             \
    X(eg_banded_pack)                                                                              \
    X(eg_banded_unpack)                                                                            \
    X(eg_banded_pack_scalar)                                                                       \
    X(eg_plan_entry_figures)                                                                       \
    X(eg_plan_counters)                                                                            \
    X(eg_runner_reset)                                                                             \
    X(eg_runner_figures)                                                                           \
    X(eg_fft_new)                                                                                  \
    X(eg_fft_run)                                                                                  \
    X(eg_fft_free)                                                                                 \
    X(eg_gemm_new)                                                                                 \
    X(eg_gemm_run)                                                                                 \
    X(eg_gemm_free)                                                                                \
    X(eg_host_node_new)                                                                            \
    X(eg_host_node_free)                                                                           \
    X(eg_host_call)

/* The door's functions of one opened library, typed by their declarations in the header. */
typedef struct {
#define X(name) __typeof__(&name) name;
    DOOR_CALLS(X)
#undef X
} DoorFunctions;

typedef struct {
    PyObject_HEAD DoorFunctions fn;
    eg_ctx* ctx; /* NULL once the context is freed */
    /* The exception a host node's Python callable raised (type, value, traceback), kept for the
     * FAULT the door answers for it; all NULL when none is kept. */
    PyObject *kept_type, *kept_value, *kept_traceback;
} Door;

typedef enum {
    KIND_UNIT,
    KIND_PLANE,
    KIND_PLAN,
    KIND_RUNNER,
    KIND_GRAPH,
    KIND_FFT,
    KIND_GEMM,
    KIND_HOST_NODE,
    KIND_EVENT
} Kind;
static const char* const kind_names[]
    = { "unit", "plane", "plan", "runner", "graph", "fft", "gemm", "host_node", "event" };

/* A handle the door made, with the Door that made it; ptr is NULL once the handle is freed. */
typedef struct {
    PyObject_HEAD Door* door;
    void* ptr;
    Kind kind;
} Handle;

static PyTypeObject* handle_type = NULL;
/* NotWritten, BadRequest, Fault: the module's classes, indexed by status - 1. */
static PyObject* door_errors[3];

/* The name a consumed capsule takes (the DLPack rule), so its producer does not free the tensor. */
static const char DLPACK_USED[] = "used_dltensor_versioned";
static const char DLPACK_NAME[] = "dltensor_versioned";

/* ---- statuses ------------------------------------------------------------------------------ */

/* Raises the exception of a non-OK status, carrying the reason the door stored in the context
 * (or, for the FAULT of a failed host node, the exception its callable raised). */
static void door_raise(Door* d, eg_status st)
{
    if (st == EG_FAULT && d->kept_type) {
        /* the FAULT is a host node's callable failing: the caller gets the callable's own exception
         */
        PyErr_Restore(d->kept_type, d->kept_value, d->kept_traceback);
        d->kept_type = d->kept_value = d->kept_traceback = NULL;
        return;
    }
    Py_CLEAR(d->kept_type);
    Py_CLEAR(d->kept_value);
    Py_CLEAR(d->kept_traceback);
    size_t len = 0;
    char* text = NULL;
    if (d->fn.eg_error(d->ctx, NULL, 0, &len) == EG_OK && len > 0) {
        text = PyMem_Malloc(len + 1);
        if (text && d->fn.eg_error(d->ctx, text, len + 1, &len) != EG_OK)
            len = 0;
    }
    if (st >= EG_NOT_WRITTEN && st <= EG_FAULT)
        PyErr_SetString(door_errors[st - 1], text && len > 0 ? text : "the door stored no reason");
    else
        PyErr_Format(PyExc_RuntimeError, "door status %d: %s", (int)st, text ? text : "");
    PyMem_Free(text);
}

/* 0 when the status is OK; else -1 with the exception set. */
static int door_check(Door* d, eg_status st)
{
    if (st == EG_OK)
        return 0;
    door_raise(d, st);
    return -1;
}

static int door_ready(Door* d)
{
    if (d->ctx)
        return 1;
    PyErr_SetString(PyExc_ValueError, "the context is freed");
    return 0;
}

/* ---- handles ------------------------------------------------------------------------------- */

static PyObject* handle_make(Door* d, Kind kind, void* ptr)
{
    Handle* h = (Handle*)PyType_GenericAlloc(handle_type, 0);
    if (!h)
        return NULL;
    Py_INCREF((PyObject*)d);
    h->door = d;
    h->ptr  = ptr;
    h->kind = kind;
    return (PyObject*)h;
}

/* The door pointer of a live handle of the kind and of this Door; else -1 with a TypeError or a
 * ValueError. */
static int handle_ptr(Door* d, PyObject* o, Kind kind, void** ptr)
{
    if (!PyObject_TypeCheck(o, handle_type) || ((Handle*)o)->kind != kind) {
        PyErr_Format(PyExc_TypeError, "expected a %s handle", kind_names[kind]);
        return -1;
    }
    Handle* h = (Handle*)o;
    if (h->door != d) {
        PyErr_Format(PyExc_TypeError, "a %s handle of another Door", kind_names[kind]);
        return -1;
    }
    if (!h->ptr) {
        PyErr_Format(PyExc_ValueError, "the %s handle is freed", kind_names[kind]);
        return -1;
    }
    *ptr = h->ptr;
    return 0;
}

/* As handle_ptr, with None giving NULL (the door's optional handles). */
static int handle_ptr_or_none(Door* d, PyObject* o, Kind kind, void** ptr)
{
    if (o == NULL || o == Py_None) {
        *ptr = NULL;
        return 0;
    }
    return handle_ptr(d, o, kind, ptr);
}

/* Calls the door's free of the kind (every free is NULL-safe and answers OK). */
static eg_status handle_release(Door* d, Kind kind, void* ptr)
{
    switch (kind) {
    case KIND_UNIT:
        return d->fn.eg_unit_free(d->ctx, ptr);
    case KIND_PLANE:
        return d->fn.eg_plane_free(d->ctx, ptr);
    case KIND_PLAN:
        return d->fn.eg_plan_free(d->ctx, ptr);
    case KIND_RUNNER:
        return d->fn.eg_runner_free(d->ctx, ptr);
    case KIND_GRAPH:
        return d->fn.eg_graph_free(d->ctx, ptr);
    case KIND_FFT:
        return d->fn.eg_fft_free(d->ctx, ptr);
    case KIND_HOST_NODE:
        return d->fn.eg_host_node_free(d->ctx, ptr);
    case KIND_EVENT:
        return d->fn.eg_event_free(d->ctx, ptr);
    default:
        return d->fn.eg_gemm_free(d->ctx, ptr);
    }
}

/* The free methods: free(handle) releases it; a second free of the same handle is refused. */
static PyObject* handle_free(PyObject* self, PyObject* args, Kind kind)
{
    Door* d = (Door*)self;
    PyObject* o;
    void* ptr;
    if (!PyArg_ParseTuple(args, "O", &o) || !door_ready(d) || handle_ptr(d, o, kind, &ptr) < 0)
        return NULL;
    if (door_check(d, handle_release(d, kind, ptr)) < 0)
        return NULL;
    ((Handle*)o)->ptr = NULL;
    Py_RETURN_NONE;
}

static PyObject* m_unit_free(PyObject* self, PyObject* args)
{
    return handle_free(self, args, KIND_UNIT);
}

static PyObject* m_plane_free(PyObject* self, PyObject* args)
{
    return handle_free(self, args, KIND_PLANE);
}

static PyObject* m_plan_free(PyObject* self, PyObject* args)
{
    return handle_free(self, args, KIND_PLAN);
}

static PyObject* m_runner_free(PyObject* self, PyObject* args)
{
    return handle_free(self, args, KIND_RUNNER);
}

static PyObject* m_graph_free(PyObject* self, PyObject* args)
{
    return handle_free(self, args, KIND_GRAPH);
}

static PyObject* m_host_node_free(PyObject* self, PyObject* args)
{
    return handle_free(self, args, KIND_HOST_NODE);
}

/* ---- argument conversion ------------------------------------------------------------------- */

/* A sequence of integers as an array of `width` bytes (4 or 8) each; None or NULL: none. The
 * array is PyMem memory the caller frees. */
static int seq_ints(PyObject* seq, size_t width, void** out, Py_ssize_t* n)
{
    *out = NULL;
    *n   = 0;
    if (seq == NULL || seq == Py_None)
        return 0;
    PyObject* t = PySequence_Tuple(seq);
    if (!t)
        return -1;
    *n               = PyTuple_Size(t);
    unsigned char* v = PyMem_Malloc((size_t)(*n ? *n : 1) * width);
    if (!v) {
        Py_DECREF(t);
        PyErr_NoMemory();
        return -1;
    }
    for (Py_ssize_t i = 0; i < *n; i++) {
        long long x = PyLong_AsLongLong(PyTuple_GetItem(t, i));
        if (x == -1 && PyErr_Occurred())
            goto fail;
        if (width == sizeof(int32_t)) {
            int32_t narrow = (int32_t)x;
            if ((long long)narrow != x) {
                PyErr_SetString(PyExc_OverflowError, "an index does not fit 32 bits");
                goto fail;
            }
            memcpy(v + (size_t)i * width, &narrow, width);
        } else {
            memcpy(v + (size_t)i * width, &x, width);
        }
    }
    Py_DECREF(t);
    *out = v;
    return 0;
fail:
    Py_DECREF(t);
    PyMem_Free(v);
    return -1;
}

/* A sequence of live handles of the kind as an array of door pointers (PyMem memory). */
static int seq_handles(Door* d, PyObject* seq, Kind kind, void*** out, Py_ssize_t* n)
{
    *out = NULL;
    *n   = 0;
    if (seq == NULL || seq == Py_None)
        return 0;
    PyObject* t = PySequence_Tuple(seq);
    if (!t)
        return -1;
    *n       = PyTuple_Size(t);
    void** v = PyMem_Malloc((size_t)(*n ? *n : 1) * sizeof(void*));
    if (!v) {
        Py_DECREF(t);
        PyErr_NoMemory();
        return -1;
    }
    for (Py_ssize_t i = 0; i < *n; i++) {
        if (handle_ptr(d, PyTuple_GetItem(t, i), kind, &v[i]) < 0) {
            Py_DECREF(t);
            PyMem_Free(v);
            return -1;
        }
    }
    Py_DECREF(t);
    *out = v;
    return 0;
}


/* The arrays behind an eg_bind_desc, and the desc over them. Each uniform value is a cell of 8
 * bytes: the bytes the caller gave, zero-padded, so the door reads a whole cell at any scalar
 * width. */
enum { UNIFORM_CELL = 8 };
typedef struct {
    struct eg_bind_desc desc;
    void* plane_slots;
    void** planes;
    void* uniform_slots;
    void** values;
    unsigned char* cells;
} Bind;

static void bind_free(Bind* b)
{
    PyMem_Free(b->plane_slots);
    PyMem_Free(b->planes);
    PyMem_Free(b->uniform_slots);
    PyMem_Free(b->values);
    PyMem_Free(b->cells);
    memset(b, 0, sizeof *b);
}

/* The uniform values as cells; `bytes` objects of 1 to UNIFORM_CELL bytes. */
static int bind_cells(PyObject* seq, Bind* b, Py_ssize_t n)
{
    if (n == 0)
        return 0;
    PyObject* t = PySequence_Tuple(seq);
    if (!t)
        return -1;
    b->cells  = PyMem_Malloc((size_t)n * UNIFORM_CELL);
    b->values = PyMem_Malloc((size_t)n * sizeof(void*));
    if (!b->cells || !b->values) {
        Py_DECREF(t);
        PyErr_NoMemory();
        return -1;
    }
    memset(b->cells, 0, (size_t)n * UNIFORM_CELL);
    for (Py_ssize_t i = 0; i < n; i++) {
        char* bytes;
        Py_ssize_t len;
        if (PyBytes_AsStringAndSize(PyTuple_GetItem(t, i), &bytes, &len) < 0) {
            Py_DECREF(t);
            return -1;
        }
        if (len < 1 || len > UNIFORM_CELL) {
            Py_DECREF(t);
            PyErr_SetString(PyExc_ValueError, "a uniform value is 1 to 8 bytes");
            return -1;
        }
        memcpy(b->cells + (size_t)i * UNIFORM_CELL, bytes, (size_t)len);
        b->values[i] = b->cells + (size_t)i * UNIFORM_CELL;
    }
    Py_DECREF(t);
    return 0;
}

/* Fills the desc from sequences of slots, planes, uniform slots and uniform values (the last two
 * may be None). Sequences that pair up must have one length. */
static int bind_parse(
    Door* d, Bind* b, PyObject* slots, PyObject* planes, PyObject* uslots, PyObject* uvalues)
{
    Py_ssize_t n_planes, n_slots, n_uslots, n_uvalues = 0;
    memset(b, 0, sizeof *b);
    if (seq_ints(slots, sizeof(int32_t), &b->plane_slots, &n_slots) < 0
        || seq_handles(d, planes, KIND_PLANE, &b->planes, &n_planes) < 0
        || seq_ints(uslots, sizeof(int32_t), &b->uniform_slots, &n_uslots) < 0)
        goto fail;
    if (uvalues != NULL && uvalues != Py_None) {
        n_uvalues = PySequence_Size(uvalues);
        if (n_uvalues < 0)
            goto fail;
    }
    if (n_slots != n_planes || n_uslots != n_uvalues) {
        PyErr_SetString(PyExc_ValueError, "slots and the values they name must have one length");
        goto fail;
    }
    if (bind_cells(uvalues, b, n_uvalues) < 0)
        goto fail;
    b->desc.struct_size    = sizeof b->desc;
    b->desc.n_planes       = (int32_t)n_planes;
    b->desc.n_uniforms     = (int32_t)n_uvalues;
    b->desc.plane_slots    = b->plane_slots;
    b->desc.planes         = (eg_plane* const*)b->planes;
    b->desc.uniform_slots  = b->uniform_slots;
    b->desc.uniform_values = (const void* const*)b->values;
    return 0;
fail:
    bind_free(b);
    return -1;
}

/* ---- context ------------------------------------------------------------------------------- */

static PyObject* Door_new(PyTypeObject* type, PyObject* args, PyObject* kw)
{
    static char* names[] = { "backend", "device", NULL };
    unsigned long long handle;
    PyObject* device = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "K|O", names, &handle, &device))
        return NULL;
    if (handle == 0) {
        PyErr_SetString(PyExc_ValueError, "Door: the backend handle is 0, not an opened backend");
        return NULL;
    }
    ob_backend* lib = (ob_backend*)(uintptr_t)handle;
    /* The first function tells a library without a runtime (nothing exported) from a broken one. */
    if (!ob_runtime_symbol(lib, "eg_ctx_new")) {
        PyErr_SetString(door_errors[EG_NOT_WRITTEN - 1],
            "Door: the backend library states no runtime (it exports no eg_ctx_new)");
        return NULL;
    }
    Door* d = (Door*)PyType_GenericAlloc(type, 0);
    if (!d)
        return NULL;
#define X(name)                                                                                    \
    d->fn.name = (__typeof__(&name))ob_runtime_symbol(lib, #name);                                 \
    if (!d->fn.name) {                                                                             \
        PyErr_SetString(door_errors[EG_FAULT - 1], "Door: the runtime lacks the function " #name); \
        goto fail;                                                                                 \
    }
    DOOR_CALLS(X)
#undef X
    struct eg_ctx_settings settings
        = { .struct_size = sizeof settings, .abi_major = EG_DOOR_VERSION_MAJOR };
    if (device != Py_None) {
        long ordinal = PyLong_AsLong(device);
        if (ordinal == -1 && PyErr_Occurred())
            goto fail;
        settings.flags  = EG_CTX_DEVICE;
        settings.device = (int32_t)ordinal;
    }
    eg_status st = d->fn.eg_ctx_new(&settings, &d->ctx);
    if (!d->ctx) {
        PyErr_SetString(door_errors[EG_FAULT - 1], "the door made no context (out of memory)");
        goto fail;
    }
    if (st != EG_OK) {
        door_raise(d, st);
        goto fail;
    }
    return (PyObject*)d;
fail:
    Py_DECREF((PyObject*)d);
    return NULL;
}

/* A context frees itself, and what is left of its handles, with its Door; in a forked child
 * that frees host memory only, which is the one call the door allows there. A Door never closes
 * the backend library: its handle belongs to the caller of backend_open. */
static void Door_dealloc(PyObject* o)
{
    PyTypeObject* t = Py_TYPE(o);
    Door* d         = (Door*)o;
    if (d->ctx)
        d->fn.eg_ctx_free(d->ctx);
    Py_XDECREF(d->kept_type);
    Py_XDECREF(d->kept_value);
    Py_XDECREF(d->kept_traceback);
    ((freefunc)PyType_GetSlot(t, Py_tp_free))(o);
    Py_DECREF((PyObject*)t);
}

static PyObject* m_ctx_free(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    if (d->ctx) {
        /* with a DLPack export outstanding the door only closes the context; the pointer is not
         * used again either way */
        d->fn.eg_ctx_free(d->ctx);
        d->ctx = NULL;
    }
    Py_RETURN_NONE;
}

/* The reason of the last refusal: "" when none is stored. */
static PyObject* m_error(PyObject* self, PyObject* args)
{
    Door* d    = (Door*)self;
    size_t len = 0;
    if (!PyArg_ParseTuple(args, "") || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_error(d->ctx, NULL, 0, &len)) < 0)
        return NULL;
    char* text = PyMem_Malloc(len + 1);
    if (!text)
        return PyErr_NoMemory();
    PyObject* res = NULL;
    if (door_check(d, d->fn.eg_error(d->ctx, text, len + 1, &len)) == 0)
        res = PyUnicode_DecodeUTF8(text, (Py_ssize_t)len, "strict");
    PyMem_Free(text);
    return res;
}

/* ctx_info(): the door's eg_ctx_info as a dict, member by member under the struct's own names.
 * The weight_ members state the class key the dispatch weights were looked up by and the row
 * taken (-1: the built-in row); they are read-only facts of the context. */
static PyObject* m_ctx_info(PyObject* self, PyObject* args)
{
    Door* d                 = (Door*)self;
    struct eg_ctx_info info = { .struct_size = sizeof info };
    if (!PyArg_ParseTuple(args, "") || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_ctx_info(d->ctx, &info)) < 0)
        return NULL;
    return Py_BuildValue("{s:I,s:I,s:i,s:K,s:i,s:i,s:L,s:i,s:i,s:i,s:i}", "abi_major",
        info.abi_major, "abi_minor", info.abi_minor, "device", info.device, "capabilities",
        (unsigned long long)info.capabilities, "host_threads", info.host_threads, "device_type",
        info.device_type, "stream", (long long)info.stream, "weight_vendor", info.weight_vendor,
        "weight_family", info.weight_family, "weight_model", info.weight_model, "weight_row",
        info.weight_row);
}

static PyObject* m_host_threads(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    int threads;
    if (!PyArg_ParseTuple(args, "i", &threads) || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_host_threads(d->ctx, threads)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

/* ---- units and planes ---------------------------------------------------------------------- */

/* unit_load(path, name, digest, deps=()): the unit of that name and 32-byte digest from the
 * library at path; deps are the loaded unit handles its stages name. */
static PyObject* m_unit_load(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* names[] = { "path", "name", "digest", "deps", NULL };
    Door* d              = (Door*)self;
    const char *path, *name, *digest;
    Py_ssize_t digest_len;
    PyObject* odeps = NULL;
    if (!PyArg_ParseTupleAndKeywords(
            args, kw, "sss#|O", names, &path, &name, &digest, &digest_len, &odeps)
        || !door_ready(d))
        return NULL;
    if (digest_len != EG_DIGEST_BYTES) {
        PyErr_Format(PyExc_ValueError, "a unit digest is %d bytes", EG_DIGEST_BYTES);
        return NULL;
    }
    void** deps;
    Py_ssize_t n_deps;
    if (seq_handles(d, odeps, KIND_UNIT, &deps, &n_deps) < 0)
        return NULL;
    struct eg_unit_ref ref = { .struct_size = sizeof ref, .name = name };
    memcpy(ref.digest, digest, EG_DIGEST_BYTES);
    struct eg_unit_load_desc desc = { .struct_size = sizeof desc,
        .path                                      = path,
        .unit                                      = &ref,
        .deps                                      = (eg_unit* const*)deps,
        .n_deps                                    = (int32_t)n_deps };
    eg_unit* unit                 = NULL;
    PyObject* res                 = NULL;
    if (door_check(d, d->fn.eg_unit_load(d->ctx, &desc, &unit)) == 0)
        res = handle_make(d, KIND_UNIT, unit);
    PyMem_Free(deps);
    return res;
}

/* The array's DLPack 1.0 capsule (a new reference), asked with max_version (1, 0) and the stream
 * when one is given; NULL with a Python error set when the array gives none. */
static PyObject* dlpack_capsule(PyObject* array, PyObject* stream)
{
    PyObject* producer = PyObject_GetAttrString(array, "__dlpack__");
    PyObject* none     = PyTuple_New(0);
    PyObject* options  = PyDict_New();
    PyObject* version  = Py_BuildValue("(ii)", 1, 0);
    PyObject* capsule  = NULL;
    if (producer && none && options && version
        && PyDict_SetItemString(options, "max_version", version) == 0
        && (stream == Py_None || PyDict_SetItemString(options, "stream", stream) == 0))
        capsule = PyObject_Call(producer, none, options);
    if (capsule && !PyCapsule_IsValid(capsule, DLPACK_NAME)) {
        PyErr_SetString(
            PyExc_TypeError, "the array gave no DLPack 1.0 capsule (dltensor_versioned)");
        Py_CLEAR(capsule);
    }
    Py_XDECREF(version);
    Py_XDECREF(options);
    Py_XDECREF(none);
    Py_XDECREF(producer);
    return capsule;
}

/* plane_wrap(array, stream=None): takes the array's DLPack 1.0 tensor as a plane. The door owns
 * the tensor once the wrap succeeded (the capsule is marked used, so its producer does not free
 * it too); on a refusal the capsule keeps it and the producer frees it with the capsule. */
static PyObject* m_plane_wrap(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* names[] = { "array", "stream", NULL };
    Door* d              = (Door*)self;
    PyObject *array, *stream = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|O", names, &array, &stream) || !door_ready(d))
        return NULL;
    PyObject* capsule = dlpack_capsule(array, stream);
    if (!capsule)
        return NULL;
    struct DLManagedTensorVersioned* tensor = PyCapsule_GetPointer(capsule, DLPACK_NAME);
    eg_plane* plane                         = NULL;
    PyObject* res                           = NULL;
    if (door_check(d, d->fn.eg_plane_wrap(d->ctx, tensor, &plane)) == 0) {
        PyCapsule_SetName(capsule, DLPACK_USED);
        res = handle_make(d, KIND_PLANE, plane);
    }
    Py_DECREF(capsule);
    return res;
}

/* A new tuple of `count` integers read from `values`; NULL with a Python error set. */
static PyObject* tuple_of_int64(const int64_t* values, int32_t count)
{
    PyObject* tuple = PyTuple_New(count);
    for (int32_t i = 0; tuple && i < count; ++i) {
        PyObject* number = PyLong_FromLongLong(values[i]);
        if (!number || PyTuple_SetItem(tuple, i, number) < 0)
            Py_CLEAR(tuple);
    }
    return tuple;
}

/* array_layout(array): what the array's DLPack 1.0 tensor states, without taking it: `shape`,
 * `strides` in elements (None when the array is compact in C order), `dtype` as (code, bits,
 * lanes), `device` as (type, id) and `read_only`. The tensor goes back to its producer. */
static PyObject* m_array_layout(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject* array;
    if (!PyArg_ParseTuple(args, "O", &array))
        return NULL;
    PyObject* capsule = dlpack_capsule(array, Py_None);
    if (!capsule)
        return NULL;
    const struct DLManagedTensorVersioned* tensor = PyCapsule_GetPointer(capsule, DLPACK_NAME);
    const DLTensor* t                             = &tensor->dl_tensor;
    PyObject* shape                               = tuple_of_int64(t->shape, t->ndim);
    PyObject* strides = t->strides ? tuple_of_int64(t->strides, t->ndim) : NULL;
    PyObject* res     = NULL;
    if (shape && (strides || !t->strides))
        res = Py_BuildValue("{s:O,s:O,s:(iii),s:(ii),s:O}", "shape", shape, "strides",
            strides ? strides : Py_None, "dtype", (int)t->dtype.code, (int)t->dtype.bits,
            (int)t->dtype.lanes, "device", (int)t->device.device_type, (int)t->device.device_id,
            "read_only", tensor->flags & DLPACK_FLAG_BITMASK_READ_ONLY ? Py_True : Py_False);
    Py_XDECREF(strides);
    Py_XDECREF(shape);
    Py_DECREF(capsule);
    return res;
}

/* ---- plans and launches -------------------------------------------------------------------- */

/* plan_new(unit, structure, policy, n_samples, part_base=None, part_count=None, split_evenly=0):
 * the partitions are given by base and count, or the range is split evenly into `split_evenly`
 * parts; not both. */
static PyObject* m_plan_new(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* names[] = { "unit", "structure", "policy", "n_samples", "part_base", "part_count",
        "split_evenly", NULL };
    Door* d              = (Door*)self;
    PyObject *ounit, *obase = NULL, *ocount = NULL;
    int structure, policy;
    long long n_samples, split_evenly = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OiiL|OOL", names, &ounit, &structure, &policy,
            &n_samples, &obase, &ocount, &split_evenly)
        || !door_ready(d))
        return NULL;
    void* unit;
    void *base, *count;
    Py_ssize_t n_base, n_count;
    if (handle_ptr(d, ounit, KIND_UNIT, &unit) < 0)
        return NULL;
    if (seq_ints(obase, sizeof(int64_t), &base, &n_base) < 0)
        return NULL;
    if (seq_ints(ocount, sizeof(int64_t), &count, &n_count) < 0) {
        PyMem_Free(base);
        return NULL;
    }
    PyObject* res = NULL;
    if (n_base != n_count) {
        PyErr_SetString(PyExc_ValueError, "part_base and part_count must have one length");
    } else if (split_evenly != 0 && n_base != 0) {
        PyErr_SetString(
            PyExc_ValueError, "split_evenly and part_base / part_count are two ways to partition");
    } else {
        struct eg_plan_desc desc = { .struct_size = sizeof desc,
            .unit                                 = unit,
            .structure                            = structure,
            .policy                               = policy,
            .n_samples                            = n_samples,
            .flags                                = split_evenly != 0 ? EG_PLAN_SPLIT_EVENLY : 0,
            .n_partitions                         = split_evenly != 0 ? split_evenly : n_base,
            .part_base                            = base,
            .part_count                           = count };
        eg_plan* plan            = NULL;
        if (door_check(d, d->fn.eg_plan_new(d->ctx, &desc, &plan)) == 0)
            res = handle_make(d, KIND_PLAN, plan);
    }
    PyMem_Free(base);
    PyMem_Free(count);
    return res;
}

/* bind(plan, plane_slots, planes, uniform_slots=None, uniform_values=None): planes and uniforms by
 * slot; a uniform value is the bytes of the scalar at the slot's declared dtype. */
static PyObject* m_bind(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* names[]
        = { "plan", "plane_slots", "planes", "uniform_slots", "uniform_values", NULL };
    Door* d = (Door*)self;
    PyObject *oplan, *slots, *planes, *uslots = NULL, *uvalues = NULL;
    if (!PyArg_ParseTupleAndKeywords(
            args, kw, "OOO|OO", names, &oplan, &slots, &planes, &uslots, &uvalues)
        || !door_ready(d))
        return NULL;
    void* plan;
    Bind b;
    if (handle_ptr(d, oplan, KIND_PLAN, &plan) < 0
        || bind_parse(d, &b, slots, planes, uslots, uvalues) < 0)
        return NULL;
    int bad = door_check(d, d->fn.eg_bind(d->ctx, plan, &b.desc)) < 0;
    bind_free(&b);
    if (bad)
        return NULL;
    Py_RETURN_NONE;
}

static PyObject* m_clear(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject *oplan, *oslots;
    if (!PyArg_ParseTuple(args, "OO", &oplan, &oslots) || !door_ready(d))
        return NULL;
    void* plan;
    void* slots;
    Py_ssize_t n;
    if (handle_ptr(d, oplan, KIND_PLAN, &plan) < 0
        || seq_ints(oslots, sizeof(int32_t), &slots, &n) < 0)
        return NULL;
    int bad = door_check(d, d->fn.eg_clear(d->ctx, plan, (int32_t)n, slots)) < 0;
    PyMem_Free(slots);
    if (bad)
        return NULL;
    Py_RETURN_NONE;
}

static PyObject* m_set_count(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject* oplan;
    unsigned long count;
    void* plan;
    if (!PyArg_ParseTuple(args, "Ok", &oplan, &count) || !door_ready(d)
        || handle_ptr(d, oplan, KIND_PLAN, &plan) < 0)
        return NULL;
    if (count > UINT32_MAX) {
        PyErr_SetString(PyExc_OverflowError, "a count does not fit 32 bits");
        return NULL;
    }
    if (door_check(d, d->fn.eg_set_count(d->ctx, plan, (uint32_t)count)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

/* launch(plan): the host tiles run (0 on a device). */
static PyObject* m_launch(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject* oplan;
    void* plan;
    uint64_t tiles = 0;
    if (!PyArg_ParseTuple(args, "O", &oplan) || !door_ready(d)
        || handle_ptr(d, oplan, KIND_PLAN, &plan) < 0)
        return NULL;
    if (door_check(d, d->fn.eg_launch(d->ctx, plan, &tiles)) < 0)
        return NULL;
    return PyLong_FromUnsignedLongLong(tiles);
}

/* ---- runners ------------------------------------------------------------------------------- */

/* runner_new(plans, max_steps, flags=0, derived_roles=None, derived_units=None, every=0,
 * window=0, memory_budget=0, reorder_theta=0.0, steps_reverse=0, events=0, on_no_event=0,
 * steps=None): the fields of eg_runner_desc, a field left out being its zero. `steps` is a sequence
 * of (graph, before_plan) pairs, passed as given: the door checks their order and the
 * recorded-runner rules. */
static int seq_steps(Door* d, PyObject* osteps, struct eg_step_graph** out, Py_ssize_t* n)
{
    *out = NULL;
    *n   = 0;
    if (osteps == NULL || osteps == Py_None)
        return 0;
    Py_ssize_t count = PySequence_Size(osteps);
    if (count < 0)
        return -1;
    struct eg_step_graph* steps = PyMem_Malloc(((size_t)count + 1) * sizeof *steps);
    if (steps == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject* pair = PySequence_GetItem(osteps, i);
        void* graph    = NULL;
        long before    = 0;
        int ok         = pair != NULL && PyTuple_Check(pair) && PyTuple_Size(pair) == 2;
        if (pair != NULL && !ok)
            PyErr_SetString(PyExc_TypeError, "each step is a (graph, before_plan) pair");
        ok = ok && handle_ptr(d, PyTuple_GetItem(pair, 0), KIND_GRAPH, &graph) == 0;
        if (ok) {
            before = PyLong_AsLong(PyTuple_GetItem(pair, 1));
            ok     = !(before == -1 && PyErr_Occurred());
        }
        Py_XDECREF(pair);
        if (!ok) {
            PyMem_Free(steps);
            return -1;
        }
        steps[i] = (struct eg_step_graph){ .graph = graph, .before_plan = (int32_t)before };
    }
    *out = steps;
    *n   = count;
    return 0;
}

static PyObject* m_runner_new(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* names[] = { "plans", "max_steps", "flags", "derived_roles", "derived_units",
        "every", "window", "memory_budget", "reorder_theta", "steps_reverse", "events",
        "on_no_event", "steps", NULL };
    Door* d              = (Door*)self;
    PyObject *oplans, *oroles = NULL, *ounits = NULL, *osteps = NULL;
    unsigned int flags = 0;
    long long max_steps, every = 0, window = 0;
    unsigned long long budget = 0;
    double theta              = 0.0;
    int steps_reverse = 0, events = 0, on_no_event = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OL|IOOLLKdiiiO", names, &oplans, &max_steps, &flags,
            &oroles, &ounits, &every, &window, &budget, &theta, &steps_reverse, &events,
            &on_no_event, &osteps)
        || !door_ready(d))
        return NULL;
    void **plans = NULL, **units = NULL;
    struct eg_step_graph* steps = NULL;
    void* roles                 = NULL;
    Py_ssize_t n_plans, n_units, n_roles, n_steps;
    PyObject* res = NULL;
    if (seq_handles(d, oplans, KIND_PLAN, &plans, &n_plans) < 0
        || seq_handles(d, ounits, KIND_UNIT, &units, &n_units) < 0
        || seq_ints(oroles, sizeof(int32_t), &roles, &n_roles) < 0
        || seq_steps(d, osteps, &steps, &n_steps) < 0)
        goto done;
    if (n_roles != n_units) {
        PyErr_SetString(PyExc_ValueError, "derived_roles and derived_units must have one length");
        goto done;
    }
    struct eg_runner_desc desc = { .struct_size = sizeof desc,
        .flags                                  = flags,
        .n_plans                                = (int32_t)n_plans,
        .n_derived                              = (int32_t)n_units,
        .plans                                  = (eg_plan* const*)plans,
        .derived_roles                          = roles,
        .derived_units                          = (eg_unit* const*)units,
        .max_steps                              = max_steps,
        .every                                  = every,
        .window                                 = window,
        .memory_budget                          = budget,
        .reorder_theta                          = theta,
        .steps_reverse                          = steps_reverse,
        .events                                 = events,
        .on_no_event                            = on_no_event,
        .step_graphs                            = steps,
        .n_step_graphs                          = (int32_t)n_steps };
    eg_runner* runner          = NULL;
    if (door_check(d, d->fn.eg_runner_new(d->ctx, &desc, &runner)) == 0)
        res = handle_make(d, KIND_RUNNER, runner);
done:
    PyMem_Free(plans);
    PyMem_Free(units);
    PyMem_Free(roles);
    PyMem_Free(steps);
    return res;
}

/* run(runner): runs to done or max_steps; the report as a dict of eg_run_report's fields. */
static PyObject* m_run(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject* orunner;
    void* runner;
    struct eg_run_report r = { .struct_size = sizeof r };
    if (!PyArg_ParseTuple(args, "O", &orunner) || !door_ready(d)
        || handle_ptr(d, orunner, KIND_RUNNER, &runner) < 0)
        return NULL;
    if (door_check(d, d->fn.eg_run(d->ctx, runner, &r)) < 0)
        return NULL;
    return Py_BuildValue("{s:I,s:i,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:d,s:d}", "flags", r.flags, "mode",
        r.mode, "n", (long long)r.n, "finished", (long long)r.finished, "launches",
        (long long)r.launches, "steps", (long long)r.steps, "compactions", (long long)r.compactions,
        "reorders", (long long)r.reorders, "mapped", (long long)r.mapped, "build_s", r.build_s,
        "slot_fill", r.slot_fill);
}

/* run_events(runner, t_stop=None, status=None): the last refine run's per-sample results. */
static PyObject* m_run_events(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject *orunner, *ot = Py_None, *os = Py_None;
    void *runner, *t_stop, *status;
    if (!PyArg_ParseTuple(args, "O|OO", &orunner, &ot, &os) || !door_ready(d)
        || handle_ptr(d, orunner, KIND_RUNNER, &runner) < 0
        || handle_ptr_or_none(d, ot, KIND_PLANE, &t_stop) < 0
        || handle_ptr_or_none(d, os, KIND_PLANE, &status) < 0)
        return NULL;
    if (door_check(d, d->fn.eg_run_events(d->ctx, runner, t_stop, status)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

/* A pair (slots, planes) as the cotangent or gradient binding of a reverse. */
static int reverse_side(Door* d, PyObject* side, Bind* b)
{
    PyObject *slots, *planes;
    if (!PyArg_ParseTuple(side, "OO", &slots, &planes))
        return -1;
    return bind_parse(d, b, slots, planes, NULL, NULL);
}

/* reverse(runner, seeds, grads): each a pair (slots, planes) by the primal unit's slots. */
static PyObject* m_reverse(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject *orunner, *oseeds, *ograds;
    void* runner;
    if (!PyArg_ParseTuple(args, "OOO", &orunner, &oseeds, &ograds) || !door_ready(d)
        || handle_ptr(d, orunner, KIND_RUNNER, &runner) < 0)
        return NULL;
    Bind seeds, grads;
    if (reverse_side(d, oseeds, &seeds) < 0)
        return NULL;
    if (reverse_side(d, ograds, &grads) < 0) {
        bind_free(&seeds);
        return NULL;
    }
    int bad = door_check(d, d->fn.eg_reverse(d->ctx, runner, &seeds.desc, &grads.desc)) < 0;
    bind_free(&seeds);
    bind_free(&grads);
    if (bad)
        return NULL;
    Py_RETURN_NONE;
}

/* ---- graph operations and recording -------------------------------------------------------- */

/* reduce(source, out, op, policy, comp=None, flags=0, index=None): index receives the lane of a min or max. */
static PyObject* m_reduce(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* names[] = { "source", "out", "op", "policy", "comp", "flags", "index", NULL };
    Door* d              = (Door*)self;
    PyObject *osource, *oout, *ocomp = Py_None, *oindex = Py_None;
    int op, policy;
    unsigned int flags = 0;
    if (!PyArg_ParseTupleAndKeywords(
            args, kw, "OOii|OIO", names, &osource, &oout, &op, &policy, &ocomp, &flags, &oindex)
        || !door_ready(d))
        return NULL;
    void *source, *out, *comp, *index;
    if (handle_ptr(d, osource, KIND_PLANE, &source) < 0 || handle_ptr(d, oout, KIND_PLANE, &out) < 0
        || handle_ptr_or_none(d, ocomp, KIND_PLANE, &comp) < 0
        || handle_ptr_or_none(d, oindex, KIND_PLANE, &index) < 0)
        return NULL;
    struct eg_reduce_desc desc = { .struct_size = sizeof desc,
        .flags                                  = flags,
        .in                                     = source,
        .out                                    = out,
        .comp                                   = comp,
        .op                                     = op,
        .policy                                 = policy,
        .index                                  = index };
    if (door_check(d, d->fn.eg_reduce(d->ctx, &desc)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

/* scan(source, out, op, flags=0, index=None): the running sum, minimum or maximum of source into out;
 * index receives the running winner's lane of a min or max. */
static PyObject* m_scan(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* names[] = { "source", "out", "op", "flags", "index", NULL };
    Door* d              = (Door*)self;
    PyObject *osource, *oout, *oindex = Py_None;
    int op;
    unsigned int flags = 0;
    if (!PyArg_ParseTupleAndKeywords(
            args, kw, "OOi|IO", names, &osource, &oout, &op, &flags, &oindex)
        || !door_ready(d))
        return NULL;
    void *source, *out, *index;
    if (handle_ptr(d, osource, KIND_PLANE, &source) < 0
        || handle_ptr(d, oout, KIND_PLANE, &out) < 0
        || handle_ptr_or_none(d, oindex, KIND_PLANE, &index) < 0)
        return NULL;
    struct eg_scan_desc desc = { .struct_size = sizeof desc,
        .flags                                = flags,
        .in                                   = source,
        .out                                  = out,
        .op                                   = op,
        .index                                = index };
    if (door_check(d, d->fn.eg_scan(d->ctx, &desc)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

/* compact(keep, kept, count): the ascending lanes a bool plane keeps into kept, and their number into count. */
static PyObject* m_compact(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject *okeep, *okept, *ocount;
    if (!PyArg_ParseTuple(args, "OOO", &okeep, &okept, &ocount) || !door_ready(d))
        return NULL;
    void *keep, *kept, *count;
    if (handle_ptr(d, okeep, KIND_PLANE, &keep) < 0 || handle_ptr(d, okept, KIND_PLANE, &kept) < 0
        || handle_ptr(d, ocount, KIND_PLANE, &count) < 0)
        return NULL;
    struct eg_compact_desc desc
        = { .struct_size = sizeof desc, .keep = keep, .kept = kept, .count = count };
    if (door_check(d, d->fn.eg_compact(d->ctx, &desc)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

/* order_by_key(keys, order, starts, bound): the stable permutation of the lanes by integer key into order, and the first place of
 * each key into starts. */
static PyObject* m_order_by_key(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject *okeys, *oorder, *ostarts;
    long long bound;
    if (!PyArg_ParseTuple(args, "OOOL", &okeys, &oorder, &ostarts, &bound) || !door_ready(d))
        return NULL;
    void *keys, *order, *starts;
    if (handle_ptr(d, okeys, KIND_PLANE, &keys) < 0 || handle_ptr(d, oorder, KIND_PLANE, &order) < 0
        || handle_ptr(d, ostarts, KIND_PLANE, &starts) < 0)
        return NULL;
    struct eg_order_desc desc = { .struct_size = sizeof desc,
        .keys = keys,
        .order = order,
        .starts = starts,
        .bound = bound };
    if (door_check(d, d->fn.eg_order_by_key(d->ctx, &desc)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

/* reorder(planes, order): every plane of the sequence permuted in place so that p[j] holds what p[order[j]] held. */
static PyObject* m_reorder(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject *oplanes, *oorder;
    if (!PyArg_ParseTuple(args, "OO", &oplanes, &oorder) || !door_ready(d))
        return NULL;
    void* order;
    if (handle_ptr(d, oorder, KIND_PLANE, &order) < 0)
        return NULL;
    Py_ssize_t count = PySequence_Size(oplanes);
    if (count < 0)
        return NULL;
    eg_plane** planes = PyMem_Calloc(count > 0 ? (size_t)count : 1, sizeof *planes);
    if (planes == NULL)
        return PyErr_NoMemory();
    Py_ssize_t i = 0;
    for (; i < count; ++i) {
        PyObject* item = PySequence_GetItem(oplanes, i);
        void* plane = NULL;
        int bad = item == NULL || handle_ptr(d, item, KIND_PLANE, &plane) < 0;
        Py_XDECREF(item);
        if (bad)
            break;
        planes[i] = plane;
    }
    PyObject* result = NULL;
    if (i == count) {
        struct eg_reorder_desc desc = { .struct_size = sizeof desc,
            .n_planes = (uint32_t)count,
            .planes = planes,
            .order = order };
        if (door_check(d, d->fn.eg_reorder(d->ctx, &desc)) == 0) {
            Py_INCREF(Py_None);
            result = Py_None;
        }
    }
    PyMem_Free(planes);
    return result;
}

static PyObject* m_copy(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject *osrc, *odst;
    void *src, *dst;
    if (!PyArg_ParseTuple(args, "OO", &osrc, &odst) || !door_ready(d)
        || handle_ptr(d, osrc, KIND_PLANE, &src) < 0 || handle_ptr(d, odst, KIND_PLANE, &dst) < 0)
        return NULL;
    if (door_check(d, d->fn.eg_copy(d->ctx, src, dst)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

static PyObject* m_record_abort(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    if (!PyArg_ParseTuple(args, "") || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_record_abort(d->ctx)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

static PyObject* m_record_begin(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    if (!PyArg_ParseTuple(args, "") || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_record_begin(d->ctx)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

/* record_cond_begin(kind, cell, n_arms=0, negate=False, max_iters=0, cell_index=0, baseline=None,
 * baseline_index=0, counter=None, limit=None, limit_index=0, brk=None, brk_index=0): opens a
 * conditional on the int32 cell plane (None only for a WHILE with a limit or a brk); the defaults
 * are the header's: no counter, limit or break. */
static PyObject* m_record_cond_begin(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* names[] = { "kind", "cell", "n_arms", "negate", "max_iters", "cell_index",
        "baseline", "baseline_index", "counter", "limit", "limit_index", "brk", "brk_index", NULL };
    Door* d              = (Door*)self;
    PyObject *ocell, *obaseline = Py_None, *ocounter = Py_None, *olimit = Py_None, *obrk = Py_None;
    int kind, n_arms = 0, negate = 0;
    long long max_iters = 0, cell_index = 0, baseline_index = 0, limit_index = 0, brk_index = 0;
    void *cell, *baseline, *counter, *limit, *brk;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "iO|ipLLOLOOLOL", names, &kind, &ocell, &n_arms,
            &negate, &max_iters, &cell_index, &obaseline, &baseline_index, &ocounter, &olimit,
            &limit_index, &obrk, &brk_index)
        || !door_ready(d) || handle_ptr_or_none(d, ocell, KIND_PLANE, &cell) < 0
        || handle_ptr_or_none(d, obaseline, KIND_PLANE, &baseline) < 0
        || handle_ptr_or_none(d, ocounter, KIND_PLANE, &counter) < 0
        || handle_ptr_or_none(d, olimit, KIND_PLANE, &limit) < 0
        || handle_ptr_or_none(d, obrk, KIND_PLANE, &brk) < 0)
        return NULL;
    struct eg_cond_desc desc = { .struct_size = sizeof desc,
        .flags                                = negate ? EG_COND_NEGATE : 0,
        .kind                                 = kind,
        .n_arms                               = n_arms,
        .cell                                 = cell,
        .max_iters                            = max_iters,
        .cell_index                           = cell_index,
        .baseline                             = baseline,
        .baseline_index                       = baseline_index,
        .counter                              = counter,
        .limit                                = limit,
        .limit_index                          = limit_index,
        .brk                                  = brk,
        .brk_index                            = brk_index };
    if (door_check(d, d->fn.eg_record_cond_begin(d->ctx, &desc)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

static PyObject* m_record_cond_arm(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    if (!PyArg_ParseTuple(args, "") || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_record_cond_arm(d->ctx)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

static PyObject* m_record_cond_end(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    if (!PyArg_ParseTuple(args, "") || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_record_cond_end(d->ctx)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

static PyObject* m_record_fork_begin(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    if (!PyArg_ParseTuple(args, "") || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_record_fork_begin(d->ctx)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

static PyObject* m_record_fork_branch(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    if (!PyArg_ParseTuple(args, "") || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_record_fork_branch(d->ctx)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

static PyObject* m_record_fork_end(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    if (!PyArg_ParseTuple(args, "") || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_record_fork_end(d->ctx)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

/* record_wait(planes): the rest of the current branch runs after the sibling branches that write
 * the planes. */
static PyObject* m_record_wait(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject* oplanes;
    void** planes;
    Py_ssize_t n;
    if (!PyArg_ParseTuple(args, "O", &oplanes) || !door_ready(d)
        || seq_handles(d, oplanes, KIND_PLANE, &planes, &n) < 0)
        return NULL;
    int bad = door_check(d, d->fn.eg_record_wait(d->ctx, (int32_t)n, (eg_plane* const*)planes)) < 0;
    PyMem_Free(planes);
    if (bad)
        return NULL;
    Py_RETURN_NONE;
}

/* A sequence of (plane, plane) pairs as two arrays of door pointers, one per member. None or
 * NULL: none. Both arrays are PyMem memory the caller frees. */
static int seq_plane_pairs(Door* d, PyObject* seq, void*** a, void*** b, Py_ssize_t* n)
{
    *a = *b = NULL;
    *n      = 0;
    if (seq == NULL || seq == Py_None)
        return 0;
    PyObject* t = PySequence_Tuple(seq);
    if (!t)
        return -1;
    *n     = PyTuple_Size(t);
    *a     = PyMem_Malloc((size_t)(*n ? *n : 1) * sizeof(void*));
    *b     = PyMem_Malloc((size_t)(*n ? *n : 1) * sizeof(void*));
    int rc = (*a && *b) ? 0 : (PyErr_NoMemory(), -1);
    for (Py_ssize_t i = 0; rc == 0 && i < *n; i++) {
        PyObject* pair = PySequence_Tuple(PyTuple_GetItem(t, i));
        if (!pair) {
            rc = -1;
        } else if (PyTuple_Size(pair) != 2) {
            PyErr_SetString(PyExc_TypeError, "expected a pair");
            rc = -1;
        } else if (handle_ptr(d, PyTuple_GetItem(pair, 0), KIND_PLANE, &(*a)[i]) < 0
            || handle_ptr(d, PyTuple_GetItem(pair, 1), KIND_PLANE, &(*b)[i]) < 0) {
            rc = -1;
        }
        Py_XDECREF(pair);
    }
    Py_DECREF(t);
    if (rc < 0) {
        PyMem_Free(*a);
        PyMem_Free(*b);
        *a = *b = NULL;
    }
    return rc;
}

/* record_end(diff=None): the graph that replays the recording. diff is a sequence of (launch
 * ordinal, vjp unit): the graph is then planned for graph_reverse. */
static PyObject* m_record_end(PyObject* self, PyObject* args)
{
    Door* d         = (Door*)self;
    eg_graph* graph = NULL;
    PyObject* odiff = Py_None;
    if (!PyArg_ParseTuple(args, "|O", &odiff) || !door_ready(d))
        return NULL;
    if (odiff == Py_None) {
        if (door_check(d, d->fn.eg_record_end(d->ctx, NULL, &graph)) < 0)
            return NULL;
        return handle_make(d, KIND_GRAPH, graph);
    }
    PyObject* t = PySequence_Tuple(odiff);
    if (!t)
        return NULL;
    Py_ssize_t n            = PyTuple_Size(t);
    struct eg_node_vjp* vjp = PyMem_Malloc((size_t)(n ? n : 1) * sizeof *vjp);
    if (!vjp) {
        Py_DECREF(t);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* pair = PySequence_Tuple(PyTuple_GetItem(t, i));
        long ordinal   = -1;
        if (!pair)
            goto fail;
        if (PyTuple_Size(pair) != 2) {
            PyErr_SetString(PyExc_TypeError, "expected a (launch ordinal, vjp unit) pair");
            Py_DECREF(pair);
            goto fail;
        }
        ordinal = PyLong_AsLong(PyTuple_GetItem(pair, 0));
        if ((ordinal == -1 && PyErr_Occurred()) || ordinal < INT32_MIN || ordinal > INT32_MAX) {
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_OverflowError, "a launch ordinal does not fit 32 bits");
            Py_DECREF(pair);
            goto fail;
        }
        vjp[i].launch = (int32_t)ordinal;
        void* unit;
        int bad = handle_ptr(d, PyTuple_GetItem(pair, 1), KIND_UNIT, &unit) < 0;
        Py_DECREF(pair);
        if (bad)
            goto fail;
        vjp[i].vjp = unit;
    }
    struct eg_record_end_desc desc = {
        .struct_size = sizeof desc, .flags = EG_RECORD_DIFF, .n_vjps = (int32_t)n, .vjps = vjp
    };
    int bad = door_check(d, d->fn.eg_record_end(d->ctx, &desc, &graph)) < 0;
    PyMem_Free(vjp);
    Py_DECREF(t);
    return bad ? NULL : handle_make(d, KIND_GRAPH, graph);
fail:
    PyMem_Free(vjp);
    Py_DECREF(t);
    return NULL;
}

/* graph_reverse(graph, seeds, grads, strict=False): seeds are (output plane, seed plane) pairs and
 * grads are (input plane, gradient plane) pairs; the door matches the planes by handle. Returns
 * the door's report: (bitwise per gradient, memory passes). */
static PyObject* m_graph_reverse(PyObject* self, PyObject* args, PyObject* kw)
{
    Door* d              = (Door*)self;
    static char* names[] = { "graph", "seeds", "grads", "strict", NULL };
    PyObject *ograph, *oseeds, *ograds;
    int strict = 0;
    void* graph;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|p", names, &ograph, &oseeds, &ograds, &strict)
        || !door_ready(d) || handle_ptr(d, ograph, KIND_GRAPH, &graph) < 0)
        return NULL;
    void **outputs, **seeds, **inputs, **grads;
    Py_ssize_t n_seeds, n_grads;
    if (seq_plane_pairs(d, oseeds, &outputs, &seeds, &n_seeds) < 0)
        return NULL;
    if (seq_plane_pairs(d, ograds, &inputs, &grads, &n_grads) < 0) {
        PyMem_Free(outputs);
        PyMem_Free(seeds);
        return NULL;
    }
    uint8_t* bitwise = PyMem_Malloc((size_t)(n_grads ? n_grads : 1));
    int32_t passes   = 0;
    if (!bitwise) {
        PyMem_Free(outputs);
        PyMem_Free(seeds);
        PyMem_Free(inputs);
        PyMem_Free(grads);
        return PyErr_NoMemory();
    }
    struct eg_graph_reverse_desc desc = { .struct_size = sizeof desc,
        .flags                                         = strict ? EG_REVERSE_STRICT : 0,
        .n_seeds                                       = (int32_t)n_seeds,
        .outputs                                       = (eg_plane* const*)outputs,
        .seeds                                         = (eg_plane* const*)seeds,
        .n_grads                                       = (int32_t)n_grads,
        .inputs                                        = (eg_plane* const*)inputs,
        .grads                                         = (eg_plane* const*)grads,
        .bitwise                                       = bitwise,
        .passes                                        = &passes };
    int bad = door_check(d, d->fn.eg_graph_reverse(d->ctx, graph, &desc)) < 0;
    PyMem_Free(outputs);
    PyMem_Free(seeds);
    PyMem_Free(inputs);
    PyMem_Free(grads);
    PyObject* flags = bad ? NULL : PyTuple_New(n_grads);
    for (Py_ssize_t i = 0; flags && i < n_grads; i++)
        PyTuple_SetItem(flags, i, PyBool_FromLong(bitwise[i]));
    PyMem_Free(bitwise);
    if (!flags)
        return NULL;
    PyObject* report = Py_BuildValue("(Oi)", flags, passes);
    Py_DECREF(flags);
    return report;
}

/* A tuple of ints from an array of n values of the given element type. */
#define INT_TUPLE(name, type)                                                                      \
    static PyObject* name(const type* v, int32_t n)                                                \
    {                                                                                              \
        PyObject* t = PyTuple_New(n);                                                              \
        for (int32_t i = 0; t && i < n; i++) {                                                     \
            PyObject* x = PyLong_FromLongLong((long long)v[i]);                                    \
            if (!x) {                                                                              \
                Py_CLEAR(t);                                                                       \
            } else {                                                                               \
                PyTuple_SetItem(t, i, x);                                                          \
            }                                                                                      \
        }                                                                                          \
        return t;                                                                                  \
    }
INT_TUPLE(access_tuple, uint8_t)
INT_TUPLE(slot_tuple, int32_t)

/* The planes the caller listed, as the door knows them: a description names a plane by its index
 * in this list, or -1 when the caller did not list it. */
typedef struct {
    void** ptr;
    Py_ssize_t n;
} PlaneList;

static long plane_index(const PlaneList* known, const eg_plane* plane)
{
    for (Py_ssize_t i = 0; i < known->n; i++)
        if (known->ptr[i] == plane)
            return (long)i;
    return -1;
}

/* The planes of a node as their indices in the caller's list. */
static PyObject* plane_tuple(const PlaneList* known, eg_plane* const* v, int32_t n)
{
    PyObject* t = PyTuple_New(n);
    for (int32_t i = 0; t && i < n; i++) {
        PyObject* x = PyLong_FromLong(plane_index(known, v[i]));
        if (!x)
            Py_CLEAR(t);
        else
            PyTuple_SetItem(t, i, x);
    }
    return t;
}

static PyObject* node_dict(const PlaneList* known, const struct eg_graph_node* n)
{
    PyObject* unit   = n->unit ? PyUnicode_FromString(n->unit) : (Py_INCREF(Py_None), Py_None);
    PyObject* planes = plane_tuple(known, n->planes, n->n_planes);
    PyObject* access = access_tuple(n->access, n->n_planes);
    PyObject* slot   = slot_tuple(n->slot, n->n_planes);
    PyObject* node   = NULL;
    if (unit && planes && access && slot)
        node = Py_BuildValue("{s:i,s:i,s:i,s:i,s:O,s:O,s:O,s:O,s:I,s:O,s:L,s:L,s:i,s:i}", "kind",
            n->kind, "parent", n->parent, "level", n->level, "arm", n->arm, "unit", unit, "planes",
            planes, "access", access, "slot", slot, "cond_flags", n->cond_flags, "has_baseline",
            n->has_baseline ? Py_True : Py_False, "max_iters", (long long)n->max_iters,
            "cell_index", (long long)n->cell_index, "n_plans", n->n_plans, "launch", n->launch);
    Py_XDECREF(unit);
    Py_XDECREF(planes);
    Py_XDECREF(access);
    Py_XDECREF(slot);
    return node;
}

static PyObject* edge_dict(const PlaneList* known, const struct eg_graph_edge* e)
{
    PyObject* plane
        = e->plane ? PyLong_FromLong(plane_index(known, e->plane)) : (Py_INCREF(Py_None), Py_None);
    if (!plane)
        return NULL;
    PyObject* edge = Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:O}", "from", e->from, "to", e->to,
        "kind", e->kind, "from_slot", e->from_slot, "to_slot", e->to_slot, "plane", plane);
    Py_DECREF(plane);
    return edge;
}

/* A list of the dictionaries a builder makes for each of n array elements, which the door lays
 * out `stride` bytes apart (its own struct size, not this file's). */
#define DICT_LIST(name, type, build)                                                               \
    static PyObject* name(const PlaneList* known, const type* v, int32_t n, uint32_t stride)       \
    {                                                                                              \
        PyObject* list = PyList_New(n);                                                            \
        for (int32_t i = 0; list && i < n; i++) {                                                  \
            PyObject* item = build(known, (const type*)((const char*)v + (size_t)i * stride));     \
            if (!item)                                                                             \
                Py_CLEAR(list);                                                                    \
            else                                                                                   \
                PyList_SetItem(list, i, item);                                                     \
        }                                                                                          \
        return list;                                                                               \
    }
DICT_LIST(node_list, struct eg_graph_node, node_dict)
DICT_LIST(edge_list, struct eg_graph_edge, edge_dict)

/* graph_describe(graph, planes): the recorded graph's nodes and edges, copied out of the door's
 * arrays. A plane appears as its index in `planes`, or -1 when the door names one not listed. */
static PyObject* m_graph_describe(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject *ograph, *oplanes;
    void* graph;
    PlaneList known;
    struct eg_graph_description desc = { .struct_size = sizeof desc };
    if (!PyArg_ParseTuple(args, "OO", &ograph, &oplanes) || !door_ready(d)
        || handle_ptr(d, ograph, KIND_GRAPH, &graph) < 0
        || seq_handles(d, oplanes, KIND_PLANE, &known.ptr, &known.n) < 0)
        return NULL;
    if (door_check(d, d->fn.eg_graph_describe(d->ctx, graph, &desc)) < 0) {
        PyMem_Free(known.ptr);
        return NULL;
    }
    PyObject* nodes = node_list(&known, desc.nodes, desc.n_nodes, desc.node_size);
    PyObject* edges = nodes ? edge_list(&known, desc.edges, desc.n_edges, desc.edge_size) : NULL;
    PyMem_Free(known.ptr);
    PyObject* out = NULL;
    if (edges)
        out = Py_BuildValue("{s:I,s:i,s:O,s:O}", "flags", desc.flags, "n_levels", desc.n_levels,
            "nodes", nodes, "edges", edges);
    Py_XDECREF(nodes);
    Py_XDECREF(edges);
    return out;
}

static PyObject* m_graph_launch(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject* ograph;
    void* graph;
    if (!PyArg_ParseTuple(args, "O", &ograph) || !door_ready(d)
        || handle_ptr(d, ograph, KIND_GRAPH, &graph) < 0)
        return NULL;
    if (door_check(d, d->fn.eg_graph_launch(d->ctx, graph)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

/* ---- device group -------------------------------------------------------------------------- */

static PyObject* m_device_count(PyObject* self, PyObject* args)
{
    Door* d       = (Door*)self;
    int32_t count = 0;
    if (!PyArg_ParseTuple(args, "") || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_device_count(d->ctx, &count)) < 0)
        return NULL;
    return PyLong_FromLong(count);
}

static PyObject* m_device_props(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    int device;
    struct eg_device_props p = { .struct_size = sizeof p };
    if (!PyArg_ParseTuple(args, "i", &device) || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_device_props(d->ctx, device, &p)) < 0)
        return NULL;
    return Py_BuildValue(
        "{s:i,s:i,s:i,s:i,s:i,s:i,s:L,s:L,s:s,s:i,s:i,s:i,s:d,s:d,s:d,s:d,s:d,s:d}", "cc_major",
        p.cc_major, "cc_minor", p.cc_minor, "sm_count", p.sm_count, "warp_size", p.warp_size,
        "regs_per_sm", p.regs_per_sm, "regs_per_block", p.regs_per_block, "shared_mem_per_block",
        (long long)p.shared_mem_per_block, "shared_mem_per_sm", (long long)p.shared_mem_per_sm,
        "name", p.name, "clock_rate_khz", p.clock_rate_khz, "memory_clock_rate_khz",
        p.memory_clock_rate_khz, "memory_bus_width_bits", p.memory_bus_width_bits,
        "peak_bytes_per_s", p.peak_bytes_per_s, "peak_flops_sp", p.peak_flops_sp, "peak_flops_dp",
        p.peak_flops_dp, "fp64_ratio", p.fp64_ratio, "ridge_flops_per_byte_sp",
        p.ridge_flops_per_byte_sp, "ridge_flops_per_byte_dp", p.ridge_flops_per_byte_dp);
}

static PyObject* m_stream_use(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    unsigned long long stream;
    if (!PyArg_ParseTuple(args, "K", &stream) || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_stream_use(d->ctx, stream)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

/* stream_new(device=None): the handle of a queue the library owns, on the device the text names. */
static PyObject* m_stream_new(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    const char* device = NULL;
    uint64_t stream = 0;
    if (!PyArg_ParseTuple(args, "|z", &device) || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_stream_new(d->ctx, device, &stream)) < 0)
        return NULL;
    return PyLong_FromUnsignedLongLong((unsigned long long)stream);
}

static PyObject* m_stream_free(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    unsigned long long stream;
    if (!PyArg_ParseTuple(args, "K", &stream) || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_stream_free(d->ctx, stream)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

static PyObject* m_stream_sync(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    unsigned long long stream;
    if (!PyArg_ParseTuple(args, "K", &stream) || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_stream_sync(d->ctx, stream)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

static PyObject* m_sync(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    if (!PyArg_ParseTuple(args, "") || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_sync(d->ctx)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

/* event_new(): a handle of an event that is not yet recorded. */
static PyObject* m_event_new(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    eg_event* event = NULL;
    if (!PyArg_ParseTuple(args, "") || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_event_new(d->ctx, &event)) < 0)
        return NULL;
    return handle_make(d, KIND_EVENT, event);
}

static PyObject* m_event_free(PyObject* self, PyObject* args)
{
    return handle_free(self, args, KIND_EVENT);
}

/* event_record(event) and event_wait(event): mark the stream in scope, or make it wait for the mark. */
static PyObject* event_call(PyObject* self, PyObject* args, int record)
{
    Door* d = (Door*)self;
    PyObject* o;
    void* event;
    if (!PyArg_ParseTuple(args, "O", &o) || !door_ready(d) || handle_ptr(d, o, KIND_EVENT, &event) < 0)
        return NULL;
    eg_status st = record ? d->fn.eg_event_record(d->ctx, event) : d->fn.eg_event_wait(d->ctx, event);
    if (door_check(d, st) < 0)
        return NULL;
    Py_RETURN_NONE;
}

static PyObject* m_event_record(PyObject* self, PyObject* args)
{
    return event_call(self, args, 1);
}

static PyObject* m_event_wait(PyObject* self, PyObject* args)
{
    return event_call(self, args, 0);
}

/* event_elapsed(start, stop): the milliseconds from the mark of start to the mark of stop. */
static PyObject* m_event_elapsed(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject *ostart, *ostop;
    void *start, *stop;
    double ms = 0.0;
    if (!PyArg_ParseTuple(args, "OO", &ostart, &ostop) || !door_ready(d)
        || handle_ptr(d, ostart, KIND_EVENT, &start) < 0 || handle_ptr(d, ostop, KIND_EVENT, &stop) < 0)
        return NULL;
    if (door_check(d, d->fn.eg_event_elapsed(d->ctx, start, stop, &ms)) < 0)
        return NULL;
    return PyFloat_FromDouble(ms);
}

/* ---- planes the door makes, banded cells, figures and reset -------------------------------- */

/* plane_new(shape, dtype_code, dtype_bits, kind=EG_MEM_HOST, flags=0, device=EG_DEVICE_DEFAULT): a plane
 * the door owns on the member `device` names; the dtype is the DLPack code (kDLInt 0, kDLUInt 1,
 * kDLFloat 2, kDLBool 6) and bit width. */
static PyObject* m_plane_new(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* names[] = { "shape", "dtype_code", "dtype_bits", "kind", "flags", "device", NULL };
    Door* d              = (Door*)self;
    PyObject* oshape;
    unsigned int code, bits, flags = 0;
    int kind = EG_MEM_HOST, device = EG_DEVICE_DEFAULT;
    if (!PyArg_ParseTupleAndKeywords(
            args, kw, "OII|iIi", names, &oshape, &code, &bits, &kind, &flags, &device)
        || !door_ready(d))
        return NULL;
    void* shape;
    Py_ssize_t ndim;
    if (seq_ints(oshape, sizeof(int64_t), &shape, &ndim) < 0)
        return NULL;
    struct eg_plane_desc desc = { .struct_size = sizeof desc,
        .flags                                 = flags,
        .kind                                  = kind,
        .ndim                                  = (int32_t)ndim,
        .dtype_code                            = code,
        .dtype_bits                            = bits,
        .shape                                 = shape,
        .device                                = device };
    eg_plane* plane           = NULL;
    PyObject* res             = NULL;
    if (door_check(d, d->fn.eg_plane_new(d->ctx, &desc, &plane)) == 0)
        res = handle_make(d, KIND_PLANE, plane);
    PyMem_Free(shape);
    return res;
}

/* Frees the tensor of a capsule nobody consumed (a consumer renames the capsule and owns it). */
static void dlpack_capsule_destructor(PyObject* capsule)
{
    if (!PyCapsule_IsValid(capsule, DLPACK_NAME))
        return;
    struct DLManagedTensorVersioned* tensor = PyCapsule_GetPointer(capsule, DLPACK_NAME);
    tensor->deleter(tensor);
}

/* plane_dlpack(plane): a `dltensor_versioned` capsule over the plane's memory. The tensor holds
 * its own reference, so the plane handle may be freed at once; its deleter runs once, from the
 * consumer or from the capsule if it is dropped unconsumed. */
static PyObject* m_plane_dlpack(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject* oplane;
    void* plane;
    if (!PyArg_ParseTuple(args, "O", &oplane) || !door_ready(d)
        || handle_ptr(d, oplane, KIND_PLANE, &plane) < 0)
        return NULL;
    struct DLManagedTensorVersioned* tensor = NULL;
    if (door_check(d, d->fn.eg_plane_dlpack(d->ctx, plane, &tensor)) < 0)
        return NULL;
    PyObject* capsule = PyCapsule_New(tensor, DLPACK_NAME, dlpack_capsule_destructor);
    if (!capsule)
        tensor->deleter(tensor);
    return capsule;
}

/* ---- host nodes ---------------------------------------------------------------------------- */

/* A host node's callables, handed to the door as the node's `user`. The door runs them on the
 * thread that called it, which holds the interpreter lock (this file never releases it), and
 * drops the node, so runs `release`, on any thread. Plain malloc memory: `release` may run
 * where PyMem is not usable until the lock is taken. */
typedef struct {
    PyObject* fn;   /* fn(views) */
    PyObject* vjp;  /* vjp(primals, adjoints_in, adjoints_out), or NULL */
    PyObject* name; /* the node's name, for the refusal of a stale view */
    Door* door;     /* where a failure of a callable is kept; not owned */
} HostNode;

/* A borrowed DLPack tensor of a plane, as the object a callable receives. It owns no memory and
 * is valid only while the callable runs: afterwards `tensor` is NULL and every use is refused. */
typedef struct {
    PyObject_HEAD struct DLManagedTensorVersioned* tensor;
    PyObject* node_name;
} HostView;

static PyTypeObject* host_view_type = NULL;

/* The deleter of the copy a `__dlpack__` call hands out: the copy is the whole allocation, and the
 * borrowed tensor it was made from is never touched. */
static void host_view_copy_free(struct DLManagedTensorVersioned* copy)
{
    free(copy);
}

/* The view as an exported DLPack 1.0 capsule: a copy of the borrowed tensor, shape and strides
 * included, that the consumer frees through its own deleter. */
static PyObject* HostView_dlpack(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* names[] = { "stream", "max_version", "dl_device", "copy", NULL };
    HostView* v          = (HostView*)self;
    PyObject *stream = Py_None, *max_version = Py_None, *dl_device = Py_None, *copy = Py_None;
    if (!PyArg_ParseTupleAndKeywords(
            args, kw, "|$OOOO", names, &stream, &max_version, &dl_device, &copy))
        return NULL;
    if (!v->tensor) {
        PyErr_Format(PyExc_ValueError,
            "host node %S: this view is valid only during the call of the node's function",
            v->node_name);
        return NULL;
    }
    int major = 0, minor = 0;
    if (max_version == Py_None || !PyArg_ParseTuple(max_version, "ii", &major, &minor)
        || major < 1) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_BufferError, "a host node view is exported as DLPack 1.0 only");
        return NULL;
    }
    const DLTensor* t = &v->tensor->dl_tensor;
    if (dl_device != Py_None) {
        int type, id;
        if (!PyArg_ParseTuple(dl_device, "ii", &type, &id))
            return NULL;
        if (type != (int)t->device.device_type || id != t->device.device_id) {
            PyErr_SetString(PyExc_BufferError, "a host node view cannot move to another device");
            return NULL;
        }
    }
    if (copy == Py_True) {
        PyErr_SetString(PyExc_BufferError, "a host node view is lent, never copied");
        return NULL;
    }
    size_t dims                          = (size_t)t->ndim * (t->strides ? 2 : 1);
    struct DLManagedTensorVersioned* out = malloc(sizeof *out + dims * sizeof(int64_t));
    if (!out)
        return PyErr_NoMemory();
    *out             = *v->tensor;
    out->manager_ctx = NULL;
    out->deleter     = host_view_copy_free;
    int64_t* axes    = (int64_t*)(out + 1);
    memcpy(axes, t->shape, (size_t)t->ndim * sizeof(int64_t));
    out->dl_tensor.shape = axes;
    if (t->strides) {
        memcpy(axes + t->ndim, t->strides, (size_t)t->ndim * sizeof(int64_t));
        out->dl_tensor.strides = axes + t->ndim;
    }
    PyObject* capsule = PyCapsule_New(out, DLPACK_NAME, dlpack_capsule_destructor);
    if (!capsule)
        free(out);
    (void)stream;
    return capsule;
}

static PyObject* HostView_device(PyObject* self, PyObject* unused)
{
    (void)unused;
    HostView* v = (HostView*)self;
    if (!v->tensor) {
        PyErr_Format(PyExc_ValueError,
            "host node %S: this view is valid only during the call of the node's function",
            v->node_name);
        return NULL;
    }
    return Py_BuildValue("(ii)", (int)v->tensor->dl_tensor.device.device_type,
        (int)v->tensor->dl_tensor.device.device_id);
}

static PyMethodDef host_view_methods[] = {
    { "__dlpack__", KEYWORD_CALL(HostView_dlpack), METH_VARARGS | METH_KEYWORDS, NULL },
    { "__dlpack_device__", HostView_device, METH_NOARGS, NULL },
    { NULL, NULL, 0, NULL },
};

static void HostView_dealloc(PyObject* o)
{
    PyTypeObject* t = Py_TYPE(o);
    Py_XDECREF(((HostView*)o)->node_name);
    ((freefunc)PyType_GetSlot(t, Py_tp_free))(o);
    Py_DECREF((PyObject*)t);
}

static PyType_Slot host_view_slots[] = { { Py_tp_dealloc, (void*)HostView_dealloc },
    { Py_tp_methods, host_view_methods }, { 0, 0 } };
static PyType_Spec host_view_spec    = { "_osprey.HostView", sizeof(HostView), 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DISALLOW_INSTANTIATION,
    host_view_slots };

/* The list of the views of `n` borrowed tensors (None for a NULL entry); every view made is also
 * appended to `made`, which the caller invalidates after the call. */
static PyObject* host_view_list(
    HostNode* node, int32_t n, struct DLManagedTensorVersioned* const* tensors, PyObject* made)
{
    PyObject* list = PyList_New(n);
    for (int32_t i = 0; list && i < n; i++) {
        PyObject* item = Py_None;
        Py_INCREF(item);
        if (tensors[i]) {
            Py_DECREF(item);
            HostView* v = (HostView*)PyType_GenericAlloc(host_view_type, 0);
            item        = (PyObject*)v;
            if (v) {
                v->tensor = tensors[i];
                Py_INCREF(node->name);
                v->node_name = node->name;
                if (PyList_Append(made, item) < 0)
                    Py_CLEAR(item);
            }
        }
        if (!item || PyList_SetItem(list, i, item) < 0)
            Py_CLEAR(list);
    }
    return list;
}

/* Calls `callable` with `lists` (each a list of n borrowed tensors, or NULL entries) as views,
 * under the interpreter lock; the views are invalidated afterwards. Zero when it returned; else 1,
 * with the exception it raised kept on the node's Door for the FAULT the door answers. */
static int32_t host_call_python(HostNode* node, PyObject* callable, int32_t n,
    struct DLManagedTensorVersioned* const* const* lists, int n_lists)
{
    PyGILState_STATE gil = PyGILState_Ensure();
    PyObject* made       = PyList_New(0);
    PyObject* call_args  = PyTuple_New(n_lists);
    for (int i = 0; made && call_args && i < n_lists; i++) {
        PyObject* list = host_view_list(node, n, lists[i], made);
        if (!list)
            Py_CLEAR(call_args);
        else
            PyTuple_SetItem(call_args, i, list);
    }
    PyObject* result = call_args ? PyObject_CallObject(callable, call_args) : NULL;
    for (Py_ssize_t i = 0; made && i < PyList_Size(made); i++)
        ((HostView*)PyList_GetItem(made, i))->tensor = NULL;
    Py_XDECREF(made);
    Py_XDECREF(call_args);
    int32_t status = 0;
    if (!result) {
        Door* d = node->door;
        Py_CLEAR(d->kept_type);
        Py_CLEAR(d->kept_value);
        Py_CLEAR(d->kept_traceback);
        /* PyErr_Fetch is deprecated from 3.12 and its replacement is not in the stable ABI of
         * 3.9, the floor of this extension; the triple is restored unchanged by door_raise. */
        PyErr_Fetch(&d->kept_type, &d->kept_value, &d->kept_traceback);
        status = 1;
    }
    Py_XDECREF(result);
    PyGILState_Release(gil);
    return status;
}

static int32_t host_fn_trampoline(
    void* user, int32_t n, struct DLManagedTensorVersioned* const* planes)
{
    HostNode* node = user;
    return host_call_python(node, node->fn, n, &planes, 1);
}

static int32_t host_vjp_trampoline(void* user, int32_t n,
    struct DLManagedTensorVersioned* const* primals,
    struct DLManagedTensorVersioned* const* adjoints_in,
    struct DLManagedTensorVersioned* const* adjoints_out)
{
    HostNode* node                                         = user;
    struct DLManagedTensorVersioned* const* const lists[3] = { primals, adjoints_in, adjoints_out };
    return host_call_python(node, node->vjp, n, lists, 3);
}

/* Drops the node's references to its callables, once, when the door drops its last reference to
 * the node: at a free, or at the end of the last graph that recorded it, on whatever thread. */
static void host_release_trampoline(void* user)
{
    HostNode* node       = user;
    PyGILState_STATE gil = PyGILState_Ensure();
    Py_DECREF(node->fn);
    Py_XDECREF(node->vjp);
    Py_DECREF(node->name);
    PyGILState_Release(gil);
    free(node);
}

/* host_node(name, fn, vjp=None): a node that runs the Python callable `fn(views)` where a launch
 * would run (`views`: a list with one view per plane declared at host_call), and, when `vjp` is
 * given, reverses through it as `vjp(primals, adjoints_in, adjoints_out)` (three such lists, an
 * entry None where the door gives none). A view is a DLPack producer valid only during the call. */
static PyObject* m_host_node(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* names[] = { "name", "fn", "vjp", NULL };
    Door* d              = (Door*)self;
    const char* text;
    PyObject *fn, *vjp = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "sO|O", names, &text, &fn, &vjp) || !door_ready(d))
        return NULL;
    if (!PyCallable_Check(fn) || (vjp != Py_None && !PyCallable_Check(vjp))) {
        PyErr_SetString(PyExc_TypeError, "host_node: fn and vjp must be callable");
        return NULL;
    }
    PyObject* name = PyUnicode_FromString(text);
    HostNode* node = name ? calloc(1, sizeof *node) : NULL;
    if (name && !node) {
        Py_DECREF(name);
        return PyErr_NoMemory();
    }
    if (!node)
        return NULL;
    node->fn                      = fn;
    node->vjp                     = vjp == Py_None ? NULL : vjp;
    node->name                    = name;
    node->door                    = d;
    struct eg_host_node_desc desc = { .struct_size = sizeof desc,
        .name                                      = text,
        .fn                                        = host_fn_trampoline,
        .vjp                                       = node->vjp ? host_vjp_trampoline : NULL,
        .user                                      = node,
        .release                                   = host_release_trampoline };
    eg_host_node* made            = NULL;
    if (door_check(d, d->fn.eg_host_node_new(d->ctx, &desc, &made)) < 0) {
        Py_DECREF(name);
        free(node);
        return NULL;
    }
    Py_INCREF(node->fn);
    Py_XINCREF(node->vjp);
    PyObject* handle = handle_make(d, KIND_HOST_NODE, made);
    if (!handle)
        d->fn.eg_host_node_free(d->ctx, made);
    return handle;
}

/* host_call(node, planes, access): runs the node now, or records it inside a recording; `access`
 * has one entry per plane: EG_ACCESS_READ, EG_ACCESS_WRITE, or both added. A callable that raised
 * leaves its own exception as the result. */
static PyObject* m_host_call(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject *onode, *oplanes, *oaccess;
    void* node;
    void** planes = NULL;
    void* access  = NULL;
    Py_ssize_t n_planes, n_access;
    PyObject* res = NULL;
    if (!PyArg_ParseTuple(args, "OOO", &onode, &oplanes, &oaccess) || !door_ready(d)
        || handle_ptr(d, onode, KIND_HOST_NODE, &node) < 0
        || seq_handles(d, oplanes, KIND_PLANE, &planes, &n_planes) < 0
        || seq_ints(oaccess, sizeof(uint32_t), &access, &n_access) < 0)
        goto done;
    if (n_planes != n_access) {
        PyErr_SetString(PyExc_ValueError, "host_call: planes and access differ in length");
        goto done;
    }
    if (door_check(d,
            d->fn.eg_host_call(
                d->ctx, node, (int32_t)n_planes, (eg_plane* const*)planes, (const uint32_t*)access))
        == 0) {
        Py_INCREF(Py_None);
        res = Py_None;
    }
done:
    PyMem_Free(planes);
    PyMem_Free(access);
    return res;
}

/* banded_pack(values): the 8-byte banded cell (an int) of each float64. */
static PyObject* m_banded_pack(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject* seq;
    if (!PyArg_ParseTuple(args, "O", &seq) || !door_ready(d))
        return NULL;
    PyObject* t = PySequence_Tuple(seq);
    if (!t)
        return NULL;
    Py_ssize_t n  = PyTuple_Size(t);
    double* in    = PyMem_Malloc((size_t)(n ? n : 1) * sizeof(double));
    uint64_t* out = PyMem_Malloc((size_t)(n ? n : 1) * sizeof(uint64_t));
    PyObject* res = NULL;
    if (!in || !out) {
        PyErr_NoMemory();
        goto done;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        in[i] = PyFloat_AsDouble(PyTuple_GetItem(t, i));
        if (in[i] == -1.0 && PyErr_Occurred())
            goto done;
    }
    if (door_check(d, d->fn.eg_banded_pack(d->ctx, in, out, (uint64_t)n)) < 0)
        goto done;
    res = PyTuple_New(n);
    for (Py_ssize_t i = 0; res && i < n; i++) {
        PyObject* cell = PyLong_FromUnsignedLongLong(out[i]);
        if (!cell) {
            Py_CLEAR(res);
            break;
        }
        PyTuple_SetItem(res, i, cell);
    }
done:
    PyMem_Free(in);
    PyMem_Free(out);
    Py_DECREF(t);
    return res;
}

/* banded_unpack(cells): the float64 of each banded cell. */
static PyObject* m_banded_unpack(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject* seq;
    if (!PyArg_ParseTuple(args, "O", &seq) || !door_ready(d))
        return NULL;
    PyObject* t = PySequence_Tuple(seq);
    if (!t)
        return NULL;
    Py_ssize_t n  = PyTuple_Size(t);
    uint64_t* in  = PyMem_Malloc((size_t)(n ? n : 1) * sizeof(uint64_t));
    double* out   = PyMem_Malloc((size_t)(n ? n : 1) * sizeof(double));
    PyObject* res = NULL;
    if (!in || !out) {
        PyErr_NoMemory();
        goto done;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        in[i] = PyLong_AsUnsignedLongLong(PyTuple_GetItem(t, i));
        if (in[i] == (uint64_t)-1 && PyErr_Occurred())
            goto done;
    }
    if (door_check(d, d->fn.eg_banded_unpack(d->ctx, in, out, (uint64_t)n)) < 0)
        goto done;
    res = PyTuple_New(n);
    for (Py_ssize_t i = 0; res && i < n; i++) {
        PyObject* value = PyFloat_FromDouble(out[i]);
        if (!value) {
            Py_CLEAR(res);
            break;
        }
        PyTuple_SetItem(res, i, value);
    }
done:
    PyMem_Free(in);
    PyMem_Free(out);
    Py_DECREF(t);
    return res;
}

static PyObject* m_banded_pack_scalar(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    double x;
    uint64_t cell = 0;
    if (!PyArg_ParseTuple(args, "d", &x) || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_banded_pack_scalar(d->ctx, x, &cell)) < 0)
        return NULL;
    return PyLong_FromUnsignedLongLong(cell);
}

/* plan_entry_figures(plan): what the driver reported for the plan's device function. */
static PyObject* m_plan_entry_figures(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject* oplan;
    void* plan;
    struct eg_entry_figures f = { .struct_size = sizeof f };
    if (!PyArg_ParseTuple(args, "O", &oplan) || !door_ready(d)
        || handle_ptr(d, oplan, KIND_PLAN, &plan) < 0)
        return NULL;
    if (door_check(d, d->fn.eg_plan_entry_figures(d->ctx, plan, &f)) < 0)
        return NULL;
    return Py_BuildValue("{s:I,s:i,s:i,s:i,s:i,s:i,s:i,s:i}", "flags", f.flags, "regs", f.regs,
        "max_threads", f.max_threads, "persist_regs", f.persist_regs, "persist_max_threads",
        f.persist_max_threads, "local_bytes", f.local_bytes, "shared_bytes", f.shared_bytes,
        "const_bytes", f.const_bytes);
}

/* Sets dict[key] to the integer `value`; -1 with a Python error set when that fails. */
static int dict_set_int64(PyObject* dict, const char* key, int64_t value)
{
    PyObject* number = PyLong_FromLongLong(value);
    if (!number)
        return -1;
    int status = PyDict_SetItemString(dict, key, number);
    Py_DECREF(number);
    return status;
}

/* plan_counters(plan): the plan's counters now: `finished` and `steps` (None when the unit has
 * none) and `mix`, the mix counters of every dispatch group that picks its branch at run time, in
 * the unit's dispatch order. */
static PyObject* m_plan_counters(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject* oplan;
    void* plan;
    struct eg_plan_counters c = { .struct_size = sizeof c };
    int32_t n_mix             = 0;
    if (!PyArg_ParseTuple(args, "O", &oplan) || !door_ready(d)
        || handle_ptr(d, oplan, KIND_PLAN, &plan) < 0)
        return NULL;
    if (door_check(d, d->fn.eg_plan_counters(d->ctx, plan, &c, NULL, 0, &n_mix)) < 0)
        return NULL;
    uint32_t* mix = n_mix > 0 ? PyMem_Malloc((size_t)n_mix * sizeof *mix) : NULL;
    if (n_mix > 0 && !mix)
        return PyErr_NoMemory();
    if (n_mix > 0
        && door_check(d, d->fn.eg_plan_counters(d->ctx, plan, &c, mix, n_mix, &n_mix)) < 0) {
        PyMem_Free(mix);
        return NULL;
    }
    PyObject* counts = PyTuple_New(n_mix);
    for (int32_t i = 0; counts && i < n_mix; ++i)
        PyTuple_SetItem(counts, i, PyLong_FromUnsignedLong(mix[i]));
    PyMem_Free(mix);
    if (!counts)
        return NULL;
    /* "O" takes its own reference of None; "N" hands over the one PyLong_FromLongLong made. */
    PyObject* res
        = Py_BuildValue("{s:O,s:O,s:N}", "finished", Py_None, "steps", Py_None, "mix", counts);
    if (res && c.finished >= 0 && dict_set_int64(res, "finished", c.finished) < 0)
        Py_CLEAR(res);
    if (res && c.steps >= 0 && dict_set_int64(res, "steps", c.steps) < 0)
        Py_CLEAR(res);
    return res;
}

/* runner_figures(runner): the mode and resolved launch of the runner's last run. */
static PyObject* m_runner_figures(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject* orunner;
    void* runner;
    struct eg_runner_figures f = { .struct_size = sizeof f };
    if (!PyArg_ParseTuple(args, "O", &orunner) || !door_ready(d)
        || handle_ptr(d, orunner, KIND_RUNNER, &runner) < 0)
        return NULL;
    if (door_check(d, d->fn.eg_runner_figures(d->ctx, runner, &f)) < 0)
        return NULL;
    return Py_BuildValue("{s:i,s:i,s:i,s:i,s:L,s:i,s:i}", "mode", f.mode, "block", f.block, "chunk",
        f.chunk, "steps_reverse", f.steps_reverse, "grid", (long long)f.grid, "reverse_passes",
        f.reverse_passes, "reverse_bitwise", f.reverse_bitwise);
}

static PyObject* m_runner_reset(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject* orunner;
    void* runner;
    if (!PyArg_ParseTuple(args, "O", &orunner) || !door_ready(d)
        || handle_ptr(d, orunner, KIND_RUNNER, &runner) < 0)
        return NULL;
    if (door_check(d, d->fn.eg_runner_reset(d->ctx, runner)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

/* ---- transforms ---------------------------------------------------------------------------- */

/* fft_new(kind, norm, dtype_bits, shape, axes, flags=0): a transform plan. Complex data is a float
 * plane whose last axis has extent 2; `shape` counts complex elements (the real side for R2C and
 * C2R) and never that trailing 2. The kinds, norms and flags are the EG_FFT_ constants. */
static PyObject* m_fft_new(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* names[] = { "kind", "norm", "dtype_bits", "shape", "axes", "flags", NULL };
    Door* d              = (Door*)self;
    int kind, norm;
    unsigned int bits, flags = 0;
    PyObject *oshape, *oaxes;
    if (!PyArg_ParseTupleAndKeywords(
            args, kw, "iiIOO|I", names, &kind, &norm, &bits, &oshape, &oaxes, &flags)
        || !door_ready(d))
        return NULL;
    void *shape, *axes;
    Py_ssize_t ndim, n_axes;
    if (seq_ints(oshape, sizeof(int64_t), &shape, &ndim) < 0)
        return NULL;
    if (seq_ints(oaxes, sizeof(int32_t), &axes, &n_axes) < 0) {
        PyMem_Free(shape);
        return NULL;
    }
    struct eg_fft_desc desc = { .struct_size = sizeof desc,
        .flags                               = flags,
        .kind                                = kind,
        .norm                                = norm,
        .dtype_bits                          = bits,
        .ndim                                = (int32_t)ndim,
        .shape                               = shape,
        .n_axes                              = (int32_t)n_axes,
        .axes                                = axes };
    eg_fft* plan            = NULL;
    PyObject* res           = NULL;
    if (door_check(d, d->fn.eg_fft_new(d->ctx, &desc, &plan)) == 0)
        res = handle_make(d, KIND_FFT, plan);
    PyMem_Free(shape);
    PyMem_Free(axes);
    return res;
}

/* fft_run(plan, source, out): runs the plan once, source -> out, both plane handles. */
static PyObject* m_fft_run(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject *oplan, *osource, *oout;
    void *plan, *source, *out;
    if (!PyArg_ParseTuple(args, "OOO", &oplan, &osource, &oout) || !door_ready(d)
        || handle_ptr(d, oplan, KIND_FFT, &plan) < 0
        || handle_ptr(d, osource, KIND_PLANE, &source) < 0
        || handle_ptr(d, oout, KIND_PLANE, &out) < 0)
        return NULL;
    if (door_check(d, d->fn.eg_fft_run(d->ctx, plan, source, out)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

/* fft_free(plan): drops the caller's reference; a recording that runs the plan keeps it. */
static PyObject* m_fft_free(PyObject* self, PyObject* args)
{
    return handle_free(self, args, KIND_FFT);
}

/* ---- matrix products ----------------------------------------------------------------------- */

/* The two extents (rows, columns) of a stored shape. */
static int gemm_extents(PyObject* shape, int64_t* rows, int64_t* cols)
{
    void* v;
    Py_ssize_t n;
    if (seq_ints(shape, sizeof(int64_t), &v, &n) < 0)
        return -1;
    if (n != 2) {
        PyMem_Free(v);
        PyErr_SetString(PyExc_ValueError, "a stored shape is (rows, columns)");
        return -1;
    }
    *rows = ((int64_t*)v)[0];
    *cols = ((int64_t*)v)[1];
    PyMem_Free(v);
    return 0;
}

/* gemm_new(a_shape, b_shape, c_shape, dtype_bits, flags=0): C = op(A) op(B) over the STORED shapes
 * (rows, columns) of three row-major matrices; flags are the EG_GEMM_ constants. */
static PyObject* m_gemm_new(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* names[] = { "a_shape", "b_shape", "c_shape", "dtype_bits", "flags", NULL };
    Door* d              = (Door*)self;
    PyObject *oa, *ob, *oc;
    unsigned int bits, flags = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOI|I", names, &oa, &ob, &oc, &bits, &flags)
        || !door_ready(d))
        return NULL;
    struct eg_gemm_desc desc = { .struct_size = sizeof desc, .flags = flags, .dtype_bits = bits };
    if (gemm_extents(oa, &desc.a_rows, &desc.a_cols) < 0
        || gemm_extents(ob, &desc.b_rows, &desc.b_cols) < 0
        || gemm_extents(oc, &desc.c_rows, &desc.c_cols) < 0)
        return NULL;
    eg_gemm* gemm = NULL;
    if (door_check(d, d->fn.eg_gemm_new(d->ctx, &desc, &gemm)) < 0)
        return NULL;
    return handle_make(d, KIND_GEMM, gemm);
}

/* gemm_run(gemm, a, b, c): runs the product once over three plane handles. */
static PyObject* m_gemm_run(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    PyObject *ogemm, *oa, *ob, *oc;
    void *gemm, *a, *b, *c;
    if (!PyArg_ParseTuple(args, "OOOO", &ogemm, &oa, &ob, &oc) || !door_ready(d)
        || handle_ptr(d, ogemm, KIND_GEMM, &gemm) < 0 || handle_ptr(d, oa, KIND_PLANE, &a) < 0
        || handle_ptr(d, ob, KIND_PLANE, &b) < 0 || handle_ptr(d, oc, KIND_PLANE, &c) < 0)
        return NULL;
    if (door_check(d, d->fn.eg_gemm_run(d->ctx, gemm, a, b, c)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

/* gemm_free(gemm): drops the caller's reference; a host recording that runs it keeps it. */
static PyObject* m_gemm_free(PyObject* self, PyObject* args)
{
    return handle_free(self, args, KIND_GEMM);
}

/* ---- the types and the module -------------------------------------------------------------- */


static PyMethodDef door_methods[] = {
    { "ctx_free", m_ctx_free, METH_VARARGS, NULL },
    { "error", m_error, METH_VARARGS, NULL },
    { "ctx_info", m_ctx_info, METH_VARARGS, NULL },
    { "host_threads", m_host_threads, METH_VARARGS, NULL },
    { "unit_load", KEYWORD_CALL(m_unit_load), METH_VARARGS | METH_KEYWORDS, NULL },
    { "unit_free", m_unit_free, METH_VARARGS, NULL },
    { "plane_wrap", KEYWORD_CALL(m_plane_wrap), METH_VARARGS | METH_KEYWORDS, NULL },
    { "plane_free", m_plane_free, METH_VARARGS, NULL },
    { "array_layout", m_array_layout, METH_VARARGS, NULL },
    { "plan_new", KEYWORD_CALL(m_plan_new), METH_VARARGS | METH_KEYWORDS, NULL },
    { "plan_free", m_plan_free, METH_VARARGS, NULL },
    { "bind", KEYWORD_CALL(m_bind), METH_VARARGS | METH_KEYWORDS, NULL },
    { "clear", m_clear, METH_VARARGS, NULL },
    { "set_count", m_set_count, METH_VARARGS, NULL },
    { "launch", m_launch, METH_VARARGS, NULL },
    { "runner_new", KEYWORD_CALL(m_runner_new), METH_VARARGS | METH_KEYWORDS, NULL },
    { "runner_free", m_runner_free, METH_VARARGS, NULL },
    { "run", m_run, METH_VARARGS, NULL },
    { "run_events", m_run_events, METH_VARARGS, NULL },
    { "reverse", m_reverse, METH_VARARGS, NULL },
    { "reduce", KEYWORD_CALL(m_reduce), METH_VARARGS | METH_KEYWORDS, NULL },
    { "scan", KEYWORD_CALL(m_scan), METH_VARARGS | METH_KEYWORDS, NULL },
    { "compact", m_compact, METH_VARARGS, NULL },
    { "order_by_key", m_order_by_key, METH_VARARGS, NULL },
    { "reorder", m_reorder, METH_VARARGS, NULL },
    { "copy", m_copy, METH_VARARGS, NULL },
    { "record_begin", m_record_begin, METH_VARARGS, NULL },
    { "record_abort", m_record_abort, METH_VARARGS, NULL },
    { "record_cond_begin", KEYWORD_CALL(m_record_cond_begin), METH_VARARGS | METH_KEYWORDS, NULL },
    { "record_cond_arm", m_record_cond_arm, METH_VARARGS, NULL },
    { "record_cond_end", m_record_cond_end, METH_VARARGS, NULL },
    { "record_fork_begin", m_record_fork_begin, METH_VARARGS, NULL },
    { "record_fork_branch", m_record_fork_branch, METH_VARARGS, NULL },
    { "record_fork_end", m_record_fork_end, METH_VARARGS, NULL },
    { "record_wait", m_record_wait, METH_VARARGS, NULL },
    { "record_end", m_record_end, METH_VARARGS, NULL },
    { "graph_launch", m_graph_launch, METH_VARARGS, NULL },
    { "graph_reverse", KEYWORD_CALL(m_graph_reverse), METH_VARARGS | METH_KEYWORDS, NULL },
    { "graph_describe", m_graph_describe, METH_VARARGS, NULL },
    { "host_node", KEYWORD_CALL(m_host_node), METH_VARARGS | METH_KEYWORDS, NULL },
    { "host_node_free", m_host_node_free, METH_VARARGS, NULL },
    { "host_call", m_host_call, METH_VARARGS, NULL },
    { "graph_free", m_graph_free, METH_VARARGS, NULL },
    { "device_count", m_device_count, METH_VARARGS, NULL },
    { "device_props", m_device_props, METH_VARARGS, NULL },
    { "stream_use", m_stream_use, METH_VARARGS, NULL },
    { "stream_new", m_stream_new, METH_VARARGS, NULL },
    { "stream_free", m_stream_free, METH_VARARGS, NULL },
    { "stream_sync", m_stream_sync, METH_VARARGS, NULL },
    { "event_new", m_event_new, METH_VARARGS, NULL },
    { "event_free", m_event_free, METH_VARARGS, NULL },
    { "event_record", m_event_record, METH_VARARGS, NULL },
    { "event_wait", m_event_wait, METH_VARARGS, NULL },
    { "event_elapsed", m_event_elapsed, METH_VARARGS, NULL },
    { "sync", m_sync, METH_VARARGS, NULL },
    { "plane_new", KEYWORD_CALL(m_plane_new), METH_VARARGS | METH_KEYWORDS, NULL },
    { "plane_dlpack", m_plane_dlpack, METH_VARARGS, NULL },
    { "banded_pack", m_banded_pack, METH_VARARGS, NULL },
    { "banded_unpack", m_banded_unpack, METH_VARARGS, NULL },
    { "banded_pack_scalar", m_banded_pack_scalar, METH_VARARGS, NULL },
    { "plan_entry_figures", m_plan_entry_figures, METH_VARARGS, NULL },
    { "plan_counters", m_plan_counters, METH_VARARGS, NULL },
    { "runner_reset", m_runner_reset, METH_VARARGS, NULL },
    { "runner_figures", m_runner_figures, METH_VARARGS, NULL },
    { "fft_new", KEYWORD_CALL(m_fft_new), METH_VARARGS | METH_KEYWORDS, NULL },
    { "fft_run", m_fft_run, METH_VARARGS, NULL },
    { "fft_free", m_fft_free, METH_VARARGS, NULL },
    { "gemm_new", KEYWORD_CALL(m_gemm_new), METH_VARARGS | METH_KEYWORDS, NULL },
    { "gemm_run", m_gemm_run, METH_VARARGS, NULL },
    { "gemm_free", m_gemm_free, METH_VARARGS, NULL },
    { NULL, NULL, 0, NULL },
};

/* What a handle is: its kind as a word, and whether it is still live. */
static PyObject* Handle_kind(PyObject* self, void* closure)
{
    (void)closure;
    return PyUnicode_FromString(kind_names[((Handle*)self)->kind]);
}

static PyObject* Handle_live(PyObject* self, void* closure)
{
    (void)closure;
    return PyBool_FromLong(((Handle*)self)->ptr != NULL);
}

static PyGetSetDef handle_getset[] = {
    { "kind", Handle_kind, NULL, NULL, NULL },
    { "live", Handle_live, NULL, NULL, NULL },
    { NULL, NULL, NULL, NULL, NULL },
};

/* A handle left alone releases nothing: the door frees it with its context. */
static void Handle_dealloc(PyObject* o)
{
    PyTypeObject* t = Py_TYPE(o);
    Py_DECREF((PyObject*)((Handle*)o)->door);
    ((freefunc)PyType_GetSlot(t, Py_tp_free))(o);
    Py_DECREF((PyObject*)t);
}

/* A handle is made by the door only (the flag that forbids instantiation is missing before 3.10).
 */
static PyObject* Handle_new(PyTypeObject* t, PyObject* a, PyObject* k)
{
    (void)t;
    (void)a;
    (void)k;
    PyErr_SetString(PyExc_TypeError, "cannot create '_osprey.Handle' instances");
    return NULL;
}

static PyType_Slot door_slots[]   = { { Py_tp_new, (void*)Door_new },
    { Py_tp_dealloc, (void*)Door_dealloc }, { Py_tp_methods, door_methods }, { 0, 0 } };
static PyType_Spec door_spec      = { "_osprey.Door", sizeof(Door), 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE, door_slots };
static PyType_Slot handle_slots[] = { { Py_tp_new, (void*)Handle_new },
    { Py_tp_dealloc, (void*)Handle_dealloc }, { Py_tp_getset, handle_getset }, { 0, 0 } };
static PyType_Spec handle_spec    = { "_osprey.Handle", sizeof(Handle), 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DISALLOW_INSTANTIATION,
    handle_slots };

/* The door's enums and flag bits, under the header's own names. */
#define DOOR_CONSTANTS(X)                                                                          \
    X(EG_STRUCTURE_HOST_TEAM)                                                                      \
    X(EG_STRUCTURE_HOST_SERIAL)                                                                    \
    X(EG_STRUCTURE_DEVICE)                                                                         \
    X(EG_REDUCE_COMPENSATED)                                                                       \
    X(EG_REDUCE_FIXED)                                                                             \
    X(EG_REDUCE_SUM)                                                                               \
    X(EG_REDUCE_MAX)                                                                               \
    X(EG_REDUCE_MIN)                                                                               \
    X(EG_REDUCE_ALL)                                                                               \
    X(EG_REDUCE_ANY)                                                                               \
    X(EG_SCAN_SUM)                                                                                 \
    X(EG_SCAN_MIN)                                                                                 \
    X(EG_SCAN_MAX)                                                                                 \
    X(EG_SCAN_EXCLUSIVE)                                                                           \
    X(EG_REDUCE_PAIR)                                                                              \
    X(EG_STEPS_REVERSE_PER_STEP)                                                                   \
    X(EG_STEPS_REVERSE_FUSED)                                                                      \
    X(EG_STEPS_REVERSE_AUTO)                                                                       \
    X(EG_EVENTS_STEP)                                                                              \
    X(EG_EVENTS_REFINE)                                                                            \
    X(EG_EVENT_PENDING)                                                                            \
    X(EG_EVENT_REFINED)                                                                            \
    X(EG_EVENT_AT_STEP)                                                                            \
    X(EG_EVENT_NONE)                                                                               \
    X(EG_EVENT_GRAZE)                                                                              \
    X(EG_EVENT_TWO)                                                                                \
    X(EG_NO_EVENT_REFUSE)                                                                          \
    X(EG_NO_EVENT_ZERO)                                                                            \
    X(EG_DERIVED_TAPE)                                                                             \
    X(EG_DERIVED_TAPE_FUSED)                                                                       \
    X(EG_DERIVED_STEP_VJP)                                                                         \
    X(EG_DERIVED_FUSED_VJP)                                                                        \
    X(EG_DERIVED_REFINE_ENTRY)                                                                     \
    X(EG_DERIVED_REFINE_NEWTON)                                                                    \
    X(EG_DERIVED_REFINE_VJP)                                                                       \
    X(EG_DERIVED_STATE_GUARD)                                                                      \
    X(EG_DERIVED_STATE_GUARD_VJP)                                                                  \
    X(EG_MODE_HOST)                                                                                \
    X(EG_MODE_BAND)                                                                                \
    X(EG_MODE_FUSED_ONE)                                                                           \
    X(EG_MODE_PERSIST)                                                                             \
    X(EG_COND_IF)                                                                                  \
    X(EG_COND_IF_ELSE)                                                                             \
    X(EG_COND_WHILE)                                                                               \
    X(EG_COND_SWITCH)                                                                              \
    X(EG_COND_NEGATE)                                                                              \
    X(EG_REVERSE_STRICT)                                                                           \
    X(EG_RECORD_DIFF)                                                                              \
    X(EG_GNODE_LAUNCH)                                                                             \
    X(EG_GNODE_RUN)                                                                                \
    X(EG_GNODE_REDUCE)                                                                             \
    X(EG_GNODE_COPY)                                                                               \
    X(EG_GNODE_FFT)                                                                                \
    X(EG_GNODE_GEMM)                                                                               \
    X(EG_GNODE_COND)                                                                               \
    X(EG_GNODE_ARM)                                                                                \
    X(EG_GNODE_FORK)                                                                               \
    X(EG_GNODE_BRANCH)                                                                             \
    X(EG_GNODE_JOIN)                                                                               \
    X(EG_GNODE_WAIT)                                                                               \
    X(EG_GNODE_SCAN)                                                                               \
    X(EG_GNODE_COMPACT)                                                                            \
    X(EG_GNODE_ORDER)                                                                              \
    X(EG_GNODE_REORDER)                                                                            \
    X(EG_GNODE_HOST)                                                                               \
    X(EG_ACCESS_READ)                                                                              \
    X(EG_ACCESS_WRITE)                                                                             \
    X(EG_GEDGE_RAW)                                                                                \
    X(EG_GEDGE_WAR)                                                                                \
    X(EG_GEDGE_WAW)                                                                                \
    X(EG_GEDGE_WAIT)                                                                               \
    X(EG_GEDGE_JOIN)                                                                               \
    X(EG_GEDGE_ORDER)                                                                              \
    X(EG_RUN_RECORD)                                                                               \
    X(EG_RUN_REORDER)                                                                              \
    X(EG_REPORT_DONE)                                                                              \
    X(EG_REPORT_EXACT_STEPS)                                                                       \
    X(EG_REPORT_SLOT_FILL)                                                                         \
    X(EG_CAP_HOST)                                                                                 \
    X(EG_CAP_DEVICE)                                                                               \
    X(EG_CAP_COND_IF_ELSE)                                                                         \
    X(EG_CAP_COND_SWITCH)                                                                          \
    X(EG_MEM_HOST)                                                                                 \
    X(EG_MEM_PINNED)                                                                               \
    X(EG_MEM_DEVICE)                                                                               \
    X(EG_PLANE_ZERO)                                                                               \
    X(EG_DEVICE_DEFAULT)                                                                           \
    X(EG_UNIT_CHECKED)                                                                             \
    X(EG_ENTRY_CHECKED)                                                                            \
    X(EG_ENTRY_DEVICE)                                                                             \
    X(EG_ENTRY_PERSIST)                                                                            \
    X(EG_FFT_C2C)                                                                                  \
    X(EG_FFT_R2C)                                                                                  \
    X(EG_FFT_C2R)                                                                                  \
    X(EG_FFT_NORM_BACKWARD)                                                                        \
    X(EG_FFT_NORM_FORWARD)                                                                         \
    X(EG_FFT_NORM_ORTHO)                                                                           \
    X(EG_FFT_INVERSE)                                                                              \
    X(EG_FFT_ADJOINT)                                                                              \
    X(EG_ENTRY_GEMM)                                                                               \
    X(EG_GEMM_TRANS_A)                                                                             \
    X(EG_GEMM_TRANS_B)                                                                             \
    X(EG_GEMM_TRANS_C)                                                                             \
    X(EG_GEMM_ACCUMULATE)

/* Adds the Door and Handle types, the door's constants and the digest width to the module; the
 * exception classes must be in it already (osprey_add_errors). */
int osprey_add_door(PyObject* module)
{
    static const char* const error_names[3] = { "NotWritten", "BadRequest", "Fault" };
    for (int i = 0; i < 3; i++) {
        door_errors[i] = PyObject_GetAttrString(module, error_names[i]);
        if (!door_errors[i])
            return -1;
    }
    PyObject* door   = PyType_FromSpec(&door_spec);
    PyObject* handle = PyType_FromSpec(&handle_spec);
    if (!door || !handle) {
        Py_XDECREF(door);
        Py_XDECREF(handle);
        return -1;
    }
    host_view_type = (PyTypeObject*)PyType_FromSpec(&host_view_spec);
    if (!host_view_type)
        return -1;
    handle_type = (PyTypeObject*)handle;
    Py_INCREF(handle);
    if (PyModule_AddObject(module, "Door", door) < 0
        || PyModule_AddObject(module, "Handle", handle) < 0)
        return -1;
#define X(name)                                                                                    \
    if (PyModule_AddIntConstant(module, #name, (long)(name)) < 0)                                  \
        return -1;
    DOOR_CONSTANTS(X)
#undef X
    return PyModule_AddIntConstant(module, "EG_DIGEST_BYTES", EG_DIGEST_BYTES);
}
