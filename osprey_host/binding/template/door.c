/* The runtime's C door (eagle-door/1, eg_door.h) as Python: one Door object per context and one
 * method per door function, named as the function without its eg_ prefix.
 *
 * The binding holds no policy. It converts arguments, calls the function, turns a status and the
 * door's reason text into an exception and keeps a handle's lifetime as the header states it:
 * the caller frees every handle (a context frees what is left), a plane's DLPack tensor belongs
 * to the door once a wrap succeeded, and a forked child may only free the context.
 *
 * The door library is opened by path when a Door is made and its functions are looked up by name,
 * so this extension links no runtime. The interpreter lock stays held during every call: the
 * deleter of a DLPack tensor, which the door runs on the calling thread, may need it. */
#include "module.h"

#include "eagle/door/eg_door.h"
#include "plugin/dlpack.h"
#include <dlfcn.h>
#include <string.h>

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
    X(eg_copy)                                                                                     \
    X(eg_record_begin)                                                                             \
    X(eg_record_cond_begin)                                                                        \
    X(eg_record_cond_arm)                                                                          \
    X(eg_record_cond_end)                                                                          \
    X(eg_record_end)                                                                               \
    X(eg_graph_launch)                                                                             \
    X(eg_graph_free)                                                                               \
    X(eg_device_count)                                                                             \
    X(eg_device_props)                                                                             \
    X(eg_stream_use)                                                                               \
    X(eg_sync)                                                                                     \
    X(eg_record_child)                                                                             \
    X(eg_plane_new)                                                                                \
    X(eg_plane_dlpack)                                                                             \
    X(eg_banded_pack)                                                                              \
    X(eg_banded_unpack)                                                                            \
    X(eg_banded_pack_scalar)                                                                       \
    X(eg_plan_entry_figures)                                                                       \
    X(eg_runner_reset)                                                                             \
    X(eg_runner_figures)                                                                           \
    X(eg_fft_new)                                                                                  \
    X(eg_fft_run)                                                                                  \
    X(eg_fft_free)                                                                                 \
    X(eg_gemm_new)                                                                                 \
    X(eg_gemm_run)                                                                                 \
    X(eg_gemm_free)

/* The door's functions of one opened library, typed by their declarations in the header. */
typedef struct {
#define X(name) __typeof__(&name) name;
    DOOR_CALLS(X)
#undef X
} DoorFunctions;

typedef struct {
    PyObject_HEAD DoorFunctions fn;
    eg_ctx* ctx; /* NULL once the context is freed */
} Door;

typedef enum {
    KIND_UNIT,
    KIND_PLANE,
    KIND_PLAN,
    KIND_RUNNER,
    KIND_GRAPH,
    KIND_FFT,
    KIND_GEMM
} Kind;
static const char* const kind_names[]
    = { "unit", "plane", "plan", "runner", "graph", "fft", "gemm" };

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

/* Raises the exception of a non-OK status, carrying the reason the door stored in the context. */
static void door_raise(Door* d, eg_status st)
{
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
    static char* names[] = { "library", "device", NULL };
    const char* path;
    PyObject* device = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "s|O", names, &path, &device))
        return NULL;
    Door* d = (Door*)PyType_GenericAlloc(type, 0);
    if (!d)
        return NULL;
    void* lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        PyErr_SetString(door_errors[EG_FAULT - 1], dlerror());
        goto fail;
    }
#define X(name)                                                                                    \
    d->fn.name = (__typeof__(&name))dlsym(lib, #name);                                             \
    if (!d->fn.name) {                                                                             \
        PyErr_Format(door_errors[EG_FAULT - 1], "%s: no function " #name, path);                   \
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
 * that frees host memory only, which is the one call the door allows there. The library is never
 * closed: the host worker team of the process may still run code in it. */
static void Door_dealloc(PyObject* o)
{
    PyTypeObject* t = Py_TYPE(o);
    Door* d         = (Door*)o;
    if (d->ctx)
        d->fn.eg_ctx_free(d->ctx);
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
    PyObject* producer = PyObject_GetAttrString(array, "__dlpack__");
    PyObject* none     = PyTuple_New(0);
    PyObject* options  = PyDict_New();
    PyObject* version  = Py_BuildValue("(ii)", 1, 0);
    PyObject *capsule = NULL, *res = NULL;
    if (!producer || !none || !options || !version
        || PyDict_SetItemString(options, "max_version", version) < 0
        || (stream != Py_None && PyDict_SetItemString(options, "stream", stream) < 0))
        goto done;
    capsule = PyObject_Call(producer, none, options);
    if (!capsule)
        goto done;
    if (!PyCapsule_IsValid(capsule, DLPACK_NAME)) {
        PyErr_SetString(
            PyExc_TypeError, "the array gave no DLPack 1.0 capsule (dltensor_versioned)");
        goto done;
    }
    struct DLManagedTensorVersioned* tensor = PyCapsule_GetPointer(capsule, DLPACK_NAME);
    eg_plane* plane                         = NULL;
    if (door_check(d, d->fn.eg_plane_wrap(d->ctx, tensor, &plane)) < 0)
        goto done;
    PyCapsule_SetName(capsule, DLPACK_USED);
    res = handle_make(d, KIND_PLANE, plane);
done:
    Py_XDECREF(capsule);
    Py_XDECREF(version);
    Py_XDECREF(options);
    Py_XDECREF(none);
    Py_XDECREF(producer);
    return res;
}

/* ---- plans and launches -------------------------------------------------------------------- */

/* plan_new(unit, structure, policy, n_samples, part_base=None, part_count=None). */
static PyObject* m_plan_new(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* names[]
        = { "unit", "structure", "policy", "n_samples", "part_base", "part_count", NULL };
    Door* d = (Door*)self;
    PyObject *ounit, *obase = NULL, *ocount = NULL;
    int structure, policy;
    long long n_samples;
    if (!PyArg_ParseTupleAndKeywords(
            args, kw, "OiiL|OO", names, &ounit, &structure, &policy, &n_samples, &obase, &ocount)
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
    } else {
        struct eg_plan_desc desc = { .struct_size = sizeof desc,
            .unit                                 = unit,
            .structure                            = structure,
            .policy                               = policy,
            .n_samples                            = n_samples,
            .n_partitions                         = n_base,
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
 * step_graph=None): the fields of eg_runner_desc, a field left out being its zero. */
static PyObject* m_runner_new(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* names[] = { "plans", "max_steps", "flags", "derived_roles", "derived_units",
        "every", "window", "memory_budget", "reorder_theta", "steps_reverse", "events",
        "on_no_event", "step_graph", NULL };
    Door* d              = (Door*)self;
    PyObject *oplans, *oroles = NULL, *ounits = NULL, *ograph = NULL;
    unsigned int flags = 0;
    long long max_steps, every = 0, window = 0;
    unsigned long long budget = 0;
    double theta              = 0.0;
    int steps_reverse = 0, events = 0, on_no_event = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OL|IOOLLKdiiiO", names, &oplans, &max_steps, &flags,
            &oroles, &ounits, &every, &window, &budget, &theta, &steps_reverse, &events,
            &on_no_event, &ograph)
        || !door_ready(d))
        return NULL;
    void **plans = NULL, **units = NULL, *graph = NULL;
    void* roles = NULL;
    Py_ssize_t n_plans, n_units, n_roles;
    PyObject* res = NULL;
    if (seq_handles(d, oplans, KIND_PLAN, &plans, &n_plans) < 0
        || seq_handles(d, ounits, KIND_UNIT, &units, &n_units) < 0
        || seq_ints(oroles, sizeof(int32_t), &roles, &n_roles) < 0
        || handle_ptr_or_none(d, ograph, KIND_GRAPH, &graph) < 0)
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
        .step_graph                             = graph };
    eg_runner* runner          = NULL;
    if (door_check(d, d->fn.eg_runner_new(d->ctx, &desc, &runner)) == 0)
        res = handle_make(d, KIND_RUNNER, runner);
done:
    PyMem_Free(plans);
    PyMem_Free(units);
    PyMem_Free(roles);
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

/* reduce(source, out, op, policy, comp=None, flags=0). */
static PyObject* m_reduce(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* names[] = { "source", "out", "op", "policy", "comp", "flags", NULL };
    Door* d              = (Door*)self;
    PyObject *osource, *oout, *ocomp = Py_None;
    int op, policy;
    unsigned int flags = 0;
    if (!PyArg_ParseTupleAndKeywords(
            args, kw, "OOii|OI", names, &osource, &oout, &op, &policy, &ocomp, &flags)
        || !door_ready(d))
        return NULL;
    void *source, *out, *comp;
    if (handle_ptr(d, osource, KIND_PLANE, &source) < 0 || handle_ptr(d, oout, KIND_PLANE, &out) < 0
        || handle_ptr_or_none(d, ocomp, KIND_PLANE, &comp) < 0)
        return NULL;
    struct eg_reduce_desc desc = { .struct_size = sizeof desc,
        .flags                                  = flags,
        .in                                     = source,
        .out                                    = out,
        .comp                                   = comp,
        .op                                     = op,
        .policy                                 = policy };
    if (door_check(d, d->fn.eg_reduce(d->ctx, &desc)) < 0)
        return NULL;
    Py_RETURN_NONE;
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

static PyObject* m_record_begin(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    if (!PyArg_ParseTuple(args, "") || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_record_begin(d->ctx)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

/* record_cond_begin(kind, cell, n_arms=0): opens a conditional on the int32 cell plane. */
static PyObject* m_record_cond_begin(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* names[] = { "kind", "cell", "n_arms", NULL };
    Door* d              = (Door*)self;
    PyObject* ocell;
    int kind, n_arms = 0;
    void* cell;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "iO|i", names, &kind, &ocell, &n_arms)
        || !door_ready(d) || handle_ptr(d, ocell, KIND_PLANE, &cell) < 0)
        return NULL;
    struct eg_cond_desc desc
        = { .struct_size = sizeof desc, .kind = kind, .n_arms = n_arms, .cell = cell };
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

/* record_end(): the graph that replays the recording. */
static PyObject* m_record_end(PyObject* self, PyObject* args)
{
    Door* d         = (Door*)self;
    eg_graph* graph = NULL;
    if (!PyArg_ParseTuple(args, "") || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_record_end(d->ctx, &graph)) < 0)
        return NULL;
    return handle_make(d, KIND_GRAPH, graph);
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
    return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i,s:L,s:L}", "cc_major", p.cc_major, "cc_minor",
        p.cc_minor, "sm_count", p.sm_count, "warp_size", p.warp_size, "regs_per_sm", p.regs_per_sm,
        "regs_per_block", p.regs_per_block, "shared_mem_per_block",
        (long long)p.shared_mem_per_block, "shared_mem_per_sm", (long long)p.shared_mem_per_sm);
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

static PyObject* m_sync(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    if (!PyArg_ParseTuple(args, "") || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_sync(d->ctx)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

static PyObject* m_record_child(PyObject* self, PyObject* args)
{
    Door* d = (Door*)self;
    unsigned long long graph;
    if (!PyArg_ParseTuple(args, "K", &graph) || !door_ready(d))
        return NULL;
    if (door_check(d, d->fn.eg_record_child(d->ctx, graph)) < 0)
        return NULL;
    Py_RETURN_NONE;
}

/* ---- planes the door makes, banded cells, figures and reset -------------------------------- */

/* plane_new(shape, dtype_code, dtype_bits, kind=EG_MEM_HOST, flags=0): a plane the door owns;
 * the dtype is the DLPack code (kDLInt 0, kDLUInt 1, kDLFloat 2, kDLBool 6) and bit width. */
static PyObject* m_plane_new(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* names[] = { "shape", "dtype_code", "dtype_bits", "kind", "flags", NULL };
    Door* d              = (Door*)self;
    PyObject* oshape;
    unsigned int code, bits, flags = 0;
    int kind = EG_MEM_HOST;
    if (!PyArg_ParseTupleAndKeywords(
            args, kw, "OII|iI", names, &oshape, &code, &bits, &kind, &flags)
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
        .shape                                 = shape };
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
    return Py_BuildValue("{s:i,s:i,s:i,s:i,s:L}", "mode", f.mode, "block", f.block, "chunk",
        f.chunk, "steps_reverse", f.steps_reverse, "grid", (long long)f.grid);
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

#define KEYWORD_CALL(f) ((PyCFunction)(void (*)(void))(f))

static PyMethodDef door_methods[] = {
    { "ctx_free", m_ctx_free, METH_VARARGS, NULL },
    { "error", m_error, METH_VARARGS, NULL },
    { "ctx_info", m_ctx_info, METH_VARARGS, NULL },
    { "host_threads", m_host_threads, METH_VARARGS, NULL },
    { "unit_load", KEYWORD_CALL(m_unit_load), METH_VARARGS | METH_KEYWORDS, NULL },
    { "unit_free", m_unit_free, METH_VARARGS, NULL },
    { "plane_wrap", KEYWORD_CALL(m_plane_wrap), METH_VARARGS | METH_KEYWORDS, NULL },
    { "plane_free", m_plane_free, METH_VARARGS, NULL },
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
    { "copy", m_copy, METH_VARARGS, NULL },
    { "record_begin", m_record_begin, METH_VARARGS, NULL },
    { "record_cond_begin", KEYWORD_CALL(m_record_cond_begin), METH_VARARGS | METH_KEYWORDS, NULL },
    { "record_cond_arm", m_record_cond_arm, METH_VARARGS, NULL },
    { "record_cond_end", m_record_cond_end, METH_VARARGS, NULL },
    { "record_end", m_record_end, METH_VARARGS, NULL },
    { "graph_launch", m_graph_launch, METH_VARARGS, NULL },
    { "graph_free", m_graph_free, METH_VARARGS, NULL },
    { "device_count", m_device_count, METH_VARARGS, NULL },
    { "device_props", m_device_props, METH_VARARGS, NULL },
    { "stream_use", m_stream_use, METH_VARARGS, NULL },
    { "sync", m_sync, METH_VARARGS, NULL },
    { "record_child", m_record_child, METH_VARARGS, NULL },
    { "plane_new", KEYWORD_CALL(m_plane_new), METH_VARARGS | METH_KEYWORDS, NULL },
    { "plane_dlpack", m_plane_dlpack, METH_VARARGS, NULL },
    { "banded_pack", m_banded_pack, METH_VARARGS, NULL },
    { "banded_unpack", m_banded_unpack, METH_VARARGS, NULL },
    { "banded_pack_scalar", m_banded_pack_scalar, METH_VARARGS, NULL },
    { "plan_entry_figures", m_plan_entry_figures, METH_VARARGS, NULL },
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
