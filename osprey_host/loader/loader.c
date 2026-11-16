/* Opens a backend library by path and drives it through its one entry symbol, osprey_v1. Owns the
   C functions of include/osprey_host.h; holds no state between calls beyond the handle the caller
   keeps. */
/* strndup and the dl functions are POSIX: a strict C11 build on glibc hides them without this. */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osprey_host.h"

/* The library, its table, and the bytes of its last render (the parts point into them). */
struct ob_backend {
    void* handle;
    const ob_table* table;
    char* bytes;
};

/* Records "<stage>: <message>" as the last error of ctx and returns status. */
static int fail(rx_ctx* ctx, int status, const char* stage, const char* message)
{
    char text[512];
    int n       = snprintf(text, sizeof text, "%s: %s", stage, message ? message : "unknown");
    size_t used = n < 0 ? 0 : (size_t)n >= sizeof text ? sizeof text - 1 : (size_t)n;
    rx_set_error(ctx, text, used);
    return status;
}

/* Copies a message the library keeps for its own context into the caller's context, tagged with
   the stage; a status of OK with a message becomes a refusal. */
static int forward(rx_ctx* ctx, const ob_table* table, rx_ctx* inner, int status, const char* stage)
{
    return fail(ctx, status == RX_OK ? RX_REFUSED : status, stage, table->last_error(inner));
}

int ob_open(rx_ctx* ctx, const char* path, ob_backend** out)
{
    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle)
        return fail(ctx, RX_REFUSED, "open", dlerror());
    const ob_table* (*entry)(void) = (const ob_table* (*)(void))dlsym(handle, "osprey_v1");
    if (!entry) {
        dlclose(handle);
        return fail(ctx, RX_REFUSED, "open", "the library has no symbol osprey_v1");
    }
    const ob_table* table = entry();
    if (!table || table->abi != OB_ABI) {
        char text[64];
        snprintf(text, sizeof text, "abi number %u, expected %d", table ? table->abi : 0u, OB_ABI);
        dlclose(handle);
        return fail(ctx, RX_REFUSED, "open", text);
    }
    if (table->contract != OB_CONTRACT) {
        char text[64];
        snprintf(
            text, sizeof text, "contract version %u, expected %d", table->contract, OB_CONTRACT);
        dlclose(handle);
        return fail(ctx, RX_REFUSED, "open", text);
    }
    ob_backend* lib = calloc(1, sizeof *lib);
    if (!lib) {
        dlclose(handle);
        return fail(ctx, RX_REFUSED, "open", "out of memory");
    }
    lib->handle = handle;
    lib->table  = table;
    *out        = lib;
    return RX_OK;
}

int ob_name(rx_ctx* ctx, ob_backend* lib, char* buf, size_t cap, size_t* len)
{
    (void)ctx;
    *len = strlen(lib->table->name);
    if (cap < *len + 1)
        return RX_TRUNC;
    memcpy(buf, lib->table->name, *len + 1);
    return RX_OK;
}

int ob_identity(rx_ctx* ctx, ob_backend* lib, char out65[65])
{
    (void)ctx;
    return lib->table->identity(out65);
}

/* Declares each dialect of the text (a manifest line, then its first-party key line) in the
   library's context. */
static int declare_dialects(rx_ctx* ctx, const ob_table* table, rx_ctx* inner, const char* text)
{
    while (*text) {
        const char* manifest_end = strchr(text, '\n');
        if (!manifest_end)
            return fail(ctx, RX_BADARG, "load", "a dialect manifest line has no key line");
        const char* key     = manifest_end + 1;
        const char* key_end = strchr(key, '\n');
        if (!key_end)
            key_end = key + strlen(key);
        char* key_copy = strndup(key, (size_t)(key_end - key));
        if (!key_copy)
            return fail(ctx, RX_REFUSED, "load", "out of memory");
        int status = table->declare(inner, text, (size_t)(manifest_end - text), key_copy);
        free(key_copy);
        if (status != RX_OK)
            return forward(ctx, table, inner, status, "load");
        text = *key_end ? key_end + 1 : key_end;
    }
    return RX_OK;
}

/* Loads the dump into the library's fresh context; every node must keep its position as its
   handle. */
static int load_unit(rx_ctx* ctx, const ob_table* table, rx_ctx* inner, const char* dump,
    size_t n_dump, const char* dialects)
{
    int status = declare_dialects(ctx, table, inner, dialects);
    if (status != RX_OK)
        return status;
    size_t roots = 0, got = 0;
    for (const char* at = dump; (at = strstr(at, "\nroot|")) != NULL && at < dump + n_dump; at++) {
        roots++;
    }
    rx_node* handles = malloc((roots ? roots : 1) * sizeof *handles);
    if (!handles)
        return fail(ctx, RX_REFUSED, "load", "out of memory");
    status = table->load(inner, dump, n_dump, handles, roots, &got);
    if (status != RX_OK)
        status = forward(ctx, table, inner, status, "load");
    for (size_t i = 0; status == RX_OK && i < got; i++) {
        if (handles[i] != i) {
            status = fail(
                ctx, RX_REFUSED, "load", "a node's handle differs from its position in the dump");
        }
    }
    free(handles);
    return status;
}

/* Loads the unit into a fresh context of the library and asks for its walk. */
static int prepare_unit(rx_ctx* ctx, const ob_table* table, rx_ctx* inner, const char* dump,
    size_t n_dump, const char* dialects, const char* declared, const char* force,
    const char* exempt, const int64_t* sinks, size_t nsinks, rx_walk* walk)
{
    rx_node* walk_sinks = malloc((nsinks ? nsinks : 1) * sizeof *walk_sinks);
    if (!walk_sinks)
        return fail(ctx, RX_REFUSED, "load", "out of memory");
    for (size_t i = 0; i < nsinks; i++)
        walk_sinks[i] = (rx_node)sinks[i];
    int status = load_unit(ctx, table, inner, dump, n_dump, dialects);
    if (status == RX_OK) {
        status = table->canonical(inner, walk_sinks, nsinks, declared, force, exempt, walk);
        if (status != RX_OK)
            status = forward(ctx, table, inner, status, "canonical");
    }
    free(walk_sinks);
    return status;
}

/* Asks the library for the unit's parts into a buffer of its own that grows only when the parts
   are longer than it; on success *bytes holds them one after the other and lens their lengths. */
static int render_parts(rx_ctx* ctx, const ob_table* table, rx_ctx* inner, rx_walk walk,
    const char* contract, size_t n_contract, char** bytes, size_t lens[4])
{
    size_t cap = (size_t)1 << 20;
    int status = RX_OK;
    *bytes     = NULL;
    while (status == RX_OK) {
        char* grown = realloc(*bytes, cap);
        if (!grown)
            return fail(ctx, RX_REFUSED, "render", "out of memory");
        *bytes = grown;
        status = table->render(inner, walk, contract, n_contract, *bytes, cap, lens);
        if (status != RX_TRUNC)
            break;
        cap    = lens[0] + lens[1] + lens[2] + lens[3] + 1;
        status = RX_OK;
    }
    if (status != RX_OK)
        status = forward(ctx, table, inner, status, "render");
    return status;
}

int ob_render(rx_ctx* ctx, ob_backend* lib, const char* dump, size_t n_dump, const char* dialects,
    const char* declared, const char* force, const char* exempt, const int64_t* sinks,
    size_t nsinks, const char* contract, size_t n_contract, ob_parts* out)
{
    const ob_table* table = lib->table;
    free(lib->bytes);
    lib->bytes    = NULL;
    rx_ctx* inner = table->ctx_new();
    if (!inner)
        return fail(ctx, RX_REFUSED, "load", "the library cannot create a context");
    rx_walk walk   = 0;
    size_t lens[4] = { 0, 0, 0, 0 };
    int status     = prepare_unit(
        ctx, table, inner, dump, n_dump, dialects, declared, force, exempt, sinks, nsinks, &walk);
    if (status == RX_OK) {
        status = render_parts(ctx, table, inner, walk, contract, n_contract, &lib->bytes, lens);
    }
    table->ctx_free(inner);
    if (status != RX_OK) {
        free(lib->bytes);
        lib->bytes = NULL;
        return status;
    }
    out->text      = lib->bytes;
    out->text_len  = lens[0];
    out->body      = out->text + lens[0];
    out->body_len  = lens[1];
    out->entry     = out->body + lens[1];
    out->entry_len = lens[2];
    out->meta      = out->entry + lens[2];
    out->meta_len  = lens[3];
    return RX_OK;
}

int ob_facts(rx_ctx* ctx, ob_backend* lib, const char* facts_dir, const char* target,
    const char* compiler, const char* stated, char* buf, size_t cap, size_t* len)
{
    const ob_table* table = lib->table;
    rx_ctx* inner         = table->ctx_new();
    if (!inner)
        return fail(ctx, RX_REFUSED, "facts", "the library cannot create a context");
    int status = table->facts(inner, facts_dir, target, compiler, stated, buf, cap, len);
    if (status != RX_OK && status != RX_TRUNC)
        status = forward(ctx, table, inner, status, "facts");
    table->ctx_free(inner);
    return status;
}

int ob_lower(rx_ctx* ctx, ob_backend* lib, const char* facts_dir, const char* dump, size_t n_dump,
    const char* dialects, const char* request, size_t n_request, char* buf, size_t cap, size_t* len)
{
    const ob_table* table = lib->table;
    rx_ctx* inner         = table->ctx_new();
    if (!inner)
        return fail(ctx, RX_REFUSED, "lower", "the library cannot create a context");
    int status = load_unit(ctx, table, inner, dump, n_dump, dialects);
    if (status == RX_OK) {
        status = table->lower(inner, facts_dir, request, n_request, buf, cap, len);
        if (status != RX_OK && status != RX_TRUNC)
            status = forward(ctx, table, inner, status, "lower");
    }
    table->ctx_free(inner);
    return status;
}

int ob_close(rx_ctx* ctx, ob_backend* lib)
{
    (void)ctx;
    free(lib->bytes);
    dlclose(lib->handle);
    free(lib);
    return RX_OK;
}
