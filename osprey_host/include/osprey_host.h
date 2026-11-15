/* Driving a backend library from a program that holds its own IR context. A backend library is
   opened by path and reached only through its one entry symbol, osprey_v1; nothing of the IR
   context on either side is shared with the other: a unit crosses as a node dump, value text and
   JSON text. Every call returns a status; the text of a failure is rx_last_error(ctx) of the
   context given, and starts with the stage it happened in: "open", "load", "canonical" or "render".

   The functions of this header belong to the loader and carry the prefix ob_ (not the IR door's
   rx_). */
#ifndef OSPREY_HOST_H
#define OSPREY_HOST_H
#include "raptor_ir.h"
#ifdef __cplusplus
extern "C" {
#endif

/* The abi number of the entry table, the one source of it: the library reads it from this file when
   it is built. Rule: a table whose shape (fields or their signatures) changes moves this number.
   ob_open refuses any other number by name. */
#define OB_ABI 6

/* The version of the typed contract the caller was built for (the JSON a render takes, with its
   closed schema). A library carries its own in the table; ob_open refuses any other version by
   name. It moves with CONTRACT_VERSION in the writers' contract/types.rs. */
#define OB_CONTRACT 1

typedef struct ob_backend ob_backend;

/* The four parts of a rendered unit: pointer and length of the text, the body inside it, the entry
   name and the meta (target, mode, vectorised, box) as JSON. The bytes are owned by the backend
   handle and stay valid until the next ob_render or ob_close on it. They are not NUL-terminated. */
typedef struct {
    const char* text;
    size_t text_len;
    const char* body;
    size_t body_len;
    const char* entry;
    size_t entry_len;
    const char* meta;
    size_t meta_len;
} ob_parts;

/* The entry table of a library, as osprey_v1 returns it. Which entries catch panics: ctx_new, load,
   canonical, declare and facts (the IR door's own guard) and render (catch_unwind in the library);
   ctx_free, last_error and identity cannot fail and rely on a valid context or buffer from the
   caller. render writes text, body, entry and meta one after the other into buf (no separators) and
   their lengths into lens; RX_TRUNC when cap is smaller than their sum, lens still filled. */
typedef struct {
    uint32_t abi;
    uint32_t contract;
    const char* name;
    rx_ctx* (*ctx_new)(void);
    void (*ctx_free)(rx_ctx*);
    const char* (*last_error)(const rx_ctx*);
    int (*load)(rx_ctx*, const char*, size_t, rx_node*, size_t, size_t*);
    int (*canonical)(
        rx_ctx*, const rx_node*, size_t, const char*, const char*, const char*, rx_walk*);
    int (*render)(rx_ctx*, rx_walk, const char*, size_t, char*, size_t, size_t[4]);
    int (*identity)(char*);
    int (*declare)(rx_ctx*, const char*, size_t, const char*);
    int (*facts)(rx_ctx*, const char*, const char*, const char*, char*, size_t, size_t*);
    int (*lower)(rx_ctx*, const char*, size_t, char*, size_t, size_t*);
} ob_table;

/* Opens the library at path and reads its entry table; a missing symbol or another abi number is
   refused by name ("open: abi number N, expected 6"), and so is another contract version
   ("open: contract version N, expected 1"). */
int ob_open(rx_ctx* ctx, const char* path, ob_backend** out);
/* The library's name (for example "host"). */
int ob_name(rx_ctx* ctx, ob_backend* lib, char* buf, size_t cap, size_t* len);
/* The library's identity: 64 hex digits. */
int ob_identity(rx_ctx* ctx, ob_backend* lib, char out65[65]);
/* Renders one unit in a fresh context of the library, once. dump holds every node of the unit (as
   rx_dump of all of them, in handle order), so a node reference of the unit is its position in the
   dump and the same number in the library's context. dialects holds one line per declaration: a
   manifest JSON line, then its first-party key text line. declared, force and exempt are the walk's
   inputs as rx_canonical takes them, sinks are node positions, contract the unit's contract JSON.
 */
int ob_render(rx_ctx* ctx, ob_backend* lib, const char* dump, size_t n_dump, const char* dialects,
    const char* declared, const char* force /* NULL = None */, const char* exempt,
    const int64_t* sinks, size_t nsinks, const char* contract, size_t n_contract, ob_parts* out);
/* The machine facts of a target ("host" or "cuda") for a compiler identity ("<path> || <first
   version line> || <digest>"), as the JSON of the contract's machine record: the cache sizes and
   instruction set come from a probe of the running machine, the rest from data files the library
   reads from the directory $OSPREY_FACTS_DIR, else "facts" beside the library file. The caller
   passes the answer back in a render's contract without reading it. The text and its length are
   written as ob_name does (RX_TRUNC when cap is too small, len filled in); a refusal says why in
   rx_last_error(ctx), starting with "facts". The library answers in a context of its own.
   stated is the machine the build is for when it is not the one running: ONE JSON record with the
   optional keys isa ("arm64" or "x86_64"), l1d_bytes, l2_bytes and cache_source (all three or
   none), host_profile (the profile asked for) and device_arch (for "cuda", for example "sm_61"); a
   key left out is probed or the card's default, any other key is refused by name. */
int ob_facts(rx_ctx* ctx, ob_backend* lib, const char* target, const char* compiler,
    const char* stated /* NULL = the machine running */, char* buf, size_t cap, size_t* len);
/* Lowers one kernel: the library loads the dump (as ob_render takes it: every node of the kernel in
   handle order, with the dialects' declaration lines) into a fresh context of its own and answers
   the request (JSON; the sinks are node positions of the dump) with the units the kernel lowers to
   as JSON: per unit its name, the digest of its walk and its text per target with the typed meta.
   The answer is written into buf as ob_facts writes its text (RX_TRUNC when cap is too small, len
   filled in). A refusal says why in rx_last_error(ctx), starting with "lower" or the stage it
   happened in. */
int ob_lower(rx_ctx* ctx, ob_backend* lib, const char* dump, size_t n_dump, const char* dialects,
    const char* request, size_t n_request, char* buf, size_t cap, size_t* len);
/* Closes the library and frees the parts of its last render. */
int ob_close(rx_ctx* ctx, ob_backend* lib);

#ifdef __cplusplus
}
#endif
#endif
