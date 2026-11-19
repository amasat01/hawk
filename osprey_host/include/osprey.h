/* Driving a backend library from a program that holds its own IR context. A backend library is
   opened by path and reached only through its one entry symbol, osprey_v1; nothing of the IR
   context on either side is shared with the other: a unit crosses as a node dump, value text and
   JSON text. Every call returns a status; the text of a failure is rx_last_error(ctx) of the
   context given, and starts with the stage it happened in: "open", "load", "canonical" or "render".

   The functions of this header belong to the loader and carry the prefix ob_ (not the IR door's
   rx_). */
#ifndef OSPREY_H
#define OSPREY_H
#include "raptor_ir.h"
#ifdef __cplusplus
extern "C" {
#endif

/* The abi number of the entry table, the one source of it: the library reads it from this file when
   it is built. Rule: a table whose shape (fields or their signatures) changes moves this number.
   ob_open refuses any other number by name. */
#define OB_ABI 11

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
    int (*facts)(
        rx_ctx*, const char*, const char*, const char*, const char*, char*, size_t, size_t*);
    int (*lower)(rx_ctx*, const char*, size_t, char*, size_t, size_t*);
    int (*set_text)(rx_ctx*, const char*, size_t, char*, size_t, size_t*);
    int (*card)(rx_ctx*, const char*, const char*, char*, size_t, size_t*);
    int (*machine)(rx_ctx*, const char*, char*, size_t, size_t*);
    int (*build)(rx_ctx*, const char*, size_t, const char*, char*, size_t, size_t*);
    int (*link)(rx_ctx*, const char*, size_t, const char*, const char*, const char*, char*, size_t,
        size_t*);
    int (*state)(rx_ctx*, const char*, const char*, char*, size_t, size_t*);
    int (*merge)(rx_ctx*, const char*, size_t, char*, size_t, size_t*);
    const char* (*last_code)(const rx_ctx*);
} ob_table;

/* Opens the library at path and reads its entry table; a missing symbol or another abi number is
   refused by name ("open: abi number N, expected 9"), and so is another contract version
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
   reads from the directory facts_dir (a NUL-terminated path; NULL = the directory "facts" beside
   the library file; a path that is no directory is refused, naming it). The caller
   passes the answer back in a render's contract without reading it. The text and its length are
   written as ob_name does (RX_TRUNC when cap is too small, len filled in); a refusal says why in
   rx_last_error(ctx), starting with "facts". The library answers in a context of its own.
   stated is the machine the build is for when it is not the one running: ONE JSON record with the
   optional keys isa ("arm64" or "x86_64"), l1d_bytes, l2_bytes and cache_source (all three or
   none), host_profile (the profile asked for) and device_arch (for "cuda", for example "sm_61"); a
   key left out is probed or the card's default, any other key is refused by name. The optional key
   cards states whole data documents in place of the library's files for this answer: a JSON object
   from a document name (see ob_card) to the text of the document; the answer states a digest of the
   documents it was made under, which ob_lower checks against the request's own cards. */
int ob_facts(rx_ctx* ctx, ob_backend* lib,
    const char* facts_dir /* NULL = "facts" beside the library */, const char* target,
    const char* compiler, const char* stated /* NULL = the machine running */, char* buf,
    size_t cap, size_t* len);
/* Lowers one kernel: the library loads the dump (as ob_render takes it: every node of the kernel in
   handle order, with the dialects' declaration lines) into a fresh context of its own and answers
   the request (JSON; the sinks are node positions of the dump) with the units the kernel lowers to
   as JSON: per unit its name, the digest of its walk and its text per target with the typed meta.
   The answer is written into buf as ob_facts writes its text (RX_TRUNC when cap is too small, len
   filled in). A refusal says why in rx_last_error(ctx), starting with "lower" or the stage it
   happened in. The data files are the ones the library ships beside itself. */
int ob_lower(rx_ctx* ctx, ob_backend* lib, const char* dump, size_t n_dump, const char* dialects,
    const char* request, size_t n_request, char* buf, size_t cap, size_t* len);
/* The one C++ text of a set of units built into one library: the headers and the self-check exports
   block once, then the library record over all the units in the order given. units is JSON: {"v":
   1, "units": [<the meta object of each unit, as an answer of ob_lower carries it>, ...],
   "exports": [five layout sizes]} where "exports" is optional (without it the block states the
   sizes the headers it is compiled with give; nothing is read from the environment). A set is built
   from units that came with the block asked off (the request member exports_block false), so the
   block is in the library once. An empty list, two units of one name and a record that does not
   read are refused, saying why in rx_last_error(ctx). The text is written into buf as ob_facts
   writes its text (RX_TRUNC when cap is too small, len filled in). */
int ob_set_text(rx_ctx* ctx, ob_backend* lib, const char* units, size_t n_units, char* buf,
    size_t cap, size_t* len);
/* The text of a data document the library ships, by name: build.json (the host build card),
   cuda_build.json (the device build card), cuda.json (the device document: the fusion row and the
   select weights), arm.json and x86.json (the instruction set documents). A caller derives its own
   document from this text (change a number, keep the rest) and states the whole of it under its
   name in the cards of ob_facts and of the request of ob_lower. facts_dir and the text protocol are
   as for ob_facts; a name outside the five is refused by name in rx_last_error(ctx). */
int ob_card(rx_ctx* ctx, ob_backend* lib,
    const char* facts_dir /* NULL = "facts" beside the library */, const char* name, char* buf,
    size_t cap, size_t* len);
/* The tool side of the library. Every entry below is JSON in and out, written into buf as ob_facts
   writes its text (RX_TRUNC when cap is too small, len filled in). A refusal says why in
   rx_last_error(ctx), "<stage>: <kind>: <message>", and ob_last_code(lib) answers its kind: one of
   bad_request, tool_absent, over_budget, timed_out, out_of_memory, tool_failed, io. */

/* The machine the library's tools run on, for the build a caller makes. ask is JSON: {"target":
   <the library's target name, as its manifest row spells the request>, "options": {<the target's
   own options>}, "opt_level": <text>, "root": <cache root folder>?}. The answer is {"key":
   <opaque>, "stated": {...}, "tools_absent": null | "<reason>"}; the machine is always stated, and
   a missing tool is the reason, not another shape. stated.asked repeats the options and the level.
   A caller hands the answer, as given, to ob_lower (the request's "machine": {<target name>:
   answer}) and to ob_build (the request's "machine"). With a root the answer also states the names
   a unit must not take (the host library) and the verdicts of its witnesses. */
int ob_machine(rx_ctx* ctx, ob_backend* lib, const char* ask, char* buf, size_t cap, size_t* len);

/* Builds one unit through the library's tool and the store under root (a folder, created when
   missing). unit is JSON: {"name", "text", "machine": <the answer of ob_machine>, "defines":
   [...]?}. The library reads its own build card. The answer is {"artifact": <file name of the
   object>, "source": <file name of the text>, "format", "key", "hit", "seconds", "files": {<file
   name>: <path>}, "figures": {<entry name>: {...}}, "launch": {"local_bytes", "local_source",
   "local_limit"} or null, "report", "link_recipe": <opaque>}; a second call for the same unit on
   the same machine runs no tool and answers "hit": true. A tool the machine lacks is refused with
   the code tool_absent. */
int ob_build(rx_ctx* ctx, ob_backend* lib, const char* unit, size_t n_unit, const char* root,
    char* buf, size_t cap, size_t* len);

/* Links the library of a set: parts is a JSON list of the answers of ob_build, as given, set_text
   the text of the set (ob_set_text), name the library's file name. The answer is {"library":
   <path>, "key", "hit", "report"}. */
int ob_link(rx_ctx* ctx, ob_backend* lib, const char* parts, size_t n_parts, const char* set_text,
    const char* name, const char* root, char* buf, size_t cap, size_t* len);

/* op "stats" answers {"hits", "misses", "tool_runs"} of this process; "reset" answers {}. */
int ob_state(rx_ctx* ctx, ob_backend* lib, const char* op, const char* root, char* buf, size_t cap,
    size_t* len);

/* Merges the answers the libraries of one build gave for one kernel (a JSON list) into one answer.
 */
int ob_merge(rx_ctx* ctx, ob_backend* lib, const char* answers, size_t n_answers, char* buf,
    size_t cap, size_t* len);

/* The manifest side. These three calls need no opened library: they say what a valid manifest is,
   so that a caller passes folder paths and raw target names and gets back checked facts. Each
   writes its text into buf as ob_facts writes its text (RX_TRUNC when cap is too small, len filled
   in); a refusal is RX_REFUSED and rx_last_error(ctx) is the whole sentence (no stage prefix),
   which names the folder or the target id and the rule. */

/* The canonical spelling of a target name: lowercase, whitespace, dashes and underscores removed
   ("Cu-Da" is "cuda"); the result must be a letter followed by letters and digits, else the name is
   refused. */
int ob_target_id(rx_ctx* ctx, const char* name, char* buf, size_t cap, size_t* len);

/* The roster of the OSPREY folders of a search. folders is a JSON list of paths; each holds a
   manifest osprey.json (members name, version, door, targets; every target row states request,
   target, about, device, entry and format). The answer is {"ospreys":
   [{"folder", "name", "version", "door", "targets": [<rows, request in its canonical spelling>]},
   ...], "refused": {<folder>: <sentence>}, "searched": [<the folders given>]}: a folder whose
   manifest is refused is listed in "refused" and the others are kept; a folder whose OSPREY has the
   name of an earlier one is skipped. Two OSPREYs serving one canonical target id are refused by
   name. */
int ob_roster(rx_ctx* ctx, const char* folders, char* buf, size_t cap, size_t* len);

/* The OSPREY that serves a target, over a roster as ob_roster answered it: {"id": <canonical id>,
   "osprey": <its name>}. target is any spelling of the id; a name that is no target id, and an id
   nobody serves, are refused (the second names the served ids, the folders searched and every
   refused folder with its reason). */
int ob_serving(
    rx_ctx* ctx, const char* roster, const char* target, char* buf, size_t cap, size_t* len);

/* The code of the last refusal of ob_machine, ob_build, ob_link or ob_state on this library (the
   kind names above); empty when the last such call succeeded. */
const char* ob_last_code(ob_backend* lib);

/* Closes the library and frees the parts of its last render. */
int ob_close(rx_ctx* ctx, ob_backend* lib);

#ifdef __cplusplus
}
#endif
#endif
