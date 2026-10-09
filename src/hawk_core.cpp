// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// hawk/_core — HAWK's HOST EXECUTION PATH in C++.
//
// WHAT LIVES HERE AND WHY IT IS C++. The host path is the one
// path the previous code generator classified HOT end to end — its Python
// equivalent framed itself as "compile -> cache -> dlopen -> ctypes-marshal -> call-once",
// 1,284 lines of Python, and EVERY launch pays the ctypes marshal even when the
// binding never changed. Tracing, the IR, autodiff and codegen are one-shot per
// kernel DEFINITION and stay in Python; what moves is exactly the part a
// propagation loop pays per step: `dlopen`/`dlsym` and the entry cache, the
// load-time `eagle_abi_tag` + `eagle_layout_sizes` self-check, the
// contiguous `void*[]` of by-value mirror PODs, the pointer->mirror marshal, and
// the SERIAL driver.
//
// THE CROSSING BUDGET IS THE DESIGN. Crossings are budgeted by
// LIFETIME, not by module. Per definition — `ArgBlock(descriptor)`,
// `HostLibrary(path)`, `.entry(name)` — any number, they happen once. Per bind —
// `rebind(slots, ptrs)` — exactly ONE, taking the whole changed set as two
// buffers rather than one Python call per slot, so the motivating consumer shape
// (a step loop whose caches are double-buffered, i.e. EVERY pointer changes every
// step) costs two crossings per launch and not N+1. Per launch — `run(argblock,
// base, count, nSamples)` — exactly one. Nothing inside `run` or `rebind` does a
// numpy or ctypes conversion, a dict lookup or a dtype check; the argument block
// is already the bytes the entry reads. Both shapes are counted.
//
// STAT_MANY IS NOT A LAUNCH PRIMITIVE. The
// AOT compile cache's warm re-verification (`hawk.compile.cache.ClosureWatch`)
// pays one `os.stat` per header its last compile reached — 317 on
// one measured shape — and profiling already showed that cost is Python call
// frames, not syscalls (`closure_unchanged`'s docstring: 317 stats of one file
// and 317 of 317 distinct files cost the same to within noise; batching by
// directory is a 10x LOSS). The lever left is doing those 317 `stat()` calls
// from C++ in ONE crossing instead of 317 Python-level ones. This is a
// PER-HIT cost, not a per-launch one — it runs once per warm compile-cache
// check, never from `run`, `bind` or `rebind` — so the per-launch budget is
// untouched, and `crossings()` counts it exactly like every other call so
// nobody has to take that on faith.
//
// THE ORACLE NEVER PARALLELISES. `run` calls the entry
// ONCE with the whole triple. That is not an omission to be optimised later: this
// path IS the serial reference every partitioned, tiled and ranked run is judged
// against, and an oracle that quietly threaded would be
// comparing a structure against itself. There is no `#pragma omp`, no <thread>,
// no pthread call and no OpenMP runtime in this file or in what it links — this is
// audited against both the source and the built object's symbols. Parallel host
// execution is eagle's `eagle::exec::HostTeam`, whose `run_serial` is deliberately
// the same shape as this one.
//
// WHY IT INCLUDES EAGLE'S HEADER. `GRefMirror`, `ScalarHandle`, `IntHandle`,
// `PartitionTriple` and `EAGLE_ABI_INDEX_T` are DEFINED in `plugin/gref_abi.h` and
// re-declaring them here would create a fourth hand-written copy of the layout —
// the very thing the "derived from its own build" forbids. The include ORDER
// below is mandatory: aether first, so `AETHER_INDEX_T` is visible when eagle's
// header binds `EAGLE_ABI_INDEX_T` to it; reversed, the index silently narrows to
// `uint32_t` and the whole parameter block shifts. This is a BUILD dependency on
// eagle, not a Python import, so the runtime severance is intact.

#include <aether/typedefs.h>     // FIRST: defines AETHER_INDEX_T
#include "plugin/gref_abi.h"     // SECOND: binds EAGLE_ABI_INDEX_T to it

#include "hawk_build_digest.h"   // generated at configure

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// ---------------------------------------------------------------------------
// The crossing counter (the instrument).
// ---------------------------------------------------------------------------
// A PLAIN `std::uint64_t`, deliberately not an atomic: an atomic here would be a
// lie about this module, which spawns no thread and holds the GIL through every
// call. It is incremented at the top of EVERY exported call — including
// `crossings()` itself, which is why the row that reads it measures the
// instrument's own cost (one crossing per read) instead of assuming it is free.
std::atomic<std::uint64_t> g_crossings{0};

// The documented FT-2 instrument: a counter that is DELIBERATELY plain, so a
// free-threading run can prove its schedule is adversarial enough to lose
// updates before it certifies anything else. Nothing real reads or writes it.
// The read-modify-write is split by a short busy gap so the lost-update window is
// wide enough to expose on every free-threaded build (a bare `++` loses ~4% on
// 3.13t, where interpreter overhead dominates the call).
volatile std::uint64_t g_unsynchronised = 0;

inline void unsynchronised_bump() {
    const std::uint64_t seen = g_unsynchronised;
    for (volatile int spin = 0; spin < 32; spin = spin + 1) {}
    g_unsynchronised = seen + 1;
}

inline void cross() { g_crossings.fetch_add(1, std::memory_order_relaxed); }

// The v2 host entry, exactly as `hawk/emit/host.py` emits it and as
// `eagle::exec::HostEntryV2` spells it: the packed role args, then the int64
// partition triple.
using HostEntryV2 = void (*)(void* const*, std::int64_t, std::int64_t, std::int64_t);

// ---------------------------------------------------------------------------
// Slot kinds — the pinned role -> by-value ABI shape map, as a wire vocabulary.
// ---------------------------------------------------------------------------
// The Python side (`hawk/runtime.py`) resolves a role to one of these names
// through the SAME map the emitter generates the entry signature from
// (`hawk.emit.aether.MIRROR_OF`), so a slot cannot be described one way to the
// compiler and another way to the marshal. An unknown name is refused here
// naming the vocabulary — never defaulted to a handle, which is how a 40-byte
// mirror ends up decoded as a 32-byte one (deterministic garbage, no crash).
enum class Kind {
    GRef,       // eagle::plugin::GRefMirror     (40 B) — out / vec_in / mat_in / vector-or-matrix mutable
    Handle,     // eagle::plugin::ScalarHandle   (32 B) — per_sample / lookup / terminated / wide_* / accum_out / scalar mutable
    IntHandle,  // eagle::plugin::IntHandle      (32 B) — an int-typed handle role
    F64,        // by value: double              — a uniform in a float64 kernel
    F32,        // by value: float               — a uniform in a float32 kernel
    Int,        // by value: long long           — an integer uniform (the emitter's `Int`)
    NSamples,   // by value: EAGLE_ABI_INDEX_T   — the nsamples ROLE, never the triple's int64
};

Kind kind_from_name(const std::string& name) {
    if (name == "gref")       return Kind::GRef;
    if (name == "handle")     return Kind::Handle;
    if (name == "int_handle") return Kind::IntHandle;
    if (name == "f64")        return Kind::F64;
    if (name == "f32")        return Kind::F32;
    if (name == "int")        return Kind::Int;
    if (name == "nsamples")   return Kind::NSamples;
    throw nb::value_error(
        ("hawk._core.ArgBlock: unknown slot kind '" + name +
         "'; the descriptor vocabulary is the map: 'gref' (40 B GRefMirror), "
         "'handle' / 'int_handle' (32 B), 'f64' / 'f32' / 'int' / 'nsamples' "
         "(by value)").c_str());
}

bool is_by_value(Kind k) {
    return k == Kind::F64 || k == Kind::F32 || k == Kind::Int || k == Kind::NSamples;
}

const char* kind_name(Kind k) {
    switch (k) {
        case Kind::GRef:      return "gref";
        case Kind::Handle:    return "handle";
        case Kind::IntHandle: return "int_handle";
        case Kind::F64:       return "f64";
        case Kind::F32:       return "f32";
        case Kind::Int:       return "int";
        case Kind::NSamples:  return "nsamples";
    }
    return "<unknown>";
}

// ---------------------------------------------------------------------------
// ArgBlock — the contiguous `void*[]` over C++-owned by-value mirror PODs.
// ---------------------------------------------------------------------------
// One storage cell per slot, each wide enough for the largest POD (40 B) and
// 8-byte aligned, so `params_[k]` is the address of the slot's OWN bytes — which
// is exactly what the emitted entry's `*static_cast<const GRefMirror*>(params[k])`
// reads. The cells are allocated ONCE in the constructor and never resized: a
// reallocation would leave every `params_` entry dangling, which is a class of
// bug that shows up as plausible numbers rather than a crash.
struct ArgBlock {
    struct Cell {
        alignas(8) unsigned char bytes[sizeof(eagle::plugin::GRefMirror)];
        Kind kind;
    };

    explicit ArgBlock(const std::vector<std::string>& descriptor) {
        cells_.resize(descriptor.size());
        params_.resize(descriptor.size(), nullptr);
        for (std::size_t k = 0; k < descriptor.size(); ++k) {
            cells_[k].kind = kind_from_name(descriptor[k]);
            std::memset(cells_[k].bytes, 0, sizeof(cells_[k].bytes));
            // A freshly-constructed mirror must be the POD's own default, not
            // zeroes: `ScalarHandle::stride` defaults to 1 and `deviceType`
            // to CUDA, and a zero stride would address every sample at offset 0.
            switch (cells_[k].kind) {
                case Kind::GRef:
                    new (cells_[k].bytes) eagle::plugin::GRefMirror{};
                    break;
                case Kind::Handle:
                    new (cells_[k].bytes) eagle::plugin::ScalarHandle{};
                    break;
                case Kind::IntHandle:
                    new (cells_[k].bytes) eagle::plugin::IntHandle{};
                    break;
                default:
                    break;   // a by-value cell holds a scalar; zero is its default
            }
            params_[k] = static_cast<void*>(cells_[k].bytes);
        }
    }

    std::size_t size() const { return cells_.size(); }

    void* const* params() const { return params_.data(); }

    // ONE slot's mirror, written in place (the per-bind, single-slot form).
    //
    // `stride` is the SAMPLE stride in elements. A GRef plane is `(width,
    // samples)` and its COMPONENT pitch is therefore `samples * stride` — the
    // contiguous rule eagle's own packer reads off an array's `.strides`
    // (`eagle.plan._gref_mirror_box`), stated here as the rule rather than read
    // from a numpy object, because nothing numpy may cross this boundary.
    void bind(std::size_t slot, std::uintptr_t ptr, std::uint64_t samples,
              std::uint64_t stride, std::int32_t device_type, std::int32_t device_id) {
        Cell& cell = at(slot);
        switch (cell.kind) {
            case Kind::GRef: {
                auto& m = *reinterpret_cast<eagle::plugin::GRefMirror*>(cell.bytes);
                m.data_        = reinterpret_cast<double*>(ptr);
                m.samples_     = samples;
                m.compStride_  = samples * stride;
                m.sampleStride_= stride;
                m.deviceType_  = device_type;
                m.deviceId_    = device_id;
                return;
            }
            case Kind::Handle: {
                auto& m = *reinterpret_cast<eagle::plugin::ScalarHandle*>(cell.bytes);
                m.data       = reinterpret_cast<void*>(ptr);
                m.samples    = samples;
                m.stride     = stride;
                m.deviceType = device_type;
                m.deviceId   = device_id;
                return;
            }
            case Kind::IntHandle: {
                auto& m = *reinterpret_cast<eagle::plugin::IntHandle*>(cell.bytes);
                m.data       = reinterpret_cast<void*>(ptr);
                m.samples    = samples;
                m.stride     = stride;
                m.deviceType = device_type;
                m.deviceId   = device_id;
                return;
            }
            default:
                // A by-value slot is bound by COPYING the bytes at `ptr`: the
                // caller's scalar is a temporary and keeping its address would
                // read freed memory at the first launch.
                copy_value(cell, ptr, slot);
                return;
        }
    }

    // The BULK rebind: the whole changed set in ONE crossing.
    void rebind(const std::int64_t* slots, const std::int64_t* ptrs, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            const std::int64_t s = slots[i];
            if (s < 0 || static_cast<std::size_t>(s) >= cells_.size())
                throw nb::index_error(
                    ("hawk._core.ArgBlock.rebind: slot " + std::to_string(s) +
                     " is outside this block's " + std::to_string(cells_.size()) +
                     " slots").c_str());
            Cell& cell = cells_[static_cast<std::size_t>(s)];
            const auto ptr = static_cast<std::uintptr_t>(ptrs[i]);
            switch (cell.kind) {
                case Kind::GRef:
                    reinterpret_cast<eagle::plugin::GRefMirror*>(cell.bytes)->data_ =
                        reinterpret_cast<double*>(ptr);
                    break;
                case Kind::Handle:
                    reinterpret_cast<eagle::plugin::ScalarHandle*>(cell.bytes)->data =
                        reinterpret_cast<void*>(ptr);
                    break;
                case Kind::IntHandle:
                    reinterpret_cast<eagle::plugin::IntHandle*>(cell.bytes)->data =
                        reinterpret_cast<void*>(ptr);
                    break;
                default:
                    // A by-value slot has no pointer to swap. Refused rather than
                    // silently reinterpreted: `rebind` is the POINTER path,
                    // and a uniform whose value changed is a `bind` (per bind),
                    // not a launch cost.
                    throw nb::value_error(
                        ("hawk._core.ArgBlock.rebind: slot " + std::to_string(s) +
                         " is a by-value '" + kind_name(cell.kind) + "' slot and "
                         "carries no pointer to rebind; set it with bind() "
                         "(rebind is the bulk POINTER path)").c_str());
            }
        }
    }

    Kind kind_of(std::size_t slot) { return at(slot).kind; }

private:
    Cell& at(std::size_t slot) {
        if (slot >= cells_.size())
            throw nb::index_error(
                ("hawk._core.ArgBlock: slot " + std::to_string(slot) +
                 " is outside this block's " + std::to_string(cells_.size()) +
                 " slots").c_str());
        return cells_[slot];
    }

    void copy_value(Cell& cell, std::uintptr_t ptr, std::size_t slot) {
        if (ptr == 0)
            throw nb::value_error(
                ("hawk._core.ArgBlock.bind: slot " + std::to_string(slot) +
                 " is a by-value slot and needs the ADDRESS of the value to "
                 "copy, got a null pointer").c_str());
        const void* src = reinterpret_cast<const void*>(ptr);
        switch (cell.kind) {
            case Kind::F64:      std::memcpy(cell.bytes, src, sizeof(double)); return;
            case Kind::F32:      std::memcpy(cell.bytes, src, sizeof(float)); return;
            case Kind::Int:      std::memcpy(cell.bytes, src, sizeof(long long)); return;
            case Kind::NSamples: std::memcpy(cell.bytes, src,
                                             sizeof(EAGLE_ABI_INDEX_T)); return;
            default: return;
        }
    }

    std::vector<Cell> cells_;
    std::vector<void*> params_;
};

// ---------------------------------------------------------------------------
// HostLibrary / HostEntry — dlopen, the / self-check, the entry cache.
// ---------------------------------------------------------------------------
// RTLD_LOCAL is deliberate and matches how eagle's own registry loads a plugin: a
// HAWK artifact is ONE self-contained TU, and a globally-visible load would let
// two artifacts' identically-named `eagle_abi_tag` / entry symbols interpose on
// each other — which would make the self-check certify the WRONG object.
struct Library {
    void* handle = nullptr;
    std::string path;

    ~Library() { if (handle) ::dlclose(handle); }
};

struct HostLibrary {
    explicit HostLibrary(const std::string& path) {
        lib_ = std::make_shared<Library>();
        lib_->path = path;
        ::dlerror();
        lib_->handle = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (lib_->handle == nullptr) {
            const char* err = ::dlerror();
            throw nb::value_error(
                ("hawk._core.HostLibrary: cannot dlopen '" + path + "': " +
                 std::string(err ? err : "(no dlerror)")).c_str());
        }
        const char* tag = symbol<const char>("eagle_abi_tag");
        const auto* sizes = symbol<const std::uint64_t>(
            eagle::plugin::kEagleLayoutSymbol);

        // The tag first, then the layout — the order eagle's own doors use, and
        // the order that produces the ACTIONABLE message: a v1 artifact is not a
        // wrong-layout v2 artifact, it is a different generation.
        abi_tag_ = std::string(tag);
        if (abi_tag_ != EAGLE_AETHER_ABI_V2)
            throw nb::value_error(
                ("hawk._core.HostLibrary: '" + path + "': exported eagle_abi_tag is '"
                 + abi_tag_ + "', this build speaks '" + EAGLE_AETHER_ABI_V2 +
                 "' (L9)").c_str());

        layout_.assign(sizes, sizes + eagle::plugin::kEagleLayoutFieldCount);
        try {
            // eagle's OWN comparison, so the refusal names the field in the one
            // spelling every other door already uses. Re-raised as a
            // ValueError: `std::runtime_error` would surface as a RuntimeError and
            // every existing HAWK/eagle caller catches ValueError on this path.
            eagle::plugin::check_layout_sizes(layout_.data(), "hawk._core: '" + path + "'");
        } catch (const std::runtime_error& exc) {
            throw nb::value_error(exc.what());
        }
    }

    // dlsym + CACHE (per kernel). The cache is what makes a second `entry()`
    // for the same name free; it is not a launch-path optimisation, because the
    // launch path never calls this at all.
    HostEntryV2 entry(const std::string& name) {
        auto it = entries_.find(name);
        if (it != entries_.end()) return it->second;
        auto fn = reinterpret_cast<HostEntryV2>(raw_symbol(name));
        entries_.emplace(name, fn);
        return fn;
    }

    const std::string& abi_tag() const { return abi_tag_; }
    const std::vector<std::uint64_t>& layout_sizes() const { return layout_; }
    const std::string& path() const { return lib_->path; }
    std::shared_ptr<Library> lib() const { return lib_; }

private:
    void* raw_symbol(const std::string& name) {
        ::dlerror();
        void* sym = ::dlsym(lib_->handle, name.c_str());
        const char* err = ::dlerror();
        if (err != nullptr || sym == nullptr)
            throw nb::value_error(
                ("hawk._core.HostLibrary: '" + lib_->path + "' exports no symbol '" +
                 name + "': " + std::string(err ? err : "(null address)")).c_str());
        return sym;
    }

    template <class T>
    const T* symbol(const std::string& name) { return static_cast<const T*>(raw_symbol(name)); }

    std::shared_ptr<Library> lib_;
    std::string abi_tag_;
    std::vector<std::uint64_t> layout_;
    std::unordered_map<std::string, HostEntryV2> entries_;
};

// One resolved entry point. It holds a share of the library so an entry can never
// outlive the object its address points into.
struct HostEntry {
    HostEntryV2 fn;
    std::string name;
    std::shared_ptr<Library> lib;

    // THE launch crossing. One call, one triple, the whole range — the
    // SERIAL oracle. No tiling, no thread, no schedule.
    void run(ArgBlock& args, std::int64_t base, std::int64_t count,
             std::int64_t n_samples) const {
        if (count <= 0) return;
        fn(args.params(), base, count, n_samples);
    }
};

std::vector<std::uint64_t> host_layout_sizes() {
    std::uint64_t out[eagle::plugin::kEagleLayoutFieldCount];
    eagle::plugin::expected_layout_sizes(out);
    return std::vector<std::uint64_t>(out, out + eagle::plugin::kEagleLayoutFieldCount);
}

// ---------------------------------------------------------------------------
// stat_many — the compile cache's batched, per-hit `stat()`.
// ---------------------------------------------------------------------------
// `paths` is read with the raw list C-API (`PyList_GET_ITEM`) rather than
// nanobind's `std::vector<std::string>` caster: the caller (`ClosureWatch`)
// pre-encodes its paths ONCE at construction into a plain Python `list` kept
// alive for the object's whole life, and a `std::vector<std::string>` caster
// would COPY every entry into a fresh vector on every one of the 317 x N
// warm-hit calls this exists to make cheap — the one Python-object identity
// this function is handed is the thing worth not throwing away.
//
// A `str` entry's UTF-8 bytes come from `PyUnicode_AsUTF8`, which CPython
// caches on the string object after the first call — so a `ClosureWatch` that
// keeps the same `str` objects across checks (it does: `_paths` is built once)
// pays that conversion at most once per path over the object's whole life, not
// once per check. `bytes` is accepted too (its buffer is already the raw
// bytes `stat()` wants), because ClosureWatch's docstring reserves either as
// the pre-encoded form and nothing here needs to prefer one.
//
// THE SENTINEL (the "missing or unreadable => miss", read at this layer).
// A path `stat()` cannot reach reports `size == -1` and `mtime_ns == 0` (the
// second field is then meaningless — the caller must not read it) instead of
// raising. `ClosureWatch.valid()` calls this ONCE over the WHOLE closure; an
// exception on the first deleted header among 317 would replace a cheap
// per-hit check with an exception-handling path over the other 316, which is
// exactly the cost class this function exists to remove.
std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>
stat_many_impl(const nb::list& paths) {
    const std::size_t n = paths.size();
    std::vector<std::int64_t> sizes(n);
    std::vector<std::int64_t> mtimes_ns(n);
    for (std::size_t i = 0; i < n; ++i) {
        PyObject* item = PyList_GET_ITEM(paths.ptr(), static_cast<Py_ssize_t>(i));
        const char* cpath = nullptr;
        if (PyBytes_Check(item)) {
            cpath = PyBytes_AS_STRING(item);
        } else if (PyUnicode_Check(item)) {
            cpath = PyUnicode_AsUTF8(item);
            if (cpath == nullptr)
                PyErr_Clear();   // an unencodable str: falls through to the sentinel
        } else {
            throw nb::type_error(
                "hawk._core.stat_many: every path must be str or bytes "
                "(ClosureWatch pre-encodes its own paths; this call should "
                "never see a fresh Python object of another type)");
        }
        struct ::stat st {};
        if (cpath != nullptr && ::stat(cpath, &st) == 0) {
            sizes[i] = static_cast<std::int64_t>(st.st_size);
            mtimes_ns[i] = static_cast<std::int64_t>(st.st_mtim.tv_sec) *
                          std::int64_t{1000000000} +
                          static_cast<std::int64_t>(st.st_mtim.tv_nsec);
        } else {
            sizes[i] = -1;         // sentinel: the "missing/unreadable => miss"
            mtimes_ns[i] = 0;      // meaningless whenever size == -1
        }
    }
    return {std::move(sizes), std::move(mtimes_ns)};
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "hawk's host execution path: dlopen + the "
              "self-check, the ArgBlock, and the SERIAL driver that is the oracle.";

    // -- the two import-time self-check readings ----------------------
    m.def("build_digest", [] {
        cross();
        return std::string(HAWK_BUILD_DIGEST);
    }, "The content digest of the `hawk/src` sources this binding was BUILT from "
       ". `hawk/__init__` recomputes it from the tree at import and refuses a "
       "disagreement, so a stale / wrong-arch / editable-shadowed binding fails at "
       "`import hawk` with a named field, never at the first launch.");

    m.def("layout_sizes", [] {
        cross();
        return host_layout_sizes();
    }, "This binding's OWN layout sizes, in the exported array's field order "
       "(GRefMirror, ScalarHandle, IntHandle, aether::idx_t, PartitionTriple). "
       "Derived from `sizeof` over eagle's PODs, never hand-written.");

    m.def("layout_field_name", [](std::size_t i) {
        cross();
        return std::string(eagle::plugin::layout_field_name(i));
    }, "i"_a, "The human name of layout field `i` — what a refusal says instead of "
             "an index.");

    m.def("build_info", [] {
        cross();
        std::map<std::string, std::string> info;
        info["build_digest"]     = HAWK_BUILD_DIGEST;
        info["compiler"]         = HAWK_COMPILER;
        info["compiler_id"]      = HAWK_COMPILER_ID;
        info["flags"]            = HAWK_COMPILE_FLAGS;
        info["aether_include"]   = HAWK_AETHER_INCLUDE;
        info["eagle_include"]    = HAWK_EAGLE_INCLUDE;
        info["abi_tag"]          = EAGLE_AETHER_ABI_V2;
        info["index_type_bytes"] = std::to_string(sizeof(EAGLE_ABI_INDEX_T));
        return info;
    }, "Compiler identity, flags and the two RESOLVED header roots this "
       "binding was built against. EXPORTED, not compared: a different compiler is "
       "not by itself a wrong binding, and a card that has to name the axis its arms "
       "differ on needs to be able to read it.");

    m.def("crossings", [] {
        cross();
        return g_crossings.load(std::memory_order_relaxed);
    }, "How many times the nanobind boundary has been crossed in this process "
       "(the instrument). Incremented at the top of EVERY exported call — this "
       "one included, so a reader measures the instrument rather than assuming it "
       "is free. A relaxed atomic: a count, never a fence.");

    // -- stat_many ------------------------------------------------
    m.def("_unsynchronised_bump", [] { unsynchronised_bump(); },
          "Free-threading canary: increments a deliberately PLAIN counter. "
          "Concurrent calls lose updates; a run that cannot observe the loss "
          "cannot certify the real counters. An instrument, not an API.");

    m.def("_unsynchronised_count", [] { return std::uint64_t{g_unsynchronised}; },
          "The canary counter `_unsynchronised_bump` increments.");

    m.def("stat_many", [](const nb::list& paths) {
        cross();
        return stat_many_impl(paths);
    }, "paths"_a,
    "`stat()` every entry of `paths` (each a `str` or `bytes`) in ONE crossing, "
    "returning `(sizes, mtimes_ns)` — parallel int64 arrays in `paths` order. "
    "This is the compile cache's PER-HIT instrument, not a launch primitive: "
    "`hawk.compile.cache.ClosureWatch.valid()`'s warm re-verification "
    "used to cost one Python `os.stat` call per reached header (317 on "
    "the A1 stage-4 closure); this collapses that into one binding crossing "
    "without changing what is checked — content still decides, this call reads "
    "only (size, mtime_ns), never bytes. the per-launch budget is untouched "
    "(nothing here is called from `run`, `bind` or `rebind`) and `crossings()` "
    "counts this call exactly like any other. A path that cannot be stat'd "
    "reports size -1 rather than raising, so one deleted header cannot turn a "
    "cheap per-hit check into an exception path over the other 316. Serial, "
    "like everything else in this module: 317 near-instant syscalls "
    "would spend more on thread synchronisation than the loop they replace.");

    // -- ArgBlock ------------------------------------------------------
    nb::class_<ArgBlock>(m, "ArgBlock",
        "The contiguous `void*[]` of by-value mirror PODs one kernel DEFINITION "
        "binds through. Built once from the slot descriptor `Walk.arg_spec` "
        "implies; after that a launch touches only `rebind` and `run`.")
        .def(nb::init<const std::vector<std::string>&>(), "descriptor"_a,
             "One kind name per slot, in `arg_spec` ORDER — the same order the "
             "emitted entry reads `params[k]` in.")
        .def("bind", [](ArgBlock& self, std::size_t slot, std::uintptr_t ptr,
                        std::uint64_t samples, std::uint64_t stride,
                        std::int32_t device_type, std::int32_t device_id) {
                cross();
                self.bind(slot, ptr, samples, stride, device_type, device_id);
             }, nb::lock_self(), "slot"_a, "ptr"_a, "samples"_a, "stride"_a = 1,
                "device_type"_a = eagle::plugin::kEagleAbiDeviceCPU, "device_id"_a = 0,
             "Write ONE slot's mirror in place (per bind, single slot). A by-value "
             "slot COPIES the bytes at `ptr`.")
        .def("rebind", [](ArgBlock& self,
                          nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig,
                                      nb::device::cpu> slots,
                          nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig,
                                      nb::device::cpu> ptrs) {
                cross();
                if (slots.shape(0) != ptrs.shape(0))
                    throw nb::value_error(
                        "hawk._core.ArgBlock.rebind: slots and ptrs must be the same "
                        "length (they are one changed set, taken in ONE crossing)");
                self.rebind(slots.data(), ptrs.data(),
                            static_cast<std::size_t>(slots.shape(0)));
             }, nb::lock_self(), "slots"_a, "ptrs"_a,
             "Write the WHOLE changed pointer set in ONE crossing. Both "
             "arguments are int64 buffers, never Python sequences of per-slot calls: "
             "the ODE shape, where every bound cache moves every step, must cost two "
             "crossings per launch and not N+1.")
        .def("kind_of", [](ArgBlock& self, std::size_t slot) {
                cross();
                return std::string(kind_name(self.kind_of(slot)));
             }, "slot"_a, "The kind name slot `slot` was described with.")
        .def_prop_ro("size", [](ArgBlock& self) { return self.size(); },
                     "How many slots this block carries.");

    // -- HostEntry -----------------------------------------------------
    nb::class_<HostEntry>(m, "HostEntry",
        "One resolved `extern \"C\"` v2 host entry. Holds a share of its library, so "
        "an entry can never outlive the object its address points into.")
        .def("run", [](const HostEntry& self, ArgBlock& args, std::int64_t base,
                       std::int64_t count, std::int64_t n_samples) {
                cross();
                self.run(args, base, count, n_samples);
             }, "argblock"_a.lock(), "base"_a, "count"_a, "n_samples"_a,
             "THE launch crossing: one call with the whole `[base, base+count)` "
             "range and the TRUE `n_samples`. SERIAL by construction — this is the "
             "reference oracle every partitioned/tiled/ranked run is compared against "
             "so it must never grow a schedule.")
        .def_ro("name", &HostEntry::name, "The exported symbol this entry resolved.");

    // -- HostLibrary ---------------------------------------------------
    nb::class_<HostLibrary>(m, "HostLibrary",
        "One dlopen'ed HAWK host artifact, self-checked at load: its OWN "
        "exported `eagle_abi_tag` and `eagle_layout_sizes` are read and compared "
        "against this build's, and a mismatch is refused NAMING THE FIELD — because "
        "the alternative is not a crash, it is deterministic garbage.")
        .def("__init__", [](HostLibrary* self, const std::string& path) {
                cross();
                new (self) HostLibrary(path);
             }, "path"_a)
        .def("entry", [](HostLibrary& self, const std::string& name) {
                cross();
                HostEntry e;
                e.fn = self.entry(name);
                e.name = name;
                e.lib = self.lib();
                return e;
             }, nb::lock_self(), "name"_a, nb::keep_alive<0, 1>(),
             "dlsym + cache (per kernel).")
        .def_prop_ro("abi_tag", [](HostLibrary& self) { return self.abi_tag(); },
                     "The tag this artifact exported — what the load-time check READ, "
                     "so a caller reports the checked value rather than re-reading it.")
        .def_prop_ro("layout_sizes",
                     [](HostLibrary& self) { return self.layout_sizes(); },
                     "The layout array this artifact exported, as read at load.")
        .def_prop_ro("path", [](HostLibrary& self) { return self.path(); });
}
