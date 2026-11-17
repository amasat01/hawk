//! The host backend library: one exported symbol, `osprey_v1`, returns the entry table. Loading and canonical forms go to the IR
//! core's door, rendering to the writer crate, and the identity is fixed at build time.
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
/// The wings crate answers the analyses the writers declare as external functions: naming it here links it, and a
/// function it lacks fails the build of the library (no symbol is left unresolved).
extern crate raptor_wings;
mod device_build;
pub mod facts;
mod host_build;
pub mod lower;
use aether_writers::refuse::{bad, fault, Refusal, RefusalKind};
use aether_writers::{host_render, RxCtx};
use raptor_ir::canon::rx_canonical;
use raptor_ir::dialect::rx_declare_dialect;
use raptor_ir::door::{
    put, rx_ctx_free, rx_ctx_new, rx_last_error, RX_BADARG, RX_OK, RX_PANIC, RX_REFUSED, RX_TRUNC,
};
use raptor_ir::dump::rx_load;
use raptor_ir::record::{Ctx, Fault};
use std::ffi::{c_char, c_int, CStr};

/// The table's abi number, read from include/osprey_host.h by the build script (the one source of it).
const ABI: u32 = include!(concat!(env!("OUT_DIR"), "/abi.rs"));
/// The library's identity: 64 hex digits, fixed at build time.
const IDENTITY: &str = include_str!(concat!(env!("OUT_DIR"), "/identity.txt"));

#[repr(C)]
/// The entry table of the library, as the header's `ob_table` declares it: the same fields in the same order.
pub struct OspreyV1 {
    /// The abi number of the table's shape.
    pub abi: u32,
    /// The version of the typed contract the library reads.
    pub contract: u32,
    /// The library's name.
    pub name: *const c_char,
    /// Makes a context with the wings' kinds declared.
    pub ctx_new: extern "C" fn() -> *mut Ctx,
    /// Frees a context.
    pub ctx_free: unsafe extern "C" fn(*mut Ctx),
    /// The text of a context's last refusal.
    pub last_error: unsafe extern "C" fn(*const Ctx) -> *const c_char,
    /// Loads a dump of nodes into a context.
    pub load:
        unsafe extern "C" fn(*mut Ctx, *const c_char, usize, *mut u32, usize, *mut usize) -> c_int,
    /// Takes the canonical walk over sinks.
    pub canonical: unsafe extern "C" fn(
        *mut Ctx,
        *const u32,
        usize,
        *const c_char,
        *const c_char,
        *const c_char,
        *mut u32,
    ) -> c_int,
    /// Renders one unit from a walk and the typed contract.
    pub render:
        extern "C" fn(*mut Ctx, u32, *const c_char, usize, *mut c_char, usize, *mut usize) -> c_int,
    /// Writes the library's identity into a 65-byte buffer.
    pub identity: extern "C" fn(*mut c_char) -> c_int,
    /// Declares the node classes of a dialect manifest in a context, so a dump that holds nodes of that dialect loads.
    pub declare: unsafe extern "C" fn(*mut Ctx, *const c_char, usize, *const c_char) -> c_int,
    /// Writes the machine facts for a target, a compiler identity and an optional stated machine (a JSON record, NULL for none),
    /// as the JSON of the contract's machine record, from the data files in the directory given (NULL: `facts` beside the library).
    pub facts: extern "C" fn(
        *mut Ctx,
        *const c_char,
        *const c_char,
        *const c_char,
        *const c_char,
        *mut c_char,
        usize,
        *mut usize,
    ) -> c_int,
    /// Lowers the kernel whose IR the context holds, from a request (JSON), into the answer (JSON): the units, their texts per
    /// target and their typed meta, from the data files in the directory given (NULL: `facts` beside the library). The caller-buffer protocol: RX_TRUNC with the length filled in when the buffer is too small.
    pub lower: extern "C" fn(
        *mut Ctx,
        *const c_char,
        *const c_char,
        usize,
        *mut c_char,
        usize,
        *mut usize,
    ) -> c_int,
    /// Writes the C++ text of a set of units built into one library (the exports block and the library record over the units) from
    /// a JSON list of their records, into the caller's buffer, as `lower` does.
    pub set_text:
        extern "C" fn(*mut Ctx, *const c_char, usize, *mut c_char, usize, *mut usize) -> c_int,
    /// Writes the text of the data document of a name that ships with the library (from the directory given, NULL: `facts` beside
    /// the library), as `facts` writes its answer.
    pub card: extern "C" fn(
        *mut Ctx,
        *const c_char,
        *const c_char,
        *mut c_char,
        usize,
        *mut usize,
    ) -> c_int,
}
unsafe impl Sync for OspreyV1 {}

/// Renders through the writer crate once and writes text, body, entry and meta one after the other into the buffer, their
/// lengths into `lens` (four numbers); a refusal becomes the context's last error, as for the door's own functions. A buffer
/// that is too small gets RX_TRUNC with the lengths filled in.
extern "C" fn render(
    ctx: *mut Ctx,
    walk: u32,
    json: *const c_char,
    n: usize,
    buf: *mut c_char,
    cap: usize,
    lens: *mut usize,
) -> c_int {
    if ctx.is_null() || lens.is_null() || (json.is_null() && n > 0) {
        return RX_BADARG;
    }
    let bytes = if n == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(json as *const u8, n) }
    };
    let Ok(contract) = std::str::from_utf8(bytes) else {
        return RX_BADARG;
    };
    let done = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        host_render(ctx as *mut RxCtx, walk, contract)
    }))
    .unwrap_or_else(|_| {
        Err(Refusal {
            kind: RefusalKind::Fault,
            reason: "panic in host_render".into(),
        })
    });
    match done {
        Ok(s) => {
            let meta = s.meta.json();
            let parts = [
                s.text.as_bytes(),
                s.body.as_bytes(),
                s.entry.as_bytes(),
                meta.as_bytes(),
            ];
            let total: usize = parts.iter().map(|p| p.len()).sum();
            for (at, part) in parts.iter().enumerate() {
                unsafe { *lens.add(at) = part.len() };
            }
            if buf.is_null() || cap < total {
                return RX_TRUNC;
            }
            let mut offset = 0;
            for part in parts {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        part.as_ptr(),
                        (buf as *mut u8).add(offset),
                        part.len(),
                    )
                };
                offset += part.len();
            }
            RX_OK
        }
        Err(refusal) => {
            unsafe { &mut *ctx }.set_err(&refusal.reason);
            status_of(refusal.kind)
        }
    }
}

/// The door's status for a kind of refusal: a construct not written is a refusal, a malformed request is a bad argument, and a
/// fault of the writers has no status of its own and is the panic status (the caller treats both as a failure of the library).
fn status_of(kind: RefusalKind) -> c_int {
    match kind {
        RefusalKind::NotWritten => RX_REFUSED,
        RefusalKind::BadRequest => RX_BADARG,
        RefusalKind::Fault => RX_PANIC,
    }
}

/// A new context of the library: the wings' kinds are declared in it before anything else, so a rewrite can make nodes of them.
extern "C" fn ctx_new() -> *mut Ctx {
    let ctx = rx_ctx_new();
    // SAFETY: `ctx` is not null and is the live context this function just made.
    if !ctx.is_null() && unsafe { raptor_wings::declare::rw_declare_kinds(ctx) } != RX_OK {
        // SAFETY: the context was just made and is not used again
        unsafe { rx_ctx_free(ctx) };
        return std::ptr::null_mut();
    }
    ctx
}

/// The refusal of a facts question that failed: a malformed question is the caller's, a broken data file is the library's fault.
fn facts_refusal(failed: facts::Failed) -> Refusal {
    match failed {
        facts::Failed::Bad(reason) => bad(reason),
        facts::Failed::Card(reason) => fault(reason),
    }
}

/// The refusal of an argument of the facts entry that is not a valid value.
fn bad_arg(why: &Fault) -> Refusal {
    match why {
        Fault::BadArg(reason) | Fault::Refused(reason) => bad(reason),
        _ => fault("a facts argument"),
    }
}

/// A NUL-terminated text argument of the facts entry.
fn text_arg<'a>(arg: *const c_char, what: &str) -> Result<&'a str, Fault> {
    if arg.is_null() {
        return Err(Fault::BadArg(format!("{what} is NULL")));
    }
    // SAFETY: the caller passes a NUL-terminated string that lives through the call.
    unsafe { CStr::from_ptr(arg) }
        .to_str()
        .map_err(|_| Fault::BadArg(format!("{what} is not UTF-8")))
}

/// The data directory a caller names, none for NULL.
fn dir_arg<'a>(arg: *const c_char) -> Result<Option<&'a std::path::Path>, Fault> {
    if arg.is_null() {
        return Ok(None);
    }
    text_arg(arg, "facts_dir").map(|path| Some(std::path::Path::new(path)))
}

/// Writes the facts answer for the target, the compiler identity and the stated machine (NULL: the probed one) into the caller's buffer (the door's length protocol: RX_TRUNC
/// with the length filled in when the buffer is too small); the reason of a refusal is the context's last error.
extern "C" fn facts(
    ctx: *mut Ctx,
    facts_dir: *const c_char,
    target: *const c_char,
    compiler: *const c_char,
    stated: *const c_char,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    if ctx.is_null() || len.is_null() {
        return RX_BADARG;
    }
    let done = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let args =
            (|| Ok::<_, Fault>((text_arg(target, "target")?, text_arg(compiler, "compiler")?)))();
        let (target, compiler) = args.map_err(|why| bad_arg(&why))?;
        let stated = if stated.is_null() {
            facts::Stated::default()
        } else {
            let text = text_arg(stated, "stated").map_err(|why| bad_arg(&why))?;
            facts::stated_from_json(text).map_err(facts_refusal)?
        };
        let given = dir_arg(facts_dir).map_err(|why| bad_arg(&why))?;
        let dir =
            facts::facts_dir(given).map_err(|reason| facts_refusal(facts::Failed::Card(reason)))?;
        facts::answer(&dir, target, compiler, &facts::probe(), &stated).map_err(facts_refusal)
    }))
    .unwrap_or_else(|_| Err(fault("panic in facts")));
    match done {
        Ok(answer) => match put(buf, cap, len, &answer) {
            Ok(()) => RX_OK,
            Err(Fault::Truncated) => RX_TRUNC,
            Err(_) => RX_BADARG,
        },
        Err(refusal) => {
            // SAFETY: `ctx` is the live context the caller holds.
            unsafe { &mut *ctx }.set_err(&refusal.reason);
            status_of(refusal.kind)
        }
    }
}

/// Lowers one kernel: reads the request, asks the lowering, and writes the answer into the caller's buffer; a refusal becomes the
/// context's last error with the status of its kind.
extern "C" fn lower(
    ctx: *mut Ctx,
    facts_dir: *const c_char,
    json: *const c_char,
    n: usize,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    if ctx.is_null() || len.is_null() || json.is_null() {
        return RX_BADARG;
    }
    // SAFETY: the caller passes `n` readable bytes at `json`.
    let bytes = unsafe { std::slice::from_raw_parts(json as *const u8, n) };
    let Ok(request) = std::str::from_utf8(bytes) else {
        return RX_BADARG;
    };
    let done = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let given = dir_arg(facts_dir).map_err(|why| bad_arg(&why))?;
        let dir = facts::facts_dir(given).map_err(|reason| Refusal {
            kind: RefusalKind::Fault,
            reason,
        })?;
        lower::lower(ctx, request, &dir)
    }))
    .unwrap_or_else(|_| {
        Err(Refusal {
            kind: RefusalKind::Fault,
            reason: "panic in lower".into(),
        })
    });
    match done {
        Ok(answer) => match put(buf, cap, len, &answer.to_json()) {
            Ok(()) => RX_OK,
            Err(Fault::Truncated) => RX_TRUNC,
            Err(_) => RX_BADARG,
        },
        Err(refusal) => {
            unsafe { &mut *ctx }.set_err(&refusal.reason);
            status_of(refusal.kind)
        }
    }
}

/// Writes the text of a set of units: reads the unit records, asks the lowering for the one text over them, and writes it into the
/// caller's buffer; a refusal becomes the context's last error with the status of its kind.
extern "C" fn set_text(
    ctx: *mut Ctx,
    json: *const c_char,
    n: usize,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    if ctx.is_null() || len.is_null() || json.is_null() {
        return RX_BADARG;
    }
    // SAFETY: the caller passes `n` readable bytes at `json`.
    let bytes = unsafe { std::slice::from_raw_parts(json as *const u8, n) };
    let Ok(input) = std::str::from_utf8(bytes) else {
        return RX_BADARG;
    };
    let done = std::panic::catch_unwind(|| lower::set_text(input)).unwrap_or_else(|_| {
        Err(Refusal {
            kind: RefusalKind::Fault,
            reason: "panic in set_text".into(),
        })
    });
    match done {
        Ok(text) => match put(buf, cap, len, &text) {
            Ok(()) => RX_OK,
            Err(Fault::Truncated) => RX_TRUNC,
            Err(_) => RX_BADARG,
        },
        Err(refusal) => {
            // SAFETY: `ctx` is the live context the caller holds.
            unsafe { &mut *ctx }.set_err(&refusal.reason);
            status_of(refusal.kind)
        }
    }
}

/// Writes the text of the shipped data document `name` into the caller's buffer; a refusal is the context's last error.
extern "C" fn card(
    ctx: *mut Ctx,
    facts_dir: *const c_char,
    name: *const c_char,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    if ctx.is_null() || len.is_null() {
        return RX_BADARG;
    }
    let done = std::panic::catch_unwind(|| {
        let given = dir_arg(facts_dir).map_err(|why| bad_arg(&why))?;
        let name = text_arg(name, "name").map_err(|why| bad_arg(&why))?;
        let dir =
            facts::facts_dir(given).map_err(|reason| facts_refusal(facts::Failed::Card(reason)))?;
        facts::shipped_card(&dir, name).map_err(facts_refusal)
    })
    .unwrap_or_else(|_| Err(fault("panic in card")));
    match done {
        Ok(text) => match put(buf, cap, len, &text) {
            Ok(()) => RX_OK,
            Err(Fault::Truncated) => RX_TRUNC,
            Err(_) => RX_BADARG,
        },
        Err(refusal) => {
            // SAFETY: `ctx` is the live context the caller holds.
            unsafe { &mut *ctx }.set_err(&refusal.reason);
            status_of(refusal.kind)
        }
    }
}

/// Writes the identity and its terminating NUL into the caller's 65-byte buffer.
extern "C" fn identity(out65: *mut c_char) -> c_int {
    if out65.is_null() {
        return RX_BADARG;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(IDENTITY.as_ptr(), out65 as *mut u8, IDENTITY.len());
        *out65.add(IDENTITY.len()) = 0
    }
    RX_OK
}

/// The table the library hands out.
static TABLE: OspreyV1 = OspreyV1 {
    abi: ABI,
    contract: aether_writers::contract::types::CONTRACT_VERSION,
    name: c"host".as_ptr(),
    ctx_new,
    ctx_free: rx_ctx_free,
    last_error: rx_last_error,
    load: rx_load,
    canonical: rx_canonical,
    render,
    identity,
    declare: rx_declare_dialect,
    facts,
    lower,
    set_text,
    card,
};

#[no_mangle]
/// The one symbol of the library: the entry table.
pub extern "C" fn osprey_v1() -> *const OspreyV1 {
    &TABLE
}

#[cfg(test)]
mod tests {
    //! The status each kind of refusal maps to.
    use super::*;

    #[test]
    fn each_kind_has_its_status() {
        assert_eq!(status_of(RefusalKind::NotWritten), RX_REFUSED);
        assert_eq!(status_of(RefusalKind::BadRequest), RX_BADARG);
        assert_eq!(status_of(RefusalKind::Fault), RX_PANIC);
    }
}
