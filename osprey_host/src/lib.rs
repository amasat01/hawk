//! The host backend library: one exported symbol, `osprey_v1`, returns the entry table. Loading and canonical forms go to the IR
//! core's door, rendering to the writer crate, and the identity is fixed at build time.
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
/// The wings crate answers the analyses the writers declare as external functions: naming it here links it, and a
/// function it lacks fails the build of the library (no symbol is left unresolved).
extern crate raptor_wings;
use aether_writers::{host_render, RxCtx};
use raptor_ir::canon::rx_canonical;
use raptor_ir::dialect::rx_declare_dialect;
use raptor_ir::door::{
    rx_ctx_free, rx_ctx_new, rx_last_error, RX_BADARG, RX_OK, RX_REFUSED, RX_TRUNC,
};
use raptor_ir::dump::rx_load;
use raptor_ir::record::Ctx;
use std::ffi::{c_char, c_int};

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
    .unwrap_or_else(|_| Err("panic in host_render".into()));
    match done {
        Ok(s) => {
            let parts = [
                s.text.as_bytes(),
                s.body.as_bytes(),
                s.entry.as_bytes(),
                s.meta.as_bytes(),
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
        Err(message) => {
            unsafe { &mut *ctx }.set_err(&message);
            RX_REFUSED
        }
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
};

#[no_mangle]
/// The one symbol of the library: the entry table.
pub extern "C" fn osprey_v1() -> *const OspreyV1 {
    &TABLE
}
