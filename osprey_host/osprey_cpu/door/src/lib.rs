//! The cpu OSPREY as a loadable object: exports the library's entry table as the one symbol, `osprey_v1`.
//! The symbol lives here, in a crate nothing links, so that the rlib other crates link beside it never defines it.
#![warn(missing_docs)]
use osprey_lowering::door::OspreyV1;

#[no_mangle]
/// The one symbol of the library: the entry table.
pub extern "C" fn osprey_v1() -> *const OspreyV1 {
    osprey_cpu_lib::table()
}
