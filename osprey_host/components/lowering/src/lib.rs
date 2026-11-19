//! The lowering engine and the door every OSPREY library exports. A backend library (`osprey_cpu`, `osprey_cuda`) states what it
//! is through the `Backend` trait and holds only its own data files; this crate holds what no backend owns: the lowering of a
//! kernel into units, the readers of the data documents, and the entry table the door hands out.
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
/// The wings crate answers the analyses the writers declare as external functions: naming it here links it, and a
/// function it lacks fails the build of the library (no symbol is left unresolved).
extern crate raptor_wings;
pub mod backend;
mod device_build;
pub mod door;
pub mod facts;
mod host_build;
pub mod index_width;
pub mod lower;
#[cfg(test)]
mod testing;
pub mod tool;

pub use backend::Backend;
