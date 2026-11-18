//! The tool side of an OSPREY, compiled: find the compiler and tools by a table, keep a content-addressed store with one build
//! per key, serve embedded headers, run the compiler under the machine guards and link a library. A backend's difference is data
//! the OSPREY passes (its tool table, its argv templates, its header trees); no function here names a backend.
//!
//! The four door functions are methods of [`Toolkit`]: [`Toolkit::machine`], [`Toolkit::build`], [`Toolkit::link`] and
//! [`Toolkit::state`]. Each takes and returns JSON text and paths; a refusal is a [`Refusal`] with a stable kind name.
pub mod compile;
pub mod device;
pub mod discover;
pub mod dl;
pub mod embed;
pub mod figures;
pub mod headers;
pub mod host;
pub mod jobs;
pub mod link;
pub mod refusal;
pub mod reserved;
pub mod run;
pub mod runtime;
pub mod shellwords;
pub mod store;
pub mod toolkit;
pub mod wire;
pub mod witness;

pub use discover::Env;
pub use headers::HeaderTree;
pub use refusal::{Kind, Refusal};
pub use toolkit::Toolkit;
