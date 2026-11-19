//! What a valid OSPREY manifest is, decided once and compiled: the members a manifest states, the one spelling of a target id,
//! the roster of the folders a search found and the answer to which OSPREY serves a target. A caller hands over folder paths and
//! raw target names and gets back checked facts as JSON text, or a refusal sentence that names the folder and the rule.
//!
//! The three functions of [`abi`] are what the binding's loader answers with; [`manifest`], [`roster`] and [`target`] hold the
//! rules and are tested without the C layer.
pub mod abi;
pub mod json;
pub mod manifest;
pub mod roster;
pub mod target;
