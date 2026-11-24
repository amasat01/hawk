//! The cpu OSPREY: the library that lowers kernels for the host. It serves the host target alone, ships the host's data
//! documents beside the library, and offers the entry table every OSPREY hands out; the `door` package beside it exports that table as the one symbol, `osprey_v1`.
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
use aether_writers::contract::types::Target;
use aether_writers::contract::Json;
use aether_writers::refuse::Refusal;
use osprey_lowering::backend::{Backend, Renderer};
use osprey_lowering::door::{table as door_table, OspreyV1};
use osprey_lowering::facts::{CardSet, Failed};
use osprey_lowering::lower::card::TargetCards;
use osprey_lowering::lower::request::KernelRequest;
use std::ffi::CStr;
use std::sync::OnceLock;
use toolkit::{Env, Toolkit};

pub mod card;
mod compiler_family;
mod host;
mod host_build;
pub mod host_facts;
pub mod host_state;
mod identity;
mod unit_build;

/// The table's abi number, read from include/osprey.h by the build script (the one source of it).
const ABI: u32 = include!(concat!(env!("OUT_DIR"), "/abi.rs"));
/// The library's identity: 64 hex digits, fixed at build time.
const IDENTITY: &str = include_str!(concat!(env!("OUT_DIR"), "/identity.txt"));
/// The data documents a caller may replace for one call: the host build card and the instruction set documents.
const CARDS: [&str; 3] = ["build.json", "arm.json", "x86.json"];

/// The header trees the library embeds, from the folders its build names.
mod trees {
    include!(concat!(env!("OUT_DIR"), "/trees.rs"));
}
/// The library's own tool table: how its tools are found, probed and limited.
const TOOL_TABLE: &str = include_str!("../facts/tools.json");

/// `toolkit` with what only this backend states: the host probe of the machine answer, the processor identity a witness key
/// carries and the grammar that tells the compiler family from its predefined macros.
#[must_use]
pub fn register(toolkit: Toolkit) -> Toolkit {
    toolkit
        .with_identity(identity::machine_identity)
        .with_probe(host::SECTION, host::probe)
        .with_grammar("gcc_family", compiler_family::gcc_family_grammar)
}

/// The backend of this library.
#[derive(Default)]
pub struct Cpu;

impl Backend for Cpu {
    type Probe = host_state::Probe;
    type Stated = host_state::Stated;
    fn name(&self) -> &'static CStr {
        c"cpu"
    }
    fn target_name(&self) -> &'static str {
        "host"
    }
    fn probe(&self) -> host_state::Probe {
        host_state::probe()
    }
    fn read_stated(&self, record: &Json) -> Result<host_state::Stated, Failed> {
        host_state::read_stated(record)
    }
    fn abi(&self) -> u32 {
        ABI
    }
    fn identity(&self) -> &'static str {
        IDENTITY
    }
    fn target(&self) -> Target {
        Target::Host
    }
    fn format(&self) -> &'static str {
        "host"
    }
    fn reserved_names_prelude(&self) -> Option<&'static str> {
        Some(aether_writers::unit::text::UNIT_INCLUDES)
    }
    fn card_names(&self) -> &'static [&'static str] {
        &CARDS
    }
    fn option_names(&self) -> &'static [&'static str] {
        &["profile"]
    }
    fn machine(
        &self,
        cards: &CardSet,
        compiler: &str,
        probe: &host_state::Probe,
        stated: &host_state::Stated,
    ) -> Result<Json, Failed> {
        host_facts::host_machine(cards, compiler, probe, stated)
    }
    fn cards(
        &self,
        docs: &CardSet,
        _request: &KernelRequest,
    ) -> Result<Box<dyn TargetCards>, Refusal> {
        card::HostCards::of_docs(docs).map(|cards| Box::new(cards) as Box<dyn TargetCards>)
    }
    fn opt_levels(&self, cards: &CardSet) -> Result<Vec<String>, Failed> {
        host_facts::opt_levels(cards)
    }
    fn default_opt_level(&self, cards: &CardSet) -> Result<String, Failed> {
        host_facts::default_opt_level(cards)
    }
    fn env_option_asks(&self) -> &'static [(&'static str, &'static str)] {
        &[("profile", "host_profile")]
    }
    fn machine_ask(&self, _options: &Json) -> Vec<(String, Json)> {
        vec![(host::SECTION.to_string(), Json::Bool(true))]
    }
    fn default_options(
        &self,
        cards: &CardSet,
        toolkit: &Toolkit,
        ask: &Json,
        options: &Json,
    ) -> Result<Vec<(String, Json)>, toolkit::Refusal> {
        unit_build::default_options(cards, toolkit, ask, options)
    }
    fn stated_extras(&self, asked: &Json) -> Result<Vec<(String, Json)>, toolkit::Refusal> {
        unit_build::stated_extras(asked)
    }
    fn stated_machine(&self, _cards: &CardSet, machine: &Json) -> Result<(String, String), Failed> {
        unit_build::stated_machine(machine)
    }
    fn build_unit(
        &self,
        req: &Json,
        machine: &Json,
        root: &std::path::Path,
    ) -> Result<String, toolkit::Refusal> {
        unit_build::build_unit(req, machine, root)
    }
    fn link(
        &self,
        parts: &str,
        set_text: &str,
        name: &str,
        root: &std::path::Path,
    ) -> Result<String, toolkit::Refusal> {
        unit_build::link(parts, set_text, name, root)
    }
    fn index_bytes(&self, defines: &[String]) -> Result<Option<i64>, Refusal> {
        osprey_lowering::index_width::of_trees(trees::TREES, defines).map(Some)
    }
    fn toolkit(&self) -> Result<&'static Toolkit, toolkit::Refusal> {
        /// The tool side, made on first use from the table, the trees and the process environment.
        static TOOLKIT: OnceLock<Result<Toolkit, toolkit::Refusal>> = OnceLock::new();
        TOOLKIT
            .get_or_init(|| Toolkit::new(TOOL_TABLE, trees::TREES, Env::process()).map(register))
            .as_ref()
            .map_err(Clone::clone)
    }
    fn render(&self) -> Renderer {
        aether_writers::host_render
    }
}

/// The shipped documents of the library, for the tests that read what they state.
#[cfg(test)]
mod testing {
    use std::path::{Path, PathBuf};

    /// Every document a caller may replace.
    pub const ALL_CARDS: &[&str] = &super::CARDS;

    /// The folder of the shipped documents.
    pub fn shipped() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("facts")
    }

    /// The text of the shipped document `name`, kept for the life of the process.
    pub fn text(name: &str) -> &'static str {
        Box::leak(
            std::fs::read_to_string(shipped().join(name))
                .unwrap()
                .into_boxed_str(),
        )
    }
}

/// The entry table of the library.
pub fn table() -> &'static OspreyV1 {
    /// The table, made on first use.
    static TABLE: OnceLock<OspreyV1> = OnceLock::new();
    TABLE.get_or_init(door_table::<Cpu>)
}

#[cfg(test)]
mod tests {
    //! The host row of the manifest names its object by the vocabulary of a bundle manifest's formats.
    use super::*;

    /// A bundle manifest's formats are `ptx`, `cubin`, `fatbin` and `host`; its reader refuses any other word, so the
    /// host target's format, in the library and in the manifest beside it, is `host`.
    #[test]
    fn the_host_format_is_the_bundle_manifest_word() {
        assert_eq!(Cpu.format(), "host");
        let manifest = include_str!("../osprey.json");
        assert!(
            manifest.contains("\"format\": \"host\""),
            "osprey.json states another format word"
        );
    }
}
