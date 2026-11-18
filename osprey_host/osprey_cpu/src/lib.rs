//! The cpu OSPREY: the library that lowers kernels for the host. It serves the host target alone, ships the host's data
//! documents beside the library, and offers the entry table every OSPREY hands out; the `door` package beside it exports that table as the one symbol, `osprey_v1`.
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
use aether_writers::contract::types::Target;
use aether_writers::contract::Json;
use aether_writers::refuse::Refusal;
use osprey_lowering::backend::{Backend, Renderer};
use osprey_lowering::door::{table as door_table, OspreyV1};
use osprey_lowering::facts::{host_machine, CardSet, Failed, Probe, Stated};
use osprey_lowering::lower::card::{host_build_of, BuildCard};
use std::ffi::CStr;
use std::sync::OnceLock;
use toolkit::{Env, Toolkit};

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

/// The backend of this library.
#[derive(Default)]
pub struct Cpu;

impl Backend for Cpu {
    fn name(&self) -> &'static CStr {
        c"cpu"
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
        probe: &Probe,
        stated: &Stated,
    ) -> Result<Json, Failed> {
        host_machine(cards, compiler, probe, stated)
    }
    fn host_card(&self, docs: &CardSet) -> Result<Option<BuildCard>, Refusal> {
        host_build_of(docs).map(Some)
    }
    fn toolkit(&self) -> Result<&'static Toolkit, toolkit::Refusal> {
        /// The tool side, made on first use from the table, the trees and the process environment.
        static TOOLKIT: OnceLock<Result<Toolkit, toolkit::Refusal>> = OnceLock::new();
        TOOLKIT
            .get_or_init(|| Toolkit::new(TOOL_TABLE, trees::TREES, Env::process()))
            .as_ref()
            .map_err(Clone::clone)
    }
    fn render(&self) -> Renderer {
        aether_writers::host_render
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
