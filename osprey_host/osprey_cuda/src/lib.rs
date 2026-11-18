//! The cuda OSPREY: the library that lowers kernels for the device. It serves the device target alone, ships the device's data
//! documents beside the library, and offers the entry table every OSPREY hands out; the `door` package beside it exports that table as the one symbol, `osprey_v1`.
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
use aether_writers::contract::types::Target;
use aether_writers::contract::Json;
use aether_writers::refuse::Refusal;
use osprey_lowering::backend::{Backend, Renderer};
use osprey_lowering::door::{table as door_table, OspreyV1};
use osprey_lowering::facts::{device_machine, CardSet, Failed, Probe, Stated};
use osprey_lowering::lower::card::{device_build_of, DeviceCard};
use std::ffi::CStr;
use std::sync::OnceLock;
use toolkit::{Env, Toolkit};

/// The table's abi number, read from include/osprey.h by the build script (the one source of it).
const ABI: u32 = include!(concat!(env!("OUT_DIR"), "/abi.rs"));
/// The library's identity: 64 hex digits, fixed at build time.
const IDENTITY: &str = include_str!(concat!(env!("OUT_DIR"), "/identity.txt"));
/// The data documents a caller may replace for one call: the device build card and the device document.
const CARDS: [&str; 2] = ["cuda_build.json", "cuda.json"];

/// The header trees the library embeds, from the folders its build names.
mod trees {
    include!(concat!(env!("OUT_DIR"), "/trees.rs"));
}
/// The library's own tool table: how its tools are found, probed and limited.
const TOOL_TABLE: &str = include_str!("../facts/tools.json");

/// The backend of this library.
#[derive(Default)]
pub struct Cuda;

impl Backend for Cuda {
    fn name(&self) -> &'static CStr {
        c"cuda"
    }
    fn abi(&self) -> u32 {
        ABI
    }
    fn identity(&self) -> &'static str {
        IDENTITY
    }
    fn target(&self) -> Target {
        Target::Device
    }
    fn format(&self) -> &'static str {
        "ptx"
    }
    fn card_names(&self) -> &'static [&'static str] {
        &CARDS
    }
    fn machine(
        &self,
        cards: &CardSet,
        compiler: &str,
        _probe: &Probe,
        stated: &Stated,
    ) -> Result<Json, Failed> {
        device_machine(cards, compiler, stated)
    }
    fn device_card(&self, docs: &CardSet) -> Result<Option<DeviceCard>, Refusal> {
        device_build_of(docs).map(Some)
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
        aether_writers::cuda_render
    }
}

/// The entry table of the library.
pub fn table() -> &'static OspreyV1 {
    /// The table, made on first use.
    static TABLE: OnceLock<OspreyV1> = OnceLock::new();
    TABLE.get_or_init(door_table::<Cuda>)
}
