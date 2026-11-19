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
    fn option_names(&self) -> &'static [&'static str] {
        &["arch", "literal_table"]
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
    fn index_bytes(&self, defines: &[String]) -> Result<Option<i64>, Refusal> {
        osprey_lowering::index_width::of_trees(trees::TREES, defines).map(Some)
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

#[cfg(test)]
mod tests {
    use super::trees::TREES;

    fn tree(name: &str) -> &'static toolkit::HeaderTree {
        TREES
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("the library embeds no tree {name:?}"))
    }

    fn file(tree: &toolkit::HeaderTree, name: &str) -> Option<&'static [u8]> {
        tree.files.iter().find(|f| f.0 == name).map(|f| f.1)
    }

    /// Oracle: the widths written by hand (`std::uint32_t` is 4 bytes, `std::uint64_t` 8): the library states the index width of
    /// its units from its own embedded header and the build's defines.
    #[test]
    fn the_index_width_is_stated_from_the_embedded_header_and_the_defines() {
        use super::Cuda;
        use osprey_lowering::backend::Backend;
        assert_eq!(Cuda.index_bytes(&[]).unwrap(), Some(4));
        let wide = ["AETHER_INDEX_T=std::uint64_t".to_string()];
        assert_eq!(Cuda.index_bytes(&wide).unwrap(), Some(8));
    }

    /// Oracle: the runtime compiler has no system C++ headers, so its served tree holds aether's seven shims under the bare standard
    /// names and eagle's layout header under the name the units include, with the layout header's own bytes.
    #[test]
    fn the_runtime_tree_holds_the_seven_bare_shims_and_the_layout_alias() {
        let rtc = tree("rtc");
        for shim in [
            "cstddef",
            "cstdint",
            "type_traits",
            "utility",
            "concepts",
            "cmath",
            "atomic",
        ] {
            assert!(
                file(rtc, shim).is_some_and(|bytes| !bytes.is_empty()),
                "{shim}"
            );
        }
        let layout =
            file(tree("eagle"), "plugin/gref_layout.h").expect("eagle's layout header is embedded");
        assert_eq!(file(rtc, "plugin/gref_abi.h"), Some(layout));
        assert_eq!(
            rtc.files.len(),
            8,
            "the tree holds the seven shims and the alias, nothing else"
        );
    }

    /// The offline compiler reads the aether, eagle and facts trees as include roots: none of them holds a bare standard name, which
    /// would shadow the host's own headers; and the card lists the runtime tree for the runtime compile only.
    #[test]
    fn the_offline_include_roots_hold_none_of_the_runtime_names() {
        for name in ["aether", "eagle", "facts"] {
            for bare in [
                "cstddef",
                "cstdint",
                "type_traits",
                "utility",
                "concepts",
                "cmath",
                "atomic",
            ] {
                assert!(file(tree(name), bare).is_none(), "{name}/{bare}");
            }
        }
        let card = include_str!("../facts/cuda_image.json");
        let runtime = &card[card.find("\"runtime\"").expect("the runtime member")..];
        let offline = &card[..card.find("\"runtime\"").unwrap()];
        assert!(runtime.contains("\"rtc\""));
        assert!(!offline.contains("\"rtc\""));
    }
}
