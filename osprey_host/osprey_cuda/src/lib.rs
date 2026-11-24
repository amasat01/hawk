//! The cuda OSPREY: the library that lowers kernels for the device. It serves the device target alone, ships the device's data
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
mod device;
mod device_build;
pub mod device_facts;
pub mod device_state;
mod figures;
mod fusion;
mod image;
mod local_memory;
mod unit_build;

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
/// What the machine answer says first when the build had no CUDA toolkit and so left the device runtime out.
const WITHOUT_RUNTIME: &str = "built without the device runtime";

/// `toolkit` with what only this backend states: the device probe of the machine answer and the two grammars that read the
/// figures of a compile (the assembler's report and the image's own resource records).
#[must_use]
pub fn register(toolkit: Toolkit) -> Toolkit {
    toolkit
        .with_probe(device::SECTION, device::probe)
        .with_grammar("ptxas_info", figures::ptxas_info_grammar)
        .with_grammar("image_resources", figures::image_resources_grammar)
}

/// The backend of this library.
#[derive(Default)]
pub struct Cuda;

impl Backend for Cuda {
    type Probe = ();
    type Stated = device_state::Stated;
    fn name(&self) -> &'static CStr {
        c"cuda"
    }
    fn target_name(&self) -> &'static str {
        "cuda"
    }
    fn probe(&self) {}
    fn read_stated(&self, record: &Json) -> Result<device_state::Stated, Failed> {
        device_state::read_stated(record)
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
    fn artifact_form(&self, request: &KernelRequest) -> Result<Option<String>, Refusal> {
        match image::form_of(request)? {
            Some(form) => Ok(Some(form.word().to_string())),
            None if request.targets.contains(&Target::Device) => Err(aether_writers::refuse::bad(
                "request key image_format: a device target needs one",
            )),
            None => Ok(None),
        }
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
        _probe: &(),
        stated: &device_state::Stated,
    ) -> Result<Json, Failed> {
        device_facts::device_machine(cards, compiler, stated)
    }
    fn cards(
        &self,
        docs: &CardSet,
        request: &KernelRequest,
    ) -> Result<Box<dyn TargetCards>, Refusal> {
        card::DeviceCards::of_request(docs, request)
            .map(|cards| Box::new(cards) as Box<dyn TargetCards>)
    }
    fn opt_levels(&self, cards: &CardSet) -> Result<Vec<String>, Failed> {
        device_facts::opt_levels(cards)
    }
    fn default_opt_level(&self, cards: &CardSet) -> Result<String, Failed> {
        device_facts::default_opt_level(cards)
    }
    fn machine_ask(&self, options: &Json) -> Vec<(String, Json)> {
        unit_build::machine_ask(options)
    }
    fn stated_machine(&self, cards: &CardSet, machine: &Json) -> Result<(String, String), Failed> {
        unit_build::stated_machine(cards, machine)
    }
    fn with_facts_request(
        &self,
        own: &Json,
        answer: &str,
        parts: &mut Vec<(String, Json)>,
    ) -> Result<(), Refusal> {
        unit_build::with_facts_request(self.option_names(), own, answer, parts)
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
            .get_or_init(|| {
                Toolkit::new(TOOL_TABLE, trees::TREES, Env::process())
                    .map(register)
                    .map(|toolkit| {
                        if cfg!(device_runtime) {
                            toolkit
                        } else {
                            toolkit.without_runtime(WITHOUT_RUNTIME)
                        }
                    })
            })
            .as_ref()
            .map_err(Clone::clone)
    }
    fn render(&self) -> Renderer {
        aether_writers::cuda_render
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

    /// Oracle: the build rule stated in build.rs. A library built with the device runtime leaves the verdict to the driver and never
    /// calls itself device-less; one built without it answers false and says why, whatever the machine offers.
    #[test]
    fn the_machine_answer_states_the_build_mode() {
        use osprey_lowering::backend::Backend;
        let answer = super::Cuda.toolkit().unwrap().machine("{}").unwrap();
        let without = answer.contains("built without the device runtime");
        assert_eq!(without, !cfg!(device_runtime), "{answer}");
        if without {
            assert!(answer.contains("\"runs_here\":false"), "{answer}");
        }
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
