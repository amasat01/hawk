//! Backends and data documents for the unit tests of this crate. The crate ships no documents of its own; the tests read the
//! shipped documents of the two libraries, merged into one directory in the temporary folder, and lower through a backend of
//! each kind that carries no identity.
use crate::backend::{Backend, Renderer};
use crate::facts::{device_machine, host_machine, CardSet, Failed, Probe, Stated};
use crate::lower::card::{device_build_of, host_build_of, BuildCard, DeviceCard};
use crate::lower::request::KernelRequest;
use aether_writers::contract::types::Target;
use aether_writers::contract::Json;
use aether_writers::refuse::Refusal;
use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Every document a caller may replace, for the host and the device together.
pub const ALL_CARDS: &[&str] = &[
    "build.json",
    "arm.json",
    "x86.json",
    "cuda_build.json",
    "cuda.json",
];

/// The folder of a library's shipped documents.
fn library_facts(library: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(library)
        .join("facts")
}

/// The shipped documents of both libraries in one directory.
pub fn shipped() -> PathBuf {
    /// The merged directory, made once per test process.
    static MERGED: OnceLock<PathBuf> = OnceLock::new();
    MERGED
        .get_or_init(|| {
            let dir =
                std::env::temp_dir().join(format!("osprey_lowering_facts_{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            for library in ["osprey_cpu", "osprey_cuda"] {
                for file in std::fs::read_dir(library_facts(library)).unwrap() {
                    let file = file.unwrap().path();
                    std::fs::copy(&file, dir.join(file.file_name().unwrap())).unwrap();
                }
            }
            dir
        })
        .clone()
}

/// The text of the shipped document `name`, kept for the life of the process.
pub fn text(name: &str) -> &'static str {
    Box::leak(
        std::fs::read_to_string(shipped().join(name))
            .unwrap()
            .into_boxed_str(),
    )
}

/// A host backend for tests.
#[derive(Default)]
pub struct TestHost;

/// A device backend for tests.
#[derive(Default)]
pub struct TestDevice;

impl Backend for TestHost {
    fn format(&self) -> &'static str {
        "host"
    }
    fn name(&self) -> &'static CStr {
        c"test_host"
    }
    fn abi(&self) -> u32 {
        0
    }
    fn identity(&self) -> &'static str {
        ""
    }
    fn target(&self) -> Target {
        Target::Host
    }
    fn card_names(&self) -> &'static [&'static str] {
        ALL_CARDS
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
    fn render(&self) -> Renderer {
        aether_writers::host_render
    }
    fn toolkit(&self) -> Result<&'static toolkit::Toolkit, toolkit::Refusal> {
        Err(toolkit::Refusal::bad("the test backend has no tool side"))
    }
}

impl Backend for TestDevice {
    fn format(&self) -> &'static str {
        "ptx"
    }
    fn name(&self) -> &'static CStr {
        c"test_device"
    }
    fn abi(&self) -> u32 {
        0
    }
    fn identity(&self) -> &'static str {
        ""
    }
    fn target(&self) -> Target {
        Target::Device
    }
    fn card_names(&self) -> &'static [&'static str] {
        ALL_CARDS
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
    fn render(&self) -> Renderer {
        aether_writers::cuda_render
    }
    fn toolkit(&self) -> Result<&'static toolkit::Toolkit, toolkit::Refusal> {
        Err(toolkit::Refusal::bad("the test backend has no tool side"))
    }
}

/// The cards `request` is lowered with, through the test backend of the target it names.
pub fn load_cards(request: &KernelRequest, docs: &CardSet) -> Result<crate::lower::Cards, Refusal> {
    match request.targets.contains(&Target::Device) {
        true => crate::lower::Cards::load(&TestDevice, request, docs),
        false => crate::lower::Cards::load(&TestHost, request, docs),
    }
}
