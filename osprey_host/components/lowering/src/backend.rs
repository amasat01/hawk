//! What a backend library states about itself: the one thing the door and the lowering ask of it. The trait is implemented by
//! the OSPREY crates; this crate never names one, so no branch on the backend exists here.
use crate::facts::{CardSet, Failed, Probe, Stated};
use crate::lower::card::{BuildCard, DeviceCard};
use aether_writers::contract::types::Target;
use aether_writers::contract::Json;
use aether_writers::refuse::{Refusal, RefusalKind};
use aether_writers::{HostSource, RxCtx};
use std::ffi::CStr;
use toolkit::Toolkit;

/// The writer a backend renders its unit with.
pub type Renderer = fn(*mut RxCtx, u32, &str) -> Result<HostSource, Refusal>;

/// A backend library: the one target it serves, its data documents, its machine facts and its writer.
pub trait Backend: Default + Sync + 'static {
    /// The library's name, as the entry table states it.
    fn name(&self) -> &'static CStr;
    /// The abi number of the entry table, read from the door's header when the library is built.
    fn abi(&self) -> u32;
    /// The library's identity: 64 hex digits, fixed when the library is built.
    fn identity(&self) -> &'static str;
    /// The one target the library serves; a request or a facts question for another is refused by name.
    fn target(&self) -> Target;
    /// The data documents the library ships and a caller may replace for one call.
    fn card_names(&self) -> &'static [&'static str];
    /// The keys of the options a caller may state for this target (`options` of a machine ask); any other key is refused by name.
    fn option_names(&self) -> &'static [&'static str] {
        &[]
    }
    /// The machine record of the target for a compiler identity, from the library's own documents.
    fn machine(
        &self,
        cards: &CardSet,
        compiler: &str,
        probe: &Probe,
        stated: &Stated,
    ) -> Result<Json, Failed>;
    /// The build card of a library that serves the host, none for another.
    fn host_card(&self, _docs: &CardSet) -> Result<Option<BuildCard>, Refusal> {
        Ok(None)
    }
    /// The build card of a library that serves the device, none for another.
    fn device_card(&self, _docs: &CardSet) -> Result<Option<DeviceCard>, Refusal> {
        Ok(None)
    }
    /// Whether the library serves the host target: its machine states the host's own facts, and its units are built to objects and
    /// linked into a shared library.
    fn host_card_wanted(&self) -> bool {
        self.target() == Target::Host
    }
    /// The word a build answer uses for the format of what the library builds.
    fn format(&self) -> &'static str;
    /// The include block of a unit's text, which the preprocess for the names a unit must not take starts from; none for a library
    /// that has no such names.
    fn reserved_names_prelude(&self) -> Option<&'static str> {
        None
    }
    /// The writer that renders a unit for the served target.
    fn render(&self) -> Renderer;
    /// The tool side of the library: its tool table and embedded header trees, made once; the one failure to make it (a table that
    /// does not parse) is the answer of every call.
    fn toolkit(&self) -> Result<&'static Toolkit, toolkit::Refusal>;
}

/// The name a facts question or a request uses for a target.
pub fn target_name(target: Target) -> &'static str {
    match target {
        Target::Host => "host",
        Target::Device => "cuda",
    }
}

/// The refusal of a target the library does not serve.
pub fn unserved<B: Backend>(backend: &B, asked: &str) -> Refusal {
    Refusal {
        kind: RefusalKind::BadRequest,
        reason: format!(
            "the library {} serves the target {}, not {asked}",
            backend.name().to_string_lossy(),
            target_name(backend.target())
        ),
    }
}
