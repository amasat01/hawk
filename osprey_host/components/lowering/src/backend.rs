//! What a backend library states about itself: the one thing the door and the lowering ask of it. The trait is implemented by
//! the OSPREY crates; this crate never names one, so no branch on the backend exists here.
use crate::facts::{CardSet, Failed};
use crate::lower::card::TargetCards;
use crate::lower::request::KernelRequest;
use aether_writers::contract::types::Target;
use aether_writers::contract::Json;
use aether_writers::refuse::{Refusal, RefusalKind};
use aether_writers::{HostSource, RxCtx};
use std::ffi::CStr;
use std::path::Path;
use toolkit::Toolkit;

/// The writer a backend renders its unit with.
pub type Renderer = fn(*mut RxCtx, u32, &str) -> Result<HostSource, Refusal>;

/// A backend library: the one target it serves, its data documents, its machine facts and its writer.
pub trait Backend: Default + Sync + 'static {
    /// What the running machine reports to this backend's facts: its own record, read by [`Backend::probe`].
    type Probe;
    /// The machine a caller states for a build that is not for the machine running, as this backend reads it from the record of
    /// [`Backend::read_stated`].
    type Stated;
    /// The library's name, as the entry table states it.
    fn name(&self) -> &'static CStr;
    /// The abi number of the entry table, read from the door's header when the library is built.
    fn abi(&self) -> u32;
    /// The library's identity: 64 hex digits, fixed when the library is built.
    fn identity(&self) -> &'static str;
    /// The one target the library serves; a request or a facts question for another is refused by name.
    fn target(&self) -> Target;
    /// The name a facts question, a machine ask and a refusal use for the target the library serves.
    fn target_name(&self) -> &'static str;
    /// Asks the running machine for what the backend's facts start from.
    fn probe(&self) -> Self::Probe;
    /// The stated machine of the record a caller states (its `cards` member already taken off by the base): a member outside the
    /// ones this backend reads and a value of the wrong kind are refused by name.
    ///
    /// # Errors
    /// Refuses a record this backend cannot read.
    fn read_stated(&self, record: &Json) -> Result<Self::Stated, Failed>;
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
        probe: &Self::Probe,
        stated: &Self::Stated,
    ) -> Result<Json, Failed>;
    /// The cards of the target the library serves, read from its documents `docs` for `request`.
    ///
    /// # Errors
    /// Refuses a document that does not match its schema and a request the cards do not allow.
    fn cards(
        &self,
        docs: &CardSet,
        request: &KernelRequest,
    ) -> Result<Box<dyn TargetCards>, Refusal>;
    /// The optimisation levels the library's documents list.
    ///
    /// # Errors
    /// Refuses a document that does not list them.
    fn opt_levels(&self, cards: &CardSet) -> Result<Vec<String>, Failed>;
    /// The optimisation level a build gets when none is asked, from the library's documents.
    ///
    /// # Errors
    /// Refuses a document that does not state it.
    fn default_opt_level(&self, cards: &CardSet) -> Result<String, Failed>;
    /// The options the environment can fill when an ask states none: each is the option's name and the name of the ask of the tool
    /// table whose variable holds its value.
    fn env_option_asks(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }
    /// The members the library adds to the toolkit's machine ask: the ask names the probes it registered call for, and the facts
    /// the options pin.
    fn machine_ask(&self, _options: &Json) -> Vec<(String, Json)> {
        vec![]
    }
    /// The options a build gets when it states none, from the documents and, where the default depends on the machine, from the
    /// toolkit's answer for the ask `ask` made with `options`.
    ///
    /// # Errors
    /// Refuses a machine or a document the default cannot be read from.
    fn default_options(
        &self,
        _cards: &CardSet,
        _toolkit: &Toolkit,
        _ask: &Json,
        _options: &Json,
    ) -> Result<Vec<(String, Json)>, toolkit::Refusal> {
        Ok(vec![])
    }
    /// The members the library adds to the `stated` record of its machine answer for what the build `asked`.
    ///
    /// # Errors
    /// Refuses an ask the library's documents cannot answer.
    fn stated_extras(&self, _asked: &Json) -> Result<Vec<(String, Json)>, toolkit::Refusal> {
        Ok(vec![])
    }
    /// The compiler identity and the stated machine (the JSON `read_stated` reads) that a machine answer of this library
    /// stands for.
    ///
    /// # Errors
    /// Refuses a machine answer that states nothing usable.
    fn stated_machine(&self, cards: &CardSet, machine: &Json) -> Result<(String, String), Failed>;
    /// States in the lowering request `parts` what the library's machine answer `own` and its facts `answer` add, and refuses what the
    /// machine cannot run.
    ///
    /// # Errors
    /// Refuses a request the machine cannot serve.
    fn with_facts_request(
        &self,
        _own: &Json,
        _answer: &str,
        _parts: &mut Vec<(String, Json)>,
    ) -> Result<(), Refusal> {
        Ok(())
    }
    /// Builds one unit: `req` is `{"name", "text", "machine", "defines"?}`; the answer is the build answer text (`artifact`, `source`,
    /// `format`, `key`, `hit`, `seconds`, `files`, `figures`, `report`, `link_recipe`).
    ///
    /// # Errors
    /// Refuses a machine without the facts the build needs, and what the tool side refuses.
    fn build_unit(
        &self,
        req: &Json,
        machine: &Json,
        root: &Path,
    ) -> Result<String, toolkit::Refusal>;
    /// Links the library `name` of a set from the build answers `parts` and the text of the set.
    ///
    /// # Errors
    /// Refuses parts the library cannot link, and what the tool side refuses.
    fn link(
        &self,
        parts: &str,
        set_text: &str,
        name: &str,
        root: &Path,
    ) -> Result<String, toolkit::Refusal>;
    /// The word a build answer uses for the format of what the library builds.
    fn format(&self) -> &'static str;
    /// The include block of a unit's text, which the preprocess for the names a unit must not take starts from; none for a library
    /// that has no such names.
    fn reserved_names_prelude(&self) -> Option<&'static str> {
        None
    }
    /// The writer that renders a unit for the served target.
    fn render(&self) -> Renderer;
    /// The width in bytes of the index type of a unit this library compiles with the build's `defines` (`NAME=value`), from the
    /// header it embeds; none for a library whose units have no host index type.
    ///
    /// # Errors
    /// Refuses a header that states no default or a type whose width the library does not know.
    fn index_bytes(&self, _defines: &[String]) -> Result<Option<i64>, Refusal> {
        Ok(None)
    }
    /// The form of the artifact the library builds for the request, as its own word (the word the request's `image_format` and the
    /// answer's `device_image_format` carry); none when the request is for no target that builds one. The base carries the word and
    /// never reads it.
    ///
    /// # Errors
    /// Refuses a request whose target needs a form it does not state, and a form the library does not build.
    fn artifact_form(&self, request: &KernelRequest) -> Result<Option<String>, Refusal> {
        let _ = request;
        Ok(None)
    }
    /// The tool side of the library: its tool table and embedded header trees, made once; the one failure to make it (a table that
    /// does not parse) is the answer of every call.
    fn toolkit(&self) -> Result<&'static Toolkit, toolkit::Refusal>;
}

/// The refusal of a target the library does not serve.
pub fn unserved<B: Backend>(backend: &B, asked: &str) -> Refusal {
    Refusal {
        kind: RefusalKind::BadRequest,
        reason: format!(
            "the library {} serves the target {}, not {asked}",
            backend.name().to_string_lossy(),
            backend.target_name()
        ),
    }
}
