//! What a backend registers when it builds its [`Toolkit`](crate::Toolkit): probes, which state facts of the machine beyond the
//! tools, and grammars, which read a tool's output or an image into figures. The toolkit asks them by the name the backend gave;
//! it knows no name of its own.
use crate::discover::{Env, Machine};
use crate::refusal::Done;
use crate::runtime::Runtime;
use rcore::json::J;

/// A reading of text (a tool's output) or of an image's bytes into a record; the grammar decides which of the two it reads.
pub type Grammar = fn(text: &str, image: &[u8]) -> J;

/// What a probe is given: the member of the tool table named like the probe (when the table has one), the machine ask, the
/// environment, the tools found, and the runtime compile library when one stands in for a tool.
pub struct ProbeInput<'a> {
    /// The table's member named like the probe.
    pub section: Option<&'a J>,
    /// The machine ask.
    pub ask: &'a J,
    /// The environment the tools run in, read now.
    pub env: &'a Env,
    /// The tools found.
    pub machine: &'a Machine,
    /// The runtime compile library that replaced a tool, when one did.
    pub runtime: Option<&'a Runtime>,
}

/// What a probe states.
pub struct ProbeAnswer {
    /// The record stated under the probe's name.
    pub facts: J,
    /// Why this machine cannot run what the backend builds, when the probe finds so.
    pub blocked: Option<String>,
}

/// A function from the machine to the record a backend states under its name. It is asked when the table has a member of that
/// name or the ask has a true member of that name.
pub type Probe = fn(&ProbeInput<'_>) -> Done<ProbeAnswer>;

/// The names of `grammars`, comma separated, for a refusal that lists what is registered.
pub(crate) fn names(grammars: &[(&str, Grammar)]) -> String {
    grammars
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join(", ")
}
