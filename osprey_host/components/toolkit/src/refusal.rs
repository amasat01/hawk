//! The named refusals of the tool side: what a caller is told when a request cannot be served.
use std::fmt;
use std::io;

/// Why a request was refused; each kind has a stable name the door carries to its caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The request or the build facts it carries are malformed.
    BadRequest,
    /// A tool the request needs is not on this machine.
    ToolAbsent,
    /// A source text is over the compile budget.
    OverBudget,
    /// A tool did not finish within the wall-clock limit.
    TimedOut,
    /// A tool ran out of memory under the address-space cap.
    OutOfMemory,
    /// A tool ran and failed.
    ToolFailed,
    /// A file or folder the tool side needs could not be read or written.
    Io,
}

impl Kind {
    /// The name of the kind, as the door states it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Kind::BadRequest => "bad_request",
            Kind::ToolAbsent => "tool_absent",
            Kind::OverBudget => "over_budget",
            Kind::TimedOut => "timed_out",
            Kind::OutOfMemory => "out_of_memory",
            Kind::ToolFailed => "tool_failed",
            Kind::Io => "io",
        }
    }
}

/// A refusal: its kind and a text that names the thing and the way out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// What kind of refusal this is.
    pub kind: Kind,
    /// The text for the person who reads it.
    pub message: String,
}

impl Refusal {
    /// A refusal of `kind` with `message`.
    pub fn new(kind: Kind, message: impl Into<String>) -> Self {
        Refusal {
            kind,
            message: message.into(),
        }
    }

    /// A malformed-request refusal.
    pub fn bad(message: impl Into<String>) -> Self {
        Self::new(Kind::BadRequest, message)
    }

    /// An I/O refusal naming what was being done and the path involved.
    #[must_use]
    pub fn io(what: &str, path: &std::path::Path, error: &io::Error) -> Self {
        Self::new(Kind::Io, format!("{what} {}: {error}", path.display()))
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind.name(), self.message)
    }
}

impl std::error::Error for Refusal {}

/// The result type of the tool side.
pub type Done<T> = Result<T, Refusal>;
