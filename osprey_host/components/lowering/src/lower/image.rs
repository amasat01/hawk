//! The command that makes the image of a unit, as the unit's answer states it. A backend reads it from its own facts answer; the
//! library never runs the compiler: the caller does.
/// The three members every unit with a device image states about the command that makes the image. One request carries one device
/// facts answer, so every unit of one answer holds the same values and no unit is checked against another.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageCommand {
    /// The compiler's short name: its tool and release (`name 12.6`).
    pub compiler: String,
    /// The flags of the command, joined by one space. The host compiler of the set's objects (`-ccbin`) is the place `{host_compiler}` of the compile template, not among them.
    pub flags: String,
    /// The version of the instruction text the image carries, as major * 10 + minor (85 is 8.5); 0 when it carries none.
    pub isa_version: i64,
}
