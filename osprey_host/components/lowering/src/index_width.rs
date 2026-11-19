//! The width of the index type a unit is compiled with: the library's embedded header names a default spelling, and a define of
//! the build can replace it. The unit exports the same width in its layout table, so a consumer reads it from the library's
//! answer and never from the binary.
use aether_writers::refuse::{bad, Refusal};

/// The macro that is the one switch of the index type, in the embedded header and in a define.
const SWITCH: &str = "AETHER_INDEX_T";
/// The macro the unit's own layout table reads, which defaults to the switch; a define of it wins.
const ABI_SWITCH: &str = "EAGLE_ABI_INDEX_T";

/// The byte width of the integer type spelled `spelling`, or none for a spelling this table does not know.
fn width_of(spelling: &str) -> Option<i64> {
    match spelling
        .trim()
        .trim_start_matches("std::")
        .trim_start_matches("::")
    {
        "uint32_t" | "int32_t" | "unsigned int" | "unsigned" | "int" => Some(4),
        "uint64_t" | "int64_t" | "unsigned long" | "unsigned long long" | "long" | "long long"
        | "size_t" => Some(8),
        _ => None,
    }
}

/// The spelling a define `NAME=value` gives `name`, from the build's defines (the last one wins, as on a command line).
fn defined<'a>(defines: &'a [String], name: &str) -> Option<&'a str> {
    defines
        .iter()
        .rev()
        .find_map(|d| d.strip_prefix(name).and_then(|rest| rest.strip_prefix('=')))
}

/// The default spelling the embedded header states: the text after `#define AETHER_INDEX_T`.
fn header_default(header: &str) -> Option<&str> {
    header.lines().find_map(|line| {
        let rest = line.trim_start().strip_prefix('#')?.trim_start();
        rest.strip_prefix("define")?
            .trim_start()
            .strip_prefix(SWITCH)?
            .strip_prefix(char::is_whitespace)
    })
}

/// The width in bytes of the index type of a unit compiled with `defines` against the header text `header` (the embedded
/// `aether/typedefs.h`).
///
/// # Errors
/// Refuses a header that states no default, and a spelling of the type this table does not know, naming it.
pub fn bytes(header: &str, defines: &[String]) -> Result<i64, Refusal> {
    let spelling = defined(defines, ABI_SWITCH)
        .or_else(|| defined(defines, SWITCH))
        .or_else(|| header_default(header))
        .ok_or_else(|| {
            bad(format!(
                "the embedded header states no default for {SWITCH}"
            ))
        })?;
    width_of(spelling).ok_or_else(|| {
        bad(format!(
            "the index type {spelling:?} is not one this library knows the width of"
        ))
    })
}

/// The width in bytes of the index type of a unit compiled with `defines` against the embedded trees `trees` (the one named
/// `aether` holds `aether/typedefs.h`).
///
/// # Errors
/// Refuses trees without that header, and what [`bytes`] refuses.
pub fn of_trees(trees: &[toolkit::HeaderTree], defines: &[String]) -> Result<i64, Refusal> {
    let header = trees
        .iter()
        .find(|tree| tree.name == "aether")
        .and_then(|tree| {
            tree.files
                .iter()
                .find(|(path, _)| *path == "aether/typedefs.h")
        })
        .and_then(|(_, bytes)| std::str::from_utf8(bytes).ok())
        .ok_or_else(|| bad("the embedded aether headers lack aether/typedefs.h"))?;
    bytes(header, defines)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = "#ifndef AETHER_INDEX_T\n#define AETHER_INDEX_T std::uint32_t\n#endif\n";

    /// Oracle: the widths written by hand (`uint32_t` is 4 bytes, `uint64_t` 8).
    #[test]
    fn the_default_of_the_header_and_a_define_of_the_build_give_the_width() {
        assert_eq!(bytes(HEADER, &[]).unwrap(), 4);
        assert_eq!(
            bytes(HEADER, &["AETHER_INDEX_T=std::uint64_t".into()]).unwrap(),
            8
        );
        assert_eq!(
            bytes(
                HEADER,
                &[
                    "EAGLE_ABI_INDEX_T=std::uint64_t".into(),
                    "AETHER_INDEX_T=std::uint32_t".into()
                ]
            )
            .unwrap(),
            8
        );
    }

    #[test]
    fn a_header_without_a_default_and_an_unknown_type_are_refused_by_name() {
        assert!(bytes("", &[])
            .unwrap_err()
            .reason
            .contains("AETHER_INDEX_T"));
        let why = bytes(HEADER, &["AETHER_INDEX_T=half".into()]).unwrap_err();
        assert!(why.reason.contains("half"), "{}", why.reason);
    }
}
