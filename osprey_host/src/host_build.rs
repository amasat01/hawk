//! The host build instructions as data: for the compiler, machine and settings a facts question states, the argv templates of the
//! three commands that make a host unit library: the compile of a unit text to an object, the compile of the set text, and the
//! link of the objects into one shared library. Every flag is read from the data files (`host_build.json` and the `host_flags`
//! of the instruction set's card); nothing is probed. Places the caller fills in are written `{compiler}`, `{source}`, `{object}`,
//! `{output}`, `{depfile}`, `{include:aether}` and `{include:eagle}` (headers in two roots), `{include:served}` (headers served into
//! one folder), `{include:facts}`, `{include:door}` and, for the link, `{objects}` (one element for each object). The place `{defines}` stands for
//! the caller's defines, one element `-D<define>` for each and none when it has none; only a unit compile has it (the set text
//! takes no defines).
//!
//! Where the headers are is the caller's fact (`headers`, `roots` or `served`); the include places of a compile follow it. With
//! roots the includes come before the facts include; served, they come after it, as the old command had them.
//!
//! A unit compile is the old one-command build (compile and link) with the link half taken out: the same flags in the same order,
//! with `-shared` and `-o <library>` left to the link and `-c` and `-o <object>` added. The set text is compiled with exactly the
//! flags of a unit (the entities its headers define with external linkage must be identical across the objects of one library)
//! and one more include root, the door's. The link is `-shared` on both systems, as the old command had it; only the Linux
//! spelling is unverified here.
use crate::facts::{closed, member, names_of, read_file, text_of, Failed, ISA_CARDS};
use aether_writers::contract::Json;
use std::path::Path;

/// What the facts question states that the commands depend on: the profile in effect, where the headers are, whether the compiler
/// is GCC, the optimisation level asked for, and for a GCC the release and the verdict of the masked-call witness (none: it did not run).
pub(crate) struct Settings<'a> {
    /// The host profile in effect.
    pub profile: &'a str,
    /// Where the headers are: `roots` (one include root for each header set) or `served` (one folder).
    pub headers: &'a str,
    /// Whether the compiler is GCC proper.
    pub gcc: bool,
    /// The optimisation level asked for, without its dash (`O2`); none means the default.
    pub opt_level: Option<&'a str>,
    /// The GCC release as `major.minor`.
    pub gcc_release: Option<&'a str>,
    /// What the masked-call witness did on this compiler: `fixed`, `miscompile` or `unrun`.
    pub masked_call_witness: Option<&'a str>,
}

/// The list of texts under a key, as a JSON array.
fn flags(items: Vec<String>) -> Json {
    Json::Arr(items.into_iter().map(Json::Str).collect())
}

/// Whether the `vector_math` table of an instruction set's `host_flags` has the profile for the compiler family.
fn carries(host: &Json, file: &str, profile: &str, gcc: bool) -> Result<bool, Failed> {
    let table = member(
        member(host, file, "vector_math")?,
        file,
        if gcc { "gcc" } else { "other" },
    )?;
    Ok(table.get(profile).is_some())
}

/// The flags a profile adds to the instruction set's flags for the compiler's family, refused when the instruction set has no such
/// profile for that family.
fn vector_flags(
    dir: &Path,
    host: &Json,
    isa_file: &str,
    s: &Settings,
) -> Result<Vec<String>, Failed> {
    let table = member(
        member(host, isa_file, "vector_math")?,
        isa_file,
        if s.gcc { "gcc" } else { "other" },
    )?;
    match table.get(s.profile) {
        Some(_) => Ok(names_of(table, isa_file, s.profile)?),
        None => Err(not_built(dir, host, isa_file, s)?),
    }
}

/// The refusal of a profile this instruction set and compiler do not build, saying where the cards do build it: with the other
/// compiler family on this instruction set, else on the instruction sets whose cards carry it.
fn not_built(dir: &Path, host: &Json, isa_file: &str, s: &Settings) -> Result<Failed, Failed> {
    let isa = text_of(&read_file(dir, isa_file)?, isa_file, "isa")?;
    if carries(host, isa_file, s.profile, !s.gcc)? {
        let (here, there) = if s.gcc {
            ("a GCC compiler", "a compiler other than GCC")
        } else {
            ("a compiler other than GCC", "a GCC compiler")
        };
        return Ok(Failed::Bad(format!(
            "host profile {} is not built on {isa} with {here}; it is built there only with {there}",
            s.profile
        )));
    }
    let mut only = vec![];
    for (name, file, _) in ISA_CARDS {
        let other = read_file(dir, file)?;
        let flags = member(&other, file, "host_flags")?;
        if carries(flags, file, s.profile, true)? || carries(flags, file, s.profile, false)? {
            only.push(name);
        }
    }
    if only.is_empty() {
        return Err(Failed::Card(format!(
            "no instruction set card builds the vector math profile {}",
            s.profile
        )));
    }
    Ok(Failed::Bad(format!(
        "host profile {} is not built on {isa}; it is built only on {}",
        s.profile,
        only.join(" and ")
    )))
}

/// Whether the release (`major.minor`) lies in one of the ranges `[major, minor, major, minor]` of the card.
fn in_ranges(file: &str, ranges: &Json, release: &str) -> Result<bool, Failed> {
    let (major, minor) = release
        .split_once('.')
        .and_then(|(a, b)| Some((a.parse::<i64>().ok()?, b.parse::<i64>().ok()?)))
        .ok_or_else(|| {
            Failed::Bad(format!(
                "stated machine: gcc_release {release} is not major.minor"
            ))
        })?;
    for range in ranges.arr() {
        let bounds: Vec<i64> = range.arr().iter().filter_map(Json::int).collect();
        let [lo_major, lo_minor, hi_major, hi_minor] = bounds[..] else {
            return Err(format!("{file}: an affected range is not four numbers").into());
        };
        if (lo_major, lo_minor) <= (major, minor) && (major, minor) <= (hi_major, hi_minor) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The masked-call flag, when the card of the instruction set asks for it on a vector math profile, the compiler is a GCC release
/// the table lists, and the witness says it compiles correctly (no verdict is as if the witness did not run).
fn masked_flags(
    card: &Json,
    file: &str,
    isa: &Json,
    vector: bool,
    s: &Settings,
) -> Result<Vec<String>, Failed> {
    let wanted = matches!(member(isa, file, "masked_call")?, Json::Bool(true));
    if !(wanted && vector && s.gcc) {
        return Ok(vec![]);
    }
    let masked = member(card, "host_build.json", "masked_call")?;
    // a release that cannot be read is no release the table lists: no flag, as the old probe of the macros gave
    let Some(release) = s.gcc_release else {
        return Ok(vec![]);
    };
    if !in_ranges(
        "host_build.json",
        member(masked, "host_build.json", "affected")?,
        release,
    )? {
        return Ok(vec![]);
    }
    match s.masked_call_witness {
        Some("fixed") => Ok(vec![text_of(masked, "host_build.json", "flag")?]),
        Some("miscompile" | "unrun") | None => Ok(vec![]),
        Some(other) => Err(Failed::Bad(format!(
            "stated machine: masked_call_witness {other} is not fixed, miscompile or unrun"
        ))),
    }
}

/// The `host_build` record for an instruction set's card (`isa_file`) and the settings: `compile_unit`, `compile_set` and `link`
/// (per system), each an argv template.
pub(crate) fn host_build(
    dir: &Path,
    isa_file: &str,
    vector: bool,
    s: &Settings,
) -> Result<Json, Failed> {
    let file = "host_build.json";
    let card = read_file(dir, file)?;
    closed(
        &card,
        file,
        &[
            "opt_flag",
            "opt_levels",
            "default_opt_level",
            "contraction",
            "always",
            "unit_tail",
            "includes",
            "defines",
            "facts_include",
            "set_includes",
            "dependency",
            "compile_only",
            "masked_call",
            "link",
        ],
    )?;
    let level = match s.opt_level {
        Some(asked) => asked.to_string(),
        None => text_of(&card, file, "default_opt_level")?,
    };
    if !names_of(&card, file, "opt_levels")?.contains(&level) {
        return Err(Failed::Bad(format!("unknown opt level {level}")));
    }
    let isa = read_file(dir, isa_file)?;
    let host = member(&isa, isa_file, "host_flags")?;
    closed(
        host,
        isa_file,
        &["native", "portable", "vector_math", "masked_call"],
    )?;
    let contraction = member(&card, file, "contraction")?;
    let mut argv = vec!["{compiler}".to_string()];
    argv.push(text_of(&card, file, "opt_flag")?.replace("{level}", &level));
    argv.push(text_of(contraction, file, s.profile)?);
    argv.extend(names_of(&card, file, "always")?);
    argv.extend(names_of(
        host,
        isa_file,
        if s.profile == "portable" {
            "portable"
        } else {
            "native"
        },
    )?);
    if vector {
        argv.extend(vector_flags(dir, host, isa_file, s)?);
    }
    argv.extend(masked_flags(&card, file, host, vector, s)?);
    argv.extend(names_of(&card, file, "unit_tail")?);
    let places = member(&card, file, "includes")?;
    closed(places, file, &["roots", "served"])?;
    if !["roots", "served"].contains(&s.headers) {
        return Err(Failed::Bad(format!(
            "stated machine: headers {} is not roots or served",
            s.headers
        )));
    }
    let includes = names_of(places, file, s.headers)?;
    let set_includes = names_of(&card, file, "set_includes")?;
    let set = [&argv[..], &includes[..], &set_includes[..]].concat();
    // a unit's text takes the caller's defines and the facts header of the machine's wing; the set text takes neither
    let facts = [
        names_of(&card, file, "defines")?,
        names_of(&card, file, "facts_include")?,
    ]
    .concat();
    if s.headers == "roots" {
        argv.extend(includes);
        argv.extend(facts);
    } else {
        argv.extend(facts);
        argv.extend(includes);
    }
    let rest = [
        names_of(&card, file, "dependency")?,
        names_of(&card, file, "compile_only")?,
    ]
    .concat();
    let with_rest = |head: &[String]| flags([head, &rest[..]].concat());
    let link = member(&card, file, "link")?;
    closed(link, file, &["linux", "macos"])?;
    let mut systems = vec![];
    for system in ["linux", "macos"] {
        let tail = names_of(link, file, system)?;
        systems.push((
            system.to_string(),
            flags([&["{compiler}".to_string()][..], &tail[..]].concat()),
        ));
    }
    Ok(Json::Obj(vec![
        ("compile_unit".into(), with_rest(&argv)),
        ("compile_set".into(), with_rest(&set)),
        ("link".into(), Json::Obj(systems)),
    ]))
}
