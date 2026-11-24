//! The host's facts: the profile lists and the select weights of the cpu card, the cards of the instruction sets, the machine record
//! of the host target and its fusion row, read from the library's own documents.
use crate::host_build;
use crate::host_state::{Probe, Stated};
use aether_writers::contract::types::Caches;
use aether_writers::contract::{Contract, Json};
use osprey_lowering::facts::{
    closed, flag_of, fusion_row, member, names_of, number, text_of, weights_json, weights_of,
    CardSet, Failed, FusionRow,
};

/// The optimisation level a build gets when none is asked: the `default_opt_level` of the document that states the compile of the
/// host (`host_build.json`).
///
/// # Errors
/// Refuses a document that does not state it.
pub fn default_opt_level(cards: &CardSet) -> Result<String, Failed> {
    let file = "host_build.json";
    Ok(text_of(&cards.json(file)?, file, "default_opt_level")?)
}

/// The optimisation levels the document that states the compile of the host (`host_build.json`) lists.
///
/// # Errors
/// Refuses a document that does not list them.
pub fn opt_levels(cards: &CardSet) -> Result<Vec<String>, Failed> {
    let file = "host_build.json";
    Ok(names_of(&cards.json(file)?, file, "opt_levels")?)
}

/// The cpu card: the profile lists and the select weights of a host build.
struct CpuCard {
    /// Every host profile there is.
    profiles: Vec<String>,
    /// The profiles whose results are correctly rounded.
    exact_profiles: Vec<String>,
    /// The profiles that route the math to the vector library.
    vector_math_profiles: Vec<String>,
    /// The select weights, `float64` then `float32`.
    select_weight: [f64; 2],
}

/// The card of one instruction set: how its math library and its compilers behave.
struct IsaCard {
    /// Whether the vector math library is correctly rounded whatever the profile.
    libm_exact: bool,
    /// Whether a GCC compiler needs powers routed through the vector spelling.
    gcc_routes_pow: bool,
    /// Whether a vector math profile asked for by name needs a GCC compiler.
    vector_math_needs_gcc: bool,
    /// The default profile with a GCC compiler.
    default_gcc: String,
    /// The default profile with any other compiler.
    default_other: String,
    /// The vector instruction set of a machine that states this instruction set and no vector set.
    default_vector_isa: String,
    /// The vector instruction sets the card has a fusion row for.
    vector_isas: Vec<String>,
}

/// Loads the cpu card and checks that its exact profiles are among its profiles.
fn load_cpu(cards: &CardSet) -> Result<CpuCard, String> {
    let file = "cpu.json";
    let card = cards.json(file)?;
    closed(
        &card,
        file,
        &[
            "profiles",
            "exact_profiles",
            "vector_math_profiles",
            "select_weight",
        ],
    )?;
    let cpu = CpuCard {
        profiles: names_of(&card, file, "profiles")?,
        exact_profiles: names_of(&card, file, "exact_profiles")?,
        vector_math_profiles: names_of(&card, file, "vector_math_profiles")?,
        select_weight: weights_of(&card, file, None)?,
    };
    match cpu
        .exact_profiles
        .iter()
        .find(|p| !cpu.profiles.contains(p))
    {
        Some(name) => Err(format!("{file}: exact profile {name} is not a profile")),
        None => Ok(cpu),
    }
}

/// The cards of the instruction sets a host build knows: the name, the data file and the name of the instruction set in a record.
pub(crate) const ISA_CARDS: [(&str, &str, &str); 2] = [
    ("arm64", "arm.json", "Arm64"),
    ("x86_64", "x86.json", "X86_64"),
];

impl IsaCard {
    /// The profile a host build gets when none is asked, for a GCC compiler or another.
    fn default_profile(&self, gcc: bool) -> &String {
        if gcc {
            &self.default_gcc
        } else {
            &self.default_other
        }
    }
}

/// Whether the host profile `profile` gives bit-exact values: the cpu card lists the profiles whose results are correctly
/// rounded. The set itself is not handed out; a caller asks about the profile it was given.
///
/// # Errors
/// Refuses a cpu card that cannot be read.
pub fn profile_is_exact(cards: &CardSet, profile: &str) -> Result<bool, Failed> {
    let cpu = load_cpu(cards).map_err(|reason| cards.defect("cpu.json", reason))?;
    Ok(cpu.exact_profiles.iter().any(|exact| exact == profile))
}

/// The profile a host build gets when none is asked, for the instruction set `isa_name` and the compiler `compiler` (its
/// identity): the instruction set card's default for GCC or for another compiler.
///
/// # Errors
/// Refuses an instruction set without a card and a compiler identity that is not `<path> || <version line> || <digest>`.
pub fn default_host_profile(
    cards: &CardSet,
    compiler: &str,
    isa_name: &str,
) -> Result<String, Failed> {
    let Some(&(_, file, _)) = ISA_CARDS.iter().find(|card| card.0 == isa_name) else {
        return Err(Failed::Bad(format!(
            "no card for the instruction set {isa_name}"
        )));
    };
    let isa = load_isa(cards, file, isa_name).map_err(|reason| cards.defect(file, reason))?;
    Ok(isa.default_profile(identity_is_gcc(compiler)?).clone())
}

/// Loads the card of an instruction set, and checks that it names that instruction set.
fn load_isa(cards: &CardSet, file: &str, isa: &str) -> Result<IsaCard, String> {
    let card = cards.json(file)?;
    let keys = [
        "isa",
        "libm_exact",
        "gcc_routes_pow",
        "vector_math_needs_gcc",
        "default_profile",
        "default_vector_isa",
        "fusion_host",
        "host_flags",
    ];
    closed(&card, file, &keys)?;
    if text_of(&card, file, "isa")? != isa {
        return Err(format!("{file} does not describe {isa}"));
    }
    let defaults = member(&card, file, "default_profile")?;
    closed(defaults, file, &["gcc", "other"])?;
    Ok(IsaCard {
        libm_exact: flag_of(&card, file, "libm_exact")?,
        gcc_routes_pow: flag_of(&card, file, "gcc_routes_pow")?,
        vector_math_needs_gcc: flag_of(&card, file, "vector_math_needs_gcc")?,
        default_gcc: text_of(defaults, file, "gcc")?,
        default_other: text_of(defaults, file, "other")?,
        default_vector_isa: text_of(&card, file, "default_vector_isa")?,
        vector_isas: match member(&card, file, "fusion_host")? {
            Json::Obj(rows) => rows.iter().map(|(name, _)| name.clone()).collect(),
            _ => return Err(format!("{file}: fusion_host is not an object")),
        },
    })
}

/// The fusion row of the host that a facts answer describes: the row of its vector instruction set in the card of its
/// instruction set.
pub fn host_fusion(cards: &CardSet, answer: &str) -> Result<FusionRow, Failed> {
    let machine = Contract::parse(answer)
        .map_err(|e| Failed::Card(format!("facts answer: {e}")))?
        .0;
    let file = match machine.get("isa").and_then(Json::str) {
        Some("Arm64") => "arm.json",
        Some("X86_64") => "x86.json",
        _ => {
            return Err(Failed::Card(
                "the machine has no host instruction set".into(),
            ))
        }
    };
    let vector = machine
        .get("vector_isa")
        .and_then(Json::str)
        .ok_or_else(|| Failed::Card("the machine states no vector instruction set".into()))?;
    let row = |card: &Json| {
        fusion_row(
            member(member(card, file, "fusion_host")?, file, vector)?,
            file,
        )
    };
    cards
        .json(file)
        .and_then(|card| row(&card))
        .map_err(|reason| cards.defect(file, reason))
}

/// Whether a compiler identity (`<path> || <first version line> || <digest>`) names GCC proper: its version line says GCC or
/// G++ and does not say clang.
fn identity_is_gcc(identity: &str) -> Result<bool, Failed> {
    let parts: Vec<&str> = identity.split(" || ").collect();
    if parts.len() != 3 {
        return Err(Failed::Bad(
            "the compiler identity is not <path> || <version line> || <digest>".into(),
        ));
    }
    let line = parts[1].to_lowercase();
    Ok((line.contains("gcc") || line.contains("g++")) && !line.contains("clang"))
}

/// The release a compiler identity's version line ends with, as `major.minor` (`g++ (GCC) 13.2.0` gives `13.2`; a major alone gives
/// minor 0, as the compiler's own macros did); none when the line does not end with a number.
fn identity_release(identity: &str) -> Option<String> {
    let last = identity.split(" || ").nth(1)?.split_whitespace().last()?;
    let mut parts = last.split('.');
    let (major, minor) = (parts.next()?, parts.next().unwrap_or("0"));
    let number = |text: &str| !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit());
    (number(major) && number(minor)).then(|| format!("{major}.{minor}"))
}

/// The caches as the wire's record; absent when the machine reported none.
fn caches_json(caches: &Option<Caches>) -> Option<(String, Json)> {
    let caches = caches
        .as_ref()
        .filter(|c| c.l1d_bytes > 0 && c.l2_bytes > 0)?;
    let record = Json::Obj(vec![
        ("l1d_bytes".into(), number(caches.l1d_bytes)),
        ("l2_bytes".into(), number(caches.l2_bytes)),
        ("source".into(), Json::Str(caches.source.clone())),
    ]);
    Some(("caches".into(), record))
}

/// The machine record of a host build: the instruction set, the profile the compiler gets by default and what follows from it.
pub fn host_machine(
    cards: &CardSet,
    compiler: &str,
    probe: &Probe,
    stated: &Stated,
) -> Result<Json, Failed> {
    let isa_name = stated.isa.as_ref().unwrap_or(&probe.isa);
    let Some(&(_, file, member_name)) = ISA_CARDS.iter().find(|card| card.0 == isa_name) else {
        return Err(Failed::Bad(format!(
            "no card for the instruction set {isa_name}"
        )));
    };
    let cpu = load_cpu(cards)?;
    let isa = load_isa(cards, file, isa_name).map_err(|reason| cards.defect(file, reason))?;
    // the probe's vector set belongs to the probed instruction set: a stated instruction set takes its card's default
    let vector_isa = match (&stated.vector_isa, &stated.isa) {
        (Some(vector), _) => vector.clone(),
        (None, Some(_)) => isa.default_vector_isa.clone(),
        (None, None) => probe.vector_isa.clone(),
    };
    if !isa.vector_isas.contains(&vector_isa) {
        return Err(Failed::Bad(format!(
            "{file} has no fusion row for the vector instruction set {vector_isa:?}"
        )));
    }
    let gcc = identity_is_gcc(compiler)?;
    for named in [&isa.default_gcc, &isa.default_other] {
        if !cpu.profiles.contains(named) {
            return Err(format!("{file}: default profile {named} is not a profile").into());
        }
    }
    let profile = match &stated.host_profile {
        Some(asked) if !cpu.profiles.contains(asked) => {
            return Err(Failed::Bad(format!("unknown host profile {asked}")));
        }
        Some(asked) => asked,
        None => isa.default_profile(gcc),
    };
    if stated.host_profile.is_some()
        && isa.vector_math_needs_gcc
        && cpu.vector_math_profiles.contains(profile)
        && !gcc
    {
        return Err(Failed::Bad(format!(
            "host profile {profile} needs a GCC compiler"
        )));
    }
    let exact = cpu.exact_profiles.contains(profile);
    let mut members = vec![
        ("isa".to_string(), Json::Str(member_name.into())),
        ("vector_isa".into(), Json::Str(vector_isa)),
        ("exact_libm".into(), Json::Bool(isa.libm_exact || exact)),
        (
            "routed_pow".into(),
            Json::Bool(isa.gcc_routes_pow && gcc && !exact),
        ),
    ];
    members.extend(caches_json(&stated.caches.clone().or(probe.caches.clone())));
    members.push(("select_weight".into(), weights_json(cpu.select_weight)));
    let release = stated
        .gcc_release
        .clone()
        .or_else(|| identity_release(compiler));
    let headers = stated.headers.as_deref().ok_or_else(|| {
        Failed::Bad("stated machine: a host build needs headers (roots or served)".into())
    })?;
    let settings = host_build::Settings {
        profile,
        headers,
        gcc,
        opt_level: stated.opt_level.as_deref(),
        gcc_release: release.as_deref(),
        masked_call_witness: stated.masked_call_witness.as_deref(),
    };
    let vector = cpu.vector_math_profiles.contains(profile);
    members.push((
        "host_build".into(),
        host_build::host_build(cards, file, vector, &settings)?,
    ));
    Ok(Json::Obj(members))
}

#[cfg(test)]
mod tests {
    //! The cpu card the library ships.
    use super::*;

    /// Oracle: the cpu card shipped in `osprey_cpu/facts/cpu.json`, which lists `native` and `portable` as exact.
    #[test]
    fn a_profile_is_exact_when_the_cpu_card_lists_it() {
        let docs = CardSet::new(&crate::testing::shipped(), &[], crate::testing::ALL_CARDS);
        assert!(profile_is_exact(&docs, "native").unwrap());
        assert!(profile_is_exact(&docs, "portable").unwrap());
        assert!(!profile_is_exact(&docs, "native-vector-math").unwrap());
    }

    /// Oracle: the argv written out by hand below, the unit compile the shipped card states for clang on an arm64 machine with
    /// neon, the headers in roots, at the card's default level and profile.
    #[test]
    fn the_unit_compile_of_the_shipped_card_is_the_hand_written_argv() {
        use crate::host_state::{Probe, Stated};
        use osprey_lowering::facts::answer;
        let probe = Probe {
            isa: "arm64".into(),
            vector_isa: "neon".into(),
            caches: None,
        };
        let stated = Stated {
            headers: Some("roots".into()),
            ..Stated::default()
        };
        let compiler = "/usr/bin/c++ || Apple clang version 21.0.0 || 0";
        let text = answer(
            &crate::testing::shipped(),
            &crate::Cpu,
            compiler,
            &[],
            &probe,
            &stated,
        )
        .unwrap();
        let machine = aether_writers::contract::Contract::parse(&text).unwrap().0;
        let argv: Vec<&str> = machine
            .get("host_build")
            .and_then(|b| b.get("compile_unit"))
            .unwrap()
            .arr()
            .iter()
            .map(|a| a.str().unwrap())
            .collect();
        assert_eq!(
            argv,
            [
                "{compiler}",
                "-O3",
                "-ffp-contract=fast",
                "-fno-math-errno",
                "-mcpu=native",
                "-std=c++23",
                "-fPIC",
                "-DAETHER_CPP_MODE",
                "-I",
                "{include:aether}",
                "-I",
                "{include:eagle}",
                "{defines}",
                "-I{include:facts}",
                "-MD",
                "-MF",
                "{depfile}",
                "-c",
                "{source}",
                "-o",
                "{object}"
            ]
        );
    }
}
