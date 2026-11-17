//! The machine facts the writers need, read from data files and from a probe of the running machine. The files (cpu, arm,
//! x86, cuda: JSON, in a directory that ships beside the library) state the profile lists, the per-instruction-set rules and
//! the select weights; the probe states the instruction set and the cache sizes. The answer is the JSON of the typed
//! contract's `machine` record, which the caller hands back to a render without reading it. A file that does not match its
//! schema is refused by name at load.
use crate::host_build;
use aether_writers::contract::types::Caches;
use aether_writers::contract::{Contract, Json};
pub use raptor_wings::chain::FusionRow;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::path::{Path, PathBuf};

/// What the running machine reports: its instruction set (as the arm and x86 cards name it) and its cache sizes.
pub struct Probe {
    /// The instruction set name: `arm64`, `x86_64`, or the platform's own name for any other.
    pub isa: String,
    /// The vector instruction set: `neon`, `sse2`, `avx2` or `avx512`; empty where the probe cannot tell.
    pub vector_isa: String,
    /// The first and second level data cache sizes, when the machine reports both.
    pub caches: Option<Caches>,
}

/// A machine the caller states, for a build that is not for the machine running: every field absent means the probed machine's
/// value (or the card's default). The host target takes `isa`, `caches`, `host_profile` and the build settings `headers` (required),
/// `opt_level`, `gcc_release` and `masked_call_witness`; the cuda target takes
/// `device_arch` and `device_release`.
#[derive(Default)]
pub struct Stated {
    /// The instruction set name.
    pub isa: Option<String>,
    /// The vector instruction set.
    pub vector_isa: Option<String>,
    /// The first and second level data cache sizes with where they came from.
    pub caches: Option<Caches>,
    /// The host profile asked for, in place of the compiler's default.
    pub host_profile: Option<String>,
    /// The device architecture (for example `sm_61`), which selects the rows of the device card.
    pub device_arch: Option<String>,
    /// The release of the device compiler as `major.minor`, in place of the one its identity's version line names. A compiler whose
    /// first version line carries no release (the device compiler driver prints it on a later line) must state it.
    pub device_release: Option<String>,
    /// Where the headers of a host build are: `roots` or `served`. A host answer needs it.
    pub headers: Option<String>,
    /// The optimisation level of the host build (`O0` to `O3`), in place of the default.
    pub opt_level: Option<String>,
    /// The release of a GCC compiler as `major.minor`, in place of the one its identity's version line ends with.
    pub gcc_release: Option<String>,
    /// What the masked-call witness did on the compiler: `fixed`, `miscompile` or `unrun`; none counts as `unrun`.
    pub masked_call_witness: Option<String>,
}

/// Why a facts question failed: the caller asked something that cannot be answered (a malformed stated machine, an unknown target
/// or profile), or a data file of the library is broken.
#[derive(Debug)]
pub enum Failed {
    /// The question is malformed or names what the cards do not have.
    Bad(String),
    /// A data file of the library cannot be read or does not hold what its schema says.
    Card(String),
}

impl From<String> for Failed {
    /// A failure of a data file, which is what every plain text failure of the loaders is.
    fn from(reason: String) -> Failed {
        Failed::Card(reason)
    }
}

impl From<Failed> for String {
    /// The reason in words.
    fn from(failed: Failed) -> String {
        match failed {
            Failed::Bad(reason) | Failed::Card(reason) => reason,
        }
    }
}

/// The stated machine of its JSON record (`isa`, `l1d_bytes`, `l2_bytes`, `cache_source`, `host_profile`, `device_arch`, `headers`, `opt_level`, `gcc_release`,
/// `masked_call_witness`, each optional); a key outside these, a value of the wrong kind and a cache record with a part missing are refused by name.
pub fn stated_from_json(text: &str) -> Result<Stated, Failed> {
    read_stated(text).map_err(Failed::Bad)
}

/// The stated machine of its JSON record, a failure as text.
fn read_stated(text: &str) -> Result<Stated, String> {
    let keys = [
        "isa",
        "vector_isa",
        "l1d_bytes",
        "l2_bytes",
        "cache_source",
        "host_profile",
        "device_arch",
        "device_release",
        "headers",
        "opt_level",
        "gcc_release",
        "masked_call_witness",
    ];
    let record = Contract::parse(text)
        .map_err(|e| format!("stated machine: {e}"))?
        .0;
    let Json::Obj(members) = &record else {
        return Err("stated machine: not an object".into());
    };
    if let Some((key, _)) = members
        .iter()
        .find(|(key, _)| !keys.contains(&key.as_str()))
    {
        return Err(format!("stated machine: unknown key {key}"));
    }
    let name = |key: &str| match record.get(key) {
        None => Ok(None),
        Some(Json::Str(text)) => Ok(Some(text.clone())),
        Some(_) => Err(format!("stated machine: {key} is not a text")),
    };
    let size = |key: &str| match record.get(key) {
        None => Ok(None),
        Some(value) => value
            .int()
            .filter(|n| *n > 0)
            .map(Some)
            .ok_or_else(|| format!("stated machine: {key} is not a positive integer")),
    };
    let caches = match (size("l1d_bytes")?, size("l2_bytes")?, name("cache_source")?) {
        (None, None, None) => None,
        (Some(l1d_bytes), Some(l2_bytes), Some(source)) => Some(Caches {
            l1d_bytes,
            l2_bytes,
            source,
        }),
        _ => return Err("stated machine: caches need l1d_bytes, l2_bytes and cache_source".into()),
    };
    Ok(Stated {
        isa: name("isa")?,
        vector_isa: name("vector_isa")?,
        caches,
        host_profile: name("host_profile")?,
        device_arch: name("device_arch")?,
        device_release: name("device_release")?,
        headers: name("headers")?,
        opt_level: name("opt_level")?,
        gcc_release: name("gcc_release")?,
        masked_call_witness: name("masked_call_witness")?,
    })
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
    /// The vector instruction sets the card has a fusion row for.
    vector_isas: Vec<String>,
}

/// Reads and parses one data file of the facts directory.
pub(crate) fn read_file(dir: &Path, file: &str) -> Result<Json, String> {
    let text =
        std::fs::read_to_string(dir.join(file)).map_err(|e| format!("cannot read {file}: {e}"))?;
    Ok(Contract::parse(&text)
        .map_err(|e| format!("{file}: {e}"))?
        .0)
}

/// The members of a card as a closed set: a key outside `keys` is refused by name.
pub(crate) fn closed<'a>(
    card: &'a Json,
    file: &str,
    keys: &[&str],
) -> Result<&'a [(String, Json)], String> {
    let Json::Obj(members) = card else {
        return Err(format!("{file} is not an object"));
    };
    match members
        .iter()
        .find(|(key, _)| !keys.contains(&key.as_str()))
    {
        Some((key, _)) => Err(format!("{file} has the unknown key {key}")),
        None => Ok(members),
    }
}

/// The value of a key a card must hold.
pub(crate) fn member<'a>(card: &'a Json, file: &str, key: &str) -> Result<&'a Json, String> {
    card.get(key)
        .ok_or_else(|| format!("{file} lacks the key {key}"))
}

/// A key of a card that must be a text.
pub(crate) fn text_of(card: &Json, file: &str, key: &str) -> Result<String, String> {
    match member(card, file, key)? {
        Json::Str(text) => Ok(text.clone()),
        _ => Err(format!("{file}: {key} is not a text")),
    }
}

/// A key of a card that must be a flag.
fn flag_of(card: &Json, file: &str, key: &str) -> Result<bool, String> {
    match member(card, file, key)? {
        Json::Bool(flag) => Ok(*flag),
        _ => Err(format!("{file}: {key} is not a flag")),
    }
}

/// A key of a card that must be a list of texts.
pub(crate) fn names_of(card: &Json, file: &str, key: &str) -> Result<Vec<String>, String> {
    let Json::Arr(items) = member(card, file, key)? else {
        return Err(format!("{file}: {key} is not a list"));
    };
    let name = |item: &Json| item.str().map(String::from);
    items
        .iter()
        .map(name)
        .collect::<Option<_>>()
        .ok_or_else(|| format!("{file}: {key} holds an item that is not a text"))
}

/// A number of a card that must be positive.
fn positive(value: &Json, what: &str) -> Result<f64, String> {
    match value {
        Json::Num(n) => n
            .parse::<f64>()
            .ok()
            .filter(|w| *w > 0.0)
            .ok_or_else(|| format!("{what} is not positive")),
        _ => Err(format!("{what} is not a number")),
    }
}

/// The `select_weight` key of a card: the weights for `float64` and `float32` (`defaults`, each positive), replaced for a
/// device `arch` by the rows (`cc`, `dtype`, `select_weight`) that name it.
fn weights_of(card: &Json, file: &str, arch: Option<&str>) -> Result<[f64; 2], String> {
    let table = member(card, file, "select_weight")?;
    closed(table, file, &["defaults", "rows"])?;
    let defaults = member(table, file, "defaults")?;
    closed(defaults, file, &["float64", "float32"])?;
    let default = |scalar: &str| {
        positive(
            member(defaults, file, scalar)?,
            &format!("{file}: select_weight {scalar}"),
        )
    };
    let mut weights = [default("float64")?, default("float32")?];
    for row in table.get("rows").map_or(&[][..], Json::arr) {
        closed(row, file, &["cc", "dtype", "select_weight"])?;
        let weight = positive(
            member(row, file, "select_weight")?,
            &format!("{file}: a select_weight row"),
        )?;
        let at = match text_of(row, file, "dtype")?.as_str() {
            "float64" => 0,
            "float32" => 1,
            other => {
                return Err(format!(
                    "{file}: a select_weight row names the type {other}"
                ))
            }
        };
        if arch == Some(text_of(row, file, "cc")?.as_str()) {
            weights[at] = weight;
        }
    }
    Ok(weights)
}

/// Loads the cpu card and checks that its exact profiles are among its profiles.
fn load_cpu(dir: &Path) -> Result<CpuCard, String> {
    let file = "cpu.json";
    let card = read_file(dir, file)?;
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

/// Loads the card of an instruction set, and checks that it names that instruction set.
fn load_isa(dir: &Path, file: &str, isa: &str) -> Result<IsaCard, String> {
    let card = read_file(dir, file)?;
    let keys = [
        "isa",
        "libm_exact",
        "gcc_routes_pow",
        "vector_math_needs_gcc",
        "default_profile",
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
        vector_isas: match member(&card, file, "fusion_host")? {
            Json::Obj(rows) => rows.iter().map(|(name, _)| name.clone()).collect(),
            _ => return Err(format!("{file}: fusion_host is not an object")),
        },
    })
}

/// A number of a fusion row that is a whole number, or null for a cost that is not measured.
fn optional_cost(row: &Json, file: &str, key: &str) -> Result<Option<f64>, String> {
    match member(row, file, key)? {
        Json::Null => Ok(None),
        value => positive(value, &format!("{file}: fusion row {key}")).map(Some),
    }
}

/// The fusion row of the host that a facts answer describes: the row of its vector instruction set in the card of its
/// instruction set.
pub fn host_fusion(dir: &Path, answer: &str) -> Result<FusionRow, String> {
    let machine = Contract::parse(answer)
        .map_err(|e| format!("facts answer: {e}"))?
        .0;
    let file = match machine.get("isa").and_then(Json::str) {
        Some("Arm64") => "arm.json",
        Some("X86_64") => "x86.json",
        _ => return Err("the machine has no host instruction set".into()),
    };
    let vector = machine
        .get("vector_isa")
        .and_then(Json::str)
        .ok_or("the machine states no vector instruction set")?;
    let card = read_file(dir, file)?;
    fusion_row(
        member(member(&card, file, "fusion_host")?, file, vector)?,
        file,
    )
}

/// The fusion row of the device card: the cost of a launch, of a plane byte and of local memory, and the registers a lane may hold.
/// The card states one row; it has no per-architecture rows yet.
pub fn device_fusion(dir: &Path) -> Result<FusionRow, String> {
    let file = "cuda.json";
    fusion_row(member(&read_file(dir, file)?, file, "fusion")?, file)
}

/// A fusion row of a card file, its keys closed and its numbers checked.
fn fusion_row(row: &Json, file: &str) -> Result<FusionRow, String> {
    let keys = [
        "regs_base",
        "regs_file",
        "launch_us",
        "plane_us_per_byte",
        "local_over_plane",
    ];
    closed(row, file, &keys)?;
    let whole = |key: &str| match member(row, file, key)?.int() {
        Some(n) if n >= 0 => Ok(n),
        _ => Err(format!("{file}: fusion row {key} is not a whole number")),
    };
    Ok(FusionRow {
        regs_base: whole("regs_base")?,
        regs_file: whole("regs_file")?,
        launch_us: optional_cost(row, file, "launch_us")?,
        plane_us_per_byte: optional_cost(row, file, "plane_us_per_byte")?,
        local_over_plane: optional_cost(row, file, "local_over_plane")?,
    })
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

/// A number as the wire writes it.
fn number(n: impl ToString) -> Json {
    Json::Num(n.to_string())
}

/// The select weights as the wire's record.
fn weights_json(weights: [f64; 2]) -> Json {
    Json::Obj(vec![
        ("float64".into(), number(weights[0])),
        ("float32".into(), number(weights[1])),
    ])
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
fn host_machine(
    dir: &Path,
    compiler: &str,
    probe: &Probe,
    stated: &Stated,
) -> Result<Json, Failed> {
    if stated.device_arch.is_some() || stated.device_release.is_some() {
        return Err(Failed::Bad(
            "stated machine: device_arch and device_release are for the cuda target".into(),
        ));
    }
    let isa_name = stated.isa.as_ref().unwrap_or(&probe.isa);
    let Some(&(_, file, member_name)) = ISA_CARDS.iter().find(|card| card.0 == isa_name) else {
        return Err(Failed::Bad(format!(
            "no card for the instruction set {isa_name}"
        )));
    };
    let cpu = load_cpu(dir)?;
    let isa = load_isa(dir, file, isa_name)?;
    let vector_isa = stated
        .vector_isa
        .clone()
        .unwrap_or(probe.vector_isa.clone());
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
        None if gcc => &isa.default_gcc,
        None => &isa.default_other,
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
        host_build::host_build(dir, file, vector, &settings)?,
    ));
    Ok(Json::Obj(members))
}

/// The tool and the release of a device compiler identity (`<path> || <version line> || <digest>`): the file name of the path and
/// the number after `release` in the version line (`Cuda compilation tools, release 12.6, V12.6.77` gives `nvcc`, `12.6`).
fn device_tool(identity: &str, stated: Option<&str>) -> Result<(String, String), Failed> {
    let parts: Vec<&str> = identity.split(" || ").collect();
    if parts.len() != 3 {
        return Err(Failed::Bad(
            "the compiler identity is not <path> || <version line> || <digest>".into(),
        ));
    }
    let tool = parts[0].rsplit('/').next().unwrap_or(parts[0]);
    let release: String = match stated {
        Some(release) => {
            let numbers: Vec<&str> = release.split('.').collect();
            let whole = |n: &&str| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit());
            if numbers.len() != 2 || !numbers.iter().all(whole) {
                return Err(Failed::Bad(format!(
                    "stated machine: device_release {release:?} is not major.minor"
                )));
            }
            release.into()
        }
        None => parts[1]
            .split("release ")
            .nth(1)
            .unwrap_or("")
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect(),
    };
    if tool.is_empty() || release.is_empty() {
        return Err(Failed::Bad(format!(
            "facts: the device compiler line {:?} names no tool or no release, and no device_release is stated",
            parts[1]
        )));
    }
    Ok((tool.into(), release))
}

/// A list of texts of a card, each with the architecture's names put in: `{sm}` is the physical name (`sm_80`) and `{virtual}` the
/// virtual one (`compute_80`), both made from the number the stated architecture ends in (`sm_80`, `compute_80` or `80a`).
fn flags_of(card: &Json, file: &str, key: &str, arch: &str) -> Result<Json, String> {
    let number = &arch[arch
        .find(|c: char| c.is_ascii_digit())
        .unwrap_or(arch.len())..];
    if number.is_empty() {
        return Err(format!("facts: {arch} is not a device architecture name"));
    }
    let (sm, virtual_arch) = (format!("sm_{number}"), format!("compute_{number}"));
    let mut flags = vec![];
    for flag in member(card, file, key)?.arr() {
        let Json::Str(text) = flag else {
            return Err(format!("{file}: {key} holds a flag that is not a text"));
        };
        let put = text
            .replace("{sm}", &sm)
            .replace("{virtual}", &virtual_arch);
        flags.push(Json::Str(put));
    }
    Ok(Json::Arr(flags))
}

/// The object formats an image can be embedded as, as `cuda_image.json` names them.
const OBJECT_FORMATS: [&str; 2] = ["elf", "macho"];

/// The recipe that turns an image file into an object of the unit library: the argv that assembles a source with the host compiler,
/// and the assembler source for each image format and object format, with its places `{symbol}`, `{size_symbol}` and
/// `{image_path}` left for the caller. The image's tail (a text image ends with a zero byte after its stated size) is written in
/// by the image format. An image format or an object format the data has no recipe for is refused by name.
fn embed_recipe(card: &Json, file: &str, formats: &[&str]) -> Result<Json, Failed> {
    let embed = member(card, file, "embed")?;
    closed(embed, file, &["compile", "image_tail", "object_formats"])?;
    let tails = member(embed, file, "image_tail")?;
    let objects = member(embed, file, "object_formats")?;
    closed(objects, file, &OBJECT_FORMATS)?;
    let mut by_image = vec![];
    for format in formats {
        if tails.get(format).is_none() {
            return Err(Failed::Bad(format!(
                "facts: {file} has no embed recipe for the image format {format}"
            )));
        }
        let tail = names_of(tails, file, format)?.join("\n");
        let mut by_object = vec![];
        for object in OBJECT_FORMATS {
            if objects.get(object).is_none() {
                return Err(Failed::Bad(format!(
                    "facts: {file} has no embed recipe for the object format {object}"
                )));
            }
            let mut source = String::new();
            for line in names_of(objects, file, object)? {
                match (line.as_str(), tail.is_empty()) {
                    ("{tail}", true) => {}
                    ("{tail}", false) => source += &format!("{tail}\n"),
                    _ => source += &format!("{line}\n"),
                }
            }
            by_object.push((object.to_string(), Json::Str(source)));
        }
        by_image.push((format.to_string(), Json::Obj(by_object)));
    }
    Ok(Json::Obj(vec![
        ("compile".into(), flags_of_names(embed, file, "compile")?),
        ("source".into(), Json::Obj(by_image)),
    ]))
}

/// A key of a card that is a list of texts, as a JSON list.
fn flags_of_names(card: &Json, file: &str, key: &str) -> Result<Json, String> {
    Ok(Json::Arr(
        names_of(card, file, key)?
            .into_iter()
            .map(Json::Str)
            .collect(),
    ))
}

/// The compile command of a device build as data: the compiler's short name (the tool and its release), the PTX version its
/// release writes, the flags of an image of each format for the architecture and the recipe that embeds an image file in the unit
/// library (`embed`), all from `cuda_image.json`. The flags carry no
/// host compiler: the caller that runs the command adds its own (`-ccbin`). A release with no row is refused by name.
fn device_build(
    dir: &Path,
    compiler: &str,
    arch: &str,
    release: Option<&str>,
) -> Result<Json, Failed> {
    let file = "cuda_image.json";
    let (tool, release) = device_tool(compiler, release)?;
    let card = read_file(dir, file)?;
    closed(&card, file, &["ptx_isa", "flags", "embed"])?;
    let isa = match member(&card, file, "ptx_isa")?.get(&release).and_then(Json::int) {
        Some(isa) if isa > 0 => isa,
        _ => {
            return Err(Failed::Bad(format!(
                "facts: the device compiler release {release} has no PTX version: add its row to {file}"
            )))
        }
    };
    let formats = member(&card, file, "flags")?;
    let names = ["ptx", "cubin", "fatbin"];
    closed(formats, file, &names)?;
    let mut flags = vec![];
    for name in names {
        flags.push((name.to_string(), flags_of(formats, file, name, arch)?));
    }
    Ok(Json::Obj(vec![
        ("compiler".into(), Json::Str(format!("{tool} {release}"))),
        ("ptx_isa".into(), number(isa)),
        ("flags".into(), Json::Obj(flags)),
        ("embed".into(), embed_recipe(&card, file, &names)?),
    ]))
}

/// The machine record of a device build: no instruction set, no caches, the device card's select weights, the architecture the
/// build is for (the one stated, else the card's default), which the lowering reads and the writers do not, and, last, the
/// compile command (`device_build`). One answer states one command: every unit built from it carries the same compiler, flags and
/// PTX version.
fn device_machine(dir: &Path, compiler: &str, stated: &Stated) -> Result<Json, Failed> {
    if stated.isa.is_some() || stated.caches.is_some() || stated.host_profile.is_some() {
        return Err(Failed::Bad(
            "stated machine: only device_arch and device_release are for the cuda target".into(),
        ));
    }
    let file = "cuda.json";
    let card = read_file(dir, file)?;
    closed(&card, file, &["select_weight", "default_arch", "fusion"])?;
    let arch = match &stated.device_arch {
        Some(asked) => asked.clone(),
        None => text_of(&card, file, "default_arch")?,
    };
    let build = device_build(dir, compiler, &arch, stated.device_release.as_deref())?;
    Ok(Json::Obj(vec![
        ("exact_libm".into(), Json::Bool(true)),
        ("routed_pow".into(), Json::Bool(false)),
        (
            "select_weight".into(),
            weights_json(weights_of(&card, file, stated.device_arch.as_deref())?),
        ),
        ("device_arch".into(), Json::Str(arch)),
        ("device_build".into(), build),
    ]))
}

/// The facts answer for a named target (`host` or `cuda`) and a compiler identity, from the cards in `dir`, the probe and the
/// machine the caller states (empty: the probed machine):
/// the JSON of the contract's `machine` record. A target without facts, a card that does not match its schema and an
/// identity that is not well formed are refused with the reason.
pub fn answer(
    dir: &Path,
    target: &str,
    compiler: &str,
    probe: &Probe,
    stated: &Stated,
) -> Result<String, Failed> {
    let machine = match target {
        "host" => host_machine(dir, compiler, probe, stated)?,
        "cuda" => device_machine(dir, compiler, stated)?,
        other => return Err(Failed::Bad(format!("no facts for the target {other}"))),
    };
    Ok(machine.to_text())
}

/// The value of a macOS system setting (a number of one to eight bytes), when it reads.
#[cfg(target_os = "macos")]
fn sysctl_number(name: &str) -> Option<i64> {
    extern "C" {
        /// The system call that reads a named setting.
        fn sysctlbyname(
            name: *const c_char,
            old: *mut c_void,
            old_len: *mut usize,
            new: *mut c_void,
            new_len: usize,
        ) -> c_int;
    }
    let name = std::ffi::CString::new(name).ok()?;
    let mut bytes = [0u8; 8];
    let mut len = bytes.len();
    // SAFETY: `bytes` holds `len` bytes, and the name is a NUL-terminated string.
    let status = unsafe {
        sysctlbyname(
            name.as_ptr(),
            bytes.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0 && len > 0).then(|| i64::from_le_bytes(bytes))
}

/// The cache sizes of a macOS machine: the performance cores' where it has two kinds of core, else the plain ones.
#[cfg(target_os = "macos")]
fn probe_caches() -> Option<Caches> {
    ["hw.perflevel0.", "hw."].iter().find_map(|prefix| {
        let l1d_bytes = sysctl_number(&format!("{prefix}l1dcachesize"))?;
        let l2_bytes = sysctl_number(&format!("{prefix}l2cachesize"))?;
        (l1d_bytes > 0 && l2_bytes > 0).then(|| Caches {
            l1d_bytes,
            l2_bytes,
            source: format!("sysctl {prefix}l1dcachesize/l2cachesize"),
        })
    })
}

/// A Linux cache size field (`32K`, `1024K`, `2M`) in bytes.
#[cfg(target_os = "linux")]
fn cache_bytes(text: &str) -> Option<i64> {
    let text = text.trim().to_uppercase();
    let scale = match text.chars().last()? {
        'K' => 1024,
        'M' => 1024 * 1024,
        'G' => 1024 * 1024 * 1024,
        _ => 1,
    };
    Some(text.trim_end_matches(['K', 'M', 'G']).parse::<i64>().ok()? * scale)
}

/// The cache sizes of a Linux machine from the first processor's cache directories: the level 1 data cache and the level 2
/// cache. Written from the kernel's file layout and not run on a Linux machine.
#[cfg(target_os = "linux")]
fn probe_caches() -> Option<Caches> {
    let root = "/sys/devices/system/cpu/cpu0/cache";
    let field = |entry: &Path, name: &str| {
        std::fs::read_to_string(entry.join(name))
            .ok()
            .map(|text| text.trim().to_string())
    };
    let mut entries: Vec<PathBuf> = std::fs::read_dir(root)
        .ok()?
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("index"))
        })
        .collect();
    entries.sort();
    let mut sizes = [None, None];
    for entry in &entries {
        let level = field(entry, "level")?.parse::<usize>().ok()?;
        let kind = field(entry, "type")?;
        let data = if level == 1 {
            kind == "Data"
        } else {
            kind != "Instruction"
        };
        if (level == 1 || level == 2) && data && sizes[level - 1].is_none() {
            sizes[level - 1] = cache_bytes(&field(entry, "size")?);
        }
    }
    let [Some(l1d_bytes), Some(l2_bytes)] = sizes else {
        return None;
    };
    (l1d_bytes > 0 && l2_bytes > 0).then(|| Caches {
        l1d_bytes,
        l2_bytes,
        source: root.into(),
    })
}

/// No cache sizes are probed on any other platform.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn probe_caches() -> Option<Caches> {
    None
}

/// Asks the running machine for its instruction set and cache sizes.
pub fn probe() -> Probe {
    let isa = match std::env::consts::ARCH {
        "aarch64" => "arm64",
        other => other,
    };
    Probe {
        isa: isa.into(),
        vector_isa: probe_vector_isa().into(),
        caches: probe_caches(),
    }
}

/// The widest vector instruction set the running machine has. The x86 branch is written from the feature names and not run on an
/// x86 machine.
fn probe_vector_isa() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    return "neon";
    #[cfg(target_arch = "x86_64")]
    return if is_x86_feature_detected!("avx512f") {
        "avx512"
    } else if is_x86_feature_detected!("avx2") {
        "avx2"
    } else {
        "sse2"
    };
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    return "";
}

/// The directory of the data files: the one the caller gives (which must be a directory), else `facts` beside the library file that
/// holds this code.
pub fn facts_dir(given: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(dir) = given {
        return if dir.is_dir() {
            Ok(dir.to_path_buf())
        } else {
            Err(format!("facts: {} is not a directory", dir.display()))
        };
    }
    /// The loader's description of an address, as `dladdr` fills it.
    #[repr(C)]
    struct DlInfo {
        /// The file the address lies in.
        file: *const c_char,
        /// Where that file is loaded.
        base: *mut c_void,
        /// The nearest symbol below the address.
        symbol: *const c_char,
        /// That symbol's address.
        symbol_address: *mut c_void,
    }
    extern "C" {
        /// Describes the file an address lies in.
        fn dladdr(address: *const c_void, info: *mut DlInfo) -> c_int;
    }
    let mut info = DlInfo {
        file: std::ptr::null(),
        base: std::ptr::null_mut(),
        symbol: std::ptr::null(),
        symbol_address: std::ptr::null_mut(),
    };
    // SAFETY: the address is one of this library's own functions and `info` is the structure `dladdr` fills.
    let found = unsafe { dladdr(facts_dir as *const c_void, &mut info) };
    if found == 0 || info.file.is_null() {
        return Err("cannot tell where the library is".into());
    }
    // SAFETY: `dladdr` answered a NUL-terminated file name.
    let file = unsafe { CStr::from_ptr(info.file) }
        .to_string_lossy()
        .into_owned();
    Path::new(&file)
        .parent()
        .map(|dir| dir.join("facts"))
        .ok_or_else(|| "the library has no directory".into())
}
