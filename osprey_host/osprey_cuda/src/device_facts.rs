//! The device's facts: the machine record of the device target, its fusion rows, the compile and embed recipes of its image and
//! the compilers a machine without a tool is written under, read from the library's own documents.
use crate::device_build;
use crate::device_state::Stated;
use aether_writers::contract::Json;
use osprey_lowering::facts::{
    closed, fusion_row, member, names_of, number, text_of, weights_json, weights_of, CardSet,
    Failed, FusionRow,
};

/// The optimisation level a build gets when none is asked: the `default_opt_level` of the document that states the compile of the
/// device (`cuda_image.json`).
///
/// # Errors
/// Refuses a document that does not state it.
pub fn default_opt_level(cards: &CardSet) -> Result<String, Failed> {
    let file = "cuda_image.json";
    Ok(text_of(&cards.json(file)?, file, "default_opt_level")?)
}

/// The optimisation levels the document that states the compile of the device (`cuda_image.json`) lists.
///
/// # Errors
/// Refuses a document that does not list them.
pub fn opt_levels(cards: &CardSet) -> Result<Vec<String>, Failed> {
    let file = "cuda_image.json";
    Ok(names_of(&cards.json(file)?, file, "opt_levels")?)
}

/// The fusion rows of the device document: the row every architecture and scalar type takes unless a row is stated for it, and the
/// rows stated for a compute capability (`cc`, such as `sm_61`) and a scalar type (`dtype`).
pub struct DeviceFusion {
    /// The row of a build that no stated row serves.
    pub defaults: FusionRow,
    /// The stated rows as (cc, dtype, row); a row states only the numbers it changes, the rest are the defaults'.
    pub rows: Vec<(String, String, FusionRow)>,
}

/// The fusion rows of the device document: the cost of a launch, of a plane byte and of local memory, and the registers a lane may
/// hold. Every row is checked as the default row is; a row's `dtype` other than `float32` and `float64` is refused by name.
pub fn device_fusion(cards: &CardSet) -> Result<DeviceFusion, Failed> {
    device_fusion_of_document(cards).map_err(|reason| cards.defect("cuda.json", reason))
}

/// The rows of `device_fusion` as the reader of the document `cuda.json` finds them, a defect as its reason.
fn device_fusion_of_document(cards: &CardSet) -> Result<DeviceFusion, String> {
    let file = "cuda.json";
    let document = cards.json(file)?;
    let defaults = member(&document, file, "fusion")?;
    let mut rows = vec![];
    for row in member(&document, file, "fusion_rows")?.arr() {
        let own = ["cc", "dtype"];
        let Json::Obj(stated) = row else {
            return Err(format!("{file}: a fusion row is not an object"));
        };
        let Json::Obj(merged) = defaults.clone() else {
            return Err(format!("{file}: fusion is not an object"));
        };
        let mut merged = merged;
        for (key, value) in stated
            .iter()
            .filter(|(key, _)| !own.contains(&key.as_str()))
        {
            match merged.iter_mut().find(|(held, _)| held == key) {
                Some(slot) => slot.1 = value.clone(),
                None => merged.push((key.clone(), value.clone())),
            }
        }
        let dtype = text_of(row, file, "dtype")?;
        if !["float32", "float64"].contains(&dtype.as_str()) {
            return Err(format!("{file}: a fusion row names the type {dtype}"));
        }
        closed(
            row,
            file,
            &[
                "cc",
                "dtype",
                "regs_base",
                "regs_file",
                "launch_us",
                "plane_us_per_byte",
                "local_over_plane",
            ],
        )?;
        rows.push((
            text_of(row, file, "cc")?,
            dtype,
            fusion_row(&Json::Obj(merged), file)?,
        ));
    }
    Ok(DeviceFusion {
        defaults: fusion_row(defaults, file)?,
        rows,
    })
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

/// The physical (`sm_80`) and the virtual (`compute_80`) name of a device architecture, both made from the number the stated
/// name ends in (`sm_80`, `compute_80` or `80a`).
pub(crate) fn arch_names(arch: &str) -> Result<(String, String), String> {
    let number = &arch[arch
        .find(|c: char| c.is_ascii_digit())
        .unwrap_or(arch.len())..];
    if number.is_empty() {
        return Err(format!("facts: {arch} is not a device architecture name"));
    }
    Ok((format!("sm_{number}"), format!("compute_{number}")))
}

/// A list of texts of a card, each with the architecture's names put in: `{sm}` is the physical name (`sm_80`) and `{virtual}` the
/// virtual one (`compute_80`), both made from the number the stated architecture ends in (`sm_80`, `compute_80` or `80a`).
pub(crate) fn flags_of(card: &Json, file: &str, key: &str, arch: &str) -> Result<Json, String> {
    let (sm, virtual_arch) = arch_names(arch)?;
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

/// The operating systems an image can be embedded for, as `cuda_image.json` and the link recipe of `host_build.json` name them.
const SYSTEMS: [&str; 2] = ["linux", "macos"];

/// The recipe that turns an image file into an object of the unit library: the argv that assembles a source with the host
/// compiler (`{host_compiler}`, the compiler of the host build, not the device compiler), and the assembler source for each
/// operating system, with its places `{symbol}`, `{size_symbol}` and `{image_path}` left for the caller. The image is embedded as
/// its file holds it, whatever its form, and the size symbol counts exactly those bytes; the loader that reads an image
/// terminates it itself. An operating system the data has no recipe for is refused by name.
fn embed_recipe(card: &Json, file: &str) -> Result<Json, Failed> {
    let embed = member(card, file, "embed")?;
    closed(embed, file, &["compile", "source"])?;
    let systems = member(embed, file, "source")?;
    closed(systems, file, &SYSTEMS)?;
    let mut by_system = vec![];
    for system in SYSTEMS {
        if systems.get(system).is_none() {
            return Err(Failed::Bad(format!(
                "facts: {file} has no embed recipe for the operating system {system}"
            )));
        }
        let source: String = names_of(systems, file, system)?
            .iter()
            .map(|line| format!("{line}\n"))
            .collect();
        by_system.push((system.to_string(), Json::Str(source)));
    }
    Ok(Json::Obj(vec![
        ("compile".into(), flags_of_names(embed, file, "compile")?),
        ("source".into(), Json::Obj(by_system)),
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

/// The compiler identity a device unit's text is written under when the machine has neither a device compiler nor a runtime compile
/// library: writing text needs no tool, only the release whose PTX version the unit states, which `cuda_image.json` names
/// (`text_compiler`).
///
/// # Errors
/// Refuses a card without the member.
pub fn device_text_compiler(cards: &CardSet) -> Result<String, Failed> {
    let file = "cuda_image.json";
    let card = cards.json(file)?;
    text_of(&card, file, "text_compiler").map_err(|reason| cards.defect(file, reason))
}

/// The compiler identity a device build is made under when the runtime compile library is the machine's device tool: the card's
/// own statement of it (`runtime.compiler`), since the library has no executable to identify.
///
/// # Errors
/// Refuses a card without the member.
pub fn device_runtime_compiler(cards: &CardSet) -> Result<String, Failed> {
    let file = "cuda_image.json";
    let card = cards.json(file)?;
    let record = device_build::runtime_record(&card, file)?;
    text_of(record, file, "compiler").map_err(|reason| cards.defect(file, reason))
}

/// The compile command of a device build as data: the compiler's short name (the tool and its release), the PTX version its
/// release writes, the form of image the build makes and the argv template of the compile that makes it (`device_build.rs`), the
/// flags of an image of each format for the architecture and the recipe that embeds an image file in the unit library
/// (`embed`), all from `cuda_image.json`. The device compiler's host compiler is the caller's (its environment), not a place
/// of the argv. A release with no row is refused by name.
fn device_build(
    cards: &CardSet,
    compiler: &str,
    arch: &str,
    stated: &Stated,
) -> Result<Json, Failed> {
    let file = "cuda_image.json";
    let (tool, release) = device_tool(compiler, stated.device_release.as_deref())?;
    let card = cards.json(file)?;
    closed(
        &card,
        file,
        &[
            "text_compiler",
            "ptx_isa",
            "opt_levels",
            "default_opt_level",
            "flags",
            "compile",
            "figures",
            "figures_read",
            "runtime",
            "image_format",
            "embed",
            "local_limit_B",
        ],
    )?;
    let isa = match member(&card, file, "ptx_isa")?.get(&release).and_then(Json::int) {
        Some(isa) if isa > 0 => isa,
        _ => {
            return Err(Failed::Bad(format!(
                "facts: the device compiler release {release} has no PTX version: add its row to {file}"
            )))
        }
    };
    let form = device_build::image_form(
        &card,
        file,
        arch,
        &release,
        stated.driver_release.as_deref(),
    )?;
    let (kind, compile, flags) = if stated.device_runtime {
        let record = device_build::runtime_record(&card, file)?;
        let compile = device_build::runtime_compile(record, file, &form, arch)?;
        (
            "runtime",
            compile,
            device_build::runtime_flags(record, file, arch)?,
        )
    } else {
        let formats = member(&card, file, "flags")?;
        let names = ["ptx", "cubin", "fatbin"];
        closed(formats, file, &names)?;
        let mut flags = vec![];
        for name in names {
            flags.push((name.to_string(), flags_of(formats, file, name, arch)?));
        }
        let level = stated.opt_level.as_deref();
        let compile = device_build::compile(&card, file, (&form, arch, level))?;
        ("offline", compile, Json::Obj(flags))
    };
    let mut members = vec![
        ("compiler".into(), Json::Str(format!("{tool} {release}"))),
        ("ptx_isa".into(), number(isa)),
        ("image_format".into(), Json::Str(form.clone())),
        ("compile_with".into(), Json::Str(kind.into())),
        ("compile".into(), compile),
        (
            "figures".into(),
            device_build::figures_grammar(&card, file, kind, &form)?,
        ),
        ("flags".into(), flags),
        ("embed".into(), embed_recipe(&card, file)?),
    ];
    if stated.device_runtime {
        let record = device_build::runtime_record(&card, file)?;
        members.push(("headers".into(), flags_of_names(record, file, "headers")?));
    }
    members.push((
        "local_limit_B".into(),
        member(&card, file, "local_limit_B")?.clone(),
    ));
    Ok(Json::Obj(members))
}

/// The machine record of a device build: no instruction set, no caches, the device card's select weights, the architecture the
/// build is for (the one stated, else the card's default), which the lowering reads and the writers do not, and, last, the
/// compile command (`device_build`). One answer states one command: every unit built from it carries the same compiler, flags
/// and PTX version.
pub fn device_machine(cards: &CardSet, compiler: &str, stated: &Stated) -> Result<Json, Failed> {
    let file = "cuda.json";
    let card = cards
        .json(file)
        .map_err(|reason| cards.defect(file, reason))?;
    closed(
        &card,
        file,
        &["select_weight", "default_arch", "fusion", "fusion_rows"],
    )
    .map_err(|reason| cards.defect(file, reason))?;
    let arch = match &stated.device_arch {
        Some(asked) => asked.clone(),
        None => {
            text_of(&card, file, "default_arch").map_err(|reason| cards.defect(file, reason))?
        }
    };
    let build = device_build(cards, compiler, &arch, stated)?;
    Ok(Json::Obj(vec![
        ("exact_libm".into(), Json::Bool(true)),
        ("routed_pow".into(), Json::Bool(false)),
        (
            "select_weight".into(),
            weights_json(
                weights_of(&card, file, stated.device_arch.as_deref())
                    .map_err(|reason| cards.defect(file, reason))?,
            ),
        ),
        ("device_arch".into(), Json::Str(arch)),
        ("device_build".into(), build),
    ]))
}

#[cfg(test)]
mod tests {
    //! The compile the shipped device documents state.

    /// Oracle: the argv written out by hand below, the image compile the shipped card states for the offline compiler 12.6 at the
    /// architecture `sm_61`, at the card's default level.
    #[test]
    fn the_image_compile_of_the_shipped_card_is_the_hand_written_argv() {
        use crate::device_state::Stated;
        use osprey_lowering::facts::answer;
        let stated = Stated {
            device_arch: Some("sm_61".into()),
            ..Stated::default()
        };
        let compiler =
            "/usr/local/cuda/bin/nvcc || Cuda compilation tools, release 12.6, V12.6.77 || 0";
        let text = answer(
            &crate::testing::shipped(),
            &crate::Cuda,
            compiler,
            &[],
            &(),
            &stated,
        )
        .unwrap();
        let machine = aether_writers::contract::Contract::parse(&text).unwrap().0;
        let argv: Vec<&str> = machine
            .get("device_build")
            .and_then(|b| b.get("compile"))
            .unwrap()
            .arr()
            .iter()
            .map(|a| a.str().unwrap())
            .collect();
        assert_eq!(
            argv,
            [
                "{compiler}",
                "-ccbin",
                "{host_compiler}",
                "-cubin",
                "-std=c++20",
                "-arch=sm_61",
                "-O3",
                "-Xptxas",
                "-O3",
                "-Xptxas",
                "-v",
                "-I",
                "{include:aether}",
                "-I",
                "{include:eagle}",
                "{defines}",
                "-I{include:facts}",
                "-MD",
                "-MF",
                "{depfile}",
                "{source}",
                "-o",
                "{output}"
            ]
        );
    }
}
