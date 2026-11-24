//! The device image build as data: for the device compiler, the architecture and the settings a facts question states, which
//! form of image the build makes (`ptx` or `cubin`) and the argv template of the one compile that makes it. Every flag and
//! every rule is read from `cuda_image.json`; nothing is probed. Places the caller fills in are written `{compiler}`,
//! `{host_compiler}` (the compiler of the set's host objects, which the device compiler is told with `-ccbin` so that one C++
//! runtime serves the whole library), `{source}`, `{output}`, `{depfile}`, `{include:aether}`, `{include:eagle}`, `{include:facts}`
//! and `{defines}`, as the host compile writes them; `{defines}` stands for one `-D<define>` element for each define and for nothing when there is none.
use crate::device_facts::{arch_names, flags_of};
use aether_writers::contract::Json;
use osprey_lowering::facts::{closed, member, names_of, text_of, Failed};

/// The release `major.minor` of a stated compiler or driver as a pair; `what` names it in the refusal.
fn release_pair(text: &str, what: &str) -> Result<(u32, u32), Failed> {
    let numbers: Vec<Option<u32>> = text.split('.').map(|n| n.parse().ok()).collect();
    match numbers[..] {
        [Some(major), Some(minor)] => Ok((major, minor)),
        _ => Err(Failed::Bad(format!(
            "stated machine: {what} {text:?} is not major.minor"
        ))),
    }
}

/// Whether the compiler of release `release` is newer than the driver of release `driver`: the PTX it writes then carries an ISA
/// version the driver's loader does not take.
pub(crate) fn compiler_newer(release: &str, driver: &str) -> Result<bool, Failed> {
    Ok(release_pair(release, "device_release")? > release_pair(driver, "driver_release")?)
}

/// The image form the build makes. A virtual architecture (`compute_<N>`) has no binary image, so it takes the card's
/// `virtual_arch` form. Otherwise the form depends on the compiler's release against the driver's: a driver that is not stated
/// takes `driver_unknown`, a compiler newer than the driver `compiler_newer` (the image's PTX would carry a version the driver
/// refuses), else `otherwise`. A form the card names that is neither `ptx` nor `cubin` is refused by name.
pub(crate) fn image_form(
    card: &Json,
    file: &str,
    arch: &str,
    release: &str,
    driver: Option<&str>,
) -> Result<String, Failed> {
    let rules = member(card, file, "image_format")?;
    let key = if arch.starts_with("compute_") {
        "virtual_arch"
    } else {
        match driver {
            None => "driver_unknown",
            Some(driver) => {
                if compiler_newer(release, driver)? {
                    "compiler_newer"
                } else {
                    "otherwise"
                }
            }
        }
    };
    let form = text_of(rules, file, key)?;
    if form != "ptx" && form != "cubin" {
        return Err(Failed::Card(format!(
            "{file}: image_format {key} is {form}, not ptx or cubin"
        )));
    }
    Ok(form)
}

/// The argv template of the image compile of `form` for the architecture `arch` (its virtual name when it is stated as one) at the
/// optimisation level `level`: the card's `compile` with `{form}`, `{arch}`, `{level}` and `{figures}` put in. `{figures}`
/// stands for the flags that make the compiler report each kernel's registers, spills and stack, which the card states for the
/// forms that have them.
pub(crate) fn compile(
    card: &Json,
    file: &str,
    (form, arch, level): (&str, &str, Option<&str>),
) -> Result<Json, Failed> {
    let level = match level {
        Some(asked) => asked.to_string(),
        None => text_of(card, file, "default_opt_level")?,
    };
    if !names_of(card, file, "opt_levels")?.contains(&level) {
        return Err(Failed::Bad(format!("unknown opt level {level}")));
    }
    let (sm, virtual_arch) = arch_names(arch)?;
    let arch = if arch.starts_with("compute_") {
        virtual_arch
    } else {
        sm
    };
    let figures = names_of(member(card, file, "figures")?, file, form)?;
    let mut argv = vec![];
    for element in names_of(card, file, "compile")? {
        match element.as_str() {
            "{figures}" => argv.extend(figures.iter().cloned()),
            _ => argv.push(
                element
                    .replace("{form}", form)
                    .replace("{arch}", &arch)
                    .replace("{level}", &level),
            ),
        }
    }
    Ok(Json::Arr(argv.into_iter().map(Json::Str).collect()))
}

/// The record of the card that describes the runtime compile library as the tool of a device build.
pub(crate) fn runtime_record<'a>(card: &'a Json, file: &str) -> Result<&'a Json, Failed> {
    let record = member(card, file, "runtime")?;
    closed(record, file, &["compiler", "flags", "compile", "headers"])?;
    Ok(record)
}

/// The flags of the runtime compile for each format it can write (`ptx` and `cubin`: it cannot write a fatbin), for `arch`.
pub(crate) fn runtime_flags(record: &Json, file: &str, arch: &str) -> Result<Json, Failed> {
    let formats = member(record, file, "flags")?;
    closed(formats, file, &["ptx", "cubin"])?;
    let mut flags = vec![];
    for name in ["ptx", "cubin"] {
        flags.push((name.to_string(), flags_of(formats, file, name, arch)?));
    }
    Ok(Json::Obj(flags))
}

/// The options template of the runtime compile of `form` for `arch`: the card's `compile` with `{flags}` standing for the flags of
/// the form. The places `{include:cccl}` and `{defines}` are left for the caller.
pub(crate) fn runtime_compile(
    record: &Json,
    file: &str,
    form: &str,
    arch: &str,
) -> Result<Json, Failed> {
    let flags = runtime_flags(record, file, arch)?;
    let mut options = vec![];
    for element in names_of(record, file, "compile")? {
        match (element.as_str(), flags.get(form)) {
            ("{flags}", Some(list)) => options.extend(list.arr().iter().cloned()),
            ("{flags}", None) => {
                return Err(Failed::Bad(format!(
                    "facts: the runtime compile writes no {form} image"
                )))
            }
            _ => options.push(Json::Str(element)),
        }
    }
    Ok(Json::Arr(options))
}

/// The grammar the figures of a `form` image are read by, for the tool kind `kind` (`offline` or `runtime`), from the card's
/// `figures_read`; null for a form whose compile states none.
pub(crate) fn figures_grammar(
    card: &Json,
    file: &str,
    kind: &str,
    form: &str,
) -> Result<Json, Failed> {
    let kinds = member(card, file, "figures_read")?;
    Ok(member(kinds, file, kind)
        .map_err(Failed::Card)?
        .get(form)
        .cloned()
        .unwrap_or(Json::Null))
}
