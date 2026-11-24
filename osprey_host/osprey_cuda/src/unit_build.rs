//! The build of one device unit, as the tool side runs it: the machine's facts state which tool makes the image (`compile_with`),
//! the offline compiler gets the argv template, the runtime compile library the options and the header trees it serves from memory;
//! the stated machine is what the device states about itself.
use crate::{device_build, device_facts, local_memory, Cuda};
use aether_writers::contract::{Contract, Json};
use aether_writers::refuse::Refusal;
use osprey_lowering::backend::Backend;
use osprey_lowering::facts::{CardSet, Failed};
use osprey_lowering::tool::{self, facts_for, parse, refuse, strings, text};
use rcore::json::J;
use std::path::Path;
use toolkit::wire;

/// The compiler identity and the stated machine (the JSON `device_state::read_stated` reads) a machine answer stands for: the
/// device's arch, driver and compiler release. A machine with neither a compiler nor a runtime compile library is written under the
/// card's `text_compiler`: text needs no tool.
pub(crate) fn stated_machine(cards: &CardSet, machine: &Json) -> Result<(String, String), Failed> {
    let stated = machine
        .get("stated")
        .ok_or_else(|| Failed::Bad("the machine answer states nothing".into()))?;
    let tool = stated.get("tools").and_then(|t| t.get("compiler"));
    let identity = match tool.and_then(|t| text(t, "identity")) {
        Some(found) => found.replace("||", " || "),
        None if stated.get("runtime").is_some() => device_facts::device_runtime_compiler(cards)?,
        None => device_facts::device_text_compiler(cards)?,
    };
    let mut out: Vec<(String, Json)> = vec![];
    let mut put = |key: &str, value: Option<Json>| out.extend(value.map(|v| (key.to_string(), v)));
    let device = stated.get("device");
    put(
        "device_arch",
        device
            .and_then(|d| text(d, "arch"))
            .map(|a| Json::Str(a.into())),
    );
    put(
        "driver_release",
        device
            .and_then(|d| text(d, "driver_release"))
            .map(|a| Json::Str(a.into())),
    );
    let release = tool
        .and_then(|t| text(t, "release"))
        .or_else(|| stated.get("runtime").and_then(|r| text(r, "version")));
    put("device_release", release.map(|r| Json::Str(r.into())));
    let runtime_only = tool.is_none() && stated.get("runtime").is_some();
    put("device_runtime", runtime_only.then_some(Json::Bool(true)));
    Ok((identity, Json::Obj(out).to_text()))
}

/// The members the library adds to the toolkit's machine ask: the device arch the options pin.
pub(crate) fn machine_ask(options: &Json) -> Vec<(String, Json)> {
    text(options, "arch")
        .map(|arch| ("device_arch".to_string(), Json::Str(arch.into())))
        .into_iter()
        .collect()
}

/// States in the lowering request `parts` what the machine answer `own` and the facts `answer` add: the literal-table setting, the
/// image format the facts name, and the refusal of a format the driver cannot load.
pub(crate) fn with_facts_request(
    declared: &[&str],
    own: &Json,
    answer: &str,
    parts: &mut Vec<(String, Json)>,
) -> Result<(), Refusal> {
    state_option_settings(declared, own, parts);
    let named = parts.iter().find(|(k, _)| k == "image_format");
    if let Some((_, Json::Str(format))) = named {
        refuse_unloadable_ptx(own, format).map_err(Failed::refusal)?;
    }
    if named.is_none() {
        if let Some(format) = image_format_of(answer)? {
            parts.push(("image_format".into(), Json::Str(format)));
        }
    }
    Ok(())
}

/// Links the library `name` of a set from the build answers `parts`.
pub(crate) fn link(
    parts: &str,
    set_text: &str,
    name: &str,
    root: &Path,
) -> Result<String, toolkit::Refusal> {
    tool::link_parts(Cuda.toolkit()?, parts, set_text, name, root)
}

/// The build of a device unit (see [`device_unit`]).
pub(crate) fn build_unit(
    req: &Json,
    machine: &Json,
    root: &Path,
) -> Result<String, toolkit::Refusal> {
    build_device(&Cuda, req, machine, root)
}

/// Builds one device unit: the request is the host's (`name`, `text`, `machine`, `defines`) with the image format the lowering
/// answered if it names one. The machine's facts state which tool makes the image (`compile_with`): the offline compiler gets the
/// argv template, the runtime compile library the options and the header trees it serves from memory. The answer is the host's
/// (`artifact`, `source`, `format`, `key`, `hit`, `seconds`, `files`, `figures`, `report`) with a `link_recipe` that carries the
/// recipe that embeds the image: [`link`] defines the image's symbols from it.
fn device_unit(
    req: &Json,
    machine: &Json,
    facts: &Json,
) -> Result<(String, J, Vec<String>, String), toolkit::Refusal> {
    let build = facts
        .get("device_build")
        .ok_or_else(|| refuse("the facts answer has no device_build"))?;
    let form =
        text(build, "image_format").ok_or_else(|| refuse("device_build states no image_format"))?;
    if let Some(named) = text(req, "image_format").filter(|n| !n.eq_ignore_ascii_case(form)) {
        return Err(refuse(format!(
            "build: the request names the image format {named}, but this machine's compile makes {form}"
        )));
    }
    let name = text(req, "name").ok_or_else(|| refuse("build: the request has no name"))?;
    let defines: Vec<String> = strings(req.get("defines"))
        .iter()
        .map(|d| format!("-D{d}"))
        .collect();
    let key = text(machine, "key").unwrap_or("");
    let compile = strings(build.get("compile"));
    let mut members = vec![
        ("name", wire::s(name)),
        (
            "text",
            wire::s(text(req, "text").ok_or_else(|| refuse("build: the request has no text"))?),
        ),
        ("source_suffix", wire::s(".cu")),
        ("output_suffix", wire::s(&format!(".{form}"))),
        ("key_terms", wire::strs(&[key])),
        ("keep_source", J::Bool(true)),
    ];
    if let Some(grammar) = text(build, "figures") {
        members.push(("figures", wire::s(grammar)));
    }
    if text(build, "compile_with") == Some("runtime") {
        let options: Vec<String> = compile
            .iter()
            .flat_map(|o| {
                if o == "{defines}" {
                    defines.clone()
                } else {
                    vec![o.clone()]
                }
            })
            .collect();
        members.push((
            "runtime",
            wire::obj(vec![
                ("options", wire::strs(&options)),
                ("target", wire::s(form)),
                ("headers", wire::strs(&strings(build.get("headers")))),
            ]),
        ));
    } else {
        members.push(("template", wire::strs(&compile)));
        members.push((
            "lists",
            wire::obj(vec![("{defines}", wire::strs(&defines))]),
        ));
    }
    Ok((
        name.to_string(),
        wire::obj(members),
        defines,
        form.to_string(),
    ))
}
/// The unit's `launch` record of a device build: the local memory one thread holds (`local_bytes`), how that was measured
/// (`local_source`) and the device's per-thread limit (`local_limit`, a fact of the card). A unit that holds more than the limit is
/// refused by name. An image that states no frame and has no text to read states no `launch` (`None`). The record is a member of the
/// build answer of its own: `figures` keeps only the entries of the unit.
fn launch_record(facts: &Json, name: &str, answer: &J) -> Result<Option<J>, toolkit::Refusal> {
    use J;
    let figures = answer.get("figures").cloned().unwrap_or(J::Null);
    let limit = facts
        .get("device_build")
        .and_then(|b| b.get("local_limit_B"))
        .and_then(Json::int)
        .and_then(|l| u64::try_from(l).ok())
        .ok_or_else(|| refuse("device_build states no local_limit_B"))?;
    let read_ptx = || {
        let output = answer.get("files")?.get("output")?.str()?.to_string();
        std::fs::read(output)
            .ok()
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
    };
    let Some(held) = local_memory::held(&figures, read_ptx) else {
        return Ok(None);
    };
    if held.0 > limit {
        return Err(refuse(local_memory::refusal(name, held, limit)));
    }
    Ok(Some(wire::obj(vec![
        ("local_bytes", wire::n(held.0)),
        ("local_source", wire::s(held.1)),
        ("local_limit", wire::n(limit)),
    ])))
}
/// The build of a device unit through the toolkit (see [`device_unit`]).
fn build_device(
    backend: &Cuda,
    req: &Json,
    machine: &Json,
    root: &Path,
) -> Result<String, toolkit::Refusal> {
    let facts = parse(
        "the facts answer",
        &facts_for(backend, machine).map_err(|f| refuse(String::from(f)))?,
    )?;
    let (name, unit, defines, form) = device_unit(req, machine, &facts)?;
    let index_bytes = backend
        .index_bytes(&strings(req.get("defines")))
        .map_err(|why| refuse(why.reason))?;
    let answer = wire::parse(
        "the toolkit's answer",
        &backend.toolkit()?.build(&wire::render(&unit), root)?,
    )?;
    let image = format!("{name}.{form}");
    let source = format!("{name}.cu");
    let file = |member: &str| answer.get("files").and_then(|f| f.get(member)).cloned();
    let null = || J::Null;
    let launch = launch_record(&facts, &name, &answer)?;
    let recipe = facts
        .get("device_build")
        .and_then(|b| b.get("embed"))
        .cloned()
        .unwrap_or(Json::Null);
    let link_recipe = Json::Obj(vec![
        ("unit".into(), Json::Str(name.clone())),
        ("embed".into(), recipe),
        (
            "machine".into(),
            Json::Str(text(machine, "key").unwrap_or("").into()),
        ),
        (
            "defines".into(),
            Json::Arr(defines.iter().map(|d| Json::Str(d.clone())).collect()),
        ),
    ]);
    Ok(wire::render(&wire::obj(vec![
        ("artifact", wire::s(&image)),
        ("source", wire::s(&source)),
        ("format", wire::s(&form)),
        (
            "index_bytes",
            index_bytes.map_or(J::Null, |bytes| wire::n(bytes.unsigned_abs())),
        ),
        ("key", answer.get("key").cloned().unwrap_or_else(null)),
        ("hit", answer.get("hit").cloned().unwrap_or_else(null)),
        (
            "seconds",
            answer
                .get("report")
                .and_then(|r| r.get("seconds"))
                .cloned()
                .unwrap_or_else(null),
        ),
        (
            "files",
            wire::obj(vec![
                (image.as_str(), file("output").unwrap_or_else(null)),
                (source.as_str(), file("source").unwrap_or_else(null)),
            ]),
        ),
        (
            "figures",
            answer.get("figures").cloned().unwrap_or_else(null),
        ),
        ("launch", launch.unwrap_or_else(null)),
        ("report", answer.get("report").cloned().unwrap_or_else(null)),
        (
            "link_recipe",
            wire::parse("recipe", &link_recipe.to_text())?,
        ),
    ])))
}
/// States in the request's `settings` the options of the machine answer that are settings of the lowering and that the library
/// declares (`declared`): the literal table of a device library. An option that is off, or that the library does not declare, states
/// nothing; a setting the request already states stays as it is.
fn state_option_settings(declared: &[&str], machine: &Json, parts: &mut Vec<(String, Json)>) {
    let asked = machine.get("stated").and_then(|s| s.get("asked"));
    let on = |name: &str| {
        declared.contains(&name)
            && asked
                .and_then(|a| a.get("options"))
                .and_then(|o| o.get(name))
                .is_some_and(|flag| matches!(flag, Json::Bool(true)))
    };
    if !on("literal_table") {
        return;
    }
    let at = parts.iter().position(|(k, _)| k == "settings");
    let Some(at) = at else {
        let flag = ("literal_table".into(), Json::Bool(true));
        parts.push(("settings".into(), Json::Obj(vec![flag])));
        return;
    };
    if let Json::Obj(members) = &mut parts[at].1 {
        if !members.iter().any(|(k, _)| k == "literal_table") {
            members.push(("literal_table".into(), Json::Bool(true)));
        }
    }
}
/// Refuses a request that names the image format `Ptx` on a machine whose driver cannot load the PTX its compiler writes: the
/// compiler (offline or runtime) is newer than the driver. A request that names no format gets `cubin` from the facts instead; a
/// machine that states no compiler or no driver release has nothing to refuse by.
fn refuse_unloadable_ptx(machine: &Json, format: &str) -> Result<(), Failed> {
    let Some(stated) = machine.get("stated").filter(|_| format == "Ptx") else {
        return Ok(());
    };
    let compiler = stated
        .get("tools")
        .and_then(|t| t.get("compiler"))
        .and_then(|c| text(c, "release"))
        .or_else(|| stated.get("runtime").and_then(|r| text(r, "version")));
    let driver = stated.get("device").and_then(|d| text(d, "driver_release"));
    let (Some(compiler), Some(driver)) = (compiler, driver) else {
        return Ok(());
    };
    if device_build::compiler_newer(compiler, driver)? {
        return Err(Failed::Bad(format!(
            "lower: the request names the image format Ptx, but the device compiler {compiler} is newer than the driver {driver}, \
             which cannot load its PTX; ask for Cubin or leave the format out"
        )));
    }
    Ok(())
}
/// The form of device image the facts answer states (`Ptx`, `Cubin` or `Fatbin`), as the request names it; none for a facts answer
/// with no device build, a host's.
fn image_format_of(answer: &str) -> Result<Option<String>, Refusal> {
    let facts = Contract::parse(answer)?.0;
    let Some(form) = facts
        .get("device_build")
        .and_then(|build| text(build, "image_format"))
    else {
        return Ok(None);
    };
    let mut letters = form.chars();
    let first = letters.next().map(|c| c.to_ascii_uppercase());
    Ok(Some(first.into_iter().chain(letters).collect()))
}

#[cfg(test)]
mod tests {
    //! The device unit the toolkit is sent, and the options of the machine answer that are settings of the lowering.
    use super::*;
    use crate::device_state::Stated;
    use osprey_lowering::facts::answer;

    /// The device unit a build of `defines` makes, as the text the toolkit is sent, for an image made by the offline compiler (`runtime`
    /// false) or by the runtime compile library.
    fn device_unit_text(defines: &[&str], runtime: bool) -> String {
        let strs =
            |items: &[&str]| Json::Arr(items.iter().map(|d| Json::Str((*d).into())).collect());
        let mut build = vec![
            ("image_format".to_string(), Json::Str("ptx".into())),
            ("compile".to_string(), strs(&["{flags}", "{defines}"])),
        ];
        if runtime {
            build.push(("compile_with".to_string(), Json::Str("runtime".into())));
        }
        let facts = Json::Obj(vec![("device_build".to_string(), Json::Obj(build))]);
        let req = Json::Obj(vec![
            ("name".to_string(), Json::Str("k".into())),
            ("text".to_string(), Json::Str("int k;".into())),
            ("defines".to_string(), strs(defines)),
        ]);
        let machine = Json::Obj(vec![("key".to_string(), Json::Str("m".into()))]);
        let (_, unit, _, _) = device_unit(&req, &machine, &facts).unwrap();
        wire::render(&unit)
    }

    #[test]
    fn a_define_of_the_build_is_in_the_offline_argv_lists_and_in_the_runtime_options() {
        for runtime in [false, true] {
            let on = device_unit_text(&["HAWK_FAST_ENTRIES=1"], runtime);
            assert!(on.contains("-DHAWK_FAST_ENTRIES=1"), "{on}");
            assert!(!device_unit_text(&[], runtime).contains("HAWK_FAST_ENTRIES"));
        }
    }

    /// The machine answer of a build asked with `options`.
    fn answer_asked(options: &str) -> Json {
        let text = format!("{{\"stated\":{{\"asked\":{{\"options\":{options}}}}}}}");
        Contract::parse(&text).unwrap().0
    }

    #[test]
    fn the_literal_table_option_of_the_machine_answer_is_the_setting_of_the_lowering() {
        let on = |declared: &[&str], options: &str, settings: Option<&str>| {
            let mut parts: Vec<(String, Json)> = settings
                .map(|text| ("settings".to_string(), Contract::parse(text).unwrap().0))
                .into_iter()
                .collect();
            state_option_settings(declared, &answer_asked(options), &mut parts);
            Json::Obj(parts).to_text()
        };
        let declared = ["arch", "literal_table"];
        let table = "{\"settings\":{\"literal_table\":true}}";
        assert_eq!(on(&declared, "{\"literal_table\":true}", None), table);
        assert_eq!(
            on(&declared, "{\"literal_table\":true}", Some("{\"warps\":2}")),
            "{\"settings\":{\"warps\":2,\"literal_table\":true}}"
        );
        assert_eq!(on(&declared, "{\"literal_table\":false}", None), "{}");
        assert_eq!(on(&declared, "{\"arch\":\"sm_61\"}", None), "{}");
        assert_eq!(on(&["profile"], "{\"literal_table\":true}", None), "{}");
    }

    /// A request cannot reach the lowering with a device document its facts call refused: the facts call states none for a document it
    /// does not accept.
    #[test]
    fn a_stated_document_the_facts_call_refuses_has_no_answer_to_lower_under() {
        let text = std::fs::read_to_string(crate::testing::shipped().join("cuda.json")).unwrap();
        let broken = [(
            "cuda.json".to_string(),
            format!("{{\"nonsense\": 1,{}", &text[1..]),
        )];
        let machine = Stated::default();
        let compiler =
            "/usr/local/cuda/bin/nvcc || Cuda compilation tools, release 12.6, V12.6.77 || 0";
        assert!(answer(
            &crate::testing::shipped(),
            &crate::Cuda,
            compiler,
            &broken,
            &(),
            &machine
        )
        .is_err());
    }
}
