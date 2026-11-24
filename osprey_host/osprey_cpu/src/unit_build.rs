//! The build of one host unit and the link of a set, as the tool side runs them: the unit's compile recipe is the facts answer's
//! `host_build`, the compile of the set and the link are carried in the part's `link_recipe`, and the stated machine is what the host
//! states about itself.
use crate::{host_facts, host_state, Cpu};
use aether_writers::contract::Json;
use osprey_lowering::backend::Backend;
use osprey_lowering::facts::{self, CardSet, Failed};
use osprey_lowering::tool::{self, facts_for, parse, refuse, strings, text, text_in, tool_ask};
use rcore::json::J;
use std::path::Path;
use toolkit::{wire, Toolkit};

/// Builds one host unit: `{"name", "text", "machine", "defines"?}` in; the toolkit's answer out with what a caller publishes and links
/// from it: `artifact` (the object's file name), `source` (the source's), `format`, `seconds`, `files` (name to path) and
/// `link_recipe`, the opaque instructions the link needs (the compile of the set, its link, the machine and the defines the unit was
/// compiled with), which the caller hands back untouched in `parts`.
///
/// # Errors
/// Refuses a request without a machine, a machine for another target, and what the tool side refuses.
pub(crate) fn build_unit(
    req: &Json,
    machine: &Json,
    root: &Path,
) -> Result<String, toolkit::Refusal> {
    let backend = &Cpu;
    let facts = parse(
        "the facts answer",
        &facts_for(backend, machine).map_err(|f| refuse(String::from(f)))?,
    )?;
    let recipe = facts
        .get("host_build")
        .ok_or_else(|| refuse("the facts answer has no host_build"))?;
    let name = text(req, "name").ok_or_else(|| refuse("build: the request has no name"))?;
    let defines: Vec<String> = strings(req.get("defines"))
        .iter()
        .map(|d| format!("-D{d}"))
        .collect();
    let key = text(machine, "key").unwrap_or("");
    let index_bytes = backend
        .index_bytes(&strings(req.get("defines")))
        .map_err(|why| refuse(why.reason))?;
    let unit = wire::obj(vec![
        ("name", wire::s(name)),
        (
            "text",
            wire::s(text(req, "text").ok_or_else(|| refuse("build: the request has no text"))?),
        ),
        ("template", wire::strs(&strings(recipe.get("compile_unit")))),
        (
            "lists",
            wire::obj(vec![("{defines}", wire::strs(&defines))]),
        ),
        ("key_terms", wire::strs(&[key])),
        ("keep_source", J::Bool(true)),
    ]);
    let answer = wire::parse(
        "the toolkit's answer",
        &backend.toolkit()?.build(&wire::render(&unit), root)?,
    )?;
    let object = format!("{name}.o");
    let source = format!("{name}.cpp");
    let path = |member: &str| answer.get("files").and_then(|f| f.get(member)).cloned();
    let report = answer.get("report");
    let seconds = report.and_then(|r| r.get("seconds")).cloned();
    let link_recipe = Json::Obj(vec![
        (
            "compile_set".into(),
            recipe.get("compile_set").cloned().unwrap_or(Json::Null),
        ),
        (
            "link".into(),
            recipe.get("link").cloned().unwrap_or(Json::Null),
        ),
        ("machine".into(), Json::Str(key.into())),
        (
            "defines".into(),
            Json::Arr(defines.iter().map(|d| Json::Str(d.clone())).collect()),
        ),
    ]);
    Ok(wire::render(&wire::obj(vec![
        ("artifact", wire::s(&object)),
        ("source", wire::s(&source)),
        ("format", wire::s(backend.format())),
        (
            "index_bytes",
            index_bytes.map_or(J::Null, |bytes| wire::n(bytes.unsigned_abs())),
        ),
        ("key", answer.get("key").cloned().unwrap_or(J::Null)),
        ("hit", answer.get("hit").cloned().unwrap_or(J::Null)),
        ("seconds", seconds.unwrap_or(J::Null)),
        (
            "files",
            wire::obj(vec![
                (object.as_str(), path("output").unwrap_or(J::Null)),
                (source.as_str(), path("source").unwrap_or(J::Null)),
            ]),
        ),
        ("figures", answer.get("figures").cloned().unwrap_or(J::Null)),
        ("report", answer.get("report").cloned().unwrap_or(J::Null)),
        (
            "link_recipe",
            wire::parse("recipe", &link_recipe.to_text())?,
        ),
    ])))
}

/// The compiler identity and the stated machine (the JSON `host_state::read_stated` reads) a machine answer stands for: the host's
/// instruction set, caches, profile, header shape (the library embeds both header trees, so `roots`), level and witness verdict.
pub(crate) fn stated_machine(machine: &Json) -> Result<(String, String), Failed> {
    let stated = machine
        .get("stated")
        .ok_or_else(|| Failed::Bad("the machine answer states nothing".into()))?;
    let tool = stated.get("tools").and_then(|t| t.get("compiler"));
    let identity = match tool.and_then(|t| text(t, "identity")) {
        Some(found) => found.replace("||", " || "),
        None => "none || no compiler found || none".to_string(),
    };
    let asked = stated.get("asked");
    let level = asked
        .and_then(|a| text(a, "opt_level"))
        .filter(|l| !l.is_empty());
    let mut out: Vec<(String, Json)> = vec![];
    let mut put = |key: &str, value: Option<Json>| out.extend(value.map(|v| (key.to_string(), v)));
    let host = stated.get("host");
    let isa = host
        .and_then(|h| text(h, "arch"))
        .map(|a| if a == "aarch64" { "arm64" } else { a });
    put("isa", isa.map(|a| Json::Str(a.into())));
    put(
        "vector_isa",
        host.and_then(|h| text(h, "vector_isa"))
            .map(|v| Json::Str(v.into())),
    );
    if let Some(h) = host.filter(|h| h.get("l1d_cache_bytes").is_some()) {
        put("l1d_bytes", h.get("l1d_cache_bytes").cloned());
        put("l2_bytes", h.get("l2_cache_bytes").cloned());
        put("cache_source", h.get("cache_source").cloned());
    }
    let profile = asked
        .and_then(|a| a.get("options"))
        .and_then(|o| text(o, "profile"));
    put("host_profile", profile.map(|p| Json::Str(p.into())));
    put("headers", Some(Json::Str("roots".into())));
    put("opt_level", level.map(|l| Json::Str(l.into())));
    let witness = stated
        .get("witnesses")
        .and_then(|w| w.get("masked_call"))
        .and_then(|w| text(w, "verdict"));
    put("masked_call_witness", witness.map(|w| Json::Str(w.into())));
    Ok((identity, Json::Obj(out).to_text()))
}

/// The `exact` member of the machine's `stated`: whether the resolved profile gives bit-exact values, by the cpu card's own list.
pub(crate) fn stated_extras(asked: &Json) -> Result<Vec<(String, Json)>, toolkit::Refusal> {
    let Some(profile) = asked.get("options").and_then(|o| text(o, "profile")) else {
        return Ok(vec![]);
    };
    let dir = facts::facts_dir(None).map_err(refuse)?;
    let cards = CardSet::new(&dir, &[], Cpu.card_names());
    let exact = host_facts::profile_is_exact(&cards, profile)
        .map_err(|why| refuse(why.refusal().reason))?;
    Ok(vec![("exact".into(), Json::Bool(exact))])
}

/// The profile a build gets when it states none: the instruction set card's default for the compiler the machine finds, so the
/// machine is asked once for the compiler before the key is made.
pub(crate) fn default_options(
    cards: &CardSet,
    toolkit: &Toolkit,
    ask: &Json,
    options: &Json,
) -> Result<Vec<(String, Json)>, toolkit::Refusal> {
    let Json::Obj(members) = options else {
        return Ok(vec![]);
    };
    if text_in(members, "profile").is_some() {
        return Ok(vec![]);
    }
    let failed = |why: Failed| refuse(why.refusal().reason);
    let found = toolkit.machine(&tool_ask(&Cpu, ask, options, vec![]).to_text())?;
    let found = parse("the toolkit's machine answer", &found)?;
    let (compiler, stated) = stated_machine(&found).map_err(failed)?;
    let (record, _) = facts::split_stated(&stated).map_err(failed)?;
    let stated = host_state::read_stated(&record).map_err(failed)?;
    let isa = stated.isa.unwrap_or_else(|| host_state::probe().isa);
    let profile = host_facts::default_host_profile(cards, &compiler, &isa).map_err(failed)?;
    Ok(vec![("profile".into(), Json::Str(profile))])
}

/// Links the library `name` of a set from the build answers `parts`.
pub(crate) fn link(
    parts: &str,
    set_text: &str,
    name: &str,
    root: &Path,
) -> Result<String, toolkit::Refusal> {
    tool::link_parts(Cpu.toolkit()?, parts, set_text, name, root)
}
