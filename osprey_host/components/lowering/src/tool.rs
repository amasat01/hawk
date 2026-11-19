//! The tool side as a caller that knows nothing of the target uses it: `machine` takes what a build asks (target, options,
//! optimisation level, a cache root), `lower` derives its facts from the machine it is given, `build` and `link` read the library's
//! own build card. No compiler flag, header shape or tool identity crosses the door in either direction; the library states the
//! machine once and every later call carries that answer back as given.
use crate::backend::{target_name, unserved, Backend};
use crate::device_build;
use crate::facts::{self, CardSet, Failed};
use crate::local_memory;
use aether_writers::contract::{Contract, Json};
use aether_writers::refuse::{bad, Refusal};
use std::path::Path;
use toolkit::{wire, Toolkit};

/// The text of the member `key` of `j`.
fn text<'a>(j: &'a Json, key: &str) -> Option<&'a str> {
    j.get(key).and_then(Json::str)
}

/// A refusal of the tool side with `reason` as a malformed request.
fn refuse(reason: impl Into<String>) -> toolkit::Refusal {
    toolkit::Refusal::bad(reason)
}

/// The parsed JSON of `source`, as a refusal of the tool side when it is not JSON.
fn parse(what: &str, source: &str) -> Result<Json, toolkit::Refusal> {
    Contract::parse(source)
        .map(|c| c.0)
        .map_err(|e| refuse(format!("{what} is not JSON: {}", e.reason)))
}

/// The machine the library's tools run on, for a build's ask: `{"target": <the library's target name>, "options": {...},
/// "opt_level": "...", "root": <cache root>?}`. The answer is the toolkit's (`key`, `stated`, `tools_absent`) with the ask echoed in
/// `stated.asked`, so the answer alone says what the build is for. The key covers the library's identity, the level and the options.
///
/// # Errors
/// Refuses an ask that is not JSON, names another target, or states an option the target does not declare.
pub fn machine<B: Backend>(backend: &B, ask: &str) -> Result<String, toolkit::Refusal> {
    let ask = parse("the machine ask", ask)?;
    let wanted = target_name(backend.target());
    if text(&ask, "target") != Some(wanted) {
        return Err(refuse(
            unserved(backend, text(&ask, "target").unwrap_or("")).reason,
        ));
    }
    if let Some(Json::Obj(stated)) = ask.get("options") {
        if let Some((key, _)) = stated
            .iter()
            .find(|(key, _)| !backend.option_names().contains(&key.as_str()))
        {
            return Err(refuse(format!(
                "the option {key:?} is declared by the target {wanted}: its options are {:?}",
                backend.option_names()
            )));
        }
    }
    let toolkit = backend.toolkit()?;
    let from_env = text(&ask, "opt_level").unwrap_or("").is_empty()
        && toolkit.ask_from_env("opt_level").is_some();
    let (level, options) = effective_ask(
        text(&ask, "opt_level").unwrap_or(""),
        ask.get("options").cloned().unwrap_or(Json::Obj(vec![])),
        toolkit.ask_from_env("opt_level"),
        toolkit
            .ask_from_env("host_profile")
            .filter(|_| backend.host_card_wanted()),
    );
    let (level, options) = resolved_ask(backend, toolkit, &ask, level, options)?;
    known_level(backend, toolkit, &level, from_env)?;
    let level = level.as_str();
    let terms = vec![
        Json::Str(backend.identity().into()),
        Json::Str(format!("opt_level:{level}")),
        Json::Str(format!("options:{}", options.to_text())),
    ];
    let tool_ask = tool_ask(backend, &ask, &options, terms);
    let answer = toolkit.machine(&tool_ask.to_text())?;
    let Json::Obj(mut parts) = parse("the toolkit's machine answer", &answer)? else {
        return Err(refuse("the toolkit's machine answer is not an object"));
    };
    let asked = Json::Obj(vec![
        ("opt_level".into(), Json::Str(level.into())),
        ("options".into(), options),
    ]);
    let exact = exact_profile(backend, &asked)?;
    for (name, value) in &mut parts {
        if let (Json::Obj(stated), "stated") = (value, name.as_str()) {
            stated.push(("asked".into(), asked.clone()));
            stated.extend(exact.clone().map(|flag| ("exact".to_string(), flag)));
        }
    }
    Ok(Json::Obj(parts).to_text())
}

/// The `exact` member of a host machine's `stated`: whether the resolved profile gives bit-exact values, by the cpu card's own
/// list. A device machine has no profile and states none.
fn exact_profile<B: Backend>(backend: &B, asked: &Json) -> Result<Option<Json>, toolkit::Refusal> {
    let profile = asked.get("options").and_then(|o| text(o, "profile"));
    let Some(profile) = profile.filter(|_| backend.host_card_wanted()) else {
        return Ok(None);
    };
    let dir = facts::facts_dir(None).map_err(refuse)?;
    let cards = CardSet::new(&dir, &[], backend.card_names());
    let exact =
        facts::profile_is_exact(&cards, profile).map_err(|why| refuse(why.refusal().reason))?;
    Ok(Some(Json::Bool(exact)))
}

/// Refuses a level the card does not list, by name: the text says where the level came from (the variable, when the ask stated
/// none) and which levels the card knows.
fn known_level<B: Backend>(
    backend: &B,
    toolkit: &Toolkit,
    level: &str,
    from_env: bool,
) -> Result<(), toolkit::Refusal> {
    let dir = facts::facts_dir(None).map_err(refuse)?;
    let cards = CardSet::new(&dir, &[], backend.card_names());
    let known = facts::opt_levels(&cards, backend.host_card_wanted())
        .map_err(|why| refuse(why.refusal().reason))?;
    if known.iter().any(|k| k == level) {
        return Ok(());
    }
    let source = match toolkit.ask_env_name("opt_level").filter(|_| from_env) {
        Some(variable) => format!("${variable}"),
        None => "the build's opt_level".to_string(),
    };
    Err(refuse(format!(
        "unknown opt level {level:?} (from {source}); the levels are {known:?}"
    )))
}

/// The ask the toolkit answers: the key's `terms` beyond what the machine states, whether the host's facts are wanted, the cache
/// root with the reserved names, and the device arch the options pin.
fn tool_ask<B: Backend>(backend: &B, ask: &Json, options: &Json, terms: Vec<Json>) -> Json {
    let mut members = vec![
        ("key_terms".to_string(), Json::Arr(terms)),
        ("host".into(), Json::Bool(backend.host_card_wanted())),
    ];
    if let Some(root) = text(ask, "root") {
        members.push(("root".into(), Json::Str(root.into())));
        if let Some(prelude) = backend.reserved_names_prelude() {
            let record = Json::Obj(vec![("prelude".into(), Json::Str(prelude.into()))]);
            members.push(("reserved_names".into(), record));
        }
    }
    if let Some(arch) = text(options, "arch") {
        members.push(("device_arch".into(), Json::Str(arch.into())));
    }
    Json::Obj(members)
}

/// The level and the options of an ask with the defaults the library's documents state put in, so that a build asked for nothing
/// and a build asked for the default are one ask: the key and the `stated.asked` record are made from the result. The level
/// default is the compile document's own; the profile default of a host build is its instruction set card's for the compiler the
/// machine finds, so the machine is asked once for the compiler before the key is made.
fn resolved_ask<B: Backend>(
    backend: &B,
    toolkit: &Toolkit,
    ask: &Json,
    level: String,
    options: Json,
) -> Result<(String, Json), toolkit::Refusal> {
    let dir = facts::facts_dir(None).map_err(refuse)?;
    let cards = CardSet::new(&dir, &[], backend.card_names());
    let failed = |why: Failed| refuse(why.refusal().reason);
    let level = if level.is_empty() {
        facts::default_opt_level(&cards, backend.host_card_wanted()).map_err(failed)?
    } else {
        level
    };
    let Json::Obj(mut members) = options else {
        return Ok((level, options));
    };
    if backend.host_card_wanted()
        && backend.option_names().contains(&"profile")
        && text_in(&members, "profile").is_none()
    {
        let found = toolkit
            .machine(&tool_ask(backend, ask, &Json::Obj(members.clone()), vec![]).to_text())?;
        let found = parse("the toolkit's machine answer", &found)?;
        let (compiler, stated) = stated_of(backend, &cards, &found).map_err(failed)?;
        let stated = facts::stated_from_json(&stated).map_err(failed)?;
        let isa = stated.isa.unwrap_or_else(|| facts::probe().isa);
        let profile = facts::default_host_profile(&cards, &compiler, &isa).map_err(failed)?;
        members.push(("profile".into(), Json::Str(profile)));
    }
    Ok((level, in_name_order(Json::Obj(members))))
}

/// The level and the options a machine is stated for: what the ask states, else what the environment says (`env_level`,
/// `env_profile`), else nothing (the card's default). The answer's `stated` and `key` follow the result, so two builds under
/// different variables never share a cache entry. A value that names no level or profile is refused by name where the lowering
/// reads it, as the same ask in the options is.
fn effective_ask(
    level: &str,
    options: Json,
    env_level: Option<String>,
    env_profile: Option<String>,
) -> (String, Json) {
    let level = if level.is_empty() {
        env_level.unwrap_or_default()
    } else {
        level.to_string()
    };
    let options = match (options, env_profile) {
        (Json::Obj(mut members), Some(profile)) if text_in(&members, "profile").is_none() => {
            members.push(("profile".into(), Json::Str(profile)));
            Json::Obj(members)
        }
        (options, _) => options,
    };
    (level, in_name_order(options))
}

/// `value` with the members of every object in name order, so the same options stated in another order are one ask: the key text
/// and the stated record are made from the result.
fn in_name_order(value: Json) -> Json {
    match value {
        Json::Obj(mut members) => {
            members.sort_by(|a, b| a.0.cmp(&b.0));
            Json::Obj(
                members
                    .into_iter()
                    .map(|(name, member)| (name, in_name_order(member)))
                    .collect(),
            )
        }
        Json::Arr(items) => Json::Arr(items.into_iter().map(in_name_order).collect()),
        other => other,
    }
}

/// The text member `key` of an object's members.
fn text_in<'a>(members: &'a [(String, Json)], key: &str) -> Option<&'a str> {
    members
        .iter()
        .find(|(name, _)| name == key)
        .and_then(|(_, value)| value.str())
}

/// The stated machine (the JSON `facts::stated_from_json` reads) a machine answer stands for, and the compiler identity the facts
/// question takes. A target's own members only: the host's instruction set, caches, profile, header shape (the library embeds both
/// header trees, so `roots`), level and witness verdict; the device's arch, driver and compiler release. A device machine with
/// neither a compiler nor a runtime compile library is written under the card's `text_compiler`: text needs no tool.
fn stated_of<B: Backend>(
    backend: &B,
    cards: &CardSet,
    machine: &Json,
) -> Result<(String, String), Failed> {
    let stated = machine
        .get("stated")
        .ok_or_else(|| Failed::Bad("the machine answer states nothing".into()))?;
    let tool = stated.get("tools").and_then(|t| t.get("compiler"));
    let identity = match tool.and_then(|t| text(t, "identity")) {
        Some(found) => found.replace("||", " || "),
        None if backend.host_card_wanted() => "none || no compiler found || none".to_string(),
        None if stated.get("runtime").is_some() => facts::device_runtime_compiler(cards)?,
        None => facts::device_text_compiler(cards)?,
    };
    let asked = stated.get("asked");
    let level = asked
        .and_then(|a| text(a, "opt_level"))
        .filter(|l| !l.is_empty());
    let mut out: Vec<(String, Json)> = vec![];
    let mut put = |key: &str, value: Option<Json>| out.extend(value.map(|v| (key.to_string(), v)));
    if backend.host_card_wanted() {
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
    } else {
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
    }
    Ok((identity, Json::Obj(out).to_text()))
}

/// The facts answer of the library for the target, from a machine answer it gave: the compiler identity, the probe, the stated
/// machine and the header shape are derived here.
///
/// # Errors
/// Refuses a machine answer that states nothing usable, or the facts question the cards cannot answer.
pub fn facts_for<B: Backend>(backend: &B, machine: &Json) -> Result<String, Failed> {
    let dir = facts::facts_dir(None).map_err(Failed::Card)?;
    let cards = CardSet::new(&dir, &[], backend.card_names());
    let (identity, stated) = stated_of(backend, &cards, machine)?;
    facts::answer(
        &dir,
        backend,
        &identity,
        &facts::probe(),
        &facts::stated_from_json(&stated)?,
    )
}

/// The request of a lowering with its facts: a request that states `facts` (the replay tools, recorded corpora) is lowered as it
/// is, its `machine` member dropped; one that states `machine` for the library's target gets the facts answer derived from it;
/// one that states neither is refused by name.
///
/// # Errors
/// Refuses a request that is not JSON, states neither member, or whose machine has no answer for the library's target.
pub fn with_facts<B: Backend>(backend: &B, request: &str) -> Result<String, Refusal> {
    let mut json = Contract::parse(request)?.0;
    let Json::Obj(parts) = &mut json else {
        return Ok(request.into());
    };
    let machine = parts
        .iter()
        .position(|(k, _)| k == "machine")
        .map(|at| parts.remove(at).1);
    if parts.iter().any(|(k, _)| k == "facts") {
        return Ok(json.to_text());
    }
    let Some(machine) = machine else {
        return Err(bad("lower: the request states neither `machine` (the answer of the library's machine) nor `facts`"));
    };
    let target = format!("{:?}", backend.target());
    let own = machine.get(&target).ok_or_else(|| {
        bad(format!(
            "lower: the request's `machine` has no answer for the target {target}"
        ))
    })?;
    let answer = facts_for(backend, own).map_err(Failed::refusal)?;
    state_option_settings(backend.option_names(), own, parts);
    let named = parts.iter().find(|(k, _)| k == "image_format");
    if let Some((_, Json::Str(format))) = named {
        refuse_unloadable_ptx(own, format).map_err(Failed::refusal)?;
    }
    if named.is_none() {
        if let Some(format) = image_format_of(&answer)? {
            parts.push(("image_format".into(), Json::Str(format)));
        }
    }
    parts.push(("facts".into(), Json::Obj(vec![(target, Json::Str(answer))])));
    Ok(json.to_text())
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

/// The strings of a JSON list.
fn strings(j: Option<&Json>) -> Vec<String> {
    j.map_or(vec![], |l| {
        l.arr()
            .iter()
            .filter_map(|s| s.str().map(str::to_string))
            .collect()
    })
}

/// Builds one unit: `{"name", "text", "machine", "defines"?}` in; the toolkit's answer out with what a caller publishes and links
/// from it: `artifact` (the object's file name), `source` (the source's), `format`, `seconds`, `files` (name to path) and
/// `link_recipe`, the opaque instructions the link needs (the compile of the set, its link, the machine and the defines the unit was
/// compiled with), which the caller hands back untouched in `parts`.
///
/// # Errors
/// Refuses a request without a machine, a machine for another target, and what the tool side refuses.
pub fn build<B: Backend>(
    backend: &B,
    request: &str,
    root: &Path,
) -> Result<String, toolkit::Refusal> {
    let req = parse("the build request", request)?;
    let machine = req
        .get("machine")
        .ok_or_else(|| refuse("build: the request carries no `machine`"))?;
    if let Some(reason) = text(machine, "tools_absent") {
        return Err(toolkit::Refusal::new(toolkit::Kind::ToolAbsent, reason));
    }
    if !backend.host_card_wanted() {
        return build_device(backend, &req, machine, root);
    }
    let facts = parse(
        "the facts answer",
        &facts_for(backend, machine).map_err(|f| refuse(String::from(f)))?,
    )?;
    let recipe = facts
        .get("host_build")
        .ok_or_else(|| refuse("the facts answer has no host_build"))?;
    let name = text(&req, "name").ok_or_else(|| refuse("build: the request has no name"))?;
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
            wire::s(text(&req, "text").ok_or_else(|| refuse("build: the request has no text"))?),
        ),
        ("template", wire::strs(&strings(recipe.get("compile_unit")))),
        (
            "lists",
            wire::obj(vec![("{defines}", wire::strs(&defines))]),
        ),
        ("key_terms", wire::strs(&[key])),
        ("keep_source", rcore::json::J::Bool(true)),
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
            index_bytes.map_or(rcore::json::J::Null, |bytes| wire::n(bytes.unsigned_abs())),
        ),
        (
            "key",
            answer.get("key").cloned().unwrap_or(rcore::json::J::Null),
        ),
        (
            "hit",
            answer.get("hit").cloned().unwrap_or(rcore::json::J::Null),
        ),
        ("seconds", seconds.unwrap_or(rcore::json::J::Null)),
        (
            "files",
            wire::obj(vec![
                (
                    object.as_str(),
                    path("output").unwrap_or(rcore::json::J::Null),
                ),
                (
                    source.as_str(),
                    path("source").unwrap_or(rcore::json::J::Null),
                ),
            ]),
        ),
        (
            "figures",
            answer
                .get("figures")
                .cloned()
                .unwrap_or(rcore::json::J::Null),
        ),
        (
            "report",
            answer
                .get("report")
                .cloned()
                .unwrap_or(rcore::json::J::Null),
        ),
        (
            "link_recipe",
            wire::parse("recipe", &link_recipe.to_text())?,
        ),
    ])))
}

/// Links the library `name` of a set: `parts` are the build answers as given, `set_text` the text of the set. The answer is the
/// toolkit's `{"library": path, ...}`.
///
/// # Errors
/// Refuses no parts, a part without its recipe, and what the tool side refuses.
pub fn link<B: Backend>(
    backend: &B,
    parts: &str,
    set_text: &str,
    name: &str,
    root: &Path,
) -> Result<String, toolkit::Refusal> {
    let parts = parse("the parts", parts)?;
    let first = parts
        .arr()
        .first()
        .ok_or_else(|| refuse("link: no parts"))?;
    let recipe = first
        .get("link_recipe")
        .ok_or_else(|| refuse("link: a part carries no link_recipe"))?;
    let machine = text(recipe, "machine").unwrap_or("");
    let mut listed = vec![];
    for part in parts.arr() {
        let artifact =
            text(part, "artifact").ok_or_else(|| refuse("link: a part has no artifact"))?;
        let output = part.get("files").and_then(|f| f.get(artifact)).cloned();
        let output = output
            .ok_or_else(|| refuse(format!("link: the part {artifact} lists no such file")))?;
        let key = part.get("key").cloned().unwrap_or(Json::Null);
        listed.push(Json::Obj(vec![
            ("files".into(), Json::Obj(vec![("output".into(), output)])),
            ("key".into(), key),
        ]));
    }
    let stem = name.rsplit_once('.').map_or(name, |(stem, _)| stem);
    let set = Json::Obj(vec![
        ("name".into(), Json::Str(format!("{stem}__set"))),
        ("text".into(), Json::Str(set_text.into())),
        (
            "template".into(),
            recipe.get("compile_set").cloned().unwrap_or(Json::Null),
        ),
        (
            "lists".into(),
            Json::Obj(vec![(
                "{defines}".into(),
                recipe.get("defines").cloned().unwrap_or(Json::Arr(vec![])),
            )]),
        ),
        (
            "key_terms".into(),
            Json::Arr(vec![Json::Str(machine.into())]),
        ),
    ]);
    let request = Json::Obj(vec![
        ("name".into(), Json::Str(name.into())),
        ("parts".into(), Json::Arr(listed)),
        ("set".into(), set),
        (
            "link".into(),
            recipe.get("link").cloned().unwrap_or(Json::Null),
        ),
        (
            "key_terms".into(),
            Json::Arr(vec![Json::Str(machine.into())]),
        ),
    ]);
    backend.toolkit()?.link(&request.to_text(), root)
}

/// Builds one device unit: the request is the host's (`name`, `text`, `machine`, `defines`) with the image format the lowering
/// answered if it names one. The machine's facts state which tool makes the image (`compile_with`): the offline compiler gets the
/// argv template, the runtime compile library the options and the header trees it serves from memory. The answer is the host's
/// (`artifact`, `source`, `format`, `key`, `hit`, `seconds`, `files`, `figures`, `report`) with `embed`, the record the host link
/// takes back, and a `link_recipe` that carries the recipe that embeds the image.
fn device_unit(
    req: &Json,
    machine: &Json,
    facts: &Json,
) -> Result<(String, rcore::json::J, Vec<String>, String), toolkit::Refusal> {
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
        ("keep_source", rcore::json::J::Bool(true)),
        (
            "embed",
            wire::obj(vec![
                ("symbol", wire::s(&format!("{name}__image"))),
                ("size_symbol", wire::s(&format!("{name}__image_size"))),
            ]),
        ),
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
fn launch_record(
    facts: &Json,
    name: &str,
    answer: &rcore::json::J,
) -> Result<Option<rcore::json::J>, toolkit::Refusal> {
    use rcore::json::J;
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
fn build_device<B: Backend>(
    backend: &B,
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
    let null = || rcore::json::J::Null;
    let launch = launch_record(&facts, &name, &answer)?;
    let recipe = facts
        .get("device_build")
        .and_then(|b| b.get("embed"))
        .cloned()
        .unwrap_or(Json::Null);
    let link_recipe = Json::Obj(vec![
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
            index_bytes.map_or(rcore::json::J::Null, |bytes| wire::n(bytes.unsigned_abs())),
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
        ("embed", answer.get("embed").cloned().unwrap_or_else(null)),
        (
            "link_recipe",
            wire::parse("recipe", &link_recipe.to_text())?,
        ),
    ])))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(profile: Option<&str>) -> Json {
        Json::Obj(
            profile
                .map(|p| ("profile".to_string(), Json::Str(p.into())))
                .into_iter()
                .collect(),
        )
    }

    fn profile_of(options: &Json) -> Option<String> {
        options.get("profile").and_then(Json::str).map(String::from)
    }

    #[test]
    fn an_option_beats_the_variable_and_the_variable_beats_the_default() {
        let env = |l: &str, p: &str| (Some(l.to_string()), Some(p.to_string()));
        let (env_level, env_profile) = env("O1", "portable");
        // the option wins
        let (level, opts) = effective_ask(
            "O2",
            options(Some("native")),
            env_level.clone(),
            env_profile.clone(),
        );
        assert_eq!(
            (level.as_str(), profile_of(&opts).as_deref()),
            ("O2", Some("native"))
        );
        // the variable decides when the ask states none
        let (level, opts) = effective_ask("", options(None), env_level, env_profile);
        assert_eq!(
            (level.as_str(), profile_of(&opts).as_deref()),
            ("O1", Some("portable"))
        );
        // with neither, nothing is stated and the card's default stands
        let (level, opts) = effective_ask("", options(None), None, None);
        assert_eq!((level.as_str(), profile_of(&opts)), ("", None));
    }

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

    #[test]
    fn an_option_the_target_does_not_declare_is_refused_by_name() {
        use crate::testing::TestHost;
        let ask = |options: &str| {
            machine(
                &TestHost,
                &format!("{{\"target\":\"host\",\"options\":{options}}}"),
            )
        };
        let why = ask("{\"bogus\":1}").unwrap_err().message;
        assert!(
            why.contains("\"bogus\"") && why.contains("declared by the target host"),
            "{why}"
        );
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

    /// Oracle: the default level and the default profile the shipped documents state (`default_opt_level`, `default_profile`).
    #[test]
    fn the_defaults_of_an_unset_ask_are_the_ones_the_documents_state() {
        let docs = CardSet::new(&crate::testing::shipped(), &[], crate::testing::ALL_CARDS);
        assert_eq!(facts::default_opt_level(&docs, true).unwrap(), "O3");
        assert_eq!(facts::default_opt_level(&docs, false).unwrap(), "O3");
        let gcc = "/usr/bin/g++ || g++ (GCC) 13.2.0 || 0";
        let clang = "/usr/bin/c++ || Apple clang version 21.0.0 || 0";
        let default = |compiler, isa| facts::default_host_profile(&docs, compiler, isa).unwrap();
        assert_eq!(default(gcc, "arm64"), "native-vector-math");
        assert_eq!(default(clang, "arm64"), "native-vector-math");
        assert_eq!(default(gcc, "x86_64"), "native-vector-math");
        assert_eq!(default(clang, "x86_64"), "native");
        assert!(facts::default_host_profile(&docs, gcc, "riscv").is_err());
    }

    /// Oracle: the cpu card shipped in `osprey_cpu/facts/cpu.json`, which lists `native` and `portable` as exact.
    #[test]
    fn a_profile_is_exact_when_the_cpu_card_lists_it() {
        let docs = CardSet::new(&crate::testing::shipped(), &[], crate::testing::ALL_CARDS);
        assert!(facts::profile_is_exact(&docs, "native").unwrap());
        assert!(facts::profile_is_exact(&docs, "portable").unwrap());
        assert!(!facts::profile_is_exact(&docs, "native-vector-math").unwrap());
    }

    #[test]
    fn the_key_terms_of_two_variables_differ() {
        let terms = |level: &str, profile: &str| {
            let (level, opts) =
                effective_ask("", options(None), Some(level.into()), Some(profile.into()));
            format!("opt_level:{level}|options:{}", opts.to_text())
        };
        assert_ne!(terms("O1", "native"), terms("O3", "native"));
        assert_ne!(terms("O1", "native"), terms("O1", "portable"));
    }

    /// Oracle: the options of an ask as the key text and the stated record are made from them (`effective_ask`'s result).
    #[test]
    fn the_same_options_in_another_order_are_one_ask_and_a_changed_value_is_another() {
        let ask = |members: &[(&str, &str)]| {
            let options = Json::Obj(
                members
                    .iter()
                    .map(|(k, v)| ((*k).to_string(), Json::Str((*v).into())))
                    .collect(),
            );
            effective_ask("", options, None, None).1.to_text()
        };
        let forward = ask(&[("arch", "sm_61"), ("profile", "native")]);
        assert_eq!(forward, ask(&[("profile", "native"), ("arch", "sm_61")]));
        assert_eq!(forward, "{\"arch\":\"sm_61\",\"profile\":\"native\"}");
        assert_ne!(forward, ask(&[("arch", "sm_89"), ("profile", "native")]));
    }
}
