//! The tool side as a caller that knows nothing of the target uses it: `machine` takes what a build asks (target, options,
//! optimisation level, a cache root), `lower` derives its facts from the machine it is given, `build` and `link` read the library's
//! own build card. No compiler flag, header shape or tool identity crosses the door in either direction; the library states the
//! machine once and every later call carries that answer back as given.
use crate::backend::{target_name, unserved, Backend};
use crate::facts::{self, Failed};
use aether_writers::contract::{Contract, Json};
use aether_writers::refuse::{bad, Refusal};
use std::path::Path;
use toolkit::wire;

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
/// Refuses an ask that is not JSON or names another target.
pub fn machine<B: Backend>(backend: &B, ask: &str) -> Result<String, toolkit::Refusal> {
    let ask = parse("the machine ask", ask)?;
    let wanted = target_name(backend.target());
    if text(&ask, "target") != Some(wanted) {
        return Err(refuse(
            unserved(backend, text(&ask, "target").unwrap_or("")).reason,
        ));
    }
    let options = ask.get("options").cloned().unwrap_or(Json::Obj(vec![]));
    let level = text(&ask, "opt_level").unwrap_or("");
    let mut tool_ask = vec![
        (
            "key_terms".to_string(),
            Json::Arr(vec![
                Json::Str(backend.identity().into()),
                Json::Str(format!("opt_level:{level}")),
                Json::Str(format!("options:{}", options.to_text())),
            ]),
        ),
        ("host".into(), Json::Bool(backend.host_card_wanted())),
    ];
    if let Some(root) = text(&ask, "root") {
        tool_ask.push(("root".into(), Json::Str(root.into())));
        if let Some(prelude) = backend.reserved_names_prelude() {
            let record = Json::Obj(vec![("prelude".into(), Json::Str(prelude.into()))]);
            tool_ask.push(("reserved_names".into(), record));
        }
    }
    if let Some(arch) = text(&options, "arch") {
        tool_ask.push(("device_arch".into(), Json::Str(arch.into())));
    }
    let answer = backend.toolkit()?.machine(&Json::Obj(tool_ask).to_text())?;
    let Json::Obj(mut parts) = parse("the toolkit's machine answer", &answer)? else {
        return Err(refuse("the toolkit's machine answer is not an object"));
    };
    let asked = Json::Obj(vec![
        ("opt_level".into(), Json::Str(level.into())),
        ("options".into(), options),
    ]);
    for (name, value) in &mut parts {
        if let (Json::Obj(stated), "stated") = (value, name.as_str()) {
            stated.push(("asked".into(), asked.clone()));
        }
    }
    Ok(Json::Obj(parts).to_text())
}

/// The stated machine (the JSON `facts::stated_from_json` reads) a machine answer stands for, and the compiler identity the facts
/// question takes. A target's own members only: the host's instruction set, caches, profile, header shape (the library embeds both
/// header trees, so `roots`), level and witness verdict; the device's arch, driver and compiler release.
fn stated_of<B: Backend>(backend: &B, machine: &Json) -> Result<(String, String), Failed> {
    let stated = machine
        .get("stated")
        .ok_or_else(|| Failed::Bad("the machine answer states nothing".into()))?;
    let tool = stated.get("tools").and_then(|t| t.get("compiler"));
    let identity = match tool.and_then(|t| text(t, "identity")) {
        Some(found) => found.replace("||", " || "),
        None if backend.host_card_wanted() => "none || no compiler found || none".to_string(),
        None if stated.get("runtime").is_some() => {
            "none || the runtime compile library || none".to_string()
        }
        None => {
            let why = text(machine, "tools_absent").unwrap_or("no device compiler");
            return Err(Failed::Bad(format!(
                "the machine states no device compiler and no runtime compile library, and a device unit's text needs the release of one: {why}"
            )));
        }
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
    }
    Ok((identity, Json::Obj(out).to_text()))
}

/// The facts answer of the library for the target, from a machine answer it gave: the compiler identity, the probe, the stated
/// machine and the header shape are derived here.
///
/// # Errors
/// Refuses a machine answer that states nothing usable, or the facts question the cards cannot answer.
pub fn facts_for<B: Backend>(backend: &B, machine: &Json) -> Result<String, Failed> {
    let (identity, stated) = stated_of(backend, machine)?;
    let dir = facts::facts_dir(None).map_err(Failed::Card)?;
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
    parts.push(("facts".into(), Json::Obj(vec![(target, Json::Str(answer))])));
    Ok(json.to_text())
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
/// `link_recipe`, the opaque instructions the link needs, which the caller hands back untouched in `parts`.
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
        return Err(refuse(
            "build: this library builds no unit through the door yet",
        ));
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
    ]);
    Ok(wire::render(&wire::obj(vec![
        ("artifact", wire::s(&object)),
        ("source", wire::s(&source)),
        ("format", wire::s(backend.format())),
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
            Json::Obj(vec![("{defines}".into(), Json::Arr(vec![]))]),
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
