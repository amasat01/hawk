//! The tool side as a caller that knows nothing of the target uses it: `machine` takes what a build asks (target, options,
//! optimisation level, a cache root), `lower` derives its facts from the machine it is given, `build` and `link` read the library's
//! own build card. No compiler flag, header shape or tool identity crosses the door in either direction; the library states the
//! machine once and every later call carries that answer back as given.
use crate::backend::{unserved, Backend};
use crate::facts::{self, CardSet, Failed};
use crate::lower::unit_meta::image_symbols;
use aether_writers::contract::{Contract, Json};
use aether_writers::refuse::{bad, Refusal};
use std::path::Path;
use toolkit::Toolkit;

/// The text of the member `key` of `j`.
pub fn text<'a>(j: &'a Json, key: &str) -> Option<&'a str> {
    j.get(key).and_then(Json::str)
}

/// A refusal of the tool side with `reason` as a malformed request.
pub fn refuse(reason: impl Into<String>) -> toolkit::Refusal {
    toolkit::Refusal::bad(reason)
}

/// The parsed JSON of `source`, as a refusal of the tool side when it is not JSON.
pub fn parse(what: &str, source: &str) -> Result<Json, toolkit::Refusal> {
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
    let wanted = backend.target_name();
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
        &backend
            .env_option_asks()
            .iter()
            .map(|(option, ask)| (*option, toolkit.ask_from_env(ask)))
            .collect::<Vec<_>>(),
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
    let extras = backend.stated_extras(&asked)?;
    for (name, value) in &mut parts {
        if let (Json::Obj(stated), "stated") = (value, name.as_str()) {
            stated.push(("asked".into(), asked.clone()));
            stated.extend(extras.clone());
        }
    }
    Ok(Json::Obj(parts).to_text())
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
    let known = backend
        .opt_levels(&cards)
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

/// The ask the toolkit answers: the key's `terms` beyond what the machine states, the cache root with the reserved names, and the
/// members the backend adds for the probes it registered.
pub fn tool_ask<B: Backend>(backend: &B, ask: &Json, options: &Json, terms: Vec<Json>) -> Json {
    let mut members = vec![("key_terms".to_string(), Json::Arr(terms))];
    if let Some(root) = text(ask, "root") {
        members.push(("root".into(), Json::Str(root.into())));
        if let Some(prelude) = backend.reserved_names_prelude() {
            let record = Json::Obj(vec![("prelude".into(), Json::Str(prelude.into()))]);
            members.push(("reserved_names".into(), record));
        }
    }
    members.extend(backend.machine_ask(options));
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
        backend.default_opt_level(&cards).map_err(failed)?
    } else {
        level
    };
    let Json::Obj(mut members) = options else {
        return Ok((level, options));
    };
    let added = backend.default_options(&cards, toolkit, ask, &Json::Obj(members.clone()))?;
    members.extend(added);
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
    env_options: &[(&str, Option<String>)],
) -> (String, Json) {
    let level = if level.is_empty() {
        env_level.unwrap_or_default()
    } else {
        level.to_string()
    };
    let options = match options {
        Json::Obj(mut members) => {
            for (name, value) in env_options {
                if let (Some(value), None) = (value, text_in(&members, name)) {
                    members.push(((*name).to_string(), Json::Str(value.clone())));
                }
            }
            Json::Obj(members)
        }
        options => options,
    };
    (level, in_name_order(options))
}

/// `value` with the members of every object in name order, so the same options stated in another order are one ask: the key text
/// and the stated record are made from the result.
pub fn in_name_order(value: Json) -> Json {
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
pub fn text_in<'a>(members: &'a [(String, Json)], key: &str) -> Option<&'a str> {
    members
        .iter()
        .find(|(name, _)| name == key)
        .and_then(|(_, value)| value.str())
}

/// The facts answer of the library for the target, from a machine answer it gave: the compiler identity, the probe, the stated
/// machine and the header shape are derived here.
///
/// # Errors
/// Refuses a machine answer that states nothing usable, or the facts question the cards cannot answer.
pub fn facts_for<B: Backend>(backend: &B, machine: &Json) -> Result<String, Failed> {
    let dir = facts::facts_dir(None).map_err(Failed::Card)?;
    let cards = CardSet::new(&dir, &[], backend.card_names());
    let (identity, stated) = backend.stated_machine(&cards, machine)?;
    let (record, docs) = facts::split_stated(&stated)?;
    facts::answer(
        &dir,
        backend,
        &identity,
        &docs,
        &backend.probe(),
        &backend.read_stated(&record)?,
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
    backend.with_facts_request(own, &answer, parts)?;
    parts.push(("facts".into(), Json::Obj(vec![(target, Json::Str(answer))])));
    Ok(json.to_text())
}

/// The strings of a JSON list.
pub fn strings(j: Option<&Json>) -> Vec<String> {
    j.map_or(vec![], |l| {
        l.arr()
            .iter()
            .filter_map(|s| s.str().map(str::to_string))
            .collect()
    })
}

/// Builds one unit: `{"name", "text", "machine", "defines"?}` in; the toolkit's answer out with what a caller publishes and links
/// from it: `artifact` (the object's file name), `source` (the source's), `format`, `seconds`, `files` (name to path) and
/// `link_recipe`, the opaque instructions the link needs, which the caller hands back untouched in `parts`. The build itself is the
/// backend's ([`Backend::build_unit`]).
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
    backend.build_unit(&req, machine, root)
}

/// Links the library `name` of a set: `parts` are the build answers as given, `set_text` the text of the set. The link is the
/// backend's ([`Backend::link`]).
///
/// # Errors
/// Refuses what the backend's link refuses.
pub fn link<B: Backend>(
    backend: &B,
    parts: &str,
    set_text: &str,
    name: &str,
    root: &Path,
) -> Result<String, toolkit::Refusal> {
    backend.link(parts, set_text, name, root)
}

/// Links the library `name` of a set: `parts` are the build answers as given, `set_text` the text of the set. A part that holds a
/// device image (its `link_recipe` names the `unit` it belongs to) is not linked as an object: the link defines the image under
/// the symbols the unit's meta names, from the embed recipe the part carries. The answer is the toolkit's `{"library": path, ...}`.
///
/// # Errors
/// Refuses no parts, no host part, a part without its recipe, and what the tool side refuses.
pub fn link_parts(
    toolkit: &Toolkit,
    parts: &str,
    set_text: &str,
    name: &str,
    root: &Path,
) -> Result<String, toolkit::Refusal> {
    let parts = parse("the parts", parts)?;
    let (mut listed, mut images, mut embed) = (vec![], vec![], None);
    let mut host_recipe = None;
    for part in parts.arr() {
        let recipe = part
            .get("link_recipe")
            .ok_or_else(|| refuse("link: a part carries no link_recipe"))?;
        let artifact =
            text(part, "artifact").ok_or_else(|| refuse("link: a part has no artifact"))?;
        let output = part.get("files").and_then(|f| f.get(artifact)).cloned();
        let output = output
            .ok_or_else(|| refuse(format!("link: the part {artifact} lists no such file")))?;
        if let Some(unit) = text(recipe, "unit") {
            let (symbol, size_symbol) = image_symbols(unit);
            images.push(Json::Obj(vec![
                ("image".into(), output),
                ("symbol".into(), Json::Str(symbol)),
                ("size_symbol".into(), Json::Str(size_symbol)),
            ]));
            embed = recipe.get("embed").cloned();
            continue;
        }
        host_recipe.get_or_insert(recipe);
        let key = part.get("key").cloned().unwrap_or(Json::Null);
        listed.push(Json::Obj(vec![
            ("files".into(), Json::Obj(vec![("output".into(), output)])),
            ("key".into(), key),
        ]));
    }
    let recipe = host_recipe.ok_or_else(|| refuse("link: no part holds a host object"))?;
    let machine = text(recipe, "machine").unwrap_or("");
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
        ("images".into(), Json::Arr(images)),
        ("embed".into(), embed.unwrap_or(Json::Null)),
        // the embed recipe assembles with the host compiler of the machine; this library's compiler is the host's
        (
            "as".into(),
            Json::Obj(vec![("host_compiler".into(), Json::Str("compiler".into()))]),
        ),
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
    toolkit.link(&request.to_text(), root)
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
            &[("profile", env_profile.clone())],
        );
        assert_eq!(
            (level.as_str(), profile_of(&opts).as_deref()),
            ("O2", Some("native"))
        );
        // the variable decides when the ask states none
        let (level, opts) =
            effective_ask("", options(None), env_level, &[("profile", env_profile)]);
        assert_eq!(
            (level.as_str(), profile_of(&opts).as_deref()),
            ("O1", Some("portable"))
        );
        // with neither, nothing is stated and the card's default stands
        let (level, opts) = effective_ask("", options(None), None, &[("profile", None)]);
        assert_eq!((level.as_str(), profile_of(&opts)), ("", None));
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

    #[test]
    fn the_key_terms_of_two_variables_differ() {
        let terms = |level: &str, profile: &str| {
            let (level, opts) = effective_ask(
                "",
                options(None),
                Some(level.into()),
                &[("profile", Some(profile.into()))],
            );
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
            effective_ask("", options, None, &[]).1.to_text()
        };
        let forward = ask(&[("arch", "gen_a"), ("profile", "native")]);
        assert_eq!(forward, ask(&[("profile", "native"), ("arch", "gen_a")]));
        assert_eq!(forward, "{\"arch\":\"gen_a\",\"profile\":\"native\"}");
        assert_ne!(forward, ask(&[("arch", "gen_b"), ("profile", "native")]));
    }
}
