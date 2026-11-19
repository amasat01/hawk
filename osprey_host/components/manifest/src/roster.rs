//! The OSPREYs a search found, and which of them serves a target.
use crate::json::{emit, quote};
use crate::manifest;
use crate::target::canonical;
use rcore::json::{parse, J};
use std::path::Path;

/// The roster of `folders` (a JSON list of paths) as the object `{"ospreys": [<checked manifest>, ...], "refused": {<folder>:
/// <why>}, "searched": [<folder>, ...]}`. A folder whose manifest is refused is isolated: the others are kept and its reason is stated. A folder whose
/// OSPREY has the name of an earlier one is skipped.
///
/// # Errors
/// The sentence naming the target id and both OSPREYs when two OSPREYs serve one canonical id, or the one saying `folders` is
/// not a JSON list of paths.
pub fn of_folders(folders: &str) -> Result<String, String> {
    let J::Arr(list) = parse(folders)? else {
        return Err("the folders are not a JSON list of paths".into());
    };
    let (mut found, mut refused): (Vec<J>, Vec<(String, J)>) = (vec![], vec![]);
    for folder in &list {
        let J::Str(folder) = folder else {
            return Err("the folders are not a JSON list of paths".into());
        };
        match manifest::read(Path::new(folder)) {
            Ok(one) => {
                if found.iter().all(|seen| {
                    seen.get("name").and_then(J::str) != one.get("name").and_then(J::str)
                }) {
                    found.push(one);
                }
            }
            Err(why) => refused.push((folder.clone(), J::Str(why))),
        }
    }
    owners(&found)?;
    Ok(emit(&J::Obj(vec![
        ("ospreys".into(), J::Arr(found)),
        ("refused".into(), J::Obj(refused)),
        ("searched".into(), J::Arr(list)),
    ])))
}

/// `(target id, OSPREY name)` of every target of `ospreys`, in order.
///
/// # Errors
/// The sentence naming the id and both OSPREYs when two of them state one id.
fn owners(ospreys: &[J]) -> Result<Vec<(String, String)>, String> {
    let mut owned: Vec<(String, String)> = vec![];
    for osprey in ospreys {
        let who = osprey.get("name").and_then(J::str).unwrap_or_default();
        for row in osprey.get("targets").map_or(&[][..], J::arr) {
            let id = row.get("request").and_then(J::str).unwrap_or_default();
            if let Some((_, first)) = owned.iter().find(|(known, _)| known == id) {
                return Err(format!(
                    "target {} is served by both {} and {}; keep one",
                    quote(id),
                    quote(first),
                    quote(who)
                ));
            }
            owned.push((id.to_string(), who.to_string()));
        }
    }
    Ok(owned)
}

/// The served ids as a tuple is spelled in a sentence, `none` for no id.
fn served(ids: &[&str]) -> String {
    match ids {
        [] => "none".into(),
        [one] => format!("({},)", quote(one)),
        many => format!(
            "({})",
            many.iter().map(|i| quote(i)).collect::<Vec<_>>().join(", ")
        ),
    }
}

/// Which OSPREY serves `target` (any spelling of the id), over `roster` as [`of_folders`] answered it: the object `{"id": the
/// canonical id, "osprey": the OSPREY's name}`.
///
/// # Errors
/// The sentence of [`canonical`] for a name that is no target id; for an id nobody serves, the sentence naming it, the ids
/// that are served, the folders searched and every folder refused with its reason.
pub fn serving(roster: &str, target: &str) -> Result<String, String> {
    let id = canonical(target)?;
    let roster = parse(roster)?;
    let ospreys = roster.get("ospreys").map_or(&[][..], J::arr);
    let owned = owners(ospreys)?;
    if let Some((id, osprey)) = owned.iter().find(|(known, _)| *known == id) {
        return Ok(emit(&J::Obj(vec![
            ("id".into(), J::Str(id.clone())),
            ("osprey".into(), J::Str(osprey.clone())),
        ])));
    }
    let ids: Vec<&str> = owned.iter().map(|(known, _)| known.as_str()).collect();
    let mut why = String::new();
    for (_, reason) in roster.get("refused").map_or(&[][..], J::obj) {
        why.push_str("; refused: ");
        why.push_str(reason.str().unwrap_or_default());
    }
    let searched: Vec<&str> = roster
        .get("searched")
        .map_or(&[][..], J::arr)
        .iter()
        .filter_map(J::str)
        .collect();
    let searched = if searched.is_empty() {
        "no folder".to_string()
    } else {
        searched.join(", ")
    };
    Err(format!(
        "target {} is served by no OSPREY found; the served targets are {} (searched: {searched}){why}",
        quote(&id),
        served(&ids)
    ))
}
