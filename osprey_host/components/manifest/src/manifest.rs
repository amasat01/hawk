//! Reading and checking the manifest `osprey.json` of one OSPREY folder.
use crate::json::name;
use crate::target::{canonical, not_an_id};
use rcore::json::{parse, J};
use std::path::Path;

/// The file a folder holds its manifest in.
pub const FILE: &str = "osprey.json";
/// The members a manifest states, in the order the checked manifest keeps them.
const MEMBERS: [&str; 4] = ["name", "version", "door", "targets"];
/// The members every row of `targets` states.
const ROW_MEMBERS: [&str; 6] = ["request", "target", "about", "device", "entry", "format"];

/// The members of `wanted` that the object `object` does not state, in the order of `wanted`.
fn lacking<'a>(object: &J, wanted: &[&'a str]) -> Vec<&'a str> {
    wanted
        .iter()
        .copied()
        .filter(|m| object.get(m).is_none())
        .collect()
}

/// The checked row for `row` of the manifest at `path`: every member present, the request in its canonical spelling.
fn checked_row(path: &str, row: &J) -> Result<J, String> {
    let J::Obj(members) = row else {
        return Err(format!("{path}: a target row is not an object"));
    };
    let missing = lacking(row, &ROW_MEMBERS);
    if !missing.is_empty() {
        let request = row.get("request").map_or_else(|| "None".to_string(), name);
        return Err(format!(
            "{path}: target {request} lacks {}",
            missing.join(", ")
        ));
    }
    let request = match row.get("request") {
        Some(J::Str(text)) => canonical(text),
        other => Err(not_an_id(&other.map_or_else(|| "None".to_string(), name))),
    }
    .map_err(|why| format!("{path}: the target request: {why}"))?;
    let pairs = members
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                if k == "request" {
                    J::Str(request.clone())
                } else {
                    v.clone()
                },
            )
        })
        .collect();
    Ok(J::Obj(pairs))
}

/// The refusal for a manifest whose members or keys are not those of an OSPREY, or `None` when its shape is right.
fn shape_refusal(path: &str, manifest: &J) -> Option<String> {
    let J::Obj(pairs) = manifest else {
        return Some(format!("{path}: the manifest is not a JSON object"));
    };
    let missing = lacking(manifest, &MEMBERS);
    if !missing.is_empty() {
        return Some(format!("{path}: the manifest lacks {}", missing.join(", ")));
    }
    if manifest.get("wings").is_some() {
        return Some(format!(
            "{path}: an OSPREY keeps its wings inside its library; a Python wing is refused"
        ));
    }
    let mut unknown: Vec<&str> = pairs
        .iter()
        .map(|(k, _)| k.as_str())
        .filter(|k| !MEMBERS.contains(k))
        .collect();
    unknown.sort_unstable();
    unknown.first().map(|key| {
        format!(
            "{path}: unknown manifest key {} (the keys are {}); a misspelt key is refused, never ignored",
            crate::json::quote(key),
            crate::json::python_list(&MEMBERS)
        )
    })
}

/// The manifest of `folder`, checked, as the object `{"folder", "name", "version", "door", "targets"}` whose target requests
/// are in their canonical spelling.
///
/// # Errors
/// The sentence that names the manifest file and the rule broken: the file is not readable JSON, a member is missing or
/// unknown, a Python wing is stated, a row lacks a member, a target id is outside the grammar, or one id is stated in more
/// than one row.
pub fn read(folder: &Path) -> Result<J, String> {
    let file = folder.join(FILE);
    let path = file.display().to_string();
    let text = std::fs::read_to_string(&file)
        .map_err(|why| format!("{path}: no readable {FILE} ({why})"))?;
    let manifest = parse(&text).map_err(|why| format!("{path}: no readable {FILE} ({why})"))?;
    if let Some(refusal) = shape_refusal(&path, &manifest) {
        return Err(refusal);
    }
    let J::Arr(rows) = manifest.get("targets").cloned().unwrap_or(J::Null) else {
        return Err(format!("{path}: the manifest member targets is not a list"));
    };
    let rows = rows
        .iter()
        .map(|row| checked_row(&path, row))
        .collect::<Result<Vec<_>, _>>()?;
    let ids: Vec<&str> = rows
        .iter()
        .filter_map(|r| r.get("request")?.str())
        .collect();
    let mut twice: Vec<&&str> = ids
        .iter()
        .filter(|id| ids.iter().filter(|o| o == id).count() > 1)
        .collect();
    twice.sort_unstable();
    if let Some(twice) = twice.first() {
        return Err(format!(
            "{path}: the manifest states the target id {} in more than one row (ids are compared in their one spelling)",
            crate::json::quote(twice)
        ));
    }
    let mut checked = vec![("folder".to_string(), J::Str(folder.display().to_string()))];
    for member in MEMBERS {
        let value = if member == "targets" {
            J::Arr(rows.clone())
        } else {
            manifest.get(member).cloned().unwrap_or(J::Null)
        };
        checked.push((member.to_string(), value));
    }
    Ok(J::Obj(checked))
}
