//! The merge of the answers of the libraries of one build: each library lowered the kernel for its own target, and the merge
//! returns the one answer a single library serving every target would have given.
use super::meta_c;
use super::request::{KernelAnswer, UnitAnswer, TARGETS};
use super::unit_meta::UnitMeta;
use aether_writers::contract::Contract;
use aether_writers::refuse::{bad, Refusal};

/// The one answer over the answers of a build's libraries, given as a JSON array of the answers as the libraries returned them.
/// Each member of a unit the answers state for one target comes from the answer of that target; every other member must be the
/// same in all of them, else the libraries did not lower to one cut and the merge is refused by the unit it differs in. The texts
/// of a unit are in the order of the targets (host, then device), and the C source of the meta is written again over the merged
/// units.
pub fn merge(answers: &str) -> Result<String, Refusal> {
    let parsed = Contract::parse(answers)?;
    let mut found = parsed
        .0
        .arr()
        .iter()
        .map(|answer| KernelAnswer::from_json(&answer.to_text()))
        .collect::<Result<Vec<_>, _>>()?;
    if found.is_empty() {
        return Err(bad("a merge of no answers"));
    }
    found.sort_by_key(order_of);
    if found
        .windows(2)
        .any(|pair| order_of(&pair[0]) == order_of(&pair[1]))
    {
        return Err(bad("merge: two answers are for the same target"));
    }
    let mut merged = found.remove(0);
    for next in found {
        merged = join(merged, next)?;
    }
    merged.peer = None;
    let metas: Vec<&UnitMeta> = merged.units.iter().map(|u| &u.meta).collect();
    merged.meta_source = meta_c::source(&merged.library, &metas)?;
    Ok(merged.to_json())
}

/// The place of the target an answer serves in the order of targets: the target of its first text (a unit without a text is a
/// source unit and has none).
fn order_of(answer: &KernelAnswer) -> usize {
    let target = answer
        .units
        .iter()
        .flat_map(|unit| &unit.texts)
        .map(|text| text.target)
        .next();
    TARGETS.iter().position(|t| Some(*t) == target).unwrap_or(0)
}

/// The member two answers state: the one that is stated, or the value they share.
fn either<T: PartialEq>(
    first: Option<T>,
    second: Option<T>,
    what: &str,
) -> Result<Option<T>, Refusal> {
    match (first, second) {
        (Some(a), Some(b)) if a != b => {
            Err(bad(format!("merge: the answers state different {what}")))
        }
        (a, b) => Ok(a.or(b)),
    }
}

/// Joins the answer of an earlier target with the answer of a later one.
fn join(first: KernelAnswer, second: KernelAnswer) -> Result<KernelAnswer, Refusal> {
    if first.units.len() != second.units.len() {
        return Err(bad(format!(
            "merge: the answers hold {} and {} units",
            first.units.len(),
            second.units.len()
        )));
    }
    if first.library != second.library || first.working_set_bytes != second.working_set_bytes {
        return Err(bad(
            "merge: the answers differ in the library record or the working set",
        ));
    }
    let units = first
        .units
        .into_iter()
        .zip(second.units)
        .map(|(a, b)| join_unit(a, b))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(KernelAnswer {
        units,
        library: first.library,
        meta_source: first.meta_source,
        working_set_bytes: first.working_set_bytes,
        wide_class: either(first.wide_class, second.wide_class, "wide classes")?,
        steps_reverse: super::unit_meta::StepsPair {
            host: either(
                first.steps_reverse.host,
                second.steps_reverse.host,
                "host steps tables",
            )?,
            device: either(
                first.steps_reverse.device,
                second.steps_reverse.device,
                "device steps tables",
            )?,
        },
        dispatch_costs: either(
            first.dispatch_costs,
            second.dispatch_costs,
            "dispatch costs",
        )?,
        peer: None,
    })
}

/// Joins the unit of an earlier target's answer with the same unit of a later one: the texts side by side, and the meta with
/// the members of each target taken from its own answer.
fn join_unit(first: UnitAnswer, second: UnitAnswer) -> Result<UnitAnswer, Refusal> {
    let name = first.meta.name.clone();
    let mut merged = first.meta.clone();
    device_members(&mut merged, &second.meta);
    let mut check = second.meta.clone();
    host_members(&mut check, &first.meta);
    merged.flags |= second.meta.flags;
    check.flags |= first.meta.flags;
    if merged != check {
        return Err(bad(format!(
            "merge: the answers do not lower the unit {name:?} to one meta"
        )));
    }
    let mut texts = first.texts;
    texts.extend(second.texts);
    Ok(UnitAnswer {
        texts,
        meta: merged,
    })
}

/// Copies into `dst` every member of a unit meta that the device's answer alone states.
fn device_members(dst: &mut UnitMeta, src: &UnitMeta) {
    dst.device_image = src.device_image.clone();
    dst.device_function = src.device_function.clone();
    dst.device_image_bytes = src.device_image_bytes;
    dst.device_image_format = src.device_image_format;
    dst.device_image_size = src.device_image_size.clone();
    dst.device_compiler = src.device_compiler.clone();
    dst.device_flags = src.device_flags.clone();
    dst.device_ptx_isa = src.device_ptx_isa;
    dst.launch.device_box = src.launch.device_box.clone();
    if let (Some(to), Some(from)) = (&mut dst.finish, &src.finish) {
        to.device_persist_function = from.device_persist_function.clone();
        to.device_range_function = from.device_range_function.clone();
        to.device_steps_reverse = from.device_steps_reverse.clone();
    }
    for (to, from) in dst.body.companions.iter_mut().zip(&src.body.companions) {
        to.device_fold_function = from.device_fold_function.clone();
    }
}

/// Copies into `dst` every member of a unit meta that the host's answer alone states.
fn host_members(dst: &mut UnitMeta, src: &UnitMeta) {
    dst.host_entry = src.host_entry.clone();
    dst.launch.host_box = src.launch.host_box.clone();
    if let (Some(to), Some(from)) = (&mut dst.finish, &src.finish) {
        to.host_steps_reverse = from.host_steps_reverse.clone();
    }
    for (to, from) in dst.body.companions.iter_mut().zip(&src.body.companions) {
        to.host_fold_entry = from.host_fold_entry.clone();
    }
}
