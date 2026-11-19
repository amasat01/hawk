//! The local memory one thread of a device unit holds, and its refusal past the device's per-thread limit.
//!
//! A cubin states the frame of each kernel in the figures the compile read (`stack_B`): the unit holds the largest, and the number
//! is the compiler's own (`ptxas`). A PTX text has no frame yet: the local depots it declares are a ceiling, because the assembler
//! can only shrink them (`ptx`). An entry runs alone, and the device functions it may call add to it. The limit is a fact of the
//! card (`local_limit_B`); nothing here knows a number.
use rcore::json::J;

/// What a unit holds per thread and how that was measured: `"ptxas"` or `"ptx"`.
pub(crate) type Held = (u64, &'static str);

/// The bytes a depot declaration on `line` reserves (`.local .align 8 .b8 __local_depot0[1000];` is 1000), when the line has one.
fn depot_bytes(line: &str) -> Option<u64> {
    let line = line.trim_start().strip_prefix(".local")?;
    let at = line.find("__local_depot")?;
    if line[..at].contains(';') {
        return None;
    }
    let rest = line[at + "__local_depot".len()..].trim_start_matches(|c: char| c.is_ascii_digit());
    let size = rest.strip_prefix('[')?.split(']').next()?;
    size.parse().ok()
}

/// Whether `line` opens a device function (`Some(true)`) or a kernel entry (`Some(false)`); `.visible` and `.weak` may precede.
fn opens(line: &str) -> Option<bool> {
    let mut words = line.split_whitespace();
    let mut word = words.next()?;
    if word == ".visible" || word == ".weak" {
        word = words.next()?;
    }
    match word.split('(').next()? {
        ".entry" => Some(false),
        ".func" => Some(true),
        _ => None,
    }
}

/// The local bytes the PTX `text` declares: the largest entry plus the device functions.
fn ptx_depots(text: &str) -> u64 {
    let (mut entries, mut functions) = (vec![], 0);
    let mut open: Option<(bool, u64)> = None;
    let mut close = |open: &mut Option<(bool, u64)>| match open.take() {
        Some((true, bytes)) => functions += bytes,
        Some((false, bytes)) => entries.push(bytes),
        None => {}
    };
    for line in text.lines() {
        if let Some(function) = opens(line) {
            close(&mut open);
            open = Some((function, 0));
        } else if let (Some((_, held)), Some(bytes)) = (open.as_mut(), depot_bytes(line)) {
            *held += bytes;
        }
    }
    close(&mut open);
    entries.into_iter().max().unwrap_or(0) + functions
}

/// The local memory per thread of the unit: the largest `stack_B` of the compile's `figures` when it read any, else, for a PTX
/// image (`ptx_text` is its text), the depots it declares; `None` for an image that states neither. `ptx_text` is asked only
/// when the figures state no frame.
pub(crate) fn held(figures: &J, ptx_text: impl FnOnce() -> Option<String>) -> Option<Held> {
    let frames: Vec<u64> = figures
        .obj()
        .iter()
        .filter_map(|(_, row)| u64::try_from(row.get("stack_B")?.int()).ok())
        .collect();
    match frames.iter().max() {
        Some(&bytes) => Some((bytes, "ptxas")),
        None => ptx_text().map(|text| (ptx_depots(&text), "ptx")),
    }
}

/// `n` with a comma between groups of three digits.
fn grouped(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// The refusal of a unit `name` that holds `bytes` per thread, measured by `source`, past `limit`.
pub(crate) fn refusal(name: &str, (bytes, source): Held, limit: u64) -> String {
    format!(
        "kernel '{name}': {} bytes of local memory per thread ({source}), past the device limit of {}; the launch would fail. \
         Hold fewer live factorisations or matrices per lane, build with linalg_form='team' so a team of threads works the matrix \
         in scratch, or use a smaller n",
        grouped(bytes),
        grouped(limit)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rcore::json::J;

    fn frames(rows: &[(&str, i64)]) -> J {
        J::Obj(
            rows.iter()
                .map(|(k, b)| {
                    (
                        (*k).to_string(),
                        J::Obj(vec![("stack_B".into(), J::Num(*b as f64))]),
                    )
                })
                .collect(),
        )
    }

    #[test]
    fn a_cubin_holds_its_largest_frame_as_the_compilers_number() {
        let got = held(&frames(&[("a", 4096), ("b", 128)]), || {
            panic!("the text is not asked")
        });
        assert_eq!(got, Some((4096, "ptxas")));
    }

    #[test]
    fn an_entry_runs_alone_and_its_device_functions_add_to_it() {
        let ptx = ".visible .entry a()\n{\n.local .align 8 .b8 __local_depot0[1000];\n}\n\
                   .visible .entry b()\n{\n.local .align 8 .b8 \t__local_depot1[3000];\n}\n\
                   .func f()\n{\n.local .align 8 .b8 __local_depot2[500];\n}\n";
        assert_eq!(
            held(&J::Obj(vec![]), || Some(ptx.into())),
            Some((3500, "ptx"))
        );
    }

    #[test]
    fn an_image_that_states_no_frame_and_has_no_text_holds_nothing_to_judge() {
        assert_eq!(held(&J::Null, || None), None);
    }

    #[test]
    fn the_refusal_names_the_unit_the_number_the_limit_and_the_way_out() {
        let said = refusal("k", (524_296, "ptx"), 524_288);
        assert!(said.starts_with("kernel 'k': 524,296 bytes of local memory per thread (ptx), past the device limit of 524,288;"));
        assert!(said.ends_with("linalg_form='team' so a team of threads works the matrix in scratch, or use a smaller n"));
    }
}
