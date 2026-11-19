//! Figures read from a tool's own output or from the image it made, by the name a request gives the grammar: `ptxas_info`, the
//! per-kernel register and frame lines of an `-Xptxas -v` report, and `image_resources`, the same figures read off the image
//! itself (the register-count and frame-size records of an ELF image's `.nv.info` section; no tool and no device needed).
use crate::refusal::{Done, Refusal};
use crate::wire;
use rcore::json::J;

/// The figures a kernel's lines state, in the order they are reported.
const FIELDS: [&str; 4] = ["regs", "stack_B", "spill_store_B", "spill_load_B"];

/// The figures a tool run states under the grammar `name`, read from its `stderr` or from the `image` it wrote; no name states none.
///
/// # Errors
/// Refuses a grammar name that is not `ptxas_info` or `image_resources`.
pub fn parse(name: Option<&str>, stderr: &str, image: &[u8]) -> Done<J> {
    match name {
        None => Ok(wire::obj(vec![])),
        Some("ptxas_info") => Ok(ptxas_info(stderr)),
        Some("image_resources") => Ok(image_resources(image)),
        Some(other) => Err(Refusal::bad(format!(
            "figures grammar {other:?} is not one of: ptxas_info, image_resources"
        ))),
    }
}

/// The record type of a `.nv.info` entry that states a kernel's register count.
const ATTR_REGCOUNT: u8 = 0x2F;
/// The record type of a `.nv.info` entry that states a kernel's frame size.
const ATTR_FRAME_SIZE: u8 = 0x11;

/// `n` bytes of `image` from `at`, little-endian as a number; `None` past the end.
fn le(image: &[u8], at: usize, n: usize) -> Option<u64> {
    let bytes = image.get(at..at.checked_add(n)?)?;
    Some(
        bytes
            .iter()
            .rev()
            .fold(0, |acc, b| (acc << 8) | u64::from(*b)),
    )
}

/// `le` as an index.
fn le_usize(image: &[u8], at: usize, n: usize) -> Option<usize> {
    usize::try_from(le(image, at, n)?).ok()
}

/// The NUL-terminated text at `at` in `image`.
fn c_text(image: &[u8], at: usize) -> Option<String> {
    let rest = image.get(at..)?;
    let end = rest.iter().position(|b| *b == 0)?;
    Some(String::from_utf8_lossy(&rest[..end]).into_owned())
}

/// A section header of an ELF64 image: name offset, type, file offset, size and link.
struct Section {
    /// Offset of the name in the section-name table.
    name: usize,
    /// Section type.
    kind: u64,
    /// Offset in the file.
    offset: usize,
    /// Size in bytes.
    size: usize,
    /// Index of the linked section (a symbol table's string table).
    link: usize,
}

/// The section headers of an ELF64 little-endian image.
fn sections(image: &[u8]) -> Option<Vec<Section>> {
    let (table, entry, count) = (
        le_usize(image, 0x28, 8)?,
        le_usize(image, 0x3A, 2)?,
        le_usize(image, 0x3C, 2)?,
    );
    (0..count)
        .map(|i| {
            let at = table + i * entry;
            Some(Section {
                name: le_usize(image, at, 4)?,
                kind: le(image, at + 4, 4)?,
                offset: le_usize(image, at + 0x18, 8)?,
                size: le_usize(image, at + 0x20, 8)?,
                link: le_usize(image, at + 0x28, 4)?,
            })
        })
        .collect()
}

/// `{kernel: {regs, stack_B, spill_store_B: null, spill_load_B: null}}` off an ELF image: the register-count and frame-size records
/// of its `.nv.info` section, each keyed by the symbol index of the kernel. The spill figures are not in the image (null). An
/// image that is not ELF, or malformed, states nothing.
#[must_use]
pub fn image_resources(image: &[u8]) -> J {
    resources(image).map_or_else(|| wire::obj(vec![]), J::Obj)
}

/// The kernels of `image` with their figures, in the order their register records appear.
fn resources(image: &[u8]) -> Option<Vec<(String, J)>> {
    if !image.starts_with(b"\x7fELF") {
        return None;
    }
    let secs = sections(image)?;
    let names = secs.get(le_usize(image, 0x3E, 2)?)?.offset;
    let info = secs
        .iter()
        .find(|s| c_text(image, names + s.name).as_deref() == Some(".nv.info"))?;
    let symtab = secs.iter().find(|s| s.kind == 2)?;
    let strings = secs.get(symtab.link)?.offset;
    let (mut regs, mut frames) = (vec![], vec![]);
    let mut at = info.offset;
    while at + 4 <= info.offset + info.size {
        let (format, attr, n) = (
            *image.get(at)?,
            *image.get(at + 1)?,
            le_usize(image, at + 2, 2)?,
        );
        if format == 4 && n == 8 && (attr == ATTR_REGCOUNT || attr == ATTR_FRAME_SIZE) {
            let record = (le(image, at + 4, 4)?, le(image, at + 8, 4)?);
            if attr == ATTR_REGCOUNT {
                &mut regs
            } else {
                &mut frames
            }
            .push(record);
        }
        at += 4 + if format == 4 { n } else { 0 };
    }
    regs.iter()
        .map(|(index, count)| {
            let symbol = symtab.offset + usize::try_from(*index).ok()? * 24;
            let name = c_text(image, strings + le_usize(image, symbol, 4)?)?;
            let frame = frames.iter().find(|f| f.0 == *index).map_or(0, |f| f.1);
            let figures = wire::obj(vec![
                ("regs", wire::n(*count)),
                ("stack_B", wire::n(frame)),
                ("spill_store_B", J::Null),
                ("spill_load_B", J::Null),
            ]);
            Some((name, figures))
        })
        .collect()
}

/// The number that stands before `unit` in `line`: `20 registers` gives 20.
fn before(line: &str, unit: &str) -> Option<u64> {
    let head = &line[..line.find(unit)?];
    head.trim_end()
        .rsplit(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

/// `{kernel: {regs, stack_B, spill_store_B, spill_load_B}}` from the `ptxas info` lines of a report; a kernel compiled for several
/// architectures keeps the largest of each figure.
fn ptxas_info(report: &str) -> J {
    let mut kernels: Vec<(String, [u64; 4])> = vec![];
    let mut current: Option<usize> = None;
    for line in report.lines() {
        if let Some(rest) = line.split("Compiling entry function '").nth(1) {
            let name = rest.split('\'').next().unwrap_or("").to_string();
            current = Some(kernels.iter().position(|k| k.0 == name).unwrap_or_else(|| {
                kernels.push((name, [0; 4]));
                kernels.len() - 1
            }));
            continue;
        }
        let Some(at) = current else { continue };
        let seen = [
            before(line, " registers").filter(|_| line.contains("Used ")),
            before(line, " bytes stack frame"),
            before(line, " bytes spill stores"),
            before(line, " bytes spill loads"),
        ];
        for (slot, value) in kernels[at].1.iter_mut().zip(seen) {
            *slot = (*slot).max(value.unwrap_or(0));
        }
    }
    J::Obj(
        kernels
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    wire::obj(
                        FIELDS
                            .iter()
                            .zip(v)
                            .map(|(f, n)| (*f, wire::n(n)))
                            .collect(),
                    ),
                )
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_and_frames_are_read_and_the_largest_over_archs_is_kept() {
        let report = "ptxas info    : Compiling entry function 'k' for 'sm_61'\n\
            ptxas info    : Function properties for k\n\
            \x20   8 bytes stack frame, 4 bytes spill stores, 2 bytes spill loads\n\
            ptxas info    : Used 32 registers, 360 bytes cmem[0]\n\
            ptxas info    : Compiling entry function 'k' for 'sm_75'\n\
            \x20   0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n\
            ptxas info    : Used 40 registers, 360 bytes cmem[0]\n";
        assert_eq!(
            wire::render(&ptxas_info(report)),
            "{\"k\":{\"regs\":40,\"stack_B\":8,\"spill_store_B\":4,\"spill_load_B\":2}}"
        );
    }

    /// Oracle: the two entry names written by hand in the report; the figures of each are the ones under its own name.
    #[test]
    fn each_entry_of_a_report_has_its_own_figures_the_fast_entries_included() {
        let report = "ptxas info    : Compiling entry function 'k' for 'sm_61'\n\
            \x20   0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n\
            ptxas info    : Used 24 registers, 360 bytes cmem[0]\n\
            ptxas info    : Compiling entry function 'k_fast' for 'sm_61'\n\
            \x20   16 bytes stack frame, 8 bytes spill stores, 8 bytes spill loads\n\
            ptxas info    : Used 48 registers, 360 bytes cmem[0]\n";
        assert_eq!(
            wire::render(&ptxas_info(report)),
            "{\"k\":{\"regs\":24,\"stack_B\":0,\"spill_store_B\":0,\"spill_load_B\":0},\
             \"k_fast\":{\"regs\":48,\"stack_B\":16,\"spill_store_B\":8,\"spill_load_B\":8}}"
        );
    }

    #[test]
    fn an_image_that_is_not_elf_or_is_cut_short_states_nothing() {
        assert_eq!(wire::render(&image_resources(b"// ptx text")), "{}");
        assert_eq!(wire::render(&image_resources(b"\x7fELF\x02\x01\x01")), "{}");
        assert_eq!(wire::render(&image_resources(&[])), "{}");
    }

    #[test]
    fn a_report_without_kernels_states_nothing_and_an_unknown_grammar_is_refused() {
        assert_eq!(wire::render(&ptxas_info("nothing here")), "{}");
        assert!(parse(Some("nope"), "", &[]).is_err());
    }
}
