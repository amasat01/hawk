//! The library's table of valid links against the runtime's own: the rows of `EG_LINK_RULES(X)` in its header and the library's
//! constant must be the same set. Runs only where the runtime's tree is found (`EAGLE_DOOR_DIR`); without it, it says so and skips.
#![cfg(feature = "old-contract")]
use osprey_host::lower::links::LINK_RULES;
use osprey_host::lower::unit_meta::{DerivedRole, LinkForm, LinkWhat};
use std::collections::BTreeSet;

/// The rows of the header's table, as the three macro names of each.
fn header_rows(header: &str) -> BTreeSet<[String; 3]> {
    let start = header
        .find("#define EG_LINK_RULES(X)")
        .expect("the header holds the link rules");
    let mut rows = BTreeSet::new();
    for line in header[start..].lines() {
        let Some(at) = line.find("X(EG_") else {
            // a line of the macro that is no row: the macro ends at the first line without a continuation
            if line.trim_end().ends_with('\\') {
                continue;
            }
            break;
        };
        let names: Vec<String> = line[at + 2..]
            .split(')')
            .next()
            .unwrap_or("")
            .split(',')
            .map(|n| n.trim().to_string())
            .collect();
        assert_eq!(names.len(), 3, "{line}");
        rows.insert([names[0].clone(), names[1].clone(), names[2].clone()]);
        if !line.trim_end().ends_with('\\') {
            break;
        }
    }
    rows
}

/// The library's rows as the same three macro names.
fn library_rows(table: &[(DerivedRole, LinkWhat, LinkForm)]) -> BTreeSet<[String; 3]> {
    table
        .iter()
        .map(|(role, what, form)| {
            [
                role.c_name().to_string(),
                what.c_name().to_string(),
                form.c_name().to_string(),
            ]
        })
        .collect()
}

/// The rows each side holds that the other does not, as lines.
fn differing(ours: &BTreeSet<[String; 3]>, theirs: &BTreeSet<[String; 3]>) -> Vec<String> {
    let mut out: Vec<String> = ours
        .difference(theirs)
        .map(|r| format!("library only: {}", r.join(", ")))
        .collect();
    out.extend(
        theirs
            .difference(ours)
            .map(|r| format!("header only: {}", r.join(", "))),
    );
    out
}

#[test]
fn the_library_table_is_the_headers_table() {
    let Some(door) = std::env::var_os("EAGLE_DOOR_DIR") else {
        eprintln!("skipped: EAGLE_DOOR_DIR is not set");
        return;
    };
    let header =
        std::fs::read_to_string(std::path::Path::new(&door).join("eagle/door/eg_unit_meta.h"))
            .expect("the runtime's header");
    let theirs = header_rows(&header);
    assert!(theirs.len() > 40, "the header's rows were not found");
    let ours = library_rows(LINK_RULES);
    assert_eq!(
        differing(&ours, &theirs),
        Vec::<String>::new(),
        "the library's link rows and the header's differ"
    );
    // with one row of the library's table removed, the comparison names it
    let fewer = library_rows(&LINK_RULES[1..]);
    assert_eq!(differing(&fewer, &theirs).len(), 1);
}
