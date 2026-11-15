//! Runs treplay on a small copy of recorded host units (with their recorded analyses) and checks the numbers.
#![cfg(feature = "old-contract")]
use std::process::Command;

fn run(args: &[&str]) -> String {
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/units_small.jsonl");
    let out = Command::new(env!("CARGO_BIN_EXE_treplay"))
        .arg(data)
        .args(args)
        .output()
        .expect("run treplay");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn small_units_walks_bodies_and_texts() {
    let text = run(&["--target", "host"]);
    // The invariants only: how many units are written grows with the writers, so those counts are not pinned here.
    for want in [
        "units 8 (excluded 0, oracle-raised 0)",
        "walk-loaded 8 / 8",
        "walk-digest-equal 8 / 8",
        "make-refused 0",
        "body written but different 0",
        "(written but different 0)",
    ] {
        assert!(text.contains(want), "missing {want:?} in:\n{text}");
    }
}

#[test]
fn ladder_reports_one_line_per_unit() {
    let text = run(&["--ladder"]);
    assert_eq!(
        text.lines()
            .filter(|l| l.contains("loaded  digest ="))
            .count(),
        7,
        "{text}"
    );
    assert!(
        text.contains(
            "axpb           nodes    6  loaded  digest =  arg_spec =  access =  body =  text ="
        ),
        "{text}"
    );
}
