//! The segmented set of an automatic group, asked for in the request (`auto_segmented_set`), against a recorded kernel whose one
//! automatic group has a static selector (`mix_auto`, three branches). Asked: the fused unit's text is the one it has without the
//! request's member, its record differs only by the branch and classify references, the set's units are added after it (the branch
//! units as segmented records over the same group, and the classify unit) and no other kernel of the call changes. Not asked: no
//! reference and no added unit. Runs where the recordings are found (`OSPREY_CORPUS`).
#![cfg(feature = "old-contract")]
mod common;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The call of the recording that holds the kernel, and the kernel.
const CALL: &str = "2582.32";
/// The recording's folder.
const RECORDING: &str = "k8_record";

/// The files of the replay's emission of the call, with the request's member set or not.
fn emitted(corpus: &Path, asked: bool, into: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("osprey_auto_set_{into}"));
    let _ = std::fs::remove_dir_all(&dir);
    let mut run = Command::new(env!("CARGO_BIN_EXE_kreplay"));
    run.arg(corpus.join(RECORDING))
        .arg("--emit")
        .arg(CALL)
        .arg(&dir);
    if asked {
        run.arg("--auto-set");
    }
    let out = run.output().expect("run kreplay");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    dir
}

/// The text of a file of the emission.
fn read(dir: &Path, name: &str) -> String {
    std::fs::read_to_string(dir.join(name))
        .unwrap_or_else(|_| panic!("{name} is not in {}", dir.display()))
}

/// The names of the files of an emission, sorted.
fn names(dir: &Path) -> Vec<String> {
    let mut found: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    found.sort();
    found
}

#[test]
fn an_asked_set_adds_its_units_and_two_references_and_changes_nothing_else() {
    let corpus = std::env::var("OSPREY_CORPUS").map_or_else(
        |_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus"),
        PathBuf::from,
    );
    if !corpus.join(RECORDING).exists() {
        return common::skipped(Some("OSPREY_CORPUS"), "no k8_record");
    }
    let (plain, asked) = (
        emitted(&corpus, false, "plain"),
        emitted(&corpus, true, "asked"),
    );
    let before = names(&plain);
    let added: Vec<String> = names(&asked)
        .into_iter()
        .filter(|n| !before.contains(n))
        .collect();
    let set = ["cls", "k0", "k1", "k2"].map(|u| format!("{CALL}_mix_auto__{u}_host.cpp"));
    assert_eq!(
        added, set,
        "the units the request adds are the branch units and the classify unit"
    );
    // the fused unit's text is the same; of the files both emissions hold, only the kernel's record differs
    let fused = format!("{CALL}_mix_auto_host.cpp");
    assert_eq!(read(&plain, &fused), read(&asked, &fused));
    // the requests state the setting and nothing else differs between them; of the other files only the kernel's record differs
    let (requests, others): (Vec<&String>, Vec<&String>) =
        before.iter().partition(|n| n.ends_with("_request.json"));
    for name in requests {
        let (plain_text, asked_text) = (read(&plain, name), read(&asked, name));
        assert!(
            !plain_text.contains("auto_segmented_set") && asked_text.contains("auto_segmented_set"),
            "{name}"
        );
        assert_eq!(
            asked_text.replace(
                r#""settings":{"auto_segmented_set":true}"#,
                r#""settings":{}"#
            ),
            plain_text,
            "{name}"
        );
    }
    let differing: Vec<&&String> = others
        .iter()
        .filter(|n| read(&plain, n) != read(&asked, n))
        .collect();
    assert_eq!(differing, [&&format!("{CALL}_mix_auto_host.c")]);
    let (record_plain, record_asked) = (
        read(&plain, &format!("{CALL}_mix_auto_host.c")),
        read(&asked, &format!("{CALL}_mix_auto_host.c")),
    );
    assert!(
        record_plain.contains("EG_DISPATCH_AUTO")
            && record_plain.contains("M(classify, NULL)")
            && record_plain.contains("M(branches, NULL)")
    );
    assert!(
        record_asked.contains("EG_DISPATCH_AUTO")
            && record_asked.contains("M(classify, &u0_d0_classify)")
            && record_asked.contains("M(branches, u0_d0_branches)")
    );
    // the fused record keeps its costs, no sweep and no class, stale or offsets slot; the branch records are segmented over its group
    assert!(record_asked.contains("M(sweep, NULL), M(branch_costs, u0_d0_costs)"));
    assert!(record_asked.contains("M(class_slot, -1), M(stale_slot, -1), M(offsets_slot, -1)"));
    assert!(
        record_asked.contains("EG_DISPATCH_SEGMENTED")
            && record_asked.contains("M(name, \"dispatch0\"), M(k, 3)")
    );
}

/// The unit number a record line of the meta source belongs to (`u<n>_` in its object names), when it has one.
fn unit_of(line: &str) -> Option<usize> {
    let at = line.find(" u")? + 2;
    let digits: String = line[at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    line[at + digits.len()..]
        .starts_with('_')
        .then(|| digits.parse().ok())?
}

/// The line with the unit number removed from its object names (`u3_d0` becomes `u_d0`).
fn without_number(line: &str) -> String {
    let mut out = String::new();
    let mut rest = line;
    while let Some(at) = rest.find('u') {
        let digits = rest[at + 1..]
            .chars()
            .take_while(char::is_ascii_digit)
            .count();
        let word_start = rest[..at]
            .chars()
            .last()
            .is_none_or(|c| !c.is_alphanumeric() && c != '_');
        let named = digits > 0 && rest[at + 1 + digits..].starts_with('_') && word_start;
        out.push_str(&rest[..at + usize::from(!named)]);
        rest = &rest[at + 1 + if named { digits } else { 0 }..];
    }
    out + rest
}

/// The records of the units of the source numbered from `first`, each as its lines without the unit number.
fn records(source: &str, first: usize, count: usize) -> Vec<Vec<String>> {
    (first..first + count)
        .map(|n| {
            source
                .lines()
                .filter(|l| unit_of(l) == Some(n))
                .map(without_number)
                .collect()
        })
        .collect()
}

#[test]
fn the_units_of_an_asked_set_are_those_of_the_segmented_form_of_the_same_kernel() {
    let corpus = std::env::var("OSPREY_CORPUS").map_or_else(
        |_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus"),
        PathBuf::from,
    );
    if !corpus.join(RECORDING).exists() {
        return common::skipped(Some("OSPREY_CORPUS"), "no k8_record");
    }
    let asked = emitted(&corpus, true, "asked_twin");
    // the twin: the same recorded request with its automatic dispatch pinned to the segmented strategy
    let dir = std::env::temp_dir().join("osprey_auto_set_twin");
    let _ = std::fs::remove_dir_all(&dir);
    let out = Command::new(env!("CARGO_BIN_EXE_kreplay"))
        .arg(corpus.join(RECORDING))
        .args(["--auto-as-segmented", "--emit", CALL])
        .arg(&dir)
        .output()
        .expect("run kreplay");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    for unit in ["cls", "k0", "k1", "k2"] {
        let file = format!("{CALL}_mix_auto__{unit}_host.cpp");
        assert_eq!(
            read(&asked, &file),
            read(&dir, &file),
            "{unit}: the emitted text"
        );
    }
    // the records: in the asked answer the fused unit is unit 0 and the set follows it, in the twin the set starts at unit 0, so the
    // only difference allowed is the number in the names of a unit's record objects (`u1_` against `u0_`), removed before comparing
    let record = format!("{CALL}_mix_auto_host.c");
    let (with_fused, alone) = (read(&asked, &record), read(&dir, &record));
    let (set, twin) = (records(&with_fused, 1, 4), records(&alone, 0, 4));
    assert!(set.iter().all(|lines| !lines.is_empty()));
    assert_eq!(set, twin);
}
