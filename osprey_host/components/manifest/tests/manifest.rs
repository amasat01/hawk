//! The rules of a manifest, of a target id and of the roster, through the Rust functions and through the C face.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a panic is the failure
use osprey_manifest::{abi, manifest, roster, target};
use rcore::json::{parse, J};
use std::ffi::{c_char, CString};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

const ROW: &str = r#"{"request":"host","target":"Host","about":"a","device":false,"entry":"host","format":"host"}"#;

/// A fresh folder holding `text` as its manifest.
fn folder_with(text: &str) -> PathBuf {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "osprey_manifest_{}_{}",
        std::process::id(),
        COUNT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("osprey.json"), text).unwrap();
    dir
}

/// The manifest text of an OSPREY `name` with the target rows `rows`.
fn manifest_of(name: &str, rows: &[&str]) -> String {
    format!(
        r#"{{"name":"{name}","version":"0","door":11,"targets":[{}]}}"#,
        rows.join(",")
    )
}

/// The refusal sentence for the manifest `text`.
fn refusal(text: &str) -> String {
    manifest::read(&folder_with(text)).unwrap_err()
}

/// A row stating the request `id`.
fn row(id: &str) -> String {
    ROW.replace("\"host\",\"target\"", &format!("\"{id}\",\"target\""))
}

/// The JSON list of the folders.
fn folders(list: &[&Path]) -> String {
    format!(
        "[{}]",
        list.iter()
            .map(|f| format!("{:?}", f.display().to_string()))
            .collect::<Vec<_>>()
            .join(",")
    )
}

#[test]
fn a_target_id_is_one_spelling_and_a_bad_one_is_refused_naming_it() {
    assert_eq!(target::canonical(" Cu-Da ").unwrap(), "cuda");
    assert_eq!(target::canonical("c u_d a").unwrap(), "cuda");
    for bad in ["Toy-X!", "", "9lives", "9x", "_-_", "x.y", "é"] {
        let why = target::canonical(bad).unwrap_err();
        assert!(
            why.starts_with(&format!(
                "the target id {} is not a target id",
                osprey_manifest::json::quote(bad)
            )),
            "{why}"
        );
    }
}

#[test]
fn a_manifest_that_is_unreadable_is_refused_naming_the_file() {
    let dir = std::env::temp_dir().join("osprey_manifest_absent");
    assert!(manifest::read(&dir)
        .unwrap_err()
        .contains("osprey_manifest_absent/osprey.json: no readable osprey.json"));
    assert!(refusal("{not json").contains("osprey.json: no readable osprey.json"));
}

#[test]
fn a_manifest_missing_a_member_or_stating_an_unknown_one_is_refused_naming_it() {
    for drop in ["name", "version", "door", "targets"] {
        let mut object = parse(&manifest_of("m", &[ROW])).unwrap();
        if let J::Obj(pairs) = &mut object {
            pairs.retain(|(k, _)| k != drop);
        }
        let why = refusal(&osprey_manifest::json::emit(&object));
        assert!(why.contains(&format!("the manifest lacks {drop}")), "{why}");
    }
    let extra = manifest_of("m", &[ROW]).replace("\"name\"", "\"overrides\":[],\"name\"");
    assert!(refusal(&extra).contains(
        "unknown manifest key 'overrides' (the keys are ['name', 'version', 'door', 'targets'])"
    ));
    let wings = manifest_of("m", &[ROW]).replace("\"name\"", "\"wings\":{},\"name\"");
    assert!(refusal(&wings).contains("a Python wing is refused"));
}

#[test]
fn a_target_row_missing_a_member_or_with_a_bad_or_repeated_id_is_refused() {
    let lacking = manifest_of("m", &[&ROW.replace("\"about\":\"a\",", "")]);
    assert!(refusal(&lacking).contains("target 'host' lacks about"));
    assert!(refusal(&manifest_of("m", &[&row("9x")]))
        .contains("the target request: the target id '9x' is not a target id"));
    let twice = refusal(&manifest_of("m", &[&row("cuda"), &row("Cu-Da")]));
    assert!(
        twice
            .contains("osprey.json: the manifest states the target id 'cuda' in more than one row"),
        "{twice}"
    );
}

#[test]
fn a_checked_manifest_keeps_its_rows_with_the_request_in_its_canonical_spelling() {
    let dir = folder_with(&manifest_of("m", &[&row("Cu-Da")]));
    let checked = manifest::read(&dir).unwrap();
    assert_eq!(checked.get("name").and_then(J::str), Some("m"));
    assert_eq!(
        checked.get("targets").unwrap().arr()[0]
            .get("request")
            .and_then(J::str),
        Some("cuda")
    );
    assert_eq!(checked.get("door").unwrap().int(), 11);
}

#[test]
fn a_roster_isolates_a_refused_folder_and_refuses_two_ospreys_serving_one_id() {
    let good = folder_with(&manifest_of("a", &[&row("host")]));
    let broken = folder_with("{not json");
    let out = parse(&roster::of_folders(&folders(&[&good, &broken])).unwrap()).unwrap();
    assert_eq!(out.get("ospreys").unwrap().arr().len(), 1);
    assert!(out
        .get("refused")
        .unwrap()
        .get(&broken.display().to_string())
        .is_some());
    let other = folder_with(&manifest_of("b", &[&row("Ho-st")]));
    let why = roster::of_folders(&folders(&[&good, &other])).unwrap_err();
    assert_eq!(why, "target 'host' is served by both 'a' and 'b'; keep one");
    let same_name = folder_with(&manifest_of("a", &[&row("host")]));
    assert!(roster::of_folders(&folders(&[&good, &same_name])).is_ok());
}

#[test]
fn a_target_in_any_spelling_resolves_and_an_unserved_one_lists_the_served_and_the_refused() {
    let good = folder_with(&manifest_of("cpu", &[&row("host")]));
    let broken = folder_with("{not json");
    let made = roster::of_folders(&folders(&[&good, &broken])).unwrap();
    assert_eq!(
        roster::serving(&made, " Ho-st ").unwrap(),
        r#"{"id":"host","osprey":"cpu"}"#
    );
    let why = roster::serving(&made, "tpu").unwrap_err();
    let searched = format!("(searched: {}, {})", good.display(), broken.display());
    let head = format!("target 'tpu' is served by no OSPREY found; the served targets are ('host',) {searched}; refused: ");
    assert!(why.starts_with(&head), "{why}");
    assert!(why.contains("osprey.json: no readable osprey.json"));
    let none = roster::of_folders("[]").unwrap();
    assert!(roster::serving(&none, "tpu")
        .unwrap_err()
        .ends_with("are none (searched: no folder)"));
    assert!(roster::serving(&made, "9x")
        .unwrap_err()
        .contains("is not a target id"));
}

#[test]
fn the_c_face_writes_the_answer_or_the_refusal_and_asks_for_more_room_when_it_is_short() {
    let name = CString::new("Cu-Da").unwrap();
    let mut buf = [0 as c_char; 16];
    let mut len = 0usize;
    unsafe {
        assert_eq!(
            abi::omf_target_id(name.as_ptr(), buf.as_mut_ptr(), buf.len(), &raw mut len),
            0
        );
        assert_eq!(len, 4);
        let bad = CString::new("9x").unwrap();
        assert_eq!(
            abi::omf_target_id(bad.as_ptr(), buf.as_mut_ptr(), 4, &raw mut len),
            3
        );
        assert!(len > 4);
        let mut big = vec![0 as c_char; len + 1];
        assert_eq!(
            abi::omf_target_id(bad.as_ptr(), big.as_mut_ptr(), big.len(), &raw mut len),
            1
        );
    }
}
