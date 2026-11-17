//! The facts oracle: the machine record the converter derives from every recorded host unit of the two newest recordings equals
//! the facts answer for the machine that recording states (its instruction set, cache sizes and host profile), so the rules of
//! the cards are what is tested. Runs only where the recordings are found (`OSPREY_CORPUS`, else the folder beside the clones).
#![cfg(feature = "old-contract")]
mod common;
use std::path::PathBuf;
use std::process::Command;

/// The replay tool's report on the recording `name`, or none when the recording is not there.
fn report(name: &str) -> Option<String> {
    let corpus = std::env::var("OSPREY_CORPUS").map_or_else(
        |_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../corpus"),
        PathBuf::from,
    );
    let texts = corpus.join(name).join("texts.jsonl");
    if !texts.exists() {
        common::skipped(
            Some("OSPREY_CORPUS"),
            &format!("no recording at {}", texts.display()),
        );
        return None;
    }
    let out = Command::new(env!("CARGO_BIN_EXE_treplay"))
        .arg(texts)
        .args(["--target", "host"])
        .output()
        .expect("run treplay");
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[test]
fn every_recorded_machine_equals_the_answer() {
    for name in ["t5_record", "e5_record"] {
        let Some(text) = report(name) else { continue };
        let line = text
            .lines()
            .find(|l| l.starts_with("facts oracle:"))
            .expect(&text);
        let numbers: Vec<usize> = line
            .split(|c: char| !c.is_ascii_digit())
            .filter_map(|n| n.parse().ok())
            .collect();
        // "equals the facts answer on E of N converted units, D differ"
        assert!(
            numbers[0] > 800 && numbers[0] == numbers[1] && numbers[2] == 0,
            "{name}: {text}"
        );
    }
}
