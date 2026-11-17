//! A caller's data documents travel in the request: stating the shipped text of every document changes no answer, and one
//! changed number moves the decision it governs. The answers come from the replay tool over the recorded calls.
#![cfg(feature = "old-contract")]
mod common;
use osprey_host::facts::CARD_NAMES;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The recorded calls the answers are made from.
const CALLS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/meta_calls.jsonl");

/// The cards file of the replay tool: each document of the closed set as the text of the file shipped, with `edit` applied to
/// each of `edits` (document name, old text, new text) applied to its document.
fn cards_file(folder: &Path, edits: &[(&str, &str, &str)]) -> PathBuf {
    let shipped = Path::new(env!("CARGO_MANIFEST_DIR")).join("facts");
    let documents: Vec<String> = CARD_NAMES
        .iter()
        .map(|name| {
            let mut text = std::fs::read_to_string(shipped.join(name)).expect("a shipped document");
            for (_, old, new) in edits.iter().filter(|edit| edit.0 == *name) {
                assert!(text.contains(old), "{name} holds {old}");
                text = text.replacen(old, new, 1);
            }
            format!("{name:?}:{text:?}")
        })
        .collect();
    let file = folder.join("cards.json");
    std::fs::write(&file, format!("{{{}}}", documents.join(","))).expect("the cards file");
    file
}

/// The files the replay tool writes for the recorded calls under the cards of `file` (none: the shipped files), by name.
fn answers(folder: &Path, file: Option<&Path>) -> Vec<(String, String)> {
    let out = folder.join(if file.is_some() { "stated" } else { "shipped" });
    let _ = std::fs::remove_dir_all(&out);
    let mut command = Command::new(env!("CARGO_BIN_EXE_kreplay"));
    command.args(["--emit-meta", CALLS, out.to_str().expect("a path")]);
    if let Some(file) = file {
        command.arg("--cards").arg(file);
    }
    let done = command.output().expect("kreplay starts");
    assert!(
        done.status.success(),
        "{}",
        String::from_utf8_lossy(&done.stderr)
    );
    let mut found: Vec<(String, String)> = std::fs::read_dir(&out)
        .expect("the answers")
        .map(|entry| {
            let path = entry.expect("an entry").path();
            let name = path
                .file_name()
                .expect("a name")
                .to_string_lossy()
                .into_owned();
            (name, std::fs::read_to_string(path).expect("an answer"))
        })
        .collect();
    found.sort();
    found
}

/// A folder of this test's own.
fn folder(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("osprey_own_cards_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a folder");
    dir
}

/// The bit the unit record states when the unit was lowered under stated cards, as the unit text writes it.
const CALLER_CARDS: &str = "M(flags, EG_UNIT_CALLER_CARDS)";

// The answers compared are the units' texts and records; the facts digest member that stated cards add is no part of them, and the
// one bit a unit record sets for stated cards is put back to the 0 of a record without cards before the comparison.
#[test]
fn the_shipped_text_of_every_document_gives_the_same_unit_answers_as_the_files_but_for_the_caller_cards_bit(
) {
    let dir = folder("same");
    let file = cards_file(&dir, &[]);
    let (plain, stated) = (answers(&dir, None), answers(&dir, Some(&file)));
    assert!(plain.len() >= 7, "{} answers", plain.len());
    assert!(plain
        .iter()
        .all(|(_, text)| !text.contains("EG_UNIT_CALLER_CARDS")));
    assert!(stated.iter().all(|(_, text)| text.contains(CALLER_CARDS)));
    let put_back: Vec<(String, String)> = stated
        .into_iter()
        .map(|(name, text)| (name, text.replace(CALLER_CARDS, "M(flags, 0u)")))
        .collect();
    assert_eq!(plain, put_back);
}

/// The `threads` member of the team record of the unit text `81812.35_total_host.c`, under the cards `edits` state.
fn team_threads(name: &str, edits: &[(&str, &str, &str)]) -> i64 {
    let dir = folder(name);
    let file = cards_file(&dir, edits);
    let text = answers(&dir, Some(&file))
        .into_iter()
        .find(|(file, _)| file == "81812.35_total_host.c")
        .expect("the unit text")
        .1;
    let record = &text[text.find("u0_st1_team = {").expect("the team record")..];
    let at = record.find("M(threads, ").expect("its threads") + "M(threads, ".len();
    record[at..]
        .split(')')
        .next()
        .and_then(|n| n.parse().ok())
        .expect("a number")
}

/// Oracle: the team size is the card's smaller team for a matrix of order up to `team_threads` and its larger team above
/// (`threads_for`, raptor wings team.rs). The recorded matrix is larger than 64 and no larger than 128, so the shipped 128 gives
/// the smaller team, a smaller bound of 64 gives the larger team, and the larger team's own size is then what the record states.
#[test]
fn the_team_size_follows_the_smaller_and_larger_teams_of_the_build_card_by_the_order_of_the_matrix()
{
    let small = (
        "build.json",
        "\"team_threads\": 128",
        "\"team_threads\": 64",
    );
    let large = (
        "build.json",
        "\"team_threads_large\": 256",
        "\"team_threads_large\": 512",
    );
    assert_eq!(team_threads("shipped", &[]), 128);
    assert_eq!(team_threads("large_only", &[large]), 128);
    assert_eq!(team_threads("small", &[small]), 256);
    assert_eq!(team_threads("both", &[small, large]), 512);
}

/// The text of an instruction set document with the register file of every fusion row set to one register.
fn with_one_register(text: &str) -> String {
    let key = "\"regs_file\": ";
    let mut out = String::new();
    let mut rest = text;
    while let Some(at) = rest.find(key) {
        let (head, tail) = rest.split_at(at + key.len());
        out.push_str(head);
        out.push('1');
        rest = tail.trim_start_matches(|c: char| c.is_ascii_digit());
    }
    out.push_str(rest);
    out
}

/// The units of a chain the replay tool reports for the recorded kernels of `folder`, as (written, equal to the recording).
fn chain_units(folder: &Path, cards: Option<&Path>) -> (usize, usize) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_kreplay"));
    command.arg(folder);
    if let Some(file) = cards {
        command.arg("--cards").arg(file);
    }
    let done = command.output().expect("kreplay starts");
    assert!(
        done.status.success(),
        "{}",
        String::from_utf8_lossy(&done.stderr)
    );
    let report = String::from_utf8_lossy(&done.stdout).into_owned();
    let line = report
        .lines()
        .find(|line| line.contains("units of a chain:"))
        .expect("the chain line");
    let numbers: Vec<usize> = line
        .split(|c: char| !c.is_ascii_digit())
        .filter_map(|n| n.parse().ok())
        .collect();
    (numbers[0], numbers[1])
}

/// Oracle: the recording was made under the shipped fusion rows, so every chain unit equals it; a fusion row of the host's
/// instruction set document that leaves one register for a lane cuts more of the chains, so fewer chain units are written.
#[test]
fn a_fusion_row_stated_in_the_instruction_set_document_moves_the_cuts_of_the_recorded_chains() {
    let corpus = std::env::var("OSPREY_CORPUS").map_or_else(
        |_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../corpus"),
        PathBuf::from,
    );
    let recorded = corpus.join("k6_record");
    if !recorded.exists() {
        common::skipped(Some("OSPREY_CORPUS"), "no recording of chains");
        return;
    }
    let dir = folder("fusion");
    for name in ["k92094.jsonl", "t92094.jsonl"] {
        std::fs::copy(recorded.join(name), dir.join(name)).expect("the recorded chains");
    }
    let shipped = Path::new(env!("CARGO_MANIFEST_DIR")).join("facts");
    let documents: Vec<String> = ["arm.json", "x86.json"]
        .iter()
        .map(|name| {
            let text = std::fs::read_to_string(shipped.join(name)).expect("a shipped document");
            format!("{name:?}:{:?}", with_one_register(&text))
        })
        .collect();
    let file = dir.join("cards.json");
    std::fs::write(&file, format!("{{{}}}", documents.join(","))).expect("the cards file");
    let (plain, equal) = chain_units(&dir, None);
    assert!(
        plain > 0 && plain == equal,
        "{plain} chain units, {equal} equal"
    );
    let (stated, _) = chain_units(&dir, Some(&file));
    assert!(
        stated < plain,
        "{stated} chain units under one register, {plain} under the shipped rows"
    );
}

/// The member `name` of the steps table record a unit text states (the text from `<name> = {` to its `;`), or none.
fn steps_record(text: &str, name: &str) -> Option<String> {
    let at = text.find(&format!("{name} = {{"))?;
    text[at..].split(';').next().map(String::from)
}

/// Oracle: the shipped host build card states the `lanes` regime and three rows, which the record of the one unit of the recorded
/// calls that finishes samples holds; a card that states `per_resident` and one row moves both members of the record.
#[test]
fn the_steps_table_of_the_unit_record_is_the_one_the_build_card_states() {
    let finishing = "83645.104_ev_untimed_host.c";
    let record = |name: &str, edits: &[(&str, &str, &str)]| {
        let dir = folder(name);
        let file = cards_file(&dir, edits);
        let text = answers(&dir, Some(&file))
            .into_iter()
            .find(|(file, _)| file == finishing)
            .expect("the finishing unit")
            .1;
        steps_record(&text, "host_steps").expect("the host steps record")
    };
    let kept = record("steps_kept", &[]);
    assert!(
        kept.contains("EG_REGIME_LANES") && kept.contains("M(n_rows, 3)"),
        "{kept}"
    );
    let row = ",\n      {\"width\": 4, \"k\": 4, \"lo\": 1, \"hi\": 64},\n      {\"width\": 2, \"k\": 8, \"lo\": 1, \"hi\": 65536}";
    let moved = record(
        "steps_moved",
        &[
            (
                "build.json",
                "\"regime\": \"lanes\"",
                "\"regime\": \"per_resident\"",
            ),
            ("build.json", row, ""),
        ],
    );
    assert!(
        moved.contains("EG_REGIME_PER_RESIDENT") && moved.contains("M(n_rows, 1)"),
        "{moved}"
    );
}
