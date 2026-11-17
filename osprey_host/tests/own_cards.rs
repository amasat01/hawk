//! A caller's data documents travel in the request: stating the shipped text of every document changes no answer, and one
//! changed number moves the decision it governs. The answers come from the replay tool over the recorded calls.
#![cfg(feature = "old-contract")]
mod common;
use aether_writers::refuse::{Refusal, RefusalKind};
use osprey_host::facts::{
    answer, device_fusion, host_fusion, stated_from_json, CardSet, Failed, Probe, Stated,
    CARD_NAMES,
};
use osprey_host::lower::card::{device_build_of, host_build_of};
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

// ---- a mistake in a document the caller states is the caller's: a bad request that names the document; a defect in a shipped
// document is the library's own fault ----

/// A reader of the documents of a set, answering as the library does: the reason of its refusal and the kind of it.
type Reader = fn(&CardSet) -> Result<(), Refusal>;

/// The shipped text of the document `name`.
fn shipped_text(name: &str) -> String {
    std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("facts")
            .join(name),
    )
    .expect("a shipped document")
}

/// The shipped text with `old` put as `new`, which must occur in it.
fn changed(name: &str, old: &str, new: &str) -> String {
    let text = shipped_text(name);
    assert!(text.contains(old), "{name} holds {old}");
    text.replacen(old, new, 1)
}

/// The shipped text with one member no reader asks for put first.
fn with_unknown_key(name: &str) -> String {
    format!(
        "{{\"nonsense\": 1,{}",
        &shipped_text(name).trim_start()[1..]
    )
}

/// The set of the shipped directory in which the caller states `name` as `text`.
fn stating(name: &str, text: String) -> CardSet {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("facts");
    CardSet::new(&dir, &[(name.to_string(), text)])
}

/// The answer of the host facts call, for an Arm64 machine under a clang compiler, as the set's documents state it.
const ARM_ANSWER: &str = r#"{"isa":"Arm64","vector_isa":"neon"}"#;

/// A document, its reader and the mistakes to state in it as (what, text).
type ReaderRow = (&'static str, Reader, Vec<(&'static str, String)>);

/// The readers the lowering has of each stated document, and the mistakes each is strict about: (document, reader, mistakes as
/// (what, text)). A reader reads only what its rule needs, so a key no reader asks for is a mistake only where a reader is closed.
fn lowering_readers() -> Vec<ReaderRow> {
    let common = |name: &str, old: &str, new: &str| {
        vec![
            ("a missing key", "{}".to_string()),
            ("a wrong type", changed(name, old, new)),
            ("text that is not JSON", "not json".to_string()),
        ]
    };
    let strict = |name: &str, rest: Vec<(&'static str, String)>| {
        [vec![("an unknown key", with_unknown_key(name))], rest].concat()
    };
    vec![
        (
            "build.json",
            |set| host_build_of(set).map(|_| ()),
            strict(
                "build.json",
                common(
                    "build.json",
                    "\"uniform_run\": 32",
                    "\"uniform_run\": \"x\"",
                ),
            ),
        ),
        (
            "cuda_build.json",
            |set| device_build_of(set).map(|_| ()),
            strict(
                "cuda_build.json",
                common(
                    "cuda_build.json",
                    "\"warp_size\": 32",
                    "\"warp_size\": \"x\"",
                ),
            ),
        ),
        (
            "cuda.json",
            |set| device_fusion(set).map(|_| ()).map_err(Failed::refusal),
            common("cuda.json", "\"regs_base\": 20", "\"regs_base\": \"x\""),
        ),
        (
            "arm.json",
            |set| {
                host_fusion(set, ARM_ANSWER)
                    .map(|_| ())
                    .map_err(Failed::refusal)
            },
            common("arm.json", "\"regs_file\": 32", "\"regs_file\": \"x\""),
        ),
    ]
}

#[test]
fn a_mistake_in_a_document_the_request_states_is_a_bad_request_that_names_the_document() {
    for (name, read, mistakes) in lowering_readers() {
        for (what, text) in mistakes {
            let why = read(&stating(name, text)).expect_err(what);
            assert_eq!(
                why.kind,
                RefusalKind::BadRequest,
                "{name}, {what}: {}",
                why.reason
            );
            assert!(why.reason.contains(name), "{name}, {what}: {}", why.reason);
        }
    }
}

#[test]
fn a_mistake_in_a_document_the_facts_call_states_is_a_bad_request_that_names_the_document() {
    let compiler =
        "/usr/local/cuda/bin/nvcc || Cuda compilation tools, release 12.6, V12.6.77 || 0";
    let probe = Probe {
        isa: "arm64".into(),
        vector_isa: "neon".into(),
        caches: None,
    };
    let mistakes = |name: &str, old: &str, new: &str| {
        vec![
            with_unknown_key(name),
            "{}".to_string(),
            changed(name, old, new),
            "not json".to_string(),
        ]
    };
    let calls = [
        (
            "cuda.json",
            "cuda",
            mistakes(
                "cuda.json",
                "\"default_arch\": \"sm_61\"",
                "\"default_arch\": 61",
            ),
        ),
        (
            "arm.json",
            "host",
            mistakes("arm.json", "\"libm_exact\": true", "\"libm_exact\": 1"),
        ),
    ];
    for (name, target, texts) in calls {
        for text in texts {
            let stated = Stated {
                headers: Some("roots".into()),
                cards: vec![(name.to_string(), text)],
                ..Stated::default()
            };
            let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("facts");
            let failed = answer(&dir, target, compiler, &probe, &stated).expect_err(name);
            let why = failed.refusal();
            assert_eq!(why.kind, RefusalKind::BadRequest, "{name}: {}", why.reason);
            assert!(why.reason.contains(name), "{name}: {}", why.reason);
        }
    }
}

#[test]
fn a_document_name_the_library_does_not_have_is_a_bad_request_that_names_it() {
    let why = stated_from_json(r#"{"cards": {"nonsense.json": "{}"}}"#)
        .err()
        .expect("an unknown document")
        .refusal();
    assert_eq!(why.kind, RefusalKind::BadRequest);
    assert!(why.reason.contains("nonsense.json"), "{}", why.reason);
}

#[test]
fn a_defect_in_a_shipped_document_is_a_fault_of_the_library() {
    let dir = folder("broken_shipped");
    let readers: [Reader; 4] = [
        |set| host_build_of(set).map(|_| ()),
        |set| device_build_of(set).map(|_| ()),
        |set| device_fusion(set).map(|_| ()).map_err(Failed::refusal),
        |set| {
            host_fusion(set, ARM_ANSWER)
                .map(|_| ())
                .map_err(Failed::refusal)
        },
    ];
    // a document with no member and one that is no object at all: a library whose own document has the wrong shape has a defect,
    // it does not answer that the build lacks something
    for broken in ["{}", "[1]"] {
        for name in ["build.json", "cuda_build.json", "cuda.json", "arm.json"] {
            std::fs::write(dir.join(name), broken).expect("a broken document");
        }
        let set = CardSet::new(&dir, &[]);
        for read in readers {
            let why = read(&set).expect_err("a broken shipped document");
            assert_eq!(why.kind, RefusalKind::Fault, "{broken}: {}", why.reason);
        }
    }
}

/// The shipped `cuda_build.json` without any architecture's hardware facts.
fn without_arch_features() -> String {
    let text = shipped_text("cuda_build.json");
    let (start, end) = (
        text.find("\"arch_features\": {")
            .expect("the hardware facts")
            + "\"arch_features\": {".len(),
        text.find("},\n  \"wide_classes\"")
            .expect("the end of the hardware facts"),
    );
    format!("{}{}", &text[..start], &text[end..])
}

/// The lookups of a device card the lowering makes at lowering time, over documents that read without a defect but state what
/// the lookup cannot satisfy: each (what, text of `cuda_build.json`).
fn unsatisfiable_device_cards() -> [(&'static str, String); 2] {
    [
        ("no hardware facts", without_arch_features()),
        (
            "a wide class of a capability with no hardware facts",
            changed(
                "cuda_build.json",
                "\"capability\": 61,",
                "\"capability\": 99,",
            ),
        ),
    ]
}

#[test]
fn a_lookup_the_stated_device_document_cannot_satisfy_is_a_bad_request_naming_the_document() {
    for (what, text) in unsatisfiable_device_cards() {
        let card = device_build_of(&stating("cuda_build.json", text)).expect("the document reads");
        let why = card.wide_class("sm_61").expect_err(what);
        assert_eq!(why.kind, RefusalKind::BadRequest, "{what}: {}", why.reason);
        assert!(
            why.reason.contains("cuda_build.json"),
            "{what}: {}",
            why.reason
        );
    }
}

#[test]
fn the_same_lookup_failing_on_the_shipped_device_document_is_a_fault() {
    for (what, text) in unsatisfiable_device_cards() {
        let dir = folder("shipped_lookup");
        std::fs::write(dir.join("cuda_build.json"), text).expect("a broken document");
        let card = device_build_of(&CardSet::new(&dir, &[])).expect("the document reads");
        let why = card.wide_class("sm_61").expect_err(what);
        assert_eq!(why.kind, RefusalKind::Fault, "{what}: {}", why.reason);
    }
}

/// The host facts call for an Arm64 machine stating `arm.json` as `text`, or the shipped directory's copies but for `arm.json`
/// being `text` in a directory of its own when `shipped` is set.
fn host_answer(text: String, shipped: bool) -> Result<String, Failed> {
    let facts = Path::new(env!("CARGO_MANIFEST_DIR")).join("facts");
    let probe = Probe {
        isa: "arm64".into(),
        vector_isa: "neon".into(),
        caches: None,
    };
    let mut stated = Stated {
        headers: Some("roots".into()),
        ..Stated::default()
    };
    let dir = if shipped {
        let dir = folder("shipped_flags");
        for entry in std::fs::read_dir(&facts).expect("the facts directory") {
            let path = entry.expect("an entry").path();
            std::fs::copy(&path, dir.join(path.file_name().expect("a name"))).expect("a copy");
        }
        std::fs::write(dir.join("arm.json"), text).expect("a broken document");
        dir
    } else {
        stated.cards = vec![("arm.json".to_string(), text)];
        facts
    };
    answer(
        &dir,
        "host",
        "/usr/bin/c++ || Apple clang version 21.0.0 || 0",
        &probe,
        &stated,
    )
}

/// Mistakes in the `host_flags` section of `arm.json`, which the host build reads: the section's own keys and the vector math
/// table.
fn host_flag_mistakes() -> [(&'static str, String); 3] {
    [
        (
            "a flag list that is no list",
            changed(
                "arm.json",
                "\"native\": [\"-mcpu=native\"]",
                "\"native\": 1",
            ),
        ),
        (
            "an unknown key",
            changed(
                "arm.json",
                "\"masked_call\": false",
                "\"masked_call\": false, \"nonsense\": 1",
            ),
        ),
        (
            "a vector math table without the compiler family",
            changed(
                "arm.json",
                "\"other\": {\"native-vector-math\": []}",
                "\"others\": {\"native-vector-math\": []}",
            ),
        ),
    ]
}

#[test]
fn a_mistake_in_the_host_flags_of_a_stated_instruction_set_document_is_a_bad_request_naming_it() {
    for (what, text) in host_flag_mistakes() {
        let why = host_answer(text, false).expect_err(what).refusal();
        assert_eq!(why.kind, RefusalKind::BadRequest, "{what}: {}", why.reason);
        assert!(why.reason.contains("arm.json"), "{what}: {}", why.reason);
    }
}

#[test]
fn the_same_mistake_in_the_host_flags_of_a_shipped_instruction_set_document_is_a_fault() {
    for (what, text) in host_flag_mistakes() {
        let why = host_answer(text, true).expect_err(what).refusal();
        assert_eq!(why.kind, RefusalKind::Fault, "{what}: {}", why.reason);
    }
}

#[test]
fn a_key_no_decision_reads_is_refused_by_name_in_a_stated_device_document() {
    let run_max = "\"run_max\": 32,";
    let linalg = "\"linalg_max_dim\": 32,";
    let stated = [
        (
            "bank_words",
            changed(
                "cuda_build.json",
                run_max,
                "\"run_max\": 32, \"bank_words\": 32,",
            ),
        ),
        (
            "eigh_team_threads",
            changed(
                "cuda_build.json",
                run_max,
                "\"run_max\": 32, \"eigh_team_threads\": 32,",
            ),
        ),
        (
            "linalg_fast_fp64",
            changed(
                "cuda_build.json",
                linalg,
                "\"linalg_max_dim\": 32, \"linalg_fast_fp64\": 8,",
            ),
        ),
        (
            "linalg_warp_entries",
            changed(
                "cuda_build.json",
                linalg,
                "\"linalg_max_dim\": 32, \"linalg_warp_entries\": [],",
            ),
        ),
    ];
    for (key, text) in stated {
        let why = device_build_of(&stating("cuda_build.json", text)).expect_err(key);
        assert_eq!(why.kind, RefusalKind::BadRequest, "{key}: {}", why.reason);
        assert!(why.reason.contains(key), "{key}: {}", why.reason);
        assert!(
            why.reason.contains("cuda_build.json"),
            "{key}: {}",
            why.reason
        );
    }
}
