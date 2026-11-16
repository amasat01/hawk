//! The derived sets of recorded calls (a primal, its taping twin and its reverse unit) answered as one table: the emitted derived
//! records against the records the runtime's owner loaded for the same set, the meta compiled as C and as C++17, and the texts and the
//! meta linked into one library. Runs only where the recordings (`OSPREY_CORPUS`, else the folder beside the clones) and the runtime's
//! tree (`EAGLE_DOOR_DIR`, which holds `eagle/door/test/derived/*.c`) are found; the link leg also needs `EAGLE_ROOT`.
#![cfg(feature = "old-contract")]
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The members that tell two records apart, as `member=value` words; the pointers are left out.
const MEMBERS: [&str; 7] = [
    "slot",
    "of_slot",
    "source",
    "role",
    "tape_on_slot",
    "fused_steps_slot",
    "n_links",
];

/// The derived, taped and link records of a meta source, each as its member words, counted.
fn records(file: &Path) -> BTreeMap<String, usize> {
    let text = std::fs::read_to_string(file).expect("a meta source");
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("static const struct ") else {
            continue;
        };
        let kind = rest.split(' ').next().unwrap_or("");
        if !["eg_unit_derived", "eg_unit_taped", "eg_unit_link"].contains(&kind) {
            continue;
        }
        let mut words = vec![kind.to_string()];
        for member in [
            "slot",
            "row0",
            "rows",
            "of_slot",
            "source",
            "reduced",
            "role",
            "of_unit",
            "tape_slot",
            "n_taped",
            "tape_rows",
            "steps_slot",
            "tape_on_slot",
            "time0_slot",
            "tau_slot",
            "fused_steps_slot",
            "n_links",
            "n_refined",
        ] {
            if let Some(at) = line.find(&format!("M({member}, ")) {
                let value = line[at + member.len() + 4..]
                    .split(')')
                    .next()
                    .unwrap_or("");
                words.push(format!("{member}={value}"));
            }
        }
        *out.entry(words.join(" ")).or_insert(0) += 1;
    }
    out
}

/// What the two sources hold that the other does not, as the words of `MEMBERS` only, sorted: (oracle only, ours only).
fn differences(oracle: &Path, ours: &Path) -> (Vec<String>, Vec<String>) {
    let (a, b) = (records(oracle), records(ours));
    let only = |x: &BTreeMap<String, usize>, y: &BTreeMap<String, usize>| {
        let mut out = vec![];
        for (record, n) in x {
            for _ in 0..n.saturating_sub(*y.get(record).unwrap_or(&0)) {
                let words: Vec<&str> = record
                    .split(' ')
                    .filter(|w| MEMBERS.iter().any(|m| w.starts_with(&format!("{m}="))))
                    .collect();
                out.push(format!(
                    "{} {}",
                    record.split(' ').next().unwrap_or(""),
                    words.join(" ")
                ));
            }
        }
        out.sort();
        out
    };
    (only(&a, &b), only(&b, &a))
}

/// A scratch folder of this test.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("osprey_derived_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Replays `call` of the recording `file` as a derived set into a folder; returns the folder and the `--derived` lines.
fn replay(file: &str, call: &str, name: &str) -> Option<(PathBuf, String, PathBuf)> {
    let corpus = std::env::var("OSPREY_CORPUS").map_or_else(
        |_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../corpus"),
        PathBuf::from,
    );
    let calls = corpus.join("k7e_record").join(file);
    let door = PathBuf::from(std::env::var_os("EAGLE_DOOR_DIR")?);
    if !calls.is_file() {
        eprintln!("skipped: no recording at {}", calls.display());
        return None;
    }
    let dir = scratch(name);
    let run = Command::new(env!("CARGO_BIN_EXE_kreplay"))
        .arg("--derived")
        .arg(&calls)
        .arg(call)
        .arg(&dir)
        .output()
        .expect("kreplay starts");
    let said = String::from_utf8_lossy(&run.stdout).into_owned();
    assert!(
        said.contains("found 1 / built 1 / refused 0"),
        "{said}{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        said.contains("different 0"),
        "a text of the set differs from its alone-lowered text: {said}"
    );
    Some((dir, said, door))
}

/// The meta compiles as C and as C++17.
fn compiles(source: &Path, door: &Path, dir: &Path) {
    let includes = door.to_str().unwrap();
    let as_c = Command::new("cc")
        .args(["-std=c11", "-I", includes, "-c"])
        .arg(source)
        .arg("-o")
        .arg(dir.join("m_c.o"))
        .status()
        .unwrap();
    let as_cc = Command::new("c++")
        .args(["-std=c++17", "-I", includes, "-c", "-x", "c++"])
        .arg(source)
        .arg("-o")
        .arg(dir.join("m_cc.o"))
        .status()
        .unwrap();
    assert!(
        as_c.success() && as_cc.success(),
        "the meta compiles as C and as C++17"
    );
}

#[test]
fn the_rr_pend_set_is_one_table_that_compiles_and_links_with_one_self_check() {
    let Some((dir, _, door)) = replay("k83646.jsonl", "83646.138", "rr") else {
        return;
    };
    let ours = dir.join("83646.138_rr_pend_set.c");
    let loaded = door.join("eagle/door/test/derived/rr_pend_meta.c");
    let (oracle, mine) = differences(&loaded, &ours);
    // the recorded twin is the K-step one: it has the extra slot `fused_steps` at 7, which moves the slots after it by one
    assert_eq!(
        oracle,
        [
            "eg_unit_derived role=EG_DERIVED_TAPE tape_on_slot=10 fused_steps_slot=-1 n_links=8",
            "eg_unit_link slot=7 of_slot=5 source=EG_SRC_PRIMAL",
            "eg_unit_link slot=8 of_slot=6 source=EG_SRC_PRIMAL",
            "eg_unit_link slot=9 of_slot=7 source=EG_SRC_PRIMAL"
        ]
    );
    assert_eq!(mine, ["eg_unit_derived role=EG_DERIVED_TAPE_FUSED tape_on_slot=11 fused_steps_slot=7 n_links=8", "eg_unit_link slot=10 of_slot=7 source=EG_SRC_PRIMAL", "eg_unit_link slot=8 of_slot=5 source=EG_SRC_PRIMAL", "eg_unit_link slot=9 of_slot=6 source=EG_SRC_PRIMAL"]);
    compiles(&ours, &door, &dir);
    let Some(eagle) = std::env::var_os("EAGLE_ROOT").map(PathBuf::from) else {
        return;
    };
    let aether = Path::new(env!("CARGO_MANIFEST_DIR")).join("../aether");
    let mut objects = vec![dir.join("m_cc.o")];
    for unit in ["rr_pend", "rr_pend_tape", "rr_pend_step_vjp"] {
        let object = dir.join(format!("{unit}.o"));
        let built = Command::new("c++")
            .args(["-std=c++23", "-fPIC", "-DAETHER_CPP_MODE", "-I"])
            .arg(&aether)
            .arg("-I")
            .arg(&eagle)
            .args(["-c", "-x", "c++"])
            .arg(dir.join(format!("83646.138_{unit}_set.cpp")))
            .arg("-o")
            .arg(&object)
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "{}",
            String::from_utf8_lossy(&built.stderr)
        );
        objects.push(object);
    }
    let library = dir.join("lib.so");
    let linked = Command::new("c++")
        .arg("-shared")
        .args(&objects)
        .arg("-o")
        .arg(&library)
        .status()
        .unwrap();
    assert!(linked.success(), "the objects link into one library");
    let symbols = String::from_utf8_lossy(
        &Command::new("nm")
            .arg("-g")
            .arg(&library)
            .output()
            .unwrap()
            .stdout,
    )
    .into_owned();
    for name in ["eagle_abi_tag", "eagle_layout_sizes"] {
        assert_eq!(
            symbols
                .lines()
                .filter(|l| l.ends_with(name) || l.ends_with(&format!("_{name}")))
                .count(),
            1,
            "{name}: {symbols}"
        );
    }
}

#[test]
fn the_time_step_set_equals_the_loaded_records_but_for_one_twin_link() {
    let Some((dir, _, door)) = replay("k83638.jsonl", "83638.59", "timed") else {
        return;
    };
    let ours = dir.join("83638.59__timed_set.c");
    let (oracle, mine) = differences(&door.join("eagle/door/test/derived/timed_meta.c"), &ours);
    // the loaded twin does not link its finished-count lookup; the converter links every slot the primal has, as the rr_pend set does
    assert_eq!(
        oracle,
        ["eg_unit_derived role=EG_DERIVED_TAPE tape_on_slot=11 fused_steps_slot=-1 n_links=8"]
    );
    assert_eq!(
        mine,
        [
            "eg_unit_derived role=EG_DERIVED_TAPE tape_on_slot=11 fused_steps_slot=-1 n_links=9",
            "eg_unit_link slot=8 of_slot=4 source=EG_SRC_PRIMAL"
        ]
    );
    compiles(&ours, &door, &dir);
}
