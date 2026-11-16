//! The event count of a kernel against the recorded kernels: for every kernel of the recorded set that has a finish, the library's
//! `n_events` must equal the number of comparisons its recorded stop rule is made of (read from the recorded nodes by the replay
//! tool's own, separate reading of the `or` chain). Runs where the recordings are found (`OSPREY_CORPUS`, else the folder beside the
//! clones); without them it says so and skips.
#![cfg(feature = "old-contract")]
use std::path::PathBuf;
use std::process::Command;

#[test]
fn every_recorded_kernel_with_a_finish_has_the_recorded_number_of_events() {
    let corpus = std::env::var("OSPREY_CORPUS").map_or_else(
        |_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../corpus"),
        PathBuf::from,
    );
    let calls = corpus.join("k7e_record");
    if !calls.is_dir() {
        eprintln!("skipped: no recording at {}", calls.display());
        return;
    }
    let run = Command::new(env!("CARGO_BIN_EXE_kreplay"))
        .arg("--events")
        .arg(&calls)
        .output()
        .expect("kreplay starts");
    let said = String::from_utf8_lossy(&run.stdout).into_owned();
    println!("{said}");
    let line = said
        .lines()
        .find(|l| l.contains("kernels with a finish compared"))
        .expect("the summary line");
    let number_after = |word: &str| -> usize {
        let at = line.find(word).expect(word) + word.len();
        line[at..]
            .trim_start()
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .and_then(|n| n.parse().ok())
            .expect(word)
    };
    let (compared, equal) = (number_after("compared"), number_after("equal"));
    assert!(compared > 0, "{said}");
    assert_eq!(
        compared, equal,
        "kernels differ from the recorded reading: {said}"
    );
    assert!(said.contains("different 0"), "{said}");
    // the oscillator with an event has exactly one
    let ev = said
        .lines()
        .find(|l| l.contains("ev_osc library counts"))
        .expect("the ev_osc line");
    assert!(
        ev.contains("[1") && !ev.contains(", 0") && !ev.contains(", 2"),
        "{ev}"
    );
}
