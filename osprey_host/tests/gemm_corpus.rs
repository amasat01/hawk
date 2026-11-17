//! The contraction kernels recorded from the old lowering (`g_record` of the corpus: the dense contraction of declared sizes, the
//! staged one counted by the launch and a matvec, each forward and, for the contractions, reversed, in both widths) are written by
//! the library as the old lowering wrote them, and the forward units of the two contractions state a matrix product beside their
//! kernel.
#![cfg(feature = "old-contract")]
mod common;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn the_recorded_contractions_are_written_as_recorded_and_the_forward_ones_state_their_product() {
    let corpus = std::env::var("OSPREY_CORPUS").map_or_else(
        |_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../corpus"),
        PathBuf::from,
    );
    let recorded = corpus.join("g_record");
    if !recorded.exists() {
        common::skipped(Some("OSPREY_CORPUS"), "no recording of contractions");
        return;
    }
    let done = Command::new(env!("CARGO_BIN_EXE_kreplay"))
        .arg(&recorded)
        .output()
        .expect("kreplay starts");
    assert!(
        done.status.success(),
        "{}",
        String::from_utf8_lossy(&done.stderr)
    );
    let report = String::from_utf8_lossy(&done.stdout).into_owned();
    let line = |start: &str| {
        report
            .lines()
            .find(|l| l.contains(start))
            .unwrap_or_else(|| panic!("{start}: {report}"))
    };
    // 14 builds: three forward kernels and two reversed ones, each in two widths, the reversed ones with their primal
    assert!(line("host units of plain")
        .contains("14: equal 14, text different 0, digest different 0, meta different 0"));
    // the dense contraction and the staged one state a product in each width; the reversed units, the matvec (one operand)
    // and the primals built beside the reversed units are the other units: 4 forward units of two kernels, and the 4 primals
    // rebuilt for the reverse recordings, are the units that state one
    assert!(
        line("stating a matrix product").ends_with(": 8"),
        "{report}"
    );
}
