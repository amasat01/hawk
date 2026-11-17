//! The dispatch records of the answer against the recorded kernels: every group's branch count equals the one the old record states
//! (and, for a segmented group, the number of its branch units, listed in order); a segmented kernel's records follow the rule of its
//! selector (a static one lists the classify unit and no sweep, a dynamic one the sweep unit, the stale slot and the sweep's own count
//! slot); and each sweep unit's text is the text of the same kernel requested with its dispatch as a switch and its guard an active set.
//! Runs where the recordings are found (`OSPREY_CORPUS`, else the folder beside the clones); each recording that is not there is skipped
//! with a line.
#![cfg(feature = "old-contract")]
mod common;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

/// The recordings that hold dispatch groups.
const RECORDINGS: [&str; 6] = [
    "k6_record",
    "k6e_record",
    "k7_record",
    "k7e_record",
    "k8_record",
    "k8e_record",
];

/// The numbers of a report line, in order.
fn numbers(line: &str) -> Vec<usize> {
    line.split(|c: char| !c.is_ascii_digit())
        .filter_map(|n| n.parse().ok())
        .collect()
}

/// The report lines that start with `prefix` (after their indentation), as the numbers each holds.
fn rows(report: &str, prefix: &str) -> Vec<Vec<usize>> {
    report
        .lines()
        .filter_map(|l| l.trim_start().strip_prefix(prefix))
        .map(numbers)
        .collect()
}

#[test]
fn every_group_states_the_recorded_branch_count_and_every_sweep_is_the_switch_form() {
    let corpus = std::env::var("OSPREY_CORPUS").map_or_else(
        |_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../corpus"),
        PathBuf::from,
    );
    // k compared / equal / different, branch lists compared / equal / different, static groups, dynamic groups, wrong records,
    // equal sweep units, groups of fewer than two branches
    let mut sum = [0usize; 12];
    let mut strategies: BTreeMap<String, usize> = BTreeMap::new();
    let mut access = [0usize; 8];
    let mut maskless = [0usize; 3];
    let mut loops: Vec<String> = vec![];
    let mut pairs: BTreeMap<String, usize> = BTreeMap::new();
    if !RECORDINGS.iter().any(|name| corpus.join(name).exists()) {
        common::skipped(
            Some("OSPREY_CORPUS"),
            &format!("no recording under {}", corpus.display()),
        );
        return;
    }
    for name in RECORDINGS {
        let folder = corpus.join(name);
        if !folder.exists() {
            common::skipped(
                Some("OSPREY_CORPUS"),
                &format!("no recording at {}", folder.display()),
            );
            continue;
        }
        let out = Command::new(env!("CARGO_BIN_EXE_kreplay"))
            .arg(&folder)
            .output()
            .expect("run kreplay");
        let report = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(out.status.success(), "{name}: {report}");
        let mut add = |at: usize, rows: Vec<Vec<usize>>, from: usize| {
            for row in rows {
                for (k, n) in row.iter().skip(from).take(3).enumerate() {
                    sum[at + k] += n;
                }
            }
        };
        add(0, rows(&report, "dispatch.k (branches of every group):"), 0);
        add(
            3,
            rows(
                &report,
                "dispatch.branches (units of a segmented group, in order):",
            ),
            0,
        );
        for row in rows(&report, "segmented groups with a record by the rule:") {
            // static, dynamic, wrong, sweep units equal; the numbers inside the words ("k branch ...", "2 ...") are skipped by position
            let tail: Vec<usize> = row.into_iter().rev().take(5).collect();
            sum[6] += tail[4];
            sum[7] += tail[3];
            sum[8] += tail[2];
            sum[9] += tail[1];
            sum[11] += tail[0];
        }
        for line in report
            .lines()
            .filter_map(|l| l.trim_start().strip_prefix("maskless dynamic kernels:"))
        {
            let counts = numbers(line.split("sweep loop:").next().unwrap());
            maskless[0] += counts[0];
            maskless[1] += counts[1];
            maskless[2] += counts[2];
            if let Some(header) = line.split("sweep loop:").nth(1).filter(|h| h.trim() != "-") {
                loops.push(header.trim().to_string());
            }
        }
        for line in report
            .lines()
            .filter_map(|l| l.trim_start().strip_prefix("access of segmented units,"))
        {
            let (counts, rest) = line
                .split_once("(branch/sweep) of dynamic kernels:")
                .unwrap();
            for (at, n) in numbers(counts).into_iter().enumerate() {
                access[at] += n;
            }
            for pair in rest.split(',').filter_map(|p| p.trim().split_once(' ')) {
                *pairs.entry(pair.0.to_string()).or_default() +=
                    pair.1.parse::<usize>().unwrap_or(0);
            }
        }
        for line in report
            .lines()
            .filter_map(|l| l.trim_start().strip_prefix("dispatch groups by strategy:"))
        {
            let counted = line.split(';').next().unwrap_or("").split(',');
            for (strategy, n) in counted.filter_map(|c| c.trim().split_once(' ')) {
                *strategies.entry(strategy.to_string()).or_default() +=
                    n.parse::<usize>().unwrap_or(0);
            }
        }
        sum[10] += rows(&report, "dispatch groups by strategy:")
            .iter()
            .map(|r| r.last().copied().unwrap_or(0))
            .sum::<usize>();
    }
    println!(
        "branch counts: compared {} equal {} different {}; branch units in order: compared {} equal {} different {}",
        sum[0], sum[1], sum[2], sum[3], sum[4], sum[5]
    );
    println!("dispatch groups by strategy: {strategies:?}");
    println!(
        "segmented groups: static {} dynamic {} wrong {}; sweep units equal to the switch form: {}; groups of fewer than 2 branches: {}",
        sum[6], sum[7], sum[8], sum[9], sum[10]
    );
    println!(
        "access: branch units against the recorded: compared {} equal {} different {}; classify units: compared {} equal {} different {}; sweep access equal to the switch form's: {} of {}; (branch/sweep) pairs: {pairs:?}",
        access[0], access[1], access[2], access[3], access[4], access[5], access[6], access[7]
    );
    println!(
        "maskless dynamic kernels: {} answered with a sweep that holds no mask: {} caller's set refused: {}; loop: {:?}",
        maskless[0], maskless[1], maskless[2], loops.first()
    );
    assert_eq!(
        maskless[0], maskless[1],
        "a maskless dynamic kernel's sweep"
    );
    assert_eq!(
        maskless[0], maskless[2],
        "a caller's set on a maskless kernel"
    );
    assert_eq!(
        access[2] + access[5],
        0,
        "a unit's access differs from the recorded unit's"
    );
    assert_eq!(
        access[6], access[7],
        "a sweep unit's access differs from the switch form's"
    );
    assert!(access[0] > 0, "no segmented unit compared");
    assert_eq!(
        sum[2] + sum[5] + sum[8],
        0,
        "a record differs from the old one"
    );
    assert_eq!(sum[0], sum[1]);
    assert_eq!(sum[3], sum[4]);
    assert_eq!(
        sum[7], sum[9],
        "a sweep unit is not the switch form over an active set"
    );
    assert_eq!(sum[10], 0, "a group of fewer than two branches");
    assert_eq!(
        sum[7], sum[11],
        "a dynamic group in a chain is refused NotWritten"
    );
}
