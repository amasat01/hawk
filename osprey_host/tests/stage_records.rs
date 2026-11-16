//! The meta records of recorded split kernels: each stage's sum cells name the planes that compensate their rows, every
//! sample-count slot is the unsigned 32-bit the runtime passes, and the classify unit of a segmented group with a static selector
//! holds a dispatch record of its own. Each kernel is replayed from its recorded call and the C source of its meta is read.
//! Runs where the recordings are found (`OSPREY_CORPUS`, else the folder beside the clones); otherwise it skips with a line.
#![cfg(feature = "old-contract")]
use std::path::PathBuf;
use std::process::Command;

/// The C source of the meta of the kernel `kernel` the recorded call (the one in `recording` whose line holds `test` and every
/// text of `units`) lowers to, or none when the recording is not there.
fn meta_source(recording: &str, test: &str, units: &[&str], kernel: &str) -> Option<String> {
    let corpus = std::env::var("OSPREY_CORPUS").map_or_else(
        |_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../corpus"),
        PathBuf::from,
    );
    let folder = corpus.join(recording);
    if !folder.exists() {
        println!("skipped: no recording at {}", folder.display());
        return None;
    }
    let mut id = None;
    for file in std::fs::read_dir(&folder)
        .unwrap()
        .map(|e| e.unwrap().path())
    {
        let named = file.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !(named.starts_with('k') && named.ends_with(".jsonl")) {
            continue;
        }
        for line in std::fs::read_to_string(&file).unwrap().lines() {
            let held = |text: &str| line.contains(&format!("\"unit\":\"{text}\""));
            if line.contains(test) && units.iter().all(|u| held(u)) {
                let at = line.find("\"call\":\"").expect("a call id") + 8;
                id = Some(line[at..].split('"').next().unwrap().to_string());
            }
        }
    }
    let id = id.unwrap_or_else(|| panic!("no recorded call of {kernel} for {test}"));
    let out = std::env::temp_dir().join(format!("osprey_stage_records_{kernel}"));
    std::fs::create_dir_all(&out).unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_kreplay"))
        .arg(&folder)
        .args(["--emit", &id, out.to_str().unwrap()])
        .output()
        .expect("run kreplay");
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    Some(
        std::fs::read_to_string(out.join(format!("{id}_{kernel}_host.c")))
            .expect("the meta source"),
    )
}

/// The numbers of the array object `name` of the source, in order.
fn numbers(source: &str, name: &str) -> Vec<i64> {
    let at = source
        .find(&format!("{name}[] = {{"))
        .unwrap_or_else(|| panic!("no array {name}"));
    let list = source[at..].split(['{', '}']).nth(1).unwrap();
    list.split(',').map(|n| n.trim().parse().unwrap()).collect()
}

/// The line of the object `name` of the source.
fn object<'a>(source: &'a str, name: &str) -> &'a str {
    source
        .lines()
        .find(|l| l.contains(&format!(" {name} = {{")))
        .unwrap_or_else(|| panic!("no object {name}"))
}

/// Checks that every sample-count slot of the source is unsigned 32-bit and the named ones are there.
fn nsamples(source: &str, named: &[&str]) {
    let all: Vec<&str> = source
        .lines()
        .filter(|l| l.contains("EG_ROLE_NSAMPLES"))
        .collect();
    assert!(
        all.iter().all(|l| l.contains("M(dtype, EG_DTYPE_UINT32)")),
        "{all:?}"
    );
    for name in named {
        assert!(
            object(source, name).contains("M(role, EG_ROLE_NSAMPLES), M(dtype, EG_DTYPE_UINT32)")
        );
    }
    assert_eq!(
        all.len(),
        named.len(),
        "the sample-count slots of the unit set"
    );
}

const STAGES: &str =
    "test_split_stages.py::test_two_stages_through_hawk_run_match_numpy[float64-1]";

#[test]
fn a_stage_names_the_planes_that_compensate_its_sum_cells_and_the_sample_count_is_unsigned_32_bit()
{
    if let Some(s) = meta_source("k7_record", STAGES, &["share", "share_2"], "share") {
        assert_eq!(numbers(&s, "u1_cell0_comp"), [1]);
        nsamples(&s, &[]);
    }
    if let Some(s) = meta_source("k7_record", STAGES, &["standard_3"], "standard") {
        assert_eq!(numbers(&s, "u1_cell0_comp"), [1]);
        assert_eq!(numbers(&s, "u2_cell0_comp"), [1]);
        assert_eq!(numbers(&s, "u3_cell0_comp"), [2]);
        nsamples(&s, &["u0_s6", "u1_s3", "u2_s4"]);
    }
    if let Some(s) = meta_source("k7_record", STAGES, &["centred_2"], "centred") {
        assert_eq!(numbers(&s, "u1_cell0_comp"), [1, 3, 5]);
        nsamples(&s, &["u0_s2", "u1_s7"]);
    }
}

#[test]
fn the_classify_unit_of_a_static_segmented_group_holds_the_record_of_its_own() {
    let test = "test_dispatch_mix_and_guard.py::test_auto_segmented_runs_unfused";
    let Some(s) = meta_source("k7_record", test, &["walker__cls"], "walker") else {
        return;
    };
    // the units are the three branches and then the classify unit: the branches list all of them and the classify unit lists none
    let record = object(&s, "u3_d0");
    let class = record
        .split("M(class_slot, ")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    assert!(class.parse::<i64>().unwrap() >= 0, "{record}");
    for held in [
        "M(strategy, EG_DISPATCH_SEGMENTED)",
        "M(mix_slot, -1)",
        "M(stale_slot, -1)",
        "M(offsets_slot, -1)",
        "M(classify, NULL)",
        "M(k, 3)",
        "M(branches, NULL)",
        "M(sweep, NULL)",
    ] {
        assert!(record.contains(held), "{held}: {record}");
    }
    let branch = object(&s, "u0_d0");
    assert!(
        branch.contains("M(k, 3)")
            && !branch.contains("M(classify, NULL)")
            && branch.contains("M(sweep, NULL)")
    );
}
