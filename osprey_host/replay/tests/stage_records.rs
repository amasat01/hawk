//! The meta records of recorded split kernels: each stage's sum cells name the planes that compensate their rows, every
//! sample-count slot is the unsigned 32-bit the runtime passes, and the classify unit of a segmented group with a static selector
//! holds a dispatch record of its own. Each kernel is replayed from its recorded call and the C source of its meta is read.
//! Runs where the recordings are found (`OSPREY_CORPUS`, else the folder beside the clones); otherwise it skips with a line.
#![cfg(feature = "old-contract")]
mod common;
use std::path::PathBuf;
use std::process::Command;

/// What one replayed recorded call wrote: the folder with every file, the call's id and the C source of the meta.
struct Emitted {
    /// The folder the replay wrote into: the meta sources and the unit texts, named by the call's id.
    dir: PathBuf,
    /// The recorded call's id.
    id: String,
    /// The C source of the meta of the replayed kernel.
    source: String,
}

/// Replays the recorded call of `recording` whose line holds `test` and every text of `units`, and reads the C source of the meta
/// of its kernel `kernel`; none when the recording is not there.
fn emit(recording: &str, test: &str, units: &[&str], kernel: &str) -> Option<Emitted> {
    let corpus = std::env::var("OSPREY_CORPUS").map_or_else(
        |_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus"),
        PathBuf::from,
    );
    let folder = corpus.join(recording);
    if !folder.exists() {
        common::skipped(
            Some("OSPREY_CORPUS"),
            &format!("no recording at {}", folder.display()),
        );
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
    let dir = std::env::temp_dir().join(format!("osprey_stage_records_{kernel}"));
    std::fs::create_dir_all(&dir).unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_kreplay"))
        .arg(&folder)
        .args(["--emit", &id, dir.to_str().unwrap()])
        .output()
        .expect("run kreplay");
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    // the record source is named after the call's lead target
    let source = ["host", "cuda"]
        .iter()
        .map(|target| dir.join(format!("{id}_{kernel}_{target}.c")))
        .find(|file| file.exists())
        .expect("the meta source");
    let source = std::fs::read_to_string(source).unwrap();
    Some(Emitted { dir, id, source })
}

/// The C source of the meta of the kernel `kernel`, as `emit` reads it.
fn meta_source(recording: &str, test: &str, units: &[&str], kernel: &str) -> Option<String> {
    emit(recording, test, units, kernel).map(|emitted| emitted.source)
}

/// The include root of the runtime's door headers, when the environment names it.
fn door_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("OSPREY_EAGLE_SRC").map(PathBuf::from);
    if dir.is_none() {
        common::skipped(Some("OSPREY_EAGLE_SRC"), "OSPREY_EAGLE_SRC is not set");
    }
    dir
}

/// The reader of a compiled record: a C program that includes the meta source and prints, for each unit, what the record states.
const READER: &str = r#"#include <stdio.h>
#include "SOURCE"
static const char* shown(const char* name) { return name ? name : "-"; }
int main(void) {
    const struct eg_library_meta* library = eg_unit_library_meta();
    for (uint32_t u = 0; u < library->n_units; ++u) {
        const struct eg_unit_meta* unit = library->units[u];
        const struct eg_unit_finish* finish = unit->finish;
        printf("unit %s persist %s range %s\n", unit->name,
               finish ? shown(finish->device_persist_function) : "-",
               finish ? shown(finish->device_range_function) : "-");
    }
    return 0;
}
"#;

/// Compiles the reader against the meta source `source` and the headers under `door`, runs it and returns what it printed.
fn read_record(door: &std::path::Path, source: &std::path::Path) -> String {
    let program = source.with_extension("reader.c");
    std::fs::write(&program, READER.replace("SOURCE", source.to_str().unwrap())).unwrap();
    let binary = source.with_extension("reader");
    let build = Command::new("cc")
        .args(["-std=c11", "-I", door.to_str().unwrap()])
        .args([program.to_str().unwrap(), "-o", binary.to_str().unwrap()])
        .output()
        .expect("cc starts");
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&binary).output().expect("the reader runs");
    assert!(run.status.success(), "the reader failed");
    String::from_utf8(run.stdout).unwrap()
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

#[test]
fn a_plane_axis_sink_states_its_payload_as_its_slots_shape() {
    let sums = "test_block_sums_match_the_exact_sum_and_the_per_entry_path[n8-team-host]";
    if let Some(s) = meta_source("k6_record", sums, &["grad"], "grad") {
        assert!(object(&s, "u0_s0").contains("M(rank, 1)"));
        assert_eq!(numbers(&s, "u0_s0_shape"), [8]);
    }
    if let Some(s) = meta_source("k6_record", sums, &["gram"], "gram") {
        assert!(object(&s, "u0_s0").contains("M(rank, 2)"));
        assert_eq!(numbers(&s, "u0_s0_shape"), [8, 8]);
    }
    let named = "test_the_sidecar_names_the_sorted_inputs_only_when_declared";
    if let Some(s) = meta_source("k6_record", named, &["by_segment"], "by_segment") {
        let slot = object(&s, "u0_s0");
        assert!(
            slot.contains("M(rank, 0)") && slot.contains("M(shape, NULL)"),
            "{slot}"
        );
    }
}

#[test]
fn each_branch_unit_of_a_segmented_kernel_lists_itself_alone_as_its_stage_with_scratch_of_its_own()
{
    // the branches are alternatives over runs of samples, never stages launched one after the other: a branch whose wide value
    // is placed in scratch records one stage (itself) and its own scratch plane, as a kernel that is not segmented does
    let test = "test_wide_placement.py::test_the_branch_units_of_a_segmented_kernel_are_placed_for_a_device[auto-fill_mask_seg]";
    let Some(s) = meta_source("k8_record", test, &["fill_mask_seg__k0"], "fill_mask_seg") else {
        return;
    };
    for (unit, name) in [
        "fill_mask_seg__k0",
        "fill_mask_seg__k1",
        "fill_mask_seg__k2",
    ]
    .iter()
    .enumerate()
    {
        let meta = object(&s, &format!("u{unit}_meta"));
        assert!(meta.contains(&format!("M(name, \"{name}\")")), "{meta}");
        let stages = object(&s, &format!("u{unit}_stages[]"));
        assert!(stages.contains(&format!("{{&u{unit}_st0}}")), "{stages}");
        let staged = object(&s, &format!("u{unit}_st0_unit"));
        assert!(staged.contains(&format!("\"{name}\"")), "{staged}");
        let planes = object(&s, &format!("u{unit}_team_planes[]"));
        assert!(planes.contains(&format!("{{&u{unit}_tp0}}")), "{planes}");
    }
}

#[test]
fn a_device_unit_states_the_two_entry_names_its_text_holds_and_a_host_unit_states_none() {
    // the oracle is the unit's own text: the symbols of its persistent and range entries
    let test = "test_dispatch_mix_and_guard.py::test_fused_loop_counts_mix[host]";
    let Some(emitted) = emit("k7_record", test, &["walker"], "walker") else {
        return;
    };
    let Some(door) = door_dir() else { return };
    let (dir, id) = (&emitted.dir, &emitted.id);
    let text = std::fs::read_to_string(dir.join(format!("{id}_walker_cuda.cu"))).unwrap();
    let device = read_record(&door, &dir.join(format!("{id}_walker_cuda.c")));
    assert!(text.contains("void walker_persist(") && text.contains("void walker_range("));
    assert!(
        device.contains("unit walker persist walker_persist range walker_range\n"),
        "{device}"
    );
    let host = read_record(&door, &dir.join(format!("{id}_walker_host.c")));
    assert!(host.contains("unit walker persist - range -\n"), "{host}");
}

const BROADCAST_REVERSE: &str =
    "test_broadcast_matrix.py::test_recorded_reverse_with_a_broadcast_matrix";

#[test]
fn a_gathered_matrix_the_request_states_no_pattern_for_is_kept_along_every_lane_axis() {
    if let Some(s) = meta_source("k8_record", BROADCAST_REVERSE, &["mscal_vjp"], "mscal_vjp") {
        let matrix = s
            .lines()
            .find(|l| l.contains("M(name, \"M\")") || l.contains("\"M\""))
            .unwrap_or_else(|| panic!("no slot M in {s}"));
        assert!(matrix.contains("EG_SLOT_GATHER"), "{matrix}");
        let keep: Vec<i64> = s
            .lines()
            .filter(|l| l.contains("_gather[] = {"))
            .flat_map(|l| {
                l.split(['{', '}'])
                    .nth(1)
                    .unwrap()
                    .split(',')
                    .map(|n| n.trim().parse().unwrap())
                    .collect::<Vec<i64>>()
            })
            .collect();
        assert_eq!(keep, [1, 1, 1], "the three lane axes of the recording");
    }
}
