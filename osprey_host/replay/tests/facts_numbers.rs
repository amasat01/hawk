//! The machine numbers the two libraries state as facts, read from their own facts files: the team dimension limit, the cell row
//! width and the scatter chunk cap. Oracle: the numbers of the files as shipped (4096; a row of one warp on the device and a power
//! of two on the host; at least one chunk).
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a panic is the failure
use rcore::json::{self, J};
use std::path::Path;

/// The facts file `name` of the library folder `library`, parsed.
fn facts(library: &str, name: &str) -> J {
    let file = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(library)
        .join("facts")
        .join(name);
    json::parse(&std::fs::read_to_string(file).unwrap()).unwrap()
}

/// The integer member `key` of `facts`, looking inside `group` when given.
fn number(facts: &J, group: Option<&str>, key: &str) -> i64 {
    let holder = group.map_or(Some(facts), |g| facts.get(g)).unwrap();
    holder.get(key).unwrap().int()
}

#[test]
fn the_team_dimension_limit_of_both_libraries_is_4096() {
    for (library, name) in [
        ("osprey_cpu", "build.json"),
        ("osprey_cuda", "cuda_build.json"),
    ] {
        assert_eq!(
            number(&facts(library, name), None, "linalg_team_max_dim"),
            4096,
            "{library}"
        );
    }
}

#[test]
fn the_device_cell_row_is_one_warp_and_the_host_row_a_power_of_two() {
    let device = facts("osprey_cuda", "cuda_build.json");
    assert_eq!(
        number(&device, None, "cells_row"),
        number(&device, None, "warp_size")
    );
    let row = number(&facts("osprey_cpu", "build.json"), None, "cells_row");
    assert!(row >= 1 && row & (row - 1) == 0, "{row}");
}

#[test]
fn a_device_block_holds_whole_warps_and_at_least_one() {
    let device = facts("osprey_cuda", "cuda_build.json");
    let (block, warp) = (
        number(&device, None, "max_threads_per_block"),
        number(&device, None, "warp_size"),
    );
    assert!(block >= warp && block % warp == 0, "{block} of {warp}");
}

#[test]
fn the_scatter_chunk_cap_of_both_libraries_is_at_least_one() {
    assert!(
        number(
            &facts("osprey_cuda", "cuda_build.json"),
            Some("scatter"),
            "max_chunks"
        ) >= 1
    );
    assert!(
        number(
            &facts("osprey_cpu", "build.json"),
            None,
            "scatter_max_chunks"
        ) >= 1
    );
}
