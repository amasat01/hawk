//! The shipped cards of the two libraries, read through each library's own crate: the device card's fold and wide numbers and the
//! build numbers each target states.
use osprey_cpu_lib::card::HostCards;
use osprey_cuda_lib::card::device_build;
use osprey_lowering::facts::CardSet;
use osprey_lowering::lower::card::TargetCards;
use std::path::Path;

/// The text of the shipped document `name` of the library `library`.
fn shipped(library: &str, name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(library)
        .join("facts")
        .join(name);
    std::fs::read_to_string(path).unwrap()
}

#[test]
fn the_device_card_loads_and_the_two_targets_read_their_own_numbers() {
    let device = device_build(&shipped("osprey_cuda", "cuda_build.json")).unwrap();
    assert_eq!((device.fold.k_max, device.slice_min), (8, 128));
    assert!(device.build().fold_limits.is_some() && !device.build().blend);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../osprey_cpu/facts");
    let docs = CardSet::new(&dir, &[], &["build.json", "arm.json", "x86.json"]);
    assert!(HostCards::of_docs(&docs).unwrap().numbers().blend);
}
