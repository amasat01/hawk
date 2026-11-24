//! The defaults an unset ask gets, from the documents the two libraries ship.
use osprey_lowering::facts::CardSet;

/// Oracle: the default level and the default profile the shipped documents state (`default_opt_level`, `default_profile`).
#[test]
fn the_defaults_of_an_unset_ask_are_the_ones_the_documents_state() {
    let docs = CardSet::new(
        &osprey_replay::merged_facts(),
        &[],
        osprey_replay::ALL_CARDS,
    );
    assert_eq!(
        osprey_cpu_lib::host_facts::default_opt_level(&docs).unwrap(),
        "O3"
    );
    assert_eq!(
        osprey_cuda_lib::device_facts::default_opt_level(&docs).unwrap(),
        "O3"
    );
    let gcc = "/usr/bin/g++ || g++ (GCC) 13.2.0 || 0";
    let clang = "/usr/bin/c++ || Apple clang version 21.0.0 || 0";
    let default = |compiler, isa| {
        osprey_cpu_lib::host_facts::default_host_profile(&docs, compiler, isa).unwrap()
    };
    assert_eq!(default(gcc, "arm64"), "native-vector-math");
    assert_eq!(default(clang, "arm64"), "native-vector-math");
    assert_eq!(default(gcc, "x86_64"), "native-vector-math");
    assert_eq!(default(clang, "x86_64"), "native");
    assert!(osprey_cpu_lib::host_facts::default_host_profile(&docs, gcc, "riscv").is_err());
}
