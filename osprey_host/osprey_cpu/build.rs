//! The library's identity and abi number, and the header trees it embeds, fixed at build time.
include!("../components/lowering/build_support.rs");

fn main() {
    write_build_facts(&PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()));
    // the folders of the headers the library's compiles include, named by the environment of this build
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("trees.rs");
    toolkit::embed::write_table(
        &[
            ("aether", "OSPREY_CPU_AETHER_HEADERS"),
            ("eagle", "OSPREY_CPU_EAGLE_HEADERS"),
            ("door", "OSPREY_CPU_EAGLE_HEADERS"),
            ("facts", "OSPREY_CPU_FACTS_HEADERS"),
        ],
        &out,
    )
    .unwrap_or_else(|why| panic!("{why}"));
}
