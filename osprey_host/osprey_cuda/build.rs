//! The library's identity and abi number, the header trees it embeds, and the one decision of how it is built: with or without the
//! device runtime.
//!
//! The rule. The library embeds eagle's device runtime (the device group of its C door) exactly when the machine that builds it has
//! a CUDA toolkit: the `compiler` tool of `facts/tools.json`, found the way the library finds it at run time (`HAWK_NVCC`, else
//! `nvcc` on `PATH`; `HAWK_DEVICE_COMPILER=nvrtc` selects the runtime compile library, which cannot build the objects, so it
//! builds without). The toolkit root is the folder above the folder of that tool, as the table's `process_env.root_of` states it.
//! Without a toolkit (a Mac, a machine with no CUDA) the library is built without the door: it exports `osprey_v1` alone and its
//! machine answer states `runs_here` false with a `why` that says it was built without the device runtime. A library built
//! without a toolkit can therefore never answer that it runs. The decision is made here once: this script sets the cfg
//! `device_runtime` for the library's own code and hands the root to the door package, whose build script compiles and links the
//! objects.
include!("../components/lowering/build_support.rs");
use toolkit::{discover, wire, Env};

/// The root of the CUDA toolkit this build can compile with, or `None` when the machine has none.
fn toolkit_root(here: &Path) -> Option<PathBuf> {
    for variable in ["HAWK_NVCC", "HAWK_DEVICE_COMPILER", "PATH"] {
        println!("cargo::rerun-if-env-changed={variable}");
    }
    let table = std::fs::read_to_string(here.join("facts/tools.json")).expect("facts/tools.json");
    let spec = wire::parse("the tool table", &table).unwrap_or_else(|why| panic!("{why}"));
    let env = Env::process();
    let machine = discover::find(&spec, &env, None).unwrap_or_else(|why| panic!("{why}"));
    machine.tool("compiler")?;
    let vars = discover::child_env(&spec, &env, &machine).unwrap_or_else(|why| panic!("{why}"));
    vars.get("CUDA_PATH").map(PathBuf::from)
}

fn main() {
    let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    write_build_facts(&here);
    println!("cargo::rustc-check-cfg=cfg(device_runtime)");
    if let Some(root) = toolkit_root(&here) {
        println!("cargo::rustc-cfg=device_runtime");
        println!("cargo::metadata=root={}", root.display());
    }
    // the folders of the headers the library's compiles include, named by the environment of this build
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("trees.rs");
    toolkit::embed::write_table(
        &[
            ("aether", "OSPREY_CUDA_AETHER_HEADERS"),
            ("eagle", "OSPREY_CUDA_EAGLE_HEADERS"),
            ("facts", "OSPREY_CUDA_FACTS_HEADERS"),
            ("rtc", "OSPREY_CUDA_RTC_HEADERS"),
        ],
        &out,
    )
    .unwrap_or_else(|why| panic!("{why}"));
}
