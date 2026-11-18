// The part of an OSPREY library's build script that is the same for every library; the script includes this file. It writes
// `abi.rs` (the abi number read from the door's header) and `identity.txt` (sha256 over the bytes of the library's own sources,
// the lowering crate's and the writer crate's) into `OUT_DIR`.
use std::path::{Path, PathBuf};

/// Collects the Rust source files under `dir`, in name order, for the identity.
fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            files(&path, out)
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path)
        }
    }
}

/// Reads the abi number from the door's header (`#define OB_ABI <n>`) into a Rust constant expression.
fn write_abi(header: &Path, out: &Path) {
    println!("cargo:rerun-if-changed={}", header.display());
    let text = std::fs::read_to_string(header).unwrap();
    let number = text
        .lines()
        .find_map(|line| line.strip_prefix("#define OB_ABI "))
        .and_then(|rest| rest.trim().parse::<u32>().ok())
        .expect("include/osprey.h defines OB_ABI");
    std::fs::write(out.join("abi.rs"), number.to_string()).unwrap();
}

/// Writes `abi.rs` and `identity.txt` for the library whose manifest directory is `here`.
fn write_build_facts(here: &Path) {
    let mut data = Vec::new();
    for dir in [
        here.join("src"),
        here.join("../components/lowering/src"),
        here.join("../../aether/rs/writers/src"),
    ] {
        println!("cargo:rerun-if-changed={}", dir.display());
        let mut found = vec![];
        files(&dir, &mut found);
        for file in found {
            data.extend_from_slice(file.strip_prefix(&dir).unwrap().to_string_lossy().as_bytes());
            data.push(0);
            data.extend_from_slice(&std::fs::read(&file).unwrap());
            data.push(0);
        }
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    write_abi(&here.join("../include/osprey.h"), &out);
    std::fs::write(out.join("identity.txt"), rcore::sha256::hex(&data)).unwrap();
}
