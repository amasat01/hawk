//! The library's identity: sha256 over the bytes of its own source files and of the writer crate's source files.
use std::path::{Path, PathBuf};

/// Collects the Rust source files under `dir`, in name order, for the identity.
fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut es: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    es.sort();
    for p in es {
        if p.is_dir() {
            files(&p, out)
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p)
        }
    }
}

/// Reads the abi number from the loader's header (`#define OB_ABI <n>`) into a Rust constant expression.
fn write_abi(here: &Path, out: &Path) {
    let header = here.join("include/osprey_host.h");
    println!("cargo:rerun-if-changed={}", header.display());
    let text = std::fs::read_to_string(&header).unwrap();
    let number = text
        .lines()
        .find_map(|line| line.strip_prefix("#define OB_ABI "))
        .and_then(|rest| rest.trim().parse::<u32>().ok())
        .expect("include/osprey_host.h defines OB_ABI");
    std::fs::write(out.join("abi.rs"), number.to_string()).unwrap();
}

fn main() {
    let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut data = Vec::new();
    for dir in [here.join("src"), here.join("../aether/rs/writers/src")] {
        println!("cargo:rerun-if-changed={}", dir.display());
        let mut fs = vec![];
        files(&dir, &mut fs);
        for f in fs {
            data.extend_from_slice(f.strip_prefix(&dir).unwrap().to_string_lossy().as_bytes());
            data.push(0);
            data.extend_from_slice(&std::fs::read(&f).unwrap());
            data.push(0);
        }
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    write_abi(&here, &out);
    std::fs::write(out.join("identity.txt"), rcore::sha256::hex(&data)).unwrap();
}
