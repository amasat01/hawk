//! Builds the host runtime into the cpu OSPREY: the sources of eagle's C door that `sources.txt` lists without a `cuda`
//! prefix are compiled with the C++ compiler `CXX` and their objects are linked into the loadable library, which then exports
//! the door's `eg_` functions beside `osprey_v1`. The device group of the door is not built; its host-only refusals are.
//!
//! The inputs, each refused by name when absent: `OSPREY_EAGLE_SRC` (the eagle source root, which holds `eagle/door/` and
//! `plugin/`), `OSPREY_AETHER_SRC` (the aether source root, whose headers the door includes), `CXX` (a C++23 compiler with
//! OpenMP) and the raptor sibling (`RAPTOR`, default the sibling folder), which holds the cpu wing's facts header and the door's
//! row header for a host-only door.
use std::path::{Path, PathBuf};
use std::process::Command;

/// The value of the environment variable `name`, or a refusal naming the variable and what it must hold.
fn input(name: &str, holds: &str) -> String {
    println!("cargo:rerun-if-env-changed={name}");
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set: it names {holds}"))
}

/// The path `path`, or a refusal naming the missing entry and the variable that locates it.
fn existing(path: PathBuf, what: &str) -> PathBuf {
    assert!(path.exists(), "{what} is missing: {}", path.display());
    path
}

/// The sources of the host group, as `sources.txt` of the door lists them: a path alone is built in every mode, a `cuda` line
/// only with the device group.
fn host_sources(door: &Path) -> Vec<String> {
    let list = std::fs::read_to_string(door.join("sources.txt")).expect("the door's sources.txt");
    list.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .filter(|line| !line.starts_with("cuda "))
        .map(str::to_string)
        .collect()
}

/// Compiles `source` of the door to an object in `out` and returns the object's path.
fn compile(cxx: &str, flags: &[String], door: &Path, source: &str, out: &Path) -> PathBuf {
    let object = out.join(source.replace('/', "_")).with_extension("o");
    let done = Command::new(cxx)
        .args(flags)
        .args(["-c"])
        .arg(door.join(source))
        .arg("-o")
        .arg(&object)
        .output()
        .unwrap_or_else(|why| panic!("CXX {cxx} did not run: {why}"));
    assert!(
        done.status.success(),
        "{source} did not compile with {cxx}:\n{}",
        String::from_utf8_lossy(&done.stderr)
    );
    object
}

/// The runtime library the compiler `cxx` links against: the file it names for `library`.
fn compiler_runtime(cxx: &str, library: &str) -> PathBuf {
    let asked = Command::new(cxx)
        .arg(format!("-print-file-name={library}"))
        .output()
        .unwrap_or_else(|why| panic!("CXX {cxx} did not run: {why}"));
    let found = PathBuf::from(String::from_utf8_lossy(&asked.stdout).trim());
    assert!(
        found.is_absolute() && found.exists(),
        "CXX {cxx} names no {library}: it must be a C++ compiler with OpenMP"
    );
    found
}

fn main() {
    let eagle = PathBuf::from(input("OSPREY_EAGLE_SRC", "the eagle source root"));
    let aether = PathBuf::from(input("OSPREY_AETHER_SRC", "the aether source root"));
    let cxx = input("CXX", "a C++23 compiler with OpenMP");
    let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let raptor =
        std::env::var("RAPTOR").map_or_else(|_| here.join("../../../raptor"), PathBuf::from);
    println!("cargo:rerun-if-env-changed=RAPTOR");
    let door = existing(eagle.join("eagle/door"), "OSPREY_EAGLE_SRC/eagle/door");
    let cards = raptor.join("rs/wings/cards");
    let facts = existing(
        cards.join("cpu/raptor_facts.h"),
        "the cpu wing's raptor_facts.h",
    );
    let rows = existing(
        cards.join("door/host/eagle_door_rows.h"),
        "the host door rows",
    );
    println!("cargo:rerun-if-changed={}", door.display());
    println!("cargo:rerun-if-changed={}", facts.display());
    println!("cargo:rerun-if-changed={}", rows.display());

    let base: Vec<String> = [
        "-std=c++23",
        "-O2",
        "-fPIC",
        "-fvisibility=hidden",
        "-fvisibility-inlines-hidden",
        "-ffp-contract=off",
        "-fopenmp",
        "-DAETHER_CPP_MODE=1",
        "-DEAGLE_CPU_ONLY=1",
    ]
    .iter()
    .map(ToString::to_string)
    .chain([
        "-include".to_string(),
        facts.display().to_string(),
        format!("-I{}", eagle.display()),
        format!("-I{}", aether.display()),
    ])
    .collect();
    // the row tables are read by Context.cpp alone
    let with_rows: Vec<String> = base
        .iter()
        .cloned()
        .chain(["-include".to_string(), rows.display().to_string()])
        .collect();
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let jobs = std::env::var("NUM_JOBS")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(2)
        .max(1);
    let sources = host_sources(&door);
    let mut objects = Vec::new();
    for batch in sources.chunks(jobs) {
        std::thread::scope(|scope| {
            let running: Vec<_> = batch
                .iter()
                .map(|source| {
                    let flags = if source == "Context.cpp" {
                        &with_rows
                    } else {
                        &base
                    };
                    scope.spawn(|| compile(&cxx, flags, &door, source, &out))
                })
                .collect();
            objects.extend(running.into_iter().map(|job| job.join().unwrap()));
        });
    }
    for object in &objects {
        println!("cargo:rustc-link-arg-cdylib={}", object.display());
    }
    if std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "macos") {
        println!(
            "cargo:rustc-link-arg-cdylib={}",
            compiler_runtime(&cxx, "libomp.dylib").display()
        );
        println!("cargo:rustc-link-arg-cdylib=-lc++");
        return;
    }
    println!(
        "cargo:rustc-link-arg-cdylib={}",
        compiler_runtime(&cxx, "libgomp.so").display()
    );
    // The objects need the C++ runtime of the compiler that made them, which may be newer than the
    // system's: the file CXX names can be a linker script that adds an archive beside it, so its
    // folder is searched too.
    let cxx_runtime = compiler_runtime(&cxx, "libstdc++.so");
    let beside = cxx_runtime.parent().expect("an absolute file has a folder");
    println!("cargo:rustc-link-arg-cdylib=-L{}", beside.display());
    println!("cargo:rustc-link-arg-cdylib={}", cxx_runtime.display());
}
