//! The host compiler of a device build, found by the library's shipped tool table: the standard variables before `PATH`.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a panic is the failure
mod common;
use common::{fake_compiler, scratch, script};
use rcore::json::J;
use std::fs;
use toolkit::{wire, Env, Toolkit};

/// The machine answer of a toolkit, as data.
fn ask(tk: &Toolkit) -> J {
    wire::parse(
        "answer",
        &tk.machine("{\"key_terms\":[\"facts-1\"]}").unwrap(),
    )
    .unwrap()
}

/// The host compiler the shipped device table finds, with `vars` set beside a device compiler and a `g++` on `PATH`.
fn shipped_host_compiler(dir: &std::path::Path, vars: &[(&str, String)]) -> String {
    let table = fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("facts/tools.json"),
    )
    .unwrap();
    let nvcc = dir.join("cuda/nvcc");
    script(
        &nvcc,
        "#!/bin/sh\necho 'Cuda compilation tools, release 12.6, V12.6.20'\n",
    );
    let bin = dir.to_str().unwrap().to_string();
    let nvcc_path = nvcc.display().to_string();
    let mut pairs = vec![("HAWK_NVCC", nvcc_path.as_str()), ("PATH", bin.as_str())];
    pairs.extend(vars.iter().map(|(k, v)| (*k, v.as_str())));
    let answer = ask(&Toolkit::new(&table, &[], Env::from_pairs(&pairs)).unwrap());
    let tools = answer.get("stated").unwrap().get("tools").unwrap();
    tools
        .get("host_compiler")
        .unwrap()
        .get("path")
        .and_then(J::str)
        .unwrap()
        .to_string()
}

#[test]
fn the_device_host_compiler_is_read_from_the_standard_variables_before_path() {
    let dir = scratch("device_host");
    for name in ["from_cxx", "from_cudahostcxx"] {
        fake_compiler(&dir.join(name), "g++");
    }
    fake_compiler(&dir, "g++");
    let at = |name: &str| dir.join(name).join("g++").display().to_string();
    // CUDAHOSTCXX first, then CXX (a command line: its first word), then the compiler on PATH
    let both = [
        ("CUDAHOSTCXX", at("from_cudahostcxx")),
        ("CXX", format!("{} -m64", at("from_cxx"))),
    ];
    assert_eq!(shipped_host_compiler(&dir, &both), at("from_cudahostcxx"));
    assert_eq!(shipped_host_compiler(&dir, &both[1..]), at("from_cxx"));
    assert_eq!(
        shipped_host_compiler(&dir, &[]),
        dir.join("g++").display().to_string()
    );
}
