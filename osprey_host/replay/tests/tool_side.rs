//! The tool side of the libraries as a caller that knows nothing of the target uses it: a device unit's text is written on a machine
//! with no device compiler (only a build needs one), and the defines a host unit was compiled with reach the link of its set and the
//! key of the build.
#![cfg(feature = "old-contract")]
mod common;
use osprey_lowering::tool;
use rcore::json::{self, J};
use std::path::PathBuf;

/// Puts the shipped documents of both libraries in the folder `facts` beside the running program, where the tool side reads them
/// (it has no argument for the folder), once per process.
fn shipped_beside_the_program() {
    /// Makes the folder once.
    static MADE: std::sync::Once = std::sync::Once::new();
    MADE.call_once(|| {
        let beside = std::env::current_exe().unwrap().with_file_name("facts");
        std::fs::create_dir_all(&beside).unwrap();
        for file in std::fs::read_dir(osprey_replay::merged_facts()).unwrap() {
            let file = file.unwrap().path();
            std::fs::copy(&file, beside.join(file.file_name().unwrap())).unwrap();
        }
    });
}

/// A fresh store root for one test.
fn store(test: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("osprey_tool_side_{}_{test}", std::process::id()));
    shipped_beside_the_program();
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

/// The host library's machine answer for this machine, none when it has no compiler.
fn host_machine(root: &std::path::Path) -> Option<String> {
    let ask = format!(
        r#"{{"target": "host", "options": {{}}, "opt_level": "O1", "root": "{}"}}"#,
        root.display()
    );
    let answer = tool::machine(&osprey_cpu::Cpu, &ask).unwrap();
    let parsed = json::parse(&answer).unwrap();
    match parsed.get("tools_absent") {
        Some(J::Str(_)) => None,
        _ => Some(answer),
    }
}

/// The host library's build of one unit whose text is `text`, compiled with `defines`: the answer's text.
fn build_unit(machine: &str, text: &str, defines: &[&str], root: &std::path::Path) -> String {
    let list: Vec<String> = defines.iter().map(|d| format!("{d:?}")).collect();
    let request = format!(
        r#"{{"name": "u", "text": {text:?}, "machine": {machine}, "defines": [{}]}}"#,
        list.join(", ")
    );
    tool::build(&osprey_cpu::Cpu, &request, root).unwrap()
}

/// Oracle: the compiler's own refusal of a mark nobody defined. A unit compiled with `MARK=7` makes a set text that reads `MARK`
/// link; the same set text is refused when its part carries no define.
#[test]
fn the_defines_of_a_unit_reach_the_compile_of_its_set() {
    let root = store("link");
    let Some(machine) = host_machine(&root) else {
        return common::skipped(None, "the machine has no host compiler");
    };
    let unit = "extern \"C\" int u() { return 1; }\n";
    let set = "#ifndef MARK\n#error the set was compiled without the define of its part\n#endif\nextern \"C\" int s() { return MARK; }\n";
    let linked = |defines: &[&str], name: &str| {
        let part = build_unit(&machine, unit, defines, &root);
        let parts = format!("[{part}]");
        tool::link(&osprey_cpu::Cpu, &parts, set, name, &root)
    };
    assert!(linked(&["MARK=7"], "with.dylib").is_ok());
    let why = linked(&[], "without.dylib").unwrap_err();
    assert!(why.message.contains("MARK"), "{}", why.message);
}

/// Oracle: two builds of one text that differ only in their defines are two compiler runs under two keys, and the same defines again
/// is a store hit.
#[test]
fn builds_that_differ_only_in_their_defines_have_two_keys() {
    let root = store("key");
    let Some(machine) = host_machine(&root) else {
        return common::skipped(None, "the machine has no host compiler");
    };
    let unit = "extern \"C\" int u() { return 1; }\n";
    let key = |answer: &str| {
        let parsed = json::parse(answer).unwrap();
        parsed.get("key").and_then(J::str).unwrap().to_string()
    };
    let hit = |answer: &str| {
        let parsed = json::parse(answer).unwrap();
        matches!(parsed.get("hit"), Some(J::Bool(true)))
    };
    let first = build_unit(&machine, unit, &["A=1"], &root);
    let other = build_unit(&machine, unit, &["A=2"], &root);
    let again = build_unit(&machine, unit, &["A=1"], &root);
    assert!(
        !hit(&first) && !hit(&other),
        "two defines, two compiler runs"
    );
    assert_ne!(key(&first), key(&other));
    assert!(hit(&again) && key(&again) == key(&first));
}

/// A device library states the machine of a host with no device compiler and still gives the facts of the text it writes: the
/// release its card names, with that release's PTX version; a build on that machine is refused as a missing tool.
#[test]
fn a_device_text_is_written_under_the_cards_release_on_a_machine_with_no_device_compiler() {
    shipped_beside_the_program();
    let machine = r#"{"key": "k", "stated": {"asked": {"opt_level": "", "options": {}}}, "tools_absent": "no device compiler found"}"#;
    let parsed = aether_writers::contract::Contract::parse(machine)
        .unwrap()
        .0;
    let facts = tool::facts_for(&osprey_cuda::Cuda, &parsed).unwrap();
    let answer = json::parse(&facts).unwrap();
    let build = answer.get("device_build").unwrap();
    assert_eq!(build.get("compiler").and_then(J::str), Some("nvcc 12.6"));
    assert_eq!(build.get("ptx_isa").map(J::int), Some(85));
    let request = format!(r#"{{"name": "u", "text": "", "machine": {machine}}}"#);
    let why = tool::build(&osprey_cuda::Cuda, &request, &store("absent")).unwrap_err();
    assert!(
        why.message.contains("no device compiler"),
        "{}",
        why.message
    );
}
