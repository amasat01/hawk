//! Reserved entry names and witnesses, with this machine's real host compiler: the names the system headers and a served tree
//! declare with C linkage, a witness program built and run, its verdict kept in the store, and a witness that cannot be built.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a panic is the failure
mod common;
use common::scratch;
use rcore::json::J;
use toolkit::{wire, Env, HeaderTree, Toolkit};

const OWN: HeaderTree = HeaderTree {
    name: "own",
    files: &[("own.h", b"extern \"C\" { int toolkit_taken(int); }\n")],
};
const WITNESS: &str = include_str!("data/gcc_masked_call.cpp");

fn table() -> String {
    r#"{"tools":{"compiler":{"rules":[{"path":"clang++"},{"path":"c++"},{"path":"g++"}]}},
        "reserved_names":{"tool":"compiler","args":["-E","-P","-x","c++","-std=c++17","-I{include:own}","{source}"],
                          "system_headers":["cstdlib","cstdio","cstring","cmath","unistd.h"]}}"#
        .to_string()
}

fn stated(tk: &Toolkit, ask: &str) -> J {
    wire::parse("a", &tk.machine(ask).unwrap())
        .unwrap()
        .get("stated")
        .unwrap()
        .clone()
}

#[test]
fn the_reserved_names_are_read_from_the_preprocessed_headers_of_the_real_compiler() {
    let dir = scratch("reserved");
    let tk = common::new_toolkit(&table(), &[OWN], Env::process()).unwrap();
    let ask = format!(
        "{{\"root\":\"{}\",\"reserved_names\":{{\"prelude\":\"#include \\\"own.h\\\"\\n\"}}}}",
        dir.display()
    );
    let names = stated(&tk, &ask).get("reserved_names").unwrap().strs();
    for expected in ["toolkit_taken", "strlen", "printf", "malloc", "sqrt"] {
        assert!(
            names.iter().any(|n| n == expected),
            "{expected} is not reserved; {} names",
            names.len()
        );
    }
    assert!(!names.iter().any(|n| n == "toolkit_free_name"));
    assert!(
        names.windows(2).all(|w| w[0] < w[1]),
        "names are sorted and unique"
    );
    // without a root the served tree cannot be written: a named refusal, not an empty list
    let refused = tk
        .machine("{\"reserved_names\":{\"prelude\":\"\"}}")
        .unwrap_err();
    assert!(
        refused.message.contains("needs a root"),
        "{}",
        refused.message
    );
}

/// A witness over the program of `source` with the verdicts of its exit codes.
fn witness(name: &str, source: &str, compile_flags: &[&str]) -> String {
    let compile: Vec<&str> = ["{compiler}"]
        .iter()
        .chain(compile_flags)
        .chain(&["-c", "{source}", "-o", "{object}"])
        .copied()
        .collect();
    wire::render(&wire::obj(vec![
        ("name", wire::s(name)),
        ("source", wire::s(source)),
        ("compile", wire::strs(&compile)),
        (
            "link",
            wire::strs(&["{compiler}", "{object}", "-o", "{output}"]),
        ),
        (
            "exit_verdicts",
            wire::obj(vec![("0", wire::s("fixed")), ("1", wire::s("miscompile"))]),
        ),
        ("timeout", wire::n(120)),
    ]))
}

#[test]
fn the_masked_call_witness_runs_on_the_real_compiler_and_its_verdict_is_kept() {
    let dir = scratch("witness");
    let tk = common::new_toolkit(&table(), &[OWN], Env::process()).unwrap();
    let flags = ["-O3", "-ffp-contract=fast", "-fno-trapping-math", "-w"];
    let ask = format!(
        "{{\"root\":\"{}\",\"witnesses\":[{},{},{},{}]}}",
        dir.display(),
        witness("masked_call", WITNESS, &flags),
        witness("wrong", "int main() { return 1; }", &["-O1"]),
        witness("does_not_build", "this is not C++", &["-O1"]),
        witness("odd_exit", "int main() { return 7; }", &["-O1"]),
    );
    let first = stated(&tk, &ask);
    let verdicts = first.get("witnesses").unwrap();
    let verdict = |name: &str| {
        verdicts
            .get(name)
            .unwrap()
            .get("verdict")
            .and_then(J::str)
            .unwrap()
            .to_string()
    };
    assert_eq!(
        (verdict("masked_call").as_str(), verdict("wrong").as_str()),
        ("fixed", "miscompile")
    );
    assert_eq!(
        (
            verdict("does_not_build").as_str(),
            verdict("odd_exit").as_str()
        ),
        ("unrun", "unrun")
    );
    assert!(verdicts
        .get("does_not_build")
        .unwrap()
        .get("why")
        .and_then(J::str)
        .unwrap()
        .contains("does not compile"));
    // verdicts are kept; an unrun one is not
    let hit = |v: &J, name: &str| {
        matches!(
            v.get("witnesses").unwrap().get(name).unwrap().get("hit"),
            Some(J::Bool(true))
        )
    };
    let second = stated(&tk, &ask);
    assert!(hit(&second, "masked_call") && hit(&second, "wrong"));
    assert!(!hit(&second, "does_not_build") && !hit(&second, "odd_exit"));
}

/// Oracle: two programs under one witness name and one store give their own verdicts, `fixed` for the one that exits 0 and
/// `miscompile` for the one that exits 1; the second is not answered with the first's kept verdict.
#[test]
fn a_verdict_kept_for_one_witness_text_is_not_read_for_another() {
    let dir = scratch("witness_text");
    let tk = common::new_toolkit(&table(), &[OWN], Env::process()).unwrap();
    let verdict_of = |source: &str| {
        let ask = format!(
            "{{\"root\":\"{}\",\"witnesses\":[{}]}}",
            dir.display(),
            witness("masked_call", source, &["-O1"])
        );
        let answer = stated(&tk, &ask);
        let found = answer.get("witnesses").unwrap().get("masked_call").unwrap();
        found.get("verdict").and_then(J::str).unwrap().to_string()
    };
    assert_eq!(verdict_of("int main() { return 0; }"), "fixed");
    assert_eq!(verdict_of("int main() { return 1; }"), "miscompile");
    assert_eq!(verdict_of("int main() { return 0; }"), "fixed");
}
