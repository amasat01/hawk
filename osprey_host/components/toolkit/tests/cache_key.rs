//! What moves the key of a build and what does not, and how often a compiler is asked who it is: a changed flag, compiler or header
//! tree is a miss; a touched file and a repeated question are not.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a panic is the failure
mod common;
use common::{fake_compiler, member, one_tool_table, runs, scratch, script};
use rcore::json::J;
use std::fs;
use std::path::Path;
use toolkit::{wire, Env, HeaderTree, Toolkit};

/// A toolkit over the compiler `fakecc` of `dir`, serving the header trees `trees`.
fn toolkit(dir: &Path, trees: &[HeaderTree]) -> Toolkit {
    let env = Env::from_pairs(&[
        ("FAKE_CXX", dir.join("fakecc").to_str().unwrap()),
        ("PATH", "/bin:/usr/bin"),
    ]);
    Toolkit::new(&one_tool_table("FAKE_CXX"), trees, env).unwrap()
}

/// The build request of a unit with the flags `extra` and an include folder of the tree `t`.
fn request(extra: &[&str]) -> String {
    wire::render(&wire::obj(vec![
        ("name", wire::s("u")),
        ("text", wire::s("int x;")),
        (
            "template",
            wire::strs(&[
                "{compiler}",
                "-I{include:t}",
                "-c",
                "{source}",
                "-o",
                "{output}",
            ]),
        ),
        ("extra", wire::strs(extra)),
    ]))
}

/// Whether the build answered from the store.
fn hit(answer: &str) -> bool {
    matches!(member(answer, "hit"), J::Bool(true))
}

const TREE_A: HeaderTree = HeaderTree {
    name: "t",
    files: &[("a.h", b"int a;")],
};
const TREE_B: HeaderTree = HeaderTree {
    name: "t",
    files: &[("a.h", b"int b;")],
};

#[test]
fn a_changed_flag_is_a_miss_and_the_same_flags_again_are_a_hit() {
    let dir = scratch("key_flag");
    fake_compiler(&dir, "fakecc");
    let tk = toolkit(&dir, &[TREE_A]);
    let store = dir.join("store");
    assert!(!hit(&tk.build(&request(&["-O3"]), &store).unwrap()));
    assert!(!hit(&tk.build(&request(&["-O0"]), &store).unwrap()));
    assert!(hit(&tk.build(&request(&["-O3"]), &store).unwrap()));
    assert_eq!(runs(&dir), 2);
}

/// Oracle: the header tree's bytes are the include root's content; a folder with the same bytes is the same key.
#[test]
fn a_changed_header_tree_is_a_miss_and_the_same_bytes_are_a_hit() {
    let dir = scratch("key_tree");
    fake_compiler(&dir, "fakecc");
    let store = dir.join("store");
    let first = toolkit(&dir, &[TREE_A])
        .build(&request(&[]), &store)
        .unwrap();
    let same = toolkit(&dir, &[TREE_A])
        .build(&request(&[]), &store)
        .unwrap();
    let changed = toolkit(&dir, &[TREE_B])
        .build(&request(&[]), &store)
        .unwrap();
    assert!(!hit(&first) && hit(&same) && !hit(&changed));
    assert_eq!(runs(&dir), 2);
}

/// A compiler script whose version text is two lines, the second `second`; it counts its version questions in `dir/asked` and its
/// builds in `dir/runs`.
fn compiler_saying(dir: &Path, second: &str) {
    script(
        &dir.join("fakecc"),
        &format!(
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo x >> \"{d}/asked\"; [ -f \"{d}/sleep\" ] && sleep \"$(cat \"{d}/sleep\")\"; echo 'fakecc 1.0'; echo '{second}'; exit 0; fi\n\
             echo run >> \"{d}/runs\"\nout=\"\"; prev=\"\"\n\
             for a in \"$@\"; do [ \"$prev\" = \"-o\" ] && out=\"$a\"; prev=\"$a\"; done\necho built > \"$out\"\n",
            d = dir.display()
        ),
    );
}

fn asked(dir: &Path) -> usize {
    fs::read_to_string(dir.join("asked")).map_or(0, |t| t.lines().count())
}

/// Oracle: two releases of one compiler print one first line and differ in a later one, as distribution builds of one version do.
#[test]
fn a_compiler_that_differs_only_past_its_first_version_line_is_another_key() {
    let dir = scratch("key_compiler");
    compiler_saying(&dir, "built with patch a");
    let tk = toolkit(&dir, &[TREE_A]);
    let store = dir.join("store");
    assert!(!hit(&tk.build(&request(&[]), &store).unwrap()));
    compiler_saying(&dir, "built with patch b");
    assert!(!hit(&tk.build(&request(&[]), &store).unwrap()));
    assert_eq!(runs(&dir), 2);
}

#[test]
fn a_touched_compiler_is_asked_again_and_is_the_same_key() {
    let dir = scratch("key_touch");
    compiler_saying(&dir, "built with patch a");
    let tk = toolkit(&dir, &[TREE_A]);
    let store = dir.join("store");
    assert!(!hit(&tk.build(&request(&[]), &store).unwrap()));
    let touched = std::process::Command::new("touch")
        .arg(dir.join("fakecc"))
        .status()
        .unwrap();
    assert!(touched.success());
    assert!(hit(&tk.build(&request(&[]), &store).unwrap()));
    assert_eq!(runs(&dir), 1);
}

#[test]
fn a_compiler_is_asked_who_it_is_once_however_often_the_machine_is_stated() {
    let dir = scratch("probe_once");
    compiler_saying(&dir, "built with patch a");
    let tk = toolkit(&dir, &[]);
    for terms in ["\"a\"", "\"b\"", "\"a\""] {
        tk.machine(&format!("{{\"key_terms\":[{terms}]}}")).unwrap();
    }
    assert_eq!(asked(&dir), 1);
}

#[test]
fn a_replaced_compiler_never_answers_with_the_old_identity() {
    let dir = scratch("probe_replaced");
    compiler_saying(&dir, "built with patch a");
    let tk = toolkit(&dir, &[]);
    let key = || member(&tk.machine("{\"key_terms\":[\"k\"]}").unwrap(), "key");
    let old = key();
    compiler_saying(&dir, "built with a longer banner of patch b");
    let new = key();
    assert_ne!(old.str(), new.str());
    assert_eq!(asked(&dir), 2);
}

/// The machine ask that keeps the compiler's answers under the store of `dir`.
fn kept_ask(dir: &Path) -> String {
    format!(
        "{{\"key_terms\":[\"k\"],\"root\":\"{}\"}}",
        dir.join("store").display()
    )
}

/// Oracle: a fresh process is a toolkit whose memo of this process is emptied (`reset`); the fake counts the questions it is asked.
#[test]
fn a_probe_answers_from_the_store_in_a_fresh_process_and_a_replaced_compiler_is_asked_once() {
    let dir = scratch("probe_disk");
    compiler_saying(&dir, "built with patch a");
    let tk = toolkit(&dir, &[]);
    let first = tk.machine(&kept_ask(&dir)).unwrap();
    tk.state("reset", &dir).unwrap();
    let fresh = tk.machine(&kept_ask(&dir)).unwrap();
    assert_eq!(asked(&dir), 1, "a kept probe must not start the compiler");
    assert_eq!(member(&first, "key").str(), member(&fresh, "key").str());
    compiler_saying(&dir, "built with a longer banner of patch b");
    tk.state("reset", &dir).unwrap();
    let replaced = tk.machine(&kept_ask(&dir)).unwrap();
    assert_eq!(asked(&dir), 2);
    assert_ne!(member(&first, "key").str(), member(&replaced, "key").str());
}

/// The body of a child process of the test below: states the machine under the folders its environment names.
#[test]
fn child_states_the_machine() {
    let Ok(dir) = std::env::var("TOOLKIT_CHILD_DIR") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    toolkit(&dir, &[]).machine(&kept_ask(&dir)).unwrap();
}

#[test]
fn three_processes_stating_one_machine_ask_its_compiler_once() {
    let dir = scratch("probe_processes");
    fs::write(dir.join("sleep"), "0.5").unwrap();
    compiler_saying(&dir, "built with patch a");
    let mut running: Vec<_> = (0..3)
        .map(|_| {
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "child_states_the_machine", "--test-threads=1"])
                .env("TOOLKIT_CHILD_DIR", &dir)
                .spawn()
                .unwrap()
        })
        .collect();
    for child in &mut running {
        assert!(child.wait().unwrap().success());
    }
    assert_eq!(asked(&dir), 1);
}
