//! Tool discovery by a table: a planted tool on a private PATH, none at all, an explicit override that does not resolve, the
//! host compiler of a device build read from a flag in the environment, and the environment a tool then runs under.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a panic is the failure
mod common;
use common::{fake_compiler, member, scratch, script};
use rcore::json::J;
use std::fs;
use toolkit::{wire, Env, Toolkit};

const TABLE: &str = r#"{"tools":{"compiler":{"rules":[
    {"env":"TEST_CXX","required":true},{"path":"fakecc"}],"fix":"install fakecc or set TEST_CXX"}}}"#;

fn ask(tk: &Toolkit) -> J {
    wire::parse(
        "answer",
        &tk.machine("{\"key_terms\":[\"facts-1\"]}").unwrap(),
    )
    .unwrap()
}

#[test]
fn a_tool_planted_on_a_private_path_is_found_and_identified() {
    let dir = scratch("planted");
    fake_compiler(&dir.join("bin"), "fakecc");
    let path = dir.join("bin").display().to_string();
    let tk = Toolkit::new(TABLE, &[], Env::from_pairs(&[("PATH", &path)])).unwrap();
    let answer = ask(&tk);
    let tool = answer
        .get("stated")
        .unwrap()
        .get("tools")
        .unwrap()
        .get("compiler")
        .unwrap();
    assert_eq!(
        tool.get("path").and_then(J::str),
        Some(dir.join("bin/fakecc").to_str().unwrap())
    );
    assert!(tool
        .get("identity")
        .and_then(J::str)
        .unwrap()
        .contains("||fakecc 1.0||"));
    assert_eq!(answer.get("key").and_then(J::str).map(str::len), Some(64));
    assert!(matches!(answer.get("tools_absent"), Some(J::Null)));
}

#[test]
fn the_key_moves_with_the_asked_terms_and_the_tool() {
    let dir = scratch("key");
    fake_compiler(&dir, "fakecc");
    let tk = Toolkit::new(
        TABLE,
        &[],
        Env::from_pairs(&[("PATH", dir.to_str().unwrap())]),
    )
    .unwrap();
    let key = |terms: &str| {
        member(
            &tk.machine(&format!("{{\"key_terms\":[{terms}]}}")).unwrap(),
            "key",
        )
    };
    let (a, b) = (key("\"x\""), key("\"y\""));
    assert_ne!(a.str(), b.str());
    assert_eq!(a.str(), key("\"x\"").str());
}

#[test]
fn no_tool_at_all_is_an_absence_that_says_what_was_tried() {
    let dir = scratch("none");
    let tk = Toolkit::new(
        TABLE,
        &[],
        Env::from_pairs(&[("PATH", dir.to_str().unwrap())]),
    )
    .unwrap();
    let answer = ask(&tk);
    let reason = answer.get("tools_absent").and_then(J::str).unwrap();
    assert!(
        reason.contains("no compiler found") && reason.contains("$TEST_CXX (unset)"),
        "{reason}"
    );
    assert!(
        reason.contains("fakecc on PATH (absent)")
            && reason.contains("install fakecc or set TEST_CXX"),
        "{reason}"
    );
    // the machine is still stated: no tool, the header digests, and a key; a build refuses with the same reason
    let stated = answer.get("stated").unwrap();
    assert_eq!(wire::render(stated.get("tools").unwrap()), "{}");
    assert_eq!(answer.get("key").and_then(J::str).map(str::len), Some(64));
    let refused = tk
        .build(&common::unit("u", "int x;"), &dir.join("store"))
        .unwrap_err();
    assert_eq!(refused.kind, toolkit::Kind::ToolAbsent);
    assert!(
        refused.message.contains("no compiler found"),
        "{}",
        refused.message
    );
    assert!(tk
        .link(
            "{\"name\":\"l\",\"parts\":[],\"link\":{\"linux\":[],\"macos\":[]}}",
            &dir.join("store")
        )
        .is_err());
}

#[test]
fn an_explicit_override_that_does_not_resolve_stops_the_search() {
    let dir = scratch("override");
    fake_compiler(&dir, "fakecc");
    let env = Env::from_pairs(&[
        ("PATH", dir.to_str().unwrap()),
        ("TEST_CXX", "no-such-compiler"),
    ]);
    let answer = ask(&Toolkit::new(TABLE, &[], env).unwrap());
    let reason = answer.get("tools_absent").and_then(J::str).unwrap();
    assert!(
        reason.contains("$TEST_CXX=\"no-such-compiler\" not found on PATH"),
        "{reason}"
    );
}

#[test]
fn a_tool_that_does_not_run_is_absent_not_found() {
    let dir = scratch("broken");
    script(&dir.join("fakecc"), "#!/bin/sh\necho trouble >&2\nexit 3\n");
    let tk = Toolkit::new(
        TABLE,
        &[],
        Env::from_pairs(&[("PATH", dir.to_str().unwrap())]),
    )
    .unwrap();
    let reason = ask(&tk)
        .get("tools_absent")
        .and_then(J::str)
        .unwrap()
        .to_string();
    assert!(
        reason.contains("was found but `--version` does not run (exit 3: trouble)"),
        "{reason}"
    );
}

#[test]
fn a_tool_that_cannot_be_started_is_absent_with_the_operating_systems_error() {
    let dir = scratch("unstartable");
    // a script whose interpreter does not exist: the system refuses to start it
    script(&dir.join("fakecc"), "#!/no/such/interpreter\n");
    let tk = Toolkit::new(
        TABLE,
        &[],
        Env::from_pairs(&[("PATH", dir.to_str().unwrap())]),
    )
    .unwrap();
    let reason = ask(&tk)
        .get("tools_absent")
        .and_then(J::str)
        .unwrap()
        .to_string();
    assert!(
        reason.contains("does not run (") && reason.contains("No such file or directory"),
        "{reason}"
    );
}

/// A device-shaped table: the device compiler with two probes, its host compiler from a flag in the environment (a folder resolves
/// to the `g++` inside it) else the device compiler's own table entry, and the environment adjustments.
const DEVICE_TABLE: &str = r#"{"tools":{
  "compiler":{"rules":[{"env":"FAKE_TOOL","required":true}],"probes":[
     {"name":"release","args":["--version"],"after":"release ","as":"version"},
     {"name":"archs","args":["--list-gpu-arch"],"after":"compute_","as":"int_list"}]},
  "host_compiler":{"rules":[{"env_flag":{"var":"TOOL_PREPEND_FLAGS","flags":["-hostbin"]},"dir_child":"g++"}]}},
 "process_env":{"strip_flags":{"vars":["TOOL_PREPEND_FLAGS"],"flags":["-hostbin"]},"remove":["TOOL_HOSTBIN"],
   "path_prepend":["compiler"],"root_of":{"KIT_PATH":"compiler"}}}"#;

#[test]
fn a_device_table_finds_the_host_compiler_in_a_flag_and_runs_tools_in_the_adjusted_environment() {
    let dir = scratch("device");
    let devtool = dir.join("kit/bin/devtool");
    script(
        &devtool,
        &format!(
            "#!/bin/sh\ncase \"$1\" in --version) echo 'Kit compilation tools, release 12.6, V12.6.20'; exit 0;; \
             --list-gpu-arch) printf 'compute_75\\ncompute_61\\ncompute_75\\n'; exit 0;; esac\n\
             echo \"$TOOL_PREPEND_FLAGS|$TOOL_HOSTBIN|$KIT_PATH|$PATH\" > \"{}/env.txt\"\ntouch \"$2\"\n",
            dir.display()
        ),
    );
    fake_compiler(&dir.join("hostcc"), "g++");
    let flags = format!("-O2 -hostbin {}", dir.join("hostcc").display());
    let env = Env::from_pairs(&[
        ("FAKE_TOOL", devtool.to_str().unwrap()),
        ("TOOL_PREPEND_FLAGS", &flags),
        ("TOOL_HOSTBIN", "/elsewhere"),
        ("PATH", "/bin:/usr/bin"),
    ]);
    let tk = Toolkit::new(DEVICE_TABLE, &[], env).unwrap();
    let answer = ask(&tk);
    let tools = answer.get("stated").unwrap().get("tools").unwrap();
    let compiler = tools.get("compiler").unwrap();
    assert_eq!(compiler.get("release").and_then(J::str), Some("12.6"));
    assert_eq!(wire::render(compiler.get("archs").unwrap()), "[61,75]");
    assert_eq!(
        tools
            .get("host_compiler")
            .unwrap()
            .get("path")
            .and_then(J::str),
        Some(dir.join("hostcc/g++").to_str().unwrap())
    );
    // the devtool run sees no -hostbin and no TOOL_HOSTBIN, the toolkit's bin first on PATH and KIT_PATH at its root
    let request = r#"{"name":"k","text":"","source_suffix":".cu","output_suffix":".img","template":["{compiler}","-image","{output}"]}"#;
    tk.build(request, &dir.join("store")).unwrap();
    let seen = fs::read_to_string(dir.join("env.txt")).unwrap();
    let parts: Vec<&str> = seen.trim().splitn(4, '|').collect();
    assert_eq!(parts[..3], ["-O2", "", dir.join("kit").to_str().unwrap()]);
    assert!(
        parts[3].starts_with(&format!("{}:", dir.join("kit/bin").display())),
        "{}",
        parts[3]
    );
}

#[test]
fn the_absence_of_a_host_compiler_names_every_compiler_tried_in_order() {
    let dir = scratch("tried");
    let table = r#"{"tools":{"compiler":{"rules":[{"env":"HAWK_CXX","required":true},{"env":"CXX","first_word":true},{"path":"g++"},{"path":"otherc++"}],
        "fix":"install g++ or set HAWK_CXX=/path/to/c++"}}}"#;
    let env = Env::from_pairs(&[("PATH", dir.to_str().unwrap()), ("CXX", "no-such-cxx -m64")]);
    let answer = ask(&Toolkit::new(table, &[], env).unwrap());
    assert_eq!(
        answer.get("tools_absent").and_then(J::str),
        Some(
            "no compiler found. Tried, in order: $HAWK_CXX (unset), $CXX (\"no-such-cxx\", did not resolve), g++ on PATH (absent), \
             otherc++ on PATH (absent); install g++ or set HAWK_CXX=/path/to/c++"
        )
    );
}

#[test]
fn a_host_compiler_in_a_quoted_flag_value_with_a_space_in_its_path_is_found() {
    let dir = scratch("quoted");
    let devtool = dir.join("devtool");
    script(&devtool, "#!/bin/sh\necho 'fake devtool'\n");
    fake_compiler(&dir.join("my hostcc"), "g++");
    let flags = format!("-O2 -hostbin '{}' -g", dir.join("my hostcc").display());
    let env = Env::from_pairs(&[
        ("FAKE_TOOL", devtool.to_str().unwrap()),
        ("TOOL_PREPEND_FLAGS", &flags),
        ("PATH", "/bin:/usr/bin"),
    ]);
    let answer = ask(&Toolkit::new(DEVICE_TABLE, &[], env).unwrap());
    let host = answer
        .get("stated")
        .unwrap()
        .get("tools")
        .unwrap()
        .get("host_compiler")
        .unwrap();
    assert_eq!(
        host.get("path").and_then(J::str),
        Some(dir.join("my hostcc/g++").to_str().unwrap())
    );
}
