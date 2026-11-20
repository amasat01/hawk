//! The host library's own tool table as shipped (`osprey_cpu/facts/tools.json`): the order in which the host compiler is found, and
//! the machine limits (address space, source size, time) the table reads from the `HAWK_COMPILE_*` variables.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a panic is the failure
mod common;
use common::{fake_compiler, scratch, unit};
use rcore::json::J;
use std::fs;
use std::path::{Path, PathBuf};
use toolkit::{wire, Env, Kind, Toolkit};

/// The shipped table of the host library.
fn shipped() -> String {
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../osprey_cpu/facts/tools.json");
    fs::read_to_string(file).unwrap()
}

/// The machine answer of the shipped table under exactly the variables `vars`.
fn machine(vars: &[(&str, &str)]) -> J {
    let tk = Toolkit::new(&shipped(), &[], Env::from_pairs(vars)).unwrap();
    wire::parse("answer", &tk.machine("{\"key_terms\":[\"t\"]}").unwrap()).unwrap()
}

/// The path of the compiler the answer states.
fn compiler_of(answer: &J) -> String {
    let tools = answer.get("stated").unwrap().get("tools").unwrap();
    tools
        .get("compiler")
        .and_then(|c| c.get("path"))
        .and_then(J::str)
        .unwrap()
        .to_string()
}

/// A folder holding a fake compiler of each of `names`, and its path.
fn folder_of(test: &str, names: &[&str]) -> PathBuf {
    let dir = scratch(test);
    for name in names {
        fake_compiler(&dir.join("bin"), name);
    }
    dir.join("bin")
}

/// Oracle: the order the table states: `$HAWK_CXX`, then the first word of `$CXX`, then `g++` on `PATH`.
#[test]
fn the_host_compiler_is_hawk_cxx_then_cxx_then_gxx_on_path() {
    let bin = folder_of("order", &["g++", "other-cxx", "hawk-cxx"]);
    let at = |name: &str| bin.join(name).display().to_string();
    let path = bin.display().to_string();
    let hawk = at("hawk-cxx");
    let other = at("other-cxx");
    let both = [
        ("PATH", path.as_str()),
        ("CXX", &other),
        ("HAWK_CXX", &hawk),
    ];
    assert_eq!(compiler_of(&machine(&both)), hawk);
    assert_eq!(compiler_of(&machine(&both[..2])), other);
    assert_eq!(compiler_of(&machine(&both[..1])), at("g++"));
}

#[test]
fn a_cxx_command_line_is_found_by_its_first_word_without_g_plus_plus_on_path() {
    let bin = folder_of("first_word", &["x86_64-conda-linux-gnu-g++"]);
    let path = bin.display().to_string();
    let cxx = "x86_64-conda-linux-gnu-g++ -O2";
    let answer = machine(&[("PATH", &path), ("CXX", cxx)]);
    assert_eq!(
        compiler_of(&answer),
        bin.join("x86_64-conda-linux-gnu-g++").display().to_string()
    );
    let absolute = bin.join("x86_64-conda-linux-gnu-g++").display().to_string();
    assert_eq!(compiler_of(&machine(&[("CXX", &absolute)])), absolute);
}

#[test]
fn a_cxx_that_does_not_resolve_falls_through_to_gxx_on_path() {
    let bin = folder_of("falls", &["g++"]);
    let path = bin.display().to_string();
    let answer = machine(&[("PATH", &path), ("CXX", "no-such-compiler-xyz --flag")]);
    assert_eq!(compiler_of(&answer), bin.join("g++").display().to_string());
}

#[test]
fn the_absence_of_every_candidate_names_each_of_them_and_the_way_out() {
    let empty = scratch("absent").display().to_string();
    let answer = machine(&[("PATH", &empty), ("CXX", "no-such-compiler-xyz --flag")]);
    let why = answer.get("tools_absent").and_then(J::str).unwrap();
    for needle in [
        "HAWK_CXX",
        "no-such-compiler-xyz",
        "g++ on PATH",
        "set HAWK_CXX to a C++23 compiler, or CXX, or put g++ on PATH",
    ] {
        assert!(why.contains(needle), "{needle:?} missing from {why}");
    }
}

/// A toolkit over the shipped table whose compiler is a fake that writes `built`, under the extra variables `vars`.
fn limited(dir: &Path, vars: &[(&str, &str)]) -> Toolkit {
    let fake = dir.join("fakecc").display().to_string();
    let mut pairs = vec![("HAWK_CXX", fake.as_str()), ("PATH", "/bin:/usr/bin")];
    pairs.extend_from_slice(vars);
    Toolkit::new(&shipped(), &[], Env::from_pairs(&pairs)).unwrap()
}

/// Oracle: the variables and the refusal texts the table and the toolkit state.
#[test]
fn the_source_budget_is_read_from_its_variable_and_zero_lifts_it() {
    let dir = scratch("budget_env");
    fake_compiler(&dir, "fakecc");
    let long = "int a_long_text_over_ten_bytes;";
    let over = limited(&dir, &[("HAWK_COMPILE_SOURCE_BUDGET", "10")])
        .build(&unit("u", long), &dir.join("s"))
        .unwrap_err();
    assert_eq!(over.kind, Kind::OverBudget);
    assert!(
        over.message.contains("over the compile budget of 10 bytes"),
        "{}",
        over.message
    );
    let bad = limited(&dir, &[("HAWK_COMPILE_SOURCE_BUDGET", "lots")])
        .build(&unit("u", "int x;"), &dir.join("s"))
        .unwrap_err();
    assert!(
        bad.message.contains(
            "$HAWK_COMPILE_SOURCE_BUDGET=\"lots\" is not a non-negative integer (0 disables it)"
        ),
        "{}",
        bad.message
    );
    let lifted = limited(&dir, &[("HAWK_COMPILE_SOURCE_BUDGET", "0")]);
    lifted.build(&unit("u", long), &dir.join("s")).unwrap();
}

#[test]
fn a_compiler_that_outlives_its_timeout_variable_is_stopped() {
    let dir = scratch("timeout_env");
    fs::write(dir.join("sleep"), "20").unwrap();
    fake_compiler(&dir, "fakecc");
    let stopped = limited(&dir, &[("HAWK_COMPILE_TIMEOUT", "1")])
        .build(&unit("u", "int x;"), &dir.join("s"))
        .unwrap_err();
    assert_eq!(stopped.kind, Kind::TimedOut);
}

#[test]
fn a_bad_address_cap_is_refused_naming_its_variable_and_zero_lifts_the_cap() {
    let dir = scratch("cap_env");
    fake_compiler(&dir, "fakecc");
    for raw in ["not-a-number", "-1"] {
        let bad = limited(&dir, &[("HAWK_COMPILE_ADDRESS_CAP", raw)])
            .build(&unit("u", "int x;"), &dir.join("s"))
            .unwrap_err();
        assert!(
            bad.message.contains(&format!(
                "$HAWK_COMPILE_ADDRESS_CAP={raw:?} is not a non-negative integer"
            )),
            "{}",
            bad.message
        );
    }
    let free = limited(&dir, &[("HAWK_COMPILE_ADDRESS_CAP", "0")]);
    free.build(&unit("u", "int x;"), &dir.join("s")).unwrap();
}

/// Oracle: a compiler that allocates 600 MB and, when the allocation fails, says what a C++ runtime says. The same compiler succeeds
/// with the cap lifted, so a cap that stopped applying fails the first half instead of passing it.
#[cfg(target_os = "linux")]
#[test]
fn a_compiler_under_the_address_cap_cannot_allocate_past_it() {
    use common::script;
    let dir = scratch("cap_bites");
    script(
        &dir.join("fakecc"),
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo hog; exit 0; fi\nout=\"\"; prev=\"\"\n\
         for a in \"$@\"; do [ \"$prev\" = \"-o\" ] && out=\"$a\"; prev=\"$a\"; done\n\
         awk 'BEGIN { s = \"xxxxxxxx\"; while (length(s) < 600000000) s = s s }' 2>/dev/null \
         || { echo 'terminate called: std::bad_alloc' >&2; exit 1; }\n\
         echo built > \"$out\"\n",
    );
    let capped = limited(&dir, &[("HAWK_COMPILE_ADDRESS_CAP", "268435456")])
        .build(&unit("u", "int x;"), &dir.join("s"))
        .unwrap_err();
    assert_eq!(capped.kind, Kind::OutOfMemory, "{}", capped.message);
    assert!(
        capped.message.contains(
            "268435456-byte address-space cap ($HAWK_COMPILE_ADDRESS_CAP, 0 disables it)"
        ),
        "{}",
        capped.message
    );
    limited(&dir, &[("HAWK_COMPILE_ADDRESS_CAP", "0")])
        .build(&unit("u", "int x;"), &dir.join("s"))
        .unwrap();
}

/// Oracle: the old guards' refusals each named the limit and the variable that changes it.
#[test]
fn the_budget_and_timeout_refusals_name_the_variable_that_changes_them() {
    let dir = scratch("limit_names");
    fs::write(dir.join("sleep"), "20").unwrap();
    fake_compiler(&dir, "fakecc");
    let over = limited(&dir, &[("HAWK_COMPILE_SOURCE_BUDGET", "10")])
        .build(
            &unit("u", "int a_long_text_over_ten_bytes;"),
            &dir.join("s"),
        )
        .unwrap_err();
    assert!(
        over.message.contains("HAWK_COMPILE_SOURCE_BUDGET"),
        "{}",
        over.message
    );
    let late = limited(&dir, &[("HAWK_COMPILE_TIMEOUT", "1")])
        .build(&unit("u", "int x;"), &dir.join("s"))
        .unwrap_err();
    assert!(
        late.message.contains("HAWK_COMPILE_TIMEOUT"),
        "{}",
        late.message
    );
}

/// Oracle: the command and the compiler's own complaint, then the folder the compile ran in, then the `fix` text of the shipped table
/// (read from the table here, so a change of its wording is not a failure and a refusal that lost it is).
#[test]
fn a_failed_compile_ends_with_the_folder_it_ran_in_and_the_way_out_the_tool_table_states() {
    let dir = scratch("failed_compile");
    common::script(
        &dir.join("fakecc"),
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo \"fakecc 1.0\"; exit 0; fi\necho 'error: boom' >&2\nexit 1\n",
    );
    let refused = limited(&dir, &[])
        .build(&unit("u", "int x;"), &dir.join("s"))
        .unwrap_err();
    assert_eq!(refused.kind, Kind::ToolFailed);
    let table = wire::parse("table", &shipped()).unwrap();
    let fix = table
        .get("tools")
        .and_then(|t| t.get("compiler"))
        .and_then(|c| c.get("fix"))
        .and_then(J::str)
        .unwrap();
    let message = &refused.message;
    assert!(
        message.contains("$ ") && message.contains("error: boom"),
        "{message}"
    );
    let tail = message.rsplit_once("ran in ").map_or("", |(_, tail)| tail);
    assert!(tail.ends_with(fix) && tail.contains("; "), "{message}");
}

#[test]
fn a_host_table_runs_here_whether_or_not_its_tools_are_found() {
    let empty = scratch("runs_here_absent").display().to_string();
    let answer = machine(&[("PATH", &empty), ("CXX", "no-such-compiler-xyz --flag")]);
    assert!(matches!(answer.get("runs_here"), Some(J::Bool(true))));
    assert_eq!(answer.get("why").and_then(J::str), Some(""));
}
