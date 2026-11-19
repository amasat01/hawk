//! The variables the toolkit reads are read on every ask of the process, not once when the toolkit was made: a change between two
//! asks moves the tool found, the asks taken from the environment and the limits, and a change back restores them.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a panic is the failure
mod common;
use common::{fake_compiler, scratch, unit};
use rcore::json::J;
use std::fs;
use std::path::Path;
use toolkit::{wire, Env, Kind, Toolkit};

/// The compiler the machine states, as the shipped host table finds it now.
fn compiler(tk: &Toolkit) -> String {
    let answer = wire::parse("a", &tk.machine("{\"key_terms\":[\"t\"]}").unwrap()).unwrap();
    let tools = answer.get("stated").unwrap().get("tools").unwrap();
    let found = tools.get("compiler").and_then(|c| c.get("path"));
    found.and_then(J::str).unwrap().to_string()
}

/// Oracle: the variables the shipped table names: `HAWK_CXX`, `HAWK_HOST_PROFILE`, `HAWK_OPT_LEVEL`, `HAWK_COMPILE_SOURCE_BUDGET`.
/// One test sets them, because the environment is the process's.
#[test]
fn a_variable_changed_between_two_asks_is_read_by_the_second_and_back_again() {
    let dir = scratch("live_env");
    for name in ["a", "b"] {
        fake_compiler(&dir.join(name), "cxx");
    }
    let at = |name: &str| dir.join(name).join("cxx").display().to_string();
    let table = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../osprey_cpu/facts/tools.json"),
    )
    .unwrap();
    let tk = Toolkit::new(&table, &[], Env::process()).unwrap();
    for name in [
        "HAWK_HOST_PROFILE",
        "HAWK_OPT_LEVEL",
        "HAWK_COMPILE_SOURCE_BUDGET",
    ] {
        std::env::remove_var(name);
    }
    // the tool found
    std::env::set_var("HAWK_CXX", at("a"));
    assert_eq!(compiler(&tk), at("a"));
    std::env::set_var("HAWK_CXX", at("b"));
    assert_eq!(compiler(&tk), at("b"));
    std::env::set_var("HAWK_CXX", at("a"));
    assert_eq!(compiler(&tk), at("a"));
    // the asks of the environment
    for (name, ask) in [
        ("HAWK_HOST_PROFILE", "host_profile"),
        ("HAWK_OPT_LEVEL", "opt_level"),
    ] {
        assert_eq!(tk.ask_from_env(ask), None);
        std::env::set_var(name, "first");
        assert_eq!(tk.ask_from_env(ask).as_deref(), Some("first"));
        std::env::set_var(name, "second");
        assert_eq!(tk.ask_from_env(ask).as_deref(), Some("second"));
        std::env::remove_var(name);
        assert_eq!(tk.ask_from_env(ask), None);
    }
    // the limits
    let build = || {
        tk.build(
            &unit("u", "int a_long_text_over_ten_bytes;"),
            &dir.join("store"),
        )
    };
    std::env::set_var("HAWK_COMPILE_SOURCE_BUDGET", "10");
    assert_eq!(build().unwrap_err().kind, Kind::OverBudget);
    std::env::remove_var("HAWK_COMPILE_SOURCE_BUDGET");
    build().unwrap();
    std::env::remove_var("HAWK_CXX");
}
