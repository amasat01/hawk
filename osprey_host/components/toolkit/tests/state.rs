//! The counters of the door's `state`: kept apart in their own test binary because they are per process.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a panic is the failure
mod common;
use common::{fake_compiler, member, one_tool_table, scratch, unit};
use toolkit::{Env, Toolkit};

#[test]
fn stats_count_hits_misses_and_tool_runs_and_reset_zeroes_them() {
    let dir = scratch("state");
    let fake = fake_compiler(&dir, "fakecc");
    let env = Env::from_pairs(&[
        ("FAKE_CXX", fake.to_str().unwrap()),
        ("PATH", "/bin:/usr/bin"),
    ]);
    let tk = Toolkit::new(&one_tool_table("FAKE_CXX"), &[], env).unwrap();
    let count = |key: &str| member(&tk.state("stats", &dir).unwrap(), key).int();
    tk.state("reset", &dir).unwrap();
    assert_eq!(
        (count("hits"), count("misses"), count("tool_runs")),
        (0, 0, 0)
    );
    tk.build(&unit("u", "int x;"), &dir.join("s")).unwrap();
    tk.build(&unit("u", "int x;"), &dir.join("s")).unwrap();
    assert_eq!(
        (count("hits"), count("misses"), count("tool_runs")),
        (1, 1, 1)
    );
    tk.state("reset", &dir).unwrap();
    assert_eq!(count("misses"), 0);
}
