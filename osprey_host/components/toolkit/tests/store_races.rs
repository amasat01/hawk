//! The store's promises under the real build path: one tool run per unit however many ask, across threads and processes, and
//! nothing a reader trusts left by a writer that was killed.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a panic is the failure
mod common;
use common::{fake_compiler, member, one_tool_table, output_of, runs, scratch, unit};
use std::fs;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::Command;
use toolkit::{Env, Toolkit};

/// A toolkit whose compiler is the fake of `dir`, in the environment of this process plus the fake's variable. The fake is
/// written by `fake_compiler` before any process runs it: a script rewritten while another process starts it cannot be executed.
fn with_fake(dir: &Path, process_env: bool) -> Toolkit {
    let env = if process_env {
        Env::process()
    } else {
        Env::from_pairs(&[
            ("FAKE_CXX", &dir.join("fakecc").display().to_string()),
            ("PATH", "/bin:/usr/bin"),
        ])
    };
    Toolkit::new(&one_tool_table("FAKE_CXX"), &[], env).unwrap()
}

/// The body of a child process of the process tests: builds the one unit in the folders its environment names.
#[test]
fn child_builds_one_unit() {
    let Ok(dir) = std::env::var("TOOLKIT_CHILD_DIR") else {
        return;
    };
    let tk = with_fake(Path::new(&dir), true);
    let answer = tk
        .build(&unit("u", "int x;"), &Path::new(&dir).join("store"))
        .unwrap();
    assert!(output_of(&answer).is_file());
}

/// A child process of this test binary that runs `child_builds_one_unit` against `dir`.
fn child(dir: &Path) -> Command {
    let mut c = Command::new(std::env::current_exe().unwrap());
    c.args(["--exact", "child_builds_one_unit", "--test-threads=1"])
        .env("TOOLKIT_CHILD_DIR", dir)
        .env("FAKE_CXX", dir.join("fakecc"))
        .process_group(0);
    c
}

#[test]
fn a_second_build_of_one_unit_runs_no_compiler() {
    let dir = scratch("second");
    fake_compiler(&dir, "fakecc");
    let tk = with_fake(&dir, false);
    let first = tk.build(&unit("u", "int x;"), &dir.join("store")).unwrap();
    let second = tk.build(&unit("u", "int x;"), &dir.join("store")).unwrap();
    assert!(matches!(member(&first, "hit"), rcore::json::J::Bool(false)));
    assert_eq!(runs(&dir), 1);
    assert!(matches!(member(&second, "hit"), rcore::json::J::Bool(true)));
    assert_eq!(output_of(&first), output_of(&second));
    assert_eq!(fs::read_to_string(output_of(&second)).unwrap(), "built\n");
    // a changed text is another unit
    tk.build(&unit("u", "int y;"), &dir.join("store")).unwrap();
    assert_eq!(runs(&dir), 2);
}

#[test]
fn six_threads_racing_on_one_key_run_one_compiler() {
    let dir = scratch("threads");
    fs::write(dir.join("sleep"), "0.3").unwrap();
    fake_compiler(&dir, "fakecc");
    let tk = with_fake(&dir, false);
    std::thread::scope(|scope| {
        for _ in 0..6 {
            scope.spawn(|| tk.build(&unit("u", "int x;"), &dir.join("store")).unwrap());
        }
    });
    assert_eq!(runs(&dir), 1);
}

#[test]
fn three_processes_racing_on_one_key_run_one_compiler() {
    let dir = scratch("processes");
    fs::write(dir.join("sleep"), "0.5").unwrap();
    fake_compiler(&dir, "fakecc");
    let mut running: Vec<_> = (0..3).map(|_| child(&dir).spawn().unwrap()).collect();
    for c in &mut running {
        assert!(c.wait().unwrap().success());
    }
    assert_eq!(runs(&dir), 1);
}

#[test]
fn a_writer_killed_mid_compile_leaves_no_slot_and_the_next_build_completes() {
    let dir = scratch("killed");
    fs::write(dir.join("sleep"), "3").unwrap();
    fake_compiler(&dir, "fakecc");
    let mut writer = child(&dir).spawn().unwrap();
    while runs(&dir) == 0 {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let group = format!("-{}", writer.id());
    assert!(Command::new("kill")
        .args(["-9", "--", &group])
        .status()
        .unwrap()
        .success());
    writer.wait().unwrap();
    let store = dir.join("store");
    assert!(
        !files_named(&store, "record.json"),
        "a killed writer published a record"
    );
    fs::write(dir.join("sleep"), "0").unwrap();
    let tk = with_fake(&dir, false);
    let answer = tk.build(&unit("u", "int x;"), &store).unwrap();
    assert_eq!(
        (
            runs(&dir),
            fs::read_to_string(output_of(&answer)).unwrap().as_str()
        ),
        (2, "built\n")
    );
    assert!(
        !files_named(&store, ".tmp."),
        "scratch of the dead writer was kept"
    );
    let again = tk.build(&unit("u", "int x;"), &store).unwrap();
    assert!(matches!(member(&again, "hit"), rcore::json::J::Bool(true)));
    assert_eq!(runs(&dir), 2);
}

/// Whether a file or folder whose name starts with `prefix` is anywhere under `dir`.
fn files_named(dir: &Path, prefix: &str) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|e| {
        let found = e.file_name().to_string_lossy().starts_with(prefix);
        found || (e.path().is_dir() && files_named(&e.path(), prefix))
    })
}

/// The largest number of fake compiler runs that overlapped, from the start and end lines the fake appends to `dir/log`.
fn most_at_once(dir: &Path) -> usize {
    let (mut now, mut most) = (0_usize, 0_usize);
    for line in fs::read_to_string(dir.join("log")).unwrap().lines() {
        if line == "start" {
            now += 1;
            most = most.max(now);
        } else {
            now -= 1;
        }
    }
    most
}

/// Six threads each build a different unit with a compiler that takes 0.25 s; returns the most that ran at once.
fn six_builds(dir: &Path, jobs: Option<u64>) -> usize {
    let tk = with_fake(dir, false);
    common::script(
        &dir.join("fakecc"),
        &format!(
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo fakecc; exit 0; fi\necho start >> \"{d}/log\"\n\
             out=\"\"; prev=\"\"; for a in \"$@\"; do [ \"$prev\" = \"-o\" ] && out=\"$a\"; prev=\"$a\"; done\n\
             sleep 0.25\necho built > \"$out\"\necho end >> \"{d}/log\"\n",
            d = dir.display()
        ),
    );
    std::thread::scope(|scope| {
        for i in 0..6 {
            let (tk, dir) = (&tk, dir);
            scope.spawn(move || {
                let mut request = common::unit(&format!("u{i}"), &format!("int x{i};"));
                if let Some(n) = jobs {
                    request.insert_str(request.len() - 1, &format!(",\"jobs\":{n}"));
                }
                tk.build(&request, &dir.join("store")).unwrap();
            });
        }
    });
    most_at_once(dir)
}

#[test]
fn six_concurrent_builds_of_different_units_never_exceed_the_default_cap_of_two() {
    let dir = scratch("jobs-default");
    assert_eq!(six_builds(&dir, None), 2);
}

#[test]
fn the_request_sets_the_cap_and_zero_lifts_it() {
    let one = scratch("jobs-one");
    assert_eq!(six_builds(&one, Some(1)), 1);
    let three = scratch("jobs-three");
    assert_eq!(six_builds(&three, Some(3)), 3);
    let free = scratch("jobs-free");
    assert!(six_builds(&free, Some(0)) > 3);
}

#[test]
fn the_cap_holds_across_processes_too() {
    let dir = scratch("jobs-processes");
    let permits: Vec<_> = (0..2)
        .map(|_| toolkit::jobs::acquire(&dir.join("store"), 2).unwrap())
        .collect();
    let mut waiting = child(&dir);
    // the child cannot take a permit while both are held here, so it has not run the compiler yet
    fake_compiler(&dir, "fakecc");
    let mut running = waiting.spawn().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(400));
    assert_eq!(runs(&dir), 0);
    drop(permits);
    assert!(running.wait().unwrap().success());
    assert_eq!(runs(&dir), 1);
}
