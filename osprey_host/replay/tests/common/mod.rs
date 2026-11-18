//! Helpers the test files share.

/// Says that the running test checked nothing: one line `skipped: <test>: <reason>`, which ends with `[<variable>]` when the
/// reason is a variable of the environment the test reads, so the gate can tell a skip it must accept (the variable is unset)
/// from one it must refuse (the variable is set and the test was still skipped). The test's name is its thread's name.
pub fn skipped(variable: Option<&str>, reason: &str) {
    let thread = std::thread::current();
    let test = thread.name().unwrap_or("?");
    match variable {
        Some(name) => println!("skipped: {test}: {reason} [{name}]"),
        None => println!("skipped: {test}: {reason}"),
    }
}
