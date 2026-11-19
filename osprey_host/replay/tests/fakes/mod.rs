//! The fakes the tool-side tests share: a store root of their own, the shipped documents beside the running program and an
//! executable script standing for a tool.

use std::path::PathBuf;

/// Puts the shipped documents of both libraries in the folder `facts` beside the running program, where the tool side reads them
/// (it has no argument for the folder), once per process.
pub fn shipped_beside_the_program() {
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
pub fn store(test: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("osprey_tool_side_{}_{test}", std::process::id()));
    shipped_beside_the_program();
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

/// Writes the executable script `body` at `path`, through a child process.
pub fn script(path: &std::path::Path, body: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    // A child writes the file: this process, whose sibling test threads fork tools, never holds a write handle to an
    // executable. A fork made while such a handle is open keeps it until its own exec, and running the script meanwhile
    // fails with "text file busy" on Linux.
    let mut writer = std::process::Command::new("sh")
        .args(["-c", "cat > \"$1\" && chmod 755 \"$1\"", "sh"])
        .arg(path)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    std::io::Write::write_all(&mut writer.stdin.take().unwrap(), body.as_bytes()).unwrap();
    assert!(
        writer.wait().unwrap().success(),
        "writing {} failed",
        path.display()
    );
}
