//! For an OSPREY's build script: turns header folders named by build-time environment variables into the Rust source of its
//! embedded trees, so the library carries the headers it compiles against and reads nothing from the machine at run time.
use crate::refusal::{Done, Refusal};
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

/// The files under `dir`, as paths relative to it, in name order.
fn files_under(dir: &Path, relative: &Path, out: &mut Vec<PathBuf>) -> Done<()> {
    let read = fs::read_dir(dir).map_err(|e| Refusal::io("cannot read", dir, &e))?;
    let mut entries: Vec<PathBuf> = read.flatten().map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        let name = relative.join(path.file_name().unwrap_or_default());
        if path.is_dir() {
            files_under(&path, &name, out)?;
        } else {
            out.push(name);
        }
    }
    Ok(())
}

/// Writes to `out` the source of `pub const TREES: &[toolkit::HeaderTree]`: one tree for each `(name, variable)` of `trees`, whose
/// files are those under the folder the build-time environment variable names, embedded by `include_bytes!`. Prints the
/// `cargo:rerun-if-changed` and `cargo:rerun-if-env-changed` lines the build script needs.
///
/// # Errors
/// Refuses when a variable is unset or names a folder that cannot be read, or `out` cannot be written.
pub fn write_table(trees: &[(&str, &str)], out: &Path) -> Done<()> {
    let mut source = String::from(
        "/// The header trees this library embeds.\npub const TREES: &[toolkit::HeaderTree] = &[\n",
    );
    for (name, variable) in trees {
        println!("cargo:rerun-if-env-changed={variable}");
        let folder = std::env::var(variable).map_err(|_| {
            Refusal::bad(format!(
                "the build needs ${variable}: the folder of the {name} headers it embeds"
            ))
        })?;
        println!("cargo:rerun-if-changed={folder}");
        let root = fs::canonicalize(&folder)
            .map_err(|e| Refusal::io("cannot read", Path::new(&folder), &e))?;
        let mut files = vec![];
        files_under(&root, Path::new(""), &mut files)?;
        let _ = writeln!(
            source,
            "    toolkit::HeaderTree {{ name: {name:?}, files: &["
        );
        for relative in files {
            let _ = writeln!(
                source,
                "        ({:?}, include_bytes!({:?}) as &[u8]),",
                relative.to_string_lossy(),
                root.join(&relative).to_string_lossy()
            );
        }
        source.push_str("    ] },\n");
    }
    source.push_str("];\n");
    fs::write(out, source).map_err(|e| Refusal::io("cannot write", out, &e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_folder_becomes_a_tree_with_its_files_in_name_order() {
        let dir = std::env::temp_dir().join(format!("toolkit-embed-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("inc/sub")).unwrap();
        fs::write(dir.join("inc/b.h"), "b").unwrap();
        fs::write(dir.join("inc/sub/a.h"), "a").unwrap();
        std::env::set_var("TOOLKIT_EMBED_TEST_DIR", dir.join("inc"));
        let out = dir.join("trees.rs");
        write_table(&[("demo", "TOOLKIT_EMBED_TEST_DIR")], &out).unwrap();
        let text = fs::read_to_string(out).unwrap();
        let (b, a) = (
            text.find("\"b.h\"").unwrap(),
            text.find("\"sub/a.h\"").unwrap(),
        );
        assert!(
            b < a && text.contains("name: \"demo\"") && text.contains("include_bytes!"),
            "{text}"
        );
        let refused = write_table(&[("x", "TOOLKIT_EMBED_UNSET")], &dir.join("o.rs")).unwrap_err();
        assert!(refused.message.contains("$TOOLKIT_EMBED_UNSET"));
    }
}
