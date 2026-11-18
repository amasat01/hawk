//! Serving an embedded header tree: the OSPREY holds the bytes of the headers it compiles against (embedded at its own build);
//! a compile needs a folder. The tree is written once into a private folder named by its digest, so the same tree is the same
//! folder for every process, a changed tree is a new folder, and the digest is a key term that needs no walk of the files.
use crate::refusal::{Done, Refusal};
use rcore::sha256;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Numbers the scratch folders of this process.
static SCRATCH: AtomicU64 = AtomicU64::new(0);
/// The file that marks a served folder as whole.
const WHOLE: &str = ".whole";

/// One header tree: a name (the place `{include:<name>}` in an argv template stands for its folder) and its files as
/// relative path and bytes.
#[derive(Clone, Copy, Debug)]
pub struct HeaderTree {
    /// The name of the tree.
    pub name: &'static str,
    /// The files: path relative to the tree's folder, and content.
    pub files: &'static [(&'static str, &'static [u8])],
}

impl HeaderTree {
    /// The digest of the tree: its file names and bytes in name order.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut order: Vec<&(&str, &[u8])> = self.files.iter().collect();
        order.sort_by_key(|f| f.0);
        let mut data = Vec::new();
        for (name, bytes) in order {
            data.extend_from_slice(name.as_bytes());
            data.push(0);
            data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            data.extend_from_slice(bytes);
        }
        sha256::hex(&data)
    }

    /// The folder of the tree under `root`, written on first use. Two processes serving the same tree at once both write a
    /// private copy and one rename wins; the loser's copy is removed.
    ///
    /// # Errors
    /// Refuses with a [`Refusal`](crate::refusal::Refusal) that names what is wrong with the request or the tool.
    pub fn serve(&self, root: &Path) -> Done<PathBuf> {
        let served = root.join("headers").join(self.digest());
        if served.join(WHOLE).is_file() {
            return Ok(served);
        }
        let parent = root.join("headers");
        fs::create_dir_all(&parent).map_err(|e| Refusal::io("cannot create", &parent, &e))?;
        let scratch = parent.join(format!(
            ".tmp.{}.{}",
            std::process::id(),
            SCRATCH.fetch_add(1, Ordering::Relaxed)
        ));
        let written = self.write(&scratch).and_then(|()| {
            fs::write(scratch.join(WHOLE), "")
                .map_err(|e| Refusal::io("cannot write", &scratch, &e))
        });
        let moved = written.and_then(|()| match fs::rename(&scratch, &served) {
            Ok(()) => Ok(()),
            Err(_) if served.join(WHOLE).is_file() => Ok(()),
            Err(e) => Err(Refusal::io("cannot publish", &served, &e)),
        });
        let _ = fs::remove_dir_all(&scratch);
        moved.map(|()| served)
    }

    /// Writes every file of the tree under `dir`; a path that leaves the folder is refused.
    fn write(&self, dir: &Path) -> Done<()> {
        for (name, bytes) in self.files {
            let relative = Path::new(name);
            if relative
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
            {
                return Err(Refusal::bad(format!(
                    "header tree {}: the path {name:?} leaves its folder",
                    self.name
                )));
            }
            let path = dir.join(relative);
            if let Some(up) = path.parent() {
                fs::create_dir_all(up).map_err(|e| Refusal::io("cannot create", up, &e))?;
            }
            fs::write(&path, bytes).map_err(|e| Refusal::io("cannot write", &path, &e))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TREE: HeaderTree = HeaderTree {
        name: "aether",
        files: &[
            ("aether/a.h", b"#define A 1\n"),
            ("plugin/b.h", b"#define B 2\n"),
        ],
    };

    fn root(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("toolkit-headers-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_served_tree_holds_its_files_and_is_named_by_its_digest() {
        let r = root("serve");
        let folder = TREE.serve(&r).unwrap();
        assert!(folder.ends_with(TREE.digest()));
        assert_eq!(
            fs::read(folder.join("plugin/b.h")).unwrap(),
            b"#define B 2\n"
        );
        assert_eq!(TREE.serve(&r).unwrap(), folder);
    }

    #[test]
    fn a_changed_file_is_a_changed_digest() {
        let other = HeaderTree {
            name: "aether",
            files: &[
                ("aether/a.h", b"#define A 2\n"),
                ("plugin/b.h", b"#define B 2\n"),
            ],
        };
        assert_ne!(other.digest(), TREE.digest());
    }

    #[test]
    fn a_path_that_leaves_the_folder_is_refused() {
        let bad = HeaderTree {
            name: "evil",
            files: &[("../x.h", b"")],
        };
        assert!(bad
            .serve(&root("bad"))
            .unwrap_err()
            .message
            .contains("leaves its folder"));
    }
}
