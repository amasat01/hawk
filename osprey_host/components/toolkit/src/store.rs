//! The content-addressed store: one slot per key under a root folder, built once however many threads or processes ask.
//!
//! A slot holds the files of one build and a record that names them. The record is written last, so a slot with a record is
//! whole; a builder works in a scratch folder inside the slot and moves each file into place, so a reader never sees half a file
//! and a builder that was killed leaves nothing a reader trusts. The first asker of a missing key takes the slot's lock file
//! (`flock`, held by the open file, so threads and processes exclude each other alike and a dead holder frees it); the others
//! wait on the lock and then find the record.
use crate::refusal::{Done, Kind, Refusal};
use crate::wire;
use rcore::json::J;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Slots found whole by an asker.
static HITS: AtomicU64 = AtomicU64::new(0);
/// Slots built by an asker.
static MISSES: AtomicU64 = AtomicU64::new(0);
/// Tool runs made for a build (probes are not counted).
static RUNS: AtomicU64 = AtomicU64::new(0);
/// Numbers the scratch folders of this process.
static SCRATCH: AtomicU64 = AtomicU64::new(0);

/// Counts one tool run of a build.
pub fn note_run() {
    RUNS.fetch_add(1, Ordering::Relaxed);
}

/// `(hits, misses, tool runs)` of this process since the last reset.
#[must_use]
pub fn stats() -> (u64, u64, u64) {
    (
        HITS.load(Ordering::Relaxed),
        MISSES.load(Ordering::Relaxed),
        RUNS.load(Ordering::Relaxed),
    )
}

/// Zeroes the counters; the slots on disk are untouched.
pub fn reset_stats() {
    for counter in [&HITS, &MISSES, &RUNS] {
        counter.store(0, Ordering::Relaxed);
    }
}

/// What a builder hands over: the files it wrote in its scratch folder (by name) and a record of the build. By the builders'
/// shared convention the first file is the unit's output and a kept source, when there is one, is the second.
#[derive(Debug)]
pub struct Built {
    /// File names inside the scratch folder.
    pub files: Vec<String>,
    /// The build's own record, kept with the slot and handed back on every hit.
    pub meta: J,
}

/// A whole slot.
#[derive(Debug)]
pub struct Entry {
    /// The slot's folder.
    pub dir: PathBuf,
    /// The published file names.
    pub files: Vec<String>,
    /// The record the builder kept.
    pub meta: J,
    /// Whether this asker found the slot instead of building it.
    pub hit: bool,
}

impl Entry {
    /// The path of the published file `name`.
    #[must_use]
    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    /// The path of the unit's output as the slot records it. A hit may hold the file of a unit built under another name (the key does
    /// not cover the name), so a consumer takes the path from here and never composes it from the name it asked with.
    ///
    /// # Errors
    /// Refuses when the slot records no file.
    pub fn output(&self) -> Done<PathBuf> {
        self.files
            .first()
            .map(|name| self.path(name))
            .ok_or_else(|| {
                Refusal::new(
                    Kind::ToolFailed,
                    format!("the slot {} records no output file", self.dir.display()),
                )
            })
    }

    /// The path of the kept source as the slot records it, when the build kept one.
    #[must_use]
    pub fn source(&self) -> Option<PathBuf> {
        self.files.get(1).map(|name| self.path(name))
    }
}

/// A store under one root folder.
#[derive(Clone, Debug)]
pub struct Store {
    /// The root folder.
    root: PathBuf,
}

impl Store {
    /// The store under `root`; the folder is made when first needed.
    #[must_use]
    pub fn new(root: &Path) -> Self {
        Store {
            root: root.to_path_buf(),
        }
    }

    /// The root folder.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The folder of the slot of `key`, refused when the key is not lower-case hex (a key is a path component).
    fn slot(&self, key: &str) -> Done<PathBuf> {
        if key.len() < 3
            || !key
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(Refusal::bad(format!(
                "store key {key:?} is not lower-case hex"
            )));
        }
        Ok(self.root.join(&key[..2]).join(key))
    }

    /// The whole slot of `key`, if there is one.
    fn find(slot: &Path) -> Option<Entry> {
        let record = wire::parse(
            "slot record",
            &fs::read_to_string(slot.join("record.json")).ok()?,
        )
        .ok()?;
        let files = record.get("files")?.strs();
        files.iter().all(|f| slot.join(f).is_file()).then(|| Entry {
            dir: slot.to_path_buf(),
            files,
            meta: record.get("meta").cloned().unwrap_or(J::Null),
            hit: true,
        })
    }

    /// The slot of `key`, built by `build` when no asker has built it. `build` receives an empty scratch folder to write its files
    /// into and states their names; the store publishes them.
    ///
    /// # Errors
    /// Refuses with a [`Refusal`](crate::refusal::Refusal) that names what is wrong with the request or the tool.
    pub fn get_or_build(&self, key: &str, build: impl FnOnce(&Path) -> Done<Built>) -> Done<Entry> {
        self.slot_built(key, build, true)
    }

    /// As [`Store::get_or_build`], for a slot that is no build (a kept probe answer): the hit and miss counters do not count it.
    pub(crate) fn keep(&self, key: &str, build: impl FnOnce(&Path) -> Done<Built>) -> Done<Entry> {
        self.slot_built(key, build, false)
    }

    /// The slot of `key`, built when no asker has built it; `counted` says whether the hit and miss counters see it.
    fn slot_built(
        &self,
        key: &str,
        build: impl FnOnce(&Path) -> Done<Built>,
        counted: bool,
    ) -> Done<Entry> {
        let count = |counter: &AtomicU64| {
            if counted {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        };
        let slot = self.slot(key)?;
        if let Some(entry) = Self::find(&slot) {
            count(&HITS);
            return Ok(entry);
        }
        let parent = slot.parent().unwrap_or(&self.root);
        fs::create_dir_all(parent).map_err(|e| Refusal::io("cannot create", parent, &e))?;
        let lock_path = slot.with_extension("lock");
        let lock =
            File::create(&lock_path).map_err(|e| Refusal::io("cannot create", &lock_path, &e))?;
        lock.lock()
            .map_err(|e| Refusal::io("cannot lock", &lock_path, &e))?;
        if let Some(entry) = Self::find(&slot) {
            count(&HITS);
            return Ok(entry);
        }
        count(&MISSES);
        publish(&slot, build)
    }
}

/// Builds into a scratch folder of `slot` and publishes: each file by rename, then the record by rename. Called with the lock held.
fn publish(slot: &Path, build: impl FnOnce(&Path) -> Done<Built>) -> Done<Entry> {
    fs::create_dir_all(slot).map_err(|e| Refusal::io("cannot create", slot, &e))?;
    clear_scratch(slot);
    let scratch = slot.join(format!(
        ".tmp.{}.{}",
        std::process::id(),
        SCRATCH.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&scratch).map_err(|e| Refusal::io("cannot create", &scratch, &e))?;
    let outcome = build(&scratch).and_then(|built| {
        for name in &built.files {
            let to = slot.join(name);
            fs::rename(scratch.join(name), &to)
                .map_err(|e| Refusal::io("cannot publish", &to, &e))?;
        }
        let record = wire::obj(vec![
            ("files", wire::strs(&built.files)),
            ("meta", built.meta.clone()),
        ]);
        let pending = scratch.join("record.json");
        fs::write(&pending, wire::render(&record))
            .map_err(|e| Refusal::io("cannot write", &pending, &e))?;
        let done = slot.join("record.json");
        fs::rename(&pending, &done).map_err(|e| Refusal::io("cannot publish", &done, &e))?;
        Ok(Entry {
            dir: slot.to_path_buf(),
            files: built.files,
            meta: built.meta,
            hit: false,
        })
    });
    let _ = fs::remove_dir_all(&scratch);
    outcome
}

/// Removes what a builder that died left in `slot`; only the lock holder calls this, so nothing removed is in use.
fn clear_scratch(slot: &Path) {
    let Ok(entries) = fs::read_dir(slot) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.file_name().to_string_lossy().starts_with(".tmp.") {
            let _ = fs::remove_dir_all(entry.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    const KEY: &str = "ab0123456789";

    fn root(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("toolkit-store-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn write_one(scratch: &Path) -> Built {
        fs::write(scratch.join("out.txt"), "body").unwrap();
        Built {
            files: vec!["out.txt".into()],
            meta: wire::s("made"),
        }
    }

    #[test]
    fn a_second_ask_for_one_key_does_not_build() {
        let store = Store::new(&root("second"));
        let builds = AtomicUsize::new(0);
        let ask = || {
            store.get_or_build(KEY, |s| {
                builds.fetch_add(1, Ordering::Relaxed);
                Ok(write_one(s))
            })
        };
        let (first, second) = (ask().unwrap(), ask().unwrap());
        assert_eq!(
            (first.hit, second.hit, builds.load(Ordering::Relaxed)),
            (false, true, 1)
        );
        assert_eq!(fs::read_to_string(second.path("out.txt")).unwrap(), "body");
        assert_eq!(second.meta.str(), Some("made"));
    }

    #[test]
    fn eight_threads_asking_one_key_build_once() {
        let store = Store::new(&root("threads"));
        let builds = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    store
                        .get_or_build(KEY, |s| {
                            builds.fetch_add(1, Ordering::Relaxed);
                            std::thread::sleep(std::time::Duration::from_millis(50));
                            Ok(write_one(s))
                        })
                        .unwrap();
                });
            }
        });
        assert_eq!(builds.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_failed_build_publishes_nothing_and_the_next_ask_builds() {
        let store = Store::new(&root("failed"));
        let failed = store.get_or_build(KEY, |s| {
            fs::write(s.join("half"), "x").unwrap();
            Err(Refusal::bad("no"))
        });
        assert!(failed.is_err());
        let slot = store.slot(KEY).unwrap();
        assert!(!slot.join("record.json").exists() && !slot.join("half").exists());
        assert!(!store.get_or_build(KEY, |s| Ok(write_one(s))).unwrap().hit);
    }

    #[test]
    fn scratch_left_by_a_dead_builder_is_cleared_and_never_read_as_a_slot() {
        let store = Store::new(&root("stale"));
        let slot = store.slot(KEY).unwrap();
        fs::create_dir_all(slot.join(".tmp.1.0")).unwrap();
        fs::write(slot.join(".tmp.1.0").join("out.txt"), "partial").unwrap();
        assert!(Store::find(&slot).is_none());
        store.get_or_build(KEY, |s| Ok(write_one(s))).unwrap();
        assert!(!slot.join(".tmp.1.0").exists());
    }

    #[test]
    fn a_key_that_is_not_hex_is_refused() {
        let store = Store::new(&root("key"));
        assert!(store.get_or_build("../x", |s| Ok(write_one(s))).is_err());
    }
}
