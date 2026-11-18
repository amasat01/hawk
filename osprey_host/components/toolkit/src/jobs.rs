//! The cap on tool processes at once: at most N under one store root, across threads and processes. A permit is a held `flock` on
//! one of N slot files under the root; a process that dies frees its permit with it.
use crate::refusal::{Done, Refusal};
use std::fs::{self, File, TryLockError};
use std::path::Path;
use std::thread;
use std::time::Duration;

/// The default number of tool processes at once.
pub const DEFAULT_JOBS: u64 = 2;

/// How long to wait before looking at the slots again.
const POLL: Duration = Duration::from_millis(5);

/// A held permit; the slot is free again when it is dropped.
#[derive(Debug)]
pub struct Permit {
    /// The slot file, locked while this is alive.
    _slot: Option<File>,
}

/// Waits for one of `limit` slots under `root` (0: no cap, no wait) and holds it.
///
/// # Errors
/// Refuses when the slot folder or a slot file cannot be created.
pub fn acquire(root: &Path, limit: u64) -> Done<Permit> {
    if limit == 0 {
        return Ok(Permit { _slot: None });
    }
    let dir = root.join("jobs");
    fs::create_dir_all(&dir).map_err(|e| Refusal::io("cannot create", &dir, &e))?;
    let slots = (0..limit)
        .map(|i| {
            let path = dir.join(format!("slot-{i}.lock"));
            File::create(&path).map_err(|e| Refusal::io("cannot create", &path, &e))
        })
        .collect::<Done<Vec<_>>>()?;
    loop {
        for slot in &slots {
            match slot.try_lock() {
                Ok(()) => {
                    let held = slot
                        .try_clone()
                        .map_err(|e| Refusal::io("cannot keep", &dir, &e))?;
                    return Ok(Permit { _slot: Some(held) });
                }
                Err(TryLockError::WouldBlock) => {}
                Err(TryLockError::Error(e)) => return Err(Refusal::io("cannot lock", &dir, &e)),
            }
        }
        thread::sleep(POLL);
    }
}
