//! The facts of the processor the tools run on, read from the system: architecture, the vector instruction set, the sizes of the
//! first two data caches and the digest naming the CPU. [`probe`] is what the library registers under the ask's `host` member.
use crate::identity::{cpuinfo, machine_identity, sysctl};
use rcore::json::J;
use std::fs;
use toolkit::refusal::Done;
use toolkit::{wire, ProbeAnswer, ProbeInput};

/// The name the host probe is registered under: the ask's member that calls for it and the member of the machine answer it states.
pub(crate) const SECTION: &str = "host";

/// The sysfs folder of CPU 0's caches (Linux).
const SYSFS_CACHES: &str = "/sys/devices/system/cpu/cpu0/cache";

/// The host record of the machine answer; it blocks nothing.
///
/// # Errors
/// Never refuses; the type is the probe's.
pub(crate) fn probe(_input: &ProbeInput<'_>) -> Done<ProbeAnswer> {
    Ok(ProbeAnswer {
        facts: facts(),
        blocked: None,
    })
}

/// The stated host record: `os`, `arch`, `target_identity`, `vector_isa` and, when the system tells them,
/// `l1d_cache_bytes`, `l2_cache_bytes` and `cache_source`.
fn facts() -> J {
    let mut pairs = vec![
        ("os", wire::s(std::env::consts::OS)),
        ("arch", wire::s(std::env::consts::ARCH)),
        ("target_identity", wire::s(&machine_identity())),
        ("vector_isa", wire::s(vector_isa())),
    ];
    if let Some((l1d, l2, source)) = caches() {
        pairs.extend([
            ("l1d_cache_bytes", wire::n(l1d)),
            ("l2_cache_bytes", wire::n(l2)),
            ("cache_source", wire::s(&source)),
        ]);
    }
    wire::obj(pairs)
}

/// The vector instruction set of the host, the key a card of host flags is read by.
fn vector_isa() -> &'static str {
    if matches!(std::env::consts::ARCH, "aarch64" | "arm64") {
        return "neon";
    }
    let flags = cpuinfo()
        .and_then(|block| {
            block
                .lines()
                .find(|l| l.starts_with("flags"))
                .map(str::to_string)
        })
        .unwrap_or_default();
    let words: Vec<&str> = flags.split_whitespace().collect();
    if words.contains(&"avx512f") {
        "avx512"
    } else if words.contains(&"avx2") {
        "avx2"
    } else {
        "sse2"
    }
}

/// CPU 0's `(l1 data, l2, source)` in bytes: the performance cores' keys of `sysctl` where the machine has them, else the plain
/// ones, else Linux's sysfs.
fn caches() -> Option<(u64, u64, String)> {
    for prefix in ["hw.perflevel0.", "hw."] {
        let size = |key: &str| {
            sysctl(&format!("{prefix}{key}"))
                .and_then(|t| t.parse::<u64>().ok())
                .filter(|n| *n > 0)
        };
        if let (Some(l1d), Some(l2)) = (size("l1dcachesize"), size("l2cachesize")) {
            return Some((l1d, l2, format!("sysctl {prefix}l1dcachesize/l2cachesize")));
        }
    }
    sysfs_caches()
}

/// A sysfs cache size (`32K`, `1024K`, `2M`) in bytes.
fn cache_bytes(text: &str) -> Option<u64> {
    let text = text.trim().to_uppercase();
    let scale = match text.chars().last()? {
        'K' => 1 << 10,
        'M' => 1 << 20,
        'G' => 1 << 30,
        _ => 1,
    };
    Some(text.trim_end_matches(['K', 'M', 'G']).parse::<u64>().ok()? * scale)
}

/// CPU 0's first-level data and second-level caches off sysfs.
fn sysfs_caches() -> Option<(u64, u64, String)> {
    let (mut l1d, mut l2) = (None, None);
    let mut dirs: Vec<_> = fs::read_dir(SYSFS_CACHES)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .collect();
    dirs.sort();
    for dir in dirs.iter().filter(|d| {
        d.file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with("index"))
    }) {
        let field = |name: &str| {
            fs::read_to_string(dir.join(name))
                .ok()
                .map(|t| t.trim().to_string())
        };
        let (level, kind) = (field("level")?, field("type")?);
        let size = cache_bytes(&field("size")?);
        match level.as_str() {
            "1" if kind == "Data" => l1d = l1d.or(size),
            "2" if kind != "Instruction" => l2 = l2.or(size),
            _ => {}
        }
    }
    Some((l1d?, l2?, SYSFS_CACHES.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_sizes_read_in_their_units() {
        assert_eq!(
            (cache_bytes("32K"), cache_bytes("2M"), cache_bytes("512")),
            (Some(32768), Some(2 << 20), Some(512))
        );
        assert_eq!(cache_bytes("x"), None);
    }

    #[test]
    fn the_host_record_names_the_machine() {
        let j = facts();
        assert_eq!(j.get("arch").and_then(J::str), Some(std::env::consts::ARCH));
        assert_eq!(
            j.get("target_identity").and_then(J::str).map(str::len),
            Some(16)
        );
    }
}
