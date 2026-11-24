//! What the host's facts start from: the machine the caller states for a build that is not for the machine running, the record
//! the running machine reports (its instruction set and cache sizes) and the probe that asks it.
use aether_writers::contract::types::Caches;
use aether_writers::contract::Json;
use osprey_lowering::facts::{stated_closed, stated_size, stated_text, Failed};
use std::ffi::{c_char, c_int, c_void};

/// What the running machine reports: its instruction set (as the arm and x86 cards name it) and its cache sizes.
pub struct Probe {
    /// The instruction set name: `arm64`, `x86_64`, or the platform's own name for any other.
    pub isa: String,
    /// The vector instruction set: `neon`, `sse2`, `avx2` or `avx512`; empty where the probe cannot tell.
    pub vector_isa: String,
    /// The first and second level data cache sizes, when the machine reports both.
    pub caches: Option<Caches>,
}

/// A machine the caller states, for a build that is not for the machine running: every field absent means the probed machine's
/// value (or the card's default). `headers` is required by a host answer.
#[derive(Default)]
pub struct Stated {
    /// The instruction set name.
    pub isa: Option<String>,
    /// The vector instruction set.
    pub vector_isa: Option<String>,
    /// The first and second level data cache sizes with where they came from.
    pub caches: Option<Caches>,
    /// The host profile asked for, in place of the compiler's default.
    pub host_profile: Option<String>,
    /// Where the headers of a build are: `roots` or `served`.
    pub headers: Option<String>,
    /// The optimisation level of a build (`O0` to `O3`), in place of the default.
    pub opt_level: Option<String>,
    /// The release of a GCC compiler as `major.minor`, in place of the one its identity's version line ends with.
    pub gcc_release: Option<String>,
    /// What the masked-call witness did on the compiler: `fixed`, `miscompile` or `unrun`; none counts as `unrun`.
    pub masked_call_witness: Option<String>,
}

/// The stated host machine of its JSON record (`isa`, `vector_isa`, `l1d_bytes`, `l2_bytes`, `cache_source`, `host_profile`,
/// `headers`, `opt_level`, `gcc_release`, `masked_call_witness`, each optional); a key outside these, a value of the wrong kind and a
/// cache record with a part missing are refused by name.
pub fn read_stated(record: &Json) -> Result<Stated, Failed> {
    stated_closed(
        record,
        &[
            "isa",
            "vector_isa",
            "l1d_bytes",
            "l2_bytes",
            "cache_source",
            "host_profile",
            "headers",
            "opt_level",
            "gcc_release",
            "masked_call_witness",
        ],
    )?;
    let caches = match (
        stated_size(record, "l1d_bytes")?,
        stated_size(record, "l2_bytes")?,
        stated_text(record, "cache_source")?,
    ) {
        (None, None, None) => None,
        (Some(l1d_bytes), Some(l2_bytes), Some(source)) => Some(Caches {
            l1d_bytes,
            l2_bytes,
            source,
        }),
        _ => {
            return Err(Failed::Bad(
                "stated machine: caches need l1d_bytes, l2_bytes and cache_source".into(),
            ))
        }
    };
    Ok(Stated {
        isa: stated_text(record, "isa")?,
        vector_isa: stated_text(record, "vector_isa")?,
        caches,
        host_profile: stated_text(record, "host_profile")?,
        headers: stated_text(record, "headers")?,
        opt_level: stated_text(record, "opt_level")?,
        gcc_release: stated_text(record, "gcc_release")?,
        masked_call_witness: stated_text(record, "masked_call_witness")?,
    })
}

/// The value of a macOS system setting (a number of one to eight bytes), when it reads.
#[cfg(target_os = "macos")]
fn sysctl_number(name: &str) -> Option<i64> {
    extern "C" {
        /// The system call that reads a named setting.
        fn sysctlbyname(
            name: *const c_char,
            old: *mut c_void,
            old_len: *mut usize,
            new: *mut c_void,
            new_len: usize,
        ) -> c_int;
    }
    let name = std::ffi::CString::new(name).ok()?;
    let mut bytes = [0u8; 8];
    let mut len = bytes.len();
    // SAFETY: `bytes` holds `len` bytes, and the name is a NUL-terminated string.
    let status = unsafe {
        sysctlbyname(
            name.as_ptr(),
            bytes.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0 && len > 0).then(|| i64::from_le_bytes(bytes))
}

/// The cache sizes of a macOS machine: the performance cores' where it has two kinds of core, else the plain ones.
#[cfg(target_os = "macos")]
fn probe_caches() -> Option<Caches> {
    ["hw.perflevel0.", "hw."].iter().find_map(|prefix| {
        let l1d_bytes = sysctl_number(&format!("{prefix}l1dcachesize"))?;
        let l2_bytes = sysctl_number(&format!("{prefix}l2cachesize"))?;
        (l1d_bytes > 0 && l2_bytes > 0).then(|| Caches {
            l1d_bytes,
            l2_bytes,
            source: format!("sysctl {prefix}l1dcachesize/l2cachesize"),
        })
    })
}

/// A Linux cache size field (`32K`, `1024K`, `2M`) in bytes.
#[cfg(target_os = "linux")]
fn cache_bytes(text: &str) -> Option<i64> {
    let text = text.trim().to_uppercase();
    let scale = match text.chars().last()? {
        'K' => 1024,
        'M' => 1024 * 1024,
        'G' => 1024 * 1024 * 1024,
        _ => 1,
    };
    Some(text.trim_end_matches(['K', 'M', 'G']).parse::<i64>().ok()? * scale)
}

/// The cache sizes of a Linux machine from the first processor's cache directories: the level 1 data cache and the level 2
/// cache. Written from the kernel's file layout and not run on a Linux machine.
#[cfg(target_os = "linux")]
fn probe_caches() -> Option<Caches> {
    use std::path::{Path, PathBuf};
    let root = "/sys/devices/system/cpu/cpu0/cache";
    let field = |entry: &Path, name: &str| {
        std::fs::read_to_string(entry.join(name))
            .ok()
            .map(|text| text.trim().to_string())
    };
    let mut entries: Vec<PathBuf> = std::fs::read_dir(root)
        .ok()?
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("index"))
        })
        .collect();
    entries.sort();
    let mut sizes = [None, None];
    for entry in &entries {
        let level = field(entry, "level")?.parse::<usize>().ok()?;
        let kind = field(entry, "type")?;
        let data = if level == 1 {
            kind == "Data"
        } else {
            kind != "Instruction"
        };
        if (level == 1 || level == 2) && data && sizes[level - 1].is_none() {
            sizes[level - 1] = cache_bytes(&field(entry, "size")?);
        }
    }
    let [Some(l1d_bytes), Some(l2_bytes)] = sizes else {
        return None;
    };
    (l1d_bytes > 0 && l2_bytes > 0).then(|| Caches {
        l1d_bytes,
        l2_bytes,
        source: root.into(),
    })
}

/// No cache sizes are probed on any other platform.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn probe_caches() -> Option<Caches> {
    None
}

/// Asks the running machine for its instruction set and cache sizes.
pub fn probe() -> Probe {
    let isa = match std::env::consts::ARCH {
        "aarch64" => "arm64",
        other => other,
    };
    Probe {
        isa: isa.into(),
        vector_isa: probe_vector_isa().into(),
        caches: probe_caches(),
    }
}

/// The widest vector instruction set the running machine has. The x86 branch is written from the feature names and not run on an
/// x86 machine.
fn probe_vector_isa() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    return "neon";
    #[cfg(target_arch = "x86_64")]
    return if is_x86_feature_detected!("avx512f") {
        "avx512"
    } else if is_x86_feature_detected!("avx2") {
        "avx2"
    } else {
        "sse2"
    };
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    return "";
}
