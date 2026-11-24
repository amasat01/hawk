//! The identity of the machine the tools run on: a digest naming the processor, the term a key needs when a result depends on the
//! instruction set. A native build is the same flag text on every processor, so the digest keeps a shared store from serving one
//! processor's binary to another. The two system queries it is made of are shared by the host facts.
use rcore::sha256;
use std::fs;
use std::process::Command;

/// The first processor block of `/proc/cpuinfo` (Linux), `None` elsewhere.
#[must_use]
pub(crate) fn cpuinfo() -> Option<String> {
    let text = fs::read_to_string("/proc/cpuinfo").ok()?;
    Some(text.split("\n\n").next().unwrap_or("").to_string())
}

/// The text `sysctl -n name` prints.
#[must_use]
pub(crate) fn sysctl(name: &str) -> Option<String> {
    let out = Command::new("sysctl").args(["-n", name]).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// A digest naming the CPU: the architecture, and the model and feature lines of `/proc/cpuinfo` or the brand string of `sysctl`.
#[must_use]
pub(crate) fn machine_identity() -> String {
    const KEYS: [&str; 10] = [
        "vendor_id",
        "model name",
        "cpu family",
        "model",
        "flags",
        "features",
        "cpu implementer",
        "cpu part",
        "isa",
        "cpu",
    ];
    let mut parts = vec![std::env::consts::ARCH.to_string()];
    if let Some(block) = cpuinfo() {
        let mut seen: Vec<&str> = vec![];
        for line in block.lines() {
            let key = line.split(':').next().unwrap_or("").trim().to_lowercase();
            if let Some(k) = KEYS.iter().find(|k| **k == key) {
                if !seen.contains(k) {
                    seen.push(k);
                    parts.push(line.trim().to_string());
                }
            }
        }
    } else {
        parts.push(sysctl("machdep.cpu.brand_string").unwrap_or_default());
    }
    sha256::hex(parts.join("\n").as_bytes())[..16].to_string()
}
