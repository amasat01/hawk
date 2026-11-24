//! What the replay tools and their tests share: the folders of the two libraries' shipped data documents. The tools link both
//! libraries as plain crates and send each request to the library of its target.
use aether_writers::contract::types::Target;
use osprey_lowering::Backend;

/// The data documents of the cpu library.
pub const CPU_FACTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../osprey_cpu/facts");
/// The data documents of the cuda library.
pub const CUDA_FACTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../osprey_cuda/facts");

/// The entry table of the library that serves `target`.
pub fn table_of(target: Target) -> &'static osprey_lowering::door::OspreyV1 {
    match target {
        Target::Host => osprey_cpu_lib::table(),
        Target::Device => osprey_cuda_lib::table(),
    }
}

/// The data documents of the library that serves `target`.
pub fn facts_of(target: Target) -> &'static str {
    match target {
        Target::Host => CPU_FACTS,
        Target::Device => CUDA_FACTS,
    }
}

/// Every document a caller may replace, of both libraries.
pub const ALL_CARDS: &[&str] = &[
    "build.json",
    "arm.json",
    "x86.json",
    "cuda_build.json",
    "cuda.json",
];

/// The shipped documents of both libraries in one directory, made once per process in the temporary folder.
pub fn merged_facts() -> std::path::PathBuf {
    /// The merged directory.
    static MERGED: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    MERGED
        .get_or_init(|| {
            let dir =
                std::env::temp_dir().join(format!("osprey_replay_facts_{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            for from in [CPU_FACTS, CUDA_FACTS] {
                for file in std::fs::read_dir(from).unwrap() {
                    let file = file.unwrap().path();
                    std::fs::copy(&file, dir.join(file.file_name().unwrap())).unwrap();
                }
            }
            dir
        })
        .clone()
}

/// The facts answer of the library that serves the target named `host` or `cuda` for the stated machine record `stated` (its
/// `cards` member included); any other name is refused. The device library has no probe, so `probe` is the host's alone.
pub fn answer_named(
    dir: &std::path::Path,
    target: &str,
    compiler: &str,
    probe: &osprey_cpu_lib::host_state::Probe,
    stated: &str,
) -> Result<String, osprey_lowering::facts::Failed> {
    use osprey_lowering::facts::{answer, split_stated, Failed};
    let (record, docs) = split_stated(stated)?;
    match target {
        "host" => {
            let stated = osprey_cpu_lib::host_state::read_stated(&record)?;
            answer(dir, &osprey_cpu_lib::Cpu, compiler, &docs, probe, &stated)
        }
        "cuda" => {
            let stated = osprey_cuda_lib::device_state::read_stated(&record)?;
            answer(dir, &osprey_cuda_lib::Cuda, compiler, &docs, &(), &stated)
        }
        other => Err(Failed::Bad(format!("no facts for the target {other}"))),
    }
}

/// The facts answer of the host library for a compiler identity, the probe and the stated machine, under the library's own documents.
pub fn answer_host(
    dir: &std::path::Path,
    compiler: &str,
    probe: &osprey_cpu_lib::host_state::Probe,
    stated: &osprey_cpu_lib::host_state::Stated,
) -> Result<String, osprey_lowering::facts::Failed> {
    osprey_lowering::facts::answer(dir, &osprey_cpu_lib::Cpu, compiler, &[], probe, stated)
}

/// The stated host machine of the JSON record `text`, as a caller of the facts entry states it.
pub fn host_stated(
    text: &str,
) -> Result<osprey_cpu_lib::host_state::Stated, osprey_lowering::facts::Failed> {
    let (record, _) = osprey_lowering::facts::split_stated(text)?;
    osprey_cpu_lib::host_state::read_stated(&record)
}

/// The stated device machine of the JSON record `text`, as a caller of the facts entry states it.
pub fn device_stated(
    text: &str,
) -> Result<osprey_cuda_lib::device_state::Stated, osprey_lowering::facts::Failed> {
    let (record, _) = osprey_lowering::facts::split_stated(text)?;
    osprey_cuda_lib::device_state::read_stated(&record)
}

/// The facts answer of the device library for a compiler identity and the stated machine, under the library's own documents.
pub fn answer_device(
    dir: &std::path::Path,
    compiler: &str,
    stated: &osprey_cuda_lib::device_state::Stated,
) -> Result<String, osprey_lowering::facts::Failed> {
    osprey_lowering::facts::answer(dir, &osprey_cuda_lib::Cuda, compiler, &[], &(), stated)
}

/// The text of the shipped document `name`, from the library that ships it.
pub fn shipped_card_named(
    dir: &std::path::Path,
    name: &str,
) -> Result<String, osprey_lowering::facts::Failed> {
    use osprey_lowering::facts::shipped_card;
    match osprey_cuda_lib::Cuda.card_names().contains(&name) {
        true => shipped_card(&osprey_cuda_lib::Cuda, dir, name),
        false => shipped_card(&osprey_cpu_lib::Cpu, dir, name),
    }
}
