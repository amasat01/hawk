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
        Target::Host => osprey_cpu::table(),
        Target::Device => osprey_cuda::table(),
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

/// The facts answer of the library that serves the target named `host` or `cuda`; any other name is refused.
pub fn answer_named(
    dir: &std::path::Path,
    target: &str,
    compiler: &str,
    probe: &osprey_lowering::facts::Probe,
    stated: &osprey_lowering::facts::Stated,
) -> Result<String, osprey_lowering::facts::Failed> {
    use osprey_lowering::facts::{answer, Failed};
    match target {
        "host" => answer(dir, &osprey_cpu::Cpu, compiler, probe, stated),
        "cuda" => answer(dir, &osprey_cuda::Cuda, compiler, probe, stated),
        other => Err(Failed::Bad(format!("no facts for the target {other}"))),
    }
}

/// The text of the shipped document `name`, from the library that ships it.
pub fn shipped_card_named(
    dir: &std::path::Path,
    name: &str,
) -> Result<String, osprey_lowering::facts::Failed> {
    use osprey_lowering::facts::shipped_card;
    match osprey_cuda::Cuda.card_names().contains(&name) {
        true => shipped_card(&osprey_cuda::Cuda, dir, name),
        false => shipped_card(&osprey_cpu::Cpu, dir, name),
    }
}
