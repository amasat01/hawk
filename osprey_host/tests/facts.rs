//! The facts answer: the profile and instruction-set rules, the refusals by name, and the schema check of the data files.
use aether_writers::contract::types::Caches;
use osprey_host::facts::{answer as ask_facts, device_fusion, stated_from_json, Probe, Stated};
use std::path::{Path, PathBuf};

/// The data files that ship with the library.
fn shipped() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("facts")
}

/// A compiler identity whose version line is `line`.
fn identity(line: &str) -> String {
    format!("/usr/bin/c++ || {line} || 0")
}

/// The version line of a device compiler of the release `release`.
fn line(release: &str) -> String {
    format!("Cuda compilation tools, release {release}, V{release}.77")
}

/// A device compiler identity of the release `release`.
fn nvcc(release: &str) -> String {
    format!(
        "/usr/local/cuda/bin/nvcc || Cuda compilation tools, release {release}, V{release}.77 || 0"
    )
}

/// A probe of a machine of the instruction set `isa` with no cache sizes.
fn machine(isa: &str) -> Probe {
    Probe {
        isa: isa.into(),
        vector_isa: if isa == "arm64" { "neon" } else { "avx2" }.into(),
        caches: None,
    }
}

/// The facts answer, a failure as its reason.
fn answer(
    dir: &Path,
    target: &str,
    compiler: &str,
    probe: &Probe,
    stated: &Stated,
) -> Result<String, String> {
    let text = ask_facts(dir, target, compiler, probe, stated).map_err(String::from)?;
    // the build commands are tested in host_build.rs; the machine record is what these tests read
    Ok(match text.find(r#","host_build":"#) {
        Some(at) => format!("{}}}", &text[..at]),
        None => text,
    })
}

/// A stated machine whose headers are in roots, which a host question must state.
fn roots() -> Stated {
    Stated {
        headers: Some("roots".into()),
        ..Stated::default()
    }
}

/// The answer for the host target, or the refusal.
fn host(isa: &str, line: &str) -> Result<String, String> {
    answer(&shipped(), "host", &identity(line), &machine(isa), &roots())
}

#[test]
fn arm_answers_follow_the_compiler() {
    let weights = r#""select_weight":{"float64":1,"float32":1}"#;
    let clang = format!(
        r#"{{"isa":"Arm64","vector_isa":"neon","exact_libm":true,"routed_pow":false,{weights}}}"#
    );
    assert_eq!(host("arm64", "Apple clang version 21.0.0"), Ok(clang));
    let gcc = format!(
        r#"{{"isa":"Arm64","vector_isa":"neon","exact_libm":true,"routed_pow":true,{weights}}}"#
    );
    assert_eq!(host("arm64", "g++ (Ubuntu 11.4.0) 11.4.0"), Ok(gcc));
}

#[test]
fn x86_answers_follow_the_compiler() {
    let gcc = host("x86_64", "g++ (Ubuntu 11.4.0) 11.4.0").unwrap();
    assert!(
        gcc.contains(r#""isa":"X86_64","vector_isa":"avx2","exact_libm":false,"routed_pow":false"#),
        "{gcc}"
    );
    let clang = host("x86_64", "Ubuntu clang version 14.0.0").unwrap();
    assert!(
        clang
            .contains(r#""isa":"X86_64","vector_isa":"avx2","exact_libm":true,"routed_pow":false"#),
        "{clang}"
    );
}

#[test]
fn caches_are_the_probes() {
    let caches = Caches {
        l1d_bytes: 65536,
        l2_bytes: 4194304,
        source: "probe".into(),
    };
    let probe = Probe {
        isa: "arm64".into(),
        vector_isa: "neon".into(),
        caches: Some(caches),
    };
    let text = answer(
        &shipped(),
        "host",
        &identity("clang version 1"),
        &probe,
        &roots(),
    )
    .unwrap();
    let want = r#""caches":{"l1d_bytes":65536,"l2_bytes":4194304,"source":"probe"}"#;
    assert!(text.contains(want), "{text}");
}

#[test]
fn a_device_has_no_instruction_set() {
    let text = answer(
        &shipped(),
        "cuda",
        &nvcc("12.6"),
        &machine("arm64"),
        &Stated::default(),
    )
    .unwrap();
    assert!(
        text.starts_with(r#"{"exact_libm":true,"routed_pow":false,"select_weight""#),
        "{text}"
    );
}

#[test]
fn refusals_name_their_cause() {
    let refused = |why: Result<String, String>| why.unwrap_err();
    assert_eq!(
        refused(host("riscv64", "gcc")),
        "no card for the instruction set riscv64"
    );
    let other = answer(
        &shipped(),
        "fast",
        &identity("gcc"),
        &machine("arm64"),
        &Stated::default(),
    );
    assert_eq!(refused(other), "no facts for the target fast");
    let bad = refused(answer(
        &shipped(),
        "host",
        "gcc",
        &machine("arm64"),
        &Stated::default(),
    ));
    assert!(bad.starts_with("the compiler identity is not"), "{bad}");
}

/// A copy of the data files in a folder of its own, with `file` replaced by `text`.
fn changed(name: &str, file: &str, text: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("osprey_facts_{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(shipped()).unwrap() {
        let path = entry.unwrap().path();
        std::fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
    }
    std::fs::write(dir.join(file), text).unwrap();
    dir
}

/// The shipped file `file` with `from` replaced by `to`.
fn edited(file: &str, from: &str, to: &str) -> String {
    let text = std::fs::read_to_string(shipped().join(file)).unwrap();
    assert!(text.contains(from), "{file} has no {from}");
    text.replace(from, to)
}

#[test]
fn a_card_that_breaks_its_schema_is_refused_by_name() {
    let probe = machine("arm64");
    let ask = |dir: &Path| {
        answer(dir, "host", &identity("clang"), &probe, &Stated::default()).unwrap_err()
    };
    let extra = changed(
        "extra",
        "cpu.json",
        r#"{"profiles":[],"exact_profiles":[],"weird":1}"#,
    );
    assert_eq!(ask(&extra), "cpu.json has the unknown key weird");
    let missing = changed("missing", "arm.json", r#"{"isa":"arm64"}"#);
    assert_eq!(ask(&missing), "arm.json lacks the key default_profile");
    let flag = edited("arm.json", "\"libm_exact\": true", "\"libm_exact\": 1");
    assert_eq!(
        ask(&changed("flag", "arm.json", &flag)),
        "arm.json: libm_exact is not a flag"
    );
    let name = edited(
        "arm.json",
        "\"gcc\": \"native-vector-math\"",
        "\"gcc\": \"fastest\"",
    );
    let unknown = changed("name", "arm.json", &name);
    assert_eq!(
        ask(&unknown),
        "arm.json: default profile fastest is not a profile"
    );
    let zero = edited("cpu.json", "\"float64\": 1.0", "\"float64\": 0");
    let weight = changed("weight", "cpu.json", &zero);
    assert_eq!(
        ask(&weight),
        "cpu.json: select_weight float64 is not positive"
    );
}

/// The answer for the stated machine `record`, probed as an Arm machine, for the target.
fn stated(target: &str, line: &str, record: &str) -> Result<String, String> {
    let stated = stated_from_json(record)?;
    answer(
        &shipped(),
        target,
        &identity(line),
        &machine("arm64"),
        &stated,
    )
}

#[test]
fn a_stated_machine_replaces_the_probed_one() {
    let x86 = stated(
        "host",
        "g++ 11",
        r#"{"headers":"roots","isa":"x86_64","vector_isa":"avx2"}"#,
    )
    .unwrap();
    assert!(
        x86.contains(r#""isa":"X86_64","vector_isa":"avx2","exact_libm":false"#),
        "{x86}"
    );
    let portable = stated(
        "host",
        "g++ 11",
        r#"{"headers":"served","isa":"x86_64","vector_isa":"avx2","host_profile":"portable"}"#,
    )
    .unwrap();
    assert!(
        portable.contains(r#""isa":"X86_64","vector_isa":"avx2","exact_libm":true"#),
        "{portable}"
    );
    let caches =
        r#"{"headers":"roots","l1d_bytes":32768,"l2_bytes":1048576,"cache_source":"stated"}"#;
    let text = stated("host", "clang", caches).unwrap();
    assert!(text.contains(r#""caches":{"l1d_bytes":32768,"l2_bytes":1048576,"source":"stated"}"#));
    let device = stated("cuda", &line("12.6"), r#"{"device_arch":"sm_61"}"#).unwrap();
    assert!(
        device.contains(r#""select_weight":{"float64":4,"float32":1}"#),
        "{device}"
    );
    assert!(device.contains(r#""device_arch":"sm_61""#), "{device}");
    let unstated = stated("cuda", &line("12.6"), "{}").unwrap();
    assert!(unstated.contains(r#""device_arch":"sm_61""#), "{unstated}");
    assert!(unstated.contains(r#""select_weight":{"float64":1,"float32":1}"#));
    let other = stated("cuda", &line("12.6"), r#"{"device_arch":"sm_80"}"#).unwrap();
    assert!(
        other.contains(r#""select_weight":{"float64":1,"float32":1}"#),
        "{other}"
    );
}

#[test]
fn a_stated_machine_is_refused_by_name() {
    let why = |line: &str, record: &str| stated("host", line, record).unwrap_err();
    assert_eq!(
        why("gcc", r#"{"speed":1}"#),
        "stated machine: unknown key speed"
    );
    assert_eq!(
        why("gcc", r#"{"isa":3}"#),
        "stated machine: isa is not a text"
    );
    let half = r#"{"l1d_bytes":1}"#;
    assert_eq!(
        why("gcc", half),
        "stated machine: caches need l1d_bytes, l2_bytes and cache_source"
    );
    assert_eq!(
        why("gcc", r#"{"host_profile":"fast"}"#),
        "unknown host profile fast"
    );
    let vector = r#"{"isa":"x86_64","vector_isa":"avx2","host_profile":"native-vector-math"}"#;
    assert_eq!(
        why("clang", vector),
        "host profile native-vector-math needs a GCC compiler"
    );
    assert_eq!(
        why("gcc", r#"{"device_arch":"sm_61"}"#),
        "stated machine: device_arch and device_release are for the cuda target"
    );
    let device = stated("cuda", &line("12.6"), r#"{"isa":"arm64"}"#).unwrap_err();
    assert_eq!(
        device,
        "stated machine: only device_arch and device_release are for the cuda target"
    );
}

#[test]
fn the_device_card_states_a_fusion_row() {
    let row = device_fusion(&shipped()).unwrap();
    assert_eq!((row.regs_base, row.regs_file), (20, 255));
    assert_eq!(row.launch_us, Some(2.7));
}

#[test]
fn a_device_answer_states_the_compile_command_as_data() {
    let stated = stated_from_json(r#"{"device_arch":"sm_80"}"#).unwrap();
    let text = answer(
        &shipped(),
        "cuda",
        &nvcc("12.9"),
        &machine("arm64"),
        &stated,
    )
    .unwrap();
    let want = concat!(
        r#""device_build":{"compiler":"nvcc 12.9","ptx_isa":88,"flags":{"#,
        r#""ptx":["-ptx","-std=c++20","-arch=sm_80","-O3","-Xptxas","-O3","-Wno-deprecated-gpu-targets"],"#,
        r#""cubin":["-cubin","-std=c++20","-arch=sm_80","-O3","-Xptxas","-O3","-Wno-deprecated-gpu-targets"],"#,
        r#""fatbin":["-fatbin","-std=c++20","-O3","-Xptxas","-O3","-Wno-deprecated-gpu-targets","-gencode","arch=compute_80,code=[sm_80,compute_80]"]},"embed":{"compile":["#
    );
    // the embed recipe follows the flags; its own test is in set_text.rs
    assert!(text.contains(want), "{text}");
}

#[test]
fn an_unknown_release_is_refused_naming_the_release_and_the_file() {
    let why = stated("cuda", &line("13.0"), "{}").unwrap_err();
    assert!(why.starts_with("facts"), "{why}");
    assert!(
        why.contains("13.0") && why.contains("cuda_image.json"),
        "{why}"
    );
}

/// The first version line of the device compiler driver names no release (the release is on a later line of its output), so a
/// build under it states the release: the answer is then the one of a line that names the same release, and without the stated
/// release the line is refused by name.
#[test]
fn a_device_compiler_line_without_a_release_takes_the_stated_one() {
    let driver = "nvcc: NVIDIA (R) Cuda compiler driver";
    let said = stated("cuda", driver, r#"{"device_release":"12.6"}"#).unwrap();
    assert_eq!(said, stated("cuda", &line("12.6"), "{}").unwrap());
    // the stated release wins over the one a line names
    let other = stated("cuda", &line("12.9"), r#"{"device_release":"12.6"}"#).unwrap();
    assert_eq!(other, said);
    assert_eq!(
        stated("cuda", driver, "{}").unwrap_err(),
        format!("facts: the device compiler line {driver:?} names no tool or no release, and no device_release is stated")
    );
    for bad in ["12", "12.x", "12.6.77", ""] {
        let record = format!(r#"{{"device_release":"{bad}"}}"#);
        assert_eq!(
            stated("cuda", driver, &record).unwrap_err(),
            format!("stated machine: device_release {bad:?} is not major.minor")
        );
    }
    assert_eq!(
        stated("host", "clang", r#"{"device_release":"12.6"}"#).unwrap_err(),
        "stated machine: device_arch and device_release are for the cuda target"
    );
}
