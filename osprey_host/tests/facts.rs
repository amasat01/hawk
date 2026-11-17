//! The facts answer: the profile and instruction-set rules, the refusals by name, and the schema check of the data files.
use aether_writers::contract::types::Caches;
use aether_writers::contract::{Contract, Json};
use osprey_host::facts::{
    answer as ask_facts, device_fusion, stated_from_json, CardSet, Probe, Stated,
};
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
        "stated machine: device_arch, device_release and driver_release are for the cuda target"
    );
    let device = stated("cuda", &line("12.6"), r#"{"isa":"arm64"}"#).unwrap_err();
    assert_eq!(
        device,
        "stated machine: only device_arch, device_release, driver_release and opt_level are for the cuda target"
    );
}

#[test]
fn the_device_card_states_a_fusion_row() {
    let row = device_fusion(&CardSet::new(&shipped(), &[]))
        .unwrap()
        .defaults;
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
    // no driver is stated, so the image is a cubin, and its compile asks for the resource figures
    let want = concat!(
        r#""device_build":{"compiler":"nvcc 12.9","ptx_isa":88,"image_format":"cubin","compile":["#,
        r#""{compiler}","-ccbin","{host_compiler}","-cubin","-std=c++20","-arch=sm_80","-O3","-Xptxas","-O3","-Xptxas","-v","#,
        r#""-I","{include:aether}","-I","{include:eagle}","{defines}","-I{include:facts}","#,
        r#""-MD","-MF","{depfile}","{source}","-o","{output}"],"flags":{"#,
        r#""ptx":["-ptx","-std=c++20","-arch=sm_80","-O3","-Xptxas","-O3","-Wno-deprecated-gpu-targets"],"#,
        r#""cubin":["-cubin","-std=c++20","-arch=sm_80","-O3","-Xptxas","-O3","-Wno-deprecated-gpu-targets"],"#,
        r#""fatbin":["-fatbin","-std=c++20","-O3","-Xptxas","-O3","-Wno-deprecated-gpu-targets","-gencode","arch=compute_80,code=[sm_80,compute_80]"]},"embed":{"compile":["{host_compiler}","#
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
        "stated machine: device_arch, device_release and driver_release are for the cuda target"
    );
}

/// The refusal of a host answer for the profile `profile` with the compiler `line` over the data in `dir`.
fn profile_refusal(dir: &Path, line: &str, profile: &str) -> String {
    let stated = Stated {
        host_profile: Some(profile.into()),
        ..roots()
    };
    let probe = machine("arm64");
    answer(dir, "host", &identity(line), &probe, &stated).unwrap_err()
}

#[test]
fn a_profile_no_card_builds_for_this_instruction_set_says_where_it_is_built() {
    let want =
        "host profile native-vector-math-nofma is not built on arm64; it is built only on x86_64";
    for line in ["g++ (GCC) 13.2.0", "Apple clang version 21.0.0"] {
        assert_eq!(
            profile_refusal(&shipped(), line, "native-vector-math-nofma"),
            want
        );
    }
}

#[test]
fn a_profile_the_other_compiler_family_builds_says_which_family() {
    // an x86 card whose non-GCC table lacks the profile (and which lets such a compiler ask for vector math)
    let only_gcc = edited(
        "x86.json",
        ",\n        \"native-vector-math-nofma\": [\"-fno-trapping-math\", \"-DAETHER_HOST_VECTOR_MATH\"]\n      }\n    },",
        "\n      }\n    },",
    )
    .replace("\"vector_math_needs_gcc\": true", "\"vector_math_needs_gcc\": false");
    let dir = changed("only_gcc", "x86.json", &only_gcc);
    let stated = Stated {
        isa: Some("x86_64".into()),
        vector_isa: Some("avx2".into()),
        host_profile: Some("native-vector-math-nofma".into()),
        ..roots()
    };
    let got = answer(
        &dir,
        "host",
        &identity("Apple clang version 21.0.0"),
        &machine("x86_64"),
        &stated,
    )
    .unwrap_err();
    assert_eq!(
        got,
        "host profile native-vector-math-nofma is not built on x86_64 with a compiler other than GCC; it is built there only with a GCC compiler"
    );
}

#[test]
fn the_shipped_text_of_each_named_document_is_its_file_and_another_name_is_refused_by_name() {
    for name in osprey_host::facts::CARD_NAMES {
        let file = std::fs::read_to_string(shipped().join(name)).unwrap();
        assert_eq!(
            osprey_host::facts::shipped_card(&shipped(), name).unwrap(),
            file
        );
    }
    let why = String::from(osprey_host::facts::shipped_card(&shipped(), "cpu.json").unwrap_err());
    assert!(why.contains("cpu.json"), "{why}");
}

/// The `device_card` member of the device facts answer made with `old` of the shipped device build card replaced by `new`, or the
/// reason the answer is refused.
fn device_card_stating(old: &str, new: &str) -> Result<Json, String> {
    let text = std::fs::read_to_string(shipped().join("cuda_build.json")).unwrap();
    assert!(text.contains(old), "the shipped card holds {old}");
    let document = Json::Str(text.replacen(old, new, 1)).to_text();
    let state =
        stated_from_json(&format!(r#"{{"cards":{{"cuda_build.json":{document}}}}}"#)).unwrap();
    let identity =
        "/usr/local/cuda/bin/nvcc || Cuda compilation tools, release 12.6, V12.6.77 || 0";
    let answer = ask_facts(
        &shipped(),
        "cuda",
        identity,
        &osprey_host::facts::probe(),
        &state,
    )
    .map_err(String::from)?;
    let machine = Contract::parse(&answer).unwrap().0;
    Ok(machine.get("device_card").expect("a device_card").clone())
}

/// Oracle: the shipped device build card states 8 for the slowest FP64 rate and two warp entries, the order 32 eigenvalue
/// solves of both scalar types.
#[test]
fn the_linalg_numbers_of_the_device_build_card_reach_the_facts_answer_as_stated() {
    let shipped_card =
        device_card_stating("\"linalg_fast_fp64\": 8", "\"linalg_fast_fp64\": 8").unwrap();
    assert_eq!(
        shipped_card.get("linalg_fast_fp64").and_then(Json::int),
        Some(8)
    );
    assert_eq!(
        shipped_card.get("linalg_warp_entries").unwrap().arr().len(),
        2
    );
    let slower =
        device_card_stating("\"linalg_fast_fp64\": 8", "\"linalg_fast_fp64\": 64").unwrap();
    assert_eq!(slower.get("linalg_fast_fp64").and_then(Json::int), Some(64));
    let narrower = device_card_stating(
        "\"n\": 32, \"scalar\": \"float64\"",
        "\"n\": 16, \"scalar\": \"float64\"",
    )
    .unwrap();
    assert_eq!(
        narrower.get("linalg_warp_entries").unwrap().arr()[1]
            .get("n")
            .and_then(Json::int),
        Some(16)
    );
}

#[test]
fn a_device_build_card_without_its_fast_fp64_rate_or_with_an_unknown_warp_scalar_is_refused_by_name(
) {
    let why = device_card_stating("  \"linalg_fast_fp64\": 8,\n", "").unwrap_err();
    assert!(why.contains("linalg_fast_fp64"), "{why}");
    let why = device_card_stating("\"scalar\": \"float64\"", "\"scalar\": \"half\"").unwrap_err();
    assert!(why.contains("half"), "{why}");
}

/// Oracle: the shipped device build card states 32 words for a row of shared memory without a bank conflict, in its `lanes`.
#[test]
fn the_bank_words_of_the_device_build_card_reach_the_facts_answer_as_stated_and_a_missing_one_is_refused(
) {
    let words = |card: Json| {
        card.get("lanes")
            .and_then(|lanes| lanes.get("bank_words"))
            .and_then(Json::int)
    };
    let kept = device_card_stating("\"bank_words\": 32", "\"bank_words\": 32").unwrap();
    assert_eq!(words(kept), Some(32));
    let narrow = device_card_stating("\"bank_words\": 32", "\"bank_words\": 16").unwrap();
    assert_eq!(words(narrow), Some(16));
    let why = device_card_stating("\"bank_words\": 32, ", "").unwrap_err();
    assert!(why.contains("bank_words"), "{why}");
}

/// Oracle: the shipped device build card states no wide row, and its default `slice_min` of 128 is four warps of 32.
#[test]
fn a_wide_row_stated_in_the_device_build_card_reaches_the_facts_answer_and_a_malformed_one_is_refused_by_name(
) {
    let row = r#"{"cc": "sm_80", "dtype": "float32", "slice_min": 64, "team_min": 64, "tile_bytes": 8192}"#;
    let kept = device_card_stating("\"wide_rows\": []", "\"wide_rows\": []").unwrap();
    assert!(kept.get("wide_rows").unwrap().arr().is_empty());
    let stated =
        device_card_stating("\"wide_rows\": []", &format!("\"wide_rows\": [{row}]")).unwrap();
    let rows = stated.get("wide_rows").unwrap().arr();
    assert_eq!(rows[0].get("tile_bytes").and_then(Json::int), Some(8192));
    let missing = row.replace(", \"team_min\": 64", "");
    let why = device_card_stating("\"wide_rows\": []", &format!("\"wide_rows\": [{missing}]"))
        .unwrap_err();
    assert!(why.contains("team_min"), "{why}");
    let half = row.replace("float32", "half");
    let why =
        device_card_stating("\"wide_rows\": []", &format!("\"wide_rows\": [{half}]")).unwrap_err();
    assert!(why.contains("half"), "{why}");
    let why = device_card_stating("\"slice_min\": 128", "\"slice_min\": 100").unwrap_err();
    assert!(why.contains("slice_min"), "{why}");
}

/// Oracle: the shipped device build card runs a dense contraction as `gemm` for both scalar types and states no row.
#[test]
fn the_contraction_and_the_rows_of_the_device_thread_box_reach_the_facts_answer_and_a_bad_one_is_refused_by_name(
) {
    let route = |card: Json, scalar: &str| {
        let boxes = card.get("thread_box").unwrap().clone();
        let routes = boxes.get("contraction").unwrap().clone();
        routes.get(scalar).and_then(Json::str).map(String::from)
    };
    let kept = device_card_stating("\"float32\": \"gemm\"", "\"float32\": \"gemm\"").unwrap();
    assert_eq!(route(kept.clone(), "float32").as_deref(), Some("gemm"));
    assert!(kept
        .get("thread_box")
        .unwrap()
        .get("rows")
        .unwrap()
        .arr()
        .is_empty());
    let direct = device_card_stating("\"float32\": \"gemm\"", "\"float32\": \"direct\"").unwrap();
    assert_eq!(route(direct, "float32").as_deref(), Some("direct"));
    let why = device_card_stating("\"float32\": \"gemm\"", "\"float32\": \"blas\"").unwrap_err();
    assert!(why.contains("contraction"), "{why}");
    let row = r#"{"cc": "sm_00", "device": "Test Card", "dtype": "float64", "contraction": "direct",
        "tile": {"kind": "BT", "SB": 128, "T": 16, "R": 1, "EB": 4, "smem": 49152}}"#;
    let stated = device_card_stating("\"rows\": []", &format!("\"rows\": [{row}]")).unwrap();
    let rows = stated
        .get("thread_box")
        .unwrap()
        .get("rows")
        .unwrap()
        .clone();
    assert_eq!(
        rows.arr()[0].get("device").and_then(Json::str),
        Some("Test Card")
    );
    let no_device = row.replace("\"device\": \"Test Card\", ", "");
    let why = device_card_stating("\"rows\": []", &format!("\"rows\": [{no_device}]")).unwrap_err();
    assert!(why.contains("device"), "{why}");
    let lanes = row.replace("\"contraction\"", "\"lanes\": 32, \"contraction\"");
    let why = device_card_stating("\"rows\": []", &format!("\"rows\": [{lanes}]")).unwrap_err();
    assert!(why.contains("lanes"), "{why}");
}

/// Oracle: the team of a stage made only of warp-per-lane eigenvalue solves is one warp, 32 threads, which the shipped device card
/// states in its `lanes` beside the two team sizes.
#[test]
fn the_eigh_team_of_the_device_build_card_reaches_the_facts_answer_as_stated_and_a_missing_one_is_refused(
) {
    let team = |card: Json| {
        card.get("lanes")
            .and_then(|lanes| lanes.get("eigh_team_threads"))
            .and_then(Json::int)
    };
    let kept =
        device_card_stating("\"eigh_team_threads\": 32", "\"eigh_team_threads\": 32").unwrap();
    assert_eq!(team(kept), Some(32));
    let wider =
        device_card_stating("\"eigh_team_threads\": 32", "\"eigh_team_threads\": 64").unwrap();
    assert_eq!(team(wider), Some(64));
    let why = device_card_stating("\"eigh_team_threads\": 32, ", "").unwrap_err();
    assert!(why.contains("eigh_team_threads"), "{why}");
}
