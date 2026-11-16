//! The set text: one C++ text for the units of one or several answers built into one library. The answers come from the recorded calls
//! of `tests/data/meta_calls.jsonl`, lowered by `kreplay --emit-meta`, once asking for the texts of a single kernel (the exports block
//! in the first text) and once for the texts of a set (no block anywhere). The compile and link legs need the aether tree beside the
//! clones, `EAGLE_ROOT` and `EAGLE_DOOR_DIR`, and a system C++ compiler; each prints that it skipped when one is absent.
#![cfg(feature = "old-contract")]
use aether_writers::contract::types::Exports;
use aether_writers::contract::Json;
use osprey_host::lower::request::KernelAnswer;
use osprey_host::lower::unit_meta::Wire;
use osprey_host::lower::{set_text, SET_HEADERS};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// The recorded calls the answers are lowered from.
const CALLS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/meta_calls.jsonl");

/// The three kernels of the set tests, as the folder of answers names them: a plain kernel, a split kernel (two stage units and one
/// that lists them) and an events kernel. All three are in double precision; the recorded calls hold no small single-precision kernel.
const KERNELS: [&str; 3] = [
    "73898.101_square_host",
    "81812.2_scaled_host",
    "83645.104_ev_untimed_host",
];

/// The folder of answers, written once by the replay tool.
fn answers_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join("osprey_set_text_answers");
        let _ = std::fs::remove_dir_all(&dir);
        let out = Command::new(env!("CARGO_BIN_EXE_kreplay"))
            .args(["--emit-meta", CALLS, dir.to_str().unwrap(), "--emit-set"])
            .output()
            .expect("kreplay starts");
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        dir
    })
}

/// The answer of a kernel; `set` is the one asked for the texts of a set.
fn answer(kernel: &str, set: bool) -> KernelAnswer {
    let suffix = if set { "_set" } else { "" };
    let text = std::fs::read_to_string(answers_dir().join(format!("{kernel}{suffix}.json")))
        .expect("an answer");
    KernelAnswer::from_json(&text).expect("the answer reads")
}

/// The input of the set text over the units of the answers, in order.
fn input(answers: &[&KernelAnswer]) -> String {
    let units = answers.iter().flat_map(|a| &a.units).map(|u| u.meta.put());
    Json::Obj(vec![
        ("v".into(), Json::Num("1".into())),
        ("units".into(), Json::Arr(units.collect())),
    ])
    .to_text()
}

/// The texts of an answer that hold the self-check exports.
fn with_block(answer: &KernelAnswer) -> usize {
    let texts = answer.units.iter().flat_map(|u| &u.texts);
    texts.filter(|t| t.text.contains("eagle_abi_tag")).count()
}

#[test]
fn a_request_without_the_block_has_it_in_no_text_and_by_default_in_every_text_of_its_kernel() {
    for kernel in KERNELS {
        // asked alone, every unit text of the kernel carries the block, as it always did
        let alone = answer(kernel, false);
        let texts = alone.units.iter().map(|u| u.texts.len()).sum::<usize>();
        assert_eq!(with_block(&alone), texts, "{kernel}");
        assert_eq!(with_block(&answer(kernel, true)), 0, "{kernel}");
    }
}

#[test]
fn the_text_of_one_answer_is_its_block_and_its_library_record_after_the_headers() {
    for kernel in KERNELS {
        let (single, set) = (answer(kernel, false), answer(kernel, true));
        let block = aether_writers::unit::exports(&Exports::Derived, false).unwrap();
        let first = single
            .units
            .iter()
            .flat_map(|u| &u.texts)
            .find(|t| t.text.contains(&block));
        assert!(
            first.is_some(),
            "{kernel}: the block is not in the answer's own text"
        );
        let text = set_text(&input(&[&single])).expect("the set text");
        assert_eq!(
            text,
            format!("{SET_HEADERS}{block}{}", single.meta_source),
            "{kernel}"
        );
        // the units of the set-asked answer have the same records, so the same text
        assert_eq!(set_text(&input(&[&set])).unwrap(), text, "{kernel}");
    }
}

#[test]
fn stated_sizes_replace_the_derived_ones() {
    let one = answer(KERNELS[0], true);
    let units = one.units.iter().map(|u| u.meta.put()).collect();
    let sizes = Json::Arr((1..=5).map(|n| Json::Num(n.to_string())).collect());
    let stated = Json::Obj(vec![
        ("v".into(), Json::Num("1".into())),
        ("units".into(), Json::Arr(units)),
        ("exports".into(), sizes),
    ]);
    let text = set_text(&stated.to_text()).unwrap();
    assert!(
        text.contains("1ull,\n    2ull,\n    3ull,\n    4ull,\n    5ull"),
        "{text}"
    );
}

#[test]
fn an_empty_list_two_units_of_one_name_and_a_record_that_does_not_read_are_refused() {
    let one = answer(KERNELS[0], true);
    let empty = r#"{"v": 1, "units": []}"#;
    let refused = |why: &str, input: &str| {
        let refusal = set_text(input).expect_err(why);
        assert_eq!(
            refusal.kind,
            aether_writers::RefusalKind::BadRequest,
            "{why}"
        );
        refusal.reason
    };
    assert!(refused("empty", empty).contains("without a unit"));
    let twice = input(&[&one, &one]);
    let said = refused("twice", &twice);
    assert!(
        said.contains("units 0 and 1") && said.contains("square"),
        "{said}"
    );
    let broken = r#"{"v": 1, "units": [{"name": "k"}]}"#;
    refused("broken", broken);
}

/// The compile flags of a host unit, as the derived-set test compiles them, and the include roots.
struct Tools {
    /// The aether tree.
    aether: PathBuf,
    /// The eagle tree.
    eagle: PathBuf,
    /// The root of the door headers.
    door: PathBuf,
}

impl Tools {
    /// The trees, when the environment names them; a line says what is missing.
    fn found() -> Option<Tools> {
        let var = |name: &str| std::env::var_os(name).map(PathBuf::from);
        let (Some(eagle), Some(door)) = (var("EAGLE_ROOT"), var("EAGLE_DOOR_DIR")) else {
            println!("set_text: skipped, EAGLE_ROOT and EAGLE_DOOR_DIR are not both set");
            return None;
        };
        let aether = Path::new(env!("CARGO_MANIFEST_DIR")).join("../aether");
        Some(Tools {
            aether,
            eagle,
            door,
        })
    }

    /// Compiles a C++ text to an object (`-c`) with the flags of a host unit and the door headers; the compiler's own words fail the test.
    fn object(&self, source: &Path, object: &Path) {
        let built = Command::new("c++")
            .args(["-std=c++23", "-fPIC", "-DAETHER_CPP_MODE", "-I"])
            .arg(&self.aether)
            .arg("-I")
            .arg(&self.eagle)
            .arg("-I")
            .arg(&self.door)
            .args(["-c", "-x", "c++"])
            .arg(source)
            .arg("-o")
            .arg(object)
            .output()
            .expect("c++ starts");
        assert!(
            built.status.success(),
            "{}",
            String::from_utf8_lossy(&built.stderr)
        );
    }
}

/// A folder of this test's own.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("osprey_set_text_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The three kernels' answers, asked for the texts of a set.
fn three() -> Vec<KernelAnswer> {
    KERNELS.iter().map(|k| answer(k, true)).collect()
}

#[test]
fn the_text_over_three_kernels_has_one_block_and_one_record_and_compiles() {
    let answers = three();
    let text = set_text(&input(&answers.iter().collect::<Vec<_>>())).expect("the set text");
    let count = |needle: &str| text.matches(needle).count();
    assert_eq!(count("const char eagle_abi_tag[]"), 1);
    assert_eq!(count("eagle_layout_sizes[5] ="), 1);
    assert_eq!(count("eg_unit_library_meta(void)"), 1);
    let Some(tools) = Tools::found() else { return };
    let dir = scratch("compile");
    std::fs::write(dir.join("set.cpp"), &text).unwrap();
    tools.object(&dir.join("set.cpp"), &dir.join("set.o"));
}

/// A program that opens the library at its first argument, reads the library record and resolves the host entry every unit names.
const DRIVER: &str = r#"
#include <dlfcn.h>
#include <cstdio>
#include "eagle/door/eg_door.h"
#include "eagle/door/eg_unit_meta.h"
int main(int, char** argv) {
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::printf("dlopen: %s\n", dlerror()); return 2; }
    auto meta = reinterpret_cast<eg_unit_library_meta_fn>(dlsym(lib, EG_UNIT_LIBRARY_META_SYMBOL));
    if (!meta) { std::printf("no record\n"); return 3; }
    const eg_library_meta* record = meta();
    int missing = 0;
    for (unsigned at = 0; at < record->n_units; ++at) {
        const eg_unit_meta* unit = record->units[at];
        void* entry = unit->host_entry ? dlsym(lib, unit->host_entry) : nullptr;
        std::printf("%s %s %s\n", unit->name, unit->host_entry ? unit->host_entry : "-", entry ? "found" : unit->host_entry ? "missing" : "none");
        missing += unit->host_entry && !entry;
    }
    return missing ? 4 : 0;
}
"#;

#[test]
fn the_units_of_three_kernels_and_the_set_text_link_into_one_library_that_resolves_every_entry() {
    let Some(tools) = Tools::found() else { return };
    if Command::new("c++").arg("--version").output().is_err() {
        println!("set_text: skipped, no system C++ compiler");
        return;
    }
    let dir = scratch("link");
    let answers = three();
    let mut objects = vec![];
    for (kernel, answer) in KERNELS.iter().zip(&answers) {
        for unit in &answer.units {
            for text in unit
                .texts
                .iter()
                .filter(|t| t.target == aether_writers::contract::types::Target::Host)
            {
                let name = format!("{kernel}_{}", unit.meta.name);
                std::fs::write(dir.join(format!("{name}.cpp")), &text.text).unwrap();
                tools.object(
                    &dir.join(format!("{name}.cpp")),
                    &dir.join(format!("{name}.o")),
                );
                objects.push(dir.join(format!("{name}.o")));
            }
        }
    }
    let set = set_text(&input(&answers.iter().collect::<Vec<_>>())).unwrap();
    std::fs::write(dir.join("set.cpp"), set).unwrap();
    tools.object(&dir.join("set.cpp"), &dir.join("set.o"));
    objects.push(dir.join("set.o"));
    let library = dir.join(format!(
        "units.{}",
        if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        }
    ));
    let linked = Command::new("c++")
        .arg("-shared")
        .args(&objects)
        .arg("-o")
        .arg(&library)
        .output()
        .unwrap();
    assert!(
        linked.status.success(),
        "{}",
        String::from_utf8_lossy(&linked.stderr)
    );
    std::fs::write(dir.join("driver.cpp"), DRIVER).unwrap();
    let built = Command::new("c++")
        .args(["-std=c++17", "-I"])
        .arg(&tools.door)
        .arg(dir.join("driver.cpp"))
        .arg("-o")
        .arg(dir.join("driver"))
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let run = Command::new(dir.join("driver"))
        .arg(&library)
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&run.stdout).into_owned();
    assert!(run.status.success(), "{said}");
    // the record lists the units in the order given, and each unit with a host text resolves
    let listed: Vec<&str> = said.lines().filter_map(|l| l.split(' ').next()).collect();
    let given: Vec<&str> = answers
        .iter()
        .flat_map(|a| &a.units)
        .map(|u| u.meta.name.as_str())
        .collect();
    assert_eq!(listed, given, "{said}");
    let found = said.lines().filter(|l| l.ends_with(" found")).count();
    assert_eq!(found, objects.len() - 1, "{said}");
    println!(
        "set_text: linked {} objects, every host entry resolved:\n{said}",
        objects.len()
    );
}
