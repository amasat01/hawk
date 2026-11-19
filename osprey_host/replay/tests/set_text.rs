//! The set text: one C++ text for the units of one or several answers built into one library. The answers come from the recorded calls
//! of `tests/data/meta_calls.jsonl`, lowered by `kreplay --emit-meta`, once asking for the texts of a single kernel (the exports block
//! in the first text) and once for the texts of a set (no block anywhere). The compile and link legs need the aether tree beside the
//! clones, `EAGLE_ROOT` and `OSPREY_EAGLE_SRC`, and a system C++ compiler; each prints that it skipped when one is absent.
#![cfg(feature = "old-contract")]
mod common;
use aether_writers::contract::types::Exports;
use aether_writers::contract::Json;
use osprey_lowering::lower::request::KernelAnswer;
use osprey_lowering::lower::unit_meta::Wire;
use osprey_lowering::lower::{set_text, SET_HEADERS};
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

/// A recorded device kernel.
const DEVICE_KERNEL: &str = "80036.18__matrix_cuda";

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
fn a_set_request_takes_the_block_from_host_texts_only() {
    let block = aether_writers::unit::exports(&Exports::Derived, false).unwrap();
    let texts = |answer: &KernelAnswer| -> Vec<String> {
        let all = answer.units.iter().flat_map(|u| &u.texts);
        all.map(|t| t.text.clone()).collect()
    };
    // a host text loses exactly the block
    let (alone, set) = (answer(KERNELS[0], false), answer(KERNELS[0], true));
    for (with, without) in texts(&alone).iter().zip(texts(&set)) {
        assert!(with.contains(&block) && !without.contains("eagle_abi_tag"));
        assert_eq!(with.replacen(&block, "", 1), without);
    }
    // a device text is the same text for both values of the member
    let (alone, set) = (answer(DEVICE_KERNEL, false), answer(DEVICE_KERNEL, true));
    let device = texts(&alone);
    assert!(device.iter().all(|t| t.contains("eagle_abi_tag")));
    assert_eq!(device, texts(&set));
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

/// The include roots and the host build instructions of this machine and the system compiler, from the facts answer.
struct Tools {
    /// The aether tree.
    aether: PathBuf,
    /// The eagle tree.
    eagle: PathBuf,
    /// The root of the door headers.
    door: PathBuf,
    /// The `host_build` record of the facts answer for the system compiler on this machine.
    build: Json,
}

impl Tools {
    /// The trees and the build instructions, when the environment names the trees; a line says what is missing.
    fn found() -> Option<Tools> {
        let var = |name: &str| std::env::var_os(name).map(PathBuf::from);
        let (Some(eagle), Some(door)) = (var("EAGLE_ROOT"), var("OSPREY_EAGLE_SRC")) else {
            let missing = if var("EAGLE_ROOT").is_none() {
                "EAGLE_ROOT"
            } else {
                "OSPREY_EAGLE_SRC"
            };
            common::skipped(Some(missing), &format!("{missing} is not set"));
            return None;
        };
        let Ok(version) = Command::new("c++").arg("--version").output() else {
            common::skipped(None, "no system C++ compiler");
            return None;
        };
        let line = String::from_utf8_lossy(&version.stdout);
        let identity = format!("c++ || {} || 0", line.lines().next().unwrap_or(""));
        let facts_dir = osprey_replay::merged_facts();
        let stated = osprey_lowering::facts::stated_from_json(r#"{"headers":"roots"}"#).unwrap();
        let answer = osprey_replay::answer_named(
            &facts_dir,
            "host",
            &identity,
            &osprey_lowering::facts::probe(),
            &stated,
        )
        .expect("the facts of this machine");
        let machine = aether_writers::contract::Contract::parse(&answer)
            .unwrap()
            .0;
        let build = machine.get("host_build").expect("host_build").clone();
        let aether = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../aether");
        Some(Tools {
            aether,
            eagle,
            door,
            build,
        })
    }

    /// Runs an argv template of the build instructions with its places filled; the compiler's own words fail the test.
    fn run(&self, template: &Json, places: &[(&str, String)], objects: &[PathBuf]) {
        let mut argv: Vec<String> = vec![];
        for item in template.arr() {
            let item = item.str().unwrap();
            match item {
                "{compiler}" | "{host_compiler}" => argv.push("c++".into()),
                "{objects}" => {
                    argv.extend(objects.iter().map(|o| o.to_string_lossy().into_owned()))
                }
                "{include:aether}" => argv.push(self.aether.to_string_lossy().into_owned()),
                "{include:eagle}" => argv.push(self.eagle.to_string_lossy().into_owned()),
                "{include:door}" => argv.push(self.door.to_string_lossy().into_owned()),
                // these builds state no defines
                "{defines}" => {}
                _ => argv.push(
                    places
                        .iter()
                        .find(|(k, _)| *k == item)
                        .map_or(item.into(), |(_, v)| v.clone()),
                ),
            }
        }
        let built = Command::new(&argv[0])
            .args(&argv[1..])
            .output()
            .expect("c++ starts");
        assert!(
            built.status.success(),
            "{argv:?}\n{}",
            String::from_utf8_lossy(&built.stderr)
        );
    }

    /// Compiles a C++ text to an object with the instructions for a unit text, or for the set text when `set`.
    fn object(&self, source: &Path, object: &Path, set: bool) {
        let template = self
            .build
            .get(if set { "compile_set" } else { "compile_unit" })
            .unwrap();
        let places = [
            ("{source}", source.to_string_lossy().into_owned()),
            ("{object}", object.to_string_lossy().into_owned()),
            (
                "{depfile}",
                object.with_extension("d").to_string_lossy().into_owned(),
            ),
        ];
        self.run(template, &places, &[]);
    }

    /// Links objects into one shared library with the instructions of this system.
    fn link(&self, objects: &[PathBuf], library: &Path) {
        let system = if cfg!(target_os = "macos") {
            "macos"
        } else {
            "linux"
        };
        let template = self.build.get("link").unwrap().get(system).unwrap();
        self.run(
            template,
            &[("{output}", library.to_string_lossy().into_owned())],
            objects,
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
    tools.object(&dir.join("set.cpp"), &dir.join("set.o"), true);
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
                    false,
                );
                objects.push(dir.join(format!("{name}.o")));
            }
        }
    }
    let set = set_text(&input(&answers.iter().collect::<Vec<_>>())).unwrap();
    std::fs::write(dir.join("set.cpp"), set).unwrap();
    tools.object(&dir.join("set.cpp"), &dir.join("set.o"), true);
    objects.push(dir.join("set.o"));
    let library = dir.join(format!(
        "units.{}",
        if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        }
    ));
    tools.link(&objects, &library);
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

/// The device facts answer's `embed` recipe for the nvcc of the stand-in identity, with the data directory `facts_dir`.
fn embed_of(facts_dir: &Path) -> Result<Json, String> {
    let identity =
        "/usr/local/cuda/bin/nvcc || Cuda compilation tools, release 12.6, V12.6.77 || 0";
    let stated = osprey_lowering::facts::stated_from_json("{}").unwrap();
    let probe = osprey_lowering::facts::probe();
    let answer = osprey_replay::answer_named(facts_dir, "cuda", identity, &probe, &stated)
        .map_err(String::from)?;
    let machine = aether_writers::contract::Contract::parse(&answer)
        .unwrap()
        .0;
    let build = machine.get("device_build").expect("device_build");
    Ok(build.get("embed").expect("embed").clone())
}

/// The shipped data directory.
fn shipped() -> PathBuf {
    osprey_replay::merged_facts()
}

/// The assembler source of an embed recipe for an operating system, with its places filled.
fn embed_source(embed: &Json, system: &str, symbol: &str, path: &Path) -> String {
    let source = embed.get("source").unwrap().get(system).unwrap();
    source
        .str()
        .unwrap()
        .replace("{size_symbol}", &format!("{symbol}_size"))
        .replace("{symbol}", symbol)
        .replace("{image_path}", &path.to_string_lossy())
}

#[test]
fn the_linux_recipe_is_what_the_data_states_it_cannot_be_assembled_here() {
    let embed = embed_of(&shipped()).unwrap();
    let text = embed_source(&embed, "linux", "u__image", Path::new("/i.bin"));
    let body = "\t.section .rodata\n\t.balign 16\n\t.globl u__image\n\t.type u__image, %object\nu__image:\n\t.incbin \"/i.bin\"\nu__image__end:\n";
    let rest = "\t.size u__image, .-u__image\n\t.globl u__image_size\n\t.type u__image_size, %object\n\t.balign 8\nu__image_size:\n\t.quad u__image__end - u__image\n\t.size u__image_size, 8\n\t.section .note.GNU-stack,\"\",%progbits\n";
    assert_eq!(text, format!("{body}{rest}"));
    let argv: Vec<&str> = embed
        .get("compile")
        .unwrap()
        .arr()
        .iter()
        .filter_map(Json::str)
        .collect();
    assert_eq!(
        argv,
        [
            "{host_compiler}",
            "-x",
            "assembler",
            "-c",
            "{source}",
            "-o",
            "{object}"
        ]
    );
}

/// A copy of the data directory with one text of `cuda_image.json` replaced.
fn edited_facts(name: &str, from: &str, to: &str) -> PathBuf {
    let dir = scratch(name);
    for entry in std::fs::read_dir(shipped()).unwrap() {
        let path = entry.unwrap().path();
        std::fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
    }
    let file = dir.join("cuda_image.json");
    let text = std::fs::read_to_string(&file).unwrap();
    assert!(text.contains(from), "{from}");
    std::fs::write(file, text.replacen(from, to, 1)).unwrap();
    dir
}

#[test]
fn each_system_embeds_one_incbin_between_the_symbol_and_its_end_with_no_byte_after_it() {
    let embed = embed_of(&shipped()).unwrap();
    // what a line may be between the end label and the size word: names, types, sizes and alignment, never data
    let layout_only = [".size", ".globl", ".type", ".balign", ".p2align"];
    for (system, prefix) in [("linux", ""), ("macos", "_")] {
        let text = embed_source(&embed, system, "u", Path::new("/i.bin"));
        let lines: Vec<&str> = text.lines().collect();
        let at = |label: &str| lines.iter().position(|l| *l == label).expect(label);
        let (start, end) = (at(&format!("{prefix}u:")), at(&format!("{prefix}u__end:")));
        let size = at(&format!("{prefix}u_size:"));
        assert_eq!(&lines[start + 1..end], ["\t.incbin \"/i.bin\""], "{system}");
        for line in &lines[end + 1..size] {
            let directive = line.split_whitespace().next().unwrap();
            assert!(layout_only.contains(&directive), "{system}: {line}");
        }
    }
}

#[test]
fn an_operating_system_without_a_recipe_and_a_key_the_data_does_not_know_are_refused_by_name() {
    let tail = edited_facts(
        "embed_tail",
        "\"compile\":",
        "\"image_tail\": {}, \"compile\":",
    );
    let said = embed_of(&tail).unwrap_err();
    assert!(
        said.contains("cuda_image.json has the unknown key image_tail"),
        "{said}"
    );
    let no_object = edited_facts("embed_object", "\"macos\": [", "\"other\": [");
    let said = embed_of(&no_object).unwrap_err();
    assert!(
        said.contains("cuda_image.json has the unknown key other"),
        "{said}"
    );
    let missing = edited_facts("embed_missing", "\"linux\": [", "\"other\": [");
    assert!(embed_of(&missing).is_err());
}

/// A program that opens the library at its first argument and, for every unit that names a device image, compares the bytes and
/// the size read through the two symbols the record names with the file `<folder>/<unit>.bin` (argv 2).
const IMAGE_READER: &str = r#"
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "eagle/door/eg_door.h"
#include "eagle/door/eg_unit_meta.h"
int main(int, char** argv) {
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::printf("dlopen: %s\n", dlerror()); return 2; }
    auto meta = reinterpret_cast<eg_unit_library_meta_fn>(dlsym(lib, EG_UNIT_LIBRARY_META_SYMBOL));
    const eg_library_meta* record = meta();
    int bad = 0;
    for (unsigned at = 0; at < record->n_units; ++at) {
        const eg_unit_meta* unit = record->units[at];
        if (!unit->device_image) continue;
        auto bytes = static_cast<const unsigned char*>(dlsym(lib, unit->device_image));
        auto size = static_cast<const uint64_t*>(dlsym(lib, unit->device_image_size));
        std::string path = std::string(argv[2]) + "/" + unit->name + ".bin";
        FILE* f = std::fopen(path.c_str(), "rb");
        std::vector<unsigned char> blob(1 << 20);
        size_t n = f ? std::fread(blob.data(), 1, blob.size(), f) : 0;
        bool same = bytes && size && *size == n && std::memcmp(bytes, blob.data(), n) == 0;
        std::printf("%s %s %zu %s\n", unit->name, same ? "same" : "DIFFERENT", n, unit->device_image);
        bad += !same;
    }
    return bad ? 4 : 0;
}
"#;

#[test]
fn images_embedded_by_the_recipe_are_read_back_byte_for_byte_through_the_symbols_the_record_names()
{
    let Some(tools) = Tools::found() else { return };
    let embed = embed_of(&shipped()).unwrap();
    let system = if cfg!(target_os = "macos") {
        "macos"
    } else {
        "linux"
    };
    let dir = scratch("embed");
    let units = ["80036.18__matrix_cuda", "81814.4_kg_fwd_cuda"];
    let answers: Vec<KernelAnswer> = units.iter().map(|k| answer(k, true)).collect();
    let mut objects = vec![];
    for (at, unit) in answers.iter().flat_map(|a| &a.units).enumerate() {
        let meta = &unit.meta;
        let symbol = meta.device_image.clone().expect("a device unit");
        // a blob of known bytes in place of the image, which needs a device compiler that is not on this machine
        // (a zero byte and a 0xff byte first, and a length that is no multiple of the 16-byte alignment)
        let mut blob: Vec<u8> = (0..4001 + 37 * at)
            .map(|i| (i * 31 + at * 7 + i / 251) as u8)
            .collect();
        blob[..2].copy_from_slice(&[0x00, 0xff]);
        let path = dir.join(format!("{}.bin", meta.name));
        std::fs::write(&path, &blob).unwrap();
        let source = dir.join(format!("{}.s", meta.name));
        std::fs::write(&source, embed_source(&embed, system, &symbol, &path)).unwrap();
        let place = [
            ("{source}", source.to_string_lossy().into_owned()),
            (
                "{object}",
                dir.join(format!("{}.o", meta.name))
                    .to_string_lossy()
                    .into_owned(),
            ),
        ];
        tools.run(embed.get("compile").unwrap(), &place, &[]);
        objects.push(dir.join(format!("{}.o", meta.name)));
    }
    let all: Vec<&KernelAnswer> = answers.iter().collect();
    std::fs::write(dir.join("set.cpp"), set_text(&input(&all)).unwrap()).unwrap();
    tools.object(&dir.join("set.cpp"), &dir.join("set.o"), true);
    objects.push(dir.join("set.o"));
    let library = dir.join(format!(
        "units.{}",
        if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        }
    ));
    tools.link(&objects, &library);
    std::fs::write(dir.join("reader.cpp"), IMAGE_READER).unwrap();
    let built = Command::new("c++")
        .args(["-std=c++17", "-I"])
        .arg(&tools.door)
        .arg(dir.join("reader.cpp"))
        .arg("-o")
        .arg(dir.join("reader"))
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let run = Command::new(dir.join("reader"))
        .arg(&library)
        .arg(&dir)
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&run.stdout).into_owned();
    assert!(run.status.success(), "{said}");
    assert_eq!(
        said.lines().filter(|l| l.contains(" same ")).count(),
        2,
        "{said}"
    );
    println!("set_text: two embedded images read back byte for byte:\n{said}");
}
