//! The device build instructions against the recorded old driver: for every line of `tests/data/device_argv_old.jsonl` (the image
//! form and the argv the old command builder made for a compiler release, a driver release, an architecture, defines and an
//! optimisation level, with its paths as placeholders), the facts answer for the same stated machine gives the same image form
//! and a `compile` template that, with its `{defines}` place filled by one `-D<define>` element for each define and with the
//! one pair `-ccbin {host_compiler}` the template adds after the device compiler left out, is the old argv element by element.
//!
//! A line the old code refused (an unknown optimisation level) must be refused by the facts question too; for such a line what is
//! compared is that both refuse, not their words, which are each side's own.
use aether_writers::contract::{Contract, Json};

/// The recorded old behaviour, one JSON line for each combination.
const RECORDED: &str = include_str!("data/device_argv_old.jsonl");

/// The element list under a key of a record.
fn texts(record: &Json, key: &str) -> Vec<String> {
    let items = record.get(key).unwrap_or_else(|| panic!("no {key}")).arr();
    items.iter().map(|i| i.str().unwrap().to_string()).collect()
}

/// The `device_build` record the facts answer gives for the stated machine of a recorded line, or why it refused.
fn asked(row: &Json) -> Result<Json, String> {
    let dir = osprey_replay::merged_facts();
    let text = |key: &str| row.get(key).and_then(Json::str).unwrap_or("").to_string();
    let mut members = vec![
        format!("\"device_arch\":\"{}\"", text("arch")),
        format!("\"device_release\":\"{}\"", text("compiler_release")),
    ];
    for (key, stated) in [
        ("driver_release", "driver_release"),
        ("opt_level", "opt_level"),
    ] {
        if !text(key).is_empty() {
            members.push(format!("\"{stated}\":\"{}\"", text(key)));
        }
    }
    let state = osprey_replay::device_stated(&format!("{{{}}}", members.join(",")))
        .map_err(String::from)?;
    let identity = "/usr/local/cuda/bin/nvcc || Cuda compilation tools || 0";
    let answer = osprey_replay::answer_device(&dir, identity, &state).map_err(String::from)?;
    let machine = Contract::parse(&answer).unwrap().0;
    Ok(machine.get("device_build").expect("a device_build").clone())
}

/// The template without its host compiler pair, or none unless the pair `-ccbin {host_compiler}` stands exactly once, right after
/// the device compiler.
fn without_host_compiler(argv: Vec<String>) -> Option<Vec<String>> {
    let pair = ["-ccbin", "{host_compiler}"];
    let once = argv
        .windows(2)
        .filter(|w| w[0] == pair[0] && w[1] == pair[1])
        .count()
        == 1;
    let after_compiler = argv
        .get(1..3)
        .is_some_and(|w| w[0] == pair[0] && w[1] == pair[1]);
    (once && after_compiler).then(|| [&argv[..1], &argv[3..]].concat())
}

#[test]
fn the_image_form_and_the_compile_of_every_recorded_combination_are_the_old_ones_and_the_compile_names_the_host_compiler_once(
) {
    let (mut compared, mut equal, mut refusals) = (0, 0, 0);
    let mut shown = vec![];
    for line in RECORDED.lines() {
        let row = Contract::parse(line).unwrap().0;
        compared += 1;
        let good = match (row.get("argv"), asked(&row)) {
            (Some(_), Ok(build)) => {
                let defines = texts(&row, "defines");
                let argv = texts(&build, "compile")
                    .into_iter()
                    .flat_map(|flag| match flag.as_str() {
                        "{defines}" => defines.iter().map(|d| format!("-D{d}")).collect(),
                        _ => vec![flag],
                    })
                    .collect();
                let argv = without_host_compiler(argv);
                let form = build.get("image_format").and_then(Json::str);
                argv.as_ref() == Some(&texts(&row, "argv"))
                    && form == row.get("image_format").and_then(Json::str)
            }
            (None, Err(_)) => {
                refusals += 1;
                true
            }
            _ => false,
        };
        if good {
            equal += 1
        } else if shown.len() < 5 {
            shown.push(line.to_string());
        }
    }
    println!(
        "device_build: combinations compared {compared} / equal {equal} / different {} (of the equal, refused by both {refusals})",
        compared - equal
    );
    assert!(
        compared > 0 && shown.is_empty(),
        "different:\n{}",
        shown.join("\n")
    );
}

#[test]
fn a_driver_release_that_is_not_major_minor_is_refused_by_name() {
    let dir = osprey_replay::merged_facts();
    let state =
        osprey_replay::device_stated(r#"{"device_release":"12.6","driver_release":"12"}"#).unwrap();
    let identity = "/usr/local/cuda/bin/nvcc || Cuda compilation tools || 0";
    let why = osprey_replay::answer_device(&dir, identity, &state)
        .map_err(String::from)
        .unwrap_err();
    assert_eq!(
        why,
        "stated machine: driver_release \"12\" is not major.minor"
    );
}
