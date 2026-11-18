//! The host build instructions against the recorded old driver: for every line of `tests/data/host_argv_old.jsonl` (the argv the old
//! command builder made in both header shapes, with its paths as placeholders), the facts answer for the same compiler, machine and settings states
//! `compile_unit` and `link` that make the same command.
//!
//! The old command compiled and linked in one: `<compiler> <flags> -shared <flags> -MD -MF <dep> <src> -o <library>`. The split rule:
//! the link-only elements of the old argv are `-shared` and the pair `-o {output}`; every other element (the compiler, the code
//! generation flags, the standard, the includes, the dependency file, the source) belongs to the compile. Then
//!  * the old argv without its link-only elements equals `compile_unit` without its own object elements (`-c` and the pair
//!    `-o {object}`), element by element;
//!  * `link` for each system is the compiler, the old link-only elements in their order, and `{objects}`.
//!
//! A line the old code refused (an unknown profile or level, a vector profile its compiler cannot build) must be refused by the
//! facts question too. Nothing is adjusted: a difference is printed and fails the test.
use aether_writers::contract::{Contract, Json};
use osprey_lowering::facts;

/// The recorded old argv, one JSON line for each combination.
const RECORDED: &str = include_str!("data/host_argv_old.jsonl");

/// The element list under a key of a record.
fn texts(record: &Json, key: &str) -> Vec<String> {
    let items = record.get(key).unwrap_or_else(|| panic!("no {key}")).arr();
    items.iter().map(|i| i.str().unwrap().to_string()).collect()
}

/// The version line a recorded compiler family prints.
fn identity(row: &Json) -> String {
    let line = row.get("identity_line").and_then(Json::str).unwrap();
    format!("c++ || {line} || 0")
}

/// The stated machine of a recorded line.
fn stated(row: &Json) -> String {
    let text = |key: &str| row.get(key).and_then(Json::str).unwrap_or("").to_string();
    let isa = text("machine");
    let vector = if isa == "arm64" { "neon" } else { "avx2" };
    let mut members = vec![
        format!("\"headers\":\"{}\"", text("headers")),
        format!("\"isa\":\"{isa}\""),
        format!("\"vector_isa\":\"{vector}\""),
    ];
    for (key, stated) in [
        ("profile", "host_profile"),
        ("opt_level", "opt_level"),
        ("witness", "masked_call_witness"),
    ] {
        if !text(key).is_empty() {
            members.push(format!("\"{stated}\":\"{}\"", text(key)));
        }
    }
    if !text("gcc_release").is_empty() {
        members.push(format!("\"gcc_release\":\"{}\"", text("gcc_release")));
    }
    format!("{{{}}}", members.join(","))
}

/// The commands a facts answer states for a recorded line, or why it refused.
fn asked(row: &Json) -> Result<Json, String> {
    let dir = osprey_replay::merged_facts();
    let state = facts::stated_from_json(&stated(row)).map_err(String::from)?;
    let answer = osprey_replay::answer_named(&dir, "host", &identity(row), &facts::probe(), &state)
        .map_err(String::from)?;
    let machine = Contract::parse(&answer).unwrap().0;
    Ok(machine
        .get("host_build")
        .expect("a host_build record")
        .clone())
}

/// Whether the old argv and the commands the answer states agree, by the rule above.
fn agree(old: &[String], defines: &[String], build: &Json) -> bool {
    let link_only =
        |at: usize| old[at] == "-shared" || old[at] == "-o" || (at > 0 && old[at - 1] == "-o");
    let (mut compile, mut link) = (vec![], vec![]);
    for (at, element) in old.iter().enumerate() {
        let side = if link_only(at) {
            &mut link
        } else {
            &mut compile
        };
        side.push(element.clone());
    }
    let unit = texts(build, "compile_unit");
    let unit_compile: Vec<String> = unit
        .into_iter()
        .filter(|flag| !["-c", "-o", "{object}"].contains(&flag.as_str()))
        .flat_map(|flag| match flag.as_str() {
            "{defines}" => defines.iter().map(|d| format!("-D{d}")).collect(),
            _ => vec![flag],
        })
        .collect();
    let systems = build.get("link").expect("a link record");
    let linked = ["linux", "macos"].iter().all(|system| {
        let argv = texts(systems, system);
        let objects = argv.iter().position(|a| a == "{objects}");
        let middle: Vec<String> = argv[1..objects.unwrap_or(argv.len())].to_vec();
        argv[0] == "{compiler}" && middle == link && objects == Some(argv.len() - 1)
    });
    compile == unit_compile && linked
}

#[test]
fn the_commands_of_every_recorded_combination_are_the_old_ones() {
    let (mut compared, mut equal, mut refusals) = (0, 0, 0);
    let mut shown = vec![];
    for line in RECORDED.lines() {
        let row = Contract::parse(line).unwrap().0;
        compared += 1;
        let good = match (row.get("argv"), asked(&row)) {
            (Some(old), Ok(build)) => {
                let old: Vec<String> = old
                    .arr()
                    .iter()
                    .map(|a| a.str().unwrap().to_string())
                    .collect();
                let defines = texts(&row, "defines");
                agree(&old, &defines, &build)
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
        "host_build: combinations compared {compared} / equal {equal} / different {} (of the equal, refused by both {refusals})",
        compared - equal
    );
    assert!(
        compared > 0 && shown.is_empty(),
        "different:\n{}",
        shown.join("\n")
    );
}

#[test]
fn a_gcc_release_is_read_from_the_identity_when_none_is_stated() {
    let dir = osprey_replay::merged_facts();
    let state = |witness: &str| {
        let text = format!("{{\"headers\":\"roots\",\"isa\":\"x86_64\",\"vector_isa\":\"avx2\",\"host_profile\":\"native-vector-math\",\"masked_call_witness\":\"{witness}\"}}");
        facts::stated_from_json(&text).unwrap()
    };
    let flag = |identity: &str, witness: &str| {
        let answer =
            osprey_replay::answer_named(&dir, "host", identity, &facts::probe(), &state(witness))
                .unwrap();
        answer.contains("-DAETHER_HVM_MASKED=1")
    };
    assert!(flag(
        "g++ || g++ (Ubuntu 13.2.0-23ubuntu4) 13.2.0 || 0",
        "fixed"
    ));
    assert!(!flag(
        "g++ || g++ (Ubuntu 13.2.0-23ubuntu4) 13.2.0 || 0",
        "miscompile"
    ));
    assert!(!flag("g++ || g++ (GCC) 13.4.0 || 0", "fixed"));
    assert!(flag("g++ || g++ (GCC) 13 || 0", "fixed"));
    assert!(!flag("g++ || g++ 11 || 0", "fixed"));
}

#[test]
fn headers_are_required_and_closed() {
    let dir = osprey_replay::merged_facts();
    let refusal = |record: &str| -> String {
        let state = facts::stated_from_json(record).unwrap();
        osprey_replay::answer_named(
            &dir,
            "host",
            "c++ || clang version 17.0.1 || 0",
            &facts::probe(),
            &state,
        )
        .map_err(String::from)
        .unwrap_err()
    };
    assert!(refusal("{}").contains("needs headers"));
    assert!(refusal(r#"{"headers":"both"}"#).contains("headers both is not roots or served"));
}

#[test]
fn the_unit_and_the_set_compile_each_have_one_place_for_defines() {
    let state = facts::stated_from_json(r#"{"headers":"served"}"#).unwrap();
    let dir = osprey_replay::merged_facts();
    let answer = osprey_replay::answer_named(
        &dir,
        "host",
        "c++ || clang version 17.0.1 || 0",
        &facts::probe(),
        &state,
    )
    .unwrap();
    let build = Contract::parse(&answer)
        .unwrap()
        .0
        .get("host_build")
        .unwrap()
        .clone();
    let count = |key: &str| {
        texts(&build, key)
            .iter()
            .filter(|a| *a == "{defines}")
            .count()
    };
    assert_eq!((count("compile_unit"), count("compile_set")), (1, 1));
}
