//! The variables that choose the host build (`HAWK_HOST_PROFILE`, `HAWK_OPT_LEVEL`) are read on every machine ask of a process, not
//! only the first: the answer, its key and the build flags that follow from it move with them, and come back with them.
#![cfg(feature = "old-contract")]
use aether_writers::contract::{Contract, Json};
use osprey_lowering::tool;

/// Puts the shipped documents of the libraries in the folder `facts` beside the running program, where the tool side reads them.
fn shipped_beside_the_program() {
    let beside = std::env::current_exe().unwrap().with_file_name("facts");
    std::fs::create_dir_all(&beside).unwrap();
    for file in std::fs::read_dir(osprey_replay::merged_facts()).unwrap() {
        let file = file.unwrap().path();
        std::fs::copy(&file, beside.join(file.file_name().unwrap())).unwrap();
    }
}

/// The host library's machine answer under the variables as they are now, and the facts answer (the build flags) it leads to.
fn ask_now() -> (Json, String) {
    let text = tool::machine(&osprey_cpu_lib::Cpu, r#"{"target": "host", "options": {}}"#).unwrap();
    let machine = Contract::parse(&text).unwrap().0;
    let facts = tool::facts_for(&osprey_cpu_lib::Cpu, &machine).unwrap();
    (machine, facts)
}

/// Oracle: the options and the level a machine answer states are the variables' values; the flag list of the level is `-O1` for `O1`.
#[test]
fn a_variable_changed_between_two_asks_of_one_process_is_read_by_the_second() {
    shipped_beside_the_program();
    std::env::remove_var("HAWK_HOST_PROFILE");
    std::env::remove_var("HAWK_OPT_LEVEL");
    let (first, first_facts) = ask_now();
    let key = |m: &Json| m.get("key").and_then(Json::str).unwrap().to_string();
    let asked = |m: &Json| {
        let stated = m.get("stated").unwrap();
        stated.get("asked").unwrap().to_text()
    };
    std::env::set_var("HAWK_HOST_PROFILE", "native");
    std::env::set_var("HAWK_OPT_LEVEL", "O1");
    let (second, second_facts) = ask_now();
    assert_ne!(key(&first), key(&second));
    assert!(
        asked(&second).contains("\"native\"") && asked(&second).contains("O1"),
        "{}",
        asked(&second)
    );
    assert!(
        second_facts.contains("-O1") && !first_facts.contains("-O1"),
        "{second_facts}"
    );
    std::env::remove_var("HAWK_HOST_PROFILE");
    std::env::remove_var("HAWK_OPT_LEVEL");
    let (again, again_facts) = ask_now();
    assert_eq!(key(&again), key(&first));
    assert_eq!((asked(&again), again_facts), (asked(&first), first_facts));
}
