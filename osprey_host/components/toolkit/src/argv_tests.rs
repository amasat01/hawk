//! The argv the build instructions of a unit template card fill to, against argv written out by hand.
use super::*;
use crate::discover::Env;
use crate::headers::HeaderTree;
use std::path::PathBuf;

const FACTS: HeaderTree = HeaderTree {
    name: "facts",
    files: &[("facts.h", b"")],
};
const SERVED: HeaderTree = HeaderTree {
    name: "served",
    files: &[("a.h", b"")],
};
const AETHER: HeaderTree = HeaderTree {
    name: "aether",
    files: &[("b.h", b"")],
};
const EAGLE: HeaderTree = HeaderTree {
    name: "eagle",
    files: &[("c.h", b"")],
};

/// The members of a unit template card a template is made of, read from the fixture `tests/data/unit_template_card.json` (the shape
/// of a backend's build card, as the toolkit fills it).
fn card() -> J {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/unit_template_card.json");
    wire::parse(
        "unit_template_card.json",
        &fs::read_to_string(path).unwrap(),
    )
    .unwrap()
}

/// A member of the card that is a list of texts.
fn list(card: &J, key: &str) -> Vec<String> {
    card.get(key).unwrap().strs()
}

/// The unit template the library states for a vector-math profile under `shape` at O3 with the instruction set flags `isa`: the
/// pieces in the order the facts state them.
fn unit_template(card: &J, shape: &str, isa: &[&str]) -> Vec<String> {
    let contraction = card
        .get("contraction")
        .unwrap()
        .get("native-vector-math")
        .and_then(J::str)
        .unwrap();
    let mut t = vec![
        "{compiler}".to_string(),
        "-O3".to_string(),
        contraction.to_string(),
    ];
    t.extend(list(card, "always"));
    t.extend(isa.iter().map(|s| (*s).to_string()));
    t.extend(list(card, "unit_tail"));
    let includes = list(card.get("includes").unwrap(), shape);
    let facts: Vec<String> = [list(card, "defines"), list(card, "facts_include")].concat();
    if shape == "roots" {
        t.extend(includes);
        t.extend(facts);
    } else {
        t.extend(facts);
        t.extend(includes);
    }
    t.extend(list(card, "dependency"));
    t.extend(list(card, "compile_only"));
    t
}

fn toolkit(trees: &[HeaderTree]) -> Toolkit {
    let spec = r#"{"tools":{"compiler":{"rules":[{"env":"CXX_FOR_TEST","required":true}]}}}"#;
    Toolkit::new(
        spec,
        trees,
        Env::from_pairs(&[("CXX_FOR_TEST", "/usr/bin/true"), ("PATH", "/usr/bin:/bin")]),
    )
    .unwrap()
}

fn job(template: &[String], extra: &[&str]) -> Job {
    let request = wire::obj(vec![
        ("name", wire::s("u")),
        ("text", wire::s("")),
        ("template", wire::strs(template)),
        (
            "lists",
            wire::obj(vec![("{defines}", wire::strs(&["-DK=1"]))]),
        ),
        ("extra", wire::strs(extra)),
    ]);
    Job::from_request(&request).unwrap()
}

fn strings(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| (*p).to_string()).collect()
}

#[test]
fn a_served_unit_template_fills_to_the_hand_written_argv() {
    let root = std::env::temp_dir().join(format!("toolkit-argv-a-{}", std::process::id()));
    let tk = toolkit(&[FACTS, SERVED]);
    let template = unit_template(&card(), "served", &["-mnative-flag"]);
    let argv = tk
        .argv_of(
            &job(&template, &["-g"]),
            &tk.tools(None).unwrap(),
            &root,
            Path::new("/s"),
        )
        .unwrap();
    let (facts, served) = (
        root.join("headers").join(FACTS.digest()),
        root.join("headers").join(SERVED.digest()),
    );
    let expected = strings(&[
        "/usr/bin/true",
        "-O3",
        "-ffp-contract=fast",
        "-fno-math-errno",
        "-mnative-flag",
        "-std=c++23",
        "-fPIC",
        "-DAETHER_CPP_MODE",
        "-DK=1",
        &format!("-I{}", facts.display()),
        &format!("-I{}", served.display()),
        "-MD",
        "-MF",
        "/s/u.d",
        "-c",
        "/s/u.cpp",
        "-o",
        "/s/u.o",
        "-g",
    ]);
    assert_eq!(argv, expected);
}

#[test]
fn a_roots_unit_template_puts_both_roots_before_the_defines_and_the_facts_include() {
    let root = std::env::temp_dir().join(format!("toolkit-argv-b-{}", std::process::id()));
    let tk = toolkit(&[FACTS, AETHER, EAGLE]);
    let template = unit_template(&card(), "roots", &["-mnative-flag"]);
    let argv = tk
        .argv_of(
            &job(&template, &[]),
            &tk.tools(None).unwrap(),
            &root,
            Path::new("/s"),
        )
        .unwrap();
    let at = |t: &HeaderTree| root.join("headers").join(t.digest()).display().to_string();
    let expected = strings(&[
        "/usr/bin/true",
        "-O3",
        "-ffp-contract=fast",
        "-fno-math-errno",
        "-mnative-flag",
        "-std=c++23",
        "-fPIC",
        "-DAETHER_CPP_MODE",
        "-I",
        &at(&AETHER),
        "-I",
        &at(&EAGLE),
        "-DK=1",
        &format!("-I{}", at(&FACTS)),
        "-MD",
        "-MF",
        "/s/u.d",
        "-c",
        "/s/u.cpp",
        "-o",
        "/s/u.o",
    ]);
    assert_eq!(argv, expected);
}

#[test]
fn a_link_template_fills_one_element_for_each_object() {
    let tk = toolkit(&[]);
    let objects = [PathBuf::from("/o/a.o"), PathBuf::from("/o/b.o")];
    let template = list(card().get("link").unwrap(), "macos");
    let template = [vec!["{compiler}".to_string()], template].concat();
    let link = Job::linking(
        &wire::obj(vec![]),
        "lib.dylib",
        &template,
        &objects,
        &strings(&["k1", "k2"]),
    )
    .unwrap();
    let argv = tk
        .argv_of(
            &link,
            &tk.tools(None).unwrap(),
            Path::new("/r"),
            Path::new("/s"),
        )
        .unwrap();
    assert_eq!(
        argv,
        strings(&[
            "/usr/bin/true",
            "-shared",
            "-o",
            "/s/lib.dylib",
            "/o/a.o",
            "/o/b.o"
        ])
    );
}

#[test]
fn a_place_nobody_fills_is_refused_by_name() {
    let tk = toolkit(&[]);
    let template = strings(&["{compiler}", "-I{include:nowhere}"]);
    let refused = tk
        .argv_of(
            &job(&template, &[]),
            &tk.tools(None).unwrap(),
            Path::new("/r"),
            Path::new("/s"),
        )
        .unwrap_err();
    assert!(
        refused.message.contains("{include:nowhere}"),
        "{}",
        refused.message
    );
}

#[test]
fn a_define_in_a_list_moves_the_key_of_the_build() {
    let tk = toolkit(&[FACTS, SERVED]);
    let machine = tk.tools(None).unwrap();
    let with = |defines: &[&str]| {
        let request = wire::obj(vec![
            ("name", wire::s("u")),
            ("text", wire::s("")),
            ("template", wire::strs(&["{compiler}", "{defines}"])),
            ("lists", wire::obj(vec![("{defines}", wire::strs(defines))])),
        ]);
        tk.job_key(&Job::from_request(&request).unwrap(), &machine)
    };
    assert_ne!(with(&["-DHAWK_FAST_ENTRIES=1"]), with(&[]));
    assert_eq!(
        with(&["-DHAWK_FAST_ENTRIES=1"]),
        with(&["-DHAWK_FAST_ENTRIES=1"])
    );
}
