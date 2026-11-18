//! One unit built to an object with this machine's real host compiler, a set text built with it, both linked into a shared
//! library, which is loaded and called.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a panic is the failure
mod common;
use common::{member, output_of, scratch};
use std::ffi::{c_char, c_int, c_void, CString};
use toolkit::{wire, Env, HeaderTree, Toolkit};

const ANSWER: HeaderTree = HeaderTree {
    name: "answer",
    files: &[("answer.h", b"#define ANSWER 42\n")],
};
const TABLE: &str =
    r#"{"tools":{"compiler":{"rules":[{"path":"clang++"},{"path":"c++"},{"path":"g++"}]}}}"#;
const RTLD_NOW: c_int = 2;

extern "C" {
    fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}

/// The unit request for a text, with the compile template of a host unit (a dependency file included).
fn request(name: &str, text: &str, extra: Vec<(&str, rcore::json::J)>) -> String {
    let mut pairs = vec![
        ("name", wire::s(name)),
        ("text", wire::s(text)),
        (
            "template",
            wire::strs(&[
                "{compiler}",
                "-std=c++17",
                "-O1",
                "-fPIC",
                "-I{include:answer}",
                "-MD",
                "-MF",
                "{depfile}",
                "-c",
                "{source}",
                "-o",
                "{object}",
            ]),
        ),
    ];
    pairs.extend(extra);
    wire::render(&wire::obj(pairs))
}

/// Calls the `int()` function `name` of the loaded library at `path`.
fn call(path: &str, name: &str) -> i32 {
    let (lib, symbol) = (CString::new(path).unwrap(), CString::new(name).unwrap());
    // SAFETY: the library was just linked from C++ units that define `extern "C" int name()`, so the symbol has that type.
    unsafe {
        let handle = dlopen(lib.as_ptr(), RTLD_NOW);
        assert!(!handle.is_null(), "dlopen failed");
        let function = dlsym(handle, symbol.as_ptr());
        assert!(!function.is_null(), "dlsym failed");
        std::mem::transmute::<*mut c_void, extern "C" fn() -> i32>(function)()
    }
}

#[test]
fn a_unit_and_a_set_text_built_and_linked_with_the_real_compiler_load_and_run() {
    let dir = scratch("e2e");
    let tk = Toolkit::new(TABLE, &[ANSWER], Env::process()).unwrap();
    let store = dir.join("store");
    let unit = tk
        .build(
            &request(
                "answer_unit",
                "#include \"answer.h\"\nextern \"C\" int toolkit_answer() { return ANSWER; }\n",
                vec![],
            ),
            &store,
        )
        .unwrap();
    assert!(output_of(&unit).is_file());
    let set = request(
        "set_text",
        "extern \"C\" int toolkit_set() { return 7; }\n",
        vec![],
    );
    let link = format!(
        "{{\"name\":\"libset.dylib\",\"parts\":[{unit}],\"set\":{set},\"link\":{{\"linux\":[\"{{compiler}}\",\"-shared\",\"-o\",\"{{output}}\",\"{{objects}}\"],\
         \"macos\":[\"{{compiler}}\",\"-shared\",\"-o\",\"{{output}}\",\"{{objects}}\"]}}}}"
    );
    let linked = tk.link(&link, &store).unwrap();
    let library = member(&linked, "library").str().unwrap().to_string();
    assert_eq!(
        (
            call(&library, "toolkit_answer"),
            call(&library, "toolkit_set")
        ),
        (42, 7)
    );
    let again = tk
        .build(
            &request(
                "answer_unit",
                "#include \"answer.h\"\nextern \"C\" int toolkit_answer() { return ANSWER; }\n",
                vec![],
            ),
            &store,
        )
        .unwrap();
    assert!(matches!(member(&again, "hit"), rcore::json::J::Bool(true)));
    assert!(matches!(
        member(&tk.link(&link, &store).unwrap(), "hit"),
        rcore::json::J::Bool(true)
    ));
}
