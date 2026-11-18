//! The in-process compile against a FAKE library built here that implements the same C entry points as a real runtime compiler
//! (the real library is not on this machine, so the real path is unproven here): selection by the table when no offline compiler is
//! found, the program calls, headers served from memory, the same store and refusals, and the figures read off the image.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a panic is the failure
mod common;
use common::{build_shared, elf_image, member, output_of, runs, scratch, script};
use rcore::json::J;
use std::fs;
use std::path::Path;
use toolkit::{wire, Env, HeaderTree, Kind, Toolkit};

const DEVICE: HeaderTree = HeaderTree {
    name: "device",
    files: &[("lib/one.h", b"// one\n"), ("two.h", b"// two\n")],
};

/// The fake runtime library: a program remembers its source, name and header names; the PTX image echoes them with the options, the
/// CUBIN image is `elf`; a source that contains `SYNTAX_ERROR` fails to compile with a log; every compile appends a line to `trace`.
fn fake_source(elf: &[u8], trace: &Path) -> String {
    let bytes: Vec<String> = elf.iter().map(ToString::to_string).collect();
    format!(
        r#"
#include <cstdio>
#include <cstring>
#include <string>
struct Program {{ std::string src, name, headers, options; bool failed = false; }};
static const unsigned char CUBIN[] = {{{bytes}}};
extern "C" {{
int fakeCreate(Program** p, const char* src, const char* name, int n, const char* const* texts, const char* const* names) {{
  *p = new Program{{src, name, "", "", false}};
  for (int i = 0; i < n; ++i) {{ (*p)->headers += names[i]; (*p)->headers += "="; (*p)->headers += texts[i]; }}
  return 0; }}
int fakeCompile(Program* p, int n, const char* const* opts) {{
  for (int i = 0; i < n; ++i) {{ p->options += opts[i]; p->options += " "; }}
  if (FILE* f = fopen("{trace}", "a")) {{ fputs("compile\n", f); fclose(f); }}
  p->failed = p->src.find("SYNTAX_ERROR") != std::string::npos;
  return p->failed ? 6 : 0; }}
int fakeLogSize(Program* p, size_t* n) {{ *n = (p->failed ? strlen("fake: syntax error") : 0) + 1; return 0; }}
int fakeLog(Program* p, char* out) {{ strcpy(out, p->failed ? "fake: syntax error" : ""); return 0; }}
static std::string ptx(Program* p) {{ return "// ptx " + p->name + "\n// options " + p->options + "\n// headers " + p->headers + "\n"; }}
int fakePtxSize(Program* p, size_t* n) {{ *n = ptx(p).size() + 1; return 0; }}
int fakePtx(Program* p, char* out) {{ strcpy(out, ptx(p).c_str()); return 0; }}
int fakeCubinSize(Program*, size_t* n) {{ *n = sizeof CUBIN; return 0; }}
int fakeCubin(Program*, char* out) {{ memcpy(out, CUBIN, sizeof CUBIN); return 0; }}
int fakeDestroy(Program** p) {{ delete *p; *p = nullptr; return 0; }}
int fakeVersion(int* a, int* b) {{ *a = 12; *b = 6; return 0; }}
int fakeArchCount(int* n) {{ *n = 3; return 0; }}
int fakeArchList(int* out) {{ out[0] = 61; out[1] = 75; out[2] = 80; return 0; }}
}}
"#,
        bytes = bytes.join(","),
        trace = trace.display()
    )
}

/// A table with no offline compiler rule that can match, the runtime library `library` standing in for the tool `compiler`.
fn table(library: &str) -> String {
    format!(
        r#"{{"tools":{{"compiler":{{"rules":[{{"env":"FAKE_NVCC","required":false}},{{"path":"no-such-offline-compiler"}}],"fix":"install the toolkit"}}}},
          "runtime_compile":{{"replaces":"compiler","libraries":["{library}"],"force":{{"env":"FORCE","tool":"offline","runtime":"runtime"}},
            "calls":{{"version":"fakeVersion","arch_count":"fakeArchCount","arch_list":"fakeArchList","create":"fakeCreate","compile":"fakeCompile",
              "log_size":"fakeLogSize","log":"fakeLog","destroy":"fakeDestroy",
              "images":{{"ptx":{{"size":"fakePtxSize","get":"fakePtx","text":true}},"cubin":{{"size":"fakeCubinSize","get":"fakeCubin"}}}}}}}}}}"#
    )
}

/// A build request for the runtime path.
fn request(text: &str, target: &str) -> String {
    wire::render(&wire::obj(vec![
        ("name", wire::s("k")),
        ("text", wire::s(text)),
        ("source_suffix", wire::s(".cu")),
        (
            "output_suffix",
            wire::s(if target == "ptx" { ".ptx" } else { ".cubin" }),
        ),
        ("figures", wire::s("image_resources")),
        (
            "runtime",
            wire::obj(vec![
                ("options", wire::strs(&["-std=c++20", "-arch={arch}"])),
                ("target", wire::s(target)),
                ("headers", wire::strs(&["device"])),
            ]),
        ),
        ("places", wire::obj(vec![("arch", wire::s("compute_75"))])),
    ]))
}

struct Fixture {
    dir: std::path::PathBuf,
    library: String,
    trace: std::path::PathBuf,
}

fn fixture(name: &str) -> Fixture {
    let dir = scratch(name);
    let trace = dir.join("trace");
    let library = build_shared(
        &dir,
        "fakenvrtc",
        &fake_source(&elf_image(&[("kern", 40, 8)]), &trace),
    )
    .display()
    .to_string();
    Fixture {
        dir,
        library,
        trace,
    }
}

fn toolkit(f: &Fixture, extra: &[(&str, &str)]) -> Toolkit {
    let mut pairs = vec![("PATH", "/bin:/usr/bin")];
    pairs.extend_from_slice(extra);
    Toolkit::new(&table(&f.library), &[DEVICE], Env::from_pairs(&pairs)).unwrap()
}

#[test]
fn with_no_offline_compiler_the_machine_states_the_runtime_library_instead() {
    let f = fixture("rt-machine");
    let answer = wire::parse("a", &toolkit(&f, &[]).machine("{}").unwrap()).unwrap();
    let stated = answer.get("stated").unwrap();
    assert!(stated.get("tools").unwrap().get("compiler").is_none());
    let runtime = stated.get("runtime").unwrap();
    assert_eq!(runtime.get("version").and_then(J::str), Some("12.6"));
    assert_eq!(wire::render(runtime.get("archs").unwrap()), "[61,75,80]");
}

#[test]
fn a_runtime_build_serves_headers_from_memory_and_a_second_build_compiles_nothing() {
    let f = fixture("rt-build");
    let tk = toolkit(&f, &[]);
    let store = f.dir.join("store");
    let first = tk
        .build(&request("__global__ void k() {}", "ptx"), &store)
        .unwrap();
    let ptx = fs::read_to_string(output_of(&first)).unwrap();
    assert!(ptx.starts_with("// ptx k\n// options -std=c++20 -arch=compute_75 \n// headers lib/one.h=// one\ntwo.h=// two\n"), "{ptx}");
    assert!(
        !ptx.contains('\0'),
        "the NUL that ends a text image is not kept"
    );
    let second = tk
        .build(&request("__global__ void k() {}", "ptx"), &store)
        .unwrap();
    assert!(matches!(member(&second, "hit"), J::Bool(true)));
    assert_eq!(fs::read_to_string(&f.trace).unwrap().lines().count(), 1);
    // another text is another unit
    tk.build(&request("__global__ void k2() {}", "ptx"), &store)
        .unwrap();
    assert_eq!(fs::read_to_string(&f.trace).unwrap().lines().count(), 2);
}

#[test]
fn a_binary_image_is_kept_whole_and_its_register_and_frame_figures_are_read_off_it() {
    let f = fixture("rt-cubin");
    let answer = toolkit(&f, &[])
        .build(
            &request("__global__ void kern() {}", "cubin"),
            &f.dir.join("store"),
        )
        .unwrap();
    assert_eq!(
        fs::read(output_of(&answer)).unwrap(),
        elf_image(&[("kern", 40, 8)])
    );
    assert_eq!(
        wire::render(&member(&answer, "figures")),
        "{\"kern\":{\"regs\":40,\"stack_B\":8,\"spill_store_B\":null,\"spill_load_B\":null}}"
    );
    assert!(matches!(
        member(&answer, "report").get("elf"),
        Some(J::Bool(true))
    ));
}

#[test]
fn a_failed_runtime_compile_is_refused_with_its_log_and_publishes_nothing() {
    let f = fixture("rt-fail");
    let tk = toolkit(&f, &[]);
    let refused = tk
        .build(&request("SYNTAX_ERROR", "ptx"), &f.dir.join("store"))
        .unwrap_err();
    assert_eq!(refused.kind, Kind::ToolFailed);
    assert!(
        refused.message.contains("fake: syntax error")
            && refused.message.contains("-arch=compute_75"),
        "{}",
        refused.message
    );
    assert!(!f
        .dir
        .join("store")
        .read_dir()
        .unwrap()
        .flatten()
        .any(|e| e.path().join("record.json").exists()));
}

#[test]
fn the_environment_chooses_between_the_offline_tool_and_the_library() {
    let f = fixture("rt-force");
    let nvcc = f.dir.join("nvcc");
    script(&nvcc, &format!("#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'fake'; exit 0; fi\necho run >> \"{}/runs\"\necho tool > \"$2\"\n", f.dir.display()));
    let both = [("FAKE_NVCC", nvcc.to_str().unwrap())];
    let offline = wire::parse("a", &toolkit(&f, &both).machine("{}").unwrap()).unwrap();
    assert!(offline
        .get("stated")
        .unwrap()
        .get("tools")
        .unwrap()
        .get("compiler")
        .is_some());
    assert!(offline.get("stated").unwrap().get("runtime").is_none());
    let forced = [("FAKE_NVCC", nvcc.to_str().unwrap()), ("FORCE", "runtime")];
    let tk = toolkit(&f, &forced);
    tk.build(
        &request("__global__ void k() {}", "ptx"),
        &f.dir.join("store"),
    )
    .unwrap();
    assert_eq!(runs(&f.dir), 0, "the forced runtime compile ran no tool");
    // forced to the offline tool with none found: absent, naming the tools and the library tried
    let none = toolkit(&f, &[("FORCE", "offline")]);
    let answer = wire::parse("a", &none.machine("{}").unwrap()).unwrap();
    assert!(answer
        .get("tools_absent")
        .and_then(J::str)
        .unwrap()
        .contains("no compiler found"));
    assert!(toolkit(&f, &[("FORCE", "neither")])
        .machine("{}")
        .unwrap_err()
        .message
        .contains("must be \"offline\" or \"runtime\""));
}

#[test]
fn with_neither_a_tool_nor_the_library_the_absence_names_everything_tried() {
    let f = fixture("rt-none");
    let tk = Toolkit::new(
        &table("libno-such-runtime.so"),
        &[DEVICE],
        Env::from_pairs(&[("PATH", "/bin")]),
    )
    .unwrap();
    let answer = wire::parse("a", &tk.machine("{}").unwrap()).unwrap();
    let reason = answer.get("tools_absent").and_then(J::str).unwrap();
    assert!(reason.contains("no compiler found. Tried, in order: $FAKE_NVCC (unset), no-such-offline-compiler on PATH (absent)"), "{reason}");
    assert!(reason.contains("libno-such-runtime.so"), "{reason}");
    let _ = f;
}

/// A runtime request whose only option is `-I{include:cccl}`, over a toolkit whose table names the include folders beside the
/// library: `include` first (absent from the fixture unless planted), then `lib`.
fn include_toolkit_and_request(f: &Fixture) -> (Toolkit, String) {
    let table = table(&f.library).replace(
        r#""calls":"#,
        r#""include":{"cccl":{"marker":"cuda/std/version","beside":["include","lib"]}},"calls":"#,
    );
    let toolkit = Toolkit::new(
        &table,
        &[DEVICE],
        Env::from_pairs(&[("PATH", "/bin:/usr/bin")]),
    )
    .unwrap();
    let request = wire::render(&wire::obj(vec![
        ("name", wire::s("k")),
        ("text", wire::s("int x;")),
        ("source_suffix", wire::s(".cu")),
        ("output_suffix", wire::s(".ptx")),
        (
            "runtime",
            wire::obj(vec![
                ("options", wire::strs(&["-I{include:cccl}"])),
                ("target", wire::s("ptx")),
                ("headers", wire::strs(&["device"])),
            ]),
        ),
    ]));
    (toolkit, request)
}

#[test]
fn an_include_folder_of_the_table_is_found_beside_the_loaded_library_and_named_in_the_options() {
    let f = fixture("rt-include");
    let (toolkit, request) = include_toolkit_and_request(&f);
    let missing = toolkit.build(&request, &f.dir.join("store")).unwrap_err();
    assert_eq!(missing.kind, Kind::ToolAbsent);
    assert!(
        missing.message.contains("cuda/std/version") && missing.message.contains("include, lib"),
        "{}",
        missing.message
    );
    fs::create_dir_all(f.dir.join("lib/cuda/std")).unwrap();
    fs::write(f.dir.join("lib/cuda/std/version"), "").unwrap();
    let built = toolkit.build(&request, &f.dir.join("store")).unwrap();
    let image = fs::read_to_string(output_of(&built)).unwrap();
    assert!(
        image.contains(&format!(
            "// options -I{}/lib ",
            f.dir.canonicalize().unwrap().display()
        )),
        "{image}"
    );
}
