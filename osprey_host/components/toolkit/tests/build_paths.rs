//! The device path with fake tools (this machine has no GPU tools: the image compile, the figures of its report and the embedding
//! of the image are exercised against scripts), and the named refusals of a build: budget, timeout, memory, failure.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a panic is the failure
mod common;
use common::{fake_compiler, member, one_tool_table, output_of, runs, scratch, script, unit};
use rcore::json::J;
use std::fs;
use std::path::PathBuf;
use toolkit::{wire, Env, Kind, Toolkit};

const TABLE: &str = r#"{"tools":{
  "compiler":{"rules":[{"env":"FAKE_NVCC","required":true}]},
  "host_compiler":{"rules":[{"env":"FAKE_HOST","required":true}]}}}"#;

/// The embed recipe of the device facts, read from the file.
fn recipe() -> J {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../osprey_cuda/facts/cuda_image.json");
    let card = wire::parse("cuda_image.json", &fs::read_to_string(path).unwrap()).unwrap();
    card.get("embed").cloned().unwrap()
}

/// A fake device compiler: writes an ELF-looking image after `-o` and a ptxas-style report on standard error.
fn fake_nvcc(dir: &std::path::Path) -> PathBuf {
    let path = dir.join("nvcc");
    let body = format!(
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'fake nvcc'; exit 0; fi\necho run >> \"{d}/runs\"\n\
         out=\"\"; prev=\"\"; for a in \"$@\"; do [ \"$prev\" = \"-o\" ] && out=\"$a\"; prev=\"$a\"; done\n\
         printf '\\177ELFimage' > \"$out\"\n\
         echo \"ptxas info    : Compiling entry function 'k' for 'sm_75'\" >&2\n\
         echo '    8 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' >&2\n\
         echo 'ptxas info    : Used 32 registers, 360 bytes cmem[0]' >&2\n",
        d = dir.display()
    );
    script(&path, &body);
    path
}

#[test]
fn a_device_image_is_built_embedded_and_linked_with_fake_tools_and_a_second_link_runs_nothing() {
    let dir = scratch("device");
    let (nvcc, host) = (fake_nvcc(&dir), fake_compiler(&dir, "g++"));
    let env = Env::from_pairs(&[
        ("FAKE_NVCC", nvcc.to_str().unwrap()),
        ("FAKE_HOST", host.to_str().unwrap()),
        ("PATH", "/bin:/usr/bin"),
    ]);
    let tk = Toolkit::new(TABLE, &[], env).unwrap();
    let store = dir.join("store");
    let image = wire::render(&wire::obj(vec![
        ("name", wire::s("k")),
        ("text", wire::s("__global__ void k() {}")),
        ("source_suffix", wire::s(".cu")),
        ("output_suffix", wire::s(".fatbin")),
        (
            "template",
            wire::strs(&[
                "{compiler}",
                "-fatbin",
                "-ccbin",
                "{host_compiler}",
                "{source}",
                "-o",
                "{output}",
            ]),
        ),
        ("figures", wire::s("ptxas_info")),
        (
            "embed",
            wire::obj(vec![
                ("symbol", wire::s("k__image")),
                ("size_symbol", wire::s("k__image_size")),
            ]),
        ),
    ]));
    let built = tk.build(&image, &store).unwrap();
    assert_eq!(
        wire::render(&member(&built, "figures")),
        "{\"k\":{\"regs\":32,\"stack_B\":8,\"spill_store_B\":0,\"spill_load_B\":0}}"
    );
    assert!(matches!(
        member(&built, "report").get("elf"),
        Some(J::Bool(true))
    ));
    let object = wire::render(&wire::obj(vec![
        ("name", wire::s("k_host")),
        ("text", wire::s("int k_host;")),
        (
            "template",
            wire::strs(&["{host_compiler}", "-c", "{source}", "-o", "{output}"]),
        ),
    ]));
    let host_part = tk.build(&object, &store).unwrap();
    let request = format!(
        "{{\"name\":\"lib.so\",\"parts\":[{host_part},{built}],\"as\":{{\"compiler\":\"host_compiler\"}},\
         \"link\":{{\"linux\":[\"{{compiler}}\",\"-shared\",\"-o\",\"{{output}}\",\"{{objects}}\"],\
         \"macos\":[\"{{compiler}}\",\"-shared\",\"-o\",\"{{output}}\",\"{{objects}}\"]}},\"embed\":{}}}",
        wire::render(&recipe())
    );
    let linked = tk.link(&request, &store).unwrap();
    let library = PathBuf::from(member(&linked, "library").str().unwrap());
    assert!(library.is_file());
    // nvcc once, the host object once, the assembly of the image once, the link once
    assert_eq!(runs(&dir), 4);
    let assembly = fs::read_to_string(dir.join("last.s")).unwrap();
    assert!(
        assembly.contains("k__image_size")
            && assembly.contains(&format!(".incbin \"{}\"", output_of(&built).display())),
        "{assembly}"
    );
    let again = tk.link(&request, &store).unwrap();
    assert!(matches!(member(&again, "hit"), J::Bool(true)));
    assert_eq!(runs(&dir), 4);
}

/// A toolkit over the fake of `dir` whose limits read the environment variables of the table.
fn limited(dir: &std::path::Path, extra: &[(&str, &str)]) -> Toolkit {
    let table = r#"{"tools":{"compiler":{"rules":[{"env":"FAKE_CXX","required":true}],"fix":"set FAKE_CXX"}},
        "limits":{"source_budget":{"env":"T_BUDGET","default":8388608},"timeout":{"env":"T_TIMEOUT","default":1800}}}"#;
    let fake = dir.join("fakecc").display().to_string();
    let mut pairs = vec![("FAKE_CXX", fake.as_str()), ("PATH", "/bin:/usr/bin")];
    pairs.extend_from_slice(extra);
    Toolkit::new(table, &[], Env::from_pairs(&pairs)).unwrap()
}

#[test]
fn a_source_over_the_budget_is_refused_before_any_compile() {
    let dir = scratch("budget");
    fake_compiler(&dir, "fakecc");
    let refused = limited(&dir, &[("T_BUDGET", "10")])
        .build(
            &unit("u", "int a_long_text_over_ten_bytes;"),
            &dir.join("s"),
        )
        .unwrap_err();
    assert_eq!((refused.kind, runs(&dir)), (Kind::OverBudget, 0));
    let bad = limited(&dir, &[("T_BUDGET", "ten")])
        .build(&unit("u", "int x;"), &dir.join("s"))
        .unwrap_err();
    assert!(
        bad.message
            .contains("$T_BUDGET=\"ten\" is not a non-negative integer"),
        "{}",
        bad.message
    );
}

#[test]
fn a_tool_over_its_time_is_refused_by_name_and_publishes_nothing() {
    let dir = scratch("timeout");
    fs::write(dir.join("sleep"), "20").unwrap();
    fake_compiler(&dir, "fakecc");
    let refused = limited(&dir, &[("T_TIMEOUT", "1")])
        .build(&unit("u", "int x;"), &dir.join("s"))
        .unwrap_err();
    assert_eq!(refused.kind, Kind::TimedOut);
    assert!(!dir
        .join("s")
        .read_dir()
        .unwrap()
        .flatten()
        .any(|e| e.path().join("record.json").exists()));
}

#[test]
fn a_failing_tool_is_refused_with_its_command_and_complaint_and_an_allocation_failure_names_the_cap(
) {
    let dir = scratch("failing");
    let tk = |complaint: &str, code: i32| {
        script(&dir.join("fakecc"), &format!("#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo f; exit 0; fi\necho '{complaint}' >&2\nexit {code}\n"));
        limited(&dir, &[])
    };
    let failed = tk("error: expected ;", 1)
        .build(&unit("u", "int"), &dir.join("s"))
        .unwrap_err();
    assert_eq!(failed.kind, Kind::ToolFailed);
    assert!(
        failed.message.contains("rc=1")
            && failed.message.contains("$ ")
            && failed.message.contains("expected ;")
            && failed.message.contains("set FAKE_CXX")
    );
    let memory = tk("terminate: std::bad_alloc", 1)
        .build(&unit("u", "int a;"), &dir.join("s"))
        .unwrap_err();
    assert_eq!(memory.kind, Kind::OutOfMemory);
    let missing = tk("", 127)
        .build(&unit("u", "int b;"), &dir.join("s"))
        .unwrap_err();
    assert_eq!(missing.kind, Kind::ToolAbsent);
    let quiet = tk("", 0)
        .build(&unit("u", "int c;"), &dir.join("s"))
        .unwrap_err();
    assert!(quiet.message.contains("wrote no u.o"), "{}", quiet.message);
}

#[test]
fn an_unknown_table_member_in_a_request_is_a_named_refusal() {
    let dir = scratch("badreq");
    fake_compiler(&dir, "fakecc");
    let tk = Toolkit::new(
        &one_tool_table("FAKE_CXX"),
        &[],
        Env::from_pairs(&[
            ("FAKE_CXX", dir.join("fakecc").to_str().unwrap()),
            ("PATH", "/bin"),
        ]),
    )
    .unwrap();
    assert_eq!(
        tk.build("{\"name\":\"u\"}", &dir.join("s"))
            .unwrap_err()
            .kind,
        Kind::BadRequest
    );
    assert_eq!(tk.state("nope", &dir).unwrap_err().kind, Kind::BadRequest);
}

#[test]
fn a_request_with_no_template_is_refused_by_name_and_a_tool_that_writes_no_output_is_refused_with_its_argv_and_log(
) {
    let dir = scratch("no_output");
    let silent = dir.join("silent");
    script(
        &silent,
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo silent; exit 0; fi\necho 'nothing to do' >&2\n",
    );
    let env = Env::from_pairs(&[
        ("FAKE_NVCC", silent.to_str().unwrap()),
        ("FAKE_HOST", silent.to_str().unwrap()),
        ("PATH", "/bin:/usr/bin"),
    ]);
    let tk = Toolkit::new(TABLE, &[], env).unwrap();
    let request = |template: &[&str]| {
        wire::render(&wire::obj(vec![
            ("name", wire::s("kern")),
            ("text", wire::s("int x;")),
            ("output_suffix", wire::s(".ptx")),
            ("template", wire::strs(template)),
        ]))
    };
    let none = tk.build(&request(&[]), &dir.join("store")).unwrap_err();
    assert!(
        none.message.contains("has no argv template"),
        "{}",
        none.message
    );
    let silent_run = tk
        .build(
            &request(&["{compiler}", "-ptx", "{source}"]),
            &dir.join("store"),
        )
        .unwrap_err();
    assert!(
        silent_run.message.contains("wrote no kern.ptx")
            && silent_run.message.contains("silent -ptx")
            && silent_run.message.contains("nothing to do"),
        "{}",
        silent_run.message
    );
}

/// A fake compiler that, like a real linker, fails when an object it is given does not exist.
fn fake_linker(dir: &std::path::Path) -> PathBuf {
    let path = dir.join("ld");
    script(
        &path,
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo \"fakeld 1.0\"; exit 0; fi\n\
         out=\"\"; prev=\"\"\n\
         for a in \"$@\"; do [ \"$prev\" = \"-o\" ] && out=\"$a\"; \
         case \"$a\" in *.o) [ -f \"$a\" ] || { echo \"cannot find $a\" >&2; exit 1; };; esac; prev=\"$a\"; done\n\
         echo linked > \"$out\"\n",
    );
    path
}

/// A link request of the library `name` over the set text `text` built under the unit name `set_name`.
fn set_link(name: &str, set_name: &str, text: &str) -> String {
    let set = unit(set_name, text);
    format!(
        "{{\"name\":\"{name}\",\"parts\":[],\"set\":{set},\"link\":{{\"linux\":[\"{{linker}}\",\"-shared\",\"-o\",\"{{output}}\",\"{{objects}}\"],\
         \"macos\":[\"{{linker}}\",\"-shared\",\"-o\",\"{{output}}\",\"{{objects}}\"]}}}}"
    )
}

#[test]
fn a_hit_on_an_object_built_under_another_unit_name_links_the_file_the_store_holds() {
    let dir = scratch("other_name");
    let (compiler, linker) = (fake_compiler(&dir, "cc"), fake_linker(&dir));
    let env = Env::from_pairs(&[
        ("FAKE_NVCC", compiler.to_str().unwrap()),
        ("FAKE_HOST", linker.to_str().unwrap()),
        ("PATH", "/bin:/usr/bin"),
    ]);
    let table = r#"{"tools":{
      "compiler":{"rules":[{"env":"FAKE_NVCC","required":true}]},
      "linker":{"rules":[{"env":"FAKE_HOST","required":true}]}}}"#;
    let tk = Toolkit::new(table, &[], env).unwrap();
    let store = dir.join("store");
    let (first_unit, second_unit) = (
        tk.build(&unit("first_set", "int helper;"), &store).unwrap(),
        tk.build(&unit("second_set", "int helper;"), &store)
            .unwrap(),
    );
    // the same job under two names is one slot, and the second answer names the file the slot holds
    assert_eq!(output_of(&second_unit), output_of(&first_unit));
    assert!(output_of(&second_unit).is_file());
    let first = tk
        .link(&set_link("one.so", "one__set", "int set_text;"), &store)
        .unwrap();
    let second = tk
        .link(&set_link("two.so", "two__set", "int set_text;"), &store)
        .unwrap();
    // the second link finds the first set object and the first library; neither path is composed from the second names
    let (first_library, second_library) = (
        PathBuf::from(member(&first, "library").str().unwrap()),
        PathBuf::from(member(&second, "library").str().unwrap()),
    );
    assert!(second_library.is_file());
    assert_eq!(second_library, first_library);
}
