//! The tool side of the libraries as a caller that knows nothing of the target uses it: a device unit's text is written on a machine
//! with no device compiler (only a build needs one), and the defines a host unit was compiled with reach the link of its set and the
//! key of the build.
#![cfg(feature = "old-contract")]
mod common;
mod fakes;
use fakes::{script, shipped_beside_the_program, store};
use osprey_lowering::tool;
use rcore::json::{self, J};

/// The host library's machine answer for this machine, none when it has no compiler.
fn host_machine(root: &std::path::Path) -> Option<String> {
    let ask = format!(
        r#"{{"target": "host", "options": {{}}, "opt_level": "O1", "root": "{}"}}"#,
        root.display()
    );
    let answer = tool::machine(&osprey_cpu_lib::Cpu, &ask).unwrap();
    let parsed = json::parse(&answer).unwrap();
    match parsed.get("tools_absent") {
        Some(J::Str(_)) => None,
        _ => Some(answer),
    }
}

/// The host library's build of one unit whose text is `text`, compiled with `defines`: the answer's text.
fn build_unit(machine: &str, text: &str, defines: &[&str], root: &std::path::Path) -> String {
    let list: Vec<String> = defines.iter().map(|d| format!("{d:?}")).collect();
    let request = format!(
        r#"{{"name": "u", "text": {text:?}, "machine": {machine}, "defines": [{}]}}"#,
        list.join(", ")
    );
    tool::build(&osprey_cpu_lib::Cpu, &request, root).unwrap()
}

/// Oracle: the compiler's own refusal of a mark nobody defined. A unit compiled with `MARK=7` makes a set text that reads `MARK`
/// link; the same set text is refused when its part carries no define.
#[test]
fn the_defines_of_a_unit_reach_the_compile_of_its_set() {
    let root = store("link");
    let Some(machine) = host_machine(&root) else {
        return common::skipped(None, "the machine has no host compiler");
    };
    let unit = "extern \"C\" int u() { return 1; }\n";
    let set = "#ifndef MARK\n#error the set was compiled without the define of its part\n#endif\nextern \"C\" int s() { return MARK; }\n";
    let linked = |defines: &[&str], name: &str| {
        let part = build_unit(&machine, unit, defines, &root);
        let parts = format!("[{part}]");
        tool::link(&osprey_cpu_lib::Cpu, &parts, set, name, &root)
    };
    assert!(linked(&["MARK=7"], "with.dylib").is_ok());
    let why = linked(&[], "without.dylib").unwrap_err();
    assert!(why.message.contains("MARK"), "{}", why.message);
}

/// Oracle: two builds of one text that differ only in their defines are two compiler runs under two keys, and the same defines again
/// is a store hit.
#[test]
fn builds_that_differ_only_in_their_defines_have_two_keys() {
    let root = store("key");
    let Some(machine) = host_machine(&root) else {
        return common::skipped(None, "the machine has no host compiler");
    };
    let unit = "extern \"C\" int u() { return 1; }\n";
    let key = |answer: &str| {
        let parsed = json::parse(answer).unwrap();
        parsed.get("key").and_then(J::str).unwrap().to_string()
    };
    let hit = |answer: &str| {
        let parsed = json::parse(answer).unwrap();
        matches!(parsed.get("hit"), Some(J::Bool(true)))
    };
    let first = build_unit(&machine, unit, &["A=1"], &root);
    let other = build_unit(&machine, unit, &["A=2"], &root);
    let again = build_unit(&machine, unit, &["A=1"], &root);
    assert!(
        !hit(&first) && !hit(&other),
        "two defines, two compiler runs"
    );
    assert_ne!(key(&first), key(&other));
    assert!(hit(&again) && key(&again) == key(&first));
}

/// A device library states the machine of a host with no device compiler and still gives the facts of the text it writes: the
/// release its card names, with that release's PTX version; a build on that machine is refused as a missing tool.
#[test]
fn a_device_text_is_written_under_the_cards_release_on_a_machine_with_no_device_compiler() {
    shipped_beside_the_program();
    let machine = r#"{"key": "k", "stated": {"asked": {"opt_level": "", "options": {}}}, "tools_absent": "no device compiler found"}"#;
    let parsed = aether_writers::contract::Contract::parse(machine)
        .unwrap()
        .0;
    let facts = tool::facts_for(&osprey_cuda_lib::Cuda, &parsed).unwrap();
    let answer = json::parse(&facts).unwrap();
    let build = answer.get("device_build").unwrap();
    assert_eq!(build.get("compiler").and_then(J::str), Some("nvcc 12.6"));
    assert_eq!(build.get("ptx_isa").map(J::int), Some(85));
    let request = format!(r#"{{"name": "u", "text": "", "machine": {machine}}}"#);
    let why = tool::build(&osprey_cuda_lib::Cuda, &request, &store("absent")).unwrap_err();
    assert!(
        why.message.contains("no device compiler"),
        "{}",
        why.message
    );
}

/// A stated device machine: the compiler release (none: the runtime compile library stands in at 12.6) and the driver release.
fn stated_device(compiler: Option<&str>, driver: &str) -> String {
    let tools = match compiler {
        Some(release) => format!(
            r#"{{"compiler": {{"identity": "/usr/local/cuda/bin/nvcc||Cuda compilation tools, release {release}, V{release}.1||0", "release": "{release}"}}}}"#
        ),
        None => "{}".to_string(),
    };
    let runtime = if compiler.is_none() {
        r#""runtime": {"library": "libnvrtc.so||runtime 12.6", "version": "12.6"},"#
    } else {
        ""
    };
    format!(
        r#"{{"key": "k", "stated": {{"tools": {tools}, {runtime} "device": {{"arch": "sm_61", "driver_release": "{driver}"}}, "asked": {{"opt_level": "", "options": {{}}}}}}, "tools_absent": null}}"#
    )
}

/// The lowering request `with_facts` gets for a stated device machine, naming `format` when given.
fn lower_request(machine: &str, format: Option<&str>) -> String {
    let named = format.map_or(String::new(), |f| format!(r#""image_format": "{f}", "#));
    format!(r#"{{{named}"machine": {{"Device": {machine}}}}}"#)
}

/// The text of the member `key` of the JSON record `text`.
fn member_text(text: &str, key: &str) -> Option<String> {
    json::parse(text)
        .unwrap()
        .get(key)
        .and_then(J::str)
        .map(str::to_string)
}

/// A device compiler newer than the driver gets the image format Cubin when the request names none, and a request that names Ptx
/// on it is refused by name; a driver as new as the compiler takes Ptx either way.
#[test]
fn a_ptx_the_driver_cannot_load_is_refused_and_an_unnamed_format_is_the_one_that_loads() {
    shipped_beside_the_program();
    let newer = stated_device(Some("12.9"), "12.6");
    let unnamed = tool::with_facts(&osprey_cuda_lib::Cuda, &lower_request(&newer, None)).unwrap();
    assert_eq!(
        member_text(&unnamed, "image_format").as_deref(),
        Some("Cubin")
    );
    let why =
        tool::with_facts(&osprey_cuda_lib::Cuda, &lower_request(&newer, Some("Ptx"))).unwrap_err();
    assert!(
        why.reason.contains("12.9") && why.reason.contains("12.6") && why.reason.contains("Ptx"),
        "{}",
        why.reason
    );
    let named = tool::with_facts(
        &osprey_cuda_lib::Cuda,
        &lower_request(&newer, Some("Cubin")),
    )
    .unwrap();
    assert_eq!(
        member_text(&named, "image_format").as_deref(),
        Some("Cubin")
    );
    let level = stated_device(Some("12.6"), "12.6");
    for format in [None, Some("Ptx")] {
        let text =
            tool::with_facts(&osprey_cuda_lib::Cuda, &lower_request(&level, format)).unwrap();
        assert_eq!(member_text(&text, "image_format").as_deref(), Some("Ptx"));
    }
}

/// With no offline compiler and the runtime compile library present, the facts name the runtime compiler and state its own option
/// list: virtual arch for PTX, the physical one for a CUBIN, no optimisation level and no assembler flags.
#[test]
fn the_runtime_compiler_states_its_own_label_and_options() {
    shipped_beside_the_program();
    let machine = aether_writers::contract::Contract::parse(&stated_device(None, "12.6"))
        .unwrap()
        .0;
    let facts = aether_writers::contract::Contract::parse(
        &tool::facts_for(&osprey_cuda_lib::Cuda, &machine).unwrap(),
    )
    .unwrap()
    .0;
    let build = facts.get("device_build").unwrap();
    assert_eq!(
        build.get("compiler").and_then(|c| c.str()),
        Some("nvrtc 12.6")
    );
    assert_eq!(
        build.get("compile_with").and_then(|c| c.str()),
        Some("runtime")
    );
    let flags = |form: &str| build.get("flags").unwrap().get(form).unwrap().to_text();
    let inf = "-D__builtin_inf()=__longlong_as_double(0x7ff0000000000000LL)";
    assert_eq!(
        flags("ptx"),
        format!(r#"["-std=c++20","-arch=compute_61","-default-device","{inf}"]"#)
    );
    assert_eq!(
        flags("cubin"),
        format!(r#"["-std=c++20","-arch=sm_61","-default-device","{inf}"]"#)
    );
    assert!(build.get("flags").unwrap().get("fatbin").is_none());
    let options = build.get("compile").unwrap().to_text();
    assert!(
        options.ends_with(r#""-I{include:cccl}","{defines}"]"#)
            && !options.contains("-O")
            && !options.contains("Xptxas"),
        "{options}"
    );
}

/// A device unit built through the door with a fake device compiler (proven against fakes, not a real nvcc): the machine states
/// the compiler it found, the build writes the image the facts name, publishes it with the recipe the host link takes back, states
/// the figures the compiler reported, and a second build runs no compiler.
#[test]
fn a_device_unit_is_built_through_the_door_with_the_found_compiler_and_a_second_build_runs_nothing()
{
    let root = store("device_unit");
    let tools = root.join("tools");
    let runs = tools.join("runs");
    script(
        &tools.join("nvcc"),
        &format!(
            "#!/bin/sh\ncase \"$1\" in --version) echo 'Cuda compilation tools, release 12.6, V12.6.77'; exit 0;; \
             --list-gpu-arch) printf 'compute_61\\ncompute_75\\n'; exit 0;; esac\necho run >> {runs}\n\
             out=\"\"; src=\"\"; prev=\"\"; for a in \"$@\"; do [ \"$prev\" = \"-o\" ] && out=\"$a\"; \
             case \"$a\" in *.cu) src=\"$a\";; esac; prev=\"$a\"; done\n\
             frame=8; grep -q HUGE_FRAME \"$src\" && frame=600000\n\
             printf '\\177ELFimage' > \"$out\"\n\
             echo \"ptxas info    : Compiling entry function 'k' for 'sm_61'\" >&2\n\
             echo \"    $frame bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\" >&2\n\
             echo 'ptxas info    : Used 32 registers, 360 bytes cmem[0]' >&2\n",
            runs = runs.display()
        ),
    );
    script(&tools.join("host/g++"), "#!/bin/sh\necho 'g++ fake'\n");
    // the library reads the process environment once, when its tool side is first used
    std::env::set_var("HAWK_NVCC", tools.join("nvcc"));
    std::env::set_var("HAWK_NVCC_CCBIN", tools.join("host"));
    let ask = r#"{"target": "cuda", "options": {"arch": "sm_61"}, "opt_level": ""}"#;
    let machine = tool::machine(&osprey_cuda_lib::Cuda, ask).unwrap();
    assert_eq!(member_text(&machine, "tools_absent"), None, "{machine}");
    let request = format!(
        r#"{{"name": "k", "text": "__global__ void k() {{}}", "machine": {machine}, "defines": ["WIDTH=4"]}}"#
    );
    let built = tool::build(&osprey_cuda_lib::Cuda, &request, &root).unwrap();
    let answer = json::parse(&built).unwrap();
    // no driver is stated here, so the image is the one that loads whatever the driver is
    assert_eq!(answer.get("artifact").and_then(J::str), Some("k.cubin"));
    assert_eq!(answer.get("format").and_then(J::str), Some("cubin"));
    let image = answer
        .get("files")
        .unwrap()
        .get("k.cubin")
        .and_then(J::str)
        .unwrap();
    assert!(std::fs::read(image).unwrap().starts_with(b"\x7fELF"));
    let figures = answer.get("figures").unwrap().get("k").unwrap();
    assert_eq!(figures.get("regs").map(J::int), Some(32));
    // the unit states the local memory one thread holds, how it was measured and the card's limit
    let launch = answer.get("launch").unwrap();
    assert_eq!(launch.get("local_bytes").map(J::int), Some(8));
    assert_eq!(launch.get("local_source").and_then(J::str), Some("ptxas"));
    assert_eq!(launch.get("local_limit").map(J::int), Some(524_288));
    // `figures` holds the entries of the unit and nothing else: the launch record is a member of its own
    let entries: Vec<&str> = answer
        .get("figures")
        .unwrap()
        .obj()
        .iter()
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(entries, ["k"]);
    // the link recipe names the unit the image belongs to and carries the recipe that embeds it
    let recipe = answer.get("link_recipe").unwrap();
    assert_eq!(recipe.get("unit").and_then(J::str), Some("k"));
    assert!(recipe.get("embed").unwrap().get("source").is_some());
    let again =
        json::parse(&tool::build(&osprey_cuda_lib::Cuda, &request, &root).unwrap()).unwrap();
    assert!(matches!(again.get("hit"), Some(J::Bool(true))));
    assert_eq!(std::fs::read_to_string(&runs).unwrap().lines().count(), 1);
    // a unit whose frame is past the card's limit is refused by name, with the number, the limit and the way out
    let huge = request.replace(
        "__global__ void k() {}",
        "__global__ void k() {} // HUGE_FRAME",
    );
    let why = tool::build(&osprey_cuda_lib::Cuda, &huge, &root).unwrap_err();
    assert!(
        why.message.starts_with(
            "kernel 'k': 600,000 bytes of local memory per thread (ptxas), past the device limit of 524,288; the launch would fail."
        ) && why.message.ends_with("or use a smaller n"),
        "{}",
        why.message
    );
    // a request that names the other format is refused by name
    let other = request.replace(r#""defines""#, r#""image_format": "Ptx", "defines""#);
    let why = tool::build(&osprey_cuda_lib::Cuda, &other, &root).unwrap_err();
    assert!(
        why.message.contains("names the image format Ptx"),
        "{}",
        why.message
    );
}
