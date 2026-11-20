//! A unit built for the device and the host links with its device image defined in the library, under the symbols the unit's meta
//! names, and a host-only unit links with none. The device compiler is a script that writes a fixed image; the host compiler and
//! the symbol tool are the machine's.
mod common;
mod fakes;
use fakes::{script, store};
use osprey_lowering::lower::unit_meta::image_symbols;
use osprey_lowering::tool;
use rcore::json::{self, J};
use std::path::Path;
use std::process::Command;

/// What the device compiler script writes as the image.
const IMAGE: &[u8] = b"\x7fELFimage";

extern "C" {
    fn dlopen(path: *const std::ffi::c_char, mode: i32) -> *mut std::ffi::c_void;
    fn dlsym(handle: *mut std::ffi::c_void, name: *const std::ffi::c_char)
        -> *mut std::ffi::c_void;
}

/// The member `key` of the JSON record `text` as text.
fn member_text(text: &str, key: &str) -> Option<String> {
    json::parse(text)
        .unwrap()
        .get(key)
        .and_then(J::str)
        .map(str::to_string)
}

/// The host library's machine answer, none when this machine has no host compiler.
fn host_machine(root: &Path) -> Option<String> {
    let ask = format!(
        r#"{{"target": "host", "options": {{}}, "opt_level": "O1", "root": "{}"}}"#,
        root.display()
    );
    let answer = tool::machine(&osprey_cpu_lib::Cpu, &ask).unwrap();
    member_text(&answer, "tools_absent")
        .is_none()
        .then_some(answer)
}

/// The device library's machine answer under a fake device compiler.
fn device_machine(root: &Path) -> String {
    let tools = root.join("tools");
    script(
        &tools.join("nvcc"),
        "#!/bin/sh\ncase \"$1\" in --version) echo 'Cuda compilation tools, release 12.6, V12.6.77'; exit 0;; \
         --list-gpu-arch) printf 'compute_61\\n'; exit 0;; esac\n\
         out=\"\"; prev=\"\"; for a in \"$@\"; do [ \"$prev\" = \"-o\" ] && out=\"$a\"; prev=\"$a\"; done\n\
         printf '\\177ELFimage' > \"$out\"\n",
    );
    script(&tools.join("host/g++"), "#!/bin/sh\necho 'g++ fake'\n");
    // the library reads the process environment once, when its tool side is first used
    std::env::set_var("HAWK_NVCC", tools.join("nvcc"));
    std::env::set_var("HAWK_NVCC_CCBIN", tools.join("host"));
    let ask = r#"{"target": "cuda", "options": {"arch": "sm_61"}, "opt_level": ""}"#;
    tool::machine(&osprey_cuda_lib::Cuda, ask).unwrap()
}

/// The answer of a build of the unit `k` by `library` with the text `text` on `machine`.
fn built<B: osprey_lowering::backend::Backend + Default>(
    library: &B,
    machine: &str,
    text: &str,
    root: &Path,
) -> String {
    let request = format!(r#"{{"name": "k", "text": {text:?}, "machine": {machine}}}"#);
    tool::build(library, &request, root).unwrap()
}

/// The names `nm` lists as defined (not undefined) in the library, without the platform's leading underscore.
fn defined_symbols(library: &str) -> Vec<String> {
    let tool = std::env::var("NM").unwrap_or_else(|_| "nm".into());
    let out = Command::new(tool).args(["-g", library]).output().unwrap();
    assert!(out.status.success(), "nm failed on {library}");
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .filter_map(|line| {
            let mut words = line.split_whitespace().rev();
            let (name, kind) = (words.next()?, words.next()?);
            (kind != "U").then(|| name.trim_start_matches('_').to_string())
        })
        .collect()
}

const SET: &str = "extern \"C\" int s(void) { return 2; }\n";
const HOST_UNIT: &str = "extern \"C\" int k_host(void) { return 1; }\n";

/// Oracle: `nm` lists the two image symbols as defined, and the library's own bytes under them are the image the device compiler
/// wrote, with the length the size symbol states. The names are those the unit's meta carries (`image_symbols`).
#[test]
fn a_unit_with_a_device_image_defines_it_under_the_symbols_its_meta_names() {
    let root = store("device_link");
    let Some(host) = host_machine(&root) else {
        return common::skipped(None, "the machine has no host compiler");
    };
    let device = device_machine(&root);
    let parts = format!(
        "[{},{}]",
        built(&osprey_cpu_lib::Cpu, &host, HOST_UNIT, &root),
        built(
            &osprey_cuda_lib::Cuda,
            &device,
            "__global__ void k() {}",
            &root
        )
    );
    let linked = tool::link(&osprey_cpu_lib::Cpu, &parts, SET, "with_image.so", &root).unwrap();
    let library = member_text(&linked, "library").unwrap();
    let (image, size) = image_symbols("k");
    let defined = defined_symbols(&library);
    assert!(
        defined.contains(&image) && defined.contains(&size),
        "{defined:?}"
    );
    let path = std::ffi::CString::new(library).unwrap();
    // SAFETY: the library was just linked by the test; its two symbols are a byte array and a 64-bit length, as read below.
    let (bytes, length) = unsafe {
        let handle = dlopen(path.as_ptr(), 2);
        assert!(!handle.is_null(), "the linked library does not load");
        let find = |name: &str| dlsym(handle, std::ffi::CString::new(name).unwrap().as_ptr());
        let length = *find(&size).cast::<u64>();
        (
            std::slice::from_raw_parts(find(&image).cast::<u8>(), length as usize),
            length,
        )
    };
    assert_eq!((bytes, length), (IMAGE, IMAGE.len() as u64));
}

/// A unit whose parts hold no device image defines neither symbol.
#[test]
fn a_host_only_unit_defines_no_device_image() {
    let root = store("host_link");
    let Some(host) = host_machine(&root) else {
        return common::skipped(None, "the machine has no host compiler");
    };
    let parts = format!("[{}]", built(&osprey_cpu_lib::Cpu, &host, HOST_UNIT, &root));
    let linked = tool::link(&osprey_cpu_lib::Cpu, &parts, SET, "without_image.so", &root).unwrap();
    let (image, size) = image_symbols("k");
    let defined = defined_symbols(&member_text(&linked, "library").unwrap());
    assert!(
        !defined.contains(&image) && !defined.contains(&size),
        "{defined:?}"
    );
}
