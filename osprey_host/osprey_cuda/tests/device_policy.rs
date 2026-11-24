//! The device arch policy against a fake driver library built here (a tiny C++ file compiled by the host compiler) and against no
//! driver at all: the GPU and release the driver states, the pin of the ask and of the environment, the clamp to the toolkit's archs.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a panic is the failure
mod common;
use common::{build_shared, scratch, script};
use rcore::json::J;
use toolkit::{wire, Env};

const DRIVER: &str = r#"
extern "C" {
int cuInit(unsigned) { return 0; }
int cuDriverGetVersion(int* v) { *v = 12060; return 0; }
int cuCtxGetDevice(int*) { return 201; }
int cuDeviceGet(int* d, int ordinal) { *d = 3 + ordinal; return 0; }
int cuDeviceGetAttribute(int* v, int attr, int dev) { if (dev != 3) return 1; *v = attr == 75 ? 7 : 5; return 0; }
}
"#;

/// A table whose driver library is `library` (a name that need not exist).
fn table(library: &str) -> String {
    format!(
        r#"{{"tools":{{"compiler":{{"rules":[{{"env":"FAKE_NVCC","required":true}}],
              "probes":[{{"name":"archs","args":["--list-gpu-arch"],"after":"compute_","as":"int_list"}}]}}}},
          "device":{{"driver":{{"libraries":["{library}"],
              "calls":{{"init":"cuInit","version":"cuDriverGetVersion","current_device":"cuCtxGetDevice","device_by_ordinal":"cuDeviceGet","attribute":"cuDeviceGetAttribute"}},
              "version_decode":[1000,10],"attributes":{{"major":75,"minor":76}},"arch_format":"sm_{{major}}{{minor}}"}},
            "arch":{{"pin_env":"TEST_ARCH","fallback":"compute_61","portable_min":60,"toolkit":{{"tool":"compiler","probe":"archs"}}}}}}}}"#
    )
}

/// The `stated.device` record, or the refusal text, of a machine ask under `env_pin` and the ask's own members.
fn device(
    dir: &std::path::Path,
    library: &str,
    env_pin: Option<&str>,
    ask_pin: Option<&str>,
) -> Result<J, String> {
    let nvcc = dir.join("nvcc");
    script(
        &nvcc,
        "#!/bin/sh\ncase \"$1\" in --version) echo 'fake nvcc release 12.6'; exit 0;; --list-gpu-arch) printf 'compute_50\\ncompute_61\\ncompute_75\\ncompute_80\\n'; exit 0;; esac\n",
    );
    let mut pairs = vec![
        ("FAKE_NVCC", nvcc.to_str().unwrap()),
        ("PATH", "/bin:/usr/bin"),
    ];
    if let Some(pin) = env_pin {
        pairs.push(("TEST_ARCH", pin));
    }
    let tk = common::new_toolkit(&table(library), &[], Env::from_pairs(&pairs)).unwrap();
    let ask = ask_pin.map_or("{}".to_string(), |p| format!("{{\"device_arch\":\"{p}\"}}"));
    let answer = tk.machine(&ask).map_err(|r| r.message)?;
    Ok(wire::parse("a", &answer)
        .unwrap()
        .get("stated")
        .unwrap()
        .get("device")
        .unwrap()
        .clone())
}

fn text(j: &J, key: &str) -> String {
    j.get(key).and_then(J::str).unwrap_or("<null>").to_string()
}

#[test]
fn the_fake_driver_states_the_gpu_and_the_release_and_the_gpu_is_the_arch() {
    let dir = scratch("driver");
    let lib = build_shared(&dir, "fakedriver", DRIVER);
    let d = device(&dir, lib.to_str().unwrap(), None, None).unwrap();
    assert_eq!(
        (
            text(&d, "arch").as_str(),
            text(&d, "gpu_arch").as_str(),
            text(&d, "driver_release").as_str()
        ),
        ("sm_75", "sm_75", "12.6")
    );
    assert_eq!(
        (
            d.get("driver_encoded").unwrap().int(),
            wire::render(d.get("toolkit_archs").unwrap())
        ),
        (12060, "[50,61,75,80]".to_string())
    );
    assert_eq!(text(&d, "why"), "the running GPU");
}

#[test]
fn pins_beat_the_gpu_and_are_clamped_to_the_toolkit() {
    let dir = scratch("pins");
    let lib = build_shared(&dir, "fakedriver", DRIVER);
    let lib = lib.to_str().unwrap();
    assert_eq!(
        text(&device(&dir, lib, Some("sm_61"), None).unwrap(), "arch"),
        "sm_61"
    );
    assert_eq!(
        text(
            &device(&dir, lib, Some("sm_61"), Some("sm_80")).unwrap(),
            "arch"
        ),
        "sm_80"
    );
    let above = device(&dir, lib, None, Some("sm_100")).unwrap();
    assert_eq!(text(&above, "arch"), "compute_80");
    assert!(text(&above, "why").contains("above the toolkit's highest arch"));
    let below = device(&dir, lib, Some("sm_35"), None).unwrap_err();
    assert!(
        below.contains("$TEST_ARCH=\"sm_35\"") && below.contains("below the minimum arch 50"),
        "{below}"
    );
}

#[test]
fn with_no_driver_library_the_arch_is_virtual_at_the_lowest_portable_toolkit_arch() {
    let dir = scratch("nodriver");
    let d = device(&dir, "libdoes-not-exist-anywhere.so", None, None).unwrap();
    assert_eq!(
        (
            text(&d, "arch").as_str(),
            text(&d, "gpu_arch").as_str(),
            text(&d, "driver_release").as_str()
        ),
        ("compute_61", "<null>", "<null>")
    );
}

/// A driver that initialises and finds no device.
const DRIVER_WITHOUT_DEVICE: &str = r#"
extern "C" {
int cuInit(unsigned) { return 0; }
int cuDriverGetVersion(int* v) { *v = 12060; return 0; }
int cuCtxGetDevice(int*) { return 201; }
int cuDeviceGet(int*, int) { return 100; }
int cuDeviceGetAttribute(int*, int, int) { return 100; }
}
"#;

/// A driver whose initialisation fails.
const DRIVER_THAT_FAILS: &str = r#"
extern "C" {
int cuInit(unsigned) { return 999; }
int cuDriverGetVersion(int*) { return 0; }
int cuCtxGetDevice(int*) { return 201; }
int cuDeviceGet(int*, int) { return 0; }
int cuDeviceGetAttribute(int*, int, int) { return 0; }
}
"#;

/// The machine answer's `runs_here` and `why` for a table whose driver library is `library`.
fn runs_here(dir: &std::path::Path, library: &str) -> (bool, String) {
    let nvcc = dir.join("nvcc");
    script(&nvcc, "#!/bin/sh\ncase \"$1\" in --version) echo 'fake nvcc release 12.6'; exit 0;; --list-gpu-arch) printf 'compute_61\\ncompute_75\\n'; exit 0;; esac\n");
    let env = Env::from_pairs(&[
        ("FAKE_NVCC", nvcc.to_str().unwrap()),
        ("PATH", "/bin:/usr/bin"),
    ]);
    let answer = common::new_toolkit(&table(library), &[], env)
        .unwrap()
        .machine("{}")
        .unwrap();
    let answer = wire::parse("a", &answer).unwrap();
    (
        matches!(answer.get("runs_here"), Some(J::Bool(true))),
        text(&answer, "why"),
    )
}

#[test]
fn the_machine_runs_a_device_target_only_when_a_driver_library_states_a_device() {
    let dir = scratch("runs_here");
    let present = build_shared(&dir, "fakedriver", DRIVER);
    assert_eq!(
        runs_here(&dir, present.to_str().unwrap()),
        (true, String::new())
    );
    let (runs, why) = runs_here(&dir, "libdoes-not-exist-anywhere.so");
    assert!(!runs && why.contains("no driver library found"), "{why}");
    let empty = build_shared(&dir, "emptydriver", DRIVER_WITHOUT_DEVICE);
    let (runs, why) = runs_here(&dir, empty.to_str().unwrap());
    assert!(!runs && why.contains("states no device"), "{why}");
    let failing = build_shared(&dir, "failingdriver", DRIVER_THAT_FAILS);
    let (runs, why) = runs_here(&dir, failing.to_str().unwrap());
    assert!(
        !runs && why.contains("did not initialise (status 999)"),
        "{why}"
    );
}

/// Oracle: the reason a library states for lacking its runtime leads `why`, and the driver's own reason follows it when there is
/// one; a working driver does not make a library without the runtime run.
#[test]
fn a_toolkit_without_the_runtime_never_runs_here_and_says_so_first() {
    let dir = scratch("runs_here_without_runtime");
    let present = build_shared(&dir, "fakedriver_without", DRIVER);
    let nvcc = dir.join("nvcc");
    script(&nvcc, "#!/bin/sh\necho 'fake nvcc release 12.6'\n");
    let env = || {
        Env::from_pairs(&[
            ("FAKE_NVCC", nvcc.to_str().unwrap()),
            ("PATH", "/bin:/usr/bin"),
        ])
    };
    let why_of = |library: &str| {
        let toolkit = common::new_toolkit(&table(library), &[], env())
            .unwrap()
            .without_runtime("built without the device runtime");
        let answer = wire::parse("a", &toolkit.machine("{}").unwrap()).unwrap();
        assert!(matches!(answer.get("runs_here"), Some(J::Bool(false))));
        text(&answer, "why")
    };
    assert_eq!(
        why_of(present.to_str().unwrap()),
        "built without the device runtime"
    );
    let why = why_of("libdoes-not-exist-anywhere.so");
    assert!(
        why.starts_with("built without the device runtime; no driver library found"),
        "{why}"
    );
}
