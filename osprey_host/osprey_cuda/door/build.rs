//! Builds the device runtime into the cuda OSPREY, when the library was built with it (the rule is in `../build.rs`): every
//! source `sources.txt` of eagle's C door lists, host group and device group, is compiled and the objects are linked into the
//! loadable library, which then exports the door's `eg_` functions beside `osprey_v1`. Without a toolkit this script builds
//! nothing and the library exports `osprey_v1` alone.
//!
//! With a toolkit the inputs are, each refused by name when absent: `OSPREY_EAGLE_SRC` (the eagle source root, which holds
//! `eagle/door/` and `plugin/`), `OSPREY_AETHER_SRC` (the aether source root), `CXX` (a C++23 compiler, also the host compiler of
//! `nvcc`) and the raptor sibling (`RAPTOR`, default the sibling folder), which holds the cpu and cuda wings' facts headers and the
//! door's host-and-device row header. The toolkit root comes from the library's build script (`DEP_OSPREY_CUDA_DEVICE_ROOT`).
//!
//! The door's own makefile (`eagle/door/Makefile`) is the law of the flags. Only the source that includes the CUDA headers is built
//! without `EAGLE_CPU_ONLY`; the `.cu` source goes to `nvcc` with device code for every served architecture; the CUDA runtime is
//! linked statically so the library asks the process for the driver alone; cuBLAS and cuFFT are loaded by the door at use.
use std::path::{Path, PathBuf};
use std::process::Command;

mod object_list;
use object_list::{gencode, sources, Source};

/// The GPU architectures the device code is built for: device code for each and the intermediate code of the newest, so a later GPU
/// can still load it. The facts cards state hardware rows by nearest capability, not a served list, so the list is stated here: the
/// door owner's default.
const SERVED_ARCHS: [u32; 4] = [61, 70, 80, 90];
/// The one source that includes the CUDA headers, so the only one built without `EAGLE_CPU_ONLY` (as the door's makefile does).
const CUDA_HEADERS_SOURCE: &str = "device/Device.cpp";
/// The one source that reads the door's row tables.
const ROWS_SOURCE: &str = "Context.cpp";

/// The value of the environment variable `name`, or a refusal naming the variable and what it must hold.
fn input(name: &str, holds: &str) -> String {
    println!("cargo::rerun-if-env-changed={name}");
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set: it names {holds}"))
}

/// The path `path`, or a refusal naming the missing entry.
fn existing(path: PathBuf, what: &str) -> PathBuf {
    assert!(path.exists(), "{what} is missing: {}", path.display());
    path
}

/// Runs `program` with `args` to make `object` from `source` and refuses with the program's own output when it fails.
fn compile(program: &str, args: &[String], source: &Path, object: &Path) {
    let done = Command::new(program)
        .args(args)
        .arg("-c")
        .arg(source)
        .arg("-o")
        .arg(object)
        .output()
        .unwrap_or_else(|why| panic!("{program} did not run: {why}"));
    assert!(
        done.status.success(),
        "{} did not compile with {program}:\n{}",
        source.display(),
        String::from_utf8_lossy(&done.stderr)
    );
}

/// The runtime library the compiler `cxx` links against: the file it names for `library`.
fn compiler_runtime(cxx: &str, library: &str) -> PathBuf {
    let asked = Command::new(cxx)
        .arg(format!("-print-file-name={library}"))
        .output()
        .unwrap_or_else(|why| panic!("CXX {cxx} did not run: {why}"));
    let found = PathBuf::from(String::from_utf8_lossy(&asked.stdout).trim());
    assert!(
        found.is_absolute() && found.exists(),
        "CXX {cxx} names no {library}"
    );
    found
}

/// What the build reads: the toolchain, the checkouts and the cards. Each path is checked to exist when this is made.
struct Inputs {
    /// The CUDA toolkit root.
    toolkit: PathBuf,
    /// The eagle source root.
    eagle: PathBuf,
    /// The aether source root.
    aether: PathBuf,
    /// The C++ compiler, also the host compiler of `nvcc`.
    cxx: String,
    /// The door's folder, `eagle/door`.
    door: PathBuf,
    /// The cpu wing's facts header, forced into every C++ source.
    cpu_facts: PathBuf,
    /// The cuda wing's facts header, forced into the `.cu` source.
    cuda_facts: PathBuf,
    /// The door's row tables for a host-and-device door.
    rows: PathBuf,
}

impl Inputs {
    /// Reads the environment and checks the files, each refused by name.
    fn read(toolkit: PathBuf) -> Self {
        let eagle = PathBuf::from(input("OSPREY_EAGLE_SRC", "the eagle source root"));
        let aether = PathBuf::from(input("OSPREY_AETHER_SRC", "the aether source root"));
        let cxx = input("CXX", "a C++23 compiler");
        let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
        let raptor =
            std::env::var("RAPTOR").map_or_else(|_| here.join("../../../raptor"), PathBuf::from);
        println!("cargo::rerun-if-env-changed=RAPTOR");
        assert!(
            std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "linux"),
            "the device runtime is built for Linux only"
        );
        let cards = raptor.join("rs/wings/cards");
        let inputs = Inputs {
            door: existing(eagle.join("eagle/door"), "OSPREY_EAGLE_SRC/eagle/door"),
            cpu_facts: existing(
                cards.join("cpu/raptor_facts.h"),
                "the cpu wing's raptor_facts.h",
            ),
            cuda_facts: existing(
                cards.join("cuda/raptor_facts.h"),
                "the cuda wing's raptor_facts.h",
            ),
            rows: existing(
                cards.join("door/host_device/eagle_door_rows.h"),
                "the host and device door rows",
            ),
            toolkit,
            eagle,
            aether,
            cxx,
        };
        for watched in [
            &inputs.door,
            &inputs.cpu_facts,
            &inputs.cuda_facts,
            &inputs.rows,
        ] {
            println!("cargo::rerun-if-changed={}", watched.display());
        }
        inputs
    }

    /// The include roots every source reads.
    fn roots(&self) -> [String; 2] {
        [
            format!("-I{}", self.eagle.display()),
            format!("-I{}", self.aether.display()),
        ]
    }
}

/// The flags `-include <file>`.
fn force_include(file: &Path) -> [String; 2] {
    ["-include".to_string(), file.display().to_string()]
}

/// The compile of one source of the door: the program and its arguments.
struct Job<'a> {
    /// The source and its group.
    source: &'a Source,
    /// The compiler that builds it.
    program: String,
    /// Its flags.
    args: Vec<String>,
}

/// The jobs for the sources of the list, each with the flags of its kind: the `.cu` source goes to `nvcc`, the source that includes
/// the CUDA headers keeps `EAGLE_CPU_ONLY` out, the one that reads the row tables gets them, and every other source is host code.
fn jobs<'a>(inputs: &Inputs, list: &'a [Source]) -> Vec<Job<'a>> {
    let host: Vec<String> = [
        "-std=c++23",
        "-O2",
        "-fPIC",
        "-fvisibility=hidden",
        "-fvisibility-inlines-hidden",
        "-ffp-contract=off",
        "-DAETHER_CPP_MODE=1",
        "-DEAGLE_CPU_ONLY=1",
    ]
    .map(ToString::to_string)
    .into_iter()
    .chain(force_include(&inputs.cpu_facts))
    .chain(inputs.roots())
    .collect();
    let with_cuda_headers: Vec<String> = host
        .iter()
        .filter(|flag| *flag != "-DEAGLE_CPU_ONLY=1")
        .cloned()
        .chain([
            "-isystem".to_string(),
            inputs.toolkit.join("include").display().to_string(),
        ])
        .collect();
    let with_rows: Vec<String> = host
        .iter()
        .cloned()
        .chain(force_include(&inputs.rows))
        .collect();
    let device_code: Vec<String> = ["-std=c++20", "-O2"]
        .map(ToString::to_string)
        .into_iter()
        .chain(gencode(&SERVED_ARCHS))
        .chain(
            [
                "-Wno-deprecated-gpu-targets",
                "-Werror",
                "all-warnings",
                "-ccbin",
                &inputs.cxx,
                "-Xcompiler",
                "-fPIC,-fvisibility=hidden,-Wall,-Wextra",
                "-DAETHER_HAS_CUDA",
                "-diag-suppress=20050",
            ]
            .map(ToString::to_string),
        )
        .chain(force_include(&inputs.cuda_facts))
        .chain(inputs.roots())
        .collect();
    let nvcc = existing(inputs.toolkit.join("bin/nvcc"), "the toolkit's nvcc");
    list.iter()
        .map(|source| {
            let (program, args) = match source.path.as_str() {
                path if path.ends_with(".cu") => (nvcc.display().to_string(), &device_code),
                CUDA_HEADERS_SOURCE => (inputs.cxx.clone(), &with_cuda_headers),
                ROWS_SOURCE => (inputs.cxx.clone(), &with_rows),
                _ => (inputs.cxx.clone(), &host),
            };
            Job {
                source,
                program,
                args: args.clone(),
            }
        })
        .collect()
}

/// Runs the jobs, as many at a time as cargo allows, and returns the objects in job order.
fn build_objects(jobs: &[Job], door: &Path) -> Vec<PathBuf> {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let parallel = std::env::var("NUM_JOBS")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(2)
        .max(1);
    let mut objects = Vec::new();
    for batch in jobs.chunks(parallel) {
        std::thread::scope(|scope| {
            let running: Vec<_> = batch
                .iter()
                .map(|job| {
                    let object = out
                        .join(job.source.path.replace('/', "_"))
                        .with_extension("o");
                    let source = door.join(&job.source.path);
                    scope.spawn(move || {
                        compile(&job.program, &job.args, &source, &object);
                        object
                    })
                })
                .collect();
            objects.extend(running.into_iter().map(|job| job.join().unwrap()));
        });
    }
    objects
}

/// Hands the linker the objects, the static CUDA runtime and the C++ runtime of the compiler that made the objects.
fn link(inputs: &Inputs, objects: &[PathBuf]) {
    let cudart = existing(
        inputs.toolkit.join("lib64/libcudart_static.a"),
        "the toolkit's static CUDA runtime",
    );
    for object in objects {
        println!("cargo::rustc-link-arg-cdylib={}", object.display());
    }
    // The CUDA runtime is linked statically (the linker script of the build keeps its symbols out of the exports), so the library
    // needs the driver alone at run time.
    for flag in [
        format!("-L{}", cudart.parent().unwrap().display()),
        "-lcudart_static".to_string(),
        "-ldl".to_string(),
        "-lrt".to_string(),
        "-pthread".to_string(),
    ] {
        println!("cargo::rustc-link-arg-cdylib={flag}");
    }
    // The objects need the C++ runtime of the compiler that made them, which may be newer than the system's: the file CXX names can
    // be a linker script that adds an archive beside it, so its folder is searched too.
    let cxx_runtime = compiler_runtime(&inputs.cxx, "libstdc++.so");
    let beside = cxx_runtime.parent().expect("an absolute file has a folder");
    println!("cargo::rustc-link-arg-cdylib=-L{}", beside.display());
    println!("cargo::rustc-link-arg-cdylib={}", cxx_runtime.display());
}

fn main() {
    let Ok(root) = std::env::var("DEP_OSPREY_CUDA_DEVICE_ROOT") else {
        return;
    };
    let inputs = Inputs::read(PathBuf::from(root));
    let list =
        std::fs::read_to_string(inputs.door.join("sources.txt")).expect("the door's sources.txt");
    let list = sources(&list).unwrap_or_else(|why| panic!("{why}"));
    let objects = build_objects(&jobs(&inputs, &list), &inputs.door);
    link(&inputs, &objects);
}
