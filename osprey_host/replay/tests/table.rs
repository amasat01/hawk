//! The entry table of each library: abi, name, identity, the render call that reaches the writer crate, the facts entry and its
//! refusals, and the refusal of the target the other library serves.
use aether_writers::contract::types::Target;
use osprey_lowering::door::OspreyV1;
use std::ffi::{c_char, CStr};

/// Both libraries: the target each serves, its name, and a stated machine and compiler its facts question takes.
const LIBRARIES: [(Target, &str, &CStr, &CStr, &CStr); 2] = [
    (
        Target::Host,
        "cpu",
        c"host",
        c"c++ || Apple clang version 21.0.0 || 0",
        c"{\"headers\":\"roots\"}",
    ),
    (
        Target::Device,
        "cuda",
        c"cuda",
        c"/usr/local/cuda/bin/nvcc || Cuda compilation tools, release 12.6, V12.6.77 || 0",
        c"{}",
    ),
];

/// The table of the library that serves `target`.
fn table(target: Target) -> &'static OspreyV1 {
    osprey_replay::table_of(target)
}

/// The data files of the library that serves `target`, NUL-terminated as the table takes a directory.
fn facts_dir(target: Target) -> std::ffi::CString {
    std::ffi::CString::new(osprey_replay::facts_of(target)).unwrap()
}

#[test]
fn each_table_forwards_and_identifies() {
    for (target, name, ..) in LIBRARIES {
        table_forwards_and_identifies(table(target), name);
    }
}

/// The checks of one library's table.
fn table_forwards_and_identifies(t: &OspreyV1, name: &str) {
    assert_eq!(
        (t.abi, unsafe { CStr::from_ptr(t.name) }.to_str().unwrap()),
        (osprey_host_abi(), name)
    );
    assert_eq!(t.contract, header_number("OB_CONTRACT"));
    let mut id = [0 as c_char; 65];
    assert_eq!((t.identity)(id.as_mut_ptr()), 0);
    assert_eq!(unsafe { CStr::from_ptr(id.as_ptr()) }.to_bytes().len(), 64);
    let c = (t.ctx_new)();
    let (j, mut lens) = ("{}", [0usize; 4]);
    assert_eq!(
        (t.render)(
            c,
            0,
            j.as_ptr() as *const c_char,
            j.len(),
            std::ptr::null_mut(),
            0,
            lens.as_mut_ptr()
        ),
        2
    );
    assert_eq!(
        unsafe { CStr::from_ptr((t.last_error)(c)) }
            .to_str()
            .unwrap(),
        if cfg!(feature = "old-contract") {
            "not in this build: an unknown call form"
        } else {
            "not in this build: an old-shape contract (a typed contract carries \"v\")"
        }
    );
    // SAFETY: `c` came from this table's `ctx_new` and is freed once.
    unsafe { (t.ctx_free)(c) };
}

/// The abi number the header defines, which the library must carry.
fn osprey_host_abi() -> u32 {
    header_number("OB_ABI")
}

/// The number the header defines under `name`.
fn header_number(name: &str) -> u32 {
    let header = include_str!("../../include/osprey.h");
    let prefix = format!("#define {name} ");
    let line = header.lines().find_map(|l| l.strip_prefix(prefix.as_str()));
    line.unwrap().trim().parse().unwrap()
}

#[test]
fn each_facts_entry_answers_and_refuses() {
    for (target, _, name, compiler, stated) in LIBRARIES {
        let dir = facts_dir(target);
        facts_entry_answers_and_refuses(table(target), (&dir, name, compiler, stated));
    }
}

#[test]
fn a_facts_question_for_the_other_librarys_target_is_refused_by_name() {
    for (target, name, _, compiler, stated) in LIBRARIES {
        let (other, other_name, ..) = LIBRARIES.iter().find(|row| row.0 != target).unwrap();
        let (t, dir) = (table(target), facts_dir(target));
        let c = (t.ctx_new)();
        let (mut buf, mut len) = (vec![0 as c_char; 1 << 16], 0usize);
        let asked = LIBRARIES.iter().find(|row| row.0 == *other).unwrap().2;
        let status = (t.facts)(
            c,
            dir.as_ptr(),
            asked.as_ptr(),
            compiler.as_ptr(),
            stated.as_ptr(),
            buf.as_mut_ptr(),
            buf.len(),
            &mut len,
        );
        assert_eq!(status, 2, "{name} asked for {other_name}'s target");
        let why = unsafe { CStr::from_ptr((t.last_error)(c)) }
            .to_str()
            .unwrap();
        assert!(
            why.contains(name) && why.contains(asked.to_str().unwrap()),
            "{why}"
        );
        // SAFETY: `c` came from this table's `ctx_new` and is freed once.
        unsafe { (t.ctx_free)(c) };
    }
}

/// The checks of one library's facts entry: its own target answers, a target nobody serves and a malformed stated machine are
/// the caller's mistakes (status 2), and a data directory that is none is the library's fault (status 4).
fn facts_entry_answers_and_refuses(
    t: &OspreyV1,
    (dir, target, compiler, stated_ok): (&CStr, &CStr, &CStr, &CStr),
) {
    let dir = dir.as_ptr();
    let c = (t.ctx_new)();
    let mut len = 0usize;
    let status = |buf: *mut c_char, cap: usize, len: &mut usize| {
        (t.facts)(
            c,
            dir,
            target.as_ptr(),
            compiler.as_ptr(),
            stated_ok.as_ptr(),
            buf,
            cap,
            len,
        )
    };
    assert_eq!(status(std::ptr::null_mut(), 0, &mut len), 3);
    let mut buf = vec![0 as c_char; len + 1];
    assert_eq!(status(buf.as_mut_ptr(), len + 1, &mut len), 0);
    let text = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_str().unwrap();
    assert!(
        text.starts_with("{\"isa\":") || text.starts_with("{\"exact_libm\":"),
        "{text}"
    );
    let refused = (t.facts)(
        c,
        dir,
        c"fast".as_ptr(),
        compiler.as_ptr(),
        std::ptr::null(),
        buf.as_mut_ptr(),
        buf.len(),
        &mut len,
    );
    assert_eq!(refused, 2);
    let why = unsafe { CStr::from_ptr((t.last_error)(c)) }
        .to_str()
        .unwrap();
    assert!(why.contains("fast"), "{why}");
    let stated = c"{\"nope\":1}";
    let malformed = (t.facts)(
        c,
        dir,
        target.as_ptr(),
        compiler.as_ptr(),
        stated.as_ptr(),
        buf.as_mut_ptr(),
        buf.len(),
        &mut len,
    );
    assert_eq!(malformed, 2);
    let none = c"/nonexistent/osprey_facts";
    let missing = (t.facts)(
        c,
        none.as_ptr(),
        target.as_ptr(),
        compiler.as_ptr(),
        std::ptr::null(),
        buf.as_mut_ptr(),
        buf.len(),
        &mut len,
    );
    assert_eq!(missing, 4);
    let why = unsafe { CStr::from_ptr((t.last_error)(c)) };
    assert_eq!(
        why.to_str().unwrap(),
        "facts: /nonexistent/osprey_facts is not a directory"
    );
    // SAFETY: `c` came from this table's `ctx_new` and is freed once.
    unsafe { (t.ctx_free)(c) };
}

/// A well-formed typed unit of the host target that holds no walk.
const UNIT: &str = r#"{"v":1,"name":"axpb","target":"Host","scalar":"Float64","kind":{"guard":"Default","lanes":false,"sink":"Plain"},"form":"Plain","sinks":[5],"walk":{"access":"SampleLocal","slots":[{"role":"Mutable","name":"y","dtype":"F64","shape":[],"index":0,"zeroed":false},{"role":"PlaneIn","name":"x","dtype":"F64","shape":[],"index":1,"zeroed":false},{"role":"Uniform","name":"a","dtype":"F64","shape":[],"index":2,"zeroed":false},{"role":"Uniform","name":"b","dtype":"F64","shape":[],"index":3,"zeroed":false}]},"body":"Walk","rows":1,"build":{"form":{"lazy_leaf_width":32,"rolled_fold_min":16,"lean_width":4,"uniform_run":32},"fold_form":"Auto","lanes":{"run_max":32,"block_run":1024,"tile":[64,1024]},"linalg_max_dim":32,"cells_row":32,"blend":true,"warps":1},"machine":{"isa":"Arm64","exact_libm":true,"routed_pow":false,"caches":{"l1d_bytes":131072,"l2_bytes":16777216,"source":"sysctl hw.perflevel0.l1dcachesize/l2cachesize"},"select_weight":{}},"exports":"Derived","literal_table":false}"#;

/// The status and reason of rendering `contract` with the walk `walk` in a fresh context of the library.
fn render_refusal(contract: &str, walk: u32) -> (i32, String) {
    let t = table(Target::Host);
    let c = (t.ctx_new)();
    let mut lens = [0usize; 4];
    let json = contract.as_ptr() as *const c_char;
    let status = (t.render)(
        c,
        walk,
        json,
        contract.len(),
        std::ptr::null_mut(),
        0,
        lens.as_mut_ptr(),
    );
    let reason = unsafe { CStr::from_ptr((t.last_error)(c)) }
        .to_str()
        .unwrap()
        .to_string();
    // SAFETY: `c` came from this table's `ctx_new` and is freed once.
    unsafe { (t.ctx_free)(c) };
    (status, reason)
}

#[test]
fn a_refusal_status_follows_its_kind() {
    // a request the library cannot read is a bad argument (status 2)
    let (status, reason) = render_refusal(r#"{"v":1,"name":"k","target":"Bogus"}"#, 0);
    assert_eq!(
        (status, reason.as_str()),
        (2, "not in this build: member Bogus of target")
    );
    // a well-formed request for a construct the build does not write is the refused status (1)
    let (status, reason) = render_refusal(UNIT, u32::MAX);
    assert_eq!(status, 1);
    assert_eq!(
        reason,
        "not in this build: a unit with no walk (team stage)"
    );
}
