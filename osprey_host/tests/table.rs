//! The entry table: abi, name, identity, and a render call that reaches the writer crate.
use std::ffi::{c_char, CStr};

#[test]
fn table_forwards_and_identifies() {
    let t = unsafe { &*osprey_host::osprey_v1() };
    assert_eq!(
        (t.abi, unsafe { CStr::from_ptr(t.name) }.to_str().unwrap()),
        (osprey_host_abi(), "host")
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
    let header = include_str!("../include/osprey_host.h");
    let prefix = format!("#define {name} ");
    let line = header.lines().find_map(|l| l.strip_prefix(prefix.as_str()));
    line.unwrap().trim().parse().unwrap()
}

/// The data files of this checkout, NUL-terminated as the table takes a directory.
const FACTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/facts\0");

#[test]
fn facts_entry_answers_and_refuses() {
    let dir = FACTS.as_ptr() as *const c_char;
    let t = unsafe { &*osprey_host::osprey_v1() };
    let c = (t.ctx_new)();
    let (target, compiler) = (c"host", c"c++ || Apple clang version 21.0.0 || 0");
    let mut len = 0usize;
    let status = |buf: *mut c_char, cap: usize, len: &mut usize| {
        (t.facts)(
            c,
            dir,
            target.as_ptr(),
            compiler.as_ptr(),
            c"{\"headers\":\"roots\"}".as_ptr(),
            buf,
            cap,
            len,
        )
    };
    assert_eq!(status(std::ptr::null_mut(), 0, &mut len), 3);
    let mut buf = vec![0 as c_char; len + 1];
    assert_eq!(status(buf.as_mut_ptr(), len + 1, &mut len), 0);
    let text = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_str().unwrap();
    assert!(text.starts_with("{\"isa\":"), "{text}");
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
    // a target the cards do not describe is a malformed question (status 2)
    assert_eq!(refused, 2);
    let why = unsafe { CStr::from_ptr((t.last_error)(c)) };
    assert_eq!(why.to_str().unwrap(), "no facts for the target fast");
    // a malformed stated machine is a malformed question too
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
    // a data directory that is none is the library's fault (status 4), and the reason names the path
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
const UNIT: &str = r#"{"v":1,"name":"axpb","target":"Host","scalar":"Float64","kind":{"guard":"Default","lanes":false,"sink":"Plain"},"form":"Plain","sinks":[5],"walk":{"access":"SampleLocal","slots":[{"role":"Mutable","name":"y","dtype":"F64","shape":[],"index":0},{"role":"PerSample","name":"x","dtype":"F64","shape":[],"index":1},{"role":"Uniform","name":"a","dtype":"F64","shape":[],"index":2},{"role":"Uniform","name":"b","dtype":"F64","shape":[],"index":3}]},"body":"Walk","rows":1,"build":{"form":{"lazy_leaf_width":32,"rolled_fold_min":16,"lean_width":4,"uniform_run":32},"fold_form":"Auto","lanes":{"run_max":32,"block_run":1024,"tile":[64,1024]},"linalg_max_dim":32,"cells_row":32,"blend":true,"warps":1},"machine":{"isa":"Arm64","exact_libm":true,"routed_pow":false,"caches":{"l1d_bytes":131072,"l2_bytes":16777216,"source":"sysctl hw.perflevel0.l1dcachesize/l2cachesize"},"select_weight":{}},"exports":"Derived","literal_table":false}"#;

/// The status and reason of rendering `contract` with the walk `walk` in a fresh context of the library.
fn render_refusal(contract: &str, walk: u32) -> (i32, String) {
    let t = unsafe { &*osprey_host::osprey_v1() };
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
