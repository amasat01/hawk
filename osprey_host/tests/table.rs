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
        1
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
