//! The C face of the crate: three functions the binding's loader answers with. Each writes its text as the door's text calls do:
//! the text and a NUL into `buf` when `cap` holds `len + 1` bytes, else only `len` is set and the status is [`TRUNC`]. The
//! status is [`OK`] with the answer as the text, or [`REFUSED`] with the refusal sentence as the text.
use std::ffi::{c_char, c_int, CStr};

use crate::{roster, target};

/// The door's status for an answer that was written.
const OK: c_int = 0;
/// The door's status for a refusal; the text is its sentence.
const REFUSED: c_int = 1;
/// The door's status for a buffer that is too small; `len` holds the size needed.
const TRUNC: c_int = 3;

/// The text of the C string `text`, or the refusal that it is not UTF-8.
///
/// # Safety
/// `text` is a NUL-terminated string that stays valid for the call.
unsafe fn read<'a>(text: *const c_char) -> Result<&'a str, String> {
    CStr::from_ptr(text)
        .to_str()
        .map_err(|_| "an argument is not UTF-8 text".to_string())
}

/// Writes `answer` into the caller's buffer by the text protocol and returns the status.
///
/// # Safety
/// `buf` holds `cap` bytes (or is null when `cap` is 0) and `len` is a valid pointer.
unsafe fn write(
    answer: Result<String, String>,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    let (text, status) = match answer {
        Ok(text) => (text, OK),
        Err(text) => (text, REFUSED),
    };
    *len = text.len();
    if cap < text.len() + 1 {
        return TRUNC;
    }
    std::ptr::copy_nonoverlapping(text.as_ptr().cast::<c_char>(), buf, text.len());
    *buf.add(text.len()) = 0;
    status
}

/// The canonical spelling of the target name `name` ([`target::canonical`]).
///
/// # Safety
/// `name` is a NUL-terminated string; `buf` holds `cap` bytes; `len` is valid.
#[no_mangle]
pub unsafe extern "C" fn omf_target_id(
    name: *const c_char,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    write(read(name).and_then(target::canonical), buf, cap, len)
}

/// The roster of a JSON list of folders ([`roster::of_folders`]).
///
/// # Safety
/// `folders` is a NUL-terminated string; `buf` holds `cap` bytes; `len` is valid.
#[no_mangle]
pub unsafe extern "C" fn omf_roster(
    folders: *const c_char,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    write(read(folders).and_then(roster::of_folders), buf, cap, len)
}

/// The OSPREY that serves a target, over a roster ([`roster::serving`]).
///
/// # Safety
/// `roster` and `target` are NUL-terminated strings; `buf` holds `cap` bytes; `len` is valid.
#[no_mangle]
pub unsafe extern "C" fn omf_serving(
    roster: *const c_char,
    target: *const c_char,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    write(
        read(roster).and_then(|r| read(target).and_then(|t| roster::serving(r, t))),
        buf,
        cap,
        len,
    )
}
