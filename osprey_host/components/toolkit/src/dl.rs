//! Loading a shared library named by a tool table and looking up its functions: the one place the tool side touches the dynamic
//! loader. A library is opened and never closed: its functions may be called again for the life of the process.
use std::ffi::{c_char, c_int, c_void, CString};
use std::mem::{size_of, transmute_copy};
use std::path::PathBuf;

/// The loader's "bind every symbol now" flag, the same value on Linux and macOS.
const RTLD_NOW: c_int = 2;

/// The record `dladdr(3)` fills: the file a symbol was loaded from and the symbol it found.
#[repr(C)]
struct DlInfo {
    /// The path of the shared object that holds the address.
    file: *const c_char,
    /// The base address the object is loaded at.
    base: *mut c_void,
    /// The name of the nearest symbol, or null.
    symbol: *const c_char,
    /// The address of that symbol.
    symbol_address: *mut c_void,
}

extern "C" {
    /// The `dladdr(3)` of glibc and macOS.
    fn dladdr(address: *const c_void, info: *mut DlInfo) -> c_int;
    /// POSIX `dlopen(3)`.
    fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
    /// POSIX `dlsym(3)`.
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}

/// An opened shared library.
#[derive(Debug)]
pub struct Library {
    /// The loader's handle.
    handle: *mut c_void,
    /// The name it was opened by.
    pub name: String,
}

impl Library {
    /// The first of `names` the loader can open (a bare name searches the system's library paths), or `None`.
    #[must_use]
    pub fn open_first(names: &[String]) -> Option<Library> {
        names.iter().find_map(|name| {
            let c = CString::new(name.as_str()).ok()?;
            // SAFETY: `c` is a valid NUL-terminated string that outlives the call; the flag is a documented one.
            let handle = unsafe { dlopen(c.as_ptr(), RTLD_NOW) };
            (!handle.is_null()).then(|| Library {
                handle,
                name: name.clone(),
            })
        })
    }

    /// The function `name` of the library as the function-pointer type `F`, or `None` when the library has no such symbol.
    ///
    /// # Safety
    /// `F` must be an `extern "C"` function-pointer type with exactly the signature the library gives `name`.
    #[must_use]
    pub unsafe fn function<F: Copy>(&self, name: &str) -> Option<F> {
        if size_of::<F>() != size_of::<*mut c_void>() {
            return None;
        }
        let c = CString::new(name).ok()?;
        // SAFETY: `handle` came from `dlopen` and is never closed; `c` is a valid NUL-terminated string.
        let found = unsafe { dlsym(self.handle, c.as_ptr()) };
        // SAFETY: the sizes match (checked above) and the caller vouches for the signature.
        (!found.is_null()).then(|| unsafe { transmute_copy::<*mut c_void, F>(&found) })
    }

    /// The folder the library was loaded from, found through the address of its symbol `name`; `None` when it has no such symbol or
    /// the loader cannot say where it came from.
    #[must_use]
    pub fn folder(&self, name: &str) -> Option<PathBuf> {
        let c = CString::new(name).ok()?;
        // SAFETY: `handle` came from `dlopen` and is never closed; `c` is a valid NUL-terminated string.
        let address = unsafe { dlsym(self.handle, c.as_ptr()) };
        let mut info = DlInfo {
            file: std::ptr::null(),
            base: std::ptr::null_mut(),
            symbol: std::ptr::null(),
            symbol_address: std::ptr::null_mut(),
        };
        // SAFETY: `info` is a valid record for `dladdr` to fill; `address` is a symbol of the open library or null, which it refuses.
        if address.is_null()
            || unsafe { dladdr(address, &raw mut info) } == 0
            || info.file.is_null()
        {
            return None;
        }
        // SAFETY: on success `file` points to the loader's NUL-terminated path, valid while the library stays loaded.
        let file = unsafe { std::ffi::CStr::from_ptr(info.file) }
            .to_str()
            .ok()?;
        PathBuf::from(file)
            .parent()
            .map(std::path::Path::to_path_buf)
    }
}
