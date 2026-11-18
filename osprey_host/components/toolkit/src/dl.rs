//! Loading a shared library named by a tool table and looking up its functions: the one place the tool side touches the dynamic
//! loader. A library is opened and never closed: its functions may be called again for the life of the process.
use std::ffi::{c_char, c_int, c_void, CString};
use std::mem::{size_of, transmute_copy};

/// The loader's "bind every symbol now" flag, the same value on Linux and macOS.
const RTLD_NOW: c_int = 2;

extern "C" {
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
}
