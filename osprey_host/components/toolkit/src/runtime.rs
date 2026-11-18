//! The in-process compile: when no offline compiler is found, the table's `runtime_compile` member names a library with a program
//! style C interface (create a program from a source and in-memory headers, compile it with options, fetch the log and the image,
//! destroy it). The library, its function names and the image kinds are the table's. The compile goes through the same store, key,
//! refusals and figures as a tool run; it has no address-space cap or timeout, because it runs in this process.
use crate::dl::Library;
use crate::refusal::{Done, Kind, Refusal};
use crate::wire;
use rcore::json::J;
use std::ffi::{c_char, c_int, c_void, CString};
use std::path::PathBuf;
use std::ptr;

/// `(program*, source, name, header count, header texts, header names) -> status`.
type Create = unsafe extern "C" fn(
    *mut *mut c_void,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> c_int;
/// `(program, option count, options) -> status`.
type Compile = unsafe extern "C" fn(*mut c_void, c_int, *const *const c_char) -> c_int;
/// `(program, size*) -> status`.
type SizeOf = unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int;
/// `(program, buffer) -> status`.
type Fetch = unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int;
/// `(program*) -> status`.
type Destroy = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
/// `(int*, int*) -> status`.
type Version = unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int;
/// `(int*) -> status`.
type OutInt = unsafe extern "C" fn(*mut c_int) -> c_int;

/// The loaded runtime compile library and the names the table gives its functions.
pub struct Runtime {
    /// The library.
    lib: Library,
    /// The table's `calls` record.
    calls: J,
    /// The status that means success.
    ok: c_int,
}

/// What a compile gave: the log always, the image when it succeeded.
pub struct Compiled {
    /// The compiler's diagnostic text.
    pub log: String,
    /// The image bytes, `None` on failure.
    pub image: Option<Vec<u8>>,
}

impl Runtime {
    /// Opens the first library of `spec["libraries"]` that loads (`spec` is the table's `runtime_compile` record); `None` when none.
    ///
    /// # Errors
    /// Refuses a record that lacks `libraries` or `calls`.
    pub fn open(spec: &J) -> Done<Option<Runtime>> {
        let Some(lib) = Library::open_first(&wire::texts(spec, "libraries", "runtime_compile")?)
        else {
            return Ok(None);
        };
        let calls = wire::need(spec, "calls", "runtime_compile")?.clone();
        let ok = c_int::try_from(spec.get("ok_status").map_or(0, J::int)).unwrap_or(0);
        Ok(Some(Runtime { lib, calls, ok }))
    }

    /// The function the table calls `key`.
    fn function<F: Copy>(&self, key: &str) -> Done<F> {
        let name = wire::text(&self.calls, key, "runtime_compile.calls")?;
        // SAFETY: the table states `name` as the function whose signature is `F`, as each caller below documents.
        unsafe { self.lib.function::<F>(name) }.ok_or_else(|| {
            Refusal::new(
                Kind::ToolAbsent,
                format!(
                    "the runtime library {} has no function {name}",
                    self.lib.name
                ),
            )
        })
    }

    /// The library's `major.minor` version.
    ///
    /// # Errors
    /// Refuses a library that lacks the version function or whose call fails.
    pub fn version(&self) -> Done<(i32, i32)> {
        let version: Version = self.function("version")?;
        let (mut major, mut minor): (c_int, c_int) = (0, 0);
        // SAFETY: both out-pointers are valid.
        if unsafe { version(&raw mut major, &raw mut minor) } != self.ok {
            return Err(Refusal::new(
                Kind::ToolFailed,
                "the runtime library's version call failed",
            ));
        }
        Ok((major, minor))
    }

    /// The archs the library can target, ascending; `None` when it cannot say.
    #[must_use]
    pub fn archs(&self) -> Option<Vec<u64>> {
        let count: OutInt = self.function("arch_count").ok()?;
        let list: OutInt = self.function("arch_list").ok()?;
        let mut n: c_int = 0;
        // SAFETY: `n` is a valid out-pointer.
        if unsafe { count(&raw mut n) } != self.ok {
            return None;
        }
        let mut archs = vec![0 as c_int; usize::try_from(n).ok()?];
        // SAFETY: `archs` has the `n` elements the library was told to write.
        if unsafe { list(archs.as_mut_ptr()) } != self.ok {
            return None;
        }
        let mut archs: Vec<u64> = archs
            .into_iter()
            .filter_map(|a| u64::try_from(a).ok())
            .collect();
        archs.sort_unstable();
        Some(archs)
    }

    /// The folder the table's `include` member names `name` for: the first of its `beside` folders (relative to the folder the
    /// library was loaded from) that holds the `marker` file. The headers such a compile reads from a real path live there.
    ///
    /// # Errors
    /// Refuses a table without the entry, and a library beside which no listed folder holds the marker.
    pub fn include_root(&self, spec: &J, name: &str) -> Done<PathBuf> {
        let rule = wire::need(
            wire::need(spec, "include", "runtime_compile")?,
            name,
            "runtime_compile.include",
        )?;
        let marker = wire::text(rule, "marker", "runtime_compile.include")?;
        let beside = wire::texts(rule, "beside", "runtime_compile.include")?;
        let anchor = wire::text(&self.calls, "version", "runtime_compile.calls")?;
        let home = self.lib.folder(anchor).ok_or_else(|| {
            Refusal::new(
                Kind::ToolAbsent,
                format!(
                    "the loader cannot say where {} was loaded from",
                    self.lib.name
                ),
            )
        })?;
        beside
            .iter()
            .map(|relative| home.join(relative))
            .find(|folder| folder.join(marker).is_file())
            .ok_or_else(|| {
                Refusal::new(
                    Kind::ToolAbsent,
                    format!(
                        "no folder with {marker} beside {} ({}): tried {}",
                        self.lib.name,
                        home.display(),
                        beside.join(", ")
                    ),
                )
            })
    }

    /// The identity term of keys: the library's name and version.
    ///
    /// # Errors
    /// As [`Runtime::version`].
    pub fn identity(&self) -> Done<String> {
        let (major, minor) = self.version()?;
        Ok(format!("{}||runtime {major}.{minor}", self.lib.name))
    }

    /// Compiles `source` (named `name`) with `options`, the `headers` served from memory as `(include name, bytes)`, fetching the
    /// image of the kind `target` (a key of the table's `images`). A program is always destroyed.
    ///
    /// # Errors
    /// Refuses a library that lacks a function, an unknown `target`, text with a NUL, or a failed fetch.
    pub fn compile(
        &self,
        name: &str,
        source: &str,
        headers: &[(String, Vec<u8>)],
        options: &[String],
        target: &str,
    ) -> Done<Compiled> {
        let bad = |what: &str| Refusal::bad(format!("runtime compile: {what} holds a NUL"));
        let source_c = CString::new(source).map_err(|_| bad("the source"))?;
        let name_c = CString::new(name).map_err(|_| bad("the name"))?;
        let texts = headers
            .iter()
            .map(|h| CString::new(h.1.clone()).map_err(|_| bad("a header")))
            .collect::<Done<Vec<_>>>()?;
        let names = headers
            .iter()
            .map(|h| CString::new(h.0.as_str()).map_err(|_| bad("a header name")))
            .collect::<Done<Vec<_>>>()?;
        let opts = options
            .iter()
            .map(|o| CString::new(o.as_str()).map_err(|_| bad("an option")))
            .collect::<Done<Vec<_>>>()?;
        let (texts, names, opts): (Vec<_>, Vec<_>, Vec<_>) = (
            texts.iter().map(|c| c.as_ptr()).collect(),
            names.iter().map(|c| c.as_ptr()).collect(),
            opts.iter().map(|c| c.as_ptr()).collect(),
        );
        let (create, compile, destroy): (Create, Compile, Destroy) = (
            self.function("create")?,
            self.function("compile")?,
            self.function("destroy")?,
        );
        let mut program: *mut c_void = ptr::null_mut();
        let header_count = c_int::try_from(texts.len())
            .map_err(|_| Refusal::bad("runtime compile: too many headers"))?;
        let option_count = c_int::try_from(opts.len())
            .map_err(|_| Refusal::bad("runtime compile: too many options"))?;
        // SAFETY: every pointer is a valid NUL-terminated string or array that outlives the call; counts match the arrays.
        let made = unsafe {
            create(
                &raw mut program,
                source_c.as_ptr(),
                name_c.as_ptr(),
                header_count,
                texts.as_ptr(),
                names.as_ptr(),
            )
        };
        if made != self.ok {
            return Err(Refusal::new(
                Kind::ToolFailed,
                format!("the runtime library could not create a program (status {made})"),
            ));
        }
        // SAFETY: `program` was just created; the option array is valid for `option_count` entries.
        let status = unsafe { compile(program, option_count, opts.as_ptr()) };
        let outcome = self.collect(program, status, target);
        // SAFETY: `program` is live and destroyed exactly once.
        unsafe { destroy(&raw mut program) };
        outcome
    }

    /// The log, and the image when `status` says the compile succeeded.
    fn collect(&self, program: *mut c_void, status: c_int, target: &str) -> Done<Compiled> {
        let call = |key: &str| wire::text(&self.calls, key, "runtime_compile.calls");
        let log = self.fetch(program, call("log_size")?, call("log")?)?;
        let log = String::from_utf8_lossy(&log)
            .trim_end_matches('\0')
            .to_string();
        if status != self.ok {
            return Ok(Compiled { log, image: None });
        }
        let images = wire::need(&self.calls, "images", "runtime_compile.calls")?;
        let kind = wire::need(images, target, "runtime_compile.calls.images")?;
        let image = self.fetch(
            program,
            wire::text(kind, "size", "image")?,
            wire::text(kind, "get", "image")?,
        )?;
        Ok(Compiled {
            log,
            image: Some(image),
        })
    }

    /// The bytes behind a size function and a fetch function of the library, given by their names.
    fn fetch(&self, program: *mut c_void, size_name: &str, get_name: &str) -> Done<Vec<u8>> {
        let (size, get): (SizeOf, Fetch) = (self.named(size_name)?, self.named(get_name)?);
        let mut n = 0_usize;
        // SAFETY: `program` is live and `n` is a valid out-pointer.
        if unsafe { size(program, &raw mut n) } != self.ok {
            return Err(Refusal::new(
                Kind::ToolFailed,
                "the runtime library could not state a size",
            ));
        }
        let mut bytes = vec![0_u8; n];
        // SAFETY: `bytes` has the `n` bytes the library stated.
        if unsafe { get(program, bytes.as_mut_ptr().cast::<c_char>()) } != self.ok {
            return Err(Refusal::new(
                Kind::ToolFailed,
                "the runtime library could not fetch its output",
            ));
        }
        Ok(bytes)
    }

    /// The function called `name` in the library.
    fn named<F: Copy>(&self, name: &str) -> Done<F> {
        // SAFETY: the caller states the signature `F` for the table's function name.
        unsafe { self.lib.function::<F>(name) }.ok_or_else(|| {
            Refusal::new(
                Kind::ToolAbsent,
                format!(
                    "the runtime library {} has no function {name}",
                    self.lib.name
                ),
            )
        })
    }
}
