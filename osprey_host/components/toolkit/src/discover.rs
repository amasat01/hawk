//! Tool discovery by a table the OSPREY passes: for each named tool an ordered list of rules (an environment variable, a name on
//! `PATH`, a flag inside another variable, another tool), then the probes that state its identity and facts. Nothing here names a
//! tool: the table does. A rule that cannot be satisfied falls through to the next, except an environment variable marked
//! `required`, which is an explicit override and refuses when it does not resolve.
use crate::refusal::{Done, Kind, Refusal};
use crate::run;
use crate::shellwords;
use crate::store::{Built, Store};
use crate::wire;
use rcore::json::J;
use rcore::sha256;
use std::collections::HashMap;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

/// The environment variables of a discovery, a snapshot so a test can state its own. The snapshot of the process environment is
/// [`fresh`](Env::fresh)ed by whoever asks, so a variable changed between two asks is read as it then is.
#[derive(Clone, Debug, Default)]
pub struct Env {
    /// Variable name to value.
    vars: HashMap<String, String>,
    /// Whether this is the environment of the process, to be read again when asked for.
    live: bool,
}

impl Env {
    /// The environment of this process.
    #[must_use]
    pub fn process() -> Self {
        Env {
            vars: std::env::vars().collect(),
            live: true,
        }
    }

    /// The environment as it is now: read again for the process environment, the same snapshot for a stated one.
    #[must_use]
    pub fn fresh(&self) -> Env {
        if self.live {
            Env::process()
        } else {
            self.clone()
        }
    }

    /// An environment of exactly `pairs`.
    #[must_use]
    pub fn from_pairs(pairs: &[(&str, &str)]) -> Self {
        Env {
            vars: pairs
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect(),
            live: false,
        }
    }

    /// The value of `name` when it is set and not empty.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&str> {
        self.vars
            .get(name)
            .map(String::as_str)
            .filter(|v| !v.is_empty())
    }

    /// The variables, for a child process.
    #[must_use]
    pub fn vars(&self) -> &HashMap<String, String> {
        &self.vars
    }

    /// The folders of `PATH`, in order.
    fn path_dirs(&self) -> Vec<PathBuf> {
        self.get("PATH")
            .map_or_else(Vec::new, |p| std::env::split_paths(p).collect())
    }
}

/// One tool found: where it is, its identity (the term a key carries) and what its probes stated.
#[derive(Clone, Debug)]
pub struct Tool {
    /// The absolute path of the executable.
    pub path: PathBuf,
    /// `<path>||<first line of its version text>||<16 hex digits of the whole text>`.
    pub identity: String,
    /// The probe results by probe name.
    pub facts: Vec<(String, J)>,
}

/// Every tool of a table, in the table's order.
#[derive(Clone, Debug, Default)]
pub struct Machine {
    /// Tool name and the tool.
    pub tools: Vec<(String, Tool)>,
    /// Tools the table lets a runtime compile library stand in for, that were not used: name and why (not found, or forced).
    pub replaced: Vec<(String, String)>,
    /// Why a tool is missing (every missing tool's reason, in order), `None` when all are found or replaced.
    pub absent: Option<String>,
}

impl Machine {
    /// The tool called `name`.
    #[must_use]
    pub fn tool(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }
}

/// What one rule gave.
enum Step {
    /// The tool is here.
    Hit(PathBuf),
    /// This rule does not apply; the note says why, for the absence text.
    Miss(String),
    /// An explicit override that does not resolve: discovery stops with this text.
    Stop(String),
}

/// How the table lets a runtime compile library stand in for a tool.
enum Swap {
    /// The tool is used when found, the library when not.
    Allow,
    /// The library is used, the tool is not looked for.
    Prefer,
    /// Only the tool is used.
    Never,
}

/// How the `runtime_compile` member of `spec` treats the tool `name`: `None` when it does not replace it. A `force` record
/// (`env`, `tool`, `runtime`) lets an environment variable choose; a value that is neither word is refused.
fn swap_of(spec: &J, env: &Env, name: &str) -> Done<Option<Swap>> {
    let Some(runtime) = spec
        .get("runtime_compile")
        .filter(|r| r.get("replaces").and_then(J::str) == Some(name))
    else {
        return Ok(None);
    };
    let Some(force) = runtime.get("force") else {
        return Ok(Some(Swap::Allow));
    };
    let var = wire::text(force, "env", "runtime_compile.force")?;
    let (tool, library) = (
        wire::text(force, "tool", "runtime_compile.force")?,
        wire::text(force, "runtime", "runtime_compile.force")?,
    );
    match env.get(var) {
        None => Ok(Some(Swap::Allow)),
        Some(v) if v == tool => Ok(Some(Swap::Never)),
        Some(v) if v == library => Ok(Some(Swap::Prefer)),
        Some(v) => Err(Refusal::bad(format!(
            "${var}={v:?} must be {tool:?} or {library:?}"
        ))),
    }
}

/// Finds every tool of `spec["tools"]` in `env`. A tool that `runtime_compile` replaces and that is not found (or is forced out) is
/// listed in [`Machine::replaced`]; any other missing tool is stated in [`Machine::absent`] and discovery goes on, so everything that
/// needs no missing tool can still be stated.
///
/// # Errors
/// Refuses a table that is malformed.
pub fn find(spec: &J, env: &Env, root: Option<&Path>) -> Done<Machine> {
    let mut machine = Machine::default();
    let mut missing: Vec<String> = vec![];
    for (name, tool_spec) in wire::need(spec, "tools", "tool table")?.obj() {
        let swap = swap_of(spec, env, name)?;
        if matches!(swap, Some(Swap::Prefer)) {
            machine.replaced.push((
                name.clone(),
                "the environment selects the runtime compile".into(),
            ));
            continue;
        }
        let found = match locate(name, tool_spec, env, &machine)? {
            Ok(path) => describe(&path, tool_spec, env, root)?,
            Err(reason) => Err(reason),
        };
        match (found, swap) {
            (Ok(tool), _) => machine.tools.push((name.clone(), tool)),
            (Err(reason), Some(Swap::Allow)) => machine.replaced.push((name.clone(), reason)),
            (Err(reason), _) => missing.push(reason),
        }
    }
    machine.absent = (!missing.is_empty()).then(|| missing.join("; "));
    Ok(machine)
}

/// The path of the tool `name` by its rules, or the reason it is absent.
fn locate(name: &str, spec: &J, env: &Env, machine: &Machine) -> Done<Result<PathBuf, String>> {
    let mut notes = vec![];
    for rule in wire::need(spec, "rules", name)?.arr() {
        match apply(rule, env, machine)? {
            Step::Hit(path) => return Ok(Ok(path)),
            Step::Miss(note) => notes.push(note),
            Step::Stop(reason) => return Ok(Err(format!("{reason}; {}", fix(spec)))),
        }
    }
    Ok(Err(format!(
        "no {name} found. Tried, in order: {}; {}",
        notes.join(", "),
        fix(spec)
    )))
}

/// The tool's own way out, as its table states it.
fn fix(spec: &J) -> &str {
    wire::text_or(
        spec,
        "fix",
        "name the tool in the environment variable of the table",
    )
}

/// Applies one rule.
fn apply(rule: &J, env: &Env, machine: &Machine) -> Done<Step> {
    let child = rule.get("dir_child").and_then(J::str);
    if let Some(var) = rule.get("env").and_then(J::str) {
        let Some(value) = env.get(var) else {
            return Ok(Step::Miss(format!("${var} (unset)")));
        };
        let value = if matches!(rule.get("first_word"), Some(J::Bool(true))) {
            value.split_whitespace().next().unwrap_or("")
        } else {
            value
        };
        return Ok(match resolve(value, child, env) {
            Some(path) => Step::Hit(path),
            None if matches!(rule.get("required"), Some(J::Bool(true))) => {
                Step::Stop(format!("${var}={value:?} not found on PATH"))
            }
            None => Step::Miss(format!("${var} ({value:?}, did not resolve)")),
        });
    }
    if let Some(name) = rule.get("path").and_then(J::str) {
        return Ok(resolve(name, child, env)
            .map_or_else(|| Step::Miss(format!("{name} on PATH (absent)")), Step::Hit));
    }
    if let Some(flag) = rule.get("env_flag") {
        return env_flag(flag, child, env);
    }
    if let Some(other) = rule.get("tool").and_then(J::str) {
        return Ok(machine.tool(other).map_or_else(
            || Step::Miss(format!("the tool {other} (absent)")),
            |t| Step::Hit(t.path.clone()),
        ));
    }
    Err(Refusal::bad(
        "a discovery rule names none of env, path, env_flag or tool",
    ))
}

/// The rule that reads a flag's value out of an environment variable holding flags (`-ccbin <x>` in `NVCC_PREPEND_FLAGS`).
fn env_flag(rule: &J, child: Option<&str>, env: &Env) -> Done<Step> {
    let var = wire::text(rule, "var", "env_flag rule")?;
    let flags = wire::texts(rule, "flags", "env_flag rule")?;
    let note = || Step::Miss(format!("a {} flag in ${var}", flags.join(" or ")));
    let Some(text) = env.get(var) else {
        return Ok(note());
    };
    Ok(flag_value(text, &flags)
        .and_then(|value| resolve(&value, child, env))
        .map_or_else(note, Step::Hit))
}

/// The value after the first of `flags` in the shell-style words of `text`, in the forms `-f v` and `-f=v`; `None` when there is
/// none or the text is not a word list (an unclosed quote).
#[must_use]
pub fn flag_value(text: &str, flags: &[String]) -> Option<String> {
    let words = shellwords::split(text)?;
    for (i, word) in words.iter().enumerate() {
        for flag in flags {
            if word == flag {
                return words.get(i + 1).cloned();
            }
            if let Some(value) = word.strip_prefix(&format!("{flag}=")) {
                return Some(value.to_string());
            }
        }
    }
    None
}

/// The shell-style words of `text` without each of `flags` and its value (both forms), joined so they split back to the same
/// words; empty when `text` is not a word list.
#[must_use]
pub fn without_flags(text: &str, flags: &[String]) -> String {
    let mut kept = vec![];
    let mut skip = false;
    for word in shellwords::split(text).unwrap_or_default() {
        if skip {
            skip = false;
        } else if flags.contains(&word) {
            skip = true;
        } else if !flags.iter().any(|f| word.starts_with(&format!("{f}="))) {
            kept.push(word);
        }
    }
    shellwords::join(&kept)
}

/// Whether `path` is a regular file with an execute bit.
fn is_executable(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

/// The executable `value` names: a path (a folder resolves to its `child` when the rule names one) or a name looked up on `PATH`.
fn resolve(value: &str, child: Option<&str>, env: &Env) -> Option<PathBuf> {
    let found = if value.contains('/') {
        let mut p = PathBuf::from(value);
        if let (true, Some(c)) = (p.is_dir(), child) {
            p.push(c);
        }
        is_executable(&p).then_some(p)
    } else {
        env.path_dirs()
            .into_iter()
            .map(|d| d.join(value))
            .find(|p| is_executable(p))
    };
    found.and_then(|p| std::path::absolute(p).ok())
}

/// The memo of probe answers: the same executable (path, size, modification time) asked the same arguments answers once.
type Memo = HashMap<(PathBuf, u64, u128, Vec<String>), String>;
/// The probe answers of this process.
static PROBES: Mutex<Option<Memo>> = Mutex::new(None);

/// Forgets every memoised probe answer.
pub fn forget_probes() {
    if let Ok(mut memo) = PROBES.lock() {
        *memo = None;
    }
}

/// The standard output of `<path> <args>`, or why it did not run: the operating system's error when it could not be started,
/// else the exit code and the tool's own complaint. A success is memoised per executable for this process and, when `root` is
/// given, kept in the store under it, keyed on the executable's path, size and modification time and the arguments, so a fresh
/// process and a second process do not ask the tool again and a replaced executable never reads the old answer.
fn probe(path: &Path, args: &[String], env: &Env, root: Option<&Path>) -> Result<String, String> {
    let unreadable = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
    let meta = std::fs::metadata(path).map_err(|e| unreadable(&e))?;
    let mtime = meta
        .modified()
        .map_err(|e| unreadable(&e))?
        .duration_since(UNIX_EPOCH)
        .map_err(|e| unreadable(&e))?
        .as_nanos();
    let key = (path.to_path_buf(), meta.len(), mtime, args.to_vec());
    let remembered = PROBES
        .lock()
        .ok()
        .and_then(|m| m.as_ref().and_then(|m| m.get(&key).cloned()));
    if let Some(hit) = remembered {
        return Ok(hit);
    }
    let ask = || {
        let mut command = vec![path.display().to_string()];
        command.extend_from_slice(args);
        let done = run::run(&command, env.vars(), 0, run::PROBE_SECONDS).map_err(|r| r.message)?;
        if done.code == Some(0) {
            return Ok(done.stdout);
        }
        let code = done
            .code
            .map_or("no exit code".to_string(), |c| format!("exit {c}"));
        Err(format!("{code}: {}", done.stderr.trim()))
    };
    let answer = match root {
        Some(root) => kept(root, &key, ask)?,
        None => ask()?,
    };
    if let Ok(mut memo) = PROBES.lock() {
        memo.get_or_insert_with(HashMap::new)
            .insert(key, answer.clone());
    }
    Ok(answer)
}

/// The answer of `ask` for the probe `key`, kept as a slot of the store under `root/probes`: built once however many processes
/// ask, published whole. A failed probe leaves no slot. A store that cannot be used (a read-only root) costs only the keeping.
fn kept(
    root: &Path,
    key: &(PathBuf, u64, u128, Vec<String>),
    ask: impl Fn() -> Result<String, String>,
) -> Result<String, String> {
    let (path, size, mtime, args) = key;
    let mut terms = vec![
        path.display().to_string(),
        size.to_string(),
        mtime.to_string(),
    ];
    terms.extend_from_slice(args);
    let slot = sha256::hex(terms.join("\0").as_bytes());
    let failed = std::cell::RefCell::new(None);
    let entry = Store::new(&root.join("probes")).keep(&slot, |_| {
        ask()
            .map_err(|why| {
                *failed.borrow_mut() = Some(why);
                Refusal::new(Kind::ToolFailed, "the probe failed")
            })
            .map(|text| Built {
                files: vec![],
                meta: wire::s(&text),
            })
    });
    match (entry, failed.into_inner()) {
        (Ok(slot), _) => slot
            .meta
            .str()
            .map(String::from)
            .ok_or_else(|| "a kept probe answer is not text".to_string()),
        (Err(_), Some(why)) => Err(why),
        (Err(_), None) => ask(),
    }
}

/// The identity of the tool at `path` and the facts of its probes, or the reason it cannot be asked.
fn describe(path: &Path, spec: &J, env: &Env, root: Option<&Path>) -> Done<Result<Tool, String>> {
    let args = match spec.get("identity_args") {
        Some(_) => wire::texts(spec, "identity_args", "tool")?,
        None => vec!["--version".to_string()],
    };
    let text = match probe(path, &args, env, root) {
        Ok(text) => text,
        Err(why) => {
            return Ok(Err(format!(
                "{} was found but `{}` does not run ({why})",
                path.display(),
                args.join(" ")
            )))
        }
    };
    let first = text.lines().next().unwrap_or("").trim();
    let digest = &sha256::hex(text.as_bytes())[..16];
    let identity = format!("{}||{first}||{digest}", path.display());
    let mut facts = vec![];
    for p in spec.get("probes").map_or(&[][..], J::arr) {
        let name = wire::text(p, "name", "probe")?;
        let answer = probe(path, &wire::texts(p, "args", "probe")?, env, root).ok();
        facts.push((name.to_string(), read_probe(p, answer.as_deref())?));
    }
    Ok(Ok(Tool {
        path: path.to_path_buf(),
        identity,
        facts,
    }))
}

/// What a probe's output states, by its `as`: `version` (the `N.M` after a marker), `int_list` (the numbers after each occurrence
/// of a marker) or `gcc_family` (read from predefined macros). No output states `null`.
fn read_probe(probe: &J, output: Option<&str>) -> Done<J> {
    let Some(out) = output else {
        return Ok(J::Null);
    };
    let marker = wire::text_or(probe, "after", "");
    Ok(match wire::text(probe, "as", "probe")? {
        "version" => version_after(out, marker).map_or(J::Null, |v| wire::s(&v)),
        "int_list" => {
            let mut all = numbers_after(out, marker);
            all.sort_unstable();
            all.dedup();
            J::Arr(all.into_iter().map(wire::n).collect())
        }
        "gcc_family" => gcc_family(out),
        other => {
            return Err(Refusal::bad(format!(
                "probe: `as` {other:?} is not version, int_list or gcc_family"
            )))
        }
    })
}

/// The `N.M` that follows `marker` in `text`.
fn version_after(text: &str, marker: &str) -> Option<String> {
    let rest = &text[text.find(marker)? + marker.len()..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(rest.len());
    let mut parts = rest[..end].split('.');
    let (major, minor) = (parts.next()?, parts.next()?);
    (!major.is_empty() && !minor.is_empty()).then(|| format!("{major}.{minor}"))
}

/// The digit runs right after each occurrence of `marker` in `text`.
fn numbers_after(text: &str, marker: &str) -> Vec<u64> {
    text.match_indices(marker)
        .filter_map(|(at, _)| {
            let rest = &text[at + marker.len()..];
            let end = rest
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len());
            rest[..end].parse().ok()
        })
        .collect()
}

/// `{"gcc": bool, "gcc_release": "major.minor" | null}` from the output of `-dM -E`: GCC proper defines `__GNUC__` and not
/// `__clang__`.
fn gcc_family(macros: &str) -> J {
    let value = |name: &str| {
        macros.lines().find_map(|l| {
            let mut w = l.split_whitespace();
            (w.next() == Some("#define") && w.next() == Some(name))
                .then(|| w.next().unwrap_or("").to_string())
        })
    };
    let gcc = value("__GNUC__").is_some() && value("__clang__").is_none();
    let release = value("__GNUC__").filter(|_| gcc).map(|major| {
        format!(
            "{major}.{}",
            value("__GNUC_MINOR__").unwrap_or_else(|| "0".into())
        )
    });
    wire::obj(vec![
        ("gcc", J::Bool(gcc)),
        ("gcc_release", release.map_or(J::Null, |r| wire::s(&r))),
    ])
}

/// The environment a tool of `machine` runs under: `env` as the table's `process_env` adjusts it: `strip_flags` takes a flag and
/// its value out of the listed variables (the one on the command line is the only one), `remove` drops variables, `path_prepend`
/// puts the folder of a tool first on `PATH` and `root_of` names a variable that is the folder above that folder.
///
/// # Errors
/// Refuses with a [`Refusal`](crate::refusal::Refusal) that names what is wrong with the request or the tool.
pub fn child_env(spec: &J, env: &Env, machine: &Machine) -> Done<HashMap<String, String>> {
    let mut vars = env.vars().clone();
    let Some(rules) = spec.get("process_env") else {
        return Ok(vars);
    };
    if let Some(strip) = rules.get("strip_flags") {
        let flags = wire::texts(strip, "flags", "strip_flags")?;
        for var in wire::texts(strip, "vars", "strip_flags")? {
            let Some(old) = vars.get(&var) else { continue };
            let rest = without_flags(old, &flags);
            if rest.is_empty() {
                vars.remove(&var);
            } else {
                vars.insert(var, rest);
            }
        }
    }
    for var in wire::texts(rules, "remove", "process_env")? {
        vars.remove(&var);
    }
    for name in wire::texts(rules, "path_prepend", "process_env")? {
        let dir = tool_dir(machine, &name)?;
        let old = vars.get("PATH").cloned().unwrap_or_default();
        vars.insert("PATH".into(), format!("{}:{old}", dir.display()));
    }
    for (var, name) in rules.get("root_of").map_or(&[][..], J::obj) {
        let dir = tool_dir(machine, name.str().unwrap_or(""))?;
        let root = dir.parent().unwrap_or(&dir);
        vars.insert(var.clone(), root.display().to_string());
    }
    Ok(vars)
}

/// The folder holding the tool `name`.
fn tool_dir(machine: &Machine, name: &str) -> Done<PathBuf> {
    machine
        .tool(name)
        .and_then(|t| t.path.parent().map(Path::to_path_buf))
        .ok_or_else(|| {
            Refusal::bad(format!(
                "process_env names the tool {name}, which the table does not find"
            ))
        })
}

/// The stated record of one tool: its path, identity and probe facts.
#[must_use]
pub fn stated(tool: &Tool) -> J {
    let mut pairs = vec![
        ("path", wire::s(&tool.path.display().to_string())),
        ("identity", wire::s(&tool.identity)),
    ];
    pairs.extend(tool.facts.iter().map(|(k, v)| (k.as_str(), v.clone())));
    wire::obj(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_are_read_and_removed_in_both_forms() {
        let flags = ["-ccbin".to_string(), "--compiler-bindir".to_string()];
        assert_eq!(
            flag_value("-O2 -ccbin /x/g++ -g", &flags).as_deref(),
            Some("/x/g++")
        );
        assert_eq!(
            flag_value("--compiler-bindir=/y", &flags).as_deref(),
            Some("/y")
        );
        assert_eq!(
            without_flags("-O2 -ccbin /x/g++ --compiler-bindir=/y -g", &flags),
            "-O2 -g"
        );
    }

    #[test]
    fn a_version_and_a_number_list_are_read_after_their_markers() {
        let text =
            "Cuda compilation tools, release 12.6, V12.6.20\n compute_61\n compute_75\n compute_61";
        assert_eq!(version_after(text, "release ").as_deref(), Some("12.6"));
        assert_eq!(numbers_after(text, "compute_"), vec![61, 75, 61]);
        assert_eq!(version_after("no marker", "release "), None);
    }

    #[test]
    fn gcc_is_told_from_clang_by_the_predefined_macros() {
        let gcc = gcc_family("#define __GNUC__ 13\n#define __GNUC_MINOR__ 2\n");
        assert_eq!(
            wire::render(&gcc),
            "{\"gcc\":true,\"gcc_release\":\"13.2\"}"
        );
        let clang = gcc_family("#define __GNUC__ 4\n#define __clang__ 1\n");
        assert_eq!(wire::render(&clang), "{\"gcc\":false,\"gcc_release\":null}");
    }
}
