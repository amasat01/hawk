//! The C function names an entry must not take: the functions the system headers and the unit's own headers declare with C linkage
//! (an entry is `extern "C"` and would redeclare them). Read from the preprocessed text of those headers by the compiler the
//! table names; the headers, flags and the include block of the unit are the table's and the ask's.
use crate::discover::{Machine, Tool};
use crate::refusal::{Done, Kind, Refusal};
use crate::toolkit::Toolkit;
use crate::witness::{Includes, ProbePlaces};
use crate::{compile, run, wire};
use rcore::json::J;
use rcore::sha256;
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// The words that start a declaration of something that is not a function.
const NOT_FUNCTIONS: [&str; 5] = ["typedef", "struct", "union", "enum", "class"];
/// The attribute and assembler-label words whose parenthesised argument is not part of the declaration.
const NOISE: [&str; 6] = [
    "__attribute__",
    "__asm__",
    "__asm",
    "asm",
    "__extension__",
    "__THROW",
];

/// The answers of this process, by key.
static MEMO: Mutex<Option<HashMap<String, Vec<String>>>> = Mutex::new(None);

/// Whether `c` can be part of an identifier.
fn word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// The byte offsets of the whole-word occurrences of `word` in `text`.
fn word_positions(text: &str, word: &str) -> Vec<usize> {
    text.match_indices(word)
        .map(|(at, _)| at)
        .filter(|at| {
            let before = text[..*at].chars().next_back();
            let after = text[at + word.len()..].chars().next();
            !before.is_some_and(word_char) && !after.is_some_and(word_char)
        })
        .collect()
}

/// `stmt` with `(name)` before an opening parenthesis written as `name`: how a header protects a function from a macro.
fn unwrap_parenthesised_names(stmt: &str) -> String {
    let mut out = String::new();
    let mut rest = stmt;
    while let Some(open) = rest.find('(') {
        out.push_str(&rest[..open]);
        let inner = rest[open + 1..].trim_start();
        let end = inner.find(|c: char| !word_char(c)).unwrap_or(inner.len());
        let (name, after) = inner.split_at(end);
        let closed = after.trim_start().strip_prefix(')');
        let wrapped = closed.filter(|tail| {
            !name.is_empty()
                && !name.starts_with(|c: char| c.is_ascii_digit())
                && tail.trim_start().starts_with('(')
        });
        if let Some(tail) = wrapped {
            out.push_str(name);
            rest = tail;
        } else {
            out.push('(');
            rest = &rest[open + 1..];
        }
    }
    out.push_str(rest);
    out
}

/// `stmt` without its first attribute or assembler-label group, or `None` when it has none.
fn without_first_noise(stmt: &str) -> Option<String> {
    let (at, word) = NOISE
        .iter()
        .flat_map(|w| word_positions(stmt, w).into_iter().map(move |at| (at, *w)))
        .min_by_key(|(at, _)| *at)?;
    let after = at + word.len();
    let Some(open) = stmt[after..].find('(').map(|j| after + j) else {
        return Some(format!("{}{}", &stmt[..at], &stmt[after..]));
    };
    let mut depth = 0_i32;
    let mut end = stmt.len();
    for (i, c) in stmt[open..].char_indices() {
        depth += i32::from(c == '(') - i32::from(c == ')');
        if depth == 0 {
            end = open + i + 1;
            break;
        }
    }
    Some(format!("{}{}", &stmt[..at], &stmt[end..]))
}

/// The first identifier of `stmt` that a `(` follows (white space allowed between): the name a declaration declares.
fn first_called(stmt: &str) -> Option<String> {
    let starts = stmt
        .char_indices()
        .filter(|(_, c)| c.is_ascii_alphabetic() || *c == '_');
    for (i, _) in starts {
        let rest = &stmt[i..];
        let end = rest.find(|c: char| !word_char(c)).unwrap_or(rest.len());
        if rest[end..].trim_start().starts_with('(') {
            return Some(rest[..end].to_string());
        }
    }
    None
}

/// The function a one-statement declaration declares, if it declares one.
fn declared_name(stmt: &str) -> Option<String> {
    if stmt.contains('{')
        || NOT_FUNCTIONS
            .iter()
            .any(|w| !word_positions(stmt, w).is_empty())
    {
        return None;
    }
    let mut stmt = unwrap_parenthesised_names(stmt);
    while let Some(shorter) = without_first_noise(&stmt) {
        stmt = shorter;
    }
    first_called(&stmt)
}

/// The text of the blocks that follow `extern "C" {`, up to their closing brace.
fn c_blocks(text: &str) -> Vec<&str> {
    let mut blocks = vec![];
    for (at, _) in text.match_indices("extern") {
        let rest = &text[at + "extern".len()..];
        let spaced = rest.trim_start();
        if spaced.len() == rest.len() {
            continue;
        }
        let Some(after_c) = spaced.strip_prefix("\"C\"") else {
            continue;
        };
        let Some(body) = after_c.trim_start().strip_prefix('{') else {
            continue;
        };
        let mut depth = 1_i32;
        let end = body
            .char_indices()
            .find(|(_, c)| {
                depth += i32::from(*c == '{') - i32::from(*c == '}');
                depth == 0
            })
            .map_or(body.len(), |(i, _)| i);
        blocks.push(&body[..end]);
    }
    blocks
}

/// The sorted function names declared inside the `extern "C" { ... }` blocks of preprocessed C++ `text`.
#[must_use]
pub fn declared_names(text: &str) -> Vec<String> {
    let mut names = BTreeSet::new();
    for block in c_blocks(text) {
        let mut depth = 0_i32;
        let mut start = 0;
        for (i, c) in block.char_indices() {
            depth += i32::from(c == '{') - i32::from(c == '}');
            if c == ';' && depth == 0 {
                names.extend(declared_name(&block[start..i]));
                start = i + 1;
            }
        }
    }
    names.into_iter().collect()
}

impl Toolkit {
    /// The reserved names under the ask's `reserved_names` record (`prelude`: the unit's include block) and the table's
    /// `reserved_names` record (`tool`, `args` with the places `{source}` and `{include:<tree>}`, `system_headers`).
    pub(crate) fn reserved_names(
        &self,
        machine: &Machine,
        prelude: &str,
        root: Option<&Path>,
    ) -> Done<Vec<String>> {
        let table = wire::need(&self.spec, "reserved_names", "the tool table")?;
        let tool_name = wire::text(table, "tool", "reserved_names")?;
        let tool = machine.tool(tool_name).ok_or_else(|| {
            Refusal::new(
                Kind::ToolAbsent,
                format!("reserved_names needs the tool {tool_name}"),
            )
        })?;
        let mut probe = prelude.to_string();
        for header in wire::texts(table, "system_headers", "reserved_names")? {
            let _ = writeln!(probe, "#include <{header}>");
        }
        let args = wire::texts(table, "args", "reserved_names")?;
        let key = sha256::hex(
            format!(
                "toolkit.reserved/1\0{}\0{probe}\0{}",
                tool.identity,
                args.join("\0")
            )
            .as_bytes(),
        );
        if let Some(hit) = MEMO
            .lock()
            .ok()
            .and_then(|m| m.as_ref().and_then(|m| m.get(&key).cloned()))
        {
            return Ok(hit);
        }
        let dir = scratch_dir()?;
        let outcome = self.preprocess(machine, tool, &args, &probe, root, &dir);
        let _ = std::fs::remove_dir_all(&dir);
        let names = declared_names(&outcome?);
        if let Ok(mut memo) = MEMO.lock() {
            memo.get_or_insert_with(HashMap::new)
                .insert(key, names.clone());
        }
        Ok(names)
    }

    /// The preprocessed text of `probe` by the table's arguments.
    fn preprocess(
        &self,
        machine: &Machine,
        tool: &Tool,
        args: &[String],
        probe: &str,
        root: Option<&Path>,
        dir: &Path,
    ) -> Done<String> {
        let source = dir.join("reserved.cpp");
        std::fs::write(&source, probe).map_err(|e| Refusal::io("cannot write", &source, &e))?;
        let mut command = vec![tool.path.display().to_string()];
        for arg in args {
            let places = ProbePlaces {
                source: &source,
                object: None,
                output: None,
                includes: Includes::Served(root),
            };
            command.push(compile::fill(arg, &mut |place| {
                self.probe_place(machine, place, &places)
            })?);
        }
        let done = run::run(&command, &self.child_env(machine)?, 0, run::PROBE_SECONDS)?;
        if done.code != Some(0) {
            return Err(Refusal::new(
                Kind::ToolFailed,
                format!(
                    "the preprocess for reserved names failed\n$ {}\n{}",
                    command.join(" "),
                    done.stderr
                ),
            ));
        }
        Ok(done.stdout)
    }
}

/// A fresh private folder for one probe.
pub(crate) fn scratch_dir() -> Done<PathBuf> {
    static COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "toolkit-probe-{}-{}",
        std::process::id(),
        COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).map_err(|e| Refusal::io("cannot create", &dir, &e))?;
    Ok(dir)
}

/// The `names` of the answer as JSON.
#[must_use]
pub fn stated(names: &[String]) -> J {
    wire::strs(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declarations_in_c_blocks_give_their_function_names() {
        let text = r#"
            int not_in_a_block(int);
            extern "C" {
              int abs(int);
              void *malloc(unsigned long) __attribute__((__malloc__(free, 1)));
              int (isnan)(double);
              typedef int foo_t;
              struct s { int a; int (*f)(int); };
              extern int __asm__("alias") weird(void);
              void exit(int) __THROW;
              double fma(double, double, double) __attribute__ ((const));
            }
            extern "C++" { int cpp_only(int); }
        "#;
        assert_eq!(
            declared_names(text),
            ["abs", "exit", "fma", "isnan", "malloc", "weird"]
        );
    }

    #[test]
    fn a_nested_block_is_read_on_its_own_and_a_statement_that_holds_a_brace_declares_nothing() {
        let text = "extern \"C\" { plain(int); extern \"C\" { inner(void); } after(void); }";
        assert_eq!(declared_names(text), ["inner", "plain"]);
    }
}
