//! JSON in and out: the request accessors that refuse by name, the value builders and the writer. The reader is the tree's own
//! (`rcore::json`); it has no writer, so this module has the one the tool side needs.
use crate::refusal::{Done, Refusal};
use rcore::json::{self, J};
use std::fmt::Write;

/// Parses `text`, refusing with the reader's own message.
///
/// # Errors
/// Refuses with a [`Refusal`](crate::refusal::Refusal) that names what is wrong with the request or the tool.
pub fn parse(what: &str, text: &str) -> Done<J> {
    json::parse(text).map_err(|e| Refusal::bad(format!("{what} is not JSON: {e}")))
}

/// The member `key` of the object `j`, refused by name when it is missing.
///
/// # Errors
/// Refuses with a [`Refusal`](crate::refusal::Refusal) that names what is wrong with the request or the tool.
pub fn need<'a>(j: &'a J, key: &str, what: &str) -> Done<&'a J> {
    j.get(key)
        .ok_or_else(|| Refusal::bad(format!("{what}: member {key} is missing")))
}

/// The text member `key` of `j`, refused by name when it is missing or not text.
///
/// # Errors
/// Refuses with a [`Refusal`](crate::refusal::Refusal) that names what is wrong with the request or the tool.
pub fn text<'a>(j: &'a J, key: &str, what: &str) -> Done<&'a str> {
    need(j, key, what)?
        .str()
        .ok_or_else(|| Refusal::bad(format!("{what}: member {key} is not text")))
}

/// The text member `key` of `j`, or `default` when it is absent.
#[must_use]
pub fn text_or<'a>(j: &'a J, key: &str, default: &'a str) -> &'a str {
    j.get(key).and_then(J::str).unwrap_or(default)
}

/// The list of texts under `key` of `j`; absent means an empty list, a non-text element is refused.
///
/// # Errors
/// Refuses with a [`Refusal`](crate::refusal::Refusal) that names what is wrong with the request or the tool.
pub fn texts(j: &J, key: &str, what: &str) -> Done<Vec<String>> {
    let Some(list) = j.get(key) else {
        return Ok(vec![]);
    };
    list.arr()
        .iter()
        .map(|x| {
            x.str().map(str::to_string).ok_or_else(|| {
                Refusal::bad(format!("{what}: {key} holds an element that is not text"))
            })
        })
        .collect()
}

/// A JSON string.
#[must_use]
pub fn s(x: &str) -> J {
    J::Str(x.to_string())
}

/// A JSON number from a count.
#[must_use]
#[allow(clippy::cast_precision_loss)] // counts here stay far below 2^53
pub fn n(x: u64) -> J {
    J::Num(x as f64)
}

/// A JSON array of strings.
#[must_use]
pub fn strs<S: AsRef<str>>(items: &[S]) -> J {
    J::Arr(items.iter().map(|x| s(x.as_ref())).collect())
}

/// A JSON object from key and value pairs, in the order given.
#[must_use]
pub fn obj(pairs: Vec<(&str, J)>) -> J {
    J::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// The JSON text of `j`, compact, keys in the order held.
#[must_use]
pub fn render(j: &J) -> String {
    let mut out = String::new();
    put(j, &mut out);
    out
}

/// Appends the JSON text of `j` to `out`.
fn put(j: &J, out: &mut String) {
    match j {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Num(x) if x.fract() == 0.0 && x.abs() < 9e15 => {
            let _ = write!(out, "{x:.0}");
        }
        J::Num(x) => {
            let _ = write!(out, "{x}");
        }
        J::Str(t) => quote(t, out),
        J::Arr(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                put(item, out);
            }
            out.push(']');
        }
        J::Obj(pairs) => {
            out.push('{');
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                quote(k, out);
                out.push(':');
                put(v, out);
            }
            out.push('}');
        }
    }
}

/// Appends `t` as a JSON string literal.
fn quote(t: &str, out: &mut String) {
    out.push('"');
    for c in t.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rendered_value_parses_back_to_the_same_text() {
        let j = obj(vec![
            ("a", s("q\"\\\n\u{1}é")),
            ("b", J::Arr(vec![n(3), J::Null, J::Bool(true)])),
            ("c", obj(vec![])),
        ]);
        let text = render(&j);
        assert_eq!(render(&parse("t", &text).unwrap()), text);
        assert_eq!(
            text,
            "{\"a\":\"q\\\"\\\\\\n\\u0001é\",\"b\":[3,null,true],\"c\":{}}"
        );
    }

    #[test]
    fn a_missing_member_is_refused_by_name() {
        let j = parse("t", "{}").unwrap();
        assert!(text(&j, "name", "unit")
            .unwrap_err()
            .message
            .contains("member name is missing"));
    }
}
