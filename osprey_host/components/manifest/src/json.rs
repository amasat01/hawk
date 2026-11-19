//! JSON text out of a parsed value, and the Python spelling of a value inside a refusal sentence.
use rcore::json::J;
use std::fmt::Write;

/// `text` as a JSON string literal.
fn quoted(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A number as JSON: a whole number without a fraction.
fn number(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{value:.0}")
    } else {
        format!("{value}")
    }
}

/// The JSON text of `value`, an object keeping the order of its keys.
#[must_use]
pub fn emit(value: &J) -> String {
    match value {
        J::Null => "null".into(),
        J::Bool(b) => b.to_string(),
        J::Num(n) => number(*n),
        J::Str(s) => quoted(s),
        J::Arr(items) => format!("[{}]", items.iter().map(emit).collect::<Vec<_>>().join(",")),
        J::Obj(pairs) => format!(
            "{{{}}}",
            pairs
                .iter()
                .map(|(k, v)| format!("{}:{}", quoted(k), emit(v)))
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

/// `text` as a sentence quotes it: single quotes, double ones when the text holds a single quote and no double one.
#[must_use]
pub fn quote(text: &str) -> String {
    let double = text.contains('\'') && !text.contains('"');
    let edge = if double { '"' } else { '\'' };
    let mut out = String::from(edge);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == edge => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(edge);
    out
}

/// A value as a sentence names it: a string quoted, anything else as its JSON text.
#[must_use]
pub fn name(value: &J) -> String {
    match value {
        J::Str(s) => quote(s),
        other => emit(other),
    }
}

/// `items` as a list is spelled in a sentence: `['a', 'b']`.
#[must_use]
pub fn python_list(items: &[&str]) -> String {
    format!(
        "[{}]",
        items
            .iter()
            .map(|i| quote(i))
            .collect::<Vec<_>>()
            .join(", ")
    )
}
