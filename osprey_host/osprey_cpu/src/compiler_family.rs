//! The grammar `gcc_family`, which a tool probe of the table names in its `as`: it tells GCC from the compilers that define its
//! macros too, by the predefined macros of `-dM -E`.
use rcore::json::J;
use toolkit::wire;

/// The grammar as the toolkit registers it: read from the tool's output; no image is read.
pub(crate) fn gcc_family_grammar(macros: &str, _image: &[u8]) -> J {
    gcc_family(macros)
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

#[cfg(test)]
mod tests {
    use super::*;

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
