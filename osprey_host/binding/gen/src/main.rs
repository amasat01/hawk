//! gen [--header H] [--out F]            writes the generated C of the extension from the door header
//! gen --check [--header H --template D --generated F --ext SO --door-header H --door-source C]   the gate
//! std only. A prototype the generator does not understand is refused, never skipped.
use std::collections::BTreeSet;
use std::fs;
use std::process::Command;

const LIFECYCLE: [&str; 6] = [
    "ob_last_code",
    "rx_abi_version",
    "rx_ctx_new",
    "rx_ctx_free",
    "rx_last_error",
    "rx_last_code",
];
const TEMPLATE_ALLOW: [&str; 7] = [
    "Ctx_new",
    "Ctx_dealloc",
    "Node_dealloc",
    "Node_new",
    "Node_richcompare",
    "Node_hash",
    "PyInit__osprey",
];
const TEMPLATE_RX: [&str; 3] = ["rx_ctx_new", "rx_ctx_free", "rx_abi_version"];
// the Node surface: python name, door function, getter (true) or method (false). A front end reads back what it made through
// these: the class, the operands, the type and the typed fields of a node (raptor_ir_front.h); every other question is the backend's
const NODE_SURFACE: [(&str, &str, bool); 13] = [
    ("cls", "rx_class", true),
    ("operands", "rx_operands", true),
    ("ttype", "rx_ttype_of", true),
    ("field_kind", "rx_field_kind", false),
    ("field_text", "rx_field_text", false),
    ("field_int", "rx_field_int", false),
    ("field_real", "rx_field_real", false),
    ("field_flag", "rx_field_flag", false),
    ("field_nodes", "rx_field_nodes", false),
    ("field_node_list", "rx_field_node_list", false),
    ("field_ints", "rx_field_ints", false),
    ("field_names", "rx_field_names", false),
    ("field_pairs", "rx_field_pairs", false),
];
// Prototypes of the headers the binding does not wrap: the runtime lookup, which the door module of the template calls. Every other shape has a rule, and a prototype without one is refused.
const NOT_WRAPPED: [&str; 1] = ["ob_runtime_symbol"];
// the entries of the tool side: a refusal of theirs carries a code (the refusal kind's name), read from the library handle
const CODED: [&str; 4] = ["ob_machine", "ob_build", "ob_link", "ob_state"];
// The `int*` outputs that are codes, not flags (a flag answers a Python bool).
const CODE_OUTPUTS: [&str; 2] = ["rx_field_kind", "rw_wide_element_kind"];
// The prefix of the readers whose status RX_ABSENT answers None.
const ABSENT_READERS: &str = "rx_field_";
// The pair readers whose answer is one row per first part: the pairs of one first part are together, and the answer is a tuple of
// (first part, tuple of the second parts).
const GROUPED_PAIRS: [&str; 2] = ["rx_walk_quantity_wires", "rx_walk_quantity_slots"];
const KEYWORDS: [&str; 5] = ["class", "def", "import", "from", "lambda"];

#[derive(Debug, Clone)]
enum P {
    NodeIn(String),
    WalkIn(String),
    Cstr(String, bool),
    Text(String, String),
    I64s(String, String),
    /// One list of nodes; `group` names the length it shares (see `NodeLists`).
    NodeList(String, usize),
    /// The length of a group of node lists, at its place in the C call.
    GroupLen(usize),
    /// A node or None (a pointer that may be NULL).
    OptNode(String),
    /// An integer of 64 bits passed by value.
    I64In(String),
    /// An int passed by value.
    IntIn(String),
    OutText,
    OutNodes,
    /// A node and then a list of nodes, both answers (the base of a chain and its links).
    OutNodeList,
    OutNode,
    OutWalk,
    OutChars(usize),
    HandleIn(String),
    OutHandle,
    OutParts,
    OutInt,
    OutCode,
    OutI64,
    OutF64,
    OutType,
    OutI64s,
    OutNames,
    OutPairs,
    /// An integer passed by pointer that may be NULL (`/* NULL */`): a Python int or None.
    OptI64(String),
    /// One scalar answer before the outputs are told apart: `i` an int, `n` a node, `l` a 64-bit integer, with its name.
    Scalar(char, String),
    /// Several scalar answers of one call, as a tuple in call order; a first one named `found` answers None when 0 and is not listed.
    OutScalars(Vec<(char, String)>),
    /// An `int` flag after a list of nodes the call made: the answer is the pair (nodes, flag).
    TrailFlag,
    /// `int* flags` after the node list of that name: one flag per node of it, as a tuple of bool.
    OutFlags(String),
    /// A `rw_key_run*`: None for no key run, else (form, width, block).
    OutKeyRun,
    /// A `rw_gather*`: None for no run, else (plane, base or None, start, step, clamped, lead nodes).
    OutGather,
    /// A `rw_compensated_target*` and the text buffer after it: None when no sink is compensated, else (target name, own column).
    OutCompensated,
    /// Rows of `rw_plane_row`, the list protocol: a tuple of (plane node, layout, width).
    OutPlaneRows,
}

struct Proto {
    name: String,
    params: Vec<P>,
    text: String,
}

/// the header without block comments; an optional-pointer comment (`/* NULL ...*/`) becomes the marker `@opt`
fn strip(src: &str) -> String {
    let mut out = String::new();
    let mut rest = src;
    while let Some(i) = rest.find("/*") {
        out.push_str(&rest[..i]);
        let j = rest[i..].find("*/").expect("unterminated comment") + i;
        let body = rest[i + 2..j].trim();
        if body.starts_with("NULL") {
            out.push_str(" @opt")
        }
        rest = &rest[j + 2..];
    }
    out.push_str(rest);
    out
}

fn ident_last(s: &str) -> String {
    s.trim_end_matches(|c: char| !(c.is_alphanumeric() || c == '_'))
        .rsplit(|c: char| !(c.is_alphanumeric() || c == '_'))
        .next()
        .unwrap_or("")
        .to_string()
}

fn parse_params(raw: &str, whole: &str) -> Result<Vec<P>, String> {
    let bad = |m: &str| format!("cannot wrap `{whole}`: {m}");
    let ps: Vec<String> = raw
        .split(',')
        .map(|p| p.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect();
    let norm = |p: &str| {
        p.replace(" *", "*")
            .replace("* ", "*")
            .replace("@opt", "")
            .trim()
            .to_string()
    };
    let shape = |p: &str| {
        let n = norm(p);
        let k = n.rfind(['*', ' ']).map_or(0, |i| i + 1);
        (n[..k].trim().to_string(), n[k..].to_string())
    };
    if ps.is_empty() || shape(&ps[0]) != ("rx_ctx*".into(), "ctx".into()) {
        return Err(bad("the first parameter is not `rx_ctx* ctx`"));
    }
    let mut out = vec![];
    let mut i = 1;
    while i < ps.len() {
        let opt = ps[i].contains("@opt");
        let (t, n) = shape(&ps[i]);
        let next = ps.get(i + 1).map(|p| shape(p));
        let next2 = ps.get(i + 2).map(|p| shape(p));
        let sz = |x: &Option<(String, String)>| x.as_ref().is_some_and(|x| x.0 == "size_t");
        let szp = |x: &Option<(String, String)>| x.as_ref().is_some_and(|x| x.0 == "size_t*");
        if t.ends_with(']') || n.contains('[') {
            // `char out65[65]`
            let nm = norm(&ps[i]);
            let (a, b) = (
                nm.find('[').ok_or_else(|| bad("array"))?,
                nm.find(']').ok_or_else(|| bad("array"))?,
            );
            if !nm.starts_with("char ") {
                return Err(bad("an array parameter that is not `char name[N]`"));
            }
            out.push(P::OutChars(
                nm[a + 1..b].parse().map_err(|_| bad("array size"))?,
            ));
            i += 1;
        } else if t == "ob_backend*" {
            out.push(P::HandleIn(n));
            i += 1
        } else if t == "int*" && n == "flags" {
            out.push(P::OutFlags(String::new()));
            i += 1
        } else if t == "int*" {
            out.push(P::Scalar('i', n));
            i += 1
        } else if t == "ob_parts*" {
            out.push(P::OutParts);
            i += 1
        } else if t == "ob_backend**" {
            out.push(P::OutHandle);
            i += 1
        } else if t == "rx_node" {
            out.push(P::NodeIn(n));
            i += 1
        } else if t == "rx_walk" {
            out.push(P::WalkIn(n));
            i += 1
        } else if t == "const char*" && sz(&next) {
            out.push(P::Text(n, next.unwrap().1));
            i += 2
        } else if t == "const char*" {
            out.push(P::Cstr(n, opt));
            i += 1
        } else if t == "const int64_t*" && opt {
            out.push(P::OptI64(n));
            i += 1
        } else if t == "const int64_t*" && sz(&next) {
            out.push(P::I64s(n, next.unwrap().1));
            i += 2
        } else if t == "const rx_node*" && opt {
            out.push(P::OptNode(n));
            i += 1
        } else if t == "const rx_node*" {
            i = node_run(&ps, i, &mut out, &bad)?;
        } else if t == "int64_t" {
            out.push(P::I64In(n));
            i += 1
        } else if t == "int" {
            out.push(P::IntIn(n));
            i += 1
        } else if t == "double*" {
            out.push(P::OutF64);
            i += 1
        } else if t == "int64_t*" && sz(&next) && szp(&next2) {
            out.push(P::OutI64s);
            i += 3
        } else if t == "int64_t*" {
            out.push(P::Scalar('l', n));
            i += 1
        } else if t == "rx_type*" {
            out.push(P::OutType);
            i += 1
        } else if t == "rw_compensated_target*"
            && next.as_ref().is_some_and(|x| x.0 == "char*")
            && ps.get(i + 2).is_some_and(|p| shape(p).0 == "size_t")
            && ps.get(i + 3).is_some_and(|p| shape(p).0 == "size_t*")
        {
            out.push(P::OutCompensated);
            i += 4
        } else if t == "rw_key_run*" {
            out.push(P::OutKeyRun);
            i += 1
        } else if t == "rw_gather*" {
            out.push(P::OutGather);
            i += 1
        } else if t == "rw_plane_row*" && sz(&next) && szp(&next2) {
            out.push(P::OutPlaneRows);
            i += 3
        } else if t == "rx_pair*" && sz(&next) && szp(&next2) {
            out.push(P::OutPairs);
            i += 3
        } else if t == "char*" && sz(&next) && szp(&next2) && szp(&ps.get(i + 3).map(|p| shape(p)))
        {
            out.push(P::OutNames);
            i += 4
        } else if t == "char*" && sz(&next) && szp(&next2) {
            out.push(P::OutText);
            i += 3
        } else if t == "rx_node*"
            && next.as_ref().is_some_and(|x| x.0 == "rx_node*")
            && ps.get(i + 2).is_some_and(|p| shape(p).0 == "size_t")
            && ps.get(i + 3).is_some_and(|p| shape(p).0 == "size_t*")
        {
            out.push(P::OutNodeList);
            i += 4
        } else if t == "rx_node*" && sz(&next) && szp(&next2) {
            out.push(P::OutNodes);
            i += 3
        } else if t == "rx_node*" {
            out.push(P::Scalar('n', n));
            i += 1
        } else if t == "rx_walk*" || t == "uint32_t*" {
            out.push(P::OutWalk);
            i += 1
        } else {
            return Err(bad(&format!(
                "parameter `{}` has a shape the generator does not know",
                ps[i]
            )));
        }
    }
    let out = group_outputs(out);
    if out
        .iter()
        .filter(|p| {
            matches!(
                p,
                P::OutText
                    | P::OutNodes
                    | P::OutNodeList
                    | P::OutNode
                    | P::OutWalk
                    | P::OutChars(_)
                    | P::OutHandle
                    | P::OutParts
                    | P::OutInt
                    | P::OutCode
                    | P::OutI64
                    | P::OutF64
                    | P::OutType
                    | P::OutI64s
                    | P::OutNames
                    | P::OutPairs
                    | P::OutScalars(..)
                    | P::OutFlags(_)
                    | P::OutKeyRun
                    | P::OutGather
                    | P::OutCompensated
                    | P::OutPlaneRows
            )
        })
        .count()
        > 1
    {
        return Err(bad("more than one output"));
    }
    Ok(out)
}

/// Settles what the scalar answers are: a lone one is the plain answer, a run of them one tuple, an int after the nodes a call
/// made a flag beside them; an `int* flags` is told which node list it has one flag per node of.
fn group_outputs(params: Vec<P>) -> Vec<P> {
    let mut out: Vec<P> = vec![];
    let mut last_list = String::new();
    for q in params {
        match q {
            P::NodeList(n, k) => {
                last_list = n.clone();
                out.push(P::NodeList(n, k))
            }
            P::OutFlags(_) => out.push(P::OutFlags(last_list.clone())),
            P::Scalar('i', _) if matches!(out.last(), Some(P::OutNodes)) => out.push(P::TrailFlag),
            P::Scalar(kind, name) => match out.pop() {
                Some(P::Scalar(k, n)) => out.push(P::OutScalars(vec![(k, n), (kind, name)])),
                Some(P::OutScalars(mut run)) => {
                    run.push((kind, name));
                    out.push(P::OutScalars(run))
                }
                Some(before) => {
                    out.push(before);
                    out.push(P::Scalar(kind, name))
                }
                None => out.push(P::Scalar(kind, name)),
            },
            other => out.push(other),
        }
    }
    out.into_iter()
        .map(|q| match q {
            P::Scalar('i', _) => P::OutInt,
            P::Scalar('n', _) => P::OutNode,
            P::Scalar(..) => P::OutI64,
            other => other,
        })
        .collect()
}

/// The shape of one parameter as (type, name), pointer stars glued to the type.
fn param_shape(p: &str) -> (String, String) {
    let n = p
        .replace(" *", "*")
        .replace("* ", "*")
        .replace("@opt", "")
        .trim()
        .to_string();
    let k = n.rfind(['*', ' ']).map_or(0, |i| i + 1);
    (n[..k].trim().to_string(), n[k..].to_string())
}

/// Reads a run of node lists starting at parameter `start`: consecutive `const rx_node*` parameters that may not be NULL. A list
/// followed by a `size_t` owns that length; a list with none shares the length of the next list that has one in the run, else of
/// the one before it (carries, inits and nexts of a loop; from and to of a substitution). Pushes the lists and the lengths in
/// the order of the C call and answers the index of the parameter after the run.
fn node_run(
    ps: &[String],
    start: usize,
    out: &mut Vec<P>,
    bad: &dyn Fn(&str) -> String,
) -> Result<usize, String> {
    // (name, index of the sized list it shares the length of)
    let mut lists: Vec<(String, usize)> = vec![];
    let mut sized: Vec<usize> = vec![];
    let mut at = start;
    while at < ps.len() {
        let (t, n) = param_shape(&ps[at]);
        if t != "const rx_node*" || ps[at].contains("@opt") {
            break;
        }
        let owns = ps.get(at + 1).is_some_and(|p| param_shape(p).0 == "size_t");
        if owns {
            sized.push(lists.len());
        }
        lists.push((n, if owns { lists.len() } else { usize::MAX }));
        at += 1 + usize::from(owns);
    }
    if sized.is_empty() {
        return Err(bad("a run of node lists with no length"));
    }
    for (i, list) in lists.iter_mut().enumerate() {
        if list.1 == usize::MAX {
            list.1 = *sized
                .iter()
                .find(|&&k| k > i)
                .unwrap_or_else(|| sized.iter().rev().find(|&&k| k < i).expect("a sized list"));
        }
    }
    let group_of = |list: usize| sized.iter().position(|&k| k == lists[list].1).unwrap_or(0);
    // the group's index is made unique over the whole call by the position of its first parameter
    let base = out.len();
    for (i, (name, _)) in lists.iter().enumerate() {
        out.push(P::NodeList(name.clone(), base * 8 + group_of(i)));
        if sized.contains(&i) {
            out.push(P::GroupLen(base * 8 + group_of(i)));
        }
    }
    Ok(at)
}

/// returns (prototypes, count of other statements understood: typedef, enum)
fn parse_header(src: &str) -> Result<(Vec<Proto>, usize), String> {
    let text = strip(src);
    let body: String = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
        .replace("extern \"C\" {", "");
    let (mut protos, mut other) = (vec![], 0);
    let mut stmts: Vec<String> = vec![];
    let (mut cur, mut depth) = (String::new(), 0);
    for c in body.chars() {
        match c {
            '{' => {
                depth += 1;
                cur.push(c)
            }
            '}' => {
                depth = (depth - 1).max(0);
                cur.push(c)
            }
            ';' if depth == 0 => stmts.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    for s in stmts {
        let s = s.replace("extern \"C\" {", "").replace('}', "");
        let s = s.trim();
        if s.is_empty() {
            continue;
        }
        if s.starts_with("typedef ") || s.starts_with("enum") {
            other += 1;
            continue;
        }
        let (a, b) = (
            s.find('(')
                .ok_or(format!("cannot wrap `{s}`: not a prototype"))?,
            s.rfind(')')
                .ok_or(format!("cannot wrap `{s}`: no closing parenthesis"))?,
        );
        let name = ident_last(&s[..a]);
        let ret = s[..a].trim_end_matches(&name[..]).trim().replace(" *", "*");
        if !(name.starts_with("rx_") || name.starts_with("rw_") || name.starts_with("ob_")) {
            return Err(format!(
                "cannot wrap `{s}`: the name does not start with rx_, rw_ or ob_"
            ));
        }
        let inner = s[a + 1..b].trim();
        if NOT_WRAPPED.contains(&name.as_str()) {
            other += 1;
            continue;
        }
        let mut params = if inner == "void" || LIFECYCLE.contains(&name.as_str()) {
            vec![]
        } else {
            parse_params(inner, s)?
        };
        if CODE_OUTPUTS.contains(&name.as_str()) {
            for q in params.iter_mut().filter(|q| matches!(q, P::OutInt)) {
                *q = P::OutCode;
            }
        }
        if !LIFECYCLE.contains(&name.as_str()) && ret != "int" {
            return Err(format!(
                "cannot wrap `{s}`: the return type is `{ret}`, not int"
            ));
        }
        protos.push(Proto {
            name,
            params,
            text: s.split_whitespace().collect::<Vec<_>>().join(" "),
        });
    }
    Ok((protos, other))
}

/// The short name of a function: the door's `rx_` dropped, the loader's `ob_` becoming `backend_`.
fn short(name: &str) -> String {
    match name.strip_prefix("ob_") {
        Some(rest) => format!("backend_{rest}"),
        None => name
            .trim_start_matches("rx_")
            .trim_start_matches("rw_")
            .to_string(),
    }
}

/// The Python name of a function.
fn pyname(rx: &str) -> String {
    let n = short(rx);
    let n = n.as_str();
    if KEYWORDS.contains(&n) {
        format!("{n}_")
    } else {
        n.to_string()
    }
}

/// The node lists of a call that share the length of group `g`, in call order.
fn groups(params: &[P], g: usize) -> Vec<String> {
    params
        .iter()
        .filter_map(|q| match q {
            P::NodeList(n, k) if *k == g => Some(n.clone()),
            _ => None,
        })
        .collect()
}

fn wrapper(p: &Proto) -> String {
    let mut decl = String::new();
    let (mut fmt, mut refs, mut conv, mut done) =
        (String::new(), vec![], String::new(), String::new());
    let mut args: Vec<String> = vec!["s->c".into()];
    let mut out: Option<&P> = None;
    let mut trail = false;
    for q in &p.params {
        match q {
            P::HandleIn(n) => {
                decl += &format!("  unsigned long long a_{n};\n");
                fmt += "K";
                refs.push(format!("&a_{n}"));
                args.push(format!("(ob_backend*)(uintptr_t)a_{n}"))
            }
            P::NodeIn(n) => {
                decl += &format!("  PyObject* o_{n}; rx_node a_{n};\n");
                fmt += "O";
                refs.push(format!("&o_{n}"));
                conv += &format!("  if (osprey_handle(s, o_{n}, &a_{n}) < 0) goto done;\n");
                args.push(format!("a_{n}"))
            }
            P::WalkIn(n) => {
                decl += &format!("  unsigned int a_{n};\n");
                fmt += "I";
                refs.push(format!("&a_{n}"));
                args.push(format!("(rx_walk)a_{n}"))
            }
            P::Cstr(n, opt) => {
                decl += &format!("  const char* a_{n};\n");
                fmt += if *opt { "z" } else { "s" };
                refs.push(format!("&a_{n}"));
                args.push(format!("a_{n}"))
            }
            P::Text(n, l) => {
                decl += &format!("  const char* a_{n}; Py_ssize_t l_{l};\n");
                fmt += "s#";
                refs.push(format!("&a_{n}"));
                refs.push(format!("&l_{l}"));
                args.push(format!("a_{n}"));
                args.push(format!("(size_t)l_{l}"))
            }
            P::I64s(n, l) => {
                decl +=
                    &format!("  PyObject* o_{n}; int64_t* a_{n} = NULL; Py_ssize_t l_{l} = 0;\n");
                fmt += "O";
                refs.push(format!("&o_{n}"));
                conv += &format!("  if (osprey_i64s(o_{n}, &a_{n}, &l_{l}) < 0) goto done;\n");
                done += &format!("  PyMem_Free(a_{n});\n");
                args.push(format!("a_{n}"));
                args.push(format!("(size_t)l_{l}"))
            }
            P::NodeList(n, _) => {
                decl +=
                    &format!("  PyObject* o_{n}; rx_node* a_{n} = NULL; Py_ssize_t l_{n} = 0;\n");
                fmt += "O";
                refs.push(format!("&o_{n}"));
                conv += &format!("  if (osprey_nodes(s, o_{n}, &a_{n}, &l_{n}) < 0) goto done;\n");
                done += &format!("  PyMem_Free(a_{n});\n");
                args.push(format!("a_{n}"))
            }
            P::GroupLen(g) => {
                let first = &groups(&p.params, *g)[0];
                args.push(format!("(size_t)l_{first}"))
            }
            P::OptNode(n) => {
                decl += &format!(
                    "  PyObject* o_{n}; rx_node a_{n} = 0; const rx_node* p_{n} = NULL;\n"
                );
                fmt += "O";
                refs.push(format!("&o_{n}"));
                conv += &format!("  if (o_{n} != Py_None) {{ if (osprey_handle(s, o_{n}, &a_{n}) < 0) goto done; p_{n} = &a_{n}; }}\n");
                args.push(format!("p_{n}"))
            }
            P::OptI64(n) => {
                decl += &format!(
                    "  PyObject* o_{n}; int64_t a_{n} = 0; const int64_t* p_{n} = NULL;\n"
                );
                fmt += "O";
                refs.push(format!("&o_{n}"));
                conv += &format!("  if (o_{n} != Py_None) {{ long long v_{n} = PyLong_AsLongLong(o_{n}); if (v_{n} == -1 && PyErr_Occurred()) goto done; a_{n} = (int64_t)v_{n}; p_{n} = &a_{n}; }}\n");
                args.push(format!("p_{n}"))
            }
            P::TrailFlag => trail = true,
            P::I64In(n) => {
                decl += &format!("  long long a_{n};\n");
                fmt += "L";
                refs.push(format!("&a_{n}"));
                args.push(format!("(int64_t)a_{n}"))
            }
            P::IntIn(n) => {
                decl += &format!("  int a_{n};\n");
                fmt += "i";
                refs.push(format!("&a_{n}"));
                args.push(format!("a_{n}"))
            }
            o => out = Some(o),
        }
    }
    for q in &p.params {
        if let P::GroupLen(g) = q {
            let members = groups(&p.params, *g);
            for m in &members[1..] {
                conv += &format!("  if (l_{m} != l_{}) {{ PyErr_SetString(PyExc_ValueError, \"the lists {m} and {} must have one length\"); goto done; }}\n", members[0], members[0]);
            }
        }
    }
    let call = |extra: &[&str]| {
        let mut a = args.clone();
        a.extend(extra.iter().map(|x| x.to_string()));
        format!("{}({})", p.name, a.join(", "))
    };
    let parse = if fmt.is_empty() {
        "  if (!PyArg_ParseTuple(args, \"\")) return NULL;\n".to_string()
    } else {
        format!(
            "  if (!PyArg_ParseTuple(args, \"{fmt}\", {})) return NULL;\n",
            refs.join(", ")
        )
    };
    let mut tail = String::new();
    match out {
        None => tail += &format!("  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  Py_INCREF(Py_None); res = Py_None;\n", call(&[])),
        Some(P::OutText) => tail += &format!("  char sb[256]; char* hb = NULL; size_t len = 0;\n  st = {};\n  if (st == RX_TRUNC) {{ hb = PyMem_Malloc(len + 1); if (!hb) {{ PyErr_NoMemory(); goto done; }} st = {}; }}\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  res = PyUnicode_DecodeUTF8(hb ? hb : sb, (Py_ssize_t)len, \"strict\");\n  PyMem_Free(hb);\n", call(&["sb", "sizeof sb", "&len"]), call(&["hb", "len + 1", "&len"])),
        Some(P::OutNodes) => {
            let (first, again, made) = if trail {
                (call(&["sn", "64", "&cnt", "&oc"]), call(&["hn", "cnt", "&cnt", "&oc"]), "Py_BuildValue(\"(NN)\", osprey_node_tuple(s, hn ? hn : sn, cnt), PyBool_FromLong(oc))")
            } else {
                (call(&["sn", "64", "&cnt"]), call(&["hn", "cnt", "&cnt"]), "osprey_node_tuple(s, hn ? hn : sn, cnt)")
            };
            let flag = if trail { " int oc = 0;" } else { "" };
            tail += &format!("  rx_node sn[64]; rx_node* hn = NULL; size_t cnt = 0;{flag}\n  st = {first};\n  if (st == RX_TRUNC) {{ hn = PyMem_Malloc((cnt ? cnt : 1) * sizeof(rx_node)); if (!hn) {{ PyErr_NoMemory(); goto done; }} st = {again}; }}\n  if (st != RX_OK) {{ osprey_fail(s, st); PyMem_Free(hn); goto done; }}\n  res = {made};\n  PyMem_Free(hn);\n")
        }
        Some(P::OutCompensated) => tail += &format!("  rw_compensated_target ot; memset(&ot, 0, sizeof ot); char sb[256]; char* hb = NULL; size_t len = 0;\n  st = {};\n  if (st == RX_TRUNC) {{ hb = PyMem_Malloc(len + 1); if (!hb) {{ PyErr_NoMemory(); goto done; }} st = {}; }}\n  if (st != RX_OK) {{ osprey_fail(s, st); PyMem_Free(hb); goto done; }}\n  if (!ot.found) {{ Py_INCREF(Py_None); res = Py_None; }}\n  else res = Py_BuildValue(\"(NN)\", PyUnicode_DecodeUTF8(hb ? hb : sb, (Py_ssize_t)len, \"strict\"), PyBool_FromLong(ot.own_column));\n  PyMem_Free(hb);\n", call(&["&ot", "sb", "sizeof sb", "&len"]), call(&["&ot", "hb", "len + 1", "&len"])),
        Some(P::OutKeyRun) => tail += &format!("  rw_key_run ok; memset(&ok, 0, sizeof ok);\n  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  res = osprey_key_run_tuple(&ok);\n", call(&["&ok"])),
        Some(P::OutGather) => tail += &format!("  rw_gather og; memset(&og, 0, sizeof og);\n  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  res = osprey_gather_tuple(s, &og);\n", call(&["&og"])),
        Some(P::OutPlaneRows) => tail += &format!("  rw_plane_row sr[16]; rw_plane_row* hr = NULL; size_t cnt = 0;\n  st = {};\n  if (st == RX_TRUNC) {{ hr = PyMem_Malloc((cnt ? cnt : 1) * sizeof(rw_plane_row)); if (!hr) {{ PyErr_NoMemory(); goto done; }} st = {}; }}\n  if (st != RX_OK) {{ osprey_fail(s, st); PyMem_Free(hr); goto done; }}\n  res = osprey_rows_tuple(s, hr ? hr : sr, cnt);\n  PyMem_Free(hr);\n", call(&["sr", "16", "&cnt"]), call(&["hr", "cnt", "&cnt"])),
        Some(P::OutFlags(list)) => tail += &format!("  int* fl = PyMem_Malloc((l_{list} ? (size_t)l_{list} : 1) * sizeof(int));\n  if (!fl) {{ PyErr_NoMemory(); goto done; }}\n  memset(fl, 0, (l_{list} ? (size_t)l_{list} : 1) * sizeof(int));\n  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); PyMem_Free(fl); goto done; }}\n  res = osprey_flags_tuple(fl, (size_t)l_{list});\n  PyMem_Free(fl);\n", call(&["fl"])),
        Some(P::OutScalars(run)) => {
            let found = run[0].1 == "found";
            let (mut decls, mut extra, mut items) = (String::new(), vec![], String::new());
            for (k, (kind, name)) in run.iter().enumerate() {
                let ctype = match kind { 'i' => "int", 'n' => "rx_node", _ => "int64_t" };
                decls += &format!(" {ctype} r{k} = 0;");
                extra.push(format!("&r{k}"));
                if found && k == 0 {
                    continue;
                }
                let item = match kind {
                    'i' => format!("PyBool_FromLong(r{k})"),
                    'n' => format!("(r{k} == UINT32_MAX ? (Py_INCREF(Py_None), Py_None) : osprey_wrap_node(s, r{k}))"),
                    _ => format!("PyLong_FromLongLong((long long)r{k})"),
                };
                items += &format!("  PyTuple_SetItem(res, {}, {item}); /* {name} */\n", k - usize::from(found));
            }
            let refs: Vec<&str> = extra.iter().map(String::as_str).collect();
            let count = run.len() - usize::from(found);
            let none = if found { "  if (!r0) { Py_INCREF(Py_None); res = Py_None; goto done; }\n" } else { "" };
            tail += &format!("{decls}\n  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n{none}  res = PyTuple_New({count});\n  if (!res) goto done;\n{items}", call(&refs));
        }
        Some(P::OutNodeList) => tail += &format!("  rx_node ob0 = 0; rx_node sn[64]; rx_node* hn = NULL; size_t cnt = 0;\n  st = {};\n  if (st == RX_TRUNC) {{ hn = PyMem_Malloc((cnt ? cnt : 1) * sizeof(rx_node)); if (!hn) {{ PyErr_NoMemory(); goto done; }} st = {}; }}\n  if (st != RX_OK) {{ osprey_fail(s, st); PyMem_Free(hn); goto done; }}\n  res = Py_BuildValue(\"(NN)\", osprey_wrap_node(s, ob0), osprey_node_tuple(s, hn ? hn : sn, cnt));\n  PyMem_Free(hn);\n", call(&["&ob0", "sn", "64", "&cnt"]), call(&["&ob0", "hn", "cnt", "&cnt"])),
        Some(P::OutNode) => tail += &format!("  rx_node oh = 0;\n  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  res = osprey_wrap_node(s, oh);\n", call(&["&oh"])),
        Some(P::OutInt) => tail += &format!("  int oi = 0;\n  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  res = PyBool_FromLong(oi);\n", call(&["&oi"])),
        Some(P::OutParts) => tail += &format!("  ob_parts parts = {{0}};\n  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  res = osprey_parts_tuple(&parts);\n", call(&["&parts"])),
        Some(P::OutHandle) => tail += &format!("  ob_backend* ob = NULL;\n  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  res = PyLong_FromVoidPtr(ob);\n", call(&["&ob"])),
        Some(P::OutWalk) => tail += &format!("  rx_walk ow = 0;\n  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  res = PyLong_FromUnsignedLong(ow);\n", call(&["&ow"])),
        Some(P::OutCode) => tail += &format!("  int oc = 0;\n  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  res = PyLong_FromLong(oc);\n", call(&["&oc"])),
        Some(P::OutI64) => tail += &format!("  int64_t oi = 0;\n  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  res = PyLong_FromLongLong(oi);\n", call(&["&oi"])),
        Some(P::OutF64) => tail += &format!("  double od = 0;\n  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  res = PyFloat_FromDouble(od);\n", call(&["&od"])),
        Some(P::OutType) => tail += &format!("  rx_type ot; memset(&ot, 0, sizeof ot);\n  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  res = osprey_type_tuple(&ot);\n", call(&["&ot"])),
        Some(P::OutI64s) => tail += &format!("  int64_t si[64]; int64_t* hi = NULL; size_t cnt = 0;\n  st = {};\n  if (st == RX_TRUNC) {{ hi = PyMem_Malloc((cnt ? cnt : 1) * sizeof(int64_t)); if (!hi) {{ PyErr_NoMemory(); goto done; }} st = {}; }}\n  if (st != RX_OK) {{ osprey_fail(s, st); PyMem_Free(hi); goto done; }}\n  res = osprey_i64_tuple(hi ? hi : si, cnt);\n  PyMem_Free(hi);\n", call(&["si", "64", "&cnt"]), call(&["hi", "cnt", "&cnt"])),
        Some(P::OutNames) => tail += &format!("  char sb[256]; char* hb = NULL; size_t len = 0, cnt = 0;\n  st = {};\n  if (st == RX_TRUNC) {{ hb = PyMem_Malloc(len + 1); if (!hb) {{ PyErr_NoMemory(); goto done; }} st = {}; }}\n  if (st != RX_OK) {{ osprey_fail(s, st); PyMem_Free(hb); goto done; }}\n  res = osprey_names_tuple(hb ? hb : sb, len);\n  PyMem_Free(hb);\n", call(&["sb", "sizeof sb", "&len", "&cnt"]), call(&["hb", "len + 1", "&len", "&cnt"])),
        Some(P::OutPairs) => {
            let rows = if GROUPED_PAIRS.contains(&p.name.as_str()) { "osprey_grouped_pairs_tuple" } else { "osprey_pairs_tuple" };
            tail += &format!("  rx_pair sp[16]; rx_pair* hp = NULL; size_t cnt = 0;\n  st = {};\n  if (st == RX_TRUNC) {{ hp = PyMem_Malloc((cnt ? cnt : 1) * sizeof(rx_pair)); if (!hp) {{ PyErr_NoMemory(); goto done; }} st = {}; }}\n  if (st != RX_OK) {{ osprey_fail(s, st); PyMem_Free(hp); goto done; }}\n  res = {rows}(hp ? hp : sp, cnt);\n  PyMem_Free(hp);\n", call(&["sp", "16", "&cnt"]), call(&["hp", "cnt", "&cnt"]), rows = rows);
        },
        Some(P::OutChars(n)) => tail += &format!("  char dg[{n}] = {{0}};\n  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  res = PyUnicode_FromString(dg);\n", call(&["dg"])),
        _ => unreachable!(),
    }
    if p.name.starts_with(ABSENT_READERS) && !CODE_OUTPUTS.contains(&p.name.as_str()) {
        // a field that holds nothing answers None
        if let Some(at) = tail.find("\n  if (st") {
            tail.insert_str(
                at,
                "\n  if (st == RX_ABSENT) { Py_INCREF(Py_None); res = Py_None; goto done; }",
            );
        }
    }
    let text = format!("/* wrapper: {n} */\n/* {t} */\nstatic PyObject* call_{n}(Ctx* s, PyObject* args) {{\n  PyObject* res = NULL; int st;\n{decl}{parse}{conv}{tail}done:\n{done}  return res;\n}}\nstatic PyObject* m_{n}(PyObject* self, PyObject* args) {{ return call_{n}((Ctx*)self, args); }}\n\n", n = short(&p.name), t = p.text);
    if CODED.contains(&p.name.as_str()) {
        text.replace(
            "osprey_fail(s, st)",
            "osprey_fail_lib(s, st, (ob_backend*)(uintptr_t)a_lib)",
        )
    } else {
        text
    }
}

const PRELUDE: &str = r#"/* Generated by gen from the headers of the door. Do not edit. */
#include "module.h"
#include "raptor_ir_front.h"
#include "raptor_ir_query.h"
#include "raptor_wings_front.h"
#include <stdio.h>
#include <string.h>

/* The exception classes of the three kinds of refusal: a construct not written and a bad request are value errors, a fault is a
   runtime error. Made at module init by osprey_add_errors; osprey_fail chooses by the door's status. NotWritten has an attribute
   `code` (a str, or None): a code is a promise to callers, who may match on it; the text is not. */
static PyObject* osprey_errors[3];

int osprey_add_errors(PyObject* module) {
  static const char* const names[3] = {"NotWritten", "BadRequest", "Fault"};
  PyObject* const bases[3] = {PyExc_ValueError, PyExc_ValueError, PyExc_RuntimeError};
  for (int i = 0; i < 3; i++) {
    char full[40];
    snprintf(full, sizeof full, "_osprey.%s", names[i]);
    /* NotWritten has a class attribute `code` (None), set on an instance when the refusal has a code. */
    PyObject* members = i == 0 ? Py_BuildValue("{sO}", "code", Py_None) : NULL;
    if (i == 0 && !members) return -1;
    osprey_errors[i] = PyErr_NewException(full, bases[i], members);
    Py_XDECREF(members);
    if (!osprey_errors[i]) return -1;
    Py_INCREF(osprey_errors[i]);
    if (PyModule_AddObject(module, names[i], osprey_errors[i]) < 0) return -1;
  }
  return 0;
}

/* Raises NotWritten carrying the code of the refusal: a code is a promise to callers, the text is not. */
static void osprey_raise_coded(const char* why, const char* code) {
  PyObject* exc = PyObject_CallFunction(osprey_errors[0], "s", why);
  PyObject* text = exc ? PyUnicode_FromString(code) : NULL;
  if (text && PyObject_SetAttrString(exc, "code", text) == 0) PyErr_SetObject(osprey_errors[0], exc);
  Py_XDECREF(text);
  Py_XDECREF(exc);
}

static inline void osprey_fail(Ctx* s, int st);
/* A refusal of the tool side: NotWritten carrying the library's code, a code being a promise to callers. */
static inline void osprey_fail_lib(Ctx* s, int st, ob_backend* lib) {
  const char* code = ob_last_code(lib);
  if (st == RX_REFUSED && *code) osprey_raise_coded(rx_last_error(s->c), code);
  else osprey_fail(s, st);
}

static inline void osprey_fail(Ctx* s, int st) {
  const char* why = rx_last_error(s->c);
  const char* code = rx_last_code(s->c);
  if (st == RX_REFUSED && *code) osprey_raise_coded(why, code);
  else if (st == RX_REFUSED) PyErr_SetString(osprey_errors[0], why);
  else if (st == RX_BADARG) PyErr_SetString(osprey_errors[1], why);
  else if (st == RX_PANIC) PyErr_SetString(osprey_errors[2], why);
  else PyErr_Format(PyExc_RuntimeError, "door status %d: %s", st, why);
}

static inline PyObject* osprey_wrap_node(Ctx* s, rx_node h) {
  Node* n = (Node*)PyType_GenericAlloc(osprey_node_type, 0);
  if (!n) return NULL;
  Py_INCREF((PyObject*)s);
  n->ctx = s;
  n->h = h;
  return (PyObject*)n;
}

static inline PyObject* osprey_parts_tuple(const ob_parts* parts) {
  const char* texts[4] = {parts->text, parts->body, parts->entry, parts->meta};
  const size_t lengths[4] = {parts->text_len, parts->body_len, parts->entry_len, parts->meta_len};
  PyObject* t = PyTuple_New(4);
  if (!t) return NULL;
  for (Py_ssize_t i = 0; i < 4; i++) {
    PyObject* part = PyUnicode_DecodeUTF8(texts[i], (Py_ssize_t)lengths[i], "strict");
    if (!part) { Py_DECREF(t); return NULL; }
    PyTuple_SetItem(t, i, part);
  }
  return t;
}

static inline PyObject* osprey_node_tuple(Ctx* s, const rx_node* v, size_t n) {
  PyObject* t = PyTuple_New((Py_ssize_t)n);
  if (!t) return NULL;
  for (size_t i = 0; i < n; i++) {
    PyObject* w = osprey_wrap_node(s, v[i]);
    if (!w) { Py_DECREF(t); return NULL; }
    PyTuple_SetItem(t, (Py_ssize_t)i, w);
  }
  return t;
}

static inline int osprey_handle(Ctx* s, PyObject* o, rx_node* h) {
  if (!PyObject_TypeCheck(o, osprey_node_type)) { PyErr_SetString(PyExc_TypeError, "expected a Node"); return -1; }
  if (((Node*)o)->ctx != s) {
    osprey_raise_coded("a Node of another Ctx: a node built in one document cannot be an operand in another", "foreign_node");
    return -1;
  }
  *h = ((Node*)o)->h;
  return 0;
}

static inline int osprey_nodes(Ctx* s, PyObject* seq, rx_node** v, Py_ssize_t* n) {
  PyObject* t = PySequence_Tuple(seq);
  if (!t) return -1;
  *n = PyTuple_Size(t);
  *v = PyMem_Malloc((size_t)(*n ? *n : 1) * sizeof(rx_node));
  if (!*v) { Py_DECREF(t); PyErr_NoMemory(); return -1; }
  for (Py_ssize_t i = 0; i < *n; i++) {
    if (osprey_handle(s, PyTuple_GetItem(t, i), &(*v)[i]) < 0) { Py_DECREF(t); PyMem_Free(*v); *v = NULL; return -1; }
  }
  Py_DECREF(t);
  return 0;
}

static inline PyObject* osprey_i64_tuple(const int64_t* v, size_t n) {
  PyObject* t = PyTuple_New((Py_ssize_t)n);
  if (!t) return NULL;
  for (size_t i = 0; i < n; i++) {
    PyObject* x = PyLong_FromLongLong((long long)v[i]);
    if (!x) { Py_DECREF(t); return NULL; }
    PyTuple_SetItem(t, (Py_ssize_t)i, x);
  }
  return t;
}

/* The texts a reader packs, each NUL-terminated and back to back in `len` bytes, as a tuple of str. */
static inline PyObject* osprey_names_tuple(const char* packed, size_t len) {
  Py_ssize_t count = 0;
  for (size_t i = 0; i < len; i++) if (packed[i] == 0) count++;
  PyObject* t = PyTuple_New(count);
  if (!t) return NULL;
  const char* at = packed;
  for (Py_ssize_t k = 0; k < count; k++) {
    size_t n = strlen(at);
    PyObject* x = PyUnicode_DecodeUTF8(at, (Py_ssize_t)n, "strict");
    if (!x) { Py_DECREF(t); return NULL; }
    PyTuple_SetItem(t, k, x);
    at += n + 1;
  }
  return t;
}

/* The dtype names, in the order of the RX_ codes of raptor_ir_front.h. */
static const char* const osprey_dtype_names[] = {"f64", "f32", "i32", "i64", "bool", "int32", "uint32"};

/* One part of a pair as None, an int or a str. */
static inline PyObject* osprey_pair_part(const rx_pair_part* part) {
  if (part->kind == 1) return PyLong_FromLongLong((long long)part->number);
  if (part->kind == 2) return PyUnicode_FromString(part->text);
  Py_INCREF(Py_None);
  return Py_None;
}

/* A tuple of pairs as a tuple of 2-tuples, an absent part None. */
static inline PyObject* osprey_pairs_tuple(const rx_pair* pairs, size_t n) {
  PyObject* out = PyTuple_New((Py_ssize_t)n);
  if (!out) return NULL;
  for (size_t i = 0; i < n; i++) {
    PyObject* item = Py_BuildValue("(NN)", osprey_pair_part(&pairs[i].first), osprey_pair_part(&pairs[i].second));
    if (!item) { Py_DECREF(out); return NULL; }
    PyTuple_SetItem(out, (Py_ssize_t)i, item);
  }
  return out;
}

/* One flag per node as a tuple of bool. */
static inline PyObject* osprey_flags_tuple(const int* flags, size_t n) {
  PyObject* t = PyTuple_New((Py_ssize_t)n);
  if (!t) return NULL;
  for (size_t i = 0; i < n; i++) PyTuple_SetItem(t, (Py_ssize_t)i, PyBool_FromLong(flags[i]));
  return t;
}

/* A key run as None or (form name, width, whether it is a block), the forms in the order of the RW_KEY_ codes of raptor_wings_front.h. */
static inline PyObject* osprey_key_run_tuple(const rw_key_run* run) {
  static const char* const forms[] = {"proven", "sorted", "tested"};
  if (!run->found) Py_RETURN_NONE;
  if (run->form < RW_KEY_PROVEN || run->form > RW_KEY_TESTED) {
    PyErr_SetString(PyExc_ValueError, "a key run form this binding does not know");
    return NULL;
  }
  return Py_BuildValue("(sLN)", forms[run->form - RW_KEY_PROVEN], (long long)run->width, PyBool_FromLong(run->block));
}

/* A gather as None or (plane node, base node or None, start, step, whether it is clamped, the lead index nodes as a tuple). */
static inline PyObject* osprey_gather_tuple(Ctx* s, const rw_gather* run) {
  if (!run->found) Py_RETURN_NONE;
  PyObject* lead = PyTuple_New((Py_ssize_t)run->nlead);
  if (!lead) return NULL;
  for (size_t i = 0; i < run->nlead; i++) PyTuple_SetItem(lead, (Py_ssize_t)i, osprey_wrap_node(s, run->lead[i]));
  PyObject* base = run->base == UINT32_MAX ? (Py_INCREF(Py_None), Py_None) : osprey_wrap_node(s, run->base);
  return Py_BuildValue("(NNLLNN)", osprey_wrap_node(s, run->plane), base, (long long)run->start, (long long)run->step, PyBool_FromLong(run->clamped), lead);
}

/* The placed planes as a tuple of (plane node, layout name, width), the layouts as the RW_LAYOUT_ codes of raptor_wings_front.h. */
static inline PyObject* osprey_rows_tuple(Ctx* s, const rw_plane_row* rows, size_t n) {
  static const char* const layouts[] = {"plane_major", "warp_tiled"};
  PyObject* out = PyTuple_New((Py_ssize_t)n);
  if (!out) return NULL;
  for (size_t i = 0; i < n; i++) {
    if (rows[i].layout < RW_LAYOUT_PLANE_MAJOR || rows[i].layout > RW_LAYOUT_WARP_TILED) {
      Py_DECREF(out);
      PyErr_SetString(PyExc_ValueError, "a plane layout this binding does not know");
      return NULL;
    }
    PyObject* item = Py_BuildValue("(NsL)", osprey_wrap_node(s, rows[i].plane), layouts[rows[i].layout], (long long)rows[i].width);
    if (!item) { Py_DECREF(out); return NULL; }
    PyTuple_SetItem(out, (Py_ssize_t)i, item);
  }
  return out;
}

/* The pairs of one first part together, as a tuple of (first part, tuple of the second parts); the pairs of a part are consecutive. */
static inline PyObject* osprey_grouped_pairs_tuple(const rx_pair* pairs, size_t n) {
  PyObject* out = PyList_New(0);
  if (!out) return NULL;
  PyObject* seconds = NULL;
  for (size_t i = 0; i < n; i++) {
    if (i == 0 || strcmp(pairs[i].first.text, pairs[i - 1].first.text) != 0) {
      seconds = PyList_New(0);
      PyObject* row = seconds ? Py_BuildValue("(NO)", osprey_pair_part(&pairs[i].first), seconds) : NULL;
      if (!row || PyList_Append(out, row) < 0) { Py_XDECREF(row); Py_XDECREF(seconds); Py_DECREF(out); return NULL; }
      Py_DECREF(row);
      Py_DECREF(seconds);
    }
    PyObject* part = osprey_pair_part(&pairs[i].second);
    if (!part || PyList_Append(seconds, part) < 0) { Py_XDECREF(part); Py_DECREF(out); return NULL; }
    Py_DECREF(part);
  }
  PyObject* rows = PyList_AsTuple(out);
  Py_DECREF(out);
  if (!rows) return NULL;
  for (Py_ssize_t k = 0; k < PyTuple_Size(rows); k++) {
    PyObject* row = PyTuple_GetItem(rows, k);
    PyObject* frozen = PyList_AsTuple(PyTuple_GetItem(row, 1));
    if (!frozen) { Py_DECREF(rows); return NULL; }
    PyTuple_SetItem(row, 1, frozen);
  }
  return rows;
}

/* A node's type as (shape tuple, dtype name, tag or None). */
static inline PyObject* osprey_type_tuple(const rx_type* ty) {
  if (ty->dtype < 0 || ty->dtype >= (int)(sizeof osprey_dtype_names / sizeof *osprey_dtype_names)) {
    PyErr_SetString(PyExc_ValueError, "a dtype code this binding does not know");
    return NULL;
  }
  PyObject* shape = PyTuple_New((Py_ssize_t)ty->rank);
  if (!shape) return NULL;
  for (uint32_t i = 0; i < ty->rank; i++) {
    PyObject* x = PyLong_FromLongLong((long long)ty->dims[i]);
    if (!x) { Py_DECREF(shape); return NULL; }
    PyTuple_SetItem(shape, (Py_ssize_t)i, x);
  }
  PyObject* tag;
  if (ty->tag[0]) tag = PyUnicode_FromString(ty->tag);
  else { Py_INCREF(Py_None); tag = Py_None; }
  if (!tag) { Py_DECREF(shape); return NULL; }
  return Py_BuildValue("(NsN)", shape, osprey_dtype_names[ty->dtype], tag);
}

static inline int osprey_i64s(PyObject* seq, int64_t** v, Py_ssize_t* n) {
  PyObject* t = PySequence_Tuple(seq);
  if (!t) return -1;
  *n = PyTuple_Size(t);
  *v = PyMem_Malloc((size_t)(*n ? *n : 1) * sizeof(int64_t));
  if (!*v) { Py_DECREF(t); PyErr_NoMemory(); return -1; }
  for (Py_ssize_t i = 0; i < *n; i++) {
    long long x = PyLong_AsLongLong(PyTuple_GetItem(t, i));
    if (x == -1 && PyErr_Occurred()) { Py_DECREF(t); PyMem_Free(*v); *v = NULL; return -1; }
    (*v)[i] = (int64_t)x;
  }
  Py_DECREF(t);
  return 0;
}

"#;

fn generate(header: &str) -> Result<(String, usize, usize), String> {
    let (protos, other) = parse_header(header)?;
    let wrapped: Vec<&Proto> = protos
        .iter()
        .filter(|p| !LIFECYCLE.contains(&p.name.as_str()))
        .collect();
    for l in LIFECYCLE {
        if !protos.iter().any(|p| p.name == l) {
            return Err(format!(
                "the lifecycle prototype {l} is missing from the header"
            ));
        }
    }
    let mut c = String::from(PRELUDE);
    for p in &wrapped {
        c += &wrapper(p)
    }
    c += "PyMethodDef osprey_ctx_methods[] = {\n";
    for p in &wrapped {
        c += &format!(
            "  {{\"{}\", m_{}, METH_VARARGS, NULL}},\n",
            pyname(&p.name),
            short(&p.name)
        )
    }
    c += "  {NULL, NULL, 0, NULL}\n};\n\n";
    for (py, rx, getter) in NODE_SURFACE {
        if !wrapped.iter().any(|p| p.name == rx) {
            return Err(format!(
                "the Node surface names {rx}, which the header does not have"
            ));
        }
        let n = short(rx);
        if getter {
            c += &format!("static PyObject* g_{py}(PyObject* self, void* closure) {{\n  (void)closure;\n  PyObject* a = PyTuple_Pack(1, self);\n  if (!a) return NULL;\n  PyObject* r = call_{n}(((Node*)self)->ctx, a);\n  Py_DECREF(a);\n  return r;\n}}\n\n");
        } else {
            c += &format!("static PyObject* n_{py}(PyObject* self, PyObject* args) {{\n  PyObject* h = PyTuple_Pack(1, self);\n  if (!h) return NULL;\n  PyObject* a = PySequence_Concat(h, args);\n  Py_DECREF(h);\n  if (!a) return NULL;\n  PyObject* r = call_{n}(((Node*)self)->ctx, a);\n  Py_DECREF(a);\n  return r;\n}}\n\n");
        }
    }
    c += "PyGetSetDef osprey_node_getset[] = {\n";
    for (py, _, g) in NODE_SURFACE {
        if g {
            c += &format!("  {{\"{py}\", g_{py}, NULL, NULL, NULL}},\n")
        }
    }
    c += "  {NULL, NULL, NULL, NULL, NULL}\n};\n\nPyMethodDef osprey_node_methods[] = {\n";
    for (py, _, g) in NODE_SURFACE {
        if !g {
            c += &format!("  {{\"{py}\", n_{py}, METH_VARARGS, NULL}},\n")
        }
    }
    c += "  {NULL, NULL, 0, NULL}\n};\n";
    Ok((c, wrapped.len(), protos.len() + other))
}

/// C source without comments, string and char literals, and preprocessor lines
fn clean_c(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let (mut o, mut i) = (String::new(), 0);
    while i < b.len() {
        if b[i] == '/' && b.get(i + 1) == Some(&'*') {
            while i + 1 < b.len() && !(b[i] == '*' && b[i + 1] == '/') {
                i += 1
            }
            i += 2;
            o.push(' ');
            continue;
        }
        if b[i] == '/' && b.get(i + 1) == Some(&'/') {
            while i < b.len() && b[i] != '\n' {
                i += 1
            }
            continue;
        }
        if b[i] == '"' || b[i] == '\'' {
            let q = b[i];
            i += 1;
            while i < b.len() && b[i] != q {
                if b[i] == '\\' {
                    i += 1
                }
                i += 1
            }
            i += 1;
            o.push('0');
            continue;
        }
        o.push(b[i]);
        i += 1;
    }
    o.lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
}

/// names of the functions a C file defines (top level, a `(` before the `{`, no `=`)
fn defined_functions(src: &str) -> Vec<String> {
    let (mut out, mut cur, mut depth) = (vec![], String::new(), 0);
    for c in clean_c(src).chars() {
        match c {
            '{' => {
                if depth == 0 && cur.contains('(') && !cur.contains('=') {
                    out.push(ident_last(cur.split('(').next().unwrap_or("")))
                }
                if depth == 0 {
                    cur.clear()
                }
                depth += 1
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    cur.clear()
                }
            }
            ';' if depth == 0 => cur.clear(),
            _ if depth == 0 => cur.push(c),
            _ => {}
        }
    }
    out
}

/// identifiers `rx_...` that are called (followed by `(`)
fn rx_calls(src: &str) -> BTreeSet<String> {
    let s = clean_c(src);
    let b: Vec<char> = s.chars().collect();
    let (mut out, mut i) = (BTreeSet::new(), 0);
    while i < b.len() {
        if b[i].is_alphabetic() || b[i] == '_' {
            let j = (i..b.len())
                .find(|&k| !(b[k].is_alphanumeric() || b[k] == '_'))
                .unwrap_or(b.len());
            let w: String = b[i..j].iter().collect();
            let k = (j..b.len())
                .find(|&k| !b[k].is_whitespace())
                .unwrap_or(b.len());
            if w.starts_with("rx_") && b.get(k) == Some(&'(') {
                out.insert(w);
            }
            i = j;
        } else {
            i += 1
        }
    }
    out
}

/// The functions the runtime's door header declares: `eg_status eg_<name>(` outside comments.
fn door_functions(header: &str) -> Vec<String> {
    strip(header)
        .split("eg_status eg_")
        .skip(1)
        .filter_map(|rest| rest.split('(').next())
        .map(|name| format!("eg_{name}"))
        .collect()
}

/// Check 6: the door source lists every function of the door header once, calls each, and lists nothing else.
fn door_wrapped(header: &str, source: &str) -> (bool, String) {
    let names = door_functions(header);
    let mut bad: Vec<String> = vec![];
    for n in &names {
        let (listed, called) = (
            source.matches(&format!("X({n})")).count(),
            source.matches(&format!("fn.{n}(")).count(),
        );
        if listed != 1 || called == 0 {
            bad.push(format!(
                "{n} is listed {listed} times and called {called} times"
            ))
        }
    }
    let extra = source.matches("    X(eg_").count() as isize - names.len() as isize;
    if extra != 0 {
        bad.push(format!("{extra} listed function(s) beyond the header"))
    }
    let what = if bad.is_empty() {
        "no violation".to_string()
    } else {
        bad.join("; ")
    };
    (
        bad.is_empty() && !names.is_empty(),
        format!("{} functions in the door header; {what}", names.len()),
    )
}

fn check(hsrc: &str, tdir: &str, genfile: &str, ext: &str, door: Option<(String, String)>) -> bool {
    let mut ok = true;
    let mut report = |name: &str, pass: bool, what: String| {
        println!("[{}] {name}: {what}", if pass { "ok" } else { "RED" });
        if !pass {
            ok = false
        }
    };
    let (fresh, nwrap, nstmts) = match generate(hsrc) {
        Ok(x) => x,
        Err(e) => {
            println!("[RED] header: {e}");
            return false;
        }
    };
    // (1)
    let have = fs::read_to_string(genfile).unwrap_or_default();
    report(
        "1 regeneration is byte-identical",
        !have.is_empty() && have == fresh,
        format!(
            "{} bytes generated, {} bytes on disk, {} statements of the header examined",
            fresh.len(),
            have.len(),
            nstmts
        ),
    );
    // (2)
    let (mut nfn, mut bad) = (0usize, vec![]);
    let mut calls = 0usize;
    for f in ["types.c", "module.h"] {
        let src = fs::read_to_string(format!("{tdir}/{f}")).unwrap_or_default();
        if src.is_empty() {
            bad.push(format!("{f} is empty or missing"))
        }
        for n in defined_functions(&src) {
            nfn += 1;
            if !TEMPLATE_ALLOW.contains(&n.as_str()) {
                bad.push(format!("{f}: function {n} is not in the allow-list"))
            }
        }
        for n in rx_calls(&src) {
            calls += 1;
            if !TEMPLATE_RX.contains(&n.as_str()) {
                bad.push(format!("{f}: calls {n}"))
            }
        }
    }
    report(
        "2 template defines only the allow-list and calls only the lifecycle",
        bad.is_empty() && nfn > 0,
        format!(
            "{nfn} functions defined (allow-list {}), {calls} rx_ names called (allowed {}); {}",
            TEMPLATE_ALLOW.join(" "),
            TEMPLATE_RX.join(" "),
            if bad.is_empty() {
                "no violation".into()
            } else {
                bad.join("; ")
            }
        ),
    );
    // (3)
    let (protos, _) = parse_header(hsrc).unwrap_or_default();
    let mut miss = vec![];
    let wrapped: Vec<&Proto> = protos
        .iter()
        .filter(|p| !LIFECYCLE.contains(&p.name.as_str()))
        .collect();
    for p in &wrapped {
        let n = short(&p.name);
        let (m, d) = (
            have.matches(&format!("/* wrapper: {n} */")).count(),
            have.matches(&format!("static PyObject* call_{n}(")).count(),
        );
        let tab = have
            .matches(&format!("{{\"{}\", m_{n},", pyname(&p.name)))
            .count();
        if (m, d, tab) != (1, 1, 1) {
            miss.push(format!(
                "{} has marker {m}, definition {d}, table entry {tab}",
                p.name
            ))
        }
    }
    let extra = have.matches("/* wrapper: ").count() as isize - wrapped.len() as isize;
    if extra != 0 {
        miss.push(format!("{extra} wrapper(s) beyond the header"))
    }
    report(
        "3 every non-lifecycle prototype has exactly one wrapper",
        miss.is_empty() && !wrapped.is_empty(),
        format!(
            "{} prototypes in the header, {} wrapped, {} lifecycle; {}",
            protos.len(),
            wrapped.len(),
            protos.len() - wrapped.len(),
            if miss.is_empty() {
                "no violation".into()
            } else {
                miss.join("; ")
            }
        ),
    );
    let _ = nwrap;
    // (4)
    // GNU nm says --defined-only where the BSD nm of macOS says -U (the same choice as check_exports.sh)
    let gnu = Command::new("nm")
        .arg("--version")
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains("GNU"));
    let nm = Command::new("nm")
        .args(["-g", if gnu { "--defined-only" } else { "-U" }, ext])
        .output();
    match nm {
        Ok(o) if o.status.success() => {
            let syms: Vec<String> = String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.split_whitespace().last().unwrap_or("").to_string())
                .collect();
            report(
                "4 the extension exports exactly one symbol",
                syms == [if cfg!(target_os = "macos") {
                    "_PyInit__osprey"
                } else {
                    "PyInit__osprey"
                }],
                format!("{} symbol(s): {}", syms.len(), syms.join(" ")),
            );
        }
        _ => report(
            "4 the extension exports exactly one symbol",
            false,
            format!("cannot run nm on {ext}"),
        ),
    }
    report(
        "5 prototypes the binding does not wrap, by rule",
        true,
        format!(
            "{} ({} of {} statements)",
            NOT_WRAPPED.join(" "),
            NOT_WRAPPED
                .iter()
                .filter(|n| hsrc.contains(&format!("{n}(")))
                .count(),
            NOT_WRAPPED.len()
        ),
    );
    if let Some((header, source)) = door {
        let read = |p: &str| fs::read_to_string(p).unwrap_or_default();
        let (pass, what) = door_wrapped(&read(&header), &read(&source));
        report(
            "6 every function of the door header has one wrapper",
            pass,
            what,
        );
    }
    ok
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |f: &str, d: &str| {
        a.iter()
            .position(|x| x == f)
            .and_then(|i| a.get(i + 1))
            .cloned()
            .unwrap_or(d.to_string())
    };
    // the door header and any further header of C functions (--header may repeat); their text is read as one
    let headers: Vec<String> = a
        .iter()
        .enumerate()
        .filter(|(_, x)| *x == "--header")
        .filter_map(|(i, _)| a.get(i + 1).cloned())
        .collect();
    if headers.is_empty() {
        eprintln!(
            "usage: gen --header H [--header H ...] --generated F [--check --template D --ext SO]"
        );
        std::process::exit(2)
    }
    let source: String = headers
        .iter()
        .map(|h| {
            fs::read_to_string(h).unwrap_or_else(|e| {
                eprintln!("cannot read {h}: {e}");
                std::process::exit(2)
            })
        })
        .collect::<Vec<_>>()
        .join("\n");
    let genfile = arg("--generated", "");
    if a.iter().any(|x| x == "--check") {
        let door = a
            .iter()
            .any(|x| x == "--door-header")
            .then(|| (arg("--door-header", ""), arg("--door-source", "")));
        let ok = check(
            &source,
            &arg("--template", ""),
            &genfile,
            &arg("--ext", ""),
            door,
        );
        println!("{}", if ok { "GATE GREEN" } else { "GATE RED" });
        std::process::exit(if ok { 0 } else { 1 });
    }
    match generate(&source) {
        Ok((c, nwrap, nstmts)) => {
            fs::write(&genfile, &c).expect("write");
            println!(
                "wrapped {nwrap} prototypes ({nstmts} statements read); wrote {genfile} ({} lines)",
                c.lines().count()
            )
        }
        Err(e) => {
            eprintln!("refused: {e}");
            std::process::exit(1)
        }
    }
}
