// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

//! gen [--header H] [--generated F]      writes the generated C of the extension from the door header
//! gen --check [--header H --template D --generated F --ext SO]   the gate
//! std only. A prototype the generator does not understand is refused, never skipped.
use std::collections::BTreeSet;
use std::fs;
use std::process::Command;

const LIFECYCLE: [&str; 4] = ["rx_abi_version", "rx_ctx_new", "rx_ctx_free", "rx_last_error"];
const TEMPLATE_ALLOW: [&str; 4] = ["Ctx_new", "Ctx_dealloc", "Node_dealloc", "PyInit__osprey"];
const TEMPLATE_RX: [&str; 3] = ["rx_ctx_new", "rx_ctx_free", "rx_abi_version"];
// the Node surface: python name, door function, getter (true) or method (false)
const NODE_SURFACE: [(&str, &str, bool); 4] = [("cls", "rx_class", true), ("ttype", "rx_ttype", true), ("operands", "rx_operands", true), ("field", "rx_field", false)];
const KEYWORDS: [&str; 5] = ["class", "def", "import", "from", "lambda"];

#[derive(Debug, Clone)]
enum P {
    NodeIn(String),
    WalkIn(String),
    Cstr(String, bool),
    Text(String, String),
    I64s(String, String),
    Nodes(String, String),
    OutText(String, String, String),
    OutNodes(String, String, String),
    OutNode(String),
    OutWalk(String),
    OutChars(String, usize),
}

struct Proto { name: String, ret: String, params: Vec<P>, text: String }

/// the header without block comments; an optional-pointer comment (`/* NULL ...*/`) becomes the marker `@opt`
fn strip(src: &str) -> String {
    let mut out = String::new();
    let mut rest = src;
    while let Some(i) = rest.find("/*") {
        out.push_str(&rest[..i]);
        let j = rest[i..].find("*/").expect("unterminated comment") + i;
        let body = rest[i + 2..j].trim();
        if body.starts_with("NULL") { out.push_str(" @opt") }
        rest = &rest[j + 2..];
    }
    out.push_str(rest);
    out
}

fn ident_last(s: &str) -> String {
    s.trim_end_matches(|c: char| !(c.is_alphanumeric() || c == '_')).rsplit(|c: char| !(c.is_alphanumeric() || c == '_')).next().unwrap_or("").to_string()
}

fn parse_params(raw: &str, whole: &str) -> Result<Vec<P>, String> {
    let bad = |m: &str| format!("cannot wrap `{whole}`: {m}");
    let ps: Vec<String> = raw.split(',').map(|p| p.split_whitespace().collect::<Vec<_>>().join(" ")).collect();
    let norm = |p: &str| p.replace(" *", "*").replace("* ", "*").replace("@opt", "").trim().to_string();
    let shape = |p: &str| { let n = norm(p); let k = n.rfind(|c: char| c == '*' || c == ' ').map_or(0, |i| i + 1); (n[..k].trim().to_string(), n[k..].to_string()) };
    if ps.is_empty() || shape(&ps[0]) != ("rx_ctx*".into(), "ctx".into()) { return Err(bad("the first parameter is not `rx_ctx* ctx`")) }
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
            let (a, b) = (nm.find('[').ok_or_else(|| bad("array"))?, nm.find(']').ok_or_else(|| bad("array"))?);
            if !nm.starts_with("char ") { return Err(bad("an array parameter that is not `char name[N]`")) }
            out.push(P::OutChars(ident_last(&nm[..a]), nm[a + 1..b].parse().map_err(|_| bad("array size"))?));
            i += 1;
        } else if t == "rx_node" { out.push(P::NodeIn(n)); i += 1 }
        else if t == "rx_walk" { out.push(P::WalkIn(n)); i += 1 }
        else if t == "const char*" && sz(&next) { out.push(P::Text(n, next.unwrap().1)); i += 2 }
        else if t == "const char*" { out.push(P::Cstr(n, opt)); i += 1 }
        else if t == "const int64_t*" && sz(&next) { out.push(P::I64s(n, next.unwrap().1)); i += 2 }
        else if t == "const rx_node*" && sz(&next) { out.push(P::Nodes(n, next.unwrap().1)); i += 2 }
        else if t == "char*" && sz(&next) && szp(&next2) { out.push(P::OutText(n, next.unwrap().1, next2.unwrap().1)); i += 3 }
        else if t == "rx_node*" && sz(&next) && szp(&next2) { out.push(P::OutNodes(n, next.unwrap().1, next2.unwrap().1)); i += 3 }
        else if t == "rx_node*" { out.push(P::OutNode(n)); i += 1 }
        else if t == "rx_walk*" { out.push(P::OutWalk(n)); i += 1 }
        else { return Err(bad(&format!("parameter `{}` has a shape the generator does not know", ps[i]))) }
    }
    if out.iter().filter(|p| matches!(p, P::OutText(..) | P::OutNodes(..) | P::OutNode(_) | P::OutWalk(_) | P::OutChars(..))).count() > 1 { return Err(bad("more than one output")) }
    Ok(out)
}

/// returns (prototypes, count of other statements understood: typedef, enum)
fn parse_header(src: &str) -> Result<(Vec<Proto>, usize), String> {
    let text = strip(src);
    let body: String = text.lines().filter(|l| !l.trim_start().starts_with('#')).collect::<Vec<_>>().join("\n").replace("extern \"C\" {", "");
    let (mut protos, mut other) = (vec![], 0);
    let mut stmts: Vec<String> = vec![];
    let (mut cur, mut depth) = (String::new(), 0);
    for c in body.chars() {
        match c {
            '{' => { depth += 1; cur.push(c) }
            '}' => { depth = (depth - 1).max(0); cur.push(c) }
            ';' if depth == 0 => { stmts.push(std::mem::take(&mut cur)) }
            _ => cur.push(c),
        }
    }
    for s in stmts {
        let s = s.replace("extern \"C\" {", "").replace('}', "");
        let s = s.trim();
        if s.is_empty() { continue }
        if s.starts_with("typedef ") || s.starts_with("enum") { other += 1; continue }
        let (a, b) = (s.find('(').ok_or(format!("cannot wrap `{s}`: not a prototype"))?, s.rfind(')').ok_or(format!("cannot wrap `{s}`: no closing parenthesis"))?);
        let name = ident_last(&s[..a]);
        let ret = s[..a].trim_end_matches(&name[..]).trim().replace(" *", "*");
        if !name.starts_with("rx_") { return Err(format!("cannot wrap `{s}`: the name does not start with rx_")) }
        let inner = s[a + 1..b].trim();
        let params = if inner == "void" { vec![] } else if LIFECYCLE.contains(&name.as_str()) { vec![] } else { parse_params(inner, s)? };
        if !LIFECYCLE.contains(&name.as_str()) && ret != "int" { return Err(format!("cannot wrap `{s}`: the return type is `{ret}`, not int")) }
        protos.push(Proto { name, ret, params, text: s.split_whitespace().collect::<Vec<_>>().join(" ") });
    }
    Ok((protos, other))
}

fn pyname(rx: &str) -> String {
    let n = rx.trim_start_matches("rx_");
    if KEYWORDS.contains(&n) { format!("{n}_") } else { n.to_string() }
}

fn wrapper(p: &Proto) -> String {
    let mut decl = String::new();
    let (mut fmt, mut refs, mut conv, mut done) = (String::new(), vec![], String::new(), String::new());
    let mut args: Vec<String> = vec!["s->c".into()];
    let mut out: Option<&P> = None;
    for q in &p.params {
        match q {
            P::NodeIn(n) => { decl += &format!("  PyObject* o_{n}; rx_node a_{n};\n"); fmt += "O"; refs.push(format!("&o_{n}")); conv += &format!("  if (osprey_handle(s, o_{n}, &a_{n}) < 0) goto done;\n"); args.push(format!("a_{n}")) }
            P::WalkIn(n) => { decl += &format!("  unsigned int a_{n};\n"); fmt += "I"; refs.push(format!("&a_{n}")); args.push(format!("(rx_walk)a_{n}")) }
            P::Cstr(n, opt) => { decl += &format!("  const char* a_{n};\n"); fmt += if *opt { "z" } else { "s" }; refs.push(format!("&a_{n}")); args.push(format!("a_{n}")) }
            P::Text(n, l) => { decl += &format!("  const char* a_{n}; Py_ssize_t l_{l};\n"); fmt += "s#"; refs.push(format!("&a_{n}")); refs.push(format!("&l_{l}")); args.push(format!("a_{n}")); args.push(format!("(size_t)l_{l}")) }
            P::I64s(n, l) => { decl += &format!("  PyObject* o_{n}; int64_t* a_{n} = NULL; Py_ssize_t l_{l} = 0;\n"); fmt += "O"; refs.push(format!("&o_{n}")); conv += &format!("  if (osprey_i64s(o_{n}, &a_{n}, &l_{l}) < 0) goto done;\n"); done += &format!("  PyMem_Free(a_{n});\n"); args.push(format!("a_{n}")); args.push(format!("(size_t)l_{l}")) }
            P::Nodes(n, l) => { decl += &format!("  PyObject* o_{n}; rx_node* a_{n} = NULL; Py_ssize_t l_{l} = 0;\n"); fmt += "O"; refs.push(format!("&o_{n}")); conv += &format!("  if (osprey_nodes(s, o_{n}, &a_{n}, &l_{l}) < 0) goto done;\n"); done += &format!("  PyMem_Free(a_{n});\n"); args.push(format!("a_{n}")); args.push(format!("(size_t)l_{l}")) }
            o => out = Some(o),
        }
    }
    let call = |extra: &[&str]| { let mut a = args.clone(); a.extend(extra.iter().map(|x| x.to_string())); format!("{}({})", p.name, a.join(", ")) };
    let parse = if fmt.is_empty() { "  if (!PyArg_ParseTuple(args, \"\")) return NULL;\n".to_string() } else { format!("  if (!PyArg_ParseTuple(args, \"{fmt}\", {})) return NULL;\n", refs.join(", ")) };
    let mut tail = String::new();
    match out {
        None => tail += &format!("  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  Py_INCREF(Py_None); res = Py_None;\n", call(&[])),
        Some(P::OutText(..)) => tail += &format!("  char sb[256]; char* hb = NULL; size_t len = 0;\n  st = {};\n  if (st == RX_TRUNC) {{ hb = PyMem_Malloc(len + 1); if (!hb) {{ PyErr_NoMemory(); goto done; }} st = {}; }}\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  res = PyUnicode_DecodeUTF8(hb ? hb : sb, (Py_ssize_t)len, \"strict\");\n  PyMem_Free(hb);\n", call(&["sb", "sizeof sb", "&len"]), call(&["hb", "len + 1", "&len"])),
        Some(P::OutNodes(..)) => tail += &format!("  rx_node sn[64]; rx_node* hn = NULL; size_t cnt = 0;\n  st = {};\n  if (st == RX_TRUNC) {{ hn = PyMem_Malloc((cnt ? cnt : 1) * sizeof(rx_node)); if (!hn) {{ PyErr_NoMemory(); goto done; }} st = {}; }}\n  if (st != RX_OK) {{ osprey_fail(s, st); PyMem_Free(hn); goto done; }}\n  res = osprey_node_tuple(s, hn ? hn : sn, cnt);\n  PyMem_Free(hn);\n", call(&["sn", "64", "&cnt"]), call(&["hn", "cnt", "&cnt"])),
        Some(P::OutNode(_)) => tail += &format!("  rx_node oh = 0;\n  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  res = osprey_wrap_node(s, oh);\n", call(&["&oh"])),
        Some(P::OutWalk(_)) => tail += &format!("  rx_walk ow = 0;\n  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  res = PyLong_FromUnsignedLong(ow);\n", call(&["&ow"])),
        Some(P::OutChars(_, n)) => tail += &format!("  char dg[{n}] = {{0}};\n  st = {};\n  if (st != RX_OK) {{ osprey_fail(s, st); goto done; }}\n  res = PyUnicode_FromString(dg);\n", call(&["dg"])),
        _ => unreachable!(),
    }
    format!("/* wrapper: {n} */\n/* {t} */\nstatic PyObject* call_{n}(Ctx* s, PyObject* args) {{\n  PyObject* res = NULL; int st;\n{decl}{parse}{conv}{tail}done:\n{done}  return res;\n}}\nstatic PyObject* m_{n}(PyObject* self, PyObject* args) {{ return call_{n}((Ctx*)self, args); }}\n\n", n = p.name.trim_start_matches("rx_"), t = p.text)
}

const PRELUDE: &str = r#"// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* Generated by gen from raptor_ir.h. Do not edit. */
#include "osprey.h"

static inline void osprey_fail(Ctx* s, int st) {
  if (st == RX_REFUSED || st == RX_BADARG) PyErr_SetString(PyExc_ValueError, rx_last_error(s->c));
  else PyErr_Format(PyExc_RuntimeError, "door status %d: %s", st, rx_last_error(s->c));
}

static inline PyObject* osprey_wrap_node(Ctx* s, rx_node h) {
  Node* n = (Node*)PyType_GenericAlloc(osprey_node_type, 0);
  if (!n) return NULL;
  Py_INCREF((PyObject*)s);
  n->ctx = s;
  n->h = h;
  return (PyObject*)n;
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
  if (((Node*)o)->ctx != s) { PyErr_SetString(PyExc_TypeError, "a Node of another Ctx"); return -1; }
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
    let wrapped: Vec<&Proto> = protos.iter().filter(|p| !LIFECYCLE.contains(&p.name.as_str())).collect();
    for l in LIFECYCLE { if !protos.iter().any(|p| p.name == l) { return Err(format!("the lifecycle prototype {l} is missing from the header")) } }
    let mut c = String::from(PRELUDE);
    for p in &wrapped { c += &wrapper(p) }
    c += "PyMethodDef osprey_ctx_methods[] = {\n";
    for p in &wrapped { c += &format!("  {{\"{}\", m_{}, METH_VARARGS, NULL}},\n", pyname(&p.name), p.name.trim_start_matches("rx_")) }
    c += "  {NULL, NULL, 0, NULL}\n};\n\n";
    for (py, rx, getter) in NODE_SURFACE {
        if !wrapped.iter().any(|p| p.name == rx) { return Err(format!("the Node surface names {rx}, which the header does not have")) }
        let n = rx.trim_start_matches("rx_");
        if getter {
            c += &format!("static PyObject* g_{py}(PyObject* self, void* closure) {{\n  (void)closure;\n  PyObject* a = PyTuple_Pack(1, self);\n  if (!a) return NULL;\n  PyObject* r = call_{n}(((Node*)self)->ctx, a);\n  Py_DECREF(a);\n  return r;\n}}\n\n");
        } else {
            c += &format!("static PyObject* n_{py}(PyObject* self, PyObject* args) {{\n  PyObject* h = PyTuple_Pack(1, self);\n  if (!h) return NULL;\n  PyObject* a = PySequence_Concat(h, args);\n  Py_DECREF(h);\n  if (!a) return NULL;\n  PyObject* r = call_{n}(((Node*)self)->ctx, a);\n  Py_DECREF(a);\n  return r;\n}}\n\n");
        }
    }
    c += "PyGetSetDef osprey_node_getset[] = {\n";
    for (py, _, g) in NODE_SURFACE { if g { c += &format!("  {{\"{py}\", g_{py}, NULL, NULL, NULL}},\n") } }
    c += "  {NULL, NULL, NULL, NULL, NULL}\n};\n\nPyMethodDef osprey_node_methods[] = {\n";
    for (py, _, g) in NODE_SURFACE { if !g { c += &format!("  {{\"{py}\", n_{py}, METH_VARARGS, NULL}},\n") } }
    c += "  {NULL, NULL, 0, NULL}\n};\n";
    Ok((c, wrapped.len(), protos.len() + other))
}

/// C source without comments, string and char literals, and preprocessor lines
fn clean_c(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let (mut o, mut i) = (String::new(), 0);
    while i < b.len() {
        if b[i] == '/' && b.get(i + 1) == Some(&'*') { while i + 1 < b.len() && !(b[i] == '*' && b[i + 1] == '/') { i += 1 } i += 2; o.push(' '); continue }
        if b[i] == '/' && b.get(i + 1) == Some(&'/') { while i < b.len() && b[i] != '\n' { i += 1 } continue }
        if b[i] == '"' || b[i] == '\'' { let q = b[i]; i += 1; while i < b.len() && b[i] != q { if b[i] == '\\' { i += 1 } i += 1 } i += 1; o.push_str("0"); continue }
        o.push(b[i]); i += 1;
    }
    o.lines().filter(|l| !l.trim_start().starts_with('#')).collect::<Vec<_>>().join("\n")
}

/// names of the functions a C file defines (top level, a `(` before the `{`, no `=`)
fn defined_functions(src: &str) -> Vec<String> {
    let (mut out, mut cur, mut depth) = (vec![], String::new(), 0);
    for c in clean_c(src).chars() {
        match c {
            '{' => { if depth == 0 && cur.contains('(') && !cur.contains('=') { out.push(ident_last(cur.split('(').next().unwrap_or(""))) } if depth == 0 { cur.clear() } depth += 1 }
            '}' => { depth -= 1; if depth == 0 { cur.clear() } }
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
            let j = (i..b.len()).find(|&k| !(b[k].is_alphanumeric() || b[k] == '_')).unwrap_or(b.len());
            let w: String = b[i..j].iter().collect();
            let k = (j..b.len()).find(|&k| !b[k].is_whitespace()).unwrap_or(b.len());
            if w.starts_with("rx_") && b.get(k) == Some(&'(') { out.insert(w); }
            i = j;
        } else { i += 1 }
    }
    out
}

fn check(header: &str, tdir: &str, genfile: &str, ext: &str) -> bool {
    let mut ok = true;
    let mut report = |name: &str, pass: bool, what: String| { println!("[{}] {name}: {what}", if pass { "ok" } else { "RED" }); if !pass { ok = false } };
    let hsrc = fs::read_to_string(header).unwrap_or_default();
    let (fresh, nwrap, nstmts) = match generate(&hsrc) { Ok(x) => x, Err(e) => { println!("[RED] header: {e}"); return false } };
    // (1)
    let have = fs::read_to_string(genfile).unwrap_or_default();
    report("1 regeneration is byte-identical", !have.is_empty() && have == fresh, format!("{} bytes generated, {} bytes on disk, {} statements of the header examined", fresh.len(), have.len(), nstmts));
    // (2)
    let (mut nfn, mut bad) = (0usize, vec![]);
    let mut calls = 0usize;
    for f in ["types.c", "osprey.h"] {
        let src = fs::read_to_string(format!("{tdir}/{f}")).unwrap_or_default();
        if src.is_empty() { bad.push(format!("{f} is empty or missing")) }
        for n in defined_functions(&src) { nfn += 1; if !TEMPLATE_ALLOW.contains(&n.as_str()) { bad.push(format!("{f}: function {n} is not in the allow-list")) } }
        for n in rx_calls(&src) { calls += 1; if !TEMPLATE_RX.contains(&n.as_str()) { bad.push(format!("{f}: calls {n}")) } }
    }
    report("2 template defines only the allow-list and calls only the lifecycle", bad.is_empty() && nfn > 0, format!("{nfn} functions defined (allow-list {}), {calls} rx_ names called (allowed {}); {}", TEMPLATE_ALLOW.join(" "), TEMPLATE_RX.join(" "), if bad.is_empty() { "no violation".into() } else { bad.join("; ") }));
    // (3)
    let (protos, _) = parse_header(&hsrc).unwrap_or_default();
    let mut miss = vec![];
    let wrapped: Vec<&Proto> = protos.iter().filter(|p| !LIFECYCLE.contains(&p.name.as_str())).collect();
    for p in &wrapped {
        let n = p.name.trim_start_matches("rx_");
        let (m, d) = (have.matches(&format!("/* wrapper: {n} */")).count(), have.matches(&format!("static PyObject* call_{n}(")).count());
        let tab = have.matches(&format!("{{\"{}\", m_{n},", pyname(&p.name))).count();
        if (m, d, tab) != (1, 1, 1) { miss.push(format!("{} has marker {m}, definition {d}, table entry {tab}", p.name)) }
    }
    let extra = have.matches("/* wrapper: ").count() as isize - wrapped.len() as isize;
    if extra != 0 { miss.push(format!("{extra} wrapper(s) beyond the header")) }
    report("3 every non-lifecycle prototype has exactly one wrapper", miss.is_empty() && !wrapped.is_empty(), format!("{} prototypes in the header, {} wrapped, {} lifecycle; {}", protos.len(), wrapped.len(), protos.len() - wrapped.len(), if miss.is_empty() { "no violation".into() } else { miss.join("; ") }));
    let _ = nwrap;
    // (4)
    // macOS: `nm -gU` lists the exported symbols with a leading underscore; Linux: `nm -D --defined-only` lists the dynamic symbols
    let (flags, want, form): (&[&str], &str, &str) = if cfg!(target_os = "macos") { (&["-gU"], "_PyInit__osprey", "nm -gU") } else { (&["-D", "--defined-only"], "PyInit__osprey", "nm -D --defined-only") };
    let nm = Command::new("nm").args(flags).arg(ext).output();
    match nm {
        Ok(o) if o.status.success() => {
            let syms: Vec<String> = String::from_utf8_lossy(&o.stdout).lines().filter(|l| !l.trim().is_empty()).map(|l| l.split_whitespace().last().unwrap_or("").to_string()).collect();
            report("4 the extension exports exactly one symbol", syms == [want], format!("{form}: {} symbol(s): {}", syms.len(), syms.join(" ")));
        }
        _ => report("4 the extension exports exactly one symbol", false, format!("cannot run {form} on {ext}")),
    }
    ok
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |f: &str, d: &str| a.iter().position(|x| x == f).and_then(|i| a.get(i + 1)).cloned().unwrap_or(d.to_string());
    // defaults: the header from RX_SRC (a raptor checkout), everything else relative to the working directory (build.sh passes all of them)
    let src = std::env::var("RX_SRC").unwrap_or_else(|_| "..".to_string());
    let header = arg("--header", &format!("{src}/rs/ir/include/raptor_ir.h"));
    let genfile = arg("--generated", "build/_osprey.c");
    if a.iter().any(|x| x == "--check") {
        let ok = check(&header, &arg("--template", "template"), &genfile, &arg("--ext", "build/_osprey.abi3.so"));
        println!("{}", if ok { "GATE GREEN" } else { "GATE RED" });
        std::process::exit(if ok { 0 } else { 1 });
    }
    let src = fs::read_to_string(&header).unwrap_or_else(|e| { eprintln!("cannot read {header}: {e}"); std::process::exit(2) });
    match generate(&src) {
        Ok((c, nwrap, nstmts)) => { fs::write(&genfile, &c).expect("write"); println!("wrapped {nwrap} prototypes ({nstmts} statements read); wrote {genfile} ({} lines)", c.lines().count()) }
        Err(e) => { eprintln!("refused: {e}"); std::process::exit(1) }
    }
}
