//! What the integration tests share: scratch folders, fake tools that count their runs, and the request builders.
#![allow(dead_code)] // each test file uses a part of this module
#![allow(clippy::cast_possible_truncation)] // the image builder's sizes are tiny
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a panic is the failure
use rcore::json::J;
use std::fs;
use std::path::{Path, PathBuf};
use toolkit::{wire, Env, HeaderTree, Refusal, Toolkit};

/// A fresh empty folder named for a test.
pub fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("toolkit-it-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Writes an executable script, through a child process.
pub fn script(path: &Path, body: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    // A child writes the file: this process, whose sibling test threads fork tools, never holds a write handle to an
    // executable. A fork made while such a handle is open keeps it until its own exec, and running the script meanwhile
    // fails with "text file busy" on Linux.
    let mut writer = std::process::Command::new("sh")
        .args(["-c", "cat > \"$1\" && chmod 755 \"$1\"", "sh"])
        .arg(path)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    std::io::Write::write_all(&mut writer.stdin.take().unwrap(), body.as_bytes()).unwrap();
    assert!(
        writer.wait().unwrap().success(),
        "writing {} failed",
        path.display()
    );
}

/// A fake compiler: answers its version, counts each real run in `dir/runs`, sleeps `dir/sleep` seconds when that file exists,
/// keeps the last assembly source it was given as `dir/last.s`, and writes `built` to the file after `-o`.
pub fn fake_compiler(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    let body = format!(
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo \"fakecc 1.0\"; exit 0; fi\n\
         echo run >> \"{d}/runs\"\nout=\"\"; prev=\"\"\n\
         for a in \"$@\"; do [ \"$prev\" = \"-o\" ] && out=\"$a\"; case \"$a\" in *.s) cp \"$a\" \"{d}/last.s\";; esac; prev=\"$a\"; done\n\
         [ -f \"{d}/sleep\" ] && sleep \"$(cat \"{d}/sleep\")\"\n\
         echo built > \"$out\"\n",
        d = dir.display()
    );
    script(&path, &body);
    path
}

/// How many real runs the fake compilers of `dir` made.
pub fn runs(dir: &Path) -> usize {
    fs::read_to_string(dir.join("runs")).map_or(0, |t| t.lines().count())
}

/// A tool table with one tool, `compiler`, found by the environment variable `var`.
pub fn one_tool_table(var: &str) -> String {
    format!("{{\"tools\":{{\"compiler\":{{\"rules\":[{{\"env\":\"{var}\",\"required\":true}}],\"fix\":\"set {var}\"}}}}}}")
}

/// A build request for the fake compiler.
pub fn unit(name: &str, text: &str) -> String {
    wire::render(&wire::obj(vec![
        ("name", wire::s(name)),
        ("text", wire::s(text)),
        (
            "template",
            wire::strs(&["{compiler}", "-c", "{source}", "-o", "{output}"]),
        ),
    ]))
}

/// The `files.output` path of a build answer.
pub fn output_of(answer: &str) -> PathBuf {
    let j = wire::parse("answer", answer).unwrap();
    PathBuf::from(
        j.get("files")
            .unwrap()
            .get("output")
            .and_then(J::str)
            .unwrap(),
    )
}

/// The text member `key` of a JSON answer.
pub fn member(answer: &str, key: &str) -> J {
    wire::parse("answer", answer)
        .unwrap()
        .get(key)
        .cloned()
        .unwrap()
}

/// Builds `source` (C++ text) into a shared library `dir/<name>.so` with the host compiler of this machine, and returns its path.
pub fn build_shared(dir: &Path, name: &str, source: &str) -> PathBuf {
    let (src, lib) = (
        dir.join(format!("{name}.cpp")),
        dir.join(format!("{name}.so")),
    );
    fs::write(&src, source).unwrap();
    let done = std::process::Command::new("c++")
        .args(["-shared", "-fPIC", "-o"])
        .arg(&lib)
        .arg(&src)
        .output()
        .unwrap();
    assert!(
        done.status.success(),
        "{}",
        String::from_utf8_lossy(&done.stderr)
    );
    lib
}

/// A little-endian ELF64 image with the sections the resource reader needs and one register-count and one frame-size record for
/// each `(kernel, registers, frame bytes)`.
pub fn elf_image(kernels: &[(&str, u32, u32)]) -> Vec<u8> {
    let shstr = b"\0.shstrtab\0.strtab\0.symtab\0.nv.info\0".to_vec();
    let (mut strtab, mut symtab, mut info) = (vec![0_u8], vec![0_u8; 24], vec![]);
    for (i, (name, regs, frame)) in kernels.iter().enumerate() {
        let (at, index) = (strtab.len() as u32, (i + 1) as u32);
        strtab.extend_from_slice(name.as_bytes());
        strtab.push(0);
        let mut symbol = vec![0_u8; 24];
        symbol[..4].copy_from_slice(&at.to_le_bytes());
        symtab.extend(symbol);
        for (attr, value) in [(0x2F_u8, regs), (0x11, frame)] {
            info.extend([4, attr, 8, 0]);
            info.extend(index.to_le_bytes());
            info.extend(value.to_le_bytes());
        }
    }
    let blocks = [shstr, strtab, symtab, info];
    let mut image = vec![0_u8; 64];
    image[..4].copy_from_slice(b"\x7fELF");
    let mut offsets = vec![];
    for block in &blocks {
        offsets.push(image.len() as u64);
        image.extend(block);
    }
    let table = image.len() as u64;
    // section 0 is null; then .shstrtab (name 1), .strtab (11), .symtab (19), .nv.info (27)
    let rows = [
        (1_u32, 3_u32, 0_u32),
        (11, 3, 0),
        (19, 2, 2),
        (27, 0x7000_0000, 0),
    ];
    image.extend(vec![0_u8; 64]);
    for (i, (name, kind, link)) in rows.iter().enumerate() {
        let mut header = vec![0_u8; 64];
        header[..4].copy_from_slice(&name.to_le_bytes());
        header[4..8].copy_from_slice(&kind.to_le_bytes());
        header[0x18..0x20].copy_from_slice(&offsets[i].to_le_bytes());
        header[0x20..0x28].copy_from_slice(&(blocks[i].len() as u64).to_le_bytes());
        header[0x28..0x2C].copy_from_slice(&link.to_le_bytes());
        image.extend(header);
    }
    image[0x28..0x30].copy_from_slice(&table.to_le_bytes());
    image[0x3A..0x3C].copy_from_slice(&64_u16.to_le_bytes());
    image[0x3C..0x3E].copy_from_slice(&5_u16.to_le_bytes());
    image[0x3E..0x40].copy_from_slice(&1_u16.to_le_bytes());
    image
}

/// A toolkit for `spec` and `trees` with what this library registers, as the library builds its own.
pub fn new_toolkit(spec: &str, trees: &[HeaderTree], env: Env) -> Result<Toolkit, Refusal> {
    Toolkit::new(spec, trees, env).map(osprey_cpu_lib::register)
}
