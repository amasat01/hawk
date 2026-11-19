//! One home per fact: a machine number (warp size, lane block, bank count, cache size, tile, SIMD width, thread count) is data of a
//! card or facts file, never an UPPER_CASE number constant in code. The scan reads each crate's sources and judges names only: an
//! item `const NAME: <integer type or array of integers> = <numbers>;` whose name carries a machine word is a hit, and the message
//! lists each `file:line`. Arithmetic inside a function cannot be judged statically and is not checked here.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a panic is the failure
use std::fs;
use std::path::{Path, PathBuf};

/// The words that make a name a machine number.
const MACHINE_WORDS: [&str; 12] = [
    "WARP", "WARPS", "LANE", "LANES", "BANK", "BANKS", "CACHE", "TILE", "TILES", "SIMD", "THREAD",
    "THREADS",
];

/// The source folders of the compiled pieces that must hold no machine number constant.
const PIECES: [&str; 5] = [
    "components/lowering/src",
    "components/toolkit/src",
    "osprey_cpu/src",
    "osprey_cuda/src",
    "../raptor/rs/wings/src",
];

/// Every `.rs` file under `dir`.
fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Whether `line` is an item that names a number constant: `const NAME: <integer type or [integer; n]> = <digits>...`.
fn machine_constant(line: &str) -> Option<String> {
    let item = line
        .trim_start()
        .trim_start_matches("pub(crate) ")
        .trim_start_matches("pub ");
    let rest = item.strip_prefix("const ")?;
    let (name, rest) = rest.split_once(':')?;
    let name = name.trim();
    let (kind, value) = rest.split_once('=')?;
    let integer = ["i32", "i64", "u32", "u64", "usize", "isize", "u8", "u16"]
        .iter()
        .any(|t| kind.contains(t));
    let numeric = value
        .trim_start()
        .trim_start_matches('[')
        .starts_with(|c: char| c.is_ascii_digit());
    let is_name = name
        .chars()
        .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit());
    (integer && numeric && is_name && name.split('_').any(|word| MACHINE_WORDS.contains(&word)))
        .then(|| item.trim_end().to_string())
}

/// The machine number constants of the folder `piece`, as `file:line: item`.
fn hits(piece: &str) -> Vec<String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(piece);
    let mut files = vec![];
    sources(&root, &mut files);
    assert!(!files.is_empty(), "no sources under {}", root.display());
    let mut out = vec![];
    for file in files {
        for (n, line) in fs::read_to_string(&file).unwrap().lines().enumerate() {
            if let Some(item) = machine_constant(line) {
                out.push(format!("{}:{}: {item}", file.display(), n + 1));
            }
        }
    }
    out
}

#[test]
fn the_scan_sees_a_number_constant_with_a_machine_word_and_not_a_text_or_a_plain_number() {
    assert!(machine_constant("const WARP_SIZE: i64 = 32;").is_some());
    assert!(machine_constant("pub const TILE_ROWS: [i64; 3] = [8, 4, 2];").is_some());
    assert!(machine_constant("const WARP_SUM: &str = \"warp_sum\";").is_none());
    assert!(machine_constant("const OTHER: i64 = 5;").is_none());
    assert!(machine_constant("const SLOT_PARAM_PLANE: u32 = 1 << 1;").is_none());
    assert!(machine_constant("const GEMM_PER_LANE: i64 = -1;").is_none());
    assert!(machine_constant("    let warp = 32;").is_none());
}

#[test]
#[ignore = "the compiled pieces hold machine number constants (lowering warp.rs TILE_ROWS, card.rs BOX_LANES_MAX, request.rs MAX_THREADS, raptor wings team.rs WARP_THREADS)"]
fn no_compiled_piece_holds_a_machine_number_constant() {
    let all: Vec<String> = PIECES.iter().flat_map(|p| hits(p)).collect();
    assert!(
        all.is_empty(),
        "{} machine number constant(s) outside the facts:\n{}",
        all.len(),
        all.join("\n")
    );
}
