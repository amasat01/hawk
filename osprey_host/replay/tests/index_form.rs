//! The affine form of an index expression equals the old `raptor.ir.access.index_form` for the index of every table read of
//! the recorded graphs, against the sample index and against loop indices, with and without quotients. The oracle is
//! `tests/data/index_forms.json`, written by raptor's `rs/support/tests/tools/gen_data_oracle.py`; the graphs are rebuilt from
//! the recording in `OSPREY_CORPUS` (else the folder beside the clones).
#![cfg(feature = "old-contract")]
mod common;
use raptor_ir::door::{rx_ctx_free, rx_ctx_new};
use raptor_support::corpus::{declare, make_nodes, manifests, Vals};
use rcore::json;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[test]
fn the_index_form_equals_the_old_function_for_every_recorded_index_expression() {
    let corpus = std::env::var("OSPREY_CORPUS").map_or_else(
        |_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus"),
        PathBuf::from,
    );
    let dir = corpus.join("k6_record");
    if !dir.exists() {
        common::skipped(
            Some("OSPREY_CORPUS"),
            &format!("no recording at {}", dir.display()),
        );
        return;
    }
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/index_forms.json");
    let rows = json::parse(&fs::read_to_string(path).unwrap()).unwrap();
    assert!(rows.arr().len() > 600);
    // the rows of one recorded line share one rebuilt graph
    let mut graphs: HashMap<(String, i64), (*mut raptor_ir::record::Ctx, Vec<u32>)> =
        HashMap::new();
    for row in rows.arr() {
        // a hand-made expression carries its own node table; a recorded one is read from the recording
        let (key, line) = match row.get("nodes") {
            Some(_) => (format!("hand-made {}", graphs.len()), 0),
            None => (
                row.get("file").unwrap().str().unwrap().to_string(),
                row.get("line").unwrap().int(),
            ),
        };
        let (ctx, handles) = graphs.entry((key.clone(), line)).or_insert_with(|| {
            let table = match row.get("nodes") {
                Some(_) => row.clone(),
                None => {
                    let text = fs::read_to_string(dir.join(&key)).unwrap();
                    json::parse(text.lines().nth(line as usize).unwrap()).unwrap()
                }
            };
            let ctx = rx_ctx_new();
            declare(ctx, &manifests(&table)).unwrap();
            (ctx, make_nodes(ctx, &table, &mut Vals::default()).unwrap())
        });
        // SAFETY: `ctx` is live until the frees below and the test thread is its only user.
        let core = unsafe { &**ctx };
        let node = handles[row.get("node").unwrap().int() as usize];
        let var = row.get("var").and_then(|v| match v {
            json::J::Num(n) => Some(handles[*n as usize]),
            _ => None,
        });
        for (quotients, want) in [true, false]
            .into_iter()
            .zip(row.get("answers").unwrap().arr())
        {
            assert_eq!(
                core.index_form(node, var, quotients).unwrap(),
                want.strs(),
                "{row:?}"
            );
        }
    }
    for (ctx, _) in graphs.into_values() {
        // SAFETY: each context came from `rx_ctx_new` and nothing refers to it after this loop.
        unsafe { rx_ctx_free(ctx) }
    }
}
