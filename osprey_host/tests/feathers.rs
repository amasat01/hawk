//! The feather question of the core equals the old `raptor.ir.nodes.has_feather` for every feather and every node of two
//! recorded graphs that the old package can build. The oracle is `tests/data/feathers.json`, written by raptor's
//! `rs/support/tests/tools/gen_data_oracle.py`; the graphs are rebuilt from the recording in `OSPREY_CORPUS` (else the folder
//! beside the clones).
#![cfg(feature = "old-contract")]
mod common;
use raptor_ir::door::{rx_ctx_free, rx_ctx_new};
use raptor_support::corpus::{declare, make_nodes, manifests, Vals};
use rcore::json::{self, J};
use std::fs;
use std::path::PathBuf;

#[test]
fn has_feather_equals_the_old_function_for_every_feather_and_every_buildable_recorded_node() {
    let corpus = std::env::var("OSPREY_CORPUS").map_or_else(
        |_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../corpus"),
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
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/feathers.json");
    let oracle = json::parse(&fs::read_to_string(path).unwrap()).unwrap();
    let feathers = oracle.get("feathers").unwrap().strs();
    let mut positives = 0;
    for row in oracle.get("rows").unwrap().arr() {
        let text = fs::read_to_string(dir.join(row.get("file").unwrap().str().unwrap())).unwrap();
        let record = text
            .lines()
            .nth(row.get("line").unwrap().int() as usize)
            .unwrap();
        let table = json::parse(record).unwrap();
        let ctx = rx_ctx_new();
        declare(ctx, &manifests(&table)).unwrap();
        let handles = make_nodes(ctx, &table, &mut Vals::default()).unwrap();
        // SAFETY: `ctx` is live until the free below and nothing else uses it.
        let core = unsafe { &*ctx };
        for answer in row.get("answers").unwrap().arr() {
            let node = handles[answer.arr()[0].int() as usize];
            for (name, want) in feathers.iter().zip(answer.arr()[1].arr()) {
                let want = matches!(want, J::Bool(true));
                assert_eq!(
                    core.has_feather(node, name).unwrap(),
                    want,
                    "node {node}: {name}"
                );
                positives += usize::from(want);
            }
        }
        // SAFETY: `ctx` came from `rx_ctx_new` and nothing refers to it after this row.
        unsafe { rx_ctx_free(ctx) }
    }
    assert!(
        positives > 0,
        "no node states a feather: the oracle covers no positive answer"
    );
}
