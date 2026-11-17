//! The structural keys of the core partition the nodes of an order exactly as the old `raptor.ir.walk.structural_keys` does,
//! over every node the old package can build in a sample of the recorded graphs. The oracle is
//! `tests/data/structural_keys.json`, written by raptor's `rs/support/tests/tools/gen_data_oracle.py`; the graphs are rebuilt
//! from the recording in `OSPREY_CORPUS` (else the folder beside the clones).
#![cfg(feature = "old-contract")]
mod common;
use raptor_ir::door::{rx_ctx_free, rx_ctx_new};
use raptor_support::corpus::{declare, make_nodes, manifests, Vals};
use rcore::json;
use std::fs;
use std::path::PathBuf;

#[test]
fn the_structural_keys_partition_every_recorded_order_as_the_old_function_does() {
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
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/structural_keys.json"
    );
    let rows = json::parse(&fs::read_to_string(path).unwrap()).unwrap();
    let mut nodes = 0;
    for row in rows.arr() {
        let text = fs::read_to_string(dir.join(row.get("file").unwrap().str().unwrap())).unwrap();
        let table = json::parse(
            text.lines()
                .nth(row.get("line").unwrap().int() as usize)
                .unwrap(),
        )
        .unwrap();
        let ctx = rx_ctx_new();
        declare(ctx, &manifests(&table)).unwrap();
        let handles = make_nodes(ctx, &table, &mut Vals::default()).unwrap();
        let order: Vec<u32> = row
            .get("order")
            .unwrap()
            .arr()
            .iter()
            .map(|i| handles[i.int() as usize])
            .collect();
        // SAFETY: `ctx` is live until the free below and nothing else uses it.
        let got = unsafe { &*ctx }.structural_keys(&order).unwrap();
        let want: Vec<i64> = row
            .get("numbers")
            .unwrap()
            .arr()
            .iter()
            .map(json::J::int)
            .collect();
        assert_eq!(
            got,
            want,
            "{}:{}",
            row.get("file").unwrap().str().unwrap(),
            row.get("line").unwrap().int()
        );
        nodes += order.len();
        // SAFETY: `ctx` came from `rx_ctx_new` and nothing refers to it after this row.
        unsafe { rx_ctx_free(ctx) }
    }
    assert!(nodes >= 30000, "the oracle covers {nodes} nodes");
}
