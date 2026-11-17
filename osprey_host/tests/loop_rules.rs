//! The loop rules of the core (the terms of an `add` chain, the nested loops a derivative cannot carry, accumulator or
//! recurrence) equal the answers of the old Python functions over every recorded loop and two hand-made nested loops. The oracle is
//! `tests/data/loop_rules_oracle.jsonl`, written by raptor's `rs/support/tests/tools/gen_loop_oracle.py` from the old
//! `raptor.ir.loops` over the kernel recording; the graphs are rebuilt from that recording in `OSPREY_CORPUS` (else the
//! folder beside the clones).
#![cfg(feature = "old-contract")]
mod common;
use raptor_ir::door::{rx_ctx_free, rx_ctx_new};
use raptor_ir::record::{Ctx, Fault};
use raptor_ir::value::Val;
use raptor_support::corpus::{declare, err, make_nodes, manifests, Vals};
use rcore::json::{self, J};
use std::fs;
use std::path::PathBuf;

/// The text of a refusal.
fn text_of(fault: Fault) -> String {
    match fault {
        Fault::Refused(text) => text,
        other => panic!("not a refusal: {other:?}"),
    }
}

/// One recorded loop with its rebuilt graph.
struct Case {
    /// The oracle row.
    row: J,
    /// The context holding the graph.
    ctx: *mut Ctx,
    /// The handle of every recorded node, by its place in the record.
    handles: Vec<u32>,
}

impl Case {
    /// The context of the case, for the questions that only read.
    fn ctx(&self) -> &Ctx {
        // SAFETY: `ctx` is live until `drop`, and the test thread is its only user.
        unsafe { &*self.ctx }
    }

    /// The context of the case, for the rule that makes nodes.
    fn ctx_mut(&mut self) -> &mut Ctx {
        // SAFETY: as `ctx`; the exclusive borrow of the case makes this the only reference.
        unsafe { &mut *self.ctx }
    }

    /// The handle of the recorded loop.
    fn lp(&self) -> u32 {
        self.handles[self.row.get("node").unwrap().int() as usize]
    }

    /// The nodes a field of the loop holds, in order.
    fn nodes_of(&self, field: &str) -> Vec<u32> {
        match self.ctx().field(self.lp(), field).unwrap() {
            Val::Tuple(items) | Val::List(items) => items
                .iter()
                .map(|v| match v {
                    Val::Node(n) => *n,
                    other => panic!("{field}: not a node: {other:?}"),
                })
                .collect(),
            other => panic!("{field}: not a list: {other:?}"),
        }
    }

    /// Whether the node `n` is the one the oracle writes as `want`: a node of the record, or an `add` the rule made.
    fn same(&self, n: u32, want: &J) -> bool {
        if let Some(index) = want.get("n") {
            return self.handles[index.int() as usize] == n;
        }
        let ctx = self.ctx();
        let operands = want.get("operands").unwrap().arr();
        let is_add = matches!(ctx.field(n, "kind"), Ok(Val::Str(k)) if k == "add");
        is_add
            && ctx.node(n).unwrap().ops.len() == operands.len()
            && ctx
                .node(n)
                .unwrap()
                .ops
                .iter()
                .zip(operands)
                .all(|(&o, w)| self.same(o, w))
    }
}

impl Drop for Case {
    fn drop(&mut self) {
        // SAFETY: `ctx` came from `rx_ctx_new` and nothing refers to it after this case.
        unsafe { rx_ctx_free(self.ctx) }
    }
}

/// Every oracle row with its graph rebuilt, or none when the recording is not there.
fn cases() -> Option<Vec<Case>> {
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
        return None;
    }
    let oracle = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/loop_rules_oracle.jsonl"
    ))
    .unwrap();
    let mut out = vec![];
    for line in oracle.lines() {
        let row = json::parse(line).unwrap();
        // a hand-made loop carries its own node table; a recorded one is read from the recording
        let table = if row.get("nodes").is_some() {
            row.clone()
        } else {
            let name = row.get("file").unwrap().str().unwrap();
            let text = fs::read_to_string(dir.join(name)).unwrap();
            let record = text
                .lines()
                .nth(row.get("line").unwrap().int() as usize)
                .unwrap();
            json::parse(record).unwrap()
        };
        let ctx = rx_ctx_new();
        let built = declare(ctx, &manifests(&table))
            .and_then(|()| make_nodes(ctx, &table, &mut Vals::default()));
        let handles = built.unwrap_or_else(|e| panic!("{e} ({})", err(ctx)));
        out.push(Case { row, ctx, handles });
    }
    assert!(!out.is_empty(), "the oracle holds no loop");
    Some(out)
}

#[test]
fn the_terms_of_every_recorded_next_value_equal_the_old_addends_function() {
    let Some(all) = cases() else { return };
    for case in &all {
        let nexts = case.nodes_of("nexts");
        let want = case.row.get("terms").unwrap().arr();
        assert_eq!(nexts.len(), want.len());
        for (next, terms) in nexts.iter().zip(want) {
            let got = case.ctx().add_terms(*next).unwrap();
            let terms = terms.arr();
            assert_eq!(got.len(), terms.len());
            assert!(got.iter().zip(terms).all(|(&n, w)| case.same(n, w)));
        }
    }
}

#[test]
fn the_class_of_every_recorded_loop_equals_the_old_classify_loop() {
    let Some(mut all) = cases() else { return };
    for case in &mut all {
        let lp = case.lp();
        let refused = case.row.get("class").unwrap().get("refused").cloned();
        if let Some(text) = refused {
            let fault = case.ctx().is_accumulator(lp).unwrap_err();
            assert_eq!(text_of(fault), text.str().unwrap());
            continue;
        }
        let accumulator =
            case.row.get("class").unwrap().get("name").unwrap().str() == Some("accumulator");
        assert_eq!(case.ctx().is_accumulator(lp).unwrap(), accumulator);
        let addends = case.ctx_mut().loop_addends(lp);
        if accumulator {
            let got = addends.unwrap();
            let want = case.row.get("class").unwrap().get("addends").unwrap().arr();
            assert_eq!(got.len(), want.len());
            assert!(got.iter().zip(want).all(|(&n, w)| case.same(n, w)));
        } else {
            assert!(addends.is_err());
        }
    }
}

#[test]
fn the_nesting_refusal_of_every_recorded_loop_equals_the_old_refuse_unsupported_nesting() {
    let Some(all) = cases() else { return };
    for case in &all {
        let got = case.ctx().refuse_nesting(case.lp());
        match case.row.get("nesting").and_then(J::str) {
            Some(text) => assert_eq!(text_of(got.unwrap_err()), text),
            None => assert!(got.is_ok()),
        }
    }
}
