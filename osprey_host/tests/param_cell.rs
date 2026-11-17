//! The parameter-cell read the core builds equals the old `raptor.ir.cells.param_cell` for the same name and type, in both
//! forms. The oracle is `tests/data/param_cells.json`, written by raptor's `rs/support/tests/tools/gen_data_oracle.py`: each
//! case holds the old node's recorded graph, which is rebuilt in a second context and compared by its dump text.
#![cfg(feature = "old-contract")]
use raptor_ir::door::{rx_ctx_free, rx_ctx_new};
use raptor_ir::record::Ctx;
use raptor_support::corpus::{declare, make_nodes, manifests, text_out, Vals};
use raptor_support::safe::rx_dump;
use rcore::json;
use std::fs;

/// The dump text of the graph under `root`.
fn dump_of(ctx: *mut Ctx, root: u32) -> String {
    text_out(ctx, |c, buf, cap, len| rx_dump(c, &root, 1, buf, cap, len))
}

#[test]
fn the_param_cell_read_equals_the_old_function_in_both_forms() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/param_cells.json");
    let cases = json::parse(&fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(cases.arr().len(), 4);
    for case in cases.arr() {
        let shape: Vec<i64> = case
            .get("shape")
            .unwrap()
            .arr()
            .iter()
            .map(|d| d.int())
            .collect();
        let (old, new) = (rx_ctx_new(), rx_ctx_new());
        declare(old, &manifests(case)).unwrap();
        let handles = make_nodes(old, case, &mut Vals::default()).unwrap();
        let want = dump_of(old, handles[case.get("node").unwrap().int() as usize]);
        // SAFETY: `new` is live until the free below and nothing else uses it.
        let built = unsafe { &mut *new }.param_cell(
            case.get("name").unwrap().str().unwrap(),
            case.get("dtype").unwrap().str().unwrap(),
            &shape,
            case.get("tag").and_then(json::J::str),
            matches!(case.get("per_cell"), Some(json::J::Bool(true))),
        );
        assert_eq!(dump_of(new, built.unwrap()), want, "{case:?}");
        // SAFETY: both contexts came from `rx_ctx_new` and nothing refers to them after this case.
        unsafe {
            rx_ctx_free(old);
            rx_ctx_free(new);
        }
    }
}
