//! The construction rules of a compound quantity equal the old `raptor.ir.compound.Quantity` constructor: sound declarations
//! are accepted and every refusal carries the old text, over slugs, packs and wire lists (including every wire role the
//! manifest names). The oracle is `tests/data/quantity.json`, written by raptor's
//! `rs/support/tests/tools/gen_data_oracle.py`.
use raptor_ir::quantity::quantity_check;
use raptor_ir::record::Fault;
use rcore::json::{self, J};
use std::fs;

#[test]
fn the_quantity_rules_equal_the_old_constructor_over_slugs_packs_and_wire_lists() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/quantity.json");
    let rows = json::parse(&fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(rows.arr().len(), 112);
    for row in rows.arr() {
        let wires: Vec<(String, String)> = row
            .get("wires")
            .unwrap()
            .arr()
            .iter()
            .map(|w| {
                (
                    w.arr()[0].str().unwrap().into(),
                    w.arr()[1].str().unwrap().into(),
                )
            })
            .collect();
        let got = quantity_check(
            row.get("slug").unwrap().str().unwrap(),
            row.get("pack").unwrap().str().unwrap(),
            &wires,
        );
        match row.get("refused").unwrap() {
            J::Str(text) => match got {
                Err(Fault::Refused(found)) => assert_eq!(&found, text, "{row:?}"),
                other => panic!("{row:?}: expected a refusal, got {other:?}"),
            },
            _ => assert!(got.is_ok(), "{row:?}: {got:?}"),
        }
    }
}
