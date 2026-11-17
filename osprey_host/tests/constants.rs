//! The constants and the contract rule the core owns equal the values of the old Python package. The oracle is
//! `tests/data/constants.json` and `tests/data/compensates.json`, written by raptor's
//! `rs/support/tests/tools/gen_data_oracle.py` from `raptor.ir`.
use raptor_ir::consts::{compensates, names};
use rcore::json::{self, J};
use std::fs;

/// A recorded oracle file, parsed.
fn oracle(file: &str) -> J {
    let path = format!("{}/tests/data/{file}", env!("CARGO_MANIFEST_DIR"));
    json::parse(&fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn every_constant_the_front_end_imports_equals_the_old_python_value() {
    let all = oracle("constants.json");
    let rows = all.obj();
    assert_eq!(rows.len(), 18);
    for (name, want) in rows {
        let want: Vec<String> = match want {
            J::Str(text) => vec![text.clone()],
            list => list.strs(),
        };
        let got = names(name).unwrap_or_else(|| panic!("the core owns no list {name}"));
        assert_eq!(got, want, "{name}");
    }
}

#[test]
fn compensates_equals_the_old_python_answer_for_every_contract_and_an_unknown_text() {
    let all = oracle("compensates.json");
    let rows = all.obj();
    assert_eq!(rows.len(), 7);
    for (contract, want) in rows {
        assert_eq!(
            compensates(contract),
            matches!(want, J::Bool(true)),
            "{contract}"
        );
    }
}
