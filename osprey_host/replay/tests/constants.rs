//! The constants and the contract rule the core owns equal the values of the old Python package. The oracle is
//! `tests/data/constants.json` and `tests/data/compensates.json`, written by raptor's
//! `rs/support/tests/tools/gen_data_oracle.py` from `raptor.ir`.
use raptor_ir::consts::names;
#[cfg(feature = "old-contract")]
use raptor_support::migrate::update_fields;
use rcore::json::{self, J};
use std::fs;

/// A recorded oracle file, parsed.
fn oracle(file: &str) -> J {
    let path = format!("{}/tests/data/{file}", env!("CARGO_MANIFEST_DIR"));
    json::parse(&fs::read_to_string(path).unwrap()).unwrap()
}

/// The constants of the oracle the core no longer owns: the accumulate contracts are the fields of an update now.
const REMOVED: [&str; 2] = ["KEYED_AUTO", "SCATTER_CONTRACTS"];

#[test]
fn every_constant_the_front_end_imports_equals_the_old_python_value() {
    let all = oracle("constants.json");
    let rows = all.obj();
    assert_eq!(rows.len(), 18);
    for (name, want) in rows
        .iter()
        .filter(|(name, _)| !REMOVED.contains(&name.as_str()))
    {
        let want: Vec<String> = match want {
            J::Str(text) => vec![text.clone()],
            list => list.strs(),
        };
        let got = names(name).unwrap_or_else(|| panic!("the core owns no list {name}"));
        assert_eq!(got, want, "{name}");
    }
}

/// Oracle: the old package's sample-sum companion suffix (`SAMPLE_SUM_COMP` of its cells module), which is not the keyed accumulate's
/// `COMPANION_SUFFIX`.
#[test]
fn the_sample_sum_companion_suffix_is_its_own_constant_beside_the_keyed_one() {
    assert_eq!(names("SAMPLE_SUM_COMP_SUFFIX"), Some(vec!["_comp"]));
    assert_eq!(names("COMPANION_SUFFIX"), Some(vec!["_c"]));
}

/// Whether an accumulate the migration makes of an old contract commits through a companion plane: an add under the compensated
/// summation.
#[cfg(feature = "old-contract")]
fn commits_through_companion(contract: &str) -> bool {
    let (op, summation, _) = update_fields(true, contract);
    op == "add" && summation == "compensated"
}

#[cfg(feature = "old-contract")]
#[test]
fn the_update_a_contract_migrates_to_compensates_as_the_old_python_answer_for_every_contract_and_an_unknown_text(
) {
    let all = oracle("compensates.json");
    let rows = all.obj();
    assert_eq!(rows.len(), 7);
    for (contract, want) in rows {
        assert_eq!(
            commits_through_companion(contract),
            matches!(want, J::Bool(true)),
            "{contract}"
        );
    }
}
