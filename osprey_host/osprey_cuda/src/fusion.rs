//! The fusion row of a device build: the row of the device document for the architecture and the scalar type, found by the card's
//! hardware distance.
use crate::device_facts::device_fusion;
use aether_writers::contract::types::Scalar;
use aether_writers::refuse::Refusal;
use osprey_lowering::facts::{CardSet, Failed, FusionRow};

/// The fusion row of a device build for the architecture and the scalar type: the stated row of that type whose architecture is
/// nearest by hardware, else the device document's default row.
pub fn device_row(arch: &str, scalar: Scalar, docs: &CardSet) -> Result<FusionRow, Refusal> {
    let rows = device_fusion(docs).map_err(Failed::refusal)?;
    let device = crate::card::device_build_of(docs)?;
    let of_type: Vec<(String, FusionRow)> = rows
        .rows
        .into_iter()
        .filter(|(_, dtype, _)| match scalar {
            Scalar::Float32 => dtype == "float32",
            Scalar::Float64 => dtype == "float64",
            Scalar::Banded => false,
        })
        .map(|(cc, _, row)| (cc, row))
        .collect();
    Ok(device
        .nearest_row(arch, &of_type)?
        .cloned()
        .unwrap_or(rows.defaults))
}

#[cfg(test)]
mod tests {
    //! The fusion row of a device build follows the rows the device document states.
    use super::*;

    /// The shipped document with the rows `rows` (JSON) stated in place of its empty list.
    fn docs_with_rows(rows: &str) -> CardSet {
        let dir = crate::testing::shipped();
        let text = std::fs::read_to_string(dir.join("cuda.json")).unwrap();
        let stated = text.replacen(
            "\"fusion_rows\": []",
            &format!("\"fusion_rows\": {rows}"),
            1,
        );
        CardSet::new(
            &dir,
            &[("cuda.json".to_string(), stated)],
            crate::testing::ALL_CARDS,
        )
    }

    /// Oracle: the shipped default row holds 255 registers. A row stated for `sm_61` and `float32` holds 64 and leaves the other
    /// numbers to the defaults; it serves that type on every architecture (it is the only row, so the nearest), and no other type.
    #[test]
    fn a_stated_row_serves_its_scalar_type_by_the_nearest_architecture_and_the_others_take_the_defaults(
    ) {
        let none = docs_with_rows("[]");
        let one = docs_with_rows(
            r#"[{"cc": "sm_61", "dtype": "float32", "regs_file": 64, "local_over_plane": null}]"#,
        );
        let regs = |docs: &CardSet, arch, scalar| device_row(arch, scalar, docs).unwrap();
        assert_eq!(regs(&none, "sm_61", Scalar::Float32).regs_file, 255);
        let row = regs(&one, "sm_80", Scalar::Float32);
        assert_eq!(
            (row.regs_file, row.local_over_plane, row.launch_us),
            (64, None, Some(2.7))
        );
        assert_eq!(regs(&one, "sm_61", Scalar::Float64).regs_file, 255);
        assert_eq!(regs(&one, "sm_61", Scalar::Banded).regs_file, 255);
    }

    /// Oracle: of two rows of one type the one of the architecture with hardware nearest the build's wins (sm_61 and sm_75 have an
    /// FP64 rate of 1/32 where sm_80 has 1/2 and sm_90 has 1/2).
    #[test]
    fn the_row_of_the_architecture_with_the_nearest_hardware_wins() {
        let docs = docs_with_rows(
            r#"[{"cc": "sm_61", "dtype": "float64", "regs_file": 64}, {"cc": "sm_90", "dtype": "float64", "regs_file": 128}]"#,
        );
        assert_eq!(
            device_row("sm_80", Scalar::Float64, &docs)
                .unwrap()
                .regs_file,
            128
        );
        assert_eq!(
            device_row("sm_75", Scalar::Float64, &docs)
                .unwrap()
                .regs_file,
            64
        );
    }

    #[test]
    fn a_stated_row_that_names_another_type_or_an_unknown_number_is_refused_by_name() {
        let other = docs_with_rows(r#"[{"cc": "sm_61", "dtype": "half"}]"#);
        let why = device_row("sm_61", Scalar::Float32, &other).unwrap_err();
        assert!(why.reason.contains("half"), "{}", why.reason);
        let unknown = docs_with_rows(r#"[{"cc": "sm_61", "dtype": "float32", "regs": 1}]"#);
        let why = device_row("sm_61", Scalar::Float32, &unknown).unwrap_err();
        assert!(why.reason.contains("regs"), "{}", why.reason);
    }
}
