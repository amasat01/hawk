//! The host build card (`build.json`): the numbers the host build reads, parsed into the record the steps decide on and answered to
//! the plan through [`HostCards`]. The same schema rules as every card: a document that does not match is refused by name.
use crate::host_facts::host_fusion;
use aether_writers::contract::types::Target;
use aether_writers::contract::types::{
    Build, FoldForm, FormSettings, LaneSettings, Scalar, DEFAULT_CELLS_ROW, DEFAULT_WARPS,
};
use aether_writers::contract::Json;
use aether_writers::refuse::{fault, Refusal};
use osprey_lowering::facts::{CardSet, FusionRow};
use osprey_lowering::lower::card::{
    box_lanes, card_json, defect, number, setting_names, steps_reverse, team_threads,
    wide_thresholds, BuildNumbers, ContractionRoute, TargetCards, WarpForm,
};
use osprey_lowering::lower::image::ImageCommand;
use osprey_lowering::lower::request::{KernelRequest, Keys};
use osprey_lowering::lower::unit_meta::{DispatchCosts, StepsPair};

/// The cards of the host target: the parsed card.
pub struct HostCards {
    /// The numbers the card states.
    numbers: BuildNumbers,
}

impl HostCards {
    /// The cards from the documents `docs`.
    ///
    /// # Errors
    /// Refuses a card that does not match its schema.
    pub fn of_docs(docs: &CardSet) -> Result<HostCards, Refusal> {
        Ok(HostCards {
            numbers: host_build_of(docs)?,
        })
    }
}

impl TargetCards for HostCards {
    fn numbers(&self) -> Build {
        build_of(&self.numbers)
    }
    fn box_lanes(&self) -> &[(String, i64)] {
        &self.numbers.box_lanes
    }
    fn settings(&self) -> &[String] {
        &self.numbers.settings
    }
    fn steps_reverse(&self) -> StepsPair {
        StepsPair {
            host: self.numbers.steps_reverse.clone(),
            device: None,
        }
    }
    fn deciding(&self) -> &BuildNumbers {
        &self.numbers
    }
    fn wide_class(&self) -> Option<&str> {
        None
    }
    fn contraction_route(&self, _scalar: Scalar) -> Option<ContractionRoute> {
        None
    }
    fn warp_form(&self, _scalar: Scalar) -> Option<WarpForm> {
        None
    }
    fn image(&self) -> Option<&ImageCommand> {
        None
    }
    fn fusion_row(&self, request: &KernelRequest, docs: &CardSet) -> Result<FusionRow, Refusal> {
        host_fusion(docs, request.facts_for(Target::Host)?)
            .map_err(osprey_lowering::facts::Failed::refusal)
    }
    fn dispatch_costs(
        &self,
        _request: &KernelRequest,
        _docs: &CardSet,
    ) -> Result<Option<DispatchCosts>, Refusal> {
        Ok(None)
    }
}

/// The build numbers of the host, with the caller's settings (fold form, warps, bounds, fold width) at their defaults.
fn build_of(numbers: &BuildNumbers) -> Build {
    Build {
        form: FormSettings {
            lazy_leaf_width: numbers.lazy_leaf_width,
            rolled_fold_min: numbers.rolled_fold_min,
            lean_width: Some(numbers.lean_width),
            uniform_run: Some(numbers.uniform_run),
        },
        fold_form: FoldForm::default(),
        lanes: LaneSettings {
            run_max: Some(numbers.run_max),
            block_run: Some(numbers.block_run),
            tile: Some((numbers.tile_min, numbers.tile_max)),
        },
        linalg_max_dim: Some(numbers.linalg_max_dim),
        cells_row: numbers.cells_row.max(DEFAULT_CELLS_ROW),
        blend: true,
        warps: DEFAULT_WARPS,
        launch_bounds: None,
        fold_k: None,
        fold_limits: None,
        wide_card: None,
    }
}

/// The host build card in effect in `docs`, the caller's own `build.json` or the shipped one.
pub fn host_build_of(docs: &CardSet) -> Result<BuildNumbers, Refusal> {
    let file = "build.json";
    host_build(&docs.text(file).map_err(fault)?).map_err(|why| defect(docs, file, why))
}

/// The bytes above which a group's scratch plane is held by the runtime. The groups of a launch run on the threads of the host
/// pool, which are plain threads whose stack is the platform's default for a secondary thread: 512 KiB on macOS, the smallest of
/// the hosts we run on (8 MiB on Linux). A plane of an eighth of that, 64 KiB, leaves the rest of the smallest stack to the
/// unit's other arrays and the frames above it; a larger plane overruns a worker's stack and ends the process.
fn group_scratch_plane_bytes(top: &mut Keys) -> Result<i64, Refusal> {
    let bytes = number(top, "group_scratch_plane_bytes")?;
    match bytes > 0 {
        true => Ok(bytes),
        false => Err(fault(format!(
            "facts: group_scratch_plane_bytes {bytes} must be positive"
        ))),
    }
}

/// Reads the text of the host build card (`build.json`); every defect is a fault, which `host_build_of` turns into the caller's
/// refusal when the text is the caller's.
pub fn host_build(text: &str) -> Result<BuildNumbers, Refusal> {
    let card = card_json(text, "build.json")?;
    let section = |keys: &mut Keys, key: &'static str| -> Result<Json, Refusal> {
        keys.req(key)
            .cloned()
            .map_err(|why| fault(format!("facts: build.json: {why}")))
    };
    let mut top = Keys::of(&card, "build card")?;
    let (form, lanes) = (section(&mut top, "form")?, section(&mut top, "lanes")?);
    let mut form = Keys::of(&form, "form")?;
    let mut lanes = Keys::of(&lanes, "lanes")?;
    let (team_threads, team_threads_large) = team_threads(&mut lanes)?;
    let found = BuildNumbers {
        lazy_leaf_width: number(&mut form, "lazy_leaf_width")?,
        rolled_fold_min: number(&mut form, "rolled_fold_min")?,
        lean_width: number(&mut form, "lean_width")?,
        uniform_run: number(&mut top, "uniform_run")?,
        run_max: number(&mut lanes, "run_max")?,
        block_run: number(&mut lanes, "block_run")?,
        tile_min: number(&mut lanes, "tile_min")?,
        tile_max: number(&mut lanes, "tile_max")?,
        linalg_max_dim: number(&mut top, "linalg_max_dim")?,
        linalg_team_max_dim: number(&mut top, "linalg_team_max_dim")?,
        warp_entries: vec![],
        cells_row: number(&mut top, "cells_row")?,
        scatter_max_chunks: number(&mut top, "scatter_max_chunks")?,
        scatter_threads: number(&mut top, "scatter_threads")?,
        warp_size: None,
        copy_threads: number(&mut lanes, "copy_threads")?,
        team_threads,
        team_threads_large,
        group_scratch_plane_bytes: Some(group_scratch_plane_bytes(&mut top)?),
        box_lanes: box_lanes(&section(&mut top, "thread_box")?, "build.json")?,
        steps_reverse: Some(steps_reverse(top.req("steps_reverse")?, "build.json")?),
        device_steps_reverse: None,
        wide: wide_thresholds(&section(&mut top, "host_wide")?)?,
        settings: setting_names(
            top.req("settings")
                .map_err(|why| fault(format!("facts: build.json: {why}")))?,
        )?,
    };
    for done in [form.done(), lanes.done(), top.done()] {
        done.map_err(|why| fault(format!("facts: build.json: {why}")))?;
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    use osprey_lowering::lower::unit_meta::StepsRegime;

    /// The shipped build card with the lean width changed to 64, as a stated document.
    fn lean_card() -> Vec<(String, String)> {
        let text = std::fs::read_to_string(crate::testing::shipped().join("build.json")).unwrap();
        vec![(
            "build.json".into(),
            text.replace("\"lean_width\": 4", "\"lean_width\": 64"),
        )]
    }

    /// Oracle: the shipped build card states a lean width of 4.
    #[test]
    fn a_build_card_stated_in_the_request_replaces_the_file_for_that_request() {
        let kept = CardSet::new(&crate::testing::shipped(), &[], crate::testing::ALL_CARDS);
        let stated = CardSet::new(
            &crate::testing::shipped(),
            &lean_card(),
            crate::testing::ALL_CARDS,
        );
        let host = |docs: &CardSet| host_build(&docs.text("build.json").unwrap()).unwrap();
        assert_eq!(host(&kept).lean_width, 4);
        assert_eq!(host(&stated).lean_width, 64);
    }

    /// Oracle: the host's own card has no warp rows (they are the device card's, read at an architecture).
    #[test]
    fn the_host_card_states_no_warp_rows() {
        let host = host_build(crate::testing::text("build.json")).unwrap();
        assert!(host.warp_entries.is_empty());
    }

    /// Oracle: the shipped host card states the `lanes` regime and the rows (4, 8), (4, 4), (2, 8); a regime the unit record has no
    /// name for, a row whose widest regime is below its narrowest and a missing rows list are refused by name.
    #[test]
    fn the_steps_table_of_the_card_is_read_strictly() {
        let text = crate::testing::text("build.json");
        let card = host_build(text).unwrap();
        let steps = card.steps_reverse.as_ref().unwrap();
        assert_eq!(steps.regime, StepsRegime::Lanes);
        assert_eq!(
            (steps.widths.clone(), steps.ks.clone()),
            (vec![4, 4, 2], vec![8, 4, 8])
        );
        for (old, new, named) in [
            (
                "\"regime\": \"lanes\"",
                "\"regime\": \"samples\"",
                "samples",
            ),
            (
                "\"lo\": 1, \"hi\": 1024",
                "\"lo\": 2000, \"hi\": 1024",
                "above hi",
            ),
            ("\"rows\": [", "\"rowz\": [", "rows"),
        ] {
            let why = host_build(&text.replacen(old, new, 1)).unwrap_err();
            assert!(why.reason.contains(named), "{}", why.reason);
        }
    }
}
