//! The numbers the builds read that are neither the caller's settings nor the machine's facts: the choices each target makes for
//! its text (how wide a fold stays unrolled, the lane runs and tiles, the widest matrix one thread holds, the row of a reduction
//! over samples). The host's are data in `build.json` and the device's in `cuda_build.json`, beside the machine cards, loaded
//! when a request is lowered; a file that does not match its schema is refused by name.
use super::request::Keys;
use super::unit_meta::{StepsRegime, StepsTable};
use crate::facts::{blame, CardSet};
use aether_writers::contract::types::{
    Build, FoldForm, FoldLimits, FormSettings, LaneSettings, WideCard, DEFAULT_CELLS_ROW,
    DEFAULT_WARPS,
};
use aether_writers::contract::{Contract, Json};
use aether_writers::refuse::{bad, fault, Refusal};
use raptor_wings::team::WarpEntry;

/// The host build's numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct BuildCard {
    /// The element count above which a read-only leaf is bound as a proxy.
    pub lazy_leaf_width: i64,
    /// The most elements a fold stays unrolled over.
    pub rolled_fold_min: i64,
    /// The widest vector whose single indexed accumulate keeps the select-chain spelling.
    pub lean_width: i64,
    /// The aligned run of lanes a uniform access is proven equal over.
    pub uniform_run: i64,
    /// The longest run of lanes the text keeps whole.
    pub run_max: i64,
    /// The block of lanes of a run.
    pub block_run: i64,
    /// The smallest tile of an interchanged loop, in samples.
    pub tile_min: i64,
    /// The largest tile of an interchanged loop, in samples.
    pub tile_max: i64,
    /// The widest matrix one thread holds.
    pub linalg_max_dim: i64,
    /// The largest extent a team of threads works on.
    pub linalg_team_max_dim: i64,
    /// The factorisations a warp takes per lane at the architecture the build is for: empty for the host's own card; the device's
    /// rows with a double row kept only where the architecture's FP64 rate is at least `1 / linalg_fast_fp64` of its FP32 rate.
    pub warp_entries: Vec<WarpEntry>,
    /// The samples one row of a reduction over samples stands for.
    pub cells_row: i64,
    /// The most chunks of an ordered scatter a launch splits the lanes into.
    pub scatter_max_chunks: i64,
    /// The team of each stage of an ordered scatter.
    pub scatter_threads: i64,
    /// The threads of the block of a stage that only copies operands into team scratch.
    pub copy_threads: i64,
    /// The threads of a team for a matrix of order up to this many.
    pub team_threads: i64,
    /// The threads of a team for a larger matrix.
    pub team_threads_large: i64,
    /// The lanes one thread serves in a thread box, by scalar type name; a type with no entry is built without a box.
    pub box_lanes: Vec<(String, i64)>,
    /// The measured rows of the host's `steps_reverse` choice, when the build has a host target.
    pub steps_reverse: Option<StepsTable>,
    /// The measured rows of the device's, when the build has a device target.
    pub device_steps_reverse: Option<StepsTable>,
    /// The thresholds the host's wide-value placement reads, by the scalar's width in bytes.
    pub host_wide: Vec<(i64, WideLimits)>,
    /// The build settings the target declares, by the names a request states them under; a setting no target of a build declares is
    /// refused by name.
    pub settings: Vec<String>,
}

/// What the placement of wide values reads of the target for one scalar width.
#[derive(Debug, Clone, PartialEq)]
pub struct WideLimits {
    /// The widest vector kept in registers when accessed at a run-time index, by access pattern (read, slot read, rotate, write,
    /// accumulate).
    pub runtime_min: [i64; 5],
    /// The widest vector with only compile-time indices that stays in registers.
    pub wide_min: i64,
}

/// The access patterns of `runtime_min`, in the order the placement takes them.
const WIDE_PATTERNS: [&str; 5] = ["read", "slot_read", "rotate", "write", "accum"];
/// The scalar widths a wide table has an entry for.
const WIDE_WIDTHS: [i64; 2] = [4, 8];

/// The most lanes a thread box holds.
const BOX_LANES_MAX: i64 = 16;

impl BuildCard {
    /// The wide-value thresholds for a scalar of `real_bytes` bytes.
    pub fn wide_limits(&self, real_bytes: i64) -> Option<&WideLimits> {
        let found = self
            .host_wide
            .iter()
            .find(|(width, _)| *width == real_bytes);
        found.map(|(_, limits)| limits)
    }
}

/// The wide-value thresholds of the card, one entry per scalar width.
fn host_wide(json: &Json) -> Result<Vec<(i64, WideLimits)>, Refusal> {
    let bad = |what: &str| fault(format!("facts: wide thresholds: {what}"));
    let table = |value: Option<&Json>, width: i64| {
        value
            .and_then(|t| t.get(&width.to_string()))
            .and_then(Json::int)
            .ok_or_else(|| bad("a threshold is missing or not an integer"))
    };
    let (runtime, minimum) = (json.get("runtime_min"), json.get("wide_min"));
    WIDE_WIDTHS
        .iter()
        .map(|&width| {
            let mut runtime_min = [0; 5];
            for (slot, pattern) in runtime_min.iter_mut().zip(WIDE_PATTERNS) {
                *slot = table(runtime.and_then(|r| r.get(pattern)), width)?;
            }
            let wide_min = table(minimum, width)?;
            Ok((
                width,
                WideLimits {
                    runtime_min,
                    wide_min,
                },
            ))
        })
        .collect()
}

/// The thread box entries of the card: each a whole number of lanes from 2 up to the most a box holds.
fn box_lanes(json: &Json) -> Result<Vec<(String, i64)>, Refusal> {
    let Json::Obj(entries) = json else {
        return Err(fault("facts: build.json: thread_box is not an object"));
    };
    entries
        .iter()
        .map(|(name, lanes)| match lanes.int() {
            Some(n) if (2..=BOX_LANES_MAX).contains(&n) => Ok((name.clone(), n)),
            _ => Err(fault(format!(
                "facts: build.json: thread_box {name} is not a lane count from 2 to {BOX_LANES_MAX}"
            ))),
        })
        .collect()
}

/// The integer under a required key of the card, refused by name when it is missing or not an integer.
fn number(keys: &mut Keys, key: &'static str) -> Result<i64, Refusal> {
    keys.req(key)
        .map_err(|why| fault(format!("facts: {why}")))?
        .int()
        .ok_or_else(|| fault(format!("facts: {key} is not an integer")))
}

/// The setting names of a card's `settings` list: a list of texts.
fn setting_names(json: &Json) -> Result<Vec<String>, Refusal> {
    json.arr()
        .iter()
        .map(|name| {
            name.str()
                .map(String::from)
                .ok_or_else(|| fault("facts: settings is not a list of names"))
        })
        .collect()
}

/// The two team sizes of the card: the smaller serves a matrix up to its own order, the larger the rest. A card whose larger
/// team is below the smaller, or whose sizes are not positive, is refused by name.
fn team_threads(lanes: &mut Keys) -> Result<(i64, i64), Refusal> {
    let small = number(lanes, "team_threads")?;
    let large = number(lanes, "team_threads_large")?;
    match small > 0 && large >= small {
        true => Ok((small, large)),
        false => Err(fault(format!(
            "facts: team_threads {small} and team_threads_large {large} must be positive and not decreasing"
        ))),
    }
}

impl BuildCard {
    /// The build numbers of the host, with the caller's settings (fold form, warps, bounds, fold width) at their defaults.
    pub fn build(&self) -> Build {
        Build {
            form: FormSettings {
                lazy_leaf_width: self.lazy_leaf_width,
                rolled_fold_min: self.rolled_fold_min,
                lean_width: Some(self.lean_width),
                uniform_run: Some(self.uniform_run),
            },
            fold_form: FoldForm::default(),
            lanes: LaneSettings {
                run_max: Some(self.run_max),
                block_run: Some(self.block_run),
                tile: Some((self.tile_min, self.tile_max)),
            },
            linalg_max_dim: Some(self.linalg_max_dim),
            cells_row: self.cells_row.max(DEFAULT_CELLS_ROW),
            blend: true,
            warps: DEFAULT_WARPS,
            launch_bounds: None,
            fold_k: None,
            fold_limits: None,
            wide_card: None,
        }
    }
}

impl DeviceCard {
    /// The build numbers of the device, with the caller's settings at their defaults. The device text has no tile, no uniform run
    /// and no blend.
    pub fn build(&self) -> Build {
        Build {
            form: FormSettings {
                lazy_leaf_width: self.lazy_leaf_width,
                rolled_fold_min: self.rolled_fold_min,
                lean_width: Some(self.lean_width),
                uniform_run: None,
            },
            fold_form: FoldForm::default(),
            lanes: LaneSettings {
                run_max: Some(self.run_max),
                block_run: Some(self.block_run),
                tile: None,
            },
            linalg_max_dim: Some(self.linalg_max_dim),
            cells_row: self.cells_row.max(DEFAULT_CELLS_ROW),
            blend: false,
            warps: DEFAULT_WARPS,
            launch_bounds: None,
            fold_k: None,
            fold_limits: Some(self.fold.clone()),
            wide_card: Some(WideCard {
                row_scalars: self.wide_rows.iter().map(|row| row.dtype.clone()).collect(),
                slice_min: Some(self.slice_min),
                forced: false,
            }),
        }
    }
}

/// The parsed JSON of the text of a card document; `file` names it in a refusal.
fn card_json(text: &str, file: &str) -> Result<Json, Refusal> {
    Ok(Contract::parse(text)
        .map_err(|why| fault(format!("facts: {file}: {why}")))?
        .0)
}

/// The refusal for a defect of the document `file` found while reading it from `docs`: the caller's bad request when the caller
/// stated the document, a fault of the library when the document ships.
fn defect(docs: &CardSet, file: &str, why: Refusal) -> Refusal {
    docs.defect(file, why.reason).refusal()
}

/// The host build card in effect in `docs`, the caller's own `build.json` or the shipped one.
pub fn host_build_of(docs: &CardSet) -> Result<BuildCard, Refusal> {
    let file = "build.json";
    host_build(&docs.text(file).map_err(fault)?).map_err(|why| defect(docs, file, why))
}

/// The device build card in effect in `docs`, the caller's own `cuda_build.json` or the shipped one.
pub fn device_build_of(docs: &CardSet) -> Result<DeviceCard, Refusal> {
    let file = "cuda_build.json";
    let card =
        device_build(&docs.text(file).map_err(fault)?).map_err(|why| defect(docs, file, why))?;
    Ok(DeviceCard {
        stated: docs.states(file),
        ..card
    })
}

/// Reads the text of the host build card (`build.json`); every defect is a fault, which `host_build_of` turns into the caller's
/// refusal when the text is the caller's.
pub fn host_build(text: &str) -> Result<BuildCard, Refusal> {
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
    let found = BuildCard {
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
        copy_threads: number(&mut lanes, "copy_threads")?,
        team_threads,
        team_threads_large,
        box_lanes: box_lanes(&section(&mut top, "thread_box")?)?,
        steps_reverse: Some(steps_reverse(top.req("steps_reverse")?, "build.json")?),
        device_steps_reverse: None,
        host_wide: host_wide(&section(&mut top, "host_wide")?)?,
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

/// The numbers the device build reads, from `cuda_build.json`: the choices for the text of a device kernel, the limits of its
/// folds and wide vectors and the lanes of its thread box. The same schema rules as the host card.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceCard {
    /// The element count above which a read-only leaf is bound as a proxy.
    pub lazy_leaf_width: i64,
    /// The most elements a fold stays unrolled over.
    pub rolled_fold_min: i64,
    /// The widest vector whose single indexed accumulate keeps the select-chain spelling.
    pub lean_width: i64,
    /// The aligned run of lanes a uniform access is proven equal over.
    pub uniform_run: i64,
    /// The longest run of lanes the text keeps whole.
    pub run_max: i64,
    /// The block of lanes of a run.
    pub block_run: i64,
    /// The smallest tile of an interchanged loop, in samples.
    pub tile_min: i64,
    /// The largest tile of an interchanged loop, in samples.
    pub tile_max: i64,
    /// The widest matrix one thread holds.
    pub linalg_max_dim: i64,
    /// The samples one row of a reduction over samples stands for.
    pub cells_row: i64,
    /// The limits of a fold.
    pub fold: FoldLimits,
    /// The lanes one thread serves in a thread box, by scalar type name.
    pub box_lanes: Vec<(String, i64)>,
    /// How a dense contraction is run, by scalar type name: `gemm` or `direct`.
    pub box_contraction: Vec<(String, String)>,
    /// The thread box choices measured on one device and scalar type, which replace the card's own for that type.
    pub box_rows: Vec<BoxRow>,
    /// The narrowest vector held sliced.
    pub slice_min: i64,
    /// The shortest sliced value from which a warp is chosen without being asked for.
    pub team_min: i64,
    /// The bytes of shared memory the output tiles of one block may take.
    pub tile_bytes: i64,
    /// The measured rows of the device's `steps_reverse` choice.
    pub steps_reverse: StepsTable,
    /// The lanes of a warp.
    pub warp_size: i64,
    /// The resident threads of one multiprocessor: the most a launch's threads times blocks may come to.
    pub max_threads_per_sm: i64,
    /// The wide thresholds measured for one architecture and scalar type, which replace the card's own for that type.
    pub wide_rows: Vec<WideRow>,
    /// The build settings the device declares, by the names a request states them under.
    pub settings: Vec<String>,
    /// The largest extent a team of threads works on.
    pub linalg_team_max_dim: i64,
    /// The most FP32 lanes per FP64 lane a device may have for a double warp row to be taken.
    pub linalg_fast_fp64: i64,
    /// The factorisations a warp takes per lane.
    pub linalg_warp_entries: Vec<WarpEntry>,
    /// The threads of the block of a stage that only copies operands into team scratch.
    pub copy_threads: i64,
    /// The threads of a team for a matrix of order up to this many.
    pub team_threads: i64,
    /// The threads of a team for a larger matrix.
    pub team_threads_large: i64,
    /// The most chunks of an ordered scatter a launch splits the lanes into.
    pub scatter_max_chunks: i64,
    /// The team of each stage of an ordered scatter.
    pub scatter_threads: i64,
    /// What the hardware of each compute capability offers, by capability.
    pub arch_features: Vec<(i64, ArchFeatures)>,
    /// The classes of device the wide thresholds were measured on.
    pub wide_classes: Vec<WideClass>,
    /// Whether the caller stated the document this card was read from, which decides what a value it cannot satisfy is.
    stated: bool,
}

/// How a dense contraction of one scalar type runs on a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractionRoute {
    /// The library states the matrix products beside the kernel, for a device plan to run through the vendor's library.
    Gemm,
    /// The unit states none and runs its own loop everywhere.
    Direct,
}

impl DeviceCard {
    /// The route of a contraction of the scalar type `scalar` (`float32` or `float64`) on the architecture `arch`: the row of the
    /// card's thread box rows for that architecture and type that states a route, else the card's route for the type, else
    /// `Gemm`. A row matches on architecture and type only: the library is told no device name.
    pub fn contraction_route(&self, arch: &str, scalar: &str) -> ContractionRoute {
        let word = self
            .box_rows
            .iter()
            .find(|row| row.cc == arch && row.dtype == scalar && row.contraction.is_some())
            .and_then(|row| row.contraction.as_deref())
            .or_else(|| {
                self.box_contraction
                    .iter()
                    .find(|(name, _)| name == scalar)
                    .map(|(_, route)| route.as_str())
            });
        // the readers refuse every other word, so a word here is one of the two
        if word == Some("direct") {
            ContractionRoute::Direct
        } else {
            ContractionRoute::Gemm
        }
    }

    /// What a lookup this card cannot satisfy is: a bad request naming the document when the caller stated it, else a fault.
    fn defect(&self, reason: &str) -> Refusal {
        blame(self.stated, "cuda_build.json", reason.into()).refusal()
    }

    /// Refuses launch bounds `(threads, blocks)` that the device cannot run: threads must be a multiple of the warp size and
    /// threads times blocks at most the resident threads of a multiprocessor.
    pub fn check_launch_bounds(&self, (threads, blocks): (i64, i64)) -> Result<(), Refusal> {
        let (warp, cap) = (self.warp_size, self.max_threads_per_sm);
        if threads < warp || threads % warp != 0 || blocks < 1 || threads * blocks > cap {
            return Err(bad(format!(
                "launch_bounds ({threads}, {blocks}) needs threads a multiple of {warp} and threads times blocks at most {cap}"
            )));
        }
        Ok(())
    }
}

/// The `steps_reverse` table of a card: the regime (`lanes` or `per_resident`) and the rows, each a state `width`, a step count `k`
/// and the narrowest and widest regime `lo` and `hi` the row serves (`lo` at most `hi`, both positive).
fn steps_reverse(json: &Json, file: &str) -> Result<StepsTable, Refusal> {
    let mut keys = Keys::of(json, "steps_reverse")?;
    let regime = match text(&mut keys, "regime")?.as_str() {
        "lanes" => StepsRegime::Lanes,
        "per_resident" => StepsRegime::PerResident,
        other => {
            return Err(fault(format!(
                "facts: {file}: steps_reverse regime {other} is not lanes or per_resident"
            )))
        }
    };
    let Json::Arr(rows) = keys
        .req("rows")
        .map_err(|why| fault(format!("facts: {file}: {why}")))?
    else {
        return Err(fault(format!(
            "facts: {file}: steps_reverse rows is not a list"
        )));
    };
    let mut table = StepsTable {
        regime,
        widths: vec![],
        ks: vec![],
        lo: vec![],
        hi: vec![],
    };
    for row in rows {
        let mut row = Keys::of(row, "steps_reverse rows")?;
        let mut bound = |key: &'static str| -> Result<f64, Refusal> {
            match row
                .req(key)
                .map_err(|why| fault(format!("facts: {file}: {why}")))?
            {
                Json::Num(n) => n.parse::<f64>().ok().filter(|v| *v > 0.0),
                _ => None,
            }
            .ok_or_else(|| {
                fault(format!(
                    "facts: {file}: steps_reverse {key} is not a positive number"
                ))
            })
        };
        let (lo, hi) = (bound("lo")?, bound("hi")?);
        if lo > hi {
            return Err(fault(format!(
                "facts: {file}: a steps_reverse row has lo {lo} above hi {hi}"
            )));
        }
        table
            .widths
            .push(positive(number(&mut row, "width")?, "steps_reverse width")?);
        table
            .ks
            .push(positive(number(&mut row, "k")?, "steps_reverse k")?);
        table.lo.push(lo);
        table.hi.push(hi);
        row.done()
            .map_err(|why| fault(format!("facts: {file}: {why}")))?;
    }
    keys.done()
        .map_err(|why| fault(format!("facts: {file}: {why}")))?;
    Ok(table)
}

/// The thread box choices measured on one device and scalar type.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxRow {
    /// The compute capability of the device (`sm_61`).
    pub cc: String,
    /// The device's name.
    pub device: String,
    /// The scalar type name the row is for.
    pub dtype: String,
    /// The lanes one thread serves, when the row changes them.
    pub lanes: Option<i64>,
    /// The way a dense contraction is run, when the row changes it.
    pub contraction: Option<String>,
    /// The tile of a block, when the device has one.
    pub tile: Option<BoxTile>,
}

/// The tile of one block of a thread box row: its kind and the sizes the mapping reads (`SB` block samples, `T` lanes per thread,
/// `R` rows, `EB` elements per block lane, `smem` shared bytes), named as the mapping that reads them names them.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxTile {
    /// The tile's kind.
    pub kind: String,
    /// The sizes `SB`, `T`, `R`, `EB` and `smem`, in that order.
    pub sizes: [i64; 5],
}

/// The scalar type names a thread box has numbers for.
const BOX_SCALARS: [&str; 2] = ["float64", "float32"];

/// Splits the device's `thread_box` section: the lane counts by scalar type and the rest of the section's keys, left in the keys.
fn split_thread_box<'a>(json: &'a Json) -> Result<(Json, Keys<'a>), Refusal> {
    let mut keys = Keys::of(json, "thread_box")?;
    let mut lanes = vec![];
    for scalar in BOX_SCALARS {
        let key: &'static str = scalar;
        lanes.push((
            key.to_string(),
            keys.req(key)
                .map_err(|why| fault(format!("facts: cuda_build.json: {why}")))?
                .clone(),
        ));
    }
    Ok((Json::Obj(lanes), keys))
}

/// A contraction route name: `gemm` or `direct`, else refused by name.
fn route(json: &Json, what: &str) -> Result<String, Refusal> {
    match json.str() {
        Some(name @ ("gemm" | "direct")) => Ok(name.to_string()),
        _ => Err(fault(format!(
            "facts: cuda_build.json: {what} contraction must be gemm or direct"
        ))),
    }
}

/// The `contraction` of the device's thread box: the route for each scalar type.
fn box_contraction(json: &Json) -> Result<Vec<(String, String)>, Refusal> {
    let mut keys = Keys::of(json, "thread_box contraction")?;
    let mut routes = vec![];
    for scalar in BOX_SCALARS {
        let key: &'static str = scalar;
        let value = keys
            .req(key)
            .map_err(|why| fault(format!("facts: {why}")))?;
        routes.push((scalar.to_string(), route(value, scalar)?));
    }
    keys.done().map_err(|why| fault(format!("facts: {why}")))?;
    Ok(routes)
}

/// The `rows` of the device's thread box: `cc`, `device` and `dtype` and, each optional, `lanes`, `contraction` and `tile`.
fn box_rows(json: &Json) -> Result<Vec<BoxRow>, Refusal> {
    let Json::Arr(items) = json else {
        return Err(fault(
            "facts: cuda_build.json: thread_box rows is not a list",
        ));
    };
    items
        .iter()
        .map(|item| {
            let mut keys = Keys::of(item, "thread_box rows")?;
            let row = BoxRow {
                cc: text(&mut keys, "cc")?,
                device: text(&mut keys, "device")?,
                dtype: text(&mut keys, "dtype")?,
                lanes: keys
                    .opt("lanes")
                    .map(|n| match n.int() {
                        Some(n) if (2..=BOX_LANES_MAX).contains(&n) => Ok(n),
                        _ => Err(fault("facts: cuda_build.json: a thread box row's lanes is not a lane count from 2 to 16")),
                    })
                    .transpose()?,
                contraction: keys.opt("contraction").map(|c| route(c, "a row's")).transpose()?,
                tile: keys.opt("tile").map(box_tile).transpose()?,
            };
            if !BOX_SCALARS.contains(&row.dtype.as_str()) {
                return Err(fault(format!(
                    "facts: cuda_build.json: a thread box row names the scalar type {}",
                    row.dtype
                )));
            }
            keys.done().map_err(|why| fault(format!("facts: {why}")))?;
            Ok(row)
        })
        .collect()
}

/// The `tile` of a thread box row: a kind and five positive sizes.
fn box_tile(json: &Json) -> Result<BoxTile, Refusal> {
    let mut keys = Keys::of(json, "thread_box tile")?;
    let kind = text(&mut keys, "kind")?;
    let mut sizes = [0; 5];
    for (slot, key) in sizes.iter_mut().zip(["SB", "T", "R", "EB", "smem"]) {
        *slot = positive(number(&mut keys, key)?, key)?;
    }
    keys.done().map_err(|why| fault(format!("facts: {why}")))?;
    Ok(BoxTile { kind, sizes })
}

/// The wide thresholds measured for one architecture and scalar type.
#[derive(Debug, Clone, PartialEq)]
pub struct WideRow {
    /// The compute capability the row was measured on (`sm_61`).
    pub cc: String,
    /// The scalar type name the row is for.
    pub dtype: String,
    /// The narrowest vector held sliced.
    pub slice_min: i64,
    /// The shortest sliced value from which a warp is chosen without being asked for.
    pub team_min: i64,
    /// The bytes of shared memory the output tiles of one block may take.
    pub tile_bytes: i64,
}

/// The `wide_rows` list of the device card: each row names its architecture and scalar type and states the three thresholds,
/// each positive; a scalar type other than `float32` and `float64` is refused by name.
fn wide_rows(json: &Json) -> Result<Vec<WideRow>, Refusal> {
    let Json::Arr(items) = json else {
        return Err(fault("facts: wide_rows is not a list"));
    };
    items
        .iter()
        .map(|item| {
            let mut keys = Keys::of(item, "wide_rows")?;
            let row = WideRow {
                cc: text(&mut keys, "cc")?,
                dtype: text(&mut keys, "dtype")?,
                slice_min: positive(number(&mut keys, "slice_min")?, "wide_rows slice_min")?,
                team_min: positive(number(&mut keys, "team_min")?, "wide_rows team_min")?,
                tile_bytes: positive(number(&mut keys, "tile_bytes")?, "wide_rows tile_bytes")?,
            };
            if !["float32", "float64"].contains(&row.dtype.as_str()) {
                return Err(fault(format!(
                    "facts: wide_rows names the scalar type {}",
                    row.dtype
                )));
            }
            keys.done().map_err(|why| fault(format!("facts: {why}")))?;
            Ok(row)
        })
        .collect()
}

/// A text member of a card section; a missing or non-text member is refused by name.
fn text(keys: &mut Keys, key: &'static str) -> Result<String, Refusal> {
    keys.req(key)
        .map_err(|why| fault(format!("facts: {why}")))?
        .str()
        .map(String::from)
        .ok_or_else(|| fault(format!("facts: {key} is not a text")))
}

/// The hardware facts the wide thresholds depend on.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchFeatures {
    /// FP32 lanes per FP64 lane.
    pub fp64_ratio: i64,
    /// The 32-bit registers of a multiprocessor.
    pub regs_per_sm: i64,
    /// The threads a multiprocessor holds resident.
    pub threads_per_sm: i64,
    /// Whether local memory goes through the first-level cache.
    pub local_cached: bool,
}

/// One class of device the wide thresholds were measured on.
#[derive(Debug, Clone, PartialEq)]
pub struct WideClass {
    /// The class's name.
    pub name: String,
    /// The compute capability whose hardware facts the class has.
    pub capability: i64,
    /// The thresholds, by the scalar's width in bytes.
    pub limits: Vec<(i64, WideLimits)>,
}

impl ArchFeatures {
    /// How far two hardware descriptions are apart: the log2 gaps of the rate, the registers and the resident threads, and one for
    /// local memory one caches and the other does not.
    fn distance(&self, other: &ArchFeatures) -> f64 {
        let gap = |a: i64, b: i64| (a as f64 / b as f64).log2().abs();
        gap(self.fp64_ratio, other.fp64_ratio)
            + gap(self.regs_per_sm, other.regs_per_sm)
            + gap(self.threads_per_sm, other.threads_per_sm)
            + f64::from(self.local_cached != other.local_cached)
    }
}

/// The compute capability an architecture name states (`sm_61` is 61, `sm_90a` is 90).
fn capability(arch: &str) -> Result<i64, Refusal> {
    let digits: String = arch
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    digits
        .parse()
        .map_err(|_| bad(format!("facts: {arch} is not a device architecture")))
}

impl DeviceCard {
    /// The hardware facts of an architecture: its own row, else the row of the nearest capability (the lower one on a tie).
    fn features_of(&self, arch: &str) -> Result<&ArchFeatures, Refusal> {
        let wanted = capability(arch)?;
        self.arch_features
            .iter()
            .min_by_key(|(known, _)| ((known - wanted).abs(), *known))
            .map(|(_, features)| features)
            .ok_or_else(|| self.defect("arch_features holds no architecture"))
    }

    /// The value of the row whose architecture has hardware nearest the architecture's among `rows` (the first on a tie), or none
    /// when there is no row.
    pub fn nearest_row<'a, T>(
        &self,
        arch: &str,
        rows: &'a [(String, T)],
    ) -> Result<Option<&'a T>, Refusal> {
        let want = self.features_of(arch)?;
        let mut best: Option<(f64, &'a T)> = None;
        for (cc, value) in rows {
            let distance = want.distance(self.features_of(cc)?);
            if best.is_none_or(|(held, _)| distance < held) {
                best = Some((distance, value));
            }
        }
        Ok(best.map(|(_, value)| value))
    }

    /// The measured class whose hardware is nearest the architecture's (the lower name on a tie).
    pub fn wide_class(&self, arch: &str) -> Result<&WideClass, Refusal> {
        let want = self.features_of(arch)?;
        let own = |class: &WideClass| {
            self.arch_features
                .iter()
                .find(|(known, _)| *known == class.capability)
                .map(|(_, features)| want.distance(features))
                .ok_or_else(|| {
                    self.defect(&format!(
                        "wide class {} has no row in arch_features",
                        class.name
                    ))
                })
        };
        let mut best: Option<(f64, &WideClass)> = None;
        for class in &self.wide_classes {
            let distance = own(class)?;
            let closer = best.is_none_or(|(held, kept)| {
                distance
                    .total_cmp(&held)
                    .then(class.name.cmp(&kept.name))
                    .is_lt()
            });
            if closer {
                best = Some((distance, class));
            }
        }
        best.map(|(_, class)| class)
            .ok_or_else(|| self.defect("wide_classes holds no class"))
    }

    /// The numbers the steps of the lowering decide on when the build has a device target: the card's own limits and the wide
    /// thresholds of the class nearest the architecture.
    pub fn deciding(&self, arch: &str) -> Result<BuildCard, Refusal> {
        let fast = self.features_of(arch)?.fp64_ratio <= self.linalg_fast_fp64;
        Ok(BuildCard {
            lazy_leaf_width: self.lazy_leaf_width,
            rolled_fold_min: self.rolled_fold_min,
            lean_width: self.lean_width,
            uniform_run: self.uniform_run,
            run_max: self.run_max,
            block_run: self.block_run,
            tile_min: self.tile_min,
            tile_max: self.tile_max,
            linalg_max_dim: self.linalg_max_dim,
            linalg_team_max_dim: self.linalg_team_max_dim,
            warp_entries: self
                .linalg_warp_entries
                .iter()
                .filter(|row| row.scalar == "float32" || fast)
                .cloned()
                .collect(),
            cells_row: self.cells_row,
            scatter_max_chunks: self.scatter_max_chunks,
            scatter_threads: self.scatter_threads,
            copy_threads: self.copy_threads,
            team_threads: self.team_threads,
            team_threads_large: self.team_threads_large,
            box_lanes: self.box_lanes.clone(),
            steps_reverse: None,
            device_steps_reverse: Some(self.steps_reverse.clone()),
            host_wide: self.wide_class(arch)?.limits.clone(),
            settings: self.settings.clone(),
        })
    }
}

/// The hardware rows of the card: a list per capability of the rate, registers, resident threads and whether local memory is cached.
fn arch_features(json: &Json) -> Result<Vec<(i64, ArchFeatures)>, Refusal> {
    let bad = |what: &str| fault(format!("facts: cuda_build.json: arch_features: {what}"));
    let Json::Obj(rows) = json else {
        return Err(bad("not an object"));
    };
    rows.iter()
        .map(|(cc, row)| {
            let cc = cc
                .parse()
                .map_err(|_| bad("a capability is not a number"))?;
            match row.arr() {
                [ratio, regs, threads, Json::Bool(cached)] => Ok((
                    cc,
                    ArchFeatures {
                        fp64_ratio: ratio.int().ok_or_else(|| bad("a rate is not an integer"))?,
                        regs_per_sm: regs.int().ok_or_else(|| bad("a count is not an integer"))?,
                        threads_per_sm: threads
                            .int()
                            .ok_or_else(|| bad("a count is not an integer"))?,
                        local_cached: *cached,
                    },
                )),
                _ => Err(bad("a row is not [rate, registers, threads, cached]")),
            }
        })
        .collect()
}

/// The warp rows of the card: each `[kind, order, scalar, outputs]`, the outputs a list of names.
fn warp_entries(json: &Json) -> Result<Vec<WarpEntry>, Refusal> {
    let bad = |what: &str| {
        fault(format!(
            "facts: cuda_build.json: linalg_warp_entries: {what}"
        ))
    };
    if !matches!(json, Json::Arr(_)) {
        return Err(bad("not a list"));
    }
    json.arr()
        .iter()
        .map(|row| match row.arr() {
            [kind, n, scalar, outputs] => Ok(WarpEntry {
                kind: kind.str().ok_or_else(|| bad("a kind is not text"))?.into(),
                n: n.int().ok_or_else(|| bad("an order is not an integer"))?,
                scalar: scalar
                    .str()
                    .ok_or_else(|| bad("a scalar is not text"))?
                    .into(),
                outputs: outputs
                    .arr()
                    .iter()
                    .map(|o| {
                        o.str()
                            .map(String::from)
                            .ok_or_else(|| bad("an output is not text"))
                    })
                    .collect::<Result<_, _>>()?,
            }),
            _ => Err(bad("a row is not [kind, order, scalar, outputs]")),
        })
        .collect()
}

/// The measured wide classes of the card: each its capability and its thresholds.
fn wide_classes(json: &Json) -> Result<Vec<WideClass>, Refusal> {
    let Json::Obj(classes) = json else {
        return Err(fault(
            "facts: cuda_build.json: wide_classes is not an object",
        ));
    };
    classes
        .iter()
        .map(|(name, class)| {
            let capability = class
                .get("capability")
                .and_then(Json::int)
                .ok_or_else(|| fault(format!("facts: wide class {name} has no capability")))?;
            Ok(WideClass {
                name: name.clone(),
                capability,
                limits: host_wide(class)?,
            })
        })
        .collect()
}

/// A card number that must be positive; a card with another is refused by name.
fn positive(value: i64, key: &str) -> Result<i64, Refusal> {
    match value > 0 {
        true => Ok(value),
        false => Err(fault(format!("facts: {key} must be positive, got {value}"))),
    }
}

/// A named section of a card.
fn part(top: &mut Keys, key: &'static str, file: &str) -> Result<Json, Refusal> {
    top.req(key)
        .cloned()
        .map_err(|why| fault(format!("facts: {file}: {why}")))
}

/// Reads the text of the device build card (`cuda_build.json`).
pub fn device_build(text: &str) -> Result<DeviceCard, Refusal> {
    let file = "cuda_build.json";
    let card = card_json(text, file)?;
    let mut top = Keys::of(&card, "device build card")?;
    let (form, lanes) = (
        part(&mut top, "form", file)?,
        part(&mut top, "lanes", file)?,
    );
    let (fold, wide) = (part(&mut top, "fold", file)?, part(&mut top, "wide", file)?);
    let (boxes, rows) = (
        part(&mut top, "thread_box", file)?,
        part(&mut top, "wide_rows", file)?,
    );
    let scatter = part(&mut top, "scatter", file)?;
    let (features, classes) = (
        part(&mut top, "arch_features", file)?,
        part(&mut top, "wide_classes", file)?,
    );
    let mut scatter = Keys::of(&scatter, "scatter")?;
    let mut form = Keys::of(&form, "form")?;
    let mut lanes = Keys::of(&lanes, "lanes")?;
    let mut fold = Keys::of(&fold, "fold")?;
    let mut wide = Keys::of(&wide, "wide")?;
    let (team_threads, team_threads_large) = team_threads(&mut lanes)?;
    let (lanes_of_box, mut boxes) = split_thread_box(&boxes)?;
    let found = DeviceCard {
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
        linalg_fast_fp64: positive(number(&mut top, "linalg_fast_fp64")?, "linalg_fast_fp64")?,
        linalg_warp_entries: warp_entries(
            top.req("linalg_warp_entries")
                .map_err(|why| fault(format!("facts: {why}")))?,
        )?,
        cells_row: number(&mut top, "cells_row")?,
        copy_threads: number(&mut lanes, "copy_threads")?,
        team_threads,
        team_threads_large,
        scatter_max_chunks: number(&mut scatter, "max_chunks")?,
        scatter_threads: number(&mut scatter, "threads")?,
        arch_features: arch_features(&features)?,
        wide_classes: wide_classes(&classes)?,
        stated: false,
        fold: FoldLimits {
            k_max: number(&mut fold, "k_max")?,
            regs_base: number(&mut fold, "regs_base")?,
            regs_column: number(&mut fold, "regs_column")?,
            residency_regs: number(&mut fold, "residency_regs")?,
        },
        box_lanes: box_lanes(&lanes_of_box)?,
        box_contraction: box_contraction(boxes.req("contraction")?)?,
        box_rows: box_rows(boxes.req("rows")?)?,
        slice_min: number(&mut wide, "slice_min")?,
        team_min: number(&mut wide, "team_min")?,
        tile_bytes: number(&mut wide, "tile_bytes")?,
        steps_reverse: steps_reverse(top.req("steps_reverse")?, file)?,
        warp_size: number(&mut top, "warp_size")?,
        max_threads_per_sm: positive(
            number(&mut top, "max_threads_per_sm")?,
            "max_threads_per_sm",
        )?,
        wide_rows: wide_rows(&rows)?,
        settings: setting_names(
            top.req("settings")
                .map_err(|why| fault(format!("facts: {file}: {why}")))?,
        )?,
    };
    if found.slice_min % found.warp_size != 0 {
        return Err(fault(format!(
            "facts: {file}: wide slice_min {} is not a whole number of warps of {}",
            found.slice_min, found.warp_size
        )));
    }
    for done in [
        form.done(),
        lanes.done(),
        fold.done(),
        wide.done(),
        scatter.done(),
        boxes.done(),
        top.done(),
    ] {
        done.map_err(|why| fault(format!("facts: {file}: {why}")))?;
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    //! The wide class of a device architecture is the measured class nearest its hardware.
    use super::*;

    /// The shipped device card.
    fn shipped() -> DeviceCard {
        device_build(crate::testing::text("cuda_build.json")).unwrap()
    }

    /// Oracle: the old rule of the warp form: a float32 row always, a float64 row only where the architecture's FP64 rate is at least
    /// an eighth of its FP32 rate (sm_80 and sm_90 at 1:2; sm_61 at 1:32 and sm_89 at 1:64 are not), the default architecture being
    /// sm_61.
    #[test]
    fn a_double_warp_row_is_kept_only_where_fp64_is_fast() {
        let card = shipped();
        for (arch, scalars) in [
            ("sm_61", vec!["float32"]),
            ("sm_89", vec!["float32"]),
            ("sm_80", vec!["float32", "float64"]),
            ("sm_90", vec!["float32", "float64"]),
        ] {
            let rows = card.deciding(arch).unwrap().warp_entries;
            let got: Vec<&str> = rows.iter().map(|r| r.scalar.as_str()).collect();
            assert_eq!(got, scalars, "{arch}");
        }
        let host = host_build(crate::testing::text("build.json")).unwrap();
        assert!(host.warp_entries.is_empty());
    }

    #[test]
    fn a_malformed_warp_row_is_refused_by_name() {
        let text = crate::testing::text("cuda_build.json");
        let rows = r#"[["eigh", 32, "float32", []], ["eigh", 32, "float64", []]]"#;
        for (bad, why) in [
            (r#"[["eigh", 32, "float32"]]"#, "a row is not"),
            (
                r#"[["eigh", "32", "float32", []]]"#,
                "an order is not an integer",
            ),
            (r#"[[1, 32, "float32", []]]"#, "a kind is not text"),
            (r#"[["eigh", 32, "float32", [1]]]"#, "an output is not text"),
            (r#"{}"#, "not a list"),
        ] {
            let broken = text.replacen(rows, bad, 1);
            assert_ne!(broken, text);
            let Err(refused) = device_build(&broken) else {
                panic!("{bad} was read");
            };
            assert!(refused.reason.contains(why), "{bad}: {}", refused.reason);
            assert!(
                refused.reason.contains("linalg_warp_entries"),
                "{}",
                refused.reason
            );
        }
    }

    /// With one measured class every architecture takes it, and its thresholds replace the host's in the deciding card.
    #[test]
    fn the_only_class_serves_every_architecture() {
        let card = shipped();
        for arch in ["sm_61", "sm_80", "sm_90a", "sm_120"] {
            assert_eq!(card.wide_class(arch).unwrap().name, "sm61", "{arch}");
        }
        let deciding = card.deciding("sm_80").unwrap();
        assert_eq!(deciding.host_wide, card.wide_classes[0].limits);
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

    /// A second measured class is chosen by the hardware distance, and a tie goes to the lower name.
    #[test]
    fn the_nearest_class_by_hardware_is_chosen() {
        let mut card = shipped();
        let other = WideClass {
            name: "sm80".into(),
            capability: 80,
            limits: card.wide_classes[0].limits.clone(),
        };
        card.wide_classes.push(other);
        for (arch, name) in [
            ("sm_61", "sm61"),
            ("sm_75", "sm61"),
            ("sm_80", "sm80"),
            ("sm_90", "sm80"),
        ] {
            assert_eq!(card.wide_class(arch).unwrap().name, name, "{arch}");
        }
        assert!(card.wide_class("cuda").is_err());
    }

    /// Oracle: the old `nearest` and `features_of` rules: a tie between two classes goes to the lower name, and an architecture
    /// between two known capabilities takes the lower one.
    #[test]
    fn ties_go_to_the_lower_name_and_the_lower_capability() {
        let mut card = shipped();
        let limits = card.wide_classes[0].limits.clone();
        for name in ["b70", "a70"] {
            card.wide_classes.push(WideClass {
                name: name.into(),
                capability: 70,
                limits: limits.clone(),
            });
        }
        // sm_80 has the hardware of capability 70: both classes are at distance 0 and the lower name wins
        assert_eq!(card.wide_class("sm_80").unwrap().name, "a70");
        // 83 is three from 80 and three from 86: the lower (80, FP64 1:2) is taken, not 86 (1:64)
        assert_eq!(card.features_of("sm_83").unwrap().fp64_ratio, 2);
    }

    /// The two team sizes are read from the card, and a card whose larger team is below the smaller is refused by name.
    #[test]
    fn the_team_sizes_are_card_numbers_and_must_not_decrease() {
        let sizes = |text: &str| {
            let json = Contract::parse(text).unwrap().0;
            team_threads(&mut Keys::of(&json, "lanes").unwrap())
        };
        assert_eq!(
            sizes(r#"{"team_threads": 128, "team_threads_large": 256}"#),
            Ok((128, 256))
        );
        let why = sizes(r#"{"team_threads": 256, "team_threads_large": 128}"#).unwrap_err();
        assert!(why.reason.contains("team_threads_large 128"), "{why}");
        assert!(sizes(r#"{"team_threads": 128}"#).is_err());
        let card = shipped();
        assert_eq!((card.team_threads, card.team_threads_large), (128, 256));
    }
}
