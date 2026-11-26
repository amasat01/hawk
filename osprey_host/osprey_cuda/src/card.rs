//! The device build card (`cuda_build.json`): the numbers the device build reads, parsed into a [`DeviceCard`] and answered to the
//! plan through [`DeviceCards`]. The same schema rules as every card: a document that does not match is refused by name.
use crate::image::{command_of_request, form_of};
use aether_writers::contract::types::{
    Build, FoldForm, FoldLimits, FormSettings, LaneSettings, Scalar, Target, WideCard,
    DEFAULT_CELLS_ROW, DEFAULT_WARPS,
};
use aether_writers::contract::{Contract, Json};
use aether_writers::refuse::{bad, fault, Refusal};
use osprey_lowering::facts::{blame, CardSet, FusionRow};
use osprey_lowering::lower::card::{
    box_lanes, card_json, defect, number, positive, setting_names, steps_reverse, team_threads,
    text, wide_thresholds, BuildNumbers, ContractionRoute, TargetCards, WarpForm, WideLimits,
    BOX_LANES_MAX,
};
use osprey_lowering::lower::image::ImageCommand;
use osprey_lowering::lower::request::{KernelRequest, Keys};
use osprey_lowering::lower::unit_meta::{DispatchCosts, StepsPair, StepsTable};
use raptor_wings::team::WarpEntry;

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
    /// The most threads one block of a launch holds.
    pub max_threads_per_block: i64,
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

    /// Refuses launch bounds `(threads, blocks)` that the device cannot run: threads must be a multiple of the warp size and at most
    /// a block holds, and threads times blocks at most the resident threads of a multiprocessor.
    pub fn check_launch_bounds(&self, (threads, blocks): (i64, i64)) -> Result<(), Refusal> {
        let (warp, cap) = (self.warp_size, self.max_threads_per_sm);
        let block = self.max_threads_per_block;
        if threads < warp
            || threads % warp != 0
            || threads > block
            || blocks < 1
            || threads * blocks > cap
        {
            return Err(bad(format!(
                "launch_bounds ({threads}, {blocks}) needs threads a multiple of {warp} and at most {block}, and threads times blocks at most {cap}"
            )));
        }
        Ok(())
    }

    /// Refuses a team size of `team_threads` threads (0 leaves it to the card) or a block of `warps` warps that the device cannot
    /// run: a block holds at most `max_threads_per_block` threads.
    pub fn check_block_settings(&self, warps: i64, team_threads: i64) -> Result<(), Refusal> {
        let (warp, block) = (self.warp_size, self.max_threads_per_block);
        if !(1..=block / warp).contains(&warps) {
            return Err(bad(format!(
                "warps {warps} is outside 1 to {}: a block holds at most {block} threads of {warp} lanes a warp",
                block / warp
            )));
        }
        if !(0..=block).contains(&team_threads) {
            return Err(bad(format!(
                "team_threads {team_threads} is outside 0 to {block}: a block holds at most {block} threads"
            )));
        }
        Ok(())
    }
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
    pub fn deciding(&self, arch: &str) -> Result<BuildNumbers, Refusal> {
        let fast = self.features_of(arch)?.fp64_ratio <= self.linalg_fast_fp64;
        Ok(BuildNumbers {
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
            warp_size: Some(self.warp_size),
            cells_row: self.cells_row,
            scatter_max_chunks: self.scatter_max_chunks,
            scatter_threads: self.scatter_threads,
            copy_threads: self.copy_threads,
            team_threads: self.team_threads,
            team_threads_large: self.team_threads_large,
            group_scratch_plane_bytes: None,
            box_lanes: self.box_lanes.clone(),
            steps_reverse: None,
            device_steps_reverse: Some(self.steps_reverse.clone()),
            wide: self.wide_class(arch)?.limits.clone(),
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
                limits: wide_thresholds(class)?,
            })
        })
        .collect()
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
        box_lanes: box_lanes(&lanes_of_box, "cuda_build.json")?,
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
        max_threads_per_block: positive(
            number(&mut top, "max_threads_per_block")?,
            "max_threads_per_block",
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

/// The cards of the device target for one request: the parsed card, and what it states for the architecture the request's device
/// facts name.
pub struct DeviceCards {
    /// The parsed card.
    card: DeviceCard,
    /// The architecture of the request's device facts answer.
    arch: String,
    /// The numbers the steps decide on at that architecture.
    deciding: BuildNumbers,
    /// The name of the wide class nearest that architecture's hardware.
    wide_class: String,
    /// The command that makes the image, when the request names an image format.
    image: Option<ImageCommand>,
}

/// The architecture the device facts answer of `request` states.
fn arch_of(request: &KernelRequest) -> Result<String, Refusal> {
    let machine = Contract::parse(request.facts_for(Target::Device)?)?.0;
    machine
        .get("device_arch")
        .and_then(Json::str)
        .map(String::from)
        .ok_or_else(|| bad("the device facts answer names no architecture"))
}

impl DeviceCards {
    /// The cards of `request` from the documents `docs`: the card in effect, the launch bounds and block settings the request states
    /// checked against it, and the numbers at the architecture of the request's device facts.
    ///
    /// # Errors
    /// Refuses a card that does not match its schema, a setting the card does not allow, and facts that name no architecture.
    pub fn of_request(docs: &CardSet, request: &KernelRequest) -> Result<DeviceCards, Refusal> {
        let card = device_build_of(docs)?;
        if let Some(bounds) = request.settings.launch_bounds {
            card.check_launch_bounds(bounds)?;
        }
        card.check_block_settings(request.settings.warps, request.settings.team_threads)?;
        let arch = arch_of(request)?;
        let deciding = card.deciding(&arch)?;
        let wide_class = card.wide_class(&arch)?.name.clone();
        let image = form_of(request)?
            .map(|form| command_of_request(request, form))
            .transpose()?;
        Ok(DeviceCards {
            card,
            arch,
            deciding,
            wide_class,
            image,
        })
    }
}

impl TargetCards for DeviceCards {
    fn numbers(&self) -> Build {
        self.card.build()
    }
    fn box_lanes(&self) -> &[(String, i64)] {
        &self.card.box_lanes
    }
    fn settings(&self) -> &[String] {
        &self.card.settings
    }
    fn steps_reverse(&self) -> StepsPair {
        StepsPair {
            host: None,
            device: Some(self.card.steps_reverse.clone()),
        }
    }
    fn deciding(&self) -> &BuildNumbers {
        &self.deciding
    }
    fn wide_class(&self) -> Option<&str> {
        Some(&self.wide_class)
    }
    fn contraction_route(&self, scalar: Scalar) -> Option<ContractionRoute> {
        let name = match scalar {
            Scalar::Float32 => "float32",
            _ => "float64",
        };
        Some(self.card.contraction_route(&self.arch, name))
    }
    fn warp_form(&self, scalar: Scalar) -> Option<WarpForm> {
        let name = match scalar {
            Scalar::Float32 => "float32",
            _ => "float64",
        };
        Some(WarpForm {
            card_row: self.card.wide_rows.iter().any(|row| row.dtype == name),
            warp_size: self.card.warp_size,
            slice_min: self.card.slice_min,
            team_min: self.card.team_min,
            tile_bytes: self.card.tile_bytes,
        })
    }
    fn image(&self) -> Option<&ImageCommand> {
        self.image.as_ref()
    }
    fn fusion_row(&self, request: &KernelRequest, docs: &CardSet) -> Result<FusionRow, Refusal> {
        crate::fusion::device_row(&self.arch, request.scalar, docs)
    }
    fn dispatch_costs(
        &self,
        request: &KernelRequest,
        docs: &CardSet,
    ) -> Result<Option<DispatchCosts>, Refusal> {
        let row = crate::fusion::device_row(&self.arch, request.scalar, docs)?;
        Ok(Some(DispatchCosts {
            launch_us: row.launch_us,
            plane_us_per_byte: row.plane_us_per_byte,
            local_over_plane: row.local_over_plane,
        }))
    }
}

#[cfg(test)]
mod tests {
    //! The wide class of a device architecture is the measured class nearest its hardware.
    use super::*;
    use crate::device_state::Stated;
    use osprey_lowering::facts::answer;

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
        assert_eq!(deciding.wide, card.wide_classes[0].limits);
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

    /// A request with no body of its own, the caller's settings all at their defaults.
    fn plain_request() -> KernelRequest {
        KernelRequest::from_json(
            r#"{"v": 1, "name": "k", "scalar": "Float64", "targets": ["Host"], "kind": {"guard": "Default", "active_set": false, "sink": "Plain"}, "body": {"Plain": {"name": "k", "sinks": [0], "declared": [], "exempt": []}}, "facts": {"Host": "{}"}}"#,
        )
        .unwrap()
    }

    #[test]
    fn a_device_request_names_one_of_three_image_forms_or_is_refused() {
        use osprey_lowering::backend::Backend;
        let mut request = device_request("{}".into());
        for form in ["Ptx", "Cubin", "Fatbin"] {
            request.image_format = Some(form.into());
            assert_eq!(
                crate::Cuda.artifact_form(&request).unwrap().as_deref(),
                Some(form)
            );
        }
        request.image_format = Some("Elf".into());
        let why = crate::Cuda.artifact_form(&request).unwrap_err();
        assert!(why.reason.contains("member Elf of image_format"));
        request.image_format = None;
        let why = crate::Cuda.artifact_form(&request).unwrap_err();
        assert!(why.reason.contains("a device target needs one"));
    }

    /// A request for the device whose facts answer states `device_facts`.
    fn device_request(device_facts: String) -> KernelRequest {
        KernelRequest {
            targets: vec![Target::Device],
            facts: vec![(Target::Device, device_facts)],
            ..plain_request()
        }
    }

    /// The device facts answer for a stated architecture (none: the card's default), under the cards `stated`.
    fn facts_of_arch(arch: Option<&str>, stated: &[(String, String)]) -> String {
        let machine = Stated {
            device_arch: arch.map(String::from),
            ..Stated::default()
        };
        let compiler =
            "/usr/local/cuda/bin/nvcc || Cuda compilation tools, release 12.6, V12.6.77 || 0";
        answer(
            &crate::testing::shipped(),
            &crate::Cuda,
            compiler,
            stated,
            &(),
            &machine,
        )
        .unwrap()
    }

    /// The device build card the cards state, with `old` of the shipped text replaced by `new`.
    fn device_card_with(old: &str, new: &str) -> DeviceCard {
        let text =
            std::fs::read_to_string(crate::testing::shipped().join("cuda_build.json")).unwrap();
        assert!(text.contains(old), "the shipped card holds {old}");
        let stated = [("cuda_build.json".to_string(), text.replacen(old, new, 1))];
        let docs = CardSet::new(
            &crate::testing::shipped(),
            &stated,
            crate::testing::ALL_CARDS,
        );
        device_build(&docs.text("cuda_build.json").unwrap()).unwrap()
    }

    /// Oracle: a launch needs threads a multiple of the warp size (32) and threads times blocks at most the resident threads of
    /// a multiprocessor, which the shipped card states as 2048; the same bounds are refused once the card states 1024.
    #[test]
    fn the_resident_threads_of_the_device_card_decide_which_launch_bounds_are_refused() {
        let shipped_card = device_card_with("\"warp_size\": 32", "\"warp_size\": 32");
        let small = device_card_with(
            "\"max_threads_per_sm\": 2048",
            "\"max_threads_per_sm\": 1024",
        );
        assert!(shipped_card.check_launch_bounds((128, 16)).is_ok());
        assert!(small.check_launch_bounds((128, 16)).is_err());
        assert!(small.check_launch_bounds((128, 8)).is_ok());
        for bad_bounds in [(100, 1), (16, 1), (128, 17), (2048, 1)] {
            let why = shipped_card.check_launch_bounds(bad_bounds).unwrap_err();
            assert!(why.reason.contains("launch_bounds"), "{}", why.reason);
        }
    }

    /// Oracle: a block holds at most the threads the shipped card states (1024) of 32 lanes a warp, so 32 warps are the most and a
    /// team of 2048 threads is refused; the same card stating 512 refuses 17 warps.
    #[test]
    fn the_threads_of_a_block_of_the_device_card_decide_which_block_settings_are_refused() {
        let shipped_card = device_card_with("\"warp_size\": 32", "\"warp_size\": 32");
        assert!(shipped_card.check_block_settings(32, 1024).is_ok());
        assert!(shipped_card.check_block_settings(0, 0).is_err());
        assert!(shipped_card.check_block_settings(33, 0).is_err());
        assert!(shipped_card.check_block_settings(4, 2048).is_err());
        let small = device_card_with(
            "\"max_threads_per_block\": 1024",
            "\"max_threads_per_block\": 512",
        );
        assert!(small.check_block_settings(16, 512).is_ok());
        let why = small.check_block_settings(17, 0).unwrap_err();
        assert!(why.reason.contains("warps 17"), "{}", why.reason);
    }

    /// Oracle: a card without the matrix dimension a thread holds is refused by the name of the number, never read as zero.
    #[test]
    fn a_device_card_without_the_linalg_dimension_is_refused_by_name() {
        let text =
            std::fs::read_to_string(crate::testing::shipped().join("cuda_build.json")).unwrap();
        assert!(text.contains("\"linalg_max_dim\": 32,"));
        let why = device_build(&text.replacen("\"linalg_max_dim\": 32,", "", 1)).unwrap_err();
        assert!(why.reason.contains("linalg_max_dim"), "{}", why.reason);
    }

    /// Oracle: the old tests of the class choice (`test_build_cards.py`): with a class measured on capability 70 beside the shipped
    /// one, sm_61 is its own class, sm_75 (FP64 1:32, 1024 threads) is one step from Pascal and sm_80 (1:2, 2048 threads) is nearest
    /// the class of capability 70.
    #[test]
    fn the_answer_names_the_wide_class_nearest_the_architecture_of_the_device_facts() {
        let stated = two_class_cards();
        let docs = CardSet::new(
            &crate::testing::shipped(),
            &stated,
            crate::testing::ALL_CARDS,
        );
        for (arch, class) in [("sm_61", "sm61"), ("sm_75", "sm61"), ("sm_80", "fake70")] {
            let request = device_request(facts_of_arch(Some(arch), &stated));
            let cards = DeviceCards::of_request(&docs, &request).unwrap();
            assert_eq!(cards.wide_class(), Some(class), "{arch}");
        }
    }

    /// Oracle: the old device card's `default_arch` is `sm_61` (a fact of the card the library applies when the caller states none),
    /// so the class of the default is the shipped one.
    #[test]
    fn a_device_request_that_states_no_architecture_takes_the_class_of_the_cards_default() {
        let stated = two_class_cards();
        let request = device_request(facts_of_arch(None, &stated));
        let cards = DeviceCards::of_request(
            &CardSet::new(
                &crate::testing::shipped(),
                &stated,
                crate::testing::ALL_CARDS,
            ),
            &request,
        )
        .unwrap();
        assert_eq!(cards.wide_class(), Some("sm61"));
    }

    #[test]
    fn a_device_facts_answer_naming_an_unknown_architecture_is_refused_by_name() {
        let request = device_request(r#"{"device_arch": "cuda"}"#.into());
        let Err(why) = DeviceCards::of_request(
            &CardSet::new(&crate::testing::shipped(), &[], crate::testing::ALL_CARDS),
            &request,
        ) else {
            panic!("an architecture that is not one was taken");
        };
        assert!(
            why.reason.contains("is not a device architecture"),
            "{}",
            why.reason
        );
    }

    /// The cards the shipped files give, plus a second wide class measured on capability 70 (a copy of the shipped class's
    /// thresholds), as the old tests of the class choice stated it: `sm_61` and `sm_75` are nearest the shipped class, `sm_80` is
    /// nearest the class of capability 70.
    fn two_class_cards() -> Vec<(String, String)> {
        let text =
            std::fs::read_to_string(crate::testing::shipped().join("cuda_build.json")).unwrap();
        let first = "\"wide_classes\": {";
        assert!(text.contains(first), "the shipped card holds {first}");
        let second = format!(
            "{first}\"fake70\": {{\"capability\": 70, \"runtime_min\": {{\"read\": {{\"4\": 16, \"8\": 2}}, \"slot_read\": {{\"4\": 16, \"8\": 4}}, \"rotate\": {{\"4\": 16, \"8\": 0}}, \"write\": {{\"4\": 96, \"8\": 8}}, \"accum\": {{\"4\": 4096, \"8\": 4096}}}}, \"wide_min\": {{\"4\": 128, \"8\": 64}}}},"
        );
        vec![("cuda_build.json".into(), text.replacen(first, &second, 1))]
    }
}
