//! The numbers the builds read that are neither the caller's settings nor the machine's facts: the choices each target makes for
//! its text (how wide a fold stays unrolled, the lane runs and tiles, the widest matrix one thread holds, the row of a reduction
//! over samples). The host's are data in `build.json` and the device's in `cuda_build.json`, beside the machine cards, loaded
//! when a request is lowered; a file that does not match its schema is refused by name.
use super::request::Keys;
use aether_writers::contract::types::{
    Build, FoldForm, FoldLimits, FormSettings, LaneSettings, WideCard, DEFAULT_CELLS_ROW,
    DEFAULT_WARPS,
};
use aether_writers::contract::{Contract, Json};
use aether_writers::refuse::{fault, Refusal};
use std::path::Path;

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
    /// The thresholds the host's wide-value placement reads, by the scalar's width in bytes.
    pub host_wide: Vec<(i64, WideLimits)>,
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
                row_scalars: self.wide_rows.clone(),
                slice_min: Some(self.slice_min),
                forced: false,
            }),
        }
    }
}

/// The parsed JSON of a card file in the facts directory.
fn card_file(dir: &Path, file: &str) -> Result<Json, Refusal> {
    let path = dir.join(file);
    let text = std::fs::read_to_string(&path)
        .map_err(|why| fault(format!("facts: {}: {why}", path.display())))?;
    Ok(Contract::parse(&text)
        .map_err(|why| fault(format!("facts: {file}: {why}")))?
        .0)
}

/// Reads `build.json` from the facts directory.
pub fn host_build(dir: &Path) -> Result<BuildCard, Refusal> {
    let card = card_file(dir, "build.json")?;
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
        cells_row: number(&mut top, "cells_row")?,
        scatter_max_chunks: number(&mut top, "scatter_max_chunks")?,
        scatter_threads: number(&mut top, "scatter_threads")?,
        copy_threads: number(&mut lanes, "copy_threads")?,
        team_threads,
        team_threads_large,
        box_lanes: box_lanes(&section(&mut top, "thread_box")?)?,
        host_wide: host_wide(&section(&mut top, "host_wide")?)?,
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
    /// The longest run of lanes the text keeps whole.
    pub run_max: i64,
    /// The block of lanes of a run.
    pub block_run: i64,
    /// The widest matrix one thread holds.
    pub linalg_max_dim: i64,
    /// The samples one row of a reduction over samples stands for.
    pub cells_row: i64,
    /// The limits of a fold.
    pub fold: FoldLimits,
    /// The lanes one thread serves in a thread box, by scalar type name.
    pub box_lanes: Vec<(String, i64)>,
    /// The narrowest vector held sliced.
    pub slice_min: i64,
    /// The shortest sliced value from which a warp is chosen without being asked for.
    pub team_min: i64,
    /// The bytes of shared memory the output tiles of one block may take.
    pub tile_bytes: i64,
    /// The lanes of a warp.
    pub warp_size: i64,
    /// The scalar names the wide card has a row of its own for.
    pub wide_rows: Vec<String>,
    /// The largest extent a team of threads works on.
    pub linalg_team_max_dim: i64,
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
        .map_err(|_| fault(format!("facts: {arch} is not a device architecture")))
}

impl DeviceCard {
    /// The hardware facts of an architecture: its own row, else the row of the nearest capability (the lower one on a tie).
    fn features_of(&self, arch: &str) -> Result<&ArchFeatures, Refusal> {
        let wanted = capability(arch)?;
        self.arch_features
            .iter()
            .min_by_key(|(known, _)| ((known - wanted).abs(), *known))
            .map(|(_, features)| features)
            .ok_or_else(|| fault("facts: cuda_build.json: no architecture has hardware facts"))
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
                    fault(format!(
                        "facts: wide class {} has no hardware row",
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
            .ok_or_else(|| fault("facts: cuda_build.json: no wide class"))
    }

    /// The numbers the steps of the lowering decide on when the build has a device target: the host's card with the device's
    /// limits and the wide thresholds of the class nearest the architecture laid over it.
    pub fn deciding(&self, host: &BuildCard, arch: &str) -> Result<BuildCard, Refusal> {
        Ok(BuildCard {
            rolled_fold_min: self.rolled_fold_min,
            linalg_max_dim: self.linalg_max_dim,
            linalg_team_max_dim: self.linalg_team_max_dim,
            copy_threads: self.copy_threads,
            team_threads: self.team_threads,
            team_threads_large: self.team_threads_large,
            cells_row: self.cells_row,
            scatter_max_chunks: self.scatter_max_chunks,
            scatter_threads: self.scatter_threads,
            host_wide: self.wide_class(arch)?.limits.clone(),
            ..host.clone()
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

/// A named section of a card.
fn part(top: &mut Keys, key: &'static str, file: &str) -> Result<Json, Refusal> {
    top.req(key)
        .cloned()
        .map_err(|why| fault(format!("facts: {file}: {why}")))
}

/// Reads `cuda_build.json` from the facts directory.
pub fn device_build(dir: &Path) -> Result<DeviceCard, Refusal> {
    let file = "cuda_build.json";
    let card = card_file(dir, file)?;
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
    let found = DeviceCard {
        lazy_leaf_width: number(&mut form, "lazy_leaf_width")?,
        rolled_fold_min: number(&mut form, "rolled_fold_min")?,
        lean_width: number(&mut form, "lean_width")?,
        run_max: number(&mut lanes, "run_max")?,
        block_run: number(&mut lanes, "block_run")?,
        linalg_max_dim: number(&mut top, "linalg_max_dim")?,
        linalg_team_max_dim: number(&mut top, "linalg_team_max_dim")?,
        cells_row: number(&mut top, "cells_row")?,
        copy_threads: number(&mut lanes, "copy_threads")?,
        team_threads,
        team_threads_large,
        scatter_max_chunks: number(&mut scatter, "max_chunks")?,
        scatter_threads: number(&mut scatter, "threads")?,
        arch_features: arch_features(&features)?,
        wide_classes: wide_classes(&classes)?,
        fold: FoldLimits {
            k_max: number(&mut fold, "k_max")?,
            regs_base: number(&mut fold, "regs_base")?,
            regs_column: number(&mut fold, "regs_column")?,
            residency_regs: number(&mut fold, "residency_regs")?,
        },
        box_lanes: box_lanes(&boxes)?,
        slice_min: number(&mut wide, "slice_min")?,
        team_min: number(&mut wide, "team_min")?,
        tile_bytes: number(&mut wide, "tile_bytes")?,
        warp_size: number(&mut top, "warp_size")?,
        wide_rows: match rows {
            Json::Arr(names) => names
                .iter()
                .map(|n| n.str().map(String::from))
                .collect::<Option<_>>()
                .ok_or_else(|| fault(format!("facts: {file}: wide_rows holds a non-name")))?,
            _ => return Err(fault(format!("facts: {file}: wide_rows is not a list"))),
        },
    };
    for done in [
        form.done(),
        lanes.done(),
        fold.done(),
        wide.done(),
        scatter.done(),
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
        device_build(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/facts"))).unwrap()
    }

    /// With one measured class every architecture takes it, and its thresholds replace the host's in the deciding card.
    #[test]
    fn the_only_class_serves_every_architecture() {
        let card = shipped();
        for arch in ["sm_61", "sm_80", "sm_90a", "sm_120"] {
            assert_eq!(card.wide_class(arch).unwrap().name, "sm61", "{arch}");
        }
        let host = host_build(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/facts"))).unwrap();
        let deciding = card.deciding(&host, "sm_80").unwrap();
        assert_eq!(deciding.host_wide, card.wide_classes[0].limits);
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
