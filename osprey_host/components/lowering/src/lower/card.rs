//! The numbers the builds read that are neither the caller's settings nor the machine's facts: the choices a target makes for its
//! text (how wide a fold stays unrolled, the lane runs and tiles, the widest matrix one thread holds, the row of a reduction over
//! samples). A backend reads its own data documents into a [`TargetCards`]; this module holds the record the steps decide on, the
//! trait that carries it, and the helpers every card reader shares. A document that does not match its schema is refused by name.
use super::image::ImageCommand;
use super::request::{KernelRequest, Keys};
use super::unit_meta::{DispatchCosts, StepsPair, StepsRegime, StepsTable};
use crate::facts::{CardSet, FusionRow};
use aether_writers::contract::types::{Build, Scalar};
use aether_writers::contract::{Contract, Json};
use aether_writers::refuse::{fault, Refusal};
use raptor_wings::team::WarpEntry;

/// The rows and lanes of the warp form a card states for a scalar type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarpForm {
    /// Whether the card has a row of its own for the type, which the warp form does not serve.
    pub card_row: bool,
    /// The lanes of a warp.
    pub warp_size: i64,
    /// The narrowest vector the form slices.
    pub slice_min: i64,
    /// The narrowest vector a team serves.
    pub team_min: i64,
    /// The bytes of the tile a block stages its vector outputs through.
    pub tile_bytes: i64,
}

/// What a backend answers about the cards of the target it serves, for one request: the numbers the plan asks. A question a kind
/// has no notion of is answered with `None` or an empty list.
pub trait TargetCards {
    /// The build numbers of the target, before the caller's settings are laid over them.
    fn numbers(&self) -> Build;
    /// The lanes one thread serves in a thread box, by scalar type name.
    fn box_lanes(&self) -> &[(String, i64)];
    /// The build settings the target declares, by the names a request states them under.
    fn settings(&self) -> &[String];
    /// The measured rows of the `steps_reverse` choice; the key of a kind the target is not is empty.
    fn steps_reverse(&self) -> StepsPair;
    /// The numbers the steps of the lowering decide on, for the request's target.
    fn deciding(&self) -> &BuildNumbers;
    /// The name of the wide class the placement of wide values reads, when the target has classes.
    fn wide_class(&self) -> Option<&str>;
    /// How a dense contraction of `scalar` runs, when the card says; none leaves it to the plan.
    fn contraction_route(&self, scalar: Scalar) -> Option<ContractionRoute>;
    /// The warp form of the target for `scalar`; none for a target that has no such form.
    fn warp_form(&self, scalar: Scalar) -> Option<WarpForm>;
    /// The command that makes the image of the request's target, when it has one.
    fn image(&self) -> Option<&ImageCommand>;
    /// The fusion row of the target for the request, from the documents `docs`.
    ///
    /// # Errors
    /// Refuses a document that does not state the row it needs.
    fn fusion_row(&self, request: &KernelRequest, docs: &CardSet) -> Result<FusionRow, Refusal>;
    /// What a dispatch of the target costs for the request; none for a target that launches nothing.
    ///
    /// # Errors
    /// Refuses a document that does not state the costs.
    fn dispatch_costs(
        &self,
        request: &KernelRequest,
        docs: &CardSet,
    ) -> Result<Option<DispatchCosts>, Refusal>;
}

/// The numbers the steps decide on.
#[derive(Debug, Clone, PartialEq)]
pub struct BuildNumbers {
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
    /// The lanes of a warp, for a build that has a device target; none for the host's own card.
    pub warp_size: Option<i64>,
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
    pub wide: Vec<(i64, WideLimits)>,
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

/// The most lanes a card may state for a thread box: a number of the text writer, which unrolls a box's lanes one by one, and of the
/// card's schema, not a property of any device. Not a machine number: the card states the lanes a device's box takes, below it.
pub const BOX_LANES_MAX: i64 = 16;

impl BuildNumbers {
    /// The wide-value thresholds for a scalar of `real_bytes` bytes.
    pub fn wide_limits(&self, real_bytes: i64) -> Option<&WideLimits> {
        let found = self.wide.iter().find(|(width, _)| *width == real_bytes);
        found.map(|(_, limits)| limits)
    }
}

/// The wide-value thresholds of the card, one entry per scalar width.
pub fn wide_thresholds(json: &Json) -> Result<Vec<(i64, WideLimits)>, Refusal> {
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

/// The integer under a required key of the card, refused by name when it is missing or not an integer.
pub fn number(keys: &mut Keys, key: &'static str) -> Result<i64, Refusal> {
    keys.req(key)
        .map_err(|why| fault(format!("facts: {why}")))?
        .int()
        .ok_or_else(|| fault(format!("facts: {key} is not an integer")))
}

/// The thread box entries of the card: each a whole number of lanes from 2 up to the most a box holds.
pub fn box_lanes(json: &Json, file: &str) -> Result<Vec<(String, i64)>, Refusal> {
    let Json::Obj(entries) = json else {
        return Err(fault(format!("facts: {file}: thread_box is not an object")));
    };
    entries
        .iter()
        .map(|(name, lanes)| match lanes.int() {
            Some(n) if (2..=BOX_LANES_MAX).contains(&n) => Ok((name.clone(), n)),
            _ => Err(fault(format!(
                "facts: {file}: thread_box {name} is not a lane count from 2 to {BOX_LANES_MAX}"
            ))),
        })
        .collect()
}
/// The setting names of a card's `settings` list: a list of texts.
pub fn setting_names(json: &Json) -> Result<Vec<String>, Refusal> {
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
pub fn team_threads(lanes: &mut Keys) -> Result<(i64, i64), Refusal> {
    let small = number(lanes, "team_threads")?;
    let large = number(lanes, "team_threads_large")?;
    match small > 0 && large >= small {
        true => Ok((small, large)),
        false => Err(fault(format!(
            "facts: team_threads {small} and team_threads_large {large} must be positive and not decreasing"
        ))),
    }
}

/// The parsed JSON of the text of a card document; `file` names it in a refusal.
pub fn card_json(text: &str, file: &str) -> Result<Json, Refusal> {
    Ok(Contract::parse(text)
        .map_err(|why| fault(format!("facts: {file}: {why}")))?
        .0)
}

/// The refusal for a defect of the document `file` found while reading it from `docs`: the caller's bad request when the caller
/// stated the document, a fault of the library when the document ships.
pub fn defect(docs: &CardSet, file: &str, why: Refusal) -> Refusal {
    docs.defect(file, why.reason).refusal()
}

/// How a dense contraction of one scalar type runs on a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractionRoute {
    /// The library states the matrix products beside the kernel, for a device plan to run through the vendor's library.
    Gemm,
    /// The unit states none and runs its own loop everywhere.
    Direct,
}

impl ContractionRoute {
    /// The route's word in an answer.
    pub fn word(self) -> &'static str {
        match self {
            ContractionRoute::Gemm => "gemm",
            ContractionRoute::Direct => "direct",
        }
    }
}

/// The `steps_reverse` table of a card: the regime (`lanes` or `per_resident`) and the rows, each a state `width`, a step count `k`
/// and the narrowest and widest regime `lo` and `hi` the row serves (`lo` at most `hi`, both positive).
pub fn steps_reverse(json: &Json, file: &str) -> Result<StepsTable, Refusal> {
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

/// A text member of a card section; a missing or non-text member is refused by name.
pub fn text(keys: &mut Keys, key: &'static str) -> Result<String, Refusal> {
    keys.req(key)
        .map_err(|why| fault(format!("facts: {why}")))?
        .str()
        .map(String::from)
        .ok_or_else(|| fault(format!("facts: {key} is not a text")))
}

/// A card number that must be positive; a card with another is refused by name.
pub fn positive(value: i64, key: &str) -> Result<i64, Refusal> {
    match value > 0 {
        true => Ok(value),
        false => Err(fault(format!("facts: {key} must be positive, got {value}"))),
    }
}
