//! The typed meta of a unit: the records of the runtime's door headers, member for member. A count the header carries beside a list
//! (`n_slots`, `rank`, `n_axes`, ...) is the list's length here; a member that may be absent is an `Option` or an empty list; a
//! closed set is an enum whose discriminants are the header's values. The same declaration gives the answer's JSON, so the records
//! cross the wire as they are, and the C source of the meta is written from them (see `meta_c`). The C source names the members it
//! writes one by one, so a member the header has no counterpart of (a fact the answer states for its reader) is documented as such
//! and stays out of the C source and of its header comparison.
use super::request::{int_of, text_of, wrong, Keys};
use aether_writers::contract::Json;
use aether_writers::refuse::{unknown, Refusal};

/// A value of a record: how it is written as JSON and read back. `None` stands for a key that is missing or null.
pub trait Wire: Sized {
    /// The value as JSON.
    fn put(&self) -> Json;
    /// The value read back from the member `key`; a type that cannot be absent refuses a missing member.
    fn get(json: Option<&Json>, key: &str) -> Result<Self, Refusal>;
}

/// The present value of a member.
fn present<'a>(json: Option<&'a Json>, key: &str) -> Result<&'a Json, Refusal> {
    json.ok_or_else(|| wrong(key, "missing"))
}

/// A whole number of a type, refused when it does not fit.
fn fits<T: TryFrom<i64>>(json: Option<&Json>, key: &str) -> Result<T, Refusal> {
    T::try_from(int_of(present(json, key)?, key)?).map_err(|_| wrong(key, "out of range"))
}

impl Wire for i64 {
    fn put(&self) -> Json {
        Json::Num(self.to_string())
    }
    fn get(json: Option<&Json>, key: &str) -> Result<Self, Refusal> {
        fits(json, key)
    }
}
impl Wire for u32 {
    fn put(&self) -> Json {
        Json::Num(self.to_string())
    }
    fn get(json: Option<&Json>, key: &str) -> Result<Self, Refusal> {
        fits(json, key)
    }
}
impl Wire for usize {
    fn put(&self) -> Json {
        Json::Num(self.to_string())
    }
    fn get(json: Option<&Json>, key: &str) -> Result<Self, Refusal> {
        fits(json, key)
    }
}
impl Wire for f64 {
    fn put(&self) -> Json {
        Json::Num(self.to_string())
    }
    fn get(json: Option<&Json>, key: &str) -> Result<Self, Refusal> {
        match present(json, key)? {
            Json::Num(n) => n.parse().map_err(|_| wrong(key, "not a number")),
            _ => Err(wrong(key, "not a number")),
        }
    }
}
impl Wire for bool {
    fn put(&self) -> Json {
        Json::Bool(*self)
    }
    fn get(json: Option<&Json>, key: &str) -> Result<Self, Refusal> {
        match present(json, key)? {
            Json::Bool(flag) => Ok(*flag),
            _ => Err(wrong(key, "not a flag")),
        }
    }
}
impl Wire for String {
    fn put(&self) -> Json {
        Json::Str(self.clone())
    }
    fn get(json: Option<&Json>, key: &str) -> Result<Self, Refusal> {
        text_of(present(json, key)?, key)
    }
}
impl<T: Wire> Wire for Option<T> {
    fn put(&self) -> Json {
        self.as_ref().map_or(Json::Null, Wire::put)
    }
    fn get(json: Option<&Json>, key: &str) -> Result<Self, Refusal> {
        json.filter(|j| **j != Json::Null)
            .map(|j| T::get(Some(j), key))
            .transpose()
    }
}
impl<T: Wire> Wire for Vec<T> {
    fn put(&self) -> Json {
        Json::Arr(self.iter().map(Wire::put).collect())
    }
    fn get(json: Option<&Json>, key: &str) -> Result<Self, Refusal> {
        present(json, key)?
            .arr()
            .iter()
            .map(|item| T::get(Some(item), key))
            .collect()
    }
}

/// A record: the struct, its JSON both ways (a closed schema: a key nobody asked for is refused by name).
macro_rules! record {
    ($(#[$doc:meta])* pub struct $name:ident { $($(#[$fdoc:meta])* pub $field:ident: $ty:ty,)* }) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Default, PartialEq)]
        pub struct $name {
            $($(#[$fdoc])* pub $field: $ty,)*
        }
        impl Wire for $name {
            fn put(&self) -> Json {
                Json::Obj(vec![$((stringify!($field).to_string(), self.$field.put()),)*])
            }
            fn get(json: Option<&Json>, key: &str) -> Result<Self, Refusal> {
                let mut keys = Keys::of(present(json, key)?, stringify!($name))?;
                let found = $name {
                    $($field: <$ty as Wire>::get(keys.opt(stringify!($field)), stringify!($field))?,)*
                };
                keys.done()?;
                Ok(found)
            }
        }
    };
}

/// A closed set: its members with the header's value and macro name; written as JSON by member name.
macro_rules! closed {
    ($(#[$doc:meta])* pub enum $name:ident { $($(#[$vdoc:meta])* $variant:ident = $value:literal => $c:literal,)* }) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub enum $name {
            $($(#[$vdoc])* $variant = $value,)*
        }
        impl $name {
            /// Every member, in the header's order.
            pub const ALL: &'static [$name] = &[$($name::$variant,)*];
            /// The header's macro name of the member.
            pub fn c_name(self) -> &'static str {
                match self {
                    $($name::$variant => $c,)*
                }
            }
        }
        impl Default for $name {
            fn default() -> Self {
                $name::ALL[0]
            }
        }
        impl Wire for $name {
            fn put(&self) -> Json {
                Json::Str(format!("{self:?}"))
            }
            fn get(json: Option<&Json>, key: &str) -> Result<Self, Refusal> {
                let name = text_of(present(json, key)?, key)?;
                $name::ALL
                    .iter()
                    .find(|member| format!("{member:?}") == name)
                    .copied()
                    .ok_or_else(|| unknown(format!("member {name} of {key}")))
            }
        }
    };
}

closed! {
    /// How a unit is loaded and launched.
    pub enum Pattern { /** The serial entry over samples. */ Pure = 0 => "EG_PATTERN_PURE", }
}
closed! {
    /// What `Real` is on the wire.
    pub enum ScalarType {
        /** 64-bit floats. */ Float64 = 0 => "EG_SCALAR_FLOAT64",
        /** 32-bit floats. */ Float32 = 1 => "EG_SCALAR_FLOAT32",
        /** Banded accumulators. */ Banded = 2 => "EG_SCALAR_BANDED",
    }
}
closed! {
    /// The format of a device image.
    pub enum ImageFormat {
        /** No image. */ None = 0 => "EG_IMAGE_NONE",
        /** Text assembly. */ Ptx = 1 => "EG_IMAGE_PTX",
        /** A binary for one architecture. */ Cubin = 2 => "EG_IMAGE_CUBIN",
        /** Binaries for several architectures. */ Fatbin = 3 => "EG_IMAGE_FATBIN",
    }
}
closed! {
    /// What a slot holds.
    pub enum SlotRole {
        /** A plane read and written in place. */ Mutable = 0 => "EG_ROLE_MUTABLE",
        /** One value per sample. */ PerSample = 1 => "EG_ROLE_PER_SAMPLE",
        /** A vector read in one piece. */ VecIn = 2 => "EG_ROLE_VEC_IN",
        /** A table indexed at run time. */ Lookup = 3 => "EG_ROLE_LOOKUP",
        /** Several values per sample out. */ WideOut = 4 => "EG_ROLE_WIDE_OUT",
        /** One value for all samples. */ Uniform = 5 => "EG_ROLE_UNIFORM",
        /** A matrix read in one piece. */ MatIn = 6 => "EG_ROLE_MAT_IN",
        /** Accumulated over samples. */ AccumOut = 7 => "EG_ROLE_ACCUM_OUT",
        /** The mask that marks a sample finished. */ Terminated = 8 => "EG_ROLE_TERMINATED",
        /** Several values per sample in. */ WideIn = 9 => "EG_ROLE_WIDE_IN",
        /** The sample count. */ Nsamples = 10 => "EG_ROLE_NSAMPLES",
        /** A plain output plane. */ Out = 11 => "EG_ROLE_OUT",
    }
}
closed! {
    /// The element type of a slot on the wire, at its real width.
    pub enum WireDtype {
        /** The unit's scalar type. */ Real = 0 => "EG_DTYPE_REAL",
        /** Signed 64-bit. */ Int64 = 1 => "EG_DTYPE_INT64",
        /** Signed 32-bit. */ Int32 = 2 => "EG_DTYPE_INT32",
        /** Unsigned 32-bit. */ Uint32 = 3 => "EG_DTYPE_UINT32",
        /** A flag. */ Bool = 4 => "EG_DTYPE_BOOL",
        /** A partial sum of a banded unit. */ BandPartial = 5 => "EG_DTYPE_BAND_PARTIAL",
    }
}
closed! {
    /// How a unit's loads and stores relate samples.
    pub enum Access {
        /** Each sample touches its own position. */ SampleLocal = 0 => "EG_ACCESS_SAMPLE_LOCAL",
        /** A sample reads other samples' positions. */ CrossSampleRead = 1 => "EG_ACCESS_CROSS_SAMPLE_READ",
        /** A sample writes other samples' positions. */ CrossSampleWrite = 2 => "EG_ACCESS_CROSS_SAMPLE_WRITE",
    }
}
closed! {
    /// What one block of a team stage's launch stands for.
    pub enum TeamGrid {
        /** Lanes. */ Lanes = 0 => "EG_GRID_LANES",
        /** Keys of a keyed plane. */ Keys = 1 => "EG_GRID_KEYS",
        /** Lane chunks of an ordered scatter. */ Chunks = 2 => "EG_GRID_CHUNKS",
        /** The cells of the scatter's target. */ Targets = 3 => "EG_GRID_TARGETS",
    }
}
closed! {
    /// What one slice of a team plane stands for.
    pub enum PlaneUnit {
        /** One per lane. */ Lane = 0 => "EG_UNIT_LANE",
        /** One per key. */ Key = 1 => "EG_UNIT_KEY",
        /** Chunks times the cells of a target. */ Target = 2 => "EG_UNIT_TARGET",
        /** One for the whole launch. */ Once = 3 => "EG_UNIT_ONCE",
    }
}
closed! {
    /// The shape of a plane the runtime makes.
    pub enum PlaneLayout {
        /** `(per_lane * units,)`. */ Flat = 0 => "EG_LAYOUT_FLAT",
        /** `(per_lane, units)`. */ PlaneMajor = 1 => "EG_LAYOUT_PLANE_MAJOR",
    }
}
closed! {
    /// Whether a plane holds a placed wide value.
    pub enum WideLayout {
        /** Not a wide value. */ None = 0 => "EG_WIDE_NONE",
        /** The scratch of a placed wide value, tiled by warp (value 1 is retired). */ WarpTiled = 2 => "EG_WIDE_WARP_TILED",
    }
}
closed! {
    /// A dispatch group's strategy.
    pub enum DispatchStrategy {
        /** Predicated branches. */ Predicated = 0 => "EG_DISPATCH_PREDICATED",
        /** A switch. */ Switch = 1 => "EG_DISPATCH_SWITCH",
        /** A segmented split. */ Segmented = 2 => "EG_DISPATCH_SEGMENTED",
        /** Decided at run time. */ Auto = 3 => "EG_DISPATCH_AUTO",
    }
}
closed! {
    /// The batch a steps-reverse row is measured in.
    pub enum StepsRegime {
        /** The sample count. */ Lanes = 0 => "EG_REGIME_LANES",
        /** The samples per resident thread. */ PerResident = 1 => "EG_REGIME_PER_RESIDENT",
    }
}
closed! {
    /// The fold of a cell.
    pub enum ReduceOp {
        /** Sum. */ Sum = 0 => "EG_REDUCE_SUM",
        /** Maximum. */ Max = 1 => "EG_REDUCE_MAX",
        /** Minimum. */ Min = 2 => "EG_REDUCE_MIN",
        /** All, over a mask. */ All = 3 => "EG_REDUCE_ALL",
    }
}

/// The flag bits of a slot, with the header's macro names.
pub const SLOT_FLAGS: [(u32, &str); 4] = [
    (1 << 0, "EG_SLOT_PRIOR"),
    (1 << 1, "EG_SLOT_PARAM_PLANE"),
    (1 << 2, "EG_SLOT_GATHER"),
    (1 << 3, "EG_SLOT_SHARED"),
];
/// The flag of a slot the body reads before writing it: a caller must give it an initial value (the door validates the bit and does
/// not act on it).
pub const SLOT_PRIOR: u32 = 1 << 0;
/// The flag of a lookup slot that is a parameter read at index 0 only: its gradient is the sum over samples into that one cell.
pub const SLOT_PARAM_PLANE: u32 = 1 << 1;
/// The flag of a parameter plane with no lane axis.
pub const SLOT_SHARED: u32 = 1 << 3;
/// The slot flag of a matrix plane the unit reads only at lane indices.
pub const SLOT_GATHER: u32 = 1 << 2;
/// The flag bits of a unit record, with the header's macro names.
pub const UNIT_FLAGS: [(u32, &str); 1] = [(UNIT_CALLER_CARDS, "EG_UNIT_CALLER_CARDS")];
/// The flag of a unit lowered under data documents the caller stated, not the ones the library ships.
pub const UNIT_CALLER_CARDS: u32 = 1 << 0;
/// The flag bits of a branch cost record that this library states, with the header's macro names.
pub const BRANCH_FLAGS: [(u32, &str); 1] = [(BRANCH_HAS_STORES, "EG_BRANCH_HAS_STORES")];
/// The flag of a branch cost whose `stores` is stated.
pub const BRANCH_HAS_STORES: u32 = 1 << 1;
/// The flag bits of a finish record, with the header's macro names.
pub const FINISH_FLAGS: [(u32, &str); 1] = [(FINISH_ENTRY_COUNTS, "EG_FINISH_ENTRY_COUNTS")];
/// The flag of a finishing unit whose fast entries count the samples finished on entry.
pub const FINISH_ENTRY_COUNTS: u32 = 1 << 0;
/// The schema major the records are written for.
pub const SCHEMA_MAJOR: u32 = 1;
/// The name of the one function a unit library exports.
pub const LIBRARY_META_SYMBOL: &str = "eg_unit_library_meta";

record! {
    /// A unit named by its name and digest.
    pub struct UnitRef {
        /// The unit's name.
        pub name: String,
        /// The digest: 64 hex digits, the 32 bytes of the unit's canonical walk digest.
        pub digest: String,
    }
}
record! {
    /// One argument slot of a unit's entry; its position in the list is the slot's index.
    pub struct Slot {
        /// The slot's flag bits.
        pub flags: u32,
        /// What the slot holds.
        pub role: SlotRole,
        /// The element type on the wire.
        pub dtype: WireDtype,
        /// An adjoint plane sized like this lookup slot.
        pub like_slot: Option<usize>,
        /// The lane axes a gathered matrix varies along (one flag per axis); the slot is gathered when present.
        pub gather: Option<Vec<bool>>,
        /// The extents of one sample's component; the rank is its length.
        pub shape: Vec<i64>,
        /// The declared name.
        pub name: String,
        /// How the slot crosses the launch boundary: `GRefMirror`, `ScalarHandle` or `value`, from the writers' one table of roles.
        /// Stated in the answer's JSON only: the C record of the header has no such member, and `meta_c` does not write it.
        pub mirror: String,
    }
}
record! {
    /// A companion plane of a compensated keyed sum.
    pub struct Companion {
        /// The companion's slot.
        pub slot: usize,
        /// The slot it compensates.
        pub target_slot: usize,
        /// The host symbol of the entry that folds it into its target.
        pub host_fold_entry: Option<String>,
        /// The fold function's name inside the device image.
        pub device_fold_function: Option<String>,
    }
}
record! {
    /// A reduction over samples kept in rows and folded into the cell between launches.
    pub struct Cell {
        /// The fold.
        pub op: ReduceOp,
        /// Whether only a later stage reads it and the caller never declared it.
        pub internal: bool,
        /// The components of the cell.
        pub columns: i64,
        /// The row-plane slot of each column in this unit, absent when the unit binds none.
        pub slots: Option<Vec<usize>>,
        /// The slot of each column's compensation of a sum, absent when the cell has none.
        pub comp_slots: Option<Vec<usize>>,
        /// The cell's name.
        pub name: String,
    }
}
record! {
    /// A plane keyed by kept lane axes, owned by the runtime between stages.
    pub struct Keyed {
        /// The entries of one key's payload.
        pub width: i64,
        /// One label per kept axis, absent when unlabelled; the axis count is its length.
        pub axis_labels: Vec<Option<String>>,
        /// One extent per kept axis, -1 for a runtime axis.
        pub axis_extents: Vec<i64>,
        /// The extents of one key's payload.
        pub payload: Vec<i64>,
        /// The plane's name.
        pub name: String,
    }
}
record! {
    /// A scratch plane the runtime makes once per kernel and lane count.
    pub struct TeamPlane {
        /// The element type.
        pub dtype: WireDtype,
        /// What a slice stands for.
        pub unit: PlaneUnit,
        /// The plane's own shape.
        pub layout: PlaneLayout,
        /// Wide placement: the unit count and whether the plane is filled.
        pub wide: WideLayout,
        /// With unit key: the index in the body's keyed table.
        pub keyed: Option<usize>,
        /// With unit target: the slot whose cells the slices stand for.
        pub target_slot: Option<usize>,
        /// The elements of one slice.
        pub per_lane: i64,
        /// The value at the start of every run, read only when `fill_promised`.
        pub fill: i64,
        /// The plane's name.
        pub name: String,
        /// With unit target: the index in the kernel's stages of the unit whose body holds `target_slot`; none: this unit.
        pub target_stage: Option<usize>,
        /// Whether `fill` is promised at the start of every run.
        pub fill_promised: bool,
    }
}
record! {
    /// One branch's operation counts per sample, as the cost model counts them. The header's vectorisation bits are not here: their
    /// verdict comes from the host loop's own writer, so the record leaves them unstated until that writer states them.
    pub struct BranchCost {
        /// Arithmetic operations at 32 bits.
        pub arith32: f64,
        /// Arithmetic operations at 64 bits.
        pub arith64: f64,
        /// Special functions at 32 bits.
        pub special32: f64,
        /// Special functions at 64 bits.
        pub special64: f64,
        /// Plane reads at the lane index.
        pub loads: f64,
        /// Reads at a computed index.
        pub gathers: f64,
        /// The longest dependent arithmetic chain at 32 bits.
        pub chain32: f64,
        /// The longest dependent arithmetic chain at 64 bits.
        pub chain64: f64,
        /// The special functions on the 32-bit chain.
        pub chain_special32: f64,
        /// The special functions on the 64-bit chain.
        pub chain_special64: f64,
        /// The plane writes; none when no sink takes the group's value directly.
        pub stores: Option<f64>,
    }
}
record! {
    /// One dispatch group of a unit.
    pub struct Dispatch {
        /// The strategy.
        pub strategy: DispatchStrategy,
        /// The branch-mix counters of an auto group.
        pub mix_slot: Option<usize>,
        /// The per-sample class plane of a segmented group.
        pub class_slot: Option<usize>,
        /// The stale mask of a segmented group with a dynamic selector.
        pub stale_slot: Option<usize>,
        /// The run offsets of a segmented group.
        pub offsets_slot: Option<usize>,
        /// The classify unit of a segmented group with a static selector.
        pub classify: Option<UnitRef>,
        /// The group's name.
        pub name: String,
        /// The group's number of branches: the dispatch's, read from the kernel's IR.
        pub k: usize,
        /// The branch units of a segmented group in selector-value order, the unit of branch `j` running over the run of class `j`.
        pub branches: Vec<UnitRef>,
        /// The fused unit of a segmented group with a dynamic selector, run over the index set of the stale samples.
        pub sweep: Option<UnitRef>,
        /// The cost of each branch of an auto group, in branch order; empty for any other group.
        pub branch_costs: Vec<BranchCost>,
    }
}
record! {
    /// The typed walk of a unit: its slots and the planes the runtime makes for it.
    pub struct Body {
        /// How loads and stores relate samples.
        pub access: Access,
        /// The samples per row of a cell, 0 without cells.
        pub cell_row: i64,
        /// The most chunks of an ordered scatter, 0 for none.
        pub scatter_chunks: i64,
        /// The argument slots in order.
        pub slots: Vec<Slot>,
        /// The compensation planes.
        pub companions: Vec<Companion>,
        /// The cells.
        pub cells: Vec<Cell>,
        /// The keyed planes.
        pub keyed: Vec<Keyed>,
        /// The scratch planes.
        pub team_planes: Vec<TeamPlane>,
        /// The dispatch groups.
        pub dispatches: Vec<Dispatch>,
    }
}
record! {
    /// The named axes of a sink.
    pub struct PlaneAxes {
        /// The sink's slot.
        pub slot: usize,
        /// One label per axis.
        pub labels: Vec<Option<String>>,
        /// One extent per axis, -1 for a runtime one.
        pub extents: Vec<i64>,
    }
}
record! {
    /// A thread box: each thread serves `t` samples `stride` apart.
    pub struct ThreadBox {
        /// The sample axis.
        pub axis: i64,
        /// The samples per thread.
        pub t: i64,
        /// The distance between a thread's samples.
        pub stride: i64,
    }
}
record! {
    /// The launch shape of a unit.
    pub struct Launch {
        /// The device threads per block a launch must use, 0 for the runtime's choice.
        pub block: i64,
        /// The active set's map slot.
        pub active_map_slot: Option<usize>,
        /// The active set's live-lane count slot.
        pub active_count_slot: Option<usize>,
        /// The caller's index-set map slot.
        pub index_map_slot: Option<usize>,
        /// The caller's prefix count slot.
        pub index_count_slot: Option<usize>,
        /// The lanes per compaction group.
        pub group: i64,
        /// One label per lane axis; the axis count is its length.
        pub lane_labels: Vec<Option<String>>,
        /// One extent per lane axis, -1 for a runtime one.
        pub lane_extents: Vec<i64>,
        /// The named axes of the sinks.
        pub plane_axes: Vec<PlaneAxes>,
        /// The host entry's box.
        pub host_box: Option<ThreadBox>,
        /// The device function's box.
        pub device_box: Option<ThreadBox>,
    }
}
record! {
    /// The measured rows of the steps-reverse choice.
    pub struct StepsTable {
        /// The batch the rows are measured in.
        pub regime: StepsRegime,
        /// The state width of each row; the row count is its length.
        pub widths: Vec<i64>,
        /// The step count of each row.
        pub ks: Vec<i64>,
        /// The narrowest regime of each row.
        pub lo: Vec<f64>,
        /// The widest regime of each row.
        pub hi: Vec<f64>,
    }
}
record! {
    /// How a finishing unit stops a sample and steps.
    pub struct Finish {
        /// The flag bits.
        pub flags: u32,
        /// The slot of the mask it marks.
        pub mask_slot: usize,
        /// The slot of the counter of newly finished samples.
        pub counter_slot: usize,
        /// The slot of the steps word of an auto count.
        pub steps_slot: Option<usize>,
        /// The steps one launch takes, -1 for auto, never 0.
        pub steps: i64,
        /// The most steps of an auto launch, 0 for none.
        pub steps_max: Option<i64>,
        /// One step's operation count, -1 for not counted.
        pub step_ops: Option<i64>,
        /// One step's 64-bit float operations in the IR, -1 for not counted.
        pub fp64_ops_ir: Option<i64>,
        /// The host's measured rows.
        pub host_steps_reverse: Option<StepsTable>,
        /// The device's measured rows.
        pub device_steps_reverse: Option<StepsTable>,
        /// The comparisons of the stop rule, counted from the kernel's IR: 0 when it has none.
        pub n_events: i64,
        /// The persistent entry of the device text, as the device writer names it; none for a unit without one.
        pub device_persist_function: Option<String>,
        /// The contiguous-range entry of the device text, as the device writer names it; none for a unit without one.
        pub device_range_function: Option<String>,
    }
}
record! {
    /// The team a stage runs on; indices are into the body of the unit that lists the stage.
    pub struct Team {
        /// What a block stands for.
        pub grid: TeamGrid,
        /// The threads of one team.
        pub threads: i64,
        /// The warps of one team of a warp per lane.
        pub warps: i64,
        /// The lanes one block serves.
        pub lanes: i64,
        /// The blocks per key.
        pub groups: i64,
        /// With grid keys: the index in the keyed table.
        pub keyed: Option<usize>,
        /// With grid chunks or targets: the slot the scatter adds into.
        pub of_slot: Option<usize>,
        /// With grid chunks: the index in the team planes of the out-of-range flags.
        pub status_plane: Option<usize>,
        /// With grid chunks or targets: the index in the kernel's stages of the unit whose body holds `of_slot`; none: the source.
        pub of_stage: Option<usize>,
    }
}
record! {
    /// One stage of a split unit.
    pub struct Stage {
        /// The stage unit.
        pub unit: UnitRef,
        /// The cells it reads.
        pub cells_in: Vec<String>,
        /// The cells it stores.
        pub cells_out: Vec<String>,
        /// The team it runs on.
        pub team: Option<Team>,
    }
}
closed! {
    /// What a derived unit is to a runner.
    pub enum DerivedRole {
        /** The one-step taping twin. */ Tape = 0 => "EG_DERIVED_TAPE",
        /** The K-step taping twin. */ TapeFused = 1 => "EG_DERIVED_TAPE_FUSED",
        /** The per-step reverse unit. */ StepVjp = 2 => "EG_DERIVED_STEP_VJP",
        /** The K-step reverse unit. */ FusedVjp = 3 => "EG_DERIVED_FUSED_VJP",
        /** The event refine entry. */ RefineEntry = 4 => "EG_DERIVED_REFINE_ENTRY",
        /** The refine entry with its derivative in the event time. */ RefineNewton = 5 => "EG_DERIVED_REFINE_NEWTON",
        /** The reverse unit of the refine entry. */ RefineVjp = 6 => "EG_DERIVED_REFINE_VJP",
        /** The state guard. */ StateGuard = 7 => "EG_DERIVED_STATE_GUARD",
        /** The reverse unit of the state guard. */ StateGuardVjp = 8 => "EG_DERIVED_STATE_GUARD_VJP",
    }
}
closed! {
    /// What quantity a link of a derived unit refers to.
    pub enum LinkWhat {
        /** The primal's slot. */ PrimalSlot = 0 => "EG_WHAT_PRIMAL_SLOT",
        /** The start copy of the primal's time plane. */ Time0 = 1 => "EG_WHAT_TIME0",
        /** A step plane. */ Steps = 2 => "EG_WHAT_STEPS",
        /** The candidate or located time inside the stop step. */ Tau = 3 => "EG_WHAT_TAU",
        /** The length of the accepted step. */ TimeStep = 4 => "EG_WHAT_TIME_STEP",
        /** An event guard, zero at the crossing. */ Guard = 5 => "EG_WHAT_GUARD",
        /** The tape rows of a primal state. */ TapeRow = 6 => "EG_WHAT_TAPE_ROW",
        /** A primal state at the candidate time. */ Refined = 7 => "EG_WHAT_REFINED",
    }
}
closed! {
    /// In which form a derived unit holds what a link refers to.
    pub enum LinkForm {
        /** The quantity itself. */ Value = 0 => "EG_FORM_VALUE",
        /** Its derivative in the candidate time. */ Dot = 1 => "EG_FORM_DOT",
        /** The cotangent arriving at it. */ AdjointIn = 2 => "EG_FORM_ADJOINT_IN",
        /** The unit's output cotangent of it. */ AdjointOut = 3 => "EG_FORM_ADJOINT_OUT",
        /** An accumulated gradient. */ Grad = 4 => "EG_FORM_GRAD",
        /** A plane filled with ones. */ Seed = 5 => "EG_FORM_SEED",
        /** Written by the unit, read by nobody. */ Discard = 6 => "EG_FORM_DISCARD",
    }
}
record! {
    /// One taped plane: rows `row0` to `row0 + rows` of the tape hold the primal's slot `slot`.
    pub struct Taped {
        /// The slot in the primal.
        pub slot: usize,
        /// The first row.
        pub row0: i64,
        /// The rows.
        pub rows: i64,
    }
}
record! {
    /// One slot of a derived unit: what it refers to and in which form.
    pub struct Link {
        /// The slot in this unit.
        pub slot: usize,
        /// What it refers to.
        pub what: LinkWhat,
        /// The slot it is of, absent for none.
        pub of_slot: Option<usize>,
        /// Whether a gradient is summed over samples.
        pub reduced: bool,
        /// The form the unit holds it in.
        pub form: LinkForm,
    }
}
/// The flag bits of a matrix product, with the header's macro names.
pub const GEMM_FLAGS: [(u32, &str); 4] = [
    (1 << 0, "EG_GEMM_TRANS_A"),
    (1 << 1, "EG_GEMM_TRANS_B"),
    (1 << 2, "EG_GEMM_TRANS_C"),
    (1 << 3, "EG_GEMM_ACCUMULATE"),
];
/// The flag of a product whose A is stored (k, m).
pub const GEMM_TRANS_A: u32 = 1 << 0;
/// The flag of a product whose B is stored (n, k).
pub const GEMM_TRANS_B: u32 = 1 << 1;
/// The flag of a product whose C is stored through a transposed view.
pub const GEMM_TRANS_C: u32 = 1 << 2;
/// The flag of a product that adds into C.
pub const GEMM_ACCUMULATE: u32 = 1 << 3;
/// An extent that is the run's lane count over the record's `lane_fanin`.
pub const GEMM_PER_LANE: i64 = -1;
record! {
    /// One matrix product of a gemm unit over three of its slots, as stored shapes.
    pub struct GemmCall {
        /// The flag bits (`GEMM_FLAGS`).
        pub flags: u32,
        /// The slot that is A.
        pub a_slot: usize,
        /// The slot that is B.
        pub b_slot: usize,
        /// The slot that is C, a written one.
        pub c_slot: usize,
        /// The stored rows of A.
        pub a_rows: i64,
        /// The stored columns of A.
        pub a_cols: i64,
        /// The stored rows of B.
        pub b_rows: i64,
        /// The stored columns of B.
        pub b_cols: i64,
        /// The stored rows of C.
        pub c_rows: i64,
        /// The stored columns of C.
        pub c_cols: i64,
    }
}
record! {
    /// The matrix products a unit that is one dense contraction is, stated beside its kernel, which it still carries in full: a
    /// host plan runs the kernel's own loop and a device plan the products.
    pub struct Gemm {
        /// The other operand's free extent: a `GEMM_PER_LANE` extent is the run's lane count over it; 0: none.
        pub lane_fanin: i64,
        /// The products, run in order.
        pub calls: Vec<GemmCall>,
    }
}
record! {
    /// What a derived unit is to its primal.
    pub struct Derived {
        /// The unit's role.
        pub role: DerivedRole,
        /// The primal's index in the library's table.
        pub of_unit: usize,
        /// The tape slot of a taping twin.
        pub tape_slot: Option<usize>,
        /// The rows of one recorded step.
        pub tape_rows: i64,
        /// The applied steps slot.
        pub steps_slot: Option<usize>,
        /// The slot that switches taping.
        pub tape_on_slot: Option<usize>,
        /// The time plane's start copy.
        pub time0_slot: Option<usize>,
        /// The word of steps a fused unit runs.
        pub fused_steps_slot: Option<usize>,
        /// The taped planes.
        pub taped: Vec<Taped>,
        /// The slots' links.
        pub links: Vec<Link>,
    }
}
record! {
    /// One unit of the library.
    pub struct UnitMeta {
        /// The flag bits of a unit (`UNIT_FLAGS`).
        pub flags: u32,
        /// The unit's name.
        pub name: String,
        /// The digest: 64 hex digits, the 32 bytes of the unit's canonical walk digest.
        pub digest: String,
        /// How the unit is loaded.
        pub pattern: Pattern,
        /// What `Real` is.
        pub scalar_type: ScalarType,
        /// The symbol of the serial entry.
        pub host_entry: Option<String>,
        /// The symbol of the device image's bytes.
        pub device_image: Option<String>,
        /// The kernel's name inside the image.
        pub device_function: Option<String>,
        /// The image's length when the size symbol is not used: written 0.
        pub device_image_bytes: i64,
        /// The format of the image.
        pub device_image_format: ImageFormat,
        /// The stages of a split unit.
        pub stages: Vec<Stage>,
        /// The typed walk.
        pub body: Body,
        /// The launch shape.
        pub launch: Launch,
        /// The finish record.
        pub finish: Option<Finish>,
        /// The symbol of the image's byte count.
        pub device_image_size: Option<String>,
        /// What the unit is to its primal, absent for a primal unit.
        pub derived: Option<Derived>,
        /// The compiler that made the image (`nvcc 12.6`), absent without an image.
        pub device_compiler: Option<String>,
        /// The flags of the compile command, as given, absent without an image.
        pub device_flags: Option<String>,
        /// The PTX version of the image's PTX as major * 10 + minor (85 is 8.5); 0 when it holds no PTX.
        pub device_ptx_isa: i64,
        /// The matrix products the unit is, when it is one dense contraction the card routes.
        pub gemm: Option<Gemm>,
    }
}
record! {
    /// The fusion row the leading library cut a chain with.
    pub struct PeerFusion {
        /// The registers a kernel uses before any value.
        pub regs_base: i64,
        /// The register file.
        pub regs_file: i64,
        /// The cost of one more launch, in microseconds.
        pub launch_us: Option<f64>,
        /// The cost of one byte through a plane, in microseconds.
        pub plane_us_per_byte: Option<f64>,
        /// The cost of a byte of local frame over a byte of plane.
        pub local_over_plane: Option<f64>,
    }
}
record! {
    /// The numbers the steps of the lowering decided the cut on.
    pub struct PeerNumbers {
        /// The most elements a fold stays unrolled over.
        pub rolled_fold_min: i64,
        /// The widest matrix one thread holds.
        pub linalg_max_dim: i64,
        /// The largest extent a team of threads works on.
        pub linalg_team_max_dim: i64,
        /// The threads of the block of a stage that only copies operands into team scratch.
        pub copy_threads: i64,
        /// The threads of a team for a matrix of order up to `team_threads_large`'s limit.
        pub team_threads: i64,
        /// The threads of a team for a larger matrix.
        pub team_threads_large: i64,
        /// The most chunks of an ordered scatter a launch splits the lanes into.
        pub scatter_max_chunks: i64,
        /// The team of each stage of an ordered scatter.
        pub scatter_threads: i64,
        /// The samples one row of a reduction over samples stands for.
        pub cells_row: i64,
    }
}
record! {
    /// What a library that lowers the same build beside the leading one must match: the numbers the cut was decided on, the fusion
    /// row of a chain, whether the leader runs a contraction directly, the build setting keys its target declares and the names of
    /// its units in order (the cut). The caller hands it unread into the other libraries' requests.
    pub struct PeerRecord {
        /// The numbers the steps decided on.
        pub numbers: PeerNumbers,
        /// The fusion row, stated for a chain of kernels.
        pub fusion: Option<PeerFusion>,
        /// Whether a dense contraction is run directly instead of as a matrix product.
        pub direct_contraction: bool,
        /// The build setting keys the leader's target declares: a companion accepts a stated setting its own card does not name
        /// when the leader's target does.
        pub settings: Vec<String>,
        /// The names of the leader's units, in order.
        pub units: Vec<String>,
    }
}
record! {
    /// The `steps_reverse` rows of a build, each stated by the library of its target.
    pub struct StepsPair {
        /// The host's rows.
        pub host: Option<StepsTable>,
        /// The device's rows.
        pub device: Option<StepsTable>,
    }
}
record! {
    /// What a dispatch of a device target costs, as the runtime prices a launch.
    pub struct DispatchCosts {
        /// The cost of one launch, in microseconds.
        pub launch_us: Option<f64>,
        /// The cost of one byte through a plane, in microseconds.
        pub plane_us_per_byte: Option<f64>,
        /// The cost of a byte of local frame over a byte of plane.
        pub local_over_plane: Option<f64>,
    }
}
record! {
    /// The top of the tree; its units are the answer's units, in order.
    pub struct LibraryMeta {
        /// The schema major.
        pub schema_major: u32,
        /// The schema minor.
        pub schema_minor: u32,
    }
}

impl LibraryMeta {
    /// The record of the schema this library writes.
    pub fn current() -> LibraryMeta {
        LibraryMeta {
            schema_major: SCHEMA_MAJOR,
            schema_minor: 0,
        }
    }
}

/// The members of the records that are not computed yet, each with the step or writer that knows the value. A member listed here is
/// written as the header's absent value (or past the record's `struct_size`) and never as a value; the next slices shrink the list.
/// Paths: raptor = rs/ir/src, aether = rs/writers/src, lower = components/lowering/src/lower.
pub const NOT_COMPUTED: &[(&str, &str)] = &[
    ("team_plane.wide of a wide OUTPUT plane (`bf`, `y`: a slot the placement made wide, not a scratch plane)", "a design choice: the meta has no member for a wide slot; the recorded launch.wide names it (4 units on k7); the warp step's own planes (`hwd*`) also come from lower/warp.rs `slice`"),
];
