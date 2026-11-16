//! What the split of a kernel into launches decided, by name: the stages in launch order, the cells and keyed planes between them
//! and the scratch planes they make. It is a step's product only: `meta_fill` turns it into the typed meta of the units (which is
//! what the answer carries) once every unit's slots are known.
use super::unit_meta::UnitMeta;

/// A cell: a reduction over samples stored in rows, finished between two launches.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CellRecord {
    /// The cell's name.
    pub(crate) name: String,
    /// The width of the cell.
    pub(crate) columns: i64,
    /// The reduction's operation.
    pub(crate) op: String,
    /// Whether only a later stage reads it and the caller never declared it.
    pub(crate) internal: bool,
}

/// One launch of a cut kernel.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StageRecord {
    /// The stage's unit name.
    pub(crate) kernel: String,
    /// The cells it reads.
    pub(crate) cells_in: Vec<String>,
    /// The cells it stores.
    pub(crate) cells_out: Vec<String>,
    /// The team the stage runs on, when it runs on one.
    pub(crate) team: Option<TeamRecord>,
}

/// The team a stage runs on and what one block of its launch is.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TeamRecord {
    /// The threads of one team.
    pub(crate) threads: i64,
    /// The lanes one block of a warp per lane serves; one otherwise.
    pub(crate) lanes: i64,
    /// The warps of one team of a warp per lane; one otherwise.
    pub(crate) warps: i64,
    /// The keys a block of the launch stands for, when it stands for one (else it stands for a lane).
    pub(crate) keys: Option<KeysGrid>,
    /// The chunk of lanes or the cell an ordered scatter's block stands for, when it stands for one.
    pub(crate) scatter: Option<ScatterGrid>,
}

/// A launch whose blocks are the chunks of lanes or the cells of an ordered scatter.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScatterGrid {
    /// `chunks` or `targets`.
    pub(crate) grid: String,
    /// The plane the scatter adds into.
    pub(crate) target: String,
    /// The plane whose flags say a key was out of range, for the chunks.
    pub(crate) status: Option<String>,
}

/// A launch whose blocks are the keys of a block sum.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct KeysGrid {
    /// The blocks of a key.
    pub(crate) groups: i64,
    /// The keyed plane whose keys they are.
    pub(crate) keyed: KeyedRecord,
}

/// A plane keyed by some lane axes that the runtime owns between stages.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct KeyedRecord {
    /// The plane's name.
    pub(crate) name: String,
    /// The entries of one key's payload.
    pub(crate) width: i64,
    /// The kept lane axes: a label and an extent, each when it has one.
    pub(crate) axes: Vec<(Option<String>, Option<i64>)>,
    /// The shape of one key's payload.
    pub(crate) payload: Vec<i64>,
}

/// A scratch plane the runtime makes once per kernel and lane count.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PlaneRecord {
    /// The plane's name.
    pub(crate) name: String,
    /// The element type: `real` (the build's scalar type), or a plain type name.
    pub(crate) dtype: String,
    /// The elements of each lane's slice.
    pub(crate) per_lane: i64,
    /// How the plane is laid out when it is not the default (`plane_major`: one plane per value of the lane).
    pub(crate) layout: Option<String>,
    /// The keys one slice stands for, when the plane holds a value per key.
    pub(crate) keys: Option<KeyedRecord>,
    /// The plane whose cells the slices stand for, when they do.
    pub(crate) per_target: Option<String>,
    /// Whether one slice serves the whole launch.
    pub(crate) once: bool,
    /// Whether the plane is the scratch of a wide value the placement made (tiled by warp, written before read, never filled).
    pub(crate) wide: bool,
    /// What the plane holds at the start of every run, when something is promised.
    pub(crate) fill: Option<i64>,
}

/// What a kernel cut into launches (or with wide values placed in scratch planes) tells its runtime.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct StageBlock {
    /// The cells the stages store, in first-stored order.
    pub(crate) cells: Vec<CellRecord>,
    /// The stages in launch order.
    pub(crate) stages: Vec<StageRecord>,
    /// The scratch planes of the stages.
    pub(crate) team_planes: Vec<PlaneRecord>,
    /// The keyed planes of the stages.
    pub(crate) keyed: Vec<KeyedRecord>,
    /// The unit of a cut kernel that lists its stages and has no entry of its own: the caller's view of the kernel.
    pub(crate) source: Option<UnitMeta>,
}
