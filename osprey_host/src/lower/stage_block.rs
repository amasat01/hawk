//! The stage block of an answer: what a kernel cut into launches tells its runtime (the stages in launch order, the cells and
//! keyed planes between them, the scratch planes it makes once) and the facts of a run that follow from it. The caller writes any
//! file from these records; nothing here knows a file format.
use super::request::{int_of, text_of, wrong, Keys};
use aether_writers::contract::Json;
use aether_writers::refuse::Refusal;

/// A cell: a reduction over samples stored in rows, finished between two launches.
#[derive(Debug, Clone, PartialEq)]
pub struct CellRecord {
    /// The cell's name.
    pub name: String,
    /// The width of the cell.
    pub columns: i64,
    /// The reduction's operation.
    pub op: String,
    /// Whether only a later stage reads it and the caller never declared it.
    pub internal: bool,
}

/// One launch of a cut kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct StageRecord {
    /// The stage's unit name.
    pub kernel: String,
    /// The cells it reads.
    pub cells_in: Vec<String>,
    /// The cells it stores.
    pub cells_out: Vec<String>,
    /// The kinds of the per-sample values it computes again.
    pub recomputed: Vec<String>,
    /// The team the stage runs on, when it runs on one.
    pub team: Option<TeamRecord>,
}

/// The team a stage runs on and what one block of its launch is.
#[derive(Debug, Clone, PartialEq)]
pub struct TeamRecord {
    /// The threads of one team.
    pub threads: i64,
    /// The lanes one block of a warp per lane serves; one otherwise.
    pub lanes: i64,
    /// The warps of one team of a warp per lane; one otherwise.
    pub warps: i64,
    /// The keys a block of the launch stands for, when it stands for one (else it stands for a lane).
    pub keys: Option<KeysGrid>,
    /// The chunk of lanes or the cell an ordered scatter's block stands for, when it stands for one.
    pub scatter: Option<ScatterGrid>,
}

/// A launch whose blocks are the chunks of lanes or the cells of an ordered scatter.
#[derive(Debug, Clone, PartialEq)]
pub struct ScatterGrid {
    /// `chunks` or `targets`.
    pub grid: String,
    /// The plane the scatter adds into.
    pub target: String,
    /// The plane whose flags say a key was out of range, for the chunks.
    pub status: Option<String>,
}

/// A launch whose blocks are the keys of a block sum.
#[derive(Debug, Clone, PartialEq)]
pub struct KeysGrid {
    /// The blocks of a key.
    pub groups: i64,
    /// The keyed plane whose keys they are.
    pub keyed: KeyedRecord,
}

/// A plane keyed by some lane axes that the runtime owns between stages.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyedRecord {
    /// The plane's name.
    pub name: String,
    /// The entries of one key's payload.
    pub width: i64,
    /// The kept lane axes: a label and an extent, each when it has one.
    pub axes: Vec<(Option<String>, Option<i64>)>,
    /// The shape of one key's payload.
    pub payload: Vec<i64>,
}

/// A scratch plane the runtime makes once per kernel and lane count.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaneRecord {
    /// The plane's name.
    pub name: String,
    /// The element type: `real` (the build's scalar type), or a plain type name.
    pub dtype: String,
    /// The elements of each lane's slice.
    pub per_lane: i64,
    /// How the plane is laid out when it is not the default (`plane_major`: one plane per value of the lane).
    pub layout: Option<String>,
    /// The keys one slice stands for, when the plane holds a value per key.
    pub keys: Option<KeyedRecord>,
    /// The plane whose cells the slices stand for, when they do.
    pub per_target: Option<String>,
    /// Whether one slice serves the whole launch.
    pub once: bool,
    /// What the plane holds at the start of every run, when something is promised.
    pub fill: Option<i64>,
}

/// What a kernel cut into launches (or with wide values placed in scratch planes) tells its runtime.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StageBlock {
    /// The cells the stages store, in first-stored order.
    pub cells: Vec<CellRecord>,
    /// The stages in launch order.
    pub stages: Vec<StageRecord>,
    /// The scratch planes of the stages.
    pub team_planes: Vec<PlaneRecord>,
    /// The keyed planes of the stages.
    pub keyed: Vec<KeyedRecord>,
}

/// The facts a run reads of a lowered kernel.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RunFacts {
    /// The samples one row of a reduction over samples stands for, when the kernel has cells.
    pub cell_row: Option<i64>,
    /// The most chunks of an ordered scatter, when the kernel has a team stage or scratch planes.
    pub scatter_chunks: Option<i64>,
}

/// A number as a value.
fn num(n: i64) -> Json {
    Json::Num(n.to_string())
}

/// A list of texts as a value.
fn names(items: &[String]) -> Json {
    Json::Arr(items.iter().map(|s| Json::Str(s.clone())).collect())
}

/// An optional value, null when absent.
fn maybe<T>(item: &Option<T>, one: impl Fn(&T) -> Json) -> Json {
    item.as_ref().map_or(Json::Null, one)
}

/// A keyed plane as a value.
fn keyed_json(k: &KeyedRecord) -> Json {
    let axis = |(label, extent): &(Option<String>, Option<i64>)| {
        Json::Arr(vec![
            maybe(label, |l| Json::Str(l.clone())),
            maybe(extent, |e| num(*e)),
        ])
    };
    Json::Obj(vec![
        ("name".into(), Json::Str(k.name.clone())),
        ("width".into(), num(k.width)),
        ("axes".into(), Json::Arr(k.axes.iter().map(axis).collect())),
        (
            "payload".into(),
            Json::Arr(k.payload.iter().map(|n| num(*n)).collect()),
        ),
    ])
}

/// A keyed plane read back.
fn keyed_read(json: &Json) -> Result<KeyedRecord, Refusal> {
    let mut k = Keys::of(json, "keyed")?;
    let axis = |a: &Json| match a.arr() {
        [label, extent] => Ok((label.str().map(String::from), extent.int())),
        _ => Err(wrong("axes", "not a pair")),
    };
    let found = KeyedRecord {
        name: text_of(k.req("name")?, "name")?,
        width: int_of(k.req("width")?, "width")?,
        axes: k
            .req("axes")?
            .arr()
            .iter()
            .map(axis)
            .collect::<Result<_, _>>()?,
        payload: k
            .req("payload")?
            .arr()
            .iter()
            .map(|n| int_of(n, "payload"))
            .collect::<Result<_, _>>()?,
    };
    k.done()?;
    Ok(found)
}

/// A team record as a value.
fn team_json(team: &TeamRecord) -> Json {
    let grid = |g: &KeysGrid| {
        Json::Obj(vec![
            ("groups".into(), num(g.groups)),
            ("keyed".into(), keyed_json(&g.keyed)),
        ])
    };
    let scatter = |g: &ScatterGrid| {
        Json::Obj(vec![
            ("grid".into(), Json::Str(g.grid.clone())),
            ("target".into(), Json::Str(g.target.clone())),
            ("status".into(), maybe(&g.status, |s| Json::Str(s.clone()))),
        ])
    };
    Json::Obj(vec![
        ("threads".into(), num(team.threads)),
        ("lanes".into(), num(team.lanes)),
        ("warps".into(), num(team.warps)),
        ("keys".into(), maybe(&team.keys, grid)),
        ("scatter".into(), maybe(&team.scatter, scatter)),
    ])
}

/// A team record read back.
fn team_read(json: &Json) -> Result<TeamRecord, Refusal> {
    let mut keys = Keys::of(json, "team")?;
    let threads = int_of(keys.req("threads")?, "threads")?;
    let lanes = keys.opt("lanes").map_or(Ok(1), |n| int_of(n, "lanes"))?;
    let warps = keys.opt("warps").map_or(Ok(1), |n| int_of(n, "warps"))?;
    let grid = |j: &Json| -> Result<KeysGrid, Refusal> {
        let mut k = Keys::of(j, "keys")?;
        let found = KeysGrid {
            groups: int_of(k.req("groups")?, "groups")?,
            keyed: keyed_read(k.req("keyed")?)?,
        };
        k.done()?;
        Ok(found)
    };
    let held = keys
        .opt("keys")
        .filter(|g| !matches!(g, Json::Null))
        .map(grid)
        .transpose()?;
    let scatter = |j: &Json| -> Result<ScatterGrid, Refusal> {
        let mut k = Keys::of(j, "scatter")?;
        let found = ScatterGrid {
            grid: text_of(k.req("grid")?, "grid")?,
            target: text_of(k.req("target")?, "target")?,
            status: k
                .opt("status")
                .filter(|s| !matches!(s, Json::Null))
                .map(|s| text_of(s, "status"))
                .transpose()?,
        };
        k.done()?;
        Ok(found)
    };
    let scattered = keys
        .opt("scatter")
        .filter(|g| !matches!(g, Json::Null))
        .map(scatter)
        .transpose()?;
    keys.done()?;
    Ok(TeamRecord {
        threads,
        lanes,
        warps,
        keys: held,
        scatter: scattered,
    })
}

impl StageBlock {
    /// The block as a value.
    pub(super) fn to_json(&self) -> Json {
        let cell = |c: &CellRecord| {
            Json::Obj(vec![
                ("name".into(), Json::Str(c.name.clone())),
                ("columns".into(), num(c.columns)),
                ("op".into(), Json::Str(c.op.clone())),
                ("internal".into(), Json::Bool(c.internal)),
            ])
        };
        let stage = |s: &StageRecord| {
            Json::Obj(vec![
                ("kernel".into(), Json::Str(s.kernel.clone())),
                ("cells_in".into(), names(&s.cells_in)),
                ("cells_out".into(), names(&s.cells_out)),
                ("recomputed".into(), names(&s.recomputed)),
                ("team".into(), maybe(&s.team, team_json)),
            ])
        };
        let plane = |p: &PlaneRecord| {
            Json::Obj(vec![
                ("name".into(), Json::Str(p.name.clone())),
                ("dtype".into(), Json::Str(p.dtype.clone())),
                ("per_lane".into(), num(p.per_lane)),
                ("layout".into(), maybe(&p.layout, |l| Json::Str(l.clone()))),
                ("keys".into(), maybe(&p.keys, keyed_json)),
                (
                    "per_target".into(),
                    maybe(&p.per_target, |t| Json::Str(t.clone())),
                ),
                ("once".into(), Json::Bool(p.once)),
                ("fill".into(), maybe(&p.fill, |n| num(*n))),
            ])
        };
        Json::Obj(vec![
            (
                "cells".into(),
                Json::Arr(self.cells.iter().map(cell).collect()),
            ),
            (
                "stages".into(),
                Json::Arr(self.stages.iter().map(stage).collect()),
            ),
            (
                "team_planes".into(),
                Json::Arr(self.team_planes.iter().map(plane).collect()),
            ),
            (
                "keyed".into(),
                Json::Arr(self.keyed.iter().map(keyed_json).collect()),
            ),
        ])
    }

    /// Reads the block back.
    pub(super) fn read(json: &Json) -> Result<StageBlock, Refusal> {
        let mut keys = Keys::of(json, "stages")?;
        let list = |j: &Json, key: &str| -> Result<Vec<String>, Refusal> {
            j.arr().iter().map(|s| text_of(s, key)).collect()
        };
        let mut block = StageBlock::default();
        for c in keys.req("cells")?.arr() {
            let mut k = Keys::of(c, "cell")?;
            block.cells.push(CellRecord {
                name: text_of(k.req("name")?, "name")?,
                columns: int_of(k.req("columns")?, "columns")?,
                op: text_of(k.req("op")?, "op")?,
                internal: matches!(k.req("internal")?, Json::Bool(true)),
            });
            k.done()?;
        }
        for s in keys.req("stages")?.arr() {
            let mut k = Keys::of(s, "stage")?;
            block.stages.push(StageRecord {
                kernel: text_of(k.req("kernel")?, "kernel")?,
                cells_in: list(k.req("cells_in")?, "cells_in")?,
                cells_out: list(k.req("cells_out")?, "cells_out")?,
                recomputed: list(k.req("recomputed")?, "recomputed")?,
                team: k
                    .opt("team")
                    .filter(|t| !matches!(t, Json::Null))
                    .map(team_read)
                    .transpose()?,
            });
            k.done()?;
        }
        for p in keys.req("team_planes")?.arr() {
            let mut k = Keys::of(p, "plane")?;
            block.team_planes.push(PlaneRecord {
                name: text_of(k.req("name")?, "name")?,
                dtype: text_of(k.req("dtype")?, "dtype")?,
                per_lane: int_of(k.req("per_lane")?, "per_lane")?,
                layout: k.opt("layout").map(|l| text_of(l, "layout")).transpose()?,
                keys: k
                    .opt("keys")
                    .filter(|j| !matches!(j, Json::Null))
                    .map(keyed_read)
                    .transpose()?,
                per_target: k
                    .opt("per_target")
                    .filter(|j| !matches!(j, Json::Null))
                    .map(|t| text_of(t, "per_target"))
                    .transpose()?,
                once: matches!(k.opt("once"), Some(Json::Bool(true))),
                fill: k
                    .opt("fill")
                    .filter(|j| !matches!(j, Json::Null))
                    .map(|n| int_of(n, "fill"))
                    .transpose()?,
            });
            k.done()?;
        }
        for q in keys.req("keyed")?.arr() {
            block.keyed.push(keyed_read(q)?);
        }
        keys.done()?;
        Ok(block)
    }
}

impl RunFacts {
    /// The facts as a value.
    pub(super) fn to_json(&self) -> Json {
        Json::Obj(vec![
            ("cell_row".into(), maybe(&self.cell_row, |n| num(*n))),
            (
                "scatter_chunks".into(),
                maybe(&self.scatter_chunks, |n| num(*n)),
            ),
        ])
    }

    /// Reads the facts back.
    pub(super) fn read(json: &Json) -> Result<RunFacts, Refusal> {
        let mut keys = Keys::of(json, "run")?;
        let one = |j: Option<&Json>, key: &str| j.map(|n| int_of(n, key)).transpose();
        let facts = RunFacts {
            cell_row: one(keys.opt("cell_row"), "cell_row")?,
            scatter_chunks: one(keys.opt("scatter_chunks"), "scatter_chunks")?,
        };
        keys.done()?;
        Ok(facts)
    }
}
