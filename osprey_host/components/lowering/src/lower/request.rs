//! The request of the lowering and its answer: one request per kernel, answered with the units the kernel lowers to. Both are
//! closed-schema JSON on the rules of the writers' contract: every key is stated, a key the reader does not know is refused by
//! name, a required key that is missing is refused by name, and the version is checked first. Nothing here knows how a caller
//! records its kernels.
use super::derived::DerivedKernel;
use super::unit_meta::{
    DispatchCosts, ImageFormat, LibraryMeta, PeerRecord, StepsPair, UnitMeta, Wire,
};
use crate::facts;
use aether_writers::contract::types::{
    Access, Dtype, FoldForm, Guard, MapBits, Role, Scalar, SinkPolicy, Target, ThreadBox,
};
use aether_writers::contract::{Contract, Json};
use aether_writers::refuse::{bad, unknown, Refusal};
use aether_writers::unit::sums::RowPlane;
use aether_writers::unit::{CompanionFact, Facts, StepFacts};
use raptor_wings::blocks::LinalgForm;
use std::fmt::Debug;

/// The version of the request and answer schema, written as `"v"` and checked first.
pub const REQUEST_VERSION: u32 = 1;

/// A slot the caller declares for a kernel's walk: the role and name the slot is keyed on and its type.
#[derive(Debug, Clone, PartialEq)]
pub struct DeclaredSlot {
    /// What the slot holds.
    pub role: Role,
    /// The slot's name.
    pub name: String,
    /// The element type.
    pub dtype: Dtype,
    /// The extents, empty for a scalar.
    pub shape: Vec<i64>,
    /// The type's tag, when it has one.
    pub tag: Option<String>,
}

/// One kernel body: the nodes it writes and how its walk is taken over them.
#[derive(Debug, Clone, PartialEq)]
pub struct Member {
    /// The member's name.
    pub name: String,
    /// The node handles it writes, in the context that holds the kernel's IR.
    pub sinks: Vec<u32>,
    /// The slots the caller declares beyond those the body binds.
    pub declared: Vec<DeclaredSlot>,
    /// The access class forced on the walk, when one is.
    pub force: Option<Access>,
    /// The planes exempt from the access check.
    pub exempt: Vec<String>,
}

/// What the kernel's body is.
#[derive(Debug, Clone, PartialEq)]
pub enum Body {
    /// One body.
    Plain(Member),
    /// A chain of bodies, cut where the lowering decides.
    Chain(Vec<Member>),
    /// Bodies fused as lanes of one unit.
    Fused(Vec<Member>),
}

/// A set of lanes the caller builds: the launch runs lane `map[t]` for launch lane `t`, or lane `t` itself for a prefix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexSet {
    /// The word size of the map.
    pub bits: MapBits,
    /// Whether the set is the first lanes up to a count, with no map.
    pub prefix: bool,
}

/// The kind of the kernel as its author states it.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestKind {
    /// How a sample is guarded.
    pub guard: Guard,
    /// Whether the guard is an active set the launch runs over.
    pub active_set: bool,
    /// Whether the runtime fills the active set, as it does for the sweep unit of a segmented group (the stale samples) and not the
    /// caller. A set a caller asks for is built from the caller's mask and follows it; the runtime's has no mask to follow. The
    /// lowering sets it on the sweep's request alone: a request cannot state it.
    pub runtime_set: bool,
    /// The index set of lanes the kind declares, when it does.
    pub lanes: Option<IndexSet>,
    /// How the sinks are summed.
    pub sink: SinkPolicy,
}

/// How wide values are held: placed in scratch planes where the target needs them (`Auto`), kept in the thread (`Thread`), or
/// shared by a warp (`Team`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum WideForm {
    /// Chosen by the lowering.
    #[default]
    Auto,
    /// Each thread keeps its wide values.
    Thread,
    /// A warp holds each lane's wide values.
    Team,
}

/// The settings the caller states for the build; a setting left out has the default stated here.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    /// How a fold is written.
    pub fold_form: FoldForm,
    /// How wide values are held.
    pub wide_form: WideForm,
    /// How the linear algebra of a lane is held: block sums kept for a team of threads, or written per entry.
    pub linalg_form: LinalgForm,
    /// Whether a keyed accumulate is routed to the ordered stages. A banded build routes whatever this says; for the other scalars it
    /// is as stated.
    pub route_to_ordered: bool,
    /// The launch bounds a device kernel states.
    pub launch_bounds: Option<(i64, i64)>,
    /// The fold's step count, when stated.
    pub fold_k: Option<i64>,
    /// The warps of a team, and of a build.
    pub warps: i64,
    /// The threads a team is forced to; zero leaves it to the lowering.
    pub team_threads: i64,
    /// Five layout sizes in place of the derived ones.
    pub exports: Option<[u64; 5]>,
    /// Whether the device text is followed by its table of literals.
    pub literal_table: bool,
    /// Whether the units of the segmented form of an automatic dispatch group are built beside the fused unit, and the fused unit's
    /// record names them. Asked for a kernel whose one automatic group has a selector no sink of the kernel writes, in a plain body,
    /// it adds one unit per branch and the classify unit; any other kernel (no automatic group, a dynamic selector, a chain or fused
    /// body, several groups) is built as if it were not asked, and that is not an error.
    pub auto_segmented_set: bool,
    /// Whether a device unit builds its fast entry points: the unit text gates them behind a macro, which the device compile defines
    /// when this is set.
    pub fast_entries: bool,
}

/// One axis the caller declares: its label and extent, each when it has one (a runtime axis has no extent).
#[derive(Debug, Clone, PartialEq)]
pub struct LaneAxis {
    /// The axis's label.
    pub label: Option<String>,
    /// The axis's extent, at least 1.
    pub extent: Option<i64>,
}

/// The axes the caller declares for a sink plane, with the plane's payload.
#[derive(Debug, Clone, PartialEq)]
pub struct SinkAxes {
    /// The plane's name.
    pub name: String,
    /// The per-sample component of the plane: its extents, empty for a scalar plane. Every entry states it (a missing member is
    /// refused, never read as a scalar). The plane is bound with the payload first and the named axes after it, and the slot's
    /// rank and shape in the unit's record are the payload.
    pub payload: Vec<i64>,
    /// The plane's axes.
    pub axes: Vec<LaneAxis>,
}

/// How many steps one launch of a finishing kernel takes, as the caller declares it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Steps {
    /// A fixed count, at least 1.
    Fixed(i64),
    /// The runner chooses, up to `max` (at least 1).
    Auto {
        /// The most steps of one launch.
        max: i64,
    },
}

/// The image formats a caller can compile a device text to.
pub const IMAGE_FORMATS: [ImageFormat; 3] =
    [ImageFormat::Ptx, ImageFormat::Cubin, ImageFormat::Fatbin];

/// One request: a kernel and what the lowering needs to know about it beyond the IR.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelRequest {
    /// The kernel's name.
    pub name: String,
    /// The scalar type.
    pub scalar: Scalar,
    /// The targets the bundle is built for.
    pub targets: Vec<Target>,
    /// The kernel's kind.
    pub kind: RequestKind,
    /// The kernel's body.
    pub body: Body,
    /// The single step of a kernel written as two paths.
    pub one_step: Option<Member>,
    /// The kernels derived from this one (a taping twin, a reverse unit), answered in the same table after its own units.
    pub derived: Vec<DerivedKernel>,
    /// The samples one lane group covers when the launch runs over an active set.
    pub group: i64,
    /// The planes the caller declares for the kernel (an output it declares is its own, not the runtime's).
    pub planes: Vec<String>,
    /// The build's settings.
    pub settings: Settings,
    /// The format the caller compiles a device text to; a request with a device target states it.
    pub image_format: Option<ImageFormat>,
    /// The steps of one launch, when the kernel finishes samples and the caller declares them.
    pub steps: Option<Steps>,
    /// The kernel's lane axes.
    pub lane_space: Vec<LaneAxis>,
    /// The axes the caller declares for the kernel's sink planes.
    pub sink_axes: Vec<SinkAxes>,
    /// The parameter planes the caller declares shared: one value for every lane.
    pub shared: Vec<String>,
    /// The kernel's broadcast pattern: for a matrix plane the kernel reads at lane indices, whether it varies along each lane axis.
    pub broadcast: Vec<(String, Vec<bool>)>,
    /// The data documents the request is lowered under in place of the library's files, by name (see `facts::CARD_NAMES`), each
    /// the whole text of the document.
    pub cards: Vec<(String, String)>,
    /// The answer of the facts entry for each target of the request, passed back as it was given.
    pub facts: Vec<(Target, String)>,
    /// Whether the first kernel's host texts carry the self-check exports block. A set of kernels built into one host library asks
    /// every kernel with false and takes the block once from the set text; a text that is compiled alone asks with true. A device
    /// text always keeps its block (each device unit's image is a module of its own, and its layout self-check reads the block).
    /// No other part of any text changes.
    pub exports_block: bool,
    /// The targets of the build that other libraries serve: the caller lists the request names of the other libraries of the build
    /// (a library never lists its own target).
    pub companions: Vec<Target>,
    /// The record the leading library of the build answered with, keyed by the target it serves, handed in unread: the cut and the
    /// numbers this library lowers to.
    pub peer: Option<(Target, PeerRecord)>,
    /// Kernel names the runtime reserves for its own units; a kernel or a derived kernel named after one is refused.
    pub reserved_names: Vec<String>,
}

/// The text a unit has for one target, with the facts the caller reads back.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetText {
    /// The target the text is for.
    pub target: Target,
    /// The whole translation-unit text.
    pub text: String,
    /// The body inside the text.
    pub body: String,
    /// The entry function's name.
    pub entry: String,
    /// The scalar mode's id.
    pub mode: String,
    /// Whether the lane loop runs its lanes side by side.
    pub vectorised: bool,
    /// The thread box, when the unit has one with an axis.
    pub thread_box: Option<ThreadBox>,
    /// What the writer decided about the unit that its meta states.
    pub facts: Facts,
}

/// The peer member of a request: one record under the name of the library that answered it.
fn peer_read(json: &Json) -> Result<(Target, PeerRecord), Refusal> {
    let Json::Obj(members) = json else {
        return Err(wrong("peer", "not an object"));
    };
    let [(leader, record)] = members.as_slice() else {
        return Err(wrong("peer", "names not exactly one library"));
    };
    let target = member_of(&TARGETS, &Json::Str(leader.clone()), "peer")?;
    Ok((target, PeerRecord::get(Some(record), "peer")?))
}

impl KernelRequest {
    /// Whether the build this request belongs to has no target but `target`.
    pub fn alone_on(&self, target: Target) -> bool {
        self.targets == [target] && self.companions.is_empty() && self.peer.is_none()
    }

    /// Whether the build this request belongs to has a library for `target`: the request's own targets, the companions the caller
    /// lists, or the leader whose record the request carries.
    pub fn build_has(&self, target: Target) -> bool {
        self.targets.contains(&target)
            || self.companions.contains(&target)
            || self
                .peer
                .as_ref()
                .is_some_and(|(leader, _)| *leader == target)
    }
}

/// The facts of a text as a value.
fn written_facts(facts: &Facts) -> Json {
    let rows = facts
        .row_planes
        .iter()
        .map(|r| {
            Json::Obj(vec![
                ("cell".into(), Json::Str(r.cell.clone())),
                ("slot".into(), Json::Num(r.slot.to_string())),
            ])
        })
        .collect();
    let companions = facts
        .companions
        .iter()
        .map(|c| {
            Json::Obj(vec![
                ("plane".into(), Json::Str(c.plane.clone())),
                ("target".into(), Json::Str(c.target.clone())),
                ("fold_entry".into(), Json::Str(c.fold_entry.clone())),
            ])
        })
        .collect();
    let mixes = facts
        .mix_planes
        .iter()
        .map(|(group, plane)| {
            Json::Obj(vec![
                ("group".into(), Json::Str(group.clone())),
                ("plane".into(), Json::Str(plane.clone())),
            ])
        })
        .collect();
    let mut o = vec![
        ("row_planes".into(), Json::Arr(rows)),
        ("companions".into(), Json::Arr(companions)),
        ("mix_planes".into(), Json::Arr(mixes)),
    ];
    for (key, name) in [
        ("persist_function", &facts.persist_function),
        ("range_function", &facts.range_function),
    ] {
        if let Some(name) = name {
            o.push((key.into(), Json::Str(name.clone())));
        }
    }
    if let Some(step) = &facts.step {
        o.push((
            "step".into(),
            Json::Obj(vec![
                ("ops".into(), Json::Num(step.ops.to_string())),
                ("entry_counts".into(), Json::Bool(step.entry_counts)),
            ]),
        ))
    }
    Json::Obj(o)
}

/// The facts of a text a value describes.
fn read_written_facts(json: &Json, symbol: String) -> Result<Facts, Refusal> {
    let mut k = Keys::of(json, "facts")?;
    let row_planes = list_of(k.req("row_planes")?, "row_planes", |j| {
        let mut r = Keys::of(j, "row plane")?;
        let found = RowPlane {
            cell: text_of(r.req("cell")?, "cell")?,
            slot: usize::try_from(int_of(r.req("slot")?, "slot")?)
                .map_err(|_| wrong("slot", "not a position"))?,
        };
        r.done()?;
        Ok(found)
    })?;
    let companions = list_of(k.req("companions")?, "companions", |j| {
        let mut c = Keys::of(j, "companion")?;
        let found = CompanionFact {
            plane: text_of(c.req("plane")?, "plane")?,
            target: text_of(c.req("target")?, "target")?,
            fold_entry: text_of(c.req("fold_entry")?, "fold_entry")?,
        };
        c.done()?;
        Ok(found)
    })?;
    let mix_planes = list_of(k.req("mix_planes")?, "mix_planes", |j| {
        let mut m = Keys::of(j, "mix plane")?;
        let found = (
            text_of(m.req("group")?, "group")?,
            text_of(m.req("plane")?, "plane")?,
        );
        m.done()?;
        Ok(found)
    })?;
    let step = k
        .opt("step")
        .map(|j| {
            let mut s = Keys::of(j, "step")?;
            let found = StepFacts {
                ops: int_of(s.req("ops")?, "ops")?,
                entry_counts: flag_of(s.req("entry_counts")?, "entry_counts")?,
            };
            s.done()?;
            Ok(found)
        })
        .transpose()?;
    let persist_function = k
        .opt("persist_function")
        .map(|j| text_of(j, "persist_function"))
        .transpose()?;
    let range_function = k
        .opt("range_function")
        .map(|j| text_of(j, "range_function"))
        .transpose()?;
    k.done()?;
    Ok(Facts {
        symbol,
        row_planes,
        companions,
        step,
        mix_planes,
        persist_function,
        range_function,
    })
}

/// One unit of the answer: its text per target and its typed meta (which names it and holds its digest).
#[derive(Debug, Clone, PartialEq)]
pub struct UnitAnswer {
    /// The unit's text per target.
    pub texts: Vec<TargetText>,
    /// The unit's meta.
    pub meta: UnitMeta,
    /// The defines the unit's compile for a target needs, as `(target, define)` pairs: a text that gates code behind a macro the
    /// request's settings turn on. The caller hands them to that target's build.
    pub defines: Vec<(Target, String)>,
}

/// The answer: the units the kernel lowers to, in order, the library record, and the C source of the meta of all the units (the
/// constant data the front end compiles next to the kernel texts).
#[derive(Debug, Clone, PartialEq)]
pub struct KernelAnswer {
    /// The units.
    pub units: Vec<UnitAnswer>,
    /// The library record above the units.
    pub library: LibraryMeta,
    /// The C source of the meta.
    pub meta_source: String,
    /// Bytes of per-lane storage the kernel's body declares for its library results (a chain or a fused body: all its members
    /// together), under the fold threshold and form the lowering decided its rolled sites with.
    pub working_set_bytes: i64,
    /// The name of the device card's wide class whose thresholds placed the kernel's wide values: the class nearest the
    /// architecture the request's device facts state. None for a request with no device target.
    pub wide_class: Option<String>,
    /// The measured rows of the `steps_reverse` choice of the target this library serves; the other target's key is empty.
    pub steps_reverse: StepsPair,
    /// The costs a dispatch of the served device target is priced with, read by the runtime; none for a host.
    pub dispatch_costs: Option<DispatchCosts>,
    /// What a library that lowers the same build beside this one must match.
    pub peer: Option<PeerRecord>,
}

/// The input of the set text: the unit records of one or several answers, and the five layout sizes the exports block states.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitSet {
    /// The `meta` objects of the units, as the answers carry them, in the order the library lists them.
    pub units: Vec<UnitMeta>,
    /// Five layout sizes in place of the ones derived from the headers the text is compiled with.
    pub exports: Option<[u64; 5]>,
}

impl UnitSet {
    /// The input a JSON text describes: `{"v": 1, "units": [<meta>, ...], "exports": [5 sizes]}`, with `exports` optional.
    pub fn from_json(source: &str) -> Result<UnitSet, Refusal> {
        let parsed = Contract::parse(source)?;
        let mut keys = Keys::of(&parsed.0, "unit set")?;
        let version = int_of(keys.req("v")?, "v")?;
        if version != i64::from(REQUEST_VERSION) {
            return Err(unknown(format!("request version {version}")));
        }
        let units = list_of(keys.req("units")?, "units", |j| {
            UnitMeta::get(Some(j), "units")
        })?;
        let exports = keys.opt("exports").map(layout_sizes).transpose()?;
        keys.done()?;
        Ok(UnitSet { units, exports })
    }
}

/// The members of a closed set, read by their names.
fn pick<T: Debug + Copy>(all: &[T], name: &str, what: &str) -> Result<T, Refusal> {
    all.iter()
        .find(|member| format!("{member:?}") == name)
        .copied()
        .ok_or_else(|| unknown(format!("member {name} of {what}")))
}

/// The members of each closed set a request names, in declaration order.
pub const SCALARS: [Scalar; 3] = [Scalar::Float64, Scalar::Float32, Scalar::Banded];
/// The targets.
pub const TARGETS: [Target; 2] = [Target::Host, Target::Device];
/// The element types.
pub const DTYPES: [Dtype; 7] = [
    Dtype::F64,
    Dtype::F32,
    Dtype::I32,
    Dtype::I64,
    Dtype::Int32,
    Dtype::Uint32,
    Dtype::Bool,
];
/// The slot roles.
pub const ROLES: [Role; 12] = [
    Role::Mutable,
    Role::PerSample,
    Role::VecIn,
    Role::Lookup,
    Role::WideOut,
    Role::Uniform,
    Role::MatIn,
    Role::AccumOut,
    Role::Terminated,
    Role::WideIn,
    Role::Nsamples,
    Role::Out,
];
/// The access classes.
pub const ACCESSES: [Access; 3] = [
    Access::SampleLocal,
    Access::CrossSampleRead,
    Access::CrossSampleWrite,
];
/// The fold forms.
pub const FOLD_FORMS: [FoldForm; 3] = [FoldForm::Auto, FoldForm::Static, FoldForm::Rolled];
/// The linear algebra forms.
pub const LINALG_FORMS: [LinalgForm; 3] = [LinalgForm::Auto, LinalgForm::Team, LinalgForm::Thread];
/// The wide forms.
pub const WIDE_FORMS: [WideForm; 3] = [WideForm::Auto, WideForm::Thread, WideForm::Team];

/// The keys of one object, handed out by name; `done` refuses a key nobody asked for.
pub(super) struct Keys<'a> {
    /// The members, in the order written.
    items: &'a [(String, Json)],
    /// The keys asked for so far.
    asked: Vec<&'static str>,
    /// What the object stands for, for the refusal text.
    scope: &'static str,
}

impl<'a> Keys<'a> {
    /// Starts reading an object; a value that is not an object is refused by name.
    pub(super) fn of(json: &'a Json, scope: &'static str) -> Result<Keys<'a>, Refusal> {
        match json {
            Json::Obj(items) => Ok(Keys {
                items,
                asked: vec![],
                scope,
            }),
            _ => Err(unknown(format!("request key {scope}: not an object"))),
        }
    }
    /// The value under an optional key; null counts as absent.
    pub(super) fn opt(&mut self, key: &'static str) -> Option<&'a Json> {
        self.asked.push(key);
        let found = self.items.iter().find(|(k, _)| k == key).map(|(_, v)| v);
        found.filter(|value| **value != Json::Null)
    }
    /// The value under a required key.
    pub(super) fn req(&mut self, key: &'static str) -> Result<&'a Json, Refusal> {
        let scope = self.scope;
        self.opt(key)
            .ok_or_else(|| unknown(format!("request key {key} missing in {scope}")))
    }
    /// Refuses a key that was not asked for.
    pub(super) fn done(self) -> Result<(), Refusal> {
        match self
            .items
            .iter()
            .find(|(k, _)| !self.asked.contains(&k.as_str()))
        {
            Some((k, _)) => Err(unknown(format!("request key {k}"))),
            None => Ok(()),
        }
    }
}

/// A value of the wrong shape, refused naming the key.
pub(super) fn wrong(key: &str, what: &str) -> Refusal {
    unknown(format!("request key {key}: {what}"))
}
/// The text of a value.
pub(super) fn text_of(json: &Json, key: &str) -> Result<String, Refusal> {
    json.str()
        .map(String::from)
        .ok_or_else(|| wrong(key, "not text"))
}
/// The integer of a value.
pub(super) fn int_of(json: &Json, key: &str) -> Result<i64, Refusal> {
    json.int().ok_or_else(|| wrong(key, "not an integer"))
}
/// The five layout sizes of a value.
pub(super) fn layout_sizes(j: &Json) -> Result<[u64; 5], Refusal> {
    let values: Vec<u64> = list_of(j, "exports", |n| {
        u64::try_from(int_of(n, "exports")?).map_err(|_| wrong("exports", "not a size"))
    })?;
    values
        .try_into()
        .map_err(|_| wrong("exports", "not five sizes"))
}
/// The flag of a value.
pub(super) fn flag_of(json: &Json, key: &str) -> Result<bool, Refusal> {
    match json {
        Json::Bool(flag) => Ok(*flag),
        _ => Err(wrong(key, "not a flag")),
    }
}
/// The member of a closed set under a key.
pub(super) fn member_of<T: Debug + Copy>(all: &[T], json: &Json, key: &str) -> Result<T, Refusal> {
    pick(
        all,
        json.str().ok_or_else(|| wrong(key, "not a name"))?,
        key,
    )
}
/// The facts answers of a request, one per target named.
fn facts_read(json: &Json) -> Result<Vec<(Target, String)>, Refusal> {
    let Json::Obj(items) = json else {
        return Err(wrong("facts", "not an object"));
    };
    items
        .iter()
        .map(|(name, answer)| Ok((pick(&TARGETS, name, "facts")?, text_of(answer, "facts")?)))
        .collect()
}
/// A list of values, each read by `one`.
pub(super) fn list_of<T>(
    json: &Json,
    key: &str,
    one: impl Fn(&Json) -> Result<T, Refusal>,
) -> Result<Vec<T>, Refusal> {
    match json {
        Json::Arr(items) => items.iter().map(one).collect(),
        _ => Err(wrong(key, "not a list")),
    }
}

/// A one-member object `{"Name": payload}`.
fn tagged<'a>(json: &'a Json, key: &str) -> Result<(&'a str, &'a Json), Refusal> {
    match json {
        Json::Obj(items) if items.len() == 1 => Ok((items[0].0.as_str(), &items[0].1)),
        _ => Err(wrong(key, "not a named member")),
    }
}

impl DeclaredSlot {
    /// A scalar lookup plane of the given name and element type.
    pub fn lookup(name: &str, dtype: Dtype) -> DeclaredSlot {
        DeclaredSlot {
            role: Role::Lookup,
            name: name.to_string(),
            dtype,
            shape: vec![],
            tag: None,
        }
    }

    /// Reads a declared slot.
    fn read(json: &Json) -> Result<DeclaredSlot, Refusal> {
        let mut keys = Keys::of(json, "slot")?;
        let slot = DeclaredSlot {
            role: member_of(&ROLES, keys.req("role")?, "role")?,
            name: text_of(keys.req("name")?, "name")?,
            dtype: member_of(&DTYPES, keys.req("dtype")?, "dtype")?,
            shape: list_of(keys.req("shape")?, "shape", |n| int_of(n, "shape"))?,
            tag: keys.opt("tag").map(|t| text_of(t, "tag")).transpose()?,
        };
        keys.done()?;
        Ok(slot)
    }
    /// Writes the declared slot.
    fn write(&self) -> Json {
        let mut o = vec![
            ("role".into(), name_of(&self.role)),
            ("name".into(), Json::Str(self.name.clone())),
            ("dtype".into(), name_of(&self.dtype)),
            ("shape".into(), ints(&self.shape)),
        ];
        if let Some(tag) = &self.tag {
            o.push(("tag".into(), Json::Str(tag.clone())))
        }
        Json::Obj(o)
    }
}

/// The name of a member of a closed set, as a value.
pub(super) fn name_of<T: Debug>(member: &T) -> Json {
    Json::Str(format!("{member:?}"))
}
/// A list of integers as a value.
pub(super) fn ints<T: ToString>(values: &[T]) -> Json {
    Json::Arr(values.iter().map(|n| Json::Num(n.to_string())).collect())
}
/// A list of texts as a value.
pub(super) fn texts(values: &[String]) -> Json {
    Json::Arr(values.iter().map(|s| Json::Str(s.clone())).collect())
}

impl LaneAxis {
    /// Reads an axis: both keys are optional, and an extent is at least 1.
    fn read(json: &Json) -> Result<LaneAxis, Refusal> {
        let mut keys = Keys::of(json, "axis")?;
        let axis = LaneAxis {
            label: keys.opt("label").map(|l| text_of(l, "label")).transpose()?,
            extent: keys
                .opt("extent")
                .map(|e| int_of(e, "extent"))
                .transpose()?,
        };
        keys.done()?;
        match axis.extent {
            Some(n) if n < 1 => Err(bad(format!("request key extent: {n} is not at least 1"))),
            _ => Ok(axis),
        }
    }
    /// Writes the axis.
    fn write(&self) -> Json {
        let mut o = vec![];
        if let Some(label) = &self.label {
            o.push(("label".into(), Json::Str(label.clone())))
        }
        if let Some(extent) = self.extent {
            o.push(("extent".into(), Json::Num(extent.to_string())))
        }
        Json::Obj(o)
    }
}

impl SinkAxes {
    /// Reads the axes and the payload of a sink plane of the kernel `kernel`.
    fn read(json: &Json, kernel: &str) -> Result<SinkAxes, Refusal> {
        let mut keys = Keys::of(json, "sink_axes")?;
        let name = text_of(keys.req("name")?, "name")?;
        let Some(payload) = keys.opt("payload") else {
            return Err(bad(format!(
                "kernel {kernel:?}: the sink {name:?} states no payload (an empty list for a scalar plane)"
            )));
        };
        let found = SinkAxes {
            name,
            payload: list_of(payload, "payload", |n| int_of(n, "payload"))?,
            axes: list_of(keys.req("axes")?, "axes", LaneAxis::read)?,
        };
        keys.done()?;
        Ok(found)
    }
    /// Writes the axes of a sink plane.
    fn write(&self) -> Json {
        Json::Obj(vec![
            ("name".into(), Json::Str(self.name.clone())),
            ("payload".into(), ints(&self.payload)),
            (
                "axes".into(),
                Json::Arr(self.axes.iter().map(LaneAxis::write).collect()),
            ),
        ])
    }
}

/// The steps a request declares from its two keys: a count or `"auto"` (which needs `steps_max`), and the cap that only an auto
/// count has.
fn steps_read(steps: Option<&Json>, max: Option<&Json>) -> Result<Option<Steps>, Refusal> {
    let cap = max.map(|m| int_of(m, "steps_max")).transpose()?;
    let positive = |key: &str, n: i64| {
        if n < 1 {
            Err(bad(format!("request key {key}: {n} is not at least 1")))
        } else {
            Ok(n)
        }
    };
    match (steps, cap) {
        (None, None) => Ok(None),
        (None, Some(_)) => Err(bad("request key steps_max: without steps")),
        (Some(Json::Str(word)), Some(max)) if word == "auto" => Ok(Some(Steps::Auto {
            max: positive("steps_max", max)?,
        })),
        (Some(Json::Str(word)), None) if word == "auto" => {
            Err(bad("request key steps_max: missing for auto steps"))
        }
        (Some(_), Some(_)) => Err(bad("request key steps_max: only auto steps have one")),
        (Some(n), None) => Ok(Some(Steps::Fixed(positive("steps", int_of(n, "steps")?)?))),
    }
}

impl Member {
    /// Reads a member.
    pub(super) fn read(json: &Json) -> Result<Member, Refusal> {
        let mut keys = Keys::of(json, "member")?;
        let member = Member {
            name: text_of(keys.req("name")?, "name")?,
            sinks: list_of(keys.req("sinks")?, "sinks", |n| {
                u32::try_from(int_of(n, "sinks")?).map_err(|_| wrong("sinks", "not a node handle"))
            })?,
            declared: list_of(keys.req("declared")?, "declared", DeclaredSlot::read)?,
            force: keys
                .opt("force")
                .map(|f| member_of(&ACCESSES, f, "force"))
                .transpose()?,
            exempt: list_of(keys.req("exempt")?, "exempt", |s| text_of(s, "exempt"))?,
        };
        keys.done()?;
        Ok(member)
    }
    /// Writes the member.
    pub(super) fn write(&self) -> Json {
        let mut o = vec![
            ("name".into(), Json::Str(self.name.clone())),
            ("sinks".into(), ints(&self.sinks)),
            (
                "declared".into(),
                Json::Arr(self.declared.iter().map(DeclaredSlot::write).collect()),
            ),
        ];
        if let Some(force) = &self.force {
            o.push(("force".into(), name_of(force)))
        }
        o.push(("exempt".into(), texts(&self.exempt)));
        Json::Obj(o)
    }
}

/// Reads the members of a request that belong to one kernel's own lowering (its kind, single step, group, planes, settings, steps,
/// lane axes, sink axes, shared planes and broadcast pattern) into `into`, each left as it is when the request does not state it.
/// The kernel and each of its derived kernels are read by this one function.
pub(super) fn read_part(keys: &mut Keys, into: &mut KernelRequest) -> Result<(), Refusal> {
    if let Some(kind) = keys.opt("kind") {
        into.kind = RequestKind::read(kind)?;
    }
    if let Some(step) = keys.opt("one_step") {
        into.one_step = Some(Member::read(step)?);
    }
    if let Some(g) = keys.opt("group") {
        into.group = int_of(g, "group")?;
    }
    if let Some(p) = keys.opt("planes") {
        into.planes = list_of(p, "planes", |n| text_of(n, "planes"))?;
    }
    if let Some(s) = keys.opt("settings") {
        into.settings = Settings::read(s)?;
    }
    let (steps, max) = (keys.opt("steps"), keys.opt("steps_max"));
    if steps.is_some() || max.is_some() {
        into.steps = steps_read(steps, max)?;
    }
    if let Some(l) = keys.opt("lane_space") {
        into.lane_space = list_of(l, "lane_space", LaneAxis::read)?;
    }
    if let Some(l) = keys.opt("sink_axes") {
        let kernel = into.name.clone();
        into.sink_axes = list_of(l, "sink_axes", |j| SinkAxes::read(j, &kernel))?;
    }
    if let Some(l) = keys.opt("shared") {
        into.shared = list_of(l, "shared", |n| text_of(n, "shared"))?;
    }
    if let Some(l) = keys.opt("broadcast") {
        into.broadcast = list_of(l, "broadcast", |p| {
            let mut keys = Keys::of(p, "broadcast")?;
            let found = (
                text_of(keys.req("plane")?, "plane")?,
                list_of(keys.req("axes")?, "axes", |b| flag_of(b, "axes"))?,
            );
            keys.done()?;
            Ok(found)
        })?;
    }
    Ok(())
}

/// Reads the derived kernels of a request over the kernel `base`: each states its own per-kernel members, and what it leaves out is
/// the default of a request (its kind is the base's); two of one name and role are refused by name.
fn derived_read(json: &Json, base: &KernelRequest) -> Result<Vec<DerivedKernel>, Refusal> {
    let list = list_of(json, "derived", |one| DerivedKernel::read(one, base))?;
    for (at, kernel) in list.iter().enumerate() {
        let same =
            |other: &DerivedKernel| other.name() == kernel.name() && other.role == kernel.role;
        if list[..at].iter().any(same) {
            return Err(bad(format!(
                "request key derived: two kernels are named {} with the role {:?}",
                kernel.name(),
                kernel.role
            )));
        }
    }
    Ok(list)
}

impl Body {
    /// Reads a body: `{"Plain": member}`, `{"Chain": [members]}` or `{"Fused": [members]}`.
    fn read(json: &Json) -> Result<Body, Refusal> {
        let members = |j: &Json| {
            let list = list_of(j, "body", Member::read)?;
            let mut seen = std::collections::HashSet::new();
            match list.iter().find(|member| !seen.insert(&member.name)) {
                Some(member) => Err(bad(format!(
                    "request key body: two members are named {}",
                    member.name
                ))),
                None => Ok(list),
            }
        };
        match tagged(json, "body")? {
            ("Plain", member) => Ok(Body::Plain(Member::read(member)?)),
            ("Chain", list) => Ok(Body::Chain(members(list)?)),
            ("Fused", list) => Ok(Body::Fused(members(list)?)),
            (other, _) => Err(unknown(format!("member {other} of body"))),
        }
    }
    /// Writes the body.
    fn write(&self) -> Json {
        let one = |name: &str, list: &[Member]| {
            Json::Obj(vec![(
                name.into(),
                Json::Arr(list.iter().map(Member::write).collect()),
            )])
        };
        match self {
            Body::Plain(member) => Json::Obj(vec![("Plain".into(), member.write())]),
            Body::Chain(list) => one("Chain", list),
            Body::Fused(list) => one("Fused", list),
        }
    }
}

impl IndexSet {
    /// Reads an index set.
    fn read(json: &Json) -> Result<IndexSet, Refusal> {
        let mut keys = Keys::of(json, "lanes")?;
        let bits = match int_of(keys.req("bits")?, "bits")? {
            32 => MapBits::B32,
            64 => MapBits::B64,
            _ => return Err(wrong("bits", "not 32 or 64")),
        };
        let set = IndexSet {
            bits,
            prefix: flag_of(keys.req("prefix")?, "prefix")?,
        };
        keys.done()?;
        Ok(set)
    }
    /// Writes the index set.
    fn write(&self) -> Json {
        let bits = if self.bits == MapBits::B64 {
            "64"
        } else {
            "32"
        };
        Json::Obj(vec![
            ("bits".into(), Json::Num(bits.into())),
            ("prefix".into(), Json::Bool(self.prefix)),
        ])
    }
}

impl RequestKind {
    /// Reads a kind.
    fn read(json: &Json) -> Result<RequestKind, Refusal> {
        let mut keys = Keys::of(json, "kind")?;
        let guard = match keys.req("guard")? {
            Json::Str(s) if s == "Default" => Guard::Default,
            Json::Str(s) if s == "DataOnly" => Guard::DataOnly,
            other => match tagged(other, "guard")? {
                ("Masks", masks) => Guard::Masks(list_of(masks, "guard", |m| text_of(m, "guard"))?),
                (name, _) => return Err(unknown(format!("member {name} of guard"))),
            },
        };
        let kind = RequestKind {
            guard,
            active_set: flag_of(keys.req("active_set")?, "active_set")?,
            runtime_set: false,
            lanes: keys.opt("lanes").map(IndexSet::read).transpose()?,
            sink: sink_of(keys.req("sink")?)?,
        };
        keys.done()?;
        Ok(kind)
    }
    /// Writes the kind.
    pub(super) fn write(&self) -> Json {
        let guard = match &self.guard {
            Guard::Default => Json::Str("Default".into()),
            Guard::DataOnly => Json::Str("DataOnly".into()),
            Guard::Masks(masks) => Json::Obj(vec![("Masks".into(), texts(masks))]),
        };
        let sink = match &self.sink {
            SinkPolicy::Plain => Json::Str("Plain".into()),
            SinkPolicy::Compensated {
                plane,
                output_slot,
                output_names,
            } => {
                let mut o = vec![("plane".into(), Json::Str(plane.clone()))];
                if let Some(slot) = output_slot {
                    o.push(("output_slot".into(), Json::Str(slot.clone())))
                }
                o.push(("output_names".into(), texts(output_names)));
                Json::Obj(vec![("Compensated".into(), Json::Obj(o))])
            }
        };
        Json::Obj(vec![
            ("guard".into(), guard),
            ("active_set".into(), Json::Bool(self.active_set)),
            (
                "lanes".into(),
                self.lanes.map_or(Json::Null, |set| set.write()),
            ),
            ("sink".into(), sink),
        ])
    }
}

/// Reads a sink policy: `"Plain"` or `{"Compensated": {plane, output_slot?, output_names}}`.
fn sink_of(json: &Json) -> Result<SinkPolicy, Refusal> {
    if matches!(json, Json::Str(s) if s == "Plain") {
        return Ok(SinkPolicy::Plain);
    }
    match tagged(json, "sink")? {
        ("Compensated", body) => {
            let mut keys = Keys::of(body, "sink")?;
            let policy = SinkPolicy::Compensated {
                plane: text_of(keys.req("plane")?, "plane")?,
                output_slot: keys
                    .opt("output_slot")
                    .map(|s| text_of(s, "output_slot"))
                    .transpose()?,
                output_names: list_of(keys.req("output_names")?, "output_names", |n| {
                    text_of(n, "output_names")
                })?,
            };
            keys.done()?;
            Ok(policy)
        }
        (name, _) => Err(unknown(format!("member {name} of sink"))),
    }
}

/// The most threads a block of a launch holds.
const MAX_THREADS: i64 = 1024;

/// The value of a setting, refused by name and value when it lies outside `low..=high`.
fn in_range(key: &str, value: i64, low: i64, high: i64) -> Result<i64, Refusal> {
    if (low..=high).contains(&value) {
        Ok(value)
    } else {
        Err(wrong(key, &format!("{value} is outside {low} to {high}")))
    }
}

impl Settings {
    /// Reads the settings; every key but the version is optional and has its default.
    pub(super) fn read(json: &Json) -> Result<Settings, Refusal> {
        let mut keys = Keys::of(json, "settings")?;
        let pair = |j: &Json| match j.arr() {
            [a, b] => Ok((int_of(a, "launch_bounds")?, int_of(b, "launch_bounds")?)),
            _ => Err(wrong("launch_bounds", "not a pair")),
        };
        let settings = Settings {
            fold_form: match keys.opt("fold_form") {
                Some(form) => member_of(&FOLD_FORMS, form, "fold_form")?,
                None => FoldForm::default(),
            },
            wide_form: match keys.opt("wide_form") {
                Some(form) => member_of(&WIDE_FORMS, form, "wide_form")?,
                None => WideForm::default(),
            },
            linalg_form: match keys.opt("linalg_form") {
                Some(form) => member_of(&LINALG_FORMS, form, "linalg_form")?,
                None => LinalgForm::default(),
            },
            route_to_ordered: match keys.opt("route_to_ordered") {
                Some(f) => flag_of(f, "route_to_ordered")?,
                None => false,
            },
            launch_bounds: keys
                .opt("launch_bounds")
                .map(pair)
                .transpose()?
                .map(|(threads, blocks)| {
                    Ok((
                        in_range("launch_bounds threads", threads, 1, MAX_THREADS)?,
                        in_range("launch_bounds blocks", blocks, 1, i64::from(i32::MAX))?,
                    ))
                })
                .transpose()?,
            fold_k: keys
                .opt("fold_k")
                .map(|k| in_range("fold_k", int_of(k, "fold_k")?, 1, i64::from(i32::MAX)))
                .transpose()?,
            warps: match keys.opt("warps") {
                Some(w) => in_range("warps", int_of(w, "warps")?, 1, MAX_THREADS / 32)?,
                None => aether_writers::contract::types::DEFAULT_WARPS,
            },
            team_threads: match keys.opt("team_threads") {
                Some(t) => in_range("team_threads", int_of(t, "team_threads")?, 0, MAX_THREADS)?,
                None => 0,
            },
            exports: keys.opt("exports").map(layout_sizes).transpose()?,
            literal_table: match keys.opt("literal_table") {
                Some(f) => flag_of(f, "literal_table")?,
                None => false,
            },
            auto_segmented_set: match keys.opt("auto_segmented_set") {
                Some(f) => flag_of(f, "auto_segmented_set")?,
                None => false,
            },
            fast_entries: match keys.opt("fast_entries") {
                Some(f) => flag_of(f, "fast_entries")?,
                None => false,
            },
        };
        keys.done()?;
        Ok(settings)
    }
    /// Writes the settings, leaving out what has its default.
    pub(super) fn write(&self) -> Json {
        let mut o: Vec<(String, Json)> = vec![];
        if self.fold_form != FoldForm::default() {
            o.push(("fold_form".into(), name_of(&self.fold_form)))
        }
        if self.wide_form != WideForm::default() {
            o.push(("wide_form".into(), name_of(&self.wide_form)))
        }
        if self.linalg_form != LinalgForm::default() {
            o.push(("linalg_form".into(), name_of(&self.linalg_form)))
        }
        if self.route_to_ordered {
            o.push(("route_to_ordered".into(), Json::Bool(true)))
        }
        if let Some((a, b)) = self.launch_bounds {
            o.push(("launch_bounds".into(), ints(&[a, b])))
        }
        if let Some(k) = self.fold_k {
            o.push(("fold_k".into(), Json::Num(k.to_string())))
        }
        if self.warps != aether_writers::contract::types::DEFAULT_WARPS {
            o.push(("warps".into(), Json::Num(self.warps.to_string())))
        }
        if self.team_threads != 0 {
            o.push((
                "team_threads".into(),
                Json::Num(self.team_threads.to_string()),
            ))
        }
        if let Some(sizes) = &self.exports {
            o.push(("exports".into(), ints(sizes)))
        }
        if self.literal_table {
            o.push(("literal_table".into(), Json::Bool(true)))
        }
        if self.auto_segmented_set {
            o.push(("auto_segmented_set".into(), Json::Bool(true)))
        }
        if self.fast_entries {
            o.push(("fast_entries".into(), Json::Bool(true)))
        }
        Json::Obj(o)
    }
}

impl KernelRequest {
    /// The request a JSON text describes: the version is checked first; a key, a member or a version this build does not know is
    /// refused by name.
    pub fn from_json(source: &str) -> Result<KernelRequest, Refusal> {
        let parsed = Contract::parse(source)?;
        let mut keys = Keys::of(&parsed.0, "request")?;
        let version = int_of(keys.req("v")?, "v")?;
        if version != i64::from(REQUEST_VERSION) {
            return Err(unknown(format!("request version {version}")));
        }
        let mut request = KernelRequest {
            name: text_of(keys.req("name")?, "name")?,
            scalar: member_of(&SCALARS, keys.req("scalar")?, "scalar")?,
            targets: list_of(keys.req("targets")?, "targets", |t| {
                member_of(&TARGETS, t, "targets").map_err(|_| {
                    bad(format!(
                        "a target this library does not serve: {}",
                        t.str().unwrap_or("not a name")
                    ))
                })
            })?,
            kind: RequestKind::read(keys.req("kind")?)?,
            body: Body::read(keys.req("body")?)?,
            one_step: None,
            derived: vec![],
            group: aether_writers::contract::types::DEFAULT_GROUP,
            planes: vec![],
            settings: Settings::read(&Json::Obj(vec![]))?,
            image_format: keys
                .opt("image_format")
                .map(|f| member_of(&IMAGE_FORMATS, f, "image_format"))
                .transpose()?,
            steps: None,
            lane_space: vec![],
            sink_axes: vec![],
            shared: vec![],
            broadcast: vec![],
            cards: keys
                .opt("cards")
                .map(|value| facts::cards_from_json(value).map_err(bad))
                .transpose()?
                .unwrap_or_default(),
            facts: facts_read(keys.req("facts")?)?,
            exports_block: match keys.opt("exports_block") {
                Some(flag) => flag_of(flag, "exports_block")?,
                None => true,
            },
            companions: match keys.opt("companions") {
                Some(list) => {
                    list_of(list, "companions", |t| member_of(&TARGETS, t, "companions"))?
                }
                None => vec![],
            },
            peer: keys.opt("peer").map(peer_read).transpose()?,
            reserved_names: match keys.opt("reserved_names") {
                Some(list) => list_of(list, "reserved_names", |n| text_of(n, "reserved_names"))?,
                None => vec![],
            },
        };
        read_part(&mut keys, &mut request)?;
        if let Some(list) = keys.opt("derived") {
            request.derived = derived_read(list, &request)?;
        }
        keys.done()?;
        for target in &request.targets {
            request.facts_for(*target)?;
        }
        if request.targets.contains(&Target::Device) && request.image_format.is_none() {
            return Err(bad("request key image_format: a device target needs one"));
        }
        Ok(request)
    }

    /// The facts answer the request holds for a target; a request that names a target without its answer is refused.
    pub fn facts_for(&self, target: Target) -> Result<&str, Refusal> {
        self.facts
            .iter()
            .find(|(named, _)| *named == target)
            .map(|(_, answer)| answer.as_str())
            .ok_or_else(|| bad(format!("no facts answer for the target {target:?}")))
    }

    /// The members of the request that belong to one kernel's own lowering, as `read_part` reads them.
    pub(super) fn part_members(&self) -> Vec<(String, Json)> {
        let mut o: Vec<(String, Json)> = vec![];
        if let Some(step) = &self.one_step {
            o.push(("one_step".into(), step.write()))
        }
        if !self.planes.is_empty() {
            o.push(("planes".into(), texts(&self.planes)))
        }
        if self.group != aether_writers::contract::types::DEFAULT_GROUP {
            o.push(("group".into(), Json::Num(self.group.to_string())))
        }
        match self.steps {
            Some(Steps::Fixed(n)) => o.push(("steps".into(), Json::Num(n.to_string()))),
            Some(Steps::Auto { max }) => {
                o.push(("steps".into(), Json::Str("auto".into())));
                o.push(("steps_max".into(), Json::Num(max.to_string())))
            }
            None => {}
        }
        if !self.lane_space.is_empty() {
            let axes = self.lane_space.iter().map(LaneAxis::write).collect();
            o.push(("lane_space".into(), Json::Arr(axes)))
        }
        if !self.sink_axes.is_empty() {
            let sinks = self.sink_axes.iter().map(SinkAxes::write).collect();
            o.push(("sink_axes".into(), Json::Arr(sinks)))
        }
        if !self.shared.is_empty() {
            o.push(("shared".into(), texts(&self.shared)))
        }
        if !self.broadcast.is_empty() {
            let planes = self.broadcast.iter().map(|(plane, axes)| {
                Json::Obj(vec![
                    ("plane".into(), Json::Str(plane.clone())),
                    (
                        "axes".into(),
                        Json::Arr(axes.iter().map(|b| Json::Bool(*b)).collect()),
                    ),
                ])
            });
            o.push(("broadcast".into(), Json::Arr(planes.collect())))
        }
        o.push(("settings".into(), self.settings.write()));
        o
    }

    /// The request as JSON text, led by the version.
    pub fn to_json(&self) -> String {
        let mut o: Vec<(String, Json)> = vec![
            ("v".into(), Json::Num(REQUEST_VERSION.to_string())),
            ("name".into(), Json::Str(self.name.clone())),
            ("scalar".into(), name_of(&self.scalar)),
            (
                "targets".into(),
                Json::Arr(self.targets.iter().map(name_of).collect()),
            ),
            ("kind".into(), self.kind.write()),
            ("body".into(), self.body.write()),
        ];
        if !self.derived.is_empty() {
            o.push((
                "derived".into(),
                Json::Arr(self.derived.iter().map(DerivedKernel::write).collect()),
            ))
        }
        if let Some(format) = self.image_format {
            o.push(("image_format".into(), name_of(&format)))
        }
        o.extend(self.part_members());
        if !self.cards.is_empty() {
            let documents = self
                .cards
                .iter()
                .map(|(name, text)| (name.clone(), Json::Str(text.clone())))
                .collect();
            o.push(("cards".into(), Json::Obj(documents)))
        }
        let answers = self
            .facts
            .iter()
            .map(|(target, answer)| (format!("{target:?}"), Json::Str(answer.clone())))
            .collect();
        o.push(("facts".into(), Json::Obj(answers)));
        if !self.exports_block {
            o.push(("exports_block".into(), Json::Bool(false)))
        }
        if !self.companions.is_empty() {
            let names = self.companions.iter().map(name_of).collect();
            o.push(("companions".into(), Json::Arr(names)))
        }
        if let Some((leader, record)) = &self.peer {
            o.push((
                "peer".into(),
                Json::Obj(vec![(format!("{leader:?}"), record.put())]),
            ))
        }
        if !self.reserved_names.is_empty() {
            o.push(("reserved_names".into(), texts(&self.reserved_names)))
        }
        Json::Obj(o).to_text()
    }
}

impl KernelAnswer {
    /// The answer as JSON text, led by the version.
    pub fn to_json(&self) -> String {
        let text = |t: &TargetText| {
            let mut o: Vec<(String, Json)> = vec![
                ("target".into(), name_of(&t.target)),
                ("text".into(), Json::Str(t.text.clone())),
                ("body".into(), Json::Str(t.body.clone())),
                ("entry".into(), Json::Str(t.entry.clone())),
                ("symbol".into(), Json::Str(t.facts.symbol.clone())),
                ("mode".into(), Json::Str(t.mode.clone())),
                ("vectorised".into(), Json::Bool(t.vectorised)),
            ];
            o.push(("facts".into(), written_facts(&t.facts)));
            if let Some(b) = &t.thread_box {
                o.push((
                    "box".into(),
                    Json::Obj(vec![
                        ("axis".into(), Json::Num(b.axis.unwrap_or(0).to_string())),
                        ("T".into(), Json::Num(b.t.to_string())),
                        ("stride".into(), Json::Num(b.stride.to_string())),
                    ]),
                ))
            }
            Json::Obj(o)
        };
        let unit = |u: &UnitAnswer| {
            let mut o = vec![
                (
                    "texts".into(),
                    Json::Arr(u.texts.iter().map(text).collect()),
                ),
                ("meta".into(), u.meta.put()),
            ];
            if !u.defines.is_empty() {
                let mut by_target: Vec<(String, Json)> = vec![];
                for (target, define) in &u.defines {
                    let name = crate::backend::target_name(*target);
                    match by_target.iter_mut().find(|(n, _)| n == name) {
                        Some((_, Json::Arr(list))) => list.push(Json::Str(define.clone())),
                        _ => by_target
                            .push((name.into(), Json::Arr(vec![Json::Str(define.clone())]))),
                    }
                }
                o.push(("defines".into(), Json::Obj(by_target)));
            }
            Json::Obj(o)
        };
        Json::Obj(vec![
            ("v".into(), Json::Num(REQUEST_VERSION.to_string())),
            (
                "units".into(),
                Json::Arr(self.units.iter().map(unit).collect()),
            ),
            ("library".into(), self.library.put()),
            ("meta_source".into(), Json::Str(self.meta_source.clone())),
            (
                "working_set_bytes".into(),
                Json::Num(self.working_set_bytes.to_string()),
            ),
            (
                "wide_class".into(),
                self.wide_class.clone().map_or(Json::Null, Json::Str),
            ),
            ("steps_reverse".into(), self.steps_reverse.put()),
            ("dispatch_costs".into(), self.dispatch_costs.put()),
            ("peer".into(), self.peer.put()),
        ])
        .to_text()
    }

    /// The answer a JSON text describes (what a caller reads back).
    pub fn from_json(source: &str) -> Result<KernelAnswer, Refusal> {
        let parsed = Contract::parse(source)?;
        let mut keys = Keys::of(&parsed.0, "answer")?;
        let version = int_of(keys.req("v")?, "v")?;
        if version != i64::from(REQUEST_VERSION) {
            return Err(unknown(format!("request version {version}")));
        }
        let read_text = |j: &Json| -> Result<TargetText, Refusal> {
            let mut k = Keys::of(j, "text")?;
            let thread_box = match k.opt("box") {
                None => None,
                Some(b) => {
                    let mut bk = Keys::of(b, "box")?;
                    let found = ThreadBox {
                        axis: Some(int_of(bk.req("axis")?, "axis")?),
                        t: int_of(bk.req("T")?, "T")?,
                        stride: int_of(bk.req("stride")?, "stride")?,
                    };
                    bk.done()?;
                    Some(found)
                }
            };
            let symbol = text_of(k.req("symbol")?, "symbol")?;
            let text = TargetText {
                target: member_of(&TARGETS, k.req("target")?, "target")?,
                text: text_of(k.req("text")?, "text")?,
                body: text_of(k.req("body")?, "body")?,
                entry: text_of(k.req("entry")?, "entry")?,
                facts: read_written_facts(k.req("facts")?, symbol)?,
                mode: text_of(k.req("mode")?, "mode")?,
                vectorised: flag_of(k.req("vectorised")?, "vectorised")?,
                thread_box,
            };
            k.done()?;
            Ok(text)
        };
        let read_unit = |j: &Json| -> Result<UnitAnswer, Refusal> {
            let mut k = Keys::of(j, "unit")?;
            let mut defines = vec![];
            if let Some(Json::Obj(by_target)) = k.opt("defines") {
                for (name, list) in by_target {
                    // the writer names a target as the libraries do (`host`, `cuda`)
                    let target = TARGETS
                        .into_iter()
                        .find(|t| crate::backend::target_name(*t) == name)
                        .ok_or_else(|| {
                            wrong("defines", &format!("{name:?} is no target of this build"))
                        })?;
                    for define in list_of(list, "defines", |d| text_of(d, "defines"))? {
                        defines.push((target, define));
                    }
                }
            }
            let unit = UnitAnswer {
                texts: list_of(k.req("texts")?, "texts", read_text)?,
                meta: UnitMeta::get(k.opt("meta"), "meta")?,
                defines,
            };
            k.done()?;
            Ok(unit)
        };
        let units = list_of(keys.req("units")?, "units", read_unit)?;
        let library = LibraryMeta::get(keys.opt("library"), "library")?;
        let meta_source = text_of(keys.req("meta_source")?, "meta_source")?;
        let working_set_bytes = int_of(keys.req("working_set_bytes")?, "working_set_bytes")?;
        let wide_class = keys
            .opt("wide_class")
            .map(|name| text_of(name, "wide_class"))
            .transpose()?;
        let steps_reverse = StepsPair::get(keys.opt("steps_reverse"), "steps_reverse")?;
        let dispatch_costs =
            Option::<DispatchCosts>::get(keys.opt("dispatch_costs"), "dispatch_costs")?;
        let peer = Option::<PeerRecord>::get(keys.opt("peer"), "peer")?;
        keys.done()?;
        Ok(KernelAnswer {
            units,
            library,
            meta_source,
            working_set_bytes,
            wide_class,
            steps_reverse,
            dispatch_costs,
            peer,
        })
    }
}

#[cfg(test)]
mod tests {
    //! The request round-trips, and the reader is strict.
    use super::*;

    /// A plain host request.
    fn request() -> KernelRequest {
        KernelRequest {
            name: "k".into(),
            scalar: Scalar::Float64,
            targets: vec![Target::Host],
            kind: RequestKind {
                guard: Guard::Masks(vec!["m".into()]),
                active_set: false,
                runtime_set: false,
                lanes: None,
                sink: SinkPolicy::Plain,
            },
            body: Body::Plain(Member {
                name: "k".into(),
                sinks: vec![3],
                declared: vec![DeclaredSlot {
                    role: Role::Mutable,
                    name: "out".into(),
                    dtype: Dtype::F64,
                    shape: vec![2],
                    tag: None,
                }],
                force: None,
                exempt: vec![],
            }),
            one_step: None,
            derived: vec![],
            group: 1,
            planes: vec![],
            settings: Settings::read(&Json::Obj(vec![])).unwrap(),
            image_format: None,
            steps: None,
            lane_space: vec![],
            sink_axes: vec![],
            shared: vec![],
            broadcast: vec![],
            cards: vec![],
            facts: vec![(Target::Host, "{}".into())],
            exports_block: true,
            companions: vec![],
            peer: None,
            reserved_names: vec![],
        }
    }

    #[test]
    fn the_segmented_set_of_an_automatic_group_is_asked_for_by_a_setting_that_is_absent_by_default()
    {
        let none = Settings::read(&Json::Obj(vec![])).unwrap();
        assert!(!none.auto_segmented_set);
        assert_eq!(none.write(), Json::Obj(vec![]));
        let asked = Json::Obj(vec![("auto_segmented_set".into(), Json::Bool(true))]);
        let read = Settings::read(&asked).unwrap();
        assert!(read.auto_segmented_set);
        assert_eq!(read.write(), asked);
        let not_a_flag = Json::Obj(vec![("auto_segmented_set".into(), Json::Num("1".into()))]);
        assert!(Settings::read(&not_a_flag).is_err());
    }

    #[test]
    fn the_cards_member_round_trips_and_an_unknown_document_is_refused_by_name() {
        let mut stated = request();
        stated.cards = vec![("build.json".into(), "{}".into())];
        assert_eq!(KernelRequest::from_json(&stated.to_json()).unwrap(), stated);
        stated.cards = vec![("build.jsn".into(), "{}".into())];
        let named = KernelRequest::from_json(&stated.to_json()).unwrap();
        let docs = crate::facts::CardSet::new(
            &crate::testing::shipped(),
            &named.cards,
            crate::testing::ALL_CARDS,
        );
        let why = docs.check_stated().unwrap_err().refusal();
        assert!(why.reason.contains("build.jsn"), "{}", why.reason);
        assert_eq!(why.kind, aether_writers::refuse::RefusalKind::BadRequest);
        stated.cards = vec![("build.json".into(), "{}".into())];
        let numbered = stated.to_json().replace("\"{}\"", "{}");
        let why = KernelRequest::from_json(&numbered).unwrap_err();
        assert!(why.reason.contains("build.json"), "{}", why.reason);
    }

    #[test]
    fn the_exports_block_is_on_unless_the_request_turns_it_off() {
        let plain = request();
        assert!(plain.exports_block && !plain.to_json().contains("exports_block"));
        let off = KernelRequest {
            exports_block: false,
            ..plain
        };
        let text = off.to_json();
        assert!(text.contains("\"exports_block\":false"));
        assert_eq!(KernelRequest::from_json(&text).unwrap(), off);
    }

    #[test]
    fn the_declared_members_round_trip_and_are_range_checked_by_name() {
        let mut full = request();
        full.steps = Some(Steps::Auto { max: 64 });
        full.image_format = Some(ImageFormat::Ptx);
        full.lane_space = vec![LaneAxis {
            label: Some("a".into()),
            extent: Some(3),
        }];
        full.shared = vec!["w".into()];
        full.broadcast = vec![("m".into(), vec![true, false])];
        let text = full.to_json();
        assert_eq!(KernelRequest::from_json(&text).unwrap(), full);
        let cases = [
            ("\"steps_max\":64", "\"steps_max\":0", "steps_max"),
            ("\"extent\":3", "\"extent\":0", "extent"),
            (
                "\"steps\":\"auto\",\"steps_max\":64",
                "\"steps\":\"auto\"",
                "steps_max",
            ),
        ];
        for (from, to, member) in cases {
            let why = KernelRequest::from_json(&text.replace(from, to)).unwrap_err();
            assert!(why.reason.contains(member), "{}", why.reason);
        }
        let device = text
            .replace("\"Host\"", "\"Device\"")
            .replace("\"image_format\":\"Ptx\",", "");
        assert!(KernelRequest::from_json(&device).is_err());
    }

    #[test]
    fn the_fast_entry_setting_is_a_flag_that_is_written_only_when_set() {
        let on =
            Settings::read(&Json::Obj(vec![("fast_entries".into(), Json::Bool(true))])).unwrap();
        assert!(on.fast_entries);
        assert!(on.write().to_text().contains("\"fast_entries\":true"));
        let off = Settings::read(&Json::Obj(vec![])).unwrap();
        assert!(!off.fast_entries && !off.write().to_text().contains("fast_entries"));
        let why = Settings::read(&Json::Obj(vec![(
            "fast_entries".into(),
            Json::Num("1".into()),
        )]))
        .unwrap_err();
        assert!(why.reason.contains("fast_entries"), "{}", why.reason);
    }

    #[test]
    fn a_setting_out_of_range_is_a_bad_request_naming_the_member_and_the_value() {
        /// A setting set out of range, the member the refusal names and the value it quotes.
        type Case<'a> = (&'a dyn Fn(&mut Settings), &'a str, &'a str);
        let cases: [Case; 7] = [
            (&|s| s.warps = 0, "warps", "0"),
            (&|s| s.warps = 33, "warps", "33"),
            (&|s| s.team_threads = -4, "team_threads", "-4"),
            (&|s| s.team_threads = 2048, "team_threads", "2048"),
            (
                &|s| s.launch_bounds = Some((0, 1)),
                "launch_bounds threads",
                "0",
            ),
            (
                &|s| s.launch_bounds = Some((128, 0)),
                "launch_bounds blocks",
                "0",
            ),
            (&|s| s.fold_k = Some(0), "fold_k", "0"),
        ];
        for (set, member, value) in cases {
            let mut bad = request();
            set(&mut bad.settings);
            let why = KernelRequest::from_json(&bad.to_json()).unwrap_err();
            assert_eq!(why.kind, aether_writers::refuse::RefusalKind::BadRequest);
            assert!(
                why.reason.contains(member) && why.reason.contains(value),
                "{}",
                why.reason
            );
        }
        let mut fine = request();
        fine.settings.launch_bounds = Some((128, 2));
        fine.settings.warps = 4;
        assert!(KernelRequest::from_json(&fine.to_json()).is_ok());
    }

    #[test]
    fn a_sink_that_is_no_node_of_the_context_is_a_bad_request_not_a_panic() {
        let mut bad = request();
        if let Body::Plain(member) = &mut bad.body {
            member.sinks = vec![7];
        }
        let table = crate::door::table::<crate::testing::TestHost>();
        let ctx = (table.ctx_new)();
        let dir = crate::testing::shipped();
        let why =
            crate::lower::lower(&crate::testing::TestHost, ctx, &bad.to_json(), &dir).unwrap_err();
        // SAFETY: `ctx` came from this table's `ctx_new` and is freed once.
        unsafe { (table.ctx_free)(ctx) };
        assert_eq!(why.kind, aether_writers::refuse::RefusalKind::BadRequest);
        assert!(
            why.reason.contains("sinks") && why.reason.contains('7'),
            "{}",
            why.reason
        );
    }

    #[test]
    fn a_team_op_inside_a_loop_of_the_body_is_a_named_refusal_not_a_fault() {
        let table = crate::door::table::<crate::testing::TestHost>();
        let ctx = (table.ctx_new)();
        // SAFETY: `ctx` came from this table's `ctx_new`; nothing else holds it until it is freed below.
        let c = unsafe { &mut *ctx };
        let leaf = |c: &mut raptor_ir::record::Ctx, role: &str, name: &str, shape: &[i64]| {
            let fields = format!("(('kind', 'vocab_read'), ('role', '{role}'), ('name', '{name}'), ('quantity', None), ('wire_index', 0), ('sorted', False))");
            c.make("Leaf", "f64", shape, None, &[], &fields).unwrap()
        };
        let (a, b) = (
            leaf(c, "mat_in", "A", &[40, 40]),
            leaf(c, "vec_in", "b", &[40]),
        );
        let index = c
            .make(
                "LoopIndex",
                "int32",
                &[],
                None,
                &[],
                "(('depth', 0), ('name', '_'), ('bounds', (0, 3)))",
            )
            .unwrap();
        let carry = c
            .make(
                "LoopCarry",
                "f64",
                &[40],
                None,
                &[],
                "(('depth', 0), ('slot', 0), ('name', 's'))",
            )
            .unwrap();
        let op = "(('kind', 'chol'), ('literal', None))";
        let factor = c.make("Op", "f64", &[40, 40], None, &[a], op).unwrap();
        let solve = "(('kind', 'cho_solve'), ('literal', None))";
        let next = c
            .make("Op", "f64", &[40], None, &[factor, carry], solve)
            .unwrap();
        let assign = "(('kind', 'assign'), ('role', 'mutable'), ('name', 'y'))";
        let store = c
            .make("Assign", "f64", &[40], None, &[next], assign)
            .unwrap();
        let parts = raptor_ir::build::LoopParts {
            index,
            carries: &[carry],
            inits: &[b],
            nexts: &[next],
            range: (0, 3, 1),
            body_sinks: &[store],
            exit_cond: None,
            exit_values: &[],
            count: None,
            count_tail: false,
            dynamic: None,
        };
        let looped = c.build_loop(&parts).unwrap();
        let mut asked = request();
        if let Body::Plain(member) = &mut asked.body {
            member.sinks = vec![looped];
            member.declared.clear();
        }
        let dir = crate::testing::shipped();
        let why = crate::lower::lower(&crate::testing::TestHost, ctx, &asked.to_json(), &dir)
            .unwrap_err();
        // SAFETY: `ctx` came from this table's `ctx_new` and is freed once.
        unsafe { (table.ctx_free)(ctx) };
        assert_ne!(
            why.kind,
            aether_writers::refuse::RefusalKind::Fault,
            "{}",
            why.reason
        );
        assert!(
            why.reason.contains("cannot sit inside a loop of the body"),
            "{}",
            why.reason
        );
    }

    /// A request names its scalar by one of the three closed names; any other is refused by name, so a caller keeps no list of its own.
    #[test]
    fn a_scalar_the_library_does_not_know_is_refused_by_name() {
        let text = |scalar: &str| {
            format!(
                r#"{{"v": 1, "name": "k", "scalar": "{scalar}", "targets": ["Host"], "kind": {{"guard": "Default", "active_set": false, "sink": "Plain"}}, "body": {{"Plain": {{"name": "k", "sinks": [0], "declared": [], "exempt": []}}}}, "facts": {{"Host": "{{}}"}}}}"#
            )
        };
        for known in ["Float64", "Float32", "Banded"] {
            KernelRequest::from_json(&text(known)).unwrap();
        }
        let why = KernelRequest::from_json(&text("Float128")).unwrap_err();
        assert_eq!(why.kind, aether_writers::refuse::RefusalKind::BadRequest);
        assert_eq!(why.reason, "not in this build: member Float128 of scalar");
    }

    #[test]
    fn an_answer_carries_its_wide_class_name_or_none_through_its_text() {
        for wide_class in [Some("sm61".to_string()), None] {
            let answer = KernelAnswer {
                units: vec![],
                library: super::super::unit_meta::LibraryMeta::current(),
                meta_source: String::new(),
                working_set_bytes: 0,
                wide_class,
                steps_reverse: Default::default(),
                dispatch_costs: Some(super::super::unit_meta::DispatchCosts {
                    launch_us: Some(2.7),
                    ..Default::default()
                }),
                peer: Some(PeerRecord {
                    units: vec!["k".into()],
                    ..Default::default()
                }),
            };
            assert_eq!(KernelAnswer::from_json(&answer.to_json()).unwrap(), answer);
        }
    }

    #[test]
    fn a_negative_slot_position_in_an_answer_is_refused_by_name() {
        let facts = Facts {
            row_planes: vec![RowPlane {
                cell: "c".into(),
                slot: 3,
            }],
            ..Facts::default()
        };
        let text = written_facts(&facts)
            .to_text()
            .replace("\"slot\":3", "\"slot\":-3");
        let parsed = Contract::parse(&text).unwrap();
        let why = read_written_facts(&parsed.0, String::new()).unwrap_err();
        assert!(why.reason.contains("slot"), "{}", why.reason);
    }

    #[test]
    fn derived_kernels_round_trip_and_two_of_one_name_and_role_are_refused() {
        use super::super::derived::DerivedFacts;
        use super::super::links::LinkFact;
        use super::super::unit_meta::{DerivedRole, LinkForm, LinkWhat};
        let kernel = |role| DerivedKernel {
            request: KernelRequest {
                body: Body::Plain(Member {
                    name: "k".into(),
                    sinks: vec![4],
                    declared: vec![],
                    force: None,
                    exempt: vec![],
                }),
                steps: Some(Steps::Fixed(3)),
                ..request()
            },
            role,
            facts: DerivedFacts {
                links: vec![LinkFact {
                    slot: "bar_x".into(),
                    what: LinkWhat::PrimalSlot,
                    form: LinkForm::Grad,
                    of: Some("x".into()),
                    reduced: true,
                }],
                ..DerivedFacts::default()
            },
        };
        let mut set = request();
        set.derived = vec![kernel(DerivedRole::Tape), kernel(DerivedRole::StepVjp)];
        assert_eq!(KernelRequest::from_json(&set.to_json()), Ok(set.clone()));
        set.derived.push(kernel(DerivedRole::Tape));
        let why = KernelRequest::from_json(&set.to_json()).unwrap_err();
        assert!(
            why.reason.contains("named k with the role Tape"),
            "{}",
            why.reason
        );
    }

    #[test]
    fn two_members_of_one_name_are_refused_by_name() {
        let member = |name: &str| Member {
            name: name.into(),
            sinks: vec![3],
            declared: vec![],
            force: None,
            exempt: vec![],
        };
        let mut chain = request();
        chain.body = Body::Chain(vec![member("a"), member("b")]);
        assert!(KernelRequest::from_json(&chain.to_json()).is_ok());
        chain.body = Body::Chain(vec![member("a"), member("b"), member("a")]);
        let why = KernelRequest::from_json(&chain.to_json()).unwrap_err();
        assert_eq!(why.kind, aether_writers::refuse::RefusalKind::BadRequest);
        assert!(why.reason.contains("named a"), "{}", why.reason);
    }

    #[test]
    fn a_target_the_library_does_not_serve_is_refused_by_name() {
        let text = request().to_json().replace("\"Host\"]", "\"c\"]");
        let why = KernelRequest::from_json(&text).unwrap_err().reason;
        assert!(why.contains("does not serve: c"), "{why}");
    }

    #[test]
    fn a_target_without_its_facts_answer_is_refused() {
        let mut both = request();
        both.targets = vec![Target::Host, Target::Device];
        let why = KernelRequest::from_json(&both.to_json())
            .unwrap_err()
            .reason;
        assert!(
            why.contains("no facts answer for the target Device"),
            "{why}"
        );
    }

    #[test]
    fn the_device_card_loads_and_the_two_targets_read_their_own_numbers() {
        let device =
            super::super::card::device_build(crate::testing::text("cuda_build.json")).unwrap();
        assert_eq!((device.fold.k_max, device.slice_min), (8, 128));
        assert!(device.build().fold_limits.is_some() && !device.build().blend);
        assert!(
            super::super::card::host_build(crate::testing::text("build.json"))
                .unwrap()
                .build()
                .blend
        );
    }

    #[test]
    fn a_request_round_trips() {
        let one = request();
        assert_eq!(KernelRequest::from_json(&one.to_json()), Ok(one));
    }

    #[test]
    fn an_unknown_key_member_or_version_is_refused_by_name() {
        let good = request().to_json();
        let extra = good.replacen("{\"v\":1,", "{\"v\":1,\"surprise\":1,", 1);
        assert_eq!(
            KernelRequest::from_json(&extra),
            Err(unknown("request key surprise"))
        );
        let member = good.replacen("\"Float64\"", "\"Float16\"", 1);
        assert_eq!(
            KernelRequest::from_json(&member),
            Err(unknown("member Float16 of scalar"))
        );
        let missing = good.replacen("\"name\":\"k\",\"scalar\"", "\"scalar\"", 1);
        assert_eq!(
            KernelRequest::from_json(&missing),
            Err(unknown("request key name missing in request"))
        );
        let version = good.replacen("{\"v\":1,", "{\"v\":2,", 1);
        assert_eq!(
            KernelRequest::from_json(&version),
            Err(unknown("request version 2"))
        );
    }

    #[test]
    fn a_sink_with_axes_states_its_payload_and_a_request_without_it_is_refused_by_name() {
        let mut stated = request();
        stated.sink_axes = vec![SinkAxes {
            name: "acc".into(),
            payload: vec![3],
            axes: vec![LaneAxis {
                label: Some("bin".into()),
                extent: Some(9),
            }],
        }];
        let text = stated.to_json();
        assert_eq!(KernelRequest::from_json(&text).unwrap(), stated);
        let why = KernelRequest::from_json(&text.replace("\"payload\":[3],", "")).unwrap_err();
        assert_eq!(why.kind, aether_writers::refuse::RefusalKind::BadRequest);
        assert!(
            why.reason.contains("\"k\"") && why.reason.contains("\"acc\""),
            "{}",
            why.reason
        );
    }
}
