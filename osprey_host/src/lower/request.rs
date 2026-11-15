//! The request of the lowering and its answer: one request per kernel, answered with the units the kernel lowers to. Both are
//! closed-schema JSON on the rules of the writers' contract: every key is stated, a key the reader does not know is refused by
//! name, a required key that is missing is refused by name, and the version is checked first. Nothing here knows how a caller
//! records its kernels.
use super::unit_meta::{LibraryMeta, UnitMeta, Wire};
use aether_writers::contract::types::{
    Access, Dtype, FoldForm, Guard, MapBits, Role, Scalar, SinkPolicy, Target, ThreadBox,
};
use aether_writers::contract::{Contract, Json};
use aether_writers::refuse::{bad, unknown, Refusal};
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
    /// Whether a keyed accumulate is routed to the ordered stages.
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
}

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
    /// The samples one lane group covers when the launch runs over an active set.
    pub group: i64,
    /// The planes the caller declares for the kernel (an output it declares is its own, not the runtime's).
    pub planes: Vec<String>,
    /// The build's settings.
    pub settings: Settings,
    /// The answer of the facts entry for each target of the request, passed back as it was given.
    pub facts: Vec<(Target, String)>,
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
}

/// One unit of the answer: its text per target and its typed meta (which names it and holds its digest).
#[derive(Debug, Clone, PartialEq)]
pub struct UnitAnswer {
    /// The unit's text per target.
    pub texts: Vec<TargetText>,
    /// The unit's meta.
    pub meta: UnitMeta,
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
/// The flag of a value.
fn flag_of(json: &Json, key: &str) -> Result<bool, Refusal> {
    match json {
        Json::Bool(flag) => Ok(*flag),
        _ => Err(wrong(key, "not a flag")),
    }
}
/// The member of a closed set under a key.
fn member_of<T: Debug + Copy>(all: &[T], json: &Json, key: &str) -> Result<T, Refusal> {
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
fn list_of<T>(
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
fn name_of<T: Debug>(member: &T) -> Json {
    Json::Str(format!("{member:?}"))
}
/// A list of integers as a value.
fn ints<T: ToString>(values: &[T]) -> Json {
    Json::Arr(values.iter().map(|n| Json::Num(n.to_string())).collect())
}
/// A list of texts as a value.
fn texts(values: &[String]) -> Json {
    Json::Arr(values.iter().map(|s| Json::Str(s.clone())).collect())
}

impl Member {
    /// Reads a member.
    fn read(json: &Json) -> Result<Member, Refusal> {
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
    fn write(&self) -> Json {
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

impl Body {
    /// Reads a body: `{"Plain": member}`, `{"Chain": [members]}` or `{"Fused": [members]}`.
    fn read(json: &Json) -> Result<Body, Refusal> {
        let members = |j: &Json| list_of(j, "body", Member::read);
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
            lanes: keys.opt("lanes").map(IndexSet::read).transpose()?,
            sink: sink_of(keys.req("sink")?)?,
        };
        keys.done()?;
        Ok(kind)
    }
    /// Writes the kind.
    fn write(&self) -> Json {
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

impl Settings {
    /// Reads the settings; every key but the version is optional and has its default.
    fn read(json: &Json) -> Result<Settings, Refusal> {
        let mut keys = Keys::of(json, "settings")?;
        let pair = |j: &Json| match j.arr() {
            [a, b] => Ok((int_of(a, "launch_bounds")?, int_of(b, "launch_bounds")?)),
            _ => Err(wrong("launch_bounds", "not a pair")),
        };
        let sizes = |j: &Json| -> Result<[u64; 5], Refusal> {
            let values: Vec<u64> = list_of(j, "exports", |n| {
                u64::try_from(int_of(n, "exports")?).map_err(|_| wrong("exports", "not a size"))
            })?;
            values
                .try_into()
                .map_err(|_| wrong("exports", "not five sizes"))
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
            launch_bounds: keys.opt("launch_bounds").map(pair).transpose()?,
            fold_k: keys
                .opt("fold_k")
                .map(|k| int_of(k, "fold_k"))
                .transpose()?,
            warps: match keys.opt("warps") {
                Some(w) => int_of(w, "warps")?,
                None => aether_writers::contract::types::DEFAULT_WARPS,
            },
            team_threads: match keys.opt("team_threads") {
                Some(t) => int_of(t, "team_threads")?,
                None => 0,
            },
            exports: keys.opt("exports").map(sizes).transpose()?,
            literal_table: match keys.opt("literal_table") {
                Some(f) => flag_of(f, "literal_table")?,
                None => false,
            },
        };
        keys.done()?;
        Ok(settings)
    }
    /// Writes the settings, leaving out what has its default.
    fn write(&self) -> Json {
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
        let request = KernelRequest {
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
            one_step: keys.opt("one_step").map(Member::read).transpose()?,
            group: match keys.opt("group") {
                Some(g) => int_of(g, "group")?,
                None => aether_writers::contract::types::DEFAULT_GROUP,
            },
            planes: match keys.opt("planes") {
                Some(p) => list_of(p, "planes", |n| text_of(n, "planes"))?,
                None => vec![],
            },
            settings: match keys.opt("settings") {
                Some(s) => Settings::read(s)?,
                None => Settings::read(&Json::Obj(vec![]))?,
            },
            facts: facts_read(keys.req("facts")?)?,
        };
        keys.done()?;
        for target in &request.targets {
            request.facts_for(*target)?;
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
        if let Some(step) = &self.one_step {
            o.push(("one_step".into(), step.write()))
        }
        if !self.planes.is_empty() {
            o.push(("planes".into(), texts(&self.planes)))
        }
        if self.group != aether_writers::contract::types::DEFAULT_GROUP {
            o.push(("group".into(), Json::Num(self.group.to_string())))
        }
        o.push(("settings".into(), self.settings.write()));
        let answers = self
            .facts
            .iter()
            .map(|(target, answer)| (format!("{target:?}"), Json::Str(answer.clone())))
            .collect();
        o.push(("facts".into(), Json::Obj(answers)));
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
                ("mode".into(), Json::Str(t.mode.clone())),
                ("vectorised".into(), Json::Bool(t.vectorised)),
            ];
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
            Json::Obj(vec![
                (
                    "texts".into(),
                    Json::Arr(u.texts.iter().map(text).collect()),
                ),
                ("meta".into(), u.meta.put()),
            ])
        };
        Json::Obj(vec![
            ("v".into(), Json::Num(REQUEST_VERSION.to_string())),
            (
                "units".into(),
                Json::Arr(self.units.iter().map(unit).collect()),
            ),
            ("library".into(), self.library.put()),
            ("meta_source".into(), Json::Str(self.meta_source.clone())),
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
            let text = TargetText {
                target: member_of(&TARGETS, k.req("target")?, "target")?,
                text: text_of(k.req("text")?, "text")?,
                body: text_of(k.req("body")?, "body")?,
                entry: text_of(k.req("entry")?, "entry")?,
                mode: text_of(k.req("mode")?, "mode")?,
                vectorised: flag_of(k.req("vectorised")?, "vectorised")?,
                thread_box,
            };
            k.done()?;
            Ok(text)
        };
        let read_unit = |j: &Json| -> Result<UnitAnswer, Refusal> {
            let mut k = Keys::of(j, "unit")?;
            let unit = UnitAnswer {
                texts: list_of(k.req("texts")?, "texts", read_text)?,
                meta: UnitMeta::get(k.opt("meta"), "meta")?,
            };
            k.done()?;
            Ok(unit)
        };
        let units = list_of(keys.req("units")?, "units", read_unit)?;
        let library = LibraryMeta::get(keys.opt("library"), "library")?;
        let meta_source = text_of(keys.req("meta_source")?, "meta_source")?;
        keys.done()?;
        Ok(KernelAnswer {
            units,
            library,
            meta_source,
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
            group: 1,
            planes: vec![],
            settings: Settings::read(&Json::Obj(vec![])).unwrap(),
            facts: vec![(Target::Host, "{}".into())],
        }
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
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("facts");
        let device = super::super::card::device_build(&dir).unwrap();
        assert_eq!((device.fold.k_max, device.slice_min), (8, 128));
        assert!(device.build().fold_limits.is_some() && !device.build().blend);
        assert!(super::super::card::host_build(&dir).unwrap().build().blend);
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
}
