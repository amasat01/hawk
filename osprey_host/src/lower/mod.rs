//! The lowering: one request per kernel in, the units the kernel lowers to out. The caller's kernel is already in the context as
//! IR nodes; the request names its sinks and states what the IR cannot (kind, scalar, settings, the facts answer). The lowering
//! takes the canonical walk, reads the walk's arguments back, builds the writers' unit with the numbers of the target's card
//! (`build.json` for the host, `cuda_build.json` for the device) and renders it. This module holds no code of the host library's table, so a second backend library can link it.
//!
//! Written so far: the plain kernel, a segmented or placed one, an active set, two paths, split stages with their team stages, a
//! chain of kernels (cut where the fusion rule of the build's target says) and fused lanes, for the host and the device target.
//! The answer carries one text per target of the request. The warp form is written for the device alone, and a target the
//! library does not serve is refused by its name.
mod active;
mod blocks;
pub mod boxed;
pub mod card;
mod chain;
pub mod derived;
pub mod events;
mod fused;
pub mod image;
pub mod links;
pub mod meta_c;
mod meta_fill;
mod placed;
pub mod request;
mod segment;
mod stage_block;
mod stages;
pub mod unit_meta;
mod warp;

use aether_writers::contract::types::{
    Access, Active, BodySource, Build, Dtype, Exports, Form, FusedLane, Kind, Machine, OneStep,
    Scalar, Segment, SelectWeight, Slot, Target, Unit, Walk, DEFAULT_ROWS,
};
use aether_writers::contract::Contract;
use aether_writers::refuse::{bad, fault, nib, not_written, Refusal};
use aether_writers::{cuda_render, host_render, RxCtx};

use crate::facts::CardSet;
use card::{BuildCard, DeviceCard};
use raptor_ir::canon::{Tt, WalkRes};
use raptor_ir::record::{Ctx, Fault};
use raptor_wings::team::TeamSpec;
use raptor_wings::working_set::working_set_bytes;
use request::{Body, DeclaredSlot, KernelAnswer, KernelRequest, Member, Settings, TargetText};
use request::{UnitAnswer, UnitSet};
use request::{ACCESSES, DTYPES, ROLES};
use std::path::Path;

/// A unit waiting to be written: the body its walk is taken over and what the steps before the write have decided for it.
pub(super) struct Pending {
    /// The body: name, sinks and how the walk is taken.
    pub member: Member,
    /// The run of a segmented input the unit covers.
    pub segment: Option<Segment>,
    /// The single step of a unit written as two paths.
    pub one_step: Option<Member>,
    /// The active set the unit runs over.
    pub active: Option<Active>,
    /// The lanes of a fused group the unit is written from.
    pub lanes: Option<Vec<FusedLane>>,
    /// Whether the unit may be written with a thread box.
    pub boxable: bool,
    /// What the answer's stage block says of the unit, when the unit is a stage of a cut or placed kernel.
    pub note: Option<stages::Note>,
    /// The team the unit runs on, when it is a team stage or a stage that copies operands into team scratch.
    pub team: Option<TeamSpec>,
    /// The warp per lane the unit is written as, when it is.
    pub warp: Option<warp::Warp>,
    /// The dispatch group a branch unit of a segmented split belongs to.
    pub group: Option<segment::Group>,
    /// The units of the segmented form of the unit's automatic group, when the request asked for them: the fused unit's record names
    /// them.
    pub auto_set: Option<segment::Group>,
    /// Whether the unit's text carries the library's self-check exports; a later unit of a table that has them already omits them.
    pub exports_block: bool,
}

/// A wing refusal as the refusal of the request.
pub(super) fn wing(why: Fault) -> Refusal {
    match why {
        Fault::Refused(reason) => bad(reason),
        other => fault(format!("wing: {other:?}")),
    }
}

/// The plane a finishing kernel counts its finished samples in.
pub(super) const FINISHED_PLANE: &str = raptor_wings::cells::FINISHED_PLANE;

/// The finished-count plane a body declares when any of its sinks finishes samples and it does not declare the plane yet.
pub(super) fn finished_slot(ctx: *mut Ctx, member: &Member) -> Option<DeclaredSlot> {
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let context = unsafe { &*ctx };
    let finishing = member.sinks.iter().any(|&sink| {
        context
            .table
            .is_a(context.nodes[sink as usize].class, "Finish")
    });
    let declared = member
        .declared
        .iter()
        .any(|slot| slot.name == FINISHED_PLANE);
    (finishing && !declared).then(|| DeclaredSlot::lookup(FINISHED_PLANE, Dtype::Uint32))
}

/// The bytes of one real of a scalar mode, the one place that says it. A banded kernel is placed and tiled as double
/// precision: the recorded units and the build this replaces read it so for wide placement and warp tiles. The one rule
/// that costed it as single precision (the fusion rule of a chain) is refused for a banded kernel instead.
pub(super) fn real_bytes(scalar: Scalar) -> i64 {
    match scalar {
        Scalar::Float32 => 4,
        Scalar::Float64 | Scalar::Banded => 8,
    }
}

/// Adds `slot` to `held`, each name once. A name held already with another role, type or shape is a conflict, returned
/// as the words that say what the two disagree on.
pub(super) fn merge_slot(held: &mut Vec<DeclaredSlot>, slot: DeclaredSlot) -> Result<(), String> {
    match held.iter().find(|other| other.name == slot.name) {
        None => held.push(slot),
        Some(other) if *other != slot => {
            return Err(format!(
                "slot {} is declared twice as {other:?} and as {slot:?}",
                slot.name
            ))
        }
        Some(_) => {}
    }
    Ok(())
}

/// Refuses a body whose forced access a merged unit would not carry: a forced access is a property of the whole walk and cannot
/// be kept for one part of a merge.
pub(super) fn carries_no_forced_access(body: &Member, what: &str) -> Result<(), Refusal> {
    match body.force {
        None => Ok(()),
        Some(_) => Err(not_written(format!(
            "{what} {} with a forced access",
            body.name
        ))),
    }
}

/// The exempt planes of the bodies of a merge that a unit over `sinks` and `declared` holds, each once. An exemption only acts
/// on a plane the unit's walk holds, so a plane that no node under the sinks carries and no slot declares needs none, and
/// leaving it out is exact.
pub(super) fn held_exempt(
    ctx: *mut Ctx,
    bodies: &[Member],
    sinks: &[u32],
    declared: &[DeclaredSlot],
) -> Result<Vec<String>, Refusal> {
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let under = raptor_wings::stages::names_under(unsafe { &*ctx }, sinks).map_err(wing)?;
    Ok(keep_held(bodies, &under, declared))
}

/// The exempt planes of `bodies` that are among the names `under` the sinks or the `declared` slots, each once.
fn keep_held(bodies: &[Member], under: &[String], declared: &[DeclaredSlot]) -> Vec<String> {
    let mut held: Vec<String> = vec![];
    for plane in bodies.iter().flat_map(|body| &body.exempt) {
        let holds = under.contains(plane) || declared.iter().any(|slot| &slot.name == plane);
        if holds && !held.contains(plane) {
            held.push(plane.clone());
        }
    }
    held
}

/// Faults on units that share a name, which the lowering itself made (the caller's own names are checked when the request is read):
/// the answer names each unit, and the stage block and the caller key on it.
fn distinct_names(pending: &[Pending]) -> Result<(), Refusal> {
    let mut seen = std::collections::HashSet::new();
    match pending.iter().find(|unit| !seen.insert(&unit.member.name)) {
        Some(unit) => Err(fault(format!(
            "the lowering made two units named {} (a piece of a chain and a stage of another piece)",
            unit.member.name
        ))),
        None => Ok(()),
    }
}

/// One step of the pipeline: the pending units in, the pending units out. A step decides on `cards.deciding`.
type Step = fn(*mut Ctx, &KernelRequest, &Cards, Vec<Pending>) -> Result<Vec<Pending>, Refusal>;

/// The steps every request goes through, in order. A step that has nothing to decide for a unit passes it on.
const STEPS: [Step; 5] = [
    segment::split,
    blocks::plain,
    warp::slice,
    stages::split,
    active::set,
];

/// The cards a request is lowered with: the host's always, the device's when a target asks for it, and the numbers the steps
/// decide on (the device's when the request has a device target, else the host's).
struct Cards {
    /// The host build card.
    host: BuildCard,
    /// The device build card, loaded when the request names the device.
    device: Option<DeviceCard>,
    /// The card the steps decide on.
    deciding: BuildCard,
    /// The command that makes the device image, when the request names the device.
    image: Option<image::DeviceImage>,
    /// The name of the device card's wide class for the architecture the request's device facts state, whose thresholds place the
    /// kernel's wide values; none when the request has no device target.
    wide_class: Option<String>,
}

impl Cards {
    /// The cards the request needs, from the data documents in `docs`.
    fn load(request: &KernelRequest, docs: &CardSet) -> Result<Cards, Refusal> {
        let host = card::host_build_of(docs)?;
        let device = request
            .targets
            .contains(&Target::Device)
            .then(|| card::device_build_of(docs))
            .transpose()?;
        if let (Some(device), Some(bounds)) = (&device, request.settings.launch_bounds) {
            device.check_launch_bounds(bounds)?;
        }
        let deciding = match &device {
            Some(device) => device.deciding(&host, &device_arch(request)?)?,
            None => host.clone(),
        };
        let wide_class = match &device {
            Some(device) => Some(device.wide_class(&device_arch(request)?)?.name.clone()),
            None => None,
        };
        let image = match request.image_format {
            Some(format) if request.targets.contains(&Target::Device) => {
                Some(image::DeviceImage::of_request(request, format)?)
            }
            _ => None,
        };
        Ok(Cards {
            host,
            device,
            deciding,
            image,
            wide_class,
        })
    }

    /// How the device card runs a dense contraction of the request's scalar type; `Gemm` for a request without a device target.
    fn contraction_route(
        &self,
        request: &KernelRequest,
    ) -> Result<card::ContractionRoute, Refusal> {
        let Some(device) = &self.device else {
            return Ok(card::ContractionRoute::Gemm);
        };
        let scalar = match request.scalar {
            Scalar::Float32 => "float32",
            _ => "float64",
        };
        Ok(device.contraction_route(&device_arch(request)?, scalar))
    }

    /// The build numbers of a target's card, before the caller's settings are laid over them.
    fn numbers(&self, target: Target) -> Result<Build, Refusal> {
        match (target, &self.device) {
            (Target::Host, _) => Ok(self.host.build()),
            (Target::Device, Some(device)) => Ok(device.build()),
            (Target::Device, None) => Err(fault("a device target without its card")),
        }
    }

    /// The thread box lanes the card of a target holds, by scalar type name.
    fn box_lanes(&self, target: Target) -> Result<&[(String, i64)], Refusal> {
        match (target, &self.device) {
            (Target::Host, _) => Ok(&self.host.box_lanes),
            (Target::Device, Some(device)) => Ok(&device.box_lanes),
            (Target::Device, None) => Err(fault("a device target without its card")),
        }
    }
}

/// Refuses a request whose facts answers were made under other cards than the request states: a request with cards needs
/// answers that state the digest of those cards, and a request without cards needs answers that state none.
fn facts_under_cards(answers: &[(Target, String)], docs: &CardSet) -> Result<(), Refusal> {
    let ours = docs.digest().map_err(fault)?;
    for (target, answer) in answers {
        let stated = Contract::parse(answer).ok().and_then(|parsed| {
            parsed
                .0
                .get("cards")
                .and_then(aether_writers::contract::Json::str)
                .map(String::from)
        });
        if stated != ours {
            return Err(bad(format!(
                "the {target:?} facts answer was made under other cards than the request's `cards`; ask the facts call with the same cards"
            )));
        }
    }
    Ok(())
}

/// The architecture the device facts answer of the request states.
pub(super) fn device_arch(request: &KernelRequest) -> Result<String, Refusal> {
    let machine = Contract::parse(request.facts_for(Target::Device)?)?.0;
    machine
        .get("device_arch")
        .and_then(aether_writers::contract::Json::str)
        .map(String::from)
        .ok_or_else(|| bad("the device facts answer names no architecture"))
}

/// Refuses a refine unit over a stop rule that cannot be refined.
fn refine_stop_rule(ctx: *mut Ctx, request: &KernelRequest) -> Result<(), Refusal> {
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    events::check_refine(unsafe { &*ctx }, request)
}

/// The answer to a request, from the kernel's IR in `ctx` and the data files in the directory `facts_dir`. A request with derived kernels is
/// answered as one table: the units of the kernel first, then those of each derived kernel in list order.
pub fn lower(ctx: *mut Ctx, request_text: &str, facts_dir: &Path) -> Result<KernelAnswer, Refusal> {
    let request = KernelRequest::from_json(request_text)?;
    let docs = &CardSet::new(facts_dir, &request.cards);
    scope(&request)?;
    known_sinks(ctx, &request)?;
    refine_stop_rule(ctx, &request)?;
    facts_under_cards(&request.facts, docs)?;
    let cards = Cards::load(&request, docs)?;
    let mut units = kernel_units(ctx, &request, &cards, docs, request.exports_block)?;
    if !request.derived.is_empty() {
        units.extend(derived_units(ctx, &request, &cards, docs, &units)?);
    }
    meta_fill::unit_digests(&mut units);
    let library = unit_meta::LibraryMeta::current();
    let meta_source = meta_c::source(&library, &units.iter().map(|u| &u.meta).collect::<Vec<_>>())?;
    Ok(KernelAnswer {
        units,
        library,
        meta_source,
        working_set_bytes: working_set(ctx, &request, &cards)?,
        wide_class: cards.wide_class.clone(),
    })
}

/// The working-set figure of the request's body: the walk over every sink of its members (the kernel as the caller wrote it,
/// before any step rewrites it), counted by the wing under the deciding card's fold threshold and the settings' fold form, the
/// same two numbers the steps decide rolled sites with.
fn working_set(ctx: *mut Ctx, request: &KernelRequest, cards: &Cards) -> Result<i64, Refusal> {
    let members = match &request.body {
        Body::Plain(member) => std::slice::from_ref(member),
        Body::Chain(members) | Body::Fused(members) => members.as_slice(),
    };
    let whole = Member {
        name: request.name.clone(),
        sinks: members
            .iter()
            .flat_map(|m| m.sinks.iter().copied())
            .collect(),
        declared: vec![],
        force: None,
        exempt: vec![],
    };
    let (walk, _) = canonical(ctx, &whole)?;
    let order = walk_of(ctx, walk)?.order.clone();
    let form = format!("{:?}", request.settings.fold_form).to_lowercase();
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let context = unsafe { &*ctx };
    working_set_bytes(context, &order, Some(cards.deciding.rolled_fold_min), &form).map_err(|why| {
        match why {
            Fault::Refused(reason) => bad(reason),
            other => fault(format!("working set: {other:?}")),
        }
    })
}

/// The C++ text of a set of units built into one library: the headers the exports block needs, the block once, and the library
/// record over all the units in the order given. The text of the units of one answer is `SET_HEADERS`, that answer's own exports
/// block (the one its first unit carries when asked with the block) and its `meta_source`. Two units of one name are refused, as is
/// an empty list; nothing is read from the environment: the sizes are the five stated or the ones the headers give.
pub fn set_text(input: &str) -> Result<String, Refusal> {
    let set = UnitSet::from_json(input)?;
    if set.units.is_empty() {
        return Err(bad("a unit set without a unit"));
    }
    for (at, unit) in set.units.iter().enumerate() {
        if let Some(first) = set.units[..at].iter().position(|u| u.name == unit.name) {
            return Err(bad(format!(
                "units {first} and {at} of the set are both named {}",
                unit.name
            )));
        }
    }
    let sizes = set.exports.map_or(Exports::Derived, Exports::Sizes);
    let block = aether_writers::unit::exports(&sizes, false)?;
    let meta = meta_c::source(
        &unit_meta::LibraryMeta::current(),
        &set.units.iter().collect::<Vec<_>>(),
    )?;
    Ok(format!("{SET_HEADERS}{block}{meta}"))
}

/// The headers the exports block of a set text needs, which a unit text has from its own prelude.
pub const SET_HEADERS: &str = "#include \"plugin/gref_layout.h\"\n";

/// The units of the kernels derived from the request's kernel, each lowered as a kernel of its own over the shared IR with the
/// request's settings, its self-check exports omitted, and its stated facts resolved against the units lowered before it.
fn derived_units(
    ctx: *mut Ctx,
    request: &KernelRequest,
    cards: &Cards,
    docs: &CardSet,
    primal: &[UnitAnswer],
) -> Result<Vec<UnitAnswer>, Refusal> {
    let single = |units: &[UnitAnswer], name: &str| match units {
        [_] => Ok(()),
        _ => Err(not_written(format!(
            "derived kernel {name}: a set over a kernel that lowers to {} units",
            units.len()
        ))),
    };
    single(primal, &request.name)?;
    let mut all: Vec<UnitAnswer> = vec![];
    let mut twin_at: Option<usize> = None;
    for kernel in &request.derived {
        let sub = kernel.request.clone();
        let mut units = kernel_units(ctx, &sub, cards, docs, false)?;
        single(&units, kernel.name())?;
        let at = primal.len() + all.len();
        let twin = twin_at.map(|i| &all[i - primal.len()].meta);
        let context = derived::Context {
            primal: &primal[0].meta,
            of_unit: 0,
            twin,
        };
        units[0].meta.derived = Some(derived::resolve(
            &kernel.stated(),
            &units[0].meta,
            &context,
        )?);
        if matches!(
            kernel.role,
            unit_meta::DerivedRole::Tape | unit_meta::DerivedRole::TapeFused
        ) {
            twin_at = Some(at);
        }
        all.extend(units);
    }
    let mut seen = std::collections::HashSet::new();
    let names = primal
        .iter()
        .map(|u| (u.meta.name.clone(), None))
        .chain(all.iter().map(|u| {
            (
                u.meta.name.clone(),
                u.meta.derived.as_ref().map(|d| d.role as i32),
            )
        }));
    for key in names {
        if !seen.insert(key.clone()) {
            return Err(fault(format!(
                "the lowering made two units named {} with one role",
                key.0
            )));
        }
    }
    Ok(all)
}

/// The units one kernel lowers to, with their meta; the first kernel of a table keeps the self-check exports in its texts, the
/// later ones omit them.
fn kernel_units(
    ctx: *mut Ctx,
    request: &KernelRequest,
    cards: &Cards,
    docs: &CardSet,
    exports_block: bool,
) -> Result<Vec<UnitAnswer>, Refusal> {
    let mut pending = match &request.body {
        Body::Plain(member) => vec![Pending::of(member, request.one_step.as_ref())],
        Body::Chain(members) => chain::expand(ctx, request, members, docs)?,
        Body::Fused(lanes) => vec![fused::expand(ctx, lanes, &request.name)?],
    };
    for unit in &mut pending {
        for body in std::iter::once(&mut unit.member).chain(unit.one_step.as_mut()) {
            body.declared.extend(finished_slot(ctx, body));
        }
    }
    for step in STEPS {
        pending = step(ctx, request, cards, pending)?;
    }
    distinct_names(&pending)?;
    for unit in &mut pending {
        unit.exports_block = exports_block;
    }
    let blocks = stages::blocks(&pending);
    let mut units = pending
        .into_iter()
        .map(|unit| write(ctx, request, cards, unit))
        .collect::<Result<Vec<_>, _>>()?;
    for block in &blocks {
        meta_fill::stages(&mut units, block, cards.deciding.scatter_max_chunks)?;
    }
    if let Some(sweep) = segment::sweep_request(ctx, request)? {
        units.extend(kernel_units(ctx, &sweep, cards, docs, exports_block)?);
    }
    Ok(units)
}

impl Pending {
    /// A unit written from one body (and the single step beside it) as a plain loop, with a thread box where the facts allow.
    pub(super) fn of(member: &Member, one_step: Option<&Member>) -> Pending {
        Pending {
            member: member.clone(),
            segment: None,
            one_step: one_step.cloned(),
            active: None,
            lanes: None,
            boxable: true,
            note: None,
            team: None,
            warp: None,
            group: None,
            auto_set: None,
            exports_block: true,
        }
    }
}

/// The single step of a two-path unit: its sinks and the walk its own arguments make.
fn single_step(ctx: *mut Ctx, step: &Member) -> Result<OneStep, Refusal> {
    let (walk, _) = canonical(ctx, step)?;
    Ok(OneStep {
        sinks: step.sinks.clone(),
        walk: Some(typed_walk(ctx, walk)?),
    })
}

/// Writes one pending unit for every target of the request: its walk once, then per target its writer unit, its box and its text.
fn write(
    ctx: *mut Ctx,
    request: &KernelRequest,
    cards: &Cards,
    pending: Pending,
) -> Result<UnitAnswer, Refusal> {
    let (walk, digest) = canonical(ctx, &pending.member)?;
    let mut texts = vec![];
    for &target in &request.targets {
        texts.push(write_target(ctx, request, cards, &pending, (walk, target))?);
    }
    let meta = meta_fill::unit(&meta_fill::Parts {
        ctx,
        request,
        card: &cards.deciding,
        image: cards.image.as_ref(),
        pending: &pending,
        walk,
        digest: &digest,
        typed: &typed_walk(ctx, walk)?,
        texts: &texts,
        route: cards.contraction_route(request)?,
    })?;
    Ok(UnitAnswer { texts, meta })
}

/// The text of one pending unit for one target, from its walk: the writer unit, its thread box and the target's writer.
fn write_target(
    ctx: *mut Ctx,
    request: &KernelRequest,
    cards: &Cards,
    pending: &Pending,
    (walk, target): (u32, Target),
) -> Result<TargetText, Refusal> {
    let mut unit = unit_of(
        request,
        &pending.member,
        typed_walk(ctx, walk)?,
        cards,
        target,
    )?;
    // a set has one host library, so its block is in the set text; a device unit's image is a module of its own and keeps its block
    if !pending.exports_block && target == Target::Host {
        unit.exports = Exports::Text(String::new());
    }
    unit.segment = pending.segment.clone();
    unit.one_step = pending
        .one_step
        .as_ref()
        .map(|step| single_step(ctx, step))
        .transpose()?;
    unit.active = pending.active.clone();
    if let Some(spec) = &pending.team {
        let (form, shape) = stages::team_form(spec, pending.active.clone())?;
        if spec.grid != raptor_wings::team::Grid::Lanes {
            // a stage over keys reads the active set itself; its launch takes none
            unit.active = None;
        }
        unit.form = form.unwrap_or(Form::Plain);
        unit.team = Some(shape);
    }
    if let Some(lanes) = &pending.lanes {
        unit.form = Form::Fused(lanes.clone());
    }
    if let Some(warp) = &pending.warp {
        warp.shape(&mut unit);
    }
    if pending.boxable {
        let walked = unit
            .walk
            .as_ref()
            .ok_or_else(|| fault("a unit without a walk"))?;
        let box_lanes = cards.box_lanes(target)?;
        unit.thread_box =
            boxed::thread_box(ctx, request, &pending.member.sinks, walked, box_lanes)?;
    }
    let json = with_facts(&unit, request.facts_for(target)?)?;
    let render = match target {
        Target::Host => host_render,
        Target::Device => cuda_render,
    };
    let source = render(ctx as *mut RxCtx, walk, &json)?;
    let meta = source.meta;
    Ok(TargetText {
        target,
        text: source.text,
        body: source.body,
        entry: source.entry,
        facts: source.facts,
        mode: meta.mode.to_string(),
        vectorised: meta.vectorised,
        thread_box: meta.thread_box.filter(|b| b.axis.is_some()),
    })
}

/// Refuses a sink handle that is no node of the context: the request names the nodes it writes, and the lowering indexes them.
fn known_sinks(ctx: *mut Ctx, request: &KernelRequest) -> Result<(), Refusal> {
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let nodes = unsafe { ctx.as_ref() }
        .ok_or_else(|| fault("a null context"))?
        .nodes
        .len();
    let members: Vec<&Member> = match &request.body {
        Body::Plain(member) => vec![member],
        Body::Chain(members) | Body::Fused(members) => members.iter().collect(),
    };
    let steps = request.one_step.iter().chain(members);
    for sink in steps.flat_map(|member| member.sinks.iter()) {
        if *sink as usize >= nodes {
            return Err(bad(format!(
                "request key sinks: node {sink} is not in the context ({nodes} nodes)"
            )));
        }
    }
    Ok(())
}

/// Refuses by name what the written shapes do not cover: a request without a target.
fn scope(request: &KernelRequest) -> Result<(), Refusal> {
    if request.targets.is_empty() {
        return Err(bad("a request without a target"));
    }
    Ok(())
}

/// The refusal a failed walk stands for: a construct the core refuses is not written, a bad argument is a bad request.
fn walk_refusal(why: Fault) -> Refusal {
    match why {
        Fault::Refused(reason) => nib(reason),
        Fault::BadArg(reason) => bad(reason),
        other => fault(format!("walk: {other:?}")),
    }
}

/// The IR core's slot key and type of a declared slot.
fn declared_leaf(slot: &DeclaredSlot) -> ((String, String), Tt) {
    (
        (slot.role.name().into(), slot.name.clone()),
        Tt {
            dtype: slot.dtype.name().into(),
            shape: slot.shape.clone(),
            tag: slot.tag.clone(),
        },
    )
}

/// The kernel's canonical walk over the member's sinks and declared slots (the wing adds the companion, finished-count and
/// branch-mix slots), and the walk's digest.
fn canonical(ctx: *mut Ctx, member: &Member) -> Result<(u32, String), Refusal> {
    let declared: Vec<_> = member.declared.iter().map(declared_leaf).collect();
    let force = member.force.map(|a| a.name());
    // SAFETY: `ctx` is the live context the caller holds; nothing else reads it during the call.
    let context = unsafe { &mut *ctx };
    let walk =
        raptor_wings::cells::kernel_walk(context, &member.sinks, &declared, force, &member.exempt)
            .map_err(walk_refusal)?;
    let digest = context.walk(walk).map_err(walk_refusal)?.digest.clone();
    Ok((walk, digest))
}

/// The walk the context holds under the handle `walk`.
pub(super) fn walk_of<'a>(ctx: *mut Ctx, walk: u32) -> Result<&'a WalkRes, Refusal> {
    // SAFETY: `ctx` is the live context the caller holds; it is only read here and outlives the lowering step.
    unsafe { &*ctx }.walk(walk).map_err(walk_refusal)
}

/// The walk's leaves read back as slots, in argument order: each with its role, name, element type, shape and tag.
fn leaf_slots(ctx: *mut Ctx, walk: u32) -> Result<Vec<DeclaredSlot>, Refusal> {
    let shape = || fault("a walk leaf with a role or an element type this library does not know");
    walk_of(ctx, walk)?
        .leaves
        .iter()
        .map(|((role, name), ttype)| {
            Ok(DeclaredSlot {
                role: *ROLES.iter().find(|r| r.name() == role).ok_or_else(shape)?,
                name: name.clone(),
                dtype: *DTYPES
                    .iter()
                    .find(|d| d.name() == ttype.dtype)
                    .ok_or_else(shape)?,
                shape: ttype.shape.clone(),
                tag: ttype.tag.clone(),
            })
        })
        .collect()
}

/// The names of the planes whose launch-start value the walk reads.
fn prior_reads(ctx: *mut Ctx, walk: u32) -> Result<Vec<String>, Refusal> {
    Ok(walk_of(ctx, walk)?.prior.clone())
}

/// The walk's arguments, access class and finish, read back as the writers' typed walk.
fn typed_walk(ctx: *mut Ctx, walk: u32) -> Result<Walk, Refusal> {
    let slots = leaf_slots(ctx, walk)?
        .into_iter()
        .enumerate()
        .map(|(index, slot)| Slot {
            role: slot.role,
            name: slot.name,
            dtype: slot.dtype,
            shape: slot.shape,
            index: index as i64,
        })
        .collect();
    let read = walk_of(ctx, walk)?;
    let access: Access = *ACCESSES
        .iter()
        .find(|a| a.name() == read.access)
        .ok_or_else(|| fault(format!("a walk with the access class {}", read.access)))?;
    Ok(Walk {
        access,
        finish: read.finish.clone(),
        slots,
    })
}

/// The writers' unit for a plain kernel on a target, with a machine that `with_facts` replaces.
fn unit_of(
    request: &KernelRequest,
    member: &Member,
    walk: Walk,
    cards: &Cards,
    target: Target,
) -> Result<Unit, Refusal> {
    let settings = &request.settings;
    Ok(Unit {
        name: member.name.clone(),
        target,
        scalar: request.scalar,
        kind: Kind {
            guard: request.kind.guard.clone(),
            lanes: request.kind.lanes.is_some(),
            sink: request.kind.sink.clone(),
        },
        form: Form::Plain,
        sinks: member.sinks.clone(),
        walk: Some(walk),
        body: BodySource::Walk,
        active: None,
        segment: None,
        thread_box: None,
        one_step: None,
        rows: DEFAULT_ROWS,
        team: None,
        build: build_of(settings, cards, target)?,
        machine: Machine {
            isa: None,
            vector_isa: None,
            exact_libm: false,
            routed_pow: false,
            caches: None,
            select_weight: SelectWeight {
                float64: None,
                float32: None,
            },
        },
        exports: settings.exports.map_or(Exports::Derived, Exports::Sizes),
        literal_table: settings.literal_table,
    })
}

/// The build's numbers for a target: the numbers of the target's card with the caller's settings laid over them.
fn build_of(settings: &Settings, cards: &Cards, target: Target) -> Result<Build, Refusal> {
    Ok(Build {
        fold_form: settings.fold_form,
        warps: settings.warps,
        launch_bounds: settings.launch_bounds,
        fold_k: settings.fold_k.filter(|k| *k != 0),
        ..cards.numbers(target)?
    })
}

/// The unit's contract JSON with its machine member replaced by the facts answer (less the architecture, the compile commands and
/// the digest of the cards an answer names).
fn with_facts(unit: &Unit, facts: &str) -> Result<String, Refusal> {
    let mut top = Contract::parse(&unit.to_json())?.0;
    let mut machine = Contract::parse(facts)?.0;
    if let aether_writers::contract::Json::Obj(stated) = &mut machine {
        // the architecture is the lowering's to read, and the digest of stated cards is no machine fact: the writers' machine
        // record is closed, so none of them may reach it
        stated.retain(|(key, _)| {
            !["device_arch", "device_build", "host_build", "cards"].contains(&key.as_str())
        });
    }
    let aether_writers::contract::Json::Obj(members) = &mut top else {
        return Err(fault("a unit that is not an object"));
    };
    let slot = members
        .iter_mut()
        .find(|(key, _)| key == "machine")
        .ok_or_else(|| fault("a unit without a machine member"))?;
    slot.1 = machine;
    Ok(top.to_text())
}

#[cfg(test)]
mod tests {
    //! The scalar width has one answer, and a merge that would drop something is refused by name.
    use super::*;
    use crate::facts::{answer, probe, Stated};

    /// The shipped facts directory.
    fn shipped() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("facts")
    }

    /// The shipped build card with the lean width changed to 64, as a stated document.
    fn lean_card() -> Vec<(String, String)> {
        let text = std::fs::read_to_string(shipped().join("build.json")).unwrap();
        vec![(
            "build.json".into(),
            text.replace("\"lean_width\": 4", "\"lean_width\": 64"),
        )]
    }

    /// The device facts answer made under the cards `stated`.
    fn device_facts(stated: &[(String, String)]) -> String {
        let machine = Stated {
            cards: stated.to_vec(),
            ..Stated::default()
        };
        let compiler =
            "/usr/local/cuda/bin/nvcc || Cuda compilation tools, release 12.6, V12.6.77 || 0";
        answer(&shipped(), "cuda", compiler, &probe(), &machine).unwrap()
    }

    /// The device build card the cards state, with `old` of the shipped text replaced by `new`.
    fn device_card_with(old: &str, new: &str) -> DeviceCard {
        let text = std::fs::read_to_string(shipped().join("cuda_build.json")).unwrap();
        assert!(text.contains(old), "the shipped card holds {old}");
        let stated = [("cuda_build.json".to_string(), text.replacen(old, new, 1))];
        let docs = CardSet::new(&shipped(), &stated);
        card::device_build(&docs.text("cuda_build.json").unwrap()).unwrap()
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
        for bad_bounds in [(100, 1), (16, 1), (128, 17)] {
            let why = shipped_card.check_launch_bounds(bad_bounds).unwrap_err();
            assert!(why.reason.contains("launch_bounds"), "{}", why.reason);
        }
    }

    /// Oracle: the shipped build card states a lean width of 4.
    #[test]
    fn a_build_card_stated_in_the_request_replaces_the_file_for_that_request() {
        let kept = CardSet::new(&shipped(), &[]);
        let stated = CardSet::new(&shipped(), &lean_card());
        let host = |docs: &CardSet| card::host_build(&docs.text("build.json").unwrap()).unwrap();
        assert_eq!(host(&kept).lean_width, 4);
        assert_eq!(host(&stated).lean_width, 64);
    }

    #[test]
    fn a_facts_answer_must_state_the_digest_of_the_request_cards_and_state_none_when_the_request_has_none(
    ) {
        let files = CardSet::new(&shipped(), &[]);
        let lean = CardSet::new(&shipped(), &lean_card());
        let stated = [(Target::Device, device_facts(&lean_card()))];
        let plain = [(Target::Device, device_facts(&[]))];
        assert!(facts_under_cards(&stated, &lean).is_ok());
        assert!(facts_under_cards(&plain, &files).is_ok());
        for (answers, docs) in [(&plain, &lean), (&stated, &files)] {
            let why = facts_under_cards(answers, docs).unwrap_err();
            assert!(why.reason.contains("other cards"), "{}", why.reason);
        }
        let hand_written = [(Target::Host, "{}".to_string())];
        assert!(facts_under_cards(&hand_written, &files).is_ok());
        assert!(facts_under_cards(&hand_written, &lean).is_err());
    }

    /// A request cannot reach the lowering with a device document its facts call refused: the answer a request needs carries the
    /// digest of the request's cards, and the facts call states none for a document it does not accept.
    #[test]
    fn a_stated_document_the_facts_call_refuses_has_no_answer_to_lower_under() {
        let text = std::fs::read_to_string(shipped().join("cuda.json")).unwrap();
        let broken = [(
            "cuda.json".to_string(),
            format!("{{\"nonsense\": 1,{}", &text[1..]),
        )];
        let machine = Stated {
            cards: broken.to_vec(),
            ..Stated::default()
        };
        let compiler =
            "/usr/local/cuda/bin/nvcc || Cuda compilation tools, release 12.6, V12.6.77 || 0";
        assert!(answer(&shipped(), "cuda", compiler, &probe(), &machine).is_err());
        let plain = [(Target::Device, device_facts(&[]))];
        let why = facts_under_cards(&plain, &CardSet::new(&shipped(), &broken)).unwrap_err();
        assert!(why.reason.contains("other cards"), "{}", why.reason);
    }

    /// Oracle: a stated copy of the shipped file has the digest of the text it states, which differs from that of another text.
    #[test]
    fn the_digest_is_none_without_stated_cards_and_follows_the_stated_text_with_them() {
        let text = std::fs::read_to_string(shipped().join("cuda.json")).unwrap();
        let same = CardSet::new(&shipped(), &[("cuda.json".into(), text)]);
        assert_eq!(CardSet::new(&shipped(), &[]).digest().unwrap(), None);
        let shipped_text = same.digest().unwrap().unwrap();
        assert_ne!(
            CardSet::new(&shipped(), &lean_card())
                .digest()
                .unwrap()
                .unwrap(),
            shipped_text
        );
    }
    use aether_writers::contract::types::Role;
    use aether_writers::refuse::RefusalKind;

    /// Oracle: `working_set_bytes` of the Python model (the old IR package), recorded per kernel in the wings' oracle file under
    /// the shipped card's fold threshold (16) and form "auto". The answer's figure is that of the kernel's own sinks.
    #[test]
    fn the_answer_states_the_working_set_the_python_model_gives_for_the_kernel() {
        use raptor_ir::door::{rx_ctx_free, rx_ctx_new};
        use raptor_ir::dump::rx_load;
        let oracle = rcore::json::parse(include_str!(
            "../../../raptor/rs/wings/tests/data/working_set_oracle.json"
        ))
        .unwrap();
        let docs = CardSet::new(&shipped(), &[]);
        let mut checked = 0;
        for case in oracle.get("cases").unwrap().arr() {
            let name = case.get("name").and_then(rcore::json::J::str).unwrap();
            let text = case.get("dump").and_then(rcore::json::J::str).unwrap();
            let ctx = rx_ctx_new();
            let (mut handles, mut n) = (vec![0u32; 4096], 0usize);
            // SAFETY: `ctx` is the live context made above; the buffer holds `cap` handles.
            let status = unsafe {
                rx_load(
                    ctx,
                    text.as_ptr().cast(),
                    text.len(),
                    handles.as_mut_ptr(),
                    handles.len(),
                    &mut n,
                )
            };
            assert_eq!(status, 0, "{name}");
            let sinks = case.get("sinks").unwrap().arr();
            let request = KernelRequest {
                body: Body::Plain(Member {
                    name: name.into(),
                    sinks: sinks.iter().map(|p| handles[p.int() as usize]).collect(),
                    declared: vec![],
                    force: None,
                    exempt: vec![],
                }),
                ..plain_request()
            };
            let cards = Cards::load(&request, &docs).unwrap();
            let want = case.get("figures").unwrap().arr()[0]
                .get("bytes")
                .unwrap()
                .int();
            assert_eq!(working_set(ctx, &request, &cards).unwrap(), want, "{name}");
            checked += 1;
            // SAFETY: the context is not used again.
            unsafe { rx_ctx_free(ctx) };
        }
        assert_eq!(checked, 15);
    }

    /// The cards the shipped files give, plus a second wide class measured on capability 70 (a copy of the shipped class's
    /// thresholds), as the old tests of the class choice stated it: `sm_61` and `sm_75` are nearest the shipped class, `sm_80` is
    /// nearest the class of capability 70.
    fn two_class_cards() -> Vec<(String, String)> {
        let text = std::fs::read_to_string(shipped().join("cuda_build.json")).unwrap();
        let first = "\"wide_classes\": {";
        assert!(text.contains(first), "the shipped card holds {first}");
        let second = format!(
            "{first}\"fake70\": {{\"capability\": 70, \"runtime_min\": {{\"read\": {{\"4\": 16, \"8\": 2}}, \"slot_read\": {{\"4\": 16, \"8\": 4}}, \"rotate\": {{\"4\": 16, \"8\": 0}}, \"write\": {{\"4\": 96, \"8\": 8}}, \"accum\": {{\"4\": 4096, \"8\": 4096}}}}, \"wide_min\": {{\"4\": 128, \"8\": 64}}}},"
        );
        vec![("cuda_build.json".into(), text.replacen(first, &second, 1))]
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
            cards: stated.to_vec(),
            device_arch: arch.map(String::from),
            ..Stated::default()
        };
        let compiler =
            "/usr/local/cuda/bin/nvcc || Cuda compilation tools, release 12.6, V12.6.77 || 0";
        answer(&shipped(), "cuda", compiler, &probe(), &machine).unwrap()
    }

    /// Oracle: the old tests of the class choice (`test_build_cards.py`): with a class measured on capability 70 beside the shipped
    /// one, sm_61 is its own class, sm_75 (FP64 1:32, 1024 threads) is one step from Pascal and sm_80 (1:2, 2048 threads) is nearest
    /// the class of capability 70.
    #[test]
    fn the_answer_names_the_wide_class_nearest_the_architecture_of_the_device_facts() {
        let stated = two_class_cards();
        let docs = CardSet::new(&shipped(), &stated);
        for (arch, class) in [("sm_61", "sm61"), ("sm_75", "sm61"), ("sm_80", "fake70")] {
            let request = device_request(facts_of_arch(Some(arch), &stated));
            let cards = Cards::load(&request, &docs).unwrap();
            assert_eq!(cards.wide_class.as_deref(), Some(class), "{arch}");
        }
    }

    /// Oracle: the old device card's `default_arch` is `sm_61` (a fact of the card the library applies when the caller states none),
    /// so the class of the default is the shipped one.
    #[test]
    fn a_device_request_that_states_no_architecture_takes_the_class_of_the_cards_default() {
        let stated = two_class_cards();
        let request = device_request(facts_of_arch(None, &stated));
        let cards = Cards::load(&request, &CardSet::new(&shipped(), &stated)).unwrap();
        assert_eq!(cards.wide_class.as_deref(), Some("sm61"));
    }

    #[test]
    fn a_device_facts_answer_naming_an_unknown_architecture_is_refused_by_name() {
        let request = device_request(r#"{"device_arch": "cuda"}"#.into());
        let Err(why) = Cards::load(&request, &CardSet::new(&shipped(), &[])) else {
            panic!("an architecture that is not one was taken");
        };
        assert!(
            why.reason.contains("is not a device architecture"),
            "{}",
            why.reason
        );
    }

    #[test]
    fn a_request_with_no_device_target_states_no_wide_class() {
        let cards = Cards::load(&plain_request(), &CardSet::new(&shipped(), &[])).unwrap();
        assert_eq!(cards.wide_class, None);
    }

    /// A host request with no body of its own, the caller's settings all at their defaults.
    fn plain_request() -> KernelRequest {
        KernelRequest::from_json(
            r#"{"v": 1, "name": "k", "scalar": "Float64", "targets": ["Host"], "kind": {"guard": "Default", "active_set": false, "sink": "Plain"}, "body": {"Plain": {"name": "k", "sinks": [0], "declared": [], "exempt": []}}, "facts": {"Host": "{}"}}"#,
        )
        .unwrap()
    }

    /// A declared slot of the given type.
    fn slot(name: &str, dtype: Dtype) -> DeclaredSlot {
        DeclaredSlot {
            role: Role::Mutable,
            name: name.into(),
            dtype,
            shape: vec![2],
            tag: None,
        }
    }

    /// A body with no sinks.
    fn body(name: &str) -> Member {
        Member {
            name: name.into(),
            sinks: vec![],
            declared: vec![],
            force: None,
            exempt: vec![],
        }
    }

    #[test]
    fn a_scalar_mode_has_one_byte_width() {
        assert_eq!(real_bytes(Scalar::Float32), 4);
        assert_eq!(real_bytes(Scalar::Float64), 8);
        assert_eq!(real_bytes(Scalar::Banded), 8);
    }

    #[test]
    fn a_slot_named_twice_is_kept_once_or_refused() {
        let mut held = vec![];
        merge_slot(&mut held, slot("out", Dtype::F64)).unwrap();
        merge_slot(&mut held, slot("out", Dtype::F64)).unwrap();
        assert_eq!(held.len(), 1);
        let why = merge_slot(&mut held, slot("out", Dtype::F32)).unwrap_err();
        assert!(why.contains("slot out is declared twice"), "{why}");
        assert_eq!(held.len(), 1);
    }

    #[test]
    fn a_forced_access_is_refused_where_it_would_be_dropped() {
        assert!(carries_no_forced_access(&body("a"), "a member").is_ok());
        let mut forced = body("a");
        forced.force = Some(Access::SampleLocal);
        let refusal = carries_no_forced_access(&forced, "a member").unwrap_err();
        assert_eq!(refusal.kind, RefusalKind::NotWritten);
        assert!(refusal.reason.contains("a member"), "{}", refusal.reason);
    }

    #[test]
    fn an_exempt_plane_goes_to_the_piece_that_holds_it() {
        let mut first = body("a");
        first.exempt = vec!["p".into(), "q".into()];
        let mut second = body("b");
        second.exempt = vec!["p".into()];
        let bodies = [first, second];
        let names = |list: &[&str]| list.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        // one piece reads p, another declares q and reads nothing else, a third holds neither
        assert_eq!(keep_held(&bodies, &names(&["p"]), &[]), names(&["p"]));
        assert_eq!(
            keep_held(&bodies, &names(&[]), &[slot("q", Dtype::F64)]),
            names(&["q"])
        );
        assert!(keep_held(&bodies, &names(&["r"]), &[]).is_empty());
    }

    #[test]
    fn two_units_of_one_name_are_refused() {
        let units = |names: &[&str]| -> Vec<Pending> {
            names
                .iter()
                .map(|name| Pending::of(&body(name), None))
                .collect()
        };
        assert!(distinct_names(&units(&["k", "k_2", "k_3"])).is_ok());
        let refusal = distinct_names(&units(&["k", "k_2", "k_2"])).unwrap_err();
        assert_eq!(refusal.kind, RefusalKind::Fault);
        assert!(refusal.reason.contains("named k_2"), "{}", refusal.reason);
    }
}
