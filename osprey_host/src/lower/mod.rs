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
use card::{BuildCard, DeviceCard};
use raptor_ir::canon::{Tt, WalkRes};
use raptor_ir::record::{Ctx, Fault};
use raptor_wings::team::TeamSpec;
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
pub(super) const FINISHED_PLANE: &str = "finished_count";

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
}

impl Cards {
    /// The cards the request needs, from the data files in `dir`.
    fn load(request: &KernelRequest, dir: &Path) -> Result<Cards, Refusal> {
        let host = card::host_build(dir)?;
        let device = request
            .targets
            .contains(&Target::Device)
            .then(|| card::device_build(dir))
            .transpose()?;
        let deciding = match &device {
            Some(device) => device.deciding(&host, &device_arch(request)?)?,
            None => host.clone(),
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
        })
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

/// The architecture the device facts answer of the request states.
fn device_arch(request: &KernelRequest) -> Result<String, Refusal> {
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

/// The answer to a request, from the kernel's IR in `ctx` and the data files in `facts_dir`. A request with derived kernels is
/// answered as one table: the units of the kernel first, then those of each derived kernel in list order.
pub fn lower(ctx: *mut Ctx, request_text: &str, facts_dir: &Path) -> Result<KernelAnswer, Refusal> {
    let request = KernelRequest::from_json(request_text)?;
    scope(&request)?;
    known_sinks(ctx, &request)?;
    refine_stop_rule(ctx, &request)?;
    let cards = Cards::load(&request, facts_dir)?;
    let mut units = kernel_units(ctx, &request, &cards, facts_dir, request.exports_block)?;
    if !request.derived.is_empty() {
        units.extend(derived_units(ctx, &request, &cards, facts_dir, &units)?);
    }
    meta_fill::unit_digests(&mut units);
    let library = unit_meta::LibraryMeta::current();
    let meta_source = meta_c::source(&library, &units.iter().map(|u| &u.meta).collect::<Vec<_>>())?;
    Ok(KernelAnswer {
        units,
        library,
        meta_source,
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
    facts_dir: &Path,
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
        let mut units = kernel_units(ctx, &sub, cards, facts_dir, false)?;
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
    facts_dir: &Path,
    exports_block: bool,
) -> Result<Vec<UnitAnswer>, Refusal> {
    let mut pending = match &request.body {
        Body::Plain(member) => vec![Pending::of(member, request.one_step.as_ref())],
        Body::Chain(members) => chain::expand(ctx, request, members, facts_dir)?,
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
    let block = stages::block(&pending);
    let mut units = pending
        .into_iter()
        .map(|unit| write(ctx, request, cards, unit))
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(block) = &block {
        meta_fill::stages(&mut units, block, cards.deciding.scatter_max_chunks)?;
    }
    if let Some(sweep) = segment::sweep_request(ctx, request)? {
        units.extend(kernel_units(ctx, &sweep, cards, facts_dir, exports_block)?);
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
    if !pending.exports_block {
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

/// The canonical walk over the member's sinks with its declared slots, and the walk's digest.
fn canonical(ctx: *mut Ctx, member: &Member) -> Result<(u32, String), Refusal> {
    let declared: Vec<_> = member.declared.iter().map(declared_leaf).collect();
    let force = member.force.map(|a| a.name());
    // SAFETY: `ctx` is the live context the caller holds; nothing else reads it during the call.
    let context = unsafe { &mut *ctx };
    let walk = context
        .canonical(&member.sinks, &declared, force, &member.exempt)
        .map_err(walk_refusal)?;
    let digest = context
        .walk(walk as u32)
        .map_err(walk_refusal)?
        .digest
        .clone();
    Ok((walk as u32, digest))
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
                compensates: None,
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

/// The unit's contract JSON with its machine member replaced by the facts answer (less the architecture and the compile command a device answer names).
fn with_facts(unit: &Unit, facts: &str) -> Result<String, Refusal> {
    let mut top = Contract::parse(&unit.to_json())?.0;
    let mut machine = Contract::parse(facts)?.0;
    if let aether_writers::contract::Json::Obj(stated) = &mut machine {
        // the architecture is the lowering's to read; the writers' machine record does not hold it
        stated.retain(|(key, _)| key != "device_arch" && key != "device_build");
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
    use aether_writers::contract::types::Role;
    use aether_writers::refuse::RefusalKind;

    /// A declared slot of the given type.
    fn slot(name: &str, dtype: Dtype) -> DeclaredSlot {
        DeclaredSlot {
            role: Role::Mutable,
            name: name.into(),
            dtype,
            shape: vec![2],
            tag: None,
            compensates: None,
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
