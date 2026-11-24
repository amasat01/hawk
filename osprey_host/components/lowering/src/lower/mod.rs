//! The lowering: one request per kernel in, the units the kernel lowers to out. The caller's kernel is already in the context as
//! IR nodes; the request names its sinks and states what the IR cannot (kind, scalar, settings, the facts answer). The lowering
//! takes the canonical walk, reads the walk's arguments back, builds the writers' unit with the numbers of the target's card
//! (the build card each backend ships) and renders it. This module holds no code of the host library's table, so a second backend library can link it.
//!
//! Written so far: the plain kernel, a segmented or placed one, an active set, two paths, split stages with their team stages, a
//! chain of kernels (cut where the fusion rule of the build's target says) and fused lanes, for the host and the device target.
//! A library lowers for the one target it serves. A build over several libraries is lowered by the leader first (the device's
//! when there is one); its answer's `peer` record states the cut, and each other library lowers to it and refuses by name a cut
//! it cannot meet. `merge` joins the answers into the one a single library serving every target would give. The warp form is
//! written for a build with no target but a device, and a target the library does not serve is refused by its name.
mod active;
mod blocks;
pub mod boxed;
pub mod card;
mod chain;
pub mod derived;
pub mod events;
mod functions;
mod fused;
mod group;
pub mod image;
pub mod links;
mod merge;
pub mod meta_c;
mod meta_fill;
mod peer;
mod placed;
pub mod request;
mod segment;
mod stage_block;
mod stages;
pub mod unit_meta;
mod warp;

pub use merge::merge;

use aether_writers::contract::types::{
    Access, Active, Axis, BodySource, Build, Dtype, Exports, Form, FusedLane, Kind, Machine,
    OneStep, Role, Scalar, Segment, SelectWeight, Slot, Target, Unit, Walk, DEFAULT_ROWS,
};
use aether_writers::contract::{Contract, Json};
use aether_writers::refuse::{bad, fault, nib, not_written, Refusal};
use aether_writers::RxCtx;

use crate::backend::{unserved, Backend, Renderer};
use crate::facts::CardSet;
use aether_writers::unit::extent_name;
use raptor_ir::canon::{Tt, WalkRes};
use raptor_ir::record::{Ctx, Fault};
use raptor_ir::value::{text_val, Val};
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

/// The cards a request is lowered with: what the backend answers for the target it serves, and the numbers the steps decide on.
pub(crate) struct Cards {
    /// The backend's answers for the request.
    target: Box<dyn card::TargetCards>,
    /// The kind of target the backend serves.
    served: Target,
    /// The name the backend gives its target in answers and refusals.
    served_name: &'static str,
    /// The form of the artifact the backend builds for the request, in the backend's own word.
    artifact: Option<String>,
    /// The writer of the backend, which renders every unit of the request.
    render: Renderer,
    /// The numbers the steps decide on: the backend's, laid over by the leader's when the library lowers beside one.
    deciding: card::BuildNumbers,
}

impl Cards {
    /// The cards the request needs, from the backend's data documents in `docs`.
    pub(crate) fn load<B: Backend>(
        backend: &B,
        request: &KernelRequest,
        docs: &CardSet,
    ) -> Result<Cards, Refusal> {
        let artifact = backend.artifact_form(request)?;
        let target = backend.cards(docs, request)?;
        let deciding = match &request.peer {
            Some((_, leader)) => peer::lay_over(target.deciding().clone(), &leader.numbers),
            None => target.deciding().clone(),
        };
        Ok(Cards {
            target,
            served: backend.target(),
            served_name: backend.target_name(),
            artifact,
            render: backend.render(),
            deciding,
        })
    }

    /// How the card runs a dense contraction of the request's scalar type; the plan's own choice when the card states none: the
    /// leader's, else `Gemm`.
    fn contraction_route(
        &self,
        request: &KernelRequest,
    ) -> Result<card::ContractionRoute, Refusal> {
        Ok(self
            .target
            .contraction_route(request.scalar)
            .unwrap_or(match &request.peer {
                Some((_, leader)) if leader.direct_contraction => card::ContractionRoute::Direct,
                _ => card::ContractionRoute::Gemm,
            }))
    }

    /// Refuses a build setting the request states that neither the library's target declares nor, for a library that lowers beside
    /// a leader, the leader's target (the keys its peer record lists).
    fn refuse_undeclared_settings(&self, request: &KernelRequest) -> Result<(), Refusal> {
        let Json::Obj(stated) = request.settings.write() else {
            return Ok(());
        };
        let leader_declares = |key: &String| {
            request
                .peer
                .as_ref()
                .is_some_and(|(_, leader)| leader.settings.contains(key))
        };
        match stated
            .iter()
            .find(|(key, _)| !self.target.settings().contains(key) && !leader_declares(key))
        {
            Some((key, _)) => Err(bad(format!(
                "build settings key {key:?} is declared by none of the targets {:?}: the target's card names the setting keys it accepts",
                request.targets.iter().map(|_| self.served_name).collect::<Vec<_>>()
            ))),
            None => Ok(()),
        }
    }

    /// The build numbers of a target's card, before the caller's settings are laid over them.
    fn numbers(&self, target: Target) -> Result<Build, Refusal> {
        self.served_by_cards(target)?;
        Ok(self.target.numbers())
    }

    /// The thread box lanes the card of a target holds, by scalar type name.
    fn box_lanes(&self, target: Target) -> Result<&[(String, i64)], Refusal> {
        self.served_by_cards(target)?;
        Ok(self.target.box_lanes())
    }

    /// Refuses a target this library's cards are not for.
    fn served_by_cards(&self, target: Target) -> Result<(), Refusal> {
        match target == self.served {
            true => Ok(()),
            false => Err(fault(format!("a {target:?} target without its card"))),
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

/// Refuses a refine unit over a stop rule that cannot be refined.
fn refine_stop_rule(ctx: *mut Ctx, request: &KernelRequest) -> Result<(), Refusal> {
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    events::check_refine(unsafe { &*ctx }, request)
}

/// The answer to a request, from the kernel's IR in `ctx` and the data files in the directory `facts_dir`. A request with derived kernels is
/// answered as one table: the units of the kernel first, then those of each derived kernel in list order.
pub fn lower<B: Backend>(
    backend: &B,
    ctx: *mut Ctx,
    request_text: &str,
    facts_dir: &Path,
) -> Result<KernelAnswer, Refusal> {
    let request = KernelRequest::from_json(request_text)?;
    let docs = &CardSet::new(facts_dir, &request.cards, backend.card_names());
    docs.check_stated().map_err(crate::facts::Failed::refusal)?;
    serves_all(backend, &request)?;
    scope(&request)?;
    known_sinks(ctx, &request)?;
    refine_stop_rule(ctx, &request)?;
    facts_under_cards(&request.facts, docs)?;
    let cards = Cards::load(backend, &request, docs)?;
    cards.refuse_undeclared_settings(&request)?;
    let mut units = kernel_units(ctx, &request, &cards, docs, request.exports_block)?;
    if !request.derived.is_empty() {
        units.extend(derived_units(ctx, &request, &cards, docs, &units)?);
    }
    meta_fill::unit_digests(&mut units);
    let library = unit_meta::LibraryMeta::current();
    let metas: Vec<&unit_meta::UnitMeta> = units.iter().map(|u| &u.meta).collect();
    peer::check_cut(&request, &metas)?;
    let meta_source = meta_c::source(&library, &metas)?;
    let fusion = match request.body {
        Body::Chain(_) => Some(chain::fusion_row(&request, cards.target.as_ref(), docs)?),
        _ => None,
    };
    let direct = cards.contraction_route(&request)? == card::ContractionRoute::Direct;
    let peer = peer::record(
        &cards.deciding,
        fusion.as_ref(),
        (direct, cards.target.settings()),
        &metas,
    );
    Ok(KernelAnswer {
        working_set_bytes: working_set(ctx, &request, &cards)?,
        wide_class: cards.target.wide_class().map(String::from),
        steps_reverse: cards.target.steps_reverse(),
        dispatch_costs: cards.target.dispatch_costs(&request, docs)?,
        index_bytes: backend.index_bytes(&[])?,
        peer: Some(peer),
        units,
        library,
        meta_source,
    })
}

/// Refuses a request that names a target the backend does not serve, naming the target and the library.
fn serves_all<B: Backend>(backend: &B, request: &KernelRequest) -> Result<(), Refusal> {
    match request.targets.iter().find(|t| **t != backend.target()) {
        Some(other) => Err(unserved(backend, request::target_word(*other))),
        None => Ok(()),
    }
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
    group::refuse_unwritten(ctx, request)?;
    let mut pending = match &request.body {
        Body::Plain(member) => vec![Pending::of(member, request.one_step.as_ref())],
        Body::Chain(members) => {
            chain::expand(ctx, request, members, (cards.target.as_ref(), docs))?
        }
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
    let (meta, contraction) = meta_fill::unit(&meta_fill::Parts {
        ctx,
        request,
        card: &cards.deciding,
        image: cards.target.image(),
        artifact: cards.artifact.as_deref(),
        pending: &pending,
        walk,
        digest: &digest,
        typed: &typed_walk(ctx, walk)?,
        texts: &texts,
        route: cards.contraction_route(request)?,
    })?;
    let defines = fast_entry_defines(
        request.settings.fast_entries,
        texts.iter().map(|t| (t.target, t.text.as_str())),
    );
    Ok(UnitAnswer {
        texts,
        meta,
        defines: defines
            .into_iter()
            .map(|(_, define)| (cards.served_name.to_string(), define))
            .collect(),
        contraction,
    })
}

/// The macro a device text gates its fast entry points behind.
const FAST_ENTRIES_MACRO: &str = "HAWK_FAST_ENTRIES";

/// The defines the settings put on the compile of each text: the fast-entry macro for a text that gates code behind it, when the
/// build asks for the fast entries. `texts` are the `(target, text)` pairs of a unit.
fn fast_entry_defines<'a>(
    asked: bool,
    texts: impl Iterator<Item = (Target, &'a str)>,
) -> Vec<(Target, String)> {
    if !asked {
        return vec![];
    }
    texts
        .filter(|(_, text)| text.contains(FAST_ENTRIES_MACRO))
        .map(|(target, _)| (target, format!("{FAST_ENTRIES_MACRO}=1")))
        .collect()
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
    unit.functions = functions::table(ctx, walk, cards.served_name)?;
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
    let source = (cards.render)(ctx as *mut RxCtx, walk, &json)?;
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

/// Refuses by name what the written shapes do not cover: a request without a target, and a banded kernel (one with cells) for the
/// host, because cells are written for device execution paths only.
fn scope(request: &KernelRequest) -> Result<(), Refusal> {
    if request.targets.is_empty() {
        return Err(bad("a request without a target"));
    }
    if let Some(own) = request
        .companions
        .iter()
        .find(|c| request.targets.contains(c))
    {
        return Err(bad(format!(
            "companions: {own:?} is served by this library, which lists only the others"
        )));
    }
    let named =
        std::iter::once(request.name.as_str()).chain(request.derived.iter().map(|d| d.name()));
    for name in named {
        if request
            .reserved_names
            .iter()
            .any(|reserved| reserved == name)
        {
            return Err(bad(format!(
                "{name:?}: the name is reserved for the runtime's own units; name the kernel otherwise"
            )));
        }
    }
    if request.scalar == Scalar::Banded && request.build_has(Target::Host) {
        return Err(bad(format!(
            "{:?}: banded runs on device execution paths; on the host use float64 (target 'host' is not a device)",
            request.name
        )));
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
            axes: axes_text(&slot.axes),
        },
    )
}

/// The axes of a plane from the text a type states them in (empty text: none).
fn axes_of(text: &str) -> Vec<request::LaneAxis> {
    let Ok(Val::Tuple(declared)) = raptor_ir::value::parse(text) else {
        return vec![];
    };
    declared
        .iter()
        .filter_map(|axis| match axis {
            Val::Tuple(pair) => match pair.as_slice() {
                [Val::Str(label), extent] => Some(request::LaneAxis {
                    label: Some(label.clone()),
                    extent: match extent {
                        Val::Int(n) => i64::try_from(*n).ok(),
                        _ => None,
                    },
                }),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

/// An axis as the writers state it.
fn axis_of(axis: &request::LaneAxis) -> Axis {
    Axis {
        label: axis.label.clone().unwrap_or_default(),
        extent: axis.extent,
    }
}

/// The axis list of a declared plane in the text the digest states it in: a tuple of `(label, extent)` pairs, an extent
/// known at run time being `None`; empty for a plane declared without axes.
fn axes_text(axes: &[request::LaneAxis]) -> String {
    if axes.is_empty() {
        return String::new();
    }
    let pair = |a: &request::LaneAxis| {
        Val::Tuple(vec![
            Val::Str(a.label.clone().unwrap_or_default()),
            a.extent.map_or(Val::None, |n| Val::Int(i128::from(n))),
        ])
    };
    text_val(&Val::Tuple(axes.iter().map(pair).collect()))
}

/// The kernel's canonical walk over the member's sinks and declared slots (the wing adds the companion, finished-count and
/// branch-mix slots), and the walk's digest.
fn canonical(ctx: *mut Ctx, member: &Member) -> Result<(u32, String), Refusal> {
    let declared = raptor_ir::canon::Declared {
        slots: member.declared.iter().map(declared_leaf).collect(),
        zeroed: member
            .declared
            .iter()
            .filter(|slot| slot.zeroed)
            .map(|slot| (slot.role.name().to_string(), slot.name.clone()))
            .collect(),
        counts: member
            .declared
            .iter()
            .filter_map(|slot| Some((slot.name.clone(), slot.counts.clone()?)))
            .collect(),
        winners: member
            .declared
            .iter()
            .filter_map(|slot| Some((slot.name.clone(), slot.winner_of.clone()?)))
            .collect(),
    };
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
    let read = walk_of(ctx, walk)?;
    let zeroed = &read.zeroed;
    read.leaves
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
                axes: axes_of(&ttype.axes),
                zeroed: zeroed.contains(&(role.clone(), name.clone())),
                counts: read
                    .counts
                    .iter()
                    .find(|(count, _)| count == name)
                    .map(|(_, of)| of.clone()),
                winner_of: read
                    .winners
                    .iter()
                    .find(|(winner, _)| winner == name)
                    .map(|(_, of)| of.clone()),
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
    let leaves = leaf_slots(ctx, walk)?;
    let mut slots: Vec<Slot> = leaves
        .iter()
        .enumerate()
        .map(|(index, slot)| Slot {
            role: slot.role,
            name: slot.name.clone(),
            dtype: slot.dtype,
            shape: slot.shape.clone(),
            index: index as i64,
            axes: (!slot.axes.is_empty()).then(|| slot.axes.iter().map(axis_of).collect()),
            zeroed: slot.zeroed,
        })
        .collect();
    // each runtime axis of a plane reaches the entry as one more argument, which the runtime fills from the bound array
    for plane in &leaves {
        for axis in plane.axes.iter().filter(|a| a.extent.is_none()) {
            let label = axis.label.clone().unwrap_or_default();
            slots.push(Slot {
                role: Role::Extent,
                name: extent_name(&plane.name, &label),
                dtype: Dtype::I64,
                shape: vec![],
                index: slots.len() as i64,
                axes: None,
                zeroed: false,
            });
        }
    }
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
        functions: None,
    })
}

/// The build's numbers for a target: the numbers of the target's card with the caller's settings laid over them.
fn build_of(settings: &Settings, cards: &Cards, target: Target) -> Result<Build, Refusal> {
    Ok(Build {
        fold_form: settings.fold_form,
        warps: settings.warps,
        launch_bounds: settings.launch_bounds,
        fold_k: settings.fold_rows.filter(|k| *k != 0),
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
    //! The scalar width has one answer, a merge that would drop something is refused by name, and the fast-entry setting becomes
    //! the macro of the texts that gate on it.
    use super::*;

    use crate::testing::{TestDevice, TestHost};

    #[test]
    fn the_fast_entry_setting_defines_the_macro_for_exactly_the_texts_that_gate_on_it() {
        let gated = "#ifndef HAWK_FAST_ENTRIES\n#define HAWK_FAST_ENTRIES 0\n#endif";
        let texts = || [(Target::Device, gated), (Target::Host, "int k;")].into_iter();
        assert_eq!(
            fast_entry_defines(true, texts()),
            vec![(Target::Device, "HAWK_FAST_ENTRIES=1".to_string())]
        );
        assert!(fast_entry_defines(false, texts()).is_empty());
        assert!(fast_entry_defines(true, [(Target::Device, "int k;")].into_iter()).is_empty());
    }

    /// The shipped facts directory.
    fn shipped() -> std::path::PathBuf {
        crate::testing::shipped()
    }

    /// A document of the fixture stated in place of the shipped one, with a number of its own.
    fn lean_card() -> Vec<(String, String)> {
        vec![("numbers.json".into(), "{\"lean_width\": 64}".into())]
    }

    /// The device facts answer made under the cards `stated`.
    fn device_facts(stated: &[(String, String)]) -> String {
        crate::testing::device_facts(None, stated)
    }

    #[test]
    fn a_request_for_the_other_kind_is_refused_naming_both_kinds_in_the_words_of_a_request() {
        let request = KernelRequest {
            targets: vec![Target::Device],
            ..plain_request()
        };
        let why = serves_all(&TestHost, &request).unwrap_err();
        assert_eq!(
            why.reason,
            "the library test_host serves the target host, not Device"
        );
    }

    #[test]
    fn a_facts_answer_must_state_the_digest_of_the_request_cards_and_state_none_when_the_request_has_none(
    ) {
        let files = CardSet::new(&shipped(), &[], crate::testing::ALL_CARDS);
        let lean = CardSet::new(&shipped(), &lean_card(), crate::testing::ALL_CARDS);
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

    /// A request cannot reach the lowering with answers made under other cards than its own: the answer a request needs carries the
    /// digest of the request's cards, and an answer made under none is refused for a request that states some.
    #[test]
    fn an_answer_made_under_no_cards_is_refused_for_a_request_that_states_some() {
        let broken = [("fusion.json".to_string(), "{\"nonsense\": 1}".to_string())];
        let plain = [(Target::Device, device_facts(&[]))];
        let why = facts_under_cards(
            &plain,
            &CardSet::new(&shipped(), &broken, crate::testing::ALL_CARDS),
        )
        .unwrap_err();
        assert!(why.reason.contains("other cards"), "{}", why.reason);
    }

    /// Oracle: a stated copy of the shipped file has the digest of the text it states, which differs from that of another text.
    #[test]
    fn the_digest_is_none_without_stated_cards_and_follows_the_stated_text_with_them() {
        let text = std::fs::read_to_string(shipped().join("fusion.json")).unwrap();
        let same = CardSet::new(
            &shipped(),
            &[("fusion.json".into(), text)],
            crate::testing::ALL_CARDS,
        );
        assert_eq!(
            CardSet::new(&shipped(), &[], crate::testing::ALL_CARDS)
                .digest()
                .unwrap(),
            None
        );
        let shipped_text = same.digest().unwrap().unwrap();
        assert_ne!(
            CardSet::new(&shipped(), &lean_card(), crate::testing::ALL_CARDS)
                .digest()
                .unwrap()
                .unwrap(),
            shipped_text
        );
    }
    use crate::lower::unit_meta::SlotRole;
    use aether_writers::contract::types::Role;
    use aether_writers::refuse::RefusalKind;
    use raptor_ir::door::{rx_ctx_free, rx_ctx_new, RX_OK};
    use raptor_ir::dump::rx_load;

    /// Oracle: `working_set_bytes` of the Python model (the old IR package), recorded per kernel in the wings' oracle file under
    /// the shipped card's fold threshold (16) and form "auto". The answer's figure is that of the kernel's own sinks.
    #[test]
    fn the_answer_states_the_working_set_the_python_model_gives_for_the_kernel() {
        use raptor_ir::door::{rx_ctx_free, rx_ctx_new};
        use raptor_ir::dump::rx_load;
        let oracle = rcore::json::parse(include_str!(
            "../../../../../raptor/rs/wings/tests/data/working_set_oracle.json"
        ))
        .unwrap();
        let docs = CardSet::new(&shipped(), &[], crate::testing::ALL_CARDS);
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
            let cards = crate::testing::load_cards(&request, &docs).unwrap();
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
        crate::testing::device_facts(arch, stated)
    }

    /// The warp rows a request's split may take, for the scalar type, targets and architecture the request states.
    fn warp_rows_of(
        scalar: Scalar,
        targets: Vec<Target>,
        arch: Option<&str>,
    ) -> Vec<(i64, String)> {
        let stated = [];
        let mut request = device_request(facts_of_arch(arch, &stated));
        request.scalar = scalar;
        request.targets = targets;
        if request.targets.contains(&Target::Host) {
            request.facts.push((Target::Host, "{}".into()));
        }
        let cards = crate::testing::load_cards(
            &request,
            &CardSet::new(&shipped(), &stated, crate::testing::ALL_CARDS),
        )
        .unwrap();
        let rows = stages::stage_facts(&request, &cards.deciding)
            .team
            .warp_eigh;
        rows.into_iter().map(|row| (row.n, row.scalar)).collect()
    }

    /// Oracle: the old rule of the warp form (the removed hawk test): a float32 build takes the row on any device; a float64 build
    /// only where the double rate is fast: the `fast` words yes, the `slow` words and no stated architecture no; a build with a
    /// host target has no warp.
    #[test]
    fn a_double_build_takes_the_warp_row_only_where_fp64_is_fast_and_a_host_target_takes_none() {
        let row = |scalar: &str| vec![(32, scalar.to_string())];
        for (arch, double) in [
            (Some("slow_a"), false),
            (Some("slow_b"), false),
            (None, false),
            (Some("fast_a"), true),
            (Some("fast_b"), true),
        ] {
            let rows = warp_rows_of(Scalar::Float64, vec![Target::Device], arch);
            assert_eq!(
                rows,
                if double { row("float64") } else { vec![] },
                "{arch:?}"
            );
            let rows = warp_rows_of(Scalar::Float32, vec![Target::Device], arch);
            assert_eq!(rows, row("float32"), "{arch:?}");
        }
        for targets in [vec![Target::Host], vec![Target::Device, Target::Host]] {
            assert_eq!(warp_rows_of(Scalar::Float32, targets, Some("fast_a")), []);
        }
        assert_eq!(
            warp_rows_of(Scalar::Banded, vec![Target::Device], Some("fast_a")),
            []
        );
    }

    #[test]
    fn a_request_with_no_device_target_states_no_wide_class() {
        let cards = crate::testing::load_cards(
            &plain_request(),
            &CardSet::new(&shipped(), &[], crate::testing::ALL_CARDS),
        )
        .unwrap();
        assert_eq!(cards.target.wide_class(), None);
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
            axes: vec![],
            zeroed: false,
            counts: None,
            winner_of: None,
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

    /// A request for `scalar` on the given targets, under the host facts answer the plain request states.
    fn scalar_request(scalar: Scalar, targets: Vec<Target>) -> KernelRequest {
        KernelRequest {
            scalar,
            targets,
            ..plain_request()
        }
    }

    /// Oracle: the sentence the old build refused a banded kernel with when a target was not a device.
    #[test]
    fn a_banded_kernel_for_the_host_is_refused_with_the_sentence_of_the_old_build() {
        for targets in [vec![Target::Host], vec![Target::Device, Target::Host]] {
            let why = scope(&scalar_request(Scalar::Banded, targets)).unwrap_err();
            assert_eq!(
                why.reason,
                "\"k\": banded runs on device execution paths; on the host use float64 (target 'host' is not a device)"
            );
        }
    }

    /// A device library that leads a build with a host library is refused a banded kernel by the same sentence, and a request that
    /// lists its own target among the companions is refused by name.
    #[test]
    fn a_banded_kernel_of_a_build_with_a_host_companion_is_refused_and_a_companion_is_never_its_own_target(
    ) {
        let mut lead = scalar_request(Scalar::Banded, vec![Target::Device]);
        scope(&lead).unwrap();
        lead.companions = vec![Target::Host];
        let why = scope(&lead).unwrap_err();
        assert!(
            why.reason.contains("banded runs on device"),
            "{}",
            why.reason
        );
        lead.scalar = Scalar::Float64;
        lead.companions = vec![Target::Device];
        let why = scope(&lead).unwrap_err();
        assert!(why.reason.contains("companions"), "{}", why.reason);
        assert!(!lead.alone_on(Target::Device) && lead.build_has(Target::Device));
    }

    #[test]
    fn a_host_kernel_without_cells_and_a_banded_kernel_for_the_device_are_not_refused() {
        scope(&scalar_request(Scalar::Float64, vec![Target::Host])).unwrap();
        scope(&scalar_request(Scalar::Banded, vec![Target::Device])).unwrap();
    }

    /// The answer for the host to the kernel with two dispatch groups over one sink (an IR dump recorded from the front end), with a
    /// vector and a matrix input declared beside the slots its body binds.
    fn lowered_two_groups() -> KernelAnswer {
        lowered_two_groups_as(|_| {}).unwrap()
    }

    /// The answer to the two-group kernel's request after `edit` has changed it.
    fn lowered_two_groups_as(
        edit: impl FnOnce(&mut KernelRequest),
    ) -> Result<KernelAnswer, Refusal> {
        const DUMP: &str = include_str!("../../tests/data/two_groups.dump");
        let ctx = rx_ctx_new();
        // SAFETY: `ctx` is the live context just made.
        assert_eq!(
            unsafe { raptor_wings::declare::rw_declare_kinds(ctx) },
            RX_OK
        );
        let mut roots = vec![0u32; 4096];
        let mut loaded = 0;
        // SAFETY: `ctx` is the live context just made, the text and the buffers outlive the call.
        let status = unsafe {
            rx_load(
                ctx,
                DUMP.as_ptr().cast(),
                DUMP.len(),
                roots.as_mut_ptr(),
                roots.len(),
                &mut loaded,
            )
        };
        assert_eq!(status, RX_OK);
        let facts = crate::testing::host_facts(&[]);
        let mut request = plain_request();
        request.facts = vec![(Target::Host, facts)];
        request.body = Body::Plain(Member {
            // the one sink of the dump is its last root, the assignment to `y`
            sinks: vec![roots[10]],
            declared: vec![
                DeclaredSlot {
                    role: Role::PlaneIn,
                    shape: vec![3],
                    ..slot("v", Dtype::F64)
                },
                DeclaredSlot {
                    role: Role::PlaneIn,
                    shape: vec![2, 2],
                    ..slot("m", Dtype::F64)
                },
            ],
            ..body("k")
        });
        edit(&mut request);
        let answer = lower(&TestHost, ctx, &request.to_json(), &shipped());
        // SAFETY: `ctx` is the live context made above and is not used after.
        unsafe { rx_ctx_free(ctx) };
        answer
    }

    /// Oracle: the old build set the routing of a banded build itself (a banded mode has cells, which route to the ordered stages), so a
    /// banded request that does not say so lowers to the answer of one that does; a request of another scalar is as it states.
    #[test]
    fn a_banded_request_routes_to_ordered_whether_or_not_it_says_so() {
        let banded = |route: bool| {
            let ctx = rx_ctx_new();
            // SAFETY: `ctx` is the live context just made, and is not used after the free below.
            let c = unsafe { &mut *ctx };
            assert_eq!(
                unsafe { raptor_wings::declare::rw_declare_kinds(ctx) },
                RX_OK
            );
            let x = c
                .make(
                    "Leaf",
                    "f64",
                    &[],
                    None,
                    &[],
                    "(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', 'x'), ('quantity', None), ('wire_index', 0), ('sorted', False))",
                )
                .unwrap();
            let sum = c
                .make(
                    "Op",
                    "f64",
                    &[],
                    None,
                    &[x, x],
                    "(('kind', 'add'), ('literal', None))",
                )
                .unwrap();
            let out = c
                .make(
                    "Assign",
                    "f64",
                    &[],
                    None,
                    &[sum],
                    "(('kind', 'assign'), ('role', 'mutable'), ('name', 'y'))",
                )
                .unwrap();
            let mut request = scalar_request(Scalar::Banded, vec![Target::Device]);
            request.facts = vec![(Target::Device, facts_of_arch(None, &[]))];
            request.image_format = Some("Blob".into());
            request.settings.route_to_ordered = route;
            request.body = Body::Plain(Member {
                sinks: vec![out],
                ..body("k")
            });
            let answer = lower(&TestDevice, ctx, &request.to_json(), &shipped());
            // SAFETY: `ctx` is the live context made above and is not used after.
            unsafe { rx_ctx_free(ctx) };
            answer.unwrap().to_json()
        };
        assert_eq!(banded(false), banded(true));
        let card = card::TargetCards::deciding(&crate::testing::FixtureCards::host()).clone();
        for (scalar, route, routed) in [
            (Scalar::Float64, false, false),
            (Scalar::Float64, true, true),
            (Scalar::Float32, false, false),
            (Scalar::Float32, true, true),
            (Scalar::Banded, false, true),
        ] {
            let mut request = scalar_request(scalar, vec![Target::Device]);
            request.settings.route_to_ordered = route;
            assert_eq!(
                stages::stage_facts(&request, &card).routed,
                routed,
                "{scalar:?} {route}"
            );
        }
    }

    /// Oracle: the kernel's own source, read by hand: `y` is the mutable output, `x`, `v` and `m` are only read. A slot's element count
    /// is the product of its shape, 1 for a scalar.
    #[test]
    fn every_slot_states_whether_the_kernel_writes_it_and_no_element_count() {
        let answer = lowered_two_groups();
        let slots: Vec<_> = answer
            .units
            .iter()
            .flat_map(|u| &u.meta.body.slots)
            .collect();
        let found = |name: &str| {
            *slots
                .iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("slot {name}"))
        };
        assert!(!found("x").written && found("y").written);
        for name in ["v", "m"] {
            let slot = found(name);
            assert!(
                slot.shape.iter().product::<i64>() > 1 && !slot.written,
                "{name}"
            );
        }
        let json = answer.to_json();
        assert_eq!(json.matches("\"elements\":").count(), 0);
        assert_eq!(json.matches("\"written\":").count(), slots.len());
    }

    /// Oracle: the table in aether's `runtime_names.rs` (`mirror_of`), read by hand: a mutable slot is a `GRefMirror` when it has a
    /// rank and a `ScalarHandle` when it has none; a vector or matrix input or an output is a `GRefMirror`; a per-sample, lookup or
    /// terminated slot is a `ScalarHandle`; a uniform or the sample count is passed by `value`.
    #[test]
    fn every_slot_of_a_lowered_kernel_states_how_it_crosses_the_launch_boundary() {
        let answer = lowered_two_groups();
        let slots: Vec<_> = answer
            .units
            .iter()
            .flat_map(|u| &u.meta.body.slots)
            .collect();
        let mirror_of = |name: &str| {
            let slot = slots
                .iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("slot {name}"));
            (slot.role, slot.shape.len(), slot.mirror.as_str())
        };
        // rank 0: a per-sample read is a scalar handle, a mutable output without rank too
        assert_eq!(mirror_of("x"), (SlotRole::PlaneIn, 0, "ScalarHandle"));
        assert_eq!(mirror_of("y"), (SlotRole::Mutable, 0, "ScalarHandle"));
        // ranked: a vector and a matrix input, one role, are each read through a mirror
        assert_eq!(mirror_of("v"), (SlotRole::PlaneIn, 1, "GRefMirror"));
        assert_eq!(mirror_of("m"), (SlotRole::PlaneIn, 2, "GRefMirror"));
        // the answer's JSON states it on every slot
        let json = answer.to_json();
        let count = slots.len();
        assert_eq!(json.matches("\"mirror\":").count(), count);
        assert_eq!(KernelAnswer::from_json(&json).unwrap(), answer);
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
