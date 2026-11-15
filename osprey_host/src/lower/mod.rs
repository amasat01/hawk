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
mod fused;
mod placed;
pub mod request;
mod segment;
pub mod stage_block;
mod stages;
mod warp;

use aether_writers::contract::types::{
    Access, Active, BodySource, Build, Dtype, Exports, Form, FusedLane, Kind, Machine, OneStep,
    Scalar, Segment, SelectWeight, Slot, Target, Unit, Walk, DEFAULT_ROWS,
};
use aether_writers::contract::Contract;
use aether_writers::door;
use aether_writers::refuse::{bad, fault, not_written, Refusal};
use aether_writers::{cuda_render, host_render, RxCtx};
use card::{BuildCard, DeviceCard};
use raptor_ir::canon::{rx_canonical, rx_digest, rx_walk_field};
use raptor_ir::door::{RX_OK, RX_TRUNC};
use raptor_ir::record::{Ctx, Fault};
use raptor_wings::team::TeamSpec;
use rcore::keytext::{parse, text_val, Val};
use request::UnitAnswer;
use request::{Body, DeclaredSlot, KernelAnswer, KernelRequest, Member, Settings, TargetText};
use request::{ACCESSES, DTYPES, ROLES};
use std::ffi::{c_char, CString};
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

/// Refuses units that share a name: the answer names each unit, and the stage block and the caller key on it.
fn distinct_names(pending: &[Pending]) -> Result<(), Refusal> {
    let mut seen = std::collections::HashSet::new();
    match pending.iter().find(|unit| !seen.insert(&unit.member.name)) {
        Some(unit) => Err(not_written(format!(
            "two units are named {} (a piece of a chain and a stage of another piece)",
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
        Ok(Cards {
            host,
            device,
            deciding,
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

/// The answer to a request, from the kernel's IR in `ctx` and the data files in `facts_dir`.
pub fn lower(ctx: *mut Ctx, request_text: &str, facts_dir: &Path) -> Result<KernelAnswer, Refusal> {
    let request = KernelRequest::from_json(request_text)?;
    scope(&request)?;
    let cards = Cards::load(&request, facts_dir)?;
    let mut pending = match &request.body {
        Body::Plain(member) => vec![Pending::of(member, request.one_step.as_ref())],
        Body::Chain(members) => chain::expand(ctx, &request, members, facts_dir)?,
        Body::Fused(lanes) => vec![fused::expand(ctx, lanes, &request.name)?],
    };
    for unit in &mut pending {
        for body in std::iter::once(&mut unit.member).chain(unit.one_step.as_mut()) {
            body.declared.extend(finished_slot(ctx, body));
        }
    }
    for step in STEPS {
        pending = step(ctx, &request, &cards, pending)?;
    }
    distinct_names(&pending)?;
    let (stages, run) = stages::block(ctx, &cards.deciding, &pending)?;
    let units = pending
        .into_iter()
        .map(|unit| write(ctx, &request, &cards, unit))
        .collect::<Result<_, _>>()?;
    Ok(KernelAnswer { units, stages, run })
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
    Ok(UnitAnswer {
        name: pending.member.name,
        digest,
        texts,
    })
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
        mode: meta.mode.to_string(),
        vectorised: meta.vectorised,
        thread_box: meta.thread_box.filter(|b| b.axis.is_some()),
    })
}

/// Refuses by name what the written shapes do not cover: a request without a target.
fn scope(request: &KernelRequest) -> Result<(), Refusal> {
    if request.targets.is_empty() {
        return Err(bad("a request without a target"));
    }
    Ok(())
}

/// The key text of a tuple of texts.
fn tuple_of(items: impl Iterator<Item = String>) -> String {
    text_val(&Val::Tuple(items.map(Val::Str).collect()))
}

/// The key text of the declared slots, as the IR core's canonical walk reads them.
fn declared_text(declared: &[DeclaredSlot]) -> String {
    let slot = |slot: &DeclaredSlot| {
        let shape = slot
            .shape
            .iter()
            .map(|n| Val::Int(i128::from(*n)))
            .collect();
        let ttype = Val::Rec(
            "TensorType".into(),
            vec![
                ("dtype".into(), Val::Str(slot.dtype.name().into())),
                ("shape".into(), Val::Tuple(shape)),
                ("tag".into(), slot.tag.clone().map_or(Val::None, Val::Str)),
            ],
        );
        Val::Rec(
            "Slot".into(),
            vec![
                ("role".into(), Val::Str(slot.role.name().into())),
                ("name".into(), Val::Str(slot.name.clone())),
                ("ttype".into(), ttype),
            ],
        )
    };
    text_val(&Val::Tuple(declared.iter().map(slot).collect()))
}

/// The canonical walk over the member's sinks with its declared slots, and the walk's digest.
fn canonical(ctx: *mut Ctx, member: &Member) -> Result<(u32, String), Refusal> {
    let text = |s: String| CString::new(s).map_err(|_| fault("a NUL in a request text"));
    let declared = text(declared_text(&member.declared))?;
    let exempt = text(tuple_of(member.exempt.iter().cloned()))?;
    let force = member.force.map(|a| text(a.name().into())).transpose()?;
    let mut walk = 0u32;
    // SAFETY: `ctx` is the live context the caller holds; the pointers live through the call.
    let status = unsafe {
        rx_canonical(
            ctx,
            member.sinks.as_ptr(),
            member.sinks.len(),
            declared.as_ptr(),
            force.as_ref().map_or(std::ptr::null(), |f| f.as_ptr()),
            exempt.as_ptr(),
            &mut walk,
        )
    };
    if status != RX_OK {
        return Err(door::failed(ctx as *mut RxCtx, status));
    }
    let mut out = [0 as c_char; 65];
    // SAFETY: `out` holds the 65 bytes the digest is written into.
    let status = unsafe { rx_digest(ctx, walk, out.as_mut_ptr()) };
    if status != RX_OK {
        return Err(door::failed(ctx as *mut RxCtx, status));
    }
    // SAFETY: `rx_digest` wrote a NUL-terminated text.
    let digest = unsafe { std::ffi::CStr::from_ptr(out.as_ptr()) };
    Ok((walk, digest.to_string_lossy().into_owned()))
}

/// One field of the walk as key text (the two-call protocol of the door).
fn walk_field(ctx: *mut Ctx, walk: u32, field: &str) -> Result<String, Refusal> {
    let name = CString::new(field).map_err(|_| fault("a NUL in a field name"))?;
    let mut cap = 4096usize;
    loop {
        let (mut buf, mut len) = (vec![0u8; cap], 0usize);
        // SAFETY: `buf` holds `cap` bytes and `len` takes one write.
        let status = unsafe {
            rx_walk_field(
                ctx,
                walk,
                name.as_ptr(),
                buf.as_mut_ptr() as *mut c_char,
                cap,
                &mut len,
            )
        };
        match status {
            RX_TRUNC => cap = len + 1,
            RX_OK => {
                buf.truncate(len);
                return String::from_utf8(buf).map_err(fault);
            }
            other => return Err(door::failed(ctx as *mut RxCtx, other)),
        }
    }
}

/// The walk's leaves read back as slots, in argument order: each with its role, name, element type, shape and tag.
fn leaf_slots(ctx: *mut Ctx, walk: u32) -> Result<Vec<DeclaredSlot>, Refusal> {
    let (Val::Tuple(leaves) | Val::List(leaves)) =
        parse(&walk_field(ctx, walk, "leaves")?).map_err(fault)?
    else {
        return Err(fault("the walk's leaves are not a tuple"));
    };
    let shape = || fault("a walk leaf without a role, a name and a type");
    let mut slots = vec![];
    for leaf in &leaves {
        let Val::Rec(_, members) = leaf else {
            return Err(shape());
        };
        let get = |key: &str| members.iter().find(|m| m.0 == key).map(|m| &m.1);
        let (Some(Val::Str(role)), Some(Val::Str(name)), Some(Val::Rec(_, ttype))) =
            (get("role"), get("name"), get("ttype"))
        else {
            return Err(shape());
        };
        let field = |key: &str| ttype.iter().find(|m| m.0 == key).map(|m| &m.1);
        let (Some(Val::Str(dtype)), Some(Val::Tuple(extents))) = (field("dtype"), field("shape"))
        else {
            return Err(shape());
        };
        let extent = |n: &Val| match n {
            Val::Int(n) => i64::try_from(*n).map_err(|_| shape()),
            _ => Err(shape()),
        };
        slots.push(DeclaredSlot {
            role: *ROLES.iter().find(|r| r.name() == role).ok_or_else(shape)?,
            name: name.clone(),
            dtype: *DTYPES
                .iter()
                .find(|d| d.name() == dtype)
                .ok_or_else(shape)?,
            shape: extents.iter().map(extent).collect::<Result<_, _>>()?,
            tag: match field("tag") {
                Some(Val::Str(tag)) => Some(tag.clone()),
                _ => None,
            },
        });
    }
    Ok(slots)
}

/// The walk's arguments, access class and finish, read back as the writers' typed walk.
fn typed_walk(ctx: *mut Ctx, walk: u32) -> Result<Walk, Refusal> {
    let read = |field: &str| parse(&walk_field(ctx, walk, field)?).map_err(fault);
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
    let Val::Str(access) = read("access")? else {
        return Err(fault("the walk's access is not a name"));
    };
    let finish = match read("finish")? {
        Val::Str(name) => Some(name),
        _ => None,
    };
    let access: Access = *ACCESSES
        .iter()
        .find(|a| a.name() == access)
        .ok_or_else(|| fault(format!("a walk with the access class {access}")))?;
    Ok(Walk {
        access,
        finish,
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

/// The unit's contract JSON with its machine member replaced by the facts answer (less the architecture a device answer names).
fn with_facts(unit: &Unit, facts: &str) -> Result<String, Refusal> {
    let mut top = Contract::parse(&unit.to_json())?.0;
    let mut machine = Contract::parse(facts)?.0;
    if let aether_writers::contract::Json::Obj(stated) = &mut machine {
        // the architecture is the lowering's to read; the writers' machine record does not hold it
        stated.retain(|(key, _)| key != "device_arch");
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
        assert_eq!(refusal.kind, RefusalKind::NotWritten);
        assert!(refusal.reason.contains("named k_2"), "{}", refusal.reason);
    }
}
