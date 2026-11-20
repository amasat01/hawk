//! The chain step: a chain of kernels is one kernel when the fusion rule fuses every boundary between its members, else a unit per
//! piece, each piece reading what an earlier piece handed on in a plane the runtime makes per lane. The IR decisions (the
//! composition, the cuts, the pieces) come from the wing analysis; this module reads the fusion row of the build's target from the facts and
//! hands the pieces to the steps that follow as pending units.
use super::request::{DeclaredSlot, KernelRequest, Member};
use super::stage_block::{PlaneRecord, StageRecord};
use super::stages::{source_unit, Note};
use super::unit_meta::UnitMeta;
use super::{carries_no_forced_access, held_exempt, merge_slot, real_bytes, wing, Pending};
use crate::facts;
use crate::facts::CardSet;
use aether_writers::contract::types::{Scalar, Target};
use aether_writers::refuse::{bad, nib, Refusal};
use raptor_ir::record::Ctx;
use raptor_wings::chain::{bind, cuts, pieces, Member as ChainMember, Piece};
use std::collections::HashSet;

/// The lanes the cut rule weighs a cut at: the extent is not known when the kernel is built.
const CHAIN_SAMPLES: i64 = 1 << 16;

/// The name of the unit of piece `index`: the chain's own name, then the name and the piece's place from the second on.
fn piece_name(name: &str, index: usize) -> String {
    match index {
        0 => name.to_string(),
        _ => format!("{name}_{}", index + 1),
    }
}

/// The pieces a chain is cut into: the whole chain composed as one piece when no boundary is cut. A compaction group runs as one
/// kernel, so it is never cut.
fn cut_pieces(
    ctx: &mut Ctx,
    request: &KernelRequest,
    members: &[ChainMember],
    row: &facts::FusionRow,
) -> Result<Vec<Piece>, Refusal> {
    let cutting = members.len() > 1 && request.group == 1;
    if cutting && request.scalar == Scalar::Banded {
        return Err(nib("the cuts of a chain of a banded kernel"));
    }
    let cut = match cutting {
        true => cuts(
            ctx,
            members,
            row,
            (real_bytes(request.scalar) == 4, CHAIN_SAMPLES),
        )
        .map_err(wing)?,
        false => vec![],
    };
    if cut.is_empty() {
        let sinks = bind(ctx, members, &HashSet::new()).map_err(wing)?.sinks;
        return Ok(vec![Piece {
            sinks,
            planes: vec![],
        }]);
    }
    pieces(ctx, members, &cut).map_err(wing)
}

/// The slots the members declare, each name once: a piece of the chain keeps the slots its members stated.
fn declared_by_members(members: &[Member]) -> Result<Vec<DeclaredSlot>, Refusal> {
    let mut declared: Vec<DeclaredSlot> = vec![];
    for slot in members.iter().flat_map(|m| &m.declared) {
        merge_slot(&mut declared, slot.clone())
            .map_err(|why| bad(format!("chain members: {why}")))?;
    }
    Ok(declared)
}

/// The fusion row of a device build for the architecture and the scalar type: the stated row of that type whose architecture is
/// nearest by hardware, else the device document's default row.
pub(super) fn device_row(
    arch: &str,
    scalar: Scalar,
    docs: &CardSet,
) -> Result<facts::FusionRow, Refusal> {
    let rows = facts::device_fusion(docs).map_err(facts::Failed::refusal)?;
    let device = super::card::device_build_of(docs)?;
    let of_type: Vec<(String, facts::FusionRow)> = rows
        .rows
        .into_iter()
        .filter(|(_, dtype, _)| match scalar {
            Scalar::Float32 => dtype == "float32",
            Scalar::Float64 => dtype == "float64",
            Scalar::Banded => false,
        })
        .map(|(cc, _, row)| (cc, row))
        .collect();
    Ok(device
        .nearest_row(arch, &of_type)?
        .cloned()
        .unwrap_or(rows.defaults))
}

/// The build's one fusion row: the leader's when this library lowers beside one, else the device's when the target is a device,
/// else this host's.
pub(super) fn fusion_row(
    request: &KernelRequest,
    docs: &CardSet,
) -> Result<facts::FusionRow, Refusal> {
    if let Some((leader, peer)) = &request.peer {
        return peer
            .fusion
            .as_ref()
            .map(super::peer::fusion_row)
            .ok_or_else(|| {
                bad(format!(
                    "{:?}: the {leader:?} library states no fusion row for a chain",
                    request.name
                ))
            });
    }
    if request.targets.contains(&Target::Device) {
        device_row(&super::device_arch(request)?, request.scalar, docs)
    } else {
        facts::host_fusion(docs, request.facts_for(Target::Host)?).map_err(facts::Failed::refusal)
    }
}

/// The unit that lists the pieces of a cut chain: the whole chain composed as one kernel, so its slots are every plane the caller
/// binds over the chain, whichever piece reads or writes it.
fn source_of(
    ctx: *mut Ctx,
    request: &KernelRequest,
    members: &[ChainMember],
    (bodies, declared): (&[Member], &[DeclaredSlot]),
) -> Result<UnitMeta, Refusal> {
    // SAFETY: the caller's context is not read elsewhere while the source is made.
    let sinks = bind(unsafe { &mut *ctx }, members, &HashSet::new())
        .map_err(wing)?
        .sinks;
    let exempt = held_exempt(ctx, bodies, &sinks, declared)?;
    let whole = Member {
        name: request.name.clone(),
        sinks,
        declared: declared.to_vec(),
        force: None,
        exempt,
    };
    source_unit(ctx, request, &whole)
}

/// The pending units of a chain, one per piece, each noting the planes it hands to the next piece. The first piece of a cut chain
/// carries the unit that lists the pieces.
pub(super) fn expand(
    ctx: *mut Ctx,
    request: &KernelRequest,
    members: &[Member],
    docs: &CardSet,
) -> Result<Vec<Pending>, Refusal> {
    let row = fusion_row(request, docs)?;
    let named: Vec<ChainMember> = members
        .iter()
        .map(|m| ChainMember {
            name: &m.name,
            sinks: &m.sinks,
        })
        .collect();
    // SAFETY: the caller's context is not read elsewhere while the pieces are made.
    let parts = cut_pieces(unsafe { &mut *ctx }, request, &named, &row)?;
    for member in members {
        carries_no_forced_access(member, "a chain member")?;
    }
    let declared = declared_by_members(members)?;
    let source = match parts.len() > 1 {
        true => Some(source_of(ctx, request, &named, (members, &declared))?),
        false => None,
    };
    let cut = source.is_some();
    let mut units = vec![];
    for (index, piece) in parts.into_iter().enumerate() {
        let exempt = held_exempt(ctx, members, &piece.sinks, &declared)?;
        let member = Member {
            name: piece_name(&request.name, index),
            sinks: piece.sinks,
            declared: declared.clone(),
            force: None,
            exempt,
        };
        let mut unit = Pending::of(&member, None);
        unit.note = cut.then_some(Note {
            stage: StageRecord {
                kernel: member.name.clone(),
                cells_in: vec![],
                cells_out: vec![],
                team: None,
            },
            cells: vec![],
            planes: piece
                .planes
                .into_iter()
                .map(|plane| PlaneRecord {
                    name: plane.name,
                    dtype: plane.dtype.to_string(),
                    per_lane: plane.per_lane,
                    layout: Some("plane_major".into()),
                    keys: None,
                    per_target: None,
                    once: false,
                    wide: false,
                    fill: None,
                })
                .collect(),
            keyed: vec![],
            source: source.clone().filter(|_| index == 0),
        });
        units.push(unit);
    }
    Ok(units)
}

#[cfg(test)]
mod tests {
    //! The fusion row of a device build follows the rows the device document states, and the first piece of a cut chain carries the
    //! unit that lists the whole chain.
    use super::*;
    use crate::lower::request::Body as RequestBody;
    use raptor_ir::door::{rx_ctx_free, rx_ctx_new, RX_OK};

    /// The shipped document with the rows `rows` (JSON) stated in place of its empty list.
    fn docs_with_rows(rows: &str) -> CardSet {
        let dir = crate::testing::shipped();
        let text = std::fs::read_to_string(dir.join("cuda.json")).unwrap();
        let stated = text.replacen(
            "\"fusion_rows\": []",
            &format!("\"fusion_rows\": {rows}"),
            1,
        );
        CardSet::new(
            &dir,
            &[("cuda.json".to_string(), stated)],
            crate::testing::ALL_CARDS,
        )
    }

    /// Oracle: the shipped default row holds 255 registers. A row stated for `sm_61` and `float32` holds 64 and leaves the other
    /// numbers to the defaults; it serves that type on every architecture (it is the only row, so the nearest), and no other type.
    #[test]
    fn a_stated_row_serves_its_scalar_type_by_the_nearest_architecture_and_the_others_take_the_defaults(
    ) {
        let none = docs_with_rows("[]");
        let one = docs_with_rows(
            r#"[{"cc": "sm_61", "dtype": "float32", "regs_file": 64, "local_over_plane": null}]"#,
        );
        let regs = |docs: &CardSet, arch, scalar| device_row(arch, scalar, docs).unwrap();
        assert_eq!(regs(&none, "sm_61", Scalar::Float32).regs_file, 255);
        let row = regs(&one, "sm_80", Scalar::Float32);
        assert_eq!(
            (row.regs_file, row.local_over_plane, row.launch_us),
            (64, None, Some(2.7))
        );
        assert_eq!(regs(&one, "sm_61", Scalar::Float64).regs_file, 255);
        assert_eq!(regs(&one, "sm_61", Scalar::Banded).regs_file, 255);
    }

    /// Oracle: of two rows of one type the one of the architecture with hardware nearest the build's wins (sm_61 and sm_75 have an
    /// FP64 rate of 1/32 where sm_80 has 1/2 and sm_90 has 1/2).
    #[test]
    fn the_row_of_the_architecture_with_the_nearest_hardware_wins() {
        let docs = docs_with_rows(
            r#"[{"cc": "sm_61", "dtype": "float64", "regs_file": 64}, {"cc": "sm_90", "dtype": "float64", "regs_file": 128}]"#,
        );
        assert_eq!(
            device_row("sm_80", Scalar::Float64, &docs)
                .unwrap()
                .regs_file,
            128
        );
        assert_eq!(
            device_row("sm_75", Scalar::Float64, &docs)
                .unwrap()
                .regs_file,
            64
        );
    }

    #[test]
    fn a_stated_row_that_names_another_type_or_an_unknown_number_is_refused_by_name() {
        let other = docs_with_rows(r#"[{"cc": "sm_61", "dtype": "half"}]"#);
        let why = device_row("sm_61", Scalar::Float32, &other).unwrap_err();
        assert!(why.reason.contains("half"), "{}", why.reason);
        let unknown = docs_with_rows(r#"[{"cc": "sm_61", "dtype": "float32", "regs": 1}]"#);
        let why = device_row("sm_61", Scalar::Float32, &unknown).unwrap_err();
        assert!(why.reason.contains("regs"), "{}", why.reason);
    }

    /// A node of `class` over `ops`, with the fields text `fields`.
    fn node(ctx: *mut Ctx, class: &str, (ops, fields): (&[u32], &str)) -> u32 {
        // SAFETY: `ctx` is the live context of the test, used by one thread.
        unsafe { &mut *ctx }
            .make(class, "f64", &[], None, ops, fields)
            .unwrap()
    }

    /// A read of the per-sample plane `name`.
    fn read(ctx: *mut Ctx, name: &str) -> u32 {
        let fields = format!(
            "(('kind', 'vocab_read'), ('role', 'per_sample'), ('name', '{name}'), ('quantity', None), ('wire_index', 0), ('sorted', False))"
        );
        node(ctx, "Leaf", (&[], &fields))
    }

    /// An operation `kind` over `ops`.
    fn op(ctx: *mut Ctx, kind: &str, ops: &[u32]) -> u32 {
        node(
            ctx,
            "Op",
            (ops, &format!("(('kind', '{kind}'), ('literal', None))")),
        )
    }

    /// The write of `value` to the mutable plane `name`.
    fn write(ctx: *mut Ctx, name: &str, value: u32) -> u32 {
        let fields = format!("(('kind', 'assign'), ('role', 'mutable'), ('name', '{name}'))");
        node(ctx, "Assign", (&[value], &fields))
    }

    /// Oracle: the chain `y = exp(x)`, `z = exp(y)`: its source unit is the caller's face of the whole chain, so it lists `x` and `z`
    /// (`z` is written by the last member only) and not `y`, which the members hand on between them.
    #[test]
    fn the_source_of_a_chain_lists_the_plane_only_the_last_member_writes() {
        let ctx = rx_ctx_new();
        // SAFETY: `ctx` is the live context just made.
        assert_eq!(
            unsafe { raptor_wings::declare::rw_declare_kinds(ctx) },
            RX_OK
        );
        let first = write(ctx, "y", op(ctx, "exp", &[read(ctx, "x")]));
        let second = write(ctx, "z", op(ctx, "exp", &[read(ctx, "y")]));
        let request = KernelRequest::from_json(&format!(
            r#"{{"v": 1, "name": "k", "scalar": "Float64", "targets": ["Host"], "kind": {{"guard": "Default", "active_set": false, "sink": "Plain"}}, "body": {{"Chain": [{{"name": "a", "sinks": [{first}], "declared": [], "exempt": []}}, {{"name": "b", "sinks": [{second}], "declared": [], "exempt": []}}]}}, "facts": {{"Host": "{{}}"}}}}"#
        ))
        .unwrap();
        let RequestBody::Chain(bodies) = &request.body else {
            panic!("a chain body");
        };
        let members: Vec<ChainMember> = bodies
            .iter()
            .map(|m| ChainMember {
                name: &m.name,
                sinks: &m.sinks,
            })
            .collect();
        let source = source_of(ctx, &request, &members, (bodies, &[])).unwrap();
        // SAFETY: `ctx` is the live context made above and is not used after.
        unsafe { rx_ctx_free(ctx) };
        let mut listed: Vec<&str> = source.body.slots.iter().map(|s| s.name.as_str()).collect();
        listed.sort_unstable();
        assert_eq!(listed, ["x", "z"]);
        assert_eq!(source.name, "k_source");
    }
}
