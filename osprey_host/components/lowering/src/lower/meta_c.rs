//! The C source of the meta: one translation unit of constant data that the unit library compiles next to the kernel texts. It
//! includes the runtime's two headers (as `eagle/door/...`), defines the whole tree as `static const` objects and defines the one exported function that
//! returns the tree. A record is written member by member in the header's order through `M(member, value)`: a designated
//! initialiser in C and a positional one in C++17 (which has no designated form), so the same text compiles as both.
//!
//! What the digest of a unit is: the unit's canonical walk digest (SHA-256 of the walk, 64 hex digits in the answer), decoded to
//! its 32 bytes. The device image is named by its bytes symbol and its size symbol; its content is never part of the source.
use super::unit_meta::{
    BranchCost, Derived, Dispatch, DispatchStrategy, Gemm, LibraryMeta, StepsTable, ThreadBox,
    UnitMeta, BRANCH_FLAGS, BRANCH_HAS_STORES, FINISH_FLAGS, GEMM_FLAGS, GEMM_PER_LANE,
    LIBRARY_META_SYMBOL, SCHEMA_MAJOR, SLOT_FLAGS, UNIT_FLAGS,
};
use aether_writers::refuse::{fault, Refusal};
use std::fmt::Write;

/// The first lines of every source: the headers and the member macro.
const PRELUDE: &str =
    "/* The meta of a unit library: constant data and one exported function. Generated. */
#include \"eagle/door/eg_door.h\"
#include \"eagle/door/eg_unit_meta.h\"
#ifdef __cplusplus
#define M(member, ...) __VA_ARGS__
#else
#define M(member, ...) .member = __VA_ARGS__
#endif
";

/// The source being written, with the unit whose objects are being named.
struct Out {
    /// The text so far.
    text: String,
    /// The prefix of the object names of the unit being written.
    unit: String,
}

/// A C string literal of a name.
fn lit(name: &str) -> String {
    let mut out = String::from("\"");
    for byte in name.bytes() {
        match byte {
            b'"' => out.push_str("\\\""),
            b'\\' => out.push_str("\\\\"),
            0x20..=0x7e => out.push(byte as char),
            other => out.push_str(&format!("\\{other:03o}")),
        }
    }
    out + "\""
}
/// A C string literal, or NULL.
fn lit_or_null(name: &Option<String>) -> String {
    name.as_deref().map_or("NULL".into(), lit)
}
/// A slot or table index, -1 when absent.
fn index(at: Option<usize>) -> String {
    at.map_or("-1".into(), |n| n.to_string())
}
/// A flag word as the OR of the header's macro names, 0u when no bit is set.
fn flags(word: u32, names: &[(u32, &str)]) -> String {
    let set: Vec<&str> = names
        .iter()
        .filter(|(bit, _)| word & bit != 0)
        .map(|(_, name)| *name)
        .collect();
    if set.is_empty() {
        "0u".into()
    } else {
        set.join(" | ")
    }
}
/// A flag as 0 or 1.
fn flag(on: bool) -> &'static str {
    if on {
        "1"
    } else {
        "0"
    }
}
/// The 32 bytes of a digest of 64 hex digits, as an initialiser list.
fn digest_bytes(digest: &str) -> Result<String, Refusal> {
    let bytes: Result<Vec<String>, Refusal> = (0..digest.len() / 2)
        .map(|at| {
            digest
                .get(2 * at..2 * at + 2)
                .and_then(|pair| u8::from_str_radix(pair, 16).ok())
                .map(|byte| format!("0x{byte:02x}"))
                .ok_or_else(|| fault(format!("the digest {digest:?} is not hex")))
        })
        .collect();
    let bytes = bytes?;
    if bytes.len() != 32 || digest.len() != 64 {
        return Err(fault(format!("the digest {digest:?} is not 32 bytes")));
    }
    Ok(format!("{{{}}}", bytes.join(", ")))
}

impl Out {
    /// The name of an object of the current unit.
    fn name(&self, what: &str) -> String {
        format!("{}_{what}", self.unit)
    }
    /// A record object: `ty` is the struct's name and `members` its members in the header's order; `struct_size` is the whole struct.
    /// Returns the object's name.
    fn record(&mut self, ty: &str, what: &str, members: &[(&str, String)]) -> String {
        let name = self.name(what);
        let mut line =
            format!("static const struct {ty} {name} = {{M(struct_size, sizeof(struct {ty}))");
        for (member, value) in members {
            let _ = write!(line, ", M({member}, {value})");
        }
        self.text += &(line + "};\n");
        name
    }
    /// An array object of `item` (a type with its qualifiers, e.g. `int64_t`); NULL when there is nothing to list.
    fn array(&mut self, item: &str, what: &str, items: &[String]) -> String {
        if items.is_empty() {
            return "NULL".into();
        }
        let name = self.name(what);
        let _ = writeln!(
            self.text,
            "static const {item} {name}[] = {{{}}};",
            items.join(", ")
        );
        name
    }
    /// An array of pointers to the records named in `items`.
    fn pointers(&mut self, ty: &str, what: &str, items: &[String]) -> String {
        let refs: Vec<String> = items.iter().map(|n| format!("&{n}")).collect();
        self.array(&format!("struct {ty}* const"), what, &refs)
    }
    /// An array of numbers.
    fn numbers<T: ToString>(&mut self, item: &str, what: &str, values: &[T]) -> String {
        let all: Vec<String> = values.iter().map(T::to_string).collect();
        self.array(item, what, &all)
    }
    /// An array of labels, an entry NULL when unlabelled.
    fn labels(&mut self, what: &str, labels: &[Option<String>]) -> String {
        let all: Vec<String> = labels.iter().map(lit_or_null).collect();
        self.array("char* const", what, &all)
    }
    /// An array of names.
    fn names(&mut self, what: &str, names: &[String]) -> String {
        let all: Vec<String> = names.iter().map(|n| lit(n)).collect();
        self.array("char* const", what, &all)
    }
    /// A unit reference object.
    fn unit_ref(&mut self, what: &str, name: &str, digest: &str) -> Result<String, Refusal> {
        let members = [
            ("reserved0", "0".into()),
            ("name", lit(name)),
            ("digest", digest_bytes(digest)?),
        ];
        Ok(self.record("eg_unit_ref", what, &members).to_string())
    }
    /// A thread box object, or NULL.
    fn thread_box(&mut self, what: &str, held: &Option<ThreadBox>) -> String {
        let Some(b) = held else { return "NULL".into() };
        let members = [
            ("axis", b.axis.to_string()),
            ("t", b.t.to_string()),
            ("stride", b.stride.to_string()),
        ];
        format!("&{}", self.record("eg_unit_box", what, &members))
    }
    /// A steps table object, or NULL.
    fn steps_table(&mut self, what: &str, held: &Option<StepsTable>) -> String {
        let Some(t) = held else { return "NULL".into() };
        let widths = self.numbers("int64_t", &format!("{what}_widths"), &t.widths);
        let ks = self.numbers("int64_t", &format!("{what}_ks"), &t.ks);
        let lo = self.numbers("double", &format!("{what}_lo"), &t.lo);
        let hi = self.numbers("double", &format!("{what}_hi"), &t.hi);
        let members = [
            ("regime", t.regime.c_name().into()),
            ("n_rows", t.widths.len().to_string()),
            ("reserved0", "0".into()),
            ("widths", widths),
            ("ks", ks),
            ("lo", lo),
            ("hi", hi),
        ];
        format!("&{}", self.record("eg_unit_steps_table", what, &members))
    }
}

/// Faults on a dispatch record the header's rules do not allow. A group that is neither segmented nor automatic names no branch,
/// classify or sweep unit; an automatic group names the branch units and the classify unit of its segmented set, both or neither. A segmented group has exactly three shapes: the record of a branch unit with a static selector (one unit per branch, the
/// classify unit, no stale slot, no sweep), that of a branch unit with a dynamic selector (one unit per branch, the stale slot and
/// the sweep unit, no classify unit), and that of the classify unit itself (the class slot, and no other slot or unit). Every
/// group has at least two branches.
fn check_dispatch(d: &Dispatch) -> Result<(), Refusal> {
    let broken = |why: &str| fault(format!("the dispatch record of {}: {why}", d.name));
    if d.k < 2 {
        return Err(broken("fewer than two branches"));
    }
    let costed = d.strategy == DispatchStrategy::Auto;
    if d.branch_costs.len() != if costed { d.k } else { 0 } {
        return Err(broken(
            "branch costs not one per branch of an auto group, or on another group",
        ));
    }
    if d.strategy == DispatchStrategy::Auto {
        // an automatic group may name the segmented set built beside it: its branch units and its classify unit, both or neither, and
        // never a sweep unit (a moving selector runs fused)
        let paired = d.branches.is_empty() == d.classify.is_none();
        let sized = d.branches.is_empty() || d.branches.len() == d.k;
        return match (d.sweep.is_none(), paired, sized) {
            (true, true, true) => Ok(()),
            (false, _, _) => Err(broken("a sweep unit on an automatic group")),
            (_, false, _) => Err(broken(
                "branch units without a classify unit, or the reverse, on an automatic group",
            )),
            _ => Err(broken("an automatic group without one unit per branch")),
        };
    }
    if d.strategy != DispatchStrategy::Segmented {
        let named = !d.branches.is_empty() || d.classify.is_some() || d.sweep.is_some();
        return if named {
            Err(broken(
                "branch, classify or sweep units on a group that is not segmented",
            ))
        } else {
            Ok(())
        };
    }
    if d.branches.is_empty() {
        let bare = d.class_slot.is_some()
            && d.mix_slot.is_none()
            && d.stale_slot.is_none()
            && d.offsets_slot.is_none()
            && d.classify.is_none()
            && d.sweep.is_none();
        return if bare {
            Ok(())
        } else {
            Err(broken(
                "a classify unit's record that holds more than the class slot, or none",
            ))
        };
    }
    if d.branches.len() != d.k {
        return Err(broken("a segmented group without one unit per branch"));
    }
    match (d.stale_slot.is_some(), d.classify.is_some(), d.sweep.is_some()) {
        (false, true, false) | (true, false, true) => Ok(()),
        _ => Err(broken("the stale slot, classify unit and sweep unit disagree on whether the selector is static")),
    }
}

/// The branch cost objects of a group, as pointers; NULL for a group that states none. A count the model produced that is not a finite
/// number is a fault: the header's counts are plain numbers.
fn branch_costs(out: &mut Out, at: usize, costs: &[BranchCost]) -> Result<String, Refusal> {
    let mut names = vec![];
    for (j, c) in costs.iter().enumerate() {
        let counts = [
            ("arith32", c.arith32),
            ("arith64", c.arith64),
            ("special32", c.special32),
            ("special64", c.special64),
            ("loads", c.loads),
            ("gathers", c.gathers),
            ("chain32", c.chain32),
            ("chain64", c.chain64),
            ("chain_special32", c.chain_special32),
            ("chain_special64", c.chain_special64),
        ];
        let stored = c.stores.unwrap_or(0.0);
        if counts.iter().any(|(_, n)| !n.is_finite()) || !stored.is_finite() {
            return Err(fault(format!(
                "the branch cost {j} of dispatch group {at}: a count is not a finite number"
            )));
        }
        let word = if c.stores.is_some() {
            BRANCH_HAS_STORES
        } else {
            0
        };
        // the vectorisation bits stay unstated: their verdict is another writer's, not the cost model's
        let mut members = vec![("flags", flags(word, &BRANCH_FLAGS))];
        members.extend(counts.map(|(member, n)| (member, format!("{n:?}"))));
        members.push(("stores", format!("{stored:?}")));
        names.push(out.record("eg_branch_cost", &format!("d{at}_cost{j}"), &members));
    }
    Ok(out.pointers("eg_branch_cost", &format!("d{at}_costs"), &names))
}

/// The objects of one unit's body; returns the body's name.
fn body(out: &mut Out, unit: &UnitMeta) -> Result<String, Refusal> {
    let b = &unit.body;
    let mut slots = vec![];
    for (at, s) in b.slots.iter().enumerate() {
        let shape = out.numbers("int64_t", &format!("s{at}_shape"), &s.shape);
        let gather = s.gather.clone().unwrap_or_default();
        let gather = out.numbers(
            "uint8_t",
            &format!("s{at}_gather"),
            &gather.iter().map(|g| u8::from(*g)).collect::<Vec<_>>(),
        );
        let members = [
            ("flags", flags(s.flags, &SLOT_FLAGS)),
            ("role", s.role.c_name().into()),
            ("dtype", s.dtype.c_name().into()),
            ("rank", s.shape.len().to_string()),
            ("like_slot", index(s.like_slot)),
            (
                "n_gather",
                s.gather.as_ref().map_or(0, Vec::len).to_string(),
            ),
            ("reserved0", "0".into()),
            ("shape", shape),
            ("gather", gather),
            ("name", lit(&s.name)),
        ];
        slots.push(out.record("eg_unit_slot", &format!("s{at}"), &members));
    }
    let mut companions = vec![];
    for (at, c) in b.companions.iter().enumerate() {
        let members = [
            ("slot", c.slot.to_string()),
            ("target_slot", c.target_slot.to_string()),
            ("reserved0", "0".into()),
            ("host_fold_entry", lit_or_null(&c.host_fold_entry)),
            ("device_fold_function", lit_or_null(&c.device_fold_function)),
        ];
        companions.push(out.record("eg_unit_companion", &format!("c{at}"), &members));
    }
    let mut cells = vec![];
    for (at, c) in b.cells.iter().enumerate() {
        let row = c.slots.clone().unwrap_or_default();
        let row = out.numbers("int32_t", &format!("cell{at}_slots"), &row);
        let comp = c.comp_slots.clone().unwrap_or_default();
        let comp = out.numbers("int32_t", &format!("cell{at}_comp"), &comp);
        let members = [
            ("op", c.op.c_name().into()),
            ("internal", flag(c.internal).into()),
            ("columns", c.columns.to_string()),
            ("slots", row),
            ("name", lit(&c.name)),
            ("comp_slots", comp),
        ];
        cells.push(out.record("eg_unit_cell", &format!("cell{at}"), &members));
    }
    let mut keyed = vec![];
    for (at, k) in b.keyed.iter().enumerate() {
        let labels = out.labels(&format!("k{at}_labels"), &k.axis_labels);
        let extents = out.numbers("int64_t", &format!("k{at}_extents"), &k.axis_extents);
        let payload = out.numbers("int64_t", &format!("k{at}_payload"), &k.payload);
        let members = [
            ("n_axes", k.axis_labels.len().to_string()),
            ("payload_rank", k.payload.len().to_string()),
            ("reserved0", "0".into()),
            ("width", k.width.to_string()),
            ("axis_labels", labels),
            ("axis_extents", extents),
            ("payload", payload),
            ("name", lit(&k.name)),
        ];
        keyed.push(out.record("eg_unit_keyed", &format!("k{at}"), &members));
    }
    let mut planes = vec![];
    for (at, p) in b.team_planes.iter().enumerate() {
        let members = [
            ("dtype", p.dtype.c_name().into()),
            ("unit", p.unit.c_name().into()),
            ("layout", p.layout.c_name().into()),
            ("wide", p.wide.c_name().into()),
            ("keyed", index(p.keyed)),
            ("target_slot", index(p.target_slot)),
            ("reserved0", "0".into()),
            ("per_lane", p.per_lane.to_string()),
            ("fill", p.fill.to_string()),
            ("name", lit(&p.name)),
            ("target_stage", index(p.target_stage)),
            ("fill_promised", flag(p.fill_promised).into()),
        ];
        planes.push(out.record("eg_unit_team_plane", &format!("tp{at}"), &members));
    }
    let mut dispatches = vec![];
    for (at, d) in b.dispatches.iter().enumerate() {
        check_dispatch(d)?;
        let classify = match &d.classify {
            Some(r) => format!(
                "&{}",
                out.unit_ref(&format!("d{at}_classify"), &r.name, &r.digest)?
            ),
            None => "NULL".into(),
        };
        let mut refs = vec![];
        for (j, r) in d.branches.iter().enumerate() {
            refs.push(out.unit_ref(&format!("d{at}_branch{j}"), &r.name, &r.digest)?);
        }
        let branches = out.pointers("eg_unit_ref", &format!("d{at}_branches"), &refs);
        let sweep = match &d.sweep {
            Some(r) => format!(
                "&{}",
                out.unit_ref(&format!("d{at}_sweep"), &r.name, &r.digest)?
            ),
            None => "NULL".into(),
        };
        let costs = branch_costs(out, at, &d.branch_costs)?;
        let members = [
            ("strategy", d.strategy.c_name().into()),
            ("mix_slot", index(d.mix_slot)),
            ("class_slot", index(d.class_slot)),
            ("stale_slot", index(d.stale_slot)),
            ("offsets_slot", index(d.offsets_slot)),
            ("classify", classify),
            ("name", lit(&d.name)),
            ("k", d.k.to_string()),
            ("reserved0", "0".into()),
            ("branches", branches),
            ("sweep", sweep),
            ("branch_costs", costs),
            // the vectorisation verdicts of the fused loop are not stated (see `BranchCost`)
            ("cost_flags", "0u".into()),
        ];
        dispatches.push(out.record("eg_unit_dispatch", &format!("d{at}"), &members));
    }
    let members = [
        ("access", b.access.c_name().into()),
        ("n_slots", slots.len().to_string()),
        ("n_companions", companions.len().to_string()),
        ("n_cells", cells.len().to_string()),
        ("n_keyed", keyed.len().to_string()),
        ("n_team_planes", planes.len().to_string()),
        ("n_dispatches", dispatches.len().to_string()),
        ("cell_row", b.cell_row.to_string()),
        ("scatter_chunks", b.scatter_chunks.to_string()),
        ("slots", out.pointers("eg_unit_slot", "slots", &slots)),
        (
            "companions",
            out.pointers("eg_unit_companion", "companions", &companions),
        ),
        ("cells", out.pointers("eg_unit_cell", "cells", &cells)),
        ("keyed", out.pointers("eg_unit_keyed", "keyed", &keyed)),
        (
            "team_planes",
            out.pointers("eg_unit_team_plane", "team_planes", &planes),
        ),
        (
            "dispatches",
            out.pointers("eg_unit_dispatch", "dispatches", &dispatches),
        ),
    ];
    Ok(out.record("eg_unit_body", "body", &members))
}

/// The launch record of a unit; returns its name.
fn launch(out: &mut Out, unit: &UnitMeta) -> String {
    let l = &unit.launch;
    let mut axes = vec![];
    for (at, a) in l.plane_axes.iter().enumerate() {
        let labels = out.labels(&format!("pa{at}_labels"), &a.labels);
        let extents = out.numbers("int64_t", &format!("pa{at}_extents"), &a.extents);
        let members = [
            ("slot", a.slot.to_string()),
            ("n_axes", a.labels.len().to_string()),
            ("reserved0", "0".into()),
            ("labels", labels),
            ("extents", extents),
        ];
        axes.push(out.record("eg_unit_plane_axes", &format!("pa{at}"), &members));
    }
    let members = [
        ("n_lane_axes", l.lane_labels.len().to_string()),
        ("block", l.block.to_string()),
        ("active_map_slot", index(l.active_map_slot)),
        ("active_count_slot", index(l.active_count_slot)),
        ("index_map_slot", index(l.index_map_slot)),
        ("index_count_slot", index(l.index_count_slot)),
        ("n_plane_axes", axes.len().to_string()),
        ("group", l.group.to_string()),
        ("lane_labels", out.labels("lane_labels", &l.lane_labels)),
        (
            "lane_extents",
            out.numbers("int64_t", "lane_extents", &l.lane_extents),
        ),
        (
            "plane_axes",
            out.pointers("eg_unit_plane_axes", "plane_axes", &axes),
        ),
        ("host_box", out.thread_box("host_box", &l.host_box)),
        ("device_box", out.thread_box("device_box", &l.device_box)),
    ];
    out.record("eg_unit_launch", "launch", &members)
}

/// The finish record of a unit, or NULL.
fn finish(out: &mut Out, unit: &UnitMeta) -> String {
    let Some(f) = &unit.finish else {
        return "NULL".into();
    };
    let members = [
        ("flags", flags(f.flags, &FINISH_FLAGS)),
        ("mask_slot", f.mask_slot.to_string()),
        ("counter_slot", f.counter_slot.to_string()),
        ("steps_slot", index(f.steps_slot)),
        ("reserved0", "0".into()),
        ("steps", f.steps.to_string()),
        ("steps_max", f.steps_max.unwrap_or(0).to_string()),
        ("step_ops", f.step_ops.unwrap_or(-1).to_string()),
        ("fp64_ops_ir", f.fp64_ops_ir.unwrap_or(-1).to_string()),
        (
            "host_steps_reverse",
            out.steps_table("host_steps", &f.host_steps_reverse),
        ),
        (
            "device_steps_reverse",
            out.steps_table("device_steps", &f.device_steps_reverse),
        ),
        ("n_events", f.n_events.to_string()),
        ("reserved1", "0".into()),
        (
            "device_persist_function",
            lit_or_null(&f.device_persist_function),
        ),
        (
            "device_range_function",
            lit_or_null(&f.device_range_function),
        ),
    ];
    format!("&{}", out.record("eg_unit_finish", "finish", &members))
}

/// The stages of a unit; returns the pointer array's name.
fn stages(out: &mut Out, unit: &UnitMeta) -> Result<String, Refusal> {
    let mut all = vec![];
    for (at, s) in unit.stages.iter().enumerate() {
        let unit_ref = out.unit_ref(&format!("st{at}_unit"), &s.unit.name, &s.unit.digest)?;
        let team = match &s.team {
            Some(t) => {
                let members = [
                    ("grid", t.grid.c_name().into()),
                    ("threads", t.threads.to_string()),
                    ("warps", t.warps.to_string()),
                    ("lanes", t.lanes.to_string()),
                    ("groups", t.groups.to_string()),
                    ("keyed", index(t.keyed)),
                    ("of_slot", index(t.of_slot)),
                    ("status_plane", index(t.status_plane)),
                    ("of_stage", index(t.of_stage)),
                ];
                format!(
                    "&{}",
                    out.record("eg_unit_team", &format!("st{at}_team"), &members)
                )
            }
            None => "NULL".into(),
        };
        let members = [
            ("n_cells_in", s.cells_in.len().to_string()),
            ("n_cells_out", s.cells_out.len().to_string()),
            ("reserved0", "0".into()),
            ("unit", format!("&{unit_ref}")),
            ("cells_in", out.names(&format!("st{at}_in"), &s.cells_in)),
            ("cells_out", out.names(&format!("st{at}_out"), &s.cells_out)),
            ("team", team),
        ];
        all.push(out.record("eg_unit_stage", &format!("st{at}"), &members));
    }
    Ok(out.pointers("eg_unit_stage", "stages", &all))
}

/// The derived record of a unit, as a pointer to it or NULL for a primal unit.
fn derived(out: &mut Out, derived: Option<&Derived>) -> String {
    let Some(d) = derived else {
        return "NULL".into();
    };
    let taped: Vec<String> = d
        .taped
        .iter()
        .enumerate()
        .map(|(at, t)| {
            let members = [
                ("slot", t.slot.to_string()),
                ("row0", t.row0.to_string()),
                ("rows", t.rows.to_string()),
            ];
            out.record("eg_unit_taped", &format!("dtp{at}"), &members)
        })
        .collect();
    let links: Vec<String> = d
        .links
        .iter()
        .enumerate()
        .map(|(at, l)| {
            let members = [
                ("slot", l.slot.to_string()),
                ("what", l.what.c_name().into()),
                ("of_slot", index(l.of_slot)),
                ("reduced", format!("{}u", flag(l.reduced))),
                ("form", l.form.c_name().into()),
            ];
            out.record("eg_unit_link", &format!("dl{at}"), &members)
        })
        .collect();
    let members = [
        ("role", d.role.c_name().into()),
        ("of_unit", d.of_unit.to_string()),
        ("tape_slot", index(d.tape_slot)),
        ("n_taped", taped.len().to_string()),
        ("tape_rows", d.tape_rows.to_string()),
        ("steps_slot", index(d.steps_slot)),
        ("tape_on_slot", index(d.tape_on_slot)),
        ("time0_slot", index(d.time0_slot)),
        ("fused_steps_slot", index(d.fused_steps_slot)),
        ("n_links", links.len().to_string()),
        ("reserved0", "0".into()),
        ("taped", out.pointers("eg_unit_taped", "taped", &taped)),
        ("links", out.pointers("eg_unit_link", "links", &links)),
    ];
    format!("&{}", out.record("eg_unit_derived", "derived", &members))
}

/// An extent of a stored plane: the header's name for the run's lane count, else the number.
fn extent(n: i64) -> String {
    if n == GEMM_PER_LANE {
        "EG_GEMM_PER_LANE".into()
    } else {
        n.to_string()
    }
}

/// The gemm record of a unit, as a pointer to it or NULL for a unit that is not one.
fn gemm(out: &mut Out, gemm: Option<&Gemm>) -> String {
    let Some(g) = gemm else {
        return "NULL".into();
    };
    let calls: Vec<String> = g
        .calls
        .iter()
        .enumerate()
        .map(|(at, c)| {
            let members = [
                ("flags", flags(c.flags, &GEMM_FLAGS)),
                ("a_slot", c.a_slot.to_string()),
                ("b_slot", c.b_slot.to_string()),
                ("c_slot", c.c_slot.to_string()),
                ("reserved0", "0".into()),
                ("a_rows", extent(c.a_rows)),
                ("a_cols", extent(c.a_cols)),
                ("b_rows", extent(c.b_rows)),
                ("b_cols", extent(c.b_cols)),
                ("c_rows", extent(c.c_rows)),
                ("c_cols", extent(c.c_cols)),
            ];
            out.record("eg_unit_gemm_call", &format!("gc{at}"), &members)
        })
        .collect();
    let members = [
        ("n_calls", calls.len().to_string()),
        ("lane_fanin", g.lane_fanin.to_string()),
        (
            "calls",
            out.pointers("eg_unit_gemm_call", "gemm_calls", &calls),
        ),
    ];
    format!("&{}", out.record("eg_unit_gemm", "gemm", &members))
}

/// One unit's objects; returns the unit record's name.
fn unit(out: &mut Out, unit: &UnitMeta) -> Result<String, Refusal> {
    let stage_list = stages(out, unit)?;
    let derived_list = derived(out, unit.derived.as_ref());
    let gemm_record = gemm(out, unit.gemm.as_ref());
    let (body, launch, finish) = (body(out, unit)?, launch(out, unit), finish(out, unit));
    let members = [
        ("flags", flags(unit.flags, &UNIT_FLAGS)),
        ("name", lit(&unit.name)),
        ("digest", digest_bytes(&unit.digest)?),
        ("pattern", unit.pattern.c_name().into()),
        ("scalar_type", unit.scalar_type.c_name().into()),
        ("host_entry", lit_or_null(&unit.host_entry)),
        ("device_image", lit_or_null(&unit.device_image)),
        ("device_function", lit_or_null(&unit.device_function)),
        ("device_image_bytes", unit.device_image_bytes.to_string()),
        (
            "device_image_format",
            unit.device_image_format.c_name().into(),
        ),
        ("n_stages", unit.stages.len().to_string()),
        ("stages", stage_list),
        ("body", format!("&{body}")),
        ("launch", format!("&{launch}")),
        ("finish", finish),
        ("device_image_size", lit_or_null(&unit.device_image_size)),
        ("derived", derived_list),
        ("device_compiler", lit_or_null(&unit.device_compiler)),
        ("device_flags", lit_or_null(&unit.device_flags)),
        ("device_ptx_isa", unit.device_ptx_isa.to_string()),
        ("reserved1", "0".into()),
        ("gemm", gemm_record),
    ];
    Ok(out.record("eg_unit_meta", "meta", &members))
}

/// The C source of the meta of one kernel's units.
pub fn source(library: &LibraryMeta, units: &[&UnitMeta]) -> Result<String, Refusal> {
    let mut out = Out {
        text: PRELUDE.to_string(),
        unit: String::new(),
    };
    if library.schema_major != SCHEMA_MAJOR {
        return Err(fault(format!(
            "the records are written for schema major {SCHEMA_MAJOR}, not {}",
            library.schema_major
        )));
    }
    let mut named = vec![];
    for (at, held) in units.iter().enumerate() {
        out.unit = format!("u{at}");
        named.push(unit(&mut out, held)?);
    }
    out.unit = "lib".into();
    let list = out.pointers("eg_unit_meta", "units", &named);
    let members = [
        ("schema_major", "EG_META_SCHEMA_MAJOR".into()),
        ("schema_minor", library.schema_minor.to_string() + "u"),
        ("n_units", named.len().to_string() + "u"),
        ("units", list),
    ];
    let lib = out.record("eg_library_meta", "meta", &members);
    let _ = writeln!(
        out.text,
        "__attribute__((visibility(\"default\"))) const struct eg_library_meta* {LIBRARY_META_SYMBOL}(void) {{ return &{lib}; }}"
    );
    Ok(out.text)
}

#[cfg(test)]
mod tests {
    //! A dispatch record that breaks the header's rules is never written.
    use super::*;
    use crate::lower::unit_meta::UnitRef;

    /// A reference to a unit of the given name.
    fn unit(name: &str) -> UnitRef {
        UnitRef {
            name: name.into(),
            digest: "00".repeat(32),
        }
    }

    /// A segmented group of two branches with a static selector.
    fn fixed() -> Dispatch {
        Dispatch {
            strategy: DispatchStrategy::Segmented,
            name: "g".into(),
            k: 2,
            branches: vec![unit("a__k0"), unit("a__k1")],
            classify: Some(unit("a__cls")),
            ..Dispatch::default()
        }
    }

    /// A segmented group of two branches with a dynamic selector.
    fn moving() -> Dispatch {
        Dispatch {
            stale_slot: Some(3),
            classify: None,
            sweep: Some(unit("a__sweep")),
            ..fixed()
        }
    }

    /// The record of the classify unit of a group with a static selector.
    fn lister() -> Dispatch {
        Dispatch {
            strategy: DispatchStrategy::Segmented,
            name: "g".into(),
            k: 2,
            class_slot: Some(1),
            ..Dispatch::default()
        }
    }

    #[test]
    fn the_three_valid_segmented_records_are_written() {
        assert!(check_dispatch(&lister()).is_ok());
        assert!(check_dispatch(&fixed()).is_ok());
        assert!(check_dispatch(&moving()).is_ok());
    }

    #[test]
    fn a_record_whose_members_disagree_is_a_fault_and_not_written() {
        let torn = [
            Dispatch {
                sweep: Some(unit("s")),
                ..fixed()
            },
            Dispatch {
                stale_slot: Some(3),
                ..fixed()
            },
            Dispatch {
                classify: Some(unit("c")),
                ..moving()
            },
            Dispatch {
                sweep: None,
                ..moving()
            },
            Dispatch {
                stale_slot: None,
                ..moving()
            },
            Dispatch {
                branches: vec![unit("a__k0")],
                ..fixed()
            },
            Dispatch {
                k: 1,
                branches: vec![unit("a__k0")],
                ..fixed()
            },
            Dispatch {
                strategy: DispatchStrategy::Switch,
                ..fixed()
            },
        ];
        for record in &torn {
            let why = check_dispatch(record).expect_err("a record that disagrees");
            assert_eq!(why.kind, aether_writers::refuse::RefusalKind::Fault);
        }
    }

    /// An automatic group of two branches that states its costs, with the segmented set built beside it.
    fn auto_with_set() -> Dispatch {
        Dispatch {
            strategy: DispatchStrategy::Auto,
            branch_costs: vec![BranchCost::default(), BranchCost::default()],
            ..fixed()
        }
    }

    #[test]
    fn an_automatic_group_names_its_segmented_set_whole_or_not_at_all() {
        let bare = Dispatch {
            branches: vec![],
            classify: None,
            ..auto_with_set()
        };
        assert!(check_dispatch(&bare).is_ok());
        assert!(check_dispatch(&auto_with_set()).is_ok());
        let torn = [
            Dispatch {
                classify: None,
                ..auto_with_set()
            },
            Dispatch {
                branches: vec![],
                ..auto_with_set()
            },
            Dispatch {
                sweep: Some(unit("a__sweep")),
                ..auto_with_set()
            },
            Dispatch {
                branches: vec![unit("a__k0")],
                ..auto_with_set()
            },
            Dispatch {
                branch_costs: vec![],
                ..auto_with_set()
            },
        ];
        for record in &torn {
            let why = check_dispatch(record).expect_err("an automatic record that disagrees");
            assert_eq!(why.kind, aether_writers::refuse::RefusalKind::Fault);
        }
    }
}
