//! The meta of the answer's units against the sidecars the old build recorded for the same units, member by member, and the mode
//! that writes the C source of the meta of recorded kernels to files (the source's compile test reads those).
use super::{
    ask, facts_all, load, name_as_built, needed_keys, osprey_host_free, recorded_broadcast,
    request_for, texts_of, unit_targets, Found, SIDES,
};
use osprey_host::lower::request::{KernelAnswer, UnitAnswer};
use osprey_host::lower::unit_meta as meta;
use rcore::json::{self, J};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::Path;

/// The members of the meta compared with the record, one line each.
const MEMBERS: [&str; 36] = [
    "name",
    "digest",
    "pattern",
    "scalar_type",
    "host_entry",
    "body.access",
    "body.slots (role, name, order)",
    "slot.dtype",
    "slot.shape (width)",
    "slot.flags PARAM_PLANE",
    "body.cell_row",
    "body.scatter_chunks",
    "body.cells",
    "meta.stages",
    "stage.team",
    "body.team_planes",
    "body.keyed",
    "launch.group",
    "launch.host_box / device_box",
    "launch.index_*, finish.mask_slot, counter_slot",
    "slot.flags SHARED",
    "finish.steps, steps_max",
    "launch.lane_labels, lane_extents",
    "launch.plane_axes",
    "slot.flags PRIOR",
    "body.cells of a unit that lists no stages",
    "team_plane.wide (names)",
    "source unit launch.lane_labels, lane_extents",
    "source unit launch.plane_axes",
    "cell.slots (row planes among the arguments)",
    "companion.slot, target_slot (names)",
    "slot.like_slot (gradient slot, lookup slot)",
    "dispatch.name, strategy, mix_slot (name)",
    "dispatch.class_slot, stale_slot, offsets_slot, classify (names)",
    "finish.step_ops, flags ENTRY_COUNTS",
    "slot.flags GATHER, slot.gather (plane, axis flags)",
];

/// What was compared of one member: the units, those equal, and the first differing ones with both values.
#[derive(Default)]
struct Row {
    /// Units compared.
    compared: usize,
    /// Units equal.
    equal: usize,
    /// Units different, with both values, as many as are shown.
    shown: Vec<String>,
    /// Units different.
    different: usize,
}

/// The meta comparison of one side of one corpus.
#[derive(Default)]
pub struct Oracle {
    /// One row per member.
    rows: BTreeMap<&'static str, Row>,
}

/// A recorded value of a sidecar by key path.
fn at<'a>(sidecar: &'a J, path: &[&str]) -> Option<&'a J> {
    path.iter()
        .try_fold(sidecar, |held, key| held.get(key))
        .filter(|j| !matches!(j, J::Null))
}
/// The text under a key path.
fn word(sidecar: &J, path: &[&str]) -> Option<String> {
    at(sidecar, path).and_then(J::str).map(String::from)
}
/// The whole number under a key path.
fn number(sidecar: &J, path: &[&str]) -> Option<i64> {
    match at(sidecar, path) {
        Some(J::Num(n)) => Some(*n as i64),
        _ => None,
    }
}
/// A header macro name as the lower-case word the record uses.
fn lower(c_name: &str, prefix: &str) -> String {
    c_name.trim_start_matches(prefix).to_lowercase()
}
/// A flag the record states, false when it does not.
fn flag(sidecar: &J, path: &[&str]) -> bool {
    matches!(at(sidecar, path), Some(J::Bool(true)))
}
/// The entries of a recorded object as key and value pairs.
fn pairs<'a>(sidecar: &'a J, path: &[&str]) -> Vec<(&'a str, &'a J)> {
    match at(sidecar, path) {
        Some(J::Obj(items)) => items.iter().map(|(k, v)| (k.as_str(), v)).collect(),
        _ => vec![],
    }
}
/// The texts in order of their words: the record keeps a dictionary's keys sorted, so the order of its entries says nothing.
fn sorted(items: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut all: Vec<String> = items.into_iter().collect();
    all.sort();
    all
}
/// A list of texts as one text.
fn joined(items: impl IntoIterator<Item = String>) -> String {
    items.into_iter().collect::<Vec<_>>().join(" ")
}

/// A keyed plane as a text: width, the kept axes and the payload.
fn keyed_text(k: &meta::Keyed) -> String {
    let axes = k
        .axis_labels
        .iter()
        .zip(&k.axis_extents)
        .map(|(label, extent)| {
            format!(
                "{}={}",
                label.as_deref().unwrap_or("-"),
                if *extent < 0 {
                    "-".into()
                } else {
                    extent.to_string()
                }
            )
        });
    format!("{} [{}] {:?}", k.width, joined(axes), k.payload)
}
/// A recorded keyed plane as the same text.
fn recorded_keyed(j: &J) -> String {
    let axes = j.get("axes").map_or(&[][..], J::arr).iter().map(|axis| {
        let (label, extent) = (axis.arr().first(), axis.arr().get(1));
        format!(
            "{}={}",
            label.and_then(J::str).unwrap_or("-"),
            match extent {
                Some(J::Num(n)) => (*n as i64).to_string(),
                _ => "-".into(),
            }
        )
    });
    let payload: Vec<i64> = j
        .get("payload")
        .map_or(&[][..], J::arr)
        .iter()
        .filter_map(|n| {
            if let J::Num(n) = n {
                Some(*n as i64)
            } else {
                None
            }
        })
        .collect();
    let width = number(j, &["width"]).unwrap_or(0);
    format!("{width} [{}] {payload:?}", joined(axes))
}

/// The element type of a slot as the record names it.
fn dtype_word(unit: &meta::UnitMeta, dtype: meta::WireDtype) -> String {
    match dtype {
        meta::WireDtype::Real => lower(unit.scalar_type.c_name(), "EG_SCALAR_"),
        other => lower(other.c_name(), "EG_DTYPE_"),
    }
}

/// The compared values of one member of a unit: what the meta says and what the sidecar recorded; none when the member is not
/// filled or not recorded for this unit.
type Pair = Option<(String, String)>;

/// The members of the first group: names, entries, access, the slot table.
fn slot_rows(unit: &meta::UnitMeta, sc: &J) -> Vec<Pair> {
    let b = &unit.body;
    let spec: Vec<String> = at(sc, &["arg_spec"])
        .map_or(&[][..], J::arr)
        .iter()
        .map(|p| {
            format!(
                "{}:{}",
                p.arr().first().and_then(J::str).unwrap_or(""),
                p.arr().get(1).and_then(J::str).unwrap_or("")
            )
        })
        .collect();
    let ours = b
        .slots
        .iter()
        .map(|s| format!("{}:{}", lower(s.role.c_name(), "EG_ROLE_"), s.name));
    let dtypes = pairs(sc, &["arg_dtypes"]);
    let dtype = (
        joined(
            b.slots
                .iter()
                .filter(|s| dtypes.iter().any(|(n, _)| *n == s.name))
                .map(|s| format!("{}={}", s.name, dtype_word(unit, s.dtype))),
        ),
        joined(
            b.slots
                .iter()
                .filter_map(|s| dtypes.iter().find(|(n, _)| *n == s.name))
                .map(|(n, d)| format!("{n}={}", d.str().unwrap_or(""))),
        ),
    );
    let widths = pairs(sc, &["arg_widths"]);
    let width = (
        joined(
            b.slots
                .iter()
                .filter(|s| widths.iter().any(|(n, _)| *n == s.name))
                .map(|s| format!("{}={}", s.name, s.shape.iter().product::<i64>())),
        ),
        joined(
            b.slots
                .iter()
                .filter_map(|s| widths.iter().find(|(n, _)| *n == s.name))
                .map(|(n, w)| format!("{n}={}", number(w, &[]).unwrap_or(0))),
        ),
    );
    let planar = |names: Vec<String>| {
        let mut names = names;
        names.sort();
        joined(names)
    };
    let param = (
        planar(
            b.slots
                .iter()
                .filter(|s| s.flags & meta::SLOT_PARAM_PLANE != 0)
                .map(|s| s.name.clone())
                .collect(),
        ),
        planar(
            at(sc, &["param_planes"])
                .map_or(&[][..], J::arr)
                .iter()
                .filter_map(|n| n.str().map(String::from))
                .collect(),
        ),
    );
    vec![
        Some((joined(ours), joined(spec))),
        Some(dtype),
        Some(width),
        Some(param),
    ]
}

/// The stage facts of the unit that lists the kernel's stages, against the sidecar of the kernel's first stage. A scatter's
/// target and a plane's target are named in the lister's body, or in the body of the stage unit the answer says holds them.
fn stage_rows(unit: &meta::UnitMeta, answer: &KernelAnswer, sc: &J) -> Vec<Pair> {
    let Some(unit) = answer.units.iter().map(|u| &u.meta).find(|lister| {
        lister
            .stages
            .first()
            .is_some_and(|s| s.unit.name == unit.name)
    }) else {
        return vec![None; 6];
    };
    let b = &unit.body;
    let cells = (
        joined(sorted(b.cells.iter().map(|c| {
            format!(
                "{}:{}:{}:{}",
                c.name,
                lower(c.op.c_name(), "EG_REDUCE_"),
                c.columns,
                c.internal
            )
        }))),
        joined(sorted(pairs(sc, &["cells"]).into_iter().map(|(n, c)| {
            format!(
                "{n}:{}:{}:{}",
                word(c, &["op"]).unwrap_or_default(),
                number(c, &["columns"]).unwrap_or(0),
                flag(c, &["internal"])
            )
        }))),
    );
    let stages = (
        joined(unit.stages.iter().map(|s| {
            format!(
                "{}[{}>{}]",
                s.unit.name,
                s.cells_in.join(","),
                s.cells_out.join(",")
            )
        })),
        joined(at(sc, &["stages"]).map_or(&[][..], J::arr).iter().map(|s| {
            let names = |key: &str| {
                s.get(key)
                    .map_or(&[][..], J::arr)
                    .iter()
                    .filter_map(J::str)
                    .collect::<Vec<_>>()
                    .join(",")
            };
            format!(
                "{}[{}>{}]",
                word(s, &["kernel"]).unwrap_or_default(),
                names("cells_in"),
                names("cells_out")
            )
        })),
    );
    let slot = |at: Option<usize>, stage: Option<usize>| {
        let slots = match stage {
            None => Some(&b.slots),
            Some(n) => unit
                .stages
                .get(n)
                .and_then(|s| answer.units.iter().find(|u| u.meta.name == s.unit.name))
                .map(|u| &u.meta.body.slots),
        };
        at.and_then(|n| slots?.get(n))
            .map_or("-".to_string(), |s| s.name.clone())
    };
    let team_ours = joined(unit.stages.iter().filter_map(|s| {
        let t = s.team.as_ref()?;
        let keyed = t
            .keyed
            .and_then(|n| b.keyed.get(n))
            .map_or("-".into(), keyed_text);
        let status = t
            .status_plane
            .and_then(|n| b.team_planes.get(n))
            .map_or("-", |p| p.name.as_str());
        Some(format!(
            "{}:{}:{}:{}:{}:{}:{}:{}:{}",
            s.unit.name,
            lower(t.grid.c_name(), "EG_GRID_"),
            t.threads,
            t.lanes,
            t.warps,
            t.groups,
            keyed,
            slot(t.of_slot, t.of_stage),
            status
        ))
    }));
    let team_recorded = joined(
        at(sc, &["stages"])
            .map_or(&[][..], J::arr)
            .iter()
            .filter_map(|s| {
                let t = s.get("team")?;
                Some(format!(
                    "{}:{}:{}:{}:{}:{}:{}:{}:{}",
                    word(s, &["kernel"]).unwrap_or_default(),
                    word(t, &["grid"]).unwrap_or_else(|| "lanes".into()),
                    number(t, &["threads"]).unwrap_or(0),
                    number(t, &["lanes"]).unwrap_or(1),
                    number(t, &["warps"]).unwrap_or(1),
                    number(t, &["groups"]).unwrap_or(1),
                    t.get("keyed").map_or("-".into(), recorded_keyed),
                    word(t, &["of"]).unwrap_or_else(|| "-".into()),
                    word(t, &["status"]).unwrap_or_else(|| "-".into()),
                ))
            }),
    );
    let planes = (
        joined(sorted(b.team_planes.iter().map(|p| {
            format!(
                "{}:{}:{}:{}:{}:{}",
                p.name,
                lower(p.dtype.c_name(), "EG_DTYPE_"),
                p.per_lane,
                p.layout == meta::PlaneLayout::PlaneMajor,
                p.unit == meta::PlaneUnit::Once,
                p.fill
            )
        }))),
        joined(sorted(pairs(sc, &["team_planes"]).into_iter().map(
            |(n, p)| {
                format!(
                    "{n}:{}:{}:{}:{}:{}",
                    word(p, &["dtype"]).unwrap_or_default(),
                    number(p, &["per_lane"]).unwrap_or(0),
                    word(p, &["layout"]).as_deref() == Some("plane_major"),
                    flag(p, &["once"]),
                    number(p, &["fill"]).unwrap_or(0)
                )
            },
        ))),
    );
    let recorded = pairs(sc, &["keyed"]);
    let keyed = (!recorded.is_empty()).then(|| {
        (
            joined(
                b.keyed
                    .iter()
                    .filter(|k| recorded.iter().any(|(n, _)| *n == k.name))
                    .map(|k| format!("{}={}", k.name, keyed_text(k))),
            ),
            joined(
                recorded
                    .iter()
                    .map(|(n, k)| format!("{n}={}", recorded_keyed(k))),
            ),
        )
    });
    vec![
        Some(cells),
        Some(stages),
        Some((team_ours, team_recorded)),
        Some(planes),
        keyed,
        Some((
            b.scatter_chunks.to_string(),
            number(sc, &["scatter_chunks"]).unwrap_or(0).to_string(),
        )),
    ]
}

/// The launch and finish members of a unit.
fn launch_rows(unit: &meta::UnitMeta, sc: &J, unboxed: bool) -> Vec<Pair> {
    let l = &unit.launch;
    let boxed = |held: &Option<meta::ThreadBox>| {
        held.as_ref().map_or("none".to_string(), |b| {
            format!("{}/{}/{}", b.axis, b.t, b.stride)
        })
    };
    let recorded = at(sc, &["launch", "box"]).map_or("none".to_string(), |b| {
        format!(
            "{}/{}/{}",
            number(b, &["axis"]).unwrap_or(0),
            number(b, &["T"]).unwrap_or(0),
            number(b, &["stride"]).unwrap_or(0)
        )
    });
    let device = word(sc, &["format"]).as_deref() == Some("cuda");
    let ours = boxed(if device { &l.device_box } else { &l.host_box });
    let slots = &unit.body.slots;
    let name = |at: Option<usize>| {
        at.and_then(|n| slots.get(n))
            .map_or("-", |s| s.name.as_str())
    };
    let width = |at: Option<usize>| {
        at.and_then(|n| slots.get(n))
            .map_or("-", |s| match s.dtype {
                meta::WireDtype::Int64 => "64",
                _ => "32",
            })
    };
    let index = match (l.index_map_slot, l.index_count_slot) {
        (Some(map), _) => format!("set:{}", width(Some(map))),
        (None, Some(_)) => "prefix".to_string(),
        _ => "none".to_string(),
    };
    // a prefix set has no map, so no word size: it is compared with the record's set by presence
    let recorded_index = at(sc, &["launch", "index_set"]).map_or("none".to_string(), |s| {
        match (index.as_str(), number(s, &["width"])) {
            ("prefix", _) => "prefix".to_string(),
            (_, Some(w)) => format!("set:{w}"),
            _ => "set".to_string(),
        }
    });
    let finish = unit.finish.as_ref().map_or("none".to_string(), |f| {
        format!("{}:{}", name(Some(f.mask_slot)), name(Some(f.counter_slot)))
    });
    let recorded_finish = if flag(sc, &["terminated_readonly"]) {
        "none".to_string()
    } else {
        format!(
            "{}:{}",
            word(sc, &["finish", "mask"]).unwrap_or_default(),
            word(sc, &["finish", "counter"]).unwrap_or_default()
        )
    };
    vec![
        Some((
            l.group.to_string(),
            number(sc, &["launch", "lane_space", "group"])
                .unwrap_or(1)
                .to_string(),
        )),
        (!unboxed).then_some((ours, recorded)),
        Some((
            format!("{index} {finish}"),
            format!("{recorded_index} {recorded_finish}"),
        )),
    ]
}

/// An axis as `label=extent`, `-` for what it lacks.
fn axis_text(label: Option<&str>, extent: Option<i64>) -> String {
    format!(
        "{}={}",
        label.unwrap_or("-"),
        extent
            .filter(|n| *n >= 0)
            .map_or("-".into(), |n| n.to_string())
    )
}

/// The members the caller declares: the shared planes, the steps, the lane space and the sink axes.
fn declared_rows(
    unit: &meta::UnitMeta,
    unsplit: bool,
    lister: Option<&meta::UnitMeta>,
    sc: &J,
) -> Vec<Pair> {
    let slots = &unit.body.slots;
    let shared = (
        joined(sorted(
            slots
                .iter()
                .filter(|s| s.flags & meta::SLOT_SHARED != 0)
                .map(|s| s.name.clone()),
        )),
        joined(sorted(
            pairs(sc, &["param_shapes"])
                .into_iter()
                .map(|(n, _)| n.to_string()),
        )),
    );
    let recorded_finish = at(sc, &["finish"]).filter(|_| !flag(sc, &["terminated_readonly"]));
    let steps = (unit.finish.is_some() || recorded_finish.is_some()).then(|| {
        let ours = unit.finish.as_ref().map_or("none".to_string(), |f| {
            let steps = match f.steps {
                -1 => "auto".to_string(),
                other => other.to_string(),
            };
            format!("{steps}:{}", f.steps_max.unwrap_or(0))
        });
        let recorded = recorded_finish.map_or("none".to_string(), |f| {
            let steps = word(f, &["steps"])
                .unwrap_or_else(|| number(f, &["steps"]).unwrap_or(0).to_string());
            format!("{steps}:{}", number(f, &["steps_max"]).unwrap_or(0))
        });
        (ours, recorded)
    });
    let l = &unit.launch;
    let lane = (
        joined(
            l.lane_labels
                .iter()
                .zip(&l.lane_extents)
                .map(|(label, extent)| axis_text(label.as_deref(), Some(*extent))),
        ),
        joined(
            at(sc, &["launch", "lane_space", "axes"])
                .map_or(&[][..], J::arr)
                .iter()
                .map(|a| axis_text(word(a, &["label"]).as_deref(), number(a, &["extent"]))),
        ),
    );
    let recorded_axes = pairs(sc, &["plane_axes"]);
    let axes = (!recorded_axes.is_empty() || !l.plane_axes.is_empty()).then(|| {
        let ours = l
            .plane_axes
            .iter()
            .filter(|p| !slots.get(p.slot).is_some_and(|s| s.name.ends_with("_c")))
            .map(|p| {
                let axes = p
                    .labels
                    .iter()
                    .zip(&p.extents)
                    .map(|(label, extent)| axis_text(label.as_deref(), Some(*extent)));
                format!(
                    "{}:{}",
                    slots.get(p.slot).map_or("?", |s| s.name.as_str()),
                    joined(axes)
                )
            });
        // a compensation plane (`<name>_c`) rides with its sum: the sum states the axes, on both sides
        let theirs = recorded_axes
            .iter()
            .filter(|(n, _)| {
                !n.strip_suffix("_c")
                    .is_some_and(|base| recorded_axes.iter().any(|(b, _)| *b == base))
            })
            .map(|(n, p)| {
                let axes = at(p, &["axes"])
                    .map_or(&[][..], J::arr)
                    .iter()
                    .map(|a| axis_text(word(a, &["label"]).as_deref(), number(a, &["extent"])));
                format!("{n}:{}", joined(axes))
            });
        (joined(sorted(ours)), joined(sorted(theirs)))
    });
    let prior = (
        joined(sorted(
            slots
                .iter()
                .filter(|s| s.flags & meta::SLOT_PRIOR != 0)
                .map(|s| s.name.clone()),
        )),
        joined(sorted(
            at(sc, &["mutables"])
                .map_or(&[][..], J::arr)
                .iter()
                .filter(|m| flag(m, &["prior"]))
                .filter_map(|m| word(m, &["name"])),
        )),
    );
    let cells = (unsplit && (!unit.body.cells.is_empty() || !pairs(sc, &["cells"]).is_empty()))
        .then(|| {
            (
                joined(sorted(unit.body.cells.iter().map(|c| {
                    format!(
                        "{}:{}:{}:{}",
                        c.name,
                        lower(c.op.c_name(), "EG_REDUCE_"),
                        c.columns,
                        c.internal
                    )
                }))),
                joined(sorted(pairs(sc, &["cells"]).into_iter().map(|(n, c)| {
                    format!(
                        "{n}:{}:{}:{}",
                        word(c, &["op"]).unwrap_or_default(),
                        number(c, &["columns"]).unwrap_or(0),
                        flag(c, &["internal"])
                    )
                }))),
            )
        });
    let recorded_wide: Vec<String> = pairs(sc, &["launch", "wide"])
        .into_iter()
        .map(|(n, _)| n.to_string())
        .filter(|n| n != "class")
        .collect();
    let wide = (lister.is_some() || !recorded_wide.is_empty()).then(|| {
        let ours = lister.map_or(vec![], |l| {
            l.body
                .team_planes
                .iter()
                .filter(|p| p.wide != meta::WideLayout::None)
                .map(|p| p.name.clone())
                .collect()
        });
        (joined(sorted(ours)), joined(sorted(recorded_wide)))
    });
    vec![
        Some(shared),
        steps,
        Some(lane),
        axes,
        Some(prior),
        cells,
        wide,
    ]
}

/// The lane space and sink axes of a source unit (the kernel as the caller saw it) against the sidecar of the kernel's first stage.
fn source_rows(lister: Option<&meta::UnitMeta>, sc: &J) -> Vec<Pair> {
    match lister.filter(|l| l.host_entry.is_none() && l.device_function.is_none()) {
        Some(source) => {
            let rows = declared_rows(source, false, None, sc);
            vec![rows[2].clone(), rows[3].clone()]
        }
        None => vec![None, None],
    }
}

/// The positions of the row planes of the cells of a unit that lists no stages, against the positions of the recorded
/// `accum_out` planes of the same names in the recorded argument spec.
fn cell_slot_row(unit: &meta::UnitMeta, unsplit: bool, sc: &J) -> Pair {
    let recorded = pairs(sc, &["cells"]);
    if !unsplit || recorded.is_empty() || unit.body.cells.is_empty() {
        return None;
    }
    let spec: Vec<(String, String)> = at(sc, &["arg_spec"])
        .map_or(&[][..], J::arr)
        .iter()
        .map(|p| {
            (
                p.arr().first().and_then(J::str).unwrap_or("").to_string(),
                p.arr().get(1).and_then(J::str).unwrap_or("").to_string(),
            )
        })
        .collect();
    let ours =
        joined(sorted(unit.body.cells.iter().map(|c| {
            format!("{}:{:?}", c.name, c.slots.clone().unwrap_or_default())
        })));
    let theirs = joined(sorted(recorded.iter().map(|(n, cell)| {
        // a cell of one column fills the plane of its own name, a wider one the planes `<name>_<k>`
        let columns = number(cell, &["columns"]).unwrap_or(1);
        let named = |name: &str| {
            if columns == 1 {
                name == *n
            } else {
                (0..columns).any(|k| name == format!("{n}_{k}"))
            }
        };
        let at: Vec<usize> = (0..spec.len())
            .filter(|&p| spec[p].0 == "accum_out" && named(&spec[p].1))
            .collect();
        format!("{n}:{at:?}")
    })));
    Some((ours, theirs))
}

/// The companion pairs of a unit, each as the names of its two slots, against the recorded `companions` map (companion to target).
fn companion_row(unit: &meta::UnitMeta, sc: &J) -> Pair {
    let name = |at: usize| unit.body.slots.get(at).map_or("?", |s| s.name.as_str());
    let ours = joined(sorted(
        unit.body
            .companions
            .iter()
            .map(|c| format!("{}:{}", name(c.slot), name(c.target_slot))),
    ));
    let theirs =
        joined(sorted(pairs(sc, &["companions"]).into_iter().map(
            |(comp, target)| format!("{comp}:{}", target.str().unwrap_or("")),
        )));
    Some((ours, theirs))
}

/// The gradient slots of a unit, each as its name and the name of the slot it is sized like, against the recorded `adjoint_like`.
fn like_row(unit: &meta::UnitMeta, sc: &J) -> Pair {
    let slots = &unit.body.slots;
    let ours = joined(sorted(slots.iter().filter_map(|s| {
        s.like_slot
            .map(|like| format!("{}:{}", s.name, slots[like].name))
    })));
    let theirs =
        joined(sorted(pairs(sc, &["adjoint_like"]).into_iter().map(
            |(plane, like)| format!("{plane}:{}", like.str().unwrap_or("")),
        )));
    Some((ours, theirs))
}

/// The dispatch groups of a unit as name, strategy and branch-mix plane, against the recorded `dispatch` list.
fn dispatch_row(unit: &meta::UnitMeta, sc: &J) -> Pair {
    let ours = joined(sorted(unit.body.dispatches.iter().map(|d| {
        let mix = d
            .mix_slot
            .map_or("", |at| unit.body.slots[at].name.as_str());
        format!(
            "{}:{}:{mix}",
            d.name,
            lower(d.strategy.c_name(), "EG_DISPATCH_")
        )
    })));
    let theirs = joined(sorted(
        at(sc, &["dispatch"])
            .map_or(&[][..], J::arr)
            .iter()
            .map(|d| {
                let field = |key: &str| word(d, &[key]).unwrap_or_default();
                format!("{}:{}:{}", field("name"), field("strategy"), field("mix"))
            }),
    ));
    Some((ours, theirs))
}

/// The planes and classify unit of the segmented groups of a unit by name. The recorded sidecar names the class and stale planes
/// of every group, but a unit holds them as slots only when the selector is dynamic: the others are left out of the record's side.
fn segment_row(unit: &meta::UnitMeta, sc: &J) -> Pair {
    let name = |at: Option<usize>| at.map_or("", |at| unit.body.slots[at].name.as_str());
    let segmented = meta::DispatchStrategy::Segmented;
    let ours = joined(sorted(
        unit.body
            .dispatches
            .iter()
            .filter(|d| d.strategy == segmented)
            .map(|d| {
                let classify = d.classify.as_ref().map_or(String::new(), |u| {
                    format!(
                        "{}{}",
                        u.name,
                        if u.digest.len() == 64 {
                            ""
                        } else {
                            " without a digest"
                        }
                    )
                });
                format!(
                    "{}:{}:{}:{}:{classify}",
                    name(d.class_slot),
                    name(d.stale_slot),
                    name(d.offsets_slot),
                    d.name
                )
            }),
    ));
    let theirs = joined(sorted(
        at(sc, &["dispatch"])
            .map_or(&[][..], J::arr)
            .iter()
            .filter(|d| word(d, &["strategy"]).as_deref() == Some("segmented"))
            .map(|d| {
                let field = |key: &str| word(d, &[key]).unwrap_or_default();
                let dynamic = !flag(d, &["static_selector"]);
                let held = |key: &str| if dynamic { field(key) } else { String::new() };
                format!(
                    "{}:{}:{}:{}:{}",
                    held("class"),
                    held("stale"),
                    field("offsets"),
                    field("name"),
                    field("classify")
                )
            }),
    ));
    Some((ours, theirs))
}

/// The step count and the entry-count flag of a finishing unit, against the recorded `step_ops` and `entry_counts_finished`.
fn finish_row(unit: &meta::UnitMeta, sc: &J) -> Pair {
    let finish = unit.finish.as_ref()?;
    let ours = format!(
        "{:?} {}",
        finish.step_ops,
        finish.flags & meta::FINISH_ENTRY_COUNTS != 0
    );
    let theirs = format!(
        "{:?} {}",
        number(sc, &["step_ops"]),
        flag(sc, &["entry_counts_finished"])
    );
    Some((ours, theirs))
}

/// The gathered planes of a unit with the lane axes each varies along (none when the request stated none), against the recorded
/// `gather_mats`.
fn gather_row(unit: &meta::UnitMeta, sc: &J) -> Pair {
    let word = |flags: Option<&Vec<bool>>| {
        flags.map_or("none".to_string(), |f| {
            f.iter().map(|b| if *b { '1' } else { '0' }).collect()
        })
    };
    let ours = joined(sorted(
        unit.body
            .slots
            .iter()
            .filter(|s| s.flags & meta::SLOT_GATHER != 0)
            .map(|s| format!("{}:{}", s.name, word(s.gather.as_ref()))),
    ));
    let theirs = joined(sorted(pairs(sc, &["gather_mats"]).into_iter().map(
        |(plane, flags)| {
            let flags = match flags {
                J::Arr(f) => Some(f.iter().map(|b| matches!(b, J::Bool(true))).collect()),
                _ => None,
            };
            format!("{plane}:{}", word(flags.as_ref()))
        },
    )));
    Some((ours, theirs))
}

impl Oracle {
    /// Compares the meta of an answer unit with the sidecar recorded for it; a unit with no sidecar is not counted.
    pub fn unit(
        &mut self,
        id: &str,
        unit: &UnitAnswer,
        answer: &KernelAnswer,
        record: &J,
        unboxed: bool,
    ) {
        let Some(sc) = record.get("sidecar").filter(|s| !matches!(s, J::Null)) else {
            return;
        };
        let m = &unit.meta;
        let host = word(sc, &["format"]).as_deref() == Some("host");
        let unsplit = !answer
            .units
            .iter()
            .any(|u| u.meta.stages.iter().any(|s| s.unit.name == m.name));
        let lister = answer
            .units
            .iter()
            .map(|u| &u.meta)
            .find(|l| l.stages.first().is_some_and(|s| s.unit.name == m.name));
        let slots = slot_rows(m, sc);
        let stages = stage_rows(m, answer, sc);
        let launch = launch_rows(m, sc, unboxed);
        let cell_row = (
            m.body.cell_row.to_string(),
            number(sc, &["cell_row"]).unwrap_or(0).to_string(),
        );
        let pairs: Vec<Pair> = [
            Some((m.name.clone(), word(sc, &["kernel"]).unwrap_or_default())),
            Some((m.digest.clone(), word(sc, &["digest"]).unwrap_or_default())),
            Some((
                lower(m.pattern.c_name(), "EG_PATTERN_"),
                word(sc, &["pattern"]).unwrap_or_default(),
            )),
            Some((
                lower(m.scalar_type.c_name(), "EG_SCALAR_"),
                word(sc, &["scalar_type"]).unwrap_or_default(),
            )),
            host.then(|| {
                (
                    m.host_entry.clone().unwrap_or_default(),
                    word(sc, &["host_entry"]).unwrap_or_default(),
                )
            }),
            Some((
                lower(m.body.access.c_name(), "EG_ACCESS_"),
                word(sc, &["exec_access"]).unwrap_or_default(),
            )),
        ]
        .into_iter()
        .chain(slots)
        .chain([Some(cell_row)])
        .chain([stages[5].clone()])
        .chain(stages[..5].iter().cloned())
        .chain(launch)
        .chain(declared_rows(m, unsplit, lister, sc))
        .chain(source_rows(lister, sc))
        .chain([
            cell_slot_row(m, unsplit, sc),
            companion_row(m, sc),
            like_row(m, sc),
            dispatch_row(m, sc),
            segment_row(m, sc),
            finish_row(m, sc),
            gather_row(m, sc),
        ])
        .collect();
        // the member order of `MEMBERS`: the scatter chunks sit after the cell row
        for (member, pair) in MEMBERS.iter().zip(pairs) {
            let Some((ours, recorded)) = pair else {
                continue;
            };
            let row = self.rows.entry(member).or_default();
            row.compared += 1;
            if ours == recorded {
                row.equal += 1;
            } else {
                row.different += 1;
                if row.shown.len() < 3 {
                    row.shown.push(format!(
                        "{id} {}: meta {ours:?} / recorded {recorded:?}",
                        m.name
                    ));
                }
            }
        }
    }

    /// Prints one line per member and the first differing units.
    pub fn print(&self, side: &str) {
        println!("meta against the recorded sidecars, {side}: members, units compared / equal / different");
        for member in MEMBERS {
            let row = self.rows.get(member);
            let (c, e, d) = row.map_or((0, 0, 0), |r| (r.compared, r.equal, r.different));
            println!("  {member}: {c} / {e} / {d}");
            for shown in row.map_or(&[][..], |r| &r.shown[..]) {
                println!("    {}", &shown[..shown.len().min(4000)]);
            }
        }
    }
}

/// Writes the C source of the meta of every kernel of the recorded calls in `file` into `dir`, one file per kernel and side, and
/// prints one line per file with what the answer holds.
///
/// With `only`, just the call of that id is written, and each unit's text per target is written beside the source; one line names
/// the test that made the call.
pub fn emit(file: &Path, dir: &Path, only: Option<&str>) {
    std::fs::create_dir_all(dir).expect("the output folder");
    let text = std::fs::read_to_string(file).expect("the calls");
    let mut machines = HashMap::new();
    // the recorded texts beside the calls hold what only the caller knew (the broadcast pattern): read as the replay reads them
    let texts = file.with_file_name(
        file.file_name()
            .map_or(String::new(), |n| n.to_string_lossy().replacen('k', "t", 1)),
    );
    let mut expected = None;
    for line in text.lines() {
        let mut call = json::parse(line).expect("a call");
        if only.is_some_and(|wanted| call.get("call").and_then(J::str) != Some(wanted)) {
            continue;
        }
        if texts.is_file() {
            let both = expected.get_or_insert_with(|| {
                let wanted = needed_keys(file);
                [
                    texts_of(&texts, &wanted, "host"),
                    texts_of(&texts, &wanted, "cuda"),
                ]
            });
            name_as_built(&mut call);
            recorded_broadcast(&mut call, both);
        }
        let id = call.get("call").and_then(J::str).unwrap_or("");
        if only.is_some() {
            println!(
                "test {id}: {}",
                call.get("test").and_then(J::str).unwrap_or("")
            );
        }
        let units = call.get("units").map_or(&[][..], J::arr);
        let mut ctx = None;
        for kernel in call.get("kernels").map_or(&[][..], J::arr) {
            let name = kernel.get("name").and_then(J::str).unwrap_or("");
            for (target, side) in SIDES {
                if !unit_targets(kernel, units).contains(&side) {
                    continue;
                }
                let found = Found {
                    call: &call,
                    id,
                    show: 0,
                    target,
                };
                let answer = (|| {
                    let facts = facts_all(&found, kernel, &mut machines)?;
                    if ctx.is_none() {
                        ctx = Some(load(&call)?);
                    }
                    let (c, handles) = ctx.as_ref().ok_or("no context")?;
                    ask(*c, &request_for(&call, kernel, facts, handles)?.to_json())
                })();
                let answer = answer.unwrap_or_else(|why| panic!("{id} {name} {side}: {why}"));
                let file = dir.join(format!("{id}_{name}_{side}.c"));
                std::fs::write(&file, &answer.meta_source).expect("the source");
                if only.is_some() {
                    for unit in &answer.units {
                        for text in unit.texts.iter().filter(|t| t.target == target) {
                            let ext = if side == "host" { "cpp" } else { "cu" };
                            let name = format!("{id}_{}_{side}.{ext}", unit.meta.name);
                            std::fs::write(dir.join(name), &text.text).expect("the text");
                        }
                    }
                }
                let held: Vec<&meta::UnitMeta> = answer.units.iter().map(|u| &u.meta).collect();
                println!(
                    "meta {}: units {} stages {} finish {} host_box {} device_box {} team {} companions {} image {}",
                    file.display(),
                    held.len(),
                    held.iter().map(|m| m.stages.len()).sum::<usize>(),
                    held.iter().filter(|m| m.finish.is_some()).count(),
                    held.iter().filter(|m| m.launch.host_box.is_some()).count(),
                    held.iter().filter(|m| m.launch.device_box.is_some()).count(),
                    held.iter().flat_map(|m| &m.stages).filter(|s| s.team.is_some()).count(),
                    held.iter().map(|m| m.body.companions.len()).sum::<usize>(),
                    held.iter().filter(|m| m.device_image.is_some()).count(),
                );
            }
        }
        if let Some((c, _)) = ctx {
            osprey_host_free(c);
        }
    }
}
