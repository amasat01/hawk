//! The C source of the unit meta compiles against the runtime's headers as C11 and as C++17, follows the headers' member order, and
//! builds a shared object whose only exported symbol is the meta function and whose data is read-only. The headers are found through
//! OSPREY_EAGLE_SRC, which names the include root that holds `eagle/door/eg_door.h` and `eagle/door/eg_unit_meta.h` (the tree's and the
//! installed form); without it each test prints that it skipped and passes.
mod common;
use osprey_lowering::lower::meta_c;
use osprey_lowering::lower::unit_meta::*;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The include root that holds `eagle/door/eg_door.h` and `eagle/door/eg_unit_meta.h`, when the environment names it.
fn door_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("OSPREY_EAGLE_SRC").map(PathBuf::from);
    if dir.is_none() {
        common::skipped(Some("OSPREY_EAGLE_SRC"), "OSPREY_EAGLE_SRC is not set");
    }
    dir
}

/// A folder of this test's own under the system's temporary folder.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("osprey_meta_c_{name}"));
    std::fs::create_dir_all(&dir).expect("a scratch folder");
    dir
}

/// Runs a program and returns its output, failing the test with the program's own words when it does not succeed.
fn run(program: &str, args: &[&str]) -> String {
    let out = Command::new(program)
        .args(args)
        .output()
        .expect("the program starts");
    assert!(
        out.status.success(),
        "{program} {args:?}:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Compiles a source for syntax only as C11 and as C++17, with warnings as errors.
fn compile_both(door: &Path, source: &Path) {
    let (door, source) = (door.to_str().unwrap(), source.to_str().unwrap());
    let flags = ["-Wall", "-Wextra", "-pedantic", "-Werror", "-fsyntax-only"];
    let c = [&["-std=c11", "-I", door][..], &flags, &[source]].concat();
    run("cc", &c);
    let cpp = [
        &["-std=c++17", "-x", "c++", "-I", door][..],
        &flags,
        &[source],
    ]
    .concat();
    run("c++", &cpp);
}

/// The text without its comments.
fn uncommented(text: &str) -> String {
    let (mut out, mut rest) = (String::new(), text);
    while let Some(at) = rest.find("/*").into_iter().chain(rest.find("//")).min() {
        out += &rest[..at];
        let end = if rest[at..].starts_with("/*") {
            "*/"
        } else {
            "\n"
        };
        rest = &rest[at..];
        rest = &rest[rest.find(end).map_or(rest.len(), |e| e + end.len())..];
    }
    out + rest
}

/// The members of a struct of the headers, in order.
fn header_members(headers: &str, name: &str) -> Vec<String> {
    let start = headers
        .find(&format!("struct {name} {{"))
        .unwrap_or_else(|| panic!("struct {name}"));
    let body = &headers[start..];
    let body = &body[body.find('{').unwrap() + 1..body.find("};").unwrap()];
    let mut members = vec![];
    for declaration in body.split(';').filter(|d| !d.trim().is_empty()) {
        for part in declaration.split(',') {
            let part = part.trim().split('[').next().unwrap().trim();
            members.push(part[part.rfind([' ', '*']).map_or(0, |at| at + 1)..].to_string());
        }
    }
    members
}

/// The structs the source writes and the members of each, in the order written.
fn written_members(source: &str) -> Vec<(String, Vec<String>)> {
    source
        .lines()
        .filter_map(|line| {
            let ty = line
                .strip_prefix("static const struct ")?
                .split(' ')
                .next()?
                .to_string();
            line.contains(" = {M(").then_some(())?;
            let members = line
                .split("M(")
                .skip(1)
                .map(|rest| rest.split(',').next().unwrap().to_string())
                .collect();
            Some((ty, members))
        })
        .collect()
}

/// Every record of the tree written at least once, every optional member present.
fn full_tree() -> (LibraryMeta, UnitMeta) {
    let slot = |name: &str, role| Slot {
        flags: SLOT_FLAGS.iter().map(|(bit, _)| bit).sum(),
        role,
        dtype: WireDtype::Int32,
        like_slot: Some(0),
        gather: Some(vec![true, false]),
        shape: vec![3, 2],
        name: name.into(),
        mirror: "GRefMirror".into(),
        written: true,
    };
    let keyed = Keyed {
        width: 6,
        axis_labels: vec![Some("key".into()), None],
        axis_extents: vec![-1, 4],
        payload: vec![3, 2],
        name: "kp".into(),
    };
    let table = StepsTable {
        regime: StepsRegime::PerResident,
        widths: vec![1, 8],
        ks: vec![4, 16],
        lo: vec![0.5, 2.0],
        hi: vec![8.0, 64.0],
    };
    let digest = "ab".repeat(32);
    let reference = UnitRef {
        name: "stage".into(),
        digest: digest.clone(),
    };
    let body = Body {
        access: Access::CrossSampleWrite,
        cell_row: 32,
        scatter_chunks: 8,
        slots: vec![slot("a", SlotRole::Mutable), slot("b", SlotRole::Lookup)],
        companions: vec![Companion {
            slot: 1,
            target_slot: 0,
            host_fold_entry: Some("fold_host".into()),
            device_fold_function: Some("fold_dev".into()),
        }],
        counts: vec![],
        winners: vec![],
        cells: vec![Cell {
            op: ReduceOp::Sum,
            internal: true,
            columns: 2,
            slots: Some(vec![0, 1]),
            comp_slots: Some(vec![2, 3]),
            position: false,
            name: "c".into(),
        }],
        keyed: vec![keyed],
        team_planes: vec![TeamPlane {
            dtype: WireDtype::BandPartial,
            unit: PlaneUnit::Target,
            layout: PlaneLayout::PlaneMajor,
            wide: WideLayout::WarpTiled,
            keyed: Some(0),
            target_slot: Some(1),
            per_lane: 5,
            fill: -1,
            name: "tp".into(),
            target_stage: Some(1),
            fill_promised: true,
        }],
        dispatches: vec![
            Dispatch {
                strategy: DispatchStrategy::Segmented,
                mix_slot: Some(0),
                class_slot: Some(1),
                stale_slot: None,
                offsets_slot: Some(1),
                classify: Some(reference.clone()),
                name: "g".into(),
                k: 2,
                branches: vec![reference.clone(), reference.clone()],
                sweep: None,
                branch_costs: vec![],
            },
            Dispatch {
                strategy: DispatchStrategy::Segmented,
                mix_slot: Some(0),
                class_slot: Some(1),
                stale_slot: Some(0),
                offsets_slot: Some(1),
                classify: None,
                name: "h".into(),
                k: 2,
                branches: vec![reference.clone(), reference.clone()],
                sweep: Some(reference.clone()),
                branch_costs: vec![],
            },
            Dispatch {
                strategy: DispatchStrategy::Auto,
                mix_slot: Some(0),
                name: "m".into(),
                k: 2,
                branch_costs: vec![
                    BranchCost {
                        arith32: 1.5,
                        stores: Some(2.0),
                        ..BranchCost::default()
                    },
                    BranchCost::default(),
                ],
                ..Dispatch::default()
            },
            // an automatic group with the segmented set built beside it: the branch units and the classify unit, and its costs
            Dispatch {
                strategy: DispatchStrategy::Auto,
                mix_slot: Some(0),
                classify: Some(reference.clone()),
                name: "n".into(),
                k: 2,
                branches: vec![reference.clone(), reference.clone()],
                branch_costs: vec![BranchCost::default(), BranchCost::default()],
                ..Dispatch::default()
            },
        ],
    };
    let team = Team {
        grid: TeamGrid::Chunks,
        threads: 32,
        warps: 2,
        lanes: 4,
        groups: 3,
        keyed: Some(0),
        of_slot: Some(1),
        status_plane: Some(0),
        of_stage: Some(1),
    };
    let unit = UnitMeta {
        flags: 0,
        name: "u".into(),
        digest,
        pattern: Pattern::Pure,
        scalar_type: ScalarType::Banded,
        host_entry: Some("u_host".into()),
        device_image: Some("u__image".into()),
        device_function: Some("u".into()),
        device_image_bytes: 0,
        device_image_format: ArtifactForm("Cubin".into()),
        stages: vec![Stage {
            unit: reference,
            cells_in: vec!["c".into()],
            cells_out: vec!["d".into(), "e".into()],
            team: Some(team),
        }],
        body,
        launch: Launch {
            block: 128,
            active_map_slot: Some(0),
            active_count_slot: Some(1),
            index_map_slot: Some(0),
            index_count_slot: Some(1),
            group: 4,
            lane_labels: vec![Some("key".into()), None],
            lane_extents: vec![-1, 5],
            plane_axes: vec![PlaneAxes {
                slot: 0,
                labels: vec![Some("k".into())],
                extents: vec![-1],
                extent_slots: vec![-1],
            }],
            host_box: Some(ThreadBox {
                axis: 0,
                t: 4,
                stride: 2,
            }),
            device_box: Some(ThreadBox {
                axis: 1,
                t: 2,
                stride: 1,
            }),
        },
        finish: Some(Finish {
            flags: 1,
            mask_slot: 0,
            counter_slot: 1,
            steps_slot: Some(1),
            steps: -1,
            steps_max: Some(64),
            step_ops: Some(41),
            fp64_ops_ir: None,
            host_steps_reverse: Some(table.clone()),
            device_steps_reverse: Some(table),
            n_events: 2,
            device_persist_function: Some("u_persist".into()),
            device_range_function: Some("u_range".into()),
        }),
        device_image_size: Some("u__image_size".into()),
        device_compiler: Some("nvcc 12.6".into()),
        device_flags: Some("-fatbin -std=c++20 -O3".into()),
        device_ptx_isa: 85,
        gemm: Some(Gemm {
            lane_fanin: 6,
            calls: vec![GemmCall {
                flags: GEMM_TRANS_A | GEMM_ACCUMULATE,
                a_slot: 1,
                b_slot: 0,
                c_slot: 2,
                a_rows: 8,
                a_cols: GEMM_PER_LANE,
                b_rows: 8,
                b_cols: 6,
                c_rows: GEMM_PER_LANE,
                c_cols: 6,
            }],
        }),
        group: Some(Group {
            membership: GroupMembership::Axes,
            phases: 2,
            key_slot: None,
            count: None,
            max_size: Some(5),
            scratch_bytes: 120,
            axes: vec![1],
            scratch: vec![GroupScratch {
                dtype: WireDtype::Real,
                shape: vec![3],
                elements: 15,
                bytes: 120,
                name: "t".into(),
                realisation: ScratchRealisation::Plane,
                slot: Some(3),
            }],
            distinct_slots: vec![0, 2],
        }),
        derived: Some(Derived {
            role: DerivedRole::StepVjp,
            of_unit: 0,
            taped: vec![Taped {
                slot: 0,
                row0: 0,
                rows: 1,
            }],
            links: vec![Link {
                slot: 1,
                what: LinkWhat::PrimalSlot,
                of_slot: Some(0),
                reduced: true,
                form: LinkForm::Grad,
            }],
            tape_rows: 1,
            ..Derived::default()
        }),
    };
    (LibraryMeta::current(), unit)
}

#[test]
fn the_records_cross_the_wire_unchanged() {
    let (_, unit) = full_tree();
    let json = aether_writers::contract::Contract::parse(&unit.put().to_text())
        .unwrap()
        .0;
    assert_eq!(UnitMeta::get(Some(&json), "meta").unwrap(), unit);
}

#[test]
fn every_closed_set_and_name_is_the_headers() {
    let Some(door) = door_dir() else { return };
    let headers = std::fs::read_to_string(door.join("eagle/door/eg_door.h")).unwrap()
        + &std::fs::read_to_string(door.join("eagle/door/eg_unit_meta.h")).unwrap();
    let defines = |name: &str| -> String {
        let at = headers
            .find(&format!("#define {name} "))
            .unwrap_or_else(|| panic!("{name}"));
        let line = headers[at..].lines().next().unwrap();
        let value = line
            .split("/**")
            .next()
            .unwrap()
            .trim_start_matches("#define ")
            .trim_start_matches(name);
        value
            .split_whitespace()
            .collect::<String>()
            .trim_end_matches('u')
            .to_string()
    };
    macro_rules! sets {
        ($($set:ident),*) => {$(for member in $set::ALL {
            assert_eq!(defines(member.c_name()), (*member as i32).to_string(), "{}", member.c_name());
        })*};
    }
    sets!(
        Pattern,
        ScalarType,
        SlotRole,
        WireDtype,
        Access,
        TeamGrid,
        PlaneUnit,
        PlaneLayout,
        WideLayout,
        GroupMembership,
        ScratchRealisation,
        DispatchStrategy,
        StepsRegime,
        ReduceOp,
        DerivedRole,
        LinkWhat,
        LinkForm
    );
    // the forms of an artifact: the base's word for none and the device library's three, numbered as the header numbers them
    for (value, form) in ["None", "Ptx", "Cubin", "Fatbin"].iter().enumerate() {
        assert_eq!(
            defines(&ArtifactForm(form.to_string()).c_name()),
            value.to_string(),
            "{form}"
        );
    }
    for (bit, name) in SLOT_FLAGS.iter().chain(&FINISH_FLAGS).chain(&BRANCH_FLAGS) {
        assert_eq!(
            defines(name),
            format!("(1u<<{})", bit.trailing_zeros()).trim_end_matches('u'),
            "{name}"
        );
    }
    assert_eq!(defines("EG_META_SCHEMA_MAJOR"), SCHEMA_MAJOR.to_string());
    assert_eq!(
        defines("EG_UNIT_LIBRARY_META_SYMBOL"),
        format!("\"{LIBRARY_META_SYMBOL}\"")
    );
}

#[test]
fn a_tree_with_every_record_follows_the_headers_and_compiles() {
    let Some(door) = door_dir() else { return };
    let (library, unit) = full_tree();
    let source = meta_c::source(&library, &[&unit]).unwrap();
    let headers = uncommented(
        &(std::fs::read_to_string(door.join("eagle/door/eg_door.h")).unwrap()
            + &std::fs::read_to_string(door.join("eagle/door/eg_unit_meta.h")).unwrap()),
    );
    let written = written_members(&source);
    let mut seen: Vec<&str> = written.iter().map(|(ty, _)| ty.as_str()).collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(
        seen.len(),
        25,
        "every record of the headers is written: {seen:?}"
    );
    for (ty, members) in &written {
        assert_eq!(
            members,
            &header_members(&headers, ty),
            "the members of {ty}"
        );
    }
    let file = scratch("full").join("full.c");
    std::fs::write(&file, source).unwrap();
    compile_both(&door, &file);
}

/// A C program that includes the meta source and prints what each dispatch group of the unit states of its costs.
const READER: &str = r#"#include <stdio.h>
#include "SOURCE"
int main(void) {
    const struct eg_unit_body* body = eg_unit_library_meta()->units[0]->body;
    for (int32_t g = 0; g < body->n_dispatches; ++g) {
        const struct eg_unit_dispatch* d = body->dispatches[g];
        printf("group %s k %d costs %s cost_flags %u", d->name, d->k,
               d->branch_costs ? "yes" : "no", d->cost_flags);
        printf(" branches %s classify %s sweep %s\n", d->branches ? "yes" : "no",
               d->classify ? d->classify->name : "none", d->sweep ? "yes" : "no");
        for (int32_t j = 0; d->branch_costs && j < d->k; ++j) {
            const struct eg_branch_cost* c = d->branch_costs[j];
            printf("branch %s %d flags %u %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g stores %.17g\n",
                   d->name, j, c->flags, c->arith32, c->arith64, c->special32, c->special64, c->loads, c->gathers,
                   c->chain32, c->chain64, c->chain_special32, c->chain_special64, c->stores);
        }
    }
    const struct eg_unit_meta* unit = eg_unit_library_meta()->units[0];
    const struct eg_unit_gemm* gemm = unit->gemm;
    printf("gemm %d fanin %lld host_entry %s\n", gemm ? gemm->n_calls : -1, gemm ? (long long)gemm->lane_fanin : -1,
           unit->host_entry ? "kept" : "dropped");
    if (gemm) {
        const struct eg_unit_gemm_call* c = gemm->calls[0];
        printf("call flags %u slots %d %d %d shape %lld %lld %lld %lld %lld %lld\n", c->flags, c->a_slot, c->b_slot,
               c->c_slot, (long long)c->a_rows, (long long)c->a_cols, (long long)c->b_rows, (long long)c->b_cols,
               (long long)c->c_rows, (long long)c->c_cols);
    }
    const struct eg_unit_group* group = unit->group;
    printf("group %d phases %d key %d count %lld max %lld bytes %lld axes %d first %d scratch %d\n", group ? group->membership : -1,
           group ? group->phases : -1, group ? group->key_slot : -1, group ? (long long)group->count : -1,
           group ? (long long)group->max_size : -1, group ? (long long)group->scratch_bytes : -1, group ? group->n_axes : -1,
           group ? group->axes[0] : -1, group ? group->n_scratch : -1);
    if (group) {
        const struct eg_unit_group_scratch* t = group->scratch[0];
        printf("scratch %s dtype %d rank %d shape %lld elements %lld bytes %lld\n", t->name, t->dtype, t->rank, (long long)t->shape[0],
               (long long)t->elements, (long long)t->bytes);
        printf("distinct %d first %d last %d\n", group->n_distinct, group->distinct_slots[0], group->distinct_slots[group->n_distinct - 1]);
    }
    return 0;
}
"#;

#[test]
fn a_c_reader_of_the_compiled_record_sees_the_branch_costs_and_the_products_that_were_written() {
    let Some(door) = door_dir() else { return };
    let (library, unit) = full_tree();
    let dir = scratch("reader");
    let source = dir.join("meta.c");
    std::fs::write(&source, meta_c::source(&library, &[&unit]).unwrap()).unwrap();
    let program = dir.join("reader.c");
    std::fs::write(&program, READER.replace("SOURCE", source.to_str().unwrap())).unwrap();
    let binary = dir.join("reader");
    run(
        "cc",
        &[
            "-std=c11",
            "-I",
            door.to_str().unwrap(),
            program.to_str().unwrap(),
            "-o",
            binary.to_str().unwrap(),
        ],
    );
    let read = run(binary.to_str().unwrap(), &[]);
    // oracle: the values `full_tree` wrote; the group `m` is the auto one, `g` and `h` are segmented and state no costs
    let want = [
        "group g k 2 costs no cost_flags 0 branches yes classify stage sweep no\n",
        "group h k 2 costs no cost_flags 0 branches yes classify none sweep yes\n",
        "group m k 2 costs yes cost_flags 0 branches no classify none sweep no\n",
        "branch m 0 flags 2 1.5 0 0 0 0 0 0 0 0 0 stores 2\n",
        "branch m 1 flags 0 0 0 0 0 0 0 0 0 0 0 stores 0\n",
        // an automatic group that names the segmented set built beside it keeps its costs and has no sweep
        "group n k 2 costs yes cost_flags 0 branches yes classify stage sweep no\n",
        "branch n 0 flags 0 0 0 0 0 0 0 0 0 0 0 stores 0\n",
        "branch n 1 flags 0 0 0 0 0 0 0 0 0 0 0 stores 0\n",
        // a unit with the member keeps its kernel: a host plan runs it, and a device plan the door cannot route does
        "gemm 1 fanin 6 host_entry kept\n",
        "call flags 9 slots 1 0 2 shape 8 -1 8 6 -1 6\n",
        // the group record: AXES over lane axis 1, bound 5, one scratch of 3-value rows (15 scalars, 120 bytes)
        "group 1 phases 2 key -1 count 0 max 5 bytes 120 axes 1 first 1 scratch 1\n",
        "scratch t dtype 0 rank 1 shape 3 elements 15 bytes 120\n",
        // the planes the unit assumes distinct: two slots, 0 and 2
        "distinct 2 first 0 last 2\n",
    ];
    assert_eq!(read, want.concat());
}

/// The exported symbols of a library and the sections of the data it holds.
fn inspect(library: &Path) {
    let lib = library.to_str().unwrap();
    if cfg!(target_os = "macos") {
        let exported = run("nm", &["-gU", lib]);
        let names: Vec<&str> = exported
            .lines()
            .filter_map(|l| l.split_whitespace().last())
            .collect();
        assert_eq!(
            names,
            [format!("_{LIBRARY_META_SYMBOL}")],
            "exported: {exported}"
        );
        let all = run("nm", &["-m", lib]);
        for line in all.lines().filter(|l| l.contains("non-external")) {
            let ok = [
                "(__TEXT,__const)",
                "(__DATA_CONST,__const)",
                "(__TEXT,__cstring)",
            ];
            assert!(ok.iter().any(|s| line.contains(s)), "writable data: {line}");
        }
    } else {
        println!("meta_c: the section check runs on macOS only (unverified here)");
        let exported = run("nm", &["-g", "--defined-only", lib]);
        assert_eq!(exported.lines().count(), 1, "exported: {exported}");
        assert!(exported.contains(LIBRARY_META_SYMBOL), "{exported}");
    }
}

#[test]
fn a_shared_object_exports_one_symbol_and_holds_read_only_data() {
    let Some(door) = door_dir() else { return };
    let (library, unit) = full_tree();
    let dir = scratch("shared");
    let file = dir.join("meta.c");
    std::fs::write(&file, meta_c::source(&library, &[&unit]).unwrap()).unwrap();
    let out = dir.join("meta.so");
    let args = [
        "-std=c11",
        "-O2",
        "-shared",
        "-fPIC",
        "-I",
        door.to_str().unwrap(),
        file.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
    ];
    run("cc", &args);
    inspect(&out);
}

/// The kernels the lowered-kernel tests use: recorded calls of different shapes.
#[cfg(feature = "old-contract")]
const CALLS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/meta_calls.jsonl");

#[cfg(feature = "old-contract")]
#[test]
fn lowered_kernels_of_every_shape_write_sources_that_compile() {
    let Some(door) = door_dir() else { return };
    let dir = scratch("kernels");
    let out = Command::new(env!("CARGO_BIN_EXE_kreplay"))
        .args(["--emit-meta", CALLS, dir.to_str().unwrap()])
        .output()
        .expect("kreplay starts");
    let lines = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "{lines}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let held = |key: &str| {
        lines
            .lines()
            .any(|l| l.contains(&format!(" {key} ")) && !l.contains(&format!(" {key} 0 ")))
    };
    for shape in [
        "stages",
        "finish",
        "host_box",
        "device_box",
        "team",
        "image",
    ] {
        assert!(held(shape), "no lowered kernel with {shape}:\n{lines}");
    }
    assert!(
        lines.lines().all(|l| l.contains(" companions 0 ")),
        "a companion is not computed yet:\n{lines}"
    );
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    files.retain(|f| f.extension().is_some_and(|e| e == "c"));
    files.sort();
    assert!(files.len() >= 7, "{files:?}");
    for file in &files {
        compile_both(&door, file);
    }
    let staged = std::fs::read_to_string(
        files
            .iter()
            .find(|f| f.to_string_lossy().contains("ev_untimed"))
            .unwrap(),
    )
    .unwrap();
    assert!(
        staged.contains("M(struct_size, sizeof(struct eg_unit_finish))")
            && staged.contains("M(steps, 1)"),
        "the finish record states the steps the request declared"
    );
    let library = dir.join("kernel.so");
    let first = files[0].to_str().unwrap();
    run(
        "cc",
        &[
            "-std=c11",
            "-O2",
            "-shared",
            "-fPIC",
            "-I",
            door.to_str().unwrap(),
            first,
            "-o",
            library.to_str().unwrap(),
        ],
    );
    inspect(&library);
    for (member, knows) in NOT_COMPUTED {
        println!("not computed yet: {member} -> {knows}");
    }
}
