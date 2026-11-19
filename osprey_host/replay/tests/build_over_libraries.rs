//! A build over the host and the device library: the device library leads and states a record, the host library lowers to it, and
//! `merge` joins the two answers. A library's texts do not change with the other's presence, a companion refuses a cut it cannot
//! meet, the answer states the steps table and the dispatch costs of its target, and a reserved kernel name is refused.
#![cfg(feature = "old-contract")]
mod fakes;
use aether_writers::contract::types::Target;
use fakes::{script, store};
use osprey_lowering::facts::{probe, Stated};
use osprey_lowering::lower::request::{KernelAnswer, KernelRequest};
use osprey_lowering::lower::unit_meta::ImageFormat;
use osprey_lowering::tool;
use raptor_ir::door::{rx_ctx_free, rx_ctx_new, RX_OK};
use raptor_ir::dump::rx_load;
use raptor_ir::record::Ctx;
use rcore::json::{self, J};
use std::path::Path;

/// The device compiler identity the device facts are asked with.
const DEVICE_COMPILER: &str =
    "/usr/local/cuda/bin/nvcc || Cuda compilation tools, release 12.6, V12.6.77 || 0";
/// The host compiler identity the host facts are asked with.
const HOST_COMPILER: &str = "/usr/bin/c++ || Apple clang version 21.0.0 || 0";

/// The kernel `name` of the working-set oracle in a fresh context, and the handle of its first sink.
fn kernel_named(name: &str) -> (*mut Ctx, u32) {
    let oracle = json::parse(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../raptor/rs/wings/tests/data/working_set_oracle.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let case = oracle
        .get("cases")
        .unwrap()
        .arr()
        .iter()
        .find(|case| case.get("name").and_then(J::str) == Some(name))
        .unwrap();
    let text = case.get("dump").and_then(J::str).unwrap();
    let ctx = rx_ctx_new();
    // SAFETY: `ctx` is the live context just made.
    assert_eq!(
        unsafe { raptor_wings::declare::rw_declare_kinds(ctx) },
        RX_OK
    );
    let (mut handles, mut n) = (vec![0u32; 4096], 0usize);
    // SAFETY: `ctx` is live; the text and the buffer outlive the call.
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
    assert_eq!(status, RX_OK);
    let sink = handles[case.get("sinks").unwrap().arr()[0].int() as usize];
    (ctx, sink)
}

/// A context holding the kernel `plain` of the working-set oracle, and the handle of its one sink.
fn plain_kernel() -> (*mut Ctx, u32) {
    kernel_named("plain")
}

/// The request of the kernel for one target, with the facts that target's library states.
fn request_for(target: Target, sink: u32) -> KernelRequest {
    let base = format!(
        r#"{{"v": 1, "name": "k", "scalar": "Float64", "targets": ["Host"], "kind": {{"guard": "Default", "active_set": false, "sink": "Plain"}}, "body": {{"Plain": {{"name": "k", "sinks": [{sink}], "declared": [], "exempt": []}}}}, "facts": {{"Host": "{{}}"}}}}"#
    );
    let mut request = KernelRequest::from_json(&base).unwrap();
    let dir = osprey_replay::merged_facts();
    let (name, compiler, stated) = match target {
        Target::Host => (
            "host",
            HOST_COMPILER,
            Stated {
                headers: Some("roots".into()),
                ..Stated::default()
            },
        ),
        Target::Device => ("cuda", DEVICE_COMPILER, Stated::default()),
    };
    let facts = osprey_replay::answer_named(&dir, name, compiler, &probe(), &stated).unwrap();
    request.targets = vec![target];
    request.facts = vec![(target, facts)];
    request.image_format = (target == Target::Device).then_some(ImageFormat::Cubin);
    request
}

/// What the library of the request's one target answers to the request, from the data files of that library's folder of the
/// checkout: its answer text, or the status and the error text as the door gives them.
fn lowered(ctx: *mut Ctx, request: &KernelRequest) -> Result<String, String> {
    let target = request.targets[0];
    let facts_dir = Path::new(osprey_replay::facts_of(target));
    let text = request.to_json();
    let done = match target {
        Target::Host => osprey_lowering::lower::lower(&osprey_cpu_lib::Cpu, ctx, &text, facts_dir),
        Target::Device => {
            osprey_lowering::lower::lower(&osprey_cuda_lib::Cuda, ctx, &text, facts_dir)
        }
    };
    match done {
        Ok(answer) => Ok(answer.to_json()),
        Err(refusal) => Err(format!(
            "{}|{}",
            osprey_lowering::door::status_of(refusal.kind),
            refusal.reason
        )),
    }
}

/// The device answer of a build whose companion is the host, and the host answer lowered to its record.
fn both(ctx: *mut Ctx, sink: u32) -> (String, String) {
    let mut lead = request_for(Target::Device, sink);
    lead.companions = vec![Target::Host];
    let lead_text = lowered(ctx, &lead).unwrap();
    let record = KernelAnswer::from_json(&lead_text).unwrap().peer.unwrap();
    let mut beside = request_for(Target::Host, sink);
    beside.peer = Some((Target::Device, record));
    (lead_text, lowered(ctx, &beside).unwrap())
}

/// The texts of every unit of an answer, with their targets.
fn texts_of(answer: &str) -> Vec<(Target, String)> {
    let parsed = KernelAnswer::from_json(answer).unwrap();
    let units = parsed.units.iter();
    units
        .flat_map(|u| u.texts.iter().map(|t| (t.target, t.text.clone())))
        .collect()
}

#[test]
fn a_library_states_the_same_texts_whether_or_not_the_other_is_in_the_build() {
    let (ctx, sink) = plain_kernel();
    let (lead, beside) = both(ctx, sink);
    let alone_device = lowered(ctx, &request_for(Target::Device, sink)).unwrap();
    let alone_host = lowered(ctx, &request_for(Target::Host, sink)).unwrap();
    assert_eq!(texts_of(&lead), texts_of(&alone_device));
    assert_eq!(texts_of(&beside), texts_of(&alone_host));
    // SAFETY: the context is not used again.
    unsafe { rx_ctx_free(ctx) };
}

#[test]
fn the_merge_of_the_two_answers_holds_both_targets_in_the_order_host_then_device() {
    let (ctx, sink) = plain_kernel();
    let (lead, beside) = both(ctx, sink);
    // the answers are given in the other order than the merge's
    let merged = osprey_lowering::lower::merge(&format!("[{lead},{beside}]")).unwrap();
    let answer = KernelAnswer::from_json(&merged).unwrap();
    let targets: Vec<Target> = answer.units[0].texts.iter().map(|t| t.target).collect();
    assert_eq!(targets, [Target::Host, Target::Device]);
    let meta = &answer.units[0].meta;
    assert!(meta.host_entry.is_some() && meta.device_function.is_some());
    assert!(answer.steps_reverse.host.is_some() && answer.steps_reverse.device.is_some());
    assert!(answer.peer.is_none() && !answer.meta_source.is_empty());
    let twice = osprey_lowering::lower::merge(&format!("[{lead},{lead}]")).unwrap_err();
    assert!(twice.reason.contains("same target"), "{}", twice.reason);
    // SAFETY: the context is not used again.
    unsafe { rx_ctx_free(ctx) };
}

#[test]
fn a_companion_refuses_a_cut_it_cannot_meet_by_naming_both_cuts() {
    let (ctx, sink) = plain_kernel();
    let mut lead = request_for(Target::Device, sink);
    lead.companions = vec![Target::Host];
    let mut record = KernelAnswer::from_json(&lowered(ctx, &lead).unwrap())
        .unwrap()
        .peer
        .unwrap();
    record.units = vec!["k".into(), "k_2".into()];
    let mut beside = request_for(Target::Host, sink);
    beside.peer = Some((Target::Device, record));
    let why = lowered(ctx, &beside).unwrap_err();
    assert!(why.contains("the cut") && why.contains("k_2"), "{why}");
    // SAFETY: the context is not used again.
    unsafe { rx_ctx_free(ctx) };
}

/// Oracle: the card of a host states no launch cost, so the merged answer of a host-only build states the record with every
/// member unstated, and a build that has a device states the device's own.
#[test]
fn the_merged_answer_states_the_dispatch_costs_of_a_host_only_build_with_every_member_unstated() {
    let (ctx, sink) = plain_kernel();
    let host = lowered(ctx, &request_for(Target::Host, sink)).unwrap();
    let merged = osprey_lowering::lower::merge(&format!("[{host}]")).unwrap();
    let costs = KernelAnswer::from_json(&merged)
        .unwrap()
        .dispatch_costs
        .expect("the merged answer states the record");
    assert_eq!(
        (
            costs.launch_us,
            costs.plane_us_per_byte,
            costs.local_over_plane
        ),
        (None, None, None)
    );
    // SAFETY: the context is not used again.
    unsafe { rx_ctx_free(ctx) };
}

/// Oracle: the shipped device documents state the launch cost 2.7 microseconds and the steps rows of width 2, and the host's own
/// documents state the host rows.
#[test]
fn each_answer_states_the_steps_table_of_its_target_and_the_device_the_dispatch_costs() {
    let (ctx, sink) = plain_kernel();
    let device =
        KernelAnswer::from_json(&lowered(ctx, &request_for(Target::Device, sink)).unwrap())
            .unwrap();
    let host =
        KernelAnswer::from_json(&lowered(ctx, &request_for(Target::Host, sink)).unwrap()).unwrap();
    let costs = device.dispatch_costs.unwrap();
    assert_eq!(
        (costs.launch_us, costs.plane_us_per_byte),
        (Some(2.7), Some(7.1e-06))
    );
    assert_eq!(device.steps_reverse.device.unwrap().widths, [2]);
    assert!(device.steps_reverse.host.is_none() && host.steps_reverse.device.is_none());
    assert!(host.steps_reverse.host.is_some() && host.dispatch_costs.is_none());
    // SAFETY: the context is not used again.
    unsafe { rx_ctx_free(ctx) };
}

#[test]
fn a_kernel_named_after_a_reserved_name_is_refused_by_name() {
    let (ctx, sink) = plain_kernel();
    let mut request = request_for(Target::Host, sink);
    request.reserved_names = vec!["k".into(), "other".into()];
    let why = lowered(ctx, &request).unwrap_err();
    assert!(why.contains("\"k\"") && why.contains("reserved"), "{why}");
    request.reserved_names = vec!["other".into()];
    lowered(ctx, &request).unwrap();
    // SAFETY: the context is not used again.
    unsafe { rx_ctx_free(ctx) };
}

#[test]
fn a_setting_the_target_does_not_declare_is_refused_by_name_and_the_target_that_declares_it_takes_it(
) {
    let (ctx, sink) = plain_kernel();
    let mut host = request_for(Target::Host, sink);
    host.settings.fold_k = Some(2);
    let why = lowered(ctx, &host).unwrap_err();
    assert!(
        why.contains("\"fold_k\"") && why.contains("declared by none"),
        "{why}"
    );
    let mut device = request_for(Target::Device, sink);
    device.settings.fold_k = Some(2);
    lowered(ctx, &device).unwrap();
    // SAFETY: the context is not used again.
    unsafe { rx_ctx_free(ctx) };
}

#[test]
fn the_fast_entry_setting_is_refused_by_the_host_card_by_name_and_taken_by_the_device_card() {
    // the host library has no fast entry points to build, so its card does not declare the setting and a host-only build is told so
    let (ctx, sink) = plain_kernel();
    let mut host = request_for(Target::Host, sink);
    host.settings.fast_entries = true;
    let why = lowered(ctx, &host).unwrap_err();
    assert!(
        why.contains("\"fast_entries\"") && why.contains("declared by none"),
        "{why}"
    );
    let mut device = request_for(Target::Device, sink);
    device.settings.fast_entries = true;
    lowered(ctx, &device).unwrap();
    // SAFETY: the context is not used again.
    unsafe { rx_ctx_free(ctx) };
}

#[test]
fn a_companion_takes_a_setting_the_leader_declares_and_refuses_one_that_neither_declares() {
    let (ctx, sink) = plain_kernel();
    let mut lead = request_for(Target::Device, sink);
    lead.companions = vec![Target::Host];
    lead.settings.fold_k = Some(2);
    let record = KernelAnswer::from_json(&lowered(ctx, &lead).unwrap())
        .unwrap()
        .peer
        .unwrap();
    assert!(record.settings.contains(&"fold_k".to_string()));
    // the host card does not name fold_k; the leader's record does
    let mut beside = request_for(Target::Host, sink);
    beside.settings.fold_k = Some(2);
    beside.peer = Some((Target::Device, record.clone()));
    lowered(ctx, &beside).unwrap();
    // a record that does not list it leaves the companion to its own card
    let mut silent = record;
    silent.settings.retain(|key| key != "fold_k");
    beside.peer = Some((Target::Device, silent));
    let why = lowered(ctx, &beside).unwrap_err();
    assert!(
        why.contains("\"fold_k\"") && why.contains("declared by none"),
        "{why}"
    );
    // SAFETY: the context is not used again.
    unsafe { rx_ctx_free(ctx) };
}

/// The device answer `lead` with the define the leader states for a text that gates its fast entry points behind a macro (the plain
/// kernel's text has no gate, so its answer states none of its own).
fn with_device_define(lead: &str) -> String {
    let mut answer = KernelAnswer::from_json(lead).unwrap();
    answer.units[0]
        .defines
        .push((Target::Device, "HAWK_FAST_ENTRIES=1".into()));
    answer.to_json()
}

/// Oracle: the define the answer of the device library states, as the writer wrote it; a merge over answers that state it keeps it
/// for the device only and one over answers that state none has no `defines` member.
#[test]
fn a_two_library_merge_keeps_the_defines_of_the_device_and_states_none_for_the_host() {
    let (ctx, sink) = plain_kernel();
    let (lead, beside) = both(ctx, sink);
    let defines = |answer: &str| {
        KernelAnswer::from_json(answer).unwrap().units[0]
            .defines
            .clone()
    };
    let stating = with_device_define(&lead);
    assert!(
        stating.contains("\"defines\":{\"cuda\":[\"HAWK_FAST_ENTRIES=1\"]}"),
        "{stating}"
    );
    let merged = osprey_lowering::lower::merge(&format!("[{stating},{beside}]")).unwrap();
    assert_eq!(
        defines(&merged),
        [(Target::Device, "HAWK_FAST_ENTRIES=1".to_string())]
    );
    let plain = osprey_lowering::lower::merge(&format!("[{lead},{beside}]")).unwrap();
    assert!(
        !plain.contains("\"defines\""),
        "an answer with no define has no member"
    );
    // SAFETY: the context is not used again.
    unsafe { rx_ctx_free(ctx) };
}

/// Oracle: each library's own answer. Every text with its facts (the written planes, their elements and shapes), the steps table, the
/// dispatch costs and the per-target members of the unit's meta in the merged answer are those the library of the target stated.
#[test]
fn a_merged_pair_states_each_member_of_a_target_as_the_library_of_that_target_did() {
    let (ctx, sink) = plain_kernel();
    let (lead, beside) = both(ctx, sink);
    let merged = KernelAnswer::from_json(
        &osprey_lowering::lower::merge(&format!("[{lead},{beside}]")).unwrap(),
    )
    .unwrap();
    let (device, host) = (
        KernelAnswer::from_json(&lead).unwrap(),
        KernelAnswer::from_json(&beside).unwrap(),
    );
    let of = |answer: &KernelAnswer, target: Target| {
        let texts = answer.units.iter().flat_map(|u| &u.texts);
        texts
            .filter(|t| t.target == target)
            .map(|t| format!("{t:?}"))
            .collect::<Vec<_>>()
    };
    for (alone, target) in [(&host, Target::Host), (&device, Target::Device)] {
        assert!(!of(alone, target).is_empty());
        assert_eq!(
            of(&merged, target),
            of(alone, target),
            "{target:?} texts and facts"
        );
    }
    assert_eq!(merged.dispatch_costs, device.dispatch_costs);
    assert_eq!(merged.steps_reverse.host, host.steps_reverse.host);
    assert_eq!(merged.steps_reverse.device, device.steps_reverse.device);
    let (m, d, h) = (
        &merged.units[0].meta,
        &device.units[0].meta,
        &host.units[0].meta,
    );
    assert_eq!(
        (
            &m.device_function,
            &m.device_image_format,
            &m.device_flags,
            &m.launch.device_box
        ),
        (
            &d.device_function,
            &d.device_image_format,
            &d.device_flags,
            &d.launch.device_box
        )
    );
    assert_eq!(
        (&m.host_entry, &m.launch.host_box),
        (&h.host_entry, &h.launch.host_box)
    );
    // SAFETY: the context is not used again.
    unsafe { rx_ctx_free(ctx) };
}

/// Oracle: the argv the fake device compiler recorded (a script that appends its arguments to a file). The device build request is
/// filled the way a caller fills it from the merged answer: the unit's device text and the defines the answer states for the
/// device. The plain kernel's text has no fast-entry gate and no kernel of the working-set oracle gates its entry points, so the
/// define is the injected member of `with_device_define`; the step proved is the one from the merged answer to the compile.
#[test]
fn the_device_build_request_filled_from_the_merged_answer_carries_the_define_in_the_compiler_argv()
{
    let (ctx, sink) = plain_kernel();
    let (lead, beside) = both(ctx, sink);
    let merged =
        osprey_lowering::lower::merge(&format!("[{},{beside}]", with_device_define(&lead)))
            .unwrap();
    let answer = KernelAnswer::from_json(&merged).unwrap();
    let defines: Vec<String> = answer.units[0]
        .defines
        .iter()
        .filter(|(target, _)| *target == Target::Device)
        .map(|(_, define)| format!("{define:?}"))
        .collect();
    assert_eq!(defines, ["\"HAWK_FAST_ENTRIES=1\""]);
    let text = texts_of(&merged)
        .into_iter()
        .find(|(target, _)| *target == Target::Device)
        .unwrap()
        .1;
    let root = store("merged_define_argv");
    let tools = root.join("tools");
    let argv = tools.join("argv");
    script(
        &tools.join("nvcc"),
        &format!(
            "#!/bin/sh\ncase \"$1\" in --version) echo 'Cuda compilation tools, release 12.6, V12.6.77'; exit 0;; \
             --list-gpu-arch) printf 'compute_61\\ncompute_75\\n'; exit 0;; esac\necho \"$@\" >> {argv}\n\
             out=\"\"; prev=\"\"; for a in \"$@\"; do [ \"$prev\" = \"-o\" ] && out=\"$a\"; prev=\"$a\"; done\n\
             printf '\\177ELFimage' > \"$out\"\n",
            argv = argv.display()
        ),
    );
    script(&tools.join("host/g++"), "#!/bin/sh\necho 'g++ fake'\n");
    // the library reads the process environment once, when its tool side is first used
    std::env::set_var("HAWK_NVCC", tools.join("nvcc"));
    std::env::set_var("HAWK_NVCC_CCBIN", tools.join("host"));
    let ask = r#"{"target": "cuda", "options": {"arch": "sm_61"}, "opt_level": ""}"#;
    let machine = tool::machine(&osprey_cuda_lib::Cuda, ask).unwrap();
    let request = format!(
        r#"{{"name": "k", "text": {text:?}, "machine": {machine}, "defines": [{}]}}"#,
        defines.join(", ")
    );
    tool::build(&osprey_cuda_lib::Cuda, &request, &root).unwrap();
    let recorded = std::fs::read_to_string(&argv).unwrap();
    assert!(
        recorded.contains("-DHAWK_FAST_ENTRIES=1"),
        "the compiler ran with: {recorded}"
    );
    // SAFETY: the context is not used again.
    unsafe { rx_ctx_free(ctx) };
}
