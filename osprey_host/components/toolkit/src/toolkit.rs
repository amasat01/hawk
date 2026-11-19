//! The handle an OSPREY holds: its tool table, its embedded header trees and the environment it was started in; and the two
//! door functions that need no compile, `machine` and `state`.
use crate::device;
use crate::discover::{self, Env, Machine};
use crate::headers::HeaderTree;
use crate::refusal::{Done, Kind, Refusal};
use crate::runtime::Runtime;
use crate::store;
use crate::wire;
use crate::{host, run};
use rcore::json::J;
use rcore::sha256;
use std::collections::HashMap;
use std::path::Path;

/// The tool side of one OSPREY.
#[derive(Debug)]
pub struct Toolkit {
    /// The OSPREY's table: `tools` (see [`discover`]), and optionally `process_env` and `limits`.
    pub(crate) spec: J,
    /// The header trees the OSPREY embeds.
    pub(crate) trees: Vec<HeaderTree>,
    /// The environment the discovery and the tools run in.
    pub(crate) env: Env,
}

impl Toolkit {
    /// A toolkit from the OSPREY's table (JSON text), its embedded trees and the environment to read.
    ///
    /// # Errors
    /// Refuses with a [`Refusal`](crate::refusal::Refusal) that names what is wrong with the request or the tool.
    pub fn new(spec: &str, trees: &[HeaderTree], env: Env) -> Done<Self> {
        let spec = wire::parse("the tool table", spec)?;
        wire::need(&spec, "tools", "the tool table")?;
        Ok(Toolkit {
            spec,
            trees: trees.to_vec(),
            env,
        })
    }

    /// The machine the tools run on, always stated: `{"key": <opaque>, "stated": {"tools", "headers", ...}, "tools_absent": <reason> |
    /// null}`. A missing tool (with no runtime compile library standing in for it) is not a different answer: the tools found, the
    /// header digests, the host facts and the arch policy are stated, and `tools_absent` names every tool tried and how each rule
    /// failed. The reserved names and the witness verdicts need a tool and are left out while one is missing. `ask` is
    /// `{"key_terms": [text...], "host": bool, "root": <store root>?, "device_arch": <pin>?, "reserved_names": {"prelude": <include
    /// block>}?, "witnesses": [<witness>...]?}`: the texts the key must cover beyond what the machine states, whether to state the
    /// host's own facts, and the facts to determine: the device arch (when the table has a `device` member), the reserved names (when
    /// the table has `reserved_names`) and the verdict of each witness. `root` is where served header trees and kept verdicts live;
    /// the facts that need it refuse without it. The key covers the whole stated record and the texts.
    ///
    /// # Errors
    /// Refuses a malformed ask or table, a pinned arch the toolkit cannot serve, and a probe whose tool fails.
    pub fn machine(&self, ask: &str) -> Done<String> {
        let ask = wire::parse("the machine ask", ask)?;
        let found = discover::find(
            &self.spec,
            &self.env.fresh(),
            ask.get("root").and_then(J::str).map(Path::new),
        )?;
        let (runtime, absent) = self.runtime_for(&found)?;
        let tools = J::Obj(
            found
                .tools
                .iter()
                .map(|(n, t)| (n.clone(), discover::stated(t)))
                .collect(),
        );
        let headers = J::Obj(
            self.trees
                .iter()
                .map(|t| (t.name.to_string(), wire::s(&t.digest())))
                .collect(),
        );
        let mut stated = vec![("tools", tools), ("headers", headers)];
        if matches!(ask.get("host"), Some(J::Bool(true))) {
            stated.push(("host", host::facts()));
        }
        if let Some(runtime) = &runtime {
            stated.push(("runtime", runtime_stated(runtime)?));
        }
        stated.extend(self.determined(&ask, &found, runtime.as_ref(), absent.is_some())?);
        let stated = wire::obj(stated);
        let terms = wire::texts(&ask, "key_terms", "the machine ask")?;
        let key = sha256::hex(
            format!(
                "toolkit.machine/3\0{}\0{}",
                wire::render(&stated),
                terms.join("\0")
            )
            .as_bytes(),
        );
        let reason = absent.map_or(J::Null, |r| wire::s(&r));
        Ok(wire::render(&wire::obj(vec![
            ("key", wire::s(&key)),
            ("stated", stated),
            ("tools_absent", reason),
        ])))
    }

    /// The runtime compile library when a tool was replaced by it, and the reason for the tools still missing: the machine's own,
    /// and the library's when a tool was replaced and no library loads.
    fn runtime_for(&self, machine: &Machine) -> Done<(Option<Runtime>, Option<String>)> {
        let Some((name, why)) = machine.replaced.first() else {
            return Ok((None, machine.absent.clone()));
        };
        let spec = wire::need(&self.spec, "runtime_compile", "the tool table")?;
        if let Some(runtime) = Runtime::open(spec)? {
            return Ok((Some(runtime), machine.absent.clone()));
        }
        let libraries = wire::texts(spec, "libraries", "runtime_compile")?.join(", ");
        let own =
            format!("{why}; and no runtime compile library loads for {name}: tried {libraries}");
        Ok((
            None,
            Some(
                machine
                    .absent
                    .iter()
                    .cloned()
                    .chain([own])
                    .collect::<Vec<_>>()
                    .join("; "),
            ),
        ))
    }

    /// The facts of an ask that need more than a lookup: the device arch, the reserved names and the witness verdicts.
    fn determined(
        &self,
        ask: &J,
        machine: &Machine,
        runtime: Option<&Runtime>,
        tool_missing: bool,
    ) -> Done<Vec<(&'static str, J)>> {
        let root = ask.get("root").and_then(J::str).map(Path::new);
        let mut out = vec![];
        if self.spec.get("device").is_some() {
            out.push(("device", self.device_stated(ask, machine, runtime)?));
        }
        if let (Some(reserved), false) = (ask.get("reserved_names"), tool_missing) {
            let names =
                self.reserved_names(machine, wire::text_or(reserved, "prelude", ""), root)?;
            out.push(("reserved_names", crate::reserved::stated(&names)));
        }
        if let (Some(list), false) = (ask.get("witnesses"), tool_missing) {
            out.push(("witnesses", self.witnesses(machine, list, root)?));
        }
        Ok(out)
    }

    /// The device record: the arch chosen and why, the GPU and driver the driver library stated, and the toolkit's archs.
    fn device_stated(&self, ask: &J, machine: &Machine, runtime: Option<&Runtime>) -> Done<J> {
        let device = wire::need(&self.spec, "device", "the tool table")?;
        let driver = match device.get("driver") {
            Some(spec) => device::probe_driver(spec)?.unwrap_or_default(),
            None => device::Driver::default(),
        };
        let env = self.env.fresh();
        let policy = wire::need(device, "arch", "device")?;
        let toolkit = match runtime {
            Some(runtime) => runtime.archs(),
            None => tool_archs(policy, machine),
        };
        let env_pin = policy
            .get("pin_env")
            .and_then(J::str)
            .and_then(|var| env.get(var).map(|v| (var, v)));
        let inputs = device::Inputs {
            asked: ask.get("device_arch").and_then(J::str),
            env_pin,
            gpu: driver.gpu_arch.as_deref(),
            toolkit: toolkit.as_deref(),
        };
        let (arch, why) = device::choose_arch(policy, &inputs)?;
        let nullable = |v: Option<String>| v.map_or(J::Null, |t| wire::s(&t));
        Ok(wire::obj(vec![
            ("arch", wire::s(&arch)),
            ("why", wire::s(&why)),
            ("gpu_arch", nullable(driver.gpu_arch)),
            ("driver_release", nullable(driver.release)),
            (
                "driver_encoded",
                driver
                    .encoded
                    .map_or(J::Null, |e| wire::n(u64::try_from(e).unwrap_or(0))),
            ),
            (
                "toolkit_archs",
                toolkit.map_or(J::Null, |t| J::Arr(t.into_iter().map(wire::n).collect())),
            ),
        ]))
    }

    /// The counters of this process (`op` `"stats"`: `{"hits", "misses", "tool_runs"}`) or their reset (`op` `"reset"`, which
    /// also forgets the memoised tool probes; the slots on disk stay). `root` is the store's root, unused by both ops today and
    /// part of the door's shape for ops that read it.
    ///
    /// # Errors
    /// Refuses with a [`Refusal`](crate::refusal::Refusal) that names what is wrong with the request or the tool.
    pub fn state(&self, op: &str, _root: &Path) -> Done<String> {
        match op {
            "stats" => {
                let (hits, misses, runs) = store::stats();
                Ok(wire::render(&wire::obj(vec![
                    ("hits", wire::n(hits)),
                    ("misses", wire::n(misses)),
                    ("tool_runs", wire::n(runs)),
                ])))
            }
            "reset" => {
                store::reset_stats();
                discover::forget_probes();
                Ok("{}".to_string())
            }
            other => Err(Refusal::bad(format!(
                "state op {other:?} is not stats or reset"
            ))),
        }
    }

    /// The tools of the table, found, or the refusal that names what is missing.
    pub(crate) fn tools(&self, root: Option<&Path>) -> Done<Machine> {
        let machine = discover::find(&self.spec, &self.env.fresh(), root)?;
        match &machine.absent {
            Some(reason) => Err(Refusal::new(
                crate::refusal::Kind::ToolAbsent,
                reason.clone(),
            )),
            None => Ok(machine),
        }
    }

    /// The environment a tool runs under.
    pub(crate) fn child_env(&self, machine: &Machine) -> Done<HashMap<String, String>> {
        discover::child_env(&self.spec, &self.env.fresh(), machine)
    }

    /// What the environment says for the ask `name` (`opt_level`, `host_profile`) when the caller states none: the value of the
    /// variable the table's `asks` names, trimmed; `None` when the table names none, the variable is unset or it is empty.
    #[must_use]
    pub fn ask_from_env(&self, name: &str) -> Option<String> {
        let var = self
            .spec
            .get("asks")
            .and_then(|asks| asks.get(name))
            .and_then(|rule| rule.get("env"))
            .and_then(J::str)?;
        let value = self.env.fresh().get(var)?.trim().to_string();
        (!value.is_empty()).then_some(value)
    }

    /// The limit `name` of the table: its environment variable's value when set, else its default; 0 disables it.
    pub(crate) fn limit(&self, name: &str, default: u64) -> Done<u64> {
        let Some(rule) = self.spec.get("limits").and_then(|l| l.get(name)) else {
            return Ok(default);
        };
        let default = u64::try_from(
            rule.get("default")
                .map_or(i64::try_from(default).unwrap_or(0), J::int),
        )
        .unwrap_or(default);
        let Some(var) = rule.get("env").and_then(J::str) else {
            return Ok(default);
        };
        let env = self.env.fresh();
        match env.get(var).map(str::trim) {
            None => Ok(default),
            Some(raw) => raw.parse::<u64>().map_err(|_| {
                Refusal::bad(format!(
                    "${var}={raw:?} is not a non-negative integer (0 disables it)"
                ))
            }),
        }
    }

    /// The words that tell a reader which variable changes the limit `name`, as `" ($VAR, 0 disables it)"`; empty when the table
    /// gives that limit no variable.
    pub(crate) fn limit_note(&self, name: &str) -> String {
        self.spec
            .get("limits")
            .and_then(|l| l.get(name))
            .and_then(|rule| rule.get("env"))
            .and_then(J::str)
            .map_or(String::new(), |var| format!(" (${var}, 0 disables it)"))
    }

    /// Runs a command of this toolkit under its limits, once it holds one of the `jobs` permits of the store `root` (the request's
    /// own number, else the table's `jobs` limit, default 2; 0 is no cap).
    pub(crate) fn run(
        &self,
        argv: &[String],
        machine: &Machine,
        root: &Path,
        jobs: Option<u64>,
    ) -> Done<run::Output> {
        let cap = self.limit("address_cap", crate::compile::ADDRESS_CAP_BYTES)?;
        let seconds = self.limit("timeout", crate::compile::TIMEOUT_SECONDS)?;
        let jobs = match jobs {
            Some(n) => n,
            None => self.limit("jobs", crate::jobs::DEFAULT_JOBS)?,
        };
        let _permit = crate::jobs::acquire(root, jobs)?;
        store::note_run();
        let env = self.child_env(machine)?;
        run::run(argv, &env, cap, seconds).map_err(|mut refused| {
            if refused.kind == Kind::TimedOut {
                refused.message.push_str(&self.limit_note("timeout"));
            }
            refused
        })
    }
}

/// The archs a tool states by the probe the table's `device.arch.toolkit` record names (`tool`, `probe`); `None` when it does not.
fn tool_archs(policy: &J, machine: &Machine) -> Option<Vec<u64>> {
    let source = policy.get("toolkit")?;
    let tool = machine.tool(source.get("tool")?.str()?)?;
    let probe = source.get("probe")?.str()?;
    let list = tool
        .facts
        .iter()
        .find(|(name, _)| name == probe)?
        .1
        .arr()
        .iter()
        .map(|a| u64::try_from(a.int()).unwrap_or(0))
        .collect();
    Some(list)
}

/// The stated record of a runtime compile library: its name, version and archs.
fn runtime_stated(runtime: &Runtime) -> Done<J> {
    let (major, minor) = runtime.version()?;
    Ok(wire::obj(vec![
        ("library", wire::s(&runtime.identity()?)),
        ("version", wire::s(&format!("{major}.{minor}"))),
        (
            "archs",
            runtime
                .archs()
                .map_or(J::Null, |a| J::Arr(a.into_iter().map(wire::n).collect())),
        ),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discover::Env;

    /// The host library's shipped tool table with the environment `pairs`.
    fn host_toolkit(pairs: &[(&str, &str)]) -> Toolkit {
        let table = include_str!("../../../osprey_cpu/facts/tools.json");
        Toolkit::new(table, &[], Env::from_pairs(pairs)).unwrap()
    }

    /// Oracle: the shipped table's `jobs` limit states a default of 2 and the variable `HAWK_COMPILE_JOBS`.
    #[test]
    fn the_job_count_is_two_unless_the_variable_says_another() {
        assert_eq!(host_toolkit(&[]).limit("jobs", 99).unwrap(), 2);
        let three = host_toolkit(&[("HAWK_COMPILE_JOBS", "3")]);
        assert_eq!(three.limit("jobs", 99).unwrap(), 3);
        // the table states that 0 disables a limit: no cap on the compiles that run at once
        let none = host_toolkit(&[("HAWK_COMPILE_JOBS", "0")]);
        assert_eq!(none.limit("jobs", 99).unwrap(), 0);
    }

    #[test]
    fn a_job_count_that_is_not_a_number_is_refused_naming_the_variable() {
        for raw in ["two", "1.5", "-1"] {
            let toolkit = host_toolkit(&[("HAWK_COMPILE_JOBS", raw)]);
            let why = toolkit.limit("jobs", 2).unwrap_err();
            assert!(why.to_string().contains("HAWK_COMPILE_JOBS"), "{why}");
        }
    }
}
