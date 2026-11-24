//! The handle an OSPREY holds: its tool table, its embedded header trees and the environment it was started in; and the two
//! door functions that need no compile, `machine` and `state`.
use crate::discover::{self, Env, Machine};
use crate::headers::HeaderTree;
use crate::refusal::{Done, Kind, Refusal};
use crate::registry::{names, Grammar, Probe, ProbeAnswer, ProbeInput};
use crate::run;
use crate::runtime::Runtime;
use crate::store;
use crate::wire;
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
    /// Why the library this toolkit belongs to cannot run a target on any machine, when its build left the runtime out.
    pub(crate) runtime_absent: Option<String>,
    /// The probes the backend registered, by name.
    probes: Vec<(&'static str, Probe)>,
    /// The grammars the backend registered, by name.
    grammars: Vec<(&'static str, Grammar)>,
    /// The digest naming the machine's processor that a witness key carries, when the backend registered one.
    identity: Option<fn() -> String>,
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
            runtime_absent: None,
            probes: vec![],
            grammars: vec![],
            identity: None,
        })
    }

    /// Registers `probe` under `name`: `machine` asks it when the table has a member `name` or the ask has a true member `name`, and
    /// states its record under `name`.
    #[must_use]
    pub fn with_probe(mut self, name: &'static str, probe: Probe) -> Self {
        self.probes.push((name, probe));
        self
    }

    /// Registers the function that digests the machine's processor; a witness key carries the digest, so one processor's verdict is
    /// never served to another.
    #[must_use]
    pub fn with_identity(mut self, identity: fn() -> String) -> Self {
        self.identity = Some(identity);
        self
    }

    /// The processor digest a witness key carries: empty when the backend registered none.
    pub(crate) fn processor_identity(&self) -> String {
        self.identity.map_or_else(String::new, |digest| digest())
    }

    /// Registers `grammar` under `name`, for the tool probes whose `as` names it and the jobs whose `figures` names it.
    #[must_use]
    pub fn with_grammar(mut self, name: &'static str, grammar: Grammar) -> Self {
        self.grammars.push((name, grammar));
        self
    }

    /// The figures a job states under the grammar `name`, read from the tool's `text` output or from the `image` it wrote; no name
    /// states none.
    ///
    /// # Errors
    /// Refuses a name no grammar was registered under.
    pub(crate) fn figures(&self, name: Option<&str>, text: &str, image: &[u8]) -> Done<J> {
        let Some(name) = name else {
            return Ok(wire::obj(vec![]));
        };
        match self.grammars.iter().find(|(known, _)| *known == name) {
            Some((_, grammar)) => Ok(grammar(text, image)),
            None => Err(Refusal::bad(format!(
                "figures grammar {name:?} is not one of: {}",
                names(&self.grammars)
            ))),
        }
    }

    /// States that the library was built without the runtime a target runs on: `machine` then answers `runs_here` false whatever
    /// the machine offers, and begins `why` with `reason`.
    #[must_use]
    pub fn without_runtime(mut self, reason: &str) -> Self {
        self.runtime_absent = Some(reason.to_string());
        self
    }

    /// The machine the tools run on, always stated: `{"key": <opaque>, "stated": {"tools", "headers", ...}, "tools_absent": <reason> |
    /// null}`. A missing tool (with no runtime compile library standing in for it) is not a different answer: the tools found, the
    /// header digests and the registered probes' records are stated, and `tools_absent` names every tool tried and how each rule
    /// failed. The reserved names and the witness verdicts need a tool and are left out while one is missing. `ask` is
    /// `{"key_terms": [text...], "root": <store root>?, "reserved_names": {"prelude": <include block>}?, "witnesses":
    /// [<witness>...]?}` and any member a registered probe reads: the texts the key must cover beyond what the machine states, and
    /// the facts to determine: the reserved names (when the table has `reserved_names`) and the verdict of each witness. A probe
    /// states its record when the table has a member of its name or the ask has a true member of its name. `root` is where served
    /// header trees and kept verdicts live; the facts that need it refuse without it. The key covers the whole stated record and
    /// the texts.
    ///
    /// # Errors
    /// Refuses a malformed ask or table and a probe that refuses.
    pub fn machine(&self, ask: &str) -> Done<String> {
        let ask = wire::parse("the machine ask", ask)?;
        let env = self.env.fresh();
        let found = discover::find(
            &self.spec,
            &env,
            ask.get("root").and_then(J::str).map(Path::new),
            &self.grammars,
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
        let input = |section| ProbeInput {
            section,
            ask: &ask,
            env: &env,
            machine: &found,
            runtime: runtime.as_ref(),
        };
        // the probes the ask calls for state before the runtime record, those of a table member after it
        let mut blocked = vec![];
        for (name, probe) in &self.probes {
            if self.spec.get(name).is_none() && matches!(ask.get(name), Some(J::Bool(true))) {
                let ProbeAnswer {
                    facts,
                    blocked: why,
                } = probe(&input(None))?;
                stated.push((*name, facts));
                blocked.extend(why);
            }
        }
        if let Some(runtime) = &runtime {
            stated.push(("runtime", runtime_stated(runtime)?));
        }
        for (name, probe) in &self.probes {
            if let Some(section) = self.spec.get(name) {
                let ProbeAnswer {
                    facts,
                    blocked: why,
                } = probe(&input(Some(section)))?;
                stated.push((*name, facts));
                blocked.extend(why);
            }
        }
        stated.extend(self.determined(&ask, &found, absent.is_some())?);
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
        let (runs_here, why) = match blocked.first() {
            Some(why) => (false, why.clone()),
            None => (true, String::new()),
        };
        let (runs_here, why) = match &self.runtime_absent {
            Some(reason) if why.is_empty() => (false, reason.clone()),
            Some(reason) => (false, format!("{reason}; {why}")),
            None => (runs_here, why),
        };
        Ok(wire::render(&wire::obj(vec![
            ("key", wire::s(&key)),
            ("stated", stated),
            ("tools_absent", reason),
            ("runs_here", J::Bool(runs_here)),
            ("why", wire::s(&why)),
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

    /// The facts of an ask that need more than a lookup: the reserved names and the witness verdicts.
    fn determined(
        &self,
        ask: &J,
        machine: &Machine,
        tool_missing: bool,
    ) -> Done<Vec<(&'static str, J)>> {
        let root = ask.get("root").and_then(J::str).map(Path::new);
        let mut out = vec![];
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
        let machine = discover::find(&self.spec, &self.env.fresh(), root, &self.grammars)?;
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

    /// The name of the variable the table's `asks` gives for the ask `name`, when it names one.
    #[must_use]
    pub fn ask_env_name(&self, name: &str) -> Option<&str> {
        self.spec
            .get("asks")
            .and_then(|asks| asks.get(name))
            .and_then(|rule| rule.get("env"))
            .and_then(J::str)
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

    /// A tool table with a `jobs` limit, with the environment `pairs`: the fixture `tests/data/limits_table.json`.
    fn host_toolkit(pairs: &[(&str, &str)]) -> Toolkit {
        let table = include_str!("../tests/data/limits_table.json");
        Toolkit::new(table, &[], Env::from_pairs(pairs)).unwrap()
    }

    /// Oracle: the fixture table's `jobs` limit states a default of 2 and the variable `HAWK_COMPILE_JOBS`.
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
    fn a_grammar_nobody_registered_is_refused_naming_the_registered_ones() {
        fn nothing(_: &str, _: &[u8]) -> J {
            wire::obj(vec![])
        }
        let toolkit = Toolkit::new(r#"{"tools":{}}"#, &[], Env::from_pairs(&[]))
            .unwrap()
            .with_grammar("known", nothing);
        assert_eq!(wire::render(&toolkit.figures(None, "", &[]).unwrap()), "{}");
        assert!(toolkit.figures(Some("known"), "", &[]).is_ok());
        let why = toolkit.figures(Some("nope"), "", &[]).unwrap_err();
        assert!(
            why.message.contains("\"nope\"") && why.message.contains("known"),
            "{why}"
        );
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
