//! Witnesses: a small program the OSPREY supplies as data, built and run (or only built) with the tools of the machine to learn a
//! fact about them that no version number states, such as whether a compiler release miscompiles a known pattern. The ask gives each
//! witness's source, its compile and link argv templates and a map from the exit code of its run to a verdict word; the answer is
//! the verdict, kept in the store under the tools' identity and the witness's digest so a witness runs once per toolchain.
use crate::compile;
use crate::discover::Machine;
use crate::refusal::{Done, Kind, Refusal};
use crate::reserved::scratch_dir;
use crate::run;
use crate::store::{Built, Store};
use crate::toolkit::Toolkit;
use crate::wire;
use rcore::json::J;
use rcore::sha256;
use std::path::Path;

/// Where an `{include:<tree>}` place of a probe points.
pub(crate) enum Includes<'a> {
    /// The served folder of the embedded tree of that name; needs the store root.
    Served(Option<&'a Path>),
    /// One empty folder for every include place: a witness reads no header.
    Empty(&'a Path),
}

/// The files and includes a probe's places stand for.
pub(crate) struct ProbePlaces<'a> {
    /// The source file.
    pub source: &'a Path,
    /// The object file, when the probe makes one.
    pub object: Option<&'a Path>,
    /// The output file (the program a link makes), when there is one.
    pub output: Option<&'a Path>,
    /// What the include places mean.
    pub includes: Includes<'a>,
}

impl Toolkit {
    /// The text a place of a probe's argv template stands for, `None` when nobody fills it.
    pub(crate) fn probe_place(
        &self,
        machine: &Machine,
        place: &str,
        places: &ProbePlaces,
    ) -> Done<Option<String>> {
        let shown = |p: Option<&Path>| p.map(|p| p.display().to_string());
        if let Some(name) = place.strip_prefix("include:") {
            return match &places.includes {
                Includes::Empty(dir) => Ok(shown(Some(dir))),
                Includes::Served(root) => {
                    let Some(tree) = self.trees.iter().find(|t| t.name == name) else {
                        return Ok(None);
                    };
                    let root = root.ok_or_else(|| {
                        Refusal::bad(
                            "the ask needs a root: a probe includes an embedded header tree",
                        )
                    })?;
                    Ok(shown(Some(&tree.serve(root)?)))
                }
            };
        }
        Ok(match place {
            "source" => shown(Some(places.source)),
            "object" => shown(places.object),
            "output" => shown(places.output),
            _ => machine.tool(place).map(|t| t.path.display().to_string()),
        })
    }

    /// The verdict of every witness of the ask, by name: `{verdict, why?, hit}`.
    pub(crate) fn witnesses(&self, machine: &Machine, list: &J, root: Option<&Path>) -> Done<J> {
        let mut out = vec![];
        for witness in list.arr() {
            let name = wire::text(witness, "name", "witness")?;
            out.push((name.to_string(), self.witness(machine, witness, root)?));
        }
        Ok(J::Obj(out))
    }

    /// One witness. A witness that could not be built or run answers `unrun` and is not kept: its cause may pass.
    fn witness(&self, machine: &Machine, witness: &J, root: Option<&Path>) -> Done<J> {
        let source = wire::text(witness, "source", "witness")?;
        let mut terms = vec![
            "toolkit.witness/1".to_string(),
            source.to_string(),
            self.processor_identity(),
        ];
        terms.extend(machine.tools.iter().map(|(_, t)| t.identity.clone()));
        for key in ["compile", "link"] {
            terms.extend(wire::texts(witness, key, "witness")?);
        }
        for (code, verdict) in wire::need(witness, "exit_verdicts", "witness")?.obj() {
            terms.push(format!("{code}={}", verdict.str().unwrap_or("")));
        }
        let key = sha256::hex(terms.join("\0").as_bytes());
        let measure = |_: &Path| {
            self.measure(machine, witness).map(|verdict| Built {
                files: vec![],
                meta: wire::s(&verdict),
            })
        };
        let entry = match root {
            Some(root) => Store::new(root).get_or_build(&key, measure),
            None => measure(Path::new("")).map(|b| crate::store::Entry {
                dir: Path::new("").into(),
                files: vec![],
                meta: b.meta,
                hit: false,
            }),
        };
        Ok(match entry {
            Ok(entry) => wire::obj(vec![("verdict", entry.meta), ("hit", J::Bool(entry.hit))]),
            Err(why) if matches!(why.kind, Kind::ToolFailed | Kind::TimedOut) => wire::obj(vec![
                ("verdict", wire::s("unrun")),
                ("why", wire::s(&why.message)),
                ("hit", J::Bool(false)),
            ]),
            Err(other) => return Err(other),
        })
    }

    /// Builds and runs the witness; a `ToolFailed` refusal says why it could not.
    fn measure(&self, machine: &Machine, witness: &J) -> Done<String> {
        let dir = scratch_dir()?;
        let outcome = self.measure_in(machine, witness, &dir);
        let _ = std::fs::remove_dir_all(&dir);
        outcome
    }

    /// `measure` in the folder `dir`.
    fn measure_in(&self, machine: &Machine, witness: &J, dir: &Path) -> Done<String> {
        let source = dir.join(format!(
            "witness{}",
            wire::text_or(witness, "source_suffix", ".cpp")
        ));
        std::fs::write(&source, wire::text(witness, "source", "witness")?)
            .map_err(|e| Refusal::io("cannot write", &source, &e))?;
        let (object, exe) = (dir.join("witness.o"), dir.join("witness.bin"));
        let places = ProbePlaces {
            source: &source,
            object: Some(&object),
            output: Some(&exe),
            includes: Includes::Empty(dir),
        };
        let seconds = u64::try_from(wire::need(witness, "timeout", "witness")?.int()).unwrap_or(0);
        let env = self.child_env(machine)?;
        for step in ["compile", "link"] {
            let mut argv = vec![];
            for element in wire::texts(witness, step, "witness")? {
                if element == "{defines}" {
                    continue;
                }
                argv.push(compile::fill(&element, &mut |place| {
                    self.probe_place(machine, place, &places)
                })?);
            }
            let built = run::run(&argv, &env, 0, seconds)?;
            if built.code != Some(0) {
                return Err(Refusal::new(
                    Kind::ToolFailed,
                    format!(
                        "the witness does not {step} with these flags\n{}",
                        built.stderr
                    ),
                ));
            }
        }
        let ran = run::run(&[exe.display().to_string()], &env, 0, seconds)?;
        let code = ran
            .code
            .map_or_else(|| "signal".to_string(), |c| c.to_string());
        match wire::need(witness, "exit_verdicts", "witness")?
            .get(&code)
            .and_then(J::str)
        {
            Some(verdict) => Ok(verdict.to_string()),
            None => Err(Refusal::new(
                Kind::ToolFailed,
                format!("the witness exited {code}, which no verdict names"),
            )),
        }
    }
}
