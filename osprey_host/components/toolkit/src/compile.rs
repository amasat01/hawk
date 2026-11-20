//! `build` and `link`: fill the argv templates a unit states, run the tool under the limits, publish the result in the store and
//! return the files, the opaque embed record of a device image, the figures the tool's output stated and a report.
//!
//! A template is a list of texts with places: `{<tool>}` (a tool of the table), `{source}`, `{output}` (also `{object}`),
//! `{depfile}`, `{include:<tree>}` (the served folder of an embedded header tree) and any place the request's `places` states. A
//! list place (`{defines}`, `{objects}`) is an element of its own and stands for one element per item. A request's `as` renames a
//! tool place (`{"compiler": "host_compiler"}` makes `{compiler}` the tool the table calls `host_compiler`). A place nobody fills
//! is refused by name. The key of a build covers the text, the template as given (not the paths it is filled with), the lists,
//! the extra flags, the request's own `key_terms`, the identity of every tool the template names and the digest of every tree it
//! includes: the same request on the same machine is the same slot.
use crate::discover::Machine;
use crate::figures;
use crate::refusal::{Done, Kind, Refusal};
use crate::store::{Built, Entry, Store};
use crate::toolkit::Toolkit;
use crate::wire;
use rcore::json::J;
use rcore::sha256;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The default ceiling on one tool's address space (12 GiB): a machine guard, so a pathological compile fails instead of taking
/// the machine down.
pub const ADDRESS_CAP_BYTES: u64 = 12 << 30;
/// The default largest source text a compile accepts (8 MiB): compile time grows with the source.
pub const SOURCE_BUDGET_BYTES: u64 = 8 << 20;
/// The default wall-clock limit of one tool run, in seconds.
pub const TIMEOUT_SECONDS: u64 = 1800;
/// How many characters of a tool's complaint a refusal carries.
const COMPLAINT_CHARS: usize = 4000;
/// What a tool's complaint says when it ran out of memory under the cap (a tool killed by a signal under a cap is the cap too).
const ALLOCATION_FAILURES: [&str; 7] = [
    "bad_alloc",
    "cannot allocate memory",
    "out of memory",
    "unable to allocate",
    "memory exhausted",
    "failed to map segment",
    "signal terminated program",
];

/// One tool run asked of the toolkit, from a request.
pub(crate) struct Job {
    /// The unit's name; it names the files.
    name: String,
    /// The source text.
    text: String,
    /// The suffix of the source file.
    source_suffix: String,
    /// The suffix of the output file.
    output_suffix: String,
    /// The argv template.
    template: Vec<String>,
    /// List places and their items.
    lists: Vec<(String, Vec<String>)>,
    /// Flags appended after the template.
    extra: Vec<String>,
    /// Texts the key covers beyond what the job states.
    key_terms: Vec<String>,
    /// The grammar of the figures to read from the tool's output.
    figures: Option<String>,
    /// Places the request fills with text.
    places: Vec<(String, String)>,
    /// Tool places renamed: place name and the tool that fills it.
    renames: Vec<(String, String)>,
    /// Whether the source text is published beside the output.
    keep_source: bool,
    /// The most tool processes at once this request allows under its store root; none: the table's limit.
    pub(crate) jobs: Option<u64>,
    /// The in-process compile the request offers for a machine with no offline compiler.
    runtime: Option<runtime_job::RuntimeJob>,
}

impl Job {
    /// The job a request states. `source_suffix` defaults to `.cpp` and `output_suffix` to `.o`.
    pub(crate) fn from_request(req: &J) -> Done<Job> {
        let pairs = |key: &str| -> Done<Vec<(String, String)>> {
            let mut out = vec![];
            for (k, v) in req.get(key).map_or(&[][..], J::obj) {
                let v = v.str().ok_or_else(|| {
                    Refusal::bad(format!("request: {key} member {k} is not text"))
                })?;
                out.push((k.clone(), v.to_string()));
            }
            Ok(out)
        };
        let mut lists = vec![];
        for (place, items) in req.get("lists").map_or(&[][..], J::obj) {
            lists.push((place.clone(), items.strs()));
        }
        Ok(Job {
            name: wire::text(req, "name", "request")?.to_string(),
            text: wire::text(req, "text", "request")?.to_string(),
            source_suffix: wire::text_or(req, "source_suffix", ".cpp").to_string(),
            output_suffix: wire::text_or(req, "output_suffix", ".o").to_string(),
            template: wire::texts(req, "template", "request")?,
            lists,
            extra: wire::texts(req, "extra", "request")?,
            key_terms: wire::texts(req, "key_terms", "request")?,
            figures: req.get("figures").and_then(J::str).map(str::to_string),
            places: pairs("places")?,
            renames: pairs("as")?,
            keep_source: matches!(req.get("keep_source"), Some(J::Bool(true))),
            jobs: req.get("jobs").map(|j| u64::try_from(j.int()).unwrap_or(0)),
            runtime: runtime_job::RuntimeJob::from_request(req)?,
        })
    }

    /// The job of a link: no source, the output is the library itself, and the keys of the parts are terms of the key.
    pub(crate) fn linking(
        req: &J,
        name: &str,
        template: &[String],
        objects: &[PathBuf],
        keys: &[String],
    ) -> Done<Job> {
        let mut job = Job::from_request(&wire::obj(vec![
            ("name", wire::s(name)),
            ("text", wire::s("")),
            ("output_suffix", wire::s("")),
            ("template", wire::strs(template)),
            (
                "key_terms",
                wire::strs(&wire::texts(req, "key_terms", "link request")?),
            ),
            (
                "places",
                req.get("places")
                    .cloned()
                    .unwrap_or_else(|| wire::obj(vec![])),
            ),
            (
                "as",
                req.get("as").cloned().unwrap_or_else(|| wire::obj(vec![])),
            ),
        ]))?;
        let paths: Vec<String> = objects.iter().map(|o| o.display().to_string()).collect();
        job.lists.push(("{objects}".to_string(), paths));
        job.jobs = req.get("jobs").map(|j| u64::try_from(j.int()).unwrap_or(0));
        job.key_terms.extend(keys.iter().cloned());
        Ok(job)
    }

    /// The job that assembles the text that embeds an image: the recipe's command over a `.s` source, keyed by the image's digest.
    pub(crate) fn assembling(
        req: &J,
        name: &str,
        text: &str,
        template: &[String],
        image_digest: &str,
    ) -> Done<Job> {
        Job::from_request(&wire::obj(vec![
            ("name", wire::s(name)),
            ("text", wire::s(text)),
            ("source_suffix", wire::s(".s")),
            ("template", wire::strs(template)),
            (
                "key_terms",
                wire::strs(&["embed".to_string(), format!("image:{image_digest}")]),
            ),
            (
                "as",
                req.get("as").cloned().unwrap_or_else(|| wire::obj(vec![])),
            ),
        ]))
    }

    /// The tool a tool place stands for, after the request's renames.
    fn tool_of<'a>(&'a self, place: &'a str) -> &'a str {
        self.renames
            .iter()
            .find(|(p, _)| p == place)
            .map_or(place, |(_, t)| t)
    }
}

/// The inner text of every place in `template`, in order of appearance.
fn places_in(template: &[String]) -> Vec<String> {
    let mut found = vec![];
    for element in template {
        let _ = fill(element, &mut |p| {
            found.push(p.to_string());
            Ok(Some(String::new()))
        });
    }
    found
}

/// `element` with each `{place}` replaced by what `value` gives; text that is not a place stays. `value` answers `None` for a
/// place nobody fills, which is refused by name.
pub(crate) fn fill(
    element: &str,
    value: &mut dyn FnMut(&str) -> Done<Option<String>>,
) -> Done<String> {
    let mut out = String::new();
    let mut rest = element;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let tail = &rest[open + 1..];
        let inner = tail
            .find('}')
            .map(|close| &tail[..close])
            .filter(|i| is_place(i));
        let Some(inner) = inner else {
            out.push('{');
            rest = tail;
            continue;
        };
        match value(inner)? {
            Some(text) => out.push_str(&text),
            None => {
                return Err(Refusal::bad(format!(
                "the build instructions name the place {{{inner}}}, which this build does not fill"
            )))
            }
        }
        rest = &tail[inner.len() + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Whether `inner` is written as a place: lower-case letters and underscores, with at most one `:` between two such words.
fn is_place(inner: &str) -> bool {
    let word = |w: &str| !w.is_empty() && w.bytes().all(|b| b.is_ascii_lowercase() || b == b'_');
    match inner.split_once(':') {
        Some((a, b)) => word(a) && word(b),
        None => word(inner),
    }
}

/// `parts` joined by NUL and digested.
fn digest(parts: &[&str]) -> String {
    sha256::hex(parts.join("\0").as_bytes())
}

/// What the report of a published tool run states.
pub(crate) fn report_of(meta: &J) -> J {
    wire::obj(vec![
        ("argv", meta.get("argv").cloned().unwrap_or(J::Null)),
        ("seconds", meta.get("seconds").cloned().unwrap_or(J::Null)),
        ("elf", meta.get("elf").cloned().unwrap_or(J::Null)),
    ])
}

impl Toolkit {
    /// Builds one unit: the request (JSON; see the module text) gives the unit; the answer is
    /// `{"files": {"output": path}, "key", "hit", "figures", "report": {"argv", "seconds", "elf"}}`; with `keep_source` the files
    /// also hold `"source"`. A second call for the same unit runs no tool.
    ///
    /// # Errors
    /// Refuses with a [`Refusal`](crate::refusal::Refusal) that names what is wrong with the request or the tool.
    pub fn build(&self, unit: &str, root: &Path) -> Done<String> {
        let req = wire::parse("the build request", unit)?;
        let job = Job::from_request(&req)?;
        let entry = self.run_job(&job, root)?;
        Ok(wire::render(&Self::answer(&job, &entry)?))
    }

    /// The answer of a build of `job` published as `entry`.
    fn answer(job: &Job, entry: &Entry) -> Done<J> {
        let output = entry.output()?;
        let mut files = vec![("output", wire::s(&output.display().to_string()))];
        if let Some(source) = entry.source().filter(|_| job.keep_source) {
            files.push(("source", wire::s(&source.display().to_string())));
        }
        let files = wire::obj(files);
        Ok(wire::obj(vec![
            ("files", files),
            ("key", wire::s(&key_of(entry))),
            ("hit", J::Bool(entry.hit)),
            (
                "figures",
                entry.meta.get("figures").cloned().unwrap_or(J::Null),
            ),
            ("report", report_of(&entry.meta)),
        ]))
    }

    /// The slot of `job`, built when no asker has built it.
    pub(crate) fn run_job(&self, job: &Job, root: &Path) -> Done<Entry> {
        let budget = self.limit("source_budget", SOURCE_BUDGET_BYTES)?;
        let size = job.text.len() as u64;
        if budget > 0 && size > budget {
            let note = self.limit_note("source_budget");
            return Err(Refusal::new(
                Kind::OverBudget,
                format!("the translation unit of {:?} is {size} bytes, over the compile budget of {budget} bytes{note}: a source this large is too wide for one compile; narrow the kernel or raise the budget for one build", job.name),
            ));
        }
        let machine = self.tools(Some(root))?;
        if let (Some(runtime), false) = (&job.runtime, machine.replaced.is_empty()) {
            return self.run_runtime_job(job, runtime, &machine, root);
        }
        if job.template.is_empty() {
            return Err(Refusal::bad(format!(
                "the request for {:?} has no argv template, and this machine's compiler is not replaced by the runtime compile: \
                 a request carries a `template` for the offline tool and a `runtime` member for the runtime compile, and the \
                 machine decides which one is used",
                job.name
            )));
        }
        for (name, why) in &machine.replaced {
            if places_in(&job.template)
                .iter()
                .any(|p| job.tool_of(p) == name)
            {
                return Err(Refusal::new(
                    Kind::ToolAbsent,
                    format!("{why}; the request offers no runtime compile for it"),
                ));
            }
        }
        let key = self.job_key(job, &machine);
        Store::new(root).get_or_build(&key, |scratch| {
            self.compile_into(job, &machine, root, scratch)
        })
    }

    /// The key of `job`: see the module text.
    pub(crate) fn job_key(&self, job: &Job, machine: &Machine) -> String {
        let mut parts: Vec<String> = vec![
            "toolkit.build/1".into(),
            job.text.clone(),
            job.source_suffix.clone(),
            job.output_suffix.clone(),
        ];
        parts.push(job.figures.clone().unwrap_or_default());
        parts.push(format!("keep_source {}", job.keep_source));
        parts.extend(job.template.iter().cloned());
        for (place, items) in &job.lists {
            parts.push(format!("list {place}"));
            parts.extend(items.iter().cloned());
        }
        parts.push("extra".into());
        parts.extend(job.extra.iter().cloned());
        parts.push("terms".into());
        parts.extend(job.key_terms.iter().cloned());
        for place in places_in(&job.template) {
            if let Some(name) = place.strip_prefix("include:") {
                if let Some(tree) = self.trees.iter().find(|t| t.name == name) {
                    parts.push(format!("tree {name} {}", tree.digest()));
                }
            } else if let Some(tool) = machine.tool(job.tool_of(&place)) {
                parts.push(format!("tool {place} {}", tool.identity));
            }
        }
        digest(&parts.iter().map(String::as_str).collect::<Vec<_>>())
    }

    /// The argv of `job` with every place filled; `output` and `scratch` are this run's own paths.
    pub(crate) fn argv_of(
        &self,
        job: &Job,
        machine: &Machine,
        root: &Path,
        scratch: &Path,
    ) -> Done<Vec<String>> {
        let shown = |p: PathBuf| Ok(Some(p.display().to_string()));
        let mut value = |place: &str| -> Done<Option<String>> {
            match place {
                "source" => shown(scratch.join(format!("{}{}", job.name, job.source_suffix))),
                "output" | "object" => {
                    shown(scratch.join(format!("{}{}", job.name, job.output_suffix)))
                }
                "depfile" => shown(scratch.join(format!("{}.d", job.name))),
                _ => {
                    if let Some(name) = place.strip_prefix("include:") {
                        return match self.trees.iter().find(|t| t.name == name) {
                            Some(tree) => shown(tree.serve(root)?),
                            None => Ok(job
                                .places
                                .iter()
                                .find(|(p, _)| p == place)
                                .map(|(_, v)| v.clone())),
                        };
                    }
                    if let Some((_, v)) = job.places.iter().find(|(p, _)| p == place) {
                        return Ok(Some(v.clone()));
                    }
                    Ok(machine
                        .tool(job.tool_of(place))
                        .map(|t| t.path.display().to_string()))
                }
            }
        };
        let mut argv = vec![];
        for element in &job.template {
            let listed = element
                .strip_prefix('{')
                .and_then(|e| e.strip_suffix('}'))
                .and_then(|inner| {
                    job.lists
                        .iter()
                        .find(|(p, _)| p.trim_matches(['{', '}']) == inner)
                });
            match listed {
                Some((_, items)) => argv.extend(items.iter().cloned()),
                None => argv.push(fill(element, &mut value)?),
            }
        }
        argv.extend(job.extra.iter().cloned());
        Ok(argv)
    }

    /// Writes the source, runs the tool and states the files and the record of the build.
    fn compile_into(
        &self,
        job: &Job,
        machine: &Machine,
        root: &Path,
        scratch: &Path,
    ) -> Done<Built> {
        let source = scratch.join(format!("{}{}", job.name, job.source_suffix));
        fs::write(&source, &job.text).map_err(|e| Refusal::io("cannot write", &source, &e))?;
        let argv = self.argv_of(job, machine, root, scratch)?;
        let start = Instant::now();
        let done = self.run(&argv, machine, root, job.jobs)?;
        let label = format!("compile of {:?}", job.name);
        if done.code != Some(0) {
            return Err(self.refuse(&argv, &done, &label, machine, scratch));
        }
        let output = format!("{}{}", job.name, job.output_suffix);
        let produced = scratch.join(&output);
        let head = fs::read(&produced).map_err(|_| {
            let log: String = done.stderr.chars().take(COMPLAINT_CHARS).collect();
            Refusal::new(
                Kind::ToolFailed,
                format!("{label} wrote no {output}\n$ {}\n{log}", argv.join(" ")),
            )
        })?;
        let meta = wire::obj(vec![
            ("argv", wire::strs(&argv)),
            ("seconds", J::Num(start.elapsed().as_secs_f64())),
            ("elf", J::Bool(head.starts_with(b"\x7fELF"))),
            (
                "figures",
                figures::parse(job.figures.as_deref(), &done.stderr, &head)?,
            ),
        ]);
        let mut files = vec![output];
        if job.keep_source {
            files.push(format!("{}{}", job.name, job.source_suffix));
        }
        Ok(Built { files, meta })
    }

    /// The way out the tool table states for the tool found at `program` (empty when the table states none).
    pub(crate) fn fix_of(&self, machine: &Machine, program: &str) -> &str {
        machine
            .tools
            .iter()
            .find(|(_, t)| t.path.display().to_string() == program)
            .and_then(|(name, _)| self.spec.get("tools")?.get(name)?.get("fix")?.str())
            .unwrap_or("")
    }

    /// The refusal of a tool that failed: out of memory under the cap, not runnable, or its complaint, then the directory it ran in
    /// and the way out of the tool.
    pub(crate) fn refuse(
        &self,
        argv: &[String],
        done: &crate::run::Output,
        label: &str,
        machine: &Machine,
        dir: &Path,
    ) -> Refusal {
        let cap = self.limit("address_cap", ADDRESS_CAP_BYTES).unwrap_or(0);
        let fix = self.fix_of(machine, &argv[0]);
        let lower = done.stderr.to_lowercase();
        if cap > 0 && ALLOCATION_FAILURES.iter().any(|m| lower.contains(m)) {
            let note = self.limit_note("address_cap");
            return Refusal::new(
                Kind::OutOfMemory,
                format!("{label} ran under a {cap}-byte address-space cap{note}, so an allocation failure here means this translation unit is too large for one compile and not that the machine is out of memory"),
            );
        }
        if matches!(done.code, Some(126 | 127)) {
            return Refusal::new(
                Kind::ToolAbsent,
                format!(
                    "{label}: {} not found or not executable (exit {}); {fix}",
                    argv[0],
                    done.code.unwrap_or(0)
                ),
            );
        }
        let rc = done
            .code
            .map_or("a signal".to_string(), |c| format!("rc={c}"));
        let complaint: String = done.stderr.chars().take(COMPLAINT_CHARS).collect();
        Refusal::new(
            Kind::ToolFailed,
            format!(
                "{label} failed ({rc}).\n$ {}\n{complaint}\nran in {}; {fix}",
                argv.join(" "),
                dir.display()
            ),
        )
    }
}

/// The key of a published slot: the name of its folder.
pub(crate) fn key_of(entry: &Entry) -> String {
    entry
        .dir
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
}

#[cfg(test)]
#[path = "argv_tests.rs"]
mod argv_tests;

#[path = "runtime_job.rs"]
mod runtime_job;
