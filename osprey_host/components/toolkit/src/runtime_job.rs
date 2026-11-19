//! A unit built by the runtime compile library instead of a tool: the same slot, key and refusals as a tool run, the figures read
//! from the image. Part of [`crate::compile`]; it reads a [`Job`]'s members directly.
use super::{Job, COMPLAINT_CHARS};
use crate::discover::Machine;
use crate::figures;
use crate::refusal::{Done, Kind, Refusal};
use crate::runtime::Runtime;
use crate::store::{self, Built, Entry, Store};
use crate::toolkit::Toolkit;
use crate::wire;
use rcore::json::J;
use rcore::sha256;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// What a request states for the runtime path (its `runtime` member).
pub(crate) struct RuntimeJob {
    /// The compile options, with places filled like a template's.
    pub options: Vec<String>,
    /// The kind of image to fetch: a key of the table's `images`.
    pub target: String,
    /// The names of the header trees served to the compile from memory.
    pub trees: Vec<String>,
}

impl RuntimeJob {
    /// The runtime member of a request, if it has one.
    pub(crate) fn from_request(req: &J) -> Done<Option<RuntimeJob>> {
        let Some(member) = req.get("runtime") else {
            return Ok(None);
        };
        Ok(Some(RuntimeJob {
            options: wire::texts(member, "options", "runtime")?,
            target: wire::text(member, "target", "runtime")?.to_string(),
            trees: wire::texts(member, "headers", "runtime")?,
        }))
    }
}

impl Toolkit {
    /// The slot of `job` compiled in this process.
    pub(crate) fn run_runtime_job(
        &self,
        job: &Job,
        runtime: &RuntimeJob,
        machine: &Machine,
        root: &Path,
    ) -> Done<Entry> {
        let spec = wire::need(&self.spec, "runtime_compile", "the tool table")?;
        let library = Runtime::open(spec)?.ok_or_else(|| {
            Refusal::new(
                Kind::ToolAbsent,
                "the runtime compile library no longer loads",
            )
        })?;
        let mut options = vec![];
        for option in &runtime.options {
            options.push(super::fill(option, &mut |place| {
                if let Some(name) = place
                    .strip_prefix("include:")
                    .filter(|n| spec.get("include").is_some_and(|i| i.get(n).is_some()))
                {
                    return Ok(Some(
                        library.include_root(spec, name)?.display().to_string(),
                    ));
                }
                let given = job
                    .places
                    .iter()
                    .find(|(p, _)| p == place)
                    .map(|(_, v)| v.clone());
                Ok(given.or_else(|| {
                    machine
                        .tool(job.tool_of(place))
                        .map(|t| t.path.display().to_string())
                }))
            })?);
        }
        let trees = runtime
            .trees
            .iter()
            .map(|name| {
                self.trees.iter().find(|t| t.name == name).ok_or_else(|| {
                    Refusal::bad(format!("runtime: the header tree {name:?} is not embedded"))
                })
            })
            .collect::<Done<Vec<_>>>()?;
        let mut parts = vec![
            "toolkit.runtime/1".to_string(),
            library.identity()?,
            job.text.clone(),
            job.name.clone(),
            job.output_suffix.clone(),
            format!("keep_source {} {}", job.keep_source, job.source_suffix),
        ];
        parts.extend([
            runtime.target.clone(),
            job.figures.clone().unwrap_or_default(),
        ]);
        parts.extend(options.iter().cloned());
        parts.extend(
            trees
                .iter()
                .map(|t| format!("tree {} {}", t.name, t.digest())),
        );
        parts.extend(job.key_terms.iter().cloned());
        let key = sha256::hex(parts.join("\0").as_bytes());
        let headers: Vec<(String, Vec<u8>)> = trees
            .iter()
            .flat_map(|t| t.files.iter().map(|(n, b)| ((*n).to_string(), b.to_vec())))
            .collect();
        let kind = spec
            .get("calls")
            .and_then(|c| c.get("images"))
            .and_then(|i| i.get(&runtime.target));
        let text_image = matches!(kind.and_then(|k| k.get("text")), Some(J::Bool(true)));
        Store::new(root).get_or_build(&key, |scratch| {
            let start = Instant::now();
            store::note_run();
            let done = library.compile(&job.name, &job.text, &headers, &options, &runtime.target)?;
            let Some(image) = done.image else {
                let complaint: String = done.log.chars().take(COMPLAINT_CHARS).collect();
                return Err(Refusal::new(
                    Kind::ToolFailed,
                    format!("compile of {:?} under the runtime compiler failed\n$ runtime {}\n{complaint}", job.name, options.join(" ")),
                ));
            };
            let image = if text_image { trim_nul(image) } else { image };
            let output = format!("{}{}", job.name, job.output_suffix);
            fs::write(scratch.join(&output), &image).map_err(|e| Refusal::io("cannot write", &scratch.join(&output), &e))?;
            let mut argv = vec!["runtime".to_string()];
            argv.extend(options.iter().cloned());
            let meta = wire::obj(vec![
                ("argv", wire::strs(&argv)),
                ("seconds", J::Num(start.elapsed().as_secs_f64())),
                ("elf", J::Bool(image.starts_with(b"\x7fELF"))),
                ("figures", figures::parse(job.figures.as_deref(), &done.log, &image)?),
            ]);
            let mut files = vec![output];
            if job.keep_source {
                // the offline path publishes the source it compiled; the answer lists it for either path
                let source = format!("{}{}", job.name, job.source_suffix);
                fs::write(scratch.join(&source), &job.text).map_err(|e| Refusal::io("cannot write", &scratch.join(&source), &e))?;
                files.push(source);
            }
            Ok(Built { files, meta })
        })
    }
}

/// `image` without its trailing NUL bytes (a text image such as PTX is NUL-terminated by the library).
fn trim_nul(mut image: Vec<u8>) -> Vec<u8> {
    while image.last() == Some(&0) {
        image.pop();
    }
    image
}
