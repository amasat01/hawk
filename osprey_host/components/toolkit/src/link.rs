//! `link`: the set library from the parts a build returned, the set text and the device images, as the instructions state.
//!
//! The request names the link template per operating system (`link`, with `linux` and `macos` members), optionally the set unit
//! (`set`, a build request: the text of the set, compiled with its own template) and the embedding recipe (`embed`: how to turn an
//! image file into an object that holds it under two symbols). The link takes the objects of the parts in order, then the objects
//! that embed the images, then the set's object. Every compile it makes goes through the store like a build.
use crate::compile::Job;
use crate::refusal::{Done, Kind, Refusal};
use crate::store::{Built, Store};
use crate::toolkit::Toolkit;
use crate::wire;
use rcore::json::J;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The key of the link templates for this operating system.
fn system() -> Done<&'static str> {
    match std::env::consts::OS {
        "macos" => Ok("macos"),
        "linux" => Ok("linux"),
        other => Err(Refusal::bad(format!(
            "the link instructions name macos and linux; this system is {other}"
        ))),
    }
}

/// The text of an embedding source for this system: a list of lines joined (one per line), or a text as it is.
fn source_text(recipe: &J) -> Done<String> {
    let found = wire::need(
        wire::need(recipe, "source", "embed recipe")?,
        system()?,
        "embed recipe source",
    )?;
    Ok(match found.str() {
        Some(text) => text.to_string(),
        None => found.strs().join("\n") + "\n",
    })
}

impl Toolkit {
    /// Links the library: the request is `{"name", "parts": [<answers of build>], "link": {"linux", "macos"}, "set": <build
    /// request>?, "embed": <recipe>?, "as": {...}?, "key_terms": [...]?}`; the answer is `{"library": path, "key", "hit",
    /// "report": {"argv", "seconds"}}`. A second call with the same parts links nothing.
    ///
    /// # Errors
    /// Refuses with a [`Refusal`](crate::refusal::Refusal) that names what is wrong with the request or the tool.
    pub fn link(&self, request: &str, root: &Path) -> Done<String> {
        let req = wire::parse("the link request", request)?;
        let name = wire::text(&req, "name", "link request")?.to_string();
        let machine = self.tools(Some(root))?;
        let store = Store::new(root);
        let (mut objects, mut keys) = (vec![], vec![]);
        let mut images = vec![];
        for part in wire::need(&req, "parts", "link request")?.arr() {
            let output = wire::text(wire::need(part, "files", "part")?, "output", "part")?;
            objects.push(PathBuf::from(output));
            keys.push(wire::text(part, "key", "part")?.to_string());
            if let Some(image) = part.get("embed").filter(|e| e.get("image").is_some()) {
                images.push(image.clone());
            }
        }
        for image in &images {
            let entry = self.embed_object(&req, image, root)?;
            objects.push(entry.path(&format!("{}__image.o", wire::text(image, "name", "embed")?)));
            keys.push(crate::compile::key_of(&entry));
        }
        if let Some(set) = req.get("set") {
            let entry = self.run_job(&Job::from_request(set)?, root)?;
            objects.push(entry.path(&format!(
                "{}{}",
                wire::text(set, "name", "set")?,
                wire::text_or(set, "output_suffix", ".o")
            )));
            keys.push(crate::compile::key_of(&entry));
        }
        let template = wire::texts(wire::need(&req, "link", "link request")?, system()?, "link")?;
        let job = Job::linking(&req, &name, &template, &objects, &keys)?;
        let key = self.job_key(&job, &machine);
        let entry = store.get_or_build(&key, |scratch| {
            self.link_into(&job, &machine, root, scratch, &name)
        })?;
        Ok(wire::render(&wire::obj(vec![
            ("library", wire::s(&entry.path(&name).display().to_string())),
            ("key", wire::s(&crate::compile::key_of(&entry))),
            ("hit", J::Bool(entry.hit)),
            ("report", crate::compile::report_of(&entry.meta)),
        ])))
    }

    /// Runs the link of `job` into `scratch`.
    fn link_into(
        &self,
        job: &Job,
        machine: &crate::discover::Machine,
        root: &Path,
        scratch: &Path,
        name: &str,
    ) -> Done<Built> {
        let argv = self.argv_of(job, machine, root, scratch)?;
        let start = Instant::now();
        let done = self.run(&argv, machine, root, job.jobs)?;
        let label = format!("link of {name:?}");
        if done.code != Some(0) {
            return Err(self.refuse(&argv, &done, &label, machine));
        }
        if !scratch.join(name).is_file() {
            return Err(Refusal::new(
                Kind::ToolFailed,
                format!("{label} wrote no {name}"),
            ));
        }
        let meta = wire::obj(vec![
            ("argv", wire::strs(&argv)),
            ("seconds", J::Num(start.elapsed().as_secs_f64())),
        ]);
        Ok(Built {
            files: vec![name.to_string()],
            meta,
        })
    }

    /// The object that holds the image of `embed` under its two symbols: the recipe's source, assembled by its own command.
    fn embed_object(&self, req: &J, embed: &J, root: &Path) -> Done<crate::store::Entry> {
        let recipe = wire::need(
            req,
            "embed",
            "link request: a part carries an image, so the request needs an embed recipe",
        )?;
        let name = wire::text(embed, "name", "embed")?;
        let image = wire::text(embed, "image", "embed")?;
        let text = source_text(recipe)?
            .replace("{size_symbol}", wire::text(embed, "size_symbol", "embed")?)
            .replace("{symbol}", wire::text(embed, "symbol", "embed")?)
            .replace("{image_path}", image);
        let digest = wire::text(embed, "digest", "embed")?;
        let job = Job::assembling(
            req,
            &format!("{name}__image"),
            &text,
            &wire::texts(recipe, "compile", "embed recipe")?,
            digest,
        )?;
        self.run_job(&job, root)
    }
}
