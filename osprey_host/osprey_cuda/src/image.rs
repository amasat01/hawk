//! The compile command of a device image, read from the device facts answer the request carries. The library never runs a
//! compiler: the answer states which compiler, which flags and which PTX version make the image, and the caller runs them.
use aether_writers::contract::types::Target;
use aether_writers::contract::{Contract, Json};
use aether_writers::refuse::{bad, unknown, Refusal};
use osprey_lowering::lower::image::ImageCommand;
use osprey_lowering::lower::request::KernelRequest;

/// The form of device image a build makes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Form {
    /// Text assembly.
    Ptx,
    /// A binary for one architecture.
    Cubin,
    /// Binaries for several architectures.
    Fatbin,
}

impl Form {
    /// The word a request and an answer use for the form.
    pub fn word(self) -> &'static str {
        match self {
            Form::Ptx => "Ptx",
            Form::Cubin => "Cubin",
            Form::Fatbin => "Fatbin",
        }
    }

    /// The key of the form's flag list in the facts answer.
    fn key(self) -> &'static str {
        match self {
            Form::Ptx => "ptx",
            Form::Cubin => "cubin",
            Form::Fatbin => "fatbin",
        }
    }
}

/// The form of image the request names, none when it names none.
///
/// # Errors
/// Refuses a form the library does not build.
pub fn form_of(request: &KernelRequest) -> Result<Option<Form>, Refusal> {
    let form = match request.image_format.as_deref() {
        None => return Ok(None),
        Some("Ptx") => Form::Ptx,
        Some("Cubin") => Form::Cubin,
        Some("Fatbin") => Form::Fatbin,
        Some(other) => return Err(unknown(format!("member {other} of image_format"))),
    };
    Ok(Some(form))
}

/// The command of the request's device image, from the `device_build` record of its device facts answer. An answer without the
/// record (made by an older library) is refused by name.
pub fn command_of_request(request: &KernelRequest, format: Form) -> Result<ImageCommand, Refusal> {
    command_of_facts(request.facts_for(Target::Device)?, format)
}

/// The command of an image of `format`, from a device facts answer.
pub fn command_of_facts(facts: &str, format: Form) -> Result<ImageCommand, Refusal> {
    let machine = Contract::parse(facts)?.0;
    let build = machine.get("device_build").ok_or_else(|| {
        bad("the device facts answer states no device_build: ask the facts again")
    })?;
    let key = format.key();
    let flags: Vec<&str> = build
        .get("flags")
        .and_then(|f| f.get(key))
        .ok_or_else(|| bad(format!("device_build states no flags for the format {key}")))?
        .arr()
        .iter()
        .map(|flag| {
            flag.str()
                .ok_or_else(|| bad("device_build: a flag is not a text"))
        })
        .collect::<Result<_, _>>()?;
    let text = |name: &str| match build.get(name) {
        Some(Json::Str(text)) => Ok(text.clone()),
        _ => Err(bad(format!("device_build states no {name}"))),
    };
    let stated = build
        .get("ptx_isa")
        .and_then(Json::int)
        .ok_or_else(|| bad("device_build states no ptx_isa"))?;
    Ok(ImageCommand {
        compiler: text("compiler")?,
        flags: flags.join(" "),
        isa_version: if format == Form::Cubin { 0 } else { stated },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_state::Stated;
    use osprey_lowering::facts::answer;

    /// The command of an image for a compiler version line and an architecture, from the shipped facts.
    fn command(line: &str, arch: &str, format: Form) -> Result<ImageCommand, String> {
        let dir = crate::testing::shipped();
        let stated = Stated {
            device_arch: Some(arch.into()),
            ..Stated::default()
        };
        let identity = format!("/usr/local/cuda/bin/nvcc || {line} || 0");
        let facts =
            answer(&dir, &crate::Cuda, &identity, &[], &(), &stated).map_err(String::from)?;
        command_of_facts(&facts, format).map_err(|why| why.reason)
    }

    #[test]
    fn the_command_is_the_stated_flags_and_the_releases_ptx_version() {
        let line =
            |release: &str| format!("Cuda compilation tools, release {release}, V{release}.77");
        let common = "-O3 -Xptxas -O3 -Wno-deprecated-gpu-targets";
        for (release, name, isa) in [("12.6", "nvcc 12.6", 85), ("12.9", "nvcc 12.9", 88)] {
            for (arch, cc) in [("sm_61", "61"), ("sm_80", "80")] {
                let flags = |format: Form| {
                    let got = command(&line(release), arch, format).unwrap();
                    assert_eq!(got.compiler, name);
                    (got.flags, got.isa_version)
                };
                let ptx = format!("-ptx -std=c++20 -arch={arch} {common}");
                assert_eq!(flags(Form::Ptx), (ptx, isa));
                let cubin = format!("-cubin -std=c++20 -arch={arch} {common}");
                assert_eq!(flags(Form::Cubin), (cubin, 0));
                let fat = format!(
                    "-fatbin -std=c++20 {common} -gencode arch=compute_{cc},code=[{arch},compute_{cc}]"
                );
                assert_eq!(flags(Form::Fatbin), (fat, isa));
            }
        }
    }

    #[test]
    fn a_release_without_a_row_is_refused_by_name() {
        let why = command(
            "Cuda compilation tools, release 13.0, V13.0.1",
            "sm_61",
            Form::Ptx,
        )
        .unwrap_err();
        assert!(why.starts_with("facts"), "{why}");
        assert!(
            why.contains("13.0") && why.contains("cuda_image.json"),
            "{why}"
        );
    }

    #[test]
    fn an_answer_without_a_command_is_refused_by_name() {
        let why = command_of_facts(r#"{"device_arch":"sm_61"}"#, Form::Ptx)
            .unwrap_err()
            .reason;
        assert!(why.contains("device_build"), "{why}");
    }
}
