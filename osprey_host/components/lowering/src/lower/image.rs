//! The compile command of a device image, read from the device facts answer the request carries. The library never runs a
//! compiler: the answer states which compiler, which flags and which PTX version make the image, and the caller runs them.
use super::request::KernelRequest;
use super::unit_meta::ImageFormat;
use aether_writers::contract::types::Target;
use aether_writers::contract::{Contract, Json};
use aether_writers::refuse::{bad, Refusal};

/// The three members every unit with a device image states about the command that makes the image. One request carries one device
/// facts answer, so every unit of one answer holds the same values and no unit is checked against another.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceImage {
    /// The compiler's short name: its tool and release (`nvcc 12.6`).
    pub compiler: String,
    /// The flags of the command, joined by one space. The host compiler of the set's objects (`-ccbin`) is the place `{host_compiler}` of the compile template, not among them.
    pub flags: String,
    /// The PTX version of the image's PTX as major * 10 + minor (85 is 8.5); 0 when the image holds no PTX.
    pub ptx_isa: i64,
}

/// The key of a format's flag list in the facts answer; an image of no format has none.
fn format_key(format: ImageFormat) -> Result<&'static str, Refusal> {
    match format {
        ImageFormat::Ptx => Ok("ptx"),
        ImageFormat::Cubin => Ok("cubin"),
        ImageFormat::Fatbin => Ok("fatbin"),
        ImageFormat::None => Err(bad("request key image_format: a device image has a format")),
    }
}

impl DeviceImage {
    /// The command of the request's device image, from the `device_build` record of its device facts answer. An answer without the
    /// record (made by an older library) is refused by name.
    pub fn of_request(
        request: &KernelRequest,
        format: ImageFormat,
    ) -> Result<DeviceImage, Refusal> {
        DeviceImage::of_facts(request.facts_for(Target::Device)?, format)
    }

    /// The command of an image of `format`, from a device facts answer.
    pub fn of_facts(facts: &str, format: ImageFormat) -> Result<DeviceImage, Refusal> {
        let machine = Contract::parse(facts)?.0;
        let build = machine.get("device_build").ok_or_else(|| {
            bad("the device facts answer states no device_build: ask the facts again")
        })?;
        let key = format_key(format)?;
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
        Ok(DeviceImage {
            compiler: text("compiler")?,
            flags: flags.join(" "),
            ptx_isa: if format == ImageFormat::Cubin {
                0
            } else {
                stated
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::{answer, Probe, Stated};

    /// The command of an image for a compiler version line and an architecture, from the shipped facts.
    fn command(line: &str, arch: &str, format: ImageFormat) -> Result<DeviceImage, String> {
        let dir = crate::testing::shipped();
        let probe = Probe {
            isa: "arm64".into(),
            vector_isa: "neon".into(),
            caches: None,
        };
        let stated = Stated {
            device_arch: Some(arch.into()),
            ..Stated::default()
        };
        let identity = format!("/usr/local/cuda/bin/nvcc || {line} || 0");
        let facts = answer(
            &dir,
            &crate::testing::TestDevice,
            &identity,
            &probe,
            &stated,
        )
        .map_err(String::from)?;
        DeviceImage::of_facts(&facts, format).map_err(|why| why.reason)
    }

    #[test]
    fn the_command_is_the_stated_flags_and_the_releases_ptx_version() {
        let line =
            |release: &str| format!("Cuda compilation tools, release {release}, V{release}.77");
        let common = "-O3 -Xptxas -O3 -Wno-deprecated-gpu-targets";
        for (release, name, isa) in [("12.6", "nvcc 12.6", 85), ("12.9", "nvcc 12.9", 88)] {
            for (arch, cc) in [("sm_61", "61"), ("sm_80", "80")] {
                let flags = |format: ImageFormat| {
                    let got = command(&line(release), arch, format).unwrap();
                    assert_eq!(got.compiler, name);
                    (got.flags, got.ptx_isa)
                };
                let ptx = format!("-ptx -std=c++20 -arch={arch} {common}");
                assert_eq!(flags(ImageFormat::Ptx), (ptx, isa));
                let cubin = format!("-cubin -std=c++20 -arch={arch} {common}");
                assert_eq!(flags(ImageFormat::Cubin), (cubin, 0));
                let fat = format!(
                    "-fatbin -std=c++20 {common} -gencode arch=compute_{cc},code=[{arch},compute_{cc}]"
                );
                assert_eq!(flags(ImageFormat::Fatbin), (fat, isa));
            }
        }
    }

    #[test]
    fn a_release_without_a_row_is_refused_by_name() {
        let why = command(
            "Cuda compilation tools, release 13.0, V13.0.1",
            "sm_61",
            ImageFormat::Ptx,
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
        let why = DeviceImage::of_facts(r#"{"device_arch":"sm_61"}"#, ImageFormat::Ptx)
            .unwrap_err()
            .reason;
        assert!(why.contains("device_build"), "{why}");
    }
}
