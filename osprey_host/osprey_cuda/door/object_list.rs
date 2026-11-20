//! What the door's source list and the served architectures turn into for the device build: which sources are built, in which
//! group, and the code-generation flags of the CUDA compiler. Read by the build script; its tests run in the crate's test target.

/// The group of the door a source belongs to.
#[derive(Debug, PartialEq, Eq)]
pub enum Group {
    /// Built in every mode: the core and the host group.
    Host,
    /// Built only with the device runtime: the lines of `sources.txt` prefixed `cuda`.
    Device,
}

/// One source of the door, by its path under `eagle/door/`.
#[derive(Debug, PartialEq, Eq)]
pub struct Source {
    /// The path, relative to `eagle/door/`.
    pub path: String,
    /// The group the list puts it in.
    pub group: Group,
}

/// The sources of the list `text` (the door's `sources.txt`), in list order. Blank lines and `#` comment lines are skipped.
///
/// # Errors
/// Refuses a line that is neither a path nor `cuda <path>`, naming it.
pub fn sources(text: &str) -> Result<Vec<Source>, String> {
    let mut found = Vec::new();
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let words: Vec<&str> = line.split_whitespace().collect();
        let source = match words.as_slice() {
            [path] => Source {
                path: (*path).to_string(),
                group: Group::Host,
            },
            ["cuda", path] => Source {
                path: (*path).to_string(),
                group: Group::Device,
            },
            _ => {
                return Err(format!(
                    "sources.txt: the line {line:?} is not `<path>` or `cuda <path>`"
                ))
            }
        };
        found.push(source);
    }
    Ok(found)
}

/// The `-gencode` flags for the architectures `archs` (compute capabilities as numbers, oldest first): device code for each, and
/// the intermediate code of the newest so a later GPU can still load the library.
pub fn gencode(archs: &[u32]) -> Vec<String> {
    let each = archs.iter().flat_map(|arch| {
        [
            "-gencode".to_string(),
            format!("arch=compute_{arch},code=sm_{arch}"),
        ]
    });
    let newest = archs.last().into_iter().flat_map(|arch| {
        [
            "-gencode".to_string(),
            format!("arch=compute_{arch},code=compute_{arch}"),
        ]
    });
    each.chain(newest).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Oracle: the list written by hand in the layout of the door's own `sources.txt`: comments, plain lines and `cuda` lines.
    const LIST: &str = "# comment\n\nContext.cpp\nhost/Runner.cpp\n# the device group\ncuda device/Fft.cpp\ncuda device/Loop.cu\ndevice/Device.cpp\n";

    #[test]
    fn plain_lines_are_host_and_cuda_lines_are_device_in_list_order() {
        let found: Vec<(String, bool)> = sources(LIST)
            .unwrap()
            .into_iter()
            .map(|s| (s.path, s.group == Group::Device))
            .collect();
        let want = [
            ("Context.cpp", false),
            ("host/Runner.cpp", false),
            ("device/Fft.cpp", true),
            ("device/Loop.cu", true),
            ("device/Device.cpp", false),
        ];
        let want: Vec<(String, bool)> = want.iter().map(|(p, d)| ((*p).to_string(), *d)).collect();
        assert_eq!(found, want);
    }

    #[test]
    fn a_line_of_another_shape_is_refused_by_name() {
        for bad in ["opencl a.cpp", "a.cpp b.cpp", "cuda a.cpp b.cpp"] {
            let why = sources(bad).unwrap_err();
            assert!(why.contains(bad), "{why}");
        }
    }

    /// Oracle: the flags the door owner's own device build line spells for the archs 61 and 90.
    #[test]
    fn gencode_builds_each_arch_and_the_intermediate_code_of_the_newest() {
        assert_eq!(
            gencode(&[61, 90]).join(" "),
            "-gencode arch=compute_61,code=sm_61 -gencode arch=compute_90,code=sm_90 \
             -gencode arch=compute_90,code=compute_90"
        );
        assert!(gencode(&[]).is_empty());
    }
}
