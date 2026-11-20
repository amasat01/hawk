//! The device arch policy, from a table's `device` member: ask the driver library the table names (by the functions the table names)
//! for the running GPU and the driver's release, take a pin from the ask or from the environment variable the table names, and clamp
//! the result to the archs the toolkit can target. Nothing here names a vendor: library names, function names, attribute numbers and
//! the spelling of an arch are the table's.
use crate::dl::Library;
use crate::refusal::{Done, Refusal};
use crate::wire;
use rcore::json::J;
use std::ffi::c_int;

/// What the driver library states.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Driver {
    /// The driver's release as the library encodes it.
    pub encoded: Option<i64>,
    /// The release decoded as `major.minor`.
    pub release: Option<String>,
    /// The arch of the running GPU, spelled by the table's `arch_format`.
    pub gpu_arch: Option<String>,
}

/// A driver function that takes an out-pointer: `(int*) -> int`.
type OutInt = unsafe extern "C" fn(*mut c_int) -> c_int;
/// A driver function that takes an unsigned flag: `(unsigned) -> int`.
type InitFn = unsafe extern "C" fn(u32) -> c_int;
/// A driver function that takes an out-pointer and an ordinal: `(int*, int) -> int`.
type ByOrdinal = unsafe extern "C" fn(*mut c_int, c_int) -> c_int;
/// A driver function that reads an attribute of a device: `(int*, int attribute, int device) -> int`.
type Attribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;

/// The text member `key` of the table's `calls` record.
fn call_name<'a>(calls: &'a J, key: &str) -> Done<&'a str> {
    wire::text(calls, key, "device.driver.calls")
}

/// Asks the driver library of `spec` (the table's `device.driver` record) about the GPU and the release; the reason, as text, when
/// no library of the list opens or it does not initialise.
///
/// # Errors
/// Refuses a table whose `driver` record lacks a member.
pub fn probe_driver(spec: &J) -> Done<Result<Driver, String>> {
    let names = wire::texts(spec, "libraries", "device.driver")?;
    let Some(lib) = Library::open_first(&names) else {
        return Ok(Err(format!(
            "no driver library found (looked for {})",
            names.join(", ")
        )));
    };
    let calls = wire::need(spec, "calls", "device.driver")?;
    // SAFETY: the table states these names as the functions with the signatures of the aliases above.
    let init = unsafe { lib.function::<InitFn>(call_name(calls, "init")?) };
    let Some(init) = init else {
        return Ok(Err(
            "the driver library has no initialisation entry".to_string()
        ));
    };
    // SAFETY: `init` was looked up with its stated signature; the argument is the table's flag value 0.
    let status = unsafe { init(0) };
    if status != 0 {
        return Ok(Err(format!(
            "the driver library did not initialise (status {status})"
        )));
    }
    let mut driver = Driver::default();
    // SAFETY: as above, the table states the signature of `version`.
    if let Some(version) = unsafe { lib.function::<OutInt>(call_name(calls, "version")?) } {
        let mut encoded: c_int = 0;
        // SAFETY: `encoded` is a valid out-pointer for the call.
        if unsafe { version(&raw mut encoded) } == 0 {
            driver.encoded = Some(i64::from(encoded));
            driver.release = Some(decode_release(spec, i64::from(encoded))?);
        }
    }
    driver.gpu_arch = gpu_arch(spec, &lib, calls)?;
    Ok(Ok(driver))
}

/// The `major.minor` of an encoded release by the table's `version_decode` `[major divisor, minor divisor]`.
fn decode_release(spec: &J, encoded: i64) -> Done<String> {
    let parts: Vec<i64> = wire::need(spec, "version_decode", "device.driver")?
        .arr()
        .iter()
        .map(J::int)
        .collect();
    let [major, minor] = parts[..] else {
        return Err(Refusal::bad(
            "device.driver: version_decode is not two numbers",
        ));
    };
    if major <= 0 || minor <= 0 {
        return Err(Refusal::bad(
            "device.driver: version_decode holds a divisor that is not positive",
        ));
    }
    Ok(format!("{}.{}", encoded / major, (encoded % major) / minor))
}

/// The arch of the current device (else device 0) by the table's attributes and `arch_format`.
fn gpu_arch(spec: &J, lib: &Library, calls: &J) -> Done<Option<String>> {
    // SAFETY: the table states these names as the functions with the signatures of the aliases above.
    let (current, ordinal, attribute) = unsafe {
        (
            lib.function::<OutInt>(call_name(calls, "current_device")?),
            lib.function::<ByOrdinal>(call_name(calls, "device_by_ordinal")?),
            lib.function::<Attribute>(call_name(calls, "attribute")?),
        )
    };
    let (Some(current), Some(ordinal), Some(attribute)) = (current, ordinal, attribute) else {
        return Ok(None);
    };
    let mut device: c_int = 0;
    // SAFETY: `device` is a valid out-pointer for both calls.
    let found =
        unsafe { current(&raw mut device) } == 0 || unsafe { ordinal(&raw mut device, 0) } == 0;
    if !found {
        return Ok(None);
    }
    let attributes = wire::need(spec, "attributes", "device.driver")?;
    let read = |key: &str| -> Done<Option<c_int>> {
        let which = c_int::try_from(wire::need(attributes, key, "device.driver.attributes")?.int())
            .map_err(|_| Refusal::bad("device.driver.attributes: a number is out of range"))?;
        let mut value: c_int = 0;
        // SAFETY: `value` is a valid out-pointer; `device` was given by the driver.
        Ok((unsafe { attribute(&raw mut value, which, device) } == 0).then_some(value))
    };
    let (Some(major), Some(minor)) = (read("major")?, read("minor")?) else {
        return Ok(None);
    };
    let format = wire::text(spec, "arch_format", "device.driver")?;
    Ok(Some(
        format
            .replace("{major}", &major.to_string())
            .replace("{minor}", &minor.to_string()),
    ))
}

/// The number in an arch name `sm_75` or `compute_75` (a trailing letter, as in `sm_90a`, is allowed), or `None`.
fn arch_number(name: &str) -> Option<u64> {
    let rest = name
        .strip_prefix("sm_")
        .or_else(|| name.strip_prefix("compute_"))?;
    let digits = rest
        .strip_suffix(|c: char| c.is_ascii_lowercase())
        .unwrap_or(rest);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// What decides the arch: a pin (from the ask, else from the environment) and what the machine has.
pub struct Inputs<'a> {
    /// The pin the ask carries.
    pub asked: Option<&'a str>,
    /// The pin the environment carries, and the variable it came from.
    pub env_pin: Option<(&'a str, &'a str)>,
    /// The arch of the running GPU.
    pub gpu: Option<&'a str>,
    /// The archs the toolkit can target, ascending; `None` when it cannot be asked.
    pub toolkit: Option<&'a [u64]>,
}

/// The arch a device build is for and why. The pin of the ask wins over the pin of the environment, which wins over the running GPU.
/// With nothing requested the answer is the table's `fallback` when the toolkit's archs are unknown, else the first toolkit arch
/// that is at least `portable_min` (a virtual `compute_` arch: no image for a guessed device). An arch above the toolkit's highest
/// is the highest, virtual (the driver compiles it forward); an arch below its lowest is refused; a name that is not an arch, or
/// no toolkit list, passes the request through unchanged.
///
/// # Errors
/// Refuses a requested arch below the toolkit's lowest, naming who requested it.
pub fn choose_arch(policy: &J, inputs: &Inputs) -> Done<(String, String)> {
    let (requested, who) = match (inputs.asked, inputs.env_pin, inputs.gpu) {
        (Some(pin), _, _) => (Some(pin), format!("device_arch={pin:?} in the ask")),
        (None, Some((var, pin)), _) => (Some(pin), format!("${var}={pin:?}")),
        (None, None, Some(gpu)) => (Some(gpu), "the running GPU".to_string()),
        (None, None, None) => (None, String::new()),
    };
    let fallback = wire::text(policy, "fallback", "device.arch")?;
    let portable_min =
        u64::try_from(wire::need(policy, "portable_min", "device.arch")?.int()).unwrap_or(0);
    let Some(requested) = requested else {
        return Ok(match inputs.toolkit.filter(|t| !t.is_empty()) {
            None => (
                fallback.to_string(),
                "no GPU reachable and the toolkit's archs unknown: the table's fallback".into(),
            ),
            Some(archs) => {
                let pick = archs
                    .iter()
                    .copied()
                    .find(|a| *a >= portable_min)
                    .unwrap_or(archs[archs.len() - 1]);
                (
                    format!("compute_{pick}"),
                    "no GPU reachable: the toolkit's lowest portable arch, virtual".into(),
                )
            }
        });
    };
    let (Some(number), Some(archs)) = (
        arch_number(requested),
        inputs.toolkit.filter(|t| !t.is_empty()),
    ) else {
        return Ok((requested.to_string(), format!("{who}, passed through")));
    };
    if number > archs[archs.len() - 1] {
        return Ok((
            format!("compute_{}", archs[archs.len() - 1]),
            format!("{who}, above the toolkit's highest arch: virtual, compiled forward"),
        ));
    }
    if number < archs[0] {
        return Err(Refusal::bad(format!(
            "{who} needs arch {requested}, below the minimum arch {} the toolkit supports; use an older toolkit for this device, or pin a supported arch",
            archs[0]
        )));
    }
    Ok((requested.to_string(), who))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> J {
        wire::parse("t", r#"{"fallback":"compute_61","portable_min":60}"#).unwrap()
    }

    fn pick(inputs: &Inputs) -> Result<String, String> {
        choose_arch(&policy(), inputs)
            .map(|(arch, _)| arch)
            .map_err(|r| r.message)
    }

    const ARCHS: [u64; 4] = [50, 61, 75, 80];

    fn inputs<'a>(
        asked: Option<&'a str>,
        gpu: Option<&'a str>,
        toolkit: Option<&'a [u64]>,
    ) -> Inputs<'a> {
        Inputs {
            asked,
            env_pin: None,
            gpu,
            toolkit,
        }
    }

    #[test]
    fn the_arch_is_the_running_gpu_unless_pinned_and_a_pin_of_the_ask_wins_over_the_environment() {
        assert_eq!(
            pick(&inputs(None, Some("sm_75"), Some(&ARCHS))),
            Ok("sm_75".into())
        );
        let both = Inputs {
            asked: Some("sm_61"),
            env_pin: Some(("PIN", "sm_80")),
            gpu: Some("sm_75"),
            toolkit: Some(&ARCHS),
        };
        assert_eq!(pick(&both), Ok("sm_61".into()));
        let env = Inputs {
            asked: None,
            env_pin: Some(("PIN", "sm_80")),
            gpu: Some("sm_75"),
            toolkit: Some(&ARCHS),
        };
        assert_eq!(
            choose_arch(&policy(), &env).unwrap(),
            ("sm_80".to_string(), "$PIN=\"sm_80\"".to_string())
        );
    }

    #[test]
    fn an_arch_above_the_toolkit_is_the_highest_virtual_and_one_below_is_refused_by_name() {
        assert_eq!(
            pick(&inputs(Some("sm_90"), None, Some(&ARCHS))),
            Ok("compute_80".into())
        );
        let low = pick(&inputs(Some("sm_35"), None, Some(&ARCHS))).unwrap_err();
        assert!(
            low.contains("sm_35") && low.contains("below the minimum arch 50"),
            "{low}"
        );
    }

    #[test]
    fn with_no_gpu_the_arch_is_virtual_at_the_lowest_portable_toolkit_arch_else_the_table_fallback()
    {
        assert_eq!(
            pick(&inputs(None, None, Some(&ARCHS))),
            Ok("compute_61".into())
        );
        assert_eq!(
            pick(&inputs(None, None, Some(&[50]))),
            Ok("compute_50".into())
        );
        assert_eq!(pick(&inputs(None, None, None)), Ok("compute_61".into()));
    }

    #[test]
    fn a_name_that_is_not_an_arch_or_an_unknown_toolkit_passes_through() {
        assert_eq!(
            pick(&inputs(Some("native"), None, Some(&ARCHS))),
            Ok("native".into())
        );
        assert_eq!(
            pick(&inputs(Some("sm_90a"), None, None)),
            Ok("sm_90a".into())
        );
        assert_eq!(arch_number("sm_90a"), Some(90));
        assert_eq!(arch_number("sm_"), None);
    }

    #[test]
    fn a_release_is_decoded_by_the_tables_divisors() {
        let spec = wire::parse("t", r#"{"version_decode":[1000,10]}"#).unwrap();
        assert_eq!(decode_release(&spec, 12060).unwrap(), "12.6");
        assert_eq!(decode_release(&spec, 11080).unwrap(), "11.8");
    }
}
