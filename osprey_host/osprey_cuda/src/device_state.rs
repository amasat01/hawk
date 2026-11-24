//! The machine the caller states for a device build that is not for the machine running.
use aether_writers::contract::Json;
use osprey_lowering::facts::{stated_closed, stated_text, Failed};

/// A machine the caller states, for a build that is not for the machine running: every field absent means the card's default or
/// the machine's own.
#[derive(Default)]
pub struct Stated {
    /// The device architecture, which selects the rows of the device card.
    pub device_arch: Option<String>,
    /// The release of the device compiler as `major.minor`, in place of the one its identity's version line names. A compiler whose
    /// first version line carries no release (the device compiler driver prints it on a later line) must state it.
    pub device_release: Option<String>,
    /// The release of the GPU driver as `major.minor`, for a device build; none means the driver is not known.
    pub driver_release: Option<String>,
    /// Whether the runtime compile library, not an offline compiler, is the device tool of the machine.
    pub device_runtime: bool,
    /// The optimisation level of a build (`O0` to `O3`), in place of the default.
    pub opt_level: Option<String>,
}

/// The stated device machine of its JSON record (`device_arch`, `device_release`, `driver_release`, `device_runtime`, `opt_level`,
/// each optional); a key outside these and a value of the wrong kind are refused by name.
pub fn read_stated(record: &Json) -> Result<Stated, Failed> {
    stated_closed(
        record,
        &[
            "device_arch",
            "device_release",
            "driver_release",
            "device_runtime",
            "opt_level",
        ],
    )?;
    Ok(Stated {
        device_arch: stated_text(record, "device_arch")?,
        device_release: stated_text(record, "device_release")?,
        driver_release: stated_text(record, "driver_release")?,
        device_runtime: match record.get("device_runtime") {
            None => false,
            Some(Json::Bool(runtime)) => *runtime,
            Some(_) => {
                return Err(Failed::Bad(
                    "stated machine: device_runtime is not a flag".into(),
                ))
            }
        },
        opt_level: stated_text(record, "opt_level")?,
    })
}
