//! Backends and data documents for the unit tests of this crate. The crate ships no documents of its own; the tests read the
//! shipped documents of the two libraries, merged into one directory in the temporary folder, and lower through a backend of
//! each kind that carries no identity.
use crate::backend::{Backend, Renderer};
use crate::facts::FusionRow;
use crate::facts::{answer, stated_text, CardSet, Failed};
use crate::lower::card::{BuildNumbers, ContractionRoute, TargetCards, WarpForm, WideLimits};
use crate::lower::image::ImageCommand;
use crate::lower::request::KernelRequest;
use crate::lower::unit_meta::{DispatchCosts, StepsPair, StepsRegime, StepsTable};
use aether_writers::contract::types::Target;
use aether_writers::contract::types::{
    Build, FoldForm, FoldLimits, FormSettings, LaneSettings, Scalar, WideCard, DEFAULT_CELLS_ROW,
    DEFAULT_WARPS,
};
use aether_writers::contract::Contract;
use aether_writers::contract::Json;
use aether_writers::refuse::Refusal;
use raptor_wings::team::WarpEntry;
use std::ffi::CStr;
use std::path::{Path, PathBuf};

/// The documents the test backends accept as stated: fixture documents of the base's own test data.
pub const ALL_CARDS: &[&str] = &["numbers.json", "fusion.json"];

/// The folder of the fixture documents.
pub fn shipped() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/facts")
}

/// The machine record a test host states: what its writer reads of the machine, as one fixture answer.
const HOST_MACHINE: &str = include_str!("../tests/data/host_machine.json");
/// The machine record a test device states, for the architecture word `{arch}`.
const DEVICE_MACHINE: &str = include_str!("../tests/data/device_machine.json");
/// The architecture word a test device states when the caller states none.
pub const DEFAULT_ARCH: &str = "slow_a";

/// A host backend for tests.
#[derive(Default)]
pub struct TestHost;

/// A device backend for tests.
#[derive(Default)]
pub struct TestDevice;

impl Backend for TestHost {
    type Probe = ();
    type Stated = ();
    fn target_name(&self) -> &'static str {
        "host"
    }
    fn probe(&self) {}
    fn read_stated(&self, _record: &Json) -> Result<(), Failed> {
        Ok(())
    }
    fn format(&self) -> &'static str {
        "host"
    }
    fn name(&self) -> &'static CStr {
        c"test_host"
    }
    fn abi(&self) -> u32 {
        0
    }
    fn identity(&self) -> &'static str {
        ""
    }
    fn target(&self) -> Target {
        Target::Host
    }
    fn card_names(&self) -> &'static [&'static str] {
        ALL_CARDS
    }
    fn machine(
        &self,
        _cards: &CardSet,
        _compiler: &str,
        _probe: &(),
        _stated: &(),
    ) -> Result<Json, Failed> {
        Ok(Contract::parse(HOST_MACHINE)
            .map_err(|why| Failed::Card(why.reason))?
            .0)
    }
    fn cards(
        &self,
        _docs: &CardSet,
        _request: &KernelRequest,
    ) -> Result<Box<dyn TargetCards>, Refusal> {
        Ok(Box::new(FixtureCards::host()))
    }
    fn opt_levels(&self, _cards: &CardSet) -> Result<Vec<String>, Failed> {
        Err(Failed::Bad("the test backend has no tool side".into()))
    }
    fn default_opt_level(&self, _cards: &CardSet) -> Result<String, Failed> {
        Err(Failed::Bad("the test backend has no tool side".into()))
    }
    fn stated_machine(
        &self,
        _cards: &CardSet,
        _machine: &Json,
    ) -> Result<(String, String), Failed> {
        Err(Failed::Bad("the test backend has no tool side".into()))
    }
    fn build_unit(
        &self,
        _req: &Json,
        _machine: &Json,
        _root: &Path,
    ) -> Result<String, toolkit::Refusal> {
        Err(toolkit::Refusal::bad("the test backend has no tool side"))
    }
    fn link(
        &self,
        _parts: &str,
        _set_text: &str,
        _name: &str,
        _root: &Path,
    ) -> Result<String, toolkit::Refusal> {
        Err(toolkit::Refusal::bad("the test backend has no tool side"))
    }
    fn render(&self) -> Renderer {
        aether_writers::host_render
    }
    fn toolkit(&self) -> Result<&'static toolkit::Toolkit, toolkit::Refusal> {
        Err(toolkit::Refusal::bad("the test backend has no tool side"))
    }
}

impl Backend for TestDevice {
    /// Nothing is probed: the test device states its architecture word or takes the default.
    type Probe = ();
    /// The architecture word the caller states, none for the default.
    type Stated = Option<String>;
    fn target_name(&self) -> &'static str {
        "accel"
    }
    fn probe(&self) {}
    fn read_stated(&self, record: &Json) -> Result<Option<String>, Failed> {
        stated_text(record, "arch")
    }
    fn format(&self) -> &'static str {
        "blob"
    }
    fn artifact_form(&self, request: &KernelRequest) -> Result<Option<String>, Refusal> {
        Ok(request.image_format.clone())
    }
    fn name(&self) -> &'static CStr {
        c"test_device"
    }
    fn abi(&self) -> u32 {
        0
    }
    fn identity(&self) -> &'static str {
        ""
    }
    fn target(&self) -> Target {
        Target::Device
    }
    fn card_names(&self) -> &'static [&'static str] {
        ALL_CARDS
    }
    fn machine(
        &self,
        _cards: &CardSet,
        _compiler: &str,
        _probe: &(),
        stated: &Option<String>,
    ) -> Result<Json, Failed> {
        let arch = stated.as_deref().unwrap_or(DEFAULT_ARCH);
        Ok(Contract::parse(&DEVICE_MACHINE.replace("{arch}", arch))
            .map_err(|why| Failed::Card(why.reason))?
            .0)
    }
    fn cards(
        &self,
        _docs: &CardSet,
        request: &KernelRequest,
    ) -> Result<Box<dyn TargetCards>, Refusal> {
        FixtureCards::device(request).map(|cards| Box::new(cards) as Box<dyn TargetCards>)
    }
    fn opt_levels(&self, _cards: &CardSet) -> Result<Vec<String>, Failed> {
        Err(Failed::Bad("the test backend has no tool side".into()))
    }
    fn default_opt_level(&self, _cards: &CardSet) -> Result<String, Failed> {
        Err(Failed::Bad("the test backend has no tool side".into()))
    }
    fn stated_machine(
        &self,
        _cards: &CardSet,
        _machine: &Json,
    ) -> Result<(String, String), Failed> {
        Err(Failed::Bad("the test backend has no tool side".into()))
    }
    fn build_unit(
        &self,
        _req: &Json,
        _machine: &Json,
        _root: &Path,
    ) -> Result<String, toolkit::Refusal> {
        Err(toolkit::Refusal::bad("the test backend has no tool side"))
    }
    fn link(
        &self,
        _parts: &str,
        _set_text: &str,
        _name: &str,
        _root: &Path,
    ) -> Result<String, toolkit::Refusal> {
        Err(toolkit::Refusal::bad("the test backend has no tool side"))
    }
    fn render(&self) -> Renderer {
        aether_writers::cuda_render
    }
    fn toolkit(&self) -> Result<&'static toolkit::Toolkit, toolkit::Refusal> {
        Err(toolkit::Refusal::bad("the test backend has no tool side"))
    }
}

/// The cards `request` is lowered with, through the test backend of the target it names.
pub fn load_cards(request: &KernelRequest, docs: &CardSet) -> Result<crate::lower::Cards, Refusal> {
    match request.targets.contains(&Target::Device) {
        true => crate::lower::Cards::load(&TestDevice, request, docs),
        false => crate::lower::Cards::load(&TestHost, request, docs),
    }
}

/// The numbers a test backend answers with, stated here and read from no card document: what the plan's steps decide on for a
/// target of each kind. Each comment says which plan behaviour the number drives.
pub struct FixtureCards {
    /// The numbers the steps decide on.
    deciding: BuildNumbers,
    /// The build numbers, before the caller's settings.
    build: Build,
    /// The measured steps-reverse rows of the kind.
    steps: StepsTable,
    /// Whether the cards are the launched kind's.
    launched: bool,
    /// The command of the image the request names, for the launched kind.
    image: Option<ImageCommand>,
}

/// The settings of the build both kinds declare; a request stating a setting no target declares is refused.
const SETTINGS: [&str; 8] = [
    "fold_form",
    "wide_form",
    "linalg_form",
    "route_to_ordered",
    "exports",
    "literal_table",
    "auto_segmented_set",
    "team_threads",
];
/// The settings only the launched kind declares.
const LAUNCHED_SETTINGS: [&str; 4] = ["launch_bounds", "fold_rows", "warps", "fast_entries"];

/// The build numbers both kinds state alike.
fn shared_numbers() -> BuildNumbers {
    // the wide-value thresholds by scalar width decide which vectors stay in registers (placement of wide values)
    let wide = [4, 8].map(|width| {
        let pick = |wide: i64, narrow: i64| if width == 4 { wide } else { narrow };
        (
            width,
            WideLimits {
                runtime_min: [pick(16, 2), pick(16, 4), pick(16, 0), pick(96, 8), 4096],
                wide_min: pick(128, 64),
            },
        )
    });
    BuildNumbers {
        // a leaf wider than this is bound as a proxy
        lazy_leaf_width: 32,
        // folds up to this many elements are unrolled
        rolled_fold_min: 16,
        lean_width: 4,
        uniform_run: 32,
        run_max: 32,
        block_run: 1024,
        tile_min: 64,
        tile_max: 1024,
        // a matrix up to this order is held by one thread; larger ones take a team
        linalg_max_dim: 32,
        linalg_team_max_dim: 4096,
        warp_entries: vec![],
        warp_size: None,
        cells_row: 32,
        scatter_max_chunks: 256,
        scatter_threads: 32,
        copy_threads: 256,
        team_threads: 128,
        team_threads_large: 256,
        box_lanes: vec![("float64".into(), 16), ("float32".into(), 8)],
        steps_reverse: None,
        device_steps_reverse: None,
        wide: wide.to_vec(),
        settings: SETTINGS.map(String::from).to_vec(),
    }
}

/// The build numbers of the state both kinds share, with the form of the kind.
fn shared_build(numbers: &BuildNumbers, launched: bool) -> Build {
    Build {
        form: FormSettings {
            lazy_leaf_width: numbers.lazy_leaf_width,
            rolled_fold_min: numbers.rolled_fold_min,
            lean_width: Some(numbers.lean_width),
            uniform_run: (!launched).then_some(numbers.uniform_run),
        },
        fold_form: FoldForm::default(),
        lanes: LaneSettings {
            run_max: Some(numbers.run_max),
            block_run: Some(numbers.block_run),
            tile: (!launched).then_some((numbers.tile_min, numbers.tile_max)),
        },
        linalg_max_dim: Some(numbers.linalg_max_dim),
        cells_row: numbers.cells_row.max(DEFAULT_CELLS_ROW),
        blend: !launched,
        warps: DEFAULT_WARPS,
        launch_bounds: None,
        fold_k: None,
        fold_limits: launched.then_some(FoldLimits {
            k_max: 8,
            regs_base: 20,
            regs_column: 2,
            residency_regs: 32,
        }),
        wide_card: launched.then_some(WideCard {
            row_scalars: vec![],
            slice_min: Some(128),
            forced: false,
        }),
    }
}

impl FixtureCards {
    /// The cards of the host kind: no warp rows and no class of wide values.
    pub fn host() -> FixtureCards {
        let mut deciding = shared_numbers();
        let steps = StepsTable {
            regime: StepsRegime::Lanes,
            widths: vec![4, 4, 2],
            ks: vec![8, 4, 8],
            lo: vec![1.0, 1.0, 1.0],
            hi: vec![1024.0, 64.0, 65536.0],
        };
        deciding.steps_reverse = Some(steps.clone());
        FixtureCards {
            build: shared_build(&deciding, false),
            deciding,
            steps,
            launched: false,
            image: None,
        }
    }

    /// The cards of the launched kind for the architecture the request's facts state: a double-precision warp row is kept only for
    /// the architectures whose word starts with `fast` (their double rate is fast), which decides whether a kernel of doubles takes the warp form.
    ///
    /// # Errors
    /// Refuses facts that state no architecture.
    pub fn device(request: &KernelRequest) -> Result<FixtureCards, Refusal> {
        let facts = Contract::parse(request.facts_for(Target::Device)?)?.0;
        let arch = facts
            .get("device_arch")
            .and_then(aether_writers::contract::Json::str)
            .ok_or_else(|| {
                aether_writers::refuse::bad("the device facts answer names no architecture")
            })?;
        let entry = |scalar: &str| WarpEntry {
            kind: "eigh".into(),
            n: 32,
            scalar: scalar.into(),
            outputs: vec![],
        };
        let fast = arch.starts_with("fast");
        let mut deciding = shared_numbers();
        deciding.warp_entries = [Some("float32"), fast.then_some("float64")]
            .into_iter()
            .flatten()
            .map(entry)
            .collect();
        deciding.warp_size = Some(32);
        deciding
            .settings
            .extend(LAUNCHED_SETTINGS.map(String::from));
        let steps = StepsTable {
            regime: StepsRegime::PerResident,
            widths: vec![2],
            ks: vec![8],
            lo: vec![0.1],
            hi: vec![6.4],
        };
        deciding.device_steps_reverse = Some(steps.clone());
        let image = request.image_format.as_ref().map(|_| ImageCommand {
            compiler: "devcc 1.0".into(),
            flags: format!("-image -arch={arch}"),
            isa_version: 85,
        });
        Ok(FixtureCards {
            build: shared_build(&deciding, true),
            deciding,
            steps,
            launched: true,
            image,
        })
    }
}

impl TargetCards for FixtureCards {
    fn numbers(&self) -> Build {
        self.build.clone()
    }
    fn box_lanes(&self) -> &[(String, i64)] {
        &self.deciding.box_lanes
    }
    fn settings(&self) -> &[String] {
        &self.deciding.settings
    }
    fn steps_reverse(&self) -> StepsPair {
        match self.launched {
            true => StepsPair {
                host: None,
                device: Some(self.steps.clone()),
            },
            false => StepsPair {
                host: Some(self.steps.clone()),
                device: None,
            },
        }
    }
    fn deciding(&self) -> &BuildNumbers {
        &self.deciding
    }
    fn wide_class(&self) -> Option<&str> {
        self.launched.then_some("wide_a")
    }
    fn contraction_route(&self, _scalar: Scalar) -> Option<ContractionRoute> {
        self.launched.then_some(ContractionRoute::Gemm)
    }
    fn warp_form(&self, _scalar: Scalar) -> Option<WarpForm> {
        self.launched.then_some(WarpForm {
            card_row: false,
            warp_size: 32,
            slice_min: 128,
            team_min: 128,
            tile_bytes: 40960,
        })
    }
    fn image(&self) -> Option<&ImageCommand> {
        self.image.as_ref()
    }
    fn fusion_row(&self, _request: &KernelRequest, _docs: &CardSet) -> Result<FusionRow, Refusal> {
        match self.launched {
            // the row both kinds fuse by: a register file, and for the launched kind the costs of a launch and of a plane byte
            true => Ok(FusionRow {
                regs_base: 20,
                regs_file: 255,
                launch_us: Some(2.7),
                plane_us_per_byte: Some(7.1e-6),
                local_over_plane: Some(1.0),
            }),
            false => Ok(FusionRow {
                regs_base: 0,
                regs_file: 32,
                launch_us: None,
                plane_us_per_byte: None,
                local_over_plane: None,
            }),
        }
    }
    fn dispatch_costs(
        &self,
        request: &KernelRequest,
        docs: &CardSet,
    ) -> Result<Option<DispatchCosts>, Refusal> {
        if !self.launched {
            return Ok(None);
        }
        let row = self.fusion_row(request, docs)?;
        Ok(Some(DispatchCosts {
            launch_us: row.launch_us,
            plane_us_per_byte: row.plane_us_per_byte,
            local_over_plane: row.local_over_plane,
        }))
    }
}

/// The compiler identity the fixture backends are asked for: they ignore it.
const FIXTURE_COMPILER: &str = "/usr/bin/tool || Tool version 1.0 || 0";

/// The facts answer of the test host under the shipped fixture cards, the documents `docs` laid over them.
pub fn host_facts(docs: &[(String, String)]) -> String {
    answer(&shipped(), &TestHost, FIXTURE_COMPILER, docs, &(), &()).unwrap()
}

/// The facts answer of the test device under the shipped fixture cards, for the architecture word `arch` (none: the default) and
/// the documents `docs` laid over them.
pub fn device_facts(arch: Option<&str>, docs: &[(String, String)]) -> String {
    answer(
        &shipped(),
        &TestDevice,
        FIXTURE_COMPILER,
        docs,
        &(),
        &arch.map(String::from),
    )
    .unwrap()
}
