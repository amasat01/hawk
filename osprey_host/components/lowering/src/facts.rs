//! The machine facts the writers need, read from data files and from a probe of the running machine. The files (JSON, in a directory that
//! ships beside the library) state the profile lists, the per-instruction-set rules and
//! the select weights; the probe states the instruction set and the cache sizes. The answer is the JSON of the typed
//! contract's `machine` record, which the caller hands back to a render without reading it. A file that does not match its
//! schema is refused by name at load.
use crate::backend::{unserved, Backend};
use aether_writers::contract::{Contract, Json};
use aether_writers::refuse::{bad, fault, Refusal};
pub use raptor_wings::chain::FusionRow;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::path::{Path, PathBuf};

/// Why a facts question failed: the caller asked something that cannot be answered (a malformed stated machine, an unknown target
/// or profile), or a data file of the library is broken.
#[derive(Debug)]
pub enum Failed {
    /// The question is malformed or names what the cards do not have.
    Bad(String),
    /// A data file of the library cannot be read or does not hold what its schema says.
    Card(String),
}

impl From<String> for Failed {
    /// A failure of a data file, which is what every plain text failure of the loaders is.
    fn from(reason: String) -> Failed {
        Failed::Card(reason)
    }
}

/// The failure of a defect found in the document `file`, which is the caller's own when `stated`: their mistake, naming the
/// document, or else a broken data file of the library.
pub fn blame(stated: bool, file: &str, reason: String) -> Failed {
    match stated {
        true => Failed::Bad(format!("the {file} the request states: {reason}")),
        false => Failed::Card(reason),
    }
}

impl Failed {
    /// The refusal of the lowering library: a malformed question is the caller's bad request, a broken data file is the
    /// library's fault.
    pub fn refusal(self) -> Refusal {
        match self {
            Failed::Bad(reason) => bad(reason),
            Failed::Card(reason) => fault(reason),
        }
    }
}

impl From<Failed> for String {
    /// The reason in words.
    fn from(failed: Failed) -> String {
        match failed {
            Failed::Bad(reason) | Failed::Card(reason) => reason,
        }
    }
}

/// Parses the text of a data document; `file` names it in a refusal.
pub fn parse_card(text: &str, file: &str) -> Result<Json, String> {
    Ok(Contract::parse(text).map_err(|e| format!("{file}: {e}"))?.0)
}

/// The text of the data document `name` as it ships in the directory `dir`; a name outside the backend's documents is refused by
/// name.
pub fn shipped_card<B: Backend>(backend: &B, dir: &Path, name: &str) -> Result<String, Failed> {
    let cards = CardSet::new(dir, &[], backend.card_names());
    cards.knows(name)?;
    Ok(cards.text(name)?)
}

/// Reads the `cards` member of a request or of a stated machine: an object from a document name to the whole text of the
/// document. A value that is no text is refused by name; whether the library ships a document of that name is asked of the
/// `CardSet` the cards go into.
pub fn cards_from_json(value: &Json) -> Result<Vec<(String, String)>, String> {
    let Json::Obj(members) = value else {
        return Err("cards is not an object".into());
    };
    members
        .iter()
        .map(|(name, document)| match document {
            Json::Str(text) => Ok((name.clone(), text.clone())),
            _ => Err(format!("cards: {name} is not the text of a document")),
        })
        .collect()
}

/// Where the data documents come from: the facts directory, with the documents a caller states for this call laid over it.
/// Every reader takes the text of its document from here, and a stated document is parsed by the reader of the file it replaces.
pub struct CardSet {
    /// The facts directory.
    dir: PathBuf,
    /// The documents the caller states, by name, each the whole text of the document.
    stated: Vec<(String, String)>,
    /// The documents the library ships, which a caller may replace for one call.
    names: &'static [&'static str],
}

impl CardSet {
    /// The documents of the facts directory `dir`, the caller's `stated` ones in place of the files of their names; `names` are
    /// the documents the library ships.
    pub fn new(dir: &Path, stated: &[(String, String)], names: &'static [&'static str]) -> CardSet {
        CardSet {
            dir: dir.to_path_buf(),
            stated: stated.to_vec(),
            names,
        }
    }

    /// Refuses a document name the library does not ship, naming the documents it does.
    pub fn knows(&self, name: &str) -> Result<(), Failed> {
        match self.names.contains(&name) {
            true => Ok(()),
            false => Err(Failed::Bad(format!(
                "card: the unknown document {name}; the documents are {}",
                self.names.join(", ")
            ))),
        }
    }

    /// Refuses every document the caller states that the library does not ship.
    pub fn check_stated(&self) -> Result<(), Failed> {
        self.stated
            .iter()
            .try_for_each(|(name, _)| self.knows(name))
    }

    /// What a defect found while reading the document `file` is: a mistake of the caller's request when the caller stated that
    /// document, a broken data file of the library otherwise. The one place that decides it, since only the set knows which
    /// documents are the caller's, and a reader's reason does not always name the document.
    pub fn defect(&self, file: &str, reason: String) -> Failed {
        blame(self.states(file), file, reason)
    }

    /// Whether the caller states the document `file`.
    pub fn states(&self, file: &str) -> bool {
        self.stated.iter().any(|(name, _)| name == file)
    }

    /// The text of the document `file`.
    pub fn text(&self, file: &str) -> Result<String, String> {
        if let Some((_, text)) = self.stated.iter().find(|(name, _)| name == file) {
            return Ok(text.clone());
        }
        std::fs::read_to_string(self.dir.join(file)).map_err(|e| format!("cannot read {file}: {e}"))
    }

    /// A digest of the documents in effect (the text of every document the library ships, the caller's or the file's), or none when
    /// the caller states no document. A facts answer states it only in that case, and a lowering request is refused when its
    /// answers state another than its own cards give.
    pub fn digest(&self) -> Result<Option<String>, String> {
        if self.stated.is_empty() {
            return Ok(None);
        }
        let mut bytes = vec![];
        for name in self.names {
            bytes.extend_from_slice(name.as_bytes());
            bytes.push(0);
            bytes.extend_from_slice(self.text(name)?.as_bytes());
            bytes.push(0);
        }
        Ok(Some(rcore::sha256::hex(&bytes)))
    }

    /// The parsed document `file`.
    pub fn json(&self, file: &str) -> Result<Json, String> {
        parse_card(&self.text(file)?, file)
    }
}

/// The members of a card as a closed set: a key outside `keys` is refused by name.
pub fn closed<'a>(
    card: &'a Json,
    file: &str,
    keys: &[&str],
) -> Result<&'a [(String, Json)], String> {
    let Json::Obj(members) = card else {
        return Err(format!("{file} is not an object"));
    };
    match members
        .iter()
        .find(|(key, _)| !keys.contains(&key.as_str()))
    {
        Some((key, _)) => Err(format!("{file} has the unknown key {key}")),
        None => Ok(members),
    }
}

/// The value of a key a card must hold.
pub fn member<'a>(card: &'a Json, file: &str, key: &str) -> Result<&'a Json, String> {
    card.get(key)
        .ok_or_else(|| format!("{file} lacks the key {key}"))
}

/// A key of a card that must be a text.
pub fn text_of(card: &Json, file: &str, key: &str) -> Result<String, String> {
    match member(card, file, key)? {
        Json::Str(text) => Ok(text.clone()),
        _ => Err(format!("{file}: {key} is not a text")),
    }
}

/// A key of a card that must be a flag.
pub fn flag_of(card: &Json, file: &str, key: &str) -> Result<bool, String> {
    match member(card, file, key)? {
        Json::Bool(flag) => Ok(*flag),
        _ => Err(format!("{file}: {key} is not a flag")),
    }
}

/// A key of a card that must be a list of texts.
pub fn names_of(card: &Json, file: &str, key: &str) -> Result<Vec<String>, String> {
    let Json::Arr(items) = member(card, file, key)? else {
        return Err(format!("{file}: {key} is not a list"));
    };
    let name = |item: &Json| item.str().map(String::from);
    items
        .iter()
        .map(name)
        .collect::<Option<_>>()
        .ok_or_else(|| format!("{file}: {key} holds an item that is not a text"))
}

/// A number of a card that must be positive.
pub fn positive(value: &Json, what: &str) -> Result<f64, String> {
    match value {
        Json::Num(n) => n
            .parse::<f64>()
            .ok()
            .filter(|w| *w > 0.0)
            .ok_or_else(|| format!("{what} is not positive")),
        _ => Err(format!("{what} is not a number")),
    }
}

/// The `select_weight` key of a card: the weights for `float64` and `float32` (`defaults`, each positive), replaced for a
/// device `arch` by the rows (`cc`, `dtype`, `select_weight`) that name it.
pub fn weights_of(card: &Json, file: &str, arch: Option<&str>) -> Result<[f64; 2], String> {
    let table = member(card, file, "select_weight")?;
    closed(table, file, &["defaults", "rows"])?;
    let defaults = member(table, file, "defaults")?;
    closed(defaults, file, &["float64", "float32"])?;
    let default = |scalar: &str| {
        positive(
            member(defaults, file, scalar)?,
            &format!("{file}: select_weight {scalar}"),
        )
    };
    let mut weights = [default("float64")?, default("float32")?];
    for row in table.get("rows").map_or(&[][..], Json::arr) {
        closed(row, file, &["cc", "dtype", "select_weight"])?;
        let weight = positive(
            member(row, file, "select_weight")?,
            &format!("{file}: a select_weight row"),
        )?;
        let at = match text_of(row, file, "dtype")?.as_str() {
            "float64" => 0,
            "float32" => 1,
            other => {
                return Err(format!(
                    "{file}: a select_weight row names the type {other}"
                ))
            }
        };
        if arch == Some(text_of(row, file, "cc")?.as_str()) {
            weights[at] = weight;
        }
    }
    Ok(weights)
}

/// A number of a fusion row that is a whole number, or null for a cost that is not measured.
pub fn optional_cost(row: &Json, file: &str, key: &str) -> Result<Option<f64>, String> {
    match member(row, file, key)? {
        Json::Null => Ok(None),
        value => positive(value, &format!("{file}: fusion row {key}")).map(Some),
    }
}

/// A fusion row of a card file, its keys closed and its numbers checked.
pub fn fusion_row(row: &Json, file: &str) -> Result<FusionRow, String> {
    let keys = [
        "regs_base",
        "regs_file",
        "launch_us",
        "plane_us_per_byte",
        "local_over_plane",
    ];
    closed(row, file, &keys)?;
    let whole = |key: &str| match member(row, file, key)?.int() {
        Some(n) if n >= 0 => Ok(n),
        _ => Err(format!("{file}: fusion row {key} is not a whole number")),
    };
    Ok(FusionRow {
        regs_base: whole("regs_base")?,
        regs_file: whole("regs_file")?,
        launch_us: optional_cost(row, file, "launch_us")?,
        plane_us_per_byte: optional_cost(row, file, "plane_us_per_byte")?,
        local_over_plane: optional_cost(row, file, "local_over_plane")?,
    })
}

/// A number as the wire writes it.
pub fn number(n: impl ToString) -> Json {
    Json::Num(n.to_string())
}

/// The select weights as the wire's record.
pub fn weights_json(weights: [f64; 2]) -> Json {
    Json::Obj(vec![
        ("float64".into(), number(weights[0])),
        ("float32".into(), number(weights[1])),
    ])
}

/// The record a caller states for a machine, split into what the base reads (`cards`: the data documents the answer is made from
/// in place of the files of their names) and the rest, which the backend reads as its own stated record.
///
/// # Errors
/// Refuses text that is not a JSON object and a `cards` member that is not an object of texts.
pub fn split_stated(text: &str) -> Result<(Json, Vec<(String, String)>), Failed> {
    let record = Contract::parse(text)
        .map_err(|e| Failed::Bad(format!("stated machine: {e}")))?
        .0;
    let Json::Obj(mut members) = record else {
        return Err(Failed::Bad("stated machine: not an object".into()));
    };
    let docs = match members.iter().position(|(key, _)| key == "cards") {
        Some(at) => cards_from_json(&members.remove(at).1)
            .map_err(|e| Failed::Bad(format!("stated machine: {e}")))?,
        None => vec![],
    };
    Ok((Json::Obj(members), docs))
}

/// Refuses a member of a stated record that `keys` do not list, by name.
///
/// # Errors
/// Refuses the first member outside `keys`.
pub fn stated_closed(record: &Json, keys: &[&str]) -> Result<(), Failed> {
    let Json::Obj(members) = record else {
        return Err(Failed::Bad("stated machine: not an object".into()));
    };
    match members
        .iter()
        .find(|(key, _)| !keys.contains(&key.as_str()))
    {
        Some((key, _)) => Err(Failed::Bad(format!("stated machine: unknown key {key}"))),
        None => Ok(()),
    }
}

/// The text member `key` of a stated record, absent when the record does not state it.
///
/// # Errors
/// Refuses a member that is no text.
pub fn stated_text(record: &Json, key: &str) -> Result<Option<String>, Failed> {
    match record.get(key) {
        None => Ok(None),
        Some(Json::Str(text)) => Ok(Some(text.clone())),
        Some(_) => Err(Failed::Bad(format!("stated machine: {key} is not a text"))),
    }
}

/// The positive whole number member `key` of a stated record, absent when the record does not state it.
///
/// # Errors
/// Refuses a member that is no positive integer.
pub fn stated_size(record: &Json, key: &str) -> Result<Option<i64>, Failed> {
    match record.get(key) {
        None => Ok(None),
        Some(value) => {
            value.int().filter(|n| *n > 0).map(Some).ok_or_else(|| {
                Failed::Bad(format!("stated machine: {key} is not a positive integer"))
            })
        }
    }
}

/// The facts answer of the backend for a compiler identity, from the cards in the directory `dir` (the caller's `docs` laid over
/// them), the backend's probe and the machine the caller states: the JSON of the contract's `machine` record and, last and only when
/// the caller states documents, the digest of the cards it was made under (`cards`). A card that does not match its schema and an
/// identity that is not well formed are refused with the reason.
pub fn answer<B: Backend>(
    dir: &Path,
    backend: &B,
    compiler: &str,
    docs: &[(String, String)],
    probe: &B::Probe,
    stated: &B::Stated,
) -> Result<String, Failed> {
    let cards = CardSet::new(dir, docs, backend.card_names());
    cards.check_stated()?;
    let mut machine = backend.machine(&cards, compiler, probe, stated)?;
    let Json::Obj(members) = &mut machine else {
        return Err(Failed::Card("the machine record is not an object".into()));
    };
    if let Some(digest) = cards.digest()? {
        members.push(("cards".into(), Json::Str(digest)));
    }
    Ok(machine.to_text())
}

/// The refusal of a facts question for a target other than the one the backend serves.
pub fn serves<B: Backend>(backend: &B, asked: &str) -> Result<(), Failed> {
    match asked == backend.target_name() {
        true => Ok(()),
        false => Err(Failed::Bad(unserved(backend, asked).reason)),
    }
}

/// The directory of the data files: the one the caller gives (which must be a directory), else `facts` beside the library file that
/// holds this code.
pub fn facts_dir(given: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(dir) = given {
        return if dir.is_dir() {
            Ok(dir.to_path_buf())
        } else {
            Err(format!("facts: {} is not a directory", dir.display()))
        };
    }
    /// The loader's description of an address, as `dladdr` fills it.
    #[repr(C)]
    struct DlInfo {
        /// The file the address lies in.
        file: *const c_char,
        /// Where that file is loaded.
        base: *mut c_void,
        /// The nearest symbol below the address.
        symbol: *const c_char,
        /// That symbol's address.
        symbol_address: *mut c_void,
    }
    extern "C" {
        /// Describes the file an address lies in.
        fn dladdr(address: *const c_void, info: *mut DlInfo) -> c_int;
    }
    let mut info = DlInfo {
        file: std::ptr::null(),
        base: std::ptr::null_mut(),
        symbol: std::ptr::null(),
        symbol_address: std::ptr::null_mut(),
    };
    // SAFETY: the address is one of this library's own functions and `info` is the structure `dladdr` fills.
    let found = unsafe { dladdr(facts_dir as *const c_void, &mut info) };
    if found == 0 || info.file.is_null() {
        return Err("cannot tell where the library is".into());
    }
    // SAFETY: `dladdr` answered a NUL-terminated file name.
    let file = unsafe { CStr::from_ptr(info.file) }
        .to_string_lossy()
        .into_owned();
    Path::new(&file)
        .parent()
        .map(|dir| dir.join("facts"))
        .ok_or_else(|| "the library has no directory".into())
}
