//! The one spelling of a target id.

/// The canonical spelling of the target name `name`: lowercase, with whitespace, dashes and underscores removed (`" Cu-Da "` is
/// `cuda`). That form must be a letter followed by letters and digits (ASCII, lowercase).
///
/// # Errors
/// The sentence that names `name` and the rule, when the canonical form is not a lowercase letter followed by lowercase
/// letters and digits.
pub fn canonical(name: &str) -> Result<String, String> {
    let spelled: String = name
        .chars()
        .filter(|c| !(c.is_whitespace() || *c == '_' || *c == '-'))
        .flat_map(char::to_lowercase)
        .collect();
    let mut chars = spelled.chars();
    let letter = chars.next().is_some_and(|c| c.is_ascii_lowercase());
    if letter && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()) {
        Ok(spelled)
    } else {
        Err(not_an_id(&crate::json::quote(name)))
    }
}

/// The refusal sentence for a target id that reads `shown` in a sentence.
#[must_use]
pub fn not_an_id(shown: &str) -> String {
    format!(
        "the target id {shown} is not a target id: lowercased, with whitespace, dashes and underscores removed, \
         it must be a letter followed by letters and digits"
    )
}
