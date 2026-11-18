//! Splitting and joining the flag variables of the environment the way a POSIX shell reads them: words by white space, single quotes
//! keep everything, double quotes keep everything but let a backslash escape `"` and `\`, and outside quotes a backslash escapes the
//! next character. A quote that is never closed, or a backslash at the end, is not a word list at all.

/// The words of `text`, or `None` when a quote is never closed or a backslash ends the text.
#[must_use]
pub fn split(text: &str) -> Option<Vec<String>> {
    let mut words = vec![];
    let mut word = String::new();
    let mut started = false;
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        match c {
            c if c.is_whitespace() => {
                if started {
                    words.push(std::mem::take(&mut word));
                    started = false;
                }
            }
            '\'' => {
                started = true;
                loop {
                    match chars.next()? {
                        '\'' => break,
                        other => word.push(other),
                    }
                }
            }
            '"' => {
                started = true;
                loop {
                    match chars.next()? {
                        '"' => break,
                        '\\' => match chars.next()? {
                            escaped @ ('"' | '\\') => word.push(escaped),
                            other => {
                                word.push('\\');
                                word.push(other);
                            }
                        },
                        other => word.push(other),
                    }
                }
            }
            '\\' => {
                started = true;
                word.push(chars.next()?);
            }
            other => {
                started = true;
                word.push(other);
            }
        }
    }
    if started {
        words.push(word);
    }
    Some(words)
}

/// One word as a shell reads it back as that word: bare when it holds only safe characters, else in single quotes.
fn quote(word: &str) -> String {
    let safe = |c: char| c.is_ascii_alphanumeric() || "@%+=:,./-_".contains(c);
    if !word.is_empty() && word.chars().all(safe) {
        word.to_string()
    } else {
        format!("'{}'", word.replace('\'', "'\"'\"'"))
    }
}

/// The words joined into text that [`split`] gives back.
#[must_use]
pub fn join(words: &[String]) -> String {
    words.iter().map(|w| quote(w)).collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        split(text).unwrap()
    }

    #[test]
    fn quotes_and_backslashes_group_and_escape_as_a_shell_does() {
        assert_eq!(words("-O2  -ccbin   /a/b"), ["-O2", "-ccbin", "/a/b"]);
        assert_eq!(
            words("-ccbin '/my dir/g++' -g"),
            ["-ccbin", "/my dir/g++", "-g"]
        );
        assert_eq!(
            words(r#"-ccbin "/my dir/g++" -D"A B""#),
            ["-ccbin", "/my dir/g++", "-DA B"]
        );
        assert_eq!(words(r"a\ b c\\d"), ["a b", r"c\d"]);
        assert_eq!(words(r#""say \"hi\" \n""#), [r#"say "hi" \n"#]);
        assert_eq!(words("'single \\ stays' x"), ["single \\ stays", "x"]);
        assert_eq!(words("a'b c'd"), ["ab cd"]);
        assert_eq!(words("'' x"), ["", "x"]);
        assert_eq!(words("   "), Vec::<String>::new());
    }

    #[test]
    fn text_that_is_not_a_word_list_is_none() {
        assert_eq!(split("'open"), None);
        assert_eq!(split("\"open"), None);
        assert_eq!(split("ends\\"), None);
    }

    #[test]
    fn joined_words_split_back_to_the_same_words() {
        let original: Vec<String> = ["-O2", "/my dir/g++", "it's", "", "a\"b"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        assert_eq!(join(&original), "-O2 '/my dir/g++' 'it'\"'\"'s' '' 'a\"b'");
        assert_eq!(split(&join(&original)).unwrap(), original);
    }
}
