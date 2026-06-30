//! Whitespace-delimited token scanning for free-format records.

use std::str::FromStr;

use crate::validate::{self, FieldError};

/// Scanner over whitespace-delimited tokens in a single line.
#[derive(Debug, Clone)]
pub(crate) struct Tokenizer<'a> {
    remaining: &'a str,
}

impl<'a> Tokenizer<'a> {
    /// Build a tokenizer from a line.
    pub(crate) fn new(line: &'a str) -> Self {
        Self { remaining: line }
    }

    /// Pull the next token and parse it as a strict finite float.
    pub(crate) fn next_f64(&mut self, field: &'static str) -> Result<f64, FieldError> {
        let token = self.next_str().ok_or(FieldError::Missing { field })?;
        validate::strict_f64(token, field)
    }

    /// Pull the next token and parse it as a strict integer.
    pub(crate) fn next_int<T: FromStr>(&mut self, field: &'static str) -> Result<T, FieldError> {
        let token = self.next_str().ok_or(FieldError::Missing { field })?;
        validate::strict_int(token, field)
    }

    /// Pull the next token as a raw satellite token string.
    pub(crate) fn next_sat(&mut self, field: &'static str) -> Result<String, FieldError> {
        let token = self.next_str().ok_or(FieldError::Missing { field })?;
        Ok(token.to_string())
    }

    /// Return the unconsumed remainder of the line.
    pub(crate) fn remainder(&self) -> &str {
        self.remaining
    }

    /// Pull the next raw token, if one is available.
    pub(crate) fn next_str(&mut self) -> Option<&'a str> {
        let trimmed = self.remaining.trim_start();
        if trimmed.is_empty() {
            self.remaining = trimmed;
            return None;
        }
        let token_end = trimmed.find(char::is_whitespace).unwrap_or(trimmed.len());
        let token = &trimmed[..token_end];
        self.remaining = &trimmed[token_end..];
        Some(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scans_free_format_tokens() {
        let mut tokenizer = Tokenizer::new("1.5  42  G05 rest words");
        assert_eq!(tokenizer.next_f64("float"), Ok(1.5));
        assert_eq!(tokenizer.next_int::<i32>("integer"), Ok(42));
        assert_eq!(tokenizer.next_sat("satellite"), Ok("G05".to_string()));
        assert!(tokenizer.remainder().contains("rest words"));
        assert_eq!(tokenizer.next_str(), Some("rest"));
        assert_eq!(tokenizer.next_str(), Some("words"));
        assert_eq!(tokenizer.next_str(), None);
    }

    #[test]
    fn reports_missing_field_at_end() {
        let mut tokenizer = Tokenizer::new("G05");
        assert_eq!(tokenizer.next_sat("satellite"), Ok("G05".to_string()));
        assert_eq!(
            tokenizer.next_f64("clock"),
            Err(FieldError::Missing { field: "clock" })
        );
    }
}
