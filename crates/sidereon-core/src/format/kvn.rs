//! KVN tokenization and field lookup helpers.

/// Tokenize `KEY = VALUE` lines into trimmed key/value pairs.
pub(crate) fn tokenize(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() {
                return None;
            }
            let (key, value) = line.split_once('=')?;
            Some((key.trim().to_string(), value.trim().to_string()))
        })
        .collect()
}

/// A keyword that occurs more than once in one scope with different values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ConflictingField {
    /// The repeated keyword.
    pub(crate) key: String,
    /// The value of its first occurrence.
    pub(crate) first: String,
    /// The first later value that differs from `first`.
    pub(crate) second: String,
}

/// A generic key/value field map shared by KVN-style readers.
///
/// The map keeps every pair in source order. A reader that treats a keyword as
/// single-valued first calls [`Self::first_conflict`], which reports a repeat
/// whose value differs, and then reads the value with [`Self::get`] or
/// [`Self::get_last`]; once no conflict remains, every occurrence of a keyword
/// carries the same text, so the two lookups differ only in how they report an
/// empty value.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct FieldMap {
    fields: Vec<(String, String)>,
}

impl FieldMap {
    /// Build a field map from already tokenized pairs.
    pub(crate) fn from_pairs(fields: Vec<(String, String)>) -> Self {
        Self { fields }
    }

    /// Tokenize text as KVN and build a field map.
    pub(crate) fn parse(text: &str) -> Self {
        Self::from_pairs(tokenize(text))
    }

    /// Return the value of the first occurrence of `key`.
    ///
    /// Returns `None` if `key` is absent or its first occurrence has an empty
    /// value; the CCSDS readers treat a blank value as absent.
    pub(crate) fn get(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .filter(|v| !v.is_empty())
    }

    /// Return the last value for `key`.
    pub(crate) fn get_last(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .rev()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Borrow the raw key/value pairs in parse order.
    pub(crate) fn pairs(&self) -> &[(String, String)] {
        &self.fields
    }

    /// Return the first keyword accepted by `single_valued` that repeats with a
    /// different value, in source order of the conflicting repeat.
    ///
    /// An exact repeat carries no new information and is not a conflict.
    /// Keywords the predicate rejects, such as `COMMENT`, may repeat freely.
    pub(crate) fn first_conflict<F>(&self, single_valued: F) -> Option<ConflictingField>
    where
        F: Fn(&str) -> bool,
    {
        let mut first_values: std::collections::HashMap<&str, &str> =
            std::collections::HashMap::new();
        for (key, value) in &self.fields {
            if !single_valued(key) {
                continue;
            }
            match first_values.get(key.as_str()) {
                Some(first) if *first != value.as_str() => {
                    return Some(ConflictingField {
                        key: key.clone(),
                        first: (*first).to_string(),
                        second: value.clone(),
                    });
                }
                Some(_) => {}
                None => {
                    first_values.insert(key.as_str(), value.as_str());
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_ignores_blank_and_non_kv_lines() {
        let fields = tokenize(
            "\n\
             OBJECT_NAME = ISS\n\
             no equals here\n\
             MEAN_MOTION= 15.5 \n",
        );
        assert_eq!(
            fields,
            vec![
                ("OBJECT_NAME".to_string(), "ISS".to_string()),
                ("MEAN_MOTION".to_string(), "15.5".to_string()),
            ]
        );
    }

    #[test]
    fn field_map_get_is_first_occurrence_get_last_is_last() {
        let map = FieldMap::from_pairs(vec![
            ("A".to_string(), String::new()),
            ("A".to_string(), "first".to_string()),
            ("B".to_string(), "only".to_string()),
            ("A".to_string(), "last".to_string()),
        ]);

        assert_eq!(map.get("A"), None);
        assert_eq!(map.get_last("A"), Some("last"));
        assert_eq!(map.get("B"), Some("only"));
        assert_eq!(map.get("missing"), None);
        assert_eq!(map.pairs().len(), 4);

        let map = FieldMap::from_pairs(vec![
            ("C".to_string(), "x".to_string()),
            ("C".to_string(), "y".to_string()),
        ]);

        assert_eq!(map.get("C"), Some("x"));
        assert_eq!(map.get_last("C"), Some("y"));
    }

    #[test]
    fn first_conflict_reports_a_differing_repeat_and_ignores_exempt_keys() {
        let map = FieldMap::from_pairs(vec![
            ("COMMENT".to_string(), "one".to_string()),
            ("A".to_string(), "1".to_string()),
            ("COMMENT".to_string(), "two".to_string()),
            ("A".to_string(), "1".to_string()),
            ("B".to_string(), String::new()),
            ("B".to_string(), "2".to_string()),
        ]);

        assert_eq!(
            map.first_conflict(|key| key != "COMMENT"),
            Some(ConflictingField {
                key: "B".to_string(),
                first: String::new(),
                second: "2".to_string(),
            })
        );
        assert_eq!(map.first_conflict(|key| key == "A"), None);
    }

    #[test]
    fn parse_calls_tokenize() {
        let map = FieldMap::parse("A = 1\nB = 2\n");
        assert_eq!(map.get("A"), Some("1"));
        assert_eq!(map.get_last("B"), Some("2"));
    }
}
