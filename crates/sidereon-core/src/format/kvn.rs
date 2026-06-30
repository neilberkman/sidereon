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

/// A generic key/value field map shared by KVN-style readers.
///
/// Supports readers that need first-wins lookup through [`Self::get`] and
/// readers that need last-wins lookup through [`Self::get_last`].
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
    /// value, matching the OMM `from_field_pairs` closure.
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
    fn parse_calls_tokenize() {
        let map = FieldMap::parse("A = 1\nB = 2\n");
        assert_eq!(map.get("A"), Some("1"));
        assert_eq!(map.get_last("B"), Some("2"));
    }
}
