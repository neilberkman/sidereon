//! Logical-record grouping helpers.

/// A header line and its continuation lines.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LogicalRecord<'a> {
    /// The header line that opened this record.
    pub(crate) header: &'a str,
    /// The one-based line number of the header.
    pub(crate) header_line_no: usize,
    /// Continuation lines belonging to this record.
    pub(crate) continuations: Vec<&'a str>,
}

impl<'a> LogicalRecord<'a> {
    /// Iterate over the header followed by each continuation line.
    pub(crate) fn lines(&self) -> impl Iterator<Item = &'a str> + '_ {
        std::iter::once(self.header).chain(self.continuations.iter().copied())
    }

    /// Return the one-based line number of the header.
    pub(crate) fn header_line_no(&self) -> usize {
        self.header_line_no
    }
}

/// Group physical lines into logical records using a continuation predicate.
pub(crate) fn group_records<'a, F>(
    lines: impl IntoIterator<Item = &'a str>,
    is_continuation: F,
) -> Vec<LogicalRecord<'a>>
where
    F: Fn(&str) -> bool,
{
    let mut records = Vec::new();
    for (line_index, line) in lines.into_iter().enumerate() {
        let header_line_no = line_index + 1;
        if records.is_empty() || !is_continuation(line) {
            records.push(LogicalRecord {
                header: line,
                header_line_no,
                continuations: Vec::new(),
            });
        } else if let Some(record) = records.last_mut() {
            record.continuations.push(line);
        }
    }
    records
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_header_lines_with_continuations() {
        let lines = ["A", " a1", " a2", "B", " b1"];
        let records = group_records(lines, |line| line.starts_with(' '));

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].header, "A");
        assert_eq!(records[0].header_line_no(), 1);
        assert_eq!(records[0].continuations, vec![" a1", " a2"]);
        assert_eq!(
            records[0].lines().collect::<Vec<_>>(),
            vec!["A", " a1", " a2"]
        );

        assert_eq!(records[1].header, "B");
        assert_eq!(records[1].header_line_no(), 4);
        assert_eq!(records[1].continuations, vec![" b1"]);
    }

    #[test]
    fn leading_continuation_like_line_starts_first_record() {
        let lines = [" first", "A", " a1"];
        let records = group_records(lines, |line| line.starts_with(' '));

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].header, " first");
        assert_eq!(records[0].header_line_no(), 1);
        assert!(records[0].continuations.is_empty());
        assert_eq!(records[1].header, "A");
        assert_eq!(records[1].continuations, vec![" a1"]);
    }
}
