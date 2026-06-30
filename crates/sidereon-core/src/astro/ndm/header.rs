//! Shared CCSDS Navigation Data Message header primitives.

use super::FieldMap;

/// Shared CCSDS NDM header fields.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct NdmHeader {
    /// CCSDS message version read through the caller-supplied version key.
    pub(crate) vers: String,
    /// Optional `CREATION_DATE` header value.
    pub(crate) creation_date: Option<String>,
    /// Optional `ORIGINATOR` header value.
    pub(crate) originator: Option<String>,
}

impl NdmHeader {
    /// Read the common CCSDS NDM header fields from a KVN field map.
    pub(crate) fn read(map: &FieldMap, vers_key: &str) -> Self {
        Self {
            vers: map.get(vers_key).unwrap_or_default().to_string(),
            creation_date: map.get("CREATION_DATE").map(str::to_string),
            originator: map.get("ORIGINATOR").map(str::to_string),
        }
    }

    /// Write the common CCSDS NDM header fields as KVN `KEY = VALUE` lines.
    pub(crate) fn write_kvn(&self, vers_key: &str) -> Vec<String> {
        vec![
            format!("{vers_key} = {}", self.vers),
            format!(
                "CREATION_DATE = {}",
                self.creation_date.as_deref().unwrap_or_default()
            ),
            format!(
                "ORIGINATOR = {}",
                self.originator.as_deref().unwrap_or_default()
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::ndm::tokenize;

    #[test]
    fn read_header_from_tokenized_field_map() {
        let map = FieldMap::from_pairs(tokenize(
            "CCSDS_OMM_VERS = 2.0\n\
             CREATION_DATE = 2026-06-17T04:32:52.099296\n\
             ORIGINATOR = SIDEREON\n",
        ));

        assert_eq!(
            NdmHeader::read(&map, "CCSDS_OMM_VERS"),
            NdmHeader {
                vers: "2.0".to_string(),
                creation_date: Some("2026-06-17T04:32:52.099296".to_string()),
                originator: Some("SIDEREON".to_string()),
            }
        );
    }

    #[test]
    fn write_header_as_kvn_lines() {
        let header = NdmHeader {
            vers: "1.0".to_string(),
            creation_date: Some("2026-06-17T04:32:52.099296".to_string()),
            originator: None,
        };

        assert_eq!(
            header.write_kvn("CCSDS_CDM_VERS"),
            vec![
                "CCSDS_CDM_VERS = 1.0".to_string(),
                "CREATION_DATE = 2026-06-17T04:32:52.099296".to_string(),
                "ORIGINATOR = ".to_string(),
            ]
        );
    }
}
