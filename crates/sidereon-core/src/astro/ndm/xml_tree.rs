//! Scoped element access shared by the CCSDS NDM/XML readers.
//!
//! Each reader looks values up inside the element that owns them (the message,
//! its header, one segment, one logical block) rather than across the whole
//! document, so an NDM combined instantiation holding several messages
//! (CCSDS 505.0-B-3 4.11) cannot merge fields from different messages.

use roxmltree::{Document, Node};

/// Every element named `tag` that is not nested inside another element of the
/// same name, in document order.
///
/// A single message has exactly one such element, its root; an NDM combined
/// instantiation holds any number of them inside `<ndm>`.
pub(crate) fn message_elements<'a, 'input>(
    doc: &'a Document<'input>,
    tag: &str,
) -> Vec<Node<'a, 'input>> {
    doc.descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == tag)
        .filter(|node| {
            !node
                .ancestors()
                .skip(1)
                .any(|ancestor| ancestor.is_element() && ancestor.tag_name().name() == tag)
        })
        .collect()
}

/// The child elements of `node`, in document order.
pub(crate) fn element_children<'a, 'input>(node: Node<'a, 'input>) -> Vec<Node<'a, 'input>> {
    node.children().filter(Node::is_element).collect()
}

/// The first child element of `node` named `tag`.
pub(crate) fn child_element<'a, 'input>(
    node: Node<'a, 'input>,
    tag: &str,
) -> Option<Node<'a, 'input>> {
    node.children()
        .find(|child| child.is_element() && child.tag_name().name() == tag)
}

/// Every child element of `node` named `tag`, in document order.
pub(crate) fn child_elements<'a, 'input>(
    node: Node<'a, 'input>,
    tag: &str,
) -> Vec<Node<'a, 'input>> {
    node.children()
        .filter(|child| child.is_element() && child.tag_name().name() == tag)
        .collect()
}

/// Every element below `node` that has no element children, in document order.
pub(crate) fn leaf_descendants<'a, 'input>(node: Node<'a, 'input>) -> Vec<Node<'a, 'input>> {
    node.descendants()
        .skip(1)
        .filter(|candidate| candidate.is_element() && !candidate.children().any(|c| c.is_element()))
        .collect()
}

/// The text content of an element: its text children joined in order, with
/// surrounding whitespace removed. An empty or self-closing element yields an
/// empty string.
pub(crate) fn leaf_text(node: Node) -> String {
    let mut text = String::new();
    for child in node.children() {
        if child.is_text() {
            if let Some(part) = child.text() {
                text.push_str(part);
            }
        }
    }
    text.trim().to_string()
}

/// The text of a `COMMENT` element: its text children joined in order, with
/// trailing whitespace removed. Leading whitespace belongs to the comment, as
/// it does in KVN (CCSDS 502.0-B-3 7.8.5), and trailing whitespace does not
/// (7.4.7).
pub(crate) fn comment_text(node: Node) -> String {
    let mut text = String::new();
    for child in node.children() {
        if child.is_text() {
            if let Some(part) = child.text() {
                text.push_str(part);
            }
        }
    }
    text.truncate(text.trim_end().len());
    text
}

/// `parent/child` naming the first element inside `node`, or `None` when it
/// holds only text. A reader that takes an element's text refuses one holding
/// an element, whose content it would otherwise not read.
pub(crate) fn nested_element(node: Node) -> Option<String> {
    node.children()
        .find(Node::is_element)
        .map(|nested| format!("{}/{}", node.tag_name().name(), nested.tag_name().name()))
}

/// Whether an element holds a value: non-whitespace text anywhere below it or
/// a child element. An empty element such as `<FOO/>` holds none.
pub(crate) fn carries_data(node: Node) -> bool {
    node.descendants().skip(1).any(|descendant| {
        descendant.is_element()
            || (descendant.is_text()
                && descendant
                    .text()
                    .is_some_and(|text| !text.trim().is_empty()))
    })
}

/// The `units` attribute of an element, when present.
pub(crate) fn units_attribute<'a>(node: Node<'a, '_>) -> Option<&'a str> {
    node.attribute("units")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_elements_skips_nested_same_name_elements() {
        let doc =
            Document::parse("<ndm><omm><x><omm/></x></omm><opm/><omm version=\"3.0\"/></ndm>")
                .expect("well-formed XML");
        let messages = message_elements(&doc, "omm");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].attribute("version"), Some("3.0"));
    }

    #[test]
    fn leaf_helpers_read_text_and_units() {
        let doc = Document::parse("<data><X units=\"km\"> 1.5 </X><block><Y/></block></data>")
            .expect("well-formed XML");
        let leaves = leaf_descendants(doc.root_element());
        let names: Vec<&str> = leaves.iter().map(|n| n.tag_name().name()).collect();
        assert_eq!(names, vec!["X", "Y"]);
        assert_eq!(leaf_text(leaves[0]), "1.5");
        assert_eq!(units_attribute(leaves[0]), Some("km"));
        assert_eq!(leaf_text(leaves[1]), "");
        assert_eq!(element_children(doc.root_element()).len(), 2);
        assert!(child_element(doc.root_element(), "block").is_some());
        assert_eq!(child_elements(doc.root_element(), "X").len(), 1);
    }

    #[test]
    fn comment_text_keeps_leading_whitespace_and_carries_data_sees_values() {
        let doc = Document::parse("<r><COMMENT>  indented  </COMMENT><E/><F> </F><G>1</G></r>")
            .expect("well-formed XML");
        let children = element_children(doc.root_element());
        assert_eq!(comment_text(children[0]), "  indented");
        assert!(!carries_data(children[1]));
        assert!(!carries_data(children[2]));
        assert!(carries_data(children[3]));
    }
}
