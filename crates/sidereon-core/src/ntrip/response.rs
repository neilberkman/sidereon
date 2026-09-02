#[derive(Clone, Debug, PartialEq, Eq)]
/// Rejection outcomes produced while interpreting an NTRIP response.
pub enum NtripRejection {
    /// `NtripClientMachine` emits this for a password-related `ERROR -` reason or an HTTP 401
    /// whose challenges are absent or include a non-Digest scheme.
    Unauthorized,
    /// [`classify_http_response`] returns this for HTTP status 404.
    MountpointNotFound,
    /// [`classify_http_response`] returns this for HTTP 401 when at least one challenge is
    /// present and every detected scheme is Digest.
    DigestRequired,
    /// The caster returned an `ERROR -` line that did not identify a password failure.
    CasterError {
        /// Text returned after the `ERROR - ` prefix.
        reason: String,
    },
    /// A successful response used a media type this client does not accept.
    UnexpectedContentType {
        /// The normalized media type from the first `Content-Type` header.
        content_type: String,
    },
    /// The machine received a non-200 ICY or SOURCETABLE status, or an HTTP status other than
    /// 200, 401, and 404.
    HttpError {
        /// Numeric status parsed from the response status line.
        status: u16,
        /// Trimmed reason phrase parsed from the response status line.
        reason: String,
    },
    /// The status line was unrecognized, or a handshake, header block, or sourcetable input
    /// exceeded its byte limit.
    MalformedHandshake {
        /// Up to 256 bytes retained from the malformed or overlong handshake input.
        prefix: Vec<u8>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Result returned after the response status and headers are classified by
/// [`classify_http_response`].
pub enum HttpClassification {
    /// A `gnss/data` response, or a response without `Content-Type`, carries stream payloads.
    Stream {
        /// Whether the body uses a comma-separated `Transfer-Encoding: chunked` token.
        chunked: bool,
    },
    /// HTTP 200 with normalized media type `gnss/sourcetable` carries sourcetable text.
    Sourcetable {
        /// Whether the body uses a comma-separated `Transfer-Encoding: chunked` token.
        chunked: bool,
    },
    /// The response must be closed with the given [`NtripRejection`].
    Rejection(NtripRejection),
}

/// Classifies an HTTP NTRIP response from its status, reason phrase, and headers.
///
/// HTTP 401 becomes [`NtripRejection::DigestRequired`] when at least one
/// `WWW-Authenticate` scheme is present and every scheme is Digest; otherwise it
/// becomes [`NtripRejection::Unauthorized`]. HTTP 404 becomes
/// [`NtripRejection::MountpointNotFound`], and other non-200 statuses become
/// [`NtripRejection::HttpError`]. A 200 response is a [`HttpClassification::Stream`]
/// for no media type or `gnss/data`, a [`HttpClassification::Sourcetable`] for
/// `gnss/sourcetable`, and an [`NtripRejection::UnexpectedContentType`] otherwise.
/// The stream and sourcetable variants report whether a comma-separated
/// `Transfer-Encoding` token is `chunked`, without regard to ASCII case.
pub fn classify_http_response(
    status: u16,
    reason: &str,
    headers: &[(String, String)],
) -> HttpClassification {
    if status == 401 {
        if digest_required(headers) {
            return HttpClassification::Rejection(NtripRejection::DigestRequired);
        }
        return HttpClassification::Rejection(NtripRejection::Unauthorized);
    }
    if status == 404 {
        return HttpClassification::Rejection(NtripRejection::MountpointNotFound);
    }
    if status != 200 {
        return HttpClassification::Rejection(NtripRejection::HttpError {
            status,
            reason: reason.to_string(),
        });
    }

    let content_type = header_value(headers, "Content-Type").map(media_type);
    match content_type.as_deref() {
        None | Some("gnss/data") => HttpClassification::Stream {
            chunked: transfer_is_chunked(headers),
        },
        Some("gnss/sourcetable") => HttpClassification::Sourcetable {
            chunked: transfer_is_chunked(headers),
        },
        Some(other) => HttpClassification::Rejection(NtripRejection::UnexpectedContentType {
            content_type: other.to_string(),
        }),
    }
}

pub(crate) fn transfer_is_chunked(headers: &[(String, String)]) -> bool {
    header_values(headers, "Transfer-Encoding").any(|value| {
        value
            .split(',')
            .any(|token| token.trim().eq_ignore_ascii_case("chunked"))
    })
}

fn digest_required(headers: &[(String, String)]) -> bool {
    let mut saw = false;
    for value in header_values(headers, "WWW-Authenticate") {
        for challenge in value.split(',') {
            let challenge = challenge.trim_start();
            let Some(scheme) = challenge.split_whitespace().next() else {
                continue;
            };
            if scheme.contains('=') {
                continue;
            }
            saw = true;
            if !scheme.eq_ignore_ascii_case("digest") {
                return false;
            }
        }
    }
    saw
}

fn header_value(headers: &[(String, String)], name: &str) -> Option<String> {
    header_values(headers, name).next().map(str::to_string)
}

fn header_values<'a>(
    headers: &'a [(String, String)],
    name: &'a str,
) -> impl Iterator<Item = &'a str> + 'a {
    headers
        .iter()
        .filter(move |(n, _)| n.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

fn media_type(value: String) -> String {
    value
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
}
