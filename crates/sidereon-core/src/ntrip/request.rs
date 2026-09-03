use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Selects the request and response revision used by the NTRIP client.
/// Request construction maps the variants to HTTP/1.0 and HTTP/1.1, while response parsing maps ICY and HTTP/1.0 to [`NtripVersion::Rev1`] and HTTP/1.1 to [`NtripVersion::Rev2`].
pub enum NtripVersion {
    /// Produces an HTTP/1.0 request and is reported for ICY or HTTP/1.0 responses.
    Rev1,
    /// Produces an HTTP/1.1 request with the Rev2-only request headers and is reported for HTTP/1.1 responses.
    Rev2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Credentials encoded into the `Basic` authorization header of an NTRIP request.
/// The request builder joins the two values with `:`, Base64-encodes the result, and rejects a username containing `:` or either value containing CR/LF.
pub struct NtripCredentials {
    /// Text placed before `:` when the request builder constructs the encoded authorization value.
    /// This value must not contain `:` or CR/LF.
    pub username: String,
    /// Text placed after `:` when the request builder constructs the encoded authorization value.
    /// CR/LF in this value is rejected.
    pub password: String,
}

#[derive(Clone, Debug, PartialEq)]
/// Inputs retained by the NTRIP client machine for request construction and optional GGA pacing.
/// The default uses port 2101 and [`NtripVersion::Rev2`], leaves the host and mountpoint empty, uses `sidereon/<package version>` as the product, and disables credentials and GGA pacing.
pub struct NtripConfig {
    /// Caster host text, paired with [`NtripConfig::port`] in the Rev2 `Host` header.
    /// CR/LF in this value causes request validation to fail.
    pub host: String,
    /// Caster port rendered in decimal after [`NtripConfig::host`] in the Rev2 `Host` header.
    /// Rev1 request headers do not use this value.
    pub port: u16,
    /// Mountpoint appended to the request path; an empty value produces `/`.
    /// ASCII controls, ASCII whitespace, `/`, and `?` in this value cause request validation to fail.
    pub mountpoint: String,
    /// Request revision selected by the request builder and retained by the client machine.
    pub version: NtripVersion,
    /// Optional credentials that add a Base64 `Basic` authorization header when present.
    /// Invalid username punctuation or CR/LF in either credential causes request construction to fail.
    pub credentials: Option<NtripCredentials>,
    /// Product text placed after `NTRIP ` in the `User-Agent` header.
    /// It must be nonempty, contain exactly one `/`, and contain no ASCII control or whitespace byte.
    pub user_agent_product: String,
    /// Optional positive, finite interval in seconds used to pace GGA messages while streaming.
    /// `None` disables GGA output; with `Some`, the first message is immediately due and later messages require at least this interval since the last one.
    pub gga_interval_s: Option<f64>,
}

impl Default for NtripConfig {
    fn default() -> Self {
        Self {
            host: String::new(),
            port: 2101,
            mountpoint: String::new(),
            version: NtripVersion::Rev2,
            credentials: None,
            user_agent_product: format!("sidereon/{}", env!("CARGO_PKG_VERSION")),
            gga_interval_s: None,
        }
    }
}

impl NtripConfig {
    /// Validates this configuration and returns the complete CRLF-terminated GET request.
    /// The request uses HTTP/1.0 and Rev1 headers for [`NtripVersion::Rev1`] or HTTP/1.1 and Rev2 headers for [`NtripVersion::Rev2`], followed by a final blank line; invalid configuration returns an error.
    pub fn request_bytes(&self) -> Result<Vec<u8>> {
        let path = self.validated_path()?;
        let headers = self.common_headers()?;
        let mut out = Vec::new();
        match self.version {
            NtripVersion::Rev1 => {
                write_line(&mut out, &format!("GET {path} HTTP/1.0"));
                for (name, value) in headers {
                    write_line(&mut out, &format!("{name}: {value}"));
                }
            }
            NtripVersion::Rev2 => {
                write_line(&mut out, &format!("GET {path} HTTP/1.1"));
                for (name, value) in headers {
                    write_line(&mut out, &format!("{name}: {value}"));
                }
            }
        }
        out.extend_from_slice(b"\r\n");
        Ok(out)
    }

    /// Returns the validated request path and header pairs for an NTRIP Rev2 request.
    /// Rev1 returns an invalid-input error because this helper is only defined for [`NtripVersion::Rev2`].
    pub fn request_headers(&self) -> Result<(String, Vec<(String, String)>)> {
        if self.version != NtripVersion::Rev2 {
            return Err(Error::InvalidInput(
                "request_headers is only defined for NTRIP rev2".into(),
            ));
        }
        Ok((self.validated_path()?, self.common_headers()?))
    }

    fn validated_path(&self) -> Result<String> {
        validate_config(self)?;
        if self.mountpoint.is_empty() {
            Ok("/".into())
        } else {
            Ok(format!("/{}", self.mountpoint))
        }
    }

    fn common_headers(&self) -> Result<Vec<(String, String)>> {
        validate_config(self)?;
        let mut headers = Vec::new();
        if self.version == NtripVersion::Rev2 {
            headers.push(("Host".into(), format!("{}:{}", self.host, self.port)));
            headers.push(("Ntrip-Version".into(), "Ntrip/2.0".into()));
        }
        headers.push((
            "User-Agent".into(),
            format!("NTRIP {}", self.user_agent_product),
        ));
        if let Some(credentials) = &self.credentials {
            let token = format!("{}:{}", credentials.username, credentials.password);
            headers.push((
                "Authorization".into(),
                format!("Basic {}", base64(token.as_bytes())),
            ));
        }
        if self.version == NtripVersion::Rev2 {
            headers.push(("Connection".into(), "close".into()));
        }
        Ok(headers)
    }
}

fn validate_config(config: &NtripConfig) -> Result<()> {
    if config.host.bytes().any(|b| b == b'\r' || b == b'\n') {
        return Err(Error::InvalidInput(
            "NTRIP host must not contain CR or LF".into(),
        ));
    }

    if config
        .mountpoint
        .bytes()
        .any(|b| b.is_ascii_control() || b.is_ascii_whitespace() || b == b'/' || b == b'?')
    {
        return Err(Error::InvalidInput(
            "NTRIP mountpoint contains a forbidden byte".into(),
        ));
    }

    let product = &config.user_agent_product;
    let slash_count = product.bytes().filter(|&b| b == b'/').count();
    if product.is_empty()
        || slash_count != 1
        || product
            .bytes()
            .any(|b| b.is_ascii_control() || b.is_ascii_whitespace())
    {
        return Err(Error::InvalidInput(
            "user_agent_product must be name/version with no whitespace".into(),
        ));
    }

    if let Some(credentials) = &config.credentials {
        if credentials.username.contains(':') {
            return Err(Error::InvalidInput(
                "NTRIP username must not contain ':'".into(),
            ));
        }
        if credentials
            .username
            .bytes()
            .chain(credentials.password.bytes())
            .any(|b| b == b'\r' || b == b'\n')
        {
            return Err(Error::InvalidInput(
                "NTRIP credentials must not contain CR or LF".into(),
            ));
        }
    }

    if let Some(interval) = config.gga_interval_s {
        if !interval.is_finite() || interval <= 0.0 {
            return Err(Error::InvalidInput(
                "gga_interval_s must be finite and positive".into(),
            ));
        }
    }

    Ok(())
}

fn write_line(out: &mut Vec<u8>, line: &str) {
    out.extend_from_slice(line.as_bytes());
    out.extend_from_slice(b"\r\n");
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        let n = ((u32::from(b0)) << 16) | ((u32::from(b1)) << 8) | u32::from(b2);
        out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[((n >> 6) & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(n & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}
