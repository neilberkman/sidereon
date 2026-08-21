use crate::ntrip::chunk::ChunkedDecoder;
use crate::ntrip::gga::{format_gga, GgaPosition};
use crate::ntrip::request::{NtripConfig, NtripVersion};
use crate::ntrip::response::{classify_http_response, HttpClassification, NtripRejection};
use crate::ntrip::sourcetable::{parse_sourcetable, Sourcetable};
use crate::Result;

const MAX_LINE: usize = 8 * 1024;
const MAX_HEADER_BLOCK: usize = 64 * 1024;
const MAX_SOURCETABLE: usize = 4 * 1024 * 1024;
const PREFIX_LIMIT: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NtripState {
    Idle,
    AwaitingStatus,
    AwaitingHeaders,
    Streaming,
    Sourcetable,
    Closed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NtripHandshake {
    pub version: NtripVersion,
    pub chunked: bool,
    pub headers: Vec<(String, String)>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum NtripEvent {
    Connected(NtripHandshake),
    Payload(Vec<u8>),
    Sourcetable(Sourcetable),
    Rejected(NtripRejection),
    StreamCorrupted { detail: String },
    StreamEnded,
}

#[derive(Clone, Debug)]
pub struct NtripClientMachine {
    config: NtripConfig,
    state: NtripState,
    carry: Vec<u8>,
    headers: Vec<(String, String)>,
    header_bytes: usize,
    status_version: Option<NtripVersion>,
    http_status: Option<u16>,
    http_reason: String,
    chunked: bool,
    pending_icy_blank_line: bool,
    sourcetable_chunked: bool,
    chunk_decoder: ChunkedDecoder,
    sourcetable_text: String,
    sourcetable_carry: Vec<u8>,
    sourcetable_records_started: bool,
    last_gga_s: Option<f64>,
}

impl NtripClientMachine {
    pub fn new(config: NtripConfig) -> Self {
        Self {
            config,
            state: NtripState::Idle,
            carry: Vec::new(),
            headers: Vec::new(),
            header_bytes: 0,
            status_version: None,
            http_status: None,
            http_reason: String::new(),
            chunked: false,
            pending_icy_blank_line: false,
            sourcetable_chunked: false,
            chunk_decoder: ChunkedDecoder::new(),
            sourcetable_text: String::new(),
            sourcetable_carry: Vec::new(),
            sourcetable_records_started: false,
            last_gga_s: None,
        }
    }

    pub fn connection_request(&mut self) -> Result<Vec<u8>> {
        let bytes = self.config.request_bytes()?;
        self.state = NtripState::AwaitingStatus;
        Ok(bytes)
    }

    pub fn push(&mut self, bytes: &[u8]) -> Vec<NtripEvent> {
        let mut events = Vec::new();
        if matches!(self.state, NtripState::Closed) {
            return events;
        }
        self.carry.extend_from_slice(bytes);

        loop {
            match self.state {
                NtripState::Idle => {
                    self.state = NtripState::AwaitingStatus;
                }
                NtripState::AwaitingStatus => {
                    if !self.parse_status(&mut events) {
                        break;
                    }
                }
                NtripState::AwaitingHeaders => {
                    if !self.parse_headers(&mut events) {
                        break;
                    }
                }
                NtripState::Streaming => {
                    self.drain_payload(&mut events);
                    break;
                }
                NtripState::Sourcetable => {
                    if !self.drain_sourcetable(&mut events) {
                        break;
                    }
                }
                NtripState::Closed => break,
            }
        }

        events
    }

    pub fn gga_message(
        &mut self,
        now_s: f64,
        position: &GgaPosition,
        utc_seconds_of_day: f64,
    ) -> Option<Vec<u8>> {
        self.try_gga_message(now_s, position, utc_seconds_of_day)
            .ok()
            .flatten()
    }

    pub fn try_gga_message(
        &mut self,
        now_s: f64,
        position: &GgaPosition,
        utc_seconds_of_day: f64,
    ) -> Result<Option<Vec<u8>>> {
        if self.state != NtripState::Streaming {
            return Ok(None);
        }
        let Some(interval) = self.config.gga_interval_s else {
            return Ok(None);
        };
        if !now_s.is_finite() {
            return Ok(None);
        }
        let due = match self.last_gga_s {
            None => true,
            Some(last) if now_s >= last => now_s - last >= interval,
            Some(_) => false,
        };
        if !due {
            return Ok(None);
        }
        let bytes = format_gga(position, utc_seconds_of_day)?;
        self.last_gga_s = Some(now_s);
        Ok(Some(bytes))
    }

    pub fn state(&self) -> NtripState {
        self.state
    }

    pub fn reset(&mut self) {
        let config = self.config.clone();
        *self = Self::new(config);
    }

    fn parse_status(&mut self, events: &mut Vec<NtripEvent>) -> bool {
        let Some(line) = self.take_line_or_reject(events) else {
            return false;
        };
        let text = String::from_utf8_lossy(&line).trim().to_string();

        if let Some(rest) = text.strip_prefix("ERROR - ") {
            let reason = rest.to_string();
            let rejection = if reason.to_ascii_lowercase().contains("password") {
                NtripRejection::Unauthorized
            } else {
                NtripRejection::CasterError { reason }
            };
            self.reject(events, rejection);
            return true;
        }

        if let Some((status, reason)) = parse_prefixed_status(&text, "ICY") {
            if status == 200 {
                self.state = NtripState::Streaming;
                self.status_version = Some(NtripVersion::Rev1);
                self.chunked = false;
                self.pending_icy_blank_line = true;
                events.push(NtripEvent::Connected(NtripHandshake {
                    version: NtripVersion::Rev1,
                    chunked: false,
                    headers: Vec::new(),
                }));
                return true;
            }
            self.reject(
                events,
                NtripRejection::HttpError {
                    status,
                    reason: reason.to_string(),
                },
            );
            return true;
        }

        if let Some((status, reason)) = parse_prefixed_status(&text, "SOURCETABLE") {
            if status == 200 {
                self.state = NtripState::Sourcetable;
                self.status_version = Some(NtripVersion::Rev1);
                return true;
            }
            self.reject(
                events,
                NtripRejection::HttpError {
                    status,
                    reason: reason.to_string(),
                },
            );
            return true;
        }

        if let Some((version, status, reason)) = parse_http_status(&text) {
            self.status_version = Some(version);
            self.headers.clear();
            self.header_bytes = 0;
            self.http_status = Some(status);
            self.http_reason = reason.to_string();
            self.state = NtripState::AwaitingHeaders;

            return true;
        }

        self.reject(
            events,
            NtripRejection::MalformedHandshake {
                prefix: self.prefix_with_line(&line),
            },
        );
        true
    }

    fn parse_headers(&mut self, events: &mut Vec<NtripEvent>) -> bool {
        let Some(line) = self.take_line_or_reject(events) else {
            return false;
        };
        if line.is_empty() {
            let status = self.http_status.unwrap_or(0);
            match classify_http_response(status, &self.http_reason, &self.headers) {
                HttpClassification::Stream { chunked } => {
                    self.chunked = chunked;
                    self.state = NtripState::Streaming;
                    events.push(NtripEvent::Connected(NtripHandshake {
                        version: self.status_version.unwrap_or(NtripVersion::Rev2),
                        chunked,
                        headers: self.headers.clone(),
                    }));
                    true
                }
                HttpClassification::Sourcetable { chunked } => {
                    self.state = NtripState::Sourcetable;
                    self.sourcetable_chunked = chunked;
                    self.sourcetable_records_started = true;
                    true
                }
                HttpClassification::Rejection(rejection) => {
                    self.reject(events, rejection);
                    true
                }
            }
        } else {
            self.header_bytes += line.len();
            if self.header_bytes > MAX_HEADER_BLOCK {
                self.reject_current_prefix(events);
                return true;
            }
            if let Some((name, value)) = split_header(&line) {
                self.headers.push((name, value));
            }
            true
        }
    }

    fn drain_payload(&mut self, events: &mut Vec<NtripEvent>) {
        if self.pending_icy_blank_line && !self.consume_pending_icy_blank_line() {
            return;
        }
        if self.carry.is_empty() {
            return;
        }
        let bytes: Vec<u8> = std::mem::take(&mut self.carry);
        if self.chunked {
            match self.chunk_decoder.push(&bytes) {
                Ok(payload) => {
                    if !payload.is_empty() {
                        events.push(NtripEvent::Payload(payload));
                    }
                    if self.chunk_decoder.finished() {
                        self.state = NtripState::Closed;
                        events.push(NtripEvent::StreamEnded);
                    }
                }
                Err(err) => {
                    self.state = NtripState::Closed;
                    events.push(NtripEvent::StreamCorrupted {
                        detail: err.to_string(),
                    });
                }
            }
        } else {
            events.push(NtripEvent::Payload(bytes));
        }
    }

    fn drain_sourcetable(&mut self, events: &mut Vec<NtripEvent>) -> bool {
        if self.sourcetable_chunked {
            if !self.carry.is_empty() {
                let bytes: Vec<u8> = std::mem::take(&mut self.carry);
                match self.chunk_decoder.push(&bytes) {
                    Ok(decoded) => self.sourcetable_carry.extend_from_slice(&decoded),
                    Err(err) => {
                        self.state = NtripState::Closed;
                        events.push(NtripEvent::StreamCorrupted {
                            detail: err.to_string(),
                        });
                        return true;
                    }
                }
            }
            if self.drain_sourcetable_lines(events) {
                return true;
            }
            if self.chunk_decoder.finished() {
                if !self.sourcetable_carry.is_empty() {
                    let line: Vec<u8> = std::mem::take(&mut self.sourcetable_carry);
                    if self.push_sourcetable_line(&line, events) {
                        return true;
                    }
                }
                self.finish_sourcetable(events);
                return true;
            }
            false
        } else {
            self.drain_sourcetable_lines(events)
        }
    }

    pub fn finish(&mut self) -> Vec<NtripEvent> {
        let mut events = Vec::new();
        if self.state == NtripState::Sourcetable {
            if self.sourcetable_chunked {
                if !self.sourcetable_carry.is_empty() {
                    let line: Vec<u8> = std::mem::take(&mut self.sourcetable_carry);
                    self.push_sourcetable_line(&line, &mut events);
                }
            } else if !self.carry.is_empty() {
                let line: Vec<u8> = std::mem::take(&mut self.carry);
                self.push_sourcetable_line(&line, &mut events);
            }
            self.finish_sourcetable(&mut events);
        }
        events
    }

    fn drain_sourcetable_lines(&mut self, events: &mut Vec<NtripEvent>) -> bool {
        loop {
            let line = if self.sourcetable_chunked {
                take_line_from(&mut self.sourcetable_carry)
            } else {
                take_line_from(&mut self.carry)
            };
            let Some(line) = line else {
                let len = if self.sourcetable_chunked {
                    self.sourcetable_carry.len()
                } else {
                    self.carry.len()
                };
                if len > MAX_LINE {
                    self.reject_current_prefix(events);
                    return true;
                }
                return false;
            };

            if self.push_sourcetable_line(&line, events) {
                return true;
            }
        }
    }

    fn push_sourcetable_line(&mut self, line: &[u8], events: &mut Vec<NtripEvent>) -> bool {
        let text = String::from_utf8_lossy(line).to_string();
        let first = text.split(';').next().unwrap_or("").trim();

        if !self.sourcetable_records_started {
            if text.is_empty() {
                return false;
            }
            if is_sourcetable_record_start(first) {
                self.sourcetable_records_started = true;
            } else if text.contains(':') {
                return false;
            } else {
                self.sourcetable_records_started = true;
            }
        }

        self.sourcetable_text.push_str(&text);
        self.sourcetable_text.push_str("\r\n");
        if self.sourcetable_text.len() > MAX_SOURCETABLE {
            self.reject_current_prefix(events);
            return true;
        }
        if first.eq_ignore_ascii_case("ENDSOURCETABLE") {
            self.finish_sourcetable(events);
            return true;
        }
        false
    }

    fn finish_sourcetable(&mut self, events: &mut Vec<NtripEvent>) {
        match parse_sourcetable(&self.sourcetable_text) {
            Ok(table) => events.push(NtripEvent::Sourcetable(table)),
            Err(err) => events.push(NtripEvent::StreamCorrupted {
                detail: err.to_string(),
            }),
        }
        self.state = NtripState::Closed;
    }

    fn take_line_or_reject(&mut self, events: &mut Vec<NtripEvent>) -> Option<Vec<u8>> {
        if let Some(pos) = self.carry.iter().position(|&b| b == b'\n') {
            let mut line: Vec<u8> = self.carry.drain(..=pos).collect();
            if line.ends_with(b"\n") {
                line.pop();
            }
            if line.ends_with(b"\r") {
                line.pop();
            }
            Some(line)
        } else if self.carry.len() > MAX_LINE {
            self.reject_current_prefix(events);
            None
        } else {
            None
        }
    }

    fn consume_pending_icy_blank_line(&mut self) -> bool {
        if self.carry.is_empty() {
            return false;
        }
        if self.carry[0] == b'\n' {
            self.carry.drain(..1);
            self.pending_icy_blank_line = false;
            return true;
        }
        if self.carry[0] == b'\r' {
            if self.carry.len() == 1 {
                return false;
            }
            if self.carry[1] == b'\n' {
                self.carry.drain(..2);
            }
            self.pending_icy_blank_line = false;
            return true;
        }
        self.pending_icy_blank_line = false;
        true
    }

    fn reject_current_prefix(&mut self, events: &mut Vec<NtripEvent>) {
        let prefix = self.carry.iter().copied().take(PREFIX_LIMIT).collect();
        self.reject(events, NtripRejection::MalformedHandshake { prefix });
    }

    fn reject(&mut self, events: &mut Vec<NtripEvent>, rejection: NtripRejection) {
        self.state = NtripState::Closed;
        events.push(NtripEvent::Rejected(rejection));
    }

    fn prefix_with_line(&self, line: &[u8]) -> Vec<u8> {
        let mut prefix = line.to_vec();
        prefix.extend_from_slice(&self.carry);
        prefix.truncate(PREFIX_LIMIT);
        prefix
    }
}

fn parse_prefixed_status<'a>(text: &'a str, prefix: &str) -> Option<(u16, &'a str)> {
    let rest = text.strip_prefix(prefix)?.trim_start();
    let mut parts = rest.splitn(2, char::is_whitespace);
    let status = parts.next()?.parse().ok()?;
    let reason = parts.next().unwrap_or("").trim();
    Some((status, reason))
}

fn parse_http_status(text: &str) -> Option<(NtripVersion, u16, &str)> {
    let rest = if let Some(rest) = text.strip_prefix("HTTP/1.1 ") {
        (NtripVersion::Rev2, rest)
    } else {
        let rest = text.strip_prefix("HTTP/1.0 ")?;
        (NtripVersion::Rev1, rest)
    };
    let mut parts = rest.1.splitn(2, char::is_whitespace);
    let status = parts.next()?.parse().ok()?;
    let reason = parts.next().unwrap_or("").trim();
    Some((rest.0, status, reason))
}

fn split_header(line: &[u8]) -> Option<(String, String)> {
    let pos = line.iter().position(|&b| b == b':')?;
    let name = String::from_utf8_lossy(&line[..pos]).trim().to_string();
    let value_bytes = if line.get(pos + 1) == Some(&b' ') {
        &line[pos + 2..]
    } else {
        &line[pos + 1..]
    };
    let value = String::from_utf8_lossy(value_bytes).to_string();
    Some((name, value))
}

fn take_line_from(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    let pos = buffer.iter().position(|&b| b == b'\n')?;
    let mut line: Vec<u8> = buffer.drain(..=pos).collect();
    if line.ends_with(b"\n") {
        line.pop();
    }
    if line.ends_with(b"\r") {
        line.pop();
    }
    Some(line)
}

fn is_sourcetable_record_start(first: &str) -> bool {
    ["STR", "CAS", "NET", "ENDSOURCETABLE"]
        .iter()
        .any(|tag| first.eq_ignore_ascii_case(tag))
}
