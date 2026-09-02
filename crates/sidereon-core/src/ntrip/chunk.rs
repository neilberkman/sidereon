use crate::{Error, Result};

const MAX_CHUNK_SIZE: usize = 16 * 1024 * 1024;
const MAX_LINE: usize = 8 * 1024;
const MAX_HEX_DIGITS: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
/// Incremental decoder for an HTTP chunked response body.
///
/// It returns data bytes as complete chunks become available, including when size lines or data span multiple [`Self::push`] calls. It accepts optional chunk extensions, LF or CRLF data terminators, and trailer lines after the zero-size chunk; lines longer than 8 KiB, size fields with more than 8 hexadecimal digits, chunks larger than 16 MiB, and oversized trailer sections produce a parse error.
pub struct ChunkedDecoder {
    buf: Vec<u8>,
    state: ChunkState,
    finished: bool,
    poison: Option<String>,
    trailer_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ChunkState {
    Size,
    Data(usize),
    DataLineEnd,
    Trailer,
}

impl Default for ChunkedDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkedDecoder {
    /// Creates an empty decoder ready to read a chunk-size line.
    ///
    /// The input buffer, completion flag, stored parse error, and trailer length start empty, false, absent, and zero, respectively.
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            state: ChunkState::Size,
            finished: false,
            poison: None,
            trailer_len: 0,
        }
    }

    /// Feeds one fragment of a chunked response body and returns decoded data bytes.
    ///
    /// The fragment may end in the middle of a size line, chunk body, data terminator, or trailer; those bytes remain buffered. After the zero-size chunk and its terminating empty trailer line are consumed, later calls return an empty vector; malformed input returns [`Error::Parse`], and later calls return the same stored parse error.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<u8>> {
        if let Some(message) = &self.poison {
            return Err(Error::Parse(message.clone()));
        }
        if self.finished {
            return Ok(Vec::new());
        }
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();

        loop {
            match self.state {
                ChunkState::Size => {
                    let Some(line) = self.read_line()? else {
                        break;
                    };
                    let size = match parse_size_line(&line) {
                        Ok(size) => size,
                        Err(err) => return self.poison_from(err),
                    };
                    if size == 0 {
                        self.state = ChunkState::Trailer;
                    } else {
                        self.state = ChunkState::Data(size);
                    }
                }
                ChunkState::Data(remaining) => {
                    if self.buf.is_empty() {
                        break;
                    }
                    let n = remaining.min(self.buf.len());
                    out.extend_from_slice(&self.buf[..n]);
                    self.buf.drain(..n);
                    let left = remaining - n;
                    self.state = if left == 0 {
                        ChunkState::DataLineEnd
                    } else {
                        ChunkState::Data(left)
                    };
                }
                ChunkState::DataLineEnd => {
                    if self.buf.is_empty() {
                        break;
                    }
                    if self.buf[0] == b'\n' {
                        self.buf.drain(..1);
                        self.state = ChunkState::Size;
                    } else if self.buf[0] == b'\r' {
                        if self.buf.len() < 2 {
                            break;
                        }
                        if self.buf[1] != b'\n' {
                            return self.poison("malformed chunk data terminator");
                        }
                        self.buf.drain(..2);
                        self.state = ChunkState::Size;
                    } else {
                        return self.poison("malformed chunk data terminator");
                    }
                }
                ChunkState::Trailer => {
                    let Some(line) = self.read_line()? else {
                        break;
                    };
                    self.trailer_len += line.len();
                    if self.trailer_len > MAX_LINE {
                        return self.poison("chunk trailer section too long");
                    }
                    if line.is_empty() {
                        self.finished = true;
                        break;
                    }
                }
            }
        }

        Ok(out)
    }

    /// Reports whether the zero-size chunk and terminating empty trailer line have been consumed.
    ///
    /// [`crate::ntrip::NtripClientMachine`] uses this result to emit `StreamEnded` for a data response or finish a chunked sourcetable.
    pub fn finished(&self) -> bool {
        self.finished
    }

    /// Discards buffered response bytes and returns the decoder to the initial state.
    ///
    /// This clears completion, trailer length, and any stored parse error, matching [`Self::new`].
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    fn read_line(&mut self) -> Result<Option<Vec<u8>>> {
        if let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let mut line: Vec<u8> = self.buf.drain(..=pos).collect();
            if line.ends_with(b"\n") {
                line.pop();
            }
            if line.ends_with(b"\r") {
                line.pop();
            }
            return Ok(Some(line));
        }
        if self.buf.len() > MAX_LINE {
            return self.poison("chunk line too long");
        }
        Ok(None)
    }

    fn poison<T>(&mut self, message: &str) -> Result<T> {
        self.poison = Some(message.to_string());
        Err(Error::Parse(message.to_string()))
    }

    fn poison_from<T>(&mut self, err: Error) -> Result<T> {
        let message = err.to_string();
        self.poison = Some(message.clone());
        Err(Error::Parse(message))
    }
}

fn parse_size_line(line: &[u8]) -> Result<usize> {
    if line.len() > MAX_LINE {
        return Err(Error::Parse("chunk size line too long".into()));
    }
    let size_part = line.split(|&b| b == b';').next().unwrap_or(line);
    let size_part = trim_ascii(size_part);
    if size_part.is_empty() {
        return Err(Error::Parse("empty chunk size".into()));
    }
    if size_part.len() > MAX_HEX_DIGITS {
        return Err(Error::Parse("chunk size has too many hex digits".into()));
    }
    let mut size = 0usize;
    for &b in size_part {
        let digit = match b {
            b'0'..=b'9' => usize::from(b - b'0'),
            b'a'..=b'f' => usize::from(b - b'a' + 10),
            b'A'..=b'F' => usize::from(b - b'A' + 10),
            _ => return Err(Error::Parse("invalid chunk size digit".into())),
        };
        size = (size << 4) | digit;
    }
    if size > MAX_CHUNK_SIZE {
        return Err(Error::Parse("chunk size exceeds limit".into()));
    }
    Ok(size)
}

fn trim_ascii(bytes: &[u8]) -> &[u8] {
    let mut start = 0;
    let mut end = bytes.len();
    while start < end && bytes[start].is_ascii_whitespace() {
        start += 1;
    }
    while end > start && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    &bytes[start..end]
}
