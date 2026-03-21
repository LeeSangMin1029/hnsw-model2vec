//! Sync LSP transport — Content-Length framed reader/writer over `Read`/`Write`.
//!
//! Ported from lspmux (EUPL-1.2, p2502/lspmux) — adapted from async to sync I/O.

use std::io::{self, BufRead, Write};

use crate::error::{LspError, Result};
use crate::lsp::jsonrpc::Message;

/// Reads LSP messages from a `BufRead` source (e.g., RA stdout).
pub struct LspReader<R> {
    reader: R,
    buffer: Vec<u8>,
}

impl<R: BufRead> LspReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: Vec::with_capacity(4096),
        }
    }

    /// Read the Content-Length header, returning the body size.
    fn read_content_length(&mut self) -> Result<Option<usize>> {
        let mut content_length = None;

        loop {
            self.buffer.clear();
            let n = self.reader.read_until(b'\n', &mut self.buffer)?;
            if n == 0 {
                return Ok(None); // EOF
            }

            let line = self.buffer.strip_suffix(b"\r\n").ok_or_else(|| {
                LspError::Protocol("malformed header: missing \\r\\n".into())
            })?;

            if line.is_empty() {
                // Empty line separates headers from body.
                break;
            }

            let text = std::str::from_utf8(line).map_err(|e| {
                LspError::Protocol(format!("header not UTF-8: {e}"))
            })?;

            if let Some((name, value)) = text.split_once(": ") {
                if name.eq_ignore_ascii_case("content-length") {
                    content_length = Some(value.parse::<usize>().map_err(|e| {
                        LspError::Protocol(format!("bad content-length: {e}"))
                    })?);
                }
                // Ignore content-type and other headers.
            }
        }

        content_length
            .map(Some)
            .ok_or_else(|| LspError::Protocol("missing Content-Length header".into()))
    }

    /// Read one LSP message. Returns `None` on EOF.
    pub fn read_message(&mut self) -> Result<Option<Message>> {
        let Some(len) = self.read_content_length()? else {
            return Ok(None);
        };

        self.buffer.clear();
        self.buffer.resize(len, 0);
        self.reader.read_exact(&mut self.buffer)?;

        let msg: Message = serde_json::from_slice(&self.buffer)?;
        Ok(Some(msg))
    }
}

/// Writes LSP messages to a `Write` sink (e.g., RA stdin).
pub struct LspWriter<W> {
    writer: W,
    buffer: Vec<u8>,
}

impl<W: Write> LspWriter<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            buffer: Vec::with_capacity(4096),
        }
    }

    /// Serialize and write one LSP message with Content-Length header.
    pub fn write_message(&mut self, message: &Message) -> io::Result<()> {
        self.buffer.clear();
        serde_json::to_writer(&mut self.buffer, message)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        write!(self.writer, "Content-Length: {}\r\n\r\n", self.buffer.len())?;
        self.writer.write_all(&self.buffer)?;
        self.writer.flush()
    }
}
