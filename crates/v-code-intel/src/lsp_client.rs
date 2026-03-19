//! Lightweight LSP client for querying type information via lspmux.
//!
//! Connects to lspmux TCP server (default 127.0.0.1:27631), sends
//! `textDocument/hover` requests, and extracts type information from responses.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::time::Duration;

/// Result of a hover query — the type string if available.
pub type HoverResult = Option<String>;

/// LSP client connected to lspmux.
pub struct LspClient {
    stream: TcpStream,
    next_id: i64,
    initialized: bool,
}

/// Batch hover query: file path + (line, col) → type string.
#[derive(Debug, Clone)]
pub struct HoverQuery {
    pub file: String,
    pub line: u32,
    pub col: u32,
    pub var_name: String,
}

impl LspClient {
    /// Try to connect to lspmux. Returns None if server is not running.
    pub fn connect(project_root: &Path) -> Option<Self> {
        let stream = TcpStream::connect_timeout(
            &"127.0.0.1:27631".parse().ok()?,
            Duration::from_secs(3),
        ).ok()?;
        stream.set_read_timeout(Some(Duration::from_secs(30))).ok()?;
        stream.set_write_timeout(Some(Duration::from_secs(5))).ok()?;

        let mut client = Self {
            stream,
            next_id: 1,
            initialized: false,
        };

        if client.initialize(project_root).is_some() {
            Some(client)
        } else {
            None
        }
    }

    /// Send initialize + initialized handshake.
    fn initialize(&mut self, project_root: &Path) -> Option<()> {
        let uri = path_to_uri(project_root);
        let id = self.next_id();

        let init = format!(
            r#"{{"jsonrpc":"2.0","id":{id},"method":"initialize","params":{{"processId":null,"capabilities":{{}},"initializationOptions":{{"lspMux":{{"version":"1","method":"connect","server":"rust-analyzer","args":[]}}}},"workspaceFolders":[{{"uri":"{uri}","name":"project"}}],"rootUri":"{uri}"}}}}"#
        );

        self.send(&init)?;
        let _response = self.recv()?;

        // Send initialized notification
        let notif = r#"{"jsonrpc":"2.0","method":"initialized","params":{}}"#;
        self.send(notif)?;

        self.initialized = true;
        Some(())
    }

    /// Query hover for a single position. Returns the type string if available.
    pub fn hover(&mut self, file: &Path, line: u32, col: u32) -> HoverResult {
        if !self.initialized {
            return None;
        }

        let uri = path_to_uri(file);
        let id = self.next_id();

        let req = format!(
            r#"{{"jsonrpc":"2.0","id":{id},"method":"textDocument/hover","params":{{"textDocument":{{"uri":"{uri}"}},"position":{{"line":{line},"character":{col}}}}}}}"#
        );

        self.send(&req)?;
        let response = self.recv()?;
        extract_type_from_hover(&response)
    }

    /// Batch hover queries. Returns var_name → type_string map.
    pub fn hover_batch(&mut self, queries: &[HoverQuery]) -> HashMap<String, String> {
        let mut results = HashMap::new();
        for q in queries {
            let path = Path::new(&q.file);
            if let Some(type_str) = self.hover(path, q.line, q.col) {
                let leaf = extract_leaf_type_from_rust(&type_str);
                if !leaf.is_empty() {
                    results.insert(q.var_name.clone(), leaf);
                }
            }
        }
        results
    }

    /// Send didChange notification for a file.
    pub fn did_change(&mut self, file: &Path, content: &str) -> Option<()> {
        let uri = path_to_uri(file);
        let escaped = content.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        let notif = format!(
            r#"{{"jsonrpc":"2.0","method":"textDocument/didChange","params":{{"textDocument":{{"uri":"{uri}","version":1}},"contentChanges":[{{"text":"{escaped}"}}]}}}}"#
        );
        self.send(&notif)
    }

    /// Shutdown the LSP connection.
    pub fn shutdown(&mut self) {
        let id = self.next_id();
        let req = format!(r#"{{"jsonrpc":"2.0","id":{id},"method":"shutdown","params":null}}"#);
        let _ = self.send(&req);
        let _ = self.recv();
        let notif = r#"{"jsonrpc":"2.0","method":"exit","params":null}"#;
        let _ = self.send(notif);
    }

    fn next_id(&mut self) -> i64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn send(&mut self, content: &str) -> Option<()> {
        let header = format!("Content-Length: {}\r\n\r\n", content.len());
        self.stream.write_all(header.as_bytes()).ok()?;
        self.stream.write_all(content.as_bytes()).ok()?;
        self.stream.flush().ok()?;
        Some(())
    }

    fn recv(&mut self) -> Option<String> {
        // Read header
        let mut header_buf = Vec::with_capacity(256);
        let mut prev = 0u8;
        let mut crlf_count = 0u8;
        loop {
            let mut byte = [0u8; 1];
            self.stream.read_exact(&mut byte).ok()?;
            header_buf.push(byte[0]);
            if prev == b'\r' && byte[0] == b'\n' {
                crlf_count += 1;
                if crlf_count == 2 {
                    break;
                }
            } else if byte[0] != b'\r' {
                crlf_count = 0;
            }
            prev = byte[0];
        }

        let header = String::from_utf8_lossy(&header_buf);
        let length: usize = header
            .lines()
            .find(|l| l.starts_with("Content-Length:"))?
            .split(':')
            .nth(1)?
            .trim()
            .parse()
            .ok()?;

        let mut body = vec![0u8; length];
        self.stream.read_exact(&mut body).ok()?;

        // Skip notification messages (no "id"), keep reading until we get a response.
        let text = String::from_utf8_lossy(&body).into_owned();
        if !text.contains("\"id\"") {
            // This is a notification, try to read next message.
            return self.recv();
        }
        Some(text)
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Convert a filesystem path to a file:// URI.
fn path_to_uri(path: &Path) -> String {
    let canonical = path.to_string_lossy().replace('\\', "/");
    if canonical.starts_with('/') {
        format!("file://{canonical}")
    } else {
        format!("file:///{canonical}")
    }
}

/// Extract type info from a hover response JSON.
///
/// rust-analyzer hover returns markdown like:
/// ```rust
/// let x: Vec<String>
/// ```
/// or
/// ```rust
/// fn foo() -> Bar
/// ```
fn extract_type_from_hover(response: &str) -> Option<String> {
    // Find "contents":{"kind":"markdown","value":"..."}
    let value_start = response.find("\"value\":\"")?;
    let value_content = &response[value_start + 9..];
    let value_end = value_content.find("\"}")?;
    let value = &value_content[..value_end];

    // Unescape
    let unescaped = value.replace("\\n", "\n").replace("\\\"", "\"");

    // Look for type annotation patterns
    // Pattern 1: `let var: Type` or `var: Type`
    if let Some(colon_pos) = unescaped.find(": ") {
        let after_colon = unescaped[colon_pos + 2..].trim();
        let type_str = after_colon
            .split('\n')
            .next()
            .unwrap_or(after_colon)
            .trim()
            .trim_end_matches('`');
        if !type_str.is_empty() {
            return Some(type_str.to_owned());
        }
    }

    // Pattern 2: `-> ReturnType`
    if let Some(arrow_pos) = unescaped.find("-> ") {
        let after_arrow = unescaped[arrow_pos + 3..].trim();
        let type_str = after_arrow
            .split('\n')
            .next()
            .unwrap_or(after_arrow)
            .trim()
            .trim_end_matches('`');
        if !type_str.is_empty() {
            return Some(type_str.to_owned());
        }
    }

    None
}

/// Extract leaf type from a full Rust type.
/// `Vec<String>` → `vec`, `HashMap<K, V>` → `hashmap`, `MyStruct` → `mystruct`
fn extract_leaf_type_from_rust(full_type: &str) -> String {
    let base = full_type.split('<').next().unwrap_or(full_type);
    let leaf = base.rsplit("::").next().unwrap_or(base);
    leaf.trim().to_lowercase()
}

/// Check if lspmux server is reachable.
pub fn is_lspmux_available() -> bool {
    TcpStream::connect_timeout(
        &"127.0.0.1:27631".parse().expect("static addr"),
        Duration::from_millis(500),
    ).is_ok()
}
