//! Client-side API for sending LSP requests via shared memory.
//!
//! Clients (v-code, agents) use this to talk to the daemon-hosted RA instance.

use std::path::Path;
use std::time::{Duration, Instant};

use crate::error::{LspError, Result};
use crate::lsp::jsonrpc::{Message, Request, RequestId, Version};
use crate::shm::ShmRing;

/// Request/response pair for shared memory IPC.
///
/// Wire format (in the shm slot):
/// ```text
/// [0]       direction: u8 (0 = request, 1 = response)
/// [1..5]    client_id: u32
/// [5..9]    request_id: u32
/// [9..]     JSON body (Message)
/// ```
/// Envelope direction: client → daemon.
pub const DIR_REQUEST: u8 = 0;
/// Envelope direction: daemon → client.
pub const DIR_RESPONSE: u8 = 1;

/// Envelope wrapping a message in shared memory.
pub struct ShmEnvelope {
    pub direction: u8,
    pub client_id: u32,
    pub request_id: u32,
    pub body: Vec<u8>,
}

impl ShmEnvelope {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(9 + self.body.len());
        buf.push(self.direction);
        buf.extend_from_slice(&self.client_id.to_le_bytes());
        buf.extend_from_slice(&self.request_id.to_le_bytes());
        buf.extend_from_slice(&self.body);
        buf
    }

    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 9 {
            return Err(LspError::Shm("envelope too short".into()));
        }
        Ok(Self {
            direction: data[0],
            client_id: u32::from_le_bytes(
                data[1..5].try_into().map_err(|_| LspError::Shm("bad client_id".into()))?,
            ),
            request_id: u32::from_le_bytes(
                data[5..9].try_into().map_err(|_| LspError::Shm("bad request_id".into()))?,
            ),
            body: data[9..].to_vec(),
        })
    }
}

/// A client that sends LSP requests through shared memory.
pub struct ShmClient {
    ring: ShmRing,
    client_id: u32,
    next_id: u32,
}

impl ShmClient {
    /// Open (or create) the shared memory ring and register as a client.
    pub fn open(shm_path: &Path, client_id: u32) -> Result<Self> {
        let ring = ShmRing::open(shm_path)?;
        Ok(Self {
            ring,
            client_id,
            next_id: 1,
        })
    }

    /// Drain any stale response messages (from previous timed-out requests).
    fn drain_stale_responses(&mut self) {
        loop {
            match self.ring.try_read_filtered(Some(DIR_RESPONSE)) {
                Ok(Some(_)) => continue, // discard stale response
                _ => break,
            }
        }
    }

    /// Send an LSP request and wait for the response.
    pub fn request(
        &mut self,
        method: &str,
        params: serde_json::Value,
        timeout: Duration,
    ) -> Result<serde_json::Value> {
        // Drain stale responses from previous timed-out requests.
        self.drain_stale_responses();

        let req_id = self.next_id;
        self.next_id += 1;

        let msg = Message::Request(Request {
            jsonrpc: Version,
            method: method.to_owned(),
            params,
            id: RequestId::Number(req_id as i64),
        });

        let body = serde_json::to_vec(&msg)?;
        let envelope = ShmEnvelope {
            direction: DIR_REQUEST,
            client_id: self.client_id,
            request_id: req_id,
            body,
        };

        self.ring.write(&envelope.encode())?;

        // Poll for response.
        let deadline = Instant::now() + timeout;
        loop {
            if Instant::now() > deadline {
                return Err(LspError::Timeout);
            }

            if let Some(data) = self.ring.try_read_filtered(Some(DIR_RESPONSE))? {
                let env = ShmEnvelope::decode(&data)?;
                if env.direction == DIR_RESPONSE
                    && env.client_id == self.client_id
                    && env.request_id == req_id
                {
                    let msg: Message = serde_json::from_slice(&env.body)?;
                    return match msg.into_response() {
                        Ok(Ok(success)) => Ok(success.result),
                        Ok(Err(err)) => Err(LspError::Protocol(format!(
                            "LSP error {}: {}",
                            err.error.code, err.error.message
                        ))),
                        Err(_) => Err(LspError::Protocol("unexpected message type".into())),
                    };
                }
                // Stale response for a different request — discard.
            }

            std::thread::sleep(Duration::from_micros(100));
        }
    }

    /// Resolve a call via goto_definition. Returns (definition_file, definition_line).
    pub fn resolve_call(
        &mut self,
        file: &str,
        call_name: &str,
        call_line: u32,
    ) -> Option<(String, u32)> {
        use std::sync::atomic::{AtomicUsize, Ordering as AtOrd};
        static TIMEOUT_COUNT: AtomicUsize = AtomicUsize::new(0);
        static PARSE_FAIL: AtomicUsize = AtomicUsize::new(0);

        let params = serde_json::json!({
            "file": file,
            "call_name": call_name,
            "call_line": call_line,
        });
        let result = match self.request("ra/resolve_call", params, Duration::from_secs(2)) {
            Ok(v) => v,
            Err(e) => {
                let tc = TIMEOUT_COUNT.fetch_add(1, AtOrd::Relaxed);
                if tc < 3 {
                    eprintln!("    [ra-client] request error: {e}");
                }
                if (tc + 1) % 5000 == 0 {
                    eprintln!("    [ra-client] total request errors: {}", tc + 1);
                }
                return None;
            }
        };
        let def_file = match result.get("file").and_then(|v| v.as_str()) {
            Some(f) => f.to_owned(),
            None => {
                let pf = PARSE_FAIL.fetch_add(1, AtOrd::Relaxed);
                if pf < 3 {
                    eprintln!("    [ra-client] parse fail, result={result}");
                }
                if (pf + 1) % 5000 == 0 {
                    eprintln!("    [ra-client] total parse fails: {}", pf + 1);
                }
                return None;
            }
        };
        let def_line = result.get("line").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        Some((def_file, def_line))
    }

    /// Hover on a variable to get its type (for return type inference).
    pub fn hover_on_var(
        &mut self,
        file: &str,
        var_name: &str,
        fn_start_line: u32,
    ) -> Option<String> {
        let params = serde_json::json!({
            "method": "hover_on_var",
            "file": file,
            "var_name": var_name,
            "fn_start_line": fn_start_line,
        });
        let result = self.request("ra/hover_var", params, Duration::from_secs(60)).ok()?;
        result.as_str().map(String::from)
    }

    /// Hover on the receiver before `.method(` to get its type.
    pub fn hover_on_receiver(
        &mut self,
        file: &str,
        method_name: &str,
        call_line: u32,
    ) -> Option<String> {
        let params = serde_json::json!({
            "method": "hover_on_receiver",
            "file": file,
            "method_name": method_name,
            "call_line": call_line,
        });
        let result = self.request("ra/hover_receiver", params, Duration::from_secs(60)).ok()?;
        result.as_str().map(String::from)
    }

    /// Fetch resolve_call diagnostic counters from the daemon.
    pub fn resolve_stats(&mut self) -> Option<serde_json::Value> {
        self.request("ra/resolve_stats", serde_json::json!({}), Duration::from_secs(5)).ok()
    }
}
