//! JSON-RPC 2.0 message types.
//!
//! Ported from lspmux (EUPL-1.2, p2502/lspmux).

use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum Message {
    Request(Request),
    Notification(Notification),
    ResponseError(ResponseError),
    ResponseSuccess(ResponseSuccess),
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub jsonrpc: Version,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
    pub id: RequestId,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Notification {
    pub jsonrpc: Version,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ResponseError {
    pub jsonrpc: Version,
    pub error: Error,
    pub id: RequestId,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ResponseSuccess {
    pub jsonrpc: Version,
    #[serde(default)]
    pub result: serde_json::Value,
    pub id: RequestId,
}

impl ResponseSuccess {
    pub fn null(id: RequestId) -> Self {
        Self {
            jsonrpc: Version,
            result: serde_json::Value::Null,
            id,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Error {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum RequestId {
    Number(i64),
    String(String),
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Number(n) => write!(f, "{n}"),
            Self::String(s) => write!(f, "{s}"),
        }
    }
}

// ── Version ZST ─────────────────────────────────────────────────────

/// ZST representation of the JSON-RPC `"2.0"` version string.
#[derive(Clone, Copy)]
pub struct Version;

impl Serialize for Version {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        serializer.serialize_str("2.0")
    }
}

impl<'de> serde::de::Visitor<'de> for Version {
    type Value = Version;

    fn expecting(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(r#"string value "2.0""#)
    }

    fn visit_str<E>(self, v: &str) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        match v {
            "2.0" => Ok(Version),
            _ => Err(E::custom("unsupported JSON-RPC version")),
        }
    }
}

impl<'de> Deserialize<'de> for Version {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        deserializer.deserialize_str(Version)
    }
}

// ── From impls ──────────────────────────────────────────────────────

impl From<Request> for Message {
    fn from(value: Request) -> Self { Self::Request(value) }
}

impl From<Notification> for Message {
    fn from(value: Notification) -> Self { Self::Notification(value) }
}

impl From<ResponseSuccess> for Message {
    fn from(value: ResponseSuccess) -> Self { Self::ResponseSuccess(value) }
}

impl From<ResponseError> for Message {
    fn from(value: ResponseError) -> Self { Self::ResponseError(value) }
}

// ── Helpers ─────────────────────────────────────────────────────────

impl Message {
    /// Try to interpret this message as a response (success or error).
    pub fn into_response(self) -> std::result::Result<
        std::result::Result<ResponseSuccess, ResponseError>,
        Self,
    > {
        match self {
            Self::ResponseSuccess(s) => Ok(Ok(s)),
            Self::ResponseError(e) => Ok(Err(e)),
            other => Err(other),
        }
    }
}

impl fmt::Debug for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match serde_json::to_string(self) {
            Ok(json) => f.write_str(&json),
            Err(e) => write!(f, "<serialization error: {e}>"),
        }
    }
}
