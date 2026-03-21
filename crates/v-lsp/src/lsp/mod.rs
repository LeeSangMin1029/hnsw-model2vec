//! LSP protocol types and sync transport.
//!
//! Ported from lspmux (EUPL-1.2) — adapted for sync I/O (no tokio).
//! Only the subset needed for type inference queries is included.

pub mod jsonrpc;
pub mod transport;

use serde::{Deserialize, Deserializer, Serialize};

// ── Initialize request/response ─────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    pub process_id: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_uri: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<serde_json::Value>,

    #[serde(
        skip_serializing_if = "Option::is_none",
        default,
        deserialize_with = "deserialize_workspace_folders"
    )]
    pub workspace_folders: Option<Vec<WorkspaceFolder>>,

    /// Pass-through for any server-specific options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initialization_options: Option<serde_json::Value>,
}

fn deserialize_workspace_folders<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<Vec<WorkspaceFolder>>, D::Error>
where
    D: Deserializer<'de>,
{
    Deserialize::deserialize(deserializer).map(|v: Option<Vec<_>>| Some(v.unwrap_or_default()))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct WorkspaceFolder {
    pub uri: String,
    pub name: String,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    pub capabilities: serde_json::Value,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_info: Option<ServerInfo>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ServerInfo {
    pub name: String,
    pub version: Option<String>,
}

// ── textDocument types ──────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TextDocumentIdentifier {
    pub uri: String,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TextDocumentPositionParams {
    pub text_document: TextDocumentIdentifier,
    pub position: Position,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Position {
    pub line: u32,
    pub character: u32,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Range {
    pub start: Position,
    pub end: Position,
}

// ── Hover ───────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct HoverParams {
    #[serde(flatten)]
    pub text_document_position: TextDocumentPositionParams,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Hover {
    pub contents: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range: Option<Range>,
}

// ── didOpen / didClose / didChange ──────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct DidOpenTextDocumentParams {
    pub text_document: TextDocumentItem,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TextDocumentItem {
    pub uri: String,
    pub language_id: String,
    pub version: u64,
    pub text: String,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct DidCloseTextDocumentParams {
    pub text_document: TextDocumentIdentifier,
}
