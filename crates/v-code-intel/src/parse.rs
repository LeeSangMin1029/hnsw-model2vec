//! Text field parser for code chunks.
//!
//! Parses the structured text field produced by `chunk_code` into a
//! [`CodeChunk`] struct for structural queries.

/// Structured representation of a code chunk's text field.
#[derive(Debug, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode)]
pub struct CodeChunk {
    pub kind: String,
    pub name: String,
    pub file: String,
    pub lines: Option<(usize, usize)>,
    pub signature: Option<String>,
    pub calls: Vec<String>,
    /// Source line (1-based) of each call in `calls` (parallel array).
    pub call_lines: Vec<u32>,
    pub types: Vec<String>,
    /// File-level import statements (loaded from payload custom fields).
    #[serde(default)]
    pub imports: Vec<String>,
    /// String literal arguments: (callee, value, line_1based, arg_position).
    #[serde(default)]
    pub string_args: Vec<(String, String, u32, u8)>,
    /// Parameter-to-callee argument flows: (param_name, param_pos, callee, callee_arg, line).
    #[serde(default)]
    pub param_flows: Vec<(String, u8, String, u8, u32)>,
}

/// Parse the text field of a code chunk into a [`CodeChunk`].
///
/// Expected format (first line is `[kind] [vis] name`):
/// ```text
/// [function] pub ReadyQueue::pop_batch
/// File: .\crates\swarm-core\src\ready_queue.rs:99-137
/// Signature: pub fn pop_batch(&mut self, dag: &Dag) -> Vec<String>
/// Types: Dag, Reverse, String, Vec
/// Calls: Reverse, Vec::new, batch.push
/// ```
pub fn parse_chunk(text: &str) -> Option<CodeChunk> {
    let mut lines_iter = text.lines();
    let first = lines_iter.next()?;

    // Must start with [kind]
    if !first.starts_with('[') {
        return None;
    }
    let bracket_end = first.find(']')?;
    let kind = first[1..bracket_end].to_owned();
    let rest = first[bracket_end + 1..].trim();

    // Name extraction: for impl/trait, take everything after optional visibility
    // prefix (e.g. "pub"); for others, take the last token.
    // "[impl] VectorIndex for HnswGraph<D>" → "VectorIndex for HnswGraph<D>"
    // "[function] pub StorageEngine::insert" → "StorageEngine::insert"
    let name = if kind == "impl" || kind == "trait" {
        // Strip leading "pub"/"pub(crate)" prefix if present
        let stripped = rest.strip_prefix("pub(crate) ")
            .or_else(|| rest.strip_prefix("pub "))
            .unwrap_or(rest);
        stripped.to_owned()
    } else {
        // Strip optional visibility prefix, then take everything as the name.
        // Cannot split by whitespace because generic names contain spaces
        // (e.g. "SimpleHybridSearcher<D, T>::search_ext").
        let stripped = rest.strip_prefix("pub(crate) ")
            .or_else(|| rest.strip_prefix("pub "))
            .or_else(|| rest.strip_prefix("export "))
            .unwrap_or(rest);
        stripped.to_owned()
    };

    let mut file = String::new();
    let mut line_range = None;
    let mut signature = None;
    let mut calls = Vec::new();
    let mut call_lines: Vec<u32> = Vec::new();
    let mut types = Vec::new();
    let mut string_args: Vec<(String, String, u32, u8)> = Vec::new();
    let mut param_flows: Vec<(String, u8, String, u8, u32)> = Vec::new();

    for line in lines_iter {
        if let Some(f) = line.strip_prefix("File: ") {
            let f = f.trim();
            // Parse "path:start-end" or just "path"
            if let Some(colon) = f.rfind(':') {
                let path_part = &f[..colon];
                let range_part = &f[colon + 1..];
                if let Some(dash) = range_part.find('-')
                    && let (Ok(s), Ok(e)) = (
                        range_part[..dash].parse::<usize>(),
                        range_part[dash + 1..].parse::<usize>(),
                    )
                {
                    file = normalize_path(path_part);
                    line_range = Some((s, e));
                    continue;
                }
            }
            file = normalize_path(f);
        } else if let Some(s) = line.strip_prefix("Signature: ") {
            signature = Some(s.trim().to_owned());
        } else if let Some(c) = line.strip_prefix("Calls: ") {
            // Parse "name@line" annotations: split off @N suffix
            for token in c.split(", ") {
                let token = token.trim();
                if let Some(at) = token.rfind('@') {
                    if let Ok(line_num) = token[at + 1..].parse::<u32>() {
                        calls.push(token[..at].to_owned());
                        call_lines.push(line_num);
                        continue;
                    }
                }
                calls.push(token.to_owned());
                call_lines.push(0);
            }
        } else if let Some(t) = line.strip_prefix("Types: ") {
            types = t.split(", ").map(|s| s.trim().to_owned()).collect();
        } else if let Some(f) = line.strip_prefix("Flows: ") {
            for token in f.split(", ") {
                // Format: "param→callee"
                if let Some(arrow) = token.find('\u{2192}') {
                    let param = token[..arrow].to_owned();
                    let callee = token[arrow + '\u{2192}'.len_utf8()..].to_owned();
                    param_flows.push((param, 0, callee, 0, 0));
                }
            }
        } else if let Some(s) = line.strip_prefix("Strings: ") {
            // Format: Command::new("claude"), env::var("API_KEY")
            for token in s.split(", ") {
                let token = token.trim();
                if let Some(paren) = token.find('(')
                    && let Some(end) = token.rfind(')')
                    && paren < end
                {
                    let callee = token[..paren].to_owned();
                    let inner = &token[paren + 1..end];
                    let value = inner.trim_matches('"').to_owned();
                    string_args.push((callee, value, 0, 0));
                }
            }
        }
    }

    if file.is_empty() {
        return None;
    }

    Some(CodeChunk {
        kind,
        name,
        file,
        lines: line_range,
        signature,
        calls,
        call_lines,
        types,
        imports: Vec::new(),
        string_args,
        param_flows,
    })
}

/// Normalize Windows backslashes and strip leading `.\` for display.
pub fn normalize_path(p: &str) -> String {
    let s = p.replace('\\', "/");
    let s = s.strip_prefix("./").unwrap_or(&s);
    s.to_owned()
}
