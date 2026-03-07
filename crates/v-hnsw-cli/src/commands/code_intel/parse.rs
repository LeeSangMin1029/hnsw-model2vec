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
    pub types: Vec<String>,
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

    // Name is the last token (may have "pub" prefix)
    let name = rest.split_whitespace().last().unwrap_or(rest).to_owned();

    let mut file = String::new();
    let mut line_range = None;
    let mut signature = None;
    let mut calls = Vec::new();
    let mut types = Vec::new();

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
            calls = c.split(", ").map(|s| s.trim().to_owned()).collect();
        } else if let Some(t) = line.strip_prefix("Types: ") {
            types = t.split(", ").map(|s| s.trim().to_owned()).collect();
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
        types,
    })
}

/// Normalize Windows backslashes and strip leading `.\` for display.
fn normalize_path(p: &str) -> String {
    let s = p.replace('\\', "/");
    let s = s.strip_prefix("./").unwrap_or(&s);
    s.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_function() {
        let text = "[function] topological_sort\n\
                     File: .\\crates\\swarm-cli\\src\\engine.rs:47-54\n\
                     Signature: fn topological_sort(dag: &Dag) -> Option<Vec<String>>\n\
                     Calls: toposort, sort_by";
        let chunk = parse_chunk(text).unwrap();
        assert_eq!(chunk.kind, "function");
        assert_eq!(chunk.name, "topological_sort");
        assert_eq!(chunk.file, "crates/swarm-cli/src/engine.rs");
        assert_eq!(chunk.lines, Some((47, 54)));
        assert_eq!(chunk.calls, vec!["toposort", "sort_by"]);
    }

    #[test]
    fn parse_pub_method() {
        let text = "[function] pub ReadyQueue::pop_batch\n\
                     File: .\\crates\\swarm-core\\src\\ready_queue.rs:99-137\n\
                     Doc comment here\n\
                     Signature: pub fn pop_batch(&mut self, dag: &Dag) -> Vec<String>\n\
                     Types: Dag, Reverse\n\
                     Calls: Vec::new, batch.push";
        let chunk = parse_chunk(text).unwrap();
        assert_eq!(chunk.name, "ReadyQueue::pop_batch");
        assert_eq!(chunk.types, vec!["Dag", "Reverse"]);
        assert_eq!(chunk.calls, vec!["Vec::new", "batch.push"]);
    }

    #[test]
    fn parse_non_code_returns_none() {
        let text = "Just a regular markdown paragraph.";
        assert!(parse_chunk(text).is_none());
    }

    #[test]
    fn normalize_backslashes() {
        assert_eq!(normalize_path(".\\crates\\foo.rs"), "crates/foo.rs");
        assert_eq!(normalize_path("./src/main.rs"), "src/main.rs");
    }
}
