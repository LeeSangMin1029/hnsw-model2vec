//! Code-specific ingestion utilities.
//!
//! Moved from `v-hnsw-cli::commands::ingest` to decouple tree-sitter
//! from the shared CLI library (v-hnsw binary doesn't need code parsing).

use std::collections::HashMap;
use std::path::Path;

use v_code_chunk as chunk_code;
use v_hnsw_cli::commands::file_index;
use v_hnsw_cli::commands::file_utils::{generate_id, get_file_mtime, normalize_source};
use v_hnsw_cli::commands::ingest::IngestRecord;

/// Intermediate data collected per code chunk before `called_by` resolution.
pub struct CodeChunkEntry {
    pub chunk: chunk_code::CodeChunk,
    pub source: String,
    pub file_path_str: String,
    pub mtime: u64,
    pub lang: &'static str,
}

/// Build `called_by` reverse index from all chunks' `calls` data.
///
/// For each call target, extracts the bare function name (last segment after
/// `::` or `.`) and maps it to the set of callers (qualified chunk names).
pub fn build_called_by_index(entries: &[CodeChunkEntry]) -> HashMap<String, Vec<String>> {
    let mut reverse: HashMap<String, Vec<String>> = HashMap::new();

    for entry in entries {
        let caller = &entry.chunk.name;
        for call in &entry.chunk.calls {
            let bare = call
                .rsplit_once("::")
                .map(|(_, name)| name)
                .or_else(|| call.rsplit_once('.').map(|(_, name)| name))
                .unwrap_or(call);

            reverse
                .entry(bare.to_owned())
                .or_default()
                .push(caller.clone());
        }
    }

    for callers in reverse.values_mut() {
        callers.sort();
        callers.dedup();
    }

    reverse
}

/// Look up `called_by` entries for a given chunk name.
///
/// Checks both the full qualified name and the bare (last segment) name
/// against the reverse index built by [`build_called_by_index`].
pub fn lookup_called_by<'a>(
    reverse: &'a HashMap<String, Vec<String>>,
    chunk_name: &str,
) -> Vec<&'a str> {
    let bare = chunk_name
        .rsplit_once("::")
        .map(|(_, name)| name)
        .unwrap_or(chunk_name);

    let mut result: Vec<&str> = Vec::new();

    if let Some(callers) = reverse.get(bare) {
        for c in callers {
            if c != chunk_name {
                result.push(c.as_str());
            }
        }
    }

    if bare != chunk_name
        && let Some(callers) = reverse.get(chunk_name)
    {
        for c in callers {
            if c != chunk_name && !result.contains(&c.as_str()) {
                result.push(c.as_str());
            }
        }
    }

    result.sort();
    result
}

/// Convert a single code chunk to an [`IngestRecord`].
///
/// Pass `called_by` slice for reverse-reference enrichment (empty slice if unavailable).
pub fn code_chunk_to_record(
    chunk: &chunk_code::CodeChunk,
    source: &str,
    file_path_str: &str,
    lang: &str,
    mtime: u64,
    chunk_total: usize,
    called_by: &[String],
) -> IngestRecord {
    let id = generate_id(source, chunk.chunk_index);
    let embed_text = chunk.to_embed_text(file_path_str, called_by);

    let is_test = source.contains("/tests/")
        || source.contains("\\tests\\")
        || source.contains("/test/")
        || source.contains("\\test\\")
        || source.ends_with("_test.rs")
        || chunk.name.starts_with("test_");

    let mut tags = vec![
        format!("kind:{}", chunk.kind.as_str()),
        format!("lang:{lang}"),
        format!("role:{}", if is_test { "test" } else { "prod" }),
    ];
    if !chunk.visibility.is_empty() {
        tags.push(format!("vis:{}", chunk.visibility));
    }
    for caller in called_by {
        tags.push(format!("caller:{caller}"));
    }

    IngestRecord {
        id,
        text: embed_text,
        source: source.to_owned(),
        title: Some(chunk.name.clone()),
        tags,
        chunk_index: chunk.chunk_index,
        chunk_total,
        source_modified_at: mtime,
        custom: chunk.to_custom_fields(called_by),
    }
}

/// Convert chunked entries into `IngestRecord`s (Pass 2 + Pass 3 combined).
///
/// Builds the called_by reverse index and generates records with caller data.
pub fn entries_to_records(entries: &[CodeChunkEntry]) -> Vec<IngestRecord> {
    let reverse_index = build_called_by_index(entries);

    let chunk_total_map: HashMap<&str, usize> = {
        let mut m: HashMap<&str, usize> = HashMap::new();
        for entry in entries {
            *m.entry(&entry.source).or_default() += 1;
        }
        m
    };

    let mut records: Vec<IngestRecord> = Vec::with_capacity(entries.len());

    for entry in entries {
        let chunk_total = chunk_total_map
            .get(entry.source.as_str())
            .copied()
            .unwrap_or(1);

        let called_by_refs = lookup_called_by(&reverse_index, &entry.chunk.name);
        let called_by: Vec<String> = called_by_refs.iter().map(|s| (*s).to_owned()).collect();

        records.push(code_chunk_to_record(
            &entry.chunk,
            &entry.source,
            &entry.file_path_str,
            entry.lang,
            entry.mtime,
            chunk_total,
            &called_by,
        ));
    }

    records
}

/// Chunk code files via tree-sitter and collect entries with file metadata.
///
/// Iterates over `code_files`, reads each file, chunks it, and populates
/// `entries` and `file_metadata_map`. Skips files that cannot be read or
/// have no supported language / empty chunks.
///
/// `is_interrupted` is polled before each file to support graceful shutdown.
pub fn chunk_code_files(
    code_files: &[impl AsRef<Path>],
    is_interrupted: impl Fn() -> bool,
    entries: &mut Vec<CodeChunkEntry>,
    file_metadata_map: &mut HashMap<String, (u64, u64, Vec<u64>)>,
) {
    for code_path in code_files {
        if is_interrupted() {
            break;
        }
        let code_path = code_path.as_ref();

        let source_code = match std::fs::read_to_string(code_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error reading {}: {e}", code_path.display());
                continue;
            }
        };

        let ext = code_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let chunks = match chunk_code::chunk_for_language(ext, &source_code) {
            Some(c) => c,
            None => continue,
        };
        if chunks.is_empty() {
            continue;
        }

        let source = normalize_source(code_path);
        let file_path_str = code_path.to_string_lossy().to_string();
        let mtime = get_file_mtime(code_path).unwrap_or(0);
        let size = file_index::get_file_size(code_path).unwrap_or(0);
        let mut chunk_ids = Vec::new();

        let lang = chunk_code::lang_for_extension(ext).unwrap_or("unknown");
        for chunk in chunks {
            let id = generate_id(&source, chunk.chunk_index);
            chunk_ids.push(id);
            entries.push(CodeChunkEntry {
                chunk,
                source: source.clone(),
                file_path_str: file_path_str.clone(),
                mtime,
                lang,
            });
        }

        file_metadata_map.insert(source, (mtime, size, chunk_ids));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(name: &str, calls: &[&str]) -> CodeChunkEntry {
        CodeChunkEntry {
            chunk: chunk_code::CodeChunk {
                name: name.to_string(),
                kind: chunk_code::CodeNodeKind::Function,
                text: String::new(),
                calls: calls.iter().map(|s| s.to_string()).collect(),
                call_lines: vec![],
                type_refs: vec![],
                signature: None,
                doc_comment: None,
                visibility: String::new(),
                chunk_index: 0,
                start_line: 0,
                end_line: 0,
                start_byte: 0,
                end_byte: 0,
                imports: vec![],
                param_types: vec![],
                return_type: None,
                ast_hash: 0,
                body_hash: 0,
                sub_blocks: vec![],
            },
            source: "test.rs".to_string(),
            file_path_str: "test.rs".to_string(),
            mtime: 0,
            lang: "rust",
        }
    }

    #[test]
    fn called_by_index_simple() {
        let entries = vec![
            make_entry("main", &["foo", "bar"]),
            make_entry("foo", &["bar"]),
        ];
        let index = build_called_by_index(&entries);

        let foo_callers = index.get("foo").unwrap();
        assert!(foo_callers.contains(&"main".to_string()));

        let bar_callers = index.get("bar").unwrap();
        assert!(bar_callers.contains(&"main".to_string()));
        assert!(bar_callers.contains(&"foo".to_string()));
    }

    #[test]
    fn called_by_index_qualified_calls() {
        let entries = vec![
            make_entry("process", &["Module::helper", "self.validate"]),
        ];
        let index = build_called_by_index(&entries);
        assert!(index.get("helper").unwrap().contains(&"process".to_string()));
        assert!(index.get("validate").unwrap().contains(&"process".to_string()));
    }

    #[test]
    fn called_by_index_deduplicates() {
        let entries = vec![
            make_entry("a", &["target"]),
            make_entry("a", &["target"]),
        ];
        let index = build_called_by_index(&entries);
        assert_eq!(index.get("target").unwrap().len(), 1);
    }

    #[test]
    fn called_by_index_empty() {
        let entries: Vec<CodeChunkEntry> = vec![];
        let index = build_called_by_index(&entries);
        assert!(index.is_empty());
    }

    #[test]
    fn lookup_called_by_bare_name() {
        let entries = vec![
            make_entry("caller_a", &["target_fn"]),
            make_entry("caller_b", &["target_fn"]),
        ];
        let reverse = build_called_by_index(&entries);
        let result = lookup_called_by(&reverse, "target_fn");
        assert!(result.contains(&"caller_a"));
        assert!(result.contains(&"caller_b"));
    }

    #[test]
    fn lookup_called_by_qualified_name() {
        let entries = vec![
            make_entry("handler", &["MyStruct::new"]),
        ];
        let reverse = build_called_by_index(&entries);
        let result = lookup_called_by(&reverse, "MyStruct::new");
        assert!(result.contains(&"handler"));
    }

    #[test]
    fn lookup_called_by_excludes_self() {
        let entries = vec![
            make_entry("recursive", &["recursive"]),
        ];
        let reverse = build_called_by_index(&entries);
        let result = lookup_called_by(&reverse, "recursive");
        assert!(result.is_empty(), "self-calls should be excluded");
    }

    #[test]
    fn lookup_called_by_no_match() {
        let entries = vec![
            make_entry("a", &["b"]),
        ];
        let reverse = build_called_by_index(&entries);
        let result = lookup_called_by(&reverse, "nonexistent");
        assert!(result.is_empty());
    }

    #[test]
    fn lookup_called_by_sorted() {
        let entries = vec![
            make_entry("z_caller", &["target"]),
            make_entry("a_caller", &["target"]),
            make_entry("m_caller", &["target"]),
        ];
        let reverse = build_called_by_index(&entries);
        let result = lookup_called_by(&reverse, "target");
        assert_eq!(result, vec!["a_caller", "m_caller", "z_caller"]);
    }
}
