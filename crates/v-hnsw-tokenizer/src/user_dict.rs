//! User dictionary support for custom tokenization.
//!
//! Allows loading custom terms from CSV files to improve tokenization
//! of domain-specific vocabulary, proper nouns, etc.

use std::io::{self, BufRead};
use std::path::Path;
use v_hnsw_core::VhnswError;

/// A user dictionary entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DictionaryEntry {
    /// The surface form (how it appears in text).
    pub term: String,
    /// Part of speech tag (e.g., "NNP" for proper noun).
    pub pos: String,
    /// Reading/pronunciation (optional, same as term if not specified).
    pub reading: String,
}

impl DictionaryEntry {
    /// Create a new dictionary entry.
    pub fn new(term: impl Into<String>, pos: impl Into<String>, reading: impl Into<String>) -> Self {
        Self {
            term: term.into(),
            pos: pos.into(),
            reading: reading.into(),
        }
    }

    /// Create an entry with term as the reading.
    pub fn simple(term: impl Into<String>, pos: impl Into<String>) -> Self {
        let term = term.into();
        let reading = term.clone();
        Self {
            term,
            pos: pos.into(),
            reading,
        }
    }

    /// Convert to Lindera CSV format.
    ///
    /// Format: `term,left_id,right_id,cost,pos,pos2,pos3,pos4,conjugation_type,conjugation_form,base_form,reading,pronunciation`
    /// For simplicity, we use a minimal format that Lindera accepts.
    pub fn to_lindera_csv(&self) -> String {
        // Lindera ko-dic format: term,cost,POS,reading
        // We use a low cost to prefer user dictionary entries
        format!("{},0,{},{}", self.term, self.pos, self.reading)
    }
}

/// A collection of user dictionary entries.
#[derive(Debug, Clone, Default)]
pub struct UserDictionary {
    entries: Vec<DictionaryEntry>,
}

impl UserDictionary {
    /// Create an empty user dictionary.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Create a user dictionary with entries.
    pub fn with_entries(entries: Vec<DictionaryEntry>) -> Self {
        Self { entries }
    }

    /// Add an entry to the dictionary.
    pub fn add_entry(&mut self, entry: DictionaryEntry) {
        self.entries.push(entry);
    }

    /// Add a simple entry (term + POS).
    pub fn add_term(&mut self, term: impl Into<String>, pos: impl Into<String>) {
        self.entries.push(DictionaryEntry::simple(term, pos));
    }

    /// Get all entries.
    pub fn entries(&self) -> &[DictionaryEntry] {
        &self.entries
    }

    /// Check if the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Load entries from a CSV file.
    ///
    /// Expected format: `term,pos,reading` (one entry per line).
    /// Lines starting with `#` are treated as comments.
    /// Empty lines are skipped.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or contains invalid entries.
    pub fn load_from_csv<P: AsRef<Path>>(path: P) -> Result<Self, VhnswError> {
        let file = std::fs::File::open(path.as_ref()).map_err(|e| {
            VhnswError::Tokenizer(format!("failed to open user dictionary: {}", e))
        })?;

        Self::load_from_reader(io::BufReader::new(file))
    }

    /// Load entries from a reader.
    ///
    /// Expected format: `term,pos,reading` or `term,pos` (one entry per line).
    /// Lines starting with `#` are treated as comments.
    /// Empty lines are skipped.
    pub fn load_from_reader<R: BufRead>(reader: R) -> Result<Self, VhnswError> {
        let mut entries = Vec::new();

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result.map_err(|e| {
                VhnswError::Tokenizer(format!("failed to read line {}: {}", line_num + 1, e))
            })?;

            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split(',').collect();
            let entry = match parts.len() {
                2 => DictionaryEntry::simple(parts[0].trim(), parts[1].trim()),
                3 => DictionaryEntry::new(
                    parts[0].trim(),
                    parts[1].trim(),
                    parts[2].trim(),
                ),
                _ => {
                    return Err(VhnswError::Tokenizer(format!(
                        "invalid entry at line {}: expected 'term,pos' or 'term,pos,reading', got '{}'",
                        line_num + 1,
                        line
                    )));
                }
            };

            entries.push(entry);
        }

        Ok(Self { entries })
    }

    /// Load entries from a string.
    pub fn load_from_str(s: &str) -> Result<Self, VhnswError> {
        Self::load_from_reader(io::Cursor::new(s))
    }

    /// Convert to Lindera CSV format.
    pub fn to_lindera_csv(&self) -> String {
        self.entries
            .iter()
            .map(|e| e.to_lindera_csv())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_entry_new() {
        let entry = DictionaryEntry::new("삼성전자", "NNP", "삼성전자");
        assert_eq!(entry.term, "삼성전자");
        assert_eq!(entry.pos, "NNP");
        assert_eq!(entry.reading, "삼성전자");
    }

    #[test]
    fn test_dictionary_entry_simple() {
        let entry = DictionaryEntry::simple("카카오톡", "NNP");
        assert_eq!(entry.term, "카카오톡");
        assert_eq!(entry.pos, "NNP");
        assert_eq!(entry.reading, "카카오톡");
    }

    #[test]
    fn test_user_dictionary_add() {
        let mut dict = UserDictionary::new();
        assert!(dict.is_empty());

        dict.add_term("네이버", "NNP");
        dict.add_entry(DictionaryEntry::new("인공지능", "NNG", "인공지능"));

        assert_eq!(dict.len(), 2);
        assert!(!dict.is_empty());
    }

    #[test]
    fn test_load_from_str() {
        let csv = r#"
# This is a comment
삼성전자,NNP,삼성전자
카카오,NNP

# Another comment
네이버,NNP,네이버
"#;

        let dict = UserDictionary::load_from_str(csv).expect("valid csv");
        assert_eq!(dict.len(), 3);

        let entries = dict.entries();
        assert_eq!(entries[0].term, "삼성전자");
        assert_eq!(entries[1].term, "카카오");
        assert_eq!(entries[1].reading, "카카오"); // auto-filled
        assert_eq!(entries[2].term, "네이버");
    }

    #[test]
    fn test_load_from_str_two_column() {
        let csv = "인공지능,NNG\n기계학습,NNG";
        let dict = UserDictionary::load_from_str(csv).expect("valid csv");
        assert_eq!(dict.len(), 2);
        assert_eq!(dict.entries()[0].term, "인공지능");
        assert_eq!(dict.entries()[0].reading, "인공지능");
    }

    #[test]
    fn test_load_from_str_invalid() {
        let csv = "invalid,entry,with,too,many,columns";
        let result = UserDictionary::load_from_str(csv);
        assert!(result.is_err());
    }

    #[test]
    fn test_to_lindera_csv() {
        let mut dict = UserDictionary::new();
        dict.add_entry(DictionaryEntry::new("삼성", "NNP", "삼성"));
        dict.add_entry(DictionaryEntry::new("전자", "NNG", "전자"));

        let csv = dict.to_lindera_csv();
        assert!(csv.contains("삼성,0,NNP,삼성"));
        assert!(csv.contains("전자,0,NNG,전자"));
    }

    #[test]
    fn test_empty_dictionary() {
        let dict = UserDictionary::new();
        assert!(dict.is_empty());
        assert_eq!(dict.len(), 0);
        assert!(dict.to_lindera_csv().is_empty());
    }
}
