//! Import command - Import data from JSONL file (same as insert, alias).

use std::path::PathBuf;

use anyhow::Result;

use super::insert;

/// Run the import command.
///
/// This is an alias for the insert command.
pub fn run(path: PathBuf, input: PathBuf) -> Result<()> {
    insert::run(path, input, "vector", false, "text", "minishlab/potion-multilingual-128M", 128)
}
