//! Extract the resolved lindera version from Cargo.lock and expose it
//! as the `LINDERA_VERSION` env var at compile time.
//!
//! This lets `dict.rs` stamp compiled dictionaries with the exact lindera
//! version, so stale dictionaries are auto-rebuilt after dependency updates.

use std::io::{BufRead, BufReader};
use std::path::Path;

fn main() {
    println!("cargo::rerun-if-changed=../../Cargo.lock");

    let version = read_lindera_version().unwrap_or_else(|| "unknown".to_owned());
    println!("cargo::rustc-env=LINDERA_VERSION={version}");
}

fn read_lindera_version() -> Option<String> {
    let lock_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("Cargo.lock");
    let file = std::fs::File::open(lock_path).ok()?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    while let Some(Ok(line)) = lines.next() {
        if line.trim() == "name = \"lindera\"" {
            if let Some(Ok(ver_line)) = lines.next() {
                return ver_line
                    .trim()
                    .strip_prefix("version = \"")
                    .and_then(|s| s.strip_suffix('"'))
                    .map(|s| s.to_owned());
            }
        }
    }
    None
}
