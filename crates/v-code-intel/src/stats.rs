//! Per-crate statistics computation from code chunks.

use std::collections::BTreeMap;

use crate::graph::is_test_chunk;
use crate::helpers::extract_crate_name;
use crate::parse::ParsedChunk;

/// Schema descriptor for stats JSON output.
const STATS_SCHEMA: &str = "p=prod_fn,t=test_fn,s=struct,e=enum";

/// Build per-crate statistics from code chunks.
///
/// Returns a map from crate name to `[prod_fn, test_fn, struct, enum]` counts.
pub fn build_stats(chunks: &[ParsedChunk]) -> BTreeMap<String, [usize; 4]> {
    let mut stats: BTreeMap<String, [usize; 4]> = BTreeMap::new();
    for c in chunks {
        let crate_name = extract_crate_name(&c.file);
        let row = stats.entry(crate_name).or_insert([0; 4]);
        let is_test = is_test_chunk(c);
        match (c.kind.as_str(), is_test) {
            ("function", false) => row[0] += 1,
            ("function", true) => row[1] += 1,
            ("struct", _) => row[2] += 1,
            ("enum", _) => row[3] += 1,
            _ => {}
        }
    }
    stats
}

/// Build stats JSON Value from a `BTreeMap` of crate stats.
pub fn stats_to_json(stats: &BTreeMap<String, [usize; 4]>) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert("_s".to_owned(), serde_json::Value::String(STATS_SCHEMA.to_owned()));
    for (name, row) in stats {
        map.insert(name.clone(), serde_json::json!({"p":row[0],"t":row[1],"s":row[2],"e":row[3]}));
    }
    serde_json::Value::Object(map)
}
