use crate::commands::intel::parse::CodeChunk;

pub fn chunk(name: &str, file: &str, calls: &[&str]) -> CodeChunk {
    CodeChunk {
        kind: "function".to_owned(),
        name: name.to_owned(),
        file: file.to_owned(),
        lines: Some((1, 10)),
        signature: Some(format!("fn {name}()")),
        calls: calls.iter().map(|s| s.to_string()).collect(),
        types: vec![],
        imports: vec![],
    }
}

pub fn test_chunk(name: &str, file: &str, calls: &[&str]) -> CodeChunk {
    CodeChunk {
        kind: "function".to_owned(),
        name: name.to_owned(),
        file: format!("src/tests/{file}"),
        lines: Some((1, 10)),
        signature: Some(format!("fn {name}()")),
        calls: calls.iter().map(|s| s.to_string()).collect(),
        types: vec![],
        imports: vec![],
    }
}
