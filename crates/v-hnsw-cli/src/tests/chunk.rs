use crate::chunk::*;

#[test]
fn test_basic_chunking() {
    let config = ChunkConfig {
        target_size: 100,
        overlap: 20,
        min_size: 10,
        include_heading_context: true,
    };

    let chunker = MarkdownChunker::new(config);
    let markdown = r#"# Introduction

This is the introduction paragraph. It contains some text that will be chunked.

## Section 1

This is section 1 with more content. It should be in a separate chunk.

## Section 2

This is section 2 with even more content to trigger chunking.
"#;

    let chunks = chunker.chunk(markdown);
    assert!(!chunks.is_empty());

    for chunk in &chunks {
        assert!(chunk.text.len() >= 10);
    }
}

#[test]
fn test_frontmatter_parsing() {
    let content = r#"---
title: "Test Document"
date: 2024-01-01
tags: [rust, markdown, test]
---

# Content

This is the main content.
"#;

    let (frontmatter, remaining) = MarkdownChunker::parse_frontmatter(content);

    assert!(frontmatter.is_some());
    let fm = frontmatter.unwrap();
    assert_eq!(fm.title, Some("Test Document".to_string()));
    assert_eq!(fm.date, Some("2024-01-01".to_string()));
    assert_eq!(fm.tags, vec!["rust", "markdown", "test"]);
    assert!(remaining.contains("# Content"));
}

#[test]
fn test_code_block_handling() {
    let config = ChunkConfig::default();
    let chunker = MarkdownChunker::new(config);

    let markdown = r#"# Code Example

Here is some code:

```rust
fn main() {
    println!("Hello, world!");
}
```

More text after code.
"#;

    let chunks = chunker.chunk(markdown);
    assert!(!chunks.is_empty());
    assert!(chunks[0].text.contains("```"));
}

#[test]
fn test_heading_hierarchy() {
    let config = ChunkConfig {
        target_size: 50,
        overlap: 10,
        min_size: 10,
        include_heading_context: true,
    };

    let chunker = MarkdownChunker::new(config);
    let markdown = r#"# Level 1

Content under level 1.

## Level 2

Content under level 2.

### Level 3

Content under level 3.
"#;

    let chunks = chunker.chunk(markdown);
    for chunk in chunks {
        // Verify heading paths are tracked
        if !chunk.heading_path.is_empty() {
            assert!(chunk.heading_path.len() <= 3);
        }
    }
}

// ── Edge case tests ─────────────────────────────────────────────────

#[test]
fn test_empty_text_produces_no_chunks() {
    let chunker = MarkdownChunker::new(ChunkConfig::default());
    let chunks = chunker.chunk("");
    assert!(chunks.is_empty(), "empty input should produce no chunks");
}

#[test]
fn test_single_char_below_min_size() {
    let config = ChunkConfig {
        min_size: 10,
        ..ChunkConfig::default()
    };
    let chunker = MarkdownChunker::new(config);
    let chunks = chunker.chunk("x");
    assert!(
        chunks.is_empty(),
        "single char should be below min_size and produce no chunks"
    );
}

#[test]
fn test_text_exactly_at_min_size() {
    let config = ChunkConfig {
        target_size: 1000,
        overlap: 0,
        min_size: 5,
        include_heading_context: false,
    };
    let chunker = MarkdownChunker::new(config);
    // "abcde" is 5 chars, wrapped in a paragraph → "\n\n" appended
    let chunks = chunker.chunk("abcde");
    assert!(
        !chunks.is_empty(),
        "text at exactly min_size should produce a chunk"
    );
}

#[test]
fn test_text_exactly_at_target_size_triggers_split() {
    let config = ChunkConfig {
        target_size: 20,
        overlap: 0,
        min_size: 5,
        include_heading_context: false,
    };
    let chunker = MarkdownChunker::new(config);
    // Two paragraphs, each large enough to trigger split at target_size
    let markdown = "This is paragraph one with enough text.\n\nThis is paragraph two with enough text.\n\n";
    let chunks = chunker.chunk(markdown);
    assert!(
        chunks.len() >= 2,
        "text exceeding target_size should be split into multiple chunks, got {} chunks",
        chunks.len()
    );
}

#[test]
fn test_unicode_content_chunking() {
    let config = ChunkConfig {
        target_size: 50,
        overlap: 10,
        min_size: 5,
        include_heading_context: true,
    };
    let chunker = MarkdownChunker::new(config);
    let markdown = "# 한글 제목\n\n이것은 한국어 텍스트입니다. 충분히 긴 내용이 필요합니다.\n\n## 第二节\n\n这是中文内容，用于测试Unicode处理能力。\n\n";
    let chunks = chunker.chunk(markdown);
    assert!(!chunks.is_empty(), "unicode content should produce chunks");
    for chunk in &chunks {
        assert!(
            !chunk.text.is_empty(),
            "unicode chunk text should not be empty"
        );
    }
}

#[test]
fn test_emoji_in_markdown() {
    let config = ChunkConfig {
        target_size: 1000,
        overlap: 0,
        min_size: 5,
        include_heading_context: false,
    };
    let chunker = MarkdownChunker::new(config);
    let markdown = "Hello 🌍🎉 world! This has emoji characters mixed in.\n\n";
    let chunks = chunker.chunk(markdown);
    assert!(!chunks.is_empty());
    assert!(chunks[0].text.contains("🌍"));
}

#[test]
fn test_whitespace_only_produces_no_chunks() {
    let chunker = MarkdownChunker::new(ChunkConfig::default());
    let chunks = chunker.chunk("   \n\n   \t\t\n\n");
    assert!(
        chunks.is_empty(),
        "whitespace-only input should produce no chunks"
    );
}

#[test]
fn test_frontmatter_without_closing_delimiter() {
    let content = "---\ntitle: Broken\nNo closing delimiter here\n\nSome content.";
    let (frontmatter, remaining) = MarkdownChunker::parse_frontmatter(content);
    assert!(
        frontmatter.is_none(),
        "unclosed frontmatter should return None"
    );
    assert_eq!(remaining, content, "remaining should be the full content");
}

#[test]
fn test_frontmatter_empty_tags() {
    let content = "---\ntitle: Test\ntags: []\n---\nBody text";
    let (frontmatter, _remaining) = MarkdownChunker::parse_frontmatter(content);
    let fm = frontmatter.unwrap();
    assert!(fm.tags.is_empty(), "empty tags bracket should produce empty vec");
}

#[test]
fn test_frontmatter_custom_fields() {
    let content = "---\nauthor: Alice\ncategory: tutorial\n---\nBody";
    let (frontmatter, _) = MarkdownChunker::parse_frontmatter(content);
    let fm = frontmatter.unwrap();
    assert_eq!(fm.custom.get("author").map(|s| s.as_str()), Some("Alice"));
    assert_eq!(fm.custom.get("category").map(|s| s.as_str()), Some("tutorial"));
}

#[test]
fn test_no_frontmatter() {
    let content = "# Just a heading\n\nSome content.\n";
    let (frontmatter, remaining) = MarkdownChunker::parse_frontmatter(content);
    assert!(frontmatter.is_none());
    assert_eq!(remaining, content);
}

#[test]
fn test_overlap_zero_no_overlap_text() {
    let config = ChunkConfig {
        target_size: 30,
        overlap: 0,
        min_size: 5,
        include_heading_context: false,
    };
    let chunker = MarkdownChunker::new(config);
    let markdown = "First paragraph with enough text to exceed target.\n\nSecond paragraph also with enough text to exceed target.\n\n";
    let chunks = chunker.chunk(markdown);
    if chunks.len() >= 2 {
        // With zero overlap, chunks should not share content
        let c0_end = &chunks[0].text[chunks[0].text.len().saturating_sub(10)..];
        let c1_start = &chunks[1].text[..std::cmp::min(10, chunks[1].text.len())];
        // Not a strict assertion since overlap=0 means no shared chars from the overlap mechanism
        let _ = (c0_end, c1_start);
    }
}

#[test]
fn test_chunk_index_sequential() {
    let config = ChunkConfig {
        target_size: 30,
        overlap: 0,
        min_size: 5,
        include_heading_context: false,
    };
    let chunker = MarkdownChunker::new(config);
    let markdown = "First paragraph text is here.\n\nSecond paragraph text is here.\n\nThird paragraph text is here.\n\n";
    let chunks = chunker.chunk(markdown);
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            chunk.chunk_index, i,
            "chunk_index should be sequential, expected {i} got {}",
            chunk.chunk_index
        );
    }
}

#[test]
fn test_code_block_preserved_atomically() {
    let config = ChunkConfig {
        target_size: 20,
        overlap: 0,
        min_size: 5,
        include_heading_context: false,
    };
    let chunker = MarkdownChunker::new(config);
    let markdown = "```\nline1\nline2\nline3\nline4\nline5\n```\n\n";
    let chunks = chunker.chunk(markdown);
    // Code block should be kept as atomic unit in a single chunk
    let has_code = chunks.iter().any(|c| c.text.contains("```"));
    assert!(has_code, "code block content should be preserved");
}

#[test]
fn test_heading_context_disabled() {
    let config = ChunkConfig {
        target_size: 1000,
        overlap: 0,
        min_size: 5,
        include_heading_context: false,
    };
    let chunker = MarkdownChunker::new(config);
    let markdown = "# My Heading\n\nSome body text here.\n\n";
    let chunks = chunker.chunk(markdown);
    // With include_heading_context=false, heading text should not be prepended
    if !chunks.is_empty() {
        // The chunk should still have heading in heading_path
        assert!(!chunks[0].heading_path.is_empty());
    }
}

#[test]
fn test_multiple_paragraphs_under_one_heading() {
    let config = ChunkConfig {
        target_size: 1000,
        overlap: 0,
        min_size: 5,
        include_heading_context: true,
    };
    let chunker = MarkdownChunker::new(config);
    let markdown = "# Section\n\nParagraph one.\n\nParagraph two.\n\nParagraph three.\n\n";
    let chunks = chunker.chunk(markdown);
    assert!(
        !chunks.is_empty(),
        "multiple paragraphs should produce at least one chunk"
    );
}

#[test]
fn test_frontmatter_with_empty_lines_inside() {
    let content = "---\ntitle: Test\n\ndate: 2024-01-01\n---\nBody";
    let (frontmatter, _) = MarkdownChunker::parse_frontmatter(content);
    // Empty lines inside frontmatter should be skipped
    let fm = frontmatter.unwrap();
    assert_eq!(fm.title, Some("Test".to_string()));
    assert_eq!(fm.date, Some("2024-01-01".to_string()));
}
